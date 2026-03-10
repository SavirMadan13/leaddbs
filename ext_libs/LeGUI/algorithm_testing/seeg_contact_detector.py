#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import io as sio
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


ELEC_NAME_RE = re.compile(r"^(.*?)(\d+)$")


@dataclass
class CaseRecord:
    subject_id: str
    ct_path: Path
    mri_path: Path
    surface_path: Path
    elec_data_path: Path


@dataclass
class CandidateSet:
    points_mm: np.ndarray
    confidences: np.ndarray
    inside_mask: np.ndarray
    near_surface_mask: np.ndarray
    intracranial_inside_mask: np.ndarray
    intracranial_near_mask: np.ndarray
    blob_scores: np.ndarray
    intensity_scores: np.ndarray
    surface_support_fraction: float
    intracranial_support_fraction: float


@dataclass
class ChainHypothesis:
    indices: tuple[int, ...]
    ordered_points_mm: np.ndarray
    ordered_scores: np.ndarray
    spacing_mm: float
    span_mm: float
    regularity: float
    brain_support: float
    inside_fraction: float
    near_fraction: float
    score: float


@dataclass
class FittedShaft:
    inferred_contacts: int
    observed_contacts: int
    score: float
    support_score: float
    evidence_score: float
    spacing_mm: float
    brain_support: float
    outside_fraction: float
    contacts_mm: np.ndarray
    seed_indices: tuple[int, ...]


class SurfaceMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if faces.min() == 1:
            faces = faces - 1
        self.faces = faces

        tri = self.vertices[self.faces]
        self.v0 = tri[:, 0, :]
        self.v1 = tri[:, 1, :]
        self.v2 = tri[:, 2, :]
        self.e1 = self.v1 - self.v0
        self.e2 = self.v2 - self.v0
        yz = tri[:, :, 1:3]
        self.ymin = yz[:, :, 0].min(axis=1)
        self.ymax = yz[:, :, 0].max(axis=1)
        self.zmin = yz[:, :, 1].min(axis=1)
        self.zmax = yz[:, :, 1].max(axis=1)
        self.xmax = tri[:, :, 0].max(axis=1)
        self.vertex_tree = cKDTree(self.vertices)

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        out = np.zeros(len(pts), dtype=bool)
        ray = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        eps = 1e-9

        for i, p in enumerate(pts):
            tri_mask = (
                (self.ymin <= p[1])
                & (p[1] <= self.ymax)
                & (self.zmin <= p[2])
                & (p[2] <= self.zmax)
                & (self.xmax > p[0])
            )
            if not np.any(tri_mask):
                continue

            v0 = self.v0[tri_mask]
            e1 = self.e1[tri_mask]
            e2 = self.e2[tri_mask]

            h = np.cross(np.broadcast_to(ray, e2.shape), e2)
            a = np.einsum("ij,ij->i", e1, h)
            ok = np.abs(a) > eps
            if not np.any(ok):
                continue

            v0 = v0[ok]
            e1 = e1[ok]
            e2 = e2[ok]
            h = h[ok]
            a = a[ok]

            f = 1.0 / a
            s = p[None, :] - v0
            u = f * np.einsum("ij,ij->i", s, h)
            ok = (u >= -eps) & (u <= 1.0 + eps)
            if not np.any(ok):
                continue

            v0 = v0[ok]
            e1 = e1[ok]
            e2 = e2[ok]
            s = s[ok]
            u = u[ok]
            f = f[ok]

            q = np.cross(s, e1)
            v = f * q[:, 0]
            ok = (v >= -eps) & (u + v <= 1.0 + eps)
            if not np.any(ok):
                continue

            e2 = e2[ok]
            q = q[ok]
            f = f[ok]
            t = f * np.einsum("ij,ij->i", e2, q)
            out[i] = bool(np.count_nonzero(t > eps) % 2)

        return out

    def nearest_vertex_distance(self, points: np.ndarray) -> np.ndarray:
        d, _ = self.vertex_tree.query(np.asarray(points, dtype=np.float64), k=1)
        return d.astype(np.float64)


def load_case_table(csv_path: Path) -> list[CaseRecord]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    cases: list[CaseRecord] = []
    for row in rows:
        cases.append(
            CaseRecord(
                subject_id=row["subject_id"],
                ct_path=Path(row["ct_path"]),
                mri_path=Path(row["mri_path"]),
                surface_path=Path(row["surface_path"]),
                elec_data_path=Path(row["elec_data_path"]),
            )
        )
    return cases


def load_expected_counts(tsv_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with tsv_path.open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            name = row["name"].strip()
            match = ELEC_NAME_RE.match(name)
            if match is None:
                raise ValueError(f"Could not parse electrode contact name: {name}")
            elec_id = match.group(1)
            counts[elec_id] = counts.get(elec_id, 0) + 1
    return counts


def load_surface(surface_path: Path) -> SurfaceMesh:
    data = sio.loadmat(surface_path, squeeze_me=True, struct_as_record=False)
    surf = data["ProjSurfRaw"]
    return SurfaceMesh(vertices=np.asarray(surf.vertices), faces=np.asarray(surf.faces))


def voxel_sizes_from_affine(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def voxel_to_world(points_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_ijk, dtype=np.float64)
    hom = np.c_[pts, np.ones(len(pts))]
    return (hom @ affine.T)[:, :3]


def world_to_voxel(points_xyz: np.ndarray, inv_affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    hom = np.c_[pts, np.ones(len(pts))]
    return (hom @ inv_affine.T)[:, :3]


def compute_blob_response(image: np.ndarray, voxel_sizes: np.ndarray) -> np.ndarray:
    sigmas_mm = (0.4, 0.7, 1.0, 1.3)
    blob = np.zeros_like(image, dtype=np.float32)
    for sigma_mm in sigmas_mm:
        sigma_vox = sigma_mm / voxel_sizes
        g1 = ndimage.gaussian_filter(image, sigma=sigma_vox, mode="nearest")
        g2 = ndimage.gaussian_filter(image, sigma=sigma_vox * 1.6, mode="nearest")
        dog = -(g2 - g1) * (sigma_mm**2)
        blob = np.maximum(blob, dog.astype(np.float32))

    pos = blob[blob > 0]
    if pos.size == 0:
        return blob
    scale = np.percentile(pos, 99.5)
    if scale > 0:
        blob = blob / scale
    return blob


def sample_trilinear(volume: np.ndarray, points_ijk: np.ndarray) -> np.ndarray:
    coords = np.asarray(points_ijk, dtype=np.float64).T
    return ndimage.map_coordinates(volume, coords, order=1, mode="nearest")


def candidate_peak_target(shape: tuple[int, ...]) -> int:
    n_vox = int(np.prod(shape))
    return int(np.clip(round(n_vox / 70000.0), 180, 320))


def candidate_cap(shape: tuple[int, ...]) -> int:
    n_vox = int(np.prod(shape))
    return int(np.clip(round(n_vox / 18000.0), 700, 1400))


def empty_candidate_set() -> CandidateSet:
    return CandidateSet(
        points_mm=np.zeros((0, 3), dtype=np.float64),
        confidences=np.zeros(0, dtype=np.float64),
        inside_mask=np.zeros(0, dtype=bool),
        near_surface_mask=np.zeros(0, dtype=bool),
        intracranial_inside_mask=np.zeros(0, dtype=bool),
        intracranial_near_mask=np.zeros(0, dtype=bool),
        blob_scores=np.zeros(0, dtype=np.float64),
        intensity_scores=np.zeros(0, dtype=np.float64),
        surface_support_fraction=0.0,
        intracranial_support_fraction=0.0,
    )


def infer_tissue_mask_paths(mri_path: Path) -> dict[str, Path]:
    stem = mri_path.stem
    anchor = "_desc-preproc_acq-iso_T1w"
    if anchor not in stem:
        return {}
    prefix = stem[: -len("_T1w")] if stem.endswith("_T1w") else stem
    parent = mri_path.parent
    out = {
        "gm": parent / f"{prefix}_label-GM_mod-iso_T1w_mask.nii",
        "wm": parent / f"{prefix}_label-WM_mod-iso_T1w_mask.nii",
        "csf": parent / f"{prefix}_label-CSF_mod-iso_T1w_mask.nii",
    }
    if not all(p.exists() for p in out.values()):
        return {}
    return out


def load_intracranial_mask(case: CaseRecord, ct_shape: tuple[int, ...], ct_affine: np.ndarray) -> np.ndarray | None:
    paths = infer_tissue_mask_paths(case.mri_path)
    if not paths:
        return None

    masks = []
    for key in ("gm", "wm", "csf"):
        img = nib.load(str(paths[key]))
        if img.shape != ct_shape or not np.allclose(img.affine, ct_affine, atol=1e-3):
            return None
        data = img.get_fdata(dtype=np.float32)
        masks.append(data > 0.5)

    intracranial = np.logical_or.reduce(masks)
    intracranial = ndimage.binary_fill_holes(intracranial)
    voxel_sizes = voxel_sizes_from_affine(ct_affine)
    margin_mm = 2.8
    radius_vox = np.maximum(1, np.ceil(margin_mm / voxel_sizes).astype(int))
    zz, yy, xx = np.mgrid[
        -radius_vox[0] : radius_vox[0] + 1,
        -radius_vox[1] : radius_vox[1] + 1,
        -radius_vox[2] : radius_vox[2] + 1,
    ]
    ell = (
        (zz / max(radius_vox[0], 1)) ** 2
        + (yy / max(radius_vox[1], 1)) ** 2
        + (xx / max(radius_vox[2], 1)) ** 2
    ) <= 1.0
    intracranial = ndimage.binary_dilation(intracranial, structure=ell, iterations=1)
    return intracranial.astype(bool)


def intracranial_prior_maps(intracranial_mask: np.ndarray, voxel_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    outside_dist_mm = ndimage.distance_transform_edt(~intracranial_mask, sampling=voxel_sizes).astype(np.float32)
    prior = np.ones(intracranial_mask.shape, dtype=np.float32)
    prior[~intracranial_mask] = np.exp(-0.5 * (outside_dist_mm[~intracranial_mask] / 2.4) ** 2)
    prior[outside_dist_mm > 7.5] = 0.0
    return prior, outside_dist_mm


def sample_intracranial_status(
    points_mm: np.ndarray,
    inv_affine: np.ndarray,
    intracranial_mask: np.ndarray | None,
    intracranial_dist_mm: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if intracranial_mask is None or intracranial_dist_mm is None or len(points_mm) == 0:
        return None, None
    vox = np.rint(world_to_voxel(points_mm, inv_affine)).astype(np.int64)
    for axis in range(3):
        vox[:, axis] = np.clip(vox[:, axis], 0, intracranial_mask.shape[axis] - 1)
    return intracranial_mask[vox[:, 0], vox[:, 1], vox[:, 2]], intracranial_dist_mm[vox[:, 0], vox[:, 1], vox[:, 2]]


def empty_candidate_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 3), dtype=np.float64),
        np.zeros(0, dtype=np.float64),
        np.zeros(0, dtype=np.float64),
        np.zeros(0, dtype=np.float64),
    )


def extract_localmax_candidate_arrays(
    ct_raw: np.ndarray,
    ct_affine: np.ndarray,
    blob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = ct_raw[np.isfinite(ct_raw)]
    if valid.size == 0:
        return empty_candidate_arrays()

    p95, p99_5, p99_8 = np.percentile(valid, [95.0, 99.5, 99.8])
    i_score = np.clip((ct_raw - p95) / max(float(p99_5 - p95), 1e-6), 0.0, 2.5)
    combo = i_score * np.maximum(blob, 0.15)
    combo_pos = combo[combo > 0]
    if combo_pos.size == 0:
        return empty_candidate_arrays()

    local_max = combo >= ndimage.maximum_filter(combo, size=3, mode="nearest")
    peak_base = local_max & (ct_raw >= p95)
    peak_values = combo[peak_base]
    if peak_values.size == 0:
        return empty_candidate_arrays()

    target_min_peaks = candidate_peak_target(ct_raw.shape)
    combo_thr = max(float(np.percentile(combo_pos, 99.1)), 0.12)
    for pct in (99.6, 99.4, 99.2, 99.0, 98.8, 98.6, 98.4, 98.2, 98.0):
        thr = max(float(np.percentile(combo_pos, pct)), 0.12)
        combo_thr = thr
        if np.count_nonzero(peak_values >= thr) >= target_min_peaks:
            break

    peak_mask = peak_base & (combo >= combo_thr)
    peak_vox = np.argwhere(peak_mask)
    if peak_vox.size == 0:
        return empty_candidate_arrays()

    confs = []
    refined_vox = []
    blob_scores = []
    intensity_scores = []
    for rc in peak_vox:
        lo = np.maximum(rc - 1, 0)
        hi = np.minimum(rc + 2, ct_raw.shape)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        local_combo = combo[sl]
        grid = np.stack(np.mgrid[sl[0], sl[1], sl[2]], axis=-1).reshape(-1, 3)
        weights = np.maximum(local_combo.reshape(-1), 0.0) + 1e-6
        center = np.average(grid, axis=0, weights=weights)
        refined_vox.append(center)

        local_blob = float(sample_trilinear(blob, center[None, :])[0])
        local_i = float(sample_trilinear(ct_raw, center[None, :])[0])
        b_score = np.clip(local_blob, 0.0, 2.0)
        i_score_local = np.clip((local_i - p95) / max(float(p99_8 - p95), 1e-6), 0.0, 2.0)
        confidence = 0.55 * i_score_local + 0.45 * b_score
        confs.append(confidence)
        blob_scores.append(b_score)
        intensity_scores.append(i_score_local)

    refined_vox = np.asarray(refined_vox, dtype=np.float64)
    points_mm = voxel_to_world(refined_vox, ct_affine)
    confs = np.asarray(confs, dtype=np.float64)
    blob_scores = np.asarray(blob_scores, dtype=np.float64)
    intensity_scores = np.asarray(intensity_scores, dtype=np.float64)

    return deduplicate_points_mm(
        points_mm,
        confs,
        blob_scores,
        intensity_scores,
        radius_mm=1.6,
    )


def component_axis_ratio(points_mm: np.ndarray) -> float:
    if len(points_mm) < 4:
        return 1.0
    mu = points_mm.mean(axis=0)
    _, s, _ = np.linalg.svd(points_mm - mu, full_matrices=False)
    if s.size == 0:
        return 1.0
    return float(s[0] / max(s[-1], 1e-6))


def merge_threshold_candidates(
    points_mm: np.ndarray,
    intensity_scores: np.ndarray,
    blob_scores: np.ndarray,
    strong_mask: np.ndarray,
    threshold_count: int,
    max_raw_candidates: int = 3000,
    cluster_radius_mm: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(points_mm) == 0:
        return empty_candidate_arrays()

    points_mm = np.asarray(points_mm, dtype=np.float64)
    intensity_scores = np.asarray(intensity_scores, dtype=np.float64)
    blob_scores = np.asarray(blob_scores, dtype=np.float64)
    strong_mask = np.asarray(strong_mask, dtype=bool)

    base_score = intensity_scores * np.maximum(blob_scores, 0.01)
    if len(points_mm) > max_raw_candidates:
        keep_mask = strong_mask.copy()
        n_remaining = max_raw_candidates - int(np.count_nonzero(keep_mask))
        if n_remaining > 0:
            extras = np.flatnonzero(~keep_mask)
            extra_order = extras[np.argsort(-base_score[extras])]
            keep_mask[extra_order[:n_remaining]] = True
        points_mm = points_mm[keep_mask]
        intensity_scores = intensity_scores[keep_mask]
        blob_scores = blob_scores[keep_mask]

    if len(points_mm) <= 1:
        cluster_labels = np.ones(len(points_mm), dtype=np.int64)
    else:
        cluster_labels = fcluster(
            linkage(pdist(points_mm), method="average"),
            t=cluster_radius_mm,
            criterion="distance",
        )

    merged_points = []
    merged_confidences = []
    merged_blob_scores = []
    merged_intensity_scores = []
    for cluster_id in np.unique(cluster_labels):
        members = cluster_labels == cluster_id
        pts = points_mm[members]
        eta = intensity_scores[members]
        blob = blob_scores[members]
        weights = eta * np.maximum(blob, 0.01) + 1e-6
        merged_points.append(np.average(pts, axis=0, weights=weights))
        merged_intensity = float(np.max(eta))
        merged_blob = float(np.max(blob))
        persistence = float(np.count_nonzero(members) / max(threshold_count, 1))
        merged_confidences.append(
            0.20 * persistence
            + 0.45 * float(np.clip(merged_intensity, 0.0, 2.5))
            + 0.35 * float(np.clip(merged_blob, 0.0, 2.0))
        )
        merged_blob_scores.append(float(np.clip(merged_blob, 0.0, 2.0)))
        merged_intensity_scores.append(float(np.clip(merged_intensity, 0.0, 2.5)))

    return (
        np.asarray(merged_points, dtype=np.float64),
        np.asarray(merged_confidences, dtype=np.float64),
        np.asarray(merged_blob_scores, dtype=np.float64),
        np.asarray(merged_intensity_scores, dtype=np.float64),
    )


def extract_component_candidate_arrays(
    ct_raw: np.ndarray,
    ct_affine: np.ndarray,
    surface: SurfaceMesh,
    blob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = ct_raw[np.isfinite(ct_raw) & (ct_raw > 0)]
    if valid.size == 0:
        return empty_candidate_arrays()

    p50, p99_9 = np.percentile(valid, [50.0, 99.9])
    eta = np.maximum(0.0, (ct_raw - p50) / max(float(p99_9 - p50), 1e-6))
    combo = eta * np.maximum(blob, 0.1)

    voxel_sizes = voxel_sizes_from_affine(ct_affine)
    voxel_volume_mm3 = float(np.prod(voxel_sizes))
    n_min = max(2, int(np.floor((0.75 * 0.15) / max(voxel_volume_mm3, 1e-6))))
    n_max = max(60, int(np.ceil((0.75 * 45.0) / max(voxel_volume_mm3, 1e-6))))
    thresholds = np.linspace(0.05, 0.90, 18)
    structure = np.ones((3, 3, 3), dtype=bool)

    raw_points = []
    raw_intensity_scores = []
    raw_blob_scores = []
    raw_strong_mask = []

    for thr in thresholds:
        labels, n_labels = ndimage.label(combo >= thr, structure=structure)
        if n_labels == 0:
            continue
        objects = ndimage.find_objects(labels)
        for label_id, comp_slice in enumerate(objects, start=1):
            if comp_slice is None:
                continue
            local_mask = labels[comp_slice] == label_id
            comp_vox_local = np.argwhere(local_mask)
            if comp_vox_local.size == 0:
                continue

            offset = np.array([sl.start for sl in comp_slice], dtype=np.int64)
            comp_vox = comp_vox_local + offset[None, :]
            comp_intensity = ct_raw[comp_vox[:, 0], comp_vox[:, 1], comp_vox[:, 2]].astype(np.float64)
            mean_intensity = float(np.mean(comp_intensity))
            n_vox = int(len(comp_vox))
            tiny_bright = 1 <= n_vox <= 3 and mean_intensity >= p99_9
            if not (n_min <= n_vox <= n_max or tiny_bright):
                continue

            comp_mm = voxel_to_world(comp_vox, ct_affine)
            if component_axis_ratio(comp_mm) > 8.0:
                continue

            weights = np.maximum(comp_intensity - p50, 0.0) + 1e-6
            centroid_vox = np.average(comp_vox.astype(np.float64), axis=0, weights=weights)
            centroid_mm = voxel_to_world(centroid_vox[None, :], ct_affine)[0]
            inside = bool(surface.contains_points(centroid_mm[None, :])[0])
            if not inside and mean_intensity < p99_9:
                continue

            raw_points.append(centroid_mm)
            raw_blob_scores.append(float(sample_trilinear(blob, centroid_vox[None, :])[0]))
            raw_intensity_scores.append(float(sample_trilinear(eta.astype(np.float32), centroid_vox[None, :])[0]))
            raw_strong_mask.append(mean_intensity >= p99_9)

    if not raw_points:
        return empty_candidate_arrays()

    return merge_threshold_candidates(
        points_mm=np.asarray(raw_points, dtype=np.float64),
        intensity_scores=np.asarray(raw_intensity_scores, dtype=np.float64),
        blob_scores=np.asarray(raw_blob_scores, dtype=np.float64),
        strong_mask=np.asarray(raw_strong_mask, dtype=bool),
        threshold_count=len(thresholds),
    )


def extract_candidates(
    ct_raw: np.ndarray,
    ct_affine: np.ndarray,
    surface: SurfaceMesh,
    intracranial_mask: np.ndarray | None = None,
) -> CandidateSet:
    voxel_sizes = voxel_sizes_from_affine(ct_affine)
    blob = compute_blob_response(ct_raw, voxel_sizes)

    intracranial_dist_mm = None
    valid = ct_raw[np.isfinite(ct_raw)]
    if intracranial_mask is not None:
        _, intracranial_dist_mm = intracranial_prior_maps(intracranial_mask, voxel_sizes)
    if valid.size == 0:
        return empty_candidate_set()

    target_min_peaks = candidate_peak_target(ct_raw.shape)
    points_mm, confs, blob_scores, intensity_scores = extract_component_candidate_arrays(
        ct_raw,
        ct_affine,
        surface,
        blob,
    )
    peak_points, peak_confs, peak_blob_scores, peak_intensity_scores = extract_localmax_candidate_arrays(
        ct_raw,
        ct_affine,
        blob,
    )
    if len(points_mm) == 0:
        points_mm, confs, blob_scores, intensity_scores = (
            peak_points,
            peak_confs,
            peak_blob_scores,
            peak_intensity_scores,
        )
    else:
        rescue_threshold = max(80, int(round(0.55 * target_min_peaks)))
        if len(points_mm) < rescue_threshold and len(peak_points) > 0:
            points_mm, confs, blob_scores, intensity_scores = deduplicate_points_mm(
                np.vstack([points_mm, peak_points]),
                np.concatenate([confs, peak_confs]),
                np.concatenate([blob_scores, peak_blob_scores]),
                np.concatenate([intensity_scores, peak_intensity_scores]),
                radius_mm=1.4,
            )
    if len(points_mm) == 0:
        points_mm, confs, blob_scores, intensity_scores = extract_localmax_candidate_arrays(
            ct_raw,
            ct_affine,
            blob,
        )
    if len(points_mm) == 0:
        return empty_candidate_set()

    intracranial_support = 0.0
    cand_ic_dist = None
    in_ic = None
    near_ic = None
    if intracranial_mask is not None:
        cand_vox = np.rint(world_to_voxel(points_mm, np.linalg.inv(ct_affine))).astype(np.int64)
        for axis in range(3):
            cand_vox[:, axis] = np.clip(cand_vox[:, axis], 0, ct_raw.shape[axis] - 1)
        in_ic = intracranial_mask[cand_vox[:, 0], cand_vox[:, 1], cand_vox[:, 2]]
        if intracranial_dist_mm is not None:
            cand_ic_dist = intracranial_dist_mm[cand_vox[:, 0], cand_vox[:, 1], cand_vox[:, 2]]
            near_ic = cand_ic_dist <= 2.5
            intracranial_support = float(np.mean(in_ic | near_ic))
        else:
            near_ic = in_ic.copy()
            intracranial_support = float(np.mean(in_ic))

    inside = surface.contains_points(points_mm)
    dist_to_surface = surface.nearest_vertex_distance(points_mm)
    near_radius = 6.0
    near_surface = dist_to_surface <= near_radius
    if in_ic is not None:
        inside = inside | in_ic
    if cand_ic_dist is not None:
        near_surface = near_surface | (cand_ic_dist <= 2.5)
    surface_support = float(np.mean(inside | near_surface))

    if len(dist_to_surface) >= 120 and surface_support < 0.30:
        near_radius = 10.0
        near_surface = dist_to_surface <= near_radius
        if cand_ic_dist is not None:
            near_surface = near_surface | (cand_ic_dist <= 2.5)
        surface_support = float(np.mean(inside | near_surface))
    elif len(dist_to_surface) >= 120 and surface_support < 0.40:
        near_radius = 8.0
        near_surface = dist_to_surface <= near_radius
        if cand_ic_dist is not None:
            near_surface = near_surface | (cand_ic_dist <= 2.5)
        surface_support = float(np.mean(inside | near_surface))

    hi_conf_thr = 1.05
    lo_conf_thr = 0.90
    if surface_support < 0.25:
        hi_conf_thr = 0.95
        lo_conf_thr = 0.78
    elif surface_support < 0.35:
        hi_conf_thr = 1.00
        lo_conf_thr = 0.84

    use_zone_prior = (
        cand_ic_dist is not None
        and (surface_support < 0.55 or intracranial_support < 0.55)
    )

    usable = inside | near_surface | (confs >= hi_conf_thr)
    if use_zone_prior and cand_ic_dist is not None:
        far_outside = (~inside) & (~near_surface) & (cand_ic_dist > 3.5)
        usable = usable & (~far_outside | (confs >= hi_conf_thr + 0.18))
    min_keep = min(max(160, int(round(0.75 * target_min_peaks))), len(confs))
    if np.count_nonzero(usable) < min_keep:
        usable = inside | near_surface | (confs >= lo_conf_thr)
        if use_zone_prior and cand_ic_dist is not None:
            far_outside = (~inside) & (~near_surface) & (cand_ic_dist > 5.0)
            usable = usable & (~far_outside | (confs >= lo_conf_thr + 0.12))

    points_mm = points_mm[usable]
    confs = confs[usable]
    inside = inside[usable]
    near_surface = near_surface[usable]
    if in_ic is not None:
        in_ic = in_ic[usable]
    if near_ic is not None:
        near_ic = near_ic[usable]
    if cand_ic_dist is not None:
        cand_ic_dist = cand_ic_dist[usable]
    blob_scores = blob_scores[usable]
    intensity_scores = intensity_scores[usable]
    if len(points_mm) == 0:
        return empty_candidate_set()

    max_candidates = candidate_cap(ct_raw.shape)
    if use_zone_prior and in_ic is not None and near_ic is not None and cand_ic_dist is not None:
        zone0 = in_ic
        zone1 = (~zone0) & near_ic
        zone2 = ~(zone0 | zone1)

        rescue_frac = 0.18
        if surface_support < 0.35:
            rescue_frac = 0.28
        elif surface_support < 0.45:
            rescue_frac = 0.22
        max_zone2 = min(max_candidates // 3, max(16, int(round(rescue_frac * max_candidates))))

        pri = np.zeros(len(confs), dtype=np.float64)
        pri[zone0] = 0.40
        pri[zone1] = 0.14
        pri[zone2] = -0.18 - 0.03 * np.clip(cand_ic_dist[zone2] - 3.5, 0.0, 8.0)
        order_all = np.argsort(-(confs + pri))
        keep = []
        zone2_kept = 0
        for idx in order_all:
            if zone2[idx]:
                if zone2_kept >= max_zone2:
                    continue
                zone2_kept += 1
            keep.append(int(idx))
            if len(keep) >= max_candidates:
                break
        order = np.asarray(keep, dtype=np.int64)
    else:
        order = np.argsort(-confs)
        order = order[: min(max_candidates, len(order))]

    return CandidateSet(
        points_mm=points_mm[order],
        confidences=confs[order],
        inside_mask=inside[order],
        near_surface_mask=near_surface[order],
        intracranial_inside_mask=in_ic[order] if in_ic is not None else np.zeros(len(order), dtype=bool),
        intracranial_near_mask=near_ic[order] if near_ic is not None else np.zeros(len(order), dtype=bool),
        blob_scores=blob_scores[order],
        intensity_scores=intensity_scores[order],
        surface_support_fraction=surface_support,
        intracranial_support_fraction=intracranial_support,
    )


def deduplicate_points_mm(
    points_mm: np.ndarray,
    *score_arrays: np.ndarray,
    radius_mm: float,
) -> tuple[np.ndarray, ...]:
    if len(points_mm) == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return (empty,) + tuple(np.zeros(0, dtype=np.float64) for _ in score_arrays)

    order = np.argsort(-score_arrays[0])
    points = points_mm[order]
    scores = [np.asarray(arr)[order] for arr in score_arrays]
    keep_mask = np.ones(len(points), dtype=bool)
    tree = cKDTree(points)

    for i in range(len(points)):
        if not keep_mask[i]:
            continue
        close = tree.query_ball_point(points[i], r=radius_mm)
        for j in close:
            if j <= i:
                continue
            keep_mask[j] = False

    kept_points = points[keep_mask]
    kept_scores = [arr[keep_mask] for arr in scores]
    return (kept_points,) + tuple(kept_scores)


def build_neighbor_map(points_mm: np.ndarray, min_step: float = 1.8, max_step: float = 6.0):
    tree = cKDTree(points_mm)
    pairs = tree.query_pairs(r=max_step)
    nbrs: list[list[tuple[int, float]]] = [[] for _ in range(len(points_mm))]
    for i, j in pairs:
        d = float(np.linalg.norm(points_mm[i] - points_mm[j]))
        if d < min_step:
            continue
        nbrs[i].append((j, d))
        nbrs[j].append((i, d))
    return nbrs


def propose_chains(cands: CandidateSet) -> list[ChainHypothesis]:
    pts = cands.points_mm
    conf = cands.confidences
    inside = cands.inside_mask
    near = cands.near_surface_mask
    if len(pts) < 3:
        return []

    nbrs = build_neighbor_map(pts)
    seeds = []
    pair_seeds = []

    for b, b_nbrs in enumerate(nbrs):
        if len(b_nbrs) < 2:
            for a, d in b_nbrs:
                if a < b:
                    pair_seeds.append((pair_seed_score(conf, a, b, d), (a, b)))
            continue

        nbr_ids = [x[0] for x in b_nbrs]
        for a, d in b_nbrs:
            if a < b:
                pair_seeds.append((pair_seed_score(conf, a, b, d), (a, b)))
        for i in range(len(nbr_ids) - 1):
            a = nbr_ids[i]
            va = pts[a] - pts[b]
            da = np.linalg.norm(va)
            if da < 1e-6:
                continue
            for j in range(i + 1, len(nbr_ids)):
                c = nbr_ids[j]
                vc = pts[c] - pts[b]
                dc = np.linalg.norm(vc)
                if dc < 1e-6 or abs(da - dc) > 1.5:
                    continue
                col = float(np.dot(va, vc) / (da * dc))
                if col > -0.90:
                    continue
                seed_score = conf[a] + conf[b] + conf[c] + (-col)
                seeds.append((seed_score, (a, b, c)))

    seeds.sort(key=lambda x: x[0], reverse=True)
    max_seeds = min(max(400, int(3.0 * len(pts))), 2400)
    seeds = seeds[: min(max_seeds, len(seeds))]
    pair_seeds.sort(key=lambda x: x[0], reverse=True)
    max_pair_seeds = min(max(600, int(4.0 * len(pts))), 3200)
    pair_seeds = pair_seeds[: min(max_pair_seeds, len(pair_seeds))]

    chains: list[ChainHypothesis] = []
    seen: set[tuple[int, ...]] = set()
    for _, triple in seeds:
        chain = grow_chain_from_seed(triple, pts, conf, inside, near)
        if chain is None:
            continue
        key = tuple(sorted(chain.indices))
        if key in seen:
            continue
        seen.add(key)
        chains.append(chain)

    for _, pair in pair_seeds:
        chain = grow_chain_from_pair(pair, pts, conf, inside, near)
        if chain is None:
            continue
        key = tuple(sorted(chain.indices))
        if key in seen:
            continue
        seen.add(key)
        chains.append(chain)

    chains.sort(key=lambda x: x.score, reverse=True)
    return chains


def merge_chain_lists(primary: list[ChainHypothesis], secondary: list[ChainHypothesis]) -> list[ChainHypothesis]:
    out: list[ChainHypothesis] = []
    seen: set[tuple[int, ...]] = set()
    for chain in list(primary) + list(secondary):
        key = tuple(sorted(int(i) for i in chain.indices))
        if key in seen:
            continue
        seen.add(key)
        out.append(chain)
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def filter_component_rescue_chains(
    peak_chains: list[ChainHypothesis],
    component_chains: list[ChainHypothesis],
    surface_support: float,
    max_chains: int,
) -> list[ChainHypothesis]:
    if not component_chains or max_chains <= 0:
        return []

    kept: list[ChainHypothesis] = []
    peak_refs = list(sorted(peak_chains, key=lambda x: x.score, reverse=True)[: min(48, len(peak_chains))])
    reg_thr = 0.22 if surface_support < 0.40 else 0.18
    span_thr = 10.0 if surface_support < 0.40 else 8.0

    for chain in sorted(component_chains, key=lambda x: x.score, reverse=True):
        if chain.regularity < reg_thr or chain.span_mm < span_thr:
            continue
        dup = False
        chain_idx = set(int(i) for i in chain.indices)
        axis_c, _, _, pts_c = shaft_axis(chain.ordered_points_mm)
        for ref in peak_refs + kept:
            ref_idx = set(int(i) for i in ref.indices)
            overlap_idx = len(chain_idx & ref_idx) / max(1, min(len(chain_idx), len(ref_idx)))
            axis_r, _, _, pts_r = shaft_axis(ref.ordered_points_mm)
            overlap_axis = common_interval_overlap(pts_c, pts_r, axis_c, axis_r)
            mean_nn, med_nn = shaft_contact_alignment(chain.ordered_points_mm, ref.ordered_points_mm)
            if overlap_idx >= 0.45:
                dup = True
                break
            if overlap_axis > 0.5 and med_nn < 2.4 and mean_nn < 4.2:
                dup = True
                break
        if dup:
            continue
        kept.append(chain)
        if len(kept) >= max_chains:
            break

    kept.sort(key=lambda x: x.score, reverse=True)
    return kept


def propose_component_chains(
    cands: CandidateSet,
    ct_raw: np.ndarray,
    ct_affine: np.ndarray,
    surface: SurfaceMesh,
    intracranial_mask: np.ndarray | None = None,
    intracranial_dist_mm: np.ndarray | None = None,
) -> list[ChainHypothesis]:
    if len(cands.points_mm) < 2:
        return []

    blob = compute_blob_response(ct_raw, voxel_sizes_from_affine(ct_affine))
    support_zone = np.isfinite(ct_raw)
    if intracranial_mask is not None and intracranial_dist_mm is not None:
        support_zone &= (intracranial_mask | (intracranial_dist_mm <= 2.5))
    valid = ct_raw[support_zone]
    if valid.size == 0:
        return []

    if intracranial_mask is not None and intracranial_dist_mm is not None:
        p99_2, p99_5, p99_7 = np.percentile(valid, [99.2, 99.5, 99.7])
        support_mask = (ct_raw >= p99_2) & (blob >= 0.12)
        core_mask = (ct_raw >= p99_7) | ((ct_raw >= p99_5) & (blob >= 0.70))
        allowed_zone = intracranial_mask | (intracranial_dist_mm <= 3.0)
        support_mask &= allowed_zone
        core_mask &= allowed_zone
    else:
        p99_4, p99_6, p99_8 = np.percentile(valid, [99.4, 99.6, 99.8])
        support_mask = (ct_raw >= p99_4) & (blob >= 0.14)
        core_mask = (ct_raw >= p99_8) | ((ct_raw >= p99_6) & (blob >= 0.80))
    structure = np.ones((3, 3, 3), dtype=bool)
    support_mask = ndimage.binary_dilation(support_mask, structure=structure, iterations=1)
    tube_mask = ndimage.binary_propagation(core_mask, structure=structure, mask=support_mask)
    if not np.any(tube_mask):
        return []

    labels, n_labels = ndimage.label(tube_mask, structure=structure)
    if n_labels == 0:
        return []
    objects = ndimage.find_objects(labels)
    inv_affine = np.linalg.inv(ct_affine)
    cand_vox = np.rint(world_to_voxel(cands.points_mm, inv_affine)).astype(np.int64)
    for axis in range(3):
        cand_vox[:, axis] = np.clip(cand_vox[:, axis], 0, ct_raw.shape[axis] - 1)
    cand_labels = labels[cand_vox[:, 0], cand_vox[:, 1], cand_vox[:, 2]]

    chains: list[ChainHypothesis] = []
    seen_labels: set[int] = set()
    for label_id in cand_labels:
        if label_id <= 0 or label_id in seen_labels:
            continue
        seen_labels.add(int(label_id))
        comp_slice = objects[int(label_id) - 1]
        if comp_slice is None:
            continue

        local = labels[comp_slice] == int(label_id)
        comp_vox_local = np.argwhere(local)
        if comp_vox_local.size == 0:
            continue
        offset = np.array([sl.start for sl in comp_slice], dtype=np.int64)
        comp_vox = comp_vox_local + offset[None, :]
        if len(comp_vox) < 10:
            continue

        cand_idx = np.flatnonzero(cand_labels == int(label_id))
        cand_keep = cands.inside_mask[cand_idx] | cands.near_surface_mask[cand_idx]
        if np.any(cands.intracranial_inside_mask):
            cand_keep = cand_keep | cands.intracranial_inside_mask[cand_idx] | cands.intracranial_near_mask[cand_idx]
        cand_idx = cand_idx[cand_keep]
        if cand_idx.size == 0:
            continue

        sample_step = max(1, len(comp_vox) // 800)
        comp_sample_vox = comp_vox[::sample_step]
        comp_sample_mm = voxel_to_world(comp_sample_vox, ct_affine)
        comp_inside = surface.contains_points(comp_sample_mm)
        comp_dist = surface.nearest_vertex_distance(comp_sample_mm)
        comp_support = float(np.mean(comp_inside | (comp_dist <= 3.5)))
        if intracranial_mask is not None and intracranial_dist_mm is not None:
            comp_ic_inside = intracranial_mask[comp_sample_vox[:, 0], comp_sample_vox[:, 1], comp_sample_vox[:, 2]]
            comp_ic_near = intracranial_dist_mm[comp_sample_vox[:, 0], comp_sample_vox[:, 1], comp_sample_vox[:, 2]] <= 2.5
            comp_support = float(max(comp_support, np.mean(comp_ic_inside | comp_ic_near)))
        comp_inside_frac = float(np.mean(comp_inside))
        if comp_support < 0.20 and np.count_nonzero(cands.inside_mask[cand_idx]) == 0:
            continue

        comp_mm = voxel_to_world(comp_vox[:: max(1, len(comp_vox) // 2000)], ct_affine)
        mu = comp_mm.mean(axis=0)
        _, _, vh = np.linalg.svd(comp_mm - mu, full_matrices=False)
        axis_vec = vh[0]
        t_comp = (comp_mm - mu) @ axis_vec
        span_mm = float(np.max(t_comp) - np.min(t_comp))
        if span_mm < 6.0:
            continue
        resid = comp_mm - (mu[None, :] + np.outer(t_comp, axis_vec))
        rad95 = float(np.percentile(np.linalg.norm(resid, axis=1), 95))
        if rad95 > 2.8:
            continue

        t_lo = float(np.percentile(t_comp, 4.0))
        t_hi = float(np.percentile(t_comp, 96.0))
        comp_endpoints = np.vstack([mu + axis_vec * t_lo, mu + axis_vec * t_hi])

        pts = cands.points_mm[cand_idx]
        conf = cands.confidences[cand_idx]
        t_pts = (pts - mu) @ axis_vec
        order = np.argsort(t_pts)
        pts = pts[order]
        conf = conf[order]
        t_pts = t_pts[order]
        cand_resid = pts - (mu[None, :] + np.outer(t_pts, axis_vec))
        cand_rad95 = float(np.percentile(np.linalg.norm(cand_resid, axis=1), 95))
        if cand_rad95 > 2.3:
            continue

        bridge_pts = [pts]
        bridge_scores = [conf]
        if t_pts[0] - t_lo > 2.0:
            bridge_pts.append(comp_endpoints[:1])
            bridge_scores.append(np.array([np.max(conf) * 0.85], dtype=np.float64))
        if t_hi - t_pts[-1] > 2.0:
            bridge_pts.append(comp_endpoints[1:])
            bridge_scores.append(np.array([np.max(conf) * 0.85], dtype=np.float64))

        path_pts = np.vstack(bridge_pts)
        path_scores = np.concatenate(bridge_scores)
        t_path = (path_pts - mu) @ axis_vec
        path_order = np.argsort(t_path)
        path_pts = path_pts[path_order]
        path_scores = path_scores[path_order]
        path_pts = deduplicate_polyline_points(path_pts, min_step_mm=1.2)
        if len(path_pts) < 3:
            continue

        spacing_guess = estimate_spacing(pts) if len(pts) >= 2 else 3.5
        spacing_guess = float(np.clip(spacing_guess, 2.6, 4.6))
        reg_obs = spacing_regularity(pts, spacing_guess) if len(pts) >= 3 else 0.0
        reg_path = spacing_regularity(path_pts, spacing_guess) if len(path_pts) >= 3 else 0.0
        regularity = float(max(reg_obs, 0.45 * reg_path, min(0.36, 0.16 + 0.009 * span_mm)))
        brain_support = float(
            max(
                np.mean(cands.inside_mask[cand_idx] | cands.near_surface_mask[cand_idx]),
                comp_support,
            )
        )
        inside_frac = float(max(np.mean(cands.inside_mask[cand_idx]), comp_inside_frac))
        near_frac = float(max(np.mean(cands.near_surface_mask[cand_idx]), comp_support))
        score = float(
            0.70 * np.mean(conf)
            + 0.04 * span_mm
            + 0.18 * len(path_pts)
            + 0.45 * brain_support
            + 0.20 * regularity
        )
        chains.append(
            ChainHypothesis(
                indices=tuple(int(i) for i in cand_idx.tolist()),
                ordered_points_mm=path_pts,
                ordered_scores=np.full(len(path_pts), np.mean(path_scores), dtype=np.float64),
                spacing_mm=spacing_guess,
                span_mm=span_mm,
                regularity=regularity,
                brain_support=brain_support,
                inside_fraction=inside_frac,
                near_fraction=near_frac,
                score=score,
            )
        )

    chains.sort(key=lambda x: x.score, reverse=True)
    return chains


def pair_seed_score(confidences: np.ndarray, a: int, b: int, d: float) -> float:
    return float(confidences[a] + confidences[b] - 0.25 * abs(d - 3.5))


def grow_chain_from_seed(
    triple: tuple[int, int, int],
    points_mm: np.ndarray,
    confidences: np.ndarray,
    inside_mask: np.ndarray,
    near_mask: np.ndarray,
) -> ChainHypothesis | None:
    seed_pts = points_mm[list(triple)]
    mu = seed_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(seed_pts - mu, full_matrices=False)
    axis = vh[0]
    order = np.argsort((seed_pts - mu) @ axis)
    ordered = [triple[i] for i in order]

    spacing = estimate_spacing(points_mm[ordered])
    if not (2.0 <= spacing <= 5.5):
        spacing = 3.5

    chain = extend_chain(ordered, points_mm, confidences, spacing, forward=True)
    chain = extend_chain(chain, points_mm, confidences, spacing, forward=False)
    chain = remove_duplicate_indices(chain)
    if len(chain) < 3:
        return None

    ordered_pts = points_mm[chain]
    ordered_scores = confidences[chain]
    spacing = estimate_spacing(ordered_pts)
    regularity = spacing_regularity(ordered_pts, spacing)
    span = polyline_length(ordered_pts)
    inside_frac = float(np.mean(inside_mask[chain]))
    near_frac = float(np.mean(near_mask[chain]))
    brain_support = float(np.mean(inside_mask[chain] | near_mask[chain]))
    if regularity < 0.22 or span < 6.0:
        return None

    score = (
        1.7 * regularity
        + 0.8 * np.mean(ordered_scores)
        + 0.24 * len(chain)
        + 0.06 * span
        + 0.45 * brain_support
        + 0.12 * inside_frac
    )
    return ChainHypothesis(
        indices=tuple(chain),
        ordered_points_mm=ordered_pts,
        ordered_scores=ordered_scores,
        spacing_mm=float(np.clip(spacing, 2.4, 5.0)),
        span_mm=float(span),
        regularity=float(regularity),
        brain_support=brain_support,
        inside_fraction=inside_frac,
        near_fraction=near_frac,
        score=float(score),
    )


def grow_chain_from_pair(
    pair: tuple[int, int],
    points_mm: np.ndarray,
    confidences: np.ndarray,
    inside_mask: np.ndarray,
    near_mask: np.ndarray,
) -> ChainHypothesis | None:
    a, b = pair
    d = float(np.linalg.norm(points_mm[b] - points_mm[a]))
    if not (1.8 <= d <= 6.2):
        return None

    chain = [a, b]
    chain = extend_chain(chain, points_mm, confidences, d, forward=True)
    chain = extend_chain(chain, points_mm, confidences, d, forward=False)
    chain = remove_duplicate_indices(chain)
    if len(chain) < 3:
        return None

    ordered_pts = points_mm[chain]
    ordered_scores = confidences[chain]
    spacing = estimate_spacing(ordered_pts)
    regularity = spacing_regularity(ordered_pts, spacing)
    span = polyline_length(ordered_pts)
    inside_frac = float(np.mean(inside_mask[chain]))
    near_frac = float(np.mean(near_mask[chain]))
    brain_support = float(np.mean(inside_mask[chain] | near_mask[chain]))
    if regularity < 0.18 or span < 6.0:
        return None

    score = (
        1.3 * regularity
        + 0.8 * np.mean(ordered_scores)
        + 0.22 * len(chain)
        + 0.05 * span
        + 0.40 * brain_support
        + 0.10 * inside_frac
    )
    return ChainHypothesis(
        indices=tuple(chain),
        ordered_points_mm=ordered_pts,
        ordered_scores=ordered_scores,
        spacing_mm=float(np.clip(spacing, 2.4, 5.0)),
        span_mm=float(span),
        regularity=float(regularity),
        brain_support=brain_support,
        inside_fraction=inside_frac,
        near_fraction=near_frac,
        score=float(score),
    )


def extend_chain(
    chain: list[int],
    points_mm: np.ndarray,
    confidences: np.ndarray,
    spacing_mm: float,
    forward: bool,
) -> list[int]:
    max_len = 22
    chain = list(chain)

    while len(chain) < max_len:
        work = chain if forward else list(reversed(chain))
        pts = points_mm[work]
        tangent = pts[-1] - pts[-3] if len(pts) >= 3 else pts[-1] - pts[-2]
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            break
        tangent = tangent / norm
        tip = pts[-1]

        best_idx = None
        best_score = -np.inf
        for idx, cand_pt in enumerate(points_mm):
            if idx in chain:
                continue
            step_vec = cand_pt - tip
            step = np.linalg.norm(step_vec)
            if step < 1.6 or step > 6.2:
                continue
            align = float(np.dot(step_vec / step, tangent))
            if align < 0.78:
                continue
            pred = tip + tangent * spacing_mm
            pred_err = float(np.linalg.norm(cand_pt - pred))
            score = (
                1.8 * confidences[idx]
                + 1.2 * align
                - 0.55 * pred_err
                - 0.18 * abs(step - spacing_mm)
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None or best_score < 0.45:
            break

        if forward:
            chain.append(best_idx)
        else:
            chain.insert(0, best_idx)

        spacing_mm = estimate_spacing(points_mm[chain])

    return chain


def remove_duplicate_indices(indices: Iterable[int]) -> list[int]:
    out = []
    seen = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def estimate_spacing(points_mm: np.ndarray) -> float:
    if len(points_mm) < 2:
        return 3.5
    t = project_onto_axis(points_mm)
    gaps = np.diff(np.sort(t))
    gaps = gaps[(gaps > 1.5) & (gaps < 6.5)]
    if gaps.size == 0:
        return 3.5
    return float(np.median(gaps))


def spacing_regularity(points_mm: np.ndarray, spacing_mm: float) -> float:
    if len(points_mm) < 3:
        return 0.0
    t = np.sort(project_onto_axis(points_mm))
    gaps = np.diff(t)
    gaps = gaps[(gaps > 1.2) & (gaps < 7.0)]
    if gaps.size == 0:
        return 0.0
    err = np.abs(gaps - spacing_mm)
    return float(np.mean(np.exp(-((err / 1.0) ** 2))))


def project_onto_axis(points_mm: np.ndarray) -> np.ndarray:
    mu = points_mm.mean(axis=0)
    _, _, vh = np.linalg.svd(points_mm - mu, full_matrices=False)
    return (points_mm - mu) @ vh[0]


def polyline_length(points_mm: np.ndarray) -> float:
    if len(points_mm) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points_mm, axis=0), axis=1)))


def select_and_fit_shafts(
    chains: list[ChainHypothesis],
    cands: CandidateSet,
    ct_raw: np.ndarray,
    ct_affine: np.ndarray,
    surface: SurfaceMesh,
    intracranial_mask: np.ndarray | None = None,
    intracranial_dist_mm: np.ndarray | None = None,
) -> list[FittedShaft]:
    if not chains:
        return []

    ct_blob = compute_blob_response(ct_raw, voxel_sizes_from_affine(ct_affine))
    surface_support = cands.surface_support_fraction
    candidate_chains = chains[: min(len(chains), 140)]

    fits: list[FittedShaft] = []
    for chain in candidate_chains:
        fitted = fit_contacts_to_chain_auto(
            chain=chain,
            ct_raw=ct_raw,
            ct_blob=ct_blob,
            ct_affine=ct_affine,
            surface=surface,
            surface_support=surface_support,
            intracranial_mask=intracranial_mask,
            intracranial_dist_mm=intracranial_dist_mm,
        )
        if fitted is not None:
            fits.append(fitted)

    if not fits:
        return []

    scores = np.asarray([s.score for s in fits], dtype=np.float64)
    score_thr = max(2.20, float(np.percentile(scores, 40)) - 0.35)
    if surface_support < 0.35:
        score_thr -= 0.20
    elif surface_support < 0.45:
        score_thr -= 0.10

    min_support = 0.34
    min_evidence = 0.34
    min_brain_support = 0.30
    max_outside_fraction = 0.45
    brain_evidence_bypass = 0.75
    if surface_support < 0.25:
        min_brain_support = 0.06
        max_outside_fraction = 0.78
        brain_evidence_bypass = 0.55
    elif surface_support < 0.35:
        min_brain_support = 0.12
        max_outside_fraction = 0.68
        brain_evidence_bypass = 0.60
    elif surface_support < 0.45:
        min_brain_support = 0.20
        max_outside_fraction = 0.56
        brain_evidence_bypass = 0.68

    fits.sort(key=lambda x: x.score, reverse=True)
    selected: list[FittedShaft] = []
    selected_idx: set[int] = set()
    used_candidates: set[int] = set()
    max_total_contacts = max(320, 2 * len(cands.points_mm))
    total_contacts = 0

    for idx, shaft in enumerate(fits):
        overlap = len(set(shaft.seed_indices) & used_candidates) / max(len(shaft.seed_indices), 1)
        if overlap > 0.58:
            continue
        if shaft.score < score_thr:
            continue
        if shaft.support_score < min_support or shaft.evidence_score < min_evidence:
            continue
        if shaft.brain_support < min_brain_support and shaft.evidence_score < brain_evidence_bypass:
            continue
        if shaft.outside_fraction > max_outside_fraction and shaft.evidence_score < 0.95:
            continue
        if is_duplicate_shaft(selected, shaft):
            continue
        if total_contacts + len(shaft.contacts_mm) > max_total_contacts:
            continue

        selected.append(shaft)
        selected_idx.add(idx)
        used_candidates.update(shaft.seed_indices)
        total_contacts += len(shaft.contacts_mm)

    relaxed_score_thr = score_thr
    extra_slots = 0
    if surface_support >= 0.85:
        relaxed_score_thr = max(4.85, score_thr - 0.90)
        extra_slots = 5
    elif surface_support >= 0.70:
        relaxed_score_thr = max(5.00, score_thr - 0.60)
        extra_slots = 3
    elif surface_support >= 0.55:
        relaxed_score_thr = max(5.10, score_thr - 0.45)
        extra_slots = 2

    if extra_slots > 0:
        for idx, shaft in enumerate(fits):
            if idx in selected_idx:
                continue
            overlap = len(set(shaft.seed_indices) & used_candidates) / max(len(shaft.seed_indices), 1)
            if overlap > 0.58:
                continue
            if shaft.score < relaxed_score_thr:
                continue
            if shaft.support_score < min_support or shaft.evidence_score < min_evidence:
                continue
            if shaft.brain_support < min_brain_support and shaft.evidence_score < brain_evidence_bypass:
                continue
            if shaft.outside_fraction > max_outside_fraction and shaft.evidence_score < 0.95:
                continue
            if shaft.inferred_contacts < 6 and shaft.evidence_score < 0.95:
                continue
            if is_duplicate_shaft(selected, shaft):
                continue
            if total_contacts + len(shaft.contacts_mm) > max_total_contacts:
                continue

            selected.append(shaft)
            selected_idx.add(idx)
            used_candidates.update(shaft.seed_indices)
            total_contacts += len(shaft.contacts_mm)
            extra_slots -= 1
            if extra_slots <= 0:
                break

    if surface_support < 0.55:
        selected = merge_collinear_shafts(selected, strict=False)
        selected = consolidate_shaft_groups(selected, ct_raw, ct_blob, ct_affine, surface, surface_support, strict=False)
    selected = inflate_selected_shafts(
        selected,
        ct_raw,
        ct_blob,
        ct_affine,
        surface,
        surface_support,
        intracranial_mask=intracranial_mask,
        intracranial_dist_mm=intracranial_dist_mm,
    )
    strict_post_merge = surface_support >= 0.60
    selected = merge_collinear_shafts(selected, strict=strict_post_merge)
    selected = consolidate_shaft_groups(selected, ct_raw, ct_blob, ct_affine, surface, surface_support, strict=strict_post_merge)
    selected = drop_redundant_fragments(selected)
    pruned: list[FittedShaft] = []
    for shaft in selected:
        span_mm = polyline_length(shaft.contacts_mm)
        if shaft.score < 5.6 and shaft.support_score < 0.55 and shaft.evidence_score < 0.90:
            continue
        if shaft.inferred_contacts <= 2 and (span_mm < 8.0 or shaft.evidence_score < 0.90):
            continue
        if shaft.inferred_contacts <= 3 and span_mm < 10.0 and shaft.score < score_thr + 0.35:
            continue
        if shaft.inferred_contacts <= 4 and span_mm < 12.0 and shaft.brain_support < 0.18 and shaft.evidence_score < 0.75:
            continue
        pruned.append(shaft)

    pruned.sort(key=lambda x: x.score, reverse=True)
    return pruned


def fit_contacts_to_chain_auto(
    chain: ChainHypothesis,
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    ct_affine: np.ndarray,
    surface: SurfaceMesh,
    surface_support: float,
    intracranial_mask: np.ndarray | None = None,
    intracranial_dist_mm: np.ndarray | None = None,
) -> FittedShaft | None:
    pts = chain.ordered_points_mm
    obs_scores = chain.ordered_scores
    obs_s = arc_lengths(pts)
    spacing_guess = float(np.clip(chain.spacing_mm, 2.6, 4.6))
    span = max(float(obs_s[-1]), spacing_guess)
    obs_count = len(chain.indices)
    base_count = max(3, int(round(span / max(spacing_guess, 1e-3))) + 1)
    gap_count = int(np.count_nonzero(np.diff(obs_s) > 1.45 * spacing_guess)) if len(obs_s) > 1 else 0
    count_min = max(3, min(obs_count, base_count) - 1)
    count_max = min(24, max(obs_count + gap_count + 4, base_count + 8, int(np.ceil((span + 18.0) / 2.6)) + 1))
    count_grid = np.arange(count_min, count_max + 1, dtype=np.int64)
    if count_grid.size == 0:
        count_grid = np.arange(max(3, obs_count), max(3, obs_count) + 3, dtype=np.int64)

    inv_affine = np.linalg.inv(ct_affine)
    p95, p998 = np.percentile(ct_raw[np.isfinite(ct_raw)], [95.0, 99.8])
    spacing_grid = np.arange(max(2.5, spacing_guess - 0.6), min(4.9, spacing_guess + 0.6) + 1e-6, 0.15)

    best: tuple[float, float, float, float, int, np.ndarray, np.ndarray] | None = None
    for spacing_mm in spacing_grid:
        for n_contacts in count_grid:
            nominal_len = spacing_mm * max(int(n_contacts) - 1, 1)
            start_min = min(float(obs_s[0]), span - nominal_len) - min(3.0, spacing_mm)
            start_max = max(float(obs_s[-1]) - nominal_len, 0.0) + min(4.5, 1.5 * spacing_mm)
            for s0 in np.arange(start_min, start_max + 1e-6, 0.5):
                grid_s = s0 + spacing_mm * np.arange(int(n_contacts), dtype=np.float64)
                grid_pts = sample_polyline_with_extrapolation(pts, grid_s)
                vox = world_to_voxel(grid_pts, inv_affine)
                ct_vals = sample_trilinear(ct_raw, vox)
                blob_vals = sample_trilinear(ct_blob, vox)
                ct_score = np.clip((ct_vals - p95) / max(p998 - p95, 1e-6), 0.0, 2.0)
                blob_score = np.clip(blob_vals, 0.0, 2.0)
                evidence_vec = 0.60 * ct_score + 0.40 * blob_score

                d_obs = np.abs(grid_s[:, None] - obs_s[None, :])
                support_obs = float(np.mean(np.exp(-((d_obs.min(axis=0) / 1.10) ** 2))))
                support_grid = float(np.mean(np.exp(-((d_obs.min(axis=1) / 1.35) ** 2))))
                support = 0.55 * support_obs + 0.45 * support_grid
                evidence = float(np.mean(evidence_vec))
                peak_fraction = float(np.mean(evidence_vec >= 0.50))
                extrap_mm = max(0.0, -float(np.min(grid_s))) + max(0.0, float(np.max(grid_s)) - float(obs_s[-1]))
                extrap_penalty = extrap_mm / max(nominal_len, spacing_mm)
                inflate_penalty = max(0.0, float(n_contacts - base_count - 2))

                score = (
                    1.70 * support_obs
                    + 0.40 * support_grid
                    + 1.80 * evidence
                    + 0.90 * peak_fraction
                    + 1.00 * chain.regularity
                    + 0.22 * chain.brain_support
                    + 0.12 * float(np.mean(obs_scores))
                    + 0.04 * min(int(n_contacts), 12)
                    - 0.08 * abs(spacing_mm - spacing_guess)
                    - 0.22 * extrap_penalty
                    - 0.06 * inflate_penalty
                )
                if best is None or score > best[0]:
                    best = (score, support, evidence, spacing_mm, int(n_contacts), grid_s, grid_pts)

    if best is None:
        return None

    support_dist, outside_dist, surface_weight, base_ev_thr, strong_ev_thr = surface_distance_params(surface_support)

    score, support, evidence, spacing_mm, n_contacts, grid_s, grid_pts = best
    max_extra_total = 1 if surface_support >= 0.65 else 2
    if evidence >= 0.95 and support >= 0.55:
        max_extra_total += 1
    if evidence >= 1.15 and support >= 0.70:
        max_extra_total += 2
    if chain.regularity >= 0.45 and obs_count >= 5:
        max_extra_total += 1
    if obs_count >= 8:
        max_extra_total += 1
    max_extra_cap = 6
    if surface_support >= 0.50:
        max_extra_cap = 2
    elif surface_support >= 0.40:
        max_extra_cap = 3
    max_extra_total = int(np.clip(max_extra_total, 1, max_extra_cap))
    ev_relax = 0.0
    strong_relax = 0.0
    if surface_support < 0.40 and evidence >= 1.15:
        ev_relax = 0.03
    if surface_support < 0.35 and evidence >= 1.25:
        strong_relax = 0.04
    grid_s, grid_pts = extend_grid_by_evidence(
        polyline_pts=pts,
        grid_s=grid_s,
        spacing_mm=spacing_mm,
        ct_raw=ct_raw,
        ct_blob=ct_blob,
        inv_affine=inv_affine,
        p95=p95,
        p998=p998,
        surface=surface,
        support_dist=support_dist,
        outside_dist=outside_dist,
        intracranial_mask=intracranial_mask,
        intracranial_dist_mm=intracranial_dist_mm,
        max_extra_total=max_extra_total,
        base_ev_thr=max(0.52, base_ev_thr - ev_relax),
        strong_ev_thr=max(base_ev_thr + 0.18, strong_ev_thr - strong_relax),
    )
    n_contacts = int(len(grid_s))
    evidence_vals = contact_evidence_values(ct_raw, ct_blob, inv_affine, p95, p998, grid_pts)
    evidence = float(np.mean(evidence_vals))
    peak_fraction = float(np.mean(evidence_vals >= 0.50))
    score = score + 0.10 * min(max(n_contacts - best[4], 0), 6) + 0.20 * peak_fraction

    inside = surface.contains_points(grid_pts)
    dist = surface.nearest_vertex_distance(grid_pts)
    brain_mask = inside | (dist <= support_dist)
    ic_inside, ic_dist = sample_intracranial_status(grid_pts, inv_affine, intracranial_mask, intracranial_dist_mm)
    if ic_inside is not None and ic_dist is not None:
        brain_mask = brain_mask | ic_inside | (ic_dist <= 2.5)
        outside_fraction = float(np.mean((~inside) & (~ic_inside) & (dist > outside_dist) & (ic_dist > 4.0)))
    else:
        outside_fraction = float(np.mean((~inside) & (dist > outside_dist)))
    brain_support = float(np.mean(brain_mask))
    score = score + surface_weight * (0.35 * brain_support - 0.55 * outside_fraction)

    if support < 0.30 or evidence < 0.30:
        return None
    if surface_support >= 0.35 and outside_fraction > 0.60 and evidence < 0.80:
        return None

    return FittedShaft(
        inferred_contacts=n_contacts,
        observed_contacts=len(chain.indices),
        score=float(score),
        support_score=float(support),
        evidence_score=float(evidence),
        spacing_mm=float(spacing_mm),
        brain_support=float(brain_support),
        outside_fraction=float(outside_fraction),
        contacts_mm=grid_pts,
        seed_indices=chain.indices,
    )


def is_duplicate_shaft(selected: list[FittedShaft], shaft: FittedShaft) -> bool:
    if not selected or len(shaft.contacts_mm) == 0:
        return False
    for other in selected:
        d = np.linalg.norm(shaft.contacts_mm[:, None, :] - other.contacts_mm[None, :, :], axis=2)
        frac_a = float(np.mean(np.min(d, axis=1) < 1.6))
        frac_b = float(np.mean(np.min(d, axis=0) < 1.6))
        if frac_a > 0.60 or frac_b > 0.60:
            return True
    return False


def contact_evidence_values(
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    inv_affine: np.ndarray,
    p95: float,
    p998: float,
    points_mm: np.ndarray,
) -> np.ndarray:
    vox = world_to_voxel(points_mm, inv_affine)
    ct_vals = sample_trilinear(ct_raw, vox)
    blob_vals = sample_trilinear(ct_blob, vox)
    ct_score = np.clip((ct_vals - p95) / max(p998 - p95, 1e-6), 0.0, 2.0)
    blob_score = np.clip(blob_vals, 0.0, 2.0)
    return 0.60 * ct_score + 0.40 * blob_score


def extend_grid_by_evidence(
    polyline_pts: np.ndarray,
    grid_s: np.ndarray,
    spacing_mm: float,
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    inv_affine: np.ndarray,
    p95: float,
    p998: float,
    surface: SurfaceMesh,
    support_dist: float,
    outside_dist: float,
    max_extra_total: int,
    base_ev_thr: float,
    strong_ev_thr: float,
    intracranial_mask: np.ndarray | None = None,
    intracranial_dist_mm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    out_s = np.asarray(grid_s, dtype=np.float64).copy()
    out_pts = sample_polyline_with_extrapolation(polyline_pts, out_s)
    for _ in range(max_extra_total):
        left_s = float(out_s[0] - spacing_mm)
        left_pt = sample_polyline_with_extrapolation(polyline_pts, np.array([left_s], dtype=np.float64))[0]
        left_ev = float(contact_evidence_values(ct_raw, ct_blob, inv_affine, p95, p998, left_pt[None, :])[0])
        left_in = bool(surface.contains_points(left_pt[None, :])[0])
        left_dist = float(surface.nearest_vertex_distance(left_pt[None, :])[0])
        left_ic_inside, left_ic_dist = sample_intracranial_status(left_pt[None, :], inv_affine, intracranial_mask, intracranial_dist_mm)
        left_ic_in = bool(left_ic_inside[0]) if left_ic_inside is not None else False
        left_ic_mm = float(left_ic_dist[0]) if left_ic_dist is not None else np.inf
        left_brain = left_in or left_dist <= support_dist or left_ic_in or left_ic_mm <= 2.5
        left_ok = left_ev >= base_ev_thr and (left_brain or left_ev >= strong_ev_thr)
        left_bad = (not left_brain) and left_dist > outside_dist and left_ic_mm > 4.0 and left_ev < strong_ev_thr

        right_s = float(out_s[-1] + spacing_mm)
        right_pt = sample_polyline_with_extrapolation(polyline_pts, np.array([right_s], dtype=np.float64))[0]
        right_ev = float(contact_evidence_values(ct_raw, ct_blob, inv_affine, p95, p998, right_pt[None, :])[0])
        right_in = bool(surface.contains_points(right_pt[None, :])[0])
        right_dist = float(surface.nearest_vertex_distance(right_pt[None, :])[0])
        right_ic_inside, right_ic_dist = sample_intracranial_status(right_pt[None, :], inv_affine, intracranial_mask, intracranial_dist_mm)
        right_ic_in = bool(right_ic_inside[0]) if right_ic_inside is not None else False
        right_ic_mm = float(right_ic_dist[0]) if right_ic_dist is not None else np.inf
        right_brain = right_in or right_dist <= support_dist or right_ic_in or right_ic_mm <= 2.5
        right_ok = right_ev >= base_ev_thr and (right_brain or right_ev >= strong_ev_thr)
        right_bad = (not right_brain) and right_dist > outside_dist and right_ic_mm > 4.0 and right_ev < strong_ev_thr

        options: list[tuple[str, float, float, np.ndarray]] = []
        if left_ok and not left_bad:
            options.append(("left", left_s, left_ev, left_pt))
        if right_ok and not right_bad:
            options.append(("right", right_s, right_ev, right_pt))
        if not options:
            break
        side, s_new, _, pt_new = max(options, key=lambda item: item[2])
        if side == "left":
            out_s = np.r_[s_new, out_s]
            out_pts = np.vstack([pt_new, out_pts])
        else:
            out_s = np.r_[out_s, s_new]
            out_pts = np.vstack([out_pts, pt_new])
    return out_s, out_pts


def inflate_selected_shafts(
    shafts: list[FittedShaft],
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    ct_affine: np.ndarray,
    surface: SurfaceMesh,
    surface_support: float,
    intracranial_mask: np.ndarray | None = None,
    intracranial_dist_mm: np.ndarray | None = None,
) -> list[FittedShaft]:
    if not shafts:
        return shafts

    inv_affine = np.linalg.inv(ct_affine)
    p95, p998 = np.percentile(ct_raw[np.isfinite(ct_raw)], [95.0, 99.8])
    support_dist, outside_dist, surface_weight, base_ev_thr, strong_ev_thr = surface_distance_params(surface_support)
    out: list[FittedShaft] = []

    for shaft in shafts:
        if surface_support >= 0.50:
            clean_high_support = (
                shaft.inferred_contacts <= 12
                and shaft.brain_support >= 0.90
                and shaft.outside_fraction <= 0.10
            )
            if not clean_high_support:
                out.append(shaft)
                continue

        max_extra_total = 0
        if shaft.inferred_contacts >= 5:
            max_extra_total = 1
            if shaft.evidence_score >= 1.05:
                max_extra_total += 2
            if shaft.evidence_score >= 1.25:
                max_extra_total += 1
            if shaft.support_score >= 0.72:
                max_extra_total += 1
            if shaft.brain_support >= 0.28:
                max_extra_total += 1
            if shaft.inferred_contacts >= 8:
                max_extra_total += 1
            if surface_support < 0.45:
                max_extra_total += 1
        max_extra_cap = 8
        if surface_support >= 0.50:
            max_extra_cap = 1
        elif surface_support >= 0.42:
            max_extra_cap = 2
        elif surface_support >= 0.35:
            max_extra_cap = 5
        max_extra_total = int(np.clip(max_extra_total, 0, max_extra_cap))
        if max_extra_total == 0:
            out.append(shaft)
            continue

        grid_s = np.arange(len(shaft.contacts_mm), dtype=np.float64) * shaft.spacing_mm
        ev_relax = 0.0
        strong_relax = 0.0
        if surface_support < 0.42 and shaft.evidence_score >= 1.15:
            ev_relax += 0.06
        if surface_support < 0.35 and shaft.support_score >= 0.82:
            ev_relax += 0.03
        if surface_support < 0.35 and shaft.evidence_score >= 1.30:
            strong_relax += 0.06
        tuned_base_ev = max(0.50, base_ev_thr - ev_relax)
        tuned_strong_ev = max(tuned_base_ev + 0.16, strong_ev_thr - strong_relax)
        grid_s_new, grid_pts_new = extend_grid_by_evidence(
            polyline_pts=shaft.contacts_mm,
            grid_s=grid_s,
            spacing_mm=shaft.spacing_mm,
            ct_raw=ct_raw,
            ct_blob=ct_blob,
            inv_affine=inv_affine,
            p95=p95,
            p998=p998,
            surface=surface,
            support_dist=support_dist,
            outside_dist=outside_dist,
            intracranial_mask=intracranial_mask,
            intracranial_dist_mm=intracranial_dist_mm,
            max_extra_total=max_extra_total,
            base_ev_thr=tuned_base_ev,
            strong_ev_thr=tuned_strong_ev,
        )
        if len(grid_pts_new) <= len(shaft.contacts_mm):
            out.append(shaft)
            continue
        out.append(
            rebuild_shaft_from_contacts(
                shaft=shaft,
                contacts_mm=grid_pts_new,
                ct_raw=ct_raw,
                ct_blob=ct_blob,
                inv_affine=inv_affine,
                p95=p95,
                p998=p998,
                surface=surface,
                support_dist=support_dist,
                outside_dist=outside_dist,
                surface_weight=surface_weight,
                intracranial_mask=intracranial_mask,
                intracranial_dist_mm=intracranial_dist_mm,
            )
        )

    out.sort(key=lambda x: x.score, reverse=True)
    return out


def rebuild_shaft_from_contacts(
    shaft: FittedShaft,
    contacts_mm: np.ndarray,
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    inv_affine: np.ndarray,
    p95: float,
    p998: float,
    surface: SurfaceMesh,
    support_dist: float,
    outside_dist: float,
    surface_weight: float,
    intracranial_mask: np.ndarray | None = None,
    intracranial_dist_mm: np.ndarray | None = None,
) -> FittedShaft:
    axis, mu, t, ordered_pts = shaft_axis(contacts_mm)
    del axis, mu
    ordered_pts = deduplicate_contacts_along_axis(ordered_pts, t, min_spacing_mm=1.5)
    spacing_mm = float(np.clip(estimate_spacing(ordered_pts), 2.4, 5.0))
    ordered_pts = fill_small_contact_gaps(ordered_pts)
    evidence_vals = contact_evidence_values(ct_raw, ct_blob, inv_affine, p95, p998, ordered_pts)
    evidence_score = float(np.mean(evidence_vals))
    inside = surface.contains_points(ordered_pts)
    dist = surface.nearest_vertex_distance(ordered_pts)
    brain_mask = inside | (dist <= support_dist)
    ic_inside, ic_dist = sample_intracranial_status(ordered_pts, inv_affine, intracranial_mask, intracranial_dist_mm)
    if ic_inside is not None and ic_dist is not None:
        brain_mask = brain_mask | ic_inside | (ic_dist <= 2.5)
        outside_fraction = float(np.mean((~inside) & (~ic_inside) & (dist > outside_dist) & (ic_dist > 4.0)))
    else:
        outside_fraction = float(np.mean((~inside) & (dist > outside_dist)))
    brain_support = float(np.mean(brain_mask))
    regularity = spacing_regularity(ordered_pts, spacing_mm)
    added_contacts = max(0, len(ordered_pts) - len(shaft.contacts_mm))
    support_score = float(np.clip(0.70 * shaft.support_score + 0.30 * regularity, 0.0, 1.0))
    score = (
        shaft.score
        + 0.18 * added_contacts
        + 0.30 * (evidence_score - shaft.evidence_score)
        + 0.20 * (support_score - shaft.support_score)
        + surface_weight * (0.25 * (brain_support - shaft.brain_support) - 0.35 * (outside_fraction - shaft.outside_fraction))
    )
    return FittedShaft(
        inferred_contacts=int(len(ordered_pts)),
        observed_contacts=shaft.observed_contacts,
        score=float(score),
        support_score=float(support_score),
        evidence_score=float(evidence_score),
        spacing_mm=float(spacing_mm),
        brain_support=float(brain_support),
        outside_fraction=float(outside_fraction),
        contacts_mm=ordered_pts,
        seed_indices=shaft.seed_indices,
    )


def bridge_intracranial_shaft_gaps(
    shafts: list[FittedShaft],
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    ct_affine: np.ndarray,
    surface: SurfaceMesh,
    surface_support: float,
) -> list[FittedShaft]:
    if not shafts:
        return shafts

    inv_affine = np.linalg.inv(ct_affine)
    p95, p998 = np.percentile(ct_raw[np.isfinite(ct_raw)], [95.0, 99.8])
    support_dist, outside_dist, _, base_ev_thr, _ = surface_distance_params(surface_support)
    bridged = [
        fill_evidence_supported_gaps(
            shaft,
            ct_raw,
            ct_blob,
            inv_affine,
            p95,
            p998,
            surface,
            support_dist,
            outside_dist,
        )
        for shaft in shafts
    ]

    changed = True
    while changed:
        changed = False
        for i in range(len(bridged) - 1):
            for j in range(i + 1, len(bridged)):
                if not should_bridge_neighbor_shafts(bridged[i], bridged[j]):
                    continue
                bridge_score = bridge_segment_support(
                    bridged[i],
                    bridged[j],
                    ct_raw,
                    ct_blob,
                    inv_affine,
                    p95,
                    p998,
                    surface,
                    support_dist,
                    outside_dist,
                )
                if bridge_score is None:
                    continue
                bridge_support, bridge_evidence = bridge_score
                if bridge_support < 0.58 or bridge_evidence < max(0.42, base_ev_thr - 0.10):
                    continue
                merged = combine_shafts(bridged[i], bridged[j])
                merged = fill_evidence_supported_gaps(
                    merged,
                    ct_raw,
                    ct_blob,
                    inv_affine,
                    p95,
                    p998,
                    surface,
                    support_dist,
                    outside_dist,
                )
                if merged.inferred_contacts <= max(bridged[i].inferred_contacts, bridged[j].inferred_contacts):
                    continue
                bridged[i] = merged
                del bridged[j]
                changed = True
                break
            if changed:
                break

    bridged.sort(key=lambda x: x.score, reverse=True)
    return bridged


def fill_evidence_supported_gaps(
    shaft: FittedShaft,
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    inv_affine: np.ndarray,
    p95: float,
    p998: float,
    surface: SurfaceMesh,
    support_dist: float,
    outside_dist: float,
) -> FittedShaft:
    axis, mu, t, ordered_pts = shaft_axis(shaft.contacts_mm)
    del axis, mu, t
    spacing_mm = float(np.clip(shaft.spacing_mm, 2.4, 5.0))
    if len(ordered_pts) < 2:
        return shaft

    out = [ordered_pts[0]]
    changed = False
    for i, gap in enumerate(np.linalg.norm(np.diff(ordered_pts, axis=0), axis=1)):
        if gap > max(1.9 * spacing_mm, 6.0) and gap < 6.0 * spacing_mm:
            n_missing = max(1, int(round(gap / spacing_mm)) - 1)
            interp = np.vstack(
                [
                    (1.0 - alpha) * ordered_pts[i] + alpha * ordered_pts[i + 1]
                    for alpha in np.linspace(0.0, 1.0, n_missing + 2)[1:-1]
                ]
            )
            evidence = contact_evidence_values(ct_raw, ct_blob, inv_affine, p95, p998, interp)
            inside = surface.contains_points(interp)
            dist = surface.nearest_vertex_distance(interp)
            bridge_support = float(np.mean(inside | (dist <= support_dist)))
            bridge_outside = float(np.mean((~inside) & (dist > outside_dist)))
            if bridge_support >= 0.60 and float(np.mean(evidence)) >= 0.42 and bridge_outside <= 0.35:
                out.extend(interp)
                changed = True
        out.append(ordered_pts[i + 1])

    if not changed:
        return shaft

    return rebuild_shaft_from_contacts(
        shaft=shaft,
        contacts_mm=np.asarray(out, dtype=np.float64),
        ct_raw=ct_raw,
        ct_blob=ct_blob,
        inv_affine=inv_affine,
        p95=p95,
        p998=p998,
        surface=surface,
        support_dist=support_dist,
        outside_dist=outside_dist,
        surface_weight=1.0,
    )


def should_bridge_neighbor_shafts(a: FittedShaft, b: FittedShaft) -> bool:
    axis_a, mu_a, _, _ = shaft_axis(a.contacts_mm)
    axis_b, mu_b, _, _ = shaft_axis(b.contacts_mm)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(axis_a, axis_b)), 0.0, 1.0))))
    if angle > 18.0:
        return False

    line_dist = shaft_line_distance(mu_a, axis_a, mu_b, axis_b)
    if line_dist > 2.4:
        return False

    gap = shaft_interval_gap(a.contacts_mm, b.contacts_mm, axis_a, axis_b)
    if gap < 4.0 or gap > 18.0:
        return False

    ends_a = np.vstack([a.contacts_mm[0], a.contacts_mm[-1]])
    ends_b = np.vstack([b.contacts_mm[0], b.contacts_mm[-1]])
    end_dist = float(np.min(np.linalg.norm(ends_a[:, None, :] - ends_b[None, :, :], axis=2)))
    return end_dist <= 18.0


def bridge_segment_support(
    a: FittedShaft,
    b: FittedShaft,
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    inv_affine: np.ndarray,
    p95: float,
    p998: float,
    surface: SurfaceMesh,
    support_dist: float,
    outside_dist: float,
) -> tuple[float, float] | None:
    ends_a = np.vstack([a.contacts_mm[0], a.contacts_mm[-1]])
    ends_b = np.vstack([b.contacts_mm[0], b.contacts_mm[-1]])
    d = np.linalg.norm(ends_a[:, None, :] - ends_b[None, :, :], axis=2)
    ia, ib = np.unravel_index(int(np.argmin(d)), d.shape)
    p0 = ends_a[ia]
    p1 = ends_b[ib]
    seg_len = float(np.linalg.norm(p1 - p0))
    if seg_len < 4.0:
        return None

    n_samples = max(3, int(np.ceil(seg_len / 1.2)))
    interp = np.vstack(
        [
            (1.0 - alpha) * p0 + alpha * p1
            for alpha in np.linspace(0.0, 1.0, n_samples + 2)[1:-1]
        ]
    )
    evidence = contact_evidence_values(ct_raw, ct_blob, inv_affine, p95, p998, interp)
    inside = surface.contains_points(interp)
    dist = surface.nearest_vertex_distance(interp)
    bridge_support = float(np.mean(inside | (dist <= support_dist)))
    bridge_outside = float(np.mean((~inside) & (dist > outside_dist)))
    if bridge_outside > 0.35:
        return None
    return bridge_support, float(np.mean(evidence))


def trim_outside_shaft_ends(
    shafts: list[FittedShaft],
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    ct_affine: np.ndarray,
    surface: SurfaceMesh,
    surface_support: float,
) -> list[FittedShaft]:
    if not shafts:
        return shafts

    inv_affine = np.linalg.inv(ct_affine)
    p95, p998 = np.percentile(ct_raw[np.isfinite(ct_raw)], [95.0, 99.8])
    support_dist, outside_dist, surface_weight, _, strong_ev_thr = surface_distance_params(surface_support)
    trimmed: list[FittedShaft] = []
    for shaft in shafts:
        pts = shaft.contacts_mm
        if len(pts) < 3:
            trimmed.append(shaft)
            continue
        evidence = contact_evidence_values(ct_raw, ct_blob, inv_affine, p95, p998, pts)
        inside = surface.contains_points(pts)
        dist = surface.nearest_vertex_distance(pts)
        keep = inside | (dist <= support_dist) | ((dist <= outside_dist) & (evidence >= strong_ev_thr))
        keep_idx = np.flatnonzero(keep)
        if keep_idx.size == 0:
            trimmed.append(shaft)
            continue
        lo = int(keep_idx[0])
        hi = int(keep_idx[-1]) + 1
        if lo == 0 and hi == len(pts):
            trimmed.append(shaft)
            continue
        if hi - lo < 3:
            trimmed.append(shaft)
            continue
        trimmed.append(
            rebuild_shaft_from_contacts(
                shaft=shaft,
                contacts_mm=pts[lo:hi],
                ct_raw=ct_raw,
                ct_blob=ct_blob,
                inv_affine=inv_affine,
                p95=p95,
                p998=p998,
                surface=surface,
                support_dist=support_dist,
                outside_dist=outside_dist,
                surface_weight=surface_weight,
            )
        )
    trimmed.sort(key=lambda x: x.score, reverse=True)
    return trimmed


def surface_distance_params(surface_support: float) -> tuple[float, float, float, float, float]:
    support_dist = 3.5
    outside_dist = 6.0
    surface_weight = 1.0
    base_ev_thr = 0.64
    strong_ev_thr = 0.94
    if surface_support < 0.25:
        support_dist = 8.0
        outside_dist = 11.0
        surface_weight = 0.25
        base_ev_thr = 0.56
        strong_ev_thr = 0.82
    elif surface_support < 0.35:
        support_dist = 6.5
        outside_dist = 9.5
        surface_weight = 0.45
        base_ev_thr = 0.58
        strong_ev_thr = 0.86
    elif surface_support < 0.45:
        support_dist = 5.0
        outside_dist = 8.0
        surface_weight = 0.70
        base_ev_thr = 0.60
        strong_ev_thr = 0.90
    return support_dist, outside_dist, surface_weight, base_ev_thr, strong_ev_thr


def merge_collinear_shafts(shafts: list[FittedShaft], strict: bool = False) -> list[FittedShaft]:
    merged = list(shafts)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged) - 1):
            for j in range(i + 1, len(merged)):
                if not should_merge_shafts(merged[i], merged[j], strict=strict):
                    continue
                merged[i] = combine_shafts(merged[i], merged[j])
                del merged[j]
                changed = True
                break
            if changed:
                break
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged


def drop_redundant_fragments(shafts: list[FittedShaft]) -> list[FittedShaft]:
    kept: list[FittedShaft] = []
    for shaft in sorted(shafts, key=lambda x: (x.inferred_contacts, x.score), reverse=True):
        redundant = any(is_redundant_fragment(shaft, other) for other in kept)
        if redundant:
            continue
        kept.append(shaft)
    kept.sort(key=lambda x: x.score, reverse=True)
    return kept


def consolidate_shaft_groups(
    shafts: list[FittedShaft],
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    ct_affine: np.ndarray,
    surface: SurfaceMesh,
    surface_support: float,
    strict: bool = False,
) -> list[FittedShaft]:
    if len(shafts) < 2:
        return shafts

    parent = list(range(len(shafts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(shafts) - 1):
        for j in range(i + 1, len(shafts)):
            if should_group_shafts(shafts[i], shafts[j], strict=strict):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(shafts)):
        groups.setdefault(find(i), []).append(i)

    out: list[FittedShaft] = []
    for members in groups.values():
        if len(members) == 1:
            out.append(shafts[members[0]])
            continue

        if strict:
            ordered_members = sorted(
                members,
                key=lambda idx: (shafts[idx].inferred_contacts, shafts[idx].score),
                reverse=True,
            )
            anchor = shafts[ordered_members[0]]
            for idx in ordered_members[1:]:
                frag = shafts[idx]
                if should_attach_fragment(anchor, frag):
                    anchor = combine_shafts(anchor, frag)
            out.append(anchor)
            continue

        fragmented_mid_support = (
            0.40 <= surface_support <= 0.52
            and len(shafts) >= 24
            and max(shafts[idx].inferred_contacts for idx in members) <= 10
        )
        if fragmented_mid_support:
            ordered_members = sorted(
                members,
                key=lambda idx: (shafts[idx].inferred_contacts, shafts[idx].score),
                reverse=True,
            )
            merged = shafts[ordered_members[0]]
            for idx in ordered_members[1:]:
                merged = combine_shafts(merged, shafts[idx])
            out.append(merged)
            continue

        pts = np.vstack([shafts[idx].contacts_mm for idx in members])
        seed_indices = tuple(sorted(set().union(*[set(shafts[idx].seed_indices) for idx in members])))
        chain = synthetic_chain_from_contacts(
            points_mm=pts,
            seed_indices=seed_indices,
            ct_raw=ct_raw,
            ct_blob=ct_blob,
            ct_affine=ct_affine,
            surface=surface,
        )
        if chain is None:
            merged = shafts[members[0]]
            for idx in members[1:]:
                merged = combine_shafts(merged, shafts[idx])
            out.append(merged)
            continue

        refit = fit_contacts_to_chain_auto(
            chain=chain,
            ct_raw=ct_raw,
            ct_blob=ct_blob,
            ct_affine=ct_affine,
            surface=surface,
            surface_support=surface_support,
        )
        if refit is None:
            merged = shafts[members[0]]
            for idx in members[1:]:
                merged = combine_shafts(merged, shafts[idx])
            out.append(merged)
        else:
            out.append(refit)

    out.sort(key=lambda x: x.score, reverse=True)
    return out


def should_group_shafts(a: FittedShaft, b: FittedShaft, strict: bool = False) -> bool:
    axis_a, mu_a, t_a, pts_a = shaft_axis(a.contacts_mm)
    axis_b, mu_b, t_b, pts_b = shaft_axis(b.contacts_mm)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(axis_a, axis_b)), 0.0, 1.0))))
    line_dist = shaft_line_distance(mu_a, axis_a, mu_b, axis_b)
    max_line_dist = 2.6 if strict else 4.5
    if line_dist > max_line_dist:
        return False

    angle_cap = 22.0 if strict else 28.0
    relaxed_angle_cap = 26.0 if strict else 35.0
    relaxed_line_dist = 1.8 if strict else 3.5
    if angle > angle_cap:
        if not (angle <= relaxed_angle_cap and line_dist <= relaxed_line_dist):
            return False

    spacing_close = abs(a.spacing_mm - b.spacing_mm) <= 1.8 or min(a.inferred_contacts, b.inferred_contacts) <= 4
    if not spacing_close:
        return False

    gap = shaft_interval_gap(pts_a, pts_b, axis_a, axis_b)
    max_gap = 10.0 if strict else 16.0
    if gap > max_gap:
        return False

    ends_a = np.vstack([pts_a[0], pts_a[-1]])
    ends_b = np.vstack([pts_b[0], pts_b[-1]])
    end_dist = float(np.min(np.linalg.norm(ends_a[:, None, :] - ends_b[None, :, :], axis=2)))
    small_pair = max(a.inferred_contacts, b.inferred_contacts) <= 6 and min(a.inferred_contacts, b.inferred_contacts) <= 4
    if small_pair and line_dist <= 1.5 and end_dist <= 6.0 and gap <= 6.0:
        return True
    spacing_ref = float(np.clip(np.median([a.spacing_mm, b.spacing_mm]), 2.4, 5.0))
    mean_nn, med_nn = shaft_contact_alignment(a.contacts_mm, b.contacts_mm)
    if gap <= 0.5 and min(a.inferred_contacts, b.inferred_contacts) >= 8:
        if strict:
            if med_nn > max(2.2, 0.70 * spacing_ref) or mean_nn > max(4.0, 1.20 * spacing_ref):
                return False
        else:
            if med_nn > max(5.5, 1.70 * spacing_ref) or mean_nn > max(10.0, 3.00 * spacing_ref):
                return False
    if strict and min(a.inferred_contacts, b.inferred_contacts) >= 10:
        if line_dist > 1.8 or gap > 6.0:
            return False
        return end_dist <= 6.0 or gap <= 4.5
    if strict:
        return end_dist <= 8.0 or gap <= 8.0
    return end_dist <= 14.0 or gap <= 14.0


def is_redundant_fragment(fragment: FittedShaft, other: FittedShaft) -> bool:
    if fragment.inferred_contacts > 5:
        return False
    if other.inferred_contacts < fragment.inferred_contacts + 4:
        return False

    axis_f, mu_f, t_f, pts_f = shaft_axis(fragment.contacts_mm)
    axis_o, mu_o, t_o, pts_o = shaft_axis(other.contacts_mm)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(axis_f, axis_o)), 0.0, 1.0))))
    if angle > 14.0:
        return False

    line_dist = shaft_line_distance(mu_f, axis_f, mu_o, axis_o)
    if line_dist > 2.8:
        return False

    gap = shaft_interval_gap(pts_f, pts_o, axis_f, axis_o)
    max_gap = max(7.5, 2.0 * max(fragment.spacing_mm, other.spacing_mm))
    if gap > max_gap:
        return False

    end_f = np.vstack([pts_f[0], pts_f[-1]])
    end_o = np.vstack([pts_o[0], pts_o[-1]])
    end_dist = float(np.min(np.linalg.norm(end_f[:, None, :] - end_o[None, :, :], axis=2)))
    if end_dist > max(8.0, 2.4 * max(fragment.spacing_mm, other.spacing_mm)) and gap > max(5.0, 1.5 * max(fragment.spacing_mm, other.spacing_mm)):
        return False

    d_contacts = np.linalg.norm(pts_f[:, None, :] - pts_o[None, :, :], axis=2)
    mean_min_contact_dist = float(np.mean(np.min(d_contacts, axis=1)))
    return mean_min_contact_dist <= 3.2


def synthetic_chain_from_contacts(
    points_mm: np.ndarray,
    seed_indices: tuple[int, ...],
    ct_raw: np.ndarray,
    ct_blob: np.ndarray,
    ct_affine: np.ndarray,
    surface: SurfaceMesh,
) -> ChainHypothesis | None:
    if len(points_mm) < 3:
        return None

    inv_affine = np.linalg.inv(ct_affine)
    p95, p998 = np.percentile(ct_raw[np.isfinite(ct_raw)], [95.0, 99.8])
    axis, mu, t, ordered_pts = shaft_axis(points_mm)
    del axis, mu
    ordered_pts = deduplicate_contacts_along_axis(ordered_pts, t, min_spacing_mm=1.5)
    if len(ordered_pts) < 3:
        return None

    spacing = estimate_spacing(ordered_pts)
    regularity = spacing_regularity(ordered_pts, spacing)
    span = polyline_length(ordered_pts)
    if regularity < 0.12 or span < 6.0:
        return None

    evidence_vals = contact_evidence_values(ct_raw, ct_blob, inv_affine, p95, p998, ordered_pts)
    inside = surface.contains_points(ordered_pts)
    dist = surface.nearest_vertex_distance(ordered_pts)
    near = dist <= 8.0
    brain_support = float(np.mean(inside | near))
    inside_frac = float(np.mean(inside))
    near_frac = float(np.mean(near))
    score = (
        1.4 * regularity
        + 0.75 * float(np.mean(evidence_vals))
        + 0.20 * len(ordered_pts)
        + 0.05 * span
        + 0.35 * brain_support
    )
    return ChainHypothesis(
        indices=seed_indices,
        ordered_points_mm=ordered_pts,
        ordered_scores=evidence_vals.astype(np.float64),
        spacing_mm=float(np.clip(spacing, 2.4, 5.0)),
        span_mm=float(span),
        regularity=float(regularity),
        brain_support=brain_support,
        inside_fraction=inside_frac,
        near_fraction=near_frac,
        score=float(score),
    )


def should_merge_shafts(a: FittedShaft, b: FittedShaft, strict: bool = False) -> bool:
    axis_a, mu_a, t_a, _ = shaft_axis(a.contacts_mm)
    axis_b, mu_b, t_b, _ = shaft_axis(b.contacts_mm)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(axis_a, axis_b)), 0.0, 1.0))))
    small_pair = max(a.inferred_contacts, b.inferred_contacts) <= 6 and min(a.inferred_contacts, b.inferred_contacts) <= 4
    if small_pair and angle <= 65.0:
        line_dist = shaft_line_distance(mu_a, axis_a, mu_b, axis_b)
        if line_dist > 1.5:
            return False
        gap = shaft_interval_gap(a.contacts_mm, b.contacts_mm, axis_a, axis_b)
        if gap > 6.0:
            return False
        end_a = np.vstack([a.contacts_mm[0], a.contacts_mm[-1]])
        end_b = np.vstack([b.contacts_mm[0], b.contacts_mm[-1]])
        end_dist = float(np.min(np.linalg.norm(end_a[:, None, :] - end_b[None, :, :], axis=2)))
        return end_dist <= 6.0
    angle_cap = 16.0 if strict else 22.0
    if angle > angle_cap:
        return False

    line_dist = shaft_line_distance(mu_a, axis_a, mu_b, axis_b)
    max_line_dist = 2.0 if strict else 3.0
    if line_dist > max_line_dist:
        return False

    gap = shaft_interval_gap(a.contacts_mm, b.contacts_mm, axis_a, axis_b)
    max_gap = 8.0 if strict else 12.0
    if gap > max_gap:
        return False

    end_a = np.vstack([a.contacts_mm[0], a.contacts_mm[-1]])
    end_b = np.vstack([b.contacts_mm[0], b.contacts_mm[-1]])
    end_dist = float(np.min(np.linalg.norm(end_a[:, None, :] - end_b[None, :, :], axis=2)))
    spacing_ref = float(np.clip(np.median([a.spacing_mm, b.spacing_mm]), 2.4, 5.0))
    mean_nn, med_nn = shaft_contact_alignment(a.contacts_mm, b.contacts_mm)
    overlap_axis = common_interval_overlap(a.contacts_mm, b.contacts_mm, axis_a, axis_b)
    if overlap_axis > 0.5:
        if strict:
            if med_nn > max(2.2, 0.70 * spacing_ref) or mean_nn > max(4.0, 1.20 * spacing_ref):
                return False
            return angle <= 8.0 and line_dist <= 1.2
        if med_nn > max(5.5, 1.70 * spacing_ref) or mean_nn > max(10.0, 3.00 * spacing_ref):
            return False
        return angle <= 16.0 and line_dist <= 2.2
    if strict:
        if min(a.inferred_contacts, b.inferred_contacts) >= 10:
            if line_dist > 1.5 or gap > 4.5:
                return False
            return end_dist <= max(4.5, 1.35 * spacing_ref)
        return end_dist <= max(6.0, 1.7 * spacing_ref)
    return end_dist <= max(9.0, 2.1 * spacing_ref)


def shaft_contact_alignment(points_a: np.ndarray, points_b: np.ndarray) -> tuple[float, float]:
    d = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)
    min_a = np.min(d, axis=1)
    min_b = np.min(d, axis=0)
    combined = np.r_[min_a, min_b]
    return float(np.mean(combined)), float(np.median(combined))


def common_interval_overlap(
    points_a: np.ndarray,
    points_b: np.ndarray,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
) -> float:
    sign = 1.0 if np.dot(axis_a, axis_b) >= 0 else -1.0
    axis = axis_a + sign * axis_b
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    ta = points_a @ axis
    tb = points_b @ axis
    return float(min(float(np.max(ta)), float(np.max(tb))) - max(float(np.min(ta)), float(np.min(tb))))


def should_attach_fragment(anchor: FittedShaft, frag: FittedShaft) -> bool:
    if frag.inferred_contacts > max(7, anchor.inferred_contacts - 4):
        return False

    axis_a, mu_a, _, _ = shaft_axis(anchor.contacts_mm)
    axis_b, mu_b, _, _ = shaft_axis(frag.contacts_mm)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(axis_a, axis_b)), 0.0, 1.0))))
    if angle > 12.0:
        return False

    line_dist = shaft_line_distance(mu_a, axis_a, mu_b, axis_b)
    spacing_ref = float(np.clip(np.median([anchor.spacing_mm, frag.spacing_mm]), 2.4, 5.0))
    if line_dist > max(1.2, 0.35 * spacing_ref):
        return False

    gap = shaft_interval_gap(anchor.contacts_mm, frag.contacts_mm, axis_a, axis_b)
    overlap_axis = common_interval_overlap(anchor.contacts_mm, frag.contacts_mm, axis_a, axis_b)
    mean_nn, med_nn = shaft_contact_alignment(anchor.contacts_mm, frag.contacts_mm)

    end_a = np.vstack([anchor.contacts_mm[0], anchor.contacts_mm[-1]])
    end_b = np.vstack([frag.contacts_mm[0], frag.contacts_mm[-1]])
    end_dist = float(np.min(np.linalg.norm(end_a[:, None, :] - end_b[None, :, :], axis=2)))

    if overlap_axis > 0.5:
        return False

    if gap > max(4.8, 1.55 * spacing_ref):
        return False

    if end_dist > max(5.5, 1.75 * spacing_ref):
        return False

    if med_nn > max(6.0, 1.9 * spacing_ref):
        return False

    return True


def combine_shafts(a: FittedShaft, b: FittedShaft) -> FittedShaft:
    all_pts = np.vstack([a.contacts_mm, b.contacts_mm])
    axis, mu, t, ordered_pts = shaft_axis(all_pts)
    del axis, mu
    merge_spacing = float(np.clip(np.median([a.spacing_mm, b.spacing_mm]), 2.4, 5.0))
    dedup_spacing = float(np.clip(0.62 * merge_spacing, 1.8, 2.4))
    merged_pts = deduplicate_contacts_along_axis(ordered_pts, t, min_spacing_mm=dedup_spacing)
    merged_pts = fill_small_contact_gaps(merged_pts, gap_lo_factor=1.8, gap_hi_factor=2.35)
    spacing_mm = estimate_spacing(merged_pts)
    score = max(a.score, b.score) + 0.18 * min(len(merged_pts), 12) / 12.0
    support = weighted_average(
        [a.support_score, b.support_score],
        [len(a.contacts_mm), len(b.contacts_mm)],
    )
    evidence = weighted_average(
        [a.evidence_score, b.evidence_score],
        [len(a.contacts_mm), len(b.contacts_mm)],
    )
    brain_support = weighted_average(
        [a.brain_support, b.brain_support],
        [len(a.contacts_mm), len(b.contacts_mm)],
    )
    outside_fraction = weighted_average(
        [a.outside_fraction, b.outside_fraction],
        [len(a.contacts_mm), len(b.contacts_mm)],
    )
    return FittedShaft(
        inferred_contacts=int(len(merged_pts)),
        observed_contacts=a.observed_contacts + b.observed_contacts,
        score=float(score),
        support_score=float(support),
        evidence_score=float(evidence),
        spacing_mm=float(spacing_mm),
        brain_support=float(brain_support),
        outside_fraction=float(outside_fraction),
        contacts_mm=merged_pts,
        seed_indices=tuple(sorted(set(a.seed_indices) | set(b.seed_indices))),
    )


def shaft_axis(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = points_mm.mean(axis=0)
    if len(points_mm) < 2:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        _, _, vh = np.linalg.svd(points_mm - mu, full_matrices=False)
        axis = vh[0]
    t = (points_mm - mu) @ axis
    order = np.argsort(t)
    return axis, mu, t[order], points_mm[order]


def shaft_line_distance(mu_a: np.ndarray, axis_a: np.ndarray, mu_b: np.ndarray, axis_b: np.ndarray) -> float:
    cross = np.cross(axis_a, axis_b)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-6:
        return float(np.linalg.norm(np.cross(mu_b - mu_a, axis_a)))
    return float(abs(np.dot(mu_b - mu_a, cross)) / cross_norm)


def shaft_interval_gap(points_a: np.ndarray, points_b: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    sign = 1.0 if np.dot(axis_a, axis_b) >= 0 else -1.0
    axis = axis_a + sign * axis_b
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    ta = points_a @ axis
    tb = points_b @ axis
    return float(max(0.0, max(float(np.min(ta)) - float(np.max(tb)), float(np.min(tb)) - float(np.max(ta)))))


def deduplicate_contacts_along_axis(points_mm: np.ndarray, t_sorted: np.ndarray, min_spacing_mm: float) -> np.ndarray:
    keep_pts = [points_mm[0]]
    keep_t = [float(t_sorted[0])]
    for pt, t in zip(points_mm[1:], t_sorted[1:]):
        if float(t) - keep_t[-1] < min_spacing_mm:
            keep_pts[-1] = 0.5 * (keep_pts[-1] + pt)
            keep_t[-1] = 0.5 * (keep_t[-1] + float(t))
            continue
        keep_pts.append(pt)
        keep_t.append(float(t))
    return np.asarray(keep_pts, dtype=np.float64)


def fill_small_contact_gaps(
    points_mm: np.ndarray,
    gap_lo_factor: float = 1.6,
    gap_hi_factor: float = 2.6,
) -> np.ndarray:
    if len(points_mm) < 3:
        return points_mm
    axis, mu, t, ordered_pts = shaft_axis(points_mm)
    del axis, mu
    gaps = np.diff(t)
    valid = gaps[(gaps > 2.0) & (gaps < 6.0)]
    if valid.size == 0:
        return ordered_pts
    spacing = float(np.median(valid))
    out = [ordered_pts[0]]
    for i, gap in enumerate(gaps):
        if gap > gap_lo_factor * spacing and gap < gap_hi_factor * spacing:
            n_missing = max(0, int(round(gap / spacing)) - 1)
            for m in range(n_missing):
                alpha = (m + 1) / (n_missing + 1)
                out.append((1.0 - alpha) * ordered_pts[i] + alpha * ordered_pts[i + 1])
        out.append(ordered_pts[i + 1])
    return np.asarray(out, dtype=np.float64)


def deduplicate_polyline_points(points_mm: np.ndarray, min_step_mm: float) -> np.ndarray:
    if len(points_mm) < 2:
        return points_mm
    out = [points_mm[0]]
    for pt in points_mm[1:]:
        if float(np.linalg.norm(pt - out[-1])) < min_step_mm:
            out[-1] = 0.5 * (out[-1] + pt)
            continue
        out.append(pt)
    return np.asarray(out, dtype=np.float64)


def weighted_average(values: list[float], weights: list[int]) -> float:
    weights_arr = np.asarray(weights, dtype=np.float64)
    if np.sum(weights_arr) <= 0:
        return float(np.mean(values))
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weights_arr))


def arc_lengths(points_mm: np.ndarray) -> np.ndarray:
    if len(points_mm) == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(len(points_mm), dtype=np.float64)
    if len(points_mm) > 1:
        out[1:] = np.cumsum(np.linalg.norm(np.diff(points_mm, axis=0), axis=1))
    return out


def sample_polyline_with_extrapolation(points_mm: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    obs_s = arc_lengths(points_mm)
    out = np.empty((len(s_query), 3), dtype=np.float64)

    if len(points_mm) == 1:
        out[:] = points_mm[0]
        return out

    start_dir = points_mm[1] - points_mm[0]
    start_dir = start_dir / max(np.linalg.norm(start_dir), 1e-6)
    end_dir = points_mm[-1] - points_mm[-2]
    end_dir = end_dir / max(np.linalg.norm(end_dir), 1e-6)

    for i, s in enumerate(s_query):
        if s <= 0:
            out[i] = points_mm[0] + start_dir * s
            continue
        if s >= obs_s[-1]:
            out[i] = points_mm[-1] + end_dir * (s - obs_s[-1])
            continue
        idx = int(np.searchsorted(obs_s, s, side="right") - 1)
        idx = min(max(idx, 0), len(points_mm) - 2)
        seg_len = max(obs_s[idx + 1] - obs_s[idx], 1e-6)
        alpha = (s - obs_s[idx]) / seg_len
        out[i] = (1.0 - alpha) * points_mm[idx] + alpha * points_mm[idx + 1]
    return out


def evaluate_case(case: CaseRecord) -> dict:
    ct_img = nib.load(str(case.ct_path))
    ct_raw = ct_img.get_fdata(dtype=np.float32)
    ct_affine = ct_img.affine

    surface = load_surface(case.surface_path)
    intracranial_mask = load_intracranial_mask(case, ct_raw.shape, ct_affine)
    intracranial_dist_mm = None
    if intracranial_mask is not None:
        _, intracranial_dist_mm = intracranial_prior_maps(intracranial_mask, voxel_sizes_from_affine(ct_affine))
    candidates = extract_candidates(ct_raw, ct_affine, surface, intracranial_mask=intracranial_mask)
    peak_chains = propose_chains(candidates)
    inside_frac = float(np.mean(candidates.intracranial_inside_mask)) if len(candidates.intracranial_inside_mask) else 0.0
    use_component_rescue = (
        candidates.surface_support_fraction <= 0.50
        or candidates.intracranial_support_fraction <= 0.45
        or inside_frac <= 0.38
    )
    component_chains: list[ChainHypothesis] = []
    if use_component_rescue:
        component_chains = propose_component_chains(
            candidates,
            ct_raw,
            ct_affine,
            surface,
            intracranial_mask=intracranial_mask,
            intracranial_dist_mm=intracranial_dist_mm,
        )
        max_component_chains = 5
        if candidates.surface_support_fraction < 0.40:
            max_component_chains = 1
        elif candidates.surface_support_fraction < 0.50:
            max_component_chains = 5
        component_chains = filter_component_rescue_chains(
            peak_chains,
            component_chains,
            surface_support=candidates.surface_support_fraction,
            max_chains=max_component_chains,
        )
    chains = merge_chain_lists(peak_chains, component_chains) if component_chains else peak_chains
    chains.sort(key=lambda x: x.score, reverse=True)
    shafts = select_and_fit_shafts(
        chains,
        candidates,
        ct_raw,
        ct_affine,
        surface,
        intracranial_mask=intracranial_mask,
        intracranial_dist_mm=intracranial_dist_mm,
    )

    expected_counts_map = load_expected_counts(case.elec_data_path)
    expected_counts = sorted(expected_counts_map.values(), reverse=True)
    predicted_counts = sorted([int(s.inferred_contacts) for s in shafts], reverse=True)
    predicted_total = int(sum(predicted_counts))
    expected_total = int(sum(expected_counts))

    n_pad = max(len(expected_counts), len(predicted_counts))
    exp_pad = np.array(expected_counts + [0] * (n_pad - len(expected_counts)), dtype=np.int64)
    pred_pad = np.array(predicted_counts + [0] * (n_pad - len(predicted_counts)), dtype=np.int64)
    count_l1 = int(np.abs(exp_pad - pred_pad).sum())

    return {
        "subject_id": case.subject_id,
        "expected_electrodes": len(expected_counts),
        "expected_total_contacts": expected_total,
        "expected_counts": expected_counts,
        "candidate_count": int(len(candidates.points_mm)),
        "inside_candidate_count": int(np.count_nonzero(candidates.inside_mask)),
        "surface_support_fraction": float(candidates.surface_support_fraction),
        "intracranial_support_fraction": float(candidates.intracranial_support_fraction),
        "chain_count": int(len(chains)),
        "peak_chain_count": int(len(peak_chains)),
        "component_chain_count": int(len(component_chains)),
        "predicted_electrodes": int(len(shafts)),
        "predicted_total_contacts": predicted_total,
        "predicted_counts": predicted_counts,
        "electrode_count_error": int(len(shafts) - len(expected_counts)),
        "total_contact_error": int(predicted_total - expected_total),
        "count_l1_error": count_l1,
        "mean_shaft_score": float(np.mean([s.score for s in shafts])) if shafts else 0.0,
        "mean_support_score": float(np.mean([s.support_score for s in shafts])) if shafts else 0.0,
        "mean_evidence_score": float(np.mean([s.evidence_score for s in shafts])) if shafts else 0.0,
    }


def run_all(csv_path: Path, subject_filter: str | None = None) -> list[dict]:
    cases = load_case_table(csv_path)
    if subject_filter:
        cases = [c for c in cases if c.subject_id == subject_filter]
    return [evaluate_case(case) for case in cases]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic count-free sEEG contact detector/evaluator.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("elec_data.csv"),
        help="Path to case table CSV.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Optional subject_id filter.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    results = run_all(args.csv, args.subject)
    for result in results:
        print(json.dumps(result, indent=2))

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
