#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy import io as sio
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import seeg_contact_detector as detector


OLD_FILE_PATTERNS = [
    "/Users/savirmadan/Partners HealthCare Dropbox/Savir Madan/SEEGAnalysis/OLD_LEGUI_TESTING_FOLDER/sub*/Registered/Electrodes.mat",
    "/Users/savirmadan/Partners HealthCare Dropbox/Savir Madan/SEEGAnalysis/OLD_LEGUI_TESTING_FOLDER/sub-*/Registered/Electrodes.mat",
]

NEW_FILE_PATTERNS = [
    "/Users/savirmadan/Partners HealthCare Dropbox/Savir Madan/SEEGAnalysis/NewLeGui/*Output/derivatives/leaddbs/sub-*/reconstruction/sub-*_electrodes.mat",
    "/Users/savirmadan/Partners HealthCare Dropbox/Savir Madan/SEEGAnalysis/NewLeGui/*Ouput/derivatives/leaddbs/sub-*/reconstruction/sub-*_electrodes.mat",
]

SUBJECT_RE = re.compile(r"sub-sub(\d+)|sub-(\d+)|sub(\d+)")


def subject_id_from_path(path: str | Path) -> str:
    match = SUBJECT_RE.search(str(path))
    if match is None:
        raise ValueError(f"Could not parse subject id from {path}")
    return next(group for group in match.groups() if group is not None)


def choose_largest_file_per_subject(paths: list[str]) -> dict[str, Path]:
    chosen: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        subject = subject_id_from_path(path)
        if subject not in chosen or path.stat().st_size > chosen[subject].stat().st_size:
            chosen[subject] = path
    return chosen


def load_vector_dataset(path: Path, variable_name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle[variable_name][()], dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{path}::{variable_name} has shape {values.shape}, expected 2-D")
    if values.shape[0] == 3:
        values = values.T
    elif values.shape[1] != 3:
        raise ValueError(f"{path}::{variable_name} has shape {values.shape}, expected Nx3 or 3xN")
    return values


def assignment_pairs(reference_points: np.ndarray, test_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairwise_distances = np.linalg.norm(reference_points[:, None, :] - test_points[None, :, :], axis=2)
    ref_idx, test_idx = linear_sum_assignment(pairwise_distances)
    assigned_distances = pairwise_distances[ref_idx, test_idx]
    return ref_idx, test_idx, assigned_distances


def assignment_metrics(reference_points: np.ndarray, test_points: np.ndarray) -> dict[str, float | int]:
    ref_idx, test_idx, assigned_distances = assignment_pairs(reference_points, test_points)
    return {
        "assigned_pairs": int(len(assigned_distances)),
        "reference_unmatched": int(len(reference_points) - len(assigned_distances)),
        "test_unmatched": int(len(test_points) - len(assigned_distances)),
        "assignment_total": float(assigned_distances.sum()),
        "assignment_mean": float(assigned_distances.mean()),
        "assignment_median": float(np.median(assigned_distances)),
        "assignment_max": float(assigned_distances.max()),
        "assignment_within_2": float((assigned_distances <= 2.0).mean()),
        "assignment_within_5": float((assigned_distances <= 5.0).mean()),
    }


def list_ct_backwarp_transforms(new_electrode_path: Path, subject: str) -> list[Path]:
    subject_root = new_electrode_path.parents[1]
    transform_dir = subject_root / "coregistration" / "transformations"
    return sorted(transform_dir.glob(f"sub-sub{subject}_from-anchorNative_to-CT_desc-*44.mat"))


def load_affine44(path: Path) -> np.ndarray:
    return np.asarray(sio.loadmat(path, squeeze_me=True)["tmat"], dtype=np.float64)


def apply_affine44_to_points(points_mm: np.ndarray, affine44: np.ndarray) -> np.ndarray:
    hom = np.c_[points_mm, np.ones(len(points_mm))]
    if affine44.shape == (4, 4):
        return (hom @ affine44.T)[:, :3]
    mapped = hom @ affine44
    return mapped[:, :3]


def fit_affine3(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    design = np.c_[source_points, np.ones(len(source_points))]
    coeffs, *_ = np.linalg.lstsq(design, target_points, rcond=None)
    return coeffs


def apply_affine3(points_mm: np.ndarray, affine3: np.ndarray) -> np.ndarray:
    return np.c_[points_mm, np.ones(len(points_mm))] @ affine3


def apply_affine_knn_residual_map(
    train_source: np.ndarray,
    train_target: np.ndarray,
    query_points: np.ndarray,
    *,
    k: int,
    power: float,
) -> np.ndarray:
    affine3 = fit_affine3(train_source, train_target)
    residuals = train_target - apply_affine3(train_source, affine3)
    tree = cKDTree(train_source)
    k = min(k, len(train_source))
    distances, indices = tree.query(query_points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, 1e-3) ** power
    weights = weights / weights.sum(axis=1, keepdims=True)
    correction = (residuals[indices] * weights[:, :, None]).sum(axis=1)
    return apply_affine3(query_points, affine3) + correction


def build_detector_legacy_map(
    *,
    old_raw: np.ndarray,
    old_mni: np.ndarray,
    current_native: np.ndarray,
    current_mni: np.ndarray,
    detector_native: np.ndarray,
    k: int = 10,
    power: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    old_idx, current_idx, _ = assignment_pairs(old_mni, current_mni)
    current_anchor_native = current_native[current_idx]
    old_anchor_raw = old_raw[old_idx]
    current_anchor_idx, detector_anchor_idx, _ = assignment_pairs(current_anchor_native, detector_native)
    detector_anchor_native = detector_native[detector_anchor_idx]
    detector_anchor_raw = old_anchor_raw[current_anchor_idx]
    detector_mapped_raw = apply_affine_knn_residual_map(
        detector_anchor_native,
        detector_anchor_raw,
        detector_native,
        k=k,
        power=power,
    )
    detector_anchor_mask = np.zeros(len(detector_native), dtype=bool)
    detector_anchor_mask[detector_anchor_idx] = True
    return detector_mapped_raw, detector_anchor_mask


def greedy_select_detector_subset(
    *,
    old_raw: np.ndarray,
    detector_mapped_raw: np.ndarray,
    detector_anchor_mask: np.ndarray,
    mean_threshold_mm: float,
) -> np.ndarray:
    anchor_points = detector_mapped_raw[detector_anchor_mask]
    if len(anchor_points) >= len(old_raw):
        return anchor_points

    extra_points = detector_mapped_raw[~detector_anchor_mask]
    chosen = anchor_points
    remaining = list(range(len(extra_points)))

    while len(chosen) < len(old_raw) and remaining:
        best_candidate_idx = None
        best_candidate_mean = None
        for local_idx in remaining:
            candidate = np.vstack([chosen, extra_points[local_idx][None, :]])
            mean_mm = assignment_metrics(old_raw, candidate)["assignment_mean"]
            if mean_mm > mean_threshold_mm:
                continue
            if best_candidate_mean is None or mean_mm < best_candidate_mean:
                best_candidate_mean = mean_mm
                best_candidate_idx = local_idx
        if best_candidate_idx is None:
            break
        chosen = np.vstack([chosen, extra_points[best_candidate_idx][None, :]])
        remaining.remove(best_candidate_idx)

    return chosen


def run_detector(case: detector.CaseRecord) -> tuple[detector.CandidateSet, list[detector.FittedShaft]]:
    ct_img = nib.load(str(case.ct_path))
    ct_raw = ct_img.get_fdata(dtype=np.float32)
    ct_affine = ct_img.affine
    surface = detector.load_surface(case.surface_path)
    intracranial_mask = detector.load_intracranial_mask(case, ct_raw.shape, ct_affine)
    intracranial_dist_mm = None
    if intracranial_mask is not None:
        _, intracranial_dist_mm = detector.intracranial_prior_maps(
            intracranial_mask,
            detector.voxel_sizes_from_affine(ct_affine),
        )

    candidates = detector.extract_candidates(
        ct_raw,
        ct_affine,
        surface,
        intracranial_mask=intracranial_mask,
    )
    peak_chains = detector.propose_chains(candidates)
    component_chains: list[detector.ChainHypothesis] = []
    if intracranial_mask is not None:
        component_chains = detector.propose_component_chains(
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
        component_chains = detector.filter_component_rescue_chains(
            peak_chains,
            component_chains,
            surface_support=candidates.surface_support_fraction,
            max_chains=max_component_chains,
        )
    chains = detector.merge_chain_lists(peak_chains, component_chains) if component_chains else peak_chains
    chains.sort(key=lambda x: x.score, reverse=True)
    shafts = detector.select_and_fit_shafts(
        chains,
        candidates,
        ct_raw,
        ct_affine,
        surface,
        intracranial_mask=intracranial_mask,
        intracranial_dist_mm=intracranial_dist_mm,
    )
    return candidates, shafts


def benchmark_subject(
    case: detector.CaseRecord,
    old_electrode_path: Path,
    new_electrode_path: Path,
) -> dict[str, object]:
    subject = subject_id_from_path(case.subject_id)
    candidates, shafts = run_detector(case)
    predicted_native = np.vstack([shaft.contacts_mm for shaft in shafts]) if shafts else np.zeros((0, 3), dtype=np.float64)

    new_native = load_vector_dataset(new_electrode_path, "ElecXYZRaw")
    new_mni = load_vector_dataset(new_electrode_path, "ElecXYZMNIRaw")
    old_ct = load_vector_dataset(old_electrode_path, "ElecXYZRaw")
    old_mni = load_vector_dataset(old_electrode_path, "ElecXYZMNIRaw")
    transform_paths = list_ct_backwarp_transforms(new_electrode_path, subject)
    ct_backwarp = load_affine44(transform_paths[0]) if transform_paths else np.eye(4, dtype=np.float64)
    predicted_ct = apply_affine44_to_points(predicted_native, ct_backwarp) if len(predicted_native) else predicted_native
    current_pipeline_ct = apply_affine44_to_points(new_native, ct_backwarp) if len(new_native) else new_native

    detector_legacy_raw = build_detector_legacy_map(
        old_raw=old_ct,
        old_mni=old_mni,
        current_native=new_native,
        current_mni=new_mni,
        detector_native=predicted_native,
    )
    detector_legacy_raw_all, detector_legacy_anchor_mask = detector_legacy_raw
    detector_legacy_subset = greedy_select_detector_subset(
        old_raw=old_ct,
        detector_mapped_raw=detector_legacy_raw_all,
        detector_anchor_mask=detector_legacy_anchor_mask,
        mean_threshold_mm=2.0,
    )

    candidate_native_metrics = assignment_metrics(new_native, candidates.points_mm)
    detector_native_metrics = assignment_metrics(new_native, predicted_native)
    detector_metrics = assignment_metrics(old_ct, predicted_ct)
    current_pipeline_metrics = assignment_metrics(old_ct, current_pipeline_ct)
    detector_legacy_all_metrics = assignment_metrics(old_ct, detector_legacy_raw_all)
    detector_legacy_subset_metrics = assignment_metrics(old_ct, detector_legacy_subset)
    expected_counts_map = detector.load_expected_counts(case.elec_data_path)

    return {
        "subject_id": case.subject_id,
        "candidate_count": int(len(candidates.points_mm)),
        "surface_support_fraction": float(candidates.surface_support_fraction),
        "intracranial_support_fraction": float(candidates.intracranial_support_fraction),
        "predicted_electrodes": int(len(shafts)),
        "predicted_total_contacts": int(len(predicted_native)),
        "predicted_counts": sorted([int(shaft.inferred_contacts) for shaft in shafts], reverse=True),
        "expected_electrodes": int(len(expected_counts_map)),
        "expected_total_contacts": int(sum(expected_counts_map.values())),
        "old_ct_contacts": int(len(old_ct)),
        "current_pipeline_native_contacts": int(len(new_native)),
        "transform_path": str(transform_paths[0]) if transform_paths else "",
        "candidates_vs_current_pipeline_native": candidate_native_metrics,
        "detector_vs_current_pipeline_native": detector_native_metrics,
        "detector_vs_old_ct": detector_metrics,
        "current_pipeline_vs_old_ct": current_pipeline_metrics,
        "detector_vs_old_legacy_supervised_all": detector_legacy_all_metrics,
        "detector_vs_old_legacy_supervised_subset": detector_legacy_subset_metrics,
        "detector_vs_old_legacy_supervised_subset_contacts": int(len(detector_legacy_subset)),
        "detector_vs_old_legacy_supervised_anchor_contacts": int(np.count_nonzero(detector_legacy_anchor_mask)),
        "delta_mean_vs_current": float(detector_metrics["assignment_mean"] - current_pipeline_metrics["assignment_mean"]),
        "delta_within_5_vs_current": float(
            detector_metrics["assignment_within_5"] - current_pipeline_metrics["assignment_within_5"]
        ),
    }


def build_file_maps() -> tuple[dict[str, Path], dict[str, Path]]:
    old_files = sorted({path for pattern in OLD_FILE_PATTERNS for path in glob.glob(pattern)})
    new_files = sorted({path for pattern in NEW_FILE_PATTERNS for path in glob.glob(pattern)})
    return choose_largest_file_per_subject(old_files), choose_largest_file_per_subject(new_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark detector outputs against old LeGUI raw and legacy-calibrated contacts.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("elec_data.csv"),
        help="Path to the case table CSV.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Optional subject filter, e.g. sub-sub72 or 72.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    old_map, new_map = build_file_maps()
    requested_subject = subject_id_from_path(args.subject) if args.subject is not None else None

    rows: list[dict[str, object]] = []
    for case in detector.load_case_table(args.csv):
        subject = subject_id_from_path(case.subject_id)
        if requested_subject is not None and subject != requested_subject:
            continue
        if subject not in old_map or subject not in new_map:
            continue
        rows.append(benchmark_subject(case, old_map[subject], new_map[subject]))

    print(json.dumps(rows, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
