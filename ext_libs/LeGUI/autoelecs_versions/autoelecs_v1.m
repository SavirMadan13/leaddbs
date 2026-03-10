function [WC, T] = LeG_autoElecs(app)
% LeG_autoElecs  Automatic sEEG electrode contact detection from CT.
%
%   [WC, T] = LeG_autoElecs(app)
%
%   Detects metallic sEEG electrode contacts using a four-phase pipeline
%   that is robust to variations in image quality, resolution, and HU
%   calibration.  Unlike pure-threshold approaches, this method combines
%   scale-space blob analysis with geometric modelling of electrode shafts.
%
%   ALGORITHM OVERVIEW
%   ------------------
%   Phase 1 – Multi-scale blob detection + adaptive thresholding
%             Laplacian-of-Gaussian (via DoG) highlights bright, compact
%             objects at the physical scale of sEEG contacts (~0.8 mm dia).
%             Candidates are extracted at multiple thresholds on a combined
%             intensity-blob score and merged across levels.
%
%   Phase 2 – RANSAC electrode-shaft discovery
%             Sequential RANSAC finds collinear groups of candidates
%             corresponding to individual electrode shafts.
%
%   Phase 3 – Regular-spacing grid fit & gap filling
%             For each shaft the inter-contact spacing is robustly
%             estimated and a phase-aligned grid is fit.  Missing contacts
%             are interpolated and validated against the local CT intensity.
%
%   Phase 4 – Deduplication, confidence ranking, output
%
%   INPUTS
%     app  – LeGUI application handle carrying at minimum:
%              .CTImg           3-D normalised CT volume
%              .CTRng           [min  p99  max  nearMax]
%              .CTInfo          NIfTI header (with .mat, .pinfo)
%              .MRInfo          MR header   (with .mat)
%              .XYZScale        voxel scaling factors
%              .ProjSurfRaw     brain-surface mesh (.vertices, .faces)
%              .detect.maxelecs upper bound on total contacts expected
%              .PatientIDStr    string identifier for diagnostics
%              .SaveDir         folder for diagnostic output
%
%   OPTIONAL (app.detect) — phase flags; if absent, default is true:
%              .detect.usePhase2  run RANSAC shaft discovery (default true)
%              .detect.usePhase3  run spacing model & gap filling (default true)
%              .detect.usePhase4  run deduplication & MaxElecs cap (default true)
%              Set any to false to stop after earlier phases (e.g. usePhase2=false
%              returns merged blob contacts only, often closer to true contact count).
%
%   OUTPUTS
%     WC  – [N x 3] detected contact positions (1-based voxel coordinates)
%     T   – representative normalised intensity threshold (backward compat.)

StartTime = tic;

fprintf('\n[AutoElec] ========== Starting electrode detection ==========\n');

% === Unpack inputs ========================================================
Img         = single(app.CTImg);
MaxElecs    = app.detect.maxelecs;
MRInfo      = app.MRInfo;
ProjSurfRaw = app.ProjSurfRaw;
imSz        = size(Img);

% Voxel dimensions in mm  [dx  dy  dz]
voxDim = sqrt(sum(app.CTInfo.mat(1:3,1:3).^2));

fprintf('[AutoElec] Image size: %d x %d x %d, voxel dims: %.2f x %.2f x %.2f mm\n', ...
    imSz(1), imSz(2), imSz(3), voxDim(1), voxDim(2), voxDim(3));
fprintf('[AutoElec] MaxElecs = %d\n', MaxElecs);

% Phase flags: which pipeline stages to run (default all on for backward compat)
if isfield(app.detect, 'usePhase2')
    usePhase2 = logical(app.detect.usePhase2);
else
    usePhase2 = true;
end
if isfield(app.detect, 'usePhase3')
    usePhase3 = logical(app.detect.usePhase3);
else
    usePhase3 = true;
end
if isfield(app.detect, 'usePhase4')
    usePhase4 = logical(app.detect.usePhase4);
else
    usePhase4 = false;
end
fprintf('[AutoElec] Phase flags: usePhase2=%d  usePhase3=%d  usePhase4=%d\n', ...
    usePhase2, usePhase3, usePhase4);

% Raw threshold sweep kept for the backward-compatible diagnostic plot
TMax  = (app.CTRng(4)-app.CTRng(1)) / (app.CTRng(2)-app.CTRng(1));
TMin  = 1;
ThrR  = linspace(TMax, TMin, 21);  ThrR(1) = [];
ThrHU = ThrR*(app.CTRng(2)-app.CTRng(1)) + app.CTRng(1);
ThrHU = (ThrHU - app.CTInfo.pinfo(2)) ./ app.CTInfo.pinfo(1);

% =========================================================================
%  PHASE 1 — MULTI-SCALE BLOB DETECTION
% =========================================================================

% 1a  Robust intensity percentiles (ignore background <= 0) ---------------
vValid = Img(Img > 0);
if isempty(vValid)
    WC = []; T = 0;
    diagPlot(app, ThrHU, zeros(size(ThrHU)), 1, toc(StartTime));
    return;
end
pct  = prctile(vValid(:), [50 75 90 95 99 99.9]);
pMed = pct(1);  p75 = pct(2);  p90 = pct(3);  p95 = pct(4);  p99 = pct(5);  p99_9 = pct(6);

fprintf('[AutoElec] Intensity percentiles — p50=%.1f  p75=%.1f  p90=%.1f  p95=%.1f  p99=%.1f  p99.9=%.1f\n', ...
    pct(1), pct(2), pct(3), pct(4), pct(5), pct(6));

% 1b  Laplacian-of-Gaussian via Difference-of-Gaussians -------------------
%     DoG ≈ sigma^2 * LoG.  Anisotropic smoothing (sigma per axis) so that
%     thick-slice / oblique electrodes still get a strong blob response.
%     Sigma in mm, then converted to voxel units per axis.
nScales = 4;
sigmas_mm = linspace(0.4, 1.2, nScales);

blob = zeros(imSz, 'single');
for s = 1:nScales
    sigma_mm = sigmas_mm(s);
    sigma_vox = sigma_mm ./ voxDim(:)';           % [sx sy sz] in voxel units per axis
    g1 = imgaussfilt3(Img, sigma_vox);
    g2 = imgaussfilt3(Img, sigma_vox * 1.6);
    blob = max(blob, -(g2 - g1) * sigma_mm^2);
end

bClip = prctile(blob(blob > 0), 99.9);
if bClip > 0
    blob = blob / bClip;
end

fprintf('[AutoElec] LoG blob detection: %d scales, anisotropic (sigma_mm = %.2f – %.2f)\n', ...
    nScales, sigmas_mm(1), sigmas_mm(end));
fprintf('[AutoElec]   Blob response range: [%.3f, %.3f], clip = %.3f\n', ...
    min(blob(:)), max(blob(:)), bClip);

% 1c  Combined score: normalised intensity x blob response -----------------
%     Scale by p99.9 so the very brightest voxels can exceed 1 (no cap).
%     That avoids missing the brightest contacts when LoG is weak (metal bloom).
%     The floor of 0.1 on blob still helps weak-blob bright spots.
intNorm   = max(0, (Img - pMed) / max(p99_9 - pMed, 1e-6));
combScore = intNorm .* max(blob, 0.1);

% 1d  Volume bounds for a single contact -----------------------------------
%     sEEG contact geometry: cylinder ~0.8 mm diameter, ~1.5 mm long
%     Physical volume ≈ pi * 0.4^2 * 1.5 ≈ 0.75 mm^3
%     Generous bounds accommodate partial-volume effects and metal bloom.
voxVol = prod(voxDim);
minVox = max(2, round(0.75 * 0.15 / voxVol));
maxVox = max(60, round(0.75 * 45 / voxVol));

fprintf('[AutoElec] Volume bounds: [%d, %d] voxels (voxVol = %.3f mm^3)\n', ...
    minVox, maxVox, voxVol);

% 1e  Threshold sweep on the combined score --------------------------------
%     By sweeping on the combined intensity-blob score (rather than raw
%     intensity alone), detection is largely independent of absolute HU
%     calibration.  Candidates from ALL thresholds are collected and merged.
nThr    = 18;
thrLevs = linspace(0.05, 0.9, nThr);

% Accumulator: each row = [row  col  slice  meanIntensity  blobVal  thrIdx]
allCand = zeros(0, 6);

fprintf('[AutoElec] Sweeping %d thresholds on combined score [%.2f – %.2f]...\n', ...
    nThr, thrLevs(1), thrLevs(end));

for k = 1:nThr
    CC = bwconncomp(combScore > thrLevs(k), 26);
    if CC.NumObjects == 0, continue; end

    nRawCC = CC.NumObjects;
    ccSz = cellfun(@numel, CC.PixelIdxList);
    ccSz = ccSz(:);   % ensure column for element-wise logic
    % Get MeanIntensity for all CCs so we can keep very bright small blobs
    propsAll = regionprops3(CC, Img, ...
        'WeightedCentroid', 'MeanIntensity', 'PrincipalAxisLength');
    miAll = propsAll.MeanIntensity(:);   % ensure column, same length as ccSz
    keep = (ccSz >= minVox & ccSz <= maxVox) | ...
           (ccSz >= 1 & ccSz <= 3 & miAll >= p99_9);
    CC.PixelIdxList = CC.PixelIdxList(keep);
    CC.NumObjects   = sum(keep);
    nVolOK = CC.NumObjects;
    if CC.NumObjects == 0, continue; end

    % Shape filter — reject very elongated blobs (streak artefacts)
    pal = propsAll.PrincipalAxisLength(keep, :);     % [N x 3] descending
    if size(pal, 2) == 3
        elongation = pal(:,1) ./ max(pal(:,3), 0.1);
        goodShape  = elongation < 8;
    else
        goodShape  = true(CC.NumObjects, 1);
    end
    nShapeOK = sum(goodShape);

    wc = propsAll.WeightedCentroid(keep, :);
    wc = wc(goodShape, :);                           % [col  row  slice]
    mi = propsAll.MeanIntensity(keep);
    mi = mi(goodShape);
    if isempty(wc), continue; end

    wc(:, [1 2]) = wc(:, [2 1]);                     % → [row  col  slice]

    % Keep only detections inside the brain / skull envelope, or very bright
    % (likely electrode at boundary — don't drop brightest contacts)
    wcMM = [wc, ones(size(wc,1),1)] * MRInfo.mat';
    wcMM(:,4) = [];
    inBrain = LeG_intriangulation(ProjSurfRaw.vertices, ...
                                  ProjSurfRaw.faces, wcMM);
    keepBrain = inBrain | (mi >= p99_9);
    wc = wc(keepBrain, :);
    mi = mi(keepBrain);
    nBrainOK = size(wc, 1);
    if isempty(wc), continue; end

    % Look up the blob value at each centroid
    rc = max(min(round(wc), imSz), 1);
    linIdx = sub2ind(imSz, rc(:,1), rc(:,2), rc(:,3));
    bv = blob(linIdx);

    allCand = [allCand; ...
        wc, double(mi), double(bv), k*ones(size(wc,1),1)]; %#ok<AGROW>

    fprintf('[AutoElec]   Thr %2d (%.2f): %d raw CC -> %d vol-OK -> %d shape-OK -> %d in-brain\n', ...
        k, thrLevs(k), nRawCC, nVolOK, nShapeOK, nBrainOK);
end

fprintf('[AutoElec] Total raw candidates across all thresholds: %d\n', size(allCand, 1));

% --- Raw-threshold detection count curve (for the diagnostic plot) --------
NumObj = zeros(numel(ThrR), 1);
for kt = 1:numel(ThrR)
    cc0  = bwconncomp(Img > ThrR(kt), 26);
    cs0  = cellfun(@numel, cc0.PixelIdxList);
    NumObj(kt) = sum(cs0 >= minVox & cs0 <= maxVox);
end

if isempty(allCand)
    WC = []; T = 0;
    diagPlot(app, ThrHU, NumObj, round(numel(ThrR)/2), toc(StartTime));
    return;
end

% 1f  Merge duplicate detections across thresholds -------------------------
%     The same physical contact is typically detected at many thresholds.
%     Agglomerative clustering with a 1.5 mm cut-off merges these, and the
%     final centroid is a confidence-weighted average of the group.
candVox  = allCand(:, 1:3);
candMM   = [candVox, ones(size(candVox,1),1)] * MRInfo.mat';
candMM(:,4) = [];
candInt  = allCand(:, 4);
candBlob = allCand(:, 5);

% Pre-filter: keep top 3000 candidates (by intensity*blob), but never drop
% the very brightest (intensity >= p99.9) so we don't miss bright electrodes.
maxNCand = 3000;
nBeforeCap = size(candMM, 1);
if size(candMM, 1) > maxNCand
    protected = (candInt >= p99_9);
    nProt = sum(protected);
    rawScore = candInt .* max(candBlob, 0.01);
    if nProt >= maxNCand
        selProt = find(protected);
        [~, sIdx] = sort(rawScore(protected), 'descend');
        sel = selProt(sIdx(1:maxNCand));
    else
        selProt = find(protected);
        rest = find(~protected);
        [~, sIdxRest] = sort(rawScore(rest), 'descend');
        nTake = min(maxNCand - nProt, numel(rest));
        sel = [selProt; rest(sIdxRest(1:nTake))];
    end
    candMM  = candMM(sel, :);
    candVox = candVox(sel, :);
    candInt = candInt(sel);
    candBlob = candBlob(sel);
    allCand  = allCand(sel, :);
    fprintf('[AutoElec] Pre-filter: capped %d candidates to %d (top by score; %d brightest protected)\n', ...
        nBeforeCap, size(candMM, 1), nProt);
end

if size(candMM, 1) > 1
    Z   = linkage(candMM, 'average');
    cID = cluster(Z, 'cutoff', 1.5, 'criterion', 'distance');
else
    cID = 1;
end

nGroups = max(cID);
mrgMM   = zeros(nGroups, 3);
mrgVox  = zeros(nGroups, 3);
mrgInt  = zeros(nGroups, 1);
mrgBlob = zeros(nGroups, 1);
mrgCnt  = zeros(nGroups, 1);

for g = 1:nGroups
    mask = cID == g;
    w = candInt(mask) .* max(candBlob(mask), 0.01);
    w = w / sum(w);
    mrgMM(g, :)  = w' * candMM(mask, :);
    mrgVox(g, :) = w' * candVox(mask, :);
    mrgInt(g)    = max(candInt(mask));
    mrgBlob(g)   = max(candBlob(mask));
    mrgCnt(g)    = sum(mask);
end

% Confidence score: persistence across thresholds + intensity + blob shape
detectFrac = mrgCnt / nThr;
intScore   = (mrgInt - min(mrgInt)) / max(max(mrgInt) - min(mrgInt), 1e-6);
confidence = 0.4 * detectFrac + 0.3 * intScore + 0.3 * mrgBlob;

fprintf('[AutoElec] Cross-threshold merge: %d candidates -> %d unique contacts\n', ...
    nBeforeCap, nGroups);
fprintf('[AutoElec] Confidence distribution: min=%.2f  median=%.2f  max=%.2f  (mean=%.2f)\n', ...
    min(confidence), median(confidence), max(confidence), mean(confidence));
fprintf('[AutoElec]   Contacts with conf > 0.5: %d,  conf > 0.3: %d,  conf <= 0.3: %d\n', ...
    sum(confidence > 0.5), sum(confidence > 0.3 & confidence <= 0.5), sum(confidence <= 0.3));
fprintf('[AutoElec] Phase 1 complete (%.1f s)\n', toc(StartTime));

% ---------- Optional early exit: Phase 1 only (merged blob contacts) ------
% Optionally keep only contacts with confidence > phase1ConfMin (e.g. 0.5) at this stage.
if ~usePhase2
    fprintf('\n[AutoElec] Phase 2 disabled — returning merged contacts from Phase 1 only\n');
    phase1ConfMin = 0.5;   % no filter by default; set app.detect.phase1ConfMin = 0.5 to keep only conf > 0.5
    if isfield(app.detect, 'phase1ConfMin') && ~isempty(app.detect.phase1ConfMin)
        phase1ConfMin = double(app.detect.phase1ConfMin(1));
    end
    if phase1ConfMin > 0
        keepConf = (confidence > phase1ConfMin);
        nBefore = numel(confidence);
        mrgMM = mrgMM(keepConf, :);
        confidence = confidence(keepConf);
        fprintf('[AutoElec] Phase 1 confidence filter (conf > %.2f): %d -> %d contacts\n', ...
            phase1ConfMin, nBefore, numel(confidence));
    end
    if isempty(mrgMM)
        WC = []; T = (TMax + TMin) / 2;
        diagPlot(app, ThrHU, NumObj, round(numel(ThrR)/2), toc(StartTime));
        fprintf('\n[AutoElec] No contacts left after confidence filter. Exiting.\n\n');
        return;
    end
    [~, sortIdx] = sort(confidence, 'descend');
    finalMM_early = mrgMM(sortIdx, :);
    if size(finalMM_early, 1) > MaxElecs
        finalMM_early = finalMM_early(1:MaxElecs, :);
        fprintf('[AutoElec] Capped at MaxElecs: %d contacts\n', MaxElecs);
    end
    voxH = MRInfo.mat \ [finalMM_early, ones(size(finalMM_early, 1), 1)]';
    WC   = round(voxH(1:3, :)');
    T    = (TMax + TMin) / 2;
    EndTime = toc(StartTime);
    [~, idxD] = min(abs(NumObj - size(WC, 1)));
    diagPlot(app, ThrHU, NumObj, idxD, EndTime);
    if phase1ConfMin > 0
        fprintf('\n[AutoElec] ========== FINAL (Phase 1 only, conf > %.2f): %d contacts in %.1f s ==========\n\n', ...
            phase1ConfMin, size(WC, 1), EndTime);
    else
        fprintf('\n[AutoElec] ========== FINAL (Phase 1 only): %d contacts in %.1f s ==========\n\n', ...
            size(WC, 1), EndTime);
    end
    return;
end

% =========================================================================
%  PHASE 2 — RANSAC ELECTRODE-SHAFT DISCOVERY
% =========================================================================
%  sEEG electrodes are approximately straight: contacts are collinear.
%  Sequential RANSAC discovers these linear structures.  Each pair of
%  candidate points proposes a line; inliers within a 2 mm tube vote.
%  After two passes of PCA refinement the final inlier set is locked in
%  and removed before finding the next shaft.

nPts = size(mrgMM, 1);
if nPts < 3
    WC = round(mrgVox);
    T  = (TMax + TMin) / 2;
    diagPlot(app, ThrHU, NumObj, round(numel(ThrR)/2), toc(StartTime));
    return;
end

inlierRadius = 2.5;            % mm – tube radius for line inlier test (slight loosen for localization jitter)
minShaftPts  = 2;               % allow 2-point shafts so underdetected electrodes are kept
minSpanMm    = 6;               % require 2-point shafts to span at least this (avoid junk shafts)
maxShafts    = ceil(MaxElecs/3);
assigned     = false(nPts, 1);
shafts       = struct('idx',{}, 'dir',{}, 'mu',{});

rngOld = rng;
rng(42, 'twister');             % reproducible RANSAC sampling

fprintf('\n[AutoElec] === PHASE 2: RANSAC shaft discovery ===\n');
fprintf('[AutoElec] %d candidates to cluster, inlier radius = %.1f mm, min %d pts/shaft (2-pt span >= %.0f mm)\n', ...
    nPts, inlierRadius, minShaftPts, minSpanMm);

for sIter = 1:maxShafts
    uIdx = find(~assigned);
    nU   = numel(uIdx);
    if nU < minShaftPts, break; end

    ptsU  = mrgMM(uIdx, :);
    confU = confidence(uIdx);

    bestInliers = [];
    bestScore   = 0;
    nIter = min(5000, nU*(nU-1)/2);

    for it = 1:nIter
        % Bias sampling toward higher-confidence points so weaker shafts get found
        w = confU(:) + 1e-3;
        w = w / sum(w);
        i1 = randsample(nU, 1, true, w);
        i2 = randsample(nU, 1, true, w);
        if i1 == i2, continue; end
        pair = [i1, i2];
        lineDir = ptsU(pair(2),:) - ptsU(pair(1),:);
        lineLen = norm(lineDir);
        if lineLen < 2, continue; end              % too close — skip
        lineDir = lineDir / lineLen;

        vecs = ptsU - ptsU(pair(1), :);
        proj = vecs * lineDir';
        perpDist = sqrt(max(sum(vecs.^2, 2) - proj.^2, 0));

        inMask = perpDist < inlierRadius;
        if sum(inMask) < minShaftPts, continue; end
        % 2-point shafts: require minimum span so we don't create nonsense shafts
        if sum(inMask) == 2
            twoPts = ptsU(inMask, :);
            span = norm(twoPts(2,:) - twoPts(1,:));
            if span < minSpanMm, continue; end
        end

        score = sum(confU(inMask));
        if score > bestScore
            bestScore   = score;
            bestInliers = uIdx(inMask);
        end
    end

    if isempty(bestInliers), break; end

    % Two-pass PCA refinement: re-fit line and re-evaluate inliers
    for pass_ = 1:2
        sPts = mrgMM(bestInliers, :);
        mu   = mean(sPts, 1);
        [~, ~, V] = svd(sPts - mu, 'econ');
        lineDir = V(:,1)';

        vecsAll  = mrgMM(uIdx, :) - mu;
        projAll  = vecsAll * lineDir';
        perpAll  = sqrt(max(sum(vecsAll.^2, 2) - projAll.^2, 0));
        refined  = uIdx(perpAll < inlierRadius);
        if numel(refined) >= minShaftPts
            bestInliers = refined;
        end
    end

    % Final PCA on the locked-in inliers
    sPts = mrgMM(bestInliers, :);
    mu   = mean(sPts, 1);
    [~, ~, V] = svd(sPts - mu, 'econ');

    shafts(end+1) = struct('idx', bestInliers, ...
                           'dir', V(:,1)', ...
                           'mu',  mu);              %#ok<AGROW>
    assigned(bestInliers) = true;

    % Shaft diagnostics
    t_diag = (sPts - mu) * V(:,1);
    shaftLen = max(t_diag) - min(t_diag);
    meanConf = mean(confidence(bestInliers));
    fprintf('[AutoElec]   Shaft %d: %d contacts, span = %.1f mm, mean conf = %.2f, dir = [%.2f, %.2f, %.2f]\n', ...
        numel(shafts), numel(bestInliers), shaftLen, meanConf, V(1,1), V(2,1), V(3,1));
end

% Extend shafts: assign unassigned points that lie on an existing shaft line
% (within inlierRadius and near the shaft span); keep margin tight to avoid sparse false contacts
extendRadius = inlierRadius;
extendMargin = 5;   % mm beyond shaft endpoint to still count as "on shaft"
uList = find(~assigned);
for i = 1:numel(uList)
    u = uList(i);
    pt = mrgMM(u, :);
    bestS = [];
    bestPerp = inf;
    for s = 1:numel(shafts)
        sPts = mrgMM(shafts(s).idx, :);
        mu = mean(sPts, 1);
        dir = shafts(s).dir;
        tShaft = (sPts - mu) * dir';
        tMin = min(tShaft); tMax = max(tShaft);
        vec = pt - mu;
        proj = vec * dir';
        perpDist = sqrt(max(sum(vec.^2) - proj^2, 0));
        if perpDist < extendRadius && proj >= tMin - extendMargin && proj <= tMax + extendMargin
            if perpDist < bestPerp
                bestPerp = perpDist;
                bestS = s;
            end
        end
    end
    if ~isempty(bestS)
        shafts(bestS).idx = [shafts(bestS).idx; u];
        assigned(u) = true;
    end
end

rng(rngOld);                    % restore RNG state

nUnassigned = sum(~assigned);
fprintf('[AutoElec] RANSAC complete: %d shafts found, %d contacts assigned, %d unassigned (shaft extension applied)\n', ...
    numel(shafts), sum(assigned), nUnassigned);
fprintf('[AutoElec] Phase 2 complete (%.1f s)\n', toc(StartTime));

% =========================================================================
%  PHASE 3 — REGULAR-SPACING MODEL FIT & GAP FILLING (or shaft-only output)
% =========================================================================

finalMM     = zeros(0, 3);
finalConf   = zeros(0, 1);
finalShaftID = zeros(0, 1);   % 1..numel(shafts) per shaft, 0 = singleton

if ~usePhase3
    % Phase 3 disabled: use only RANSAC-assigned contacts per shaft (no grid/gap fill)
    fprintf('\n[AutoElec] Phase 3 disabled — using RANSAC shaft contacts only (no gap filling)\n');
    for s = 1:numel(shafts)
        sIdx = shafts(s).idx;
        nC = numel(sIdx);
        finalMM      = [finalMM;      mrgMM(sIdx, :)];  %#ok<AGROW>
        finalConf    = [finalConf;    confidence(sIdx)]; %#ok<AGROW>
        finalShaftID = [finalShaftID; s*ones(nC, 1)];  %#ok<AGROW>
    end
    unassigned = find(~assigned);
    nSingletons = 0;
    for u = 1:numel(unassigned)
        idxU = unassigned(u);
        if confidence(idxU) > 0.6 || mrgInt(idxU) >= p99_9
            finalMM      = [finalMM;      mrgMM(idxU, :)];       %#ok<AGROW>
            finalConf    = [finalConf;    confidence(idxU)];      %#ok<AGROW>
            finalShaftID = [finalShaftID; 0];                     %#ok<AGROW>
            nSingletons = nSingletons + 1;
        end
    end
    fprintf('[AutoElec] Shafts-only: %d contacts from shafts + %d singletons = %d total\n', ...
        size(finalMM, 1) - nSingletons, nSingletons, size(finalMM, 1));
    fprintf('[AutoElec] Phase 3 skipped (%.1f s)\n', toc(StartTime));
else
%  Phase 3 enabled: GAP-BASED filling only.
%  Purpose: We have N detected contacts. Real electrodes have regular
%  spacing; sometimes one contact is missed (weak blob). So we should ONLY
%  add interpolated contacts inside gaps that are clearly 2x or 3x the
%  spacing (one or two missed contacts). We must NOT lay a full grid over
%  the whole span — that wrongly assumes a contact every baseSp mm and
%  creates hundreds of false positives.
%  Steps:
%    3a. Estimate inter-contact spacing from observed gaps.
%    3b. For each consecutive pair of detected contacts, if gap ≈ k*baseSp
%        with k>=2, add (k-1) interpolated positions in that gap, each
%        validated by local intensity. Cap fill per gap and per shaft.

fprintf('\n[AutoElec] === PHASE 3: Gap-based fill only (no full-span grid) ===\n');

for s = 1:numel(shafts)
    sIdx = shafts(s).idx;
    sDir = shafts(s).dir;
    sMu  = shafts(s).mu;

    sPts  = mrgMM(sIdx, :);
    sConf = confidence(sIdx);

    % Project onto shaft axis
    t = (sPts - sMu) * sDir';
    [t, order] = sort(t);
    sPts  = sPts(order, :);
    sConf = sConf(order);
    nS    = numel(t);

    if nS < 2
        finalMM      = [finalMM;      sPts];   %#ok<AGROW>
        finalConf    = [finalConf;    sConf];   %#ok<AGROW>
        finalShaftID = [finalShaftID; s*ones(nS, 1)]; %#ok<AGROW>
        continue;
    end

    % --- 3a  Robust spacing estimation ------------------------------------
    gaps    = diff(t);
    posGaps = gaps(gaps > 0.5);          % ignore tiny residual gaps
    if isempty(posGaps)
        finalMM      = [finalMM;      sPts];   %#ok<AGROW>
        finalConf    = [finalConf;    sConf];   %#ok<AGROW>
        finalShaftID = [finalShaftID; s*ones(nS, 1)]; %#ok<AGROW>
        continue;
    end

    % Candidate spacings: observed gaps and their integer sub-divisions
    candSp = posGaps(:);
    for div = 2:4
        candSp = [candSp; posGaps(:)/div]; %#ok<AGROW>
    end
    candSp = candSp(candSp >= 1.5 & candSp <= 12);  % sEEG range (mm)

    if isempty(candSp)
        baseSp = median(posGaps);
    else
        scores = zeros(size(candSp));
        for cs = 1:numel(candSp)
            ratios   = posGaps / candSp(cs);
            rounded  = max(round(ratios), 1);
            residual = abs(ratios - rounded);
            scores(cs) = sum(residual < 0.25) - 0.5 * sum(residual);
        end
        [~, bestIdx] = max(scores);
        baseSp = candSp(bestIdx);
    end
    if baseSp < 1, baseSp = 3.5; end    % safe fallback

    % --- 3b  Gap-based fill only ------------------------------------------
    % Output list: start with first detected contact, then for each
    % consecutive pair (t_i, t_{i+1}), if gap suggests missed contacts
    % (gap in [1.5*baseSp, 2.5*baseSp] -> 1 missed, [2.5*baseSp, 3.5*baseSp] -> 2, etc.),
    % insert that many interpolated positions. Cap at maxFillPerGap
    % and ensure total contacts per shaft <= nS + reasonable.
    maxFillPerGap = 3;                  % allow up to 3 filled contacts per gap (dimmer mid-lead)
    maxTotalShaft  = nS + 8;           % total contacts on shaft at most nS+8

    shaftMM   = sPts(1, :);             % first contact
    shaftConf = sConf(1);

    nInterp   = 0;
    nInterpFail = 0;

    for i = 1:(nS - 1)
        gap = t(i+1) - t(i);
        % How many contacts are missing between t(i) and t(i+1)?
        % gap ≈ (nMissing+1)*baseSp  =>  nMissing = round(gap/baseSp) - 1
        nMissing = round(gap / baseSp) - 1;
        % Only fill if gap is clearly larger than one spacing (residual small)
        ratio = gap / baseSp;
        if nMissing < 1 || ratio < 1.35
            % No fill: gap is ~1 spacing or less
            shaftMM   = [shaftMM;   sPts(i+1, :)]; %#ok<AGROW>
            shaftConf = [shaftConf; sConf(i+1)];   %#ok<AGROW>
            continue;
        end
        nMissing = min(nMissing, maxFillPerGap);
        % Cap so shaft total stays <= maxTotalShaft (already have shaftMM; will add nMissing + rest of sPts)
        remainingDetected = nS - i;  % sPts(i+1)..sPts(nS) still to add
        if size(shaftMM, 1) + nMissing + remainingDetected > maxTotalShaft
            nMissing = max(0, maxTotalShaft - size(shaftMM, 1) - remainingDetected);
        end
        if nMissing < 1
            shaftMM   = [shaftMM;   sPts(i+1, :)]; %#ok<AGROW>
            shaftConf = [shaftConf; sConf(i+1)];   %#ok<AGROW>
            continue;
        end
        % Insert nMissing positions evenly between t(i) and t(i+1)
        for k = 1:nMissing
            tau = t(i) + (gap * k / (nMissing + 1));
            interpPt  = sMu + tau * sDir;
            interpVox = MRInfo.mat \ [interpPt'; 1];
            interpVox = round(interpVox(1:3))';

            if all(interpVox >= 1) && ...
               interpVox(1) <= imSz(1) && ...
               interpVox(2) <= imSz(2) && ...
               interpVox(3) <= imSz(3)

                [di, dj, dk] = ndgrid(-1:1, -1:1, -1:1);
                nbr = interpVox + [di(:), dj(:), dk(:)];
                nbr = max(nbr, 1);
                nbr(:,1) = min(nbr(:,1), imSz(1));
                nbr(:,2) = min(nbr(:,2), imSz(2));
                nbr(:,3) = min(nbr(:,3), imSz(3));
                nbrIdx    = sub2ind(imSz, nbr(:,1), nbr(:,2), nbr(:,3));
                localMean = mean(Img(nbrIdx));
                localBlob = mean(blob(nbrIdx));

                % Require clearly elevated intensity and blob; no dimmer tier to avoid spurious contacts
                if localMean > p90 && localBlob > 0.03
                    shaftMM   = [shaftMM;   interpPt]; %#ok<AGROW>
                    shaftConf = [shaftConf; 0.25];     %#ok<AGROW>
                    nInterp = nInterp + 1;
                else
                    nInterpFail = nInterpFail + 1;
                end
            else
                nInterpFail = nInterpFail + 1;
            end
        end

        shaftMM   = [shaftMM;   sPts(i+1, :)]; %#ok<AGROW>
        shaftConf = [shaftConf; sConf(i+1)];   %#ok<AGROW>
    end

    nDetected = nS;
    nTotal = size(shaftMM, 1);
    fprintf('[AutoElec]   Shaft %d: %d detected, %d gap-filled -> %d total (%d interp rejected)\n', ...
        s, nDetected, nInterp, nTotal, nInterpFail);

    finalMM      = [finalMM;      shaftMM];   %#ok<AGROW>
    finalConf    = [finalConf;    shaftConf];  %#ok<AGROW>
    finalShaftID = [finalShaftID; s*ones(nTotal, 1)]; %#ok<AGROW>
end

fprintf('[AutoElec] All shafts processed: %d contacts from shafts\n', size(finalMM, 1));

% Add singletons: only confident or very bright unassigned (avoid sparse false positives)
unassigned = find(~assigned);
nSingletons = 0;
for u = 1:numel(unassigned)
    idxU = unassigned(u);
    if confidence(idxU) > 0.6 || mrgInt(idxU) >= p99_9
        finalMM      = [finalMM;      mrgMM(idxU, :)];       %#ok<AGROW>
        finalConf    = [finalConf;    confidence(idxU)];      %#ok<AGROW>
        finalShaftID = [finalShaftID; 0];                     %#ok<AGROW>
        nSingletons = nSingletons + 1;
    end
end

fprintf('[AutoElec] Singletons: %d unassigned, %d added (conf > 0.6 or intensity >= p99.9)\n', ...
    numel(unassigned), nSingletons);
fprintf('[AutoElec] Entering Phase 4 with %d contacts total\n', size(finalMM, 1));
fprintf('[AutoElec] Phase 3 complete (%.1f s)\n', toc(StartTime));

end  % usePhase3

% Snapshot for diagnostics: contacts after Phase 3 (before bone/linearity filter)
finalMM_afterPhase3   = finalMM;
finalConf_afterPhase3 = finalConf;
finalShaftID_afterPhase3 = finalShaftID;

% =========================================================================
%  BONE / SURFACE FILTER — remove contacts outside brain and curved "shafts"
% =========================================================================
%  (1) Contact-level: load brain surface; drop every contact that lies
%      OUTSIDE the brain (bone, skull). Then drop shafts with < 2 contacts.
%  (2) Linearity: drop shafts whose contacts form a curve/arc (e.g. bone
%      following skull) rather than a straight line (real sEEG).
if size(finalMM, 1) > 0
    % --- (1) Contact-level brain filter ---
    surPath = app.SurfacesFile;
    loaded = load(surPath);
    if isfield(loaded, 'BrainSurfRaw')
        BrainSurfRaw = loaded.BrainSurfRaw;
    elseif isfield(loaded, 'k') && isfield(loaded.k, 'BrainSurfRaw')
        BrainSurfRaw = loaded.k.BrainSurfRaw;
    else
        fns = fieldnames(loaded);
        if ~isempty(fns) && isfield(loaded.(fns{1}), 'BrainSurfRaw')
            BrainSurfRaw = loaded.(fns{1}).BrainSurfRaw;
        else
            BrainSurfRaw = [];
        end
    end
    if ~isempty(BrainSurfRaw) && isfield(BrainSurfRaw, 'vertices') && isfield(BrainSurfRaw, 'faces')
        BrainV = BrainSurfRaw.vertices;
        BrainF = BrainSurfRaw.faces;
        % Per-contact: keep only contacts INSIDE the brain (drop bone)
        inBrain = LeG_intriangulation(BrainV, BrainF, finalMM);
        keep = inBrain;
        % Drop shafts that have fewer than 2 contacts left after removing bone
        shaftIDs = unique(finalShaftID(finalShaftID > 0));
        for s = shaftIDs(:)'
            if sum(keep & (finalShaftID == s)) < 2
                keep(finalShaftID == s) = false;
            end
        end
        % Singletons outside brain are also dropped (likely bone)
        nRemoved = sum(~keep);
        if nRemoved > 0
            finalMM      = finalMM(keep, :);
            finalConf    = finalConf(keep);
            finalShaftID = finalShaftID(keep);
            fprintf('[AutoElec] Brain filter: removed %d contacts outside brain (bone/surface)\n', nRemoved);
        end
    end
elseif isfield(app, 'SurfacesFile') && ~isempty(app.SurfacesFile)
    fprintf('[AutoElec] SurfacesFile not found or invalid, skipping brain filter\n');

    % --- (2) Linearity filter: drop shafts that form a curve/arc (bone) ---
    % Real sEEG contacts lie on a straight line; bone often follows skull contour.
    shaftIDs = unique(finalShaftID(finalShaftID > 0));
    if ~isempty(shaftIDs)
        maxCurvature = 0.25;   % max allowed (max perpendicular dist / span); above = curved → remove
        removeShaftCurve = false(max(shaftIDs), 1);
        for s = shaftIDs(:)'
            idx = (finalShaftID == s);
            pts = finalMM(idx, :);
            if size(pts, 1) < 3
                continue;
            end
            mu = mean(pts, 1);
            [~, ~, V] = svd(pts - mu, 'econ');
            proj = (pts - mu) * V(:, 1);
            span = max(proj) - min(proj);
            if span < 8
                continue;   % very short shaft, skip linearity check
            end
            perpDist = sqrt(max(sum((pts - mu).^2, 2) - proj.^2, 0));
            curvature = max(perpDist) / span;
            if curvature > maxCurvature
                removeShaftCurve(s) = true;
            end
        end
        keep = true(size(finalMM, 1), 1);
        for s = shaftIDs(:)'
            if removeShaftCurve(s)
                keep(finalShaftID == s) = false;
            end
        end
        nCurve = sum(~keep);
        if nCurve > 0
            finalMM      = finalMM(keep, :);
            finalConf    = finalConf(keep);
            finalShaftID = finalShaftID(keep);
            fprintf('[AutoElec] Linearity filter: removed %d contacts (%d curved shaft(s), likely bone)\n', ...
                nCurve, sum(removeShaftCurve));
        end
    end
end

% =========================================================================
%  PHASE 4 — DEDUPLICATION, RANKING, OUTPUT
% =========================================================================

fprintf('\n[AutoElec] === PHASE 4: Deduplication & output ===\n');

if isempty(finalMM)
    WC = []; T = 0;
    diagPlot(app, ThrHU, NumObj, round(numel(ThrR)/2), toc(StartTime));
    return;
end

nAfterBoneFilter = size(finalMM, 1);   % for diagnostic funnel (after bone/linearity filter)

% Sort by confidence (highest first) — always applied
[~, sortIdx] = sort(finalConf, 'descend');
finalMM      = finalMM(sortIdx, :);
finalConf    = finalConf(sortIdx);
finalShaftID = finalShaftID(sortIdx);

if usePhase4
    % Enforce minimum inter-contact distance (sEEG spacing is typically 3–5 mm;
    % avoid unrealistically close contacts by removing the lower-confidence of any pair too close)
    minInterContactMm = 2.0;
    nBeforeDedup = size(finalMM, 1);
    if size(finalMM, 1) > 1
        D = pdist2(finalMM, finalMM);
        D(logical(eye(size(D)))) = Inf;
        toRemove = false(size(finalMM, 1), 1);

        for i = 1:size(D, 1)
            if toRemove(i), continue; end
            tooClose = find(D(i, :) < minInterContactMm & ~toRemove');
            for j = tooClose
                if j <= i, continue; end
                if finalConf(j) <= finalConf(i)
                    toRemove(j) = true;
                else
                    toRemove(i) = true;
                    break;
                end
            end
        end

        nDedupRemoved = sum(toRemove);
        finalMM      = finalMM(~toRemove, :);
        finalConf    = finalConf(~toRemove);
        finalShaftID = finalShaftID(~toRemove);
        fprintf('[AutoElec] Min inter-contact %.1f mm: removed %d too-close contacts (%d -> %d)\n', ...
            minInterContactMm, nDedupRemoved, nBeforeDedup, size(finalMM, 1));
    end

    % Cap at MaxElecs
    nBeforeCap2 = size(finalMM, 1);
    if size(finalMM, 1) > MaxElecs
        finalMM      = finalMM(1:MaxElecs, :);
        finalConf    = finalConf(1:MaxElecs);
        finalShaftID = finalShaftID(1:MaxElecs);
        fprintf('[AutoElec] MaxElecs cap: trimmed %d -> %d contacts\n', ...
            nBeforeCap2, MaxElecs);
    end
else
    fprintf('[AutoElec] Phase 4 disabled — no dedup, no MaxElecs cap\n');
end

% Convert mm back to 1-based voxel coordinates
voxH = MRInfo.mat \ [finalMM, ones(size(finalMM, 1), 1)]';
WC   = round(voxH(1:3, :)');

% Nominal threshold for backward compatibility
T = (TMax + TMin) / 2;

% === Diagnostic plots ======================================================
EndTime = toc(StartTime);
[~, idxD] = min(abs(NumObj - size(WC, 1)));
diagPlot(app, ThrHU, NumObj, idxD, EndTime);
diagPlotsAutoElecs(app, ThrHU, NumObj, mrgMM, confidence, assigned, ...
    finalMM_afterPhase3, finalConf_afterPhase3, finalShaftID_afterPhase3, ...
    nAfterBoneFilter, finalMM, finalConf, finalShaftID, EndTime);

fprintf('\n[AutoElec] ========== FINAL: %d contacts detected in %.1f s ==========\n\n', ...
    size(WC, 1), EndTime);

end


% =========================================================================
%  diagPlot — backward-compatible diagnostic figure
% =========================================================================
function diagPlot(app, ThrHU, NumObj, idx, elapsedTime)
    fH = figure('Position', [50 50 400 400], ...
                'Name',     app.PatientIDStr);
    aH = axes('Parent', fH);
    plot(aH, ThrHU, NumObj, '-');
    hold(aH, 'on');

    if idx >= 1 && idx <= numel(ThrHU)
        plot(aH, ThrHU(idx), NumObj(idx), 'or');
    end

    xlabel(aH, 'Threshold (HU)');
    ylabel(aH, '# Detections');

    TMaxHU = (app.CTRng(4) - app.CTInfo.pinfo(2)) ./ app.CTInfo.pinfo(1);
    TMinHU = (app.CTRng(3) - app.CTInfo.pinfo(2)) ./ app.CTInfo.pinfo(1);
    title(aH, sprintf('%0.1f s  (%0.0f, %0.0f) HU', ...
          elapsedTime, TMinHU, TMaxHU));
end


% =========================================================================
%  diagPlotsAutoElecs — diagnostic figures for sensitivity / scan comparison
% =========================================================================
%  Opens figures: confidence distribution, 3D contacts (colored by confidence/shaft),
%  optional surface overlay, inter-contact distance distribution.
function diagPlotsAutoElecs(app, ThrHU, NumObj, mrgMM, confidence, assigned, ...
    finalMM_afterPhase3, finalConf_afterPhase3, finalShaftID_afterPhase3, ...
    nAfterBoneFilter, finalMM, finalConf, finalShaftID, elapsedTime)
    nC = size(finalMM, 1);
    nP1 = size(mrgMM, 1);
    sz = max(8, 80 - nC);

    % Load brain surface: SurfacesFile first (BrainSurfRaw), then ProjSurfRaw
    surfV = []; surfF = [];
    if isfield(app, 'SurfacesFile') && ~isempty(app.SurfacesFile)
        try
            loaded = load(app.SurfacesFile);
            if isfield(loaded, 'BrainSurfRaw'), S = loaded.BrainSurfRaw;
            elseif isfield(loaded, 'k') && isfield(loaded.k, 'BrainSurfRaw'), S = loaded.k.BrainSurfRaw;
            else, fns = fieldnames(loaded); if ~isempty(fns) && isfield(loaded.(fns{1}), 'BrainSurfRaw'), S = loaded.(fns{1}).BrainSurfRaw; else, S = []; end; end
            if ~isempty(S) && isfield(S, 'vertices') && isfield(S, 'faces')
                surfV = S.vertices; surfF = S.faces;
            end
        catch
        end
    end
    if isempty(surfV) && isfield(app, 'ProjSurfRaw') && ~isempty(app.ProjSurfRaw) && isfield(app.ProjSurfRaw, 'vertices') && isfield(app.ProjSurfRaw, 'faces')
        surfV = app.ProjSurfRaw.vertices;
        surfF = app.ProjSurfRaw.faces;
    end

    % ----- 1) Pipeline funnel: contact count at each stage -----
    figure('Position', [50 50 520 380], 'Name', [app.PatientIDStr ' — Pipeline funnel']);
    nAssigned = sum(assigned);
    nAfterP3 = size(finalMM_afterPhase3, 1);
    counts = [nP1; nAssigned; nAfterP3; nAfterBoneFilter; nC];
    labels = {'Phase 1 (merged)', 'Assigned to shaft', 'After Phase 3', 'After filter', 'Final'};
    bar(counts, 'FaceColor', [0.35 0.55 0.8], 'EdgeColor', 'none');
    set(gca, 'XTickLabel', labels);
    ylabel('Number of contacts');
    title(sprintf('Pipeline: %d -> %d -> %d -> %d -> %d', counts(1), counts(2), counts(3), counts(4), counts(5)));
    grid on;

    % ----- 2) 3D Phase 1: all candidates + surface, colored by confidence -----
    sz1 = max(6, 60 - round(nP1/20));
    figure('Position', [60 60 600 480], 'Name', [app.PatientIDStr ' — Phase 1 candidates']);
    ax = gca;
    if ~isempty(surfV), patch(ax, 'Vertices', surfV, 'Faces', surfF, 'FaceColor', [0.85 0.85 0.92], 'FaceAlpha', 0.4, 'EdgeColor', 'none'); hold(ax, 'on'); end
    scatter3(ax, mrgMM(:,1), mrgMM(:,2), mrgMM(:,3), sz1, confidence, 'filled');
    colormap(ax, 'parula');
    cb = colorbar(ax); cb.Label.String = 'Confidence';
    xlabel(ax, 'X (mm)'); ylabel(ax, 'Y (mm)'); zlabel(ax, 'Z (mm)');
    title(ax, sprintf('Phase 1 merged (n=%d) — colored by confidence', nP1));
    axis(ax, 'equal'); grid(ax, 'on'); view(ax, 3);

    % ----- 3) Phase 1 outcome: shaft / singleton kept / dropped + surface -----
    status = zeros(nP1, 1);
    status(assigned) = 1;
    for i = find(~assigned)'
        if min(pdist2(mrgMM(i,:), finalMM)) <= 3.0, status(i) = 2; else, status(i) = 3; end
    end
    figure('Position', [70 70 600 480], 'Name', [app.PatientIDStr ' — Phase 1 outcome']);
    ax = gca;
    if ~isempty(surfV), patch(ax, 'Vertices', surfV, 'Faces', surfF, 'FaceColor', [0.85 0.85 0.92], 'FaceAlpha', 0.4, 'EdgeColor', 'none'); hold(ax, 'on'); end
    col = [0 0.7 0; 0 0.4 0.9; 0.9 0.25 0.2];
    for k = 1:3
        idx = (status == k);
        if any(idx), scatter3(ax, mrgMM(idx,1), mrgMM(idx,2), mrgMM(idx,3), sz1, col(k,:), 'filled'); end
    end
    legend(ax, 'Shaft', 'Singleton kept', 'Dropped', 'Location', 'best');
    xlabel(ax, 'X (mm)'); ylabel(ax, 'Y (mm)'); zlabel(ax, 'Z (mm)');
    title(ax, sprintf('Phase 1 outcome: %d shaft, %d kept, %d dropped', sum(status==1), sum(status==2), sum(status==3)));
    axis(ax, 'equal'); grid(ax, 'on'); view(ax, 3);

    % ----- 4) 3D After Phase 3 + surface -----
    if nAfterP3 > 0
        figure('Position', [80 80 600 480], 'Name', [app.PatientIDStr ' — After Phase 3']);
        ax = gca;
        if ~isempty(surfV), patch(ax, 'Vertices', surfV, 'Faces', surfF, 'FaceColor', [0.85 0.85 0.92], 'FaceAlpha', 0.4, 'EdgeColor', 'none'); hold(ax, 'on'); end
        scatter3(ax, finalMM_afterPhase3(:,1), finalMM_afterPhase3(:,2), finalMM_afterPhase3(:,3), max(6, sz), finalConf_afterPhase3, 'filled');
        colormap(ax, 'parula'); colorbar(ax);
        xlabel(ax, 'X (mm)'); ylabel(ax, 'Y (mm)'); zlabel(ax, 'Z (mm)');
        title(ax, sprintf('After Phase 3 (n=%d) — before bone/linearity filter', nAfterP3));
        axis(ax, 'equal'); grid(ax, 'on'); view(ax, 3);
    end

    % ----- 4b) Kept vs dropped after Phase 3 (two colors, same figure) -----
    if nAfterP3 > 0 && nC > 0
        matchMm = 5.0;
        distToFinal = pdist2(finalMM_afterPhase3, finalMM);
        kept = min(distToFinal, [], 2) <= matchMm;
        nKept = sum(kept);
        nDropped = sum(~kept);
        figure('Position', [82 82 600 480], 'Name', [app.PatientIDStr ' — Kept vs dropped after Phase 3']);
        ax = gca;
        hold(ax, 'on');
        if ~isempty(surfV), patch(ax, 'Vertices', surfV, 'Faces', surfF, 'FaceColor', [0.85 0.85 0.92], 'FaceAlpha', 0.4, 'EdgeColor', 'none'); end
        szP3 = max(6, 60 - round(nAfterP3/20));
        % Plot both: red = dropped, green = kept. Build legend only for series we add.
        legEntries = {};
        if nDropped > 0
            scatter3(ax, finalMM_afterPhase3(~kept,1), finalMM_afterPhase3(~kept,2), finalMM_afterPhase3(~kept,3), szP3, [0.85 0.2 0.15], 'filled', 'DisplayName', sprintf('Dropped (n=%d)', nDropped));
            legEntries{end+1} = sprintf('Dropped (n=%d)', nDropped);
        end
        if nKept > 0
            scatter3(ax, finalMM_afterPhase3(kept,1), finalMM_afterPhase3(kept,2), finalMM_afterPhase3(kept,3), szP3, [0 0.65 0.2], 'filled', 'DisplayName', sprintf('Kept (n=%d)', nKept));
            legEntries{end+1} = sprintf('Kept (n=%d)', nKept);
        end
        if ~isempty(legEntries), legend(ax, legEntries, 'Location', 'best'); end
        xlabel(ax, 'X (mm)'); ylabel(ax, 'Y (mm)'); zlabel(ax, 'Z (mm)');
        title(ax, 'After Phase 3: kept (green) vs dropped by bone/linearity/Phase 4 (red)');
        axis(ax, 'equal'); grid(ax, 'on'); view(ax, 3);
    end

    % ----- 5) Confidence: Phase 1 vs Final -----
    figure('Position', [85 85 560 420], 'Name', [app.PatientIDStr ' — Confidence (Phase 1 vs Final)']);
    subplot(2,1,1);
    histogram(confidence, 25, 'FaceColor', [0.5 0.6 0.9], 'EdgeColor', 'none');
    xlabel('Confidence'); ylabel('Count');
    title(sprintf('Phase 1 merged (n=%d) — mean %.2f', nP1, mean(confidence)));
    grid on;
    subplot(2,1,2);
    histogram(finalConf, 25, 'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'none');
    xlabel('Confidence'); ylabel('Count');
    title(sprintf('Final (n=%d) — mean %.2f, median %.2f', nC, mean(finalConf), median(finalConf)));
    grid on;

    % ----- 6) 3D Final + surface, colored by confidence -----
    figure('Position', [95 95 600 480], 'Name', [app.PatientIDStr ' — Final contacts + surface']);
    ax = gca;
    if ~isempty(surfV), patch(ax, 'Vertices', surfV, 'Faces', surfF, 'FaceColor', [0.85 0.85 0.92], 'FaceAlpha', 0.4, 'EdgeColor', 'none'); hold(ax, 'on'); end
    scatter3(ax, finalMM(:,1), finalMM(:,2), finalMM(:,3), sz, finalConf, 'filled');
    colormap(ax, 'parula');
    cb = colorbar(ax); cb.Label.String = 'Confidence';
    xlabel(ax, 'X (mm)'); ylabel(ax, 'Y (mm)'); zlabel(ax, 'Z (mm)');
    title(ax, sprintf('Final contacts (n=%d) — colored by confidence', nC));
    axis(ax, 'equal'); grid(ax, 'on'); view(ax, 3);

    % ----- 7) 3D Final by shaft ID + surface -----
    if ~isempty(finalShaftID) && numel(finalShaftID) == nC
        figure('Position', [105 105 600 480], 'Name', [app.PatientIDStr ' — Final by shaft + surface']);
        ax = gca;
        if ~isempty(surfV), patch(ax, 'Vertices', surfV, 'Faces', surfF, 'FaceColor', [0.85 0.85 0.92], 'FaceAlpha', 0.4, 'EdgeColor', 'none'); hold(ax, 'on'); end
        scatter3(ax, finalMM(:,1), finalMM(:,2), finalMM(:,3), sz, finalShaftID, 'filled');
        colormap(ax, 'lines');
        cb = colorbar(ax); cb.Label.String = 'Shaft ID (0=singleton)';
        xlabel(ax, 'X (mm)'); ylabel(ax, 'Y (mm)'); zlabel(ax, 'Z (mm)');
        title(ax, sprintf('Final (n=%d) — colored by shaft', nC));
        axis(ax, 'equal'); grid(ax, 'on'); view(ax, 3);
    end

    % ----- 8) Inter-contact distance distribution -----
    if nC > 1
        D = pdist2(finalMM, finalMM);
        D(logical(eye(size(D)))) = Inf;
        minDist = min(D, [], 2);
        figure('Position', [100 100 500 350], 'Name', [app.PatientIDStr ' — Inter-contact distance']);
        histogram(minDist, 25, 'FaceColor', [0.3 0.6 0.4], 'EdgeColor', 'none');
        xlabel('Min distance to nearest contact (mm)');
        ylabel('Count');
        title(sprintf('Inter-contact spacing (n=%d) — median %.1f mm', nC, median(minDist)));
        grid on;
    end

    % ----- 6) Threshold vs count curve (which threshold “matches” final count) -----
    figure('Position', [120 120 500 350], 'Name', [app.PatientIDStr ' — Threshold curve']);
    plot(ThrHU, NumObj, '-', 'LineWidth', 1.5);
    hold on;
    [~, idxD] = min(abs(NumObj - nC));
    if idxD >= 1 && idxD <= numel(ThrHU)
        plot(ThrHU(idxD), NumObj(idxD), 'ro', 'MarkerSize', 10);
        legend('Raw threshold count', sprintf('Closest to final n=%d', nC), 'Location', 'best');
    end
    xlabel('Threshold (HU)');
    ylabel('# Connected components (raw)');
    title(sprintf('Detection curve vs threshold — final %d contacts in %.1f s', nC, elapsedTime));
    grid on;
end
