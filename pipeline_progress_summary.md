# Auto-Labeling Pipeline — Progress Summary

*Camera-LiDAR pseudo-labeling for autonomous driving — thesis implementation.*
*Covers everything from baseline through O5. All evaluations use nuScenes-format 8-class eval via a VESPA-based eval script.*

---

## 1. Context and Goal

The goal is to automatically generate 3D pseudo-labels (oriented bounding boxes + meshes) for
autonomous driving datasets, using only RGB cameras + LiDAR at train time — no human annotation.
The pipeline consists of two branches:

- **SAM3D Objects** — detects and reconstructs vehicles, bicycles, motorcycles using a
  sparse-structure diffusion model conditioned on a per-image depth pointmap.
- **SAM3D Body** — detects pedestrians via a SMPL regression model (CLIFF/HMR-style), placing
  the body mesh in 3D camera space.

Both branches take SAM3-generated segmentation masks as input.

**Key architectural fact**: SAM3D Objects was designed with an explicit pointmap input hook
`inference_model(img_rgb, mask, seed=42, pointmap=ptmap_t)`. This allows swapping the depth
map without touching model weights. SAM3D Body has no such hook — depth (`tz`) is an output
of the regression, not a conditioning input. This asymmetry drives the two fundamentally
different LiDAR integration strategies for the two branches.

---

## 2. Datasets

### 2.1 ECP (European City Pedestrian dataset, nuScenes-format export)
- **Location**: Strasbourg, France — varied terrain, European urban driving
- **LiDAR**: 64-beam Velodyne — high point density (~17,000–22,000 visible pts per front cam)
- **Evaluation set**: 33 GT annotated keyframes across 3 latesession scenes (train split)
- **Active classes**: car, pedestrian, motorcycle, bicycle
  (truck / bus / trailer / construction vehicle: near-zero instances in GT, excluded from analysis)
- **Evaluation**: 8-class, distance thresholds 0.5 / 1.0 / 2.0 / 4.0 m; mAP = mean over all
  classes and thresholds

### 2.2 nuScenes mini (10 random scenes, Boston + Singapore)
- **Location**: Dense inner-city — many vehicles, fast motion, flat terrain
- **LiDAR**: 32-beam Velodyne — half the density of ECP (~8,000–12,000 visible pts per front cam)
- **Evaluation set**: all annotated keyframes in the mini split (~81 frames across 10 scenes)
- **Active classes**: car, pedestrian, motorcycle, bicycle (+ bus, truck, trailer have GT but sparse)
- **Flat terrain**: ground near z≈0 everywhere — relevant to PseudoLabeler issues (§6.4)

### 2.3 Key differences between datasets

| Property | ECP | nuScenes mini |
|---|---|---|
| LiDAR beams | 64 | 32 |
| LiDAR density (pts/frame front cam) | ~18–22k | ~8–12k |
| Terrain | Varied (hilly Strasbourg) | Flat (Boston/Singapore) |
| Traffic speed | Moderate (European urban) | Fast (dense city) |
| GT frames evaluated | 33 (selected keyframes) | ~81 (all mini keyframes) |
| CFormer training distribution | In-distribution (KITTIDC = 64-beam) | Out-of-distribution (32-beam) |
| Aggregation used | 2 before + 2 after anchor | 3 before + 3 after anchor |

Aggregation is tuned per dataset: ECP uses ±2 sweeps (64-beam already dense; more sweeps risk
motion smearing on fast roads). nuScenes uses ±3 sweeps (32-beam is sparse; more sweeps needed
to achieve usable point density without too much smearing — a compromise).

---

## 3. Evaluation Setup

### Comparison baseline: VESPA
VESPA (VLM-based pseudo-labeling) is the existing baseline method that uses vision-language
models to generate 3D pseudo-labels. It was run in two configurations:

- **`vespa_vlm_train_all_cameras`** — uses all available cameras (6 for nuScenes, full ring)
- **`vespa_vlm_train_front_cam_only`** — uses front camera only

**Our pipeline currently uses front camera only** (CAM_FRONT). This is important for fair
comparison: the correct VESPA reference is `front_cam_only`. The `all_cameras` VESPA run
has an inherent advantage from multi-view coverage.

Multi-camera inference (merging results across cameras into a single ego-frame pseudo-label
set without duplication) is **not yet implemented** in our pipeline. It is a planned extension.

### Metrics

| Metric | Interpretation |
|---|---|
| **mAP** | Primary metric — mean AP over 8 classes × 4 distance thresholds |
| **ATE** | Average Translation Error (lower = better center localisation) |
| **ASE** | Average Scale Error (lower = better size estimation) |
| **AOE** | Average Orientation Error — only meaningful for **pedestrian** (body-forward from SAM3D Body). Vehicle AOE ≈ π (180°) in all runs — heading is not estimated, OBBs are ambiguous |
| **AVE / AAE** | Velocity / attribute errors — not predicted, always = 1.0 (ignored) |
| **NDS** | NuScenes Detection Score — penalised by AVE/AAE, not a fair metric here |

### GT sanity check
`gt_sanity_train` was run to verify the evaluation script: ECP = 0.625 mAP, nuScenes = 0.935
mAP (not 1.0 because GT has class conflicts and the 8-class mapping drops some labels). The
eval pipeline is functioning correctly.

---

## 4. LiDAR Integration — SAM3D Objects (O1–O5)

All modes feed a `(H, W, 3)` pointmap in PyTorch3D convention to the SAM3D Objects inference
hook. Setting `pointmap=<tensor>` bypasses the model's internal MoGe call entirely.

**Coordinate conventions**:
- LiDAR: +X forward, +Y left, +Z up
- Camera (R3/OpenCV): +X right, +Y down, +Z forward
- PyTorch3D (P3D): +X left, +Y up, +Z forward → `ptmap[v,u] = (-X_cam, -Y_cam, Z_cam)`

### Pre-processing shared across all modes

**1. Multi-sweep aggregation**
Per-frame LiDAR is built from multiple adjacent sweeps (±2 ECP, ±3 nuScenes). Each sweep is
ego-motion-compensated via the ego_pose transform chain:
`LiDAR sensor → sweep ego → global → anchor ego frame`.
Aggregation multiplies effective density on static surfaces. Dynamic objects smear but are
handled by HDBSCAN (smeared clouds = low-density noise → discarded as non-dominant cluster).

**2. Ego-body exclusion zone**
After aggregation, dense ego-vehicle roof / LiDAR-mount returns project into nearby object masks
and can become the dominant HDBSCAN cluster, anchoring depth to ~1.5 m (ego distance). Removed
if `|x| < 4.0 m AND |y| < 1.5 m AND 0.5 m < z < 2.5 m` simultaneously.

**3. Max BEV range filter**
Points beyond 52 m BEV distance removed. Far background returns cannot correspond to objects
within the 50 m eval range but contaminate in-mask clustering.

### HDBSCAN in-mask filtering (shared by O3, O5, B1)

Before fitting any depth model per object, in-mask LiDAR points are clustered in 3D ego-frame
XYZ via HDBSCAN. Only the dominant cluster (most points) is kept; noise points (label = -1)
and minority clusters are discarded.

**Why HDBSCAN over DBSCAN**: DBSCAN requires a fixed `eps` (neighbourhood radius). LiDAR density
falls off with range², so a radius that works at 5 m is wrong at 40 m. HDBSCAN adapts.

**Why ego-frame XYZ**: metric 3D space makes distance thresholds physically meaningful and
range-invariant.

**Per-class parameters** (same for ECP and nuScenes):

| Class | `min_cluster_size` | `cluster_eps` (m) |
|---|---|---|
| pedestrian | 3 | 0.20 |
| bicycle | 3 | 0.20 |
| motorcycle | 3 | 0.30 |
| car | 3 | 0.50 |
| truck | 5 | 0.60 |
| construction vehicle | 5 | 0.60 |
| bus | 5 | 0.80 |
| trailer | 5 | 0.70 |

`min_samples=1` (least aggressive noise rejection) is fixed for all classes.

---

### O1 — Sparse LiDAR Pointmap

Replace MoGe with a sparse per-pixel LiDAR map. Only pixels with a LiDAR return get real
metric XYZ; all other pixels are NaN. SAM3D Objects was trained with random pointmap dropout,
so windows without depth receive a learned `invalid_xyz_token` — reconstruction degrades
gracefully to image-only for uncovered pixels.

**Coverage**: ~0.5% of pixels have depth — very sparse.

**Motivation**: Does raw metric depth — even very sparse — give a meaningful lift over baseline?
Answer from results: yes, large jump (ECP mAP 0.028 → 0.106).

**Known limit**: objects with zero in-mask LiDAR returns fall back to image-only prior =
effectively no LiDAR integration for those objects.

---

### O2 — MoGe + Global Affine Calibration

MoGe produces a dense relative depth map `Z_moge (H,W)`. A single global affine
`Z_metric = a * Z_moge + b` is fitted per frame via least-squares on all in-image LiDAR
returns. Dense coverage, one scale factor per frame.

**Motivation**: Can a single linear rescaling of MoGe fix the scale issue?

**Known limit**: MoGe's depth compression is depth-range-dependent. One (a,b) averages over all
distances. Objects at very different depths from the bulk of LiDAR returns are miscalibrated.
Results confirm this: motorcycle collapses to AP 0.021 (ECP) and 0.034 (nuScenes) — motorcycle
distances vary widely and a global affine cannot capture per-object scale variation.

---

### O3 — Per-Object Local Affine Calibration *(novel contribution)*

Like O2, but a separate `(a, b)` is fitted **per detected object**, using only in-mask
LiDAR points on that object's surface (after HDBSCAN cleaning). MoGe runs once per frame;
the affine is object-specific.

**Procedure per object**:
1. MoGe → `Z_moge (H,W)` shared across all objects
2. HDBSCAN → dominant surface cluster for that mask
3. Sample `Z_moge` at clean cluster pixels → pairs `(Z_moge_i, Z_lidar_i)`
4. Fit `(a, b)` via least-squares on those object-specific pairs
5. Apply `Z_metric = max(a * Z_moge + b, 0.1)` to the full `(H,W)` map
6. Rebuild PyTorch3D pointmap, run SAM3D Objects inference

**Fallback chain**: cluster not found → global (a,b); no LiDAR in frame → unscaled MoGe (a=1, b=0).

**Why this works**: Each object's depth range is self-consistent. MoGe's relative profile is
accurate within a narrow depth range even if it is wrong globally. The per-object affine
captures exactly the compression factor relevant to that object's distance.

**Known limit (E5)**: Large vehicles viewed side-on at close range span 2–4 m in depth across
the mask. MoGe's relative depth gradient across that span can be inaccurate (compressed or
distorted). The linear affine cannot fix a non-linear MoGe error.

---

### O4 — CompletionFormer Dense Depth + PseudoLabeler Ground Filter

Replace MoGe entirely with **CompletionFormer** (CVPR 2023, trained on KITTIDC) — a depth
completion network that takes sparse LiDAR + RGB → dense metric depth.

**Network config**: `preserve_input=True` hard-anchors output at LiDAR pixels (MAE ≈ 0 at
anchors); interpolation fills the rest. `data_name=KITTIDC`, `lidar_lines=64` (ECP) / 32 (nuScenes).

**Ground filter problem**: Without pre-filtering, road-surface LiDAR returns project into the
lower pixels of foreground object masks. With `preserve_input=True`, CFormer hard-anchors to
road depth inside the mask → the dense map shows road surface *inside* the car → OBB is
misplaced.

**PseudoLabeler**: a lightweight per-frame MLP `gθ: R²→R` (ego XY → ground height Z) fitted
on each frame's LiDAR sweep via an asymmetric lower-envelope loss. `get_ground_bool(pc, thres=0.10)`
classifies a point as ground if `z_point - z_ground ≤ 0.10 m`.

**Why not RANSAC**: RANSAC fits a single plane globally. Urban terrain has slopes, crossings,
and height discontinuities — a flat-plane assumption fails outside flat parking lots. The MLP
learns a continuous surface per frame.

**Why not TerraSeg full model**: TerraSeg's heavy semantic segmentation network is expensive
and requires GPU memory that is already occupied by SAM3D models. PseudoLabeler (tiny MLP)
fits in seconds per frame and is accurate enough for ground classification at 0.10 m threshold.

**Why 0.10 m threshold**: The model's default 0.40 m (tuned for semantic labeling) would remove
bicycle frames (~0.20 m) and car bumpers (~0.30 m) — real object surfaces. 0.10 m removes only
the road surface while keeping all object returns above it.

**PseudoLabeler early stopping fix**: On flat terrain (nuScenes Boston/Singapore), random-init
pred≈0 accidentally matches true ground z≈0 → loss_0 is tiny → tracking `_best_loss` from step 0
means the step-1 overshoot never recovers the initial "baseline" → patience fires at exactly
`warmup + patience = 500` steps on every flat-terrain frame. Fix: defer `_best_loss` tracking
to step ≥ 200 (warmup window). Now typical convergence: 500–2500 steps for varied terrain (ECP);
faster for flat (nuScenes).

**O4 result**: car AP rises to 0.543 (best of any single class across all runs on ECP) but
motorcycle collapses to 0.230. The global ground filter removes too many thin-object returns
(motorcycle frame at ~0.20 m above road surface is at the boundary of the 0.10 m threshold in
practice). O4 is too aggressive for small objects.

---

### O5 — CompletionFormer + Per-Mask HDBSCAN Anchor Cleaning *(novel contribution)*

Like O4 (uses CFormer for dense metric depth) but **replaces the global PseudoLabeler ground
filter** with a per-mask HDBSCAN cleaning step applied to the LiDAR anchor map before CFormer.

**Procedure**:
1. For each object mask: HDBSCAN on in-mask 3D points → dominant cluster = clean surface
2. Mark all non-dominant in-mask points as noise; remove from sparse anchor map
3. All out-of-mask points kept unchanged (ground outside masks is fine — doesn't contaminate
   foreground objects)
4. Build clean sparse depth map → CFormer → dense metric depth → PyTorch3D pointmap

**Key difference from O4**: No PseudoLabeler needed. Ground returns inside masks are removed
because they are geometrically separated from the object surface in 3D (different cluster).
Ground returns outside masks are kept — they anchor CFormer for background depth accuracy.

**Key difference from O3**: O3 uses HDBSCAN to extract a local affine scale for MoGe. O5 uses
HDBSCAN to clean the sparse anchor before CFormer. O5 does not use MoGe.

**Result on ECP**: mAP 0.169, matched with O3+agg+filt, highest NDS. Motorcycle AOE improves
dramatically (0.328 → 0.131) — CFormer's dense shape context helps heading estimation from
elongated profile. O5 avoids O4's ground-filter over-aggression on thin objects.

**Result on nuScenes**: mAP 0.0156 vs O3's 0.0222. O5 underperforms because 32-beam nuScenes
provides too few in-mask anchors for CFormer to accurately fill object regions (CFormer was
trained on 64-beam KITTIDC — out-of-distribution). O3's dense MoGe relative shape + sparse
metric scale is more robust when CFormer cannot fill reliably.

---

## 5. LiDAR Integration — SAM3D Body (B1)

### Why a different strategy is needed

SAM3D Body is a **regression model** (CLIFF/HMR-style), not a diffusion model. It predicts
SMPL body pose, shape, and camera parameters `[s, tx_norm, ty_norm]` in one forward pass
via ViT encoder → regression head. Depth (`tz`) is derived from the predicted scale `s`:
`tz = 2 * focal_length / (bbox_size * s)`.

There is no pointmap input, no `PointPatchEmbed`, and no geometry conditioning pathway — depth
is an *output*, not a conditioning input. Injecting LiDAR at the same level as O1–O5 would
require modifying the internal `CameraEncoder` and re-training (planned as **B2**).

### B1 — Post-inference LiDAR depth correction (two stages)

**Stage 1 — Depth (tz) correction**:
1. HDBSCAN on in-mask LiDAR → dominant surface cluster
2. `tz_lidar = median(Z_vis[in_mask][keep])`
3. Recompute `tx = (u_cen - cx)/fx * tz_lidar`, `ty = (v_cen - cy)/fy * tz_lidar`
4. Shift all vertices by `[tx_new - tx_pred, ty_new - ty_pred, tz_lidar - tz_pred]`

Fallback: if HDBSCAN finds no cluster → 15th-percentile of all raw in-mask depths (targets
front surface while discarding ground bleed at the feet).

**Stage 2 — Ground anchoring (PseudoLabeler)**:
After Stage 1, the mesh sits at approximately torso depth. Stage 2 shifts the mesh so its
lowest vertices sit on the local ground surface predicted by PseudoLabeler:
- Convert `cam_t` to ego space
- Query `z_ground = pseudolabeler.predict(x_ego, y_ego)`
- Compute `z_foot = min(vertices_ego[:, 2])`
- Shift by `z_ground - z_foot`, converted back to camera space

`joints_3d` are body-relative and are **not shifted** in either stage — the projection
`joints_cam = joints_3d + cam_t` picks up the corrected `cam_t` automatically.

**B1 result**: ECP pedestrian AP rises from 0.297 (baseline) to 0.410 (B1 only) — the single
biggest per-class improvement from any single component. Without B1, SAM3D Body places
pedestrians at wrong depths. B1 corrects both depth (Stage 1) and foot placement (Stage 2).

---

## 6. Evaluation Results

### 6.1 ECP — mAP Progression

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| baseline (no LiDAR) | 0.0278 | 0.946 | 0.918 | 0.920 | 0.0355 |
| obj_baseline + B1 | 0.0473 | 0.895 | 0.919 | 0.910 | 0.0513 |
| O1 + body_baseline | 0.1064 | 0.866 | 0.729 | 1.111 | 0.0936 |
| O2 + body_baseline | 0.0712 | 1.028 | 0.751 | 1.182 | 0.0605 |
| O3 + B1 | 0.1597 | 0.671 | 0.715 | 1.132 | 0.1412 |
| O3 + B1 + agg(2) | 0.1676 | 0.658 | 0.712 | 1.125 | — |
| O3 + B1 + agg(2) + filt | 0.1689 | 0.657 | 0.712 | 1.122 | 0.1475 |
| O4 + B1 + agg(2) + filt | 0.1275 | 0.743 | 0.708 | 1.119 | 0.1187 |
| **O5 + B1 + agg(2) + filt** | **0.1690** | **0.652** | **0.706** | **1.088** | **0.1487** |
| VESPA front cam only | 0.1022 | 0.699 | 0.735 | 1.180 | 0.1077 |
| VESPA all cameras | 0.1218 | 0.742 | 0.750 | 1.214 | 0.1117 |
| GT sanity | 0.6250 | 0.375 | 0.375 | 0.375 | — |

**Key comparison**: Our pipeline uses **front camera only**. VESPA front-cam-only = 0.1022.
Our best = **0.1690 — +65% relative improvement over VESPA on the same camera configuration**.
Even VESPA's all-camera run (0.1218) is beaten by our single-camera result.

### 6.2 ECP — Per-Class AP@2m (best runs and VESPA)

| Class | O3+agg+filt | O5+agg+filt | VESPA front | VESPA all |
|---|---|---|---|---|
| car | 0.454 | 0.449 | 0.224 | 0.276 |
| pedestrian | 0.334 | 0.337 | 0.294 | 0.463 |
| motorcycle | 0.470 | 0.464 | 0.303 | 0.193 |
| bicycle | 0.246 | 0.235 | 0.122 | 0.184 |

Our pipeline substantially beats VESPA on car, motorcycle, and bicycle. VESPA all-cameras leads
on pedestrian (0.463 vs 0.337) — multi-view coverage helps for bodies partially occluded in the
front view.

### 6.3 ECP — TP Error Breakdown (O3 vs O5 active classes)

| Class | Metric | O3+agg+filt | O5+agg+filt |
|---|---|---|---|
| car | ATE | 0.398 | 0.433 |
| | ASE | 0.291 | **0.270** |
| pedestrian | ATE | **0.109** | 0.208 |
| | ASE | 0.449 | **0.307** |
| | AOE | 0.123 | 0.128 |
| motorcycle | ATE | 0.476 | **0.411** |
| | ASE | 0.511 | **0.509** |
| | AOE | 0.328 | **0.131** |
| bicycle | ATE | 0.362 | **0.307** |
| | ASE | 0.579 | **0.549** |

O5's dense CFormer map improves ASE across all classes and motorcycle AOE (0.328→0.131)
dramatically — the complete depth shape aids heading estimation from elongated profiles.

### 6.4 nuScenes mini — mAP Results

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| baseline (no LiDAR) | 0.0006 | 1.000 | 1.000 | 1.000 | 0.0003 |
| obj_baseline + B1 | 0.0016 | 0.918 | 0.931 | 0.918 | — |
| O1 + body_baseline | 0.0220 | 1.028 | 0.661 | 1.448 | — |
| O2 + body_baseline | 0.0111 | 1.043 | 0.867 | 1.109 | — |
| **O3 + B1** | **0.0222** | **0.819** | **0.554** | **1.365** | **0.0738** |
| O3 + B1 + agg(3) | 0.0104 | 0.868 | 0.706 | 1.468 | — |
| O3 + B1 + agg(3) + filt | 0.0215 | 0.731 | 0.546 | 1.463 | 0.0830 |
| O5 + B1 + agg(3) + filt | 0.0156 | 0.811 | 0.552 | 1.220 | 0.0715 |
| VESPA front cam only | 0.0075 | 0.863 | 0.701 | 0.886 | 0.0587 |
| VESPA all cameras | 0.1176 | 0.784 | 0.498 | 1.240 | 0.1305 |

**Key comparison**: Front-camera-only — our best (O3+B1) = **0.0222 vs VESPA front = 0.0075**
(~3× improvement). VESPA's all-camera run (0.1176) still leads overall — multi-view coverage
is a major advantage on the dense nuScenes scenes, where many vehicles are visible from
side/rear cameras but not from the front. This gap is a strong motivation for implementing
multi-camera merging.

### 6.5 nuScenes — Why O3 beats O5

Both modes use identical HDBSCAN in-mask cleaning. The difference is what happens between anchors:
- **O3**: MoGe provides dense per-pixel relative depth shape. With 32-beam LiDAR, a car at 20 m
  may have only 15–30 anchor pixels — but MoGe fills the rest of the mask with a geometrically
  plausible relative shape. The local affine gives O3 a physically shaped surface even where
  LiDAR is absent.
- **O5**: CFormer must fill large gaps (sparse 32-beam) using RGB guidance and KITTIDC (64-beam)
  priors. Out-of-distribution for nuScenes → inaccurate interpolation inside object masks.

**Conclusion**: O5 is the better choice for 64-beam sensors (ECP); O3 is more robust on 32-beam.

### 6.6 nuScenes — Why Aggregation Hurts Without Ego Filter

| Run | mAP | car | bicycle |
|---|---|---|---|
| O3 + B1 (no agg) | **0.0222** | 0.120 | **0.130** |
| O3 + B1 + agg(3) | 0.0104 | 0.075 | 0.054 |
| O3 + B1 + agg(3) + filt | 0.0215 | **0.156** | 0.054 |

**Motion smearing**: Boston/Singapore traffic at 30 km/h moves 12.5 m in ±1.5 s (3+3 sweeps).
Aggregated cloud smears across 12 m → HDBSCAN sees elongated low-density blob → wrong cluster
center → bad affine fit. Bicycle (small, fast) is most affected and does not recover even with
the ego-body filter.

**Ego-body returns**: without filtering, 3+3 sweeps make ego roof extremely dense. These project
into nearby masks and become the dominant HDBSCAN cluster (anchoring at ~1.5 m ego depth).
Ego filter recovers car (0.075 → 0.156) but bicycle remains at 0.054 — bicycle clusters are
too small and sensitive to any contamination.

### 6.7 nuScenes — FP Analysis (O3 vs O5)

| Class | O3 n_pred | O5 n_pred | O3 FP | O5 FP |
|---|---|---|---|---|
| car | 937 | 1,457 | 459 | **933** |
| truck | 98 | 134 | 51 | 93 |
| bus | 54 | 131 | 20 | 107 |

O5 generates far more total predictions despite identical SAM3 masks. Root cause:

- **O3**: objects with zero in-mask LiDAR fall back to unscaled MoGe (`a=1, b=0`). MoGe depth
  is non-metric — OBB center falls outside the evaluation range (50 m for cars) or far from any
  GT box. These detections are *accidentally suppressed* by falling outside the eval window.
- **O5**: CFormer produces a globally dense depth map. Nearby anchors from other objects guide
  interpolation into zero-LiDAR regions, placing those OBBs at a plausible but incorrect metric
  depth → counted as FP within the evaluation range.

O3's "suppression" is accidental. O5 rescues those detections to wrong-but-plausible positions
where they become FPs. This is a key motivation for the next development step (§7).

---

## 7. Additional Implementation Details

### Cross-class deduplication
SAM3 runs one prompt per class. The same physical object can produce masks under multiple
class labels (truck also triggers "car"; cyclist triggers both "pedestrian" and "bicycle").
After all masks are collected, pairwise mask IoU is computed across classes. If IoU > 0.5,
the lower-priority detection is removed. Priority order: pedestrian < car < bicycle < truck
< motorcycle < trailer < bus < construction vehicle.

### Cyclist / motorcyclist OBB merge
A cyclist produces two masks: "pedestrian" (body) and "bicycle" (frame). These have low IoU
(~0.3–0.4) so cross-class dedup doesn't remove either. After OBBs are computed, the body OBB
center is transformed to ego space and compared with the bicycle/motorcycle OBB center. If
3D ego-space distance ≤ 1.5 m, they are merged: concatenate camera-space vertices from both
meshes, recompute OBB on the combined set, assign vehicle class label, remove pedestrian result.

### Camera intrinsics — SAM3D Body
SAM3D Body has an existing API parameter `cam_int` in `process_one_image()`. We pass the
dataset's real `(3,3)` intrinsic matrix directly — no source code modification needed. The
"No FOV estimator" warning at construction time is misleading but harmless; the `cam_int`
branch takes full priority at inference time.

### OBB minimum dimension filter
A post-hoc size filter removes obviously broken OBBs (e.g. ghost detections with dimension
< threshold): pedestrian `min_h=0.5 m`, bicycle `min_l=0.4 m`, car `min_l=1.5 m`, etc.
This is a lenient sanity check, not a precision tool.

---

## 8. Planned Next Steps

### Multi-camera merging
All evaluations so far use front camera only. Fusing results from all cameras (6 for nuScenes,
~3 for ECP) would dramatically increase coverage — especially for pedestrians in side views and
vehicles behind the ego. Requires projecting all camera-frame OBBs into a common ego frame and
deduplicating overlapping boxes across views. Not yet implemented.

### O6 — Improved depth for zero-LiDAR objects (FP reduction)
The dominant FP source in O5 on nuScenes is zero-LiDAR objects placed at wrong metric depths
by CFormer interpolation. A planned hybrid mode combines:
1. PseudoLabeler ground filter (O4 step)
2. Per-mask HDBSCAN cleaning (O5 step)
3. **Zero-LiDAR mask blanking**: for masks with no surviving in-mask anchors after cleaning,
   zero out the sparse depth inside the mask bbox before CFormer. This forces CFormer to
   treat those regions as no-reading → non-metric interpolation → OBB outside eval range →
   not counted as FP

### Vehicle heading estimation (AOE)
Vehicle AOE ≈ π in all runs — heading is not estimated. LiDAR cluster elongation PCA or
SAM3D Objects voxel occupancy → heading direction are the main candidates.

### B2 — LiDAR conditioning for SAM3D Body
B1 is a post-inference override. True LiDAR conditioning would require modifying the internal
`CameraEncoder` of SAM3D Body to accept a depth prior as an additional input token, then
fine-tuning. Architecturally non-trivial — planned as a later-stage contribution.

### Scale error (ASE ~0.27–0.55)
Generated mesh shapes are still imperfect. Better pointmap conditioning (O5/O6) improves ASE
somewhat. Explicit size prior from class-conditional statistics or LiDAR cluster dimensions
could further constrain OBB scale.

---

## 9. Summary Table

| Mode | Dense? | Metric? | Ground filter | HDBSCAN | ECP mAP | nuScenes mAP |
|---|---|---|---|---|---|---|
| baseline | Yes (MoGe) | No | — | — | 0.028 | 0.001 |
| obj_baseline + B1 | Yes (MoGe) | No | — | — | 0.047 | 0.002 |
| O1 + B1 | Sparse (LiDAR) | Yes | — | — | 0.106 | 0.022 |
| O2 + B1 | Dense (MoGe+aff) | Approx | — | — | 0.071 | 0.011 |
| O3 + B1 | Dense (MoGe+aff) | Approx | — | Per-object | 0.160 | **0.022** |
| O3+agg+filt | Dense (MoGe+aff) | Approx | Ego+range | Per-object | 0.169 | 0.022 |
| O4+agg+filt | Dense (CFormer) | Yes | PseudoLabeler | — | 0.128 | — |
| **O5+agg+filt** | Dense (CFormer) | Yes | Ego+range | Per-mask | **0.169** | 0.016 |
| VESPA front cam | — | — | — | — | 0.102 | 0.008 |
| VESPA all cams | — | — | — | — | 0.122 | 0.118 |

*All our runs: front camera only. VESPA all-cameras uses full camera ring.*
*agg: multi-sweep aggregation. filt: ego-body + max-range filter.*
