# Clean Pipeline — Design Overview

*Reference summary of the rebuilt single-frame notebook (`testing/autolabeling_pipeline_clean.ipynb`).
Supersedes O1/O2/O4/O5/MC/hull-anchoring/volume-filter parts of the old pipeline for this rebuild —
those still exist in `autolabeling/src/autolabeling/` if ever needed again.*

## Stage list

| # | Stage | Input | Process | Output |
|---|---|---|---|---|
| 0 | Data loading | dataset root, scene/frame/camera | Build `FrameRecord`: intrinsics, extrinsics, image + LiDAR paths | 1 frame record |
| 1a | LiDAR full aggregation (`pts_ego_full`) | raw sweeps around anchor keyframe | `n_before`/`n_after` sweep aggregation (ego-motion compensated) → **ego-body exclusion filter only**. Ground included, on purpose — feeds PseudoLabeler only. | fully aggregated, ego-filtered point cloud (ego frame, ground included) |
| 1b | TerraSeg ground removal, **per sweep before aggregation** (`pts_ego`) | each raw sweep, in its own native ego frame | Ego-body filter (own frame) → TerraSeg classify (own frame, in-distribution) → keep non-ground only → transform to anchor frame → concatenate across sweeps. See reasoning below — order matters. | ground-free aggregated point cloud (ego frame). Feeds B1 + O3 only. |
| 2 | PseudoLabeler ground-surface fit | `pts_ego_full` (Section 1a), once/frame | Fit MLP `gθ(x,y)→z_ground` (asymmetric loss); only needed when B2 is on. Needs ground points in its input — the opposite requirement from B1/O3. | queryable continuous ground surface |
| 4 | SAM3 segmentation | RGB image, text prompts | Per-prompt SAM3 masks, score/area thresholds, cross-class IoU dedup (priority order) | `{prompt: [masks]}` |
| 5 | SAM3D Body inference | pedestrian masks + ViTDet gap-fill boxes | Regress mesh/pose from **bbox only** (`use_mask=False` — mask is not seen by the model) | mesh + `cam_t` (possibly wrong depth/height, coupled) |
| 6 | B1 — depth correction | body output + in-mask LiDAR | HDBSCAN dominant cluster (falls back per §"O3 fallback ladder" logic) → median depth → rigid shift along camera axis. **Independent toggle from B2.** | mesh at corrected depth, mask-centroid-aligned |
| 7 | B2 — ground anchoring | B1 output + PseudoLabeler | Shift mesh vertically so lowest vertex sits on `z_ground` at pedestrian's ego (x,y). **Independent toggle from B1.** | grounded mesh |
| 8 | SAM3D Objects — O3 pointmap | MoGe relative depth + in-mask LiDAR | Mask erosion → **TerraSeg ground removal (new, before HDBSCAN)** → HDBSCAN dominant cluster → **fallback ladder (new)** → per-object affine `Z=a·Z_moge+b` → rebuild pointmap → inference | mesh per object |
| 9 | Orientation + OBB | body/object results | Pedestrian: skeleton-based facing (unambiguous). Objects: PCA footprint + ego-heading disambiguation (180° ambiguity resolved via road-alignment) | OBBs + yaw |
| 10 | Rider merge | pedestrian + bicycle/motorcycle OBBs | Ego-space proximity match (≤1.5 m) → combine vertices, recompute one OBB, keep vehicle label | merged riders |
| 11 | Visualization | OBBs + image + LiDAR | 2D reprojection, BEV top-down. **No OBB output filter of any kind runs** (not even ego-vehicle exclusion) — every detection is shown, including O3 global/unscaled fallbacks, on purpose, to see the pipeline's raw shortcomings for analysis | plots |

## Explicitly dropped from the old pipeline

- Motion compensation (ICP tracking, TerraSeg-per-sweep ground removal for tracking, dynamic/static classification, velocity/heading from trajectory)
- Pointmap modes O1, O2, O4, O5 (only O3 kept)
- Hull anchoring (was O4/O5-only anyway; mask erosion is the O3-relevant equivalent and is kept)
- **All OBB output filters** — volume/dimension filter *and* the ego-vehicle-exclusion filter (which operates on fitted boxes, not LiDAR points — unrelated to Section 5's LiDAR-level ego-body filter, which is kept). Deliberately unfiltered output for now, to see the pipeline's raw shortcomings directly.
- Max-BEV-range LiDAR filter
- `BEV_DROP_NON_LOCAL` visualization toggle
- Cross-camera merge (single-camera scope for this rebuild)

## New relative to the old pipeline

1. **TerraSeg-based ground removal, run per sweep before aggregation** (not on the aggregated cloud, not PseudoLabeler-based) — see the dedicated reasoning section below, this is the one decision worth being able to defend precisely later. Produces a ground-free `pts_ego` used by both B1 and O3; PseudoLabeler gets its own separately-built, still fully-aggregated `pts_ego_full`.
2. **O3 fallback ladder** (replaces "HDBSCAN fails → jump straight to global fit"):
   HDBSCAN dominant cluster → raw in-mask points (uncleaned, skip clustering) → retry with small mask dilation → global affine (whole-image LiDAR) → unscaled MoGe. Global/unscaled are now true last resorts instead of the primary fallback.
3. **B1 and B2 decoupled into independent toggles** — lets us empirically test whether ground-anchoring is earning its keep, rather than assuming it (see reasoning below).

## Key design reasoning (why, not just what)

### Why TerraSeg runs per sweep, before aggregation — the decision to be able to defend later

**TerraSeg ground removal: per sweep, before aggregation.** TerraSeg (PTv3, trained on
OmniLiDAR) was trained and evaluated strictly on single, independent scans with no
multi-frame accumulation. Its input features (normalized height, horizontal range) are
defined relative to each sweep's own sensor origin. We therefore run TerraSeg on each sweep
in its native sensor frame and only then transform the non-ground points into the anchor
frame and aggregate. Running it once on the aggregated cloud would be out-of-distribution,
would compute range and height for non-anchor points relative to the wrong origin, and
would expose the model to motion-smeared trails of dynamic objects. Per-sweep labels
propagate through the ego-motion transform, so the aggregated cloud is ground-free at no
extra cost.

**This is the opposite of the PseudoLabeler choice**: its continuous height map
`z_ground(x, y)` is fitted on the aggregated cloud, because as an offline per-frame
optimization it benefits from the extra density. Otherwise it would have to extrapolate
where single sweeps leave gaps, and O4 (the earlier pipeline's precedent for this) applies
it to aggregated points anyway.

**Practical consequence**: there are now two separate aggregated point clouds. `pts_ego_full`
(Section 1a) — ground included, feeds PseudoLabeler (Section 2) and B2's RANSAC fallback
(B2 needs ground points to find the ground; giving it the ground-free cloud would leave it
with nothing to fit). `pts_ego` (Section 1b) — ground-free by construction, feeds B1 and O3.
No separate ground/non-ground boolean mask is threaded through O3 anymore — the ground was
never aggregated into `pts_ego` in the first place, so there's nothing to mask out at
candidate-selection time.

(Superseded: an earlier version of this notebook ran TerraSeg once on the aggregated cloud,
or on the anchor sweep alone with 1-nearest-neighbor label propagation onto the aggregated
cloud. Both were replaced by the per-sweep approach above once the OmniLiDAR/native-frame
training distribution argument was made explicit.)

- **Why TerraSeg over PseudoLabeler for point removal**: PseudoLabeler fits one *continuous surface* — good for querying "what's the ground height at this specific (x,y)" (needed by B2), but not the right tool for classifying *individual existing LiDAR points* as ground/not — TerraSeg does that natively and per-point, trained on real ground/non-ground labels rather than an unsupervised asymmetric-loss surface fit.
- **Why B1 alone isn't enough (the ground-anchoring question)**: SAM3D Body only sees a bounding box (`use_mask=False`) — the 2D mask is never a constraint on its output. Its depth (`tz`) and its own regressed body height are *coupled*: a taller mesh at a farther depth and a shorter mesh at a closer depth can both reproject to the same bbox size — this is the classic monocular height–depth ambiguity. B1 fixes depth via a **rigid translation only** (no rescaling), so if the model's original height guess was off (which is exactly what produced its wrong depth guess in the first place), the corrected-depth mesh no longer reprojects to a self-consistent size — feet can float or sink. B1 also only pins the mask *centroid*, not the full vertical extent, so any residual shape/pose error shows up directly as head/foot misalignment. B2's ground-snap is a cheap, robust proxy fix for this (no explicit height re-estimation needed, since LiDAR returns on a person are usually too sparse to measure height directly).
- **Why O3 (LiDAR+MoGe, mask-specific) beats O5 (CompletionFormer, full-frame fusion) despite fusing fewer modalities per-object**: CompletionFormer does one global dense forward pass per frame — each object's depth depends on how well that generalizes nearby, and it's trained on 64-beam KITTI (out-of-distribution for 32-beam nuScenes). O3 divides responsibility: MoGe only needs *local relative* shape (reliable even when globally miscalibrated), and the *absolute scale* comes from a closed-form least-squares fit on that object's own clean LiDAR — no learned interpolation for the number that matters most.
- **Mask constraint in SAM3D Objects**: the mask is a soft conditioning signal (alpha-channel + crop-to-bbox), *not* a hard silhouette constraint on the generated mesh. A real mask-IoU + pointmap-alignment refinement step exists (`layout_post_optimization` / `layout_post_optimization_method_GS` in the SAM3D Objects repo) — but it's **Meta's own default to disable it**: `with_layout_postprocess=False` was set in the very commit that implemented the feature (`0f5f9bd`, "Add Layout Post-Optimization"), not something the thesis project changed. It renders the mesh (or GS, which is decoded by default and takes priority when present), computes IoU against the mask, and optimises rotation/translation/scale — for the mesh path, a manual alignment against pointmap-derived points runs first. It has its own occlusion detection (`check_occlusion`: mask touches image border, OR a depth-discontinuity edge against the pointmap, OR the mask has an internal hole/is fragmented) and **skips optimisation entirely** for any object that trips one of those — no attempt to compensate, just leaves the original pose. Has no free-space term. Full details, prior-art positioning, and the concrete next step logged in `contribution_ideas/contribution_plan.md` §10 (Tier 1) and §30 (open questions, item 8) — check that mechanism as a cheap baseline before building anything from scratch there.

## Open items for later (not blocking the clean notebook)

- Quantify whether sweep aggregation actually improves PseudoLabeler's ground-surface fit quality (no longer applicable to TerraSeg, which is deliberately single-sweep-only now).
- SAM3D Objects `layout_post_optimization` investigation — tracked in `contribution_ideas/contribution_plan.md` (§10, §30).
- LiDAR-conditioned SAM3D Body (architectural "B2", requires modifying `CameraEncoder` + fine-tuning) — tracked in `contribution_ideas/sam3d_body_depth_conditioning.md`.
- TerraSeg ground labels could also clean PseudoLabeler's training set (supervised ground points instead of relying purely on the asymmetric loss) — free synergy once both models are loaded for other reasons.

---

## Production pipeline port (2026-09-23)

The clean-notebook design now lives in `autolabeling/src/autolabeling/pipeline.py`, config-driven
via `configs/nuscenes.yaml` / `configs/ecp.yaml` (same keys in both).

**Select the mode:** `sam3d_objects.pointmap_mode: 1..11` (or the name) — grew from the original 4 to 11 candidates
during the mode-decision investigation; full list and what each one does in `notes/sam3d_objects_mode_decision.md`.
**Final decision: mode 11 (`ldcm_full`)**, chosen after a full-dataset, all-camera comparison — see that doc for
the results and reasoning. Run `run_pipeline.py --config ...` once per mode (change `--run-name`/`output_dir` to
keep the JSONs apart), or `--pointmap-mode` to override without editing the config.

**Data flow per keyframe**
- Per sweep, in its own ego frame: ego-body filter (only filter) -> TerraSeg -> keep non-ground -> to anchor frame -> concatenate = `pts_ego` (ground-free; B1 + all modes).
- Full aggregated cloud with ground (`pts_ego_full`) -> PseudoLabeler + B2 RANSAC fallback only.
- `lidar_aggregation.use_aggregation: false` -> one sweep everywhere (TerraSeg still runs).
- Ground-free cloud is cached per keyframe and shared across cameras.

**Removed:** motion compensation (module moved to `autolabeling/legacy/`), all OBB filters (`obb_filter`), range filter (`max_range_m`),
hull anchoring, PseudoLabeler ground filter (O4), baseline / global-affine modes, `correction_mode` (B1/B2 are now independent booleans).
Old configs still load and print which keys are ignored; old mode names raise with a hint.

**Checkpoints:** stage directories are named by a hash of the config that affects them, so changing mode / aggregation / filters
never resumes stale results. Old checkpoint dirs are not reused. Object checkpoints now persist masks (needed by cross-camera merge).

**Mode 4 caveat:** CompletionFormer previously also got out-of-mask *ground* points as sparse anchors; with per-sweep ground removal
those are gone, so background/ground depth in mode 4 is completed from object-area anchors and non-ground structure only. Compare
mode 4 against 2/3 with this in mind; per-mask HDBSCAN cleaning is still applied on top.

**Tests (CPU, mocked models):** `tests/test_clean_pipeline.py`, `test_pipeline_orchestration.py`, `test_configs.py`.
Not yet run end-to-end on GPU.
