# Auto-Labeling Pipeline — Implementation Notes

Implementation notes, design decisions, and hyperparameters for the single-frame,
camera-only auto-labeling pipeline.  Sections are added as features are implemented.

---

## 1. Cross-Class Deduplication

### Problem
SAM3 runs one text prompt per class, so the same physical object can produce masks
under multiple prompts.  A truck, for example, often also triggers the "car" prompt.
Without deduplication, downstream models generate redundant meshes and the final
submission has double-counted detections.

### Approach
After all per-class masks are collected for a frame, pairwise mask IoU is computed
across every cross-class combination.  If IoU > `iou_thresh` (default 0.5), one of the
two detections is removed: the one with the lower *priority* value.

### Priority order  *(lower = removed first)*
| Class | Priority | Rationale |
|---|---|---|
| `pedestrian` | 0 | Most generic person label |
| `car` | 1 | Generic vehicle |
| `bicycle` | 2 | Bike frame; rider handled separately (see §2) |
| `truck` | 3 | More specific than car |
| `motorcycle` | 4 | Motorised rider > bicycle |
| `trailer` | 5 | Vehicle-type specific |
| `bus` | 6 | Large vehicle > truck |
| `construction vehicle` | 7 | Most specific / heavy equipment |

Within each pair the higher-priority class survives.  Equal-priority pairs are not compared
(no two classes have the same value).

### Config
```yaml
cross_class_dedup:
  iou_thresh: 0.5
  priority:
    pedestrian:           0
    car:                  1
    bicycle:              2
    truck:                3
    motorcycle:           4
    trailer:              5
    bus:                  6
    construction vehicle: 7
```

### Code
`SAM3Segmentor` in `AutoLabeling/src/autolabeling/models/sam3_segmentor.py` performs
deduplication at the end of each frame's inference, before results are checkpointed.

---

## 2. Cyclist / Motorcyclist OBB Merge

### Problem
A cyclist produces two SAM3 masks:
- `"pedestrian"` — the person's body (just the rider)
- `"bicycle"` — the bike frame

These masks have **low IoU** (~0.3–0.4, one is a subset of the other), so cross-class
dedup does not remove either.  Without further handling, SAM3D Body reconstructs the
rider and SAM3D Objects reconstructs the bike, yielding two separate results for one
physical object — double-counting in evaluation.

### Why not "bicycle and cyclist" as a single prompt
Using a combined prompt gives SAM3 a mask covering both person and bike as one unit.
SAM3D Objects then reconstructs a joint mesh, but the human-body quality is worse than
SAM3D Body's dedicated SMPL reconstruction.  Splitting gives a better body mesh.

### Design (Option A)
1. Use `"bicycle"` as the SAM3 prompt so the mask focuses on the bike frame.
2. The rider is detected as `"pedestrian"` → SAM3D Body → high-quality body mesh + skeleton.
3. Both meshes get OBBs fitted normally (`_postprocess_frame`).
4. **OBB merge step** (`_merge_rider_obbs`, run immediately after postprocessing):
   - Transform body OBB center from **camera space** to **ego space** via `R_c2e, t_c2e`
     (body OBBs from `compute_obb_pedestrian` are in camera space; object OBBs from
     `compute_obb_gravity_aligned` are already in ego space).
   - If 3-D ego-space distance ≤ `dist_thresh`, the pedestrian is a rider of that vehicle.
   - **Merge**: concatenate camera-space vertices of both meshes, recompute OBB with
     `compute_obb_gravity_aligned` on the combined vertex set.
   - The merged result keeps the **vehicle class label** (`bicycle` or `motorcycle`).
   - The pedestrian body result is **removed** from `body_results`.

The same logic applies to `"motorcycle"` — a motorcyclist rider detected as `"pedestrian"`
is merged into the motorcycle OBB.

### Distance threshold
`dist_thresh = 1.5 m`

The body OBB center (torso) sits at ~1.0 m height; the bicycle OBB center (frame) sits at
~0.5 m height; horizontal offset depends on viewing angle.  In 3-D ego space the distance
is typically 0.6–1.0 m for a directly mounted rider, up to ~1.2 m for side-on views.
1.5 m is tight enough to avoid accidentally merging a pedestrian walking next to a parked
bike (>1.5 m ego distance when they are not touching), while robust to all mounting angles.

### Code
- Full pipeline: `_merge_rider_obbs()` in `AutoLabeling/src/autolabeling/pipeline.py`
- Notebook reference: `Testing/autolabeling_pipeline.ipynb`

---

## 3. LiDAR Integration in SAM3D Objects

### Background
SAM3D Objects conditions its sparse-structure diffusion generator on a per-image **pointmap**
`(H, W, 3)` in PyTorch3D space via `PointPatchEmbed`.  In the baseline pipeline this comes from
**MoGe**, a monocular depth model: every pixel gets a depth value in arbitrary (non-metric) units.
As a result, the model's predicted translation and scale are in MoGe units — absolute distance
and physical object size are unreliable.

All LiDAR options feed alternative pointmaps into this hook:
```python
inference_model(img_rgb, binary_mask, seed=42, pointmap=ptmap_t)
```
Setting `pointmap=<tensor>` bypasses the internal MoGe call entirely.

### Coordinate spaces
| Space | Convention | Notes |
|---|---|---|
| LiDAR sensor | +X forward, +Y left, +Z up | raw `.bin` file |
| Camera / R3 | +X right, +Y down, +Z forward (OpenCV) | projection math |
| PyTorch3D | +X left, +Y up, +Z forward | SAM3D Objects input |
| Ego | +X forward, +Y left, +Z up (vehicle) | OBB fitting, merge |

Conversion R3 → PyTorch3D: negate X and Y.
```
ptmap[v, u] = (-X_cam, -Y_cam, Z_cam)
```

### Config flag
```yaml
sam3d_objects:
  pointmap_mode: baseline   # 'baseline' | 'o1_lidar' | 'o2_moge_affine' | 'o3_local_affine'
                            # | 'o4_ground_filter' | 'o5_mask_hdbscan'
```

---

### Pre-processing: Multi-Sweep LiDAR Aggregation

Before any of the O1/O2/O3/B1 paths run, the per-frame LiDAR point cloud can be built from
multiple temporally adjacent sweeps instead of a single sweep.  All sweeps are ego-motion
compensated and concatenated into one cloud in the **anchor ego frame** (the frame whose camera
image is being processed).

**Motivation**: A single 32-beam (nuScenes) or 64-beam (ECP) sweep often has too few hits on
small or far-away objects.  Aggregating 3–7 sweeps (the anchor ±2–3) multiplies the effective
point density on static surfaces (parked cars, building facades, standing pedestrians) and
substantially improves HDBSCAN clustering and affine-fit conditioning.

**Transform chain per sweep**:
```
LiDAR sensor → sweep ego frame  (R_l2e, t_l2e from calibrated_sensor)
               → global frame    (R_e2g_sw, t_e2g_sw from ego_pose at sweep timestamp)
               → anchor ego frame (R_e2g_anchor⁻¹ from ego_pose at anchor timestamp)
```
Each sweep is transformed independently so that the ego's motion between sweeps is fully
compensated.  Dynamic objects (moving vehicles, cyclists) will "smear" across sweeps, but
static surfaces gain density as intended.  HDBSCAN naturally discards smeared dynamic-object
points as low-density noise.

**Duplicate-path handling (ECP)**: ECP's NuScenes-format export sometimes has adjacent sample_data
entries pointing to the same `.bin` file.  The sweep-walking code (`_walk_lidar_tokens`) skips
any token whose resolved file path has already been seen, so each physical sweep is included
exactly once.

**Config** (per dataset — values tuned to LiDAR beam count):
```yaml
lidar_aggregation:
  use_aggregation: true
  n_before: 3    # nuScenes (32-beam): more sweeps to compensate sparse density
  n_after:  3

lidar_aggregation:
  use_aggregation: true
  n_before: 2    # ECP (64-beam): single sweep already dense; fewer extras needed
  n_after:  2
```

**Code**: `load_lidar_pts_aggregated()` in `AutoLabeling/src/autolabeling/utils/lidar.py`.
Called from the `_load_pts_ego` closure in `pipeline.py`; the closure transparently switches
between single-sweep and aggregated loading based on the config.

---

### Pre-processing: Point Cloud Pre-filters

After loading/aggregating, two global filters are applied to the full ego-frame cloud **before**
camera projection and HDBSCAN in-mask clustering.  These are dataset-specific and live in
`lidar_filters` in the per-dataset YAML.

#### Ego-body exclusion zone

Removes returns from the ego vehicle's own roof, windshield, and LiDAR mounting hardware.
After sweep aggregation these points are fixed to a consistent ego-frame position (because the
ego body moves with the sensor coordinate frame), forming a dense horizontal strip at
approximately sensor height.  If an object's projected mask happens to overlap this strip, the
ego returns can become the dominant HDBSCAN cluster and produce a wildly wrong depth estimate.

A point is removed only if it satisfies **all three** conditions simultaneously:
```
|x| < ego_box_half_x   (fore/aft)
|y| < ego_box_half_y   (left/right)
ego_box_z_min < z < ego_box_z_max
```
The Z band avoids removing ground-level returns (`z < ego_box_z_min`) and very tall objects
that happen to be at the same XY position (`z > ego_box_z_max`).

#### Max BEV range filter

Removes points further than `max_range_m` BEV distance (`sqrt(x² + y²)`) from ego.  Far
background returns at 60–100 m cannot correspond to objects within the 50 m evaluation range,
but their projections can fall inside distant objects' masks and degrade clustering.

**Config** (same values for both datasets; may need per-dataset tuning for ECP):
```yaml
lidar_filters:
  use_ego_body_filter: true
  ego_box_half_x: 4.0    # half-length fore/aft [m]
  ego_box_half_y: 1.5    # half-width left/right [m]
  ego_box_z_min:  0.5    # lower Z — preserves close ground returns [m]
  ego_box_z_max:  2.5    # upper Z — preserves tall objects above [m]
  max_range_m:   52.0    # 50 m eval limit + 2 m margin [m]
```

**Code**: `filter_lidar_pts()` in `AutoLabeling/src/autolabeling/utils/lidar.py`,
called inside the `_load_pts_ego` closure in `pipeline.py` so both B1 and O3 automatically
receive a filtered cloud.

---

### Step 0 — In-Mask LiDAR Point Filtering (HDBSCAN)

Used internally by O3 (and available for experimentation in O2).  A SAM3 binary mask is not
perfectly tight: ground-plane returns, adjacent-object bleed, and occlusion artifacts can
project inside the mask boundary.  HDBSCAN on the 3-D ego-frame coordinates of in-mask
LiDAR points isolates the genuine object surface.

**Why HDBSCAN over DBSCAN**: DBSCAN needs a fixed `eps` (neighbourhood radius).  LiDAR
point density falls off with range², so a radius that works at 5 m is wrong at 40 m.
HDBSCAN needs only `min_cluster_size`; it builds a full density hierarchy and selects
clusters adaptively.  Reference: Campello et al. (PAKDD 2013).

**Why ego space, not image plane**: Clustering in metric 3-D space means the distance
threshold is physically meaningful and range-invariant.

**`cluster_selection_epsilon`**: Prevents the hierarchy from splitting a single object
surface into many subclusters (roof, hood, sides).  Set to 0.5 m — within a car's surface
spread, less than the gap to an adjacent object.

**Post-filtering**: Only the **largest cluster** is kept; all other clusters and noise points
(HDBSCAN label = -1) are discarded.

**Per-class parameters**: `min_cluster_size` and `cluster_selection_epsilon` are tuned per class
and per dataset, because expected LiDAR point counts differ by object size and sensor beam count.
`min_samples=1` (least aggressive noise rejection) is fixed for all classes.

| Class | `min_cluster_size` | `cluster_eps` (m) | Rationale |
|---|---|---|---|
| `pedestrian` | 3 | 0.20 | Small, sparse returns; tight eps to avoid ground bleed |
| `bicycle` | 3 | 0.20 | Similar size to pedestrian |
| `motorcycle` | 3 | 0.30 | Slightly wider frame |
| `car` | 5 | 0.50 | Larger surface; eps merges roof/hood/sides |
| `truck` | 5 | 0.60 | Wider body than car |
| `construction vehicle` | 5 | 0.60 | Similar to truck |
| `bus` | 5 | 0.80 | Very long body; wide eps needed to merge end-to-end returns |
| `trailer` | 5 | 0.70 | Long, flat surface |

nuScenes (32-beam) uses the same values; ECP (64-beam) currently uses the same defaults —
per-dataset tuning is possible via the `hdbscan` block in `ecp.yaml` / `nuscenes.yaml`.

**Config** (under `sam3d_objects.hdbscan` in the per-dataset YAML):
```yaml
hdbscan:
  pedestrian:
    min_cluster_size: 3
    min_samples: 1
    cluster_eps: 0.20
  car:
    min_cluster_size: 5
    min_samples: 1
    cluster_eps: 0.50
  # … one block per class
```

**Code**: `filter_inmask_lidar_hdbscan()` in `AutoLabeling/src/autolabeling/utils/lidar.py`.
Called from both O3 (`sam3d_objects.py`) and B1 Stage 1 (`pipeline.py`).
Tuning notebook: `Testing/lidar_filtering_exploration.ipynb`.

---

### O1 — Sparse LiDAR Pointmap

**`pointmap_mode: o1_lidar`**

Replaces the MoGe pointmap with a sparse per-pixel LiDAR pointmap.  Pixels with a LiDAR
return get real metric XYZ; all other pixels are `NaN`.

**Coverage**: ~5 000–15 000 pixels out of ~1.7 M (~0.5 %) for a 1920×900 image.

**NaN handling inside the model**: `PointPatchEmbed` processes 8×8 pixel windows.  Windows
with no valid XYZ receive a learned `invalid_xyz_token` embedding — the model was trained
with random pointmap dropout so it can reconstruct from image features alone where depth
is absent.

| Property | Baseline (MoGe) | O1 (LiDAR) |
|---|---|---|
| Depth scale | Relative (arbitrary) | Metric (metres) |
| Coverage | Dense (~100 %) | Sparse (~0.5 %) |
| Object distance tz | Model prior | Directly from LiDAR Z |
| Object size / scale | Model prior | Constrained by metric scale |

**Known limitations**
- Covariate shift: the model was trained on dense MoGe maps; sparse LiDAR is OOD for
  windows that do receive data.
- Objects with zero visible in-mask LiDAR points fall back to `invalid_xyz_token` everywhere,
  equivalent to `pointmap=None`.

**Code**: `SAM3DObjectsModel._compute_lidar_pointmap()` in
`AutoLabeling/src/autolabeling/models/sam3d_objects.py`.

---

### O2 — MoGe + Global Affine Calibration

**`pointmap_mode: o2_moge_affine`**

MoGe produces a dense relative depth map Z_moge.  A single global affine transform
`Z_metric = a * Z_moge + b` is fitted per frame via least-squares on all in-image LiDAR
returns, converting the full map to metric scale.

**Fitting procedure**
1. Run MoGe → relative depth map `Z_moge (H, W)`.
2. Project LiDAR sweep into camera frame → visible metric depths `Z_vis` at pixels `(u, v)`.
3. Sample `Z_moge` at the LiDAR pixel locations → pairs `(Z_moge_i, Z_lidar_i)`.
4. Filter: keep pairs with `0.5 m < Z_lidar < 80.0 m` (discard near clutter and far noise).
5. Fit via `np.linalg.lstsq`:  `[a, b] = argmin ‖A [a,b]ᵀ − Z_lidar‖²`  where `A = [Z_moge, 1]`.
6. Apply `Z_metric = max(a * Z_moge + b, 0.1)` to full map (clip unphysical negatives).
7. Rebuild XYZ from metric Z and camera intrinsics K.

Using all in-image LiDAR (not just in-mask) gives a better-conditioned fit: points span a
wider depth range (~2–60 m vs. a narrow per-object band).

| Property | Baseline (MoGe) | O1 (LiDAR) | O2 (Global affine) |
|---|---|---|---|
| Depth scale | Relative | Metric | Metric |
| Coverage | Dense | Sparse | Dense |
| Fit scope | — | — | One (a,b) per frame |
| Handles occluded objects | Yes | No (no returns) | Yes |

**Known limitation**: MoGe's depth compression is depth-range-dependent.  A single global
(a,b) averages over all distances in the frame.  Objects at a very different depth from the
bulk of the LiDAR returns may be miscalibrated.

**Code**: `SAM3DObjectsModel._compute_moge_affine_pointmap()`.

---

### O3 — Per-Object Local Affine Calibration (Novel Contribution)

**`pointmap_mode: o3_local_affine`**

Like O2, but a separate `(a, b)` is fitted for each detected object, using only LiDAR
points on that object's surface.  The motivation: MoGe's relative-to-metric compression
factor changes with depth.  A car at 30 m has a different scale factor than a bus at 8 m.
A per-object fit captures the depth range that is actually relevant to that object.

This is an original contribution; MoGe (Wang et al. 2024) and HDBSCAN (Campello et al.
2013) are cited as tools.

**Procedure (per object, per frame)**
1. MoGe runs **once** per frame → `Z_moge (H, W)` (shared across all objects).
2. For each detected object with SAM3 mask `M`:
   a. Find in-mask LiDAR pixels: `u_vis, v_vis` where `M[v_int, u_int] == True`.
   b. Run HDBSCAN (Step 0 params) on the 3-D ego-frame coordinates of those points
      → dominant cluster = clean object surface.
   c. Sample `Z_moge` at the clean surface pixels → pairs `(Z_moge_i, Z_lidar_i)`.
   d. Fit `(a, b)` via least-squares on these object-specific pairs.
   e. Apply `Z_metric = max(a * Z_moge_full + b, 0.1)` to the **full** `(H, W)` map
      (not just in-mask — SAM3D Objects crops to the bounding box internally).
   f. Rebuild full-frame `(H, W, 3)` XYZ pointmap in PyTorch3D space.
   g. Run SAM3D Objects inference with this object-specific pointmap.
   h. Discard the pointmap; repeat for next object.

**Fallback chain**
| Condition | Fallback |
|---|---|
| HDBSCAN finds no dominant cluster | Use all in-image LiDAR → global (a,b) |
| Too few clean points for reliable fit | Use all in-image LiDAR → global (a,b) |
| No LiDAR at all for this frame | Unscaled MoGe (a=1, b=0) |

The fallback mode is logged per object: `[local]`, `[global_fallback]`, `[unscaled_fallback]`.

**Why full-frame pointmap (not in-mask only)**: SAM3D Objects internally crops to a
bounding box larger than the SAM3 mask.  Context pixels outside the mask also receive the
calibrated depth, improving reconstruction of partially occluded objects and background
context used by the diffusion model.

**HDBSCAN params** (same as Step 0):
`min_cluster_size=3`, `min_samples=1`, `cluster_selection_epsilon=0.5 m`, `metric='euclidean'`.

**Minimum points for affine fit**: `min_pts=4` (after clustering).

**Known limitation — large objects viewed side-on at close range**

The affine `Z_metric = a·Z_moge + b` is a **global linear transform per object**.  It works
well when the object surface is roughly equidistant from the camera (head-on car, flat wall),
because all in-mask LiDAR points are at similar depths and MoGe's relative values are nearly
uniform.

When a large vehicle is close and viewed side-on, the surface spans a substantial depth range
(front door at ~3 m, rear door at ~5–6 m for a van at 4 m lateral distance, fully visible from
the side).  In this regime:
- Many in-mask LiDAR points exist (the dense side surface is well-illuminated by the sensor).
- But MoGe's relative depth profile across that side surface is not guaranteed to be accurate
  — MoGe was trained on diverse internet images and may compress/distort the within-object
  depth gradient for large close objects that fill much of the field of view.
- A linear `(a, b)` cannot compensate for a non-linear or inconsistent MoGe profile.
- The resulting OBB — fitted to the reconstructed mesh vertices — will be mis-scaled or
  mis-oriented compared to the true vehicle footprint.

More in-mask points improve the statistical conditioning of the linear fit, but they cannot
fix the fundamental problem if MoGe's within-object depth shape is wrong.  This is a
motivation for O4 (dense depth completion), which would replace MoGe with a depth map
that is globally metric and accurate for close large objects.

See also edge case E5.

**Code**
- Full pipeline: `SAM3DObjectsModel._compute_local_affine_ptmap_for_object()` in
  `AutoLabeling/src/autolabeling/models/sam3d_objects.py`.
- Notebook reference: `Testing/autolabeling_pipeline.ipynb`.

---

### O4 — Dense Depth Completion  *(implemented — CompletionFormer)*

**Network choice**: CompletionFormer (Zhang et al., CVPR 2023).  
**Checkpoint**: KITTIDC_L1L2.pt (~334 MB), pretrained on KITTI Depth Completion.

Full reasoning for the network choice (CompletionFormer vs. MapAnything vs. BP-Net vs. OGNI-DC), the global-per-camera vs. per-mask workflow decision, the parallax filtering strategy, fallback chain, and expected wins/losses vs. O3 are documented in `lidar_integration_plan.md` § O4.

Directly addresses the O3 limitation for large close side-on vehicles (§ 3.O3 known limitation and § 5.E5): a depth completion network trained on real LiDAR+image pairs produces accurate per-pixel metric depth across the visible side surface where MoGe's relative gradient is inaccurate.

**Implementation**:
- `Models/CompletionFormer/` — network source (patched for PyTorch 2.x: `pvt.py` stubs removed `mmseg`/`mmcv` imports and skips missing pretrained backbone files; DCNv2 CUDA kernel patched for removed THC headers)
- `Testing/depth_completion.ipynb` — standalone prototype: sparse LiDAR → CompletionFormer → dense depth → PyTorch3D pointmap, with Plotly 3D visualisation
- `Testing/autolabeling_pipeline.ipynb` — integrated as `POINTMAP_MODE = 'o4_ground_filter'`
- `AutoLabeling/src/autolabeling/models/sam3d_objects.py` — `_load_cformer()`, `_compute_dense_completion_pointmap()`, branch in `run_frame()`
- `AutoLabeling/configs/{nuscenes,ecp}.yaml` — `pointmap_mode: o4_ground_filter`, `cformer_ckpt`, `lidar_lines`

**Key config values**:
```
data_name      = 'KITTIDC'
prop_time      = 6
affinity       = 'TGASS'
affinity_gamma = 0.5
conf_prop      = True
preserve_input = True    ← hard-anchors LiDAR pixels in the output (MAE → ~0 at anchors)
lidar_lines    = 32      (nuScenes) / 64 (ECP)
```

**Workflow per camera**:
1. Pre-filtered aggregated LiDAR (`pts_ego`) passed through PseudoLabeler ground filter (step below)
2. Filtered cloud projected into camera → sparse `(H,W)` depth map using `np.minimum.at` on `np.full(..., np.inf)` (min-depth per pixel, foreground wins)
3. CompletionFormer: `{'rgb': (1,3,H,W), 'dep': (1,1,H,W)} → {'pred': (1,1,H,W)}`
4. Back-project dense depth → `(H,W,3)` PyTorch3D pointmap `(-X_cam, -Y_cam, Z_cam)`
5. Same fixed pointmap passed to all SAM3D Objects calls for that camera — no HDBSCAN, no MoGe

**Fallback**: if no LiDAR is available for a frame, falls back to MoGe baseline.

---

#### O4 Ground Filter — PseudoLabeler

Ground-plane LiDAR returns (road surface between ego and a distant object) project into the
lower pixels of foreground object masks.  With `preserve_input=True`, CompletionFormer
hard-anchors its output at those pixels to road depth — so the completed map shows the road
surface *inside* the object's image region, causing the object to appear incorrectly close
and the resulting OBB to be misplaced.

**Fix**: remove ground-classified returns from `pts_ego` before building the sparse depth
map.  The **PseudoLabeler** (see Stage 2 above) predicts ground height `z_ground(x,y)` at
every LiDAR point location.  `get_ground_bool(pc, inlier_thres=0.10)` classifies a point as
ground if `z_point − z_ground ≤ inlier_thres`.

**Why 0.10 m (not the model default 0.40 m)**: The model's default 0.40 m threshold is
tuned for semantic segmentation and removes points up to 40 cm above the predicted surface —
this would remove bicycle frame returns at ~0.20 m and bumper returns at ~0.30 m.  0.10 m
targets road-surface returns only, keeping all object surfaces.

**Why not strict `z > z_ground` (threshold 0.0)**:  PseudoLabeler is trained with an
asymmetric lower-envelope loss (`z_predicted ≤ actual z_ground`), so all true ground points
have positive clearance `z − z_ground > 0`.  A strict threshold of 0 removes nothing.
`inlier_thres = 0.10` captures the true ground surface by accepting the small positive bias.

**PseudoLabeler pre-fitting** (shared between O4 and B1 Stage 2):
- A single pre-fitting pass runs over all frames **before** the Body and Objects pipeline
  stages.  Each frame's fitted model state_dict (~10 KB) is stored in `pl_states: Dict[int,
  Optional[dict]]` in RAM.
- O4 and B1 Stage 2 both call `_restore_pseudolabeler(pl_states[i], device, dev_root)` to
  reconstruct the model — no redundant fitting per stage.
- Only frames where at least one Body or Objects result is still pending trigger a fit; frames
  with existing checkpoints skip fitting (`pl_states[i] = None`).
- **O5 does NOT need PseudoLabeler** — when `pointmap_mode == 'o5_mask_hdbscan'` the ground
  filter step is skipped entirely; the pre-fitting loop only runs if O4 or B1 is active.

**PseudoLabeler training** (`_fit_pseudolabeler()` in `pipeline.py`):
- Optimiser: AdamW, lr=1e-2, weight_decay=1e-4
- Scheduler: CosineAnnealingLR, T_max=2500, eta_min=1e-4
- Max steps: 2500
- Early stopping: patience=300 steps, **with warmup window of 200 steps**
  - Both best-loss tracking AND patience counting are deferred until after the warmup.
    During steps 0–199 the optimizer runs freely without updating `_best_loss`.
  - **Why defer tracking (not just patience)**: random init produces `pred ≈ 0`; for flat
    terrain (nuScenes Boston/Singapore) the ground is also near `z ≈ 0`, so `loss_0` is
    accidentally low.  If `_best_loss` is set at step 0, the step-1 overshoot (lr=1e-2)
    raises loss and the recovered model may never beat the random-init baseline.  Patience
    then fires at exactly `warmup + patience = 500` steps on every frame.
  - With tracking deferred to step 200, `_best_loss` is set from the actual trained state
    after the overshoot/recovery phase.  Typical step counts: 500–2500 for varied terrain
    (ECP Strasbourg), converging faster for flat terrain (nuScenes Boston/Singapore).

**Config**: `sam3d_objects.pl_ground_inlier_thres: 0.10  # m`

**Code**:
- Pre-fitting: `_fit_pseudolabeler()` / `_restore_pseudolabeler()` and the `pl_states` dict in `AutoLabeling/src/autolabeling/pipeline.py`
- O4 integration: `SAM3DObjectsModel._filter_above_ground()` called from `run_frame()` in `AutoLabeling/src/autolabeling/models/sam3d_objects.py`
- Notebook: `Testing/autolabeling_pipeline.ipynb` — cell `pseudolabeler_ground` (O4 only, guarded by `POINTMAP_MODE == 'o4_ground_filter'`)

**`preserve_input=True` rationale**: CompletionFormer's default behaviour uses LiDAR as soft guidance. Setting `preserve_input=True` hard-anchors the output at LiDAR pixel locations (overrides network prediction with the measured depth there), making MAE at LiDAR pixels ≈ 0. The network still interpolates freely between anchors. This is the correct choice for metric accuracy — LiDAR provides ground-truth depth at those pixels and the network should not deviate from it.

---

### O5 — CompletionFormer + Per-Mask HDBSCAN Anchor Cleaning  *(implemented)*

**`pointmap_mode: o5_mask_hdbscan`**

Like O4, uses CompletionFormer to produce a dense metric depth map fed as the SAM3D Objects
pointmap.  Unlike O4, O5 **skips the PseudoLabeler ground filter** and instead cleans the
sparse LiDAR anchor map using per-mask HDBSCAN before passing it to CompletionFormer.

**Motivation**: O4's ground filter is the most expensive and brittle step — it requires
fitting the PseudoLabeler MLP per frame, which adds latency and can fail if the LiDAR sweep
is sparse.  The actual harm from ground returns is that they project inside foreground masks
and anchor CompletionFormer to the wrong (road) depth.  HDBSCAN already isolates in-mask
surface clusters for O3's affine fit — the same logic can be applied here to simply remove
non-surface points from the LiDAR anchor map before completing the depth map.  Points
outside all object masks are kept as-is (ground returns outside masks do not corrupt
foreground object depths).

**Workflow per camera**:
1. Pre-filtered aggregated LiDAR → projected into camera frame → `u_vis, v_vis, Z_vis, pts_ego_vis`.
2. For each detected object mask (only `pipeline_type == 'objects'` masks):
   a. Find in-mask LiDAR indices.
   b. Run HDBSCAN on `pts_ego_vis[in_mask]` using per-class params (same as O3/Step 0).
   c. Label the dominant cluster (most points); mark all other in-mask points as noise.
   d. Set `keep[noise_idx] = False`.
3. Build cleaned sparse depth map from `u_vis[keep], v_vis[keep], Z_vis[keep]`
   and **all out-of-mask points** (unaffected).
4. CompletionFormer: cleaned sparse map + RGB → dense metric depth → PyTorch3D pointmap.
5. Same fixed pointmap passed to all SAM3D Objects calls for that camera.

**Key difference from O4**: No PseudoLabeler needed; ground returns inside masks are
removed by HDBSCAN isolation rather than predicted ground-height thresholding.

**Key difference from O3**: O3 uses HDBSCAN per-object to extract a local affine scale for
MoGe.  O5 uses HDBSCAN to clean the LiDAR anchor before CompletionFormer — it does not
use MoGe at all.

**Observed performance vs. O3 (nuScenes mini, 8-class)**:
- mAP nearly identical (~flat): detection recall and centre localisation are both anchored
  by the same HDBSCAN-cleaned LiDAR depth, so neither metric changes substantially.
- TP errors improve — notably motorcycle AOE (0.319 → 0.092) and bicycle/car ASE.
  CompletionFormer produces a shape-complete depth map that the diffusion model uses to
  recover better extent and orientation, even when the LiDAR surface is sparse.

**Fallback**: if no LiDAR is available for a frame, falls back to MoGe baseline (same as O4).

**Config**:
```yaml
sam3d_objects:
  pointmap_mode: o5_mask_hdbscan
  cformer_ckpt: /path/to/KITTIDC_L1L2.pt   # same checkpoint as O4
  lidar_lines: 64    # ECP: 64 | nuScenes: 32
  hdbscan:           # same per-class params as O3 (Step 0)
    pedestrian: {min_cluster_size: 3, min_samples: 1, cluster_eps: 0.20}
    # … one block per class
```

**Implementation**:
- `SAM3DObjectsModel._build_o5_sparse_points()` in `AutoLabeling/src/autolabeling/models/sam3d_objects.py`
- Branch `elif self.pointmap_mode == 'o5_mask_hdbscan':` in `run_frame()`
- Notebook: `Testing/autolabeling_pipeline.ipynb` — `build_o5_sparse_points()` function, `POINTMAP_MODE = 'o5_mask_hdbscan'`
- `pipeline.py`: `_needs_pseudolabeler` check excludes O5; `_mode_labels` includes O5 label

---

### SS Correction — Mid-Pipeline Voxel Suppression  *(implemented, flag-guarded)*

**Not an alternative to O1–O5** — SS Correction is an **orthogonal dimension** that can be
layered on top of any O4 or O5 run.  It is activated by a separate config flag:

```yaml
sam3d_objects:
  ss_correction: false   # true = enable; false = strict no-op (identical to unmodified code)
```

**Motivation**

O5 (and O4) produce far more false positives than O3 on nuScenes.  The root cause: objects
with zero in-mask LiDAR returns (hidden, far-away, or between scan lines) receive plausible
metric depths from CompletionFormer (it interpolates based on surrounding returns and image
features).  SAM3D Objects then predicts a confident 3D box — but the object was never there,
it was background or an object outside the evaluation range.  O3 accidentally suppresses
these by leaving the depth at garbage MoGe-relative units, outside metric evaluation range.

SS Correction targets this by suppressing **voxels that are placed behind the visible surface**
according to the LiDAR-anchored CFormer depth map.  It intercepts SAM3D Objects mid-pipeline:
after the sparse-structure (SS) diffusion generates a voxel grid (`coords`) but before the
latent `SLAT` is decoded into a mesh.  Voxels at `Z > Z_surface + 0.3 m` (the median depth
of in-mask LiDAR-anchored pixels) are dropped from `coords`.

**Confidence mask design**

LiDAR-anchored pixels (where `preserve_input=True` hard-anchors CFormer) are given full
suppression trust — if a voxel projects behind this depth, it is behind the real surface.
CFormer-interpolated pixels (no LiDAR anchor) receive no suppression trust (asymmetric).
Only the LiDAR-anchored pixels form the `anchor_mask`.

**Voxel projection math**

```
vox_norm = coords[:, 1:] / 64.0 - 0.5          # [0,63]^3 → [-0.5, 0.5]^3
vox_p3d  = scale * (vox_norm @ R.T) + trans     # object-local → P3D camera space
vox_Z    = vox_p3d[:, 2]                        # camera-space depth
suppress_mask = vox_Z > Z_surface + 0.3         # metres behind surface
coords = coords[~suppress_mask]
```

where `R, trans, scale` are from `ss_return_dict` (SAM3D Objects output dict from Stage 1
diffusion).  `coords` is shape `(N, 4)` — `[batch_idx, z, y, x]` in voxel indices `[0, 63]`.

---

#### Touched files and line numbers

All changes are wrapped in `# ── THESIS MODIFICATION: SS LiDAR Correction` /
`# ── END THESIS MODIFICATION` comment blocks (or `# THESIS:` inline comments), making them
easy to identify and revert.  Setting `ss_correction: false` in the config guarantees a
**strict no-op** — every code path is guarded by either `if ss_correction_fn is not None:`
or `if self.ss_correction:`.

---

**`AutoLabeling/configs/nuscenes.yaml`** and **`AutoLabeling/configs/ecp.yaml`**
- Added `ss_correction: false` under `sam3d_objects:` (disables SS Correction by default).

---

**`AutoLabeling/src/autolabeling/models/sam3d_objects.py`**

| Line | Change |
|------|--------|
| L54 | `ss_correction: bool = False` added to `__init__` signature |
| L66 | `self.ss_correction = ss_correction` stored |
| L538–L642 | `_build_ss_correction_fn(self, dense_depth, anchor_mask, K, binary_mask)` method added (THESIS MODIFICATION block) |
| L675–L676 | `_dense_depth_ss = None; _anchor_mask_ss = None` initialised (O4 branch) |
| L687–L693 | Build `_anchor_mask_ss` from HDBSCAN-cleaned LiDAR pixels when `ss_correction=True` (O4 branch) |
| L699–L700 | Same initialisers (O5 branch) |
| L711–L716 | Same anchor mask build (O5 branch) |
| L722–L723 | `_dense_depth_ss = None; _anchor_mask_ss = None` for O3/baseline branches (no SS Correction available) |
| L761–L767 | Build `_ss_fn` per-mask and pass `ss_correction_fn=_ss_fn` to inference call |

---

**`AutoLabeling/src/autolabeling/pipeline.py`**

| Line | Change |
|------|--------|
| L844 | `_ss_correction = bool(getattr(_lidar_cfg, 'ss_correction', False))` — read config flag |
| L855 | `ss_correction=_ss_correction` passed to `SAM3DObjectsModel(...)` |

---

**`Models/SAM3D/sam-3d-objects/sam3d_objects/pipeline/inference_pipeline_pointmap.py`**

*(SAM3D Objects source — modified in-place)*

| Change | Location |
|--------|----------|
| `ss_correction_fn=None` parameter added to `run()` | THESIS MODIFICATION block in `run()` signature |
| Intercept block after `coords = ss_return_dict["coords"]` that calls `ss_correction_fn(coords, ss_return_dict)` and updates `coords` | THESIS MODIFICATION block after SS generation, before SLAT decode |

---

**`Models/SAM3D/sam-3d-objects/notebook/inference.py`**

*(SAM3D Objects notebook API — modified in-place)*

| Change | Location |
|--------|----------|
| `ss_correction_fn=None` added to `Inference.__call__()` | THESIS MODIFICATION block in `__call__` signature (L110–L113) |
| `ss_correction_fn=ss_correction_fn` forwarded to `self._pipeline.run(...)` | L127 |

---

**`Testing/autolabeling_pipeline.ipynb`** — cell 20 (`run_sam3d_objects` + dispatch)

| Cell-20 line | Change |
|-------------|--------|
| L159–L218 | `build_ss_correction_fn(dense_depth, anchor_mask, binary_mask)` helper added (THESIS MODIFICATION block) |
| L221 | `ss_data=None` added to `run_sam3d_objects` signature |
| L260–L268 | Per-mask `_ss_fn = build_ss_correction_fn(...)` built and passed as `ss_correction_fn=_ss_fn` to `inference_model(...)` |
| L501–L512 | O4 dispatch: `_ss_data_o4` built from `_anch_o4` when `SS_CORRECTION=True`, passed as `ss_data=_ss_data_o4` |
| L529–L540 | O5 dispatch: same pattern → `_ss_data_o5` / `ss_data=_ss_data_o5` |

Config cell (cell 4) previously added: `SS_CORRECTION = False`

---

## 4. SAM3D Body — LiDAR Options

### Why the SAM3D Objects pointmap strategy cannot be applied to SAM3D Body

SAM3D Objects and SAM3D Body are **architecturally incompatible** with respect to LiDAR
integration, which is why the two pipelines use different strategies.

**SAM3D Objects** was designed with an explicit external depth input.  A public API hook
accepts a `(H, W, 3)` pointmap tensor:
```python
inference_model(img_rgb, mask, seed=42, pointmap=ptmap_t)
```
Internally, `PointPatchEmbed` converts this into depth feature tokens that condition the
sparse-structure **diffusion generator** at every cross-attention step.  The architecture
was built to receive and incorporate external metric geometry — replacing or calibrating
the pointmap directly changes what the model generates.

**SAM3D Body** is a **regression model** (based on CLIFF/HMR), not a diffusion model.
It takes an image crop, passes it through a ViT encoder, and a regression head directly
predicts SMPL body pose, shape, and camera parameters `[s, tx_norm, ty_norm]` in one
forward pass.  There is no pointmap input, no `PointPatchEmbed`, and no geometry
conditioning pathway.  Depth (`tz`) is an *output* derived from the predicted scale `s`
via the CLIFF formula — it is not something the model is conditioned on during inference.

To inject LiDAR at the same architectural level as in SAM3D Objects, you would need to
modify the internal `CameraEncoder` to accept a depth prior as an additional input token,
and re-train or fine-tune — this is the planned **B2** option.  **B1** is the practical
alternative: accept the model's depth estimate and then override it post-inference with
a direct LiDAR measurement.

---

### Background: how SAM3D Body estimates depth (tz)

SAM3D Body uses the **CLIFF** camera model to place the estimated SMPL body in 3D camera
space.  The 3D translation of the body root (pelvis) is:

```
tz = 2 * focal_length / (bbox_size * s)
tx = (u_center - cx) / fx * tz
ty = (v_center - cy) / fy * tz
```

where `s` is a scale parameter predicted from the image crop by appearance alone.
`tz` is the primary source of error: a small distant person and a large close person
look identical in a crop, so `s` — and therefore `tz` — can be badly wrong.
LiDAR provides a direct, metric measurement of the correct depth.

This has **nothing to do with MoGe**. MoGe is only used in SAM3D Objects.

---

### B1 — LiDAR-based Position Correction  *(two-stage, post-inference)*

**Overview**: B1 is a two-stage post-inference correction applied to each pedestrian mesh.
Stage 1 fixes the depth and lateral position using in-mask LiDAR; Stage 2 anchors the
mesh feet to the local ground surface using the **PseudoLabeler**.  Both stages are
post-inference overrides — the model weights and forward pass are not changed.

---

#### Stage 1 — Depth (tz) correction

SAM3D Body computes `tz = 2 * focal_length / (bbox_size * s)` where `s` is predicted from
image appearance alone.  A small distant person and a large close person look identical
in the crop, so `tz` — and thus `tx`, `ty` — can be badly wrong.  Stage 1 replaces `tz`
with a direct LiDAR measurement and recomputes the full `cam_t`.

**Step by step**
1. Project LiDAR sweep into camera frame — `u_vis, v_vis, Z_vis, pts_ego_vis`.
2. For each pedestrian, get its SAM3 binary mask (or rectangular bbox fallback for
   ViTDet-only detections that have no SAM3 mask).
3. Find LiDAR pixels inside the mask: `in_mask = mask[v_int, u_int]`.
4. Run `filter_inmask_lidar_hdbscan()` on `pts_ego_vis[in_mask]` → dominant surface cluster.
5. `tz_lidar = median(Z_vis[in_mask][keep])`.
6. Recompute `tx_new = (u_cen - cx) / fx * tz_lidar`, `ty_new = (v_cen - cy) / fy * tz_lidar`
   from the mask centroid `(u_cen, v_cen)`.
7. `delta = [tx_new, ty_new, tz_lidar] - cam_t_pred`
8. Shift `vertices += delta`.  Update `cam_t = [tx_new, ty_new, tz_lidar]`.
   **`joints_3d` are body-relative and are NOT shifted** — the projection
   `joints_cam = joints_3d + cam_t` picks up the corrected `cam_t` automatically.

**Fallback chain**

| Condition | Behaviour | Mode tag |
|---|---|---|
| No LiDAR in mask | Skip — keep predicted tz | `no_lidar` |
| HDBSCAN finds no cluster | 15th-percentile of all in-mask depths (see below) | `+p15_fallback` |
| Normal | Median of dominant HDBSCAN cluster | `+hdbscan` |

**15th-percentile fallback rationale**

When HDBSCAN finds no coherent cluster, the correction still applies using the 15th
percentile of all raw in-mask LiDAR depths:

- In-mask returns should be on or behind the object surface — not in front (assuming
  approximate LiDAR/camera time alignment).  The closest return is the front surface.
- The 15th percentile (not the minimum) discards the nearest ~15% to guard against stray
  ground bleed at the feet, timing artifacts, and mask boundary imprecision.
- The median of all in-mask points would land in the middle of the body depth, which is
  worse than targeting the front surface.

---

#### Stage 2 — Ground anchoring

After Stage 1, `cam_t` sits at the mask-centroid depth (approximately the torso/pelvis).
The mesh may still float or clip depending on body pose.  Stage 2 shifts the mesh so its
lowest vertices sit on the local ground surface.

**PseudoLabeler ground estimation**

The PseudoLabeler is a lightweight per-frame neural network that fits a continuous ground
height surface to the raw LiDAR sweep (see TerraSeg / PseudoLabeler scripts).  It is fit
on every frame before B1 runs and produces a callable `ground_z(x, y)` — given any ego-
frame XY position it returns the predicted ground height Z.

**Step by step**
1. Fit PseudoLabeler on the full LiDAR sweep for this frame → ground height function.
2. Convert each pedestrian's `cam_t` to ego space: `ped_ego = R_c2e @ cam_t + t_c2e`.
3. Query ground height: `z_ground = pseudolabeler.predict(ped_ego[0], ped_ego[1])`.
4. Convert vertices to ego space; `z_foot = min(vertices_ego[:, 2])`.
5. `z_shift = z_ground - z_foot`.
6. Convert back to camera space: `delta_cam = R_c2e.T @ [0, 0, z_shift]`.
7. Shift `vertices += delta_cam` and `cam_t += delta_cam`.
   **`joints_3d` are NOT shifted** — same reasoning as Stage 1.
8. Refit OBB from shifted vertices.

---

**Config flags** in `sam3d_body_pipeline.ipynb`:
```python
B1_LIDAR_CORRECTION  = True   # enable/disable entire B1 (both stages)
B2_GROUND_CORRECTION = True   # enable/disable Stage 2 independently
```

**Code**
- Stage 1: `_apply_b1_depth_correction()` in `AutoLabeling/src/autolabeling/pipeline.py`
- Stage 2: `_apply_b2_ground_anchoring()` in `AutoLabeling/src/autolabeling/pipeline.py`
- PseudoLabeler: `TerraSeg/PseudoLabeler_scripts/pseudolabeler_model.py`
- Notebook reference: `Testing/sam3d_body_pipeline.ipynb`

**HDBSCAN params** (Stage 1): uses the pedestrian entry from the per-class HDBSCAN config
(`sam3d_objects.hdbscan.pedestrian`), currently `min_cluster_size=3`, `min_samples=1`,
`cluster_eps=0.20 m`.  The same `filter_inmask_lidar_hdbscan()` utility is shared with O3;
see Step 0 above for the full per-class table and rationale.
Shared utility: `autolabeling.utils.lidar.filter_inmask_lidar_hdbscan`.

---

## 5. Known Edge Cases

Observed failure modes and ambiguous situations that currently produce incorrect or
degraded results.  No fixes are implemented yet — this section exists to collect cases
before solutions are designed.

---

### E1 — Proximity-triggered false rider merge

**Description**: A pedestrian walks close to a parked or slow-moving bicycle, motorbike,
or motorcycle without riding it.  The OBB merge step (§2) only checks 3-D ego-space
distance between the body OBB center and the vehicle OBB center.  If the pedestrian
happens to be within `dist_thresh = 1.5 m`, they are incorrectly merged into the vehicle
detection and the pedestrian label is removed from `body_results`.

**Trigger conditions**
- Pedestrian walking alongside a parked bicycle or motorbike at the kerb
- Pedestrian stopped next to a motorcycle at a traffic light
- Dense crowd where a person stands near a bike even though neither is riding

**Why not caught by existing logic**: The merge heuristic has no pose or orientation
check — it doesn't verify that the pedestrian's feet are on the pedals or that the
facing direction aligns with the vehicle's heading.

---

### E2 — SAM3 masks from reflections in shop windows / glass facades

**Description**: SAM3 occasionally produces a high-confidence mask for a pedestrian or
vehicle that is actually a reflection in a shop window or glass building facade.
The reflection appears visually plausible but has no corresponding 3-D object at the
inferred depth — LiDAR returns come from the glass surface, not from the reflected
person.

**Observed consequences**
- SAM3D Body reconstructs a full SMPL mesh behind the glass (at roughly the glass depth
  after B1 tz correction, not at the reflected object's true depth).
- B1 Stage 1 will attempt to correct tz using LiDAR returns on the glass surface,
  placing the mesh flat against the window.
- The OBB has physically impossible dimensions (very shallow depth, correct height/width).

---

### E3 — Ground anchoring failure at pavement–road surface transitions

**Description**: B1 Stage 2 anchors pedestrian feet to the local ground height predicted
by the **PseudoLabeler** — a per-frame neural network that fits a continuous ground
surface to the LiDAR sweep.  When a pedestrian stands at the boundary between a raised
pavement (kerb) and the road surface, the PseudoLabeler must interpolate across a sharp
height discontinuity.  Its prediction at the boundary will be a smoothed-over blend,
placing the predicted ground height somewhere between the two surfaces.  The mesh is then
shifted to this intermediate height, causing slight floating above the road or clipping
into the pavement.

**Trigger conditions**
- Pedestrian at or crossing a kerb edge
- Steep ramps, speed bumps, or other abrupt ground-level changes beneath the pedestrian
- Scenes with multi-level ground structure (e.g. underpasses, parking garage entrances)

---

### E5 — O3 affine breakdown for large vehicles viewed side-on at close range

**Description**: When a large vehicle (van, truck, bus) is close to the camera and visible
from the side, the SAM3 mask contains hundreds of in-mask LiDAR returns spanning the full
side surface.  Despite this high point density, the resulting OBB is often poorly aligned.

**Root cause**: The O3 local affine `Z_metric = a·Z_moge + b` is a single global linear
transform per object.  It can only scale and shift MoGe's relative depth profile.  For a
large close vehicle the depth varies substantially across the mask (front to rear of the
side surface spans 2–4 m in the camera Z direction).  If MoGe's internal depth gradient
across that region does not accurately match the true geometry (e.g. it compresses the
gradient, predicting a too-flat side surface), the linear affine cannot fix this — the fit
absorbs the average discrepancy and leaves systematic error across the mask.  The resulting
mesh is reconstructed from an inaccurate pointmap and the OBB inherits the error.

**Distinguishing indicator**: BEV visualization shows a dense, spatially large HDBSCAN
cluster (correct: the sensor is seeing the whole side), but the projected OBB footprint is
rotated or scaled differently from neighboring vehicles that have sparser, head-on LiDAR.

**Counterintuitive detail**: Objects with *fewer* in-mask LiDAR points but at a consistent
single depth (e.g. a car at 20 m head-on) tend to produce better OBBs than objects with
*many* points spanning multiple depths, because MoGe's relative depth profile is accurate
for near-uniform-depth regions.

**Potential fixes**:
- O4 (`o4_ground_filter`) — replaces MoGe with CompletionFormer (LiDAR+RGB trained) depth
  map, with PseudoLabeler ground filtering before building the sparse anchor.
- O5 (`o5_mask_hdbscan`) — same CompletionFormer backend but uses per-mask HDBSCAN to
  clean ground returns from the sparse anchor; no PseudoLabeler needed.  Implemented and
  evaluated — TP errors (AOE, ASE) improve over O3; mAP is comparable.

---

### E4 — SAM3 multi-class detections for the same physical object

**Description**: SAM3 is run separately for each text prompt, so the same physical object
can generate confident masks under multiple class prompts simultaneously.  Examples:
a bicycle detected under both `"bicycle"` and `"motorcycle"`; a large van triggering both
`"car"` and `"truck"`; a pickup truck matching `"truck"` and `"construction vehicle"`.

**Current mitigation**: Cross-class deduplication (§1) removes the lower-priority
duplicate when mask IoU exceeds `iou_thresh = 0.5`.  This handles clean cases where the
two masks overlap well.

**Remaining failure modes**
- If the two masks are slightly offset (e.g. person-on-bike yields a `"pedestrian"` mask
  covering the upper body and a `"bicycle"` mask covering the lower half), IoU can fall
  below the threshold and both survive deduplication as separate detections.
- The priority ordering is a hand-coded heuristic; the "correct" class is not always the
  higher-priority one (e.g. a heavy-duty pickup could legitimately be either `"truck"` or
  `"construction vehicle"` depending on context).
- A better long-term solution might involve confidence-weighted selection or a
  classification head that re-scores competing masks in context.
