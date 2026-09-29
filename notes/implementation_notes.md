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

### Pre-processing: Per-Object Motion Compensation (ICP)

Before the aggregated multi-sweep point cloud is fed to SAM3D Objects, dynamic objects
are motion-compensated per mask.  Without this step, a moving car smears across all
aggregated sweeps, producing an elongated blob that misleads both HDBSCAN clustering
and SAM3D Objects reconstruction.  Motion compensation collapses all sweeps of a
dynamic object back to the anchor frame position, yielding a clean dense shape.

The compensation runs in two phases followed by classification.

---

#### Ground removal before Phase 1

Before Phase 1 runs, every sweep's point cloud is ground-filtered so ROI crops contain
only object-surface returns.

**Exploration notebook** (`lidar_aggregation.ipynb`): **TerraSeg-S** (PointTransformerV3
backbone, trained on OmniLiDAR) assigns every point a binary label (0 = ground,
1 = non-ground).  The resulting `sweep_nonground_pts[i]` lists are what Phase 1 crops
from.  TerraSeg is semantically accurate and handles slopes and kerbs correctly.

**Pipeline notebook** (`autolabeling_pipeline.ipynb`): TerraSeg is not yet integrated
into the ICP cell.  A simple Z-threshold fallback (`_ICP_Z_GROUND_MIN = 0.15 m`) drops
points below 15 cm ego-frame height.  This removes most road-surface returns but is
less robust on sloped roads.  The per-mask Z ceiling filter (`anchor_z_min +
CLASS_MAX_HEIGHT_M`) applied inside Phase 1 provides a secondary safeguard against
tall background objects regardless.

**Note**: PseudoLabeler is **not** used for this step.  PseudoLabeler is only used
in the O4 pointmap pipeline (CompletionFormer ground anchor cleaning) and B1 Stage 2
(pedestrian foot anchoring) — both of which require a continuous ground-height function
`z_ground(x, y)` rather than a per-point binary label.

---

#### Phase 1 — Tracking (no alignment)

Starting from the anchor sweep (t=0), walks **forward** (t+1, t+2, …) then **backward**
(t-1, t-2, …), propagating a search centre sweep-by-sweep.  In each sweep:

1. Crop a local ROI around the last known centroid using `CLASS_SEARCH_RADIUS_M`.
2. **Z pre-filter before HDBSCAN**: remove points outside
   `[anchor_z_min − Z_FLOOR_SLACK, anchor_z_min + CLASS_MAX_HEIGHT_M]`.
   - **Floor** (`anchor_z_min` = bottom of anchor cluster): the object bottom is always
     LiDAR-visible (it sits on the ground).  Robust, dataset-agnostic.
   - **Ceiling** (`anchor_z_min + CLASS_MAX_HEIGHT_M`): class-specific cap avoids stealing
     the cluster from a truck or overpass above the car.  Anchored to the bottom (not the
     top) so it works even when the roof is not visible in the anchor sweep.
3. Run HDBSCAN → pick dominant cluster nearest to last search centre (top-3 by size,
   then min centroid distance).
4. Update search centre if the centroid moved less than `CLASS_MAX_SPEED_MPS × dt`.

Outputs per sweep: `_sw_pts[i]` (cluster points), `_sw_cents[i]` (cluster centroid).
No alignment happens here — this phase is purely about locating the object.

---

#### Turning detection (between phases)

The centroid trail from Phase 1 is analysed for heading change before ICP runs:

```python
def _trajectory_yaw_rate(sw_cents, sw_data):
    # Split trail into two halves
    # Fit lstsq velocity direction to each half
    # Compute net heading change between the two halves
    # Divide by total time → deg/s
```

Using **two-half regression** (not step-by-step accumulation) makes this robust to
per-sweep centroid noise: random jitter averages within each half instead of accumulating
into a spuriously large total.

- `TURNING_YAW_RATE_DEG_S = 5.0 deg/s` — below this: straight/lane-change → yaw stripped
- Above threshold: turning → yaw applied in ICP (`allow_yaw=True`)

**Why lane changes don't need yaw**: a lane change is a lateral translation with near-zero
heading change.  The centroid trail stays parallel → net angle ≈ 0 → no yaw in ICP,
which is correct (the car body doesn't rotate during a lane change).

---

#### Phase 2 — ICP alignment (growing target)

Sweeps are processed in alternating order **t+1, t-1, t+2, t-2, …** (`zip_longest` of
forward and backward index lists).  Each sweep is aligned to `_agg_pts` — a **growing
target cloud** that starts as the anchor cluster and accumulates every successfully
compensated sweep.

**Why growing target**: the anchor alone is sparse (one LiDAR sweep hits only one side of
a car).  As sweeps accumulate, `_agg_pts` grows to cover more surfaces → ICP has richer
correspondences for later sweeps.

**Why alternating order**: sweeps at t+1 and t-1 are temporally close to the anchor and
move least; their compensation is most accurate.  Their points improve `_agg_pts` before
sweeps at t+2, t-2 (larger motion) align to it.

**Gate**: sweeps where Phase 1 found no valid cluster (`_sw_cents[i] is None`) are skipped
entirely.  Without the gate, the raw ROI crop (background noise, wrong centroid) would be
passed to `centroid_T`, producing a wildly wrong initialisation.

**Initialisation**: `centroid_T` translates the source cloud so its XY centroid matches the
target centroid.  This coarse shift handles the full object displacement; ICP then only
refines the residual shape-level misalignment.  Because `centroid_T` is used, the tight
`ICP_MAX_CORRESP = 0.4 m` is valid — ICP only needs to close the small remaining gap.

---

#### ICP metric — point-to-point vs. point-to-plane

| Metric | When to use | Why |
|---|---|---|
| **Point-to-point** | pedestrian, bicycle, motorcycle | 3D body/frame structure; distinct corners and edges provide good correspondences from all directions. Normals are unreliable on sparse, non-planar clouds. |
| **Point-to-plane** | car, van, truck, bus, construction vehicle | Flat side-wall panels. Point-to-point has no constraint perpendicular to the wall (any slide is equally good → lateral smear). Point-to-plane penalises offset along the surface normal, directly constraining the lateral direction. |

Normals for point-to-plane are estimated on the **target** cloud (`_agg_pts`) using a
neighbourhood of `ICP_MAX_CORRESP × 3 m` and oriented toward the ego vehicle at `(0,0,100)`
(upper hemisphere → consistent orientation regardless of object position in the scene).
Normals on the growing cloud improve as more sweeps accumulate.

**`allow_yaw` flag**: for straight-driving objects, the ICP rotation output is discarded
(only translation used).  Without this, ICP on an elongated car body finds a spurious local
minimum that includes large yaw — the car shape looks the same after a 180° rotation.

---

#### Final cloud

| Object state | Final cloud |
|---|---|
| **Dynamic** | Concatenation of all ICP-compensated per-sweep clusters |
| **Static** | All raw in-mask points from every sweep → HDBSCAN dominant cluster (no ICP; raw aggregation is correct for stationary objects) |

---

#### Downstream routing (static vs dynamic)

After motion compensation, the final cloud feeds two different downstream paths based on object class.

**Non-pedestrian objects → SAM3D Objects**

`run_frame` checks the `is_dynamic` flag from the ICP result per mask:
- **Static**: the raw aggregated ego-frame cloud (`o3_data['pts_ego_vis']`) is passed directly to
  `_compute_local_affine_ptmap_for_object`.  No ICP is involved.  HDBSCAN runs once *inside*
  that function to isolate the dominant surface cluster before the local affine fit.
- **Dynamic**: the ICP-compensated cloud (`_icp_meta['pts_ego_vis']`) replaces the aggregated
  cloud as input to the same function.  HDBSCAN still runs once inside `_compute_local_affine_ptmap_for_object`
  to clean the ICP output before fitting.

In both cases HDBSCAN runs exactly once inside `_compute_local_affine_ptmap_for_object`.
There is no additional clustering step between the ICP output and SAM3D Objects inference.

**Pedestrian objects → SAM3D Body / B1 Stage 1**

The same static/dynamic split applies, but the cloud is consumed by B1 depth correction:
- **Static**: the aggregated in-mask ego-frame cloud is filtered by `filter_inmask_lidar_hdbscan`
  → dominant cluster → `median(Z_vis[keep])` → `tz_lidar`.  No ICP lookup.
- **Dynamic**: the ICP cloud is retrieved from `mc_pts_by_mask_id` (a dict keyed by
  `id(binary_mask)` for each SAM3-sourced pedestrian mask, built from dynamic-only ICP results)
  → filtered by `filter_inmask_lidar_hdbscan` → surviving cluster is projected from ego to camera
  frame via `R_c2e.T @ (pts − t_c2e).T` → the Z (depth) column of the result → `tz_lidar`.

The ego→camera projection is a 3-D coordinate frame transform, not a 2-D pixel projection.
Only the Z component (camera-space depth) is needed.

`mc_pts_by_mask_id` is populated before B1 runs and contains only **dynamic** pedestrian results.
Static pedestrians have no ICP entry and use the aggregated sweep directly.

---

#### Fallback

When `USE_ICP=False` or no ICP sweep data is available, the cell falls back to the
pre-ICP pipeline: project the single anchor sweep's in-mask LiDAR into camera, run
HDBSCAN on the anchor-sweep points, and use the dominant cluster as `pts_comp_all`.
This keeps the output dict format identical so all downstream cells are unaffected.

---

#### Config (`Testing/lidar_aggregation.ipynb` cell `cell-motion`, pipeline cell `icp-motion-cell`)

| Parameter | Value | Meaning |
|---|---|---|
| `ICP_MAX_CORRESP_DIST` | 0.4 m | Max point-pair distance in ICP (valid because centroid init handles coarse offset) |
| `ICP_MAX_ITER` | 60 | ICP convergence iterations |
| `MIN_PTS_ICP` | 8 | Min points to attempt ICP (else centroid-only) |
| `Z_ANCHOR_FLOOR_SLACK` | 0.20 m | Extra slack below anchor bottom for Z pre-filter |
| `CLASS_MAX_HEIGHT_M` | per class | Ceiling above anchor bottom for Z pre-filter |
| `TURNING_YAW_RATE_DEG_S` | 5.0 deg/s | Threshold for turning detection |
| `CLASS_ICP_METRIC` | per class | `'p2l'` or `'p2p'` per class |

**Code**:
- Exploration notebook: `Testing/lidar_aggregation.ipynb` — cell `cell-motion`
- Pipeline: `Testing/autolabeling_pipeline.ipynb` — cells `icp-motion-hdr`, `icp-motion-cell`

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

**Sweep counts per dataset**:

| Dataset | LiDAR rate | Keyframe rate | Sweeps between keyframes |
|---------|-----------|---------------|--------------------------|
| nuScenes | 20 Hz | 2 Hz (annotated) | 9 non-keyframe + 1 keyframe = **10 total** |
| ECP | ~20 Hz | ~20 Hz (every frame is a keyframe in the nuScenes export) | 0 non-keyframe sweeps; N_BEFORE walks to prior annotated keyframes directly |

For nuScenes, `N_BEFORE=10` / `N_AFTER=10` captures **exactly one full keyframe interval** in each direction: the 9 intermediate non-annotated sweeps plus the neighbouring annotated keyframe.  This is the natural maximum — going beyond 10 would cross into the keyframe interval two steps away.

ECP annotates only a sparse subset of keyframes (~33 annotated out of thousands).  Between two consecutive annotated ECP frames there are many non-annotated keyframes.  `N_BEFORE` / `N_AFTER` walk the LiDAR linked list from the annotated anchor and pick up the nearest non-annotated sweeps in each direction.

**Config** (per dataset — values tuned to LiDAR beam count):
```yaml
lidar_aggregation:
  use_aggregation: true
  n_before: 3    # nuScenes (32-beam): more sweeps to compensate sparse density
  n_after:  3
  # n_before=10 / n_after=10 covers exactly one full keyframe interval (±500 ms)

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

**Mask erosion before HDBSCAN (O3)**

Before selecting in-mask LiDAR points for HDBSCAN, the SAM3 binary mask is eroded by
`MASK_ERODE_PX` pixels using an elliptical structuring element.  Erosion shrinks the mask
inward so that LiDAR returns near the mask boundary — which may belong to the adjacent
background or a neighbouring object that bleeds into the mask edge — are excluded from the
clustering step.  The un-eroded mask is still used for the final pointmap build and the
global fallback affine fit.

```
eroded_mask = cv2.erode(binary_mask, ellipse(2·MASK_ERODE_PX+1))
in_mask     = eroded_mask[v_int, u_int]   ← only eroded mask selects HDBSCAN points
```

Erosion is skipped when `MASK_ERODE_PX = 0` or when the mask area is below
`MASK_ERODE_MIN_PX` pixels (to avoid over-eroding very small masks to zero).

**Config** (Cell 4 in notebook):
```python
MASK_ERODE_PX     = 3   # pixels to erode (0 = disabled)
MASK_ERODE_MIN_PX = 0   # min mask area [px] to apply erosion (0 = always)
```

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

#### Hull Anchoring  *(O4 / O5 / O6)*

**Problem it solves**: CompletionFormer is run with `preserve_input=True`, which
hard-anchors the output at every pixel that has a LiDAR return — the network's predicted
depth there is overridden with the raw measured depth.  Ground-plane LiDAR returns that
project inside a foreground object's mask are therefore anchored to road depth (~2–8 m)
even though the object itself is further away.  The network then interpolates from these
wrong anchor points outward inside the mask, producing a depth map that is too shallow
across part of the object interior.  The resulting SAM3D OBB is shifted toward the camera.

**Fix**: after CompletionFormer produces the dense depth map, for each object mask:
1. Find in-mask LiDAR returns and run HDBSCAN to isolate the dominant cluster
   (same per-class params and erosion as O3 / Step 0).
2. Compute `min_depth = 5th-percentile of cluster depths − HULL_Z_MARGIN`.
   The 5th percentile is used (not the minimum) to guard against stray near outliers.
3. Build the convex hull of the SAM3 mask pixels in image space.
4. Clamp: `dense_depth[hull_pixels] = max(dense_depth[hull_pixels], min_depth)`.
   This raises any pixel inside the hull that is shallower than the LiDAR evidence allows.

The hull fill (step 3–4) uses the **full** (un-eroded) mask; erosion only applies to
HDBSCAN point selection (step 1), for the same border-bleed reason as in O3.

**Why hull, not just the mask pixels**: the HDBSCAN-cleaned LiDAR cluster may only cover
part of the mask (e.g. the front face of a car).  The convex hull of the mask ensures the
floor is applied over the entire projected object region, not just where LiDAR points land.

**Config**:
```python
HULL_ANCHORING = True    # enable/disable
HULL_Z_MARGIN  = 0       # [m] safety margin subtracted from 5th-pct depth
```

**Why NOT hull anchoring for O3**

O3 uses MoGe + per-object affine scaling, not CompletionFormer.  The problem hull
anchoring fixes — wrong hard-anchored pixels from ground returns — does not arise in O3:

- O3 does not hard-anchor anything.  The affine transform `Z_metric = a·Z_moge + b` is
  applied uniformly across the full map; there is no `preserve_input` mechanism.
- MoGe produces a smoothly-varying relative depth surface.  There are no specific pixels
  that are forced to road depth.  The pathology hull anchoring corrects simply does not exist.
- O3's main error mode is an inaccurate affine scale for large close objects (§ E5) — a
  per-object scale-and-shift issue.  Setting a depth floor inside the hull would be a
  no-op in most cases (the affine-scaled depth is already at the right level) and could
  not fix the underlying scale error where the fit is bad.

In summary: hull anchoring targets a CompletionFormer-specific artifact caused by
`preserve_input=True`.  O3 has a different error profile that affine scaling addresses
directly; adding a depth floor on top would not help and could introduce new artifacts.

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

---

#### Mask identification — `sam3_mask_idx` and `binary_mask`

Each SAM3D Body result dict carries two fields that enable O(1) mask lookup in B1 and the
cross-camera merge, replacing the previous fragile IoU-based search:

| Field | Content | When populated |
|---|---|---|
| `sam3_mask_idx` | Integer index into `ped_dets` (the SAM3 pedestrian detection list) | Always; `None` for ViTDet-only detections |
| `binary_mask` | `ped_dets[sam3_mask_idx]['binary_mask']` | SAM3-sourced detections only; `None` for ViTDet-only |

B1 Stage 1 uses `sam3_mask_idx` for a direct `ped_dets[sam3_mask_idx]['binary_mask']` lookup.
No IoU search is needed; the association is exact by construction since `SAM3DBodyModel.run_frame`
iterates the same `ped_dets` list in order when building bounding boxes from SAM3 masks.

**ViTDet-only detections** are those appended after all SAM3-sourced boxes — ViTDet is a
bbox-only detector that provides no segmentation mask:
- `sam3_mask_idx = None`, `binary_mask = None`
- B1 Stage 1: a rectangular mask is constructed from the ViTDet bbox for in-mask LiDAR selection
- Dynamic ICP path: skipped (no `binary_mask` means no lookup key in `mc_pts_by_mask_id`)
- Cross-camera merge Stage 1 border check: silently skipped (no `binary_mask` to test)

---

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

---

## 6. Multi-Camera Setup

### Overview

The pipeline processes every available camera for a given frame and aggregates detections
across all views.  A single shared ego-frame LiDAR cloud is loaded once and then projected
separately into each camera's image plane.

### Camera configurations

| Dataset | Cameras | Notes |
|---|---|---|
| nuScenes | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT` | Full 360° coverage, 6 cameras |
| ECP | `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT` | Frontal arc only, 3 cameras |

### nuScenes mini — train/val split

The pipeline runs on `split: train` → `mini_train` (8 scenes, ~323 keyframes).
`mini_val` (2 scenes) is kept untouched for evaluation.

**mini_train** (pseudo-label generation):

| Scene | Description |
|---|---|
| scene-0061 | Parked truck, construction, intersection, turn left |
| scene-0553 | Wait at intersection, bicycle, large truck, peds crossing |
| scene-0655 | Parking lot, parked cars, jaywalker, bendy bus, gardening vehicle |
| scene-0757 | Arrive at busy intersection, bus, wait at intersection, bicycle |
| scene-0796 | Scooter, peds on sidewalk, bus, cars, truck |
| scene-1077 | Night, big street, bus stop, high speed, construction vehicle |
| scene-1094 | Night, after rain, many peds, PMD, ped with bag, jaywalker |
| scene-1100 | Night, peds in sidewalk, peds cross crosswalk, scooter, PMD |

**mini_val** (evaluation only — not labelled by pipeline):

| Scene | Description |
|---|---|
| scene-0103 | Many peds right, wait for turning car, long bike rack left |
| scene-0916 | Parking lot, bicycle rack, parked bicycles, bus, many peds |

ECP also contains `CAM_FRONT2` (a stereo camera at the same position as `CAM_FRONT`), but this
channel is intentionally **excluded**.  Including it would produce duplicate detections in the
same FoV that the cross-camera merge (§7) would not catch, since the two images have the same
viewing direction and no object would be at an image border.

### Config

Multi-camera mode is enabled by including a `cameras:` list in the YAML config.  Removing or
commenting it out falls back to the single `camera:` field.

```yaml
camera: CAM_FRONT         # single-camera fallback / evaluation
cameras:                  # multi-camera mode; comment out to use single camera only
  - CAM_FRONT_LEFT
  - CAM_FRONT
  - CAM_FRONT_RIGHT
```

When `cameras:` has more than one entry the full pipeline automatically uses
`run_multi_camera_pipeline` and `build_submission_multi_cam`.  No CLI flag is needed.

### Pipeline behaviour

`run_multi_camera_pipeline` in `AutoLabeling/src/autolabeling/pipeline.py` loops
`run_pipeline` once per camera, using a per-camera checkpoint subdirectory
(`checkpoint_dir/<cam_name>/`) to avoid collisions.  Results are stored in per-camera dicts:
`body_results_all[cam]`, `obj_results_all[cam]`.

LiDAR is loaded once in ego frame inside each `run_pipeline` call and projected into that
camera independently using its `R_c2e`, `t_c2e` calibration.  OBBs for objects are computed
in ego space and are directly comparable across cameras.  Body OBBs are in camera space and
are transformed to ego via `R_c2e @ corners.T + t_c2e` whenever ego-space comparison is
needed (rider merge, cross-camera merge).

After all per-camera pipelines finish, `cross_camera_merge` is called if
`cross_camera_merge.enabled: true` in the config (§7).

Single-camera mode (no `cameras:` field, or `cameras:` contains only one entry) runs only
the single `camera:` value and skips the cross-camera merge.

### Code

- Notebook: `Testing/autolabeling_pipeline.ipynb` — config cell (Cell 4), multi-cam config
  cell (Cell 5), all inference cells loop over `CAMERAS`
- Full pipeline entry point: `AutoLabeling/run_pipeline.py` — reads `cameras:` from YAML,
  computes `_multi_cam = bool(_cameras_list and len(_cameras_list) > 1)`, routes to
  `run_multi_camera_pipeline` or `run_pipeline`
- Pipeline: `AutoLabeling/src/autolabeling/pipeline.py` — `run_multi_camera_pipeline()`,
  `run_pipeline()`
- Configs: `AutoLabeling/configs/ecp.yaml`, `AutoLabeling/configs/nuscenes.yaml`

---

## 7. Cross-Camera Duplicate Suppression

### Problem

An object near the seam between two adjacent cameras is often detected independently in both
views — once in the left camera (mask at the right image edge) and once in the right camera
(mask at the left image edge).  Without suppression this yields two separate OBBs for the
same physical object.

### Why not VESPA's merging strategy

VESPA (`/home/lleba/VESPA/src/image/object_merge.py`) implements two merging strategies:

1. **Camera-mask + LiDAR cluster match**: checks border pixel positions (same idea as our
   Stage 1), then computes minimum Euclidean distance between the raw **LiDAR point clusters**
   of the two candidates, or counts exact-coordinate overlapping points (distance < 1e-6 m)
   between clusters.
2. **VLM-to-clustering merge**: fuses camera-based VLM detections with a separate
   LiDAR-clustering detection pathway, using LiDAR point-ratio thresholds between both sets.

Both strategies fundamentally depend on **per-object LiDAR point clusters**.  VESPA retains the
set of raw LiDAR returns that were assigned to each detection throughout its pipeline — these
are available because VESPA runs LiDAR spatial clustering as a parallel detection branch.

Our pipeline does not produce per-object LiDAR clusters.  LiDAR is used only for depth
calibration inside each camera's `run_pipeline` call (affine scale fit for O3, dense
completion for O4/O5); the outputs are OBBs and meshes, not point clouds.  At merge time
there is no LiDAR cluster to compare.

Additionally, VESPA's VLM-to-clustering strategy is not applicable here because we have no
separate LiDAR clustering branch — all detections come from SAM3 + SAM3D.

**Our alternative** is therefore built entirely on what we do have: the fitted OBBs (for BEV
geometric overlap) and the original camera crops (for appearance embedding fallback).  BEV OBB
overlap is arguably a stronger signal than raw LiDAR cluster distance anyway — it operates in
the same evaluation space used by nuScenes mAP and is not affected by LiDAR sparsity.

### Stage 0 — OBB post-filters (ego-body exclusion + volume)

Before any cross-camera matching, two filters are applied to every per-camera detection.
Both run at the end of `run_pipeline`, after OBB fitting and rider merge, in this order:

#### 0a — Ego-body exclusion

Catches ego-vehicle parts (hood, bumper, trunk) that SAM3 segments as a real object —
most commonly the front hood appearing as `"car"` in `CAM_FRONT`.

For each camera, the filter checks whether an OBB center lies within `D` metres of the
camera along its **optical axis** in ego frame:

```
fwd_ego   = R_c2e @ [0, 0, 1]          # camera forward direction in ego space
d_along   = dot(obb_center_ego − t_c2e, fwd_ego)
excluded  = 0.0 ≤ d_along ≤ D
```

`D` is read per camera name from `obb_filter.ego_obb_depth` in the YAML config; a fallback
`ego_obb_depth_default` is used for any unlisted camera name.  Body OBB centers (camera
space) are first transformed to ego space via `R_c2e @ center + t_c2e`.

**Why camera-extrinsic-based, not a hardcoded box**: the exclusion zone origin is placed
exactly where the camera sits on the vehicle (from `calibrated_sensor` metadata) and extends
in the camera's actual viewing direction.  This is dataset-agnostic — any vehicle whose
cameras are described by nuScenes-format calibration automatically gets the correct zone
geometry.  Only the depth `D` (how far ahead the camera can see ego bodywork) is a tuned
constant; ego vehicle dimensions themselves are not stored in nuScenes metadata.

**Limitation**: `D` is still a fixed approximation per camera position.  A fully data-driven
approach would derive it from the LiDAR point cloud (ego-body returns define the vehicle
boundary), but this adds complexity without substantially improving accuracy given the
narrow physical range of passenger car dimensions.

**Per-camera depth values (current)**

| Camera | D (m) | Rationale |
|---|---|---|
| `CAM_FRONT` | 1.0 | Hood fully visible |
| `CAM_BACK` | 1.0 | Trunk visible |
| `CAM_FRONT_LEFT/RIGHT` | 1.0 | Diagonal — hood corner |
| `CAM_BACK_LEFT/RIGHT` | 1.0 | Diagonal — trunk corner |
| `CAM_LEFT/RIGHT` | 0.5 | Side cameras — nearly at car edge |
| *(default)* | 1.0 | Fallback for unlisted cameras |

#### 0b — Volume filter (min and max)

`OBB volume = L × W × H` (ego space for objects; camera space for bodies) must fall within
a per-class `[min_volume, max_volume]` range.

- **Min** catches collapsed/micro-sized boxes from bad depth estimates or very thin masks.
- **Max** catches inflated/ballooned boxes from close objects with poorly constrained mesh
  reconstruction (e.g. a nearby car that MoGe reconstructs as an enormous slab).

Per-axis dimension checks (`min_l`, `min_h`) are not used in the full pipeline — the volume
range is stricter and simpler.

**Per-class volumes (current values)**

| Class | Min (m³) | Max (m³) |
|---|---|---|
| `pedestrian` | 0.1 | 8.0 |
| `bicycle` | 1.5 | 15.0 |
| `motorcycle` | 1.5 | 15.0 |
| `car` | 10.0 | 70.0 |
| `truck` | 15.0 | 300.0 |
| `bus` | 15.0 | 300.0 |
| `trailer` | 15.0 | 300.0 |
| `construction vehicle` | 15.0 | 300.0 |

### Stage 1 — Border check

For each adjacent camera pair `(left_cam, right_cam)`, border-touching candidates are
collected:

- **Left camera**: detection mask must reach the right image edge —
  `rightmost_pixel_x / W ≥ 1 − border_threshold` (default `0.15`).
- **Right camera**: detection mask must touch the left image edge —
  `leftmost_pixel_x / W ≤ border_threshold`.

Both cameras are checked simultaneously per pair — the seam is fully covered in one pass.
Detections without a `binary_mask` (e.g. ViTDet-only pedestrian body results, which have
`binary_mask=None` because ViTDet provides no segmentation) are silently skipped; they
cannot participate in the merge.  SAM3-sourced body results carry a valid `binary_mask`
(populated via `sam3_mask_idx` during SAM3D Body inference) and do participate.

### Stage 2 — BEV overlap matching (main pass)

For all border-touching candidate pairs that pass a class-compatibility check
(`[car, truck]`, `[motorcycle, bicycle]`, `[truck, trailer, bus]`, or identical labels),
BEV OBB footprint overlap is computed and pairs are assigned greedily by descending overlap.

**BEV overlap**: `overlap_fraction = intersection_area / area(smaller OBB)` in the
top-down (XY) bird's-eye-view projection.  Implemented via `shapely.Polygon.intersection`
on the convex hull of the 8 OBB corners projected to the XY plane.  Body OBB corners are
first transformed from camera space to ego space (`R_c2e @ corners.T + t_c2e`) before
the BEV projection.

**Main pass**: pairs are sorted by descending BEV overlap.  A pair is accepted when
`overlap_fraction ≥ bev_overlap_thresh` (default `0.10` = 10 %).  Each detection can
only appear in one accepted pair (1-to-1 assignment).

**Why BEV instead of 3D volume overlap**: height estimation from SAM3D is the least
reliable dimension (depth ambiguity is mostly vertical).  BEV overlap is also the standard
evaluation metric in autonomous driving.

### Stage 3 — Appearance embedding fallback

After the main pass, any border-touching candidate that is still unmatched is paired with
the best remaining candidate by **ResNet18 cosine similarity** — no threshold applied.
A ResNet18 (truncated at global pool, eval mode) embeds each candidate's image crop; the
most similar unmatched partner (greedy, descending similarity) is accepted.

This fallback handles cases where BEV overlap is near-zero due to poor depth estimation but
the two crops are visually identical (same object, two camera angles).  It also ensures
every border detection gets a partner if one is available, rather than leaving a duplicate
alive.  The fallback is skipped if there are no remaining unmatched border candidates.

### Decision: which detection to keep

The detection whose **SAM3 mask covers more pixels** (`binary_mask.sum()`) is kept.
A larger mask means SAM3 captured more of the object surface and passed more image
information downstream to SAM3D — so that camera view yields better 3D reconstruction.
The smaller-mask detection is suppressed (removed from `body_results_all` / `obj_results_all`).

### Adjacent camera pairs (ordered left → right)

**nuScenes** (6 pairs, full circle):
`CAM_FRONT_LEFT↔CAM_FRONT`, `CAM_FRONT↔CAM_FRONT_RIGHT`, `CAM_FRONT_RIGHT↔CAM_BACK_RIGHT`,
`CAM_BACK_RIGHT↔CAM_BACK`, `CAM_BACK↔CAM_BACK_LEFT`, `CAM_BACK_LEFT↔CAM_FRONT_LEFT`

**ECP** (2 pairs, frontal arc):
`CAM_FRONT_LEFT↔CAM_FRONT`, `CAM_FRONT↔CAM_FRONT_RIGHT`

### Config (full pipeline YAML)

```yaml
obb_filter:
  ego_obb_depth:
    CAM_FRONT:       1.0
    CAM_BACK:        1.0
    CAM_FRONT_LEFT:  1.0
    CAM_FRONT_RIGHT: 1.0
    CAM_BACK_LEFT:   1.0
    CAM_BACK_RIGHT:  1.0
    CAM_LEFT:        0.5
    CAM_RIGHT:       0.5
  ego_obb_depth_default: 1.0
  min_volume:
    pedestrian:           0.1
    bicycle:              1.5
    motorcycle:           1.5
    car:                 10.0
    truck:               15.0
    bus:                 15.0
    trailer:             15.0
    construction vehicle: 15.0
  max_volume:
    pedestrian:            8.0
    bicycle:              15.0
    motorcycle:           15.0
    car:                  70.0
    truck:               300.0
    bus:                 300.0
    trailer:             300.0
    construction vehicle: 300.0

cross_camera_merge:
  enabled: true
  border_threshold: 0.15
  bev_overlap_thresh: 0.10
  compatible_class_groups:
    - [car, truck]
    - [motorcycle, bicycle]
    - [truck, trailer, bus]
  adjacent_cam_pairs:               # nuScenes example
    - [CAM_FRONT_LEFT,  CAM_FRONT]
    - [CAM_FRONT,       CAM_FRONT_RIGHT]
    - [CAM_FRONT_RIGHT, CAM_BACK_RIGHT]
    - [CAM_BACK_RIGHT,  CAM_BACK]
    - [CAM_BACK,        CAM_BACK_LEFT]
    - [CAM_BACK_LEFT,   CAM_FRONT_LEFT]
```

### Code

- Notebook: `Testing/autolabeling_pipeline.ipynb` — Cell 4 (config: `EGO_OBB_DEPTH`, `OBB_MIN/MAX_VOLUME`),
  Cell 29 (`obb_in_ego_excl()` + combined filter loop + BEV visualisation with exclusion wedges)
- Full pipeline ego-excl filter: `_apply_ego_obb_excl_filter()` in `AutoLabeling/src/autolabeling/pipeline.py`
  — runs first, before volume filter, at end of `run_pipeline()`
- Full pipeline volume filter: `_apply_obb_filter()` in `AutoLabeling/src/autolabeling/pipeline.py`
  — accepts both `min_volume` and `max_volume` dicts; called immediately after ego-excl filter
- Full pipeline merge: `AutoLabeling/src/autolabeling/cross_camera_merge.py` —
  `cross_camera_merge()`, `_bev_overlap()`, `_embed_model()`, `_get_embed()`
- Called from: `run_multi_camera_pipeline()` in `AutoLabeling/src/autolabeling/pipeline.py`

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
