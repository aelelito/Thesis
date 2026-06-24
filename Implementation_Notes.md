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
- Notebook:     `merge_rider_obbs()` in `AutoLabeling/autolabeling_pipeline.ipynb`, cell 25

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
lidar:
  pointmap_mode: baseline   # 'baseline' | 'o1_lidar' | 'o2_moge_affine' | 'o3_local_affine'
```

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

**Post-filtering**: Only the **largest cluster** is kept.

| Parameter | Value | Rationale |
|---|---|---|
| `min_cluster_size` | 3 | Low enough for far/sparse objects |
| `min_samples` | 1 | Least aggressive noise rejection |
| `cluster_selection_epsilon` | 0.5 m | Merges sub-clusters on the same surface |

Validated on ECP→nuScenes scene 10, frame 570 (CAM_FRONT).
Tuning notebook: `AutoLabeling/lidar_filtering_exploration.ipynb`.

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

**Code**
- Full pipeline: `SAM3DObjectsModel._compute_moge_local_affine_pointmap_for_object()`
  *(to be ported from notebook after testing)*
- Notebook: `compute_local_affine_ptmap_for_object()` in `autolabeling_pipeline.ipynb`, cell 19.

---

### O4 — Dense Depth Completion  *(planned)*

Use a depth completion network (e.g. CompletionFormer, CVPR 2023) to produce a dense
metric depth map from sparse LiDAR + RGB image.  The completed map replaces MoGe entirely,
giving both metric scale and dense coverage without the affine approximation.

*To be documented when implemented.*

---

### O5 — Mid-Pipeline Voxel Correction  *(planned)*

Insert a LiDAR-based voxel correction step between the sparse-structure and SLAT stages
of SAM3D Objects' diffusion pipeline.  Anchors the generated 3-D structure to metric LiDAR
geometry before the final mesh is decoded.

*To be documented when implemented.*

---

## 4. SAM3D Body — LiDAR Options  *(planned)*

Options for correcting pedestrian translation (tz) and camera encoder inputs using LiDAR
depth.  To be documented when implemented.
