# In-Mask LiDAR Point Filtering

How the pipeline goes from raw aggregated LiDAR sweeps to a clean set of
object-surface points that feed SAM3D Objects (O5) and SAM3D Body (B1).

---

## 1. LiDAR Loading and Aggregation

Before any per-mask logic, the full scene point cloud is built.

**Aggregation** (`load_lidar_pts_aggregated`): N_before + 1 anchor + N_after
sweeps are combined in the anchor ego frame via ego-motion compensation.
Each sweep goes through: LiDAR sensor frame → sweep ego frame → global frame
→ anchor ego frame.

nuScenes default: 3 + 1 + 3 = 7 sweeps  
ECP default: 2 + 1 + 2 = 5 sweeps

---

## 2. Per-Sweep Ego-Body Filter

**Applied per sweep, in that sweep's own ego frame, before the global
transform.**

Why per-sweep and not after aggregation: at 40 km/h the vehicle travels
≈ 3.7 m in 333 ms. If the ego-body box is applied after transformation into
the anchor ego frame, non-anchor sweeps have already moved, so their
ego-body returns (roof rack, windshield) escape the box by up to several
metres and contaminate the aggregated cloud.

Applying the filter while the vehicle is still at the origin (per-sweep ego
frame) guarantees the box always catches those returns.

**What is removed:** points inside the box AND within the Z band.

```
|x| < ego_box_half_x  AND  |y| < ego_box_half_y
AND  ego_box_z_min < z < ego_box_z_max
```

The Z band is intentional: it preserves close ground-level returns (z < 0.5 m)
and very tall objects above (z > 2.5 m). Only the vehicle-body height band
is removed.

**Parameters (both datasets):**

| Param | Value | Meaning |
|---|---|---|
| `ego_box_half_x` | 4.0 m | half-length fore/aft |
| `ego_box_half_y` | 1.5 m | half-width left/right |
| `ego_box_z_min` | 0.5 m | lower Z cutoff |
| `ego_box_z_max` | 2.5 m | upper Z cutoff |

---

## 3. Max-Range Filter

After aggregation, points beyond a BEV range threshold are dropped.

```
sqrt(x² + y²) > max_range_m  →  remove
```

**Parameter:** `max_range_m = 52.0 m` (2 m above the 50 m evaluation limit).

Distant background points form spurious HDBSCAN clusters inside far-projected
masks and are not useful for depth estimation.

---

## 4. Camera Projection

The surviving ego-frame cloud is projected into camera space:

1. Ego → camera via extrinsics (rotation + translation).
2. Keep only points in front of the camera (Z_cam > 0).
3. Project to pixel coords via intrinsics K; keep only points whose pixel
   falls inside the image.
4. When multiple points project to the same pixel, keep the nearest (min-depth).

Result: `u_vis, v_vis, Z_vis, pts_ego_vis` — only LiDAR returns visible in
this camera.

---

## 5. Per-Mask HDBSCAN Filtering (O5 sparse map)

**Used in:** `o5_mask_hdbscan` pointmap mode, before CompletionFormer.

**Goal:** for each SAM3 segmentation mask, keep only the points belonging to
the object's surface. Discard ground returns, background walls, and occluder
bleeds that happen to project inside the mask.

### 5.1 Mask Erosion (pre-HDBSCAN)

Before selecting in-mask points, the binary mask is morphologically eroded
by `mask_erode_px` pixels using an elliptical kernel. This removes boundary
pixels where mask inaccuracy lets adjacent occluders (signs, poles, parked
vehicles) bleed in at the mask edge.

Erosion is applied equally to all connected components of the mask — small
occluder blobs at the boundary may disappear entirely, which is the desired
outcome.

Erosion is skipped when `binary_mask.sum() < mask_erode_min_px`.

**Parameters:**

| Param | Value |
|---|---|
| `mask_erode_px` | 3 px |
| `mask_erode_min_px` | 0 (always erode) |

### 5.2 HDBSCAN Clustering

Points that project inside the (eroded) mask are extracted and clustered in
**3D ego-frame coordinates** (metres). Operating in metric space makes the
distance threshold physically meaningful and range-invariant — a 0.5 m gap
between a person and a background wall is the same gap regardless of how far
away they are.

```python
HDBSCAN(
    min_cluster_size = mcs,
    min_samples      = ms,
    metric           = 'euclidean',
    cluster_selection_epsilon = eps,
)
```

`cluster_selection_epsilon` merges sub-clusters whose centroids are within
`eps` metres. This is critical for large vehicles (car, bus, truck) whose
LiDAR returns are split into front-face, wheel-arch, and roof subclusters
that should be treated as one object.

**Per-class parameters (nuScenes / ECP):**

| Class | min_cluster_size | min_samples | cluster_eps |
|---|---|---|---|
| pedestrian | 3 | 1 | 0.20 m |
| bicycle | 3 | 1 | 0.20 m |
| motorcycle | 3 | 1 | 0.30 m |
| car | 3 | 1 | 0.50 m |
| truck | 5 | 1 | 0.60 m |
| construction vehicle | 5 | 1 | 0.60 m |
| bus | 5 | 1 | 0.80 m |
| trailer | 5 | 1 | 0.70 m |

### 5.3 Cluster Selection

After clustering, one dominant cluster is selected as the object surface.

**Default:** pick the largest cluster.

**Proximity selection** (override): if the two largest clusters are similar
in size — biggest ≥ `proximity_min_pts` AND second ≥ `proximity_ratio ×
biggest` — pick the **closer** one by ego-frame centroid distance to the
vehicle origin.

Why: when a sign or pole stands partially in front of a car, both can
accumulate roughly equal LiDAR hits inside the mask. The sign is closer. If
we always pick the largest, we risk anchoring to the sign. Proximity
selection picks the foreground cluster (the sign, being closer) which has
the correct minimum depth for the object region.

**Parameters:**

| Param | Value | Meaning |
|---|---|---|
| `proximity_min_pts` | 30 | biggest cluster must be ≥ this to trigger |
| `proximity_ratio` | 0.70 | second must be ≥ 70 % of biggest to trigger |

### 5.4 Failure Handling

| Condition | Action |
|---|---|
| `n_in_mask < min_cluster_size` | Remove all in-mask pts → CFormer uses image priors |
| HDBSCAN: all noise | Remove all in-mask pts → CFormer uses image priors |
| Success | Keep dominant cluster pts; remove everything else in mask |

The "remove on failure" behavior is deliberate: a failed HDBSCAN means the
in-mask returns are likely scattered noise or an occluder that cannot be
meaningfully isolated. Keeping them would anchor CompletionFormer to wrong
depths. Removing them lets CFormer infer depth from surrounding out-of-mask
context (road, buildings) and image appearance — which is often more
accurate than a contaminated anchor.

### 5.5 Output

```
clean_sparse = (HDBSCAN-filtered in-mask pts) ∪ (all out-of-mask pts)
```

Out-of-mask points (road surface, buildings, background) are always kept
unchanged — they provide valid background depth context for CompletionFormer.

---

## 6. CompletionFormer (CFormer)

The clean sparse depth map from step 5 is fed into CompletionFormer to
produce a dense metric depth image `dense_depth` (H × W, float32, metres).

CFormer operates entirely in **camera-frame depth** (Z_cam = depth along the
optical axis). The sparse input and dense output are both in this space.

---

## 7. Hull Anchoring (post-CFormer depth clamp)

**Used in:** O4 and O5 modes, after CFormer produces `dense_depth`.

**Goal:** prevent occluder LiDAR depth from bleeding into object interiors
through CFormer interpolation. Even after O5 HDBSCAN cleaning, CFormer can
still interpolate a nearby occluder's depth across the object's convex hull
region if the occluder sits adjacent to the mask boundary.

### 7.1 Per-Mask HDBSCAN (with erosion)

The same mask erosion (step 5.1) and HDBSCAN logic (steps 5.2–5.4) is run
again on the full (post-aggregation) in-mask LiDAR, independently of the
O5 sparse map step. This gives a fresh `dominant cluster` for each mask.

The erosion here is important for the same reason: sign/pole points at the
mask boundary would otherwise make `min_depth` too shallow (the sign depth
rather than the car depth).

Failure cases (too-few pts, all-noise) → skip hull anchoring for this mask
(let CFormer result stand unchanged).

### 7.2 min_depth

```python
min_depth = float(z_dominant.min())
```

The minimum camera-frame depth of all points in the dominant cluster. This
is the nearest point of the object's LiDAR surface as seen from the camera.

### 7.3 Convex Hull Fill

The SAM3 binary mask pixels are used to build a 2D convex hull
(`cv2.convexHull`). The hull is rasterised to a boolean image. This is
larger than the mask itself for non-convex objects but always fully encloses
the mask.

The full (non-eroded) mask is used for the convex hull — the depth clamp
should cover the complete object area, not just the eroded interior.

### 7.4 Depth Clamp

```python
dense_depth[hull] = max(dense_depth[hull], min_depth)
```

Inside the hull, any pixel whose CFormer depth is shallower than `min_depth`
is pushed back to `min_depth`. Pixels already deeper than `min_depth` are
unchanged.

This is a one-sided clamp (floor, not ceiling): it prevents CFormer from
placing object pixels in front of the nearest known LiDAR surface point, but
does not compress the depth range of pixels that are legitimately deeper
(e.g. the far side of a large vehicle).

**Hull anchoring uses the same proximity and erosion params as O5.**

| Param | Value |
|---|---|
| `hull_anchoring` | true |
| `mask_erode_px` | 3 px |
| `mask_erode_min_px` | 0 |
| `proximity_min_pts` | 30 |
| `proximity_ratio` | 0.70 |

---

## 8. SAM3D Body B1 — HDBSCAN for Depth Correction

B1 uses HDBSCAN differently from O5: instead of building a sparse depth map,
it uses the dominant cluster's **median depth** to override the SAM3D Body
predicted translation `tz`.

### Flow

1. For each pedestrian detection, select its mask (SAM3 binary mask if
   available and IoU > 0.3 with ViTDet bbox; otherwise a rectangular bbox mask).
2. Extract in-mask LiDAR points from the aggregated cloud.
3. Run the same HDBSCAN + proximity selection (no mask erosion applied).
4. `tz_lidar = median(Z_vis[dominant_cluster])`.
5. If HDBSCAN fails → fallback: `tz_lidar = 15th-percentile(Z_anchor_inmask)`
   using the anchor-sweep-only cloud to avoid moving-pedestrian displacement bias.
6. Recompute `tx, ty` from mask centroid: `tx = (u_cen - cx)/fx * tz_lidar`.

### HDBSCAN params for B1

Uses the `pedestrian` entry from the HDBSCAN config:

| Param | Value |
|---|---|
| `min_cluster_size` | 3 |
| `min_samples` | 1 |
| `cluster_eps` | 0.20 m |
| `proximity_min_pts` | 30 |
| `proximity_ratio` | 0.70 |

No mask erosion is applied in B1 (pedestrian masks are tight, and there is
no equivalent CFormer interpolation problem to guard against).

---

## 9. Summary: What Each Mode Uses

| Step | O1 | O5 | B1 |
|---|---|---|---|
| Per-sweep ego-body filter | yes | yes | yes |
| Max-range filter | yes | yes | yes |
| Camera projection | yes | yes | yes |
| Mask erosion | — | yes | — |
| Per-mask HDBSCAN (O5 sparse) | — | yes | — |
| Proximity selection | — | yes | yes |
| Failure → remove pts | — | yes | — (fallback p15) |
| CompletionFormer | — | yes | — |
| Hull anchoring HDBSCAN | — | yes | — |
| Hull depth clamp | — | yes | — |
| HDBSCAN median → tz override | — | — | yes |

---

## 10. Key Rationale Summary

| Design choice | Why |
|---|---|
| Cluster in ego frame, not image | Metric distances; gap between person and wall is the same at any range |
| Per-sweep ego-body filter | Vehicle moves ~3.7 m in 333 ms; post-aggregation filter misses non-anchor sweeps |
| cluster_eps per class | Large vehicles need wider merging; pedestrians must not merge with adjacent people |
| Proximity selection (closer wins) | Occluder in front of object accumulates similar hit count; picking closer = foreground |
| Failure → remove (O5) | Scattered noise anchors CFormer worse than no anchor at all |
| Failure → p15 fallback (B1) | No CFormer involved; a rough depth is still better than the uncorrected prediction |
| Mask erosion before HDBSCAN | Boundary pixels leak occluder returns; 3 px strip removes them without shrinking much |
| Hull uses full mask | Depth clamp must cover the entire object, not just the eroded interior |
| min_depth = cluster min, not percentile | Physically meaningful: nearest surface point; no arbitrary margin needed |
| One-sided clamp (floor only) | Preserves depth variation inside object; only prevents occluder bleed-through |
