# LiDAR Integration Plan for SAM3D Autolabeling Pipeline

> Written: June 2026  
> Purpose: Systematic plan for integrating LiDAR depth information into the SAM3D Body and SAM3D Objects autolabeling pipeline.  
> The core motivation: SAM3D models were trained on internet imagery without metric depth. LiDAR provides accurate, metric, sensor-measured 3D geometry that can ground the models in real-world scale, localization, shape, and orientation.

---

## 0. Both models are open source and modifiable

Both SAM3D Body (`SAM3D/sam-3d-body/`) and SAM3D Objects (`SAM3D/sam-3d-objects/`) are locally cloned Meta repos. The SAM License (section 1a) explicitly grants rights to **"create derivative works of and make modifications to the SAM Materials"**, and section 5a gives ownership of those modifications. Full Python source is available including all model internals.

---

## 1. Baseline Pipeline: SAM3D Objects (current, no LiDAR)

### What it does
Given a 2D image crop of an object (e.g. a car, cyclist, traffic cone), it generates a full 3D mesh of that object. It works purely from appearance — the model has never seen metric LiDAR in its inference loop.

### Step-by-step

```
Input: RGBA image crop (RGB + alpha segmentation mask from SAM3)
       The alpha channel = background removed, only the object is visible
```

**Step 1 — Preprocessing** (`preprocess_utils.py`, `sam3d_objects/data/`)
- Crop tightly around the SAM3 mask
- Remove background (alpha mask baked in)
- Pad to square, resize to 518×518
- Normalize pixel values

**Step 2 — Condition embedding** (`condition_embedders["ss_condition_embedder"]`)
- The preprocessed RGBA image passes through a vision encoder (DINOv3-style ViT)
- Output: a sequence of image feature tokens, each token representing a patch of the image
- These tokens are what the 3D generator will "look at" when deciding the shape

**Step 3 — Sparse Structure (SS) generation** (`models["ss_generator"]` → `SparseStructureFlowTdfyWrapper`)
- This is the most important step. It runs a **flow-based diffusion model** for ~25 reverse steps
- The model generates a **3D binary voxel grid**: imagine a 3D cube of cells, each cell is either ON (occupied) or OFF (empty)
- At each diffusion step, cross-attention blocks attend to the image tokens from Step 2
- Output: a coarse 3D skeleton/silhouette of the object — correct overall shape, bounding extent, and structure
- This is what determines width, height, depth, and whether the object looks like a boxy van vs. a rounded car

**Step 4 — SLAT (Structured Latent) generation** (`models["slat_generator"]`)
- Takes the SS voxel grid + image tokens as input
- Runs another diffusion model that fills in detailed latent features for each occupied voxel
- These latents encode fine geometry, surface normals, and appearance

**Step 5 — Decoding**
- `slat_decoder_mesh`: converts SLAT latents → triangle mesh (vertices + faces)
- `slat_decoder_gs`: converts SLAT latents → 3D Gaussian splats (for rendering)

### What the model does NOT know (without LiDAR)
- **Metric scale**: the output mesh has correct shape but arbitrary/relative scale. The model doesn't know if it's a 1m object or a 4m object.
- **Absolute 3D position**: the mesh is generated in object-centric normalized space, not placed in the scene.
- **True depth from camera**: the model infers scale purely from image appearance (a large close object vs. a small distant one look similar in a crop).
- **Precise surface geometry**: where the image is ambiguous (far side of a car), the model hallucinates based on learned priors.

### With the pointmap pipeline (already available — `InferencePipelinePointMap`)
Between Steps 1 and 3, the pipeline also runs:
- A **depth model** on the image to estimate per-pixel depth → produces a pointmap tensor of shape (3, H, W) = (X, Y, Z) per pixel
- This pointmap goes through `PointPatchEmbed` → depth feature tokens
- The SS generator's conditioning in Step 3 gets BOTH image tokens AND depth tokens
- Now the model knows the metric scale of the scene and generates a correctly-scaled voxel grid

The depth model in the base pointmap pipeline is a **monocular depth estimator** — it guesses depth from appearance alone. This is better than nothing but still not accurate. **The key thesis contribution is replacing or augmenting this with real LiDAR data.**

---

## 2. Baseline Pipeline: SAM3D Body (current, no LiDAR)

### What it does
Given a full image containing people, it detects each person and estimates their complete 3D body mesh — the full SMPL body model with 70 joints and ~6890 vertices, placed in metric 3D camera space.

### Step-by-step

```
Input: Full RGB image (e.g. 1600×900 nuScenes camera frame)
       Camera intrinsics matrix K (from nuScenes calibration — already provided)
       Person bounding boxes (from SAM3 masks + ViTDet supplementary detection)
```

**Step 1 — Per-person crop and alignment** (`prepare_batch`, `TopdownAffine`)
- For each detected person bounding box, crop the image
- Apply affine transform to normalize the crop to a fixed input size (e.g. 192×256)
- The crop center and scale are recorded (needed later for deprojection)

**Step 2 — Backbone feature extraction** (`create_backbone` → DINOv3 ViT)
- Patch-based ViT processes the person crop
- Output: 2D grid of patch embeddings (B × H × W × C)
- These are appearance features — texture, pose silhouette, clothing

**Step 3 — Camera ray encoding** (`CameraEncoder`, `camera_embed.py`)
- For each pixel in the crop, compute its **camera ray direction**: the 3D direction vector from the camera center through that pixel into the world
- These directions are 2D (azimuth + elevation) — the z-component is a constant 1 (dummy depth)
- Ray directions are Fourier-encoded into high-dimensional embeddings (99 dimensions)
- Mixed into the backbone features via a 1×1 conv: `embed_dim + 99 → embed_dim`
- **Purpose**: the backbone features now know which direction each patch is looking, not just what color/texture it has. This helps the model reason about 3D geometry from 2D features.
- **Key limitation**: direction only, no metric depth. The model knows the ray points toward the person but not how far along that ray the person actually is.

**Step 4 — Decoder with CLIFF conditioning** (`forward_decoder`)
- Creates initial pose + camera tokens (learnable embeddings, start at zero/neutral pose)
- CLIFF condition info is computed: `(cx - W/2)/f`, `(cy - H/2)/f`, `bbox_size/f`  
  These three numbers encode "how far off-center is this person crop, and how large relative to focal length?" This is crucial for correct perspective deprojection.
- The **promptable transformer decoder** (cross-attention to backbone features) iteratively refines the pose token over multiple passes
- Optional: keypoint prompts can be injected here

**Step 5 — Head predictions**

*Pose head* (`head_pose`):
- Outputs SMPL body parameters: joint rotation angles for all 70 body joints
- These define how the body is posed (arms up/down, legs bent, etc.)

*Camera head* (`PerspectiveHead`, `camera_head.py`):
- Outputs 3 camera parameters: `(s, tx, ty)`
  - `s` = scale — how large the person appears relative to the image (predicted from appearance)
  - `tx`, `ty` = lateral offsets (left/right and up/down in image space)
- Then computes the full 3D translation:
  ```
  tz = 2 * focal_length / (bbox_size * s)   ← depth, derived from predicted scale
  tx_3d = tx + (bbox_center_x - W/2) / s
  ty_3d = ty + (bbox_center_y - H/2) / s
  pred_cam_t = [tx_3d, ty_3d, tz]
  ```
- `pred_cam_t` is the 3D position of the person's body root (pelvis) in camera space
- **This is where LiDAR helps most**: `tz` is derived purely from `s` (predicted scale), which is guessed from appearance. A small distant person and a large close person can look identical in a crop — the model cannot tell them apart reliably.

**Step 6 — SMPL body model**
- Takes joint rotations + shape parameters → outputs 3D mesh vertices and 3D joint positions
- The mesh is in root-relative space initially, then shifted by `pred_cam_t` to place it in camera space

### Output
- `pred_vertices`: (N_persons, 6890, 3) — 3D body mesh vertices in camera coordinates
- `pred_keypoints_3d`: (N_persons, 70, 3) — 3D joint positions
- `pred_cam_t`: (N_persons, 3) — [tx, ty, tz] body root position in camera space

### What the model does NOT know (without LiDAR)
- **Accurate depth (tz)**: the biggest source of error. Persons at 5m and 15m look similar in a small crop.
- **Metric scale verification**: body dimensions are guessed from appearance priors.
- **True 3D position**: the whole body mesh placement depends on `tz` being correct.

---

## 3. Integration Options

### Prerequisite for all options: in-mask LiDAR point filtering

Before any LiDAR is passed to either model, you need to select and clean the LiDAR points that are relevant to the object. This means:

1. **Project** all LiDAR points from the LiDAR frame into the camera image using the calibrated extrinsic + intrinsic matrices
2. **Select** only points whose projected pixel coordinates fall within the SAM3 segmentation mask — these are the in-mask LiDAR points belonging to this object
3. **Filter** the selected points to remove outliers: background bleeds, ground returns, noise

Step 3 was previously done with a self-developed depth-window + MAD filter. The professor's direction is to use **established, published filtering techniques** instead. The best candidates for this use case:

- **Statistical Outlier Removal (SOR)**: for each point, compute the mean distance to its K nearest neighbors; remove points whose mean distance exceeds the global mean by N standard deviations. Standard in PCL, well-cited in LiDAR processing literature.
- **Radius Outlier Removal (ROR)**: remove points with fewer than N neighbors within radius R. Simpler and more interpretable than SOR.
- **DBSCAN clustering**: cluster all in-mask points; keep only the largest cluster. Naturally rejects isolated outliers and stray hits from adjacent objects that bleed through the mask boundary. Clean conceptual justification: the dominant cluster IS the object surface.

**Recommendation**: DBSCAN is the most principled for this case — it makes no assumptions about the distance distribution and directly targets the problem (stray points from other objects or ground). SOR is also fine and more commonly cited in depth estimation papers. Either can be justified by citing existing work.

**This filtering step is implemented once and reused for all options below.**

---

### Relationship to previous work (`sam3d_objects_LiDAR.ipynb`)

The previous notebook implemented: **MoGe (monocular depth estimator) + in-mask LiDAR global affine calibration** → fused metric pointmap → SAM3D Objects. This is now formalized as **O2** — the simplest dense baseline, already implemented, slotting naturally between sparse LiDAR (O1) and the per-object variant (O3).

The five options cover a progression of increasing integration depth:

| Option | Density | LiDAR role |
|---|---|---|
| O1 | Sparse | Geometry directly, only where LiDAR hits |
| O2 (MoGe + global affine) | Dense | Scene-wide affine correction — one `(a,b)` per frame |
| O3 (MoGe + local affine) | Dense | Per-object affine using HDBSCAN-filtered in-mask LiDAR |
| O4 (depth completion) | Dense | Learned per-pixel fusion — LiDAR anchors geometry locally |
| O5 (voxel correction) | Dense+voxel | Mid-pipeline structural correction of 3D occupancy |

**Why O2 is the simplest dense baseline**: the affine model `Z_metric = a·Z_moge + b` fitted on all in-image LiDAR is stable (many points, wide depth range) but averages over all objects and depth ranges in the scene. **O3** addresses this directly: by fitting a separate `(a,b)` per object using only its HDBSCAN-filtered in-mask LiDAR returns, the calibration adapts to each object's actual depth range — where MoGe's non-linear compression is most relevant. O4 replaces hand-crafted calibration entirely with a learned dense completion network.

This five-way ablation answers: "does a dense monocular prior help over sparse LiDAR? Does per-object calibration improve over scene-wide? Does learned per-pixel fusion beat hand-crafted calibration? Does mid-pipeline structural correction add further gains?" Each step is independently motivated.

---

### A note on LiDAR sparsity and temporal aggregation

Raw LiDAR is sparse. In a single nuScenes sweep, a car at 20m might have 30–100 LiDAR points on its surface. A pedestrian at 15m might have 5–20 points. This is enough for localization (depth) and rough scale, but not much for detailed surface geometry.

**Temporal LiDAR aggregation** (future extension): accumulate multiple sweeps over time. For static objects (parked cars, signs), 5–10 sweeps gives a very dense point cloud and dramatically improves shape recovery. For moving objects, you need to:
1. Use ego-motion compensation (nuScenes provides ego poses)
2. Use object tracking to identify which points belong to the same object across frames
3. For rigid dynamic objects (moving cars): align using the tracked 3D bounding box pose
4. For non-rigid objects (pedestrians): body shape aggregation is much harder and not straightforward

Every option below benefits from denser LiDAR — temporal aggregation is a natural future extension for all of them.

---

### SAM3D Objects Options

---

#### O1 — Direct sparse LiDAR pointmap (built-in pathway, zero code change)

**What it is**: The `InferencePipelinePointMap.compute_pointmap()` method already accepts a pre-computed pointmap tensor. If you pass a `pointmap` argument, it skips the internal depth model entirely and uses your tensor directly. You project your nuScenes LiDAR scan into camera space (each LiDAR point (X,Y,Z) in LiDAR frame → transform to camera frame → get XYZ per pixel), apply in-mask filtering, and hand the resulting sparse XYZ-per-pixel tensor in.

**What it improves**: The model now knows rough metric depth and scale for voxels that have LiDAR support. Better than pure monocular guessing.

**What it does NOT fix**: LiDAR is sparse. At object range, many 8×8 pixel windows in the crop will have zero LiDAR points, receiving the `invalid_xyz_token` ("I have no depth here"). The model still has to hallucinate a lot of the shape from image priors.

**Compared to O2**: O1 is simpler — no dense estimator at all. It establishes a clean lower bound on "what LiDAR alone gives us" and isolates the LiDAR contribution from any monocular depth prior.

**Code change**: None. Just add LiDAR projection + filtering logic in your pipeline wrapper.

**Temporal**: More sweeps → denser per-pixel coverage → fewer empty windows → better.

**Novelty**: Low. It's an intended use case. Value is as a clean ablation baseline.

---

#### O2 — MoGe + global affine calibration (dense monocular prior, LiDAR-calibrated)

**What it is**: Run MoGe on the full image to get a dense, per-pixel relative depth map. Then use the in-mask LiDAR returns to fit a global affine transform `Z_metric = a·Z_moge + b` via least-squares over all in-mask LiDAR points. Apply the fitted transform to convert MoGe's relative depth to metric depth for every pixel in the crop. The resulting dense metric pointmap replaces the sparse LiDAR input.

**Why this is better than O1**: O1 leaves most `PointPatchEmbed` windows empty (no LiDAR support). O2 gives every window a plausible metric depth — MoGe fills in the gaps between LiDAR returns, and the affine fit grounds the whole map in metric scale.

**Known limitations**: The affine model assumes a linear relationship between MoGe's relative depth and true metric depth. MoGe compresses depth non-linearly at range (a 2m depth span at 20m may map to 0.5m of relative depth), and a single `(a, b)` per object cannot fix spatially-varying errors within the mask. With very few in-mask LiDAR returns the fit is also unstable. These limitations directly motivate O3.

**Implementation status**: Already implemented in `SAM3D/sam3d_objects_LiDAR.ipynb`. Needs to be ported into the pipeline wrapper.

**Code change**: ~10–20 lines in the pipeline wrapper: run MoGe on the crop, project in-mask LiDAR, fit `(a, b)` with `np.linalg.lstsq`, apply transform, pass as pointmap.

**Temporal**: More LiDAR returns → more stable affine fit, especially for objects with few in-image returns per frame.

**Novelty**: Low — already implemented as prior work. Value is as the simplest dense baseline and as the first step toward dense metric depth.

---

#### O3 — MoGe + per-object local affine calibration (novel contribution)

**What it is**: A per-object variant of O2. Instead of fitting one `(a, b)` per frame using all in-image LiDAR, fit a separate `(a, b)` for each detected object using only its HDBSCAN-filtered in-mask LiDAR surface points. MoGe runs once per frame; the affine calibration and full-frame pointmap rebuild happen independently per object before each SAM3D Objects inference call.

**Why per-object calibration matters**: MoGe's depth compression is non-linear and depth-range-dependent. A car at 30m and a bus at 8m in the same scene will have different MoGe→metric relationships. O2's global `(a, b)` is pulled toward the average across all objects and ranges in the frame, potentially mis-calibrating any individual object. O3 fits the affine transform in the exact depth range occupied by the object being reconstructed, giving SAM3D Objects a more accurately scaled pointmap for that specific instance.

**HDBSCAN filtering role**: Raw in-mask LiDAR is contaminated with ground returns, background bleeds through mask boundaries, and adjacent object hits. HDBSCAN clusters the in-mask points in 3D ego-frame space and keeps only the dominant cluster — physically the object surface. This clean set is then used for the affine fit. Operating in metric 3D space makes the cluster distance threshold physically meaningful and scale-invariant across distances (unlike image-plane DBSCAN where `eps` would need to grow with depth).

**Pointmap construction**: The fitted `(a, b)` is applied to the full frame `Z_moge_map` to get a full-frame `Z_metric` map. SAM3D Objects internally crops this to the object's bounding box — so the context pixels outside the mask but inside the crop also benefit from the calibration. A fresh calibrated full-frame pointmap is built for each object; the previous object's pointmap is discarded.

**Fallback**: if HDBSCAN finds fewer than `min_pts` clean in-mask points (e.g. small/distant objects at the edge of LiDAR range), fall back to O2's global affine for that object.

**This is a novel contribution**: the per-object affine calibration strategy and the HDBSCAN-based surface isolation are your own design. Cite MoGe (Wang et al., 2024) as the depth estimator and HDBSCAN (Campello et al., PAKDD 2013) as the clustering algorithm.

**Code change**: ~40 lines. Add `_compute_moge_local_affine_pointmap(frame, img_rgb, binary_mask)` method to `SAM3DObjectsModel`. Call it per-object inside `run_frame()`. MoGe result is cached per frame to avoid re-running it for every object in the same frame.

**Temporal**: More in-mask LiDAR returns → more stable per-object affine fit, fewer fallbacks to global.

**Novelty**: Medium-high. Novel application of per-instance depth calibration grounded in physically-motivated clustering.

---

#### O4 — Depth completion as dense LiDAR backend (network of choice: CompletionFormer)

**What it is**: Instead of passing the raw sparse LiDAR as the pointmap, first run a **LiDAR depth completion** network. Depth completion takes your sparse LiDAR scan + the RGB image and fills in a dense depth map — every pixel gets a metrically grounded depth estimate. The completed dense depth map becomes the pointmap input.

**Why this is better than O3**: O3 still relies on MoGe's relative depth structure and corrects it with a linear transform. Depth completion networks are trained end-to-end to fuse sparse LiDAR geometry directly with RGB features — the LiDAR contributes geometry, not just a scale factor. The resulting depth maps preserve LiDAR accuracy at measurement locations and use image structure to interpolate between them. Crucially, the correction is per-pixel and locally faithful, learned from data, not hand-crafted. This directly targets E5: for large close side-on vehicles, per-pixel LiDAR anchoring across the visible side surface grounds the depth gradient that MoGe was getting wrong.

**What it improves — effect on mesh and OBB**:
- **Metric scale (dominant)**: mesh comes out at correct absolute size.
- **Depth aspect ratio (secondary)**: the SS generator's voxel grid can extend correctly along the camera Z axis when the pointmap respects the true within-object depth spread. For E5 objects, this makes the mesh's front-to-back extent match reality instead of being compressed by MoGe's inaccurate within-object gradient. The mesh becomes longer in world coordinates (along the vehicle's own length axis after OBB fitting).
- **Silhouette/category shape (minimal)**: the pointmap has minimal effect on whether the mesh looks like a sedan vs. van vs. truck — this is dominated by SAM 3D Objects' learned priors (the paper's own LVIS ablation reports 48/52 preference, i.e. essentially no shape effect). Don't expect qualitative silhouette changes; expect OBB dimensions and yaw to improve.

**Choice of completion network: CompletionFormer**

Alternatives considered:
- **CompletionFormer** (Zhang et al., CVPR 2023) — chosen
- **BP-Net** (Tang et al., CVPR 2024) — higher KITTI RMSE rank, deferred as first-line replacement if CFormer underperforms
- **OGNI-DC** (Zuo & Deng, ECCV 2024) — stronger sparsity-robust generalisation, deferred as fallback if aggregation quality proves inconsistent
- **MapAnything** — multi-view depth densification, used in AutoBox; deferred to a separate multi-view ablation phase

Reasons for CompletionFormer over MapAnything (the main alternative):

1. **Single-camera scope of this pipeline stage**: the current pipeline processes one camera view at a time. MapAnything's core advantage is multi-view cross-camera attention across the 6-camera surround rig — this benefit is unavailable in the single-cam setting. Multi-view integration is deferred to a later ablation phase; when it comes, MapAnything is a natural fit for that separate comparison but should not be conflated with O4.

2. **Locally faithful metric depth via per-pixel LiDAR anchoring**: MapAnything uses a single learnable scale token per scene, which is why AutoBox requires an additional per-instance percentile-based alignment post-hoc — MapAnything can be locally metric-inaccurate for individual objects. CompletionFormer treats projected LiDAR points as near-hard per-pixel constraints, producing locally faithful metric depth at every LiDAR anchor without a post-hoc alignment step. For E5-style close side-on vehicles, per-pixel LiDAR anchoring across the visible side surface is exactly the architectural bias needed.

3. **Cleaner conceptual contrast with MoGe**: O2/O3 use MoGe (monocular relative depth) with a global or per-object affine correction. O4 with CompletionFormer occupies the same architectural slot (single-view depth model → dense pointmap) but swaps the depth model itself. This gives a clean ablation story: O2 vs O3 vs O4 all share the "single-view depth model" slot; only the model changes. Using MapAnything in O4 would additionally change the ablation to include multi-view integration, muddying the isolation of what causes any improvement.

4. **Sparsity regime matches pretrained weights**: nuScenes provides ~1–2% pixel coverage from a single 32-beam sweep; with ±3 sweep aggregation (see "Pre-processing: Multi-Sweep LiDAR Aggregation" in `implementation_notes.md`) this rises to ~5–7%, closely matching KITTI's 5.9% coverage that CompletionFormer was trained on. KITTI pretrained checkpoints therefore operate in-distribution on the sparse-input side, minimising the need for domain-specific retraining before validating the approach end-to-end. This is a practical cost/risk argument, not a quality argument — but it matters for iteration speed.

5. **Distinct from AutoBox**: AutoBox already demonstrates MapAnything in a multi-camera pseudo-labeling setup. Reusing MapAnything in O4 would dilute the contribution of this thesis. CompletionFormer positions O4 as a distinct densification route worth comparing against AutoBox rather than being a component of it.

**Workflow**

Because SAM 3D Objects consumes a full-image pointmap, depth completion runs **globally per camera view**, not per-object mask. The dense output naturally supports per-mask extraction downstream. Reasons for the global approach:
- Depth completion networks are trained on full images and leverage scene-wide context (sky, ground plane, horizon lines, vanishing structure). Object crops lose this context and degrade quality.
- One forward pass per camera view (~50–150 ms on modern GPU) is compute-efficient vs. N passes per view.
- Preserves alignment with the MoGe pipeline: MoGe also produces a full-frame pointmap, so O4 becomes a drop-in slot replacement.

Per-camera steps:

1. **Temporal aggregation**: ±3 sweeps for nuScenes / ±2 for ECP, ego-motion compensated to anchor frame. Already implemented — see "Pre-processing: Multi-Sweep LiDAR Aggregation" in `implementation_notes.md`.

2. **Global pre-filters**: ego-body exclusion zone, max BEV range 52 m. Already implemented — see "Pre-processing: Point Cloud Pre-filters" in `implementation_notes.md`.

3. **Projection to camera view**: standard extrinsic + intrinsic projection. Where multiple aggregated points project to the same pixel (common with 5–7 sweeps), keep the **minimum depth per pixel** — this preserves the foreground surface at every pixel, since the camera can only see the closest surface and min is the only operation that guarantees foreground over background at occlusion boundaries. Average or max would produce depths that don't correspond to any real surface.

    *Note on anchor-frame-only as an alternative*: using only the anchor sweep (N_BEFORE=N_AFTER=0) avoids any multi-sweep collision entirely and is guaranteed correct for dynamic objects at their exact timestamp. It is conceptually cleaner but gives ~1–2% pixel coverage on nuScenes (32-beam), well below the ~5.9% KITTI coverage that CompletionFormer was trained on, degrading completion quality at distance. The aggregated approach (~5–7% coverage) is preferred. Switching to N_BEFORE=N_AFTER=0 in the config is the easiest way to compare single-sweep vs. aggregated completion quality.

4. **Parallax/occlusion filter** *(deferred — apply only if artifacts are observed)*: Because the LiDAR sits ~1.8 m above the cameras on the nuScenes vehicle, a LiDAR beam can bypass a foreground object and strike the background, but when projected into the camera image appears inside the foreground object's 2D mask. This creates a false background anchor for CompletionFormer inside that object's region.

    In practice this is rare and CompletionFormer is more robust to isolated outliers than O3's affine fit (which is directly corrupted by a single bad point). The global pre-filters (ego-body exclusion, 52 m range) already suppress the most severe cases. **Skip this filter in the prototype** and add it only if visible boundary artifacts appear. If needed, reuse the existing `filter_inmask_lidar_hdbscan` utility per SAM3 mask — this is strictly stronger than the 1.5×median rule and is already implemented and tested. The filter is applied to the *sparse input depth map* before CompletionFormer, not to the completed output.

5. **Depth completion forward pass**: `CompletionFormer(rgb_full, sparse_depth_full) → dense_depth (H, W)`.

6. **Back-project to pointmap**: `p = d * K^{-1} * [u, v, 1]^T` per pixel → `(H, W, 3)` pointmap in camera frame. Convert to PyTorch3D (negate X, Y) → SAM 3D Objects input.

7. **SAM 3D Objects inference**: standard call with `pointmap=<completed>`. No affine calibration step (depth is already metric). Masks gate which pixels contribute to which object mesh downstream.

8. **(Optional) Post-completion mesh sanity**: HDBSCAN can still be applied to extracted mesh vertices per object to reject boundary artifacts from completion. Cheap post-hoc insurance.

**Moving-object aggregation caveat**: For dynamic classes (car, truck, pedestrian, bicycle, motorcycle) the ±k sweeps smear across a spatial trail because ego-motion compensation cannot account for the object's own motion. In practice the smeared points from non-anchor sweeps project to *different* image pixels than the anchor-frame points (the object has moved), so pixel collisions are rare and the min-depth rule naturally prefers whichever sweep's point lands on a foreground surface. Class-aware aggregation (anchor-only inside dynamic masks, full aggregation elsewhere) is a deferred option if metrics regress.

**Fallback chain for O4**:

| Condition | Behaviour | Mode tag |
|---|---|---|
| No LiDAR in frame | Fall back to O3 (per-object local affine). | `o3_fallback` |
| CompletionFormer produces mostly invalid depth (>50% pixels) | Fall back to O3 for that frame. | `o3_fallback` |
| Normal | Use completed depth pointmap. | `o4_local` |

**Expected wins and losses vs. O3**:
- **Large close side-on vehicles (E5)**: significant improvement expected — per-pixel LiDAR anchoring directly grounds the depth gradient MoGe was getting wrong.
- **Head-on vehicles at medium range**: roughly on par with O3 — O3's affine already works well when within-object depth is near-uniform.
- **Distant sparse objects (>30 m)**: possibly worse than O3 — CompletionFormer with sparse anchors degrades toward monocular depth, whereas O3's affine can still lock a small number of LiDAR points to a good scale.

Evaluation should stratify by object distance, view angle, and size class to isolate where O4 wins vs. where O3 wins. Depth-quality check against held-out LiDAR (removed from input, ~1000 pixels per frame, RMSE per range bin 0–15/15–30/30–50 m) is worth running before the full SAM 3D Objects end-to-end, so the depth quality is characterised independently of the downstream mesh generation.

**Code change**: New wrapper class (~50–80 lines) — `LiDARCompletionDepthModel` matching the `depth_model` interface of `InferencePipelinePointMap`. Loads KITTI-pretrained CompletionFormer weights. Feeds (RGB, per-pixel sparse depth built from steps 1–4) → returns `(H, W, 3)` pointmap. No changes to SAM 3D Objects source. Detailed implementation, config values, and code paths to be added to `implementation_notes.md` once first working version is in place.

**Temporal**: Denser LiDAR from temporal aggregation → completion network works better → better pointmap → better SS generation. Already leveraged via existing aggregation pre-processing.

**Novelty**: Medium-high. Replaces a monocular depth estimator with a sensor-fused network; positioning as a distinct architectural slot from AutoBox's multi-view route (MapAnything) is part of the contribution.

---

#### O5 — Mid-pipeline LiDAR correction of the sparse structure (novel contribution)

**What it is**: After the SS generator produces its voxel grid (Step 3 in the pipeline), and **before** the SLAT generator runs (Step 4), you intervene and correct the voxel occupancy using LiDAR evidence.

**Why "mid-pipeline"**: The first stage (SS generator) has already run and produced a 3D skeleton. The remaining stages (SLAT generator, decoders) still need to run. You are injecting LiDAR between stage 1 and stage 2. Not pre-inference. Not post-inference. Between the two generative stages.

**What LiDAR tells you about the voxel grid**:
- **Occupied but no LiDAR support**: If a voxel is marked ON by the model but no LiDAR ray passes through that region (and LiDAR had line-of-sight), the voxel is likely a hallucination → turn it OFF
- **Not occupied but LiDAR hits here**: If LiDAR clearly returns a point inside a voxel that the model marked OFF, the model missed it → turn it ON
- **Uncertain region (far side, occluded)**: LiDAR has no evidence here either way → leave the model's estimate intact

**What it improves**:
- Shape: voxel corrections directly change which 3D cells are filled, changing the outline and dimensions of the generated mesh
- Dimensions: if the model generates a car that is too narrow (a common failure mode), and LiDAR shows hits on the far side, those hits will expand the voxel grid
- Orientation: if the model guesses the car is facing the wrong direction, LiDAR points on the visible face will reinforce the correct orientation

**What SLAT and mesh decoders inherit**: the SLAT generator fills in surface detail for each occupied voxel. If the voxel set is more accurate due to LiDAR correction, the resulting mesh surfaces are built on a more accurate skeleton.

**Code change**: ~100 lines. Write a function `correct_voxels_with_lidar(ss_coords, lidar_points_in_voxel_space, K, threshold)` that takes the voxel occupancy output from `sample_sparse_structure()` and returns a corrected version. Call it between `sample_sparse_structure()` and `sample_slat()` in your pipeline runner.

**Temporal**: More LiDAR sweeps → denser point cloud → more confident voxel corrections → better shape recovery. Especially powerful for static objects where full surface can be accumulated.

**Novelty**: High. This is a novel mid-pipeline intervention that directly couples LiDAR geometry with the generative 3D reconstruction process. O4 improves the depth input; O5 corrects the 3D structure itself.

---

---

### SAM3D Body Options

---

#### B1/B2 — LiDAR-corrected translation (post-inference depth override)

**What it is**: After `run_frame()` returns body mesh results, override the `tz` depth component of `pred_cam_t` with the actual LiDAR depth measured at the person's location. B1 and B2 are mathematically equivalent — treat them as one option.

**What `pred_cam_t` is**: The 3D position vector [tx, ty, tz] of the person's body root (pelvis) in camera coordinates. tx = lateral, ty = vertical, tz = depth. "Translation" in 3D graphics simply means position/placement in 3D space. Without LiDAR, tz is computed from a predicted scale parameter `s` that the model guesses from image appearance — the core source of error.

**How to get the LiDAR-correct tz**: 
1. Project all LiDAR points into the image for this frame
2. Find the points that land inside the person's bounding box
3. Take the median depth of those points (robust to outliers)
4. This is your `tz_lidar`
5. Recompute `tx` and `ty` accordingly (they depend on `tz` in the CLIFF formula)

**What it improves**: Localization. The body mesh is now placed at the correct metric distance. Everything downstream that uses `pred_cam_t` (3D bounding box fitting, tracking, scene reconstruction) benefits.

**Code change**: ~20 lines in your pipeline wrapper after calling `run_frame()`.

**Temporal**: Over multiple frames, you accumulate LiDAR observations of the same person. Kalman filtering or temporal smoothing of depth estimates makes them more stable and robust to noisy individual frames.

**Novelty**: Low. Trivial post-processing. Establish this as your baseline to measure improvement.

---

#### B3/B4 — LiDAR-depth-enhanced CameraEncoder (low-level architectural integration, novel contribution)

**What it is**: Upgrade the `CameraEncoder` in `camera_embed.py` to use 3D spatial positions from LiDAR instead of 2D ray directions.

**What CameraEncoder currently does**: For each pixel in the person crop, it computes a camera ray direction (dx, dy, 1.0) — a unit vector pointing "which direction is this pixel looking?" This is Fourier-encoded and mixed into the backbone patch features via a 1×1 conv. The result: each patch embedding "knows" which direction in 3D space it corresponds to.

**The problem**: direction alone cannot resolve depth. The ray for a pixel at (u,v) points in one direction but could intersect a person at 2m or 10m. The model must guess the scale from appearance features — this is exactly what goes wrong.

**What the change does**: Replace the dummy z=1 with real LiDAR depth. The ray (dx, dy, 1.0) becomes a 3D position (X, Y, Z) in camera space — where that pixel physically is in the real world. Now:
- Each patch embedding "knows" its metric 3D position, not just its direction
- The backbone features carry both appearance AND geometry at a per-pixel level
- The decoder can reason about true metric scale when estimating body dimensions
- The camera head no longer needs to guess `s` from appearance — it can learn that "this crop is at Z=4.2m" from the conditioned features

**Code change in `camera_embed.py`**:
- Line 35: replace `ones_like(rays[..., :1])` with projected LiDAR depth per pixel
- Adjust `FourierPositionEncoding(n=3, ...)` (already n=3, no change needed)
- The conv `embed_dim + 99 → embed_dim` stays the same — same dimensionality, richer content
- New function call signature: `CameraEncoder.forward(img_embeddings, rays, depth_map=None)` where `depth_map` is projected LiDAR depths (NaN where no LiDAR)

**Also connect to B3 (init_estimate for camera head)**: While you're at it, pass the median LiDAR depth as `init_estimate` to `PerspectiveHead.forward()`. This is the residual architecture — the head predicts a correction on top of your LiDAR prior rather than predicting from scratch. The two changes together are B3+B4 combined.

**What it improves**:
- Depth (tz): directly grounded by LiDAR at feature level
- Scale: body dimension estimation grounded in metric space
- The improvement propagates through the entire decoder, not just the camera head

**Temporal**: Aggregated LiDAR gives denser per-pixel depth in the crop. Pedestrians have few LiDAR returns — even one or two valid depth pixels in the crop provide a strong metric anchor for the `CameraEncoder`.

**Novelty**: High. Low-level architectural integration. The model's feature representation is physically grounded. Clear motivation from first principles.

---

## 4. Summary: Shortlisted Options

| ID | Model | Integration Level | What it fixes | Code change | Novelty |
|---|---|---|---|---|---|
| O1 | Objects | Pre-inference (sparse LiDAR) | Scale, rough depth — lower bound baseline | None (+ filtering) | Low |
| O2 | Objects | Pre-inference (MoGe + global affine) | Scale, metric grounding — simplest dense baseline | ~15 lines | Low |
| O3 | Objects | Pre-inference (MoGe + local affine) | Per-object depth calibration, HDBSCAN surface isolation | ~40 lines | Med |
| O4 | Objects | Pre-inference (depth completion) | Scale, depth, per-pixel metric accuracy | ~50 lines wrapper | Med-high |
| O5 | Objects | Mid-pipeline (voxel correction) | Shape, dimensions, orientation | ~100 lines | High |
| B1/B2 | Body | Post-inference | Localization (tz) | ~20 lines | Low |
| B3/B4 | Body | In-model (CameraEncoder) | Depth, scale, feature-level 3D grounding | Small source edit | High |

---

## 5. Temporal Extension Plan

All options above work on a single frame. Once the basics work, every option benefits from temporal aggregation:

| Option | Temporal benefit | How to implement |
|---|---|---|
| O1 | More LiDAR hits per object crop | Accumulate N sweeps with ego-motion compensation |
| O2 | More in-image returns → stabler global affine fit | Same accumulation, pass denser scan to affine fitting |
| O3 | More in-mask returns → stabler per-object fit, fewer fallbacks to O2 | Same accumulation; HDBSCAN scales naturally to denser input |
| O4 | Completion network gets denser input → better fill | Same accumulation, pass denser scan to completion network |
| O5 | More confident voxel corrections, fewer "no LiDAR" holes | Denser accumulated cloud → better voxel occupancy evidence |
| B1/B2 | Temporal depth filtering → stable tz per person | Kalman filter on tracked person depth over frames |
| B3/B4 | Denser depth pixels in crop → stronger metric anchor | Aggregate nearby frames for each pedestrian crop |

For **static objects** (parked cars, infrastructure): temporal aggregation is straightforward. Use ego-pose from nuScenes to transform all sweeps to a common reference frame, stack the point clouds.

For **moving rigid objects** (driving cars): transform each sweep's points using the tracked 3D box pose (nuScenes provides tracking annotations). Align to the object frame, stack, then transform back.

For **non-rigid objects** (pedestrians): full body aggregation is hard (body changes shape). Only position/depth averaging makes sense. Pose aggregation is a research problem in itself.

---

## 6. Implementation Order

### Step 0 — In-mask LiDAR filtering (prerequisite, do first)

**HDBSCAN is the chosen method**, validated in `AutoLabeling/lidar_filtering_exploration.ipynb`. It clusters in-mask points in 3D ego-frame space, making the distance threshold physically meaningful and range-invariant. The dominant cluster is kept as the clean object surface; noise and background bleeds are discarded. Cite: Campello et al., *Density-Based Clustering Based on Hierarchical Density Estimates*, PAKDD 2013. This filtering is used by O3 and is available as a utility for all options.

### Phase 1 — Baselines (validate the gap before fixing it)

1. **Measure the current pipeline** on nuScenes mini without any LiDAR. What is the depth error of `pred_cam_t` vs. LiDAR ground truth for Body? What does the SAM3D Objects mesh scale error look like vs. GT 3D boxes? Establish numbers before touching anything.

2. **B1/B2**: Implement LiDAR-corrected translation for Body. ~20 lines. Measure improvement in 3D localization.

3. **O1**: Plug filtered sparse LiDAR as pointmap for Objects. No SAM3D source change — just LiDAR projection in your wrapper.

4. **O2**: Port MoGe + global affine from the notebook into the pipeline wrapper. Already implemented — just wire it up. Measure improvement over O1.

5. **O3**: Per-object local affine with HDBSCAN. Add `_compute_moge_local_affine_pointmap()` to `SAM3DObjectsModel`. Compare against O2 — the difference isolates the effect of per-object vs. scene-wide calibration.

### Phase 2 — Learned dense depth integration

6. **O4**: Integrate a depth completion network (CompletionFormer). Measure improvement over O3. This becomes the strong pre-inference baseline for Objects.

### Phase 3 — Novel contributions (architectural integration)

7. **B3/B4**: Upgrade CameraEncoder to use 3D ray positions from LiDAR. Requires editing `camera_embed.py` and `sam3d_body.py`. Test with and without — the difference should be larger than B1/B2 alone.

8. **O5**: Mid-pipeline voxel correction for Objects. Implement LiDAR voxel projection and occupancy correction. Insert between `sample_sparse_structure()` and `sample_slat()` in your pipeline runner.

### Phase 4 — Temporal extension (if time allows)

9. **Run O1/O2/O3/O4/O5 with N-sweep accumulated LiDAR**. Static objects first (no motion compensation needed). Moving objects second.

### Evaluation at each phase

At every phase, evaluate on:
- nuScenes mini: 3D localization (translation error vs. GT 3D boxes), mesh scale (IoU of predicted vs. GT 3D box volume)
- ECP: 2D metrics (since ECP has no depth annotations, evaluate body pose projection quality)
- Qualitative: visualize meshes overlaid on LiDAR point cloud to see alignment

---

## 7. Literature & Citation Suggestions

### O1 — Sparse LiDAR pointmap
No method paper needed — this is an ablation baseline using the built-in `InferencePipelinePointMap` pathway. Cite:
- **nuScenes dataset**: Caesar et al., *nuScenes: A multimodal dataset for autonomous driving*, CVPR 2020.
- **SAM3D Objects** (the model whose pointmap pathway we use): cite the SAM3D Objects paper directly.

### O2 — MoGe + global affine calibration
- **MoGe**: Wang et al., *MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision*, arXiv:2410.19115, 2024.
  - MoGe is explicitly designed with affine-invariant supervision — fitting `Z = a·Z_moge + b` matches its stated calibration protocol. This is the primary justification for the method.
  - *Confidence: high.*
- **Scale-shift (affine) calibration protocol**: Ranftl et al., *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer*, IEEE TPAMI 2022 (MiDaS).
  - This is the standard citation for the least-squares `Z_metric = a·Z_pred + b` alignment approach in monocular depth.
  - *Confidence: high.*
- **Earlier affine calibration origin** (verify — may predate MiDaS): Eigen et al., *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network*, NeurIPS 2014. Introduced scale-invariant depth error, often cited alongside affine calibration.
  - *Confidence: medium — verify whether this is the right Eigen et al. citation for affine calibration specifically.*

### O3 — Depth completion (learned dense fusion)
- **CompletionFormer**: Zhang et al., *CompletionFormer: Depth Completion with Convolutions and Vision Transformers*, CVPR 2023.
  - *Confidence: high on title/venue.*
- **PENet**: Hu et al., *PENet: Towards Precise and Efficient Image Guided Depth Completion*, ICRA 2021.
  - *Confidence: high on title/venue.*
- **KITTI depth completion benchmark** (standard eval set all completion papers use): Uhrig et al., *Sparsity Invariant CNNs*, 3DV 2017.
  - *Confidence: medium — verify this is the right citation for the KITTI depth completion split.*
- **Additional candidates to check**: NLSPN (Park et al., ECCV 2020), CFormer, IP-Basic (Ku et al., CRV 2018). Look at what the CompletionFormer paper itself cites for a curated list.

### Prompt for verifying / finding citations
If unsure, use this with a search-enabled AI:

> I'm writing a thesis chapter on LiDAR integration into a 3D object reconstruction pipeline. I need verified citations (full title, authors, venue, year) for:
> 1. **MoGe** — monocular geometry estimation with affine-invariant supervision, arXiv:2410.19115, 2024. Confirm authors and title.
> 2. **Scale-shift affine calibration for monocular depth** — fitting `Z = a·Z_pred + b` by least-squares on sparse GT. My candidate citation is MiDaS (Ranftl et al., TPAMI 2022). Is there an earlier / more canonical source (e.g. Eigen et al. 2014)?
> 3. **LiDAR depth completion** — the two or three most-cited RGB-guided sparse-to-dense completion papers for autonomous driving. My candidates: CompletionFormer (CVPR 2023), PENet (ICRA 2021). Confirm and suggest others.
> 4. **KITTI depth completion benchmark split** — the paper that defined it. My candidate: Uhrig et al., *Sparsity Invariant CNNs*, 3DV 2017. Confirm.

---

## 8. Key File Locations

```
SAM3D Body:
  Source root:         SAM3D/sam-3d-body/sam_3d_body/
  CameraEncoder:       sam_3d_body/models/modules/camera_embed.py
  PerspectiveHead:     sam_3d_body/models/heads/camera_head.py
  Full model:          sam_3d_body/models/meta_arch/sam3d_body.py
  Estimator entry:     sam_3d_body/sam_3d_body_estimator.py
  Our wrapper:         AutoLabeling/src/autolabeling/models/sam3d_body.py

SAM3D Objects:
  Source root:         SAM3D/sam-3d-objects/sam3d_objects/
  Base pipeline:       sam3d_objects/pipeline/inference_pipeline.py
  Pointmap pipeline:   sam3d_objects/pipeline/inference_pipeline_pointmap.py
  PointPatchEmbed:     sam3d_objects/model/backbone/dit/embedder/pointmap.py
  SS flow model:       sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py
  Preprocess utils:    sam3d_objects/pipeline/preprocess_utils.py
  Our wrapper:         AutoLabeling/src/autolabeling/models/sam3d_objects.py

Autolabeling pipeline:
  Our pipeline:        AutoLabeling/autolabeling_pipeline.ipynb
  Config:              AutoLabeling/configs/ecp.yaml
```
