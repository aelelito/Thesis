> BACKLOG of ideas (too big to follow as a plan). The simple working plan is `sam3d_mask_freespace_plan.md`.

# Investigation plan: does SAM3D Objects respect the mask, and does its output enter LiDAR-certified free space?

Status: 2026-09-25, PLAN ONLY (nothing below has been run yet). Companion notes: `sam3d_objects_input_analysis.md`
(how SAM3D reads its inputs), `contribution_ideas/phase0/results/phase0_results.md` (free-space overshoot of OBBs),
`contribution_ideas/contribution_plan.md` (Tier 1/Tier 2 ideas).

Legend: **[code]** read in source, **[measured]** measured on our data, **[hypothesis]** not yet tested, **[open]** unknown.

---------------------------------------------------------------------------------------------------------------------

## 0. The question from the professor, stated precisely

Professor: LiDAR should also drive the 3D *shape* (accepted). Free space: "method 1 already covers this; which information
would we use beyond what SAM3D already has?"

His implicit argument: mask + correct depth => mesh reprojects into the mask, sits at the right depth, has the right
scale => it cannot touch certified-free space. Phase 0 saw OBBs touching free space. Three things can break the argument,
and they are different claims that must be tested separately:

| link in the argument | what could be wrong | test |
|---|---|---|
| A. "the mesh fits the mask" | the mask is only conditioning, not a constraint [code: alpha channel + crop + DINO, no loss/projection at inference; layout post-opt is off by default] | Section 2 (silhouette check + mask-perturbation test) |
| B. "mask + depth pin the object" | the mask pins only the 2D silhouette; extent ALONG the viewing ray (front/back), and the object's true size, are not pinned by a silhouette. Depth enters only as a median (shift) and as soft tokens | Section 3 (depth agreement + pointmap-perturbation test) |
| C. "then it cannot enter free space" | (i) it can, or (ii) our measurement is wrong: the OBB (not the mesh) touches free space, LiDAR/camera time offset, LiDAR noise near surfaces, box corners are legitimately empty air | Section 4 (mesh-level free-space test with controls) |
| D. "free space adds nothing beyond method 1" | may be true if free space is only implied by the visible surface; may be false where free space carries information the visible surface does not (see 5) | Section 5 (information-content ladder) |

I do NOT assume the outcome of any of these. Section 6 lists what each outcome would mean for the thesis.

---------------------------------------------------------------------------------------------------------------------

## 1. What we already know and what we do not (audit before building)

Known [code]/[measured] (details in `sam3d_objects_input_analysis.md`):
- The mask enters as (a) alpha channel -> cropped mask + full mask, each fed to a second DINOv2 as an IMAGE; (b) it selects
  the pointmap pixels that define the SHIFT (median 3D point). `rembg` (multiplying the image by the mask) is NOT in the
  pointmap pipeline config, so the RGB crop is not background-cleared. No mask-based loss or projection at inference.
- Stage 1 predicts rotation, scale, translation in normalised units; metres = predicted * scale + shift. The mesh is
  generated in a canonical cube and then placed by that pose.
- Crop pointmap tokens matter for size (drop probe: sizes move 10-44%), full-image tokens barely.
- `layout_post_optimization` (mask-IoU render-compare) exists, is OFF by default, and skips occluded/truncated objects.

Traps in the existing evidence:
1. **Phase 0 used OBBs from the OLD pipeline (O3+B1, all cameras).** The clean pipeline (per-sweep TerraSeg, mode 2/6...)
   has not been measured against free space. Numbers must be redone with the current pipeline.
2. **An OBB is not the mesh.** The OBB is a PCA fit over the mesh (plus rider merge for bikes). Stray vertices or the
   footprint fit can extend the box beyond the surface. Phase 0 cannot tell "mesh enters free space" from "box is looser
   than the mesh". Phase 0 partly controls for box looseness by subtracting the GT-box floor, but that does not remove
   the mesh-vs-box question.
3. **Stored meshes are slimmed** (`mesh_points: 20000`, faces dropped, checkpoints hold camera-frame R3 vertices + mask +
   affine/ssi info, not the raw pose). Silhouette rendering and inside/outside tests need FULL meshes with faces.
   => the analysis run must use `mesh_points: 0` (keep full) on the selected frames.
4. **LiDAR-camera time offset** (up to ~50 ms; moving objects shift 0.2-0.5 m) and rolling-shutter-like sweep smear are
   not corrected (Phase 0 limitation 6). A moving car can "enter free space" purely because the sweep and the image see it
   at different moments. Must be stratified by object speed (nuScenes: velocity from consecutive annotations; parked =
   speed ~0) before any claim.
5. GT is OBB only. Mesh vs GT can only be judged through the GT box (mesh outside the GT box, GT box free-space floor).

---------------------------------------------------------------------------------------------------------------------

## 2. Step 1: Does the mesh fit the mask? (backprojection)   [answers link A]

### 2.1 Measurement (per object, no assumptions about occlusion)
Rasterise the FULL posed mesh with the dataset intrinsics (pytorch3d is already in the container) into a silhouette and a
depth map at image resolution. Check that the projection model matches SAM3D's (pointmap built from K, PyTorch3D
convention (-X,-Y,Z); `_mesh_to_r3` already converts into camera space) by first projecting the pointmap's own points
and confirming they land on their pixels.

Three quantities, kept separate because occlusion makes them asymmetric:
- **Recall = |mask AND mesh| / |mask|**: mask pixels the mesh does not cover. Not explainable by occlusion (the mask
  pixel is visible object). This is a hard mismatch (mesh too small/misplaced or mask wrong).
- **Leak = |mesh AND NOT mask| / |mesh|** split into: (a) leak behind an occluder (another object's mask or a closer
  surface: legitimate), (b) leak in free view (mesh visible where the segmentation says background: mismatch).
  Occluder test: pixel belongs to another SAM3 mask, or in-image depth (LiDAR / dense map) is closer than the mesh depth.
- **IoU** for the headline number, plus boundary distance (Chamfer between silhouette contour and mask contour, px and
  as metres at the object's depth).
Stratify by: truncation by image edge, occlusion (leak-behind-occluder share), mask size, range, class, mode (2 vs 6).

### 2.2 Reference points, so the number is interpretable
- **Meta's own refinement as a reference:** run `layout_post_optimization` on the same objects (flag exists; note the
  wrapper hard-codes ICP off; GS path takes priority, so force `decode_formats=['mesh']`). What IoU does the shipped
  refinement reach and how much does it move the pose? That is the cheapest existing "mask as a constraint" baseline
  and must be cited/beaten either way.
- **Noise floor of the mask itself:** mask boundary jitter (SAM3 edge quality) limits any IoU. Estimate by comparing
  SAM3 masks between two prompts/runs of the same object, or by the IoU of the mask with its own 1-px dilation.

### 2.3 Causal test: is the mask a constraint or a hint?   [the decisive experiment for link A]
Same object, same pointmap, same seed, change ONLY the mask and re-run stage 1 (and 2 for the mesh):
1. mask eroded / dilated by k px (k = 5, 10, 20),
2. mask translated by a few px,
3. mask replaced by its bounding box (filled),
4. mask of a different object (control: does the mesh follow the mask or the image?),
5. mask dropped: `force_drop_modalities` for `mask` / `rgb_image_mask` (check the key names in the embedder; the pointmap
   keys are `pointmap`, `rgb_pointmap`).
Measure IoU of the new mesh silhouette with the NEW mask and with the ORIGINAL mask, and pose change. If the mask is a
hard constraint the mesh follows the new mask; if it is a hint the mesh keeps following the image (DINO-RGB) and its own
prior. Note the shift also changes when the mask changes (median over a different pixel set) -> log shift so pose changes
caused by the shift are not misread as shape changes.
Also known from the training code: masks were augmented by boundary and translation perturbations
(`perturb_mask_boundary`, `perturb_mask_translation`), i.e. SAM3D was trained to be tolerant of imperfect masks. That is
[code]; it suggests softness by design, but the perturbation test above is the evidence.

Outcome of Step 1 = a table (IoU/recall/leak by stratum, mode) + a figure per object: image with mask contour, mesh
silhouette contour, colour-coded disagreement regions.

---------------------------------------------------------------------------------------------------------------------

## 3. Step 2: How does the mesh relate to the depth it was given?   [answers link B]

### 3.1 Depth agreement (occupied-side evidence, no free space yet)
From the rasterised depth, for every LiDAR return inside the mask (the cleaned points that defined the affine, and ALL
raw in-mask returns separately): signed residual `z_lidar - z_mesh_front(pixel)`. Positive = mesh surface in front of the
LiDAR surface (mesh too near / too fat towards the camera), negative = mesh behind the return (LiDAR point floats in
front of the mesh). Report median, IQR, per object; also vs the pointmap SAM3D was fed (mesh depth vs in-mask pointmap
depth). This answers "does SAM3D reproduce the depth it was given" without any voxel grid.
Also the bias of the median: since shift = MEDIAN in-mask depth, a symmetric object's front surface should sit in front
of the shift by ~half its depth extent; check the sign/size of the residual against that expectation (a systematic,
explainable offset is not an error).

### 3.2 Causal test: pointmap perturbation (crop stream)
Same object/seed, perturb only the in-crop pointmap and measure mesh change:
- add a depth ramp / tilt inside the mask, flatten depth to the median, stretch the depth range around the shift by
  0.5x/1.5x (keeps the shift, changes the "thickness" the tokens describe),
- constant offset along the ray (moves the shift: expected to translate the mesh 1:1, a sanity check of link B).
If the mesh thickness/extent along the ray follows the stretched map, the network can use depth shape; if not, depth only
places the object and shape comes from the prior. Relates to the token-drop probe already done (that showed crop tokens
matter, but not whether they act as a constraint).

Outcome of Step 2 = depth-residual distributions + response curves (mesh change vs perturbation).

---------------------------------------------------------------------------------------------------------------------

## 4. Step 3: Free-space evidence for meshes, with controls   [answers link C]

### 4.1 Free-space definition for the test (do not reuse Phase 0 blindly)
- Source: single LiDAR sweep nearest to the CAMERA timestamp (not the aggregated cloud: aggregation smears movers and
  its rays come from other times). Phase 0 ray casting (Amanatides-Woo, numba, `experiment_a_batch.py`) can be reused;
  keep its threshold logic (one traversal certifies free; Phase 0 section 3.4 bug history).
- **Margin near returns:** a beam grazes surfaces and range noise is ~2-3 cm. Count a mesh sample as violating only if
  the free evidence is at least `d` in FRONT of the nearest return along the ray (test d = 0.1, 0.2, 0.3 m). Report
  the curve over d, not one threshold.
- Voxelisation is only an implementation detail. Also implement the ray form: for each LiDAR beam, the free segment is
  [sensor, return - d]; the mesh violates it if the segment intersects the mesh (ray-mesh intersection, penetration
  length). This needs no grid and is the form to show the professor. Use the grid version to cross-check.

### 4.2 Measures (per mesh)
- Fraction of mesh SURFACE samples in certified-free space, and penetration depth (m) along the violating beams.
- Fraction of mesh VOLUME in free space (inside test via ray parity / winding number; check watertightness first,
  otherwise use surface samples only).
- Same on the OBB fitted from that mesh (`obb_raw`, full mesh) and on the GT OBB -> separates mesh from box effect.
- Mesh outside the GT OBB (dilated by 0.2 m): volume fraction (uses GT as a reference where no GT mesh exists).

### 4.3 Controls (without them the numbers are not evidence)
1. **GT floor:** the same free-space test on the GT box (shrunk by a few cm for its own rounding) gives the false-alarm
   rate from sensor noise, time offset and box corners. A mesh is only "overshooting" beyond that floor (Phase 0's
   GT-correction logic, now per object).
2. **Speed stratification:** static (parked) vs moving objects. If violations concentrate on movers the cause is timing,
   not shape.
3. **Near-surface exclusion sweep** (the d curve above).
4. **Synthetic sanity check:** take a GT box, inflate it by 10/20/30 cm on one side, and confirm the detector flags it;
   deflate and confirm it does not. Establishes sensitivity/specificity of the test on known truth.
5. **Attribution of each violation:** does the violating region lie (a) along the viewing ray inside the mask (front/back
   extent, invisible to a silhouette), (b) sideways/beyond the silhouette (would mean link A is broken), (c) near the
   ground? Colour the violating surface patches by that class. This separates "mask not respected" from "mask respected
   but ray-extent wrong".

Outcome of Step 3 = a per-object table (mesh, mesh-OBB, GT-OBB violation, speed class, range, class), a histogram with the
GT floor as reference line, and 3D/BEV/side pictures with the violating patches highlighted.

---------------------------------------------------------------------------------------------------------------------

## 5. Step 4: What information does free space add beyond "LiDAR for shape"?   [answers link D]

The professor's objection deserves a direct test rather than an argument. Two parts.

### 5.1 Argument to test (hypotheses, [hypothesis])
- H-implied: for a pixel with a LiDAR return on the object, "space in front of that return is empty" is implied by "the
  surface is at that depth". A method that forces the mesh surface to match in-mask returns already excludes that free
  space. => free space adds nothing THERE.
- H-viewpoint: the LiDAR sits at a different origin/height than the camera (and, with multi-sweep, several origins), so
  its free segments cross the object volume from directions the camera silhouette + depth do not constrain (e.g. beams
  passing between wheels/under a truck bed, over the hood, beside a pedestrian).
- H-gaps: in-mask pixels without a return (sparse LiDAR, 94% of tokens invalid) carry no depth constraint, yet beams
  nearby may certify space in front of / around those pixels.
- H-authority: even where implied, a *measured* hard constraint may be stronger than a soft token the network can ignore
  (Step 2 quantifies how well it is followed).

### 5.2 Experiment: information ladder (post-hoc, no retraining)
Starting from the SAM3D mesh, apply increasingly informed post-hoc alignments and measure after each step (a) GT error
(centre, size ratios, yaw vs GT OBB), (b) residual free-space violation, (c) mask IoU:
- L0: as generated (mode 2).
- L1: + silhouette only (mask-IoU render-compare; this is what `layout_post_optimization` does).
- L2: + occupied evidence (in-mask LiDAR returns must lie on the surface: point-to-mesh distance; this stands for
  "method 1" in the professor's terms).
- L3: + free-space term.
The question is whether L3 improves or removes violations that L2 leaves, i.e. whether free space carries independent
information. If L2 already drives free-space violation to the GT floor, the professor is right for this failure.
If violations remain after L2 and L3 fixes them without hurting GT error, we have the evidence he asked for. Keep the
optimiser simple (9-DoF: 3 rot, 3 trans, 3 anisotropic scale; CMA-ES or coordinate descent) — this is a probe, Tier 1
in `contribution_plan.md` section 10, not the final method.
Additionally count violations by kind: violating beam ends INSIDE the mask (implied by L2) vs OUTSIDE the mask / not
visible from the camera (not implied) -> directly quantifies H-implied vs H-viewpoint/H-gaps with no optimiser at all.

---------------------------------------------------------------------------------------------------------------------

## 6. How to read the outcomes (decided in advance, to avoid post-hoc story telling)

| Finding | Meaning |
|---|---|
| Step 1: mesh silhouette matches the mask (IoU high, small leak, mask-perturbation: mesh follows new mask) | link A holds; free-space violation cannot be blamed on the silhouette |
| Step 1: IoU clearly lower, mesh ignores perturbed masks | mask is a hint; a silhouette term is a legitimate first contribution and explains part of the violation |
| Step 2: mesh reproduces input depth within ~few cm and follows stretched maps | depth is used as a constraint already; "use LiDAR for shape" (accepted idea) has little headroom |
| Step 2: large residuals, mesh ignores depth shape | depth only places the object; the accepted "LiDAR for shape" idea has headroom and is directly demonstrated |
| Step 3: violation on GT-OBB about equal to mesh/OBB violation, or concentrated on movers | Phase 0 signal was sensor/timing/box artefact; do not claim overshoot from it |
| Step 3: violation clearly above the GT floor on STATIC objects, at mesh level | real, defensible evidence that the output contradicts the sensor |
| Step 4: L2 leaves violations, L3 removes them without worse GT error | free space carries information beyond method 1 -> answer to the professor |
| Step 4: L2 already removes them | drop the free-space term as a separate contribution; fold into "LiDAR for shape" |
Kill/redirect criterion: if the mesh-level violation on static objects does not exceed the GT floor, the free-space
contribution is not supported by our data and we say so.

---------------------------------------------------------------------------------------------------------------------

## 7. Practical setup

- **New notebook** `testing/sam3d_mask_freespace_analysis.ipynb`, one section per step above; single frame first (a
  static parked-car scene in nuScenes-mini train split, the scenes already used for the probe: scene-0655 f3,
  scene-1094 f22), then batch.
- **New module** `autolabeling/src/autolabeling/utils/mesh_consistency.py` (silhouette/depth rasteriser, occluder logic,
  depth residuals, ray-mesh free-space test, GT floor, synthetic checks) with CPU tests in `autolabeling/tests/`, reusing
  Phase 0's ray caster (`contribution_ideas/phase0/scripts/experiment_a_batch.py`) and `utils/diagnostics.py` (LiDAR-based
  GT matching, min fraction 0.3).
- **Data:** inference must keep FULL meshes and the raw pose (rotation/translation/scale, shift/scale) -> a run with
  `sam3d_objects.mesh_points: 0` and the per-object pose stored; existing slim checkpoints are not enough for Steps 1-3.
  Perturbation experiments (2.3, 3.2) run SAM3D Objects directly in the notebook, same seed (42) as the pipeline.
- Rules from earlier sessions: nuScenes-mini TRAIN split only (8 scenes), LiDAR aggregation +-3 sweeps for the object
  cloud (free space uses the single sweep, see 4.1), run everything through `container/run_in_container.sh`, keep the
  user's notebook settings.
- Evidence format for the professor: (1) one table per step with medians + bootstrap CI + GT floor, (2) per-object
  figure panels (image+contours; BEV/side view with free voxels, LiDAR, mesh, GT box; violation heat), (3) response
  curves for the perturbation tests, (4) the information-ladder table.

## 8. Order of work

1. Data run with full meshes + pose on 2-3 static-car frames; projection sanity check (30 min of work, gate for all else).
2. Step 1 measurement (2.1) + Meta refinement reference (2.2) -> first figure/table.
3. Step 3 measurement with controls (4.1-4.3), because it decides whether the free-space story stands at all.
4. Step 1 causal test (2.3) and Step 2 (3.1, 3.2).
5. Step 4 ladder (5.2) once 1-3 are clear.
6. Batch over the train split, then write the evidence pack. Sam3d Body later reuses the same module.

Open decisions for the author: which frames form the pilot; whether to spend GPU on the Meta-refinement reference now;
tolerance d for free-space margins (decided by the synthetic check in 4.3).
