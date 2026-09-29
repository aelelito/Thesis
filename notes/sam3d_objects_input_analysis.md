# SAM3D Objects: what it takes as input, what it relies on, and what we should feed it

Status: 2026-09-25. Written from the code (verified by reading it), from measurements on real data, and from the
per-object experiments in `testing/autolabeling_pipeline_clean.ipynb`. The batch sweep `cmp2` is queued/running; its
results go into section 7. Companion note: `clean_pipeline_overview.md` (pipeline design, per-sweep TerraSeg).

Legend: **[code]** read in the SAM3D source, **[measured]** measured on our data, **[inferred]** conclusion drawn from
results (not proven), **[open]** not yet known.

---------------------------------------------------------------------------------------------------------------------

## 1. What SAM3D Objects consumes  [code]

Files: `models/SAM3D/sam-3d-objects/checkpoints/hf/ss_generator.yaml`, `pipeline.yaml`,
`sam3d_objects/data/dataset/tdfy/{preprocessor,img_and_mask_transforms,pose_target}.py`,
`sam3d_objects/model/backbone/dit/embedder/{pointmap,embedder_fuser}.py`,
`sam3d_objects/pipeline/inference_pipeline_pointmap.py`, `notebook/inference.py`.

### 1.1 Six conditioning inputs (stage 1, the "sparse structure" generator)

| # | input | encoder |
|---|---|---|
| 1 | RGB image, **cropped** around the object | DINOv2 |
| 2 | RGB image, **full** | DINOv2 (same encoder) |
| 3 | object mask, cropped | a second DINOv2 (the mask is fed as an image) |
| 4 | object mask, full | that second DINOv2 |
| 5 | pointmap, cropped | `PointPatchEmbed` |
| 6 | pointmap, full | `PointPatchEmbed` |

The mask is also used outside the network: it selects which pixels define the *shift* (section 1.3).

### 1.2 How the pointmap is tokenised

- **Crop stream:** the mask bounding box scaled by `box_size_factor: 1.2`, padded to a square, **nearest-neighbour**
  resized to 256x256. **Full stream:** the whole image padded to a square, nearest-resized to 256x256 (about 6x
  downsampling for a 1600-px image).
- Each stream is cut into 8x8 windows -> 32x32 = 1024 tokens. Inside a window a one-layer transformer attends over the
  64 points; the CLS output is the token.
- **NaN pixels** (and padding) become a learned "invalid" token. The window attention mixes valid and invalid points.
- `remap_output: linear`: coordinates go through the linear projection unchanged (after normalisation).
- Because of the nearest-neighbour resize, a sparse pointmap loses most of its points.

### 1.3 Normalisation and pose decoding

`ObjectCentricSSI` (`use_scene_scale: true`), computed on the FULL-image pointmap before cropping:

- **shift** = median 3D point over the pixels inside the object mask (all three axes).
- **scale** ("ruler") = median, over ALL valid pixels of the whole map, of `max(|dx|, |dy|, |dz|)` measured from the shift.
- Both streams are divided by that scale, i.e. the network sees `(p - shift) / scale`.

The stage-1 generator outputs pose **in normalised units** (rotation, translation offset from the shift, size).
`ScaleShiftInvariant.to_instance_pose` then decodes `metres = predicted * scale + shift`
(`pipeline.pose_decoder(..., scene_scale, scene_shift)`). Stage 2 adds mesh detail and texture.

### 1.4 Other facts

- **Pointmap dropout in training:** `EmbedderFuser` zeroes the two pointmap streams **together** in 10% of training
  samples (`drop_modalities_weight`, `dropout_prob: 0.1`). The pointmap is therefore an optional hint. At inference,
  `force_drop_modalities` zeroes named inputs the same way (`pointmap` = crop stream, `rgb_pointmap` = full stream).
- **Intrinsics are unused:** `Inference.__call__` hard-codes `with_layout_postprocess=False`; intrinsics are only read
  in the layout post-optimisation. Our pointmaps are built from the dataset K, so the geometry is right regardless.
- With no pointmap given, the pipeline runs MoGe itself to get one.

---------------------------------------------------------------------------------------------------------------------

## 2. The three jobs of the pointmap  [code + inferred]

1. **Shift = the absolute position anchor.** Depth of the object = median depth of the map inside the mask (plus a
   predicted offset). An error in the in-mask depth passes into the object's position 1:1. **[code]**
2. **Scale = a unit of length, not a measurement of the object.** It only has to be computed identically in training
   and at inference and to grow linearly with the map. If the network reads size from the normalised tokens, the ruler
   **cancels** (`(size/ruler) * ruler`). The map is then the only source of metric truth, and a whole-map error by a
   factor k gives everything k times too far and too big. **[code + measured, see 4.4]**
3. **Conditioning tokens.** The network reads the tokens to decide size, offset and (to a lesser degree) rotation.
   **[measured: token-drop probe, section 5]**

---------------------------------------------------------------------------------------------------------------------

## 3. Our pointmap modes (`sam3d_objects.pointmap_mode`, 1-8)

All modes read the ground-free aggregated LiDAR (TerraSeg per sweep) for object fits; modes 6/7/8 use the full cloud
(ground included) for background fits.

| mode | name | what SAM3D sees |
|---|---|---|
| 1 | `sparse_lidar` | raw LiDAR only, NaN elsewhere |
| 2 | `moge_affine_local` | MoGe depth with ONE affine `Z = a*Zm + b` per object, fitted on that object's cleaned in-mask LiDAR (erode mask, HDBSCAN), applied to the whole image |
| 3 | `moge_affine_local_masked` | mode 2, NaN outside the mask (dropped: out-of-distribution) |
| 4 | `completionformer_full` | CompletionFormer dense depth (KITTI-trained, blurry edges) |
| 5 | `moge_affine_local_raw` | mode 2 on all in-mask points (no erosion/HDBSCAN) |
| 6 | `moge_affine_local_piecewise` | per object: own affine inside the mask, robust background affine (fitted around the object) outside, 2 px feather |
| 7 | `moge_affine_composite` | ONE map per frame: each mask (all classes) its own affine, nearer painted over farther; background = spatially varying affine (4x3 grid of robust fits, bilinearly blended) |
| 8 | `moge_affine_local_regslope` | mode 6, but an ill-conditioned object slope (<= 0, or off by more than 2x from the background slope) is replaced by the background slope; the offset is refitted so the depth at the mask's median MoGe value (hence the shift) is unchanged |

`Z = a*Zm + b` converts MoGe's relative depth `Zm` to metres. `b` sets where the object is; `a` sets how many metres one
unit of MoGe's depth difference is worth, i.e. the object's depth shape. `a` is unidentifiable when the LiDAR depths
inside the mask barely vary (a car seen side-on): the fit then returns a slope near 0 or negative (a "flat car").
This happened for about 7 of 17 object fits in our three test frames. **[measured]**

---------------------------------------------------------------------------------------------------------------------

## 4. Measurements on our inputs

### 4.1 Sparse LiDAR is almost "no pointmap"  [measured, scene-0103, 12 keyframes, 109 objects]
After SAM3D's crop and nearest resize only **1.0%** of the crop pixels and **6%** of the 8x8 windows contain a valid
point (about 94% invalid tokens). Dense MoGe-based maps: 100%.

### 4.2 The scene-scale statistic depends on what is in the map  [measured]
For sparse maps the ruler was 38.5 m with ground and 25.7 m without (ratio 0.71). For a dense NN-filled proxy 30.7 m.
The old-vs-new "ground removal helps" comparison of the front-camera runs (AP 0.019 -> 0.023) is confounded (body B1/B2,
aggregation, filters all changed) and the AP range of the front-camera runs is 0.018-0.029, so it shows nothing.

### 4.3 Front-camera-only AP is a poor discriminator
GT covers 360 degrees; one camera caps recall (all-cameras AP was 0.36). We compare **per object against GT** instead.

### 4.4 Ruler cancellation  [measured]
Objects whose ruler changed by 2-3x between modes changed size by only 2-12% and not proportionally
(scene 2 #6: ruler 3.44 vs 11.69 m, sizes within 5%; ECP #13: ruler 6.84 vs 13.26, sizes -7..-13%, opposite direction).
So the network reads normalised geometry rather than scaling its answer with the ruler.

### 4.5 Ground truth matching (how objects are scored)  `utils/diagnostics.py`
- Each detection is matched to a GT box **once, independent of the mode**: the GT box (3D) containing most of the
  single-sweep LiDAR points that fall on the detection's mask. Requires >= 3 points and >= 30% of the in-mask points inside
  the box; a runner-up with >= 50% of the points makes it ambiguous (dropped); if two detections claim the same GT box the
  larger mask keeps it.
- Why not 2D mask overlap: on 934 real objects the two methods agree for 270 of 292 objects matched by both; the median
  center error was 0.86 m for LiDAR matches vs 2.02 m for mask-overlap matches (many wrong claims). The LiDAR rule
  drops small far masks with few points, so results describe near and mid range (median GT range 26 m).
- Errors: center / range (signed, along the ray from ego) / lateral [m]; long, short, height ratio to GT; yaw [deg]
  (front/back flips ignored); `size_err` = mean |ratio - 1|.
- Known artefacts: objects cut by the image edge are wrong in every mode (1-2 m, height ratio 0.4-0.8); ECP bicycle
  sizes are dominated by the rider merge; GPU nondeterminism moves a repeated run by up to 3 mm.
- OBB fitting: every object's OBB is fitted on the FULL mesh right after inference and stored in the checkpoint; only then
  the mesh is reduced to 20,000 vertices (faces dropped; bicycles and motorcycles keep all vertices because their box is
  refitted with the rider). Slimming alone changed boxes by <= 1.4 cm size / 0.7 cm center (scene-0103).

---------------------------------------------------------------------------------------------------------------------

## 5. Token-drop probe (mode 2 re-run with parts of the pointmap tokens zeroed)  [measured]

Same objects, same seed, so differences come from the zeroed tokens. Numbers are medians over the matched objects;
`repeat` (nothing zeroed) differs by <= 3 mm.

| scene (n) | run | center shift | size change (long / height) | error vs GT: range | center |
|---|---|---|---|---|---|
| nuScenes 0655 f3 (6) | drop_both | 0.47 m | x1.22 / x1.30 | 0.32 -> 0.39 | 0.39 -> 0.52 |
| | drop_full | 0.05 m | x0.99 / x1.00 | | |
| | drop_crop | 0.57 m | x1.22 / x1.23 | | |
| nuScenes 1094 f22 (5) | drop_both | 0.32 m | x0.90 / x0.86 | 0.15 -> 0.40 | 0.29 -> 0.41 |
| | drop_full | 0.06 m | x0.96 / x0.96 | | |
| | drop_crop | 0.27 m | x1.06 / x1.04 | | |
| ECP scene 10 f560 (6) | drop_both | 0.69 m | x0.58 / x0.56 | 0.31 -> 0.96 | 0.53 -> 1.04 |
| | drop_full | 0.06 m | x0.85 / x0.88 | | |
| | drop_crop | 1.35 m (one object 59.9 m off) | x0.97 / x0.72 | | |

Reading: the **crop tokens are used** (range error worsens in all three scenes; sizes move 10-44%, direction differs by
scene, so it is not a fixed size prior). The **full-image tokens matter little** (center about 0.06 m everywhere; sizes 1-4%
on nuScenes, 12-15% on ECP). Caveat: training only ever dropped both streams together, so `drop_full` and `drop_crop`
alone are somewhat off-distribution.

---------------------------------------------------------------------------------------------------------------------

## 6. Per-object results, three frames  [measured; 14 objects, so indicative]

Frames: nuScenes scene-0655 f3 CAM_FRONT_RIGHT, scene-1094 f22 CAM_BACK_RIGHT, ECP scene 10 f560 CAM_FRONT.
Pooled over 14 matched objects (one false match removed); size over the 11 cars/trucks not cut by the image edge.

| | mode 1 | mode 2 | mode 6 |
|---|---|---|---|
| median center error [m] | 0.34 | 0.37 | 0.36 |
| median abs range error [m] | 0.22 | 0.26 | 0.22 |
| median size error (mean abs ratio-1) | 0.15 | 0.11 | 0.10 |
| long / short / height ratio (median) | 1.01 / 1.25 / 1.11 | 0.92 / 1.13 / 1.02 | 0.92 / 1.14 / 1.01 |
| median yaw error [deg] | 2.9 | 2.7 | 2.7 |

- Center error: modes 2 vs 1, 6 vs 1 each better on 7 of 14 (a coin flip); 6 vs 2: 7 better, 5 worse, 2 ties.
- Size error: dense (2 or 6) closer to GT than sparse on 8 of 11 objects; 6 vs 2: 5 vs 5, 1 tie.
- ECP frame with four modes (n = 4 matched): median center 0.52 (m2), 0.44 (m6), 0.48 (m7 composite), 0.46 (m8 regslope).
  Mode 7 painted 17 fits (all detections had one), all 12 background sections had their own fit (frame-wide background
  slope 4.89 vs about 6.4 locally). Mode 8 replaced slopes 0.70 -> 6.45, 0.57 -> 6.35, -1.11 -> 6.07 and left
  `shift_z` unchanged (as designed). No evidence that 7 or 8 beats 6 or 2. One signal: far small car #13 (3,143 px,
  24.7 m) had width/height ratios 1.26/1.26 in mode 2 (flat, wrong background) vs 1.10-1.15 / 1.14-1.17 in modes 6/7/8.
- Mode 8 caveat: it regularises toward the per-object LOCAL background slope; for bicycle #17 that local slope was
  itself off (2.89 vs about 6.4 elsewhere).

---------------------------------------------------------------------------------------------------------------------

## 7. Conclusions so far

### 7.1 Sweep `cmp2` on nuScenes-mini scene-0061 (39 keyframes, 118 matched objects; 2026-09-25)  [measured]
Matched objects: 58 car, 40 truck, 12 construction vehicle, 5 bicycle, 3 other; 19 are cut by the image edge and left out of
the medians below (n = 99; size n = 94). Paired differences are run minus mode 2 (negative = better), 95% bootstrap CI.

| median | m1 sparse | m2 | m6 | m7 composite | m8 regslope |
|---|---|---|---|---|---|
| center error [m] | 0.846 | 0.615 | 0.576 | 0.631 | 0.576 |
| abs range error [m] | 0.615 | 0.371 | 0.336 | 0.373 | 0.379 |
| size error (mean abs ratio-1) | 0.190 | 0.151 | 0.149 | 0.149 | 0.149 |
| long / short / height ratio | 0.95 / 1.26 / 1.06 | 0.87 / 1.16 / 0.96 | 0.88 / 1.16 / 0.96 | 0.87 / 1.13 / 0.95 | 0.88 / 1.14 / 0.95 |
| yaw error [deg] | 3.6 | 4.5 | 4.5 | 4.5 | 4.5 |

1. **Dense MoGe input beats sparse LiDAR for position**: m2 vs m1 center -0.162 m [-0.224, -0.070], better on 68/99, p<0.001
   (range -0.168 m). By range: < 15 m -0.196 (16/19), 15-30 m -0.011 (n.s.), > 30 m -0.200 (32/43). Size: not significant
   overall (-0.009 [-0.021, +0.002]). **This corrects the earlier 14-object statement that position is the same in all modes**
   (those objects were near and well covered, where sparse LiDAR is already a good anchor).
2. **m2, m6, m8 are equivalent overall** (center within 0.003 m, size within 0.002). m6 vs m2 range -0.011 [-0.039, -0.000]
   is borderline (p = 0.044, many tests).
3. **Composite (m7) does not help**: vs m2 center +0.011 [-0.008, +0.027]; vs m6 slightly worse (+0.013 m center, +0.018 m range,
   p < 0.01, tiny).
4. **Slope regularisation helps where it applies**: 31% of the object fits are ill-conditioned (36/118). On those, m8 has lower
   size error than m2 (-0.016 [-0.047, -0.002], 21/30, p = 0.043) and than m6 (-0.010, 24/30, p = 0.001); small masks
   (< 5k px): m8 vs m2 size -0.042 (14/17, p = 0.013). Effects are 1-4 points of size error, found in subgroup tests:
   suggestive, not conclusive. Position is unaffected.
5. **Truncated objects** are the worst in every mode (center 0.88 vs 0.61 m).
6. **Systematic footprint bias in every dense mode**: length ratio ~0.87, width ratio ~1.14 (predicted footprint squarer than
   GT). Not a pointmap effect; it is the largest remaining size error. **[open]** cause (SAM3D shape prior vs our PCA OBB fit).

### 7.2 Conclusions from the three single frames (earlier, n = 14)  [measured, superseded where noted]
- The **crop tokens are used, the full-image tokens barely** (probe, section 5); the global scale cancels (4.4).
  Still valid; the sweep (m2 = m6 = m7 background differences) agrees.
- ~~Position is the same for all modes~~ -> superseded by 7.1 item 1.

### 7.3 Current recommendation (SUPERSEDED by `sam3d_objects_mode_decision.md`, which includes ECP and pooled statistics: decision = mode 2)
Default **mode 2**; **mode 8** is a low-risk refinement (never significantly worse, better for ill-conditioned fits and small
masks). Drop mode 7 unless ECP says otherwise. Mode 1 only as a fast baseline. Next lever is the footprint bias, not the pointmap.
Wait for the ECP sweep before fixing the default. **[open]**: ECP results; whether the small-mask effect of m8 replicates.

### Running now: sweep `cmp2` (modes 1 2 6 7 8), prepare-once workflow
Everything before SAM3D Objects (SAM3, TerraSeg, PseudoLabeler, Body + B1/B2) does not depend on the pointmap mode, so one
"prepare" job per dataset computes it into `<dataset>/cmp2_shared`, and each mode job (2 h limit) only runs SAM3D Objects
against those cached results (`--checkpoint-dir`, checkpoints keep `objects__<mode name>__<hash>` per mode).
- nuScenes-mini scene-0061 (all 39 keyframes): prepare 12931839, modes 12931840 (m1), 41 (m2), 42 (m6), 43 (m7), 44 (m8).
- ECP (33 annotated keyframes): prepare 12931845, modes 12931846 (m1), 47 (m2), 48 (m6), 49 (m7), 50 (m8).
- Three modes at a time per dataset; mode jobs start only after the prepare job succeeded.
- Evaluate (`--baseline m2`; strata by range, mask size, image-edge truncation and ill-conditioned affines are printed):
  `bash container/run_in_container.sh python compare_modes.py --dataset nuscenes_mini --scenes scene-0061 --shared-run cmp2_shared --modes 1 2 6 7 8 --baseline m2`
  `bash container/run_in_container.sh python compare_modes.py --dataset ecp --shared-run cmp2_shared --modes 1 2 6 7 8 --baseline m2`
- Decision rules agreed for reading it: a mode is better only if the paired 95% bootstrap CI of the median
  difference (run - baseline) excludes 0 on center or size error; the composite map (7) is worth keeping only if it
  beats modes 2 and 6 in that sense, otherwise the background does not matter and mode 2 stays.
  Check the small-mask stratum specifically (section 6, signal on ECP #13).
- GPUs: V100 is unusable (TerraSeg needs FlashAttention = Ampere or newer); L40 (48 GB) probe pending (job 12931684).

ECP results (cmp2, 33 annotated keyframes): _pending_ (modes 1, 2, 6 were running, 7 and 8 queued).

---------------------------------------------------------------------------------------------------------------------

## 8. Reproduce

- Notebook (one frame, modes side by side, per-object GT table, token-drop probe, BEV per mode):
  `testing/autolabeling_pipeline_clean.ipynb` (set `POINTMAP_MODES_TO_RUN`, `RUN_TOKEN_PROBE`, `SELECT_IDS`).
- Batch: `bash container/submit_sweep.sh <dataset> <sweep> "<modes>" [--frame-end N]`, then `compare_modes.py`.
- Code: `autolabeling/src/autolabeling/models/sam3d_objects.py` (modes), `utils/diagnostics.py` (GT matching),
  `utils/compare.py` (statistics), tests in `autolabeling/tests/`.
