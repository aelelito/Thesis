# Mesh-level mask fit / free-space / ground evidence — full-dataset spec

Status: 2026-09-29 spec, **run completed 2026-10-01** (see §"Status" at the bottom and `results/REPORT.md`).
Relationship to `contribution_ideas/phase0/`: that work measured
**OBBs** from the **old** pipeline (O3+B1, all cameras) against certified free space, with a GT floor correction.
This work measures the **mesh** (and the pipeline's own OBB) from the **current clean pipeline**, adds a GT-floor
control at the shape level, and adds the "would in-mask LiDAR depth already have caught this" classification that
Phase 0 didn't have. It supersedes Phase 0's free-space claim specifically; Phase 0's resolution-sensitivity study
(§3 of `phase0_results.md`) and its distributional-vs-matched methodology comparison (§6) stand on their own and are
not repeated here. Pilot work (4 hand-picked frames) and the causal mask-perturbation test live in
`testing/sam3d_mask_freespace_analysis.ipynb` and `notes/sam3d_mask_freespace_plan.md` — this doc is the full-scale
follow-up, agreed with the user over several rounds of discussion (see that notebook's chat history for the
reasoning behind each choice below; only the conclusions are restated here).

## Open decisions (resolved; kept for the record)

1. **Pointmap mode** — at spec time, still being decided by the `cmp7` sweep. The **final decision for the
   pipeline is mode 11 (LDCM)**, made after a later all-camera comparison (see `notes/sam3d_objects_mode_decision.md`).
   This investigation deliberately runs on **mode 2**'s full meshes instead, captured before that final decision —
   the user's explicit call (proving mask/free-space fit doesn't depend on which pointmap mode generated the mesh;
   mode 2's full-mesh data was already available, mode 11's was not). Not an oversight.
2. **Cross-camera merge** — was undecided at spec time; did not block starting, since merge is a post-processing
   step on top of SAM3D's output, not part of inference.
3. **Rider merge for bicycles/motorcycles** — resolved as described in "Rider merge" below.

The queue-drain caution below no longer applies (the `cmp7` sweep finished and `mode_selection`/`cmp7_*` are final),
kept only as a historical note of why edits were held off at the time: configs are read fresh at task execution
time, not copied at submission, so editing them while an array task is still pending can corrupt it.

## Scope

- **Datasets:** nuScenes-mini train split (8 scenes, per `configs/scenes_nuscenes_mini_train.txt`), ECP annotated
  keyframes (33). Same scope the project already evaluates on everywhere else.
- **Cameras:** all cameras (3 for ECP, 6 for nuScenes). No cross-camera dedup applied for this measurement — a
  detection that looks worse from a side camera than the front camera is itself a finding, not noise to remove.
- **Meshes:** full, `mesh_points: 0`, faces kept. Existing mode-decision checkpoints CANNOT be reused for this —
  slimming happens before the checkpoint is written (`pipeline.py`: `slim_object_results` runs before `_save`), so
  every checkpoint on disk today has only 20k vertices and no faces. A fresh SAM3D Objects pass is required.
  SAM3, TerraSeg and PseudoLabeler ARE reusable via `--checkpoint-dir` pointing at an existing prepare-shared folder
  (`--prepare-only` in `submit_sweep.sh`'s pattern) — only SAM3D Objects needs to be rerun.
- **Classes:** everything SAM3D Objects fits (car, truck, bus, trailer, construction vehicle, bicycle, motorcycle).
  Pedestrians are out of scope for this thread (SAM3D Objects only, for now — SAM3D Body is a later thread).

## Rider merge — resolved

- **Mesh-level metrics** (mask IoU/recall, free-space of the mesh itself): bike-only mesh vs bike-only SAM3 mask.
  That's what SAM3D Objects actually saw and produced — correct comparison, no merge.
- **Pipeline-OBB-vs-GT metrics** (free-space and below-ground of "what would actually ship as a label"): the OBB
  IS rider-merged for bicycle/motorcycle, because the GT box for those classes includes the rider on both datasets.
  Comparing a bike-only OBB against a rider-inclusive GT box is apples-to-oranges regardless of anything SAM3D did.
  This needs SAM3D Body to run on the matched pedestrian mask, only for detections that get merged — cheap relative
  to the rest of the run.

## The five numbers (final — everything else discussed was cut for being redundant or not decisive)

Cut from earlier discussion, with the reason: depth-residual median (redundant with the free-space penetration
depth, which does the same job with a proper ray-cast instead of sparse in-mask points); the free-space *margin*
(checked against real data — 0.1 m vs 0.2 m vs 0.3 m changed the fraction by only ~5-15% relative — so we use the
plain free/occupied/unknown classification with no extra distance threshold); the fine-grained 5-way violation
classification (collapsed to one number, #5 below); the ray-shift existence-proof (a question for when the fix is
being built, not for proving the problem exists); the "at GT size" variant of the floor control (position+yaw is
enough to isolate shape); per-frame occupancy map visualizations (deferred to hand-picked scenes later, not batch).

Matching: the existing LiDAR-based rule (`utils/diagnostics.associate_gt_lidar`) — GT box containing the most
in-mask LiDAR returns, >= 3 points and >= 30% of them inside the box, duplicates resolved by mask size, ambiguous
matches (a close runner-up) dropped. Already validated against 2D mask-overlap matching on 934 objects elsewhere in
the project (0.86 m vs 2.02 m median error where they disagreed) — reused as-is, not re-derived.

1. **Mask fit.** `recall` for every matched object (occlusion-robust: mask pixels are guaranteed visible object, a
   correct mesh should always cover them). `IoU` reported only on the clean subset (not touching the image border,
   no other detected object's mask overlapping) — plus what fraction of objects qualify as clean, since that's
   informative on its own.
2. **Free-space touch**, mesh (surface + filled volume) and pipeline OBB, against the **GT-box floor**. Both a
   fraction (share of surface/volume classified free) and an absolute free volume in m^3 — Phase 0 already showed
   the fraction alone can hide the worst cases (its own §6.1 case study: a 56x-oversized box scored as *better than
   average* on the fraction metric).
3. **Below-ground reach**, mesh and OBB vs GT, using PseudoLabeler's fitted ground surface (already in the pipeline,
   `pl_model(xy)`) rather than the pilot notebook's ad hoc LiDAR-ring estimate. One median [m], one "% of objects
   sinking more than a few cm" number.
4. **Shape-isolated free-space (the floor control).** The mesh re-centred to the GT position and yaw, own shape kept,
   free-space fraction re-measured. If it drops to near the GT floor, the violation was placement (already
   LiDAR-driven, already accepted by the professor). If it doesn't, the residual is coming from the generated shape
   itself — this is the number that argues LiDAR should guide shape, not just position.
5. **"Not implied by in-mask depth."** For the mesh's violating surface: the percentage that a LiDAR return *inside
   the mask* does NOT already imply (i.e., no in-mask return lying further back along the same ray). This is the
   direct, quantified answer to "SAM3D already has the in-mask depth — why would free space add anything?"

## What gets built once the mode is final

- New config copies (NOT edits to the shared ones): `configs/nuscenes_meshfreespace.yaml`,
  `configs/ecp_meshfreespace.yaml` — `mesh_points: 0`, all cameras uncommented, final `pointmap_mode` filled in.
- `scripts/mesh_freespace_batch.py`, reusing `autolabeling/src/autolabeling/{pilot.py, utils/freespace.py,
  utils/mesh_mask.py, utils/controls.py}` almost unchanged — one CSV row per matched object with the 5 numbers
  above. Needs the same frame-index <-> real-frame reconstruction `run_pipeline.py` / `merge_submissions.py` use
  (checkpoints are indexed by position in the `collect_frames_multi_cam` list, not by sample token — must rebuild
  the identical frame list, same scene order and frame range, to know which checkpoint is which frame).
- A submit script mirroring `submit_sweep.sh`'s prepare-once pattern, writing to a fresh output directory (not
  `mode_selection/`, not `cmp7_*`) so it can never collide with the mode-decision runs.
- The notebook here is for spot-checking whatever the batch script flags as extreme, not for the aggregate numbers.

## Reporting

Per dataset (and category where n allows it): the five numbers above, paired against the GT floor with a bootstrap
CI and sign test where a paired comparison applies (mirrors `notes/sam3d_objects_mode_decision.md`'s standard, not
a plain mean/percentage). A handful of hand-picked frames get the full visual treatment (mask overlay, occupancy
map, side view) afterward, chosen from whatever this batch run flags as clear and convincing.

## Status (2026-10-01): full run + both matching protocols done, see results/REPORT.md
LiDAR-containment matching (`results/summary.md`) and nuScenes-devkit-style center-distance matching at all 4
official thresholds (`results/summary_nuscenes_matching.md`, built by `scripts/rematch.py` from the cheap
`scripts`/`autolabeling.batch_boxes` box-only extraction -- no mesh re-render needed) now agree on every claim.
`results/REPORT.md` is the consolidated write-up. Bug fixed along the way: `rematch.py`'s ECP category-mapping
fallback was gated on a condition ("does any category contain an underscore") that's false for ECP's category
names, silently dropping all ECP GT boxes -- fixed, verified ECP rows now populate correctly across all thresholds.
Outstanding: claim 4 (floor control) under the nuScenes-style match (needs reopening meshes for the matched subset,
cheap but not yet built); the PseudoLabeler ground-baseline bias check noted in claim 3.
