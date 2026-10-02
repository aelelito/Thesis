# Simple plan: does the SAM3D Objects mesh enter LiDAR-certified free space?

Goal: first evidence, on a handful of hand-picked objects. Everything else is in `sam3d_mask_freespace_ideas_backlog.md`.

## Step 0 - pick examples (user)
Static (parked) objects only, so camera/LiDAR timing cannot fake a result:
1. clearly visible car, no occlusion
2. partly occluded car
3. bicycle (parked ideally; box includes rider merge)
4. car very close to the camera
Note scene, frame, camera, object id per example.

## Step 1 - run the pipeline as is (new notebook)
Run the clean pipeline on those frames, keep the FULL mesh (`mesh_points: 0`, faces kept) and the mask per selected object.

## Step 2 - look at the mesh vs the mask
Project the mesh into the image, draw its outline over the mask. One picture per object. Report IoU and how much of the
mask is uncovered / how much of the mesh lies outside it.

## Step 3 - free-space check
- LiDAR: the single sweep closest to the camera time, ego-motion compensated to the camera timestamp.
- Free space by ray casting (reuse the Phase 0 ray caster). Ignore free space within ~0.1-0.2 m in front of a return (noise).
- Measure for each object: share of mesh surface in free space, and how deep (m). Do it for the mesh, its OBB, AND the GT
  box of the same object. The GT box number is the baseline: only the excess over it counts.
- Pictures: BEV and side view with free space, LiDAR points, mesh, GT box.

## Step 4 - decide together
Read the results, then choose what to do next (e.g. mask perturbation test, depth residuals, post-hoc fix).

## Status (2026-09-25)
Built, not yet run on real SAM3D output: `testing/sam3d_mask_freespace_analysis.ipynb` (sections: load 5 frames, run pipeline, GT match, occupancy
maps, mask fit, depth residuals, free space of mesh / OBB / GT box, per-object pictures, optional mask perturbation, CSV export).
Code: `autolabeling/src/autolabeling/{pilot.py, utils/freespace.py, utils/freespace_viz.py, utils/mesh_mask.py}`; tests `tests/test_freespace.py`
(12 checks) and `tests/dry_run_freespace_notebook.py` (all analysis cells on CPU with a fake pipeline).
Also added `SAM3DObjectsModel.mask_hook` (analysis only; default off, production unchanged).
Measured on real data while building: GT boxes of near vehicles (7-18 m) already hold 7-18% (about 1-3 m3) certified-free volume, so the GT-box baseline is mandatory.
Known limits: ECP has no annotation velocity (motion = unknown); pedestrians are not segmented in this run (occluders = other vehicles only);
free space uses a fixed-step ray march (a beam can clip a voxel corner); filled mesh volume is about half a voxel per side too big.

## Status update (2026-09-25, after the ECP + nuScenes pilot frames)
Findings so far (ecp10_f560, nusc3_kf3): the mask sets mesh scale and lateral position (mask-bbox perturbation: size follows the bbox almost exactly, depth anchor frozen, same on both
datasets); on a clear car the mesh fits the mask (IoU 0.85) but not for cut-off / heavily occluded objects; depth agrees with LiDAR to about 0.1-0.3 m; free-space share of the mesh surface
6-17% (only car #1 of ECP and the nuScenes cars are observed), volume-wise no excess over a GT box; the free-space share tracks mask-induced size (dose-response), so it works as an oversize detector.
NOT established: that the mesh "obviously" overshoots into free space (glass / underbody may produce a floor). Controls added (utils/controls.py, notebook 6.4 / 7.0): GT-placed floor,
shift along the LiDAR ray, and the split of violating points by whether the in-mask depth would have seen them (the professor's "what does free space add" question).
Next: run all four frames, paste `summary_all_frames` and figures, then decide.

## Full-scale follow-up (2026-09-29, run completed 2026-10-01)
The pilot above (4 hand-picked frames) is done. The full-dataset, all-cameras version is specced in
`contribution_ideas/phase0_mesh_freespace/PLAN.md` — five final metrics (mask recall/IoU, free-space touch of mesh
and OBB vs GT floor, below-ground reach vs GT via PseudoLabeler, the GT-floor shape-isolation control, and the
"not implied by in-mask depth" share) — **and has now been run**; see `contribution_ideas/phase0_mesh_freespace/results/REPORT.md`.
It used mode 2's full meshes (captured before the mode decision was finalized; the pipeline's final mode is 11 —
see `notes/sam3d_objects_mode_decision.md` — but this investigation is about mask/free-space fit in general, not
mode-specific, so mode 2's already-available full meshes were used deliberately rather than redone for mode 11).
Rider-merge: resolved (bike-only for mesh metrics, rider-merged for the pipeline-OBB-vs-GT metric only).
