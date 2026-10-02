# Baseline comparison: VESPA vs. the clean pipeline (mode 11)

VESPA is the only baseline with real, already-computed numbers on the exact same evaluation set
(same datasets, same nuScenes-style mAP/NDS metric, same class groupings). It is also the only
*external* pipeline available (the method predates and is independent of the SAM3D-based pipeline
this thesis builds), so it is the fair comparison point for an Experiments/baselines section.

## What VESPA is

**VESPA** ("Towards un(Human)supervised Open-World Pointcloud Labeling for Autonomous Driving",
this lab's own prior work, arXiv:2507.20397) generates pseudo-labels with a fundamentally different
recipe: appearance-based clustering (UNION) or VLM/LLM-supervised fusion, DBSCAN-based point
denoising, **class sizes queried from an LLM** rather than measured, L-shape box fitting to the
LiDAR points, and multi-camera merging by shared points and border adjacency. No generative shape
model — boxes are fit directly to the (sparse) point cloud, inflated to a class-typical size where
the points don't cover the full object.

This is the same family of method the clean pipeline's whole design departs from: VESPA fits a box
to what LiDAR *returned*; the clean pipeline *generates* a complete object with SAM3D Objects and
only uses LiDAR to calibrate that generation. The comparison below is therefore a direct test of
whether that departure is worth it.

## Results — mAP (NDS omitted; see `notes/sam3d_objects_mode_decision.md` §2.3 for why)

**Front camera only:**

| | nuScenes 1c / 3c / 8c | ECP 1c / 3c / 8c |
|---|---|---|
| VESPA | .0130 / .0092 / .0075 | .284 / .202 / .102 |
| **Clean pipeline (mode 11)** | **.0336 / .0290 / .0283** | **.356 / .332 / .177** |

**All cameras:**

| | nuScenes 1c / 3c / 8c | ECP 1c / 3c / 8c |
|---|---|---|
| VESPA | .2527 / .2122 / .1176 | .4006 / .2803 / .1218 |
| **Clean pipeline (mode 11)** | **.3749 / .3276 / .2399** | **.5043 / .4687 / .2425** |

The clean pipeline leads on every number, every granularity, both datasets, both camera scopes —
roughly 2–4x on nuScenes front-camera, and a clear, consistent margin everywhere else. (Front-camera
and all-camera numbers are each internally comparable but not to each other — all-camera naturally
has more detections per scene from more viewpoints.)

## How to read this for the thesis

- This is evidence the **generative-shape-plus-sensor-calibration** approach is a real improvement
  over **fit-a-box-to-points-plus-class-size-prior**, not just a different way of doing the same
  thing.
- It is *not* yet evidence for the specific free-space contribution (`contribution_ideas/`) — that
  is a separate, still-open question about this pipeline's own remaining errors (see
  `contribution_ideas/novelty_brief_for_thesis.md`). This table is "does the new pipeline beat the
  old one," not "is the new pipeline's shape correct."
- Source data: `evaluation/old_eval_results/{nuscenes_mini,ecp}/vespa_vlm_train_{front_cam_only,all_cameras}/{1class,3class,8class}/metrics_summary.json`
  (VESPA) vs. `evaluation/eval_results/{nuscenes_mini,ecp}/{front_cam_mode_11,all_cam_mode_11}/{1class,3class,8class}/metrics_summary.json`
  (clean pipeline).
