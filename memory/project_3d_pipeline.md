---
name: 3D Object Dimension Estimation Pipeline
description: Overall pipeline goal, current state, blocked items, key file paths
type: project
---

Sensor-Constrained Generative Auto-Labeling thesis contribution. Goal: use LiDAR free-space
evidence (rays passing through a volume and returning from further away = that volume is empty)
to constrain the TRELLIS-based 3D mesh generation, removing the need for class-size priors.

**Why:** LiDAR free space is ignored by all auto-labeling literature. It is the missing constraint
that makes a generative shape prior trustworthy on poorly-observed objects.

**Current state:** Phase 0 notebook created at
`Contribution/notebooks/phase0_freespace.ipynb`. Not yet run on real data — waiting for
user to specify a specific ECP scene/frame path (ECP GT frames: scenes 10, 11, 15).

**Key existing results:**
- O3 + B1 + hull + agg(2) + filt: ECP 8-class mAP 0.1795
- O5 + B1 + hull + agg(2) + filt: ECP 8-class mAP 0.1866 (best)
- Hull anchoring = accidental early validation of the free-space thesis

**Key file paths:**
- Full plan: `Contribution/contribution_plan.md`
- Phase 0 notebook: `Contribution/notebooks/phase0_freespace.ipynb`
- Pipeline: `AutoLabeling/src/autolabeling/pipeline.py`
- SAM3D Objects model: `AutoLabeling/src/autolabeling/models/sam3d_objects.py`
- SS→SLAT intercept: `Models/SAM3D/sam-3d-objects/sam3d_objects/pipeline/inference_pipeline_pointmap.py` L428–473
- Test notebook: `Testing/autolabeling_pipeline.ipynb`
- Configs: `AutoLabeling/configs/{ecp,nuscenes}.yaml`

**Data roots:**
- ECP: `/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes` (v1.0-trainval, scenes 10/11/15 have GT)
- nuScenes mini: `/media/lleba/ECP_Nuscenes_01/nuScenes_mini` (v1.0-mini)

**Roadmap phases:**
- Phase 0 (current): Free-space map + BEV viz + Experiment A (overshoot vs undershoot)
- Phase 1: Observability score (7 components) + Experiment B
- Phase 2: Tier 1 — 9-DoF post-hoc alignment
- Phase 3: Regime C — ground-contact + anchored affine metric propagation
- Phase 4: Tier 2 — guidance inside SS generation
- Phase 5: consequences (FP rejection, multi-cam merge, CenterPoint weighted loss)

**Why:** Fix the "unconstrained generative prior inflates boxes at close range, has wrong depth at long range" problem — the key failure mode not addressed by any method in the literature.
**How to apply:** When the user asks about next steps, refer to §28 roadmap in contribution_plan.md.
