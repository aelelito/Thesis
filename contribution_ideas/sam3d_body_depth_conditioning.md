# SAM3D Body — Depth-Conditioning Contribution Idea

*Working note. Separate from `contribution_plan.md`, which is scoped to SAM3D Objects and the
free-space constraint. This is a distinct model, distinct architecture, distinct fix. Full
reasoning for why B1/B2 exist as post-hoc corrections today is in
`notes/clean_pipeline_overview.md` ("Why B1 alone isn't enough") — read that first, this note
only covers the idea for removing the need for them.*

## The idea

SAM3D Body (CLIFF/HMR-style regressor) has no depth-conditioning input — depth (`tz`) is *derived*
from the predicted weak-perspective scale, not something the model is given. This is the source of
the classic monocular height–depth ambiguity: a taller mesh at a farther depth and a shorter mesh
at a closer depth project to the same 2D bbox, so the model's own regressed body height and its
derived depth are coupled, and either can be wrong while still looking self-consistent in 2D.

B1 (LiDAR-corrected depth) and B2 (ground-anchoring) exist today as two independent post-hoc
patches for exactly this: B1 fixes the depth-scale ambiguity via a rigid translation (no
rescaling); B2 separately snaps the feet to the local ground surface to absorb the residual
height/shape error B1's translation-only fix can't touch.

**The idea**: give SAM3D Body a real metric-depth conditioning input, analogous to SAM3D Objects'
pointmap hook (`PointPatchEmbed` feeding the SS generator's cross-attention). If depth is a genuine
training-time conditioning signal rather than a free variable the model gets to choose, the model
can no longer trade height against depth to satisfy the 2D reprojection loss — the *only* thing
left to explain the observed silhouette size is the mesh's actual proportions. That would force the
model to learn correct height/shape *jointly* with correct depth, fixing both B1's and B2's jobs at
the source rather than patching them after the fact.

## Why this is a real project, not a quick change

- SAM3D Body ships with no such hook. Unlike SAM3D Objects (trained with pointmap dropout
  specifically so it can be conditioned flexibly at inference time with no LiDAR, sparse LiDAR, or
  dense completed depth), SAM3D Body's `CameraEncoder` was never built or trained to accept a depth
  prior as an additional input token.
- Adding the hook means modifying the `CameraEncoder` architecture and then fine-tuning — needs
  paired (image, metric depth, ground-truth body) training data, not just an inference-time flag
  flip like SAM3D Objects' `pointmap=` argument.
- Even if it works, some residual shape/pose regression noise likely remains (the network's shape
  head has finite capacity/data), so a lightweight ground-snap safety net (a much smaller-scope
  version of B2) might still earn a role — just correcting a far smaller residual than it does today.

## Before committing to this

Worth quantifying how much of the current error B1 vs. B2 each actually remove, since that's the
size of the prize this idea is chasing. The clean notebook
(`testing/autolabeling_pipeline_clean.ipynb`) already has `USE_B1`/`USE_B2` as independent toggles
for exactly this — run pedestrian AP/ATE with B1-only, B2-only, both, and neither before deciding
whether this is worth the retraining investment.

## Sequencing

After the clean pipeline is validated and after the SAM3D Objects `layout_post_optimization` /
mesh-mask reprojection investigation (see `contribution_plan.md` §10, §30) — this is explicitly a
later-stage contribution, not blocking either of those.
