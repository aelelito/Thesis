---
name: Regime C — accumulating cross-scene size/relation memory (deferred idea)
description: Optional future extension to §13 Regime C — a persistent, growing size/relation lookup built across frames/scenes during inference, kept distinct from C4
type: project
---

**Idea (user's, 2026-08-30):** instead of only intra-scene same-class size consistency (§13 C4),
maintain a memory that accumulates reliable size/shape/class evidence across frames and scenes as
inference progresses, growing into an increasingly trustworthy lookup table for Regime C objects.

**Why kept separate from C4, not merged into it:** C4 is deliberately scoped as per-scene, per-frame
evidence so the thesis can claim "no external size priors — sizes are derived from data observed in
the same scene" (see [[project_3d_pipeline]], §13). A cross-scene accumulating table is
architecturally an *online-learned external prior* — same category as VESPA/OVM3D-Det's LLM-sourced
tables, just self-supervised instead of hand-curated. Folding it into C4 would quietly weaken the "no
external priors" claim into "no hand-curated priors," which is a different, weaker claim needing its
own evidence (e.g. self-learned prior vs. LLM-sourced prior comparison).

**Cross-frame (same scene, same tracked instance) risk:** the plan's §25.2 sweep-accumulation ablation
already found moving objects smear when points/evidence are aggregated across frames (car AP 0.500 →
0.287 with aggregation, ECP). A mesh or size estimate accumulated across frames of a moving
pedestrian/vehicle would hit the same failure mode — blending a mid-stride pose is worse than one
clean frame. If implemented, should be scoped to static objects, or to a scalar size estimate rather
than the full mesh, to dodge this.

**Also risks crowding into the offboard/temporal-refinement literature** the plan explicitly steers
around (3DAL, Auto4D, CTRL, DetZero, MPPNet — "the crowded area you should avoid"). The orthogonality
argument used elsewhere in the plan — fixing single-frame amodal extent that track refinement cannot
solve when the object is never seen complete — breaks if this method itself starts refining across a
track.

**Verdict:** legitimate optional experiment/ablation for later, not part of the core §13 cascade as
currently scoped. If implemented, frame it explicitly as a "self-learned online prior vs. no prior"
ablation, separate from the C4 "no external priors" claim — don't let it quietly absorb C4's framing.
