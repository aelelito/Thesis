# Novelty / contribution — distilled for the thesis draft

*Source: `contribution_ideas/contribution_plan.md` (the full, exploratory working document — 2300+
lines, written as an evolving discussion with corrections and audits). This file extracts only what
a reader needs: the problem, the proposed solution, the evidence already gathered, and the related
work. Written for the "one added page about ideas on the novelty/contribution" part of the thesis
draft — not for implementation.*

## 1. The current issue

A LiDAR sensor sees an object from one side only. A car 30 m away might return 40 points, all on its
rear bumper. The front half is never measured. Fit a box tightly to those points and it comes out
**too short** and **shifted toward the sensor** — the box centre lands on the visible surface, not
the object's true centre.

Every prior auto-labeling method patches this the same way: look up a class-average size and
stretch the box (MODEST, OYSTER, LISO, LiSe, UNION, CPD, VESPA, AnnofreeOD, CM3D, OVM3D-Det — all of
them). This requires a hand-curated or LLM-queried size table per class, which quietly breaks the
"open-vocabulary" claim these methods otherwise make. Two papers from this lab document the same
limitation directly: **CPD** measures that ~65% of objects on Waymo never get full LiDAR scan
coverage; **AutoBox**'s own appendix states that occluded objects produce undersized boxes with "no
special handling in the current implementation," and names class-size priors as the only proposed
fix. The most directly comparable published method, **LabelAny3D** (2026, also TRELLIS-based),
states the same gap as its own future work.

**This pipeline is different: SAM3D Objects *generates* a complete object instead of fitting a box
to visible points.** That is the structural advantage this thesis exploits — but an unconstrained
generative model imagines freely, which is exactly why the pipeline's current boxes come out the
wrong shape in a way no pointmap mode fixes (`notes/sam3d_objects_mode_decision.md` §5: boxes are
~13% too short and ~15% too wide on nuScenes, ~20–26% too wide on ECP, regardless of which pointmap
mode is used — 5–10x larger than any difference *between* modes).

**One-sentence framing:** everyone in this literature uses LiDAR *returns*; nobody uses LiDAR *free
space* — the volume a laser beam passed through and proved empty — and free space is the missing
constraint that would make a generative shape prior trustworthy on the majority of objects that are
never fully observed.

### Why the generated shape is currently unconstrained — two independent pieces of evidence

1. **SAM3D's own ablation.** The SAM3D Objects paper's own appendix (E.5) reports a human
   preference test: meshes generated *with* a pointmap vs. *without* one are preferred equally often
   (48% vs. 48%) — the pointmap affects *where* the object is placed, not *what shape* it takes.
2. **This thesis's own conditioning-input probe** (`notes/sam3d_objects_mode_decision.md` §3.4–3.5)
   independently found the same thing from a completely different angle: dropping the pointmap only
   changes metric placement/scale, and the model relies overwhelmingly on the *crop-scale* geometry
   for that placement — consistent with the pointmap being a *position* signal, not a *shape*
   constraint. Two independent measurements (the model authors' internal ablation, and this thesis's
   external probe) agree.

## 2. The proposed fix

**Use LiDAR free space, not just LiDAR returns, to constrain what SAM3D Objects generates.** A LiDAR
beam is a line: if it leaves the sensor and returns from something 50 m away, everything up to that
point is *certified empty*. This partitions space, per object, into three regions:

| Region | Meaning | Who decides shape there |
|---|---|---|
| Occupied | A return landed here | The measurement — mesh must pass through |
| Certified empty | A beam passed through and returned further away | The measurement — mesh must *not* enter |
| Unknown | Never reached by any beam (the object's own shadow) | The generative prior — free to imagine |

The prior is only allowed to invent geometry in the *Unknown* region; everywhere else the sensor
decides. This single idea generalizes to several concrete mechanisms (full detail and literature
positioning in `contribution_ideas/contribution_plan.md`):

- **Tier 1 — post-hoc alignment (low risk).** After SAM3D generates a mesh, optimise a 9-parameter
  transform (rotation, translation, anisotropic scale) to maximise agreement with three measurement
  terms: occupancy (the mesh surface should touch LiDAR returns, not swallow them), free space (the
  mesh must not occupy certified-empty voxels), and an occlusion-aware silhouette (the mesh must
  cover the visible mask, and must not spill past a *real* object boundary — but occlusion/truncation
  boundaries don't count, since the real object continues behind them). A drop-in box refiner,
  applicable to any pipeline's output (including VESPA's, see `evaluation/baseline_comparison.md`),
  not just this one.
- **Tier 2 — guidance inside generation (higher risk, higher novelty).** Rather than correct the
  mesh after the fact, steer SAM3D Objects' own diffusion process at each denoising step toward
  measurement-consistent shapes, using the same three terms as a gradient. Formally: sample from
  `p(shape | image, LiDAR) ∝ p_gen(shape | image) · p(LiDAR | shape)`, where the generator is frozen
  and unchanged, and only the *sampling* is steered. This class of technique (diffusion posterior
  sampling / training-free latent intervention) is already validated on this exact model family —
  **SpaceControl** (2025) does training-free geometric control of TRELLIS, the architecture SAM3D
  Objects is built on.
- **Nothing is trained.** Every component (SAM3, SAM3D, MoGe, CompletionFormer) stays frozen; Tier 1
  is a per-object optimisation at inference time, Tier 2 steers sampling without touching weights.
  This matters for the thesis's open-vocabulary, training-free framing: it transfers across sensors
  (ECP 64-beam ↔ nuScenes 32-beam) with no retraining.

## 3. Evidence already gathered (this is not speculative — it was measured)

A full-dataset investigation (`contribution_ideas/phase0_mesh_freespace/results/REPORT.md`, both
datasets, 7445 detections, two independent matching protocols for robustness) tested exactly the
go/no-go question this proposal depends on: **does the current pipeline's output actually touch
certified-free space more than a correct box would?** Results, honestly reported including the
complication:

- **Yes, confirmed and robust.** On nuScenes, the shipped OBB touches significantly more free space
  than its matched GT box (≈2 percentage points / 0.2–0.3 m³ median excess, p<0.0001, confirmed by
  two independent matching protocols at every one of 4 distance thresholds). This is the headline
  go/no-go result, and it replicates under the official nuScenes-devkit matching criterion, not just
  this project's own rule — directly pre-empting the "is this a matching artifact" objection.
- **About half of the violation is new information, not something a simpler fix already catches.**
  For every mesh surface point sitting in free space, checking whether an in-mask LiDAR return
  already implies that point is wrong gives a clean, matching-independent number: ~50% (0.49–0.51 on
  both datasets, both matching protocols). That is the direct, quantified answer to "doesn't the
  pipeline's own in-mask depth already cover this?" — no, roughly half the problem would survive that
  simpler fix.
- **The honest complication: on nuScenes, most of the excess is a *placement* effect, not a pure
  *shape* effect.** Re-centring the generated mesh onto the correct GT position/heading (keeping its
  own shape) removes most of the violation — the mesh's shape alone then touches free space *at or
  below* the true GT box's level for every well-populated category. This means Tier 1's
  position/scale correction (already partly informed by LiDAR via the pointmap mode) is doing more of
  the work than a pure shape constraint would need to — a useful, honest finding for scoping how much
  of the gain Tier 2 (shape-level guidance) can realistically add on top of Tier 1 (position/scale
  alignment). A close-range follow-up (stratifying by range) confirms this holds even for the most
  visually striking near-field failures — it sharpens rather than overturns the main finding.
  Below-ground reach (does the box sink below the real ground) is confirmed on ECP, not supported on
  nuScenes — dataset-split, not a universal claim.

**Bottom line for the thesis page:** the free-space constraint has real, measured signal (claims 1
and 2 above) — the go/no-go gate this proposal depends on passes — but the honest picture is that
position/scale correction (closer to Tier 1) currently explains more of the gap than pure shape
correction would (closer to what Tier 2 would add), which should shape how the "future work" framing
is scoped.

## 4. Related work (condensed; full citation list and justification table in `contribution_plan.md` §24, §26)

| Who | What they do | How this differs |
|---|---|---|
| **LabelAny3D** (2026) | Closest published relative — TRELLIS-based generative auto-labeling. States the LiDAR-conditioning gap as their own future work; finds size priors beat generation on low-visibility objects. | This thesis measures the same gap concretely (the free-space experiments above) and proposes LiDAR free space, not RGBD, as the conditioning signal; uses a continuous blend rather than their binary visibility switch. |
| **AutoBox** (this lab) | Same generative-auto-labeling family. Appendix documents undersized boxes under occlusion with no handling; fence-occluded objects require a heuristic mask-overlap filter that discards 6.8% of evaluations. | Free-space evidence handles both the undersized-box case and the fence-occlusion case as one mechanism rather than a separate filter. |
| **SDFLabel** (CVPR 2020), **Segment, Lift and Fit / SLF** (ECCV 2024) | Fit a category-specific learned shape prior to LiDAR + mask by optimisation — the closest Tier-1-style prior art. | Category-specific trained shape space (cars only) vs. this thesis's open-vocabulary generative prior; neither uses free space, only returns + silhouette. |
| **CPD** (CVPR 2024) | Measures ~65% of Waymo objects lack full scan coverage; proposes retrieval-based shape prototypes instead. | Different fix to the same measured problem — prototypes vs. a sensor-constrained generative prior. |
| **VESPA** (this lab) | DBSCAN clustering + LLM-sourced class sizes + L-shape fitting. The experimental baseline (`evaluation/baseline_comparison.md`). | No generative shape model; boxes are fit to visible points, not generated and sensor-corrected. |
| **SpaceControl** (2025) | Training-free, test-time geometric control of TRELLIS via latent intervention (for artist-supplied primitives, not sensors). | Proves the Tier 2 mechanism works on this exact model family; this thesis's novelty is the measurement being a *sensor* (free space + occupancy) in a *driving* context. |
| **GLENet** (IJCV 2023), **MEDL-U** | Learned, labelled-data-dependent per-object label uncertainty. | This thesis's (future) observability score is geometric and annotation-free — transfers with zero adaptation; not part of the core free-space claim above, listed as the natural next extension. |

## 5. What's proposed vs. what's measured — keep this distinction explicit in the writeup

- **Measured, with real data:** the problem exists (box-shape error independent of pointmap mode);
  the generative shape is currently placement-only, not shape-constrained (two independent pieces of
  evidence); the free-space violation is real, robust, and ~50% not already covered by a simpler fix;
  the violation is more placement- than shape-dominated on the dataset with the most evidence.
- **Proposed, not yet built:** Tier 1 (post-hoc 9-DoF alignment) and Tier 2 (guidance inside SAM3D's
  own generation). The full phased roadmap (Phase 0 of which is now complete) is in
  `contribution_ideas/contribution_plan.md` §28 if more detail is needed than fits on one page.
