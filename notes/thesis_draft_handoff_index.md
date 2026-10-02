# Thesis draft handoff — index of what to give Claude web, and why

**Goal:** an 8-page thesis draft (excl. references): Abstract, Introduction (contributions),
Related Work, Methodology (current clean pipeline), Experiments (setup/datasets/eval/baselines/
ablations), Conclusions, and one added page on novelty/contribution ideas (current issues +
proposed fixes + supporting evidence).

This file is the map. Hand it to Claude web first, then the files below in roughly this order.
Everything listed is either an existing, already-accurate reference doc, or one of two files
written specifically for this handoff (marked **NEW**).

## 1. Methodology — the current clean pipeline

| File | What it contains |
|---|---|
| `notes/clean_pipeline_overview.md` | Stage-by-stage architecture of the production pipeline (SAM3 → TerraSeg/PseudoLabeler → SAM3D Body/Objects → OBB). The primary methodology source. States the final pointmap mode (11) and points to the decision doc. |
| `notes/lidar_inmask_filtering.md` | How raw LiDAR sweeps become the clean per-object point set fed to SAM3D Objects/Body (aggregation, HDBSCAN-based cleaning). Supporting mechanism detail for the methodology section. |
| `notes/sam3d_objects_mode_decision.md` §1–2 | What the pointmap is, why it matters, and the full list of candidate pointmap modes (1–11) with a one-line description each — useful for describing *why* this is a methodological choice, not an implementation detail. |

## 2. Experiments — setup, datasets, evaluation, baselines, ablations

| File | What it contains |
|---|---|
| `notes/sam3d_objects_mode_decision.md` (whole doc) | **The core ablation study.** What was tested (11 pointmap modes; a 6-input-channel conditioning probe), the method (full nuScenes-mini train split + full ECP annotated set, both datasets, nuScenes-style mAP at 3 class granularities), the results tables, and the final decision (mode 11) with reasoning. Already structured as what-we-tested → results → conclusions. |
| `evaluation/README_evaluation.md` | Formal definition of the evaluation protocol: matching rule, mAP computation, metric definitions. Use for the "evaluation details" subsection. |
| `evaluation/baseline_comparison.md` **(NEW)** | VESPA (this lab's prior, non-generative pipeline) vs. the final pipeline (mode 11), both datasets, front-camera and all-camera scope. The final pipeline leads on every number. This is your "baselines" row. |
| `contribution_ideas/phase0_mesh_freespace/results/REPORT.md` | Completed mesh/mask/free-space experimental results (7445 detections, both datasets, two independent matching protocols). Could be cited briefly here as additional diagnostic evidence, and is also the evidence base for the novelty page (§3 below) — don't duplicate, just note it's used in both places. |
| `contribution_ideas/phase0/results/phase0_results.md` | Earlier, independently-still-valid free-space results (resolution sensitivity, matching-methodology comparison) that the above report explicitly does not repeat. |

## 3. Introduction (contributions) + the added novelty/contribution page

| File | What it contains |
|---|---|
| `contribution_ideas/novelty_brief_for_thesis.md` **(NEW)** | **Read this one, not the 2300-line original.** Distilled specifically for this page: the current issue (generative shape is placement-only, not shape-constrained — with two independent pieces of evidence), the proposed fix (free-space-constrained generation, Tier 1 post-hoc alignment + Tier 2 guidance-inside-generation), the evidence already gathered (the completed free-space experiments, reported honestly including the "mostly placement, not pure shape" complication), and a condensed related-work table. Explicitly separates what's *measured* from what's *proposed*. |
| `contribution_ideas/sam3d_body_depth_conditioning.md` | A smaller, secondary future-work idea (giving SAM3D Body a real depth-conditioning input) — mention briefly if there's room, it's explicitly sequenced as later-stage work. |
| `contribution_ideas/contribution_plan.md` | The full original planning document (2300+ lines) — only hand this over if Claude web needs more depth than the brief provides for a specific claim. Not needed for a first draft. |

## 4. Related Work

Drawn from the condensed table in `contribution_ideas/novelty_brief_for_thesis.md` §4 — covers
LabelAny3D, AutoBox, SDFLabel/SLF, CPD, VESPA, SpaceControl, GLENet/MEDL-U, with links available in
`contribution_ideas/contribution_plan.md` §24 if full citations are needed.

## 5. Abstract / Conclusions

No dedicated source — synthesize from the mode-decision doc's conclusions (methodology worked,
final mode chosen, what it beat) and the novelty brief's one-sentence framing (free space as the
unused constraint).

## What's deliberately left out (and why)

- `notes/implementation_notes.md`, `notes/lidar_integration_plan.md`, `notes/pipeline_progress_summary.md`,
  `notes/sam3d_objects_input_analysis.md`, `notes/sam3d_mask_freespace_ideas_backlog.md` — all
  deleted (`git rm`, recoverable from history if ever needed). They described an older pipeline
  version, an earlier/superseded analysis, or a brainstorming scratchpad whose conclusions are
  already captured in the files above.
- `notes/sam3d_mask_freespace_plan.md`, `contribution_ideas/phase0_mesh_freespace/PLAN.md` — the
  *spec* for the free-space experiments. Useful if Claude web needs to understand *how* the
  REPORT.md numbers were produced, but the results doc alone should be enough for the thesis text.
- `notes/project_regime_c_memory_idea.md` — a deferred, not-yet-pursued idea; skip unless the draft
  specifically needs it.
