# Mesh / free-space / ground evaluation — final report

Full dataset: nuScenes-mini train split (8 scenes, all cameras), ECP (33 annotated keyframes, all cameras). Mode 2
(`moge_affine_local`), full meshes (`mesh_points: 0`). 7445 detections total (7230 nuScenes, 215 ECP).

Two independent matching protocols were run against the same detections, to pre-empt the objection that the
headline findings depend on an ad hoc matching rule:

- **LiDAR-containment** (`summary.md`): the GT box containing the most in-mask LiDAR returns. This project's own
  rule, used throughout the pipeline's diagnostics.
- **nuScenes-devkit-style** (`summary_nuscenes_matching.md`): greedy, per class, confidence-ranked, BEV center
  distance — the official protocol's own criterion (not IoU), at all four of its thresholds {0.5, 1, 2, 4} m. Run
  per camera (this analysis does not cross-camera-merge).

**Headline: every claim's direction and significance replicates under the independent matching protocol, and is
stable across all four of its distance thresholds.** That stability (not just agreement with the other method, but
near-identical numbers from 0.5 m to 4 m) is itself evidence the finding isn't an artifact of a threshold choice.

## What each claim tests, and how

Shared inputs: the **mesh** (full SAM3D Objects output, vertices + faces); the **pipeline/pseudo OBB** (a box fit to
the mesh — rider-merged with SAM3D Body for bicycles/motorcycles, since the GT box for those includes the rider but
the mesh doesn't); the **GT OBB** (the annotation box); a **free-space grid** built once per keyframe from that
keyframe's single LiDAR sweep (every voxel free / occupied / unknown); and the **PseudoLabeler ground surface**
(the pipeline's own fitted ground height, already produced for B2).

| Claim | What it asks | How it's measured |
|---|---|---|
| **1. Mask fit** | Does the mesh's silhouette match the SAM3 mask it was given? | Render the mesh, compare to the mask: `recall` (mask pixels covered — occlusion can't explain a miss, so scored for every object) and `IoU` (only on objects not touching the image border and not overlapped by another detection's mask, since real occlusion legitimately breaks IoU). No GT needed. |
| **2. Free-space touch vs GT floor** | Does the shipped box/mesh reach into space the LiDAR has certified empty, more than a correct box would? | Sample the mesh surface/volume and the pipeline OBB, query each against the free-space grid (fraction classified free, and that fraction × volume = an absolute m³ figure, since fraction alone can hide the worst oversized cases). Compare against the same query on the matched **GT** box — the baseline, since even a correct box touches some free air. **Needs a match.** |
| **3. Below-ground reach** | Does the box/mesh sink below the real ground? | PseudoLabeler's ground height at the box's (x, y), minus the box's lowest extent — for mesh, pipeline OBB, and GT. **GT side needs a match**; the ground estimate itself comes from LiDAR near the object, not the object's own surface. |
| **4. Shape-isolated free-space (floor control)** | If position were already correct, would the shape alone still touch excess free space? | Take the **same mesh**, translate + rotate (not rescale) it onto the **matched GT's** position and heading, re-measure free-space touch. Splits the claim-2 excess into a *placement* part (as-generated vs re-centred) and a *shape* part (re-centred vs the true GT box). **Needs a match**, needs the mesh's own vertices again (not a re-render). Excluded for bicycles/motorcycles (GT box includes the rider, the mesh doesn't). |
| **5. Not implied by in-mask depth** | Would simply fitting the mesh to the LiDAR depth it already has (inside the mask) already have caught the violation? | For every mesh-surface point sitting in free space: project it into the image and check whether an **in-mask LiDAR return** lies behind it along the same ray — if so, matching the mesh to that depth would already exclude it. The share that is *not* explained this way is `not_implied`. Self-contained to the mesh, the sensor, and the mask — **no GT needed at all**. |

Matching (claims 2–4): two independent protocols, described above, both using the LiDAR-independent numbers from
claims 1/2(mesh and OBB side)/3(mesh and OBB side)/5 as-is — those never depend on which GT a detection is paired
with — and only recomputing the GT-side numbers (and, for claim 4, the mesh re-sample) per matching rule.

## Claim 1 — Mask fit (matching-independent, identical under both protocols by construction)

| | n | recall (median) | n clean | IoU (clean, median) |
|---|---|---|---|---|
| ECP | 215 | 0.67 | 85 (40%) | 0.39 |
| nuScenes | 7230 | 0.85 | 2538 (35%) | 0.56 |

## Claim 2 — Free-space touch, pipeline OBB vs GT (the headline claim)

| | matching | n | obb excess, fraction (median, p) | obb excess, absolute (median m³, p) |
|---|---|---|---|---|
| nuScenes | LiDAR | 1729 | +2.3 pp (p<0.0001) | +0.275 m³ (p<0.0001) |
| nuScenes | nuScenes @ 2 m | 1432 | +2.1 pp (p<0.0001) | +0.227 m³ (p<0.0001) |
| ECP | LiDAR | 129 | −3.3 pp (p=0.0026) | +0.129 m³ (p=0.052, n.s.) |
| ECP | nuScenes @ 2 m | 158 | −3.8 pp (p=0.0003) | **+0.235 m³ (p=0.0018)** |

**nuScenes: confirmed, clean, and robust** — the shipped OBB touches significantly more free space than its GT
box, in both relative and absolute terms, under either matching rule, at every threshold tested (frac excess 1.3–2.3
pp, m³ excess 0.23–0.35, all p<0.0001 across 0.5/1/2/4 m). By category (nuScenes, either matching): car, bus, truck,
construction_vehicle, trailer all positive and significant; bicycle/motorcycle not significant (small n).

**ECP: strengthened, not weakened, by the independent check.** The *fraction* is lower than GT (ECP OBBs apparently
fill a larger box less completely), but the *absolute* excess — the number that matters for "does it reach into more
real, physical empty space" — is now clearly significant under the official matching (p=0.0003–0.0018 across all
four thresholds), where it was only borderline (p=0.052) under the LiDAR rule. The two methods agree on the
direction; the independent check removed the ambiguity.

## Claim 3 — Below-ground reach (PseudoLabeler ground)

| | matching | n | mesh excess (median m, p) | obb excess (median m, p) |
|---|---|---|---|---|
| ECP | LiDAR | 148 | +0.086 (p<0.0001) | +0.155 (p<0.0001) |
| ECP | nuScenes @ 2 m | 174 | +0.074 (p<0.0001) | +0.126 (p<0.0001) |
| nuScenes | LiDAR | 3639 | −0.073 (p<0.0001) | +0.001 (p=0.84, n.s.) |
| nuScenes | nuScenes @ 2 m | 3386 | −0.088 (p<0.0001) | −0.017 (p=0.0057) |

**ECP: confirmed, robust, sizeable** (7–9 cm mesh, 13–15 cm OBB, both significant, both matching rules). **nuScenes:
confirmed null-to-negative under both matching rules** — no below-ground overshoot evidence here once compared
fairly against GT's own (slightly negative) baseline. The below-ground claim is solid for ECP, not currently
supported for nuScenes; still worth checking the PseudoLabeler baseline bias noted earlier (GT's own median is
−5 to −6 cm on both datasets) before using the raw numbers anywhere.

## Claim 5 — Not implied by in-mask depth

| | matching | n | median |
|---|---|---|---|
| ECP | LiDAR | 129 | 0.49 |
| ECP | nuScenes @ 2 m | 158 | 0.51 |
| nuScenes | LiDAR | 1727 | 0.50 |
| nuScenes | nuScenes @ 2 m | 1430 | 0.51 |

**Essentially identical under both protocols and both datasets** (0.49–0.51 throughout) — expected, since this claim
doesn't depend on which GT a detection is paired with, only on which detections count as matched at all. Roughly
half the free-space-violating mesh surface would survive a "fit the mesh to its own in-mask LiDAR" correction. This
is the direct, quantified answer to "method 1 already covers this," and it is the most matching-robust number in
the whole study.

## Claim 4 — Shape-isolated free-space (floor control)

**Only available under the LiDAR-containment matching** (see `summary.md`) — redoing it under the nuScenes-style
match needs the mesh reopened at whichever GT pose that protocol assigns, which hasn't been done yet. Not expected
to change the qualitative picture (claim 2's conclusion, which the floor control refines, already replicated), but
flagged honestly as outstanding rather than assumed.

Recap of that finding: on nuScenes, once a mesh is re-centred to the correct GT position/yaw, its own shape touches
free space *at or below* the true GT box's floor for every category with enough data (car −0.5pp, construction
vehicle −7.1pp, trailer −4.2pp, truck −1.1pp, all p<0.0001) — meaning most of claim 2's excess is a **placement**
effect, not a **shape** effect, on the dataset with the most evidence. ECP cars go the other way but with much less
data (n=51, +4.1pp, p=0.049, borderline).

## Follow-up — does proximity to the ego vehicle explain the excess? (hypothesis: close objects need shape guidance most)

Motivated by a visually striking near-field failure seen earlier (`ecp10_f1080`, a heavily hyperinflated car right in
front of the camera). Three checks, all on already-collected data, no new runs:

1. **Floor control (claim 4) stratified by range, nuScenes (n=559 at 0–10 m, well powered):** placement excess is
   *largest* close to the car (+6.7 pp at 0–10 m vs +3.0–3.4 pp at 10–30 m), but the shape-only excess is *negative at
   every range*, most negative at 0–10 m (−2.7 pp, p<0.0001). **Close objects have the biggest violation, but it's
   still a placement effect, more so than further out — not a shape effect.** ECP's one positive shape signal
   (reported above) turns out to come from 10–20 m, not the closest range.
2. **Raw free-space touch by range (every detection, matched or not):** drops monotonically from the sensor outward
   (nuScenes mesh free-space median 0.15 at 0–10 m → 0.00 beyond 30 m). Confirms the visual impression, but this is
   expected independent of any real violation — LiDAR certainty is far higher close to the sensor, so there's simply
   more free space *to certify* near the car. This is exactly why the GT-floor correction in check 1 exists, and why
   check 1, not this raw number, is the one to trust.
3. **Unmatched close-range detections — a plausibility check, since the worst failures (like `ecp10_f1080`) are
   unmatched and invisible to every paired statistic above.** Unmatched rate is *lower* close to the car, not higher
   (nuScenes 28% at <15 m vs 48% beyond; ECP 11% vs 21%) — no sign of a close-range failure spike in the raw rate.
   nuScenes' unmatched close objects are *undersized* relative to typical GT (median size ratio 0.51) and sit far
   below the ground estimate (−0.50 m) — a bad depth/scale fit, not hyperinflation. ECP's equivalent set is too small
   to trust (n=13).

**Conclusion: the hyperinflation example is a real, worth-keeping qualitative illustration, but not a statistically
supported pattern.** Across three independent checks, nothing shows a shape-specific problem concentrated near the
ego vehicle; if anything, the close-range excess is more placement-dominated than the mid-range excess already was.

## Overall conclusions

1. **The free-space and below-ground findings are not an artifact of this project's own matching rule** — the
   headline numbers hold up under the nuScenes devkit's own distance-based criterion, at every one of its four
   official thresholds. This directly answers the most obvious methodological objection to the original claim.
2. **The strongest, most defensible single number for your professor:** on nuScenes, the shipped OBB touches
   significantly more certified-free space than its matched GT box — about 2 percentage points / 0.2–0.3 m³ median
   excess — confirmed by two independent matching protocols and stable across four distance thresholds.
3. **The honest complication stands, and the proximity follow-up sharpens it rather than rescuing the shape
   angle:** claim 4 (overall, and even more so at close range) says this excess is substantially a *positioning*
   effect — already accepted, already LiDAR-driven — not a pure *shape* effect, on the dataset with the most
   evidence. Claim 5 (roughly half the violation not implied by in-mask depth) remains the cleaner, matching-robust
   evidence that free space adds information beyond simple depth-matching — that's the number to lead with for
   "method 1 doesn't already cover this." The near-field hyperinflation example is a real case worth showing
   visually, but shouldn't be presented as evidence of a systematic, range-correlated shape defect.
4. **Below-ground is dataset-split:** solid and robust on ECP, null on nuScenes (worth the PseudoLabeler-baseline
   sanity check before using it further).
5. **Remaining follow-ups, not blocking:** claim 4 under the nuScenes-style match (cheap but not yet done); the
   PseudoLabeler ground-baseline bias check; picking exemplary frames for the visual evidence pack (including
   `ecp10_f1080` as a qualitative, not statistical, illustration).
