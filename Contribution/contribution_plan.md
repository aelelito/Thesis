# Sensor-Constrained Generative Auto-Labeling — Full Plan

*Working document. Covers the proposed contribution, every sub-idea discussed, answers to open
questions, the evaluation design, concrete next steps, and annotated literature with links.*

---

## Table of contents

0. [How to read this](#0-how-to-read-this)
1. [Glossary — every abbreviation used](#1-glossary)
2. [The problem in one page](#2-the-problem-in-one-page)
3. [The three regimes — the backbone of the paper](#3-the-three-regimes)
4. [What TRELLIS / SAM3D Objects actually is](#4-what-trellis--sam3d-objects-actually-is)
5. [What LiDAR really measures: returns *and* free space](#5-what-lidar-really-measures)
6. [Why free space is easier than in-mask point assignment](#6-why-free-space-is-easier-than-in-mask-point-assignment)
7. [The three constraints, precisely defined](#7-the-three-constraints)
8. [Occlusion-aware silhouette: lower bound vs upper bound](#8-occlusion-aware-silhouette)
9. [Asymmetric trust — what I actually meant](#9-asymmetric-trust)
10. [Tier 1 — post-hoc sensor-consistent alignment](#10-tier-1--post-hoc-alignment)
11. [Tier 2 — LiDAR guidance inside the generative process](#11-tier-2--guidance-inside-generation)
12. [Regime A: objects too close (over-inflation)](#12-regime-a-objects-too-close)
13. [Regime C: objects too far (no LiDAR) — scene-level metric propagation](#13-regime-c-objects-too-far)
14. [Observability — what it is, what goes in it](#14-observability)
15. [Generative-ensemble uncertainty](#15-generative-ensemble-uncertainty)
16. [Downstream detector training with label weights — explained plainly](#16-downstream-detector-training)
17. [Using the SAM3 class label during inference](#17-using-the-class-label)
18. [Free space as a false-positive filter](#18-free-space-as-a-false-positive-filter)
19. [TokenGraph3D — what it is and where it fits here](#19-tokengraph3d)
20. [Do we need to train anything? (No.)](#20-do-we-need-to-train-anything)
21. [Evaluation plan](#21-evaluation-plan)
22. [Risks and fallbacks](#22-risks-and-fallbacks)
23. [Next steps, in order](#23-next-steps-in-order)
24. [Literature with links](#24-literature)

**Part II — added after the second review pass**

25. [Corrections and clarifications](#25-corrections-and-clarifications)
26. [Design choice → literature justification](#26-design-choice--literature-justification)
27. [Claude Code handoff — initial setup prompt](#27-claude-code-handoff--initial-setup-prompt)
28. [**The ordered roadmap**](#28-the-ordered-roadmap)
29. [Quick reference — what each constraint needs](#29-quick-reference--what-each-constraint-needs)
30. [Open questions to resolve as you go](#30-open-questions-to-resolve-as-you-go)
31. [SS Correction audit findings](#31-ss-correction-audit-findings-from-claude-code)

---

## 0. How to read this

Sections 2–9 are **concepts** — read these first, they're the foundation.
Sections 10–19 are **the actual things to build**, roughly in priority order.
Sections 20–23 are **practical**: what needs training (nothing), how to evaluate, what to do Monday.
Section 24 is the reading list.

Nothing from our earlier discussion has been dropped. Several things have been **added** because of
your near/far observation, which turned out to be the most useful thing you've said so far — it
gives the work a much cleaner structure.

---

## 1. Glossary

Every abbreviation used in this document, in one place.

| Term | Meaning |
|---|---|
| **Amodal** | Reasoning about the *whole* object including the parts you cannot see. Opposite of *modal*, which is only the visible part. An amodal box wraps the whole car; a modal box wraps only the visible rear bumper. |
| **AOE** | Average Orientation Error — nuScenes metric, yaw error in radians. |
| **ASE** | Average Scale Error — nuScenes metric, `1 − IoU3D` after aligning centres and orientation. Pure size error. |
| **ATE** | Average Translation Error — nuScenes metric, centre distance in metres. |
| **BEV** | Bird's-Eye View — top-down 2D projection. |
| **CFormer** | CompletionFormer — a depth-completion network that turns sparse LiDAR + RGB into a dense depth map. You use it in O4/O5. |
| **Depth completion** | Filling in a dense depth map from sparse depth measurements + an image. |
| **DPS** | Diffusion Posterior Sampling — a technique for steering a diffusion model at sampling time so its output matches a measurement. |
| **Fitting formulation** | Using LiDAR points to *solve for* parameters (e.g. affine transform coefficients). Needs precision AND recall. One bad point corrupts the solution. See §6. |
| **Free space** | Volume that a sensor has proven to be *empty*, because a laser ray passed through it and returned from something further away. |
| **HDBSCAN** | Hierarchical DBSCAN — density-based clustering algorithm you currently use to clean in-mask LiDAR points. |
| **Metric depth** | Depth in real units (metres), as opposed to relative/affine-invariant depth which is only correct up to an unknown scale and offset. |
| **MoGe** | Monocular Geometry estimation model. Produces *affine-invariant* (relative) depth + point maps from a single image. You use it as your baseline pointmap. |
| **Modal** | The visible part of an object only (see *Amodal*). |
| **Observability** | (Our term.) A per-object score in [0,1] measuring how much of the object the sensors actually got to see. Defined in §14. |
| **OBB** | Oriented Bounding Box — a 3D box with a yaw angle, not axis-aligned. |
| **Pointmap** | An (H, W, 3) array giving an XYZ position for every pixel. SAM3D Objects accepts one as input, which is your integration hook. |
| **Ray casting** | Tracing a straight line from the sensor origin through space, to determine what it passes through. |
| **Rectified flow** | A generative modelling technique closely related to diffusion. TRELLIS uses this rather than classic denoising diffusion. Practically: same idea of iterative refinement from noise, slightly different maths. |
| **SLAT** | Structured LATents — TRELLIS's second stage: feature vectors attached to occupied voxels, decoded into a mesh. |
| **SS** | Sparse Structure — TRELLIS's first stage: a coarse voxel grid saying which cells the object occupies. |
| **Self-occlusion** | An object hiding its own far side from the sensor. Every solid object self-occludes. |
| **Silhouette / visual hull** | The 3D cone swept out by a 2D mask when back-projected from the camera. The object must lie inside it. |
| **Testing formulation** | Using LiDAR points to *evaluate* a candidate shape (does this mesh agree with these points?). Needs precision only. Zero points = no test, not garbage. See §6. |
| **Truncation** | An object running off the edge of the image, so its mask is cut by the image border. |
| **VFM** | Vision Foundation Model — a large pretrained model like SAM3, DINOv3, CLIP. |

---

## 2. The problem in one page

A LiDAR sees an object from **one side only**. A car 30 m away might return 40 points, all on its
rear bumper and left flank. The front half is never measured — not sparsely, not badly, *not at all*.

If you fit a box tightly to those 40 points you get a box that is:
- **too short** (you measured 2 m of a 4.5 m car), and
- **shifted toward the sensor** (the box centre lands on the visible surface, not the object centre).

Every method in the auto-labeling literature patches this the same way: look up an average size for
the class and stretch the box. MODEST, OYSTER, LISO, LiSe, UNION, CPD, VESPA, AnnofreeOD, CM3D,
OVM3D-Det — all of them. This works for cars and fails everywhere else, and it quietly destroys the
"open-vocabulary" claim: if you need a size table per class, you are not open-vocabulary.

**Three independent papers state this as an unsolved problem, two of them from your own lab:**

- **CPD (CVPR 2024)** measures it: ~65% of objects on Waymo never get full scan coverage, so their
  pseudo-label sizes are wrong, and temporal consistency cannot fix it because the object is never
  seen complete in any frame.
- **AutoBox (your lab)** — in the conclusion: densification still leaves self-occluded parts
  incomplete, which limits scale and centre accuracy relative to methods that inflate boxes with
  class priors. And in the appendix, under box-fitting edge cases: heavily occluded objects produce
  undersized boxes, there is **no special handling** in the current implementation, and future work
  should use class-specific size priors or temporal consistency.
- **LabelAny3D (Jan 2026)** — the closest published relative of your pipeline. Uses TRELLIS to
  imagine the whole object, then fits a box to the imagined mesh. Their limitation section says the
  generated meshes have ambiguous depth along the viewing direction, misaligning them with the RGBD
  point cloud, and their stated future work is to condition the 3D generation on RGBD data.

**Your pipeline is the only one in this literature that generates a complete object rather than
fitting a box to visible points.** That is your structural advantage. But an unconstrained generative
model imagines *freely* — which is exactly why your ASE sits at 0.27–0.55 and why your very close
objects come out inflated.

> **One-sentence thesis:** Everyone uses LiDAR *returns* and nobody uses LiDAR *free space*, and free
> space is the missing constraint that makes a generative shape prior trustworthy on the majority of
> objects you cannot see properly.

---

## 3. The three regimes

> **Revised.** The first version of this section used *range* as the axis (near / medium / far). That
> was wrong: an object truncated at the image edge at 3 m and one truncated at 40 m have completely
> different problems. The correct axis is **which evidence you have**, not how far away you are.

### The taxonomy: evidence, not range

|  | **Lots of LiDAR** | **Little / no LiDAR** |
|---|---|---|
| **Good image evidence** | **B — works.** Mild correction only. Don't break it. | **C — metric propagation** (§13). Shape is fine; depth and scale are unanchored. |
| **Poor image evidence** (truncated or heavily occluded) | **A — trust the sensor.** Use out-of-FOV returns (§12); substitute finite bounds for the lost silhouette (§8). | **D — reject.** No evidence in either modality. |

**Cell D is where your false positives live.** The O5 car FP explosion (459 → 933 on nuScenes) is
cell D: objects with no in-mask returns receiving plausible-looking interpolated depths. O3 hides
them accidentally by leaving them at non-metric depths outside the eval range — which is suppression
by luck, not by decision. Having a *named* cell for "we should not emit a label here" is worth a lot,
and free space is what lets you detect it deliberately.

The regime letters below keep their names for continuity, but they are defined by the table above,
not by distance.

Your pipeline does not have *one* failure mode — it has three distinct behaviours plus a rejection
case, and they share a single common cause and a single common fix.

### Regime A — Object too close

**Symptom:** grossly inflated box.

**Why:**
1. **Truncation.** A close object runs off the image edge. SAM3 gives a clipped mask. SAM3D Objects
   was trained on crops where the object roughly fills the frame, so it interprets the visible sliver
   as the whole object and generates something the wrong size.
2. **Strong perspective.** At 3 m an object subtends a huge angle. The in-mask depth spread is
   enormous (front bumper at 2 m, rear at 6 m). Any single-number summary — median depth, a local
   affine fit — is a terrible description of that surface.
3. **Contamination.** Ego-vehicle roof returns and ground returns project heavily into close masks
   (you already fight this with ego-body filtering).

**Key insight:** in this regime **you have hundreds of LiDAR points covering two or three sides of
the object**. The measurement nearly determines the box on its own. The generative prior should be
contributing almost nothing here — and yet in your current pipeline it dominates unconditionally.
*That is the bug.* The fix is not a new module; it is letting the evidence outvote the prior.

Free space is also at its **strongest** here — dense rays, short range, high angular resolution. So
the constraint bites hardest exactly where the problem is worst. That's a lucky alignment.

### Regime B — Object at medium range, well visible, some LiDAR

**This already works.** Your O3/O5 numbers on ECP cars, motorcycles, bicycles come from here. Don't
break it. The constraint should be a mild correction, not a rewrite.

### Regime C — Object far away, visible in camera, zero LiDAR returns

**Symptom:** correct-looking shape at completely wrong depth and therefore wrong metric size. This is
also your dominant false-positive source on nuScenes (car FPs 459 → 933 going from O3 to O5).

**Why — and this is the deep reason:** for a pinhole camera, **size and depth are entangled**. An
object twice as far away and twice as large projects to *exactly the same pixels*. Monocular depth
models resolve this by learning priors ("cars are about 4.5 m"), which is precisely the crutch you
want to remove. With no LiDAR return, there is no measurement to break the tie.

**Therefore any solution for Regime C must import an anchor from outside the object.** Three
label-free ways to do that, all developed in §13:
- the ground plane the object stands on,
- other objects in the same image that *do* have reliable LiDAR,
- the relative-depth ordering from MoGe, which stays valid even when its scale doesn't.

### Why this framing is good for the paper

Three symptoms, one cause: **the balance between measurement and prior is fixed, when it should
depend on how much was actually measured.** One mechanism fixes all three: an explicit,
geometrically-derived **observability** score that continuously hands authority from prior to
measurement. That's a clean, single-idea paper rather than a bag of tricks.

---

## 4. What TRELLIS / SAM3D Objects actually is

You asked what TRELLIS is. It matters, because Tier 2 (§11) operates inside it.

**TRELLIS** ("Structured 3D Latents for Scalable and Versatile 3D Generation", Microsoft Research,
CVPR 2025 Spotlight) is a model that turns a single image (or text) into a 3D object. SAM3D Objects
is built on this architecture family, which is why it has a pointmap input hook.

It generates in **two stages**:

**Stage 1 — Sparse Structure (SS).**
Imagine a 64×64×64 grid of cells around where the object will be. The SS stage starts from noise and
iteratively refines a binary decision per cell: *occupied or empty*. The output is a coarse voxel
blob — the object's rough shape and, crucially, **its extent**. No texture, no detail. Just "the
object is roughly here, and roughly this big."

**Stage 2 — Structured Latents (SLAT).**
Only the cells marked occupied by Stage 1 get a feature vector attached. These vectors are generated
by a second iterative process, then decoded into a detailed mesh (or Gaussians, or a radiance field).

Two implementation notes that matter:

- TRELLIS uses **rectified flow** rather than classic denoising diffusion. Practically this is still
  "start from noise, refine over N steps." Guidance techniques carry over; the maths is in terms of a
  velocity field rather than a score function. Don't let the terminology confuse you.
- It was trained on ~500K clean, isolated, complete 3D assets (Objaverse-style). It has **never seen
  a partially-observed, occluded, truncated street object**. That domain gap is a large part of why
  your close-range and occluded cases fail.

**Why this two-stage split is useful to you:** the box you care about is determined almost entirely
by **Stage 1**. Extent lives in the voxel grid; Stage 2 only adds surface detail. So if you want to
inject geometric evidence, Stage 1 is the right place — and it's exactly where you already have an
intercept point in `inference_pipeline_pointmap.py`.

### How LiDAR currently enters SAM3D Objects

LiDAR data flows into SAM3D Objects exclusively through the **pointmap** — an (H, W, 3) array
giving a 3D position for every pixel in the image crop. The model receives this alongside the
RGB crop; it does not receive the raw point cloud.

The five operational pointmap modes (O1–O5) differ only in how that array is constructed:

| Mode | Construction | LiDAR role |
|---|---|---|
| O1 `lidar` | Sparse LiDAR projected into image pixels | Direct; accurate but sparse |
| O2 `moge_affine` | MoGe relative-depth, global affine fit to LiDAR returns | LiDAR calibrates MoGe scale/offset |
| O3 `local_affine` | MoGe relative-depth, per-object affine fit | As above, locally |
| O4 `ground_filter` | CFormer depth completion + ground filter | LiDAR seeds the completer |
| O5 `mask_hdbscan` | CFormer depth completion + HDBSCAN foreground | As above, with clustering |

In every mode, once the pointmap is passed to SAM3D Objects, LiDAR's role ends.

Architecturally, the pointmap **does** enter the SS generator's cross-attention at every
denoising step — the `PointPatchEmbed` module converts it into depth feature tokens that
condition the sparse-structure diffusion alongside the image tokens. So the model *could*, in
principle, use depth information to influence shape. **But it doesn't.** The SAM3D authors
measured this directly (Appendix E.5): in a head-to-head human preference test for *shape*,
the version conditioned on pointmaps and the version without are each preferred 48% of the
time — statistically indistinguishable. The pointmap influences **where** the object is placed
in 3D space (translation, scale), not **what shape** it takes.

Concretely: LiDAR currently constrains depth position only, not size or shape. If the
generator produces a mesh 20% too wide, nothing in the pipeline catches it. The SAM3D authors
built a depth input pathway, observed it doesn't help geometry, and moved on. They do not
perform ray casting, free-space reasoning, occupancy testing, geometric consistency checking,
or any form of constraint enforcement on the generated mesh — none of this appears in their
paper, appendix, or codebase.

Free-space evidence is the first mechanism that could catch such errors: a voxel that is
certified empty by LiDAR ray traversal but contains part of the generated mesh is a hard
geometric contradiction, not a soft prior disagreement. That is the gap this contribution
fills.

---

**Why your earlier SS-correction attempt likely underperformed:** the audit (§27) found that
`Z_surface = median(Z_anch)` — a **single scalar** per object — was used as the depth reference for
suppression. For a car viewed at an angle, the visible surface spans several metres of depth. The
median sits in the middle; legitimate near-face voxels behind `median + 0.3 m` are deleted; the back
half of every angled object is carved away. Meanwhile, the code can only **suppress** voxels, never
add them. The failure was in the evidence model, not in the architecture. **SLAT was never harmed**
— it starts from fresh noise sized to whatever coordinate count it receives, so the "mutilated grid"
theory does not apply. Full audit: §27.

---

## 5. What LiDAR really measures

This is the single most important concept in the plan.

Every paper in your literature spreadsheet uses LiDAR **returns**: "a point came back at (x,y,z),
therefore there is surface there."

But a LiDAR beam is a **line**. If a beam leaves the sensor, travels through a region, and returns
from something 50 m away, then **everything along that line up to 50 m is empty**. That is a
measurement — a very reliable one — and it is essentially unused in this literature.

For a given object you can now partition space into three:

| Label | Meaning | Who decides the geometry there |
|---|---|---|
| **Occupied** | A return landed here. Surface exists. | The measurement. Mesh must pass through. |
| **Certified empty** | A beam passed through and returned further away. | The measurement. Mesh must *not* enter. |
| **Unknown** | The shadow behind the visible surface. No beam ever reached it. | The generative prior. Free to imagine. |

**This is the whole idea.** The prior is only allowed to invent geometry in the *Unknown* region.
Everywhere else, the sensor decides.

And note what the three regions do:
- Returns constrain the object **from behind you** — where the visible surface is.
- Free space constrains the object **toward the sensor and laterally** — where it cannot extend.
- The mask (§7, §8) constrains it **laterally in image space**.
- The only unconstrained direction is **directly behind the visible surface**, which is exactly the
  part no method can measure and exactly what a shape prior is for.

**Range dependence.** Free-space evidence is dense at short range (many beams per solid angle) and
sparse at long range. So its constraining power decays with distance — which is another reason it
maps onto the three regimes: strongest in Regime A (where you need it most), weakest in Regime C
(where §13 takes over instead).

---

## 6. Why free space is easier than in-mask point assignment

You raised the right objection: *"defining occupancy is exactly the hard problem I've been fighting
with HDBSCAN and mask erosion."* Here's why free space largely dodges it.

### The key asymmetry

**Free space is computed in 3D from the sensor, not in 2D from the mask.**

To ray-cast you need only two things per beam: its direction, and the range at which it returned.
You do **not** need to know which points belong to which object. You do not need HDBSCAN. You do not
need mask erosion. The computation is:

```
for each LiDAR beam:
    mark every voxel along the beam, from sensor origin up to (return_range − ε), as EMPTY
    mark the voxel at return_range as OCCUPIED
```

That's it. Object-agnostic, mask-agnostic, parameter-light. The mask enters only later, to decide
*which* region of the resulting map is relevant to *this* object.

So the hard problem you've been fighting sits on the **occupancy** side, and free space gives you a
whole second constraint that bypasses it entirely.

### Free space has two error modes — one benign, one dangerous

### First, the decision that makes this work

> **Build the occupancy / free-space map ONCE PER FRAME, globally, in 3D, straight from the raw
> sweep — object-agnostic and mask-agnostic. Then QUERY it per object.**
>
> Do **not** build a map per mask from that mask's points. If you do, every foreground bleed gets
> attributed to the wrong object and the objection below becomes real. The global map never
> attributes anything to anything, so there is nothing to mis-attribute.

### Free space has two error modes — one benign, one dangerous

**Benign: foreground bleeding.** Trace what actually happens. A fence returns at 8 m; the car is at
12 m. That beam certifies:

- voxels 0 → 8 m along the ray: **EMPTY**
- the voxel at 8 m: **OCCUPIED**
- voxels beyond 8 m: **UNKNOWN** — the beam was blocked, we learned nothing

The car's volume lies entirely in the UNKNOWN region. The occupied voxel is at 8 m, *in front of* the
car, not inside it. So the free-space map is not corrupted where the car is — it simply has **less
evidence** there. You **lose** evidence; you do not **gain** wrong evidence. That is what
"conservative" means here.

The severe version of foreground bleeding — a fence point winning the min-depth competition and
anchoring the object at 8 m — is a **depth-anchoring** and **occupancy-term** problem, not a
free-space problem, and it is handled by the high-precision core below.

**Bonus, and this answers "how do I know which part is occluded":** the UNKNOWN region inside an
object's frustum **is** the occluded volume, measured in 3D. You don't have to infer an occlusion
direction from 2D mask adjacency — you have the actual shadow. See §8.

**Do not ground-filter the cloud you ray-cast with.** Ground returns are *excellent* free-space
evidence — a beam skimming along the road certifies an enormous volume empty. TerraSeg's binary
ground/non-ground output should gate the **object occupancy core**, not the ray casting. Different
constraints get different preprocessing; this is a recurring theme (§8, §13 C1).

**Dangerous: background bleeding.** A beam that passes through the car's mask cone but returns from a
wall 20 m behind it would certify the car's own volume as empty — carving away a real object. This
happens when:
- the mask slightly over-extends past the true object boundary, or
- the beam genuinely passes *through* the object (between wheels, through windows — LiDAR does pass
  through glass, which is a real effect).

**Four mitigations, all cheap:**

1. **Require ray-bundle agreement.** Don't let a single beam carve a voxel. Require *k* beams
   (e.g. k ≥ 3) traversing the same voxel to agree it's empty. Isolated see-through beams are then
   ignored. This is the standard fix in occupancy mapping and it works.
2. **Erode the mask for carving only.** You already have mask erosion. Use the *eroded* mask to
   define which beams are allowed to carve, and the *full* mask for the silhouette bound. Different
   constraints get different levels of caution.
3. **Log-odds accumulation instead of hard decisions.** Rather than binary EMPTY/OCCUPIED, accumulate
   evidence per voxel: each traversal adds negative log-odds, each return adds positive. Only voxels
   whose accumulated evidence passes a threshold get to constrain the generator. This makes the whole
   thing robust to individual bad beams and is standard practice (octomap-style).
4. **Never let a single constraint be fatal.** In Tier 2, free space is a *soft* penalty in a
   likelihood term, not a hard deletion. A voxel with weak free-space evidence gets a weak push, not
   a death sentence. This is also the answer to §9.

### Occupancy: precision over recall

You have been trying to find **all** the object's points. For this method you don't need all of them
— you need **a few you are certain about**. The occupancy constraint is satisfied by any subset.

So flip your objective: instead of "recover every object point," aim for "**recover a high-precision
core**." Concretely:

```
core(obj) = in_mask(eroded by k px)
          ∩ dominant_HDBSCAN_cluster
          ∩ robust_depth_band(median ± 1.5·MAD)
          ∩ NOT inside any nearer mask        ← cheap, kills foreground bleed directly
          ∩ [Phase 3] same TokenGraph3D instance (§19)
```

The fourth line is worth adding immediately if it isn't already there. SAM3 gives you **all** masks. A
point projecting inside both the car mask and a nearer pedestrian mask, whose depth matches the
pedestrian, is not the car's point. Two lines of code, and it directly attacks the failure mode that
motivated O5.

Anything ambiguous, throw away. You lose recall and you don't care.

**And if the core comes out empty, that is a valid outcome.** Set `α = 0` (occupancy weight) for that
object and let free space + silhouette + prior carry it. O3 has no such option — it *must* fit an
affine transform, so an empty or degenerate core produces garbage rather than a known-unknown. That
is precisely the Regime C leak in your current pipeline.

### Why your in-mask filtering felt unsolvable: fitting vs. testing

**Your current formulation uses the points to FIT something.** O3 fits `Z = a·Z_moge + b` per object.
A fit needs **precision** (one wrong point corrupts the coefficients) **and recall** (too few points,
or too little depth spread, and the fit is unstable or degenerate). That joint requirement is why
HDBSCAN tuning felt endless — and why any failure yielded garbage rather than nothing.

**The constraint formulation uses the points to TEST a hypothesis.** Does this mesh agree with these
points? A test needs **precision only**. Fewer points is a *weaker* test, not a *corrupted* one. Zero
points is a **known state you can handle**, not silent garbage.

So the burden halves. You can erode aggressively, discard anything ambiguous, and be left with five
points you'd bet money on — and that's sufficient. In O3, five points isn't enough to fit anything.

This also explains your hull-anchoring pedestrian regression: erosion cost you points the *fit*
needed. In the testing formulation, that erosion is free.

> **This reframing alone may be worth the exercise:** the reason your in-mask filtering felt like an
> unsolvable problem is that you were solving the *fitting* version of it when the method only needs
> the *testing* version.

---

## 7. The three constraints

For an object with class *c*, mask *M*, and the LiDAR sweep, define a score for any candidate 3D
shape *S*:

### (1) Occupancy term — "the measured surface must lie ON the mesh surface"

> **Revised.** The first version said points must lie *inside* the mesh. That throws away the signal
> that detects over-inflation. LiDAR returns come from the **surface**, so a point sitting deep inside
> the mesh means the mesh is too big in that direction — which is exactly the Regime A signature.

Two-sided, asymmetric, with a tolerance band:

```
L_occ(S) = Σ_p [ w_out · max(0,  sdf_S(p))²        # outside the mesh: bad, strong penalty
               + w_in  · max(0, −sdf_S(p) − τ)² ]   # further than τ inside: also bad, weaker
```

- `sdf_S(p)` positive outside the mesh.
- `w_out > w_in` — being outside is worse than being inside, because mesh detail is coarse.
- `τ` is a tolerance band of a few centimetres. **It exists because beams genuinely penetrate glass**
  and return from seats and dashboards 30–50 cm inside the shell. Set `τ` per class: larger for
  vehicles with windows, near-zero for pedestrians, bicycles, poles.
- Robust loss (Huber / Geman-McClure) on both halves.

**Why this matters:** a mesh that is uniformly too large has *all* of its points inside. This term,
not free space, is what detects that. Regime A is fixed here as much as in §5.

### (2) Free-space term — "the mesh must not enter proven-empty space"

For each voxel *v* with accumulated free-space evidence `w_v > 0`, penalise mesh occupancy there:

```
L_free(S) = Σ_v  w_v · occupancy_S(v)
```

`w_v` is the log-odds confidence from §6, so weakly-certified voxels push weakly.

### (3) Silhouette term — "the mesh must project consistently with the mask"

Render the mesh from the camera, get a rendered mask `M_render`. Two separate pieces — see §8, this
is where your question about "limiting the mesh to the hull of the mask" gets answered properly:

```
L_sil(S) = λ_in  · |M \ M_render|          (mesh must cover the whole visible mask — always)
         + λ_out · |M_render \ M|_restricted (mesh must not spill — only past REAL boundaries)
```

### Combined

```
score(S) = − [ α·L_occ + β·L_free + γ·L_sil ]
```

with α, β, γ set per class or per regime (§17). This score is used by Tier 1 (optimise a transform to
maximise it) and Tier 2 (use its gradient to steer generation).

---

## 8. Occlusion-aware silhouette

**Your question:** *"when an object is occluded partly, we can limit the mesh to the hull of the mask,
because the object in reality extends further laterally than the mask only."*

You spotted the problem and then half-stated the solution. The correct answer is that the mask gives
you **two different constraints of different strength**:

### Lower bound — always valid

**Wherever the mask says "object", the object IS there.** So the rendered mesh must cover the entire
visible mask. If the mesh fails to project onto a mask pixel, it's too small or misplaced. This
constraint holds unconditionally, no matter how occluded or truncated the object is.

### Upper bound — only valid at *real* boundaries

**"The mesh must not extend past the mask" is only true where the mask boundary is a genuine object
boundary.** Where the boundary is caused by an occluder or the image edge, the object genuinely
continues beyond it and the constraint must be switched off.

So classify every mask boundary pixel into one of three types — you already have everything needed:

| Boundary type | How to detect | Enforce upper bound? |
|---|---|---|
| **Real object boundary** | Adjacent region is background / far depth | **Yes** |
| **Occlusion boundary** | Adjacent region belongs to another SAM3 mask that is *nearer* in depth | **No** — mesh may extend behind it |
| **Truncation boundary** | Pixel lies on the image border | **No** — mesh may extend off-image |

Implementation: dilate the mask by a few pixels, look at what's in the ring, compare depths. Cheap.

### Never remove a bound — always substitute a weaker but finite one

The obvious worry: if you switch the upper bound off for occluded and truncated objects, doesn't a
close truncated object inflate without limit again? Yes — if you *remove* the bound. So don't.

| Boundary type | Upper bound applied |
|---|---|
| **Real object boundary** | Mesh must not project outside `M`. **Strong.** |
| **Occlusion boundary** | Mesh may extend behind the occluder — but **not past where the occluder ends** into visible background. Bound = silhouette of `M ∪ occluders`. Weaker, still finite. |
| **Image truncation** | No image evidence exists. Bound comes from **LiDAR outside the camera FOV** (§12) plus a soft class-extent regulariser. |

The truncation row is the important one: a truncated object loses image evidence but *gains* LiDAR
evidence — it's close, and the LiDAR is 360° while the camera is not. The two modalities complement
each other exactly where each fails. This is why §12 (use the full point cloud, not just in-camera
points) is not an optimisation but a requirement.

### Better: use the 3D unknown region, not the 2D direction

The 2D boundary classification above is the cheap detector, and it feeds the occlusion component of
observability (§14). But for the *constraint itself*, use the free-space map: **the UNKNOWN voxels
inside the object's frustum are the occluded volume, measured in 3D.** You don't need to infer which
direction is occluded — you have the actual shadow, with its actual extent. Let the mesh occupy
UNKNOWN freely, penalise it in EMPTY, and the occlusion handling falls out with no special case.

A split mask (an object divided in two by a pole) is a strong occlusion signal in its own right: the
gap between the parts is where the occluder is. Keep your convex-hull rejoining logic for this.

**Why this matters a lot:** an occluded car whose upper bound is wrongly enforced gets squashed to
the visible sliver — which is precisely the "undersized box" failure AutoBox describes and has no
handling for. Getting this right is a large part of the Regime-B/occluded-object win, and it costs
you maybe 60 lines of code.

**Bonus:** the same classification gives you the occlusion component of the observability score
(§14) for free. One computation, two uses.

---

## 9. Asymmetric trust

**Your question:** *"are you saying it is better to purely trust LiDAR and not use any depth inference
model such as MoGe or CompletionFormer?"*

**No.** Keep MoGe. Keep CompletionFormer. The point is subtler and it's about *permissions*, not
*inclusion*.

### The idea

Different sources of geometric evidence should be allowed to do different things, because **the cost
of their errors is different**.

| Evidence source | May it say "surface is here"? | May it say "nothing is here" (delete geometry)? |
|---|---|---|
| **Real LiDAR return** | Yes, strongly | **Yes** — the ray is a physical measurement |
| **CFormer depth at a LiDAR-anchored pixel** | Yes (it equals the LiDAR value with `preserve_input=True`) | **Yes** |
| **CFormer depth interpolated between anchors** | Yes, weakly | **No** |
| **MoGe depth (locally affine-fitted)** | Yes, weakly | **No** |
| **MoGe depth (no LiDAR anchor at all)** | Very weakly — shape only, not position | **No** |

### Why the asymmetry

The two errors have wildly different costs:

- **False permission** (letting the prior imagine something that isn't there): mild. The other
  constraints and the class prior will mostly clean it up, and you lose a bit of precision.
- **False suppression** (deleting real geometry because a *predicted* depth was wrong): catastrophic.
  You destroy a true positive. It's gone; nothing downstream can recover it.

Since the costs are asymmetric, the permissions should be too. **Only physical measurements get veto
power.** Predictions get a vote, not a veto.

### What this means concretely

- Your `w_v` free-space weights (§6) come **only** from real beams. CFormer/MoGe never contribute
  free-space evidence.
- MoGe and CFormer remain essential for: filling the *shape* of the depth map between anchors,
  supplying the pointmap that SAM3D conditions on, providing relative-depth ordering for Regime C
  (§13), and handling frames where LiDAR fails entirely.
- You had this instinct already — it's written in your `lidar_integration_plan.md` (LiDAR-anchored
  pixels can confirm *and* suppress; CFormer-interpolated pixels can only confirm). That instinct was
  correct. This section is just the general principle behind it, stated so you can defend it in a
  paper.

---

## 10. Tier 1 — post-hoc alignment

**Risk: low. Novelty: medium. Time: ~2 weeks. Build this first.**

### What it does

Run SAM3D Objects exactly as now. You get a mesh. **Before** taking its OBB, place it properly.

Optimise a **9-parameter transform**:
- 3 for rotation (or just yaw, if you want to keep it simple and let AutoBox-style heading handle the rest)
- 3 for translation
- 3 for **anisotropic** scale (separate x, y, z stretch — not a single uniform scale, because the
  generator often gets proportions wrong along one axis, particularly depth, which is LabelAny3D's
  stated failure mode)

Maximise the score from §7. Use gradient descent (differentiable rendering for the silhouette term,
a signed-distance query for the others) or, if you want something bulletproof, coordinate descent /
CMA-ES over 9 dimensions — that's small enough for derivative-free optimisation.

### Why it's safe

It cannot break your pipeline. Initialise at the identity transform. Worst case it doesn't move.
You get numbers in two weeks and you learn whether the constraints have signal.

### Why it's a contribution and not just engineering

It's a **drop-in box refiner** that needs only (mask, class, LiDAR, camera calibration) — the exact
inputs every method in your spreadsheet already has. So you can run it on:
- your own O3/O5 output,
- **VESPA's** boxes,
- **AutoBox's** boxes,
- UNION's boxes, if you can get them.

If it improves all of them, it is a *module*, not a system. That's exactly the "plug it into other
pipelines" property you said you wanted.

### Prior art you must cite and differentiate from

Two papers do "fit a shape prior to LiDAR + mask by optimisation":

- **SDFLabel** (Zakharov et al., CVPR 2020) — differentiable rendering of DeepSDF shape priors.
- **Segment, Lift and Fit / SLF** (ECCV 2024) — lifts 2D masks to 3D shapes, gradient descent on pose
  and shape until the projection matches the mask and the surface conforms to nearby LiDAR points.

**Your three differences:**
1. They use a **category-specific** learned SDF shape space, trained on synthetic cars. You use an
   **open-vocabulary generative foundation model** — you work on classes that have no shape space.
2. **Neither uses free space.** Both use returns + silhouette only. This is your novel constraint.
3. They evaluate on KITTI cars. You do multi-class driving with genuinely sparse 32-beam LiDAR, and
   you stratify by observability (§21), which nobody does.

---

## 11. Tier 2 — guidance inside generation

**Risk: medium. Novelty: high. This is the paper.**

### What it does, and why it is needed

> **Important correction (from the §27 audit).** The earlier diagnosis — "post-hoc editing breaks
> SLAT because it receives a mutilated grid" — was **wrong**. SLAT allocates a fresh noise tensor
> sized to the post-correction coordinate count and starts from scratch; there is no stale latent
> issue. The actual problem with the existing `ss_correction` is the **evidence model**: a single
> scalar (`median(Z_anch)`) collapses all per-pixel LiDAR information into one number per object,
> which simultaneously over-suppresses legitimate near-face voxels and misses hallucinated voxels
> near the median depth. See §27 for the full audit.

> **The argument for guidance over post-hoc correction therefore rests on expressiveness, not on
> architectural failure:**
>
> - Post-hoc correction can only **remove** voxels. It cannot add missing structure or shift the
>   spatial distribution. An object whose voxels are all 20 cm too far stays 20 cm too far.
> - **Tier 1.25** (§28 item 0.9) upgrades the evidence model from a scalar to the per-voxel
>   three-constraint score. This fixes the diagnosed flaw and may be *sufficient*. Test it first.
> - **Tier 2 (guidance)** can steer the entire voxel distribution toward measurement-consistent
>   positions *during generation*. Strictly more expressive: it can add, remove, and shift.
>   Worth building only if Tier 1.25 is demonstrably insufficient.

### What "guidance" means, simply

A generative model like TRELLIS starts from noise and refines over ~25 steps until an object appears.

**What you did before:** let all 25 steps finish, then delete voxels that disagreed with LiDAR. The
model had already committed to a coherent shape; you punched holes in it; SLAT then decoded something
it had never seen anything like. It broke.

**What to do instead:** at *each* of the 25 steps, look at the partially-formed shape, compute how
well it agrees with the LiDAR evidence, take the gradient of that agreement with respect to the
current latent, and add a small nudge in that direction before the next refinement step.

The shape is then **born** consistent rather than corrected afterwards. The model stays inside its
own distribution the entire time, so SLAT never sees anything unfamiliar.

### The formal statement (worth having, it's how you'd write it up)

You want to sample from the posterior:

```
p(shape | image, LiDAR)  ∝  p_gen(shape | image)  ·  p(LiDAR | shape)
                            └──────┬──────┘        └───────┬───────┘
                            SAM3D Objects,          your measurement
                            frozen, unchanged       model — THE NEW BIT
```

The measurement model `p(LiDAR | shape)` is `exp(score(shape))` from §7. Its gradient is what you
inject. Nothing is trained. The generator's weights never change.

### Why this should work — the mechanism is already validated

**SpaceControl** (arXiv 2512.05343) does training-free, test-time geometric control of **exactly this
model family** (TRELLIS) by intervening in the latent space, without fine-tuning. Their application is
artists supplying coarse primitives for asset creation. The machinery is proven; what's new is
(a) the measurement is a **sensor**, (b) the domain is **driving**, and (c) the likelihood includes
**free space**, which has no analogue in their setting.

The general technique is **Diffusion Posterior Sampling / reconstruction guidance / classifier
guidance** — a well-established family. Note TRELLIS uses rectified flow, so you're perturbing a
velocity field rather than a score; the adaptation is standard but worth checking carefully.

### Where to inject: Stage 1 (SS), not Stage 2

Extent lives in the sparse-structure voxel grid. That's what determines your box. Inject there.
You already have the intercept point.

### Practical details

- **Tier 1.25 first (from the audit).** Before building guidance, try upgrading the existing
  `ss_correction` intercept point with the per-voxel three-constraint score instead of the scalar
  median. This is cheap (~1 day), sits at the same code location, and may be *sufficient*. If it is,
  Tier 2 becomes a nice-to-have. If it isn't — typically because shifting and adding voxels matters
  more than suppressing them — then guidance is justified and the Tier 1.25 result is the ablation
  baseline.
- **Guidance strength schedule.** Start weak, ramp up. Early steps set global structure; too much
  early guidance derails the sample. Late steps are refinement; strong guidance there is safe. A
  linear or cosine ramp is a fine first attempt.
- **Soft, not hard.** Every constraint is a penalty with a weight, never a deletion. This is what
  makes §9's asymmetric trust implementable — weak evidence gives a weak gradient.
- **Guaranteed fallback.** If guidance strength = 0, the pipeline is byte-identical to today. Keep it
  flag-guarded exactly as you already do with `ss_correction`.
- **Escape hatch.** If gradients through the SS stage turn out to be painful, there is a
  cheaper approximation: **resampling / rejection**. Generate K samples, score each with §7, keep the
  best. Much dumber, no gradients, and it still demonstrates the principle. Consider it a Tier 1.5.

### The probing experiment that de-risks this

Before building guidance, run this (it's also interesting in its own right, in the style of
TokenGraph3D):

**At each denoising step, how much does the object's extent still change?** Decode the intermediate
SS grid, measure its bounding box, plot box dimensions vs. step index.

- If extent is settled by step 5 of 25 → guidance must be injected **early**, and your old post-hoc
  edit was hopeless by construction (which would explain the failure).
- If extent keeps drifting until step 20 → guidance late is fine and easier.

This is one figure, maybe three days of work, and it tells you exactly where to inject. It is also
the kind of "we looked inside a frozen model and found out how it works" result that your lab
demonstrably values (see §19).

---

## 12. Regime A: objects too close

Diagnosed in §3. Four concrete fixes, all inference-time:

### A0. Use the FULL point cloud, not only points that project into the camera

**This is the highest-priority item in this section.** Your LiDAR is 360°; your camera is not. A
truncated object **has returns outside the camera FOV**, and your pipeline currently discards them by
working from the visible-projection set (`pts_ego_vis`).

**Change:** define the object's point set **in 3D, not by mask projection**. Seed from the in-mask
core (§6), then region-grow in 3D — or take the full HDBSCAN cluster in the *unrestricted* cloud — to
pick up returns beyond the image edge.

Two payoffs:
1. Regime A recovers the extent evidence its truncated silhouette lost. This is the bound that §8
   says must replace the missing image bound.
2. Objects at **camera seams** — your multi-camera merge problem — become complete in LiDAR even when
   incomplete in every individual image. The merge stops being a 2D reconciliation problem.

### A1. Detect truncation explicitly and handle it

Mask touches the image border → flag it. For truncated objects:
- replace the silhouette upper bound at the border with the LiDAR-derived bound from A0 (§8) —
  **replace, never remove**,
- down-weight the generative prior's extent along the truncated axis,
- up-weight the LiDAR occupancy term, since you have plenty of points.

Note: truncation is **not** the same as proximity (§3). An object truncated at 40 m is in cell C or D,
not A. Keep them as separate observability components (§14).

### A2. Let observability hand authority to the measurement

This is the core fix. At very high observability (close, unoccluded, hundreds of points, multiple
sides visible), the prior should contribute almost nothing to extent. Currently it dominates
unconditionally. The continuous blend in §14 handles this without a special case.

### A3. Stop summarising in-mask depth with one number

For a close object the in-mask depth spread is metres. Median depth, or a single local affine fit, is
a bad model of that surface. Instead, use the full 3D point set directly in the occupancy term
(§7.1). You're already computing it; just stop collapsing it.

### A4. Exploit that free space is strongest here

Short range = dense beams = high-confidence free-space carving. An inflated mesh will be punished
hard. Expect the largest Tier 1 gains in this regime, which is a good early sanity check: **if the
constraint doesn't fix close-range inflation, something is wrong with your implementation.**

---

## 13. Regime C: objects too far

Your idea — *"put far objects into relation to well-visible close objects that have reliable LiDAR,
and thereby learn their proper size and distance"* — is correct and is a genuinely publishable
sub-contribution. Here's how to make it concrete.

**Call it: reliability-anchored intra-frame metric propagation.**

### The theoretical core (state this in the paper)

For a pinhole camera, **size and depth are entangled**: an object 2× further away and 2× larger
projects identically. With zero LiDAR returns there is no measurement to break the tie. Every method
breaks it with an **external anchor**. Existing methods use a class size prior. You will use three
anchors that require no class table.

### C1. Ground-plane anchoring (strongest, do this first)

You already fit a ground plane per frame (the PseudoLabeler in O4). For an object that touches the
ground:

1. Take the bottom edge of its mask — the contact point.
2. Cast the camera ray through that pixel.
3. Intersect with the ground plane.
4. That intersection **is** the object's metric depth.

This works at *any* range and requires no LiDAR on the object itself. It is a classic monocular cue
and it is extremely strong.

**This is fragile if done naively, and you were right to be sceptical.** Four mitigations:

1. **Use the FULL mask, not the eroded one.** Erosion serves the occupancy core; the contact point
   needs the true silhouette. Each constraint gets its own preprocessing.
2. **Never use a single pixel.** Take the bottom 5–10% of mask pixels and use a robust statistic, or
   fit a line to the lower boundary and intersect that.
3. **Check whether the bottom boundary is occluded** using §8's classifier. Car behind another car →
   bottom boundary invalid → skip this cue entirely for that object.
4. **Cross-validate against C2/C3.** Ground contact and anchored-affine are *independent* estimates.
   Agreement → high confidence. Disagreement → low confidence, which feeds observability (§14).
   Disagreement is information, not failure.

**The strongest argument for it: you already do this and it works.** B1 foot anchoring with
PseudoLabeler is exactly ground contact, and it gave +0.021 mAP concentrated in pedestrian depth.
Extending it from Body to Objects is a small step with in-house evidence behind it.

**Honest caveat:** best for pedestrians and thin objects with clean ground contact; worst for cars in
dense traffic where the bottom is usually occluded. That's why it's one stage of a cascade (C-fallback
ordering below), not the answer on its own.

**Other caveats:** the object must actually touch the ground (fails for objects on truck beds, hanging
signs); the ground model must extrapolate correctly far away — your PseudoLabeler is fitted per frame,
so check its residuals at range and fall back to a robust planar fit if the learned model
extrapolates badly. Use **TerraSeg's binary ground/non-ground** output for point classification and
**TerraSeg's PseudoLabeler** for the continuous ground surface, as you already do.

### C2. Anchored global affine on MoGe (this explains why your O2 failed)

MoGe produces **affine-invariant** depth: the true depth relates to MoGe's output by an unknown scale
and offset, usually in **disparity** (inverse-depth) space:

```
1/depth_true  ≈  a · moge_disparity + b
```

Your O2 fitted this globally over everything and it collapsed (motorcycle AP 0.021). Almost certainly
because the fit was dominated by ground points and near-field clutter, which have a different error
structure from objects.

**The fix: fit a and b using only high-observability objects as anchors.** Take the objects with
plenty of clean LiDAR, use their measured depths as targets, robustly regress (RANSAC or Huber) for
a and b, then read off the depth of every far object from the same map.

This is a small change from O2 but a principled one, and you can *measure* it: report O2 (all points)
vs. O2-anchored (reliable objects only). If the gap is large, that's a clean ablation row and an
honest explanation of an earlier negative result.

### C3. Pairwise relative-depth propagation

MoGe's depth **ordering** is reliable even when its scale isn't. So for a far object *f* and an
anchored near object *n* of any class:

```
depth_f  ≈  depth_n × (moge_depth_f / moge_depth_n)
```

Do this against several anchors and take a robust average. This is literally "putting them into
relation" and it needs no ground plane and no class knowledge.

### C4. Intra-scene class-size consistency

If a near car with reliable LiDAR measures 4.4 m long, and a far object is also classified "car" by
SAM3, then the far object's *generated mesh* can be scaled so its metric size is consistent with the
same-class sizes **observed in this scene**, rather than a global lookup table.

This is subtly but importantly different from a class size prior:
- A class prior is **external, fixed, and hand-curated** (or LLM-queried, as VESPA does).
- This is **derived from the data at hand, per scene, at inference time**, adapts to the dataset
  automatically, and works for any class SAM3 can name — including ones nobody wrote a size for.

You can state: *"we use no external size priors; where size cannot be measured, it is transferred
from same-class instances measured in the same scene."* That is a strong, checkable claim and it's
the removal-of-a-prior move that §19 explains your lab likes.

### Considered and deferred: cross-scene accumulating memory

An earlier idea was to let C4 accumulate size/shape evidence across frames and scenes as inference
progresses, growing into a persistent lookup table rather than staying per-scene. **Deferred, not
adopted** — it would quietly convert C4 from "no external size priors" into "no *hand-curated* size
priors," a weaker, different claim (an online-learned prior is still external the moment it
outlives the scene it was observed in). It also reintroduces the moving-object smearing failure
already diagnosed in §25.2 if applied across frames of the same instance, and risks crossing into the
offboard/temporal-refinement territory this plan explicitly avoids. Kept as an optional future
ablation, not part of the core cascade — see `memory/project_regime_c_memory_idea.md` for the full
reasoning if you want to prototype it later.

### C5. Propagation confidence

Confidence in the propagated scale decays with:
- distance from the nearest anchor object,
- how few anchors were available,
- the residual of the affine fit in C2.

Feed this into observability (§14). A far object with three good anchors nearby is more trustworthy
than one alone at the horizon, and the pipeline should know that.

### Fallback ordering

Cascade, most-trusted first: **LiDAR return → ground contact (C1) → anchored affine (C2/C3) →
intra-scene class consistency (C4) → external class prior (last resort, flagged as such)**. Report
what fraction of objects each stage handles. That table is itself a useful contribution — nobody has
quantified how far you can get before you're forced to use a size table.

---

## 14. Observability

**Your question:** *"isn't observability just about 2D, e.g. if an object is occluded? Or is it also
about how many LiDAR points are available per object?"*

**Both, and more.** It is explicitly multi-modal. That's the point — it's a single number (or small
vector) summarising *all* the ways evidence can be missing.

### Components

| # | Factor | How to compute | Which regime it catches |
|---|---|---|---|
| 1 | **2D occlusion** | Fraction of mask boundary classified as occlusion boundary (§8) | occluded objects |
| 2 | **2D truncation** | Fraction of mask boundary on the image border | Regime A |
| 3 | **LiDAR point count** | Number of high-precision object points (§6), normalised by expected count at that range and class | Regime C, distant objects |
| 4 | **Angular coverage** | How many distinct *sides* were seen. Compute surface normals of the object points, or bin the azimuth of points around the object centroid. Seeing 3 sides ≫ seeing 1 side, even with the same point count. | the "40 points all on the rear bumper" case |
| 5 | **Free-space coverage** | Fraction of the object's frustum that is certified empty vs. unknown | how much the constraint can actually constrain |
| 6 | **Depth-source reliability** | Which stage of the §13 cascade supplied this object's depth | Regime C |
| 7 | **Range** | Distance to ego | correlates with 3, 5 |

### Scalar or vector?

Compute all seven. Then:
- Report the **vector** in analysis — it tells you *why* a label is bad, which is far more useful for
  the paper than a single number.
- Use a **scalar** for the blending weight and detector loss. Simplest sensible combination:
  `obs = (1 − occlusion) · (1 − truncation) · min(1, points/expected) · coverage_factor`.
  A product is right because these are conjunctive: any one being zero should kill your confidence.

Do **not** learn the combination weights — that would need labels and would break the training-free
claim. Keep it geometric and hand-specified, and justify it.

### Factor 4 deserves emphasis

**Angular coverage is the one that actually predicts extent error**, and nobody uses it. Point count
is a poor proxy: 200 points all on one flat side tell you nothing about length. 30 points spread
across the rear and the side tell you a lot. This is a small idea with a big payoff, and it is
directly testable in Experiment B (§23).

### Where it's used — four places, one computation

1. **Blending prior vs. measurement** (§3, §12) — replaces LabelAny3D's binary visibility switch with
   a continuous handover.
2. **Multi-camera merging** (your bullet 4) — when a car appears in two cameras with two boxes, pick
   or weight by observability. This is the correct generalisation of both heuristics you tried
   ("bigger mask" and "more LiDAR points" are components 1 and 3 respectively, used in isolation).
3. **Temporal fusion** (your bullet 1) — across a track, fuse per-frame extent estimates weighted by
   observability, so the one frame where the car was seen unoccluded from the side dominates the
   fifty frames where it was a distant blob. This keeps you out of the crowded "another tracking
   paper" territory: it's an *application* of your score, not a new tracker.
4. **Downstream detector loss weighting** — §16.

---

## 15. Generative-ensemble uncertainty

The generator is stochastic. Run SAM3D Objects K times with K different seeds on the same object and
you get K different meshes. The **spread** of their dimensions, centres and headings is a free,
direct measure of how uncertain the model is.

### Two uses

**A second, complementary confidence.** Observability says *"I couldn't see much."* Ensemble spread
says *"even given what I saw, the model can't decide."* Different failure modes — an object can be
well-observed but genuinely ambiguous (a symmetric box truck), or poorly observed but stereotyped (a
distant sedan). Report both.

**A diagnostic for Tier 2 — this is the better use.** If adding the LiDAR constraint **reduces the
ensemble spread**, that is direct evidence the constraint is doing what you claim: it collapsed the
hypothesis space. Plot spread-before vs. spread-after per observability bin. That's a compelling
figure that no baseline can produce, because no baseline has a distribution to collapse.

### Honest positioning

**GLENet** (IJCV 2023) already does "generative label uncertainty" with a conditional VAE, modelling
the one-to-many relationship between an object's points and plausible boxes, as a plug-and-play
module for probabilistic detectors. **MEDL-U** does uncertainty-aware 3D auto-annotation via
evidential deep learning. Both **need labelled data to train the uncertainty model**. Yours is
training-free and falls out of the sampler. That's your angle. Make it a supporting result, not the
headline.

**Cost note:** K samples means K× inference. Do this on a subset for analysis, not on the full
dataset, unless you find K=3 is enough.

---

## 16. Downstream detector training

**Your question:** *"I don't really get what you mean here."* Let me redo it from scratch.

### The point of auto-labeling

You are not producing boxes for their own sake. The purpose of an auto-labeler is to **generate
training data for a real detector**. The standard protocol in this whole literature is:

1. Run your auto-labeler over an unlabelled dataset → get pseudo-labels (boxes).
2. Train a normal 3D detector (usually **CenterPoint**) on those pseudo-labels *as if they were
   ground truth*.
3. Evaluate that detector against real ground truth.

That final number is the field's actual currency. VESPA, AutoBox, UNION, CPD, MODEST, LiSe, LISO all
report it. Your literature review says you plan to do this too.

### The problem with step 2

The detector is trained with a loss that says "predict this box." **Every box is treated as equally
true.** But your boxes are not equally true:

- a car at 8 m, unoccluded, 400 LiDAR points across three sides → your box is probably within 20 cm
- a car at 55 m, half behind a bus, zero LiDAR returns → your box is a decorated guess

Training on both with equal weight means the guesses actively teach the detector wrong things. This
is measurable — AutoBox's own Oracle experiment shows that removing false-positive labels entirely
raises downstream performance by +3.6 mAP, and LabelAny3D found that training from scratch on noisy
OVM3D-Det labels **failed to converge at all**.

### The fix

Export your observability score **alongside each box**, and weight the detector's regression loss by
it:

```
L = Σ_i  obs_i · L_box(prediction_i, pseudo_label_i)
```

Confident labels push hard on the weights. Uncertain labels push gently. Bad labels do proportionally
less damage. A slightly more sophisticated version treats each pseudo-label as a Gaussian with
variance `σ² ∝ 1/obs` and uses a KL-divergence loss against a probabilistic detector head — this is
what MEDL-U does, and the machinery exists.

### Why it's defensible but not the headline

Loss weighting by label uncertainty is not new. **3DIoUMatch** filters pseudo-labels by predicted
IoU. **FixMatch** thresholds by confidence. **GLENet** and **MEDL-U** weight by learned uncertainty.
There's a 2024 paper doing evidential-learning-based auto-label verification.

**But all of those learn the uncertainty from labelled data.** Yours is computed from sensor geometry
with zero annotations, which is the only setting that matters for a fully annotation-free pipeline.

**Deliverable:** one extra table. *"CenterPoint trained on our labels: uniform weighting → X mAP;
observability-weighted → Y mAP."* If Y > X, it's a clean result and it validates observability as a
meaningful quantity. If Y ≈ X, drop it and say nothing. Low risk either way.

---

## 17. Using the class label

Your bullet: *"the SAM3 info about what class an object belongs to could be very beneficial and should
be used during inference, not just at the end."* Here is where it enters, without becoming a size
prior:

1. **As the generation prompt** — already happening via SAM3's class-specific prompts.
2. **As per-class constraint weights.** A pedestrian's silhouette is highly informative and its
   returns are few → raise γ (silhouette), lower α (occupancy). A truck has plentiful returns and an
   often-truncated silhouette → the reverse. Small, principled, easy to ablate.
3. **As the shape prior for occlusion handling** (your bullet 6). Class-conditional expectations —
   a pedestrian is vertical and compact, a car is a horizontal slab — come out of the generative
   prior *automatically*. That's a much cleaner answer than tuning HDBSCAN hyperparameters per class,
   and it's an argument for why the generative approach is right.
4. **As the intra-scene size anchor** (§13 C4) — measured within the scene, not looked up.
5. **As the last-resort fallback** when observability ≈ 0 and no anchor is available (§13 cascade).
   This is the *only* place a class size table belongs, and now you can say exactly that: *"we use
   class statistics only in the regime where the sensors provide no information whatsoever, and we
   quantify the size of that regime."* Far stronger than "we inflate boxes by class."

---

## 18. Free space as a false-positive filter

AutoBox lists two failure cases it cannot handle:

- **Window reflections** produce false-positive masks that project into 3D boxes on building facades.
  They state reflective surfaces remain unhandled.
- **Vehicles behind permissive occluders** (chain-link fences) get valid 2D masks, but the LiDAR
  projection anchors depth to the nearer fence surface. Their fix is a filter: discard any instance
  whose mask overlaps a "fence" mask by ≥0.1. Their own audit found this discarded 6.8% of all mask
  evaluations across 100 scenes, with some scenes losing 30%.

**Ray evidence handles both naturally:**

- A **reflection** produces a mask whose frustum is entirely **certified empty** — beams pass straight
  through to the wall behind. No solid object can live there. Reject.
- A **fence-occluded car** produces returns at the fence, but the free space *beyond* the fence is
  inconsistent with a solid car body sitting right at the fence plane. The evidence disambiguates
  where a mask-overlap heuristic cannot.

This also attacks the **O5 false-positive explosion** you measured (car FPs 459 → 933 on nuScenes).
Those are objects with zero in-mask returns receiving plausible-looking interpolated depths from
CompletionFormer. Free space is exactly the evidence that would suppress them, and it does so
*without* needing the "zero-LiDAR mask blanking" hack you sketched in your O6 plan.

**Deliverable:** a comparison against AutoBox's 6.8% discard rate — *"we recover N% of the instances
their fence filter discards, and reject M% of reflection false positives they cannot handle."* Direct,
quantitative, against a labmate's published number. Very concrete.

---

## 19. TokenGraph3D

### What it is, simply

Normally, to find individual objects in a LiDAR scan without labels, you delete the ground and run
DBSCAN/HDBSCAN: "points that are close together and densely packed belong to the same object." That's
a **handcrafted geometric prior** — a rule a human wrote.

TokenGraph3D asks whether that rule is needed at all.

They take **Utonia**, a 137M-parameter point transformer pretrained with no labels on a mixture of
indoor and outdoor point clouds. They freeze it. They push one LiDAR scan through it. Then, instead
of using the network's *final output features* — which is what everybody normally consumes — they
reach **inside** the network and take the **attention keys** from a chosen encoder block.

The grouping rule is then trivially simple: connect two points if their key vectors are
cosine-similar above a threshold τ. Every connected component of that graph is one object. **No
density, no proximity, no motion, no 2D.** Ground is removed by TerraSeg, itself self-supervised, so
the whole thing is human-label-free.

### The four findings (which are the real paper)

1. Instance information lives in the **queries and keys**, not in the output features.
2. Output features **semantically collapse** — they merge two adjacent same-class objects that the
   keys keep distinct.
3. The signal is **bimodal in depth** — strongest at the shallowest (enc0) and deepest (enc4) encoder
   stages, weakest in between.
4. It is driven by the **rotary position encoding (RoPE)** — remove RoPE and the advantage vanishes.
   So the signal is positional, not content-based.

The measured effect is large: on nuScenes without any geometric prior, the output feature reaches
only 0.273 association score while the enc4 key nearly doubles it to 0.517.

### Why it's a contribution — and this is the template you should study

1. **It's a discovery, not a system.** The headline is "we found something nobody knew was there,"
   not "we combined A, B and C." The segmenter exists to prove the finding.
2. **It ports a 2D result to 3D non-trivially.** In 2D this is known — DINO's keys contain object
   structure, and LOST/TokenCut/CutLER exploit it with global operations like Normalized Cut over a
   few hundred tokens. But PTv3 (Utonia's architecture) uses patch attention with points serialised
   along space-filling curves, has no global CLS token and no scene-wide attention matrix. They had
   to invent a different grouping mechanism. That's what makes it publishable rather than a port.
3. **It removes a prior instead of adding a module**, and designs the metric (a proximity-free,
   clustering-free seed-IoU probe) so that only the thing being claimed can influence the result.
4. **It's honest about limits** — adding proximity back absorbs most of the advantage, putting
   learned features level with DBSCAN. Stating this *strengthens* the paper.

### What to steal from it

Not the method — the **shape of the contribution**. Ted Lentsch is your daily supervisor (he's
acknowledged in AutoBox), so this tells you what the group rewards:

- **A hypothesis you can test** beats a system you can build.
- **Probing a frozen model's internals** is valued. You have a frozen model nobody has looked inside:
  SAM3D Objects' SS diffusion. §11's probing experiment is directly in this spirit.
- **Removing a prior is a contribution.** Theirs is "no density, no proximity." Yours can be
  **"no class size priors"** — removing the exact crutch that AnnofreeOD, VESPA, CM3D, CPD and
  OVM3D-Det all depend on. Same rhetorical move, and §13 C4 makes it defensible.
- **Design the metric so the effect is visible.** Their seed-IoU probe exists purely to isolate the
  claim. Your equivalent is the observability-stratified evaluation in §21.

### Should you use it for your clustering / segmentation?

**Your question, answered honestly: not now — but yes, later, as an ablation.**

**Against using it now:**
- It is class-agnostic and mask-agnostic: it segments the *whole scan* into instances. You'd still
  need to associate its instances to your SAM3 masks. (Easy — project and check overlap — but it's
  another moving part.)
- Its nuScenes numbers are good but not dominant: 0.517 prior-free, 0.546 with proximity, versus
  DBSCAN at 0.544. It doesn't obviously beat what you have.
- It changes nothing about your contribution. It's a swap-in for a component.
- Its own limitations section flags two failure modes — over-segmenting very large objects, and
  merging distant same-class objects through chains of feature-coherent edges — both of which would
  hurt you.

**For using it later — and there is a genuinely elegant reason:**

TokenGraph3D gives 3D instance masks derived **only** from LiDAR. SAM3 gives 2D masks derived
**only** from images. These are two *independent* modalities agreeing or disagreeing.

> **Intersect them.** Points that are inside the SAM3 mask **and** inside the same TokenGraph3D
> instance are high-precision object points, confirmed by two independent sensors.

That is *exactly* what the occupancy constraint needs (§6: precision over recall), and it solves your
in-mask filtering problem from a completely different direction — no HDBSCAN tuning, no mask erosion.
Their disagreement is also informative: it flags exactly the mask-bleeding cases you've been fighting.

**Verdict:** Phase 3. Cheap to test once the rest works, gives you a strong ablation row
("HDBSCAN in-mask cleaning vs. SAM3 ∩ TokenGraph3D"), cites a labmate's brand-new paper, and might
just work better. Don't let it delay Phase 1.

---

## 20. Do we need to train anything?

**Your question. The answer is no, and this is a selling point.**

| Component | Trained? |
|---|---|
| SAM3 (masks) | Frozen, off the shelf |
| SAM3D Objects / Body | **Frozen.** Weights never touched. |
| MoGe | Frozen |
| CompletionFormer | Frozen (pretrained checkpoint) |
| Ray casting / free space | Pure geometry. No parameters beyond a voxel size and a log-odds threshold. |
| Tier 1 alignment | Per-object **optimisation** at inference time. Nothing is learned across objects. |
| Tier 2 guidance | **Training-free** — gradients are used to steer sampling, not to update weights. |
| Observability | A hand-specified geometric formula |
| §13 metric propagation | Robust regression per frame. No learned parameters. |
| PseudoLabeler ground fit | Per-frame optimisation (already in your pipeline) |
| **CenterPoint (§16)** | **Trained** — but this is the *downstream consumer* of your labels, not part of the labeler. |

So: **the entire auto-labeler is training-free.** Say this loudly. It means:
- no risk of overfitting to a dataset,
- immediate cross-sensor transfer (ECP 64-beam ↔ nuScenes 32-beam) with no retraining,
- it composes with any future generative model — swap TRELLIS for its successor and the method still
  applies.

It is also the same claim TokenGraph3D makes, which is not a coincidence — it's what this group
values.

---

## 21. Evaluation plan

Three things, in strict priority order.

### E1. Stratify everything by observability — this is the money figure

Plot **ASE** (and separately ATE) on the y-axis against **observability bins** on the x-axis, with
three curves:

1. box-fitting baseline (VESPA-style, or your own OBB-from-points)
2. unconstrained generation (your current O3/O5)
3. **sensor-constrained generation** (the proposal)

**Your claim is not "mAP goes up."** Your claim is:

> *Generative shape priors currently only help where the object is well observed. Sensor constraints
> extend that benefit into the poorly-observed regime, which is where the majority of objects live.*

If that figure looks the way I expect — curve 2 beating curve 1 on the right, losing on the left, and
curve 3 beating both everywhere — the paper writes itself. It also directly answers LabelAny3D's
finding that size priors beat generation on low-visibility objects, which makes it a *response to a
published result* rather than a bare improvement.

Additionally stratify by the three regimes (§3) explicitly, since that's your narrative.

### E2. Cross-pipeline transfer

**You do not have AutoBox source**, so the three targets are:

1. **VESPA** (you have it) — a genuinely independent pipeline with a different box-fitting route
2. **Your O3** — MoGe local affine
3. **Your O5** — CompletionFormer dense

Three pipelines is enough to claim modularity, and VESPA being external is what carries the claim.

**Proxy for the prior-based family:** take your own boxes and apply class-size-prior inflation
(AutoBox / AnnofreeOD / CM3D style). That gives you a fair "prior-based" curve for the E1 plot without
needing their code, and you cite their described procedure. Request AutoBox source when you can, but
don't block on it.

### E3. Isolated ablation of the free-space term

| Configuration | ASE | ATE | mAP |
|---|---|---|---|
| no constraints (current) | | | |
| + occupancy only | | | |
| + occupancy + **free space** | | | |
| + occupancy + free space + silhouette | | | |
| + occlusion-aware silhouette (§8) | | | |

If the free-space row is where the jump happens, your novelty is isolated in one table, which is
exactly what reviewers look for.

### Supporting experiments

- **Downstream CenterPoint** trained on your labels, uniform vs. observability-weighted (§16).
- **Cross-dataset**: ECP (64-beam) and nuScenes (32-beam) with **one** configuration, no retuning.
  This is where training-free pays off.
- **Regime C cascade coverage**: what fraction of objects are handled by LiDAR return / ground contact
  / anchored affine / intra-scene consistency / external prior. Nobody has quantified this.
- **Ensemble spread before vs. after constraining** (§15).
- **FP reduction** on reflections and fence-occluded objects vs. AutoBox's 6.8% discard rate (§18).
- **Multi-camera**: observability-weighted merge vs. your two earlier heuristics.

---

## 22. Risks and fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| Free space turns out to be mostly *unknown* rather than *certified empty* for typical objects (too few beams) | Medium | Aggregate sweeps to densify ray coverage (you already do this); fall back to Tier 1 with occupancy + silhouette only; lean harder on §13 |
| Gradients through the SS stage are painful | Medium | Tier 1.5: generate K samples, score, pick the best. No gradients, same principle, still novel |
| SS extent is decided in the first few steps and guidance destabilises the sample | Medium | Found by the §11 probing experiment *before* you build anything. Adjust the schedule, or fall back to Tier 1 |
| Background bleeding causes free space to carve real objects | Medium | §6 mitigations: ray-bundle agreement, eroded mask for carving, log-odds thresholds, soft penalties |
| Tier 1 alignment barely moves the boxes | Low | Then Experiment A (§23) already told you, and you'd have pivoted to §13 + §14 as the main contribution instead |
| Someone publishes LiDAR-conditioned TRELLIS first | Low-Medium | The field moves fast. Hunyuan3D-Omni already does RGBD-conditioned generation (cited as future work by LabelAny3D). Move quickly, and note that free space + the driving/auto-labeling framing is still yours |
| Reviewers say "this is just SLF with a different shape prior" | Medium | Pre-empt: free space is genuinely new, open-vocabulary is genuinely new, guidance-inside-generation is genuinely new. Have E3 ready |

---

## 23. Next steps, in order

### This week — two experiments that decide everything

**Experiment A — is the error overshoot or undershoot?** *(~2 days)*

Take your true positives on ECP and build the free-space map for each. Check: does the predicted box
extend into certified-empty voxels (**overshoot**), or does it fall short of the ground-truth extent
(**undershoot**)?

**Run this on both O3 and O5.** O3 is your working baseline and the mode you should keep reporting.
But O3 leaves zero-LiDAR objects at garbage MoGe-relative depths, which places them outside the
metric evaluation range so they silently disappear — your own notes call this suppression
"accidental." Those vanished objects *are* Regime C. If you only diagnose on O3 you will be analysing
exactly the objects that already work. O5 makes them visible as measurable boxes.

**Related check while you're there:** compare **recall** (not AP) between O3 and O5. If O5's recall is
substantially higher and its extra detections are false positives that free space can suppress, then
switching base later is justified — O5 *tries* on hard objects and gets them wrong, whereas O3
declines to try, and free space is exactly the tool that lets you try and then reject. O5 also already
gives better ASE across all classes and halves motorcycle AOE, and ASE is what this whole plan
targets. Report the base choice as an ablation either way.

Report it **stratified by range**, because you already predicted the answer differs:
- close objects → expect overshoot (inflation) → free space will bite hard → Tier 1 is live
- far objects → expect wrong depth entirely → §13 is the answer for them
- if there's *no* overshoot anywhere → free space has no signal, pivot to §13 + §14 as the main story

Do this first. Everything downstream depends on it.

**Experiment B — does observability predict error?** *(~2 days)*

Compute all seven observability components (§14) for every object in your 33 ECP GT frames.
Correlate each, and the combined scalar, against per-object IoU-with-GT.

Deliverables: your fusion weight, your stratification axis, and half of Figure 1 — all validated
before you write a line of method code. Pay particular attention to whether **angular coverage**
(component 4) predicts extent error better than raw point count. I expect it does, and that's a small
publishable insight on its own.

### Next week — the probe

**Experiment C — when does SAM3D decide the extent?** *(~3 days)*

Instrument the SS stage. At each step, decode the intermediate voxel grid and measure its bounding
box. Plot dimensions vs. step index across a few hundred objects.

Tells you where guidance must be injected, explains your earlier failure, and is a figure in the
TokenGraph3D style.

### Weeks 3–4 — Tier 1

Build the 9-DoF alignment (§10) with all three constraints, including the occlusion-aware silhouette
(§8). Evaluate with E1 and E3. This is your first real result.

### Weeks 5–6 — Regime C

Implement the §13 cascade: ground-plane anchoring, anchored affine, pairwise propagation,
intra-scene class consistency. Report cascade coverage. This alone may be worth as much as Tier 1 on
nuScenes, where far objects dominate.

### Weeks 7–10 — Tier 2

Guidance inside the SS stage. Flag-guarded, strict no-op when disabled. Compare against Tier 1 —
if guidance beats post-hoc alignment, that's the headline result.

### Later / opportunistic

- Multi-camera merge weighted by observability
- Temporal fusion weighted by observability
- Downstream CenterPoint with weighted loss
- TokenGraph3D ∩ SAM3 for high-precision occupancy points (§19)
- Ensemble-spread analysis (§15)
- FP reduction vs. AutoBox's fence filter (§18)

### One non-technical step

**Read LabelAny3D properly and decide your positioning.** It is the closest published relative of your
pipeline and it is not in your spreadsheet. Add it. A reviewer will find it, so you want to be the
one who framed the relationship. Your differentiators are real — LiDAR instead of monocular
pseudo-depth, driving instead of in-the-wild, and constraining generation rather than aligning after
it — but you must state them yourself.

---

## 24. Literature

Grouped by why it matters. ✅ = already in your project folder.

### The three papers that define your gap

| Paper | Why | Link |
|---|---|---|
| **LabelAny3D: Label Any Object 3D in the Wild** (2026) | Closest published relative. TRELLIS-based auto-labeling. Names your exact gap as future work; shows size priors beat generation on low-visibility objects. **Not in your spreadsheet — add it.** | https://arxiv.org/abs/2601.01676 |
| **AutoBox** (your lab, both versions) | Appendix: occluded objects → undersized boxes, no handling, future work = size priors. Conclusion: self-occlusion limits scale/centre. Fence filter discards 6.8%. | ✅ in project |
| **CPD: Commonsense Prototype for Outdoor Unsupervised 3D Object Detection** (CVPR 2024) | Measures that ~65% of objects lack full scan coverage. Retrieval-based prototypes as the alternative solution. | ✅ in project · code: https://github.com/hailanyi/CPD |

### Generative 3D — the tools

| Paper | Why | Link |
|---|---|---|
| **TRELLIS: Structured 3D Latents** (CVPR 2025) | The architecture underneath SAM3D Objects. Two-stage SS → SLAT. Read §3.3 for the sparse-structure generation. | https://arxiv.org/abs/2412.01506 · https://github.com/microsoft/TRELLIS |
| **SpaceControl: Test-Time Spatial Control for 3D Generative Modeling** | Training-free geometric control of TRELLIS via latent intervention. **Proves the Tier 2 mechanism works.** Read this carefully. | https://arxiv.org/abs/2512.05343 |
| **Diffusion Posterior Sampling** (Chung et al., ICLR 2023) | The general theory of steering a generative model with a measurement likelihood. The maths behind Tier 2. | https://arxiv.org/abs/2209.14687 |
| **GENA3D: Generative Amodal 3D Modeling** | Amodal 3D generation under occlusion. Relevant to §8. | https://arxiv.org/abs/2511.21945 |
| **SAM 3D (Objects / Bodies)** | Your generators. | ✅ in project |

### Shape-prior fitting — your Tier 1 prior art

| Paper | Why | Link |
|---|---|---|
| **SDFLabel: Autolabeling 3D Objects with Differentiable Rendering of SDF Shape Priors** (CVPR 2020) | The original "fit a shape prior to LiDAR + mask." Category-specific, cars. | [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zakharov_Autolabeling_3D_Objects_With_Differentiable_Rendering_of_SDF_Shape_Priors_CVPR_2020_paper.pdf) |
| **Segment, Lift and Fit (SLF)** (ECCV 2024) | Modern version. Gradient descent on pose + shape until projection matches mask and surface matches LiDAR. **Your closest Tier 1 competitor.** No free space. | https://arxiv.org/abs/2407.11382 |
| **Towards Learning to Complete Anything in LiDAR** | Zero-shot shape completion in LiDAR. Note their point that modal recognition only localises the visible part. | https://arxiv.org/abs/2504.12264 |

### Auto-labeling baselines — the ones you compete with

| Paper | Key idea | Link |
|---|---|---|
| **VESPA** (your lab) | VLM-supervised, DBSCAN denoising with LLM-derived class widths, L-shape fitting, multi-camera merging by shared points + border adjacency | ✅ in project |
| **UNION** (NeurIPS 2024) | Appearance-based pseudo-classes, no self-training | ✅ in project |
| **AnnofreeOD** | 2D-to-3D distillation, all 10 nuScenes classes, DINOv2 orientation diagrams, commonsense box refinement | ✅ in project |
| **CM3D / Shelf-Supervised Cross-Modal Pre-Training** | Popularised "shelf-supervised". LLM-sourced amodal sizes, HD-map orientation | ✅ in project · https://arxiv.org/abs/2406.10115 |
| **MODEST** (CVPR 2022) | Persistence/ephemerality score, self-training. Also the source of the closeness-based box fitting AutoBox uses | ✅ in project |
| **OYSTER** (CVPR 2023) | Temporal consistency for near-to-far generalisation | (referenced widely; not in project) |
| **LISO** (ECCV 2024) | Trajectory-regularised self-training, scene-flow based | ✅ in project |
| **LiSe** (ECCV 2024) | LiDAR-2D self-paced learning, adaptive sampling, weak model aggregation. Note their finding that image-based and LiDAR-based boxes *conflict* below 10 m — relevant to your Regime A | ✅ in project |
| **ViLGOD** | First class-aware unsupervised LiDAR detection; handles static objects via CLIP | https://arxiv.org/abs/2408.03790 |
| **OVM3D-Det** | Monocular, LLM size priors. The baseline LabelAny3D compares against | (in your spreadsheet) |

### Offboard / temporal refinement — the crowded area you should avoid

**3DAL, Auto4D, CTRL, DetZero, MPPNet, Immortal Tracker, MS3D++** — all ✅ in project. Read them so
you can say confidently in related work *why* your contribution is orthogonal (they refine tracks of
already-detected objects; you fix the single-frame amodal extent problem that track refinement cannot
solve when the object is never seen complete — which is CPD's 65% point).

### Label uncertainty — §15 and §16 prior art

| Paper | Why | Link |
|---|---|---|
| **GLENet** (IJCV 2023) | Generative label uncertainty via CVAE; plug-and-play for probabilistic detectors. **Your closest §15 competitor.** Needs labels. | https://arxiv.org/abs/2207.02466 |
| **MEDL-U** | Evidential uncertainty for 3D auto-annotation; KL-divergence loss weighting. The template for §16. | https://arxiv.org/abs/2309.09599 |
| **Uncertainty Estimation for 3D Object Detection via Evidential Learning** | Uncertainty-driven verification in an auto-labeling loop | https://arxiv.org/abs/2410.23910 |
| **3DIoUMatch** | IoU-based pseudo-label filtering | ✅ in project |
| **Not Every Side Is Equal** | Per-side localisation uncertainty for SSL 3D detection. Conceptually close to observability being a *vector* not a scalar (§14). | https://arxiv.org/abs/2312.10390 |
| **FixMatch** | Confidence thresholding, the ancestor of all of this | ✅ in project |

### Depth and geometry

| Paper | Why | Link |
|---|---|---|
| **MoGe** | Your relative-depth backbone. §13 C2/C3 depend on understanding its affine-invariance. | https://arxiv.org/abs/2410.19115 |
| **Depth Pro** | Metric depth; LabelAny3D scales MoGe to it | https://arxiv.org/abs/2410.02073 |
| **UniDepth** | Universal metric depth (CVPR 2024) | ✅ in project |
| **CompletionFormer** (CVPR 2023) | Your O4/O5 depth completion | (search title) |
| **MapAnything** | AutoBox's densifier | (in your spreadsheet) |

### Your lab's recent work — read for style as much as content

| Paper | Why | Link |
|---|---|---|
| **TokenGraph3D: Emergent 3D Instance Segmentation from Self-Supervised Point Transformers** (ECCV 2026 DriveX) | §19. The template for how this group frames a contribution. Also a candidate component for your occupancy stage. | ✅ in project · https://arxiv.org/abs/2608.15796 |
| **TerraSeg** | Self-supervised ground segmentation. Used by both AutoBox and TokenGraph3D. Would replace your PseudoLabeler ground fit and keep everything label-free. | ✅ in project |
| **ECP2.0 / ECP2.5** | Your dataset lineage. ECP2.5 also predicts per-object distance *uncertainty* — directly relevant to §14/§16, and it's in-house. | ✅ in project |

### Also worth a look

- **Frustum PointNet** (CVPR 2018) — https://arxiv.org/abs/1711.08488 — the ancestor of in-mask point
  classification with a class one-hot. Relevant to why §6's reframing (precision over recall) matters.
- **Better Call SAL / Segment Anything in LiDAR** (ECCV 2024) and **SAM4D** (2025) — the modern
  promptable LiDAR segmenters. Cite if you touch point segmentation.
- **LOGen** — https://arxiv.org/abs/2412.07385 — LiDAR object generation by point diffusion. Different
  goal but adjacent.

---

## Appendix: the plan in one table

| # | Idea | Risk | Novelty | Role |
|---|---|---|---|---|
| §10 | Tier 1: post-hoc sensor-consistent alignment | Low | Medium | First result; the transferable module |
| §11 | Tier 2: guidance inside the SS generation | Medium | **High** | **The headline** |
| §5–7 | Free space as a first-class signal | Low | **High** | The key technical ingredient |
| §8 | Occlusion-aware silhouette (lower vs upper bound) | Low | Medium | Makes occluded objects work |
| §13 | Reliability-anchored intra-frame metric propagation | Low | Medium-High | Fixes Regime C; enables "no class priors" claim |
| §14 | Observability score (multi-modal, geometric) | Low | Medium-High | Unifies bullets 1, 4, 7; the paper's dial |
| §12 | Regime A handling (truncation, evidence dominance) | Low | Medium | Fixes close-range inflation |
| §18 | Free space as FP / reflection / fence filter | Low | Medium | Concrete win vs. AutoBox |
| §15 | Generative-ensemble uncertainty | Low | Medium | Supporting figure; Tier 2 diagnostic |
| §16 | Observability-weighted detector training | Low | Low-Medium | Extra table |
| §17 | Class label used during inference | Low | Low | Woven throughout |
| §19 | TokenGraph3D ∩ SAM3 for occupancy points | Low | Medium | Phase 3 ablation |
| §11 | Probing the SS stage internals | Low | Medium-High | De-risks Tier 2; standalone figure |

---

# Part II — Clarifications, Justifications, and the Ordered Roadmap

*Added after the second review pass. Sections 25–30.*

---

## 25. Corrections and clarifications

Points that were wrong, imprecise, or under-explained in Part I.

### 25.1 Hull anchoring + updated HDBSCAN is already a weak version of this thesis

> **Note:** the +0.011 / +0.018 mAP improvements come from the `hull` runs, which bundle **two
> changes**: updated HDBSCAN parameters **and** hull depth-floor anchoring. Without an isolated
> ablation, the gain cannot be attributed to hull anchoring alone — the HDBSCAN update may be the
> larger contributor. The directional point still holds: both changes improve how LiDAR evidence
> constrains depth, which is a weak form of the thesis.

Your results:

| Config | ECP 8-class mAP | Δ from hull |
|---|---|---|
| O3 + B1 + agg(2) + filt | 0.1689 | — |
| O3 + B1 + agg(2) + filt + **hull** | 0.1795 | **+0.011** |
| O5 + B1 + agg(2) + filt | 0.1690 | — |
| O5 + B1 + agg(2) + filt + **hull** | **0.1866** | **+0.018** |

Hull anchoring applies a per-mask depth floor from LiDAR cluster evidence. **That is a free-space
statement** — "the object cannot be nearer than this" — applied to the depth map rather than the mesh.

So you have already accidentally validated a degenerate form of the thesis, and it is your single
biggest intervention. This is the strongest available evidence that Experiment A comes back positive.

Your own noted failure — pedestrian regression, "mask-level floor likely too conservative near mask
edges" — is **exactly** what a per-voxel 3D version fixes. One floor per mask is too blunt; per-voxel
ray evidence isn't.

**Paper narrative:** *we observed that a per-mask depth floor derived from LiDAR helps substantially;
we generalise it to a proper ray-based free-space constraint in 3D, which removes the mask-edge
artefacts and extends from the depth map to the generated mesh itself.*

### 25.2 Sweep accumulation — stay single-sweep

Two cases:

- **Static object, ego drives past** → genuinely new surface from new angles. Strictly good. Angular
  coverage rises, occupancy strengthens, free space carves from more directions. This is what CPD and
  Auto4D exploit.
- **Moving object** → smearing. Points are not on the object surface at the anchor timestamp.
  Corrupts occupancy **and** free space (a later sweep carves out where the object was earlier).

Your ablation already decided this: car AP 0.500 (no agg) → 0.287 (MC + agg 6). ECP cars move.

**Decision: stay single-sweep. Do not reopen aggregation.** It also removes smearing from the critical
path entirely and makes free-space maps clean.

**For later, two notes:**
- Free space is the *principled* place to reintroduce accumulation, because log-odds evidence
  accumulates correctly where stacked points don't.
- The carve-out signature (a later sweep certifying empty where an earlier sweep saw a return) is a
  **static/dynamic detector for free** — MODEST's ephemerality idea, obtained from the ray machinery
  rather than a separate module.
- Regime C objects have *zero* returns, and accumulation is the only way to get any. Range-conditional
  accumulation is a legitimate future ablation.

**For Tier 2: denser is better only if correct. Corrupt density is worse than sparse truth.**

### 25.3 Association score is not mAP

TokenGraph3D's 0.273 → 0.517 numbers are an **instance-segmentation association metric** in [0,1]:
how well points belonging to the same GT instance get grouped together, and how well points from
different instances stay apart. HOTA-family in spirit.

No boxes, no confidence ranking, no IoU thresholds, no precision-recall curve. **Not comparable to any
mAP number in your tables.** Check the paper's metric definition section for the exact formula before
citing it.

### 25.4 TRELLIS vs. SAM3D Objects — they are not alternatives

**SAM3D Objects is built on the TRELLIS architecture.** Not a competitor — the ancestor. The SAM3D
Objects paper describes improving on "the original SLAT VAE design in Xiang et al." — Xiang et al. *is*
the TRELLIS paper — and names their variant Depth-VAE.

You use SAM3D Objects, which uses TRELLIS's two-stage SS → SLAT design with modifications.

**Why this is good news:**
- **SpaceControl controls TRELLIS.** Shared architecture ⇒ direct evidence Tier 2 works on your model.
- **LabelAny3D uses TRELLIS directly.** Their documented failure modes are your failure modes, which
  is why you can cite their limitations as motivation for your design choices.
- "Swap TRELLIS for its successor" meant the method binds to the *architecture family* (two-stage
  sparse-voxel-then-latent), not to a checkpoint. TRELLIS, SAM3D Objects, Hunyuan3D all qualify.

### 25.5 "Plot against observability bins" — what that means

> **Two unrelated uses of "bin" in this document.** TRELLIS voxel cells (the 64×64×64 grid in SS
> Stage 1, where each cell is occupied or empty) define the object's 3D shape. **Observability bins**
> are statistical buckets for grouping objects when making a plot — a pure evaluation/analysis
> technique with no connection to TRELLIS. The word "bin" below always means statistical bucket.

1. Compute observability per object → a scalar in [0,1].
2. Sort objects into bins: [0.0–0.2), [0.2–0.4), [0.4–0.6), [0.6–0.8), [0.8–1.0].
3. Within each bin, average ASE (and separately ATE) over the objects in that bin.
4. Plot bin centre on x, mean ASE on y. One line per method.

You are plotting **per-bin means**, not a per-object scatter. It shows *where along the difficulty
spectrum* each method wins.

**Overlay a histogram of object counts per bin behind the curves.** A bin containing three objects is
noise and the reader must be able to see that.

### 25.6 One result to sanity-check before citing

`O5 + B1 + agg(2) + filt + hull` at ECP 1-class: mAP **0.3696** (best front-cam) but ATE **0.800** and
AOE **0.924** — wildly out of line with every neighbouring row (ATE 0.06–0.26 elsewhere).

Either an eval artefact or a systematic shift that happens to sit inside the 4 m threshold and so
inflates mAP while wrecking the TP error metrics. **Check this before it appears in a thesis table.**

### 25.7 Is this a real contribution, or engineering?

Honest answer: **the idea is methodologically sound; whether the paper is a contribution depends on
execution discipline.**

**What makes it real:**
- A **principled formulation** — posterior sampling under a measurement likelihood,
  `p(shape | image, LiDAR) ∝ p_gen(shape | image) · p(LiDAR | shape)`. A statement, not a heuristic.
- **One physical insight** — free space is a measurement everyone discards.
- It **removes a prior** (class size tables) rather than adding a module.
- It **answers a published open problem** (LabelAny3D's stated future work) and two limitations your
  own lab has written down (AutoBox conclusion + appendix).
- It is **falsifiable in two days** (Experiment A).
- Hull anchoring is **early empirical evidence** the direction has signal.

**What would make it hacky:**
- Twelve terms with twelve tuned weights.
- Implementing §12–§18 as parallel special cases rather than consequences of one mechanism.
- Wins that come from tuning rather than from the free-space term specifically.

> **The discipline test:** you must be able to run one ablation that removes free space and watch the
> effect disappear. If you can't, you built a system, not a discovery. Everything else — observability,
> merging, temporal fusion, FP rejection — must be a **consequence of the same ray machinery**, never
> a parallel heuristic. That distinction is decided by *how* you build it, not by *what* you build.

---

## 26. Design choice → literature justification

Every non-obvious decision, with the paper and the specific failure mode that justifies it. This is
the table you build your thesis's motivation section from — each row is "someone documented this
problem, therefore we did this."

| Design choice | Justified by | The specific statement |
|---|---|---|
| **Free-space constraint at all** | AutoBox appendix; CPD | Occluded objects → undersized boxes, *no handling in current implementation*; ~65% of objects lack full scan coverage |
| **Generative prior instead of class size tables** | AnnofreeOD, CM3D, VESPA, OVM3D-Det | All require external per-class size statistics; AnnofreeOD notes LiDAR-only methods are limited to ~3 classes because classification relies on box shape and size rules |
| **Anisotropic scale, with extra freedom along camera Z** | LabelAny3D limitations | Generated meshes have ambiguous depth *along the viewing direction*, misaligning with the RGBD cloud. A **directional** error — uniform scale cannot fix it |
| **Guidance instead of post-hoc voxel editing** | LabelAny3D future work; SpaceControl | "Future work could condition 3D generation on RGBD data"; SpaceControl proves training-free latent intervention works on this exact architecture |
| **Occlusion-aware silhouette upper bound** | AutoBox appendix | Partial point clouds produce undersized boxes; enforcing the modal mask as an upper bound reproduces that error |
| **Continuous observability blend, not a binary switch** | LabelAny3D | Their generative pipeline loses to size priors on low-visibility objects (nuScenes 6.69 vs 8.50 AP₃D); their fix is a hard visibility-threshold ensemble |
| **Precision-over-recall point core** | Your own hull-anchoring result | Pedestrian AP regression, "mask-level floor likely too conservative near mask edges" — the fit needed points that erosion removed |
| **Surface-touching (two-sided) occupancy loss** | Physics + your Regime A observation | Returns are surface samples; an inflated mesh contains all its points strictly inside |
| **Free space for FP rejection** | AutoBox appendix | Fence filter discards 6.8% of mask evaluations (up to 30% in some scenes); reflections explicitly unhandled |
| **Intra-scene class size transfer, not lookup tables** | VESPA, CM3D | Both query an LLM for per-class dimensions — external, fixed, closed-set |
| **Single-sweep, no aggregation** | Your own ICP/MC ablation | Car AP 0.500 → 0.287 with MC + agg(6); every multi-sweep variant loses |
| **Ray-based static/dynamic (later)** | MODEST | Persistence/ephemerality via repeated traversal |
| **Full 360° point cloud for truncated objects** | Geometry + AutoBox panoramic motivation | Camera FOV is a subset of LiDAR FOV; truncation loses image evidence but not LiDAR evidence |
| **Class label used during inference, not just at output** | Frustum PointNet | Class conditioning improves 3D localisation from 2D regions — established since 2018 |

**Use this table actively while writing.** A design choice with an empty right-hand column is either
unmotivated or you haven't found its justification yet. Both are worth knowing.

---

## 27. Claude Code handoff — initial setup prompt

The SS correction audit prompt that was here has been completed. Results are in §31.

**Below is the prompt to paste into Claude Code in VS Code to begin Phase 0.** It contains the full
context Claude Code needs: what the project is, where the code lives, what to build first, and the
rules of engagement.

```
# Context: Sensor-Constrained Generative Auto-Labeling

Read the file at /media/leander/ECP_Nuscenes_01/Thesis/Contribution/contribution_plan.md — it is the
full plan document for this work. Sections 0–9 are concepts, 10–19 are the things to build,
20–24 are practical, 25–31 are clarifications and the ordered roadmap. Read all of it before
doing anything.

## What this project is

I have a working auto-labeling pipeline for 3D object detection in autonomous driving. It uses:
- SAM3 for 2D instance segmentation (masks + class labels)
- SAM3D Objects (built on the TRELLIS architecture) for amodal 3D mesh generation per object
- SAM3D Body for pedestrian pose estimation
- MoGe for monocular depth estimation (affine-invariant)
- CompletionFormer for LiDAR depth completion
- HDBSCAN for in-mask LiDAR point filtering
- TerraSeg (PseudoLabeler + binary ground filter) for ground removal

The pipeline runs per-frame, per-camera. It produces oriented 3D bounding boxes (OBBs) from
the generated meshes. Current best config is O3 (MoGe + per-object local affine calibration
using HDBSCAN-filtered in-mask LiDAR) with B1 (LiDAR-corrected pedestrian translation).

## What we are building

A contribution based on using LiDAR free-space evidence (rays that pass through a volume and
return from further away prove that volume is empty) to constrain the generative 3D
reconstruction. The full plan with all sub-ideas, constraints, regimes, evaluation, and
literature is in the contribution_plan.md.

## Directory layout

```
/media/leander/ECP_Nuscenes_01/Thesis/
├── AutoLabeling/          # The existing full pipeline — DO NOT MODIFY until Phase 5+
│   ├── src/autolabeling/  # Pipeline source
│   │   ├── pipeline.py
│   │   ├── models/
│   │   │   ├── sam3d_objects.py
│   │   │   └── ...
│   │   └── ...
│   ├── configs/           # nuscenes.yaml, ecp.yaml
│   └── ...
├── Models/
│   └── SAM3D/
│       └── sam-3d-objects/
│           └── sam3d_objects/
│               └── pipeline/
│                   └── inference_pipeline_pointmap.py  # SS→SLAT intercept
├── Testing/
│   └── autolabeling_pipeline.ipynb  # Existing test notebook
├── Contribution/          # NEW — all experimental work goes here
│   ├── contribution_plan.md
│   └── notebooks/         # Create this — all Phase 0/1/2/... experiments
└── Data/                  # Scene data, will be specified per experiment
```

## Rules

1. **All new code goes under /media/leander/ECP_Nuscenes_01/Thesis/Contribution/.** Do not touch
   AutoLabeling/, Models/, or Testing/ until explicitly told to integrate.
2. **Use Jupyter notebooks (.ipynb)** for all experiments. One notebook per Phase or sub-phase.
3. **The existing pipeline code is reference only.** Import from it, inspect it, but don't
   edit it. Copy what you need into Contribution/.
4. **Nothing is trained.** All models are frozen. No gradient updates to any model weights.
   Per-object optimisation at inference time is fine (Tier 1). Guidance gradients flow to the
   latent, never to weights.
5. **Start with Phase 0 from the roadmap (§28).** The first deliverable is a global per-frame
   occupancy/free-space voxel map built by ray-casting the raw LiDAR sweep.
6. I work on two datasets: **ECP** (64-beam Velodyne, 3 front-arc cameras, 33 GT frames from
   3 Strasbourg scenes) and **nuScenes mini** (32-beam Velodyne, 6 cameras 360°, 10 scenes
   Boston + Singapore). Both are active — build everything to run on either. Start
   experiments on ECP (denser LiDAR, better chance of seeing free-space signal), then
   immediately verify on nuScenes mini (sparser LiDAR, the stress case that tells us if
   32-beam is enough).
7. When I say "visualise," I mean inline in the notebook — matplotlib, open3d if needed for
   3D, or save to file and display.

## Phase 0 — first task

Create a notebook `Contribution/notebooks/phase0_freespace.ipynb` that:

1. Loads one ECP frame AND one nuScenes mini frame (I will specify which — for now set up
   the loading scaffold for both datasets using the same data loading utilities the existing
   pipeline uses, look at Testing/autolabeling_pipeline.ipynb for how frames are loaded).
   Make the dataset selectable via a config cell at the top of the notebook.
2. Gets the raw LiDAR sweep (all points, NOT ground-filtered, NOT visibility-filtered)
3. Implements ray-casting into a voxel grid with log-odds accumulation:
   - For each LiDAR beam: trace from sensor origin to return point
   - Mark traversed voxels as FREE (negative log-odds)
   - Mark the return voxel as OCCUPIED (positive log-odds)
   - Voxel size: start with 0.2 m, make it configurable
   - Use a simple dense 3D array or a sparse representation if memory is tight
4. Visualises the result:
   - BEV (bird's eye view) slice at vehicle height (~1 m above ground) showing
     EMPTY (blue) / OCCUPIED (red) / UNKNOWN (grey)
   - Overlay GT boxes if available
   - Overlay camera FOV frustums
5. For 3–5 specific objects (I will specify mask indices later, for now just pick objects
   that have GT boxes): compute and display the free-space / occupied / unknown volume
   breakdown within each object's GT box footprint

The ray-casting should use Bresenham's line algorithm or similar for voxel traversal (e.g.
Amanatides & Woo fast voxel traversal). Don't use a library that hides the ray logic — we
need to understand and potentially modify it later.

After Phase 0.1-0.2 are working on ECP, **repeat on a nuScenes mini frame** — this is
critical because 32-beam may not have enough rays to produce useful free-space evidence.
The side-by-side comparison (64-beam vs 32-beam free-space maps) is itself an important
early result. Then we'll do Experiment A (overshoot vs undershoot) on both datasets.

Start by reading the contribution_plan.md, then set up the notebook structure. Ask me for
the specific ECP scene/frame paths when you're ready to load data.
```

### After the initial setup

When Phase 0 is working, come back to this conversation (or start a new one in this project)
to discuss results and plan next steps. The roadmap in §28 has the full sequence. Each phase
can be handed to Claude Code with a prompt like:

```
Continue with Phase 1 from contribution_plan.md (§28). The Phase 0 notebook is at
Contribution/notebooks/phase0_freespace.ipynb — read its outputs first. Create a new
notebook Contribution/notebooks/phase1_observability.ipynb for this phase.
```

### What Claude Code needs to know about the existing pipeline

The most important files to read (as reference, not to edit):

| File | What it tells you |
|---|---|
| `AutoLabeling/src/autolabeling/pipeline.py` | The full frame processing loop |
| `AutoLabeling/src/autolabeling/models/sam3d_objects.py` | How SAM3D Objects is called, how pointmaps are built, the HDBSCAN filtering, the ss_correction closure |
| `Models/SAM3D/sam-3d-objects/sam3d_objects/pipeline/inference_pipeline_pointmap.py` | The SS → SLAT intercept point (L428–473) |
| `Testing/autolabeling_pipeline.ipynb` | How data is loaded, how the pipeline is invoked |
| `AutoLabeling/configs/ecp.yaml` and `nuscenes.yaml` | All configuration parameters |

---

## 28. The ordered roadmap

Everything from this document, sequenced. Check items off as you go.

Notation: **[E]** experiment · **[A]** ablation · **[P]** plot/figure · **[T]** table ·
**[C]** code · **[W]** writing

---

### Phase 0 — Go/no-go diagnostics (week 1)

*Purpose: decide whether the free-space direction is live before building anything.*

| # | Task | Type | Notes |
|---|---|---|---|
| 0.1 | Build the global per-frame occupancy/free-space map: ray-cast the raw sweep into a voxel grid with log-odds accumulation | [C] | Object-agnostic, mask-agnostic. **Do not ground-filter the cloud used for casting.** ~150 lines. Everything depends on this. |
| 0.2 | Visualise it for 3–5 frames — EMPTY / OCCUPIED / UNKNOWN as three colours, with GT boxes overlaid | [P] | Sanity check. You should *see* object shadows. |
| 0.3 | **Experiment A**: for every TP, is the predicted box overshooting into certified-EMPTY voxels, or undershooting the GT extent? | [E] | Run on **both O3 and O5**. Stratify by range and by the §3 2×2 cell. |
| 0.4 | Plot overshoot/undershoot distribution per class and per cell | [P] | The go/no-go figure. |
| 0.5 | Compare **recall** (not AP) between O3 and O5 | [T] | Decides whether O5's extra detections are recoverable FPs or noise. |
| 0.6 | Run the §27 Claude Code audit of `ss_correction` | [E] | Confirms or refutes the Tier 2 diagnosis. Cheap, do it in parallel. |
| 0.7 | Sanity-check the anomalous `O5+hull` 1-class row (§25.6) — focus on the ATE=0.800, not AOE (orientation is not estimated) | [E] | Before it reaches a thesis table. |
| 0.8 | **Actually run `ss_correction: true`** on 10–20 objects across the three regimes, visualise the meshes before/after | [E] | The audit found it was likely never tested. Two hours. |
| 0.9 | If 0.8 shows half-carving (likely): prototype **Tier 1.25** — replace the scalar `median(Z_anch)` with per-voxel three-constraint scoring on the SS grid | [C][E] | Reuses Phase 0.1's free-space map. Cheaper than full Tier 2 guidance. |

**Decision gate.** Mostly overshoot → free space has signal → proceed to Phase 1 as written.
Mostly undershoot → free space can't help directly → §13 and §14 become the main contribution and
Tier 1/2 drop to supporting roles. Either way you have a plan; you just need to know which.

---

### Phase 1 — Observability and the evidence taxonomy (week 2)

*Purpose: build the measurement that everything else uses.*

| # | Task | Type | Notes |
|---|---|---|---|
| 1.1 | Implement mask-boundary classification: real / occlusion / truncation (§8) | [C] | Dilate, inspect ring, compare depths. Feeds both silhouette and observability. |
| 1.2 | Implement the high-precision point core (§6), including the **NOT-in-nearer-mask** filter | [C] | Add the nearer-mask filter first — it's two lines and attacks foreground bleed directly. |
| 1.3 | Implement the full-point-cloud object set: seed from in-mask core, region-grow in 3D beyond the camera FOV (§12 A0) | [C] | Required for Regime A and for camera seams. |
| 1.4 | Implement all 7 observability components (§14), including **angular coverage** | [C] | Keep the vector; also compute the product scalar. |
| 1.5 | **Experiment B**: correlate each component, and the scalar, against per-object IoU-with-GT | [E] | On the 33 ECP GT frames. |
| 1.6 | Specifically test: does angular coverage predict extent error better than raw point count? | [E] | If yes, a small publishable insight on its own. |
| 1.7 | Assign every GT-matched object to a §3 2×2 cell; report the population of each cell | [T] | Establishes that cells A/C/D are not edge cases. |
| 1.8 | Quantify cell D: how many of your FPs live there? | [T] | Directly explains the O5 459→933 FP result. |

---

### Phase 2 — Tier 1, post-hoc sensor-consistent alignment (weeks 3–4)

*Purpose: your first real result, and the transferable module.*

| # | Task | Type | Notes |
|---|---|---|---|
| 2.1 | Implement the two-sided occupancy term with per-class tolerance band τ (§7.1) | [C] | The revised, surface-touching version. |
| 2.2 | Implement the free-space term with log-odds weights (§7.2) | [C] | Query the Phase 0 global map. |
| 2.3 | Implement the occlusion-aware silhouette term (§7.3, §8) | [C] | Lower bound always; upper bound substituted per boundary type. |
| 2.4 | Implement the 9-DoF optimiser (rotation, translation, anisotropic scale) | [C] | Extra freedom along camera Z, justified by LabelAny3D (§26). Start with CMA-ES if gradients are painful. |
| 2.5 | Run on your O3 base; evaluate | [E] | |
| 2.6 | **E1 — the money figure**: ASE and ATE vs. observability bins, three curves (box-fit baseline / unconstrained generation / constrained), with the count histogram behind | [P] | §25.5 explains the construction. This is Figure 1 of the paper. |
| 2.7 | **E3 — the isolation ablation**: none → +occupancy → +free space → +silhouette → +occlusion-aware silhouette | [A][T] | If free space isn't where the jump is, the claim is wrong. This is the discipline test (§25.7). |
| 2.8 | Repeat E1/E3 stratified by the §3 2×2 cell | [P][T] | Shows the mechanism fixes the cells it should. |
| 2.9 | Apply Tier 1 to **VESPA's** boxes | [E] | The modularity claim depends on this. |
| 2.10 | Apply Tier 1 to your **O5** boxes | [E] | Third pipeline. |
| 2.11 | Build the class-prior-inflation proxy baseline (§E2) | [C][E] | Stands in for AutoBox / AnnofreeOD, which you can't run. |
| 2.12 | Compare against your existing **hull anchoring** — is the 3D version strictly better, especially for pedestrians? | [A][T] | Directly tests the §25.1 narrative. |

---

### Phase 3 — Regime C, metric propagation (weeks 5–6)

*Purpose: fix the cell your current base silently drops. Load-bearing, not optional.*

| # | Task | Type | Notes |
|---|---|---|---|
| 3.1 | Implement C1 ground-contact anchoring with all four mitigations (§13) | [C] | Full mask, robust bottom statistic, occlusion check, cross-validation. |
| 3.2 | Implement C2 anchored global affine — fit `(a,b)` on high-observability objects only, robust regression in disparity space | [C] | |
| 3.3 | **The O2 ablation you asked for**: O2 (all points) vs. O2-anchored (reliable objects only) | [A][T] | Honest explanation of an earlier negative result. Explicitly requested. |
| 3.4 | Implement C3 pairwise relative-depth propagation | [C] | |
| 3.5 | Implement C4 intra-scene class-size transfer | [C] | This is what licenses the "no external size priors" claim. |
| 3.6 | Implement C5 propagation confidence, feed into observability | [C] | |
| 3.7 | Wire the cascade: LiDAR return → C1 → C2/C3 → C4 → external prior (flagged) | [C] | Replaces O3's silent local-affine failure. |
| 3.8 | **Cascade coverage table**: what fraction of objects each stage handles | [T] | Explicitly requested. Nobody has quantified how far you get before needing a size table. |
| 3.9 | Measure: does the cascade recover the objects O3 was silently dropping? | [E][T] | Recall change, and FP change. |
| 3.10 | Re-run the O3 vs. O5 base decision now that Regime C is handled | [E] | The §23 decision, properly informed. |

---

### Phase 4 — Tier 2, guidance inside generation (weeks 7–10)

*Purpose: the headline.*

| # | Task | Type | Notes |
|---|---|---|---|
| 4.1 | **Experiment C — the probe**: at each SS denoising step, decode the intermediate grid and measure its bounding box | [E][P] | Tells you *where* to inject. Do this before writing guidance code. |
| 4.2 | Plot extent dimensions vs. step index across a few hundred objects | [P] | Standalone figure in the TokenGraph3D style. |
| 4.3 | Implement the measurement likelihood as a differentiable function of the SS latent | [C] | Reuse the §7 terms. |
| 4.4 | Implement guidance with a strength schedule (weak → strong ramp) | [C] | Flag-guarded; strict no-op at strength 0. |
| 4.5 | **Tier 1.5 fallback if gradients are painful**: generate K samples, score with §7, keep the best | [C] | No gradients, same principle, still novel. |
| 4.6 | Evaluate Tier 2 vs. Tier 1 vs. baseline on E1/E3 | [E][P][T] | |
| 4.6b | **Hybrid row: Tier 1 applied on top of Tier 2's output**, vs. Tier 2 alone | [A][T] | Tier 2's guidance is deliberately damped/regularised to stay in-distribution, so it likely undercorrects; Tier 1 has no such constraint and can polish the residual for free (identity-init, provably non-destructive). Predicted ordering: hybrid ≥ Tier 2 alone ≥ Tier 1 alone — keep all three rows separate so the paper can attribute gains correctly rather than only shipping the hybrid. |
| 4.7 | **Guidance strength ablation** | [A][P] | Shows it's a controlled knob, not a lucky setting. |
| 4.8 | **Ensemble spread before vs. after constraining**, per observability bin (§15) | [E][P] | Direct evidence the constraint collapsed the hypothesis space. No baseline can produce this figure. |

---

### Phase 5 — Consequences and extras (weeks 11+)

*Purpose: show one mechanism has many implications. Every item here must be a consequence of the ray
machinery, never a parallel heuristic (§25.7).*

| # | Task | Type | Notes |
|---|---|---|---|
| 5.1 | FP rejection: reflections (frustum certified empty) and fence-occluded objects (§18) | [E][T] | Compare against AutoBox's 6.8% discard rate. |
| 5.2 | Multi-camera merge weighted by observability | [E][T] | Vs. your two earlier heuristics (bigger mask / more points), which are single components of the same vector. |
| 5.3 | Temporal fusion of extent weighted by observability | [E][T] | Application of the score, *not* a new tracker. |
| 5.4 | Downstream CenterPoint: uniform vs. observability-weighted loss (§16) | [E][T] | Drop silently if it doesn't help. |
| 5.5 | Cross-dataset: ECP (64-beam) and nuScenes (32-beam), **one config, no retuning** | [E][T] | Where training-free pays off. |
| 5.6 | TokenGraph3D ∩ SAM3 masks as the high-precision occupancy core (§19) | [A][T] | Two independent modalities agreeing. Ablation row vs. HDBSCAN. |
| 5.7 | Range-conditional accumulation for Regime C only | [A] | The one legitimate reason to reopen aggregation. |
| 5.8 | Ray-based static/dynamic detection from carve-out signatures | [E] | Falls out of the machinery; MODEST's idea for free. |

---

### Writing, in parallel throughout

| # | Task | When |
|---|---|---|
| W.1 | Read LabelAny3D properly; add to the Excel; write the positioning paragraph | Week 1 |
| W.2 | Fill in §26's justification table as you make each design choice | Continuous |
| W.3 | Related work: why offboard/temporal refinement (3DAL, Auto4D, CTRL, DetZero, MPPNet) is **orthogonal** — they refine tracks of detected objects; you fix single-frame amodal extent, which track refinement cannot solve when the object is never seen complete (CPD's 65%) | Weeks 3–4 |
| W.4 | Related work: differentiate from SDFLabel and SLF — open-vocabulary prior, free space, multi-class sparse driving LiDAR | Weeks 3–4 |
| W.5 | Related work: differentiate from GLENet / MEDL-U — training-free, annotation-free uncertainty | Weeks 7–8 |
| W.6 | Method section from the posterior-sampling formulation, not from the implementation | Weeks 7–10 |

---

### The minimum viable paper

If time runs short, this is the smallest set that still constitutes a contribution:

- Phase 0 complete (the free-space map exists and Experiment A is positive)
- Phase 1 complete (observability exists and predicts error)
- Phase 2 items 2.1–2.9 (Tier 1 works, on your pipeline **and** VESPA)
- Phase 3 items 3.1–3.3, 3.7, 3.8 (Regime C cascade + the coverage table)
- Figures: E1 (money figure), E3 (isolation ablation), cascade coverage table
- Tier 2 as "future work," honestly labelled

That is a complete, defensible paper without ever touching the SAM3D source code. Tier 2 upgrades it
from good to strong, but it is not load-bearing.

---

## 29. Quick reference — what each constraint needs

A recurring source of confusion: different constraints want differently preprocessed inputs. This
table is the answer.

| Constraint | Mask to use | Points to use | Ground-filtered? |
|---|---|---|---|
| **Free-space map** (§5) | none — object-agnostic | **entire raw sweep**, all returns | **No** — ground returns are excellent free-space evidence |
| **Occupancy core** (§6) | eroded | high-precision core only, incl. out-of-FOV (§12 A0) | **Yes** — TerraSeg binary |
| **Silhouette lower bound** (§8) | full mask | — | — |
| **Silhouette upper bound** (§8) | full mask + occluder union | — | — |
| **Ground contact** (§13 C1) | **full** mask, bottom 5–10% | — | uses TerraSeg **PseudoLabeler** surface |
| **Anchored affine** (§13 C2) | — | high-observability objects' cores only | Yes |
| **Observability** (§14) | full mask + all other masks | core count + angular spread | Yes for counting |

---

## 30. Open questions to resolve as you go

Things neither of us knows yet. Worth tracking explicitly.

1. **Does the free-space map have enough certified-EMPTY volume around typical objects at 32 beams?**
   Phase 0.2's side-by-side visualisation (ECP 64-beam vs nuScenes mini 32-beam) answers this
   directly. If 32-beam is too sparse, constraint weights may need to be range- or density-adaptive
   rather than fixed.
2. **Is extent decided early or late in SS sampling?** Phase 4.1. Determines whether guidance is
   feasible.
3. **Does angular coverage beat point count as an error predictor?** Phase 1.6.
4. **Is the τ tolerance band (glass penetration) actually needed, or is HDBSCAN already removing
   interior returns?** Check empirically before adding a hyperparameter.
5. **Where does VESPA's box-fitting fail relative to yours?** Phase 2.9 tells you whether Tier 1
   transfers or whether it's exploiting something specific to generative meshes.
6. **Can free space alone reject cell-D objects, or do you need an explicit rejection rule?**
   Phase 5.1.
7. **Does the O3 local-affine fallback, once replaced by the §13 cascade, make O3 strictly dominate
   O5?** Phase 3.10.

---

## 31. SS Correction audit findings (from Claude Code)

*Audit completed. Key findings summarised here for reference.*

### Architecture

The correction sits at the correct intercept point — **after** `sample_sparse_structure()` finishes
all denoising steps, **before** `sample_slat()`. SLAT allocates a fresh noise tensor sized to the
post-correction coordinate count and starts from scratch. **There is no stale latent issue.**

### What it does

Hard-deletes rows from the `(N, 4)` coords tensor. Binary: a voxel survives or is removed. No soft
weighting, no additions, no logit modification. Non-differentiable (the entire forward pass runs
under `torch.no_grad()`).

### The diagnosed flaw

`Z_surface = float(np.median(Z_anch))` — a **single scalar** per object. The suppression rule:
`vox_Z > Z_surface + 0.3 m → delete`. For any object with depth extent (which is every object not
perfectly perpendicular to the camera), this simultaneously:

- **Over-suppresses** legitimate far-face voxels (behind the median + margin),
- **Under-suppresses** hallucinated voxels near the median depth.

The effective failure mode is *half-carving* — shaving off the back of every angled object.

### Asymmetric trust

Partially implemented. Only LiDAR-anchored pixels participate in suppression (`binary_mask &
anchor_mask`). CFormer-interpolated pixels are excluded. The confirmation path (adding voxels) was
planned but never built. Suppression-only polarity.

### Was it ever tested?

**Probably not.** Both configs have `ss_correction: false`. No saved outputs, no before/after
comparisons, no evaluation results, no failure diagnosis anywhere in the codebase. The feature is
listed as "Implementation in progress" in `lidar_integration_plan.md`. The only log output is a
print statement that no notebook or results file quotes.

### Implication for the plan

1. **Run it.** Set `ss_correction: true`, visualise 10–20 objects. This has likely never been done.
2. **If it half-carves (expected):** upgrade the evidence model from scalar-median to per-voxel
   three-constraint scoring (**Tier 1.25**). Same intercept point, same architecture, much better
   evidence. This is now Phase 0.8–0.9 in the roadmap.
3. **The motivation for Tier 2 shifts** from "post-hoc breaks SLAT" (incorrect) to "post-hoc can
   only subtract; guidance can steer" (correct, weaker, but still real and defensible).
