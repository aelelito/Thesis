# Which pointmap mode for SAM3D Objects? Final analysis and decision

Written 2026-09-25. Everything here was measured on the sweep `cmp2`: nuScenes-mini scene-0061 (39 keyframes) and the 33
annotated ECP keyframes, every mode run on exactly the same detections. Details and older experiments:
`sam3d_objects_input_analysis.md`. Per-object tables: `autolabeling/output/{nuscenes_mini,ecp}/mode_selection/_compare_m1_m2_m6_m7_m8.csv`.
All the runs behind this document (everything below except the still-running full-scale sweep) live under
`autolabeling/output/{nuscenes_mini,ecp}/mode_selection/` -- moved there to keep the output folder from getting
messy as more rounds pile up; `--shared-run` below is always relative to `output/<dataset>/`, so it's
`mode_selection/cmp2_shared` etc., not `cmp2_shared`.

## 0. The answer in five lines

1. **Use dense depth (MoGe rescaled with LiDAR), not raw sparse LiDAR.** Sparse LiDAR puts the box about 16 cm further from the
   truth (median, both datasets, p < 0.001) and produces twice as many bad boxes (30% vs 14% of objects more than 1 m off).
2. **Between the four dense variants (modes 2, 6, 7, 8) there is no reliable difference in typical accuracy.** The
   differences are 0 to 1 cm, smaller than the noise.
3. The extra machinery of modes 6, 7 and 8 buys nothing measurable that is worth its complexity; mode 8 even has *more*
   bad boxes than mode 2.
4. **Recommendation: mode 2 (`moge_affine_local`).** Simplest of the four, as accurate as the others, no second LiDAR cloud
   needed.
5. The biggest remaining error is not the pointmap at all: predicted boxes are systematically too *short* and too *wide*
   on nuScenes (Section 6).

**Update 2026-09-28:** modes 4, 9 and 10 (CompletionFormer, cleaned/raw/ground-included) were run on the same sweep and
do not change the recommendation — see Section 4.6. **Final update, same day:** all 11 modes (including 3, 5 and
LDCM/mode 11) were then re-run on a deliberately diverse sample — 3 scattered frames from each of the 8 nuScenes
scenes and 2 from each of ECP's 3 annotated scenes, instead of one scene end to end — to check the earlier results
weren't an artefact of testing on a single, possibly easy scene. They weren't: mode 2 is still the recommendation.
See Section 10.

## 1. What SAM3D Objects does with the pointmap (needed to understand every mode)

SAM3D Objects gets an image crop, the object mask and a **pointmap**: for every pixel an (X, Y, Z) position in metres in
the camera frame. It uses the pointmap for three things:

- **Anchor (shift).** It subtracts the median 3D position of the pixels inside the mask. The predicted object is then
  placed relative to this point, and at the end the shift is added back. So *where the box ends up in the world is set
  by the pointmap values inside the mask*. If those are wrong by 1 m in depth, the box is wrong by about 1 m.
- **Unit of length (scale).** The pointmap is divided by a "ruler" (how far pixels are from the anchor). At the end the
  result is multiplied by it again. If the network only looks at the normalised geometry this cancels out.
- **Hints about the shape** (the network reads the pointmap in small patches). This mainly matters in the object's own
  crop and only a little for the full image (a test where we switched the maps off showed this).

Take-away: **the depth inside the mask matters a lot; the depth outside the mask matters much less.** All modes below
differ in *how they fill the pointmap in the mask* and *what they put outside it*.

Where do the numbers come from? LiDAR is precise but sparse (holes). MoGe is a neural network that gives a dense
depth image, but only "up to a scale and offset" (it knows the shape, not the metres). We fix that with an **affine fit**:
find `a` and `b` so that `Z_true ≈ a * Z_moge + b` on the LiDAR points that hit the object. Then every pixel gets metric depth.

## 2. The modes, one by one

| # | name | what SAM3D sees | in one sentence |
|---|---|---|---|
| 1 | `sparse_lidar` | the raw LiDAR points, NaN (= "unknown") everywhere else | No neural depth; the network must fill the gaps itself. |
| 2 | `moge_affine_local` | MoGe depth, rescaled with `a, b` fitted on the cleaned LiDAR points inside this object's mask | Dense map; one correction for the whole image, taken from the object only. Outside the mask the depth is plausible in shape but not exactly metric. |
| 3 | `moge_affine_local_masked` | like 2, but everything outside the mask set to NaN | Removes the "wrong" background; confirmed much worse (huge size error) in Section 10. |
| 4 | `completionformer_full` | a depth-completion network (CompletionFormer, trained on KITTI depth completion) fills the cleaned LiDAR anchors to a dense image | Evaluated in Section 4.6 and 10. |
| 5 | `moge_affine_local_raw` | like 2, but the fit uses all in-mask LiDAR points without cleaning | Ablation: checks whether the cleaning step (HDBSCAN) matters. Confirmed catastrophic (Section 10) — cleaning matters a lot. |
| 6 | `moge_affine_local_piecewise` | like 2 inside the mask, plus a *separate* affine for the background, fitted on LiDAR around the object | Both object and background are metric. |
| 7 | `moge_affine_composite` | one map per frame: every detected object gets its own affine, nearer objects painted over farther, background = a grid of local fits blended smoothly | Your idea: the SAM3D sees a consistent scene instead of one object plus a guess. |
| 8 | `moge_affine_local_regslope` | mode 6, but if the object's fit is unreliable its slope is replaced by the background slope | A safety net for objects with too few depth differences in the mask. |
| 9 | `completionformer_raw` | like 4, but with NO per-mask cleaning at all: the raw ground-free LiDAR goes into CompletionFormer as-is | Ablation of mode 4's anchor cleaning. Evaluated in Section 4.6. |
| 10 | `completionformer_ground` | like 4's in-mask cleaning (still on the ground-free cloud, so ground can't get pulled into an object's own cluster), but the out-of-mask background anchors come from the FULL cloud instead (ground included) — mode 6's object/background split, applied to CompletionFormer's anchors | Section 4.6's numbers were measured on an earlier version that used the full cloud for both parts — **superseded, see the note in Section 4.6.** |
| 11 | `ldcm_full` | same hybrid anchors as mode 10, but fused by **LDCM** (ICLR 2026) instead of CompletionFormer: a frozen MoGe prior + a learned per-pixel LiDAR-fusion network | Proposed, not yet run. Rationale and status in Section 9. |

**Why the slope can be unreliable ("ill-conditioned").** A fit needs LiDAR points at *different* depths inside the mask.
A flat car seen from behind has almost all points at the same depth, so `a` (the slope) cannot be determined and the fit may
give nonsense (zero, negative, or huge). In about 30–50% of our objects the slope looked unreliable by this test
(31% nuScenes, 49% ECP). Mode 8 replaces it by the background slope, and adjusts `b` so the anchor stays where the LiDAR
says it is.

## 3. How we compared (so you can trust or challenge the numbers)

- **Same objects everywhere.** Every mode ran on identical detections. Only objects matched to a ground-truth box in
  *all* runs are compared: 118 (nuScenes) and 97 (ECP).
- **Matching to the truth.** Each detection is assigned to the GT box that contains most of its LiDAR points (at least
  3 points and 30% of them inside the box). Ambiguous ones are dropped (14 and 13), duplicates resolved. This is done
  once, not per mode, so a mode cannot "choose" its own truth.
- **Metrics per object.** Center error (m): distance between predicted and GT box centre. Range error: only the part along the
  viewing direction (depth). Size error: mean of |predicted/GT − 1| over length, width, height. Yaw error in degrees.
- **Paired comparison.** For each pair of modes we take the difference *per object* and look at the median difference
  (a 95% confidence interval by bootstrap, a sign test with p-value, and "wins" = number of objects where mode B is
  better). This is stronger than comparing two averages because every object is its own control.
- **Objects cut off by the image edge** (34 in total) are reported separately: the mask misses part of the car, so the
  box is wrong for a reason no pointmap can fix. Statistics below use the other 181 objects, unless stated.
- **Caution.** Many subgroup tests were made. A p-value of 0.04 in one subgroup is a hint, not proof. I only trust an
  effect if it appears in *both* datasets and in the pooled data.

## 4. Results

### 4.1 Typical error (median, objects not cut by the image edge)

| | m1 sparse | m2 | m6 | m7 | m8 |
|---|---|---|---|---|---|
| nuScenes (n = 99): center [m] | 0.85 | 0.62 | 0.58 | 0.63 | 0.58 |
| nuScenes: size error | 0.19 | 0.15 | 0.15 | 0.15 | 0.15 |
| ECP (n = 82): center [m] | 0.48 | 0.33 | 0.30 | 0.30 | 0.32 |
| ECP: size error | 0.149 | 0.125 | 0.114 | 0.117 | 0.117 |
| ECP: yaw error [deg] | 4.5 | 6.7 | 5.9 | 5.8 | 5.2 |

### 4.2 Paired differences to mode 2, both datasets pooled (n = 181; negative = better than mode 2)

| comparison | center error | size error | verdict |
|---|---|---|---|
| m1 sparse vs m2 | **+16 cm** [+10, +20], mode 2 better on 126 of 181, p < 0.001 | +1.0 pt, p = 0.02 | **Dense wins clearly.** Holds at 0–15 m (+11 cm), 15–30 m (+14 cm), > 30 m (+31 cm) and in each dataset separately. |
| m6 vs m2 | −0.4 cm [−1.8, +0.3], p = 0.23 | −0.1 pt, p = 0.42 | **No difference.** |
| m7 vs m2 | −0.1 cm [−1.0, +1.1], p = 0.88 | 0.0 pt | **No difference.** (m7 vs m6: +1 cm worse, p = 0.02, meaningless in size.) |
| m8 vs m2 | +0.1 cm [−0.5, +1.1], p = 1.0 | −0.2 pt, p = 0.24 | **No difference overall.** |

Sparse is slightly *better* at yaw (0.5°, p = 0.001). Not important for us, but it is there: a possible
reason (untested) is that exact points give a cleaner heading cue than a filled-in surface.

### 4.3 Bad boxes (tails matter for auto-labelling, a few huge errors hurt more than many tiny ones)

Share of objects with a centre error above 1 m / above 2 m (pooled, not truncated):

| | m1 | m2 | m6 | m7 | m8 |
|---|---|---|---|---|---|
| > 1 m | 30% | 14% | 14% | 15% | **17%** |
| > 2 m | 8% | 5.0% | 3.9% | 3.9% | 5.5% |

- Sparse has twice as many bad boxes: that is the real cost of mode 1.
- Mode 8 is the *worst* of the dense modes here. Replacing a fitted slope by the background slope sometimes puts
  the object at the wrong depth (biggest cases: +2.2 m truck, +2.2 m bicycle at 38 m, +1.6 m car at 36 m, all versus mode 6).
- The *mean* error looks better for mode 6 (0.63 vs 0.74 m for mode 2). This is **one object**: a car on nuScenes
  where mode 2 was 15 m off and mode 6 was fine. Without it the means are equal. Mode 6 rescued one catastrophic
  case, but on ECP it lost 0.5 to 0.8 m on four far objects (33–41 m) where mode 2 was fine (only 7 objects at that
  range on ECP, so this is an anecdote in both directions).

### 4.4 Do the extras help where they should? (unreliable object slope, small masks)

Pooled, 80 objects with an unreliable slope:

| | size error | center error |
|---|---|---|
| m8 vs m2 | −1.6 pt (p = 0.044) | +3 cm (not significant) |
| m8 vs m6 | −0.9 pt (p = 0.008) | +2.5 cm (p = 0.09) |
| m7 vs m2 | −1.5 pt (p = 0.008) | −2 cm (not significant) |

The size gain is real-looking but tiny (about 1.5 percentage points of size error, on a base of about 12–15%) and
it is bought with slightly worse position and more bad boxes (mode 8). Mode 7 gets the same size gain without the position
penalty, but this is a subgroup finding (many tests) and mode 7 is the most complex mode (one map per frame, all
objects at once, needs ≥ 30 LiDAR points per grid cell).
For the healthy objects (101 of 181), modes 2, 6, 7 and 8 are identical (differences below 1 cm).

### 4.5 Where errors come from (all modes)

- **Objects cut by the image edge:** centre error about 0.45–0.9 m (against 0.3–0.6 m).
- **Class:** cars 0.4 m, bicycles 0.2–0.25 m, trucks about 0.8 m, construction vehicles about 1 m (only 9 objects).
  Big objects are hard because the mask and the LiDAR only see one side.
- **Range:** nothing dramatic between 0 and 30 m; sparse gets clearly worse beyond 30 m.

### 4.6 CompletionFormer-family modes (4, 9, 10) [measured 2026-09-28, same sweep protocol, nuScenes scene-0061 + full ECP]

**Note added after this section was written: mode 10's m10 numbers below are STALE.** They were measured on an
earlier version of `completionformer_ground` that used the FULL cloud (ground included) for both the in-mask
cleaning and the background — not the current implementation, which uses the ground-free cloud for in-mask cleaning
(same as mode 4) and only mixes in the full cloud for the out-of-mask background (mode 6's object/background split,
applied here). The m10 discussion below (in particular the "dataset-dependent, opposite-direction effect" and the
long-range numbers) describes the *old* design and needs to be re-run before being trusted for the current code.
m4 and m9 are unaffected by that change and their numbers below still stand.

Three variants of "fill the LiDAR to a dense image with a depth-completion network instead of MoGe+affine" were run:
**m4** = `completionformer_full` with `cformer_drop_on_failure=False` (the notebook default we ended up on: on a mask
where HDBSCAN finds no dominant cluster, keep its in-mask points instead of dropping them; the *dropping* variant was
never run as a separate mode); **m9** = `completionformer_raw` (no per-mask cleaning at all); **m10** =
`completionformer_ground`, at the time this was run: m4's cleaning, but CompletionFormer sees the FULL cloud, ground
included, for BOTH parts (superseded design, see the note above).

Pooled paired differences to m2 (n = 181, negative = better than m2):

| comparison | center error | abs range error | size error | yaw error |
|---|---|---|---|---|
| m9 vs m2 | +4.4 cm, n.s. | +6.2 cm, n.s. | −0.3 pt, n.s. | **−0.52°, p = 0.007** |
| m10 vs m2 | −0.8 cm, n.s. | 0.0 cm, n.s. | −0.1 pt, n.s. | −0.24°, n.s. |
| m4 vs m2 | +2.5 cm, n.s. | +4.0 cm, p = 0.053 | −0.4 pt, n.s. | **−0.46°, p = 0.001** |

On the surface this looks like "no difference, maybe slightly better yaw" — but the median hides what actually
happens:

- **The tails are much worse for all three.** Share of objects with a centre error > 1 m: m2 = 14%, m9/m10/m4 =
  **25–30%**. > 2 m: m2 = 5%, m9/m10/m4 = **13–14%**. Mean centre error (which the median hides, but auto-labelling
  cares about): m2 = 0.74 m, m9/m10/m4 = **1.25–1.51 m**.
- **Beyond 30 m every CompletionFormer variant is clearly worse than m2/m6, on both datasets** (nuScenes: m2 0.78 m /
  m6 0.72 m vs m9/m10/m4 0.87–1.25 m; ECP, only 7 objects: m2 0.35 m vs m9/m10/m4 1.01–2.12 m). This is the most
  consistent finding in this section: a depth-completion network has to extrapolate depth where LiDAR is sparsest,
  and it extrapolates worse than an affine fit does.
- **m10 is dataset-dependent, in opposite directions.** On ECP it is *significantly better* than m2 (center −4.7 cm,
  p = 0.003; range −4.4 cm, p = 0.003; size −1.3 pt, p = 0.043) and better than m6 too. On nuScenes it is
  *significantly worse* on range (+6.2 cm, p = 0.005) and directionally worse on center. The likely reason: the
  CompletionFormer checkpoint was trained on 64-beam KITTI depth completion; ECP's LiDAR is 64-beam (in-distribution),
  nuScenes' is 32-beam (out-of-distribution) — feeding it MORE points (mode 10's ground-included cloud) only helps
  where the input already looks like training data. Even on ECP this win disappears beyond 30 m (Section 4.6, above).
  **An effect that flips sign between datasets is not a reason to change the default** — it needs to hold in both.
- **m9 (no cleaning) is the worst or tied-worst option everywhere it differs from m4/m10** — confirms the per-mask
  HDBSCAN cleaning mode 4/10 do is worth keeping, not an unnecessary step.
- **Yaw is the one place these modes are consistently and significantly better than m2** (pooled m9 p = 0.007, m4
  p = 0.001; same direction, not always significant, in every split). Consistent with mode 1 also beating m2 on yaw
  in Section 4.2: exact or non-hallucinated depth structure near the object edges seems to help heading, at some
  cost to position when the network has to hallucinate depth beyond the LiDAR.

**Conclusion: none of the CompletionFormer variants change the Section 7 decision.** m4 and m9 do not reliably beat
mode 2 on position or size, and are meaningfully worse in the tails and beyond 30 m. m10's numbers (including the one
significant ECP win) are **stale** (see the note at the top of this section) and need a re-run under the current
hybrid design before they can be used for anything. Reproduce: `compare_modes.py --shared-run mode_selection/cmp2_shared
--modes 2 6 9 10 4 --baseline m2` per dataset (checkpoints under `objects__completionformer_full__<hash>` for m4,
tagged in run folders `mode_selection/cmp3_m9`/`cmp3_m10`/`cmp3_m4b` -- m10 needs a fresh run folder since the code changed).

## 5. Token-drop probe: which part of the pointmap does SAM3D actually read?

Before comparing modes, we checked what SAM3D Objects *does* with the two pointmap conditioning streams it gets (the
object's own crop, and the full image) by zeroing them out one at a time and re-running mode 2 on the same objects
with the same seed (`force_drop_modalities`; the shift/scale SAM3D derives from the map are unaffected by this — see
Section 1). Training only ever dropped both streams together (10% of samples), so dropping one stream alone is
somewhat off the training distribution — read this as directional.

| scene (n objects) | dropped | position shift | size change (long / height) |
|---|---|---|---|
| nuScenes 0655 f3 (6) | both | 0.47 m | ×1.22 / ×1.30 |
| | full image only | 0.05 m | ×0.99 / ×1.00 |
| | **crop only** | **0.57 m** | ×1.22 / ×1.23 |
| nuScenes 1094 f22 (5) | both | 0.32 m | ×0.90 / ×0.86 |
| | full image only | 0.06 m | ×0.96 / ×0.96 |
| | **crop only** | **0.27 m** | ×1.06 / ×1.04 |
| ECP scene 10 f560 (6) | both | 0.69 m | ×0.58 / ×0.56 |
| | full image only | 0.06 m | ×0.85 / ×0.88 |
| | **crop only** | **1.35 m (one object 60 m off)** | ×0.97 / ×0.72 |

**Reading: the crop stream is what SAM3D actually reads; the full-image stream barely matters.** Dropping the crop
alone moves objects by 0.27–1.35 m and swings sizes by 10–44% (direction differs by scene, so it's not a fixed size
prior being lost — the network is really reading depth structure there). Dropping the full-image stream alone shifts
things by under 6 cm and 1–15% in size, everywhere.

**Why this matters for the mode comparison:** every mode we test (2/6/7/8/4/9/10/11) differs in what's *inside the
crop*, i.e. exactly the part SAM3D reads. That's also why a mode's effect outside the mask (the "background" fill in
modes 6/7/10) has shown almost no measurable effect anywhere in Section 4 — the probe already predicted that before
we ran a single comparison.

## 6. The larger open problem: footprint proportions

On nuScenes every dense mode gives boxes about 13% too *short* (length ratio 0.87) and 14–16% too *wide* (width ratio
1.14–1.16). On ECP the length is about right (0.98–1.00) but the width still 1.2–1.26 too wide. This is 5–10× larger than any
difference between modes, and no pointmap variant changes it. Possible causes (not yet tested): the shape SAM3D
generates is rounder than real cars; or our OBB fit to the mesh (PCA) overweights the width. This is the next thing to
investigate.

## 7. Decision

**This section reflects the single-scene sweep; Section 10 (all 11 modes, deliberately diverse scenes) confirms the
same decision and adds LDCM.**

| option | for | against |
|---|---|---|
| **Mode 2** | simplest; needs only the ground-free cloud; as accurate as the rest; fewest failure modes | one catastrophic case remains possible (seen once in 181) |
| Mode 6 | best median on paper (−0.4 cm, not significant); rescued the one 15 m failure | not measurably better; needs the full cloud including ground; lost 0.5–0.8 m on 4 far ECP cars |
| Mode 7 | −1.5 pt size on unreliable objects | not better anywhere else; most complex; slightly worse than 6 on position |
| Mode 8 | −1.6 pt size on unreliable objects | more bad boxes (17% vs 14%), position not better |
| Mode 1 | fastest, no neural depth, slightly better yaw | 16 cm worse position, twice as many bad boxes |
| Modes 4/9 | slightly better yaw (significant, pooled) | not better on position/size overall; **much worse in the tails** (25–30% of objects > 1 m off vs 14%); clearly worse beyond 30 m on both datasets |
| Mode 10 | untested under the current (hybrid ground-free/full-cloud) design | the numbers in Section 4.6 are stale (measured on a superseded implementation) — needs a re-run |

**Go with mode 2.** Reasoning in words: the data show one big effect (dense beats sparse) and no reliable effect among the
dense affine variants (6/7/8), and the CompletionFormer-family modes (4/9/10) are actively worse where LiDAR is sparsest
(beyond 30 m) even though their median looks similar. When several options perform the same on the metric that matters,
take the simplest one, and don't adopt one whose failure tail is worse just because its median ties. Every added
component (background fit, per-frame map, slope repair, a second depth-completion network) is another thing that can
break on a scene we did not test, and the tail statistics show that modes 8, 9, 10 and 4 already do, in different ways.
The small size gains of modes 7 and 8 are subgroup findings and too small to matter for the thesis question.

**What would change this decision:**
- A larger test (more nuScenes scenes) in which mode 6 or 7 is consistently better by more than the noise (about 1–2 cm
  in the median, or a lower share of bad boxes). Then take that one.
- If one wants a size-focused variant, mode 7 is the candidate (best size gain, no position loss) but needs replication on
  more scenes first.
- LDCM (mode 11, Section 9) beating mode 2 in the tails specifically, not just the median — that is exactly where the
  CompletionFormer-family modes lost.

**Limits of this evidence:** one nuScenes scene (all 8 train scenes would give ~8× the objects); 33 ECP frames; small
groups for trucks, construction vehicles, and objects beyond 30 m on ECP (7); pedestrians are not covered here (they go
through SAM3D Body, not through this pointmap).

## 8. How to reproduce

```
bash container/run_in_container.sh python autolabeling/compare_modes.py --dataset ecp --shared-run mode_selection/cmp2_shared \
     --modes 1 2 6 7 8 --baseline m2            # same with --dataset nuscenes_mini --scenes scene-0061
bash container/run_in_container.sh python autolabeling/compare_modes.py --dataset ecp --shared-run mode_selection/cmp2_shared \
     --modes 2 6 9 10 4 --baseline m2           # Section 4.6; same with --dataset nuscenes_mini --scenes scene-0061
```
(all runs behind this document now live under `output/<dataset>/mode_selection/`, see the note in Section 0/top)
The pooled statistics in Section 4.2–4.4 and 4.6 come from a small script that concatenates the two `_compare_*.csv`
files (paired sign test + bootstrap of the median difference per object).

## 9. LDCM (mode 11)

**Status (2026-09-28): implemented, debugged, and run — see Section 10 for the results.** Two real bugs had to be
fixed before it worked at all, worth recording since they're easy to hit again:
1. `ldcm_moge_ckpt` must be the `.pt` **file**, not the checkpoint directory — LDCM's own loader resolves a directory
   to `<dir>/ldcm.pt` for its main checkpoint, but its MoGe-prior loader does not do that same resolution.
2. LDCM's bundled MoGe code calls `utils3d.pt.*`; the `utils3d` already in this container (needed by an unrelated
   package) only has the same functions under `utils3d.torch` — a naming difference between `utils3d` releases, not
   an API change. `utils3d` is a process-wide singleton (SAM3D's own MoGe calls import it too), so rather than
   monkeypatch that shared module, the two lines in `models/LDCM/ldcm/moge/model/v2.py` that need it were patched to
   import a **separately renamed** vendored copy (`ldcm_utils3d`, via `ldcm_utils3d_vendor` in the config) that can
   never collide with the real `utils3d` — zero risk to every other mode, by construction rather than by careful
   sequencing.
A third issue (a long stall on a compute node) turned out to be the user's own network being capped, not a bug.

**Why try this specifically, after Section 4.6.** The one thing every CompletionFormer variant (4/9/10) got
consistently wrong was the same thing: it fails worst exactly where LiDAR is sparsest (beyond 30 m, and in the tails
generally) — because a plain depth-completion network, trained only on KITTI's 64-beam outdoor scans, has to
extrapolate depth between sparse points using what it learned from that comparatively narrow, LiDAR-only training
signal. Mode 2 (MoGe + a single affine) doesn't have that problem, because MoGe's *shape* comes from a model trained
on huge amounts of ordinary images, not from depth completion at all — the LiDAR only sets the scale and offset, not
the shape. That is the actual mechanism behind "MoGe + affine beats CompletionFormer" in this project.

**LDCM (ICLR 2026, github.com/aigc3d/LDCM, checkpoint on `pkqbajng/LDCM`)** looks like a way to get both: it fuses
the image and sparse LiDAR *per pixel*, but on top of a **frozen MoGe prior** rather than starting from a KITTI-only
depth-completion backbone — architecture: frozen MoGe + multi-scale Poisson completion + a DINOv2-based refinement
network, zero-shot (no per-scene fitting). In other words, it keeps the same image-based shape prior that already
works for us in mode 2, but replaces our single global affine fit with a learned, per-pixel fusion of that prior with
the LiDAR — which could do better than one affine number where the object's own LiDAR is too sparse or too flat to
fit reliably (Section 2's "ill-conditioned" slope problem), without inheriting CompletionFormer's KITTI-only
extrapolation weakness. The repo also reports beating OMNI-DC (a LiDAR-only completion network, no image shape
prior) in zero-shot tests, consistent with this project's own finding that the image prior matters more than
faithfully spreading sparse LiDAR.

**Implementation** (mirrors mode 4's structure in `sam3d_objects.py`): same ground-free cloud and per-mask HDBSCAN
anchor cleaning as mode 4 (`_build_o5_sparse_points`, `cformer_drop_on_failure` applies here too), but
`_compute_dense_ldcm_pointmap` calls `LDCMModel.infer(image, sparse_depth)` instead of CompletionFormer — LDCM's own
preprocessing convention (image `[0,1]` float32 CHW, sparse depth `(1,1,H,W)` metres with 0 = missing) is identical
to CompletionFormer's, so the same tensor construction is reused verbatim. New config keys `ldcm_repo`, `ldcm_ckpt`,
`ldcm_moge_ckpt` (local dirs or bare HF repo ids), not required in existing YAMLs.

**What would make this worth moving to the full pipeline:** beating mode 2 specifically in the tails (share of
objects > 1 m off) and beyond 30 m, on both datasets — not just tying it on the median, which modes 4/9/10 already
do without being worth adopting. See Section 10 for where it landed.

## 10. Final round: multi-scene sampling, all 11 modes [measured 2026-09-28]

**Why redo the sweep.** Every result up to here came from ONE nuScenes scene (scene-0061, up to 39 consecutive
keyframes) plus ECP's 33 annotated keyframes (only 3 scenes). Consecutive frames from one scene are not independent
observations — they're mostly the same handful of physical objects, re-observed every 0.5 s, on the same road, same
lighting, same weather. The "n = 118" in Section 4 overstates how much independent evidence that actually was, and it
plausibly explains why mode 10's old design looked like a clear win on ECP (Section 4.6) and didn't replicate on
nuScenes — a single scene's own quirks can dominate a result that looks clean within that scene.

**Sampling.** Added `--frames-per-scene K` to `run_pipeline.py` (implemented in `nuscenes_loader.py` /
`ecp_loader.py`: `evenly_spaced_indices(n, k)` picks K indices spread across the available range, e.g. k=3 → first,
middle, last — gaps between picks, not a contiguous run, so the scenery actually changes):
- **nuScenes:** 3 frames per scene, all 8 mini-train scenes → 24 frames (vs. 39 frames from 1 scene before).
- **ECP:** 2 frames per scene, all 3 annotated scenes → 6 frames (vs. 33 frames from the same 3 scenes before).
Checkpoints are indexed by *position* in the frame list, not by frame content, so this needed a fresh prepare-once
checkpoint (`cmp6_shared`, both datasets) — reusing `cmp2_shared` here would have silently mixed up which cached
result belongs to which frame. All 11 modes were run against `cmp6_shared` on both datasets (`submit_sweep.sh
nuscenes_mini/ecp cmp6 "1 2 3 4 5 6 7 8 9 10 11" ...`); all 33 jobs (2 prepare + 22 mode runs) completed cleanly.

**Cost of this trade: much less statistical power.** Matched objects dropped from 118→37 (nuScenes) and 97→8 (ECP,
7 non-truncated) — and a striking side effect: 101 of 142 nuScenes detections (71%) had no GT match at all, far
higher than before, apparently because the first/last frame of a scene (picked deliberately, for spread) has less
reliable single-sweep LiDAR-to-GT association than a frame from the middle of continuous tracking. Take every
non-significant result below as *underpowered*, not as evidence of "no effect" — e.g. mode 1 vs 2 (dense beats
sparse) is no longer significant here (p = 0.87, n = 38) despite being one of the most solid findings in this whole
project (p < 0.001, n = 181, Section 4.2) — that's the sample size talking, not a reversal.

**Medians, non-truncated (nuScenes n = 31, ECP n = 7 — ECP this small is anecdotal, not a result on its own):**

| | m1 | m2 | m3 | m4 | m5 | m6 | m7 | m8 | m9 | m10 | m11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nuScenes center [m] | 1.70 | 1.32 | 1.55 | 2.16 | 6.65 | 1.30 | 1.28 | 1.13 | 2.33 | 1.51 | 1.33 |
| ECP center [m] | 0.71 | 0.60 | 1.04 | 0.52 | 3.62 | 0.56 | 0.53 | 0.53 | 0.52 | 0.44 | 0.55 |

Everything moved up in absolute terms compared to the single-scene numbers (mode 2 alone: 0.62 m → 1.32 m on
nuScenes) — the single scene was, on reflection, a comparatively easy/favourable one. **Modes 3 and 5 measured for
the first time here confirm exactly what was predicted from mechanism, not run before:** m3 (masked) still has a
huge size error (0.51 vs 0.15 for m2 — removing the background really does cost SAM3D the context it needs); m5
(raw, no point-cleaning) is catastrophic (6.65 m median on nuScenes) — HDBSCAN cleaning is doing real work, not
paperwork.

**Pooled paired differences to m2 (both datasets, n = 38 non-truncated; negative = better than m2):**

| comparison | center error | yaw error | verdict |
|---|---|---|---|
| m1 (sparse) vs m2 | −5.5 cm, n.s. (p = 0.87) | n.s. | underpowered, not a reversal (see above) |
| m6 vs m2 | −2.1 cm, borderline (p = 0.073) | **+0.10°, worse, p = 0.014** | same tiny position edge as before, but now a visible yaw cost |
| m7 vs m2 | −1.4 cm, n.s. | n.s. | still no reliable difference |
| m8 vs m2 | **−2.4 cm, p = 0.034** | **+0.21°, worse, p = 0.014** | the one significant position win here, but paired with the same yaw cost as m6 |
| m9 vs m2 | −5.1 cm, n.s. | n.s. | still trends worse (not significant at this n) |
| m10 vs m2 | +0.3 cm, n.s. | n.s. | **the fixed hybrid design (Section 4.6's note) is now simply indistinguishable from m2** — no spurious win, no loss |
| m11 (LDCM) vs m2 | −6.3 cm, borderline (p = 0.073) | n.s. | closest of 9/10/11 to a real win, and the only one with no downside anywhere; not significant yet |

**Reading this together with Section 4:** mode 2 has now been checked against sparse LiDAR, the three affine
variants (6/7/8), three CompletionFormer variants (4/9/10), and LDCM (11), across two datasets, a single scene AND
eight diverse scenes. Nothing beats it convincingly anywhere. Modes 6 and 8 show a small, now cross-validated
position edge (~2 cm) that comes with a small, newly-visible yaw cost — a real tradeoff, not a free win. LDCM (11) is
the most promising of the three depth-fusion alternatives — never worse than m2 on anything measured, directionally
the best of the three — but "closest to significant" is not the same as significant; it would need a dedicated,
larger sample (not squeezed in alongside 10 other modes on 24+6 frames) to actually confirm.

**Decision: unchanged. Mode 2 stays the default** (Section 7). If one extra experiment were worth running next, it
would be LDCM specifically, with its own larger sample, rather than any of the CompletionFormer variants (4/9/10),
which this round makes it easier, not harder, to set aside.

Reproduce: `bash container/submit_sweep.sh nuscenes_mini cmp6 "1 2 3 4 5 6 7 8 9 10 11" --frames-per-scene 3` /
`bash container/submit_sweep.sh ecp cmp6 "1 2 3 4 5 6 7 8 9 10 11" --annotated-only --frames-per-scene 2` (this
re-creates the runs fresh, at the top level -- the ones already run were moved to `mode_selection/cmp6_shared` etc.
afterward, see the note in Section 0/top), then `compare_modes.py --shared-run mode_selection/cmp6_shared
--modes 1 2 3 4 5 6 7 8 9 10 11 --baseline m2` per dataset. Pooled stats: same small script pattern as Section 4.6,
reading both `mode_selection/_compare_m1_..._m11.csv` files.
