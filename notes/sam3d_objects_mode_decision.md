# SAM3D Objects: choosing the pointmap input

## 1. Background

For every detected object, SAM3D Objects (the model that turns a 2D detection into a 3D mesh and box) is given a
**pointmap**: for every pixel, a 3D position (X, Y, Z) in metres in the camera frame. This pointmap is what tells the
model *where in the real world* the object is and roughly *how big* it is — get it wrong and the resulting 3D box is
wrong by about the same amount.

The raw material for the pointmap is LiDAR, which is accurate but sparse (large gaps between points), and MoGe, a
monocular depth network that produces a dense depth image for every pixel but only "up to scale" — it knows the
*shape* of the scene, not the metres. Turning that into a usable pointmap means combining the two: fit a per-pointmap
`Z_true ≈ a · Z_moge + b` correction using the LiDAR points, then apply it densely.

There is more than one reasonable way to do this fit — how much of the image it should be fit on, what to do where
there's no LiDAR to fit against, whether to fall back to a different network entirely. This document compares those
choices ("modes") and separately asks a more basic question: does SAM3D Objects even rely on the pointmap enough for
this choice to matter?

## 2. What we tested

### 2.1 The pointmap modes

| # | mode | idea |
|---|---|---|
| 1 | sparse LiDAR | no neural depth at all — the raw, sparse LiDAR points, with everything else left unknown |
| 2 | MoGe + local affine | one metric correction, fitted only on the LiDAR inside this object's own mask |
| 3 | MoGe + local affine, masked | like 2, but everything outside the mask is blanked out instead of left as plausible-but-uncalibrated depth |
| 4 | CompletionFormer (cleaned) | a depth-completion network fills in the LiDAR to a dense map, instead of MoGe + affine |
| 5 | MoGe + local affine, uncleaned | like 2, but skips the step that removes outlier LiDAR points before fitting |
| 6 | MoGe + local + background affine | like 2, plus a *separate* correction fitted for the background around the object |
| 7 | MoGe + composite | one shared map per frame: every object gets its own correction, the background is a smoothly blended grid of local corrections |
| 8 | MoGe + local, with slope repair | like 6, but replaces an unreliable object-only fit with the background's fit |
| 9 | CompletionFormer (uncleaned) | like 4, without the outlier-cleaning step |
| 10 | CompletionFormer (hybrid anchors) | like 4, but the background anchors additionally include ground points |
| 11 | LDCM | a newer network (ICLR 2026) that fuses a frozen MoGe prior with the LiDAR per pixel, instead of a single affine number or a KITTI-only completion network |

Modes 2 and 6/7/8 all use the same underlying idea (MoGe rescaled by a LiDAR-fitted correction) and only differ in
how much of the image gets its own correction vs. a shared/fallback one. Modes 4/9/10 replace that whole idea with a
learned depth-completion network. Mode 11 is a newer alternative that tries to combine both approaches. Modes 1, 3
and 5 are deliberately "broken" baselines, included to check that the pieces of mode 2 (dense depth, background
context, outlier cleaning) are actually necessary rather than just complexity for no reason.

### 2.2 The conditioning-input probe

Separately from *which pointmap*, we asked a more basic question: **does SAM3D Objects even use the pointmap enough
for the above comparison to matter?** For every detected object, the model is actually given three kinds of input —
**image** (RGB), **mask**, and **pointmap** — each supplied at two scales: a **crop** and the **full** camera frame.
That is six independent input channels. Three terms that sound similar but mean different things:

- **Full** — the whole camera image (or pointmap), covering the entire scene, not just this object.
- **Crop** — a zoomed-in window centred on the object, covering the object *plus a margin of its immediate
  surroundings* — it still contains background pixels, just the nearby ones rather than the whole scene.
- **Mask** — a separate, pixel-exact indicator of which pixels (within either the crop or the full frame) are the
  object itself, as opposed to anything around it, near or far.

So "crop" and "mask" are independent concepts: the crop is a *region* (object + some surrounding background), the
mask is a *pixel-level selection* (object only, no background) that can be applied within that region or within the
full frame. This distinction matters for Section 3.5.

We tested whether the model still works when each channel is forcibly zeroed out, one at a time and in combination:
each of the three modalities with crop only, full only, or both dropped (9 combinations), plus all three crops
dropped together and all three fulls dropped together (2 more) — 11 combinations in total. Everything else about the
run (detections, masks, the pointmap mode) stayed exactly as normal; only the listed channel(s) were blanked out
before the model saw them.

### 2.3 Method

Both the mode comparison and the probe were evaluated on the same scope: all 8 nuScenes mini-train scenes and all 33
annotated ECP frames. Predictions were scored with the standard nuScenes detection metric, **mAP**, at three label
granularities (1-class, 3-class, 8-class vehicle/pedestrian groupings). We deliberately do not use NDS (the combined
score nuScenes benchmarks usually report): it folds in velocity error and attribute error, neither of which this
pipeline predicts (attribute is never populated, velocity is not estimated at all), and orientation error, which we
consider unreliable to assess here — so NDS would penalize modes for things unrelated to the pointmap choice, not
measure them fairly. mAP alone is the decision metric throughout.

The mode comparison ran in three rounds, widening in scope each time: first a small, single-scene round using
per-object position/size/yaw error instead of mAP (this is where modes 3, 4, 5, 9 and the original version of mode
10 were ruled out — Section 3.2); then the full-dataset, front-camera-only round (Section 3.1); then an all-camera
round for the three remaining candidates (Section 3.3), which produced the final decision.

## 3. Results

### 3.1 Mode comparison, front camera only (mAP, full evaluation set)

| mode | nuScenes 1c / 3c / 8c | ECP 1c / 3c / 8c |
|---|---|---|
| 1 (sparse LiDAR) | .0228 / .0207 / .0271 | .348 / .284 / .141 |
| 2 (local affine) | .0305 / .0251 / .0272 | .364 / .340 / .181 |
| 6 (local + background) | .0304 / .0251 / .0275 | .366 / .335 / .185 |
| 7 (composite) | .0312 / .0257 / .0273 | .363 / .341 / .186 |
| 8 (slope repair) | .0304 / .0266 / .0272 | .354 / .326 / .170 |
| 10 (CompletionFormer, hybrid) | .0218 / .0179 / .0198 | .360 / .324 / .176 |
| 11 (LDCM) | .0336 / .0290 / .0283 | .356 / .332 / .177 |

Modes 3, 4, 5 and 9 are not in this table — they were excluded before it was worth spending full-dataset compute on
them (Section 3.2).

### 3.2 Why modes 3, 4, 5 and 9 were excluded

These four were tested in the smaller, earlier round and consistently failed on mechanism, not just on the margins:

- **Mode 3** (blanking the background instead of leaving it uncalibrated) roughly triples the size error compared to
  mode 2 — SAM3D genuinely uses the surrounding context, even when it isn't metrically correct.
- **Mode 5** (no outlier cleaning before the affine fit) is catastrophic: several-metre median position error,
  because a handful of bad LiDAR points can dominate a fit with nothing to filter them out.
- **Modes 4 and 9** (CompletionFormer, cleaned/uncleaned) look similar to mode 2 *on average* but produce roughly
  twice the rate of large failures (>1 m off) and are consistently worse beyond 30 m on both datasets. A
  depth-completion network trained only on KITTI's 64-beam outdoor LiDAR has to extrapolate depth exactly where the
  input LiDAR is sparsest, and does so worse than a simple affine fit on top of MoGe's image-based shape prior.

### 3.3 All-camera round — the deciding comparison (mAP)

Front-camera-only testing showed no consistent winner among 2, 6, 7, 8 and 11, so the three strongest candidates
(2, 7, 11 — 6 dropped for being indistinguishable from 7, 8 for showing no consistent benefit) were compared using
every camera (6 for nuScenes, 3 for ECP) instead of just the front one, with cross-camera duplicate suppression.
This is a different, larger detection scope than Section 3.1 (more boxes from more viewpoints), so these numbers are
not directly comparable to the front-camera table above — only to each other, within this round.

| mode | nuScenes 1c / 3c / 8c | ECP 1c / 3c / 8c |
|---|---|---|
| 2 (local affine) | .3617 / .3088 / .2345 | .5201 / .4531 / .2357 |
| 7 (composite) | .3665 / .3147 / **.2410** | .5075 / .4605 / .2453 |
| 11 (LDCM) | **.3749** / **.3276** / .2399 | .5043 / .4687 / .2425 |

ECP alone shows no clean winner (mode 2 leads 1-class, 11 leads 3-class, 7 leads 8-class, margins small). nuScenes —
the more decisive test (see below) — is clearer: **mode 11 leads on 1-class and 3-class, by the largest margins of
the three modes, and is within 0.5% of mode 7's narrow 8-class lead.** Mode 2, the simplest option, is behind both 7
and 11 on every granularity on nuScenes.

**Why nuScenes is the more decisive test.** ECP's LiDAR is 64-beam; nuScenes' is 32-beam and sparser. A denser input
gives every mode more points to fit against, which narrows the gap between them — consistent with modes 4/9/10's
KITTI-64-beam-trained network doing better on ECP than on nuScenes (Section 3.2).

**Decision: mode 11 (LDCM).** It wins the more informative dataset on two of three granularities by a clear margin
and ties mode 7 on the third — the only candidate that is never clearly behind on nuScenes. This supersedes the
mode-2 lean from the front-camera-only round (Section 3.1): once tested at full multi-camera scale on the sparser,
more discriminating dataset, mode 2 is consistently the weakest of the three remaining candidates.

### 3.4 Conditioning-input probe (relative change in 8-class mAP vs. nothing dropped)

| dropped | nuScenes | ECP |
|---|---|---|
| **all three crops** (only full-scale left) | **−76%** | **−44%** |
| all three fulls (only crop-scale left) | −11% | −2% |
| image — crop only | −24% | −11% |
| image — full only | −5% | −1% |
| image — both | −41% | −12% |
| mask — crop only | −22% | −10% |
| mask — full only | −6% | −10% |
| mask — both | −18% | −11% |
| **pointmap — crop only** | **−52%** | **−40%** |
| pointmap — full only | −2% | +2% (noise) |
| pointmap — both | −15% | −6% |

Same ranking at every label granularity, on both datasets.

### 3.5 Reconciling this with mode 3

At first glance, "drop the full pointmap but keep the crop pointmap" sounds like it should be the same thing as mode
3 (Section 2.1), which also keeps geometry only near the object — and mode 3 performed badly (Section 3.2), while
dropping the full pointmap barely hurt (Section 3.4). These don't actually contradict each other, because they
remove different things:

- **Mode 3** blanks every pixel *outside the object's exact mask* in the pointmap the crop is later taken from — so
  the crop used by mode 3 *also* has its immediate background blanked, because the crop is just a zoomed-in slice of
  that same array. Mode 3 removes the object's own nearby background.
- **Dropping the full-pointmap channel** (Section 3.4) leaves the crop pointmap completely untouched — the object's
  immediate surroundings inside the crop window keep their real depth values. Only the separate, wider full-frame
  view is removed.

So the two results agree rather than conflict: **what actually hurts is losing the object's own nearby background,
not losing the wide-scene view beyond the crop.** Mode 3 is the one setting that removes the former; dropping
pointmap-full only removes the latter — which is exactly why one is costly and the other isn't.

## 4. Conclusions

1. **Dense depth clearly beats sparse LiDAR alone** (mode 1 worst on both datasets, every metric) — filling in the
   gaps between LiDAR points is worth doing.
2. **Removing background context (3) or outlier cleaning (5) breaks the model outright.** These aren't real
   alternatives to mode 2, they're evidence that both of those ingredients are load-bearing.
3. **A learned depth-completion network in place of MoGe (4/9/10) does not help and sometimes actively hurts**,
   specifically where LiDAR is sparsest (long range, and the tail of large failures generally) — its KITTI-only
   training doesn't generalize as well as MoGe's much broader image-based shape prior.
4. **On the front camera alone, 2, 6, 7, 8 and 11 were indistinguishable** (differences within noise, no mode
   consistently ahead on both datasets). Modes 6 and 8 were dropped at that stage (6 for being statistically
   indistinguishable from 7, 8 for showing no consistent benefit), leaving 2, 7 and 11 to be re-tested at full
   multi-camera scale (Section 3.3). There, **mode 11 (LDCM) comes out ahead** — it leads nuScenes (the more
   discriminating dataset, sparser LiDAR) on two of three label granularities by a clear margin and is tied with
   mode 7 on the third, while mode 2 is the weakest of the three on every nuScenes granularity. **Mode 11 is the
   final choice.**
5. **The conditioning-input probe explains why the pointmap choice matters at all, and where to focus it:** SAM3D
   Objects overwhelmingly relies on the *cropped* version of its inputs — dropping any full-frame input barely
   changes the output, while dropping any crop input hurts substantially. The **crop-scale pointmap is the single
   most important input the model has** (bigger effect than dropping the crop image or crop mask). This is also why
   changes made *outside* the mask — the background-fitting machinery in modes 6, 7 and 10 — have such a small,
   inconsistent effect on the final result: the model mostly isn't looking there.

## 5. Open issue: box proportions

Independent of pointmap mode, predicted boxes are systematically the wrong shape: about 13% too short and 14–16%
too wide on nuScenes, and about 20–26% too wide on ECP (length about right there). This is 5–10× larger than any
difference between pointmap modes, and no pointmap variant changes it. Not yet investigated: whether SAM3D's
generated mesh is systematically rounder than the real object, or whether the PCA-based oriented-box fit around the
mesh overweights width.

## 6. Where the runs and results live

- Mode comparison, full-dataset evaluation: `autolabeling/output/{nuscenes_mini,ecp}/front_cam_mode_<N>/`, scored
  into `evaluation/eval_results/{nuscenes_mini,ecp}/front_cam_mode_<N>/{1class,3class,8class}/metrics_summary.json`.
- Earlier, smaller-scale round (used to exclude modes 3/4/5/9): `autolabeling/output/{nuscenes_mini,ecp}/mode_selection/`.
- Conditioning-input probe: `autolabeling/output/{nuscenes_mini,ecp}/cmp7_probe_<variant>/`, scored into
  `evaluation/eval_results/{nuscenes_mini,ecp}/probe_<variant>/{1class,3class,8class}/metrics_summary.json`.
- All-camera deciding round (modes 2/7/11, final choice: 11): `autolabeling/output/{nuscenes_mini,ecp}/all_cam_mode_<N>/`.
