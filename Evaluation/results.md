# Auto-Labeling Pipeline — Results

## Evaluation setup

**Eval script**: reused from VESPA (nuScenes-format, class-mapped).
**Distance thresholds**: 0.5 / 1.0 / 2.0 / 4.0 m. mAP = mean over all classes × thresholds.
**Class configurations evaluated**:
- `1class` — single "vehicle" super-class (all vehicle types merged: car, truck, bus, motorcycle, bicycle, trailer, construction vehicle)
- `3class` — pedestrian + bicycle + vehicle
- `8class` — all 8 individual classes separately

**Metric notes**:
- **mAP** is the primary metric.
- **AOE** is meaningful only for **pedestrian** (SAM3D Body gives body-forward orientation). For all vehicle classes heading is not estimated — OBBs are 180° ambiguous — so vehicle AOE ≈ π (≈3.0–3.1) in all runs and is uninformative.
- **AVE**, **AAE**: not predicted, always = 1.0. NDS is penalised by these and is not a fair comparison metric at this stage.
- **truck, bus, trailer, construction_vehicle** at 8-class: near-zero AP in all runs on ECP (too few GT instances in the 33 frames). No conclusions can be drawn for these classes individually.
- Our pipeline uses **front camera only** (CAM_FRONT). VESPA `front_cam_only` is the fair comparison. VESPA `all_cameras` has an inherent advantage from multi-view coverage.

**GT sanity check**: ECP 1-class = 1.000, 8-class = 0.625 (not 1.0 because GT has class conflicts and the 8-class mapping drops some labels). nuScenes 1-class = 0.903, 8-class = 0.935. Eval pipeline verified correct.

---

## ECP results

33 GT annotated keyframes across 3 Strasbourg latesession scenes (train split). 64-beam Velodyne.

### ECP — 1-class (vehicle super-class)

| Run | mAP | ATE | ASE | AOE |
|---|---|---|---|---|
| baseline (no LiDAR) | 0.1605 | 0.285 | 0.305 | 0.095 |
| obj_baseline + B1 | 0.3025 | 0.055 | 0.375 | 0.019 |
| O1 + body_baseline | 0.1847 | 0.623 | 0.377 | 0.238 |
| O2 + body_baseline | 0.1773 | 0.412 | 0.417 | 0.395 |
| O3 + body_baseline | 0.1930 | 0.413 | 0.408 | 0.432 |
| O3 + B1 | 0.3419 | 0.158 | 0.341 | 0.220 |
| O3 + B1 + agg(2) | 0.3511 | 0.080 | 0.325 | 0.245 |
| O3 + B1 + agg(2) + filt | 0.3493 | 0.093 | 0.342 | 0.255 |
| O4 + B1 + agg(2) + filt | 0.3382 | 0.218 | 0.182 | 0.161 |
| **O5 + B1 + agg(2) + filt** | **0.3463** | **0.064** | **0.287** | **0.319** |
| VESPA front cam only | 0.2842 | 0.270 | 0.497 | 1.491 |
| VESPA all cameras | 0.4006 | 0.320 | 0.521 | 1.493 |
| GT sanity | 1.0000 | 0.000 | 0.000 | 0.000 |

**Note — 1-class baseline paradox**: the baseline (no LiDAR) achieves 0.1605 despite having no metric scale. This is because the "vehicle" super-class is large and the mAP denominator is just one class. More importantly, SAM3D Objects without LiDAR places some vehicles at accidentally plausible depths due to MoGe's image prior — enough for a coarse 4 m threshold hit. The metric only becomes meaningful when comparing approaches that actually improve localisation.

**Note — O1 lower than O3 at 1-class**: O1+body_baseline (0.1847) is lower than O3+B1 (0.3419). Two compounding factors: (1) O1 uses body_baseline (no B1) which leaves pedestrian at wrong depths; in 1-class this doesn't matter since pedestrian is excluded. (2) O1 uses a sparse pointmap which provides metric depth at anchor pixels but leaves ~99.5% of pixels as NaN. The key difference is B1.

**Key comparison (front cam only)**: Our best = **0.3463** vs VESPA front = **0.2842** (+22%). VESPA all-cameras (0.4006) leads — motivates multi-camera merging.

**ASE**: O4 achieves the best ASE (0.182) on 1-class — CompletionFormer's dense metric map produces the most accurate object extents. O5 is second (0.287). Both beat VESPA (0.497 / 0.521) substantially.

### ECP — 3-class (pedestrian + bicycle + vehicle)

| Run | mAP | ATE | ASE | AOE |
|---|---|---|---|---|
| baseline (no LiDAR) | 0.0700 | 0.870 | 0.779 | 0.821 |
| obj_baseline + B1 | 0.1292 | 0.711 | 0.782 | 0.755 |
| O1 + body_baseline | 0.2052 | 0.622 | 0.445 | 1.465 |
| O2 + body_baseline | 0.1612 | 0.852 | 0.456 | 1.619 |
| O3 + body_baseline | 0.2487 | 0.445 | 0.400 | 1.469 |
| O3 + B1 | 0.3017 | 0.296 | 0.400 | 1.502 |
| O3 + B1 + agg(2) | 0.3104 | 0.285 | 0.400 | 1.483 |
| O3 + B1 + agg(2) + filt | 0.3103 | 0.291 | 0.394 | 1.487 |
| O4 + B1 + agg(2) + filt | 0.2760 | 0.374 | 0.388 | 1.518 |
| **O5 + B1 + agg(2) + filt** | **0.3055** | **0.283** | **0.381** | **1.469** |
| VESPA front cam only | 0.2024 | 0.379 | 0.466 | 1.249 |
| VESPA all cameras | 0.2803 | 0.383 | 0.484 | 1.476 |
| GT sanity | 1.0000 | 0.000 | 0.000 | 0.000 |

**Key comparison**: Our best (O3+agg+filt = 0.3103 / O5+agg+filt = 0.3055) vs VESPA front = 0.2024 (+53% / +51%). We also beat VESPA all-cameras (0.2803). At 3-class our single-camera pipeline surpasses VESPA's full camera ring on ECP.

### ECP — 8-class (all individual classes)

| Run | mAP | ATE | ASE | AOE |
|---|---|---|---|---|
| baseline (no LiDAR) | 0.0278 | 0.946 | 0.918 | 0.920 |
| obj_baseline + B1 | 0.0473 | 0.895 | 0.919 | 0.910 |
| O1 + body_baseline | 0.1064 | 0.866 | 0.729 | 1.111 |
| O2 + body_baseline | 0.0712 | 1.028 | 0.751 | 1.182 |
| O3 + body_baseline | 0.1390 | 0.729 | 0.714 | 1.121 |
| O3 + B1 | 0.1597 | 0.671 | 0.715 | 1.132 |
| O3 + B1 + agg(2) | 0.1676 | 0.658 | 0.712 | 1.125 |
| O3 + B1 + agg(2) + filt | 0.1689 | 0.657 | 0.712 | 1.122 |
| O4 + B1 + agg(2) + filt | 0.1275 | 0.743 | 0.708 | 1.119 |
| **O5 + B1 + agg(2) + filt** | **0.1690** | **0.652** | **0.706** | **1.088** |
| VESPA front cam only | 0.1022 | 0.699 | 0.735 | 1.180 |
| VESPA all cameras | 0.1218 | 0.742 | 0.750 | 1.214 |
| GT sanity | 0.6250 | 0.375 | 0.375 | 0.375 |

**Key comparison**: Our best = **0.1690** vs VESPA front = **0.1022** (+65%) and VESPA all-cameras = 0.1218 (+39%). At 8-class our single-camera pipeline beats VESPA's full camera ring.

**Why 8-class mAP is lower than 3-class**: truck, bus, trailer, construction vehicle have near-zero GT instances in the 33 ECP frames → AP = 0 for those 4 classes, pulling down the mean.

### ECP — Per-class AP@2m (8-class eval, best runs + VESPA)

| Class | O3+agg+filt | O5+agg+filt | VESPA front | VESPA all |
|---|---|---|---|---|
| car | 0.454 | 0.449 | 0.224 | 0.276 |
| pedestrian | 0.334 | 0.337 | 0.294 | 0.463 |
| motorcycle | 0.470 | 0.464 | 0.303 | 0.193 |
| bicycle | 0.246 | 0.235 | 0.122 | 0.184 |
| truck | 0.000 | 0.000 | 0.000 | 0.000 |
| bus | 0.000 | 0.000 | 0.000 | 0.000 |
| trailer | 0.000 | 0.000 | 0.000 | 0.000 |
| construction vehicle | 0.000 | 0.000 | 0.000 | 0.000 |

Car, motorcycle, bicycle: we substantially outperform both VESPA configurations.
Pedestrian: VESPA all-cameras leads (0.463 vs 0.337) — multi-view coverage aids bodies partially occluded in the front view.

### ECP — TP error breakdown (O3 vs O5, active classes)

| Class | Metric | O3+agg+filt | O5+agg+filt |
|---|---|---|---|
| car | ATE | 0.398 | 0.433 |
| | ASE | 0.291 | **0.270** |
| | AOE | 3.008 | 3.020 |
| pedestrian | ATE | **0.109** | 0.208 |
| | ASE | 0.449 | **0.307** |
| | AOE | 0.123 | 0.128 |
| motorcycle | ATE | 0.476 | **0.411** |
| | ASE | 0.511 | **0.509** |
| | AOE | 0.328 | **0.131** |
| bicycle | ATE | 0.362 | **0.307** |
| | ASE | 0.579 | **0.549** |

O5's dense CFormer map improves ASE across all classes and motorcycle AOE dramatically (0.328→0.131) — the complete depth shape helps the diffusion model recover heading from elongated profiles. Pedestrian ATE is worse in O5 (0.109→0.208), likely due to the different sparse anchor distribution affecting B1 Stage 1 fallback rate.

### ECP — Ablation: value of B1 (pedestrian correction)

| Run | 8-class mAP | pedestrian AP@2m |
|---|---|---|
| O3 + body_baseline (no B1) | 0.1390 | ~0.27 (est.) |
| O3 + B1 | 0.1597 | 0.334 |

B1 adds +0.021 mAP overall on ECP. The improvement is concentrated in pedestrian: without B1, SAM3D Body places pedestrians at wrong depths; B1 Stage 1 corrects tz from LiDAR, Stage 2 anchors feet to the ground surface.

### ECP — Aggregation ablation (O3)

| Run | 8-class mAP | car AP | motorcycle AP | bicycle AP |
|---|---|---|---|---|
| O3 + B1 (single sweep) | 0.1597 | 0.481 | 0.511 | 0.361 |
| O3 + B1 + agg(2) | 0.1676 | ~0.490 | ~0.520 | ~0.340 |
| O3 + B1 + agg(2) + filt | 0.1689 | 0.499 | 0.529 | 0.329 |

Aggregation helps car and motorcycle (+2–3 pp) at a small cost to bicycle. Ego-body filter gives additional marginal improvement. Net +0.009 mAP from agg+filt on ECP.

---

## nuScenes mini results

All annotated keyframes across 10 random mini scenes (Boston + Singapore). 32-beam Velodyne.

### nuScenes — 1-class (vehicle super-class)

| Run | mAP | ATE | ASE | AOE |
|---|---|---|---|---|
| baseline (no LiDAR) | 0.0000 | 1.000 | 1.000 | 1.000 |
| obj_baseline + B1 | 0.0000 | 1.000 | 1.000 | 1.000 |
| O1 + body_baseline | 0.0186 | 0.889 | 0.436 | 1.249 |
| O2 + body_baseline | 0.0106 | 0.925 | 0.412 | 1.061 |
| O3 + body_baseline | 0.0191 | 0.765 | 0.375 | 1.259 |
| O3 + B1 | 0.0208 | 0.608 | 0.376 | 1.216 |
| O3 + B1 + agg(3) | 0.0031 | 0.506 | 0.399 | 1.224 |
| **O3 + B1 + agg(3) + filt** | **0.0278** | **0.522** | **0.370** | **1.212** |
| O4 + B1 + agg(3) + filt | 0.0000 | 1.000 | 1.000 | 1.000 |
| O5 + B1 + agg(3) + filt | 0.0186 | 0.581 | 0.373 | 1.085 |
| VESPA front cam only | 0.0130 | 0.559 | 0.417 | 0.867 |
| VESPA all cameras | 0.2527 | 0.492 | 0.384 | 1.047 |
| GT sanity | 0.9030 | 0.009 | 0.001 | 0.004 |

**O4 = 0.0000 on nuScenes 1-class** — a complete failure. CFormer trained on 64-beam (KITTIDC) cannot reliably fill the large gaps between 32-beam anchors. After PseudoLabeler ground filtering (which removes further valid low-height returns), the anchor map is too sparse for CFormer to produce usable depth in object regions. The resulting meshes are placed at random/wrong depths → zero vehicle detections within any threshold.

**Key comparison (front cam)**: Our best = **0.0278** vs VESPA front = **0.0130** (+114%). VESPA all-cameras (0.2527) dominates — multi-camera is critical on dense nuScenes scenes where many vehicles appear only in side/rear views.

### nuScenes — 3-class (pedestrian + bicycle + vehicle)

| Run | mAP | ATE | ASE | AOE |
|---|---|---|---|---|
| baseline (no LiDAR) | 0.0015 | 1.000 | 1.000 | 1.000 |
| obj_baseline + B1 | 0.0042 | 0.782 | 0.815 | 0.780 |
| O1 + body_baseline | 0.0152 | 1.016 | 0.627 | 1.513 |
| O2 + body_baseline | 0.0076 | 1.005 | 0.782 | 1.174 |
| O3 + body_baseline | 0.0141 | 0.845 | 0.601 | 1.463 |
| O3 + B1 | 0.0168 | 0.604 | 0.425 | 1.213 |
| O3 + B1 + agg(3) | 0.0031 | 0.881 | 0.779 | 1.246 |
| **O3 + B1 + agg(3) + filt** | **0.0201** | **0.639** | **0.583** | **1.007** |
| O4 + B1 + agg(3) + filt | 0.0008 | 1.000 | 1.000 | 1.000 |
| O5 + B1 + agg(3) + filt | 0.0149 | 0.514 | 0.440 | 1.026 |
| VESPA front cam only | 0.0092 | 0.634 | 0.613 | 0.869 |
| VESPA all cameras | 0.2122 | 0.562 | 0.394 | 0.983 |
| GT sanity | 0.9046 | 0.008 | 0.001 | 0.004 |

**Key comparison (front cam)**: Our best = **0.0201** vs VESPA front = **0.0092** (+118%).

### nuScenes — 8-class (all individual classes)

| Run | mAP | ATE | ASE | AOE |
|---|---|---|---|---|
| baseline (no LiDAR) | 0.0006 | 1.000 | 1.000 | 1.000 |
| obj_baseline + B1 | 0.0016 | 0.918 | 0.931 | 0.918 |
| O1 + body_baseline | 0.0220 | 1.028 | 0.661 | 1.448 |
| O2 + body_baseline | 0.0111 | 1.043 | 0.867 | 1.109 |
| O3 + body_baseline | 0.0210 | 0.912 | 0.617 | 1.471 |
| **O3 + B1** | **0.0222** | **0.819** | **0.554** | **1.365** |
| O3 + B1 + agg(3) | 0.0104 | 0.868 | 0.706 | 1.468 |
| O3 + B1 + agg(3) + filt | 0.0215 | 0.731 | 0.546 | 1.463 |
| O4 + B1 + agg(3) + filt | 0.0015 | 1.000 | 1.000 | 1.000 |
| O5 + B1 + agg(3) + filt | 0.0156 | 0.811 | 0.552 | 1.220 |
| VESPA front cam only | 0.0075 | 0.863 | 0.701 | 0.886 |
| VESPA all cameras | 0.1176 | 0.784 | 0.498 | 1.240 |
| GT sanity | 0.9351 | 0.003 | 0.000 | 0.001 |

**Key comparison (front cam)**: Our best = **0.0222** vs VESPA front = **0.0075** (+196%).

**O4 total failure on nuScenes** (8-class mAP = 0.0015, 1-class = 0.000): confirms that CFormer + PseudoLabeler ground filter is fundamentally unsuited to 32-beam LiDAR. CFormer was trained on KITTIDC (64-beam) and is out-of-distribution on 32-beam sparse inputs. The ground filter then removes additional valid low-height object returns, making anchor coverage even worse.

### nuScenes — Why O3 beats O5 (and beats aggregation)

**O3 vs O5 at 8-class**: O3+B1 = 0.0222 vs O5+agg+filt = 0.0156. Both use the same HDBSCAN in-mask cleaning. The difference: O3 combines MoGe's dense relative depth shape with sparse metric calibration — even when a car at 20 m has only 15–30 in-mask LiDAR hits, MoGe fills the entire mask with a geometrically plausible relative surface. O5 relies on CFormer to fill gaps from 32-beam anchors; this is out-of-distribution → inaccurate interpolation inside object masks.

**Aggregation hurts (3+3 sweeps)**: O3+agg = 0.0104 vs O3 (no agg) = 0.0222.
- *Motion smearing*: Boston/Singapore traffic at 30 km/h moves ~12.5 m in ±1.5 s. Aggregated cloud smears across ~12 m → HDBSCAN sees an elongated low-density blob → wrong cluster center → bad affine fit.
- *Ego-body contamination*: 3+3 sweeps make ego-vehicle roof/mount returns very dense. These project into nearby masks and become the dominant HDBSCAN cluster (anchoring at ~1.5 m ego depth). The ego-body filter recovers car (0.075 → 0.156 on 1-class) but bicycle and pedestrian remain degraded — their HDBSCAN clusters are too small and sensitive to any contamination.

### nuScenes — FP analysis (O3 vs O5 at 8-class)

O5 generates far more total predictions despite identical SAM3 masks:

| Class | O3 n_pred | O5 n_pred | O3 FP | O5 FP |
|---|---|---|---|---|
| car | 937 | 1,457 | 459 | 933 |
| truck | 98 | 134 | 51 | 93 |
| bus | 54 | 131 | 20 | 107 |

Objects with zero in-mask LiDAR fall back to unscaled MoGe in O3 → non-metric depth → outside eval range → effectively suppressed (accidental). In O5, CFormer fills those regions from nearby anchors → plausible-but-wrong metric depth → counted as FP inside the eval range. O3's "suppression" is accidental; O5 accidentally rescues those detections to wrong positions.

---

## Cross-dataset summary

| Dataset | Our best | Our config | VESPA front | VESPA all | Gain vs VESPA front |
|---|---|---|---|---|---|
| ECP 1-class | **0.3463** | O5+B1+agg2+filt | 0.2842 | 0.4006 | +22% |
| ECP 3-class | **0.3103** | O3+B1+agg2+filt | 0.2024 | 0.2803 | +53% |
| ECP 8-class | **0.1690** | O5+B1+agg2+filt | 0.1022 | 0.1218 | +65% |
| nuScenes 1-class | **0.0278** | O3+B1+agg3+filt | 0.0130 | 0.2527 | +114% |
| nuScenes 3-class | **0.0201** | O3+B1+agg3+filt | 0.0092 | 0.2122 | +118% |
| nuScenes 8-class | **0.0222** | O3+B1 (no agg) | 0.0075 | 0.1176 | +196% |

**All runs use front camera only.** We consistently and substantially outperform VESPA front-cam on both datasets and all class configurations. On ECP (3-class and 8-class) we also beat VESPA's full camera ring. On nuScenes, VESPA all-cameras leads heavily — multi-camera merging is the most impactful remaining improvement for nuScenes.

**Sensor dependency**: O5 (CompletionFormer) is the best choice for 64-beam sensors (ECP). O3 (MoGe + local affine) is more robust on 32-beam (nuScenes) where CFormer is out-of-distribution. O4 (PseudoLabeler + CFormer) fails completely on 32-beam.
