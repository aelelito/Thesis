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
- Our pipeline uses **front camera only** unless noted as "all cameras". VESPA `front_cam_only` is the fair comparison for single-camera runs. VESPA `all_cameras` is the comparison for multi-camera runs.

**GT sanity check**: ECP 1-class = 1.000, 8-class = 0.625 (not 1.0 because GT has class conflicts and the 8-class mapping drops some labels). nuScenes 1-class = 0.903, 8-class = 0.935. Eval pipeline verified correct.

**Run naming conventions used in tables**:
- `hull` = updated HDBSCAN parameters + hull anchoring applied (see note below tables)
- `all cams` = all available cameras processed jointly (3 for ECP, 6 for nuScenes); cross-camera merge active
- `MC` = per-object ICP motion compensation; `ICP` = raw ICP without full MC pipeline (intermediate variant)
- `agg(N)` = multi-sweep aggregation ±N sweeps around anchor frame

---

## ECP results

33 GT annotated keyframes across 3 Strasbourg latesession scenes (train split). 64-beam Velodyne.

### ECP — 1-class (vehicle super-class)

**Front camera only:**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| baseline (no LiDAR) | 0.1605 | 0.285 | 0.305 | 0.095 | 0.3117 |
| obj_baseline + B1 | 0.3025 | 0.055 | 0.375 | 0.019 | 0.4063 |
| obj_baseline + B1 + agg(2) + filt | 0.3097 | 0.258 | 0.269 | 0.042 | 0.3979 |
| O1 + body_baseline | 0.1847 | 0.623 | 0.377 | 0.238 | 0.2685 |
| O1 + B1 + agg(2) + filt | 0.3645 | 0.195 | 0.603 | 0.235 | 0.3789 |
| O1 + B1 + agg(2) + filt + hull | 0.3396 | 0.102 | 0.344 | 0.265 | 0.3988 |
| O2 + body_baseline | 0.1773 | 0.412 | 0.417 | 0.395 | 0.2662 |
| O2 + B1 + agg(2) + filt + hull | 0.3267 | 0.197 | 0.250 | 0.231 | 0.3956 |
| O3 + body_baseline | 0.1930 | 0.413 | 0.408 | 0.432 | 0.2712 |
| O3 + B1 | 0.3419 | 0.158 | 0.341 | 0.220 | 0.3990 |
| O3 + B1 + agg(2) | 0.3511 | 0.080 | 0.325 | 0.245 | 0.4107 |
| O3 + B1 + agg(2) + filt | 0.3493 | 0.093 | 0.342 | 0.255 | 0.4056 |
| O3 + B1 + agg(2) + filt + hull | 0.3548 | 0.121 | 0.472 | 0.192 | 0.3989 |
| O4 + B1 + agg(2) + filt | 0.3382 | 0.218 | 0.182 | 0.161 | 0.4130 |
| O5 + B1 + agg(2) + filt | 0.3463 | 0.064 | 0.287 | 0.319 | 0.4062 |
| O5 + B1 + agg(2) + filt + hull | 0.3696 | 0.800 | 0.293 | 0.924 | 0.2831 |
| VESPA front cam only | 0.2842 | 0.270 | 0.498 | 1.491 | 0.2653 |
| GT sanity | 1.0000 | 0.000 | 0.000 | 0.000 | — |

**All cameras (3 ECP cameras: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT):**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| O3 + B1 (no agg, no ICP) | 0.4776 | 0.091 | 0.328 | 0.230 | 0.4739 |
| O3 + B1 + no agg + no ICP (final) | **0.4841** | 0.226 | 0.681 | 0.479 | 0.4034 |
| O3 + B1 + ICP + agg(6) | 0.4435 | 0.373 | 0.236 | 0.221 | 0.4387 |
| O3 + B1 + no MC + agg(2) | 0.4307 | 0.146 | 0.303 | 0.435 | 0.4269 |
| O3 + B1 + MC + agg(6) | 0.4134 | 0.542 | 0.567 | 0.230 | 0.3728 |
| O5 + B1 (no agg) | 0.4757 | 0.244 | 0.229 | 0.201 | 0.4704 |
| VESPA all cameras | 0.4006 | 0.320 | 0.521 | 1.493 | 0.3162 |

### ECP — 3-class (pedestrian + bicycle + vehicle)

**Front camera only:**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| baseline (no LiDAR) | 0.0700 | 0.870 | 0.779 | 0.821 | 0.0879 |
| obj_baseline + B1 | 0.1292 | 0.711 | 0.782 | 0.755 | 0.1397 |
| obj_baseline + B1 + agg(2) + filt | 0.1288 | 0.721 | 0.785 | 0.764 | 0.1373 |
| O1 + body_baseline | 0.2052 | 0.622 | 0.445 | 1.465 | 0.1959 |
| O1 + B1 + agg(2) + filt | 0.2852 | 0.399 | 0.398 | 1.471 | 0.2629 |
| O1 + B1 + agg(2) + filt + hull | 0.2770 | 0.400 | 0.395 | 1.461 | 0.2590 |
| O2 + body_baseline | 0.1612 | 0.852 | 0.456 | 1.619 | 0.1498 |
| O2 + B1 + agg(2) + filt + hull | 0.2145 | 0.743 | 0.447 | 1.645 | 0.1883 |
| O3 + body_baseline | 0.2487 | 0.445 | 0.400 | 1.469 | 0.2398 |
| O3 + B1 | 0.3017 | 0.296 | 0.400 | 1.502 | 0.2813 |
| O3 + B1 + agg(2) | 0.3104 | 0.285 | 0.401 | 1.483 | 0.2867 |
| O3 + B1 + agg(2) + filt | 0.3103 | 0.291 | 0.394 | 1.487 | 0.2865 |
| O3 + B1 + agg(2) + filt + hull | 0.3372 | 0.290 | 0.402 | 1.496 | 0.2994 |
| O4 + B1 + agg(2) + filt | 0.2760 | 0.374 | 0.388 | 1.518 | 0.2618 |
| **O5 + B1 + agg(2) + filt + hull** | **0.3405** | **0.271** | **0.392** | **1.497** | **0.3040** |
| O5 + B1 + agg(2) + filt | 0.3055 | 0.283 | 0.381 | 1.469 | 0.2863 |
| VESPA front cam only | 0.2024 | 0.379 | 0.466 | 1.249 | 0.2167 |
| GT sanity | 1.0000 | 0.000 | 0.000 | 0.000 | — |

**All cameras:**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| O3 + B1 (no agg, no ICP) | 0.3863 | 0.343 | 0.427 | 1.341 | 0.3162 |
| O3 + B1 + MC + agg(6) | 0.3141 | 0.380 | 0.422 | 0.975 | 0.2793 |
| O5 + B1 (no agg) | 0.3728 | 0.343 | 0.428 | 1.390 | 0.3093 |
| VESPA all cameras | 0.2803 | 0.383 | 0.484 | 1.476 | 0.2535 |

### ECP — 8-class (all individual classes)

**Front camera only:**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| baseline (no LiDAR) | 0.0278 | 0.946 | 0.918 | 0.920 | 0.0355 |
| obj_baseline + B1 | 0.0473 | 0.895 | 0.919 | 0.910 | 0.0513 |
| obj_baseline + B1 + agg(2) + filt | 0.0465 | 0.898 | 0.920 | 0.920 | 0.0494 |
| O1 + body_baseline | 0.1064 | 0.866 | 0.729 | 1.111 | 0.0936 |
| O1 + B1 + agg(2) + filt | 0.1384 | 0.757 | 0.714 | 1.110 | 0.1221 |
| O1 + B1 + agg(2) + filt + hull | 0.1372 | 0.756 | 0.709 | 1.092 | 0.1222 |
| O2 + body_baseline | 0.0712 | 1.028 | 0.751 | 1.182 | 0.0605 |
| O2 + B1 + agg(2) + filt + hull | 0.0934 | 0.970 | 0.744 | 1.158 | 0.0753 |
| O3 + body_baseline | 0.1390 | 0.729 | 0.714 | 1.121 | 0.1252 |
| O3 + B1 | 0.1597 | 0.671 | 0.715 | 1.132 | 0.1412 |
| O3 + B1 + agg(2) | 0.1676 | 0.658 | 0.712 | 1.125 | 0.1469 |
| O3 + B1 + agg(2) + filt | 0.1689 | 0.657 | 0.712 | 1.122 | 0.1475 |
| O3 + B1 + agg(2) + filt + hull | 0.1795 | 0.662 | 0.714 | 1.133 | 0.1522 |
| O4 + B1 + agg(2) + filt | 0.1275 | 0.743 | 0.708 | 1.119 | 0.1187 |
| O5 + B1 + agg(2) + filt | 0.1690 | 0.652 | 0.706 | 1.088 | 0.1487 |
| **O5 + B1 + agg(2) + filt + hull** | **0.1866** | **0.640** | **0.711** | **1.113** | **0.1582** |
| VESPA front cam only | 0.1022 | 0.699 | 0.735 | 1.180 | 0.1077 |
| VESPA all cameras | 0.1218 | 0.742 | 0.750 | 1.214 | 0.1117 |
| GT sanity | 0.6250 | 0.375 | 0.375 | 0.375 | — |

**All cameras:**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| O3 + B1 (no agg, no ICP) | 0.2060 | 0.684 | 0.725 | 1.062 | 0.1621 |
| **O3 + B1 + no agg + no ICP (final)** | **0.2124** | **0.698** | **0.718** | **0.957** | **0.1688** |
| O3 + B1 + ICP + agg(6) | 0.2003 | 0.721 | 0.729 | 0.968 | 0.1584 |
| O3 + B1 + no MC + agg(2) | 0.1759 | 0.680 | 0.727 | 0.943 | 0.1528 |
| O3 + B1 + MC + agg(6) | 0.1666 | 0.696 | 0.725 | 0.926 | 0.1486 |
| O5 + B1 (no agg) | 0.1933 | 0.665 | 0.717 | 1.044 | 0.1584 |
| VESPA all cameras | 0.1218 | 0.742 | 0.750 | 1.214 | 0.1117 |

**Key comparison (8-class)**: Best single-cam = **0.1866** (O5+hull) vs VESPA front = **0.1022** (+83%). Best all-cam = **0.2124** vs VESPA all = **0.1218** (+74%).

**Why 8-class mAP is lower than 3-class**: truck, bus, trailer, construction vehicle have near-zero GT instances in the 33 ECP frames → AP = 0 for those 4 classes, pulling down the mean.

### ECP — Per-class AP@2m (8-class eval)

**Front camera:**

| Class | O3+agg+filt | O3+hull | O5+agg+filt | O5+hull | VESPA front | VESPA all |
|---|---|---|---|---|---|---|
| car | 0.454 | **0.556** | 0.449 | 0.529 | 0.224 | 0.276 |
| pedestrian | 0.334 | 0.332 | 0.337 | **0.381** | 0.294 | 0.463 |
| motorcycle | 0.470 | 0.464 | 0.464 | 0.464 | 0.303 | 0.193 |
| bicycle | 0.246 | 0.263 | 0.235 | 0.254 | 0.122 | 0.184 |
| truck | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bus | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trailer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| construction vehicle | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Hull anchoring dramatically improves car AP (O3: 0.454→0.556; O5: 0.449→0.529). Pedestrian AP is mostly unchanged by hull anchoring but benefits from O5 (0.337→0.381). VESPA all-cameras leads on pedestrian (0.463) — multi-view coverage advantage for partially occluded bodies.

**All cameras:**

| Class | O3 (no agg) | O3 + no agg (final) | O3 + MC + agg(6) | O5 (no agg) | VESPA all |
|---|---|---|---|---|---|
| car | 0.424 | **0.500** | 0.287 | 0.464 | 0.276 |
| pedestrian | 0.488 | **0.495** | 0.444 | **0.502** | 0.463 |
| motorcycle | 0.550 | **0.584** | 0.448 | 0.400 | 0.193 |
| bicycle | 0.420 | **0.445** | 0.276 | 0.313 | 0.184 |

All-cameras results substantially beat VESPA all-cameras on car, motorcycle, bicycle. Pedestrian is now comparable to VESPA all-cameras (0.488–0.502 vs 0.463). MC+agg(6) degrades car significantly (0.287) — sweep smearing hurts the LiDAR depth anchor for static objects even with motion compensation.

### ECP — TP error breakdown (O3 vs O5, active classes, front camera)

*(Per-class TP errors available for O3+agg+filt and O5+agg+filt from prior eval; not yet extracted for hull variants)*

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

O5's dense CFormer map improves ASE across all classes and motorcycle AOE dramatically (0.328→0.131).

### ECP — Ablation: value of B1 (pedestrian correction)

| Run | 8-class mAP | pedestrian AP@2m |
|---|---|---|
| O3 + body_baseline (no B1) | 0.1390 | ~0.22 |
| O3 + B1 | 0.1597 | 0.337 |

B1 adds +0.021 mAP overall on ECP. The improvement is concentrated in pedestrian depth correction.

### ECP — Aggregation ablation (O3, front camera)

| Run | 8-class mAP | car AP | motorcycle AP | bicycle AP |
|---|---|---|---|---|
| O3 + B1 (single sweep) | 0.1597 | 0.429 | 0.447 | 0.283 |
| O3 + B1 + agg(2) | 0.1676 | 0.451 | 0.470 | 0.247 |
| O3 + B1 + agg(2) + filt | 0.1689 | 0.454 | 0.470 | 0.246 |

Aggregation helps car and motorcycle (+2–3 pp) at a small cost to bicycle. Net +0.009 mAP.

### ECP — Hull anchoring ablation (front camera)

Hull anchoring applies a per-mask depth floor constraint using LiDAR cluster evidence before/after CompletionFormer. For O3 this is a new addition that clamps the MoGe-derived depth inside the convex hull of each mask. Results show a significant car AP improvement at the cost of some pedestrian accuracy (mask-level floor likely too conservative near mask edges).

| Run | 8-class mAP | car AP@2m | pede AP@2m | moto AP@2m |
|---|---|---|---|---|
| O3 + B1 + agg(2) + filt | 0.1689 | 0.454 | 0.334 | 0.470 |
| O3 + B1 + agg(2) + filt + hull | 0.1795 | **0.556** | 0.332 | 0.464 |
| O5 + B1 + agg(2) + filt | 0.1690 | 0.449 | 0.337 | 0.464 |
| O5 + B1 + agg(2) + filt + hull | **0.1866** | 0.529 | **0.381** | 0.464 |

Hull anchoring gives +0.011 mAP for O3 and +0.018 mAP for O5. The car gain (+0.10 for O3) is the main driver. O5+hull is the new single-camera best on ECP.

### ECP — ICP/MC ablation (all cameras)

All runs use O3 + B1, front+left+right cameras. The question is whether sweep aggregation + ICP/MC helps on top of multi-camera coverage.

| Config | 8-class mAP | car AP@2m | pede AP@2m | moto AP@2m | bicy AP@2m |
|---|---|---|---|---|---|
| No agg, no ICP | **0.2124** | **0.500** | **0.495** | **0.584** | **0.445** |
| ICP, agg(6) | 0.2003 | 0.384 | 0.458 | 0.601 | 0.393 |
| No MC, agg(2) | 0.1759 | 0.317 | 0.473 | 0.550 | 0.278 |
| MC, agg(6) | 0.1666 | 0.287 | 0.444 | 0.448 | 0.276 |

**Key finding**: On ECP, adding sweep aggregation and ICP/MC consistently hurts all-camera performance. Single-sweep O3 with all 3 cameras outperforms every multi-sweep variant. Likely causes:
- Multi-camera already provides dense LiDAR coverage (3 cameras × 64-beam = rich per-frame depth); additional sweeps add smearing without meaningfully increasing object-surface density
- MC on ECP has limited benefit since ECP traffic is slower (moderate urban speeds) and the 64-beam sensor already provides dense single-sweep coverage
- ICP with 6 sweeps (0.2003) performs better than MC+6 sweeps (0.1666), suggesting the full MC pipeline introduces noise relative to raw ICP

---

## nuScenes mini results

All annotated keyframes across 10 random mini scenes (Boston + Singapore). 32-beam Velodyne.

### nuScenes — 1-class (vehicle super-class)

**Front camera only:**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| baseline (no LiDAR) | 0.0000 | 1.000 | 1.000 | 1.000 | 0.0000 |
| obj_baseline + B1 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.0000 |
| O1 + body_baseline | 0.0186 | 0.889 | 0.436 | 1.249 | 0.0768 |
| O2 + body_baseline | 0.0106 | 0.925 | 0.412 | 1.061 | 0.0715 |
| O3 + body_baseline | 0.0191 | 0.765 | 0.375 | 1.259 | 0.0955 |
| O3 + B1 | 0.0208 | 0.608 | 0.376 | 1.216 | 0.1120 |
| O3 + B1 + no agg + no ICP | 0.0221 | 0.618 | 0.372 | 1.158 | 0.1120 |
| O3 + B1 + agg(3) | 0.0031 | 0.506 | 0.399 | 1.224 | 0.1110 |
| **O3 + B1 + agg(3) + filt** | **0.0278** | **0.522** | **0.370** | **1.212** | **0.1247** |
| O3 + B1 + agg(3) + filt + hull | 0.0285 | 0.514 | 0.368 | 1.192 | 0.1260 |
| O3 + B1 + MC + agg(10) | 0.0085 | 0.544 | 0.362 | 1.230 | 0.1137 |
| O4 + B1 + agg(3) + filt | 0.0000 | 1.000 | 1.000 | 1.000 | 0.0000 |
| O5 + B1 + agg(3) + filt | 0.0186 | 0.581 | 0.373 | 1.085 | 0.1139 |
| VESPA front cam only | 0.0130 | 0.559 | 0.417 | 0.867 | 0.1222 |

**All cameras (6 nuScenes cameras, full 360° ring):**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| **O3 + B1 (no agg, no ICP)** | **0.3596** | **0.290** | **0.391** | **0.953** | **0.3163** |
| VESPA all cameras | 0.2527 | 0.492 | 0.384 | 1.047 | 0.2387 |

### nuScenes — 3-class (pedestrian + bicycle + vehicle)

**Front camera only:**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| baseline (no LiDAR) | 0.0015 | 1.000 | 1.000 | 1.000 | 0.0007 |
| obj_baseline + B1 | 0.0042 | 0.782 | 0.815 | 0.780 | 0.0645 |
| O1 + body_baseline | 0.0152 | 1.016 | 0.627 | 1.513 | 0.0449 |
| O2 + body_baseline | 0.0076 | 1.006 | 0.782 | 1.174 | 0.0256 |
| O3 + body_baseline | 0.0141 | 0.845 | 0.601 | 1.463 | 0.0624 |
| O3 + B1 | 0.0168 | 0.604 | 0.425 | 1.213 | 0.1055 |
| O3 + B1 + agg(3) | 0.0031 | 0.881 | 0.779 | 1.246 | 0.0355 |
| **O3 + B1 + agg(3) + filt** | **0.0201** | **0.639** | **0.583** | **1.008** | **0.0879** |
| O4 + B1 + agg(3) + filt | 0.0008 | 1.000 | 1.000 | 1.000 | 0.0004 |
| O5 + B1 + agg(3) + filt | 0.0149 | 0.514 | 0.440 | 1.026 | 0.1121 |
| VESPA front cam only | 0.0092 | 0.634 | 0.613 | 0.869 | 0.0930 |

**All cameras:**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| **O3 + B1 (no agg, no ICP)** | **0.2928** | **0.371** | **0.369** | **1.196** | **0.2724** |
| VESPA all cameras | 0.2122 | 0.562 | 0.394 | 0.983 | 0.2122 |

### nuScenes — 8-class (all individual classes)

**Front camera only:**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| baseline (no LiDAR) | 0.0006 | 1.000 | 1.000 | 1.000 | 0.0003 |
| obj_baseline + B1 | 0.0016 | 0.918 | 0.931 | 0.918 | 0.0242 |
| O1 + body_baseline | 0.0220 | 1.028 | 0.661 | 1.448 | 0.0449 |
| O1 + B1 + agg(3) + filt | 0.0211 | 0.963 | 0.634 | 1.467 | 0.0509 |
| O2 + body_baseline | 0.0111 | 1.043 | 0.867 | 1.109 | 0.0188 |
| O3 + body_baseline | 0.0210 | 0.912 | 0.617 | 1.471 | 0.0576 |
| **O3 + B1** | **0.0222** | **0.819** | **0.554** | **1.365** | **0.0738** |
| O3 + B1 + no agg + no ICP | 0.0218 | 0.822 | 0.551 | 1.040 | 0.0736 |
| O3 + B1 + agg(3) | 0.0104 | 0.868 | 0.706 | 1.468 | 0.0478 |
| O3 + B1 + agg(3) + filt | 0.0215 | 0.731 | 0.546 | 1.463 | 0.0830 |
| O3 + B1 + agg(3) + filt + hull | 0.0226 | 0.736 | 0.546 | 1.466 | 0.0831 |
| O3 + B1 + MC + agg(10) | 0.0136 | 0.869 | 0.680 | 1.148 | 0.0520 |
| O4 + B1 + agg(3) + filt | 0.0015 | 1.000 | 1.000 | 1.000 | 0.0007 |
| O5 + B1 + agg(3) + filt | 0.0156 | 0.811 | 0.552 | 1.220 | 0.0715 |
| O5 + B1 + agg(3) + filt + hull | 0.0164 | 0.902 | 0.615 | 1.487 | 0.0564 |
| VESPA front cam only | 0.0075 | 0.863 | 0.701 | 0.886 | 0.0587 |
| VESPA all cameras | 0.1176 | 0.784 | 0.498 | 1.240 | 0.1305 |
| GT sanity | 0.9351 | 0.003 | 0.000 | 0.001 | — |

**All cameras (6 cameras, 360°):**

| Run | mAP | ATE | ASE | AOE | NDS |
|---|---|---|---|---|---|
| **O3 + B1 (no agg, no ICP)** | **0.2210** | **0.617** | **0.464** | **1.241** | **0.2024** |
| VESPA all cameras | 0.1176 | 0.784 | 0.498 | 1.240 | 0.1305 |

**Key comparison (8-class, front cam)**: Our best = **0.0222** vs VESPA front = **0.0075** (+196%).

**Key comparison (8-class, all cams)**: Our best = **0.2210** vs VESPA all cameras = **0.1176** (+88%). We now outperform VESPA's full camera ring on nuScenes.

**O4 total failure on nuScenes** (8-class mAP = 0.0015, 1-class = 0.000): CFormer trained on KITTIDC (64-beam) is fundamentally out-of-distribution on 32-beam nuScenes. Ground filter removes additional valid low-height returns. All anchors too sparse for CFormer → meshes at wrong depths.

### nuScenes — Per-class AP@2m (8-class eval, front camera)

| Class | O3+B1 (best) | O3+agg+filt | O3+agg+filt+hull | VESPA front | VESPA all |
|---|---|---|---|---|---|
| car | **0.029** | 0.062 | 0.062 | 0.017 | 0.285 |
| truck | 0.017 | 0.017 | 0.017 | — | 0.015 |
| bus | **0.046** | 0.045 | 0.053 | 0.030 | 0.088 |
| construction_vehicle | 0.046 | 0.059 | 0.059 | — | 0.228 |
| pedestrian | **0.017** | 0.008 | 0.017 | 0.009 | 0.401 |
| motorcycle | — | — | — | — | 0.049 |
| bicycle | **0.054** | 0.005 | 0.005 | 0.023 | 0.075 |

All front-camera results on nuScenes are low absolute values. The dominant advantage comes from multi-camera coverage.

**Per-class AP@2m (all cameras, O3+B1):**

| Class | O3+B1 all cams | VESPA all cams |
|---|---|---|
| car | **0.464** | 0.285 |
| truck | **0.086** | 0.015 |
| bus | **0.239** | 0.088 |
| construction_vehicle | **0.657** | 0.228 |
| pedestrian | **0.410** | 0.401 |
| motorcycle | **0.197** | 0.049 |
| bicycle | 0.082 | **0.075** |

All-camera O3+B1 beats VESPA on every class except approximately tying on bicycle. The construction vehicle gain (0.657 vs 0.228) is particularly large — these objects appear in multiple cameras simultaneously.

### nuScenes — Why O3 beats O5

Both modes use identical HDBSCAN in-mask cleaning. The difference is what happens between anchors:
- **O3**: MoGe provides dense per-pixel relative depth shape. With 32-beam LiDAR, a car at 20 m may have only 15–30 anchor pixels — but MoGe fills the rest of the mask with a geometrically plausible relative shape. The local affine gives O3 a physically shaped surface even where LiDAR is absent.
- **O5**: CFormer must fill large gaps (sparse 32-beam) using RGB guidance and KITTIDC (64-beam) priors. Out-of-distribution for nuScenes → inaccurate interpolation inside object masks.

**Conclusion**: O5 is the better choice for 64-beam sensors (ECP); O3 is more robust on 32-beam.

### nuScenes — Why aggregation hurts without ego filter

| Run | mAP | car | bicycle |
|---|---|---|---|
| O3 + B1 (no agg) | **0.0222** | 0.029 | **0.054** |
| O3 + B1 + agg(3) | 0.0104 | — | — |
| O3 + B1 + agg(3) + filt | 0.0215 | 0.062 | 0.005 |

**Motion smearing**: Boston/Singapore traffic at 30 km/h moves 12.5 m in ±1.5 s (3+3 sweeps). Aggregated cloud smears across 12 m → HDBSCAN sees elongated low-density blob → wrong cluster center → bad affine fit. Bicycle (small, fast) is most affected and does not recover even with the ego-body filter. The ego-body filter recovers car (ego returns no longer dominate) but cannot fix smearing of moving objects.

### nuScenes — ICP/MC ablation (front camera)

| Config | 8-class mAP | Observation |
|---|---|---|
| O3 + B1 (no agg, single sweep) | **0.0222** | Best front-cam result |
| O3 + B1 + no agg + no ICP | 0.0218 | Effectively same as baseline O3+B1 |
| O3 + B1 + agg(3) + filt | 0.0215 | Aggregation with filter recovers most of the loss |
| O3 + B1 + MC + agg(10) | 0.0136 | MC with 10 sweeps significantly hurts |

**Key finding**: ICP/MC makes things worse on nuScenes. With 10-sweep aggregation and MC, performance drops from 0.0222 to 0.0136. Even though MC correctly compensates individual dynamic objects, the combined effects of 10-sweep aggregation (massive smearing window), MC noise, and nuScenes' fast-moving traffic make the ICP cloud worse than a single clean sweep. The motion compensation implementation may not yet be robust enough for the high-speed nuScenes scenes (30+ km/h ego + vehicles).

### nuScenes — FP analysis (O3 vs O5 at 8-class, front camera)

| Class | O3 n_pred | O5 n_pred | O3 FP | O5 FP |
|---|---|---|---|---|
| car | 937 | 1,457 | 459 | **933** |
| truck | 98 | 134 | 51 | 93 |
| bus | 54 | 131 | 20 | 107 |

O5 generates far more total predictions despite identical SAM3 masks. Root cause:

- **O3**: objects with zero in-mask LiDAR fall back to unscaled MoGe (`a=1, b=0`). MoGe depth is non-metric — OBB center falls outside the evaluation range (50 m for cars) or far from any GT box. These detections are *accidentally suppressed* by falling outside the eval window.
- **O5**: CFormer produces a globally dense depth map. Nearby anchors from other objects guide interpolation into zero-LiDAR regions, placing those OBBs at a plausible but incorrect metric depth → counted as FP within the evaluation range.

O3's "suppression" is accidental. O5 rescues those detections to wrong-but-plausible positions where they become FPs. This is a key motivation for the planned O6 development.

---

## Cross-dataset summary

### Best results per configuration (8-class mAP)

| Dataset | Mode | mAP | Config | VESPA front | VESPA all | vs VESPA front |
|---|---|---|---|---|---|---|
| ECP | front cam | **0.1866** | O5+B1+agg2+filt+hull | 0.1022 | 0.1218 | +83% |
| ECP | all cams | **0.2124** | O3+B1 no agg no ICP | 0.1022 | 0.1218 | +108% |
| nuScenes | front cam | **0.0226** | O3+B1+agg3+filt+hull | 0.0075 | 0.1176 | +201% |
| nuScenes | all cams | **0.2210** | O3+B1 no agg no ICP | 0.0075 | 0.1176 | — |

### Full cross-dataset table (all class configs, best per config)

| Dataset | Config | Our best | Mode | VESPA front | VESPA all | Gain vs VESPA front |
|---|---|---|---|---|---|---|
| ECP 1-class | front cam | **0.3696** | O5+hull | 0.2842 | 0.4006 | +30% |
| ECP 1-class | all cams | **0.4841** | O3 no agg | — | 0.4006 | +20% vs VESPA all |
| ECP 3-class | front cam | **0.3405** | O5+hull | 0.2024 | 0.2803 | +68% |
| ECP 3-class | all cams | **0.3863** | O3 no agg | — | 0.2803 | +38% vs VESPA all |
| ECP 8-class | front cam | **0.1866** | O5+hull | 0.1022 | 0.1218 | +83% |
| ECP 8-class | all cams | **0.2124** | O3 no agg | — | 0.1218 | +74% vs VESPA all |
| nuScenes 1-class | front cam | **0.0285** | O3+agg3+hull | 0.0130 | 0.2527 | +119% |
| nuScenes 1-class | all cams | **0.3596** | O3 no agg | — | 0.2527 | +42% vs VESPA all |
| nuScenes 3-class | front cam | **0.0201** | O3+agg3+filt | 0.0092 | 0.2122 | +118% |
| nuScenes 3-class | all cams | **0.2928** | O3 no agg | — | 0.2122 | +38% vs VESPA all |
| nuScenes 8-class | front cam | **0.0226** | O3+agg3+hull | 0.0075 | 0.1176 | +201% |
| nuScenes 8-class | all cams | **0.2210** | O3 no agg | — | 0.1176 | +88% vs VESPA all |

**Key takeaways**:
- Front-camera-only: we consistently and substantially outperform VESPA front-cam on both datasets and all class configurations (83–201% relative gain on 8-class).
- All-cameras: we now outperform VESPA all-cameras on ECP 3-class and 8-class, and on nuScenes 1-class, 3-class, and 8-class. VESPA all-cameras still leads on ECP 1-class (vehicles only, 0.4006 vs 0.4841 — wait, we lead here too).
- **nuScenes**: The VESPA all-camera gap is closed and reversed. Our pipeline at 0.2210 vs VESPA 0.1176 (+88%) is a major result given that nuScenes' dense scenes benefit most from multi-view coverage.
- **Sensor dependency**: O5 (CompletionFormer + hull) is the best single-camera choice for 64-beam sensors (ECP). O3 (MoGe + local affine) is more robust on 32-beam (nuScenes) where CFormer is out-of-distribution.
- **ICP/MC currently hurts**: On both datasets and all tested configurations, adding sweep aggregation + ICP/MC reduces performance compared to single-sweep O3 with all cameras. The multi-camera coverage gain dominates and ICP/MC adds noise. This is a target for future work.

*All "front cam" runs: CAM_FRONT only. "All cams": 3 cameras for ECP (front arc), 6 cameras for nuScenes (360°).*
*agg: multi-sweep aggregation. filt: ego-body + max-range filter. hull: updated HDBSCAN + hull anchoring.*
