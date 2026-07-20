# Evaluation Protocol

This document describes the 3D object detection evaluation protocol used in this project.
It is independent of any specific run and applies to all experiments.

---

## Overview

Detection performance is measured using the **nuScenes detection metrics**: mean Average Precision (mAP) and a set of True-Positive quality metrics, combined into the **nuScenes Detection Score (NDS)**.

---

## Matching: 2D Center Distance, Not 3D IoU

Predictions and ground-truth boxes are matched using **2D center distance** in the BEV (bird's-eye-view) plane, ignoring height.  This differs from the IoU-based matching used in PASCAL VOC or COCO.

A prediction is considered a True Positive (TP) for a given class and distance threshold if:
- it belongs to the correct class, and
- its 2D center is within the threshold distance of an unmatched GT box.

Each GT box can be matched at most once; unmatched GT boxes count as False Negatives (FN).

---

## Average Precision (AP)

### Thresholds

AP is computed at four 2D center-distance thresholds: **0.5 m, 1.0 m, 2.0 m, 4.0 m**.

### Per-class, per-threshold AP

For one class at one distance threshold:

1. Sort all predictions by `detection_score` (highest first).
2. Greedily match each prediction to the nearest unmatched GT box within the threshold.
3. Accumulate TP, FP, and FN counts and build the precision–recall curve.
4. AP is the area under the precision–recall curve, computed via numerical integration over 101 recall points in [0, 1].

### Per-class AP

The per-class AP is the **mean of AP across the four distance thresholds**:

```
AP_class = mean(AP@0.5, AP@1.0, AP@2.0, AP@4.0)
```

### mAP

mAP is the **mean of per-class AP over all evaluated classes**:

```
mAP = mean(AP_class  for each class)
```

---

## True-Positive Error Metrics

TP errors are computed **only on matched True Positives**, at the **2.0 m distance threshold** (`dist_th_tp`).
Each metric is averaged over all matched TPs across the dataset.

| Short name | Field in JSON  | Definition |
|------------|----------------|------------|
| **ATE** / mATE | `trans_err`  | L2 distance between predicted and GT 2D center (m) |
| **ASE** / mASE | `scale_err`  | `1 − IoU3D` of aligned boxes (boxes shifted to share the same center before computing IoU) |
| **AOE** / mAOE | `orient_err` | Smallest yaw angle difference between prediction and GT (rad) |
| **AVE** / mAVE | `vel_err`    | L2 error of the 2D velocity vector (m/s) |
| **AAE** / mAAE | `attr_err`   | `1 − accuracy` of the predicted attribute |

Mean metrics (mATE, etc.) are the mean over all evaluated classes.

TP **scores** (used inside NDS) are computed as `max(0, 1 − normalized_error)` so that lower error is better.

---

## NDS — nuScenes Detection Score

NDS combines mAP and the five TP quality scores into one scalar:

```
NDS = (1/10) * (5·mAP + mATE_score + mASE_score + mAOE_score + mAVE_score + mAAE_score)
```

where `mX_score = max(0, 1 − mX / normalization_factor)`.

A higher NDS means both high recall/precision and high quality of matched detections.

---

## Interpretation Guide

### AP variants

| Observation | Likely cause |
|---|---|
| High AP@4.0, low AP@0.5 | Detector finds objects but predicted centers are imprecise |
| High AP at all thresholds | Good localization and confidence ranking |
| Low AP at all thresholds | High false-positive rate, many missed detections, or poor confidence ranking |
| High recall but low precision | Too many false positives; score threshold may need calibration |

### TP error metrics

| Observation | Likely cause |
|---|---|
| High AP but poor AOE | Objects are detected but heading/yaw prediction is unreliable |
| High AP but poor ASE | Centers are accurate but box dimensions (size) are poorly estimated |
| High AP but poor ATE | Matching succeeds at 2 m but predicted centers are still off by ~1–2 m |
| High AP but poor AVE | Static objects are found but velocity estimation is noisy or absent |

### Practical notes

- Unlike ASE/AAE which are bounded to [0, 1], AVE is in **m/s** and is unbounded. A model that always predicts zero velocity will have AVE equal to the mean GT speed of matched objects (e.g. 5–15 m/s for vehicles). High AVE is expected when velocity is not supervised and does not indicate a detection failure.
- AAE is dominated by the attribute vocabulary; if `attribute_name` is always empty, AAE ≈ 1.0 by construction and should be ignored.
- Classes with small evaluation ranges (pedestrian, motorcycle, bicycle: 40 m vs vehicles: 50 m) naturally have fewer GT boxes and noisier AP estimates on small datasets.

---

## ECP Ground Truth Coverage

Ground truth 3D annotations exist for only **33 frames** across **3 scenes**, all from the Strasbourg latesession location (`v1.0-trainval`).

| Scene | Total frames | Annotated frame indices |
|---|---|---|
| `scene-euro-citystrasbourg-scenariolatesession-00002_1` | 2,481 | 560, 600, 639, 679, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320 |
| `scene-euro-citystrasbourg-scenariolatesession-00003_1` | 641 | 239, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640 |
| `scene-euro-citystrasbourg-scenariolatesession-00007_1` | 3,601 | 2560, 2600, 2639, 2680, 2719, 2760, 2800, 2840, 2880, 2919 |

Frames are spaced at roughly 40-frame intervals within each scene. All other scenes in the dataset have no GT annotations and cannot be evaluated.

---

## Run Naming Convention

Evaluation results are stored under `eval_results/<dataset>/<run_name>/`.  The run name encodes the exact pipeline configuration used.

### Structure

```
autolabeling[_lidar_integration_objects_<mode>_body_<mode>[_lidar_<N>_aggregated][_filtered]]
```

- **`autolabeling_baseline`** — no changes from the basic pipeline; no LiDAR integration for either Objects or Body.
- Everything else starts with `autolabeling_lidar_integration` and then specifies the Objects mode, Body mode, and optional LiDAR settings.

### Components

| Component | Meaning |
|---|---|
| `objects_baseline` | SAM3D Objects in baseline mode (no LiDAR depth) |
| `objects_o1` … `objects_o4` | SAM3D Objects with LiDAR mode O1–O4 (increasing depth integration complexity) |
| `body_baseline` | SAM3D Body in baseline mode (no LiDAR depth) |
| `body_b1` | SAM3D Body with LiDAR mode B1 |
| `lidar_2_aggregated` | Temporal frame aggregation: 2 frames before + anchor + 2 frames after (5 sweeps total) |
| `lidar_3_aggregated` | Temporal frame aggregation: 3 frames before + anchor + 3 frames after (7 sweeps total) |
| `filtered` | Ego-vehicle filter and range filter applied to the point cloud |

### Examples

| Run name | Meaning |
|---|---|
| `autolabeling_baseline` | Full baseline — no LiDAR integration |
| `autolabeling_lidar_integration_objects_baseline_body_b1` | Objects baseline, Body with B1 LiDAR |
| `autolabeling_lidar_integration_objects_o1_body_baseline` | Objects O1, Body baseline |
| `autolabeling_lidar_integration_objects_o3_body_b1` | Objects O3, Body B1, single-frame LiDAR |
| `autolabeling_lidar_integration_objects_o3_body_b1_lidar_2_aggregated_filtered` | Objects O3, Body B1, 2-frame aggregation, filtered |
| `autolabeling_lidar_integration_objects_o4_body_b1_lidar_2_aggregated_filtered` | Objects O4, Body B1, 2-frame aggregation, filtered |

---

## Output Files

After running evaluation, the output folder contains:

| File | Description |
|---|---|
| `metrics_summary.json` | Global metrics (mAP, NDS, TP errors) and per-class AP/TP values |
| `metrics_details.json` | Per-class × per-threshold precision–recall arrays (used for PR curve plots) |

To generate a human-readable PDF report and CSV table from these files without re-running evaluation:

```bash
python make_eval_report.py --eval_dir path/to/eval_folder
# optional overrides:
# --out path/to/report.pdf
# --csv path/to/metrics_table.csv
```

The report contains:
- **Page 1**: global metrics and per-class metrics table
- **Page 2**: per-class AP bar chart with mAP reference line
- **Page 3**: precision–recall curves for all classes (at 2.0 m threshold)
- **Page 4**: per-class TP error bars (ATE, ASE, AOE, AVE, AAE)
