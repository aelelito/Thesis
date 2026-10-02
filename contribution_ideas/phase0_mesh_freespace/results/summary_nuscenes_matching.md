# nuScenes-style (center-distance) matching -- claims 1/2/3/5 at each threshold

Greedy, per class, confidence-ranked, BEV center distance -- the devkit's own criterion, NOT IoU. Run per camera (this analysis does not cross-camera-merge). Claim 4 (the floor control) is not redone here -- it needs the mesh at the matched GT's specific pose; see summary.md for that one, matched by LiDAR containment instead.

## Threshold 0.5 m -- 1393 / 7445 detections matched
### 1. Mask fit
| dataset       |    n |   recall_median |   n_clean |   clean_frac |   iou_median |
|:--------------|-----:|----------------:|----------:|-------------:|-------------:|
| ecp           |  215 |          0.6677 |        85 |       0.3953 |       0.3925 |
| nuscenes_mini | 7230 |          0.8495 |      2538 |       0.351  |       0.5577 |

### 2. Free-space vs GT floor
| dataset       |   n |   mesh_frac_median |   mesh_excess_frac_median |   mesh_excess_frac_p |   mesh_excess_m3_median |   mesh_excess_m3_p |   obb_frac_median |   obb_excess_frac_median |   obb_excess_frac_p |   obb_excess_m3_median |   obb_excess_m3_p |   gt_frac_median |
|:--------------|----:|-------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|------------------:|-------------------------:|--------------------:|-----------------------:|------------------:|-----------------:|
| ecp           | 112 |             0.371  |                   -0.0702 |               0.0002 |                 -0.7105 |                  0 |            0.3774 |                  -0.0426 |              0.0002 |                 0.253  |            0.0017 |           0.3623 |
| nuscenes_mini | 688 |             0.0691 |                   -0.0096 |               0      |                 -0.4855 |                  0 |            0.0974 |                   0.0133 |              0      |                 0.3455 |            0      |           0.0813 |

### 3. Below-ground
| dataset       |    n |   mesh_median |   mesh_excess_median |   mesh_excess_p |   obb_median |   obb_excess_median |   obb_excess_p |   gt_median |
|:--------------|-----:|--------------:|---------------------:|----------------:|-------------:|--------------------:|---------------:|------------:|
| ecp           |  118 |       -0.0177 |               0.0644 |          0.0006 |       0.04   |              0.1184 |         0      |     -0.0517 |
| nuscenes_mini | 1275 |       -0.1428 |              -0.0767 |          0      |      -0.0624 |              0.0021 |         0.6541 |     -0.0683 |

### 5. Not implied by in-mask depth
| dataset       |   n |   not_implied_median |   frac_majority_not_implied |
|:--------------|----:|---------------------:|----------------------------:|
| ecp           | 112 |               0.4522 |                      0.4196 |
| nuscenes_mini | 687 |               0.438  |                      0.4105 |

### 2. Free-space vs GT floor (by category)
| dataset       | cls                  |   n |   mesh_frac_median |   mesh_excess_frac_median |   mesh_excess_frac_p |   mesh_excess_m3_median |   mesh_excess_m3_p |   obb_frac_median |   obb_excess_frac_median |   obb_excess_frac_p |   obb_excess_m3_median |   obb_excess_m3_p |   gt_frac_median |
|:--------------|:---------------------|----:|-------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|------------------:|-------------------------:|--------------------:|-----------------------:|------------------:|-----------------:|
| ecp           | bicycle              |  61 |             0.5663 |                   -0.1002 |               0.0001 |                 -0.677  |             0      |            0.5805 |                  -0.0857 |              0      |                 0.114  |            0.2    |           0.6765 |
| ecp           | car                  |  38 |             0.129  |                   -0.0744 |               0.0051 |                 -1.2785 |             0      |            0.2177 |                  -0.0133 |              0.4177 |                 0.396  |            0.1433 |           0.2499 |
| ecp           | motorcycle           |  13 |             0.3848 |                    0.0953 |               0.0225 |                 -0.148  |             0.0002 |            0.4306 |                   0.1135 |              0.0034 |                 0.527  |            0.0002 |           0.2895 |
| nuscenes_mini | bicycle              |  30 |             0.1381 |                    0.0067 |               0.8555 |                 -0.117  |             0      |            0.1258 |                   0.0147 |              0.3616 |                 0.034  |            0.0161 |           0.1309 |
| nuscenes_mini | bus                  |   6 |             0.0913 |                    0.0202 |               0.0312 |                 -2.7775 |             0.2188 |            0.0866 |                   0.0192 |              0.0312 |                 6.213  |            0.0312 |           0.0711 |
| nuscenes_mini | car                  | 584 |             0.0645 |                   -0.01   |               0      |                 -0.483  |             0      |            0.093  |                   0.0137 |              0      |                 0.3675 |            0      |           0.0763 |
| nuscenes_mini | construction vehicle |  18 |             0.1112 |                   -0.0001 |               1      |                 -1.8915 |             0      |            0.1116 |                   0.0052 |              0.2379 |                 6.404  |            0.0075 |           0.1174 |
| nuscenes_mini | motorcycle           |  27 |             0.1438 |                   -0.0353 |               0.1221 |                 -0.417  |             0      |            0.2083 |                  -0.0109 |              0.7011 |                -0.002  |            1      |           0.236  |
| nuscenes_mini | truck                |  23 |             0.0701 |                   -0.015  |               0.0347 |                 -1.42   |             0      |            0.0974 |                   0.0196 |              0.0931 |                 0.354  |            0.21   |           0.0848 |

## Threshold 1.0 m -- 2527 / 7445 detections matched
### 1. Mask fit
| dataset       |    n |   recall_median |   n_clean |   clean_frac |   iou_median |
|:--------------|-----:|----------------:|----------:|-------------:|-------------:|
| ecp           |  215 |          0.6677 |        85 |       0.3953 |       0.3925 |
| nuscenes_mini | 7230 |          0.8495 |      2538 |       0.351  |       0.5577 |

### 2. Free-space vs GT floor
| dataset       |    n |   mesh_frac_median |   mesh_excess_frac_median |   mesh_excess_frac_p |   mesh_excess_m3_median |   mesh_excess_m3_p |   obb_frac_median |   obb_excess_frac_median |   obb_excess_frac_p |   obb_excess_m3_median |   obb_excess_m3_p |   gt_frac_median |
|:--------------|-----:|-------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|------------------:|-------------------------:|--------------------:|-----------------------:|------------------:|-----------------:|
| ecp           |  146 |             0.379  |                   -0.0702 |               0.0006 |                 -0.6645 |                  0 |            0.3838 |                  -0.0356 |              0.0006 |                 0.2575 |            0.0003 |           0.359  |
| nuscenes_mini | 1171 |             0.0734 |                   -0.0044 |               0.0001 |                 -0.549  |                  0 |            0.0999 |                   0.0186 |              0      |                 0.271  |            0      |           0.0827 |

### 3. Below-ground
| dataset       |    n |   mesh_median |   mesh_excess_median |   mesh_excess_p |   obb_median |   obb_excess_median |   obb_excess_p |   gt_median |
|:--------------|-----:|--------------:|---------------------:|----------------:|-------------:|--------------------:|---------------:|------------:|
| ecp           |  161 |        0.02   |               0.0737 |               0 |       0.049  |              0.125  |         0      |     -0.0489 |
| nuscenes_mini | 2366 |       -0.1428 |              -0.0854 |               0 |      -0.0652 |             -0.0084 |         0.1814 |     -0.0642 |

### 5. Not implied by in-mask depth
| dataset       |    n |   not_implied_median |   frac_majority_not_implied |
|:--------------|-----:|---------------------:|----------------------------:|
| ecp           |  146 |               0.4871 |                      0.4932 |
| nuscenes_mini | 1170 |               0.4906 |                      0.4889 |

### 2. Free-space vs GT floor (by category)
| dataset       | cls                  |   n |   mesh_frac_median |   mesh_excess_frac_median |   mesh_excess_frac_p |   mesh_excess_m3_median |   mesh_excess_m3_p |   obb_frac_median |   obb_excess_frac_median |   obb_excess_frac_p |   obb_excess_m3_median |   obb_excess_m3_p |   gt_frac_median |
|:--------------|:---------------------|----:|-------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|------------------:|-------------------------:|--------------------:|-----------------------:|------------------:|-----------------:|
| ecp           | bicycle              |  78 |             0.5595 |                   -0.1014 |               0.0009 |                 -0.474  |             0      |            0.5843 |                  -0.074  |              0      |                 0.129  |            0.0169 |           0.6732 |
| ecp           | car                  |  50 |             0.129  |                   -0.0799 |               0.0003 |                 -1.4065 |             0      |            0.2177 |                  -0.0188 |              0.1189 |                 0.2595 |            0.4799 |           0.252  |
| ecp           | motorcycle           |  18 |             0.4096 |                    0.0804 |               0.0013 |                 -0.1225 |             0      |            0.4158 |                   0.0932 |              0.0013 |                 0.5685 |            0.0001 |           0.3135 |
| nuscenes_mini | bicycle              |  36 |             0.1381 |                    0.014  |               0.405  |                 -0.108  |             0      |            0.1258 |                   0.0159 |              0.1325 |                 0.029  |            0.0288 |           0.1107 |
| nuscenes_mini | bus                  |  15 |             0.0685 |                    0.0202 |               0.001  |                 -2.213  |             0.0352 |            0.0795 |                   0.0235 |              0.0001 |                 2.476  |            0.0001 |           0.0455 |
| nuscenes_mini | car                  | 999 |             0.0695 |                   -0.005  |               0      |                 -0.528  |             0      |            0.0978 |                   0.0191 |              0      |                 0.282  |            0      |           0.0791 |
| nuscenes_mini | construction vehicle |  43 |             0.0975 |                    0.0041 |               0.5424 |                 -2.386  |             0      |            0.1156 |                   0.0008 |              0.7608 |                 4.665  |            0      |           0.1117 |
| nuscenes_mini | motorcycle           |  36 |             0.149  |                   -0.0243 |               0.0652 |                 -0.35   |             0      |            0.192  |                  -0.0095 |              0.8679 |                 0.059  |            0.243  |           0.2156 |
| nuscenes_mini | truck                |  42 |             0.0717 |                   -0.0124 |               0.1641 |                 -1.415  |             0      |            0.0956 |                   0.0203 |              0.0003 |                 0.329  |            0.0884 |           0.0825 |

## Threshold 2.0 m -- 3560 / 7445 detections matched
### 1. Mask fit
| dataset       |    n |   recall_median |   n_clean |   clean_frac |   iou_median |
|:--------------|-----:|----------------:|----------:|-------------:|-------------:|
| ecp           |  215 |          0.6677 |        85 |       0.3953 |       0.3925 |
| nuscenes_mini | 7230 |          0.8495 |      2538 |       0.351  |       0.5577 |

### 2. Free-space vs GT floor
| dataset       |    n |   mesh_frac_median |   mesh_excess_frac_median |   mesh_excess_frac_p |   mesh_excess_m3_median |   mesh_excess_m3_p |   obb_frac_median |   obb_excess_frac_median |   obb_excess_frac_p |   obb_excess_m3_median |   obb_excess_m3_p |   gt_frac_median |
|:--------------|-----:|-------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|------------------:|-------------------------:|--------------------:|-----------------------:|------------------:|-----------------:|
| ecp           |  158 |             0.372  |                   -0.0641 |               0.0003 |                 -0.6625 |                  0 |            0.3826 |                  -0.0375 |              0.0003 |                 0.235  |            0.0018 |           0.359  |
| nuscenes_mini | 1432 |             0.0744 |                   -0.0004 |               0.6533 |                 -0.568  |                  0 |            0.0988 |                   0.0208 |              0      |                 0.2265 |            0      |           0.0803 |

### 3. Below-ground
| dataset       |    n |   mesh_median |   mesh_excess_median |   mesh_excess_p |   obb_median |   obb_excess_median |   obb_excess_p |   gt_median |
|:--------------|-----:|--------------:|---------------------:|----------------:|-------------:|--------------------:|---------------:|------------:|
| ecp           |  174 |        0.0207 |               0.0743 |               0 |       0.0516 |              0.1262 |         0      |     -0.0488 |
| nuscenes_mini | 3386 |       -0.1425 |              -0.0878 |               0 |      -0.0697 |             -0.0168 |         0.0057 |     -0.0594 |

### 5. Not implied by in-mask depth
| dataset       |    n |   not_implied_median |   frac_majority_not_implied |
|:--------------|-----:|---------------------:|----------------------------:|
| ecp           |  158 |               0.5076 |                      0.519  |
| nuscenes_mini | 1430 |               0.507  |                      0.5091 |

### 2. Free-space vs GT floor (by category)
| dataset       | cls                  |    n |   mesh_frac_median |   mesh_excess_frac_median |   mesh_excess_frac_p |   mesh_excess_m3_median |   mesh_excess_m3_p |   obb_frac_median |   obb_excess_frac_median |   obb_excess_frac_p |   obb_excess_m3_median |   obb_excess_m3_p |   gt_frac_median |
|:--------------|:---------------------|-----:|-------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|------------------:|-------------------------:|--------------------:|-----------------------:|------------------:|-----------------:|
| ecp           | bicycle              |   84 |             0.5465 |                   -0.1014 |               0.0006 |                 -0.4655 |             0      |            0.5837 |                  -0.0784 |              0      |                 0.129  |            0.0116 |           0.6696 |
| ecp           | car                  |   54 |             0.1361 |                   -0.0744 |               0.0002 |                 -1.487  |             0      |            0.2177 |                  -0.0193 |              0.0759 |                 0.171  |            0.8919 |           0.2551 |
| ecp           | motorcycle           |   20 |             0.4096 |                    0.0804 |               0.0026 |                 -0.1285 |             0      |            0.4158 |                   0.0932 |              0.0026 |                 0.4485 |            0.0026 |           0.2932 |
| nuscenes_mini | bicycle              |   39 |             0.1429 |                    0.0283 |               0.1996 |                 -0.112  |             0      |            0.133  |                   0.0178 |              0.0533 |                 0.027  |            0.0533 |           0.106  |
| nuscenes_mini | bus                  |   24 |             0.0872 |                    0.021  |               0.0015 |                 -1.213  |             0.0639 |            0.083  |                   0.023  |              0      |                 3.108  |            0.0003 |           0.0554 |
| nuscenes_mini | car                  | 1182 |             0.07   |                   -0.0006 |               0.399  |                 -0.517  |             0      |            0.0953 |                   0.0211 |              0      |                 0.213  |            0      |           0.0762 |
| nuscenes_mini | construction vehicle |   92 |             0.0951 |                    0.004  |               0.3481 |                 -1.266  |             0      |            0.1097 |                   0.0165 |              0.0001 |                 5.7005 |            0      |           0.0898 |
| nuscenes_mini | motorcycle           |   37 |             0.1451 |                   -0.0227 |               0.047  |                 -0.346  |             0      |            0.1883 |                  -0.0081 |              1      |                 0.051  |            0.1877 |           0.2127 |
| nuscenes_mini | truck                |   58 |             0.0717 |                   -0.0079 |               0.237  |                 -1.4295 |             0      |            0.0956 |                   0.0203 |              0      |                 0.207  |            0.3581 |           0.0796 |

## Threshold 4.0 m -- 4247 / 7445 detections matched
### 1. Mask fit
| dataset       |    n |   recall_median |   n_clean |   clean_frac |   iou_median |
|:--------------|-----:|----------------:|----------:|-------------:|-------------:|
| ecp           |  215 |          0.6677 |        85 |       0.3953 |       0.3925 |
| nuscenes_mini | 7230 |          0.8495 |      2538 |       0.351  |       0.5577 |

### 2. Free-space vs GT floor
| dataset       |    n |   mesh_frac_median |   mesh_excess_frac_median |   mesh_excess_frac_p |   mesh_excess_m3_median |   mesh_excess_m3_p |   obb_frac_median |   obb_excess_frac_median |   obb_excess_frac_p |   obb_excess_m3_median |   obb_excess_m3_p |   gt_frac_median |
|:--------------|-----:|-------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|------------------:|-------------------------:|--------------------:|-----------------------:|------------------:|-----------------:|
| ecp           |  165 |             0.3613 |                   -0.059  |               0.0006 |                 -0.578  |                  0 |            0.3747 |                  -0.0344 |               0.001 |                 0.246  |             0.001 |           0.3523 |
| nuscenes_mini | 1568 |             0.0743 |                   -0.0008 |               0.4046 |                 -0.5895 |                  0 |            0.0966 |                   0.0204 |               0     |                 0.2335 |             0     |           0.0823 |

### 3. Below-ground
| dataset       |    n |   mesh_median |   mesh_excess_median |   mesh_excess_p |   obb_median |   obb_excess_median |   obb_excess_p |   gt_median |
|:--------------|-----:|--------------:|---------------------:|----------------:|-------------:|--------------------:|---------------:|------------:|
| ecp           |  182 |        0.0216 |               0.0749 |               0 |       0.0532 |              0.1314 |         0      |     -0.0506 |
| nuscenes_mini | 4065 |       -0.1393 |              -0.0843 |               0 |      -0.0681 |             -0.0111 |         0.0642 |     -0.0583 |

### 5. Not implied by in-mask depth
| dataset       |    n |   not_implied_median |   frac_majority_not_implied |
|:--------------|-----:|---------------------:|----------------------------:|
| ecp           |  165 |               0.509  |                      0.5212 |
| nuscenes_mini | 1563 |               0.5112 |                      0.5163 |

### 2. Free-space vs GT floor (by category)
| dataset       | cls                  |    n |   mesh_frac_median |   mesh_excess_frac_median |   mesh_excess_frac_p |   mesh_excess_m3_median |   mesh_excess_m3_p |   obb_frac_median |   obb_excess_frac_median |   obb_excess_frac_p |   obb_excess_m3_median |   obb_excess_m3_p |   gt_frac_median |
|:--------------|:---------------------|-----:|-------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|------------------:|-------------------------:|--------------------:|-----------------------:|------------------:|-----------------:|
| ecp           | bicycle              |   90 |             0.5319 |                   -0.0849 |               0.0021 |                 -0.445  |             0      |            0.5821 |                  -0.0707 |              0      |                 0.154  |            0.008  |           0.6569 |
| ecp           | car                  |   55 |             0.1337 |                   -0.0773 |               0.0001 |                 -1.435  |             0      |            0.2099 |                  -0.0195 |              0.0581 |                 0.244  |            0.7877 |           0.2573 |
| ecp           | motorcycle           |   20 |             0.4096 |                    0.0804 |               0.0026 |                 -0.1285 |             0      |            0.4158 |                   0.0932 |              0.0026 |                 0.4485 |            0.0026 |           0.2932 |
| nuscenes_mini | bicycle              |   41 |             0.1429 |                    0.0283 |               0.211  |                 -0.112  |             0      |            0.1265 |                   0.0173 |              0.1173 |                 0.023  |            0.1173 |           0.1019 |
| nuscenes_mini | bus                  |   34 |             0.0824 |                    0.0227 |               0.0029 |                 -1.42   |             0.0243 |            0.0864 |                   0.0247 |              0      |                 3.243  |            0.0008 |           0.0631 |
| nuscenes_mini | car                  | 1244 |             0.0718 |                   -0.0005 |               0.5143 |                 -0.5205 |             0      |            0.0944 |                   0.0211 |              0      |                 0.205  |            0      |           0.0764 |
| nuscenes_mini | construction vehicle |   98 |             0.0951 |                    0.0055 |               0.1888 |                 -1.266  |             0      |            0.1114 |                   0.0187 |              0      |                 5.4815 |            0      |           0.0898 |
| nuscenes_mini | motorcycle           |   43 |             0.1543 |                   -0.0098 |               0.3604 |                 -0.354  |             0      |            0.2211 |                   0.009  |              0.5424 |                 0.042  |            0.5424 |           0.2185 |
| nuscenes_mini | truck                |  108 |             0.0592 |                   -0.027  |               0      |                 -3.814  |             0      |            0.0731 |                   0.0039 |              0.773  |                 1.0115 |            0.0003 |           0.1358 |
