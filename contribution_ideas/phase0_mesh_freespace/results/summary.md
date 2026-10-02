# Mesh / free-space / ground evaluation -- summary

n = 7445 objects total, 3787 GT-matched, 1858 GT-matched AND observed (surf_unknown < 0.95). `**` marks p < 0.05 (sign test). Categories need at least 20 matched objects to get their own row.

## 1. Mask fit
### per dataset
| dataset       |    n |   recall_median |   n_clean |   clean_frac |   iou_median |
|:--------------|-----:|----------------:|----------:|-------------:|-------------:|
| ecp           |  215 |          0.6677 |        85 |       0.3953 |       0.3925 |
| nuscenes_mini | 7230 |          0.8495 |      2538 |       0.351  |       0.5577 |

### per dataset x category
| dataset       | cls                  |    n |   recall_median |   n_clean |   clean_frac |   iou_median |
|:--------------|:---------------------|-----:|----------------:|----------:|-------------:|-------------:|
| ecp           | bicycle              |  103 |          0.5647 |        68 |       0.6602 |       0.3699 |
| ecp           | car                  |   84 |          0.8926 |         8 |       0.0952 |       0.7116 |
| ecp           | motorcycle           |   26 |          0.7368 |         9 |       0.3462 |       0.564  |
| nuscenes_mini | bicycle              |   95 |          0.4548 |        67 |       0.7053 |       0.3645 |
| nuscenes_mini | bus                  |  325 |          0.8796 |        73 |       0.2246 |       0.5539 |
| nuscenes_mini | car                  | 6061 |          0.8442 |      2188 |       0.361  |       0.5654 |
| nuscenes_mini | construction vehicle |  171 |          0.819  |        36 |       0.2105 |       0.5713 |
| nuscenes_mini | motorcycle           |   67 |          0.7329 |        61 |       0.9104 |       0.4651 |
| nuscenes_mini | trailer              |   77 |          0.9625 |        11 |       0.1429 |       0.772  |
| nuscenes_mini | truck                |  434 |          0.9332 |       102 |       0.235  |       0.6238 |

## 2. Free-space touch vs the GT-box floor
### per dataset
| dataset       |    n |   mesh_frac_median |   mesh_excess_frac_median |   mesh_excess_frac_p |   mesh_excess_m3_median |   mesh_excess_m3_p |   obb_frac_median |   obb_excess_frac_median |   obb_excess_frac_p |   obb_excess_m3_median |   obb_excess_m3_p |   gt_frac_median |   gt_m3_median |
|:--------------|-----:|-------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|------------------:|-------------------------:|--------------------:|-----------------------:|------------------:|-----------------:|---------------:|
| ecp           |  129 |             0.3516 |                   -0.0571 |                0.008 |                  -0.815 |                  0 |            0.3624 |                  -0.033  |              0.0026 |                  0.129 |            0.0523 |           0.3375 |          1.269 |
| nuscenes_mini | 1729 |             0.0776 |                    0.0016 |                0.149 |                  -0.628 |                  0 |            0.0996 |                   0.0233 |              0      |                  0.275 |            0      |           0.0812 |          1.382 |

### per dataset x category
| dataset       | cls                  |    n |   mesh_frac_median |   mesh_excess_frac_median |   mesh_excess_frac_p |   mesh_excess_m3_median |   mesh_excess_m3_p |   obb_frac_median |   obb_excess_frac_median |   obb_excess_frac_p |   obb_excess_m3_median |   obb_excess_m3_p |   gt_frac_median |   gt_m3_median |
|:--------------|:---------------------|-----:|-------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|------------------:|-------------------------:|--------------------:|-----------------------:|------------------:|-----------------:|---------------:|
| ecp           | bicycle              |   60 |             0.5884 |                   -0.0951 |               0.0273 |                 -0.586  |             0      |            0.5815 |                  -0.0856 |              0      |                 0.0735 |            0.3663 |           0.6605 |         0.768  |
| ecp           | car                  |   51 |             0.1386 |                   -0.0773 |               0.0002 |                 -1.539  |             0      |            0.2256 |                  -0.0195 |              0.0489 |                -0.06   |            1      |           0.2573 |         2.482  |
| ecp           | motorcycle           |   18 |             0.4497 |                    0.1031 |               0.0013 |                 -0.1495 |             0      |            0.4427 |                   0.0994 |              0.0013 |                 0.4485 |            0.0001 |           0.3366 |         0.481  |
| nuscenes_mini | bicycle              |   20 |             0.1509 |                    0.0497 |               0.1153 |                 -0.1025 |             0      |            0.1128 |                   0.0134 |              0.8238 |                 0.033  |            0.1153 |           0.1004 |         0.1415 |
| nuscenes_mini | bus                  |   40 |             0.0733 |                    0.021  |               0.0007 |                 -1.6635 |             0.0064 |            0.0864 |                   0.0227 |              0      |                 2.666  |            0.0022 |           0.0653 |         6.663  |
| nuscenes_mini | car                  | 1273 |             0.0737 |                    0.0002 |               0.7793 |                 -0.514  |             0      |            0.0976 |                   0.0227 |              0      |                 0.203  |            0      |           0.0764 |         1.172  |
| nuscenes_mini | construction vehicle |  113 |             0.0958 |                    0.0088 |               0.0235 |                 -1.234  |             0      |            0.1103 |                   0.0229 |              0      |                 4.491  |            0      |           0.0885 |         2.394  |
| nuscenes_mini | motorcycle           |   38 |             0.1638 |                   -0.0212 |               0.2559 |                 -0.3885 |             0      |            0.2147 |                  -0.0017 |              1      |                 0.007  |            0.8714 |           0.2229 |         0.5445 |
| nuscenes_mini | trailer              |   63 |             0.0918 |                    0.0371 |               0.0003 |                 -2.357  |             0      |            0.109  |                   0.0446 |              0      |                 1.751  |            0.0111 |           0.0717 |         4.644  |
| nuscenes_mini | truck                |  182 |             0.0756 |                   -0.0055 |               0.0636 |                 -1.415  |             0      |            0.0938 |                   0.0241 |              0      |                 1.6015 |            0      |           0.097  |         2.374  |

## 3. Below-ground reach (PseudoLabeler)
### per dataset
| dataset       |    n |   mesh_median |   mesh_frac_below_1cm |   mesh_excess_median |   mesh_excess_p |   obb_median |   obb_frac_below_1cm |   obb_excess_median |   obb_excess_p |   gt_median |
|:--------------|-----:|--------------:|----------------------:|---------------------:|----------------:|-------------:|---------------------:|--------------------:|---------------:|------------:|
| ecp           |  148 |        0.0308 |                0.5608 |               0.0862 |               0 |       0.0634 |               0.6486 |              0.1547 |         0      |     -0.0476 |
| nuscenes_mini | 3639 |       -0.1295 |                0.3053 |              -0.0734 |               0 |      -0.0564 |               0.3996 |              0.0013 |         0.8423 |     -0.0556 |

### per dataset x category
| dataset       | cls                  |    n |   mesh_median |   mesh_frac_below_1cm |   mesh_excess_median |   mesh_excess_p |   obb_median |   obb_frac_below_1cm |   obb_excess_median |   obb_excess_p |   gt_median |
|:--------------|:---------------------|-----:|--------------:|----------------------:|---------------------:|----------------:|-------------:|---------------------:|--------------------:|---------------:|------------:|
| ecp           | bicycle              |   62 |       -0.01   |                0.5    |               0.0575 |          0.0032 |       0.04   |               0.5645 |              0.1176 |         0      |     -0.0471 |
| ecp           | car                  |   68 |        0.1097 |                0.7206 |               0.1337 |          0      |       0.1794 |               0.8382 |              0.2061 |         0      |     -0.0279 |
| ecp           | motorcycle           |   18 |       -0.1994 |                0.1667 |              -0.0665 |          0.8145 |      -0.1772 |               0.2222 |             -0.0033 |         1      |     -0.1312 |
| nuscenes_mini | bicycle              |   25 |       -0.1776 |                0.12   |               0.0554 |          0.2295 |      -0.0645 |               0.24   |              0.1007 |         0.0433 |     -0.1752 |
| nuscenes_mini | bus                  |  236 |        0.0781 |                0.5466 |               0.2402 |          0.005  |       0.2392 |               0.6017 |              0.3725 |         0      |     -0.0898 |
| nuscenes_mini | car                  | 2736 |       -0.1449 |                0.2617 |              -0.0877 |          0      |      -0.0732 |               0.3564 |             -0.0183 |         0.0005 |     -0.0539 |
| nuscenes_mini | construction vehicle |  155 |       -0.0623 |                0.4387 |              -0.0422 |          0.0769 |       0.0268 |               0.5226 |              0.023  |         0.5206 |      0.0088 |
| nuscenes_mini | motorcycle           |   48 |       -0.1393 |                0.1875 |               0.014  |          0.8854 |      -0.0305 |               0.2917 |              0.0932 |         0.0021 |     -0.1399 |
| nuscenes_mini | trailer              |   74 |       -0.0526 |                0.3378 |              -0.122  |          0      |       0.0574 |               0.6351 |              0.0228 |         0.2954 |      0.0118 |
| nuscenes_mini | truck                |  365 |       -0.0726 |                0.4411 |              -0.0012 |          1      |       0.0206 |               0.5178 |              0.0691 |         0.0008 |     -0.0608 |

## 4. Shape-isolated free-space (GT-floor control)
### per dataset
| dataset       |    n |   surf_free_median |   floor_free_median |   gt_free_median |   placement_excess_median |   placement_p |   shape_excess_median |   shape_p |
|:--------------|-----:|-------------------:|--------------------:|-----------------:|--------------------------:|--------------:|----------------------:|----------:|
| ecp           |   51 |             0.2092 |              0.2588 |           0.2573 |                   -0.0337 |        0.0489 |                0.0414 |    0.0489 |
| nuscenes_mini | 1671 |             0.0985 |              0.0642 |           0.0797 |                    0.0392 |        0      |               -0.0094 |    0      |

### per dataset x category
| dataset       | cls                  |    n |   surf_free_median |   floor_free_median |   gt_free_median |   placement_excess_median |   placement_p |   shape_excess_median |   shape_p |
|:--------------|:---------------------|-----:|-------------------:|--------------------:|-----------------:|--------------------------:|--------------:|----------------------:|----------:|
| ecp           | car                  |   51 |             0.2092 |              0.2588 |           0.2573 |                   -0.0337 |        0.0489 |                0.0414 |    0.0489 |
| nuscenes_mini | bus                  |   40 |             0.0914 |              0.0579 |           0.0653 |                    0.0351 |        0.0064 |               -0.0008 |    0.8746 |
| nuscenes_mini | car                  | 1273 |             0.099  |              0.0672 |           0.0764 |                    0.0321 |        0      |               -0.0049 |    0      |
| nuscenes_mini | construction vehicle |  113 |             0.0979 |              0.0316 |           0.0885 |                    0.0641 |        0      |               -0.0706 |    0      |
| nuscenes_mini | trailer              |   63 |             0.1093 |              0.0316 |           0.0717 |                    0.0802 |        0      |               -0.0418 |    0      |
| nuscenes_mini | truck                |  182 |             0.0887 |              0.0797 |           0.097  |                    0.0498 |        0      |               -0.011  |    0      |

## 5. Not implied by in-mask depth
### per dataset
| dataset       |    n |   not_implied_median |   frac_majority_not_implied |
|:--------------|-----:|---------------------:|----------------------------:|
| ecp           |  129 |               0.4891 |                      0.4961 |
| nuscenes_mini | 1727 |               0.4977 |                      0.4974 |

### per dataset x category
| dataset       | cls                  |    n |   not_implied_median |   frac_majority_not_implied |
|:--------------|:---------------------|-----:|---------------------:|----------------------------:|
| ecp           | bicycle              |   60 |               0.4671 |                      0.45   |
| ecp           | car                  |   51 |               0.5703 |                      0.6275 |
| ecp           | motorcycle           |   18 |               0.2932 |                      0.2778 |
| nuscenes_mini | bicycle              |   20 |               0.4699 |                      0.4    |
| nuscenes_mini | bus                  |   40 |               0.5539 |                      0.625  |
| nuscenes_mini | car                  | 1271 |               0.526  |                      0.5382 |
| nuscenes_mini | construction vehicle |  113 |               0.4169 |                      0.3894 |
| nuscenes_mini | motorcycle           |   38 |               0.5975 |                      0.6842 |
| nuscenes_mini | trailer              |   63 |               0.4614 |                      0.3651 |
| nuscenes_mini | truck                |  182 |               0.3799 |                      0.2692 |
