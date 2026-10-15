# Bumper C7 full support — bounded static training comparison

A bounded initial comparison of six configurations from five model families on a fixed full-support physical-field task; no universal ranking, convergence or design-use accuracy claim.

Suite status: **completed**. Model evidence: 6 complete, 0 failed, 0 missing.

Each configuration had a 240 s synchronized optimizer-compute budget and a 5,000-update maximum. Minimum 100 updates and two complete epochs are report-only quality gates. A final update may cross the time cap; its actual duration is retained.

| Configuration | Evidence | Batch | Updates | Samples | Complete epochs | Selected epoch | Compute at selection (s) | Actual compute / cap (s) | Overshoot (s) | Quality gate | Best validation normalized MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Transolver | complete | 5 | 4,120 | 20,063 | 179 | 149 | 201.456 | 240.037 / 240 | 0.036535 | yes | 0.115927 |
| FLARE | complete | 5 | 1,488 | 7,248 | 64 | 58 | 215.169 | 240.002 / 240 | 0.0023755 | yes | 0.18577 |
| GeoTransolver GALE | complete | 5 | 3,579 | 17,430 | 155 | 153 | 236.16 | 240.009 / 240 | 0.0085684 | yes | 0.108496 |
| GeoTransolver GALE_FA | complete | 5 | 1,237 | 6,026 | 53 | 48 | 214.347 | 240.108 / 240 | 0.107589 | yes | 0.171868 |
| MeshGraphNet | complete | 2 | 3,387 | 6,774 | 60 | 48 | 190.524 | 240.058 / 240 | 0.0582404 | yes | 0.156061 |
| FIGConvUNet | complete | 5 | 680 | 3,313 | 29 | 29 | 235.53 | 240.261 / 240 | 0.260716 | yes | 0.126917 |

A normalized validation loss is used only for checkpoint selection within its own domain. It is not a physical accuracy metric and is not comparable between the two domains.

| Configuration | Selected updates | Selected case observations | Selected epoch | Selected compute (s) | Last complete training normalized MSE | Last eligible validation improved |
|---|---:|---:|---:|---:|---:|---|
| Transolver | 3,427 | 16,688 | 149 | 201.456 | 0.124758 | no |
| FLARE | 1,334 | 6,496 | 58 | 215.169 | 0.221908 | no |
| GeoTransolver GALE | 3,519 | 17,136 | 153 | 236.16 | 0.104523 | no |
| GeoTransolver GALE_FA | 1,104 | 5,376 | 48 | 214.347 | 0.209204 | no |
| MeshGraphNet | 2,688 | 5,376 | 48 | 190.524 | 0.149748 | no |
| FIGConvUNet | 667 | 3,248 | 29 | 235.53 | 0.141118 | yes |

| Configuration | Parameters | Training CUDA allocated peak (MiB) | CUDA reserved peak (MiB) | Observed RSS peak lower bound (MiB) | Minimum observed available host RAM (MiB) | Observed solver PIDs |
|---|---:|---:|---:|---:|---:|---|
| Transolver | 43,246 | 575.424 | 694 | 1365.64 | 603.668 | — |
| FLARE | 38,854 | 561.694 | 698 | 1252.14 | 48.3516 | — |
| GeoTransolver GALE | 45,456 | 625.28 | 784 | 1393.08 | 213.223 | — |
| GeoTransolver GALE_FA | 41,064 | 714.491 | 902 | 1391.38 | 490.164 | — |
| MeshGraphNet | 82,006 | 618.936 | 734 | 1545.41 | 1499.51 | — |
| FIGConvUNet | 587,782 | 1516.83 | 1894 | 1718.62 | 275.504 | — |

| Configuration | Test CUDA allocated peak (MiB) | Test CUDA reserved peak (MiB) | Test observed RSS peak lower bound (MiB) |
|---|---:|---:|---:|
| Transolver | 29.1084 | 44 | 1242.42 |
| FLARE | 28.0649 | 44 | 1154.02 |
| GeoTransolver GALE | 34.1572 | 44 | 1273.03 |
| GeoTransolver GALE_FA | 33.3105 | 44 | 1270.81 |
| MeshGraphNet | 87.9531 | 96 | 1286 |
| FIGConvUNet | 77.0024 | 104 | 1570.99 |

CUDA allocated and reserved peaks are PyTorch allocator counters, not sampled physical device VRAM. NVML/device-level VRAM sampling is **unmeasured** in these runs. Host RSS is a sampled peak lower bound; available host RAM is a separately sampled system-wide quantity. Its minimum is the lowest observed value, not a continuous minimum. Test-process available-RAM minimum was not separately measured.

| Configuration | Optimizer compute (s) | Training wall (s) | Setup including import (s) | Monitoring (s) | Test process wall (s) |
|---|---:|---:|---:|---:|---:|
| Transolver | 240.037 | 901.421 | 36.3101 | 298.262 | 29.4546 |
| FLARE | 240.002 | 537.195 | 24.9591 | 142.124 | 30.4314 |
| GeoTransolver GALE | 240.009 | 787.704 | 25.5786 | 249.902 | 30.0417 |
| GeoTransolver GALE_FA | 240.108 | 471.925 | 23.1855 | 110.817 | 29.809 |
| MeshGraphNet | 240.058 | 495.465 | 29.6985 | 119.22 | 30.6437 |
| FIGConvUNet | 240.261 | 397.828 | 23.7106 | 64.957 | 31.377 |

| Configuration | Training read / target checks (s) | Normalization / CPU stacking (s) | Host-to-device (s) | CPU snapshots / checkpoint I/O (s) | Validation (s) |
|---|---:|---:|---:|---:|---:|
| Transolver | 130.309 | 146.303 | 22.1288 | 0.586013 | 6.83864 |
| FLARE | 50.1342 | 57.266 | 8.00103 | 0.30245 | 6.16641 |
| GeoTransolver GALE | 109.574 | 120.322 | 17.7455 | 0.840065 | 7.01842 |
| GeoTransolver GALE_FA | 37.4703 | 41.6133 | 6.07207 | 0.382672 | 6.56951 |
| MeshGraphNet | 38.3062 | 44.6021 | 7.81453 | 0.393183 | 6.45526 |
| FIGConvUNet | 20.7261 | 23.2948 | 3.31342 | 1.21207 | 16.9254 |

These observed timing categories are not an exhaustive disjoint partition: setup includes initial monitoring, while end verification and other work are not all separately timed. The 240 s optimizer cap is not a wall-time cap. Monitoring and data preparation overhead are reported rather than attributed to model compute. Memory samples are lower bounds; concurrent solver intervals remain in SUMMARY.json.

| Configuration | Test state | Forward total (s) | Displacement relative L2 | Vector RMSE (mm) | Vector error P95 (mm) | Vector error max (mm) | Centroid-removed relative L2 | Fresh validation max difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Transolver | completed | 0.354009 | 6.321% | 9.23126 | 18.6756 | 64.4009 | 10.99% | 0 |
| FLARE | completed | 0.305758 | 7.17% | 10.4711 | 21.547 | 72.4813 | 11.89% | 0 |
| GeoTransolver GALE | completed | 0.409585 | 6.003% | 8.76741 | 17.2305 | 74.3556 | 10.84% | 0 |
| GeoTransolver GALE_FA | completed | 0.454299 | 6.987% | 10.2042 | 20.81 | 70.6036 | 11.57% | 0 |
| MeshGraphNet | completed | 0.347729 | 6.39% | 9.33293 | 18.3363 | 49.8202 | 11.45% | 1.90735e-06 |
| FIGConvUNet | completed | 1.06978 | 8.337% | 12.1765 | 23.1931 | 118.743 | 14.82% | 0 |

The forward time contains one synchronized batch-size-one forward per held-out case. It excludes input preparation, transfer, saved-output I/O and metric/figure generation; it is a shared-workstation observation, not an isolated system benchmark.

| Configuration | Test setup (s) | Input read / checks (s) | Normalize / stack (s) | Host-to-device (s) | Forward (s) | Output transfer / unscale / save / truth read (s) | Metrics / figures (s) | Test wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transolver | 19.3189 | 0.0123198 | 0.0039951 | 0.0047715 | 0.354009 | 0.186392 | 7.69671 | 29.4546 |
| FLARE | 20.2032 | 0.0112165 | 0.0039395 | 0.0044448 | 0.305758 | 0.193291 | 7.78211 | 30.4314 |
| GeoTransolver GALE | 19.5487 | 0.0127097 | 0.0044071 | 0.0046623 | 0.409585 | 0.240165 | 7.95934 | 30.0417 |
| GeoTransolver GALE_FA | 20.0169 | 0.0093518 | 0.00309 | 0.0042643 | 0.454299 | 0.184728 | 7.29297 | 29.809 |
| MeshGraphNet | 21.4504 | 0.0092658 | 0.0035114 | 0.0043205 | 0.347729 | 0.181006 | 6.82285 | 30.6437 |
| FIGConvUNet | 20.4367 | 0.0103344 | 0.0035044 | 0.0044715 | 1.06978 | 0.201275 | 7.71497 | 31.377 |

These are sums of the original one-pass records. Standalone held-out truth-read time is **unmeasured**: it shares a timer with output transfer, unscaling and saved-array I/O. Setup includes framework import and frozen-source/cache checks. Other end verification and report hashing remain in total wall time; the categories are not an exhaustive disjoint partition. Per-case timings are preserved in SUMMARY.json.

| Configuration | Field | Unit | Relative L2 | RMSE | Absolute error P95 | Absolute error max | Raw prediction negatives | Raw minimum | Truth exact nonzero / count |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Transolver | displacement_x | mm | 4.327% | 6.24312 | 13.0328 | 63.9301 | — | -323.911 | 957,250 / 957,250 |
| Transolver | displacement_y | mm | 27.22% | 6.10375 | 14.2275 | 42.6874 | — | -71.991 | 949,690 / 957,250 |
| Transolver | displacement_z | mm | 90.08% | 2.99729 | 6.0465 | 33.0892 | — | -1.62637 | 949,690 / 957,250 |
| Transolver | inplane_vm_upper | GPa | 26.08% | 0.0750727 | 0.154383 | 0.468666 | 4,239 | -0.0528848 | 957,250 / 957,250 |
| Transolver | inplane_vm_lower | GPa | 25.99% | 0.0754425 | 0.155505 | 0.460967 | 5,942 | -0.0618487 | 957,250 / 957,250 |
| Transolver | EPSP_upper | dimensionless | 42.95% | 0.00982976 | 0.0160344 | 0.388137 | 201,590 | -0.0116303 | 361,429 / 957,250 |
| Transolver | EPSP_lower | dimensionless | 43.25% | 0.00987118 | 0.0158136 | 0.31572 | 205,166 | -0.0112891 | 371,515 / 957,250 |
| FLARE | displacement_x | mm | 4.906% | 7.07847 | 14.8068 | 70.3339 | — | -303.663 | 957,250 / 957,250 |
| FLARE | displacement_y | mm | 31.49% | 7.06263 | 16.2202 | 39.8786 | — | -48.8049 | 949,690 / 957,250 |
| FLARE | displacement_z | mm | 93.4% | 3.10766 | 6.4475 | 32.6593 | — | -1.85155 | 949,690 / 957,250 |
| FLARE | inplane_vm_upper | GPa | 29.27% | 0.0842738 | 0.170988 | 0.40823 | 658 | -0.0196634 | 957,250 / 957,250 |
| FLARE | inplane_vm_lower | GPa | 28.82% | 0.0836515 | 0.169693 | 0.430911 | 1,124 | -0.0310127 | 957,250 / 957,250 |
| FLARE | EPSP_upper | dimensionless | 61.82% | 0.0141467 | 0.0189947 | 0.552182 | 479,915 | -0.0110313 | 361,429 / 957,250 |
| FLARE | EPSP_lower | dimensionless | 62.83% | 0.0143398 | 0.018749 | 0.385891 | 488,649 | -0.0117948 | 371,515 / 957,250 |
| GeoTransolver GALE | displacement_x | mm | 4.305% | 6.21081 | 12.5135 | 71.916 | — | -311.17 | 957,250 / 957,250 |
| GeoTransolver GALE | displacement_y | mm | 23.33% | 5.23269 | 12.1749 | 48.0263 | — | -59.5257 | 949,690 / 957,250 |
| GeoTransolver GALE | displacement_z | mm | 99.28% | 3.30337 | 6.98547 | 34.0376 | — | -5.97583 | 949,690 / 957,250 |
| GeoTransolver GALE | inplane_vm_upper | GPa | 24.58% | 0.0707767 | 0.145003 | 0.442057 | 1,889 | -0.031263 | 957,250 / 957,250 |
| GeoTransolver GALE | inplane_vm_lower | GPa | 24.46% | 0.0710058 | 0.145506 | 0.426051 | 3,330 | -0.0500956 | 957,250 / 957,250 |
| GeoTransolver GALE | EPSP_upper | dimensionless | 44.52% | 0.0101871 | 0.0145095 | 0.466138 | 274,291 | -0.0111009 | 361,429 / 957,250 |
| GeoTransolver GALE | EPSP_lower | dimensionless | 44.89% | 0.0102462 | 0.0142328 | 0.32259 | 250,132 | -0.0125147 | 371,515 / 957,250 |
| GeoTransolver GALE_FA | displacement_x | mm | 4.438% | 6.40294 | 13.1166 | 68.3893 | — | -308.52 | 957,250 / 957,250 |
| GeoTransolver GALE_FA | displacement_y | mm | 32.13% | 7.20497 | 16.183 | 42.5741 | — | -53.3272 | 949,690 / 957,250 |
| GeoTransolver GALE_FA | displacement_z | mm | 100.7% | 3.34903 | 6.5145 | 36.1145 | — | -10.9569 | 949,690 / 957,250 |
| GeoTransolver GALE_FA | inplane_vm_upper | GPa | 28.24% | 0.0812956 | 0.165726 | 0.430272 | 140 | -0.0120607 | 957,250 / 957,250 |
| GeoTransolver GALE_FA | inplane_vm_lower | GPa | 27.74% | 0.0805346 | 0.163195 | 0.420846 | 197 | -0.0154465 | 957,250 / 957,250 |
| GeoTransolver GALE_FA | EPSP_upper | dimensionless | 58.08% | 0.0132912 | 0.0192633 | 0.523933 | 328,149 | -0.0118456 | 361,429 / 957,250 |
| GeoTransolver GALE_FA | EPSP_lower | dimensionless | 59.22% | 0.0135174 | 0.0186342 | 0.362972 | 334,074 | -0.0117575 | 371,515 / 957,250 |
| MeshGraphNet | displacement_x | mm | 4.844% | 6.98918 | 14.959 | 47.6756 | — | -310.628 | 957,250 / 957,250 |
| MeshGraphNet | displacement_y | mm | 23.85% | 5.3492 | 11.284 | 33.8835 | — | -51.1112 | 949,690 / 957,250 |
| MeshGraphNet | displacement_z | mm | 93.32% | 3.10499 | 6.30118 | 32.3932 | — | -5.48879 | 949,690 / 957,250 |
| MeshGraphNet | inplane_vm_upper | GPa | 26.62% | 0.0766337 | 0.159747 | 0.457032 | 0 | 0.00200863 | 957,250 / 957,250 |
| MeshGraphNet | inplane_vm_lower | GPa | 26.31% | 0.0763774 | 0.157637 | 0.479882 | 46 | -0.00842375 | 957,250 / 957,250 |
| MeshGraphNet | EPSP_upper | dimensionless | 56.56% | 0.0129428 | 0.0194542 | 0.497936 | 429,689 | -0.00480746 | 361,429 / 957,250 |
| MeshGraphNet | EPSP_lower | dimensionless | 58.34% | 0.0133168 | 0.0193742 | 0.331838 | 407,520 | -0.00503366 | 371,515 / 957,250 |
| FIGConvUNet | displacement_x | mm | 6.508% | 9.38892 | 19.1947 | 117.446 | — | -297.706 | 957,250 / 957,250 |
| FIGConvUNet | displacement_y | mm | 31.23% | 7.00315 | 15.0361 | 46.8867 | — | -62.8629 | 949,690 / 957,250 |
| FIGConvUNet | displacement_z | mm | 100% | 3.32747 | 6.6058 | 33.4183 | — | -12.0843 | 949,690 / 957,250 |
| FIGConvUNet | inplane_vm_upper | GPa | 25.77% | 0.0741857 | 0.151935 | 0.426986 | 6 | -0.00488558 | 957,250 / 957,250 |
| FIGConvUNet | inplane_vm_lower | GPa | 26.05% | 0.0756051 | 0.15433 | 0.436031 | 25 | -0.00727086 | 957,250 / 957,250 |
| FIGConvUNet | EPSP_upper | dimensionless | 47.86% | 0.0109523 | 0.0157882 | 0.440192 | 392,711 | -0.00769604 | 361,429 / 957,250 |
| FIGConvUNet | EPSP_lower | dimensionless | 48.53% | 0.0110773 | 0.0153317 | 0.320863 | 371,146 | -0.00990279 | 371,515 / 957,250 |

Relative L2 is undefined when the truth norm is zero (shown as —). Scalar values retain their physical units. Negative counts apply to declared nonnegative fields and refer to raw outputs; displacement sign is unrestricted.

| Configuration | EPSP field | Truth positive / exact zero / total | Prediction negatives | Prediction decreases / increments | Max prediction decrease | Truth decreases / increments | Exact support TP / FP / FN | Exact support IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Transolver | EPSP_upper | 361,429 / 595,821 / 957,250 | 201,590 | 258,776 / 861,525 | 0.00526176 | 0 / 861,525 | 328,742 / 426,918 / 32,687 | 41.7% |
| Transolver | EPSP_lower | 371,515 / 585,735 / 957,250 | 205,166 | 263,206 / 861,525 | 0.00528122 | 0 / 861,525 | 329,720 / 422,364 / 41,795 | 41.53% |
| FLARE | EPSP_upper | 361,429 / 595,821 / 957,250 | 479,915 | 358,977 / 861,525 | 0.00382134 | 0 / 861,525 | 284,652 / 192,682 / 76,777 | 51.37% |
| FLARE | EPSP_lower | 371,515 / 585,735 / 957,250 | 488,649 | 354,601 / 861,525 | 0.00387328 | 0 / 861,525 | 288,225 / 180,376 / 83,290 | 52.22% |
| GeoTransolver GALE | EPSP_upper | 361,429 / 595,821 / 957,250 | 274,291 | 378,649 / 861,525 | 0.00661356 | 0 / 861,525 | 306,531 / 376,428 / 54,898 | 41.54% |
| GeoTransolver GALE | EPSP_lower | 371,515 / 585,735 / 957,250 | 250,132 | 390,680 / 861,525 | 0.00536919 | 0 / 861,525 | 311,586 / 395,532 / 59,929 | 40.62% |
| GeoTransolver GALE_FA | EPSP_upper | 361,429 / 595,821 / 957,250 | 328,149 | 345,633 / 861,525 | 0.00866064 | 0 / 861,525 | 309,510 / 319,591 / 51,919 | 45.45% |
| GeoTransolver GALE_FA | EPSP_lower | 371,515 / 585,735 / 957,250 | 334,074 | 354,332 / 861,525 | 0.00749798 | 0 / 861,525 | 313,876 / 309,300 / 57,639 | 46.1% |
| MeshGraphNet | EPSP_upper | 361,429 / 595,821 / 957,250 | 429,689 | 373,156 / 861,525 | 0.00430577 | 0 / 861,525 | 315,221 / 212,340 / 46,208 | 54.94% |
| MeshGraphNet | EPSP_lower | 371,515 / 585,735 / 957,250 | 407,520 | 394,220 / 861,525 | 0.00406411 | 0 / 861,525 | 328,352 / 221,377 / 43,163 | 55.38% |
| FIGConvUNet | EPSP_upper | 361,429 / 595,821 / 957,250 | 392,711 | 329,761 / 861,525 | 0.0106633 | 0 / 861,525 | 325,120 / 239,419 / 36,309 | 54.11% |
| FIGConvUNet | EPSP_lower | 371,515 / 585,735 / 957,250 | 371,146 | 342,031 / 861,525 | 0.0166814 | 0 / 861,525 | 330,181 / 255,923 / 41,334 | 52.62% |

EPSP support uses exact >0 on the projected nodal labels and raw predictions. Strict negative increments include roundoff; their magnitude distributions and the same truth statistics remain in SUMMARY.json. No clipping, support threshold tuning, monotonic correction or principal reordering is applied.

## Saved architecture settings

These are the settings of the fitted model, copied from each frozen training report.

| Configuration | Saved model_config |
|---|---|
| Transolver | `{"dropout":0.0,"embedding_dim":3,"functional_dim":4,"mlp_ratio":2,"n_head":4,"n_hidden":48,"n_layers":2,"out_dim":70,"plus":false,"slice_num":8,"structured_shape":null,"time_input":false,"unified_pos":false,"use_te":false}` |
| FLARE | `{"dropout":0.0,"embedding_dim":3,"functional_dim":4,"mlp_ratio":2,"n_head":4,"n_hidden":48,"n_layers":2,"out_dim":70,"slice_num":8,"structured_shape":null,"time_input":false,"unified_pos":false,"use_te":false}` |
| GeoTransolver GALE | `{"attention_type":"GALE","concrete_dropout":false,"dropout":0.0,"functional_dim":4,"geometry_dim":3,"global_dim":3,"include_local_features":false,"mlp_ratio":2,"n_head":4,"n_hidden":48,"n_layers":2,"out_dim":70,"plus":false,"slice_num":8,"state_mixing_mode":"weighted","structured_shape":null,"time_input":false,"use_te":false}` |
| GeoTransolver GALE_FA | `{"attention_type":"GALE_FA","concrete_dropout":false,"dropout":0.0,"functional_dim":4,"geometry_dim":3,"global_dim":3,"include_local_features":false,"mlp_ratio":2,"n_head":4,"n_hidden":48,"n_layers":2,"out_dim":70,"plus":false,"slice_num":8,"state_mixing_mode":"weighted","structured_shape":null,"time_input":false,"use_te":false}` |
| MeshGraphNet | `{"aggregation":"sum","checkpoint_offloading":false,"do_concat_trick":false,"hidden_dim_edge_encoder":48,"hidden_dim_node_decoder":48,"hidden_dim_node_encoder":48,"hidden_dim_processor":48,"input_dim_edges":4,"input_dim_nodes":7,"mlp_activation_fn":"relu","norm_type":"LayerNorm","num_layers_edge_encoder":2,"num_layers_edge_processor":2,"num_layers_node_decoder":2,"num_layers_node_encoder":2,"num_layers_node_processor":2,"num_processor_checkpoint_segments":0,"output_dim":70,"processor_size":3,"recompute_activation":false}` |
| FIGConvUNet | `{"aabb_max":[1.0,1.0,1.0],"aabb_min":[0.0,0.0,0.0],"communication_types":["sum"],"has_input_features":true,"hidden_channels":[16,24,32],"in_channels":7,"kernel_size":3,"knn_k":8,"neighbor_search_type":"knn","num_down_blocks":1,"num_levels":2,"num_up_blocks":1,"out_channels":70,"pos_encode_dim":8,"reductions":["mean"],"resolution_memory_format_pairs":[["b_xc_y_z",[2,16,16]],["b_yc_x_z",[16,2,16]],["b_zc_x_y",[16,16,2]]],"to_point_sample_method":"graphconv","use_rel_pos":true,"use_rel_pos_embed":true,"use_scalar_output":false,"voxel_size":null}` |

## Evidence and interpretation

- **Transolver**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=no.
- **FLARE**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=no.
- **GeoTransolver GALE**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=no.
- **GeoTransolver GALE_FA**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=no.
- **MeshGraphNet**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=no.
- **FIGConvUNet**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=yes.

| Separate CPU workload | PID | Start UTC | End UTC | Wall (s) | Observed peak RSS (MiB) | Result | Recorded static phase |
|---|---:|---|---|---:|---:|---|---|
| CPU-only full numerical elasticity DDM sidecar | 2548 | 2026-10-15T07:25:45.2599244Z | 2026-10-15T07:26:05.9350072Z | 20.675 | 199.125 | Partial numerical attempt; shared host-memory guard stop, not solver convergence/error or completed DDM result | bumper/transolver/train |
| CPU-only reproduction metadata and cached-wheel hashing | 14924 | 2026-10-15T07:38:09.537Z | 2026-10-15T07:38:34.567331Z | 25.0298 | 51.9102 | Official metadata for90 exact package versions resolved with0 errors; no framework imports, installation or wheel downloads | bumper/flare/train |
| CPU-only cached-wheel copy and official Torch wheel download | 43576 | 2026-10-15T07:40:14.937926Z | 2026-10-15T07:41:59.034696Z | 104.095 | 30.5 | Shared host-memory guard stopped file preservation;89 small wheels verified, official Torch wheel remains partial. No installation, framework import or GPU access | bumper/flare/train |

This sidecar records an actual shared-workstation CPU overlap for the overall suite; the recorded phase identifies which model was active. It does not imply that the CPU workload overlapped every model. A guard-stopped partial numerical attempt is not a completed result.

- A single fixed seed and small models under a short compute budget do not establish convergence or an optimized recipe; later improvements may remain possible.
- Different approved batch sizes change optimizer trajectories and observations per update. Compare the recorded compute, samples, complete epochs and quality gates together.
- Phone labels describe only the initial 2–20 µs on a fixed assembly and two condition axes with a 5/2/2 simulation split. Bumper uses its fixed 112/12/7 split. These splits were previously evaluated by separate ROM methods and are not globally unseen research holdouts.
- Fields keep their own physical units and denominators. Displacement accuracy can be dominated by rigid translation; inspect centroid-removed errors and each stress/EPSP field separately.
- Nodal projection and stored-time fields do not establish original integration-point peaks, damage, full rebound, failure prediction or new-geometry generalization.
- Fresh validation checks checkpoint serialization at the validation anchor; it is not an independent physical accuracy test. Held-out performance uses one frozen-checkpoint pass, with no test-driven selection.
- Observed GPU timings and sampled memory may include contention from concurrent local solver workloads; memory samples are lower bounds rather than continuous host-memory peaks.
- The original bumper C7 export contains displacement, upper/lower in-plane VM and upper/lower EPSP; total-strain Frobenius was not exported and no total-strain metric is invented.

This summary reads only the saved reports and suite metadata. It does not rerun inference or open checkpoint, FEM target, or saved prediction arrays. Input-report and summarizer hashes are recorded in SUMMARY.json.
