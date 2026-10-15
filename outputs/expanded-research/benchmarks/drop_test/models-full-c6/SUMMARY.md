# Phone C6 full support — bounded static training comparison

A bounded initial comparison of six configurations from five model families on a fixed full-support physical-field task; no universal ranking, convergence or design-use accuracy claim.

Suite status: **completed**. Model evidence: 6 complete, 0 failed, 0 missing.

Each configuration had a 240 s synchronized optimizer-compute budget and a 5,000-update maximum. Minimum 100 updates and two complete epochs are report-only quality gates. A final update may cross the time cap; its actual duration is retained.

| Configuration | Evidence | Batch | Updates | Samples | Complete epochs | Selected epoch | Compute at selection (s) | Actual compute / cap (s) | Overshoot (s) | Quality gate | Best validation normalized MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Transolver | complete | 5 | 795 | 3,975 | 795 | 754 | 227.813 | 240.244 / 240 | 0.244071 | yes | 0.0490526 |
| FLARE | complete | 5 | 247 | 1,235 | 247 | 247 | 240.232 | 240.232 / 240 | 0.231578 | yes | 0.114191 |
| GeoTransolver GALE | complete | 5 | 682 | 3,410 | 682 | 682 | 240.296 | 240.296 / 240 | 0.29587 | yes | 0.0452166 |
| GeoTransolver GALE_FA | complete | 5 | 208 | 1,040 | 208 | 208 | 241.127 | 241.127 / 240 | 1.12684 | yes | 0.146723 |
| MeshGraphNet | complete | 1 | 764 | 764 | 152 | 150 | 235.793 | 240.298 / 240 | 0.297642 | yes | 0.0506853 |
| FIGConvUNet | complete | 2 | 403 | 672 | 134 | 133 | 237.739 | 240.231 / 240 | 0.231428 | yes | 0.121919 |

A normalized validation loss is used only for checkpoint selection within its own domain. It is not a physical accuracy metric and is not comparable between the two domains.

| Configuration | Selected updates | Selected case observations | Selected epoch | Selected compute (s) | Last complete training normalized MSE | Last eligible validation improved |
|---|---:|---:|---:|---:|---:|---|
| Transolver | 754 | 3,770 | 754 | 227.813 | 0.205323 | no |
| FLARE | 247 | 1,235 | 247 | 240.232 | 0.281953 | yes |
| GeoTransolver GALE | 682 | 3,410 | 682 | 240.296 | 0.201735 | yes |
| GeoTransolver GALE_FA | 208 | 1,040 | 208 | 241.127 | 0.309114 | yes |
| MeshGraphNet | 750 | 750 | 150 | 235.793 | 0.206175 | no |
| FIGConvUNet | 399 | 665 | 133 | 237.739 | 0.28413 | no |

| Configuration | Parameters | Training CUDA allocated peak (MiB) | CUDA reserved peak (MiB) | Observed RSS peak lower bound (MiB) | Minimum observed available host RAM (MiB) | Observed solver PIDs |
|---|---:|---:|---:|---:|---:|---|
| Transolver | 42,660 | 3004.27 | 3296 | 1380 | 1772.5 | 34268 |
| FLARE | 38,268 | 2924.01 | 3580 | 1268.43 | 675.469 | 34268 |
| GeoTransolver GALE | 44,870 | 3281.3 | 3678 | 1408.3 | 548.742 | 34268 |
| GeoTransolver GALE_FA | 40,478 | 3774.95 | 4568 | 1405.31 | 365.094 | 34268 |
| MeshGraphNet | 81,468 | 2880.43 | 3300 | 1647 | 421.945 | — |
| FIGConvUNet | 587,164 | 3185.76 | 3856 | 1728.97 | 549.484 | — |

| Configuration | Test CUDA allocated peak (MiB) | Test CUDA reserved peak (MiB) | Test observed RSS peak lower bound (MiB) |
|---|---:|---:|---:|
| Transolver | 122.467 | 176 | 1314.11 |
| FLARE | 117.804 | 136 | 1226.79 |
| GeoTransolver GALE | 147.005 | 176 | 1349.81 |
| GeoTransolver GALE_FA | 147.06 | 218 | 1346.22 |
| MeshGraphNet | 788.201 | 1102 | 1363.1 |
| FIGConvUNet | 374.224 | 526 | 1642.69 |

CUDA allocated and reserved peaks are PyTorch allocator counters, not sampled physical device VRAM. NVML/device-level VRAM sampling is **unmeasured** in these runs. Host RSS is a sampled peak lower bound; available host RAM is a separately sampled system-wide quantity. Its minimum is the lowest observed value, not a continuous minimum. Test-process available-RAM minimum was not separately measured.

| Configuration | Optimizer compute (s) | Training wall (s) | Setup including import (s) | Monitoring (s) | Test process wall (s) |
|---|---:|---:|---:|---:|---:|
| Transolver | 240.244 | 923.502 | 25.8195 | 372.581 | 39.7314 |
| FLARE | 240.232 | 620.732 | 22.023 | 262.421 | 44.3261 |
| GeoTransolver GALE | 240.296 | 970.101 | 25.5945 | 445.671 | 41.3876 |
| GeoTransolver GALE_FA | 241.127 | 596.348 | 24.2713 | 248.544 | 38.9698 |
| MeshGraphNet | 240.298 | 487.469 | 25.169 | 162.15 | 39.745 |
| FIGConvUNet | 240.231 | 504.418 | 32.6939 | 167.769 | 43.5393 |

| Configuration | Training read / target checks (s) | Normalization / CPU stacking (s) | Host-to-device (s) | CPU snapshots / checkpoint I/O (s) | Validation (s) |
|---|---:|---:|---:|---:|---:|
| Transolver | 83.0111 | 155.755 | 21.0255 | 2.41537 | 4.76332 |
| FLARE | 26.2549 | 51.0403 | 7.81686 | 0.778352 | 5.18893 |
| GeoTransolver GALE | 77.5126 | 137.157 | 18.4669 | 3.26655 | 5.84486 |
| GeoTransolver GALE_FA | 23.0164 | 41.9939 | 5.87883 | 1.11475 | 6.26278 |
| MeshGraphNet | 13.0087 | 28.562 | 3.90915 | 0.842502 | 7.46081 |
| FIGConvUNet | 12.9234 | 25.5702 | 3.35075 | 2.99058 | 14.9437 |

These observed timing categories are not an exhaustive disjoint partition: setup includes initial monitoring, while end verification and other work are not all separately timed. The 240 s optimizer cap is not a wall-time cap. Monitoring and data preparation overhead are reported rather than attributed to model compute. Memory samples are lower bounds; concurrent solver intervals remain in SUMMARY.json.

| Configuration | Test state | Forward total (s) | Displacement relative L2 | Vector RMSE (mm) | Vector error P95 (mm) | Vector error max (mm) | Centroid-removed relative L2 | Fresh validation max difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Transolver | completed | 0.412113 | 1.303% | 0.000876161 | 0.00101882 | 0.077281 | 67.44% | 0 |
| FLARE | completed | 0.43814 | 1.78% | 0.00119731 | 0.00147935 | 0.0851537 | 86.46% | 0 |
| GeoTransolver GALE | completed | 0.472299 | 1.442% | 0.000969468 | 0.0012397 | 0.0648064 | 74.07% | 0 |
| GeoTransolver GALE_FA | completed | 0.849996 | 2.824% | 0.00189903 | 0.0024967 | 0.0719103 | 145.5% | 0 |
| MeshGraphNet | completed | 0.583439 | 1.455% | 0.000978672 | 0.00119697 | 0.0467048 | 67.14% | 5.72205e-06 |
| FIGConvUNet | completed | 1.15596 | 2.461% | 0.0016551 | 0.00216906 | 0.0902396 | 127% | 0 |

The forward time contains one synchronized batch-size-one forward per held-out case. It excludes input preparation, transfer, saved-output I/O and metric/figure generation; it is a shared-workstation observation, not an isolated system benchmark.

| Configuration | Test setup (s) | Input read / checks (s) | Normalize / stack (s) | Host-to-device (s) | Forward (s) | Output transfer / unscale / save / truth read (s) | Metrics / figures (s) | Test wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transolver | 19.4993 | 0.0039924 | 0.0095556 | 0.001904 | 0.412113 | 0.234398 | 18.4392 | 39.7314 |
| FLARE | 20.6299 | 0.0048485 | 0.0113127 | 0.0019481 | 0.43814 | 0.806894 | 21.2426 | 44.3261 |
| GeoTransolver GALE | 19.711 | 0.0034495 | 0.0110536 | 0.0020948 | 0.472299 | 0.289275 | 19.7598 | 41.3876 |
| GeoTransolver GALE_FA | 18.9075 | 0.0042709 | 0.0108598 | 0.0020069 | 0.849996 | 0.363535 | 17.7208 | 38.9698 |
| MeshGraphNet | 20.5068 | 0.0039353 | 0.0104065 | 0.0019684 | 0.583439 | 0.261601 | 17.2093 | 39.745 |
| FIGConvUNet | 20.3864 | 0.0042465 | 0.0109267 | 0.0017933 | 1.15596 | 0.282192 | 20.5238 | 43.5393 |

These are sums of the original one-pass records. Standalone held-out truth-read time is **unmeasured**: it shares a timer with output transfer, unscaling and saved-array I/O. Setup includes framework import and frozen-source/cache checks. Other end verification and report hashing remain in total wall time; the categories are not an exhaustive disjoint partition. Per-case timings are preserved in SUMMARY.json.

| Configuration | Field | Unit | Relative L2 | RMSE | Absolute error P95 | Absolute error max | Raw prediction negatives | Raw minimum | Truth exact nonzero / count |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Transolver | displacement_x | mm | 1.356% | 0.000538096 | 0.000629209 | 0.0497032 | — | -0.0680625 | 1,521,280 / 1,521,300 |
| Transolver | displacement_y | mm | 1.271% | 0.000553177 | 0.000664946 | 0.0543253 | — | -0.0745946 | 1,521,280 / 1,521,300 |
| Transolver | displacement_z | mm | 1.278% | 0.000414857 | 0.000461075 | 0.0312816 | — | -0.0544797 | 1,521,280 / 1,521,300 |
| Transolver | von_mises_GPS | MPa | 33.8% | 0.15067 | 0.0996018 | 18.2135 | 600,081 | -0.214176 | 1,354,540 / 1,521,300 |
| Transolver | total_strain_frobenius | 1 | 36.49% | 7.96288e-05 | 2.08302e-05 | 0.0103479 | 520,256 | -0.000142252 | 1,354,412 / 1,521,300 |
| Transolver | epsp | 1 | 96.08% | 3.31994e-06 | 1.91356e-07 | 0.00263193 | 658,778 | -7.88672e-06 | 27 / 1,521,300 |
| FLARE | displacement_x | mm | 1.727% | 0.000685461 | 0.000967065 | 0.0497843 | — | -0.0659522 | 1,521,280 / 1,521,300 |
| FLARE | displacement_y | mm | 1.697% | 0.00073857 | 0.000896792 | 0.0556693 | — | -0.0730955 | 1,521,280 / 1,521,300 |
| FLARE | displacement_z | mm | 1.993% | 0.000646691 | 0.000701724 | 0.0459598 | — | -0.0536246 | 1,521,280 / 1,521,300 |
| FLARE | von_mises_GPS | MPa | 55.2% | 0.246047 | 0.204074 | 22.9693 | 73,367 | -0.0431562 | 1,354,540 / 1,521,300 |
| FLARE | total_strain_frobenius | 1 | 57.78% | 0.000126067 | 4.58679e-05 | 0.0130658 | 79,744 | -3.43591e-05 | 1,354,412 / 1,521,300 |
| FLARE | epsp | 1 | 99.35% | 3.43303e-06 | 1.68605e-07 | 0.00270335 | 560,827 | -7.51048e-07 | 27 / 1,521,300 |
| GeoTransolver GALE | displacement_x | mm | 1.421% | 0.000563939 | 0.000795003 | 0.0404296 | — | -0.067216 | 1,521,280 / 1,521,300 |
| GeoTransolver GALE | displacement_y | mm | 1.382% | 0.00060171 | 0.000793514 | 0.0477397 | — | -0.0733397 | 1,521,280 / 1,521,300 |
| GeoTransolver GALE | displacement_z | mm | 1.57% | 0.000509692 | 0.000649343 | 0.025923 | — | -0.059595 | 1,521,280 / 1,521,300 |
| GeoTransolver GALE | von_mises_GPS | MPa | 32.01% | 0.142688 | 0.0752589 | 18.175 | 942,977 | -0.291643 | 1,354,540 / 1,521,300 |
| GeoTransolver GALE | total_strain_frobenius | 1 | 35.15% | 7.66938e-05 | 1.78089e-05 | 0.0101853 | 999,934 | -0.000198562 | 1,354,412 / 1,521,300 |
| GeoTransolver GALE | epsp | 1 | 96.95% | 3.34985e-06 | 2.26295e-07 | 0.00265482 | 804,561 | -1.17723e-05 | 27 / 1,521,300 |
| GeoTransolver GALE_FA | displacement_x | mm | 2.897% | 0.00114965 | 0.00110784 | 0.0524612 | — | -0.0697579 | 1,521,280 / 1,521,300 |
| GeoTransolver GALE_FA | displacement_y | mm | 2.575% | 0.00112088 | 0.00139953 | 0.0482033 | — | -0.0719199 | 1,521,280 / 1,521,300 |
| GeoTransolver GALE_FA | displacement_z | mm | 3.124% | 0.00101402 | 0.00149311 | 0.0399718 | — | -0.0561618 | 1,521,280 / 1,521,300 |
| GeoTransolver GALE_FA | von_mises_GPS | MPa | 60.65% | 0.270377 | 0.141295 | 25.2642 | 624,458 | -0.197893 | 1,354,540 / 1,521,300 |
| GeoTransolver GALE_FA | total_strain_frobenius | 1 | 67.71% | 0.000147752 | 2.48973e-05 | 0.0139505 | 535,991 | -5.15217e-05 | 1,354,412 / 1,521,300 |
| GeoTransolver GALE_FA | epsp | 1 | 99.98% | 3.45449e-06 | 3.69873e-07 | 0.00271476 | 753,515 | -3.3051e-06 | 27 / 1,521,300 |
| MeshGraphNet | displacement_x | mm | 1.378% | 0.000546782 | 0.000832306 | 0.0279211 | — | -0.0680779 | 1,521,280 / 1,521,300 |
| MeshGraphNet | displacement_y | mm | 1.223% | 0.000532502 | 0.00067373 | 0.0357958 | — | -0.0740472 | 1,521,280 / 1,521,300 |
| MeshGraphNet | displacement_z | mm | 1.887% | 0.000612592 | 0.000766318 | 0.0194034 | — | -0.0536247 | 1,521,280 / 1,521,300 |
| MeshGraphNet | von_mises_GPS | MPa | 36.77% | 0.163903 | 0.145086 | 11.645 | 662,183 | -0.176727 | 1,354,540 / 1,521,300 |
| MeshGraphNet | total_strain_frobenius | 1 | 34.25% | 7.47424e-05 | 2.67748e-05 | 0.00591581 | 311,559 | -5.00083e-05 | 1,354,412 / 1,521,300 |
| MeshGraphNet | epsp | 1 | 97.12% | 3.35574e-06 | 2.76164e-07 | 0.00260534 | 649,161 | -2.66069e-06 | 27 / 1,521,300 |
| FIGConvUNet | displacement_x | mm | 2.547% | 0.00101073 | 0.00109202 | 0.0543894 | — | -0.0717677 | 1,521,280 / 1,521,300 |
| FIGConvUNet | displacement_y | mm | 2.181% | 0.000949285 | 0.00103038 | 0.0618545 | — | -0.0757289 | 1,521,280 / 1,521,300 |
| FIGConvUNet | displacement_z | mm | 2.784% | 0.000903688 | 0.0010092 | 0.0418037 | — | -0.0543528 | 1,521,280 / 1,521,300 |
| FIGConvUNet | von_mises_GPS | MPa | 53.08% | 0.236634 | 0.0828615 | 24.1223 | 445,194 | -0.193491 | 1,354,540 / 1,521,300 |
| FIGConvUNet | total_strain_frobenius | 1 | 63.16% | 0.000137823 | 1.98627e-05 | 0.0135966 | 344,739 | -6.4801e-05 | 1,354,412 / 1,521,300 |
| FIGConvUNet | epsp | 1 | 99.53% | 3.43901e-06 | 1.62464e-07 | 0.00270806 | 761,186 | -1.75884e-06 | 27 / 1,521,300 |

Relative L2 is undefined when the truth norm is zero (shown as —). Scalar values retain their physical units. Negative counts apply to declared nonnegative fields and refer to raw outputs; displacement sign is unrestricted.

| Configuration | EPSP field | Truth positive / exact zero / total | Prediction negatives | Prediction decreases / increments | Max prediction decrease | Truth decreases / increments | Exact support TP / FP / FN | Exact support IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Transolver | epsp | 27 / 1,521,273 / 1,521,300 | 658,778 | 593,109 / 1,369,170 | 3.84686e-06 | 0 / 1,369,170 | 27 / 862,495 / 0 | 0.00313% |
| FLARE | epsp | 27 / 1,521,273 / 1,521,300 | 560,827 | 523,845 / 1,369,170 | 5.68968e-07 | 0 / 1,369,170 | 27 / 960,446 / 0 | 0.002811% |
| GeoTransolver GALE | epsp | 27 / 1,521,273 / 1,521,300 | 804,561 | 713,476 / 1,369,170 | 6.54098e-06 | 0 / 1,369,170 | 27 / 716,712 / 0 | 0.003767% |
| GeoTransolver GALE_FA | epsp | 27 / 1,521,273 / 1,521,300 | 753,515 | 692,819 / 1,369,170 | 4.24376e-06 | 0 / 1,369,170 | 27 / 767,758 / 0 | 0.003517% |
| MeshGraphNet | epsp | 27 / 1,521,273 / 1,521,300 | 649,161 | 758,493 / 1,369,170 | 4.84202e-06 | 0 / 1,369,170 | 27 / 872,112 / 0 | 0.003096% |
| FIGConvUNet | epsp | 27 / 1,521,273 / 1,521,300 | 761,186 | 682,393 / 1,369,170 | 1.92438e-06 | 0 / 1,369,170 | 27 / 760,087 / 0 | 0.003552% |

EPSP support uses exact >0 on the projected nodal labels and raw predictions. Strict negative increments include roundoff; their magnitude distributions and the same truth statistics remain in SUMMARY.json. No clipping, support threshold tuning, monotonic correction or principal reordering is applied.

| Configuration | Contact-free ballistic displacement L2 | Gravity-only displacement L2 | Model SSE / ballistic SSE |
|---|---:|---:|---:|
| Transolver | 1.935% | 100% | 0.453267 |
| FLARE | 1.935% | 100% | 0.846446 |
| GeoTransolver GALE | 1.935% | 100% | 0.554949 |
| GeoTransolver GALE_FA | 1.935% | 100% | 2.12935 |
| MeshGraphNet | 1.935% | 100% | 0.565536 |
| FIGConvUNet | 1.935% | 100% | 1.61747 |

The ballistic comparator uses each case's hash-verified source-deck initial velocity and signed acceleration, with stored time converted to seconds: u=v0·t+0.5·a·t²=v0·t−0.5·g·t². It omits contact, deformation and constraints; its error is not a solver validation measure.

## Saved architecture settings

These are the settings of the fitted model, copied from each frozen training report.

| Configuration | Saved model_config |
|---|---|
| Transolver | `{"dropout":0.0,"embedding_dim":3,"functional_dim":3,"mlp_ratio":2,"n_head":4,"n_hidden":48,"n_layers":2,"out_dim":60,"plus":false,"slice_num":8,"structured_shape":null,"time_input":false,"unified_pos":false,"use_te":false}` |
| FLARE | `{"dropout":0.0,"embedding_dim":3,"functional_dim":3,"mlp_ratio":2,"n_head":4,"n_hidden":48,"n_layers":2,"out_dim":60,"slice_num":8,"structured_shape":null,"time_input":false,"unified_pos":false,"use_te":false}` |
| GeoTransolver GALE | `{"attention_type":"GALE","concrete_dropout":false,"dropout":0.0,"functional_dim":4,"geometry_dim":3,"global_dim":2,"include_local_features":false,"mlp_ratio":2,"n_head":4,"n_hidden":48,"n_layers":2,"out_dim":60,"plus":false,"slice_num":8,"state_mixing_mode":"weighted","structured_shape":null,"time_input":false,"use_te":false}` |
| GeoTransolver GALE_FA | `{"attention_type":"GALE_FA","concrete_dropout":false,"dropout":0.0,"functional_dim":4,"geometry_dim":3,"global_dim":2,"include_local_features":false,"mlp_ratio":2,"n_head":4,"n_hidden":48,"n_layers":2,"out_dim":60,"plus":false,"slice_num":8,"state_mixing_mode":"weighted","structured_shape":null,"time_input":false,"use_te":false}` |
| MeshGraphNet | `{"aggregation":"sum","checkpoint_offloading":false,"do_concat_trick":false,"hidden_dim_edge_encoder":48,"hidden_dim_node_decoder":48,"hidden_dim_node_encoder":48,"hidden_dim_processor":48,"input_dim_edges":4,"input_dim_nodes":6,"mlp_activation_fn":"relu","norm_type":"LayerNorm","num_layers_edge_encoder":2,"num_layers_edge_processor":2,"num_layers_node_decoder":2,"num_layers_node_encoder":2,"num_layers_node_processor":2,"num_processor_checkpoint_segments":0,"output_dim":60,"processor_size":3,"recompute_activation":false}` |
| FIGConvUNet | `{"aabb_max":[1.0,1.0,1.0],"aabb_min":[0.0,0.0,0.0],"communication_types":["sum"],"has_input_features":true,"hidden_channels":[16,24,32],"in_channels":6,"kernel_size":3,"knn_k":8,"neighbor_search_type":"knn","num_down_blocks":1,"num_levels":2,"num_up_blocks":1,"out_channels":60,"pos_encode_dim":8,"reductions":["mean"],"resolution_memory_format_pairs":[["b_xc_y_z",[2,16,16]],["b_yc_x_z",[16,2,16]],["b_zc_x_y",[16,16,2]]],"to_point_sample_method":"graphconv","use_rel_pos":true,"use_rel_pos_embed":true,"use_scalar_output":false,"voxel_size":null}` |

## Evidence and interpretation

- **Transolver**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=no.
- **FLARE**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=yes.
- **GeoTransolver GALE**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=yes.
- **GeoTransolver GALE_FA**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=yes.
- **MeshGraphNet**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=no.
- **FIGConvUNet**: complete; stop=synchronized_optimizer_time_cap; minimum updates met=yes; last eligible validation improved=no.

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

This summary reads only the saved reports and suite metadata. It does not rerun inference or open checkpoint, FEM target, or saved prediction arrays. Input-report and summarizer hashes are recorded in SUMMARY.json.
