# Dose-2 specificity replication summary

Gap reduction is baseline gap minus control gap; positive values reduce the gap.
Intervals are 95% paired bootstraps over question IDs (templates averaged within question/framing).

| Cell | Baseline gap | Learned reduction [95% CI] | Eval gain | Deploy loss | Max abs. random | Max abs. off-window |
|---|---:|---:|---:|---:|---:|---:|
| gemma2b_refusal | 0.350 | 0.329 [0.171, 0.493] | 0.286 | 0.043 | 0.011 | 0.118 |
| gemma2b_sandbag | 0.280 | 0.185 [0.090, 0.278] | 0.225 | -0.040 | 0.010 | 0.025 |
| gemma2b_syc | 0.492 | 0.150 [0.062, 0.242] | 0.158 | -0.008 | 0.017 | 0.017 |
| gemma9b_refusal | 0.618 | 0.639 [0.411, 0.839] | 0.646 | -0.007 | 0.000 | 0.043 |
| gemma9b_sandbag | 0.555 | 0.355 [0.280, 0.430] | 0.360 | -0.005 | 0.003 | 0.035 |
| gemma9b_syc | 0.679 | 0.421 [0.346, 0.500] | 0.421 | 0.000 | 0.008 | 0.025 |
| llama8b_refusal | 0.618 | 0.196 [0.096, 0.307] | 0.196 | 0.000 | 0.011 | 0.239 |
| llama8b_sandbag | 0.353 | 0.220 [0.145, 0.292] | 0.220 | 0.000 | 0.000 | 0.118 |
| llama8b_syc | 0.617 | -0.008 [-0.025, 0.000] | 0.000 | -0.008 | 0.000 | 0.013 |
| qwen7b_refusal | 0.768 | 0.779 [0.586, 0.939] | 0.693 | 0.086 | 0.004 | 0.014 |
| qwen7b_sandbag | 0.585 | 0.405 [0.330, 0.480] | 0.410 | -0.005 | 0.007 | 0.080 |
| qwen7b_syc | 0.738 | 0.558 [0.467, 0.646] | 0.542 | 0.017 | 0.008 | 0.008 |

## Qwen sandbag training seeds

| Seed | Learned reduction [95% CI] | Max abs. random | Max abs. off-window |
|---:|---:|---:|---:|
| 42 | 0.405 [0.330, 0.480] | 0.007 | 0.080 |
| 43 | 0.4875 [0.405, 0.568] | 0.008 | 0.020 |
| 44 | 0.5325 [0.438, 0.623] | 0.010 | 0.055 |

## Every operator-matched random clamp

- gemma2b_refusal: +0.004, +0.011, +0.004, +0.004, +0.000
- gemma2b_sandbag: -0.002, -0.007, -0.002, -0.010, -0.002
- gemma2b_syc: +0.004, +0.017, +0.000, +0.004, +0.012
- gemma9b_refusal: +0.000, +0.000, +0.000, +0.000, +0.000
- gemma9b_sandbag: +0.000, +0.000, +0.000, -0.003, +0.000
- gemma9b_syc: +0.000, +0.000, -0.004, +0.000, +0.008
- llama8b_refusal: +0.004, +0.011, +0.007, +0.004, +0.004
- llama8b_sandbag: +0.000, +0.000, +0.000, +0.000, +0.000
- llama8b_syc: +0.000, +0.000, +0.000, +0.000, +0.000
- qwen7b_refusal: +0.004, +0.000, -0.004, +0.000, +0.004
- qwen7b_sandbag: +0.000, -0.005, -0.005, -0.007, -0.005
- qwen7b_syc: +0.000, +0.000, +0.000, +0.008, +0.004

## Every width-matched off-window clamp

- gemma2b_refusal: ctl_clamp_L20_21_22_23: +0.118, ctl_clamp_L4_5_6_7: +0.029
- gemma2b_sandbag: ctl_clamp_L20_21_22_23: -0.025, ctl_clamp_L4_5_6_7: -0.017
- gemma2b_syc: ctl_clamp_L20_21_22_23: +0.012, ctl_clamp_L4_5_6_7: -0.017
- gemma9b_refusal: ctl_clamp_L30_31_32_33: +0.043, ctl_clamp_L8_9_10_11: +0.036
- gemma9b_sandbag: ctl_clamp_L30_31_32_33: +0.007, ctl_clamp_L8_9_10_11: +0.035
- gemma9b_syc: ctl_clamp_L30_31_32_33: +0.000, ctl_clamp_L8_9_10_11: -0.025
- llama8b_refusal: ctl_clamp_L24_25_26_27: +0.239, ctl_clamp_L6_7_8_9: +0.050
- llama8b_sandbag: ctl_clamp_L24_25_26_27: +0.118, ctl_clamp_L6_7_8_9: +0.105
- llama8b_syc: ctl_clamp_L24_25_26_27: -0.004, ctl_clamp_L6_7_8_9: +0.013
- qwen7b_refusal: ctl_clamp_L24_25_26_27: +0.004, ctl_clamp_L4_5_6_7: -0.014
- qwen7b_sandbag: ctl_clamp_L24_25_26_27: +0.080, ctl_clamp_L4_5_6_7: -0.005
- qwen7b_syc: ctl_clamp_L24_25_26_27: -0.004, ctl_clamp_L4_5_6_7: +0.008
