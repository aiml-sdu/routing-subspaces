| Model | Params (B) | Layer | Sandbag gap (pp) | Pool Gini | Eff. pool (50pct) | Eff. pool (90pct) | Routing window | Attractor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2-2B-it | 2.0 | L19 | +37.5 | 0.512 | 10 | 70 | L14-17 (94.1% recovery via path patching) | yes |
| Gemma-2-9B-it | 9.0 | L30 | +58.3 | 0.377 | 20 | 77 | L17-27 (90.3% recovery) | yes |
| Qwen-1.5B-Instruct | 1.5 | L20 | +28.1 | 0.345 | 27 | 92 | present (analogous window) | yes (partial) |
| Llama-3-8B-Instruct | 8.0 | L16 (divergence) | -24.4 | - | - | - | n/a | no (inverted gap) |
| Qwen-2.5-3B-Instruct | 3.0 | n/a | -57.8 | - | - | - | n/a | no (behavior untrained) |
| Qwen-3.5-2B-Instruct | 2.0 | n/a | +0.0 | - | - | - | n/a | no (collapse on at least one framing) |
