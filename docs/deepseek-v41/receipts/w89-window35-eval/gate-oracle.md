# W89 training-free gate-oracle route eval (window-35 real trace)

Real router weights (`layers.L.ffn.gate.{weight,bias}`, 384 experts, top-6, 5120-d) via the port's exact `_gate_prefix_impl` (sqrtsoftplus + noaux_tc bias). Test = 256 held-out decode rows; mean over 40 layers.

| predictor | prec@6 | rec@6 | missRed@6 | missRed@8 | missRed@10 | missRed@12 | missRed@16 | missRed@24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a' gate_L(router_in_L) vs top6_L      (recompute; alignment ~1.0) | 0.999 | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| b' gate_L(layer_in_{L-1}) vs top6_L   (one layer ahead, no training) | 0.622 | 0.622 | 0.622 | 0.693 | 0.736 | 0.766 | 0.805 | 0.848 |
| c' gate_L(layer_in_{L-2}) vs top6_L   (two layers ahead, no training) | 0.565 | 0.565 | 0.565 | 0.635 | 0.677 | 0.710 | 0.753 | 0.802 |

**Residual cosine** cos(layer_in_L, layer_in_{L-1}) mean over layers = **0.9209** (worst: L1=0.766, L19=0.853, L15=0.877, L17=0.879, L14=0.879)

## Worst-5 layers by missRed@6

- **a_prime**: L18=1.00, L1=1.00, L29=1.00, L4=1.00, L37=1.00
- **b_prime**: L1=0.26, L2=0.34, L3=0.34, L4=0.38, L5=0.48
- **c_prime**: L2=0.25, L3=0.27, L4=0.30, L5=0.35, L6=0.42

## Verdict
- One layer ahead (b'), **mean miss_reduction clears 0.70 at prefetch width k=10** (missRed@10=0.736); at k=6 it is 0.622.
- Explained by the residual stream's layer-to-layer stability (cos=0.921); it is lowest in the early layers, which are the per-layer floor (worst-5 above).
