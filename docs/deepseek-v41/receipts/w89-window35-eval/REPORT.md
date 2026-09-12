# W89 Route Predictor -- Window 35 Real-Trace Evaluation

Source: real-model trace `route-traces-w35` (DeepSeek-V4.1-Flash, cell16k, 16,384-ctx, prefill-tail 2048 / decode 256). n_experts=384, top_k=6, hidden=5120, 40 layers.

Train on the 2048 prefill-tail rows, test on the 256 held-out decode rows, per layer, per predictor type. `precision@6 == recall@6` (both sets size 6). `miss_red_at_K` = fraction of that layer's true top-6 experts covered by a width-K one-layer-ahead prefetch (the feasibility number from W89_ROUTE_PREDICTOR.md).

## Run cost (wall time / peak RSS, worker guard = 1.5 GB target / 2 GB hard kill)

| model | wall time | peak RSS | exit | layers processed (a/b/c/d) | failures |
|---|---|---|---|---|---|
| ridge | 101.7s | 801.5 MB | 0 | 40/39/38/39 | none |
| logistic | 569.1s | 369.0 MB | 0 | 40/39/38/39 | none |
| mlp | 869.2s | 409.8 MB | 0 | 40/39/38/39 | none |

## Model: ridge

### Mean over layers

| predictor | n_layers | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|---|
| (a) router_in(L)->L (sanity ceiling) | 40 | 0.4061 | 0.4061 | 0.4061 | 0.4597 | 0.5302 |
| (b) layer_in(L-1)->L (one layer ahead) | 39 | 0.3565 | 0.3565 | 0.3565 | 0.4063 | 0.4746 |
| (c) layer_in(L-2)->L (two layers ahead) | 38 | 0.3391 | 0.3391 | 0.3391 | 0.3872 | 0.4536 |
| (d) prev-route+token->L (cheap features) | 39 | 0.3608 | 0.3608 | 0.3608 | 0.4104 | 0.4785 |

### Worst 5 layers by miss_red@6

**(a) router_in(L)->L (sanity ceiling)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.1790 | 0.1790 | 0.1790 | 0.2161 | 0.2747 |
| L18 | 0.2396 | 0.2396 | 0.2396 | 0.2754 | 0.3262 |
| L17 | 0.2422 | 0.2422 | 0.2422 | 0.2904 | 0.3555 |
| L16 | 0.2526 | 0.2526 | 0.2526 | 0.3014 | 0.3828 |
| L10 | 0.2708 | 0.2708 | 0.2708 | 0.3086 | 0.3698 |

**(b) layer_in(L-1)->L (one layer ahead)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.1608 | 0.1608 | 0.1608 | 0.1999 | 0.2689 |
| L17 | 0.1914 | 0.1914 | 0.1914 | 0.2285 | 0.2852 |
| L16 | 0.2090 | 0.2090 | 0.2090 | 0.2572 | 0.3242 |
| L13 | 0.2135 | 0.2135 | 0.2135 | 0.2520 | 0.3223 |
| L18 | 0.2181 | 0.2181 | 0.2181 | 0.2611 | 0.3105 |

**(c) layer_in(L-2)->L (two layers ahead)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.1530 | 0.1530 | 0.1530 | 0.1849 | 0.2422 |
| L18 | 0.1712 | 0.1712 | 0.1712 | 0.2057 | 0.2565 |
| L16 | 0.1921 | 0.1921 | 0.1921 | 0.2272 | 0.2865 |
| L17 | 0.2044 | 0.2044 | 0.2044 | 0.2448 | 0.3073 |
| L13 | 0.2116 | 0.2116 | 0.2116 | 0.2461 | 0.3027 |

**(d) prev-route+token->L (cheap features)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.1738 | 0.1738 | 0.1738 | 0.2057 | 0.2617 |
| L18 | 0.1803 | 0.1803 | 0.1803 | 0.2090 | 0.2747 |
| L16 | 0.1849 | 0.1849 | 0.1849 | 0.2070 | 0.2572 |
| L17 | 0.2103 | 0.2103 | 0.2103 | 0.2546 | 0.3105 |
| L13 | 0.2376 | 0.2376 | 0.2376 | 0.2721 | 0.3268 |

## Model: logistic

### Mean over layers

| predictor | n_layers | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|---|
| (a) router_in(L)->L (sanity ceiling) | 40 | 0.2276 | 0.2276 | 0.2276 | 0.2580 | 0.3036 |
| (b) layer_in(L-1)->L (one layer ahead) | 39 | 0.2224 | 0.2224 | 0.2224 | 0.2538 | 0.2997 |
| (c) layer_in(L-2)->L (two layers ahead) | 38 | 0.2176 | 0.2176 | 0.2176 | 0.2495 | 0.2954 |
| (d) prev-route+token->L (cheap features) | 39 | 0.1073 | 0.1073 | 0.1073 | 0.1229 | 0.1487 |

### Worst 5 layers by miss_red@6

**(a) router_in(L)->L (sanity ceiling)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.0371 | 0.0371 | 0.0371 | 0.0495 | 0.0697 |
| L13 | 0.0592 | 0.0592 | 0.0592 | 0.0710 | 0.0866 |
| L17 | 0.0885 | 0.0885 | 0.0885 | 0.1035 | 0.1335 |
| L12 | 0.0931 | 0.0931 | 0.0931 | 0.1146 | 0.1484 |
| L16 | 0.1042 | 0.1042 | 0.1042 | 0.1243 | 0.1628 |

**(b) layer_in(L-1)->L (one layer ahead)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.0391 | 0.0391 | 0.0391 | 0.0566 | 0.0794 |
| L13 | 0.0527 | 0.0527 | 0.0527 | 0.0716 | 0.0892 |
| L12 | 0.0749 | 0.0749 | 0.0749 | 0.0964 | 0.1289 |
| L17 | 0.0762 | 0.0762 | 0.0762 | 0.0957 | 0.1224 |
| L1 | 0.0801 | 0.0801 | 0.0801 | 0.0879 | 0.1087 |

**(c) layer_in(L-2)->L (two layers ahead)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.0365 | 0.0365 | 0.0365 | 0.0495 | 0.0749 |
| L13 | 0.0540 | 0.0540 | 0.0540 | 0.0684 | 0.0853 |
| L16 | 0.0664 | 0.0664 | 0.0664 | 0.0872 | 0.1276 |
| L12 | 0.0710 | 0.0710 | 0.0710 | 0.0938 | 0.1302 |
| L17 | 0.0781 | 0.0781 | 0.0781 | 0.0951 | 0.1204 |

**(d) prev-route+token->L (cheap features)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.0215 | 0.0215 | 0.0215 | 0.0267 | 0.0397 |
| L19 | 0.0260 | 0.0260 | 0.0260 | 0.0352 | 0.0475 |
| L16 | 0.0319 | 0.0319 | 0.0319 | 0.0391 | 0.0560 |
| L21 | 0.0326 | 0.0326 | 0.0326 | 0.0404 | 0.0566 |
| L13 | 0.0345 | 0.0345 | 0.0345 | 0.0423 | 0.0488 |

## Model: mlp

### Mean over layers

| predictor | n_layers | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|---|
| (a) router_in(L)->L (sanity ceiling) | 40 | 0.1616 | 0.1616 | 0.1616 | 0.1949 | 0.2452 |
| (b) layer_in(L-1)->L (one layer ahead) | 39 | 0.2230 | 0.2230 | 0.2230 | 0.2592 | 0.3151 |
| (c) layer_in(L-2)->L (two layers ahead) | 38 | 0.2166 | 0.2166 | 0.2166 | 0.2535 | 0.3079 |
| (d) prev-route+token->L (cheap features) | 39 | 0.1218 | 0.1218 | 0.1218 | 0.1577 | 0.2028 |

### Worst 5 layers by miss_red@6

**(a) router_in(L)->L (sanity ceiling)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.0208 | 0.0208 | 0.0208 | 0.0358 | 0.0521 |
| L13 | 0.0397 | 0.0397 | 0.0397 | 0.0456 | 0.0508 |
| L19 | 0.0410 | 0.0410 | 0.0410 | 0.0540 | 0.0957 |
| L12 | 0.0462 | 0.0462 | 0.0462 | 0.0618 | 0.0924 |
| L18 | 0.0482 | 0.0482 | 0.0482 | 0.0553 | 0.0892 |

**(b) layer_in(L-1)->L (one layer ahead)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.0260 | 0.0260 | 0.0260 | 0.0410 | 0.0573 |
| L13 | 0.0384 | 0.0384 | 0.0384 | 0.0475 | 0.0547 |
| L12 | 0.0391 | 0.0391 | 0.0391 | 0.0469 | 0.0879 |
| L18 | 0.0456 | 0.0456 | 0.0456 | 0.0521 | 0.0833 |
| L16 | 0.0495 | 0.0495 | 0.0495 | 0.0632 | 0.0866 |

**(c) layer_in(L-2)->L (two layers ahead)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.0299 | 0.0299 | 0.0299 | 0.0430 | 0.0586 |
| L12 | 0.0391 | 0.0391 | 0.0391 | 0.0449 | 0.0866 |
| L13 | 0.0397 | 0.0397 | 0.0397 | 0.0456 | 0.0540 |
| L18 | 0.0436 | 0.0436 | 0.0436 | 0.0469 | 0.0768 |
| L16 | 0.0456 | 0.0456 | 0.0456 | 0.0566 | 0.0794 |

**(d) prev-route+token->L (cheap features)**

| layer | precision@6 | recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|---|
| L14 | 0.0150 | 0.0150 | 0.0150 | 0.0358 | 0.0449 |
| L24 | 0.0189 | 0.0189 | 0.0189 | 0.1660 | 0.2559 |
| L12 | 0.0384 | 0.0384 | 0.0384 | 0.0645 | 0.0762 |
| L19 | 0.0384 | 0.0384 | 0.0384 | 0.0391 | 0.0560 |
| L13 | 0.0397 | 0.0397 | 0.0397 | 0.0482 | 0.0540 |

## Cross-model headline: predictor (b) layer_in(L-1)->L (one layer ahead)

| model | precision@6=recall@6 | miss_red@6 | miss_red@8 | miss_red@12 |
|---|---|---|---|---|
| ridge | 0.3565 | 0.3565 | 0.4063 | 0.4746 |
| logistic | 0.2224 | 0.2224 | 0.2538 | 0.2997 |
| mlp | 0.2230 | 0.2230 | 0.2592 | 0.3151 |

