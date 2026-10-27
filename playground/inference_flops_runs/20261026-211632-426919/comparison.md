# Inference FLOPs comparison

ML-1M checkpoints; B=1 user, L=200 valid history items, T=211 padded positions, N=3,883 shared candidates.
FP32 eager inference; no incremental history cache. One multiply–accumulate = 2 FLOPs.
Totals count logical add/subtract/multiply/divide arithmetic. Special functions are reported separately below.

## Headline results

| Model | User MFLOPs | Candidate preparation MFLOPs | Strict scoring FLOPs/pair | All-candidate scoring MFLOPs | Entire cached MFLOPs | Entire uncached MFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| SASRec | 31.399121 | 0.578567 | 100 | 0.388300 | 31.787421 | 32.365988 |
| SASRec + MoL | 32.007067 | 185.871444 | 21,117 | 81.997311 | 114.004378 | 299.875822 |
| HSTU | 116.266090 | 0.578567 | 100 | 0.388300 | 116.654390 | 117.232957 |
| HSTU + MoL | 116.874036 | 185.871444 | 21,117 | 81.997311 | 198.871347 | 384.742791 |

Cached entire model = user + strict scoring. Uncached entire model additionally prepares the shared catalog once.
Query/component projections and query-only gates are charged to the user. Candidate projections and item-only gates are charged to candidate preparation.
Ranking, filtering, input transfers, and cache construction infrastructure are excluded.

## Ratios versus SASRec

| Model | User arithmetic | Strict scoring/pair | Entire cached | Entire uncached |
|---|---:|---:|---:|---:|
| SASRec | 1.000× | 1.000× | 1.000× | 1.000× |
| SASRec + MoL | 1.019× | 211.170× | 3.586× | 9.265× |
| HSTU | 3.703× | 1.000× | 3.670× | 3.622× |
| HSTU + MoL | 3.722× | 211.170× | 6.256× | 11.887× |

## Special-function evaluations (not included in arithmetic FLOPs)

| Model | User | Candidate preparation | Strict scoring, all candidates |
|---|---|---|---|
| SASRec | exp: 89,042, rsqrt: 844, sqrt: 211 | sqrt: 3,883 | none |
| SASRec + MoL | exp: 89,042, rsqrt: 1,055, sigmoid: 640, sqrt: 8 | sigmoid: 497,024, sqrt: 15,532 | exp: 124,256, sigmoid: 621,280 |
| HSTU | log: 356,168, rsqrt: 3,200, sigmoid: 1,032,336, sqrt: 211 | sqrt: 3,883 | none |
| HSTU + MoL | log: 356,168, rsqrt: 3,411, sigmoid: 1,032,976, sqrt: 8 | sigmoid: 497,024, sqrt: 15,532 | exp: 124,256, sigmoid: 621,280 |

## Validation and interpretation

- 128 cases validated against the original inference path; independent cached/uncached counting reconciled exactly.
- Strict scorers made zero calls to history encoding, query/item projections, or query-only/item-only gate MLPs.
- Scoring matrix FLOPs/pair: baseline 100; MoL 20,480 (4,096 component contractions + 16,384 joint gate MLP).
- MoL adds bias/activation/scaling/softmax/renormalization/reduction arithmetic; the configured inference combiner renormalizes even though dropout is disabled.
- SASRec and HSTU have identical baseline scoring cost; both MoL variants have identical strict scoring cost.
- Input width is fixed at 211. Causal masks do not eliminate the executed dense attention products. HSTU pointwise work follows valid lengths, while its attention remains padded.
- These are logical-operation counts, not GPU instructions or measured latency. Special-function costs are not priced as hardware-equivalent FLOPs.
- See sweep.csv, operation_ledger.json, validation.json, coverage.md, and manifest.json for details.
