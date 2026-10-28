# ML-1M benchmark results

Resumed from an epoch-boundary checkpoint. Original checkpoint has no RNG state; model/optimizer state and cumulative resources are retained, while stochastic streams restart.

Updated: 2026-10-27 06:20:19 PM PDT. Run status: complete; active model: None.

Seed 42; supplied configurations; full-catalog exact scoring.
Prefix and all-position supervision have different target exposure; compare matched protocols.
Best epoch is selected on the evaluation set, not an independent validation set.
Epoch numbers are zero-based. Training time includes evaluation and checkpoint saving.
MRR uses the repository's original rank cutoff of 2500.

| Model | Selection | Epoch | Epochs completed | HR@1 | HR@10 | HR@50 | HR@200 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LCAR-SASRec | final | 100 | 101 | 0.0238 | 0.1318 | 0.3280 | 0.5674 | 0.0622 |
| LCAR-SASRec | best_eval_hr10 | 99 | 101 | 0.0245 | 0.1343 | 0.3396 | 0.5795 | 0.0626 |
| LCAR-HSTU | final | 100 | 101 | 0.0323 | 0.1714 | 0.3801 | 0.6081 | 0.0792 |
| LCAR-HSTU | best_eval_hr10 | 94 | 101 | 0.0318 | 0.1742 | 0.3868 | 0.6159 | 0.0806 |

## HR@10 differences and resources

Differences are absolute reproduced-minus-paper values. Memory is peak allocated CUDA memory.

| Model | Final HR@10 difference | Best HR@10 difference | Training + evaluation minutes | Peak GiB |
|---|---:|---:|---:|---:|
| LCAR-SASRec | — | — | 22.20 | 3.73 |
| LCAR-HSTU | — | — | 166.73 | 10.68 |

## NDCG

| Model | Selection | NDCG@10 | NDCG@50 |
|---|---|---:|---:|
| LCAR-SASRec | final | 0.0689 | 0.1116 |
| LCAR-SASRec | best_eval_hr10 | 0.0694 | 0.1140 |
| LCAR-HSTU | final | 0.0910 | 0.1365 |
| LCAR-HSTU | best_eval_hr10 | 0.0925 | 0.1388 |

All paper targets and per-metric differences are included in comparison.csv.
Reference: https://arxiv.org/html/2407.15462v3#S4.T3

## LCAR comparison against paper Table 3

Paper rows use the published protocol; LCAR uses sampled-prefix supervision.
Each LCAR epoch presents one target per user, unlike all-position baseline training.
These are single-seed experiments; best_eval_hr10 selects on the evaluation set.

| Model | Selection | HR@1 | HR@10 | HR@50 | HR@200 | MRR |
|---|---|---:|---:|---:|---:|---:|
| SASRec | paper | 0.0610 | 0.2818 | 0.5470 | 0.7540 | 0.1352 |
| SASRec + MoL | paper | 0.0697 | 0.3036 | 0.5617 | 0.7667 | 0.1441 |
| HSTU | paper | 0.0750 | 0.3332 | 0.5956 | 0.7824 | 0.1579 |
| HSTU + MoL | paper | 0.0884 | 0.3465 | 0.6022 | 0.7935 | 0.1712 |
| LCAR-SASRec | final | 0.0238 | 0.1318 | 0.3280 | 0.5674 | 0.0622 |
| LCAR-SASRec | best_eval_hr10 | 0.0245 | 0.1343 | 0.3396 | 0.5795 | 0.0626 |
| LCAR-HSTU | final | 0.0323 | 0.1714 | 0.3801 | 0.6081 | 0.0792 |
| LCAR-HSTU | best_eval_hr10 | 0.0318 | 0.1742 | 0.3868 | 0.6159 | 0.0806 |

Differences below are LCAR minus the corresponding paper baseline, in percentage points.

| LCAR model | Selection | Paper baseline | ΔHR@1 | ΔHR@10 | ΔHR@50 | ΔHR@200 | ΔMRR |
|---|---|---|---:|---:|---:|---:|---:|
| LCAR-SASRec | final | SASRec | -3.72 | -15.00 | -21.90 | -18.66 | -7.30 |
| LCAR-SASRec | final | SASRec + MoL | -4.59 | -17.18 | -23.37 | -19.93 | -8.19 |
| LCAR-SASRec | best_eval_hr10 | SASRec | -3.65 | -14.75 | -20.74 | -17.45 | -7.26 |
| LCAR-SASRec | best_eval_hr10 | SASRec + MoL | -4.52 | -16.93 | -22.21 | -18.72 | -8.15 |
| LCAR-HSTU | final | HSTU | -4.27 | -16.18 | -21.55 | -17.43 | -7.87 |
| LCAR-HSTU | final | HSTU + MoL | -5.61 | -17.51 | -22.21 | -18.54 | -9.20 |
| LCAR-HSTU | best_eval_hr10 | HSTU | -4.32 | -15.90 | -20.88 | -16.65 | -7.73 |
| LCAR-HSTU | best_eval_hr10 | HSTU + MoL | -5.66 | -17.23 | -21.54 | -17.76 | -9.06 |