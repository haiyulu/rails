# ML-1M benchmark results

Updated: 2026-10-27 09:43:44 PM PDT. Run status: training; active model: sasrec_lcar.

**Short-run comparison: early-training results, without a convergence claim. Paper rows use a different training budget.**

Seed 42; supplied configurations; full-catalog exact scoring.
Scheduled training budget: 3 epochs per model.
Prefix and all-position supervision have different target exposure; compare matched protocols.
Best epoch is selected on the evaluation set, not an independent validation set.
Epoch numbers are zero-based. Training time includes evaluation and checkpoint saving.
MRR uses the repository's original rank cutoff of 2500.

| Model | Selection | Epoch | Epochs completed | HR@1 | HR@10 | HR@50 | HR@200 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LCAR-SASRec | final | 2 | 3 | 0.0134 | 0.0586 | 0.1523 | 0.3245 | 0.0312 |
| LCAR-SASRec | best_eval_hr10 | 0 | 3 | 0.0129 | 0.0596 | 0.1430 | 0.3046 | 0.0305 |
| SASRec | paper | — | — | 0.0610 | 0.2818 | 0.5470 | 0.7540 | 0.1352 |
| SASRec | final | 2 | 3 | 0.0060 | 0.0333 | 0.0988 | 0.2434 | 0.0181 |
| SASRec | best_eval_hr10 | 2 | 3 | 0.0060 | 0.0333 | 0.0988 | 0.2434 | 0.0181 |

## HR@10 differences and resources

Differences are absolute reproduced-minus-paper values. Memory is peak allocated CUDA memory.

| Model | Final HR@10 difference | Best HR@10 difference | Training + evaluation minutes | Peak GiB |
|---|---:|---:|---:|---:|
| LCAR-SASRec | — | — | 1.11 | 3.88 |
| SASRec | -0.2485 | -0.2485 | 0.31 | 0.28 |

## NDCG

| Model | Selection | NDCG@10 | NDCG@50 |
|---|---|---:|---:|
| LCAR-SASRec | final | 0.0321 | 0.0524 |
| LCAR-SASRec | best_eval_hr10 | 0.0322 | 0.0504 |
| SASRec | final | 0.0175 | 0.0315 |
| SASRec | best_eval_hr10 | 0.0175 | 0.0315 |

All paper targets and per-metric differences are included in comparison.csv.
Reference: https://arxiv.org/html/2407.15462v3#S4.T3

## LCAR comparison against paper Table 3

Paper rows use the published protocol; each run's supervision policy is recorded above and in comparison.csv.
Each LCAR epoch presents one target per user, unlike all-position baseline training.
These are single-seed experiments; best_eval_hr10 selects on the evaluation set.

| Model | Selection | HR@1 | HR@10 | HR@50 | HR@200 | MRR |
|---|---|---:|---:|---:|---:|---:|
| SASRec | paper | 0.0610 | 0.2818 | 0.5470 | 0.7540 | 0.1352 |
| SASRec + MoL | paper | 0.0697 | 0.3036 | 0.5617 | 0.7667 | 0.1441 |
| HSTU | paper | 0.0750 | 0.3332 | 0.5956 | 0.7824 | 0.1579 |
| HSTU + MoL | paper | 0.0884 | 0.3465 | 0.6022 | 0.7935 | 0.1712 |
| LCAR-SASRec | final | 0.0134 | 0.0586 | 0.1523 | 0.3245 | 0.0312 |
| LCAR-SASRec | best_eval_hr10 | 0.0129 | 0.0596 | 0.1430 | 0.3046 | 0.0305 |

Differences below are LCAR minus the corresponding paper baseline, in percentage points.

| LCAR model | Selection | Paper baseline | ΔHR@1 | ΔHR@10 | ΔHR@50 | ΔHR@200 | ΔMRR |
|---|---|---|---:|---:|---:|---:|---:|
| LCAR-SASRec | final | SASRec | -4.76 | -22.32 | -39.47 | -42.95 | -10.40 |
| LCAR-SASRec | final | SASRec + MoL | -5.63 | -24.50 | -40.94 | -44.22 | -11.29 |
| LCAR-SASRec | best_eval_hr10 | SASRec | -4.81 | -22.22 | -40.40 | -44.94 | -10.47 |
| LCAR-SASRec | best_eval_hr10 | SASRec + MoL | -5.68 | -24.40 | -41.87 | -46.21 | -11.36 |