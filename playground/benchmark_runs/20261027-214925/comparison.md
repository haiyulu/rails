# ML-1M benchmark results

Updated: 2026-10-27 10:31:41 PM PDT. Run status: complete; active model: None.

Seed 42; supplied configurations; full-catalog exact scoring.
Scheduled training budget: 101 epochs per model.
Prefix and all-position supervision have different target exposure; compare matched protocols.
Best epoch is selected on the evaluation set, not an independent validation set.
Epoch numbers are zero-based. Training time includes evaluation and checkpoint saving.
MRR uses the repository's original rank cutoff of 2500.

| Model | Selection | Epoch | Epochs completed | HR@1 | HR@10 | HR@50 | HR@200 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LCAR-SASRec | final | 100 | 101 | 0.0078 | 0.0389 | 0.1017 | 0.2522 | 0.0208 |
| LCAR-SASRec | best_eval_hr10 | 0 | 101 | 0.0129 | 0.0596 | 0.1430 | 0.3046 | 0.0305 |
| SASRec | paper | — | — | 0.0610 | 0.2818 | 0.5470 | 0.7540 | 0.1352 |
| SASRec | final | 100 | 101 | 0.0614 | 0.2750 | 0.5510 | 0.7538 | 0.1315 |
| SASRec | best_eval_hr10 | 89 | 101 | 0.0626 | 0.2755 | 0.5498 | 0.7526 | 0.1318 |
| SASRec + MoL | paper | — | — | 0.0697 | 0.3036 | 0.5617 | 0.7667 | 0.1441 |
| SASRec + MoL | final | 100 | 101 | 0.0722 | 0.2997 | 0.5684 | 0.7657 | 0.1454 |
| SASRec + MoL | best_eval_hr10 | 82 | 101 | 0.0735 | 0.3020 | 0.5657 | 0.7632 | 0.1477 |

## HR@10 differences and resources

Differences are absolute reproduced-minus-paper values. Memory is peak allocated CUDA memory.

| Model | Final HR@10 difference | Best HR@10 difference | Training + evaluation minutes | Peak GiB |
|---|---:|---:|---:|---:|
| LCAR-SASRec | — | — | 29.53 | 3.88 |
| SASRec | -0.0068 | -0.0063 | 1.82 | 2.67 |
| SASRec + MoL | -0.0039 | -0.0016 | 7.40 | 13.90 |

## NDCG

| Model | Selection | NDCG@10 | NDCG@50 |
|---|---|---:|---:|
| LCAR-SASRec | final | 0.0210 | 0.0344 |
| LCAR-SASRec | best_eval_hr10 | 0.0322 | 0.0504 |
| SASRec | final | 0.1528 | 0.2137 |
| SASRec | best_eval_hr10 | 0.1531 | 0.2137 |
| SASRec + MoL | final | 0.1696 | 0.2289 |
| SASRec + MoL | best_eval_hr10 | 0.1718 | 0.2305 |

All paper targets and per-metric differences are included in comparison.csv.
Reference: https://arxiv.org/html/2407.15462v3#S4.T3

## Table 3 reproduction check

Declared band: ±1.00 percentage points for each final HR@1, HR@10, HR@50, HR@200 and MRR. This is a practical check, not a statistical confidence interval.
SASRec + MoL must also improve HR@10 and HR@50 over SASRec. Baselines retain their released all-position configurations; LCAR uses full-prefix supervision.

| Model | Final-check status | Largest absolute metric difference (pp) |
|---|---|---:|
| SASRec | PASS | 0.680 |
| SASRec + MoL | PASS | 0.668 |

MoL gain check: PASS; ΔHR@10=+2.467 pp, ΔHR@50=+1.738 pp.

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
| LCAR-SASRec | final | 0.0078 | 0.0389 | 0.1017 | 0.2522 | 0.0208 |
| LCAR-SASRec | best_eval_hr10 | 0.0129 | 0.0596 | 0.1430 | 0.3046 | 0.0305 |

Differences below are LCAR minus the corresponding paper baseline, in percentage points.

| LCAR model | Selection | Paper baseline | ΔHR@1 | ΔHR@10 | ΔHR@50 | ΔHR@200 | ΔMRR |
|---|---|---|---:|---:|---:|---:|---:|
| LCAR-SASRec | final | SASRec | -5.32 | -24.29 | -44.53 | -50.18 | -11.44 |
| LCAR-SASRec | final | SASRec + MoL | -6.19 | -26.47 | -46.00 | -51.45 | -12.33 |
| LCAR-SASRec | best_eval_hr10 | SASRec | -4.81 | -22.22 | -40.40 | -44.94 | -10.47 |
| LCAR-SASRec | best_eval_hr10 | SASRec + MoL | -5.68 | -24.40 | -41.87 | -46.21 | -11.36 |