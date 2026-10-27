# ML-1M benchmark results

Seed 42; one RTX 5090; supplied configurations; full-catalog exact scoring.
Best epoch is selected on the evaluation set, not an independent validation set.
Epoch numbers are zero-based. Training time includes evaluation and checkpoint saving.
MRR uses the repository's original rank cutoff of 2500.

| Model | Selection | Epoch | Epochs completed | HR@1 | HR@10 | HR@50 | HR@200 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SASRec | paper | — | — | 0.0610 | 0.2818 | 0.5470 | 0.7540 | 0.1352 |
| SASRec | final | 100 | 101 | 0.0614 | 0.2750 | 0.5510 | 0.7538 | 0.1315 |
| SASRec | best_eval_hr10 | 89 | 101 | 0.0626 | 0.2755 | 0.5498 | 0.7526 | 0.1318 |
| SASRec + MoL | paper | — | — | 0.0697 | 0.3036 | 0.5617 | 0.7667 | 0.1441 |
| SASRec + MoL | final | 100 | 101 | 0.0722 | 0.2997 | 0.5684 | 0.7657 | 0.1454 |
| SASRec + MoL | best_eval_hr10 | 82 | 101 | 0.0735 | 0.3020 | 0.5657 | 0.7632 | 0.1477 |
| HSTU | paper | — | — | 0.0750 | 0.3332 | 0.5956 | 0.7824 | 0.1579 |
| HSTU | final | 100 | 101 | 0.0772 | 0.3280 | 0.6017 | 0.7922 | 0.1591 |
| HSTU | best_eval_hr10 | 98 | 101 | 0.0773 | 0.3290 | 0.5987 | 0.7917 | 0.1592 |
| HSTU + MoL | paper | — | — | 0.0884 | 0.3465 | 0.6022 | 0.7935 | 0.1712 |
| HSTU + MoL | final | 100 | 101 | 0.0805 | 0.3328 | 0.5944 | 0.7829 | 0.1617 |
| HSTU + MoL | best_eval_hr10 | 79 | 101 | 0.0818 | 0.3392 | 0.5970 | 0.7922 | 0.1637 |

## HR@10 differences and resources

Differences are absolute reproduced-minus-paper values. Memory is peak allocated CUDA memory.

| Model | Final HR@10 difference | Best HR@10 difference | Training + evaluation minutes | Peak GiB |
|---|---:|---:|---:|---:|
| SASRec | -0.0068 | -0.0063 | 1.83 | 2.67 |
| SASRec + MoL | -0.0039 | -0.0016 | 7.47 | 13.90 |
| HSTU | -0.0052 | -0.0042 | 4.72 | 3.94 |
| HSTU + MoL | -0.0137 | -0.0073 | 10.38 | 15.18 |

All paper targets and per-metric differences are included in comparison.csv.
Reference: https://arxiv.org/html/2407.15462v3#S4.T3
