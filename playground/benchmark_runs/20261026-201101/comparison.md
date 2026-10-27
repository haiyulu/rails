# ML-1M benchmark results

Seed 42; one RTX 5090; supplied configurations; full-catalog exact scoring.
Best epoch is selected on the evaluation set, not an independent validation set.
Epoch numbers are zero-based. Training time includes evaluation and checkpoint saving.
MRR uses the repository's original rank cutoff of 2500.

| Model | Selection | Epoch | Epochs completed | HR@1 | HR@10 | HR@50 | HR@200 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sasrec | final | 1 | 2 | 0.0043 | 0.0366 | 0.1288 | 0.3328 | 0.0186 |
| sasrec | best_eval_hr10 | 1 | 2 | 0.0043 | 0.0366 | 0.1288 | 0.3328 | 0.0186 |

Paper targets and absolute differences are included in comparison.csv.
Reference: https://arxiv.org/html/2407.15462v3#S4.T3
