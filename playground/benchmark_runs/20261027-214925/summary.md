# Full ML-1M benchmark: SASRec, SASRec + MoL, SASRec + LCAR

Completed: fresh seed-42 runs, 101 epochs per model, one NVIDIA RTX 5090. Primary results below use the fixed final epoch (index 100).

## Final retrieval quality

| Model | HR@1 | HR@10 | HR@50 | HR@200 | MRR |
|---|---:|---:|---:|---:|---:|
| SASRec | 0.0614 | 0.2750 | 0.5510 | 0.7538 | 0.1315 |
| SASRec + MoL | 0.0722 | 0.2997 | 0.5684 | 0.7657 | 0.1454 |
| SASRec + LCAR v4 | 0.0078 | 0.0389 | 0.1017 | 0.2522 | 0.0208 |

## Table 3 reproduction

Reference: [RAILS Table 3](https://arxiv.org/html/2407.15462v3#S4.T3). Before training, the acceptance band was declared as ±1.00 percentage point for each final metric, plus positive MoL gains at HR@10 and HR@50. This is a practical single-seed check, not a paper confidence interval.

| Baseline | Metric | Paper | Reproduced | Difference (pp) |
|---|---|---:|---:|---:|
| SASRec | hr@1 | 0.0610 | 0.0614 | +0.042 |
| SASRec | hr@10 | 0.2818 | 0.2750 | -0.680 |
| SASRec | hr@50 | 0.5470 | 0.5510 | +0.399 |
| SASRec | hr@200 | 0.7540 | 0.7538 | -0.019 |
| SASRec | mrr | 0.1352 | 0.1315 | -0.374 |
| SASRec + MoL | hr@1 | 0.0697 | 0.0722 | +0.249 |
| SASRec + MoL | hr@10 | 0.3036 | 0.2997 | -0.393 |
| SASRec + MoL | hr@50 | 0.5617 | 0.5684 | +0.668 |
| SASRec + MoL | hr@200 | 0.7667 | 0.7657 | -0.097 |
| SASRec + MoL | mrr | 0.1441 | 0.1454 | +0.130 |

**PASS:** both baselines meet every declared metric band. MoL also improves both primary hit-rate cutoffs.

Reproduced MoL gains: HR@10 +2.467 pp and HR@50 +1.738 pp. Paper gains are +2.180 pp and +1.470 pp respectively.

## Resource use and supervision

| Model | Supervision | Target exposures | Optimizer steps | Parameters | Train + eval (minutes) | Peak allocated GPU GiB |
|---|---|---:|---:|---:|---:|---:|
| SASRec | all_positions | 65,861,898 | 4,848 | 238,800 | 1.82 | 2.67 |
| SASRec + MoL | all_positions | 65,861,898 | 4,848 | 950,128 | 7.40 | 13.90 |
| SASRec + LCAR v4 | full_prefix | 610,040 | 4,848 | 322,600 | 29.53 | 3.88 |

The baselines retain the released all-position training protocol. LCAR retains its adopted full-prefix policy (one target per user per epoch). Both perform 4,848 optimizer steps, but LCAR sees about 108 times fewer supervised targets. These rows are an evaluation of the configured systems, not a scorer-only ablation. Lower LCAR quality cannot by itself isolate architecture effects from supervision and other configuration differences.

Recorded time is training-loop wall time including epoch/intra-epoch evaluation, checkpoint writes, and first-pass compilation; startup, smoke runs and separate checkpoint re-evaluation are excluded. Peak memory is allocated CUDA memory. These timings are not isolated serving-latency measurements.

## Best evaluation-HR@10 checkpoints

| Model | Epoch (one-based) | HR@10 | HR@50 | MRR |
|---|---:|---:|---:|---:|
| SASRec | 90 | 0.2755 | 0.5498 | 0.1318 |
| SASRec + MoL | 83 | 0.3020 | 0.5657 | 0.1477 |
| SASRec + LCAR v4 | 1 | 0.0596 | 0.1430 | 0.0305 |

Best checkpoints are selected on the evaluation set, without an independent validation split. They do not replace the fixed-final-epoch reproduction checks.

## Protocol and verification

- Same chronological ML-1M split and exact full-catalog evaluation for all models; 6,040 evaluation users, seen-item filtering, MRR rank cutoff 2,500.
- Batch 128; history limit 200; 128 local sampled negatives; original per-model normalization, temperatures and auxiliary losses retained.
- Released-code catalog uses 3,883 movie metadata IDs; paper Table 2 reports 3,649 items. The catalog was not changed to fit results.
- LCAR v4 uses four experts, 128 atoms/expert, at most 16 active atoms/expert, 16 selected original events/atom, 32 recent originals, global token and four source types.
- Environment differs from the paper: modern locally verified PyTorch/FBGEMM on RTX 5090; source/environment snapshots, operative configs and data hashes are retained.
- Both best and final checkpoints for every model were independently reloaded and re-evaluated. All seven comparison metrics agree with saved epoch records within 1e-5.
- All runs used the original seed without resuming or changing hyperparameters during training.

## Artifacts

- [Detailed comparison](comparison.md) and [comparison CSV](comparison.csv)
- [Reproduction check evidence](paper_consistency.json)
- [Predeclared protocol](protocol.md) and [run manifest](manifest.json)
- [Per-epoch learning curves CSV](learning_curves.csv)
- Per-model folders contain configs, operative Gin settings, best.pt, last.pt and epochs.jsonl; logs/ contains smoke, training and checkpoint evaluation logs.
