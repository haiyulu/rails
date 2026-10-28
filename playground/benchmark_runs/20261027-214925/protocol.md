# Full ML-1M benchmark protocol

Fresh training runs: SASRec, SASRec + MoL, SASRec + LCAR v4; 101 epochs per model, seed 42, one NVIDIA RTX 5090. Existing experiment artifacts are not reused or overwritten.

## Published baseline reproduction

Reference: [RAILS Table 3](https://arxiv.org/html/2407.15462v3#S4.T3).

| Baseline | HR@1 | HR@10 | HR@50 | HR@200 | MRR |
|---|---:|---:|---:|---:|---:|
| SASRec | 0.0610 | 0.2818 | 0.5470 | 0.7540 | 0.1352 |
| SASRec + MoL | 0.0697 | 0.3036 | 0.5617 | 0.7667 | 0.1441 |

The released baseline configurations retain all-position next-item supervision, 128 local sampled negatives, batch size 128, sequence limit 200, original scorer normalization/temperature, optimizer and regularizers. The final epoch is the primary reproduction check; best evaluation-HR@10 checkpoints are also reported, without substituting them for final results.

Before training, declare an absolute band of 0.01 (one percentage point) for EACH of the five Table 3 metrics. SASRec + MoL must additionally improve HR@10 and HR@50 over SASRec. This is a practical single-seed reproduction criterion, not a confidence interval reported by the paper. The runner checks independently reloaded final checkpoints and stops on a failed criterion; it will not silently relabel a failed reproduction as consistent.

## LCAR and comparability

LCAR v4 uses four item experts, 128 atoms per expert, at most 16 active atoms per expert, 16 selected original events per atom, 32 recent originals, the global-history token and four source embeddings. It retains the adopted full-prefix next-item policy: one target per user per epoch. Baselines use all-position targets, so equal epoch counts do not imply equal target exposure. Per-model exposures and optimizer steps are recorded. Differences between these rows are not a scorer-only ablation.

## Evaluation, environment and artifacts

Every epoch evaluates all 6,040 users with exact full-catalog scoring and seen-item filtering. The released loader uses all 3,883 movie metadata IDs as candidates before filtering; Table 2 reports 3,649 items, and all processed histories contain 3,706 distinct item IDs. We retain the released-code catalog rather than altering it to fit results. MRR uses the repository cutoff of 2,500. There is no independent validation split; best checkpoint selection uses the evaluation set.

Hardware/software differ from the paper: RTX 5090 and the locally verified modern PyTorch/FBGEMM environment; TF32 is enabled as in the supplied configs. Two persistent data workers with prefetch factor 4 preserve the loader RNG convention. The manifest includes environment information, data hashes, configs, source patch and source snapshots. All models are trained sequentially on the same idle GPU.

Finite-loss/gradient smoke runs precede training. Best and final checkpoints are independently reloaded and re-evaluated; recorded metrics must agree within 1e-5. Live results are in comparison.md/comparison.csv, the reproduction policy and pass/fail evidence in paper_consistency.json, and process details in manifest.json/logs/.
