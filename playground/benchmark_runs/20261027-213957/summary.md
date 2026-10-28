# SASRec vs SASRec + LCAR: short ML-1M benchmark

Completed on one NVIDIA RTX 5090. The table uses the final checkpoint after exactly three epochs for each model.

| Model | HR@10 | HR@50 | NDCG@10 | MRR | Training + evaluation (s) | Peak allocated GPU GiB | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| SASRec | 0.0333 | 0.0988 | 0.0175 | 0.0181 | 18.5 | 0.28 | 238,800 |
| SASRec + LCAR v4 | 0.0586 | 0.1523 | 0.0321 | 0.0312 | 66.6 | 3.88 | 322,600 |

LCAR increases final HR@10 by 2.53 percentage points (76.1% relative). Recorded training-plus-evaluation wall time is 3.61 times SASRec.

## Protocol

- ML-1M chronological repository split; 6,040 training prefixes and 6,040 evaluation users; 3,883 catalog candidates before seen-item filtering.
- Full-prefix next-item training for both models: 18,120 target exposures and 144 optimizer steps each. This differs from the original all-position SASRec baseline protocol.
- Seed 42; batch size 128; history limit 200; 128 local sampled negatives; AdamW learning rate 0.001, weight decay 0; dropout 0.2; user/item L2 normalization; TF32 enabled.
- Model-specific default loss temperatures retained: SASRec 0.05, LCAR 1.0. Initializations and stochastic streams can differ across architectures.
- SASRec backbone: two blocks, one head, width 50, FFN width 50. LCAR v4 adds four experts, 128 atoms/expert, up to 16 active atoms/expert, 16 selected events/atom, 32 recent events, global token and four source types.
- Full-catalog exact scoring and seen-item filtering for both models. MRR follows the repository rank cutoff of 2,500.
- Time is the recorded training-loop wall time including evaluation, checkpoint writes and first-pass compilation; it excludes startup, smoke runs and separate checkpoint verification. Memory is peak allocated CUDA memory.
- Single-seed early-training comparison. Three epochs do not establish converged quality or a paper-baseline improvement. Best checkpoints are selected on the evaluation set, without independent validation.

## Verification and artifacts

Both full-size smoke runs had finite losses/gradients. Best and final checkpoints for both models were reloaded and evaluated over the full evaluation set; all recorded comparison metrics matched within 1e-5.

- [SASRec configuration](sasrec/sasrec-sampled-softmax-n128-final.gin)
- [SASRec + LCAR configuration](sasrec_lcar/sasrec-lcar.gin)
- [Full results and per-epoch selections](comparison.md)
- [Machine-readable results](comparison.csv)
- [Run manifest](manifest.json), source snapshot and environment lock alongside this report.
