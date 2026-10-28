# Reproduce the ML-1M model comparison on the RTX 5090

## Target

Train **SASRec, SASRec+MoL, HSTU, and HSTU+MoL** from scratch on MovieLens-1M, using seed 42 and the supplied configurations.

Use **Table 3 of the paper**, rather than the older README results, as the reference:

| Model | HR@1 | HR@10 | HR@50 | HR@200 | MRR |
|---|---:|---:|---:|---:|---:|
| SASRec | .0610 | .2818 | .5470 | .7540 | .1352 |
| SASRec+MoL | .0697 | .3036 | .5617 | .7667 | .1441 |
| HSTU | .0750 | .3332 | .5956 | .7824 | .1579 |
| HSTU+MoL | .0884 | .3465 | .6022 | .7935 | .1712 |

Source: [RAILS paper, Table 3](https://arxiv.org/html/2407.15462v3#S4.T3). Exact numerical equality is not guaranteed across software versions and hardware.

## Environment and preflight

- Create an isolated Python 3.12 environment with `uv venv --python 3.12 .venv`. Use uv for environment creation and dependency installation. Start with PyTorch 2.9.0 and FBGEMM-GPU 1.4.0 CUDA 12.8 wheels; this pairing is listed in the [official compatibility matrix](https://docs.pytorch.org/FBGEMM/general/Releases). Validate the required operators on the actual GPU before training.
- Install dependencies with `uv pip install --python .venv/bin/python`. Install Gin, pandas, NumPy, absl-py, and TensorBoard; freeze the resolved environment with `uv pip freeze --python .venv/bin/python` after successful checks. FAISS is unnecessary for these exact-scoring accuracy comparisons.
- Verify CUDA matrix multiplication, compiled MoL functions, and forward/backward execution of the FBGEMM jagged operators. Stop before long training if compatibility fails; do not silently substitute operators.
- Reuse the existing prepared ML-1M data. Record data hashes, repository revision, GPU/driver information, and resolved configurations.
- Evaluate the included November HSTU+MoL checkpoint first. Its source-documented HR@10 is approximately **0.3474**, a separate sanity reference from the paper's **0.3465**.

## Training and evaluation

- Use the four existing files under `configs/ml-1m/`. Preserve sequence length 200, embedding width 50, batch size 128, 128 sampled negatives, optimizer settings, normalization, dropout, and TF32 settings.
- Train sequentially on `CUDA_VISIBLE_DEVICES=0`, using the existing `train.py` entry point and 101 epochs per configuration.
- Add a small benchmark runner that creates derived Gin configs with a unique run identifier and checkpoint saving every epoch. Preserve the supplied configs and pretrained checkpoints.
- For this host, use two persistent data-loading workers and prefetch factor 4 in the derived configs to avoid repeated worker startup. Preserve the parent RNG draws of the original iterator and verify sample-order/RNG equivalence; keep all model and optimizer settings unchanged.
- Record full evaluation metrics each epoch. Retain the final checkpoint and the checkpoint with highest HR@10; break ties by choosing the earlier epoch and delete other newly generated checkpoints after recording their metrics.
- Report **final-epoch results** and **best observed HR@10 epoch results** separately. The existing dataset setup holds out the final interaction and provides no independent validation split, so describe the latter explicitly as selection on the evaluation set.
- For checkpoint evaluation, use `MIPSBruteForceTopK` for dot-product models and `MoLBruteForceTopK` for MoL models. Set timing and brute-force-reference evaluation flags to false, use all users, and retain configured precision.
- Extend the evaluator's output to include HR@200 and MRR, which are already computed internally but omitted from its CSV summary.

Example training entry point, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  --gin_config_file=configs/ml-1m/hstu-mol-sampled-softmax-n128-8x4x64-rails-final.gin \
  --master_port=12355
```

The runner will use derived configs to enable checkpoint retention and unique run naming.

## Validation and deliverables

- Run one forward/backward optimizer step for all four variants before full training; require finite loss and gradients.
- Check that causal supervision predicts the next item, padding is excluded, and previously seen items are filtered during evaluation.
- Verify metric aggregation against a small, manually ranked example, including HR@200 and MRR.
- Confirm saved-checkpoint evaluation matches the corresponding training evaluation within numerical tolerance.
- Produce a CSV and Markdown comparison containing paper targets, reproduced metrics, differences, selected epochs, training duration, and peak GPU memory.
- Preserve complete logs and a machine-readable run manifest. Report discrepancies without changing hyperparameters to chase the paper's numbers.

Initial scope excludes QA, retrieval-latency benchmarks, MI ablations, and additional seeds. Extend the validated runner to ML-20M and Books afterward; Books requires a separate decision about preserving the original two-GPU training batch.
