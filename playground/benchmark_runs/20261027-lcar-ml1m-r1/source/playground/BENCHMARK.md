# ML-1M reproduction

Run from the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.9.0 fbgemm-gpu==1.4.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python gin-config==0.5.0 pandas==2.2.2 numpy==2.0.1 absl-py==2.1.0 tensorboard tqdm
.venv/bin/python playground/run_ml1m_benchmark.py
```

The runner uses GPU 0 and runs the following stages sequentially:

1. Verify jagged forward/backward results, metric calculations, seen-item filtering, causal attention, padding, and persistent-worker RNG equivalence.
2. Perform one real optimizer step for each of SASRec, SASRec+MoL, HSTU, and HSTU+MoL; require finite loss and gradients.
3. Evaluate the included November HSTU+MoL checkpoint on all 6,040 ML-1M users. Require HR@10 within 0.005 of the documented 0.3474.
4. Train each variant for 101 epochs with the supplied model and optimizer settings and seed 42.
5. Reload each model's best and final checkpoints, verify their metrics, and export comparison tables.

Each run creates `playground/benchmark_runs/<run-id>/`, containing:

- `manifest.json`: environment, data hashes, commands, status, and deviations.
- `requirements.lock.txt` and `source.patch`: installed package versions and tracked code changes.
- `logs/`: validation, training, and independent checkpoint-evaluation logs.
- Per-model folders: derived/operative Gin configs, smoke results, full per-epoch metrics, and `best.pt`/`last.pt` checkpoints.
- `comparison.csv` and `comparison.md`: final and best evaluation epoch metrics, paper targets, differences, duration, and peak allocated CUDA memory. Written after each completed model, and also on failure.

To regenerate an interim report from completed epochs while training is running:

```bash
.venv/bin/python playground/run_ml1m_benchmark.py --report-only --run-dir playground/benchmark_runs/<run-id>
```

The output records the number of completed epochs; a partial run's “final” row is its latest recorded epoch and must not be treated as a completed 101-epoch result. Run directories are never overwritten, and interrupted runs are not automatically resumed.

## Interpretation and runtime changes

- Reference: [RAILS Table 3](https://arxiv.org/html/2407.15462v3#S4.T3), rather than the older README table.
- Evaluation uses the full item catalog, filters seen items, and retains the original MRR cutoff of 2500.
- Best HR@10 is selected on the existing evaluation split; there is no separate validation split. Report it separately from the final epoch.
- This is a single-seed comparison, not a statistical significance study.
- The RTX 5090 requires a newer CUDA/PyTorch/FBGEMM environment than the repository's historical pins. The validation checks analytic jagged gradients, including when FBGEMM emits autograd-registration warnings.
- Derived configs use two persistent workers per loader and prefetch factor 4 to avoid repeated worker startup and reduce host memory usage. A loader wrapper preserves the parent RNG draw that a fresh iterator would consume. ML-1M dataset loading is deterministic with the configured sampling ratio of 1.
- Model dimensions, batch size, negatives, dropout, learning rate, weight decay, and training precision remain those in the supplied ML-1M configs.

The checkpoint evaluator now supports SASRec and `--metrics_output=<path.json>`. Training gains optional `train_fn.benchmark_output_dir` and `train_fn.smoke_steps`; their defaults preserve normal training behavior. Benchmark mode stores only best/latest checkpoints in its own directory.
