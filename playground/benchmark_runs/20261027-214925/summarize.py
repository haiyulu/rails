"""Summarize the completed benchmark from saved epoch and checkpoint metrics."""
import csv
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
NAMES = {'sasrec': 'SASRec', 'sasrec_mol': 'SASRec + MoL', 'sasrec_lcar': 'SASRec + LCAR v4'}
METRICS = ('hr@1', 'hr@10', 'hr@50', 'hr@200', 'mrr')
manifest = json.loads((RUN / 'manifest.json').read_text())
assert manifest['status'] == 'complete', 'Wait for all training and checkpoint checks to finish'
policy = json.loads((RUN / 'paper_consistency.json').read_text())
assert all(row['passed'] for row in policy['results'].values())
assert policy['sasrec_mol_gain']['passed']
records = {name: [json.loads(line) for line in (RUN / name / 'epochs.jsonl').read_text().splitlines() if line.strip()] for name in NAMES}
final = {name: rows[-1] for name, rows in records.items()}
for name, rows in records.items():
    assert [row['epoch'] for row in rows] == list(range(101))
    for choice in ('best', 'last'):
        expected = max(rows, key=lambda r: r['metrics']['hr@10']) if choice == 'best' else rows[-1]
        checked = json.loads((RUN / 'logs' / f'{name}_{choice}_metrics.json').read_text())
        assert all(abs(checked['metrics'][key] - expected['metrics'][key]) <= 1e-5 for key in METRICS + ('ndcg@10', 'ndcg@50'))

lines = ['# Full ML-1M benchmark: SASRec, SASRec + MoL, SASRec + LCAR', '',
         'Completed: fresh seed-42 runs, 101 epochs per model, one NVIDIA RTX 5090. Primary results below use the fixed final epoch (index 100).', '',
         '## Final retrieval quality', '',
         '| Model | HR@1 | HR@10 | HR@50 | HR@200 | MRR |', '|---|---:|---:|---:|---:|---:|']
for name, label in NAMES.items():
    lines.append('| ' + ' | '.join([label] + [f'{final[name]["metrics"][key]:.4f}' for key in METRICS]) + ' |')
lines += ['', '## Table 3 reproduction', '',
          'Reference: [RAILS Table 3](https://arxiv.org/html/2407.15462v3#S4.T3). Before training, the acceptance band was declared as ±1.00 percentage point for each final metric, plus positive MoL gains at HR@10 and HR@50. This is a practical single-seed check, not a paper confidence interval.', '',
          '| Baseline | Metric | Paper | Reproduced | Difference (pp) |', '|---|---|---:|---:|---:|']
for name in ('sasrec', 'sasrec_mol'):
    check = policy['results'][name]
    for key in METRICS:
        lines.append(f'| {NAMES[name]} | {key} | {check["targets"][key]:.4f} | {check["metrics"][key]:.4f} | {100*check["deltas"][key]:+.3f} |')
lines += ['', '**PASS:** both baselines meet every declared metric band. MoL also improves both primary hit-rate cutoffs.', '',
          f'Reproduced MoL gains: HR@10 {100*policy["sasrec_mol_gain"]["gains"]["hr@10"]:+.3f} pp and HR@50 {100*policy["sasrec_mol_gain"]["gains"]["hr@50"]:+.3f} pp. Paper gains are +2.180 pp and +1.470 pp respectively.', '',
          '## Resource use and supervision', '',
          '| Model | Supervision | Target exposures | Optimizer steps | Parameters | Train + eval (minutes) | Peak allocated GPU GiB |', '|---|---|---:|---:|---:|---:|---:|']
for name, label in NAMES.items():
    row = final[name]
    lines.append(f'| {label} | {row["supervision_mode"]} | {row["target_exposures"]:,} | {row["steps"]:,} | {row["parameter_count"]:,} | {row["elapsed_seconds"]/60:.2f} | {row["peak_memory_bytes"]/2**30:.2f} |')
lines += ['', 'The baselines retain the released all-position training protocol. LCAR retains its adopted full-prefix policy (one target per user per epoch). Both perform 4,848 optimizer steps, but LCAR sees about 108 times fewer supervised targets. These rows are an evaluation of the configured systems, not a scorer-only ablation. Lower LCAR quality cannot by itself isolate architecture effects from supervision and other configuration differences.', '',
          'Recorded time is training-loop wall time including epoch/intra-epoch evaluation, checkpoint writes, and first-pass compilation; startup, smoke runs and separate checkpoint re-evaluation are excluded. Peak memory is allocated CUDA memory. These timings are not isolated serving-latency measurements.', '',
          '## Best evaluation-HR@10 checkpoints', '',
          '| Model | Epoch (one-based) | HR@10 | HR@50 | MRR |', '|---|---:|---:|---:|---:|']
for name, label in NAMES.items():
    best = max(records[name], key=lambda row: row['metrics']['hr@10'])
    lines.append('| ' + ' | '.join([label, str(best['epoch']+1)] + [f'{best["metrics"][key]:.4f}' for key in ('hr@10', 'hr@50', 'mrr')]) + ' |')
lines += ['', 'Best checkpoints are selected on the evaluation set, without an independent validation split. They do not replace the fixed-final-epoch reproduction checks.', '',
          '## Protocol and verification', '',
          '- Same chronological ML-1M split and exact full-catalog evaluation for all models; 6,040 evaluation users, seen-item filtering, MRR rank cutoff 2,500.',
          '- Batch 128; history limit 200; 128 local sampled negatives; original per-model normalization, temperatures and auxiliary losses retained.',
          '- Released-code catalog uses 3,883 movie metadata IDs; paper Table 2 reports 3,649 items. The catalog was not changed to fit results.',
          '- LCAR v4 uses four experts, 128 atoms/expert, at most 16 active atoms/expert, 16 selected original events/atom, 32 recent originals, global token and four source types.',
          '- Environment differs from the paper: modern locally verified PyTorch/FBGEMM on RTX 5090; source/environment snapshots, operative configs and data hashes are retained.',
          '- Both best and final checkpoints for every model were independently reloaded and re-evaluated. All seven comparison metrics agree with saved epoch records within 1e-5.',
          '- All runs used the original seed without resuming or changing hyperparameters during training.', '',
          '## Artifacts', '',
          '- [Detailed comparison](comparison.md) and [comparison CSV](comparison.csv)',
          '- [Reproduction check evidence](paper_consistency.json)',
          '- [Predeclared protocol](protocol.md) and [run manifest](manifest.json)',
          '- [Per-epoch learning curves CSV](learning_curves.csv)',
          '- Per-model folders contain configs, operative Gin settings, best.pt, last.pt and epochs.jsonl; logs/ contains smoke, training and checkpoint evaluation logs.', '']
(RUN / 'summary.md').write_text('\n'.join(lines))
with (RUN / 'learning_curves.csv').open('w', newline='') as handle:
    fields = ['model', 'epoch', 'supervision_mode', 'target_exposures', 'steps', 'elapsed_seconds', 'peak_memory_bytes'] + list(METRICS) + ['ndcg@10', 'ndcg@50']
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for name, rows in records.items():
        for row in rows:
            writer.writerow(dict(model=name, epoch=row['epoch']+1, **{k: row[k] for k in fields[2:7]}, **{k: row['metrics'][k] for k in fields[7:]}))
print(RUN / 'summary.md')
