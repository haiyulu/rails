"""Run explicitly selected ML-1M configurations sequentially on one GPU.

Run from the repository root with .venv/bin/python playground/run_ml1m_benchmark.py.
Outputs are self-contained under playground/benchmark_runs/<run-id>/.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from playground.model_registry import ABLATIONS, LCAR_ABLATIONS, MODEL_SPECS, MODELS, LABELS, DEFAULT_MODELS
METRICS = ["hr@1", "hr@10", "hr@50", "hr@200", "mrr"]
EXTRA_METRICS = ["ndcg@10", "ndcg@50"]
TARGETS = {
    "sasrec": [.0610, .2818, .5470, .7540, .1352],
    "sasrec_mol": [.0697, .3036, .5617, .7667, .1441],
    "hstu": [.0750, .3332, .5956, .7824, .1579],
    "hstu_mol": [.0884, .3465, .6022, .7935, .1712],
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(args, log, env, manifest):
    entry = {"command": args, "log": str(log), "started": time.time()}
    manifest["commands"].append(entry)
    dump(log.parent.parent / "manifest.json", manifest)
    print(f"START {log.name}", flush=True)
    with log.open("w") as out:
        process = subprocess.run(args, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    entry.update(returncode=process.returncode, duration_seconds=time.time() - entry["started"])
    dump(log.parent.parent / "manifest.json", manifest)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}): {log}\n{log.read_text()[-6000:]}")
    print(f"DONE {log.name} ({entry['duration_seconds']:.1f}s)", flush=True)


def evaluate(config, checkpoint, output, env, manifest, model_name):
    method = MODEL_SPECS[model_name].exact_top_k
    eval_config = output.with_suffix(".gin")
    eval_config.write_text("\n".join(line for line in config.read_text().splitlines()
        if not line.startswith(("train_fn.benchmark_output_dir", "train_fn.random_seed"))) + "\n")
    command([sys.executable, "eval_from_checkpoint.py", f"--gin_config_file={eval_config}",
             f"--inference_from_ckpt={checkpoint}", f"--top_k_method={method}",
             "--eval_batch_size=128", "--limit_eval_to_first_n=0", "--include_eval_time=false",
             "--eval_against_brute_force=false", f"--metrics_output={output}",
             "--master_port=12356"], output.with_suffix(".log"), env, manifest)
    return json.loads(output.read_text())


def report(run):
    rows = []
    for model in MODELS:
        epochs_file = run / model / "epochs.jsonl"
        if not epochs_file.exists():
            continue
        epochs = [json.loads(line) for line in epochs_file.read_text().splitlines()]
        best = max(epochs, key=lambda row: row["metrics"]["hr@10"])
        for selection, row in [("final", epochs[-1]), ("best_eval_hr10", best)]:
            rows.append({"model": model, "selection": selection, "epoch": row["epoch"],
                         "epochs_completed": len(epochs), **{m: row["metrics"][m] for m in METRICS},
                         **{m: row["metrics"][m] for m in EXTRA_METRICS},
                         **{f"paper_{m}": t for m, t in zip(METRICS, TARGETS.get(model, [None] * len(METRICS)))},
                         **{f"delta_{m}": row["metrics"][m] - t if t is not None else None
                            for m, t in zip(METRICS, TARGETS.get(model, [None] * len(METRICS)))},
                         "supervision_mode": row.get("supervision_mode", MODEL_SPECS[model].supervision_mode),
                         "target_exposures": row.get("target_exposures"),
                         "parameter_count": row.get("parameter_count"),
                         "training_seconds": epochs[-1]["elapsed_seconds"],
                         "peak_memory_bytes": epochs[-1]["peak_memory_bytes"]})
    if not rows:
        return
    with (run / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# ML-1M benchmark results", "",
             "Seed 42; supplied configurations; full-catalog exact scoring.",
             "Prefix and all-position supervision have different target exposure; compare matched protocols.",
             "Best epoch is selected on the evaluation set, not an independent validation set.",
             "Epoch numbers are zero-based. Training time includes evaluation and checkpoint saving.",
             "MRR uses the repository's original rank cutoff of 2500.", "",
             "| Model | Selection | Epoch | Epochs completed | HR@1 | HR@10 | HR@50 | HR@200 | MRR |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        if model in TARGETS:
            lines.append("| " + " | ".join([LABELS[model], "paper", "—", "—"] +
                [f"{v:.4f}" for v in TARGETS[model]]) + " |")
        for row in model_rows:
            values = [LABELS[model], row["selection"], str(row["epoch"]), str(row["epochs_completed"])]
            values += [f"{row[m]:.4f}" for m in METRICS]
            lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## HR@10 differences and resources", "",
              "Differences are absolute reproduced-minus-paper values. Memory is peak allocated CUDA memory.", "",
              "| Model | Final HR@10 difference | Best HR@10 difference | Training + evaluation minutes | Peak GiB |",
              "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if model_rows:
            final, best = model_rows
            delta = lambda value: "—" if value is None else f"{value:+.4f}"
            lines.append(f"| {LABELS[model]} | {delta(final['delta_hr@10'])} | {delta(best['delta_hr@10'])} | "
                f"{final['training_seconds'] / 60:.2f} | {final['peak_memory_bytes'] / 2**30:.2f} |")
    lines += ["", "## NDCG", "", "| Model | Selection | NDCG@10 | NDCG@50 |",
              "|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {LABELS[row['model']]} | {row['selection']} | {row['ndcg@10']:.4f} | {row['ndcg@50']:.4f} |")
    lines += ["", "All paper targets and per-metric differences are included in comparison.csv.",
              "Reference: https://arxiv.org/html/2407.15462v3#S4.T3", ""]
    lcar_rows = [row for row in rows if MODEL_SPECS[row["model"]].scorer == "LCAR"]
    if lcar_rows:
        lines += ["## LCAR comparison against paper Table 3", "",
                  "Paper rows use the published protocol; LCAR uses sampled-prefix supervision.",
                  "Each LCAR epoch presents one target per user, unlike all-position baseline training.",
                  "These are single-seed experiments; best_eval_hr10 selects on the evaluation set.", "",
                  "| Model | Selection | HR@1 | HR@10 | HR@50 | HR@200 | MRR |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name, values in TARGETS.items():
            lines.append("| " + " | ".join([LABELS[name], "paper"] + [f"{v:.4f}" for v in values]) + " |")
        for row in lcar_rows:
            lines.append("| " + " | ".join([LABELS[row["model"]], row["selection"]] +
                         [f"{row[m]:.4f}" for m in METRICS]) + " |")
        differences = []
        lines += ["", "Differences below are LCAR minus the corresponding paper baseline, in percentage points.", "",
                  "| LCAR model | Selection | Paper baseline | ΔHR@1 | ΔHR@10 | ΔHR@50 | ΔHR@200 | ΔMRR |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for row in lcar_rows:
            backbone = row["model"].removesuffix("_lcar")
            for reference in (backbone, backbone + "_mol"):
                deltas = {m: row[m] - value for m, value in zip(METRICS, TARGETS[reference])}
                differences.append({"model": row["model"], "selection": row["selection"],
                                    "paper_baseline": reference, **deltas})
                lines.append("| " + " | ".join([LABELS[row["model"]], row["selection"], LABELS[reference]] +
                             [f"{100*deltas[m]:+.2f}" for m in METRICS]) + " |")
        with (run / "lcar_vs_paper.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(differences[0]))
            writer.writeheader()
            writer.writerows(differences)
    (run / "comparison.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(DEFAULT_MODELS))
    parser.add_argument("--supervision-mode", choices=("auto", "all_positions", "sampled_prefix"), default="auto")
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="full")
    args = parser.parse_args()
    run = args.run_dir or ROOT / "playground/benchmark_runs" / datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    run = run.resolve()
    if args.report_only:
        report(run)
        return
    run.mkdir(parents=True, exist_ok=False)
    (run / "logs").mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8")
    manifest = {"run_id": run.name, "status": "preflight", "seed": 42, "commands": [],
                "python": sys.version, "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "gpu": subprocess.check_output(["nvidia-smi"], text=True),
                "models": {name: asdict(MODEL_SPECS[name]) for name in args.models},
                "supervision_override": args.supervision_mode, "ablation": args.ablation,
                "data_sha256": {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                    [Path("tmp/ml-1m/sasrec_format.csv"), Path("tmp/processed/ml-1m/movies.csv")]},
                "deviations": ["Modern uv environment for RTX 5090; original model/optimizer settings retained.",
                    "Two persistent data workers with prefetch factor 4; original iterator RNG draws preserved.",
                    "Single seed; no independent validation split; final and best evaluation epoch reported separately.",
                    "Checkpoint evaluator extended to support SASRec and export full metrics."]}
    dump(run / "manifest.json", manifest)
    (run / "requirements.lock.txt").write_text(subprocess.check_output(
        ["uv", "pip", "freeze", "--python", sys.executable], text=True))
    (run / "source.patch").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True))
    # Include new modules/configs, which an ordinary git diff does not capture.
    sources = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    source_hashes = {}
    for relative in sources:
        source = ROOT / relative
        if not relative or source.suffix not in (".py", ".gin", ".md", ".toml") or not source.is_file():
            continue
        if relative.startswith("playground/benchmark_runs/") or relative.startswith("playground/inference_flops_runs/"):
            continue
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    dump(run / "source_sha256.json", source_hashes)
    try:
        command([sys.executable, "playground/validate_ml1m_benchmark.py"], run / "logs/validation.log", env, manifest)
        configs = {}
        for model in args.models:
            filename = MODELS[model]
            model_dir = run / model
            model_dir.mkdir()
            original = ROOT / "configs/ml-1m" / filename
            config = model_dir / filename
            config.write_text(original.read_text() + "\n" +
                f'train_fn.custom_date_str = "benchmark-{run.name}"\n' +
                f'train_fn.benchmark_output_dir = "{model_dir}"\n' +
                'train_fn.random_seed = 42\ntrain_fn.save_ckpt_every_n = 1\n' +
                'create_data_loader.num_workers = 2\ncreate_data_loader.prefetch_factor = 4\n' +
                'create_data_loader.persistent_workers = True\n')
            extra = ""
            if args.supervision_mode != "auto":
                if MODEL_SPECS[model].scorer in ("ExpertHistory","LCAR") and args.supervision_mode != "sampled_prefix":
                    raise ValueError(f"{MODEL_SPECS[model].scorer} requires sampled_prefix supervision")
                extra += f'train_fn.supervision_mode = "{args.supervision_mode}"\n'
            if MODEL_SPECS[model].scorer == "ExpertHistory":
                extra += ABLATIONS[args.ablation]
            elif MODEL_SPECS[model].scorer == "LCAR":
                if args.ablation not in LCAR_ABLATIONS:
                    raise ValueError(f"Ablation {args.ablation} does not apply to LCAR")
                extra += LCAR_ABLATIONS[args.ablation]
            config.write_text(config.read_text() + extra)
            configs[model] = config
            smoke = model_dir / "smoke.gin"
            smoke.write_text(config.read_text() + '\ntrain_fn.smoke_steps = 1\n')
            command([sys.executable, "train.py", f"--gin_config_file={smoke}", "--master_port=12355"],
                    run / "logs" / f"{model}_smoke.log", env, manifest)
        checkpoint = next((ROOT / "ckpts/ml-1m-l200").glob("*2024-11-06_ep72"), None)
        if checkpoint is not None and "hstu_mol" in configs:
            sanity = evaluate(configs["hstu_mol"], checkpoint, run / "logs/pretrained_metrics.json", env, manifest, "hstu_mol")
            manifest["pretrained_sanity"] = sanity
            if abs(sanity["metrics"]["hr@10"] - .3474) > .005:
                raise RuntimeError("Pretrained HR@10 differs by more than 0.005 from documented result")
        manifest["status"] = "training"
        for model, config in configs.items():
            manifest["active_model"] = model
            command([sys.executable, "train.py", f"--gin_config_file={config}", "--master_port=12355"],
                    run / "logs" / f"{model}_train.log", env, manifest)
            epochs = [json.loads(line) for line in (run / model / "epochs.jsonl").read_text().splitlines()]
            assert len(epochs) == 101, f"Expected 101 epochs for {model}"
            best = max(epochs, key=lambda row: row["metrics"]["hr@10"])
            for selection, expected in [("best", best), ("last", epochs[-1])]:
                actual = evaluate(config, run / model / f"{selection}.pt",
                                  run / "logs" / f"{model}_{selection}_metrics.json", env, manifest, model)
                for metric in METRICS + EXTRA_METRICS:
                    if abs(actual["metrics"][metric] - expected["metrics"][metric]) > 1e-5:
                        raise RuntimeError(f"Checkpoint metric mismatch: {model}/{selection}/{metric}")
            report(run)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        report(run)
        dump(run / "manifest.json", manifest)
    print(f"COMPLETE {run}", flush=True)


if __name__ == "__main__":
    main()
