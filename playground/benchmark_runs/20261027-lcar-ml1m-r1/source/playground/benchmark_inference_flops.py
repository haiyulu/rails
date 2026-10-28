"""Reproducible logical inference FLOPs, with strict user/item boundaries."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fbgemm_gpu  # noqa: F401
import gin
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

import train  # Registers existing Gin bindings, including saved training configs.
from data.reco_dataset import get_reco_dataset
from modeling.sequential.embedding_modules import LocalEmbeddingModule
from modeling.sequential.encoder_utils import get_sequential_encoder
from modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
from modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor, LayerNormEmbeddingPostprocessor
from modeling.similarity_utils import get_similarity_function
from playground.inference_flops_counter import ArithmeticLedger, sum_totals
from playground.model_registry import BASELINE_MODELS, MODEL_SPECS, MODELS, LABELS


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(model_name, checkpoint_root, dataset):
    config = checkpoint_root / model_name / MODELS[model_name]
    checkpoint = checkpoint_root / model_name / "best.pt"
    gin.clear_config()
    gin.parse_config_file(str(config))
    d = gin.query_parameter("train_fn.item_embedding_dim")
    interaction, _ = get_similarity_function(
        module_type=gin.query_parameter("train_fn.interaction_module_type"),
        query_embedding_dim=d, item_embedding_dim=d)
    post = L2NormEmbeddingPostprocessor if gin.query_parameter("train_fn.user_embedding_norm") == "l2_norm" else LayerNormEmbeddingPostprocessor
    with redirect_stdout(io.StringIO()):
        model = get_sequential_encoder(
            module_type=gin.query_parameter("train_fn.main_module"), max_sequence_length=200,
            max_output_length=11, embedding_module=LocalEmbeddingModule(dataset.max_item_id, d),
            interaction_module=interaction,
            input_preproc_module=LearnablePositionalEmbeddingInputFeaturesPreprocessor(211, d, .2),
            output_postproc_module=post(d, 1e-6), verbose=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state["model_state_dict"].items()})
    return model.float().cuda().eval(), {"config": str(config), "config_sha256": sha(config),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "epoch": state["epoch"], "dimension": d, "operative_config": gin.operative_config_str()}


class SplitInference:
    FORBIDDEN = ("history.encode", "_ndp_module._query_embeddings_fn._query_emb_proj_module",
        "_ndp_module._item_embeddings_fn._item_emb_proj_module",
        "_ndp_module._gating_fn._query_only_partial_module",
        "_ndp_module._gating_fn._item_only_partial_module")

    def __init__(self, model, is_mol):
        self.model = model
        self.mol = is_mol
        self.calls = Counter()
        self.stack = []
        self.hooks = []
        for name, module in model.named_modules():
            def before(module, args, name=name):
                self.calls[name] += 1
                self.stack.append(name or "model")
            def after(module, args, output):
                self.stack.pop()
            self.hooks.append(module.register_forward_pre_hook(before))
            self.hooks.append(module.register_forward_hook(after, always_call=True))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def encode(self, fixture):
        self.calls["history.encode"] += 1
        ids, lengths, payloads = fixture
        return self.model.encode(past_lengths=lengths, past_ids=ids,
            past_embeddings=self.model.get_item_embeddings(ids), past_payloads=payloads)

    def prepare_user(self, fixture):
        h = self.encode(fixture)
        if not self.mol:
            return {"h": h, "q": h}
        mol = self.model._ndp_module
        q, _ = mol.get_query_component_embeddings(h, user_ids=fixture[2]["user_ids"])
        a = mol._gating_fn._query_only_partial_module(h)
        return {"h": h, "q": q, "a": a}

    def prepare_items(self, ids):
        e = self.model.get_item_embeddings(ids)
        if not self.mol:
            v = e / torch.clamp(torch.linalg.norm(e, ord=2, dim=-1, keepdim=True), min=1e-6)
            return {"e": e, "v": v}
        mol = self.model._ndp_module
        v, _ = mol.get_item_component_embeddings(e)
        b = mol._gating_fn._item_only_partial_module(e)
        return {"e": e, "v": v, "b": b}

    def score(self, user, items):
        if not self.mol:
            return user["q"] @ items["v"].T
        mol = self.model._ndp_module
        z = torch.einsum("brd,ntd->bnrt", user["q"], items["v"])
        z = z.reshape(user["q"].shape[0], items["v"].shape[0], -1) / mol._temperature
        c = mol._gating_fn._qi_partial_module(z)
        g = user["a"].unsqueeze(1) * items["b"].unsqueeze(0) + c
        p = F.softmax(g * torch.sigmoid(g), dim=-1)
        combiner = mol._gating_fn._normalization_fn
        if combiner._dropout_rate > 0:
            p = p / torch.clamp(p.sum(-1, keepdim=True), min=combiner._eps)
        return (p * z).sum(-1)

    def original(self, fixture, ids):
        h = self.encode(fixture)
        e = self.model.get_item_embeddings(ids)
        if not self.mol:
            e = e / torch.clamp(torch.linalg.norm(e, ord=2, dim=-1, keepdim=True), min=1e-6)
        scores, _ = self.model.similarity_fn(h, ids.unsqueeze(0), e.unsqueeze(0), **fixture[2])
        return scores


class ExpertHistoryInference(SplitInference):
    FORBIDDEN = SplitInference.FORBIDDEN + ("history.prepare", "item.prepare", "model.prepare")

    def __init__(self, model, item_l2_norm=True, normalization_eps=1e-6, candidate_chunk_size=256):
        super().__init__(model, False)
        self.item_l2_norm = item_l2_norm
        self.normalization_eps = normalization_eps
        self.candidate_chunk_size = candidate_chunk_size
        self.FORBIDDEN = self.FORBIDDEN + tuple(
            name for name, _ in model.named_modules()
            if name.startswith(("_ndp_module.raw_key", "_ndp_module.raw_value",
                                "_ndp_module.context_key", "_ndp_module.context_value", "_history_adapters.",
                                "_ndp_module.item_experts.", "_ndp_module.memory_projections.",
                                "_ndp_module.value_query", "_ndp_module.expert_gate",
                                "_ndp_module.sparse_coder")))

    def prepare_model(self):
        self.calls["model.prepare"] += 1
        return self.model.prepare_history_context()

    def prepare_user(self, fixture):
        self.calls["history.encode"] += 1
        self.calls["history.prepare"] += 1
        ids, lengths, payloads = fixture
        h, memory = self.model.encode_with_history(
            lengths, ids, self.model.get_item_embeddings(ids), payloads)
        return {"h": h, "memory": memory}

    def prepare_items(self, ids):
        self.calls["item.prepare"] += 1
        embedding = self.model.get_item_embeddings(ids)
        if self.item_l2_norm:
            embedding = F.normalize(embedding, dim=-1, eps=self.normalization_eps)
        return self.model._ndp_module.prepare_items(embedding.unsqueeze(0), ids.unsqueeze(0))

    def score(self, user, items):
        return torch.cat([self.model._ndp_module.score_prepared(
            user["memory"], items.slice_candidates(start, start + self.candidate_chunk_size))[0]
            for start in range(0, items.num_candidates, self.candidate_chunk_size)], dim=1)

    def original(self, fixture, ids):
        user = self.prepare_user(fixture)
        embedding = self.model.get_item_embeddings(ids)
        if self.item_l2_norm:
            embedding = F.normalize(embedding, dim=-1, eps=self.normalization_eps)
        return torch.cat([self.model.similarity_fn(
            user["h"], ids[start:start + self.candidate_chunk_size].unsqueeze(0),
            embedding[start:start + self.candidate_chunk_size].unsqueeze(0),
            user_memory=user["memory"])[0]
            for start in range(0, ids.numel(), self.candidate_chunk_size)], dim=1)


class LCARInference(ExpertHistoryInference):
    """Prepared-state accounting for the independent LCAR module; never runs on import."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.FORBIDDEN=self.FORBIDDEN+tuple(name for name,_ in self.model.named_modules()
            if name.startswith(('_ndp_module.token_projections.','_ndp_module.summary_projections.',
                                '_ndp_module.raw_projection','_ndp_module.memory_key',
                                '_ndp_module.memory_value','_ndp_module.query_projection')))


def fixture(catalog, batch, length):
    ids = torch.zeros(batch, 211, dtype=torch.long, device="cuda")
    index = torch.arange(batch * length, device="cuda").reshape(batch, length) % catalog.numel()
    ids[:, :length] = catalog[index]
    timestamps = torch.zeros_like(ids)
    timestamps[:, :length] = 1_000_000_000 + 3600 * torch.arange(length, device="cuda")
    return ids, torch.full((batch,), length, dtype=torch.long, device="cuda"), {
        "timestamps": timestamps, "user_ids": torch.arange(1, batch + 1, device="cuda")}


def counted(adapter, function, bucket):
    before = adapter.calls.copy()
    ledger = ArithmeticLedger(adapter.stack, bucket=bucket)
    with ledger:
        result = function()
    assert not adapter.stack
    evidence = {name: adapter.calls[name] - before[name] for name in adapter.FORBIDDEN}
    return result, ledger, evidence


def write_csv(path, rows):
    import csv
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure_latency_ms(function, repeats=10):
    """Optional benchmark measurement, called only by --measure-latency."""
    if repeats < 1:
        raise ValueError("latency repeats must be positive")
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repeats


def make_report(output, results, catalog_size):
    flat = []
    for result in results:
        row = {key: result[key] for key in ("model", "B", "L", "T", "N")}
        row.update(user_cache_bytes=result.get("user_cache_bytes"),
                   item_cache_bytes=result.get("item_cache_bytes"),
                   peak_memory_bytes=result.get("peak_memory_bytes"))
        for name in ("user", "items", "scoring", "cached_catalog", "uncached_catalog"):
            row[f"{name}_latency_ms"] = result.get("latency_ms", {}).get(name)
        for bucket in ("user", "items", "scoring", "cached_total", "uncached_total"):
            for key, value in result[bucket].items():
                row[f"{bucket}_{key}"] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        row["scoring_flops_per_pair"] = result["scoring"]["arithmetic_flops"] / (result["B"] * result["N"])
        row["cached_flops_per_user"] = result["cached_total"]["arithmetic_flops"] / result["B"]
        row["uncached_flops_per_user"] = result["uncached_total"]["arithmetic_flops"] / result["B"]
        flat.append(row)
    write_csv(output / "sweep.csv", flat)
    headline = [row for row in flat if row["B"] == 1 and row["L"] == 200 and row["N"] == catalog_size]
    baseline = next((row for row in headline if row["model"] == "sasrec"), headline[0])
    for row in headline:
        row["reference_model"] = baseline["model"]
        for metric in ("user_arithmetic_flops", "scoring_flops_per_pair", "cached_total_arithmetic_flops", "uncached_total_arithmetic_flops"):
            row[f"{metric}_vs_reference"] = row[metric] / baseline[metric]
    write_csv(output / "comparison.csv", headline)
    lines = ["# Inference FLOPs comparison", "",
        f"ML-1M checkpoints; B=1 user, L=200 valid history items, T=211 padded positions, N={catalog_size:,} shared candidates.",
        "FP32 eager inference; no incremental history cache. One multiply–accumulate = 2 FLOPs.",
        "Totals count logical add/subtract/multiply/divide arithmetic. Special functions are reported separately below.", "",
        "## Headline results", "",
        "| Model | User MFLOPs | Candidate preparation MFLOPs | Strict scoring FLOPs/pair | All-candidate scoring MFLOPs | Entire cached MFLOPs | Entire uncached MFLOPs |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in headline:
        columns = [LABELS[row["model"]], f'{row["user_arithmetic_flops"]/1e6:.6f}',
            f'{row["items_arithmetic_flops"]/1e6:.6f}', f'{row["scoring_flops_per_pair"]:,.0f}',
            f'{row["scoring_arithmetic_flops"]/1e6:.6f}', f'{row["cached_total_arithmetic_flops"]/1e6:.6f}',
            f'{row["uncached_total_arithmetic_flops"]/1e6:.6f}']
        lines.append("| " + " | ".join(columns) + " |")
    lines += ["", "Cached entire model = user + strict scoring. Uncached entire model additionally prepares the shared catalog once.",
        "Query/component projections and query-only gates are charged to the user. Candidate projections and item-only gates are charged to candidate preparation.",
        "Sorting, support selection, comparisons, and gathers are reported separately from floating arithmetic.",
        "LCAR includes shared native history blocks, independent latent reads, first-block selection, and joint scoring. Legacy ExpertHistory uses preconditioning. Model-only preparation is recorded separately.", "",
        f"## Ratios versus {LABELS[baseline['model']]}", "",
        "| Model | User arithmetic | Strict scoring/pair | Entire cached | Entire uncached |",
        "|---|---:|---:|---:|---:|"]
    for row in headline:
        lines.append("| " + " | ".join([LABELS[row["model"]]] + [f'{row[k+"_vs_reference"]:.3f}×' for k in
            ("user_arithmetic_flops", "scoring_flops_per_pair", "cached_total_arithmetic_flops", "uncached_total_arithmetic_flops")]) + " |")
    lines += ["", "## Special-function evaluations (not included in arithmetic FLOPs)", "",
        "| Model | User | Candidate preparation | Strict scoring, all candidates |",
        "|---|---|---|---|"]
    for row in headline:
        columns = [LABELS[row["model"]]]
        for bucket in ("user", "items", "scoring"):
            values = json.loads(row[f"{bucket}_special_functions"])
            columns.append(", ".join(f"{name}: {count:,}" for name, count in sorted(values.items())) or "none")
        lines.append("| " + " | ".join(columns) + " |")
    lines += ["", "## Validation and interpretation", "",
        f"- {len(results)} cases validated against the original inference path; independent cached/uncached counting reconciled exactly.",
        "- Strict scorers made zero calls to history encoding, query/item projections, or query-only/item-only gate MLPs.",
        "- Scoring matrix FLOPs/pair: baseline 100; MoL 20,480 (4,096 component contractions + 16,384 joint gate MLP).",
        "- MoL adds bias/activation/scaling/softmax/renormalization/reduction arithmetic; the configured inference combiner renormalizes even though dropout is disabled.",
        "- SASRec and HSTU have identical baseline scoring cost; both MoL variants have identical strict scoring cost.",
        "- Input width is fixed at 211. Causal masks do not eliminate the executed dense attention products. HSTU pointwise work follows valid lengths, while its attention remains padded.",
        "- These are logical-operation counts, not GPU instructions or measured latency. Special-function costs are not priced as hardware-equivalent FLOPs.",
        "- See sweep.csv, operation_ledger.json, validation.json, coverage.md, and manifest.json for details.", ""]
    if any(result.get("latency_ms") for result in results):
        lines += ["", "## Optional measured latency", "",
                  "Catalog scoring includes candidate chunks and score concatenation; ranking and seen-item filtering are separate.",
                  "| Model | User ms | Item preparation ms | Scoring ms | Cached catalog ms | Uncached catalog ms | Peak GiB |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in headline:
            times = [row[f"{key}_latency_ms"] for key in ("user", "items", "scoring", "cached_catalog", "uncached_catalog")]
            if all(value is not None for value in times):
                lines.append("| " + " | ".join([LABELS[row["model"]]] + [f"{value:.4f}" for value in times]
                             + [f"{row['peak_memory_bytes'] / 2**30:.3f}"]) + " |")
    (output / "comparison.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "playground/benchmark_runs/20261026-201653")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="One headline case per model for development")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(BASELINE_MODELS))
    parser.add_argument("--measure-latency", action="store_true", help="Explicitly enable CUDA-event timing")
    parser.add_argument("--latency-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.latency_repeats < 1:
        parser.error("--latency-repeats must be positive")
    output = (args.output_dir or ROOT / "playground/inference_flops_runs" /
        datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S-%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = get_reco_dataset("ml-1m", 200, chronological=True)
    catalog = torch.tensor(dataset.all_item_ids, dtype=torch.long, device="cuda")
    catalog_size = catalog.numel()
    manifest = {"status": "running", "smoke": args.smoke, "output": str(output),
        "measure_latency": args.measure_latency, "latency_repeats": args.latency_repeats,
        "checkpoint_root": str(args.checkpoint_root.resolve()), "torch": torch.__version__,
        "python": sys.version, "gpu": torch.cuda.get_device_name(0), "dtype": "float32",
        "tf32": False, "seed": 42, "catalog_size": catalog_size, "models": {},
        "convention": "MAC=2; arithmetic separate from special functions; cached items shared across B users",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
            [Path(__file__).resolve(), ROOT / "playground/inference_flops_counter.py",
             ROOT / "modeling/sequential/sasrec.py", ROOT / "modeling/sequential/hstu.py",
             ROOT / "modeling/sequential/latent_attention.py", ROOT / "modeling/similarity_module.py",
             ROOT / "rails/similarities/mol/similarity_fn.py", ROOT / "modeling/similarity_utils.py",
             ROOT / "rails/similarities/expert_history/similarity_fn.py",
             ROOT / "rails/similarities/expert_history/sparse_coding.py",
             ROOT / "rails/similarities/expert_history/states.py", ROOT / "playground/model_registry.py"]}}
    for name in ('__init__.py','backbones.py','encoding.py','similarity_fn.py','sparse_coding.py','states.py'):
        source=ROOT/'rails/similarities/lcar'/name
        manifest['source_sha256'][str(source.relative_to(ROOT))]=sha(source)
    dump(output / "manifest.json", manifest)
    (output / "requirements.lock.txt").write_text(subprocess.check_output(
        ["uv", "pip", "freeze", "--python", sys.executable], text=True))
    results, validations, ledgers = [], [], []
    observed = set()
    try:
        with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            for model_name in args.models:
                with torch.inference_mode(False):
                    model, provenance = load_model(model_name, args.checkpoint_root, dataset)
                manifest["models"][model_name] = provenance
                spec = MODEL_SPECS[model_name]
                prepared_adapter=LCARInference if spec.inference_adapter=='lcar' else ExpertHistoryInference
                adapter = (prepared_adapter(model, gin.query_parameter("train_fn.item_l2_norm"),
                                                   gin.query_parameter("train_fn.l2_norm_eps"))
                           if spec.inference_adapter in ("expert_history","lcar")
                           else SplitInference(model, spec.inference_adapter == "mol"))
                if isinstance(adapter, ExpertHistoryInference):
                    model_state, model_ledger, _ = counted(adapter, adapter.prepare_model, "model_only")
                    manifest["models"][model_name]["model_only_preparation"] = model_ledger.totals()
                    manifest["models"][model_name]["model_only_cache_bytes"] = (
                        model_state.num_bytes() + sum(layer.cache_num_bytes() for layer in model._history_adapters))
                    manifest["models"][model_name]["architecture"] = model._ndp_module.get_extra_state()
                manifest["models"][model_name].update(
                    scorer=spec.scorer, exact_top_k=spec.exact_top_k,
                    inference_adapter=spec.inference_adapter,
                    parameter_count=sum(p.numel() for p in model.parameters()))
                shapes = [(1, 200, catalog_size)] if args.smoke else itertools.product(
                    [1, 32], [10, 50, 100, 200], [1, 100, 1000, catalog_size])
                for batch, length, n in shapes:
                    print(f"COUNT {model_name} B={batch} L={length} N={n}", flush=True)
                    inputs, ids = fixture(catalog, batch, length), catalog[:n]
                    torch.cuda.reset_peak_memory_stats()
                    user, u, u_calls = counted(adapter, lambda: adapter.prepare_user(inputs), "U")
                    items, i, i_calls = counted(adapter, lambda: adapter.prepare_items(ids), "I")
                    scores, s, s_calls = counted(adapter, lambda: adapter.score(user, items), "S")
                    assert all(v == 0 for v in s_calls.values()), s_calls
                    assert u_calls["history.encode"] == 1 and i_calls["history.encode"] == 0
                    if adapter.mol:
                        for key in (adapter.FORBIDDEN[1], adapter.FORBIDDEN[3]):
                            assert u_calls[key] == 1 and i_calls[key] == 0
                        for key in (adapter.FORBIDDEN[2], adapter.FORBIDDEN[4]):
                            assert i_calls[key] == 1 and u_calls[key] == 0
                    original = adapter.original(inputs, ids)
                    torch.testing.assert_close(scores, original, rtol=1e-4, atol=1e-5)
                    error = float((scores - original).abs().max())
                    def uncached():
                        return adapter.score(adapter.prepare_user(inputs), adapter.prepare_items(ids))
                    def cached():
                        return adapter.score(adapter.prepare_user(inputs), items)
                    full_scores, full, _ = counted(adapter, uncached, "composed_uncached")
                    cache_scores, cache, _ = counted(adapter, cached, "composed_cached")
                    torch.testing.assert_close(full_scores, scores, rtol=0, atol=0)
                    torch.testing.assert_close(cache_scores, scores, rtol=0, atol=0)
                    assert full.totals() == sum_totals(u.totals(), i.totals(), s.totals())
                    assert cache.totals() == sum_totals(u.totals(), s.totals())
                    if not isinstance(adapter, ExpertHistoryInference):
                        expected_matrix = (20480 if adapter.mol else 100) * batch * n
                        expected_arithmetic = (21117 if adapter.mol else 100) * batch * n
                        assert s.totals()["matrix_flops"] == expected_matrix, s.totals()
                        assert s.totals()["arithmetic_flops"] == expected_arithmetic, s.totals()
                    if adapter.mol:
                        assert s.totals()["special_functions"] == {"sigmoid": 160 * batch * n, "exp": 32 * batch * n}
                    profiler = {}
                    for bucket, function, ledger in [("U", lambda: adapter.prepare_user(inputs), u),
                            ("I", lambda: adapter.prepare_items(ids), i),
                            ("S", lambda: adapter.score(user, items), s)]:
                        with FlopCounterMode(display=False) as counter:
                            function()
                        profiler[bucket] = counter.get_total_flops()
                        assert profiler[bucket] == ledger.totals()["matrix_flops"], (bucket, profiler, ledger.totals())
                    case = {"model": model_name, "B": batch, "L": length, "T": 211, "N": n}
                    resources = ({"user_cache_bytes": user["memory"].num_bytes(),
                                  "item_cache_bytes": items.num_bytes(),
                                  "diagnostics": {k: float(v) for k, v in items.diagnostics(model._ndp_module.dictionary_size).items()}}
                                 if isinstance(adapter, ExpertHistoryInference) else {})
                    if args.measure_latency:
                        resources["latency_ms"] = {
                            "user": measure_latency_ms(lambda: adapter.prepare_user(inputs), args.latency_repeats),
                            "items": measure_latency_ms(lambda: adapter.prepare_items(ids), args.latency_repeats),
                            "scoring": measure_latency_ms(lambda: adapter.score(user, items), args.latency_repeats),
                            "cached_catalog": measure_latency_ms(cached, args.latency_repeats),
                            "uncached_catalog": measure_latency_ms(uncached, args.latency_repeats)}
                    resources["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
                    results.append(case | {"user": u.totals(), "items": i.totals(), "scoring": s.totals(),
                        "cached_total": cache.totals(), "uncached_total": full.totals(), **resources})
                    validations.append(case | {"max_absolute_score_error": error,
                        "boundary_calls": {"U": u_calls, "I": i_calls, "S": s_calls},
                        "reconciliation": "exact", "profiler_matrix_crosscheck": profiler})
                    for ledger in (u, i, s):
                        ledgers.append(case | {"bucket": ledger.bucket, "operations": list(ledger.rows.values())})
                        observed.update(ledger.observed)
                    dump(output / "validation.json", validations)
                    del user, items, scores, original, full_scores, cache_scores, inputs, ids
                adapter.close()
                del adapter, model
                dump(output / "manifest.json", manifest)
            # Validate scaling and cross-model invariants over the complete result matrix.
            for first, second in itertools.combinations(results, 2):
                if first["model"] == second["model"]:
                    if (first["B"], first["L"]) == (second["B"], second["L"]):
                        assert first["user"] == second["user"]
                    assert first["items"]["arithmetic_flops"] * second["N"] == second["items"]["arithmetic_flops"] * first["N"]
                    if MODEL_SPECS[first["model"]].scorer not in ("ExpertHistory","LCAR"):
                        assert first["scoring"]["arithmetic_flops"] * second["B"] * second["N"] == second["scoring"]["arithmetic_flops"] * first["B"] * first["N"]
                if (MODEL_SPECS[first["model"]].scorer == MODEL_SPECS[second["model"]].scorer
                    and MODEL_SPECS[first["model"]].scorer not in ("ExpertHistory","LCAR")
                    and (first["B"], first["N"]) == (second["B"], second["N"])):
                    assert first["scoring"] == second["scoring"]
        dump(output / "operation_ledger.json", ledgers)
        dump(output / "results.json", results)
        make_report(output, results, catalog_size)
        (output / "coverage.md").write_text("# FLOP coverage\n\nAll observed floating-point operators were explicitly classified. "
            "An unknown floating operator aborts the run. Matrix counts matched PyTorch FlopCounterMode in every case. "
            "PyTorch omits bias/elementwise arithmetic and special functions from its matrix count; these are included in the separate ledger. "
            "Custom jagged conversions are classified as data movement. LayerNorm, L2 norm, softmax and SiLU use the plan's canonical decompositions.\n\n"
            "Observed operators:\n\n" + "\n".join(f"- `{op}`" for op in sorted(observed)) + "\n")
        manifest.update(status="complete", cases=len(results), max_absolute_score_error=max(v["max_absolute_score_error"] for v in validations))
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        dump(output / "manifest.json", manifest)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
