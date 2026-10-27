"""Correctness tests only: no training runs, timing loops, or benchmark artifacts.

Run: .venv/bin/python -m unittest playground.test_expert_history -v
"""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from rails.similarities.expert_history import ExpertHistorySimilarity
from rails.similarities.expert_history.sparse_coding import NonnegativeSparseCoding
from rails.indexing.learned_top_k import LearnedBruteForceTopK
from modeling.sequential.features import SequentialFeatures
from modeling.sequential.prefix_sampling import sample_causal_prefix


def scorer(**overrides):
    settings = dict(query_embedding_dim=6, item_embedding_dim=6, num_item_experts=2,
                    dictionary_size=5, max_active_latents=2, events_per_slot=2,
                    attention_dim=3, value_dim=3, sparse_coding_steps=8)
    settings.update(overrides)
    return ExpertHistorySimilarity(**settings)


class ExpertHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def direct_reads(self, model, history, mask, memory, items, expert):
        """Independent full-history reference, including the exact tail complement."""
        keys = model.history_key(history).float() / model.attention_dim ** 0.5
        values = model.history_value(history).float()
        raw, tail = [], []
        for u in range(history.shape[0]):
            rrow, trow = [], []
            for n in range(items.num_candidates):
                item_u = 0 if items.queries.shape[0] == 1 else u
                me = 0 if model.share_expert_memory else expert
                atoms = items.active_indices[item_u, n, expert]
                active = items.active_mask[item_u, n, expert]
                positions = []
                for atom in atoms[active]:
                    idx = memory.selected_indices[u, me, atom]
                    valid = memory.selected_mask[u, me, atom]
                    positions += memory.positions[u, idx[valid]].tolist()
                selected = torch.zeros(history.shape[1], device=history.device, dtype=torch.bool)
                selected[list(set(positions))] = True
                rest = mask[u] & ~selected
                if selected.any():
                    q = items.queries[item_u, n, expert]
                    p = (keys[u, selected] @ q).softmax(0)
                    rrow.append(p @ values[u, selected])
                else:
                    rrow.append(values.new_zeros(model.value_dim))
                summary = values.new_zeros(model.value_dim)
                for k, atom in enumerate(atoms):
                    if active[k] and rest.any():
                        p = (keys[u, rest].double() @ memory.memory_queries[me, atom].double()).softmax(0)
                        z = p @ values[u, rest].double()
                        summary += (items.activation_weights[item_u, n, expert, k] * z).float()
                trow.append(summary)
            raw.append(torch.stack(rrow))
            tail.append(torch.stack(trow))
        return torch.stack(raw), torch.stack(tail)

    def test_shared_and_per_user_candidates_match_direct_reads(self):
        for candidate_batch in (1, 2):
            model = scorer().eval()
            h = torch.randn(2, 11, 6)
            mask = torch.arange(11)[None, :] < torch.tensor([11, 7])[:, None]
            ids = torch.tensor([[3, 3, 9, 0, 0, 0, 0, 0, 0, 0, 0]]).expand(2, -1)
            memory = model.prepare_user(h, mask, {"item_ids": ids})
            items = model.prepare_items(torch.randn(candidate_batch, 3, 6))
            for e in range(2):
                raw, tail, _, _ = model._read_expert(memory, items, e)
                expected_raw, expected_tail = self.direct_reads(model, h, mask, memory, items, e)
                torch.testing.assert_close(raw, expected_raw, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(tail, expected_tail, atol=2e-6, rtol=2e-5)
            scores, _ = model.score_prepared(memory, items)
            self.assertEqual(scores.shape, (2, 3))
            self.assertIn("item_ids", memory.event_payloads)

    def test_concentrated_tail_preserves_small_omitted_mass(self):
        model = scorer(query_embedding_dim=2, item_embedding_dim=2, attention_dim=2,
                       value_dim=2, num_item_experts=1, dictionary_size=2, events_per_slot=1).eval()
        with torch.no_grad():
            model.history_key.weight.copy_(torch.eye(2))
            model.history_key.bias.zero_()
            model.sparse_coder.dictionary.copy_(torch.eye(2).unsqueeze(0))
            model.memory_projections[0].weight.copy_(torch.eye(2) * 2000)
            model.memory_projections[0].bias.zero_()
        history = torch.tensor([[[10., 1.], [1., 10.], [0., 0.], [-1., -1.]]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        memory = model.prepare_user(history, mask)
        items = model.prepare_items(torch.randn(1, 1, 2))
        items = replace(items, active_indices=torch.tensor([[[[0, 1]]]]),
                        active_mask=torch.ones(1, 1, 1, 2, dtype=torch.bool),
                        activation_weights=torch.tensor([[[[.4, .6]]]]))
        _, tail, _, present = model._read_expert(memory, items, 0)
        _, expected = self.direct_reads(model, history, mask, memory, items, 0)
        self.assertTrue(present.all())
        self.assertTrue(torch.isfinite(tail).all())
        torch.testing.assert_close(tail, expected, atol=1e-5, rtol=1e-5)

    def test_empty_short_and_fully_selected_histories(self):
        for length in (0, 1, 2, 5):
            model = scorer(events_per_slot=8)
            history = torch.randn(1, length, 6, requires_grad=True)
            memory = model.prepare_user(history, torch.ones(1, length, dtype=torch.bool))
            items = model.prepare_items(torch.randn(1, 1, 6))
            scores, aux = model.score_prepared(memory, items)
            self.assertEqual(scores.shape, (1, 1))
            self.assertTrue(torch.isfinite(scores).all())
            for e in range(2):
                _, tail, _, present = model._read_expert(memory, items, e)
                self.assertFalse(present.any())
                self.assertEqual(tail.abs().sum().item(), 0)
            (scores.sum() + aux["sparse_code_loss"]).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_dictionary_constraints_support_and_zero_fallback(self):
        coding = NonnegativeSparseCoding(2, 5, 3, 2, l1_weight=100)
        indices, mask, coefficients, weights, fallback, loss = coding(torch.randn(2, 4, 2, 3))
        self.assertTrue(fallback.all())
        self.assertTrue((coefficients == 0).all())
        self.assertTrue((mask.sum(-1) == 1).all())
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        with torch.no_grad():
            coding.dictionary.mul_(10)
        coding.project_dictionary_()
        self.assertTrue((coding.dictionary.norm(dim=-1) <= 1.000001).all())
        self.assertTrue(torch.isfinite(loss).all())

    def test_sparse_solver_reduces_objective_and_has_gradients(self):
        coding = NonnegativeSparseCoding(1, 7, 4, 7, steps=16)
        q = torch.randn(2, 3, 1, 4, requires_grad=True)
        _, _, coefficients, _, _, loss = coding(q)
        self.assertTrue((coefficients >= 0).all())
        self.assertLess(loss.mean().item(), .5)
        loss.mean().backward()
        self.assertGreater(q.grad.abs().sum().item(), 0)
        self.assertGreater(coding.dictionary.grad.abs().sum().item(), 0)

    def test_user_memory_independent_and_explicit(self):
        model = scorer().eval()
        h, mask = torch.randn(2, 8, 6), torch.ones(2, 8, dtype=torch.bool)
        first = model.prepare_user(h, mask)
        model.prepare_items(torch.randn(1, 4, 6))
        second = model.prepare_user(h, mask)
        for name in ("keys", "values", "selected_indices", "summary_u", "summary_z", "deep_u"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "explicit user_memory"):
            model(torch.randn(2, 6), torch.randn(1, 4, 6))

    def test_chunking_preserves_gradients_and_auxiliary_weighting(self):
        model = scorer()
        h, embeddings = torch.randn(2, 9, 6), torch.randn(2, 5, 6)
        def backward(chunk):
            model.zero_grad(set_to_none=True)
            memory = model.prepare_user(h, torch.ones(2, 9, dtype=torch.bool))
            items = model.prepare_items(embeddings)
            scores = torch.cat([model.score_prepared(memory, items.slice_candidates(i, i+chunk),
                                                     return_auxiliary_losses=False)[0]
                                for i in range(0, 5, chunk)], dim=1)
            loss = scores.square().mean() + items.auxiliary_losses()["sparse_code_loss"]
            loss.backward()
            return scores.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        full, full_grad = backward(5)
        chunked, grad = backward(2)
        torch.testing.assert_close(full, chunked)
        self.assertEqual(full_grad.keys(), grad.keys())
        for name in grad:
            torch.testing.assert_close(grad[name], full_grad[name], atol=2e-6, rtol=2e-4, msg=name)

    def test_same_support_different_weights_changes_tail(self):
        model = scorer().eval()
        history = torch.randn(1, 15, 6)
        memory = model.prepare_user(history, torch.ones(1, 15, dtype=torch.bool))
        items = model.prepare_items(torch.randn(1, 1, 6))
        items = replace(items, active_indices=torch.tensor([[[[0, 1], [0, 1]]]]),
                        active_mask=torch.ones(1, 1, 2, 2, dtype=torch.bool))
        a = replace(items, activation_weights=torch.tensor([[[[.9, .1], [.9, .1]]]]))
        b = replace(items, activation_weights=torch.tensor([[[[.1, .9], [.1, .9]]]]))
        self.assertGreater((model._read_expert(memory, a, 0)[1] - model._read_expert(memory, b, 0)[1]).abs().sum().item(), 1e-6)

    def test_prepared_slicing_reload_and_staleness(self):
        model = scorer().eval()
        h, mask, x = torch.randn(3, 8, 6), torch.ones(3, 8, dtype=torch.bool), torch.randn(3, 4, 6)
        memory, items = model.prepare_user(h, mask), model.prepare_items(x)
        expected = model.score_prepared(memory, items)[0]
        actual = model.score_prepared(memory.index_batch(slice(1, 3)), items.index_batch(slice(1, 3)))[0]
        torch.testing.assert_close(actual, expected[1:])
        saved = io.BytesIO()
        torch.save(model.state_dict(), saved)
        saved.seek(0)
        other = scorer().eval()
        other.load_state_dict(torch.load(saved, weights_only=True))
        torch.testing.assert_close(other.score_prepared(other.prepare_user(h, mask), other.prepare_items(x))[0], expected)
        with torch.no_grad():
            model.history_key.weight.add_(.1)
        with self.assertRaisesRegex(ValueError, "Stale"):
            model.score_prepared(memory, items)

    def test_exact_chunked_topk_and_seen_filtering(self):
        from indexing.candidate_index import CandidateIndex
        model = scorer().eval()
        with torch.inference_mode():
            history = torch.randn(2, 9, 6)
            memory = model.prepare_user(history, torch.ones(2, 9, dtype=torch.bool))
            embeddings, ids = torch.randn(1, 9, 6), torch.arange(101, 110)[None, :]
            full = model.score_prepared(memory, model.prepare_items(embeddings, ids))[0]
            adapter = LearnedBruteForceTopK(model, embeddings, ids, candidate_chunk_size=2)
            scores, result_ids = adapter(history[:, 0], 4, user_memory=memory)
            expected, indices = full.topk(4, dim=1)
            torch.testing.assert_close(scores, expected)
            self.assertTrue(torch.equal(result_ids, ids.expand(2, -1).gather(1, indices)))
            seen = result_ids[:, :1]
            index = CandidateIndex(ids, embeddings)
            actual_ids, _, _ = index.get_top_k_outputs(history[:, 0], 3, {"user_memory": memory}, adapter, seen)
            self.assertFalse((actual_ids == seen).any())

    def test_shared_expert_memory_and_multiple_attention_heads(self):
        for settings in ({"share_expert_memory": True}, {"attention_heads_per_expert": 2},
                         {"use_tail_summaries": False}, {"activation_weighting": "uniform"}):
            model = scorer(**settings)
            memory = model.prepare_user(torch.randn(2, 8, 6), torch.ones(2, 8, dtype=torch.bool))
            items = model.prepare_items(torch.randn(1, 3, 6))
            scores, aux = model.score_prepared(memory, items)
            (scores.sum() + aux["sparse_code_loss"]).backward()
            self.assertTrue(torch.isfinite(scores).all())


class PrefixTests(unittest.TestCase):
    def test_seeded_cuts_mask_future_inputs_without_mutation(self):
        ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 0, 0, 0, 0]])
        features = SequentialFeatures(torch.tensor([4, 2]), ids, None,
                                      {"timestamps": ids * 100, "user_ids": torch.tensor([9, 10])})
        generator = torch.Generator().manual_seed(17)
        prefix, targets = sample_causal_prefix(features, torch.tensor([7, 8]), generator)
        other, other_targets = sample_causal_prefix(features, torch.tensor([7, 8]), torch.Generator().manual_seed(17))
        self.assertTrue(torch.equal(prefix.past_ids, other.past_ids))
        self.assertTrue(torch.equal(targets, other_targets))
        self.assertTrue(torch.equal(features.past_ids, ids))
        for row in range(2):
            cut = prefix.past_lengths[row]
            self.assertEqual(prefix.past_ids[row, cut:].count_nonzero(), 0)
            self.assertEqual(prefix.past_payloads["timestamps"][row, cut:].count_nonzero(), 0)


def small_backbone(backbone="SASRec", device="cpu", **scorer_options):
    import fbgemm_gpu  # noqa: F401
    from modeling.sequential.embedding_modules import LocalEmbeddingModule
    from modeling.sequential.encoder_utils import get_sequential_encoder
    from modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
    from modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor
    with contextlib.redirect_stdout(io.StringIO()):
        model = get_sequential_encoder(
            module_type=backbone, max_sequence_length=8, max_output_length=3,
            embedding_module=LocalEmbeddingModule(40, 6), interaction_module=scorer(**scorer_options),
            input_preproc_module=LearnablePositionalEmbeddingInputFeaturesPreprocessor(11, 6, 0),
            output_postproc_module=L2NormEmbeddingPostprocessor(6, 1e-6), verbose=False)
    return model.to(device)


def configured_backbone(config, num_items, device="cpu"):
    import gin
    import train  # Gin registration only.
    from modeling.sequential.embedding_modules import LocalEmbeddingModule
    from modeling.sequential.encoder_utils import get_sequential_encoder
    from modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
    from modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor, LayerNormEmbeddingPostprocessor
    from modeling.similarity_utils import get_similarity_function
    gin.clear_config()
    gin.parse_config_file(str(config))
    dim = gin.query_parameter("train_fn.item_embedding_dim")
    interaction, _ = get_similarity_function(gin.query_parameter("train_fn.interaction_module_type"), dim, dim)
    post = (L2NormEmbeddingPostprocessor if gin.query_parameter("train_fn.user_embedding_norm") == "l2_norm"
            else LayerNormEmbeddingPostprocessor)
    length = gin.query_parameter("train_fn.max_sequence_length")
    with contextlib.redirect_stdout(io.StringIO()):
        model = get_sequential_encoder(
            module_type=gin.query_parameter("train_fn.main_module"), max_sequence_length=length,
            max_output_length=11, embedding_module=LocalEmbeddingModule(num_items, dim),
            interaction_module=interaction,
            input_preproc_module=LearnablePositionalEmbeddingInputFeaturesPreprocessor(length + 11, dim, .2),
            output_postproc_module=post(dim, 1e-6), verbose=False)
    gin.clear_config()
    return model.to(device)


class IntegrationTests(unittest.TestCase):
    def test_sampled_prefix_training_inside_ddp_forward(self):
        import tempfile
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        from modeling.sequential.autoregressive_losses import LocalNegativesSampler
        from modeling.sequential.losses.sampled_softmax import SampledSoftmaxLoss
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group("gloo", init_method="file://" + tmp + "/group", rank=0, world_size=1)
            try:
                model = small_backbone(use_tail_summaries=False)
                ddp = DistributedDataParallel(model, find_unused_parameters=True)
                sampler = LocalNegativesSampler(40, model._embedding_module._item_emb,
                                                list(range(1, 41)), True, 1e-6)
                criterion = SampledSoftmaxLoss(3, 1., model, candidate_chunk_size=2)
                optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
                ids = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0],
                                    [5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0]])
                for _ in range(2):
                    optimizer.zero_grad()
                    loss, aux = ddp(torch.tensor([4, 3]), ids, model.get_item_embeddings(ids),
                                    {"user_ids": torch.tensor([1, 2])}, sampled_targets=torch.tensor([8, 9]),
                                    negatives_sampler=sampler, sampled_loss=criterion)
                    total = loss + .1 * aux["sparse_code_loss"]
                    total.backward()
                    self.assertTrue(torch.isfinite(total))
                    optimizer.step()
                    model._ndp_module.post_optimizer_step()
            finally:
                dist.destroy_process_group()

    @unittest.skipUnless(torch.cuda.is_available(), "HSTU jagged kernels require CUDA")
    def test_both_backbones_real_batch_train_reload_and_evaluate(self):
        import gin
        import fbgemm_gpu  # noqa: F401
        from data.reco_dataset import get_reco_dataset
        from data.eval import get_eval_state, eval_metrics_v2_from_tensors
        from indexing.utils_rails import get_top_k_module
        from modeling.sequential.features import movielens_seq_features_from_row
        from modeling.sequential.autoregressive_losses import LocalNegativesSampler
        from modeling.sequential.losses.sampled_softmax import SampledSoftmaxLoss
        from torch.utils.data._utils.collate import default_collate
        if not (ROOT / "tmp/ml-1m/sasrec_format.csv").exists():
            self.skipTest("Local ML-1M fixture not present; no downloads in correctness tests")
        with torch.compiler.set_stance("force_eager"):
            dataset = get_reco_dataset("ml-1m", 200, chronological=True)
            row = default_collate([dataset.train_dataset[i] for i in (0, 1)])
            features, targets, _ = movielens_seq_features_from_row(row, "cuda", 11)
            for backbone in ("SASRec", "HSTU"):
                gin.clear_config()
                model = configured_backbone(ROOT / f"configs/ml-1m/{backbone.lower()}-expert-history.gin",
                                             dataset.max_item_id, "cuda")
                prefix, next_ids = sample_causal_prefix(features, targets,
                                                        torch.Generator(device="cuda").manual_seed(7))
                sampler = LocalNegativesSampler(dataset.max_item_id, model._embedding_module._item_emb,
                                                dataset.all_item_ids, True, 1e-6).cuda()
                criterion = SampledSoftmaxLoss(3, 1., model, activation_checkpoint=True,
                                               candidate_chunk_size=2)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                model.train()
                loss, aux = model(prefix.past_lengths, prefix.past_ids,
                                  model.get_item_embeddings(prefix.past_ids), prefix.past_payloads,
                                  sampled_targets=next_ids, negatives_sampler=sampler, sampled_loss=criterion)
                total = loss + .1 * aux["sparse_code_loss"]
                total.backward()
                self.assertTrue(torch.isfinite(total))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertIsNotNone(model._embedding_module._item_emb.weight.grad)
                self.assertGreater(model._embedding_module._item_emb.weight.grad.abs().sum().item(), 0)
                encoder_parameters = (list(model.attention_layers.parameters()) if backbone == "SASRec"
                                      else list(model._hstu.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0
                                    for p in encoder_parameters))
                optimizer.step()
                model._ndp_module.post_optimizer_step()
                model.eval()
                with torch.inference_mode():
                    h, memory = model.encode_with_history(prefix.past_lengths, prefix.past_ids,
                                                          model.get_item_embeddings(prefix.past_ids), prefix.past_payloads)
                    ids = torch.tensor(dataset.all_item_ids[:12], device="cuda")[None, :]
                    items = model._ndp_module.prepare_items(sampler.normalize_embeddings(model.get_item_embeddings(ids)), ids)
                    scores = model._ndp_module.score_prepared(memory, items)[0]
                    checkpoint = io.BytesIO()
                    torch.save(model.state_dict(), checkpoint)
                    checkpoint.seek(0)
                    model.load_state_dict(torch.load(checkpoint, weights_only=True))
                    h, memory = model.encode_with_history(prefix.past_lengths, prefix.past_ids,
                                                          model.get_item_embeddings(prefix.past_ids), prefix.past_payloads)
                    items = model._ndp_module.prepare_items(sampler.normalize_embeddings(model.get_item_embeddings(ids)), ids)
                    torch.testing.assert_close(model._ndp_module.score_prepared(memory, items)[0], scores)
                    state = get_eval_state(model, dataset.all_item_ids[:12], sampler,
                                           lambda emb, ids: get_top_k_module("LearnedBruteForceTopK", model, emb, ids,
                                                                            candidate_chunk_size=3), "cuda")
                    target = ids[:, :1].expand(2, -1)
                    first = eval_metrics_v2_from_tensors(state, model, prefix, target, user_max_batch_size=1)
                    second = eval_metrics_v2_from_tensors(state, model, prefix, target, user_max_batch_size=2)
                    for key in first:
                        torch.testing.assert_close(first[key], second[key])

    def test_published_checkpoints_still_load_strictly(self):
        import gin
        from playground.model_registry import BASELINE_MODELS, MODEL_SPECS
        from indexing.utils_rails import get_exact_top_k_method
        directory = ROOT / "playground/benchmark_runs/20261026-201653"
        if not all((directory / name / "best.pt").exists() for name in BASELINE_MODELS):
            self.skipTest("Local baseline checkpoints not present")
        for name in BASELINE_MODELS:
            state = torch.load(directory / name / "best.pt", map_location="cpu", weights_only=True)
            weights = {key.removeprefix("module."): value for key, value in state["model_state_dict"].items()}
            n = weights["_embedding_module._item_emb.weight"].shape[0] - 1
            model = configured_backbone(ROOT / "configs/ml-1m" / MODEL_SPECS[name].config, n)
            model.load_state_dict(weights, strict=True)
            self.assertEqual(get_exact_top_k_method(model), MODEL_SPECS[name].exact_top_k)
        gin.clear_config()

    def test_encoder_changes_invalidate_history_memory(self):
        model = small_backbone().eval()
        ids = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]])
        h, memory = model.encode_with_history(torch.tensor([3]), ids, model.get_item_embeddings(ids), {})
        items = model._ndp_module.prepare_items(torch.randn(1, 2, 6))
        with torch.no_grad():
            model._embedding_module._item_emb.weight.add_(.01)
        with self.assertRaisesRegex(ValueError, "Stale history encoder"):
            model._ndp_module.score_prepared(memory, items)

    def test_embedding_changes_invalidate_catalog_cache(self):
        from indexing.utils_rails import get_top_k_module
        model = small_backbone().eval()
        ids = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]])
        with torch.inference_mode():
            catalog_ids = torch.arange(1, 9)[None, :]
            adapter = get_top_k_module("LearnedBruteForceTopK", model,
                                       model.get_item_embeddings(catalog_ids), catalog_ids)
            model._embedding_module._item_emb.weight.add_(.01)
            h, memory = model.encode_with_history(torch.tensor([3]), ids, model.get_item_embeddings(ids), {})
            with self.assertRaisesRegex(ValueError, "Stale catalog embedding"):
                adapter(h, 2, user_memory=memory)

    @unittest.skipUnless(torch.cuda.is_available(), "HSTU jagged kernels require CUDA")
    def test_future_events_do_not_change_prefix_memory(self):
        with torch.compiler.set_stance("force_eager"), torch.inference_mode():
            for backbone in ("SASRec", "HSTU"):
                model = small_backbone(backbone, "cuda").eval()
                ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0]], device="cuda")
                lengths = torch.tensor([3], device="cuda")
                first = model.encode_with_history(lengths, ids, model.get_item_embeddings(ids), {})[1]
                changed = ids.clone()
                changed[:, 3:] = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27], device="cuda")
                second = model.encode_with_history(lengths, changed, model.get_item_embeddings(changed), {})[1]
                for name in ("keys", "values", "summary_u", "summary_z", "selected_indices"):
                    torch.testing.assert_close(getattr(first, name), getattr(second, name))

    def test_gin_configs_construct_new_scorer(self):
        import gin
        import train  # Register training bindings without calling its entry point.
        from modeling.similarity_utils import get_similarity_function
        for name in ("sasrec", "hstu"):
            gin.clear_config()
            gin.parse_config_file(str(ROOT / f"configs/ml-1m/{name}-expert-history.gin"))
            dim = gin.query_parameter("train_fn.item_embedding_dim")
            module, description = get_similarity_function("ExpertHistory", dim, dim)
            self.assertEqual(module.num_item_experts, 2)
            self.assertIn("ExpertHistory", description)
            self.assertEqual(gin.query_parameter("train_fn.supervision_mode"), "sampled_prefix")
        gin.clear_config()

    def test_prepared_operation_accounting_and_boundaries(self):
        from playground.inference_flops_counter import ArithmeticLedger, sum_totals
        from torch.utils.flop_counter import FlopCounterMode
        model = scorer().eval()
        h, mask, x = torch.randn(2, 7, 6), torch.ones(2, 7, dtype=torch.bool), torch.randn(1, 3, 6)
        with torch.inference_mode():
            model.prepare_model()
            with ArithmeticLedger(bucket="U") as u:
                memory = model.prepare_user(h, mask)
            with ArithmeticLedger(bucket="I") as i:
                items = model.prepare_items(x)
            forbidden = []
            hooks = [module.register_forward_pre_hook(lambda *args: forbidden.append(True))
                     for module in (model.history_key, model.history_value, model.value_query,
                                    model.expert_gate, model.sparse_coder, *model.item_experts,
                                    *model.memory_projections)]
            try:
                with ArithmeticLedger(bucket="S") as s:
                    actual = model.score_prepared(memory, items)[0]
                self.assertFalse(forbidden)
            finally:
                for hook in hooks:
                    hook.remove()
            with ArithmeticLedger() as total:
                composed = model.score_prepared(model.prepare_user(h, mask), model.prepare_items(x))[0]
            torch.testing.assert_close(actual, composed)
            self.assertEqual(total.totals(), sum_totals(u.totals(), i.totals(), s.totals()))
            with FlopCounterMode(display=False) as profiler:
                model.score_prepared(memory, items)
            self.assertEqual(profiler.get_total_flops(), s.totals()["matrix_flops"])

    def test_inference_adapter_uses_explicit_memory_and_item_boundaries(self):
        from playground.benchmark_inference_flops import ExpertHistoryInference, counted
        model = small_backbone().eval()
        ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0],
                            [6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]])
        fixture = ids, torch.tensor([5, 3]), {"user_ids": torch.tensor([1, 2])}
        candidates = torch.arange(1, 5)
        adapter = ExpertHistoryInference(model, candidate_chunk_size=2)
        try:
            with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
                counted(adapter, adapter.prepare_model, "model_only")
                user, _, _ = counted(adapter, lambda: adapter.prepare_user(fixture), "U")
                items, _, _ = counted(adapter, lambda: adapter.prepare_items(candidates), "I")
                actual, _, calls = counted(adapter, lambda: adapter.score(user, items), "S")
                self.assertTrue(all(value == 0 for value in calls.values()))
                torch.testing.assert_close(actual, adapter.original(fixture, candidates))
        finally:
            adapter.close()


class ToolingTests(unittest.TestCase):
    def test_checkpoint_evaluator_accepts_both_configs_without_execution(self):
        import subprocess
        code = """
import gin
import eval_from_checkpoint
from modeling.similarity_utils import get_similarity_function
for name in ('sasrec', 'hstu'):
    gin.clear_config()
    gin.parse_config_file(f'configs/ml-1m/{name}-expert-history.gin')
    scorer, _ = get_similarity_function('ExpertHistory', 50, 50)
    assert scorer.num_item_experts == 2
    assert gin.query_parameter('train_fn.supervision_mode') == 'sampled_prefix'
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reports_handle_new_models_without_published_targets(self):
        import json
        import tempfile
        from playground.run_ml1m_benchmark import report, METRICS, EXTRA_METRICS
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("sasrec", "sasrec_expert_history"):
                folder = directory / name
                folder.mkdir()
                row = {"epoch": 0, "metrics": {key: .1 for key in METRICS + EXTRA_METRICS},
                       "elapsed_seconds": 1., "peak_memory_bytes": 1,
                       "target_exposures": 2, "supervision_mode": "sampled_prefix"}
                (folder / "epochs.jsonl").write_text(json.dumps(row) + "\n")
            report(directory)
            text = (directory / "comparison.md").read_text()
            self.assertIn("SASRec + ExpertHistory", text)
            self.assertIn("NDCG", text)

    def test_batch_evaluation_quotes_paths_and_selects_exact_method(self):
        import shlex
        from eval_batch import get_cmd, eval as batch_eval
        command = get_cmd("/tmp/config with spaces.gin", "/tmp/ckpt $(literal).pt",
                          2, "LearnedBruteForceTopK", 0, "")
        self.assertIn("--inference_from_ckpt=/tmp/ckpt $(literal).pt", shlex.split(command))
        with self.assertRaisesRegex(ValueError, "requires its config and checkpoint"):
            batch_eval("ml-1m", 2, scorer_type="ExpertHistory")


if __name__ == "__main__":
    unittest.main()
