"""Shared retrieval fixtures and baseline compatibility tests (no benchmarks)."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
import torch
from modeling.sequential.features import SequentialFeatures
from modeling.sequential.prefix_sampling import sample_causal_prefix, full_causal_prefix
from rails.similarities.dot_product_similarity_fn import DotProductSimilarity
ROOT = Path(__file__).resolve().parents[1]


class PrefixTests(unittest.TestCase):
    def test_full_prefix_keeps_history_and_supplied_label_without_future_payloads(self):
        ids=torch.tensor([[1,2,3,4,99,0],[5,6,99,0,0,0]])
        times=ids*100
        features=SequentialFeatures(torch.tensor([4,2]),ids,None,
            {'timestamps':times,'user_ids':torch.tensor([9,10])})
        supplied=torch.tensor([[7],[8]])
        prefix,targets=full_causal_prefix(features,supplied)
        self.assertTrue(torch.equal(prefix.past_lengths,features.past_lengths))
        self.assertTrue(torch.equal(targets,torch.tensor([7,8])))
        self.assertTrue(torch.equal(features.past_ids,ids))
        self.assertTrue(torch.equal(features.past_payloads['timestamps'],times))
        for row,length in enumerate(prefix.past_lengths):
            self.assertTrue(torch.equal(prefix.past_ids[row,:length],ids[row,:length]))
            self.assertEqual(prefix.past_ids[row,length:].count_nonzero(),0)
            self.assertEqual(prefix.past_payloads['timestamps'][row,length:].count_nonzero(),0)

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

@torch.inference_mode(False)
def small_backbone(backbone="SASRec", device="cpu", activation_checkpoint=False,
                   interaction=None):
    import fbgemm_gpu  # noqa: F401
    from modeling.sequential.embedding_modules import LocalEmbeddingModule
    from modeling.sequential.encoder_utils import get_sequential_encoder
    from modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
    from modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor
    with contextlib.redirect_stdout(io.StringIO()):
        model = get_sequential_encoder(
            module_type=backbone, max_sequence_length=8, max_output_length=3,
            embedding_module=LocalEmbeddingModule(40, 6),
            interaction_module=DotProductSimilarity() if interaction is None else interaction,
            input_preproc_module=LearnablePositionalEmbeddingInputFeaturesPreprocessor(11, 6, 0),
            output_postproc_module=L2NormEmbeddingPostprocessor(6, 1e-6), verbose=False,
            activation_checkpoint=activation_checkpoint)
    return model.to(device)

@torch.inference_mode(False)
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

class BaselineTests(unittest.TestCase):
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

class ToolingTests(unittest.TestCase):
    def test_checkpoint_evaluator_accepts_both_configs_without_execution(self):
        import subprocess
        code = """
import gin
import eval_from_checkpoint
from modeling.similarity_utils import get_similarity_function
for name in ('sasrec', 'hstu'):
    gin.clear_config()
    gin.parse_config_file(f'configs/ml-1m/{name}-lcar.gin')
    scorer, _ = get_similarity_function('LCAR', 50, 50)
    assert scorer.num_item_experts == 4
    assert gin.query_parameter('train_fn.supervision_mode') == 'full_prefix'
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reports_handle_new_models_without_published_targets(self):
        import json
        import tempfile
        from playground.run_ml1m_benchmark import report, METRICS, EXTRA_METRICS
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("sasrec", "sasrec_lcar"):
                folder = directory / name
                folder.mkdir()
                row = {"epoch": 0, "metrics": {key: .1 for key in METRICS + EXTRA_METRICS},
                       "elapsed_seconds": 1., "peak_memory_bytes": 1,
                       "target_exposures": 2, "supervision_mode": "sampled_prefix"}
                (folder / "epochs.jsonl").write_text(json.dumps(row) + "\n")
            report(directory)
            text = (directory / "comparison.md").read_text()
            self.assertIn("LCAR-SASRec", text)
            self.assertIn("NDCG", text)

    def test_batch_evaluation_quotes_paths_and_selects_exact_method(self):
        import shlex
        from eval_batch import get_cmd, eval as batch_eval
        command = get_cmd("/tmp/config with spaces.gin", "/tmp/ckpt $(literal).pt",
                          2, "LearnedBruteForceTopK", 0, "")
        self.assertIn("--inference_from_ckpt=/tmp/ckpt $(literal).pt", shlex.split(command))
        with self.assertRaisesRegex(ValueError, "requires its config and checkpoint"):
            batch_eval("ml-1m", 2, scorer_type="LCAR")
