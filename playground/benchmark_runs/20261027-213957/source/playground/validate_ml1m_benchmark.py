"""Numerical and protocol checks for the ML-1M reproduction."""

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fbgemm_gpu  # noqa: F401
import gin
import torch
import train  # Registers configurable training entry points.
from data.eval import eval_metrics_v2_from_tensors
from indexing.candidate_index import CandidateIndex
from rails.indexing.mips_top_k import MIPSBruteForceTopK
from modeling.sequential.features import SequentialFeatures
from modeling.sequential.embedding_modules import LocalEmbeddingModule
from modeling.sequential.encoder_utils import get_sequential_encoder
from modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
from modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor, LayerNormEmbeddingPostprocessor
from modeling.similarity_utils import get_similarity_function
from trainer.data_loader import ReusableDataLoader


def loader_check():
    def collect(persistent):
        torch.manual_seed(123)
        loader = ReusableDataLoader(torch.utils.data.TensorDataset(torch.arange(20)),
            batch_size=4, num_workers=1, persistent_workers=persistent)
        result = []
        for _ in range(3):
            result.append(([batch[0].tolist() for batch in loader], torch.rand(3).tolist()))
        return result
    assert collect(False) == collect(True)
    print("PASS persistent-worker sample order and parent RNG equivalence", flush=True)


def jagged_check():
    x = torch.randn(5, 4, device="cuda", requires_grad=True)
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(torch.tensor([2, 3], device="cuda"))
    padded = torch.ops.fbgemm.jagged_to_padded_dense(x, [offsets], [3])
    restored = torch.ops.fbgemm.dense_to_jagged(padded, [offsets])[0]
    torch.testing.assert_close(restored, x)
    restored.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x)
    print("PASS CUDA jagged roundtrip and analytic gradient", flush=True)


def metrics_check():
    class Model:
        _ndp_module = SimpleNamespace(requires_user_memory=False)

        def get_item_embeddings(self, ids):
            return ids.float().unsqueeze(-1)

        def encode(self, **kwargs):
            return torch.zeros(4, 1, device="cuda")

    class Candidates:
        ids = torch.arange(1, 2502, device="cuda").unsqueeze(0)

        def get_top_k_outputs(self, **kwargs):
            ids = torch.arange(1, 2501, device="cuda").expand(4, -1)
            return ids, torch.zeros_like(ids), None

    features = SequentialFeatures(torch.ones(4, device="cuda", dtype=torch.long),
        torch.zeros(4, 2, device="cuda", dtype=torch.long), None, {})
    state = SimpleNamespace(all_item_ids=set(range(1, 2502)), candidate_index=Candidates(), top_k_module=None)
    out = eval_metrics_v2_from_tensors(state, Model(), features,
        torch.tensor([[1], [10], [200], [2501]], device="cuda"))
    assert out["hr@10"].tolist() == [True, True, False, False]
    assert out["hr@200"].tolist() == [True, True, True, False]
    torch.testing.assert_close(out["mrr"], torch.tensor([1., .1, .005, 1 / 2501], device="cuda"))
    ids = torch.arange(1, 6, device="cuda").unsqueeze(0)
    embeddings = torch.arange(5., 0., -1., device="cuda").reshape(1, 5, 1)
    candidates = CandidateIndex(ids=ids, embeddings=embeddings)
    top_ids, _, _ = candidates.get_top_k_outputs(torch.ones(1, 1, device="cuda"), 2, {},
        MIPSBruteForceTopK(embeddings, ids), torch.tensor([[1, 2]], device="cuda"))
    assert top_ids.tolist() == [[3, 4]]
    print("PASS known-rank HR/MRR and seen-item filtering", flush=True)


def causal_checks():
    for config in sorted(Path("configs/ml-1m").glob("*.gin")):
        gin.clear_config()
        gin.parse_config_file(str(config))
        dimension = gin.query_parameter("train_fn.item_embedding_dim")
        norm = gin.query_parameter("train_fn.user_embedding_norm")
        interaction, _ = get_similarity_function(
            module_type=gin.query_parameter("train_fn.interaction_module_type"),
            query_embedding_dim=dimension, item_embedding_dim=dimension)
        preproc = LearnablePositionalEmbeddingInputFeaturesPreprocessor(211, dimension, .2)
        model = get_sequential_encoder(
            module_type=gin.query_parameter("train_fn.main_module"),
            max_sequence_length=200, max_output_length=11,
            embedding_module=LocalEmbeddingModule(3952, dimension),
            interaction_module=interaction, input_preproc_module=preproc,
            output_postproc_module=(L2NormEmbeddingPostprocessor if norm == "l2_norm"
                else LayerNormEmbeddingPostprocessor)(dimension, 1e-6), verbose=False).cuda().eval()
        ids = torch.zeros(2, 211, device="cuda", dtype=torch.long)
        ids[:, :6] = torch.arange(1, 7, device="cuda")
        lengths = torch.tensor([6, 6], device="cuda")
        payloads = {"timestamps": torch.arange(211, device="cuda").expand(2, -1),
                    "user_ids": torch.tensor([1, 2], device="cuda")}
        with torch.no_grad():
            emb = model.get_item_embeddings(ids)
            _, prepared, mask = preproc.cuda()(lengths, ids, emb, payloads)
            assert torch.count_nonzero(prepared[:, 6:]) == 0
            assert torch.count_nonzero(mask[:, 6:]) == 0
            before = model(lengths, ids, emb, payloads)
            changed = ids.clone()
            changed[:, 4:6] += 10
            after = model(lengths, changed, model.get_item_embeddings(changed), payloads)
            torch.testing.assert_close(before[:, :4], after[:, :4], rtol=1e-5, atol=1e-6)
        print(f"PASS padding mask and causal prefix: {config.name}", flush=True)
        del model


if __name__ == "__main__":
    loader_check()
    jagged_check()
    metrics_check()
    causal_checks()
    print("ALL_VALIDATION_OK", flush=True)
