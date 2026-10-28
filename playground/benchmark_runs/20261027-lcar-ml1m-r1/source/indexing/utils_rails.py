# Retrieval with Learned Similarities (RAILS).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Defines an utility function to invoke different TopK modules.

import torch
import gin

from indexing.candidate_index import CandidateIndex
from rails.indexing.candidate_index import TopKModule
from rails.indexing.mol_top_k import MoLBruteForceTopK, MoLNaiveTopK, MoLAvgTopK, MoLCombTopK
from rails.indexing.mips_top_k import MIPSBruteForceTopK
from rails.indexing.learned_top_k import LearnedBruteForceTopK


def get_exact_top_k_method(model):
    scorer = getattr(model, "_ndp_module", model)
    method = getattr(scorer, "exact_top_k_method", None)
    if method is not None:
        return method
    from rails.similarities.dot_product_similarity_fn import DotProductSimilarity
    from rails.similarities.mol.similarity_fn import MoLSimilarity
    if isinstance(scorer, MoLSimilarity):
        return "MoLBruteForceTopK"
    if isinstance(scorer, DotProductSimilarity):
        return "MIPSBruteForceTopK"
    raise ValueError(f"No exact retrieval method for {type(scorer).__name__}")


@gin.configurable
def get_top_k_module(top_k_method: str, model: torch.nn.Module, item_embeddings: torch.Tensor, item_ids: torch.Tensor, candidate_chunk_size: int = 256) -> TopKModule:
    if top_k_method == "LearnedBruteForceTopK":
        return LearnedBruteForceTopK(model._ndp_module, item_embeddings, item_ids,
                                     candidate_chunk_size, source_model=model)
    if top_k_method == "MIPSBruteForceTopK":
        top_k_module = MIPSBruteForceTopK(
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
    elif top_k_method == "MoLBruteForceTopK":
        top_k_module = MoLBruteForceTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
    elif top_k_method == "MoLNaiveTopK25":
        top_k_module = MoLNaiveTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group=25,
        )
    elif top_k_method == "MoLNaiveTopK5":
        top_k_module = MoLNaiveTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group=5,
        )
    elif top_k_method == "MoLNaiveFaissTopK5":
        top_k_module = MoLNaiveTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group=5,
            use_faiss=True,
        )
    elif top_k_method == "MoLNaiveTopK10":
        top_k_module = MoLNaiveTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group=10,
        )
    elif top_k_method == "MoLNaiveTopK50":
        top_k_module = MoLNaiveTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group=50,
        )
    elif top_k_method == "MoLNaiveTopK75":
        top_k_module = MoLNaiveTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group=75,
        )
    elif top_k_method == "MoLNaiveTopK100":
        top_k_module = MoLNaiveTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group=100,
        )
    elif top_k_method == "MoLAvgTopK100":
        top_k_module = MoLAvgTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            avg_top_k=100,
        )
    elif top_k_method == "MoLAvgTopK200":
        top_k_module = MoLAvgTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            avg_top_k=200,
        )
    elif top_k_method == "MoLAvgTopK500":
        top_k_module = MoLAvgTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            avg_top_k=500,
        )
    elif top_k_method == "MoLAvgTopK1000":
        top_k_module = MoLAvgTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            avg_top_k=1000,
        )
    elif top_k_method == "MoLAvgTopK2000":
        top_k_module = MoLAvgTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            avg_top_k=2000,
        )
    elif top_k_method == "MoLAvgTopK2500":
        top_k_module = MoLAvgTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            avg_top_k=2500,
        )
    elif top_k_method == "MoLAvgTopK3000":
        top_k_module = MoLAvgTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            avg_top_k=3000,
        )
    elif top_k_method == "MoLAvgTopK4000":
        top_k_module = MoLAvgTopK(
        mol_module=model._ndp_module,
        item_embeddings=item_embeddings,
        item_ids=item_ids,
        avg_top_k=4000,
    )
    elif top_k_method == "MoLCombTopK5_100":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 5,
            avg_top_k=100,
        )
    elif top_k_method == "MoLCombTopK5_200":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 5,
            avg_top_k=200,
        )
    elif top_k_method == "MoLCombTopK5_500":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 5,
            avg_top_k=500,
        )
    elif top_k_method == "MoLCombTopK1_100":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 1,
            avg_top_k=100,
        )
    elif top_k_method == "MoLCombTopK10_100":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 10,
            avg_top_k=100,
        )
    elif top_k_method == "MoLCombTopK1_500":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 1,
            avg_top_k=500,
        )
    elif top_k_method == "MoLCombTopK10_100":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 10,
            avg_top_k=100,
        )
    elif top_k_method == "MoLCombTopK10_500":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 10,
            avg_top_k=500,
        )
    elif top_k_method == "MoLCombTopK50_500":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 50,
            avg_top_k=500,
        )
    elif top_k_method == "MoLCombTopK50_1000":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 50,
            avg_top_k=1000,
        )
    elif top_k_method == "MoLCombTopK100_1000":
        top_k_module = MoLCombTopK(
            mol_module=model._ndp_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group = 100,
            avg_top_k=1000,
        )
    else:
        raise ValueError(f"Invalid top-k method {top_k_method}")
    return top_k_module
