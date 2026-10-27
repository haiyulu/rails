# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import abc
from dataclasses import replace
import weakref
from typing import Dict, Optional, Tuple

import torch

from rails.similarities.module import SimilarityModule


class SequentialEncoderWithLearnedSimilarityModule(torch.nn.Module):
    def __init__(
        self,
        ndp_module: SimilarityModule,
    ) -> None:
        super().__init__()

        self._ndp_module: SimilarityModule = ndp_module

    def history_version(self):
        return tuple((id(p), None if p.is_inference() else p._version, p.device, p.dtype)
                     for name, p in self.named_parameters() if not name.startswith("_ndp_module."))

    def prepare_history(self, encoded_embeddings, past_lengths, past_ids, past_payloads):
        """Prepare one explicit state for a causal prefix, never a candidate-dependent cache."""
        if encoded_embeddings.shape[:2] != past_ids.shape:
            raise ValueError("History memory requires one encoder state per event")
        positions = torch.arange(past_ids.shape[1], device=past_ids.device)
        valid = (positions[None, :] < past_lengths[:, None]) & (past_ids != 0)
        last = (past_lengths - 1).clamp_min(0)
        current = encoded_embeddings[torch.arange(past_ids.shape[0], device=past_ids.device), last]
        current = current * (past_lengths > 0).unsqueeze(-1)
        memory = None
        if getattr(self._ndp_module, "requires_user_memory", False):
            payloads = dict(past_payloads)
            payloads["item_ids"] = past_ids
            memory = self._ndp_module.prepare_user(encoded_embeddings, valid, payloads)
            memory = replace(memory, encoder_ref=weakref.ref(self), encoder_version=self.history_version())
        return current, memory

    def encode_with_history(self, past_lengths, past_ids, past_embeddings, past_payloads):
        states = self.forward(past_lengths, past_ids, past_embeddings, past_payloads)
        return self.prepare_history(states, past_lengths, past_ids, past_payloads)

    def sampled_prefix_loss(self, states, past_lengths, past_ids, past_payloads,
                            sampled_targets, negatives_sampler, sampled_loss):
        current, memory = self.prepare_history(states, past_lengths, past_ids, past_payloads)
        return sampled_loss.prefix_forward(current, sampled_targets, negatives_sampler,
                                           user_memory=memory, **past_payloads)

    @abc.abstractmethod
    def debug_str(
        self,
    ) -> str:
        pass

    def similarity_fn(
        self,
        query_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        item_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        torch._assert(
            len(query_embeddings.size()) == 2, "len(query_embeddings.size()) must be 2"
        )
        torch._assert(len(item_ids.size()) == 2, "len(item_ids.size()) must be 2")
        if item_embeddings is None:
            item_embeddings = self.get_item_embeddings(item_ids)
        torch._assert(
            len(item_embeddings.size()) == 3, "len(item_embeddings.size()) must be 3"
        )

        return self._ndp_module(
            query_embeddings=query_embeddings,  # (B, query_embedding_dim)
            item_embeddings=item_embeddings,  # (1/B, X, item_embedding_dim)
            # item_sideinfo=self.get_item_sideinfo(item_ids=item_ids),  # (1/B, X, item_sideinfo_dim)
            item_ids=item_ids,
            **kwargs,
        )
