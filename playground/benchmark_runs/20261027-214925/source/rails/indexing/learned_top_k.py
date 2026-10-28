"""Exact catalog retrieval using explicitly prepared history and item states."""

import torch
import weakref

from .candidate_index import TopKModule


class LearnedBruteForceTopK(TopKModule):
    def __init__(self, scorer, item_embeddings, item_ids, candidate_chunk_size=256, source_model=None):
        super().__init__()
        if candidate_chunk_size < 1:
            raise ValueError("candidate_chunk_size must be positive")
        if scorer.training:
            raise ValueError("Catalog caches require an evaluation-mode scorer")
        self.scorer = scorer
        self.candidate_chunk_size = candidate_chunk_size
        self._source_model = weakref.ref(source_model) if source_model is not None else None
        self._source_version = source_model.history_version() if source_model is not None else None
        with torch.inference_mode():
            self.items = scorer.prepare_items(item_embeddings, item_ids)

    @torch.inference_mode()
    def forward(self, query_embeddings, k, sorted=True, **kwargs):
        memory = kwargs.get("user_memory")
        if self.items.version != self.scorer.version():
            raise ValueError("Stale catalog state; rebuild after changing scorer weights")
        if self._source_model is not None:
            source = self._source_model()
            if source is None or source.history_version() != self._source_version:
                raise ValueError("Stale catalog embedding state; rebuild the evaluation index")
        if memory is None or memory.batch_size != query_embeddings.shape[0]:
            raise ValueError("Exact history retrieval requires aligned user_memory")
        if not 0 <= k <= self.items.num_candidates:
            raise ValueError("k must lie between zero and catalog size")
        batch = memory.batch_size
        best_scores = query_embeddings.new_empty(batch, 0, dtype=torch.float32)
        best_ids = self.items.item_ids.new_empty(batch, 0)
        for start in range(0, self.items.num_candidates, self.candidate_chunk_size):
            chunk = self.items.slice_candidates(start, start + self.candidate_chunk_size)
            scores, _ = self.scorer.score_prepared(memory, chunk, return_auxiliary_losses=False)
            ids = chunk.item_ids.expand(batch, -1)
            candidates = torch.cat([best_scores, scores], dim=1)
            candidate_ids = torch.cat([best_ids, ids], dim=1)
            best_scores, index = torch.topk(candidates, min(k, candidates.shape[1]),
                                           dim=1, sorted=sorted)
            best_ids = candidate_ids.gather(1, index)
        return best_scores, best_ids
