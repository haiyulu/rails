"""User-only sparse-code memory and actual-item, expert-specific history reads."""

import math
from typing import Optional

import torch
from torch import nn

from rails.similarities.module import SimilarityModule
from .sparse_coding import NonnegativeSparseCoding
from .states import ItemState, UserMemory


def masked_softmax(logits, mask):
    logits = logits.masked_fill(~mask, -torch.inf)
    maximum = logits.amax(-1, keepdim=True)
    maximum = torch.where(mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
    weights = (logits - maximum).exp()
    return weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


class ExpertHistorySimilarity(SimilarityModule):
    requires_user_memory = True
    scorer_type = "ExpertHistory"
    exact_top_k_method = "LearnedBruteForceTopK"

    @torch.inference_mode(False)
    def __init__(self, query_embedding_dim, item_embedding_dim, num_item_experts=2,
                 dictionary_size=32, max_active_latents=2, events_per_slot=16,
                 sparse_coding_steps=8, sparse_l1_weight=0.01, attention_dim=32,
                 value_dim=32, hidden_dim=64, attention_heads_per_expert=1,
                 use_tail_summaries=True, share_expert_memory=False, eps=1e-8,
                 activation_weighting="coefficient"):
        super().__init__()
        if min(num_item_experts, events_per_slot, attention_dim, value_dim, hidden_dim,
               attention_heads_per_expert) < 1:
            raise ValueError("Expert/history dimensions must be positive")
        self.num_item_experts = num_item_experts
        self.dictionary_size = dictionary_size
        self.max_active_latents = max_active_latents
        self.events_per_slot = events_per_slot
        self.num_heads = attention_heads_per_expert
        self.attention_dim = attention_dim * self.num_heads
        self.value_dim = value_dim * self.num_heads
        self.use_tail_summaries = use_tail_summaries
        self.share_expert_memory = share_expert_memory
        self.eps = eps
        if activation_weighting not in ("coefficient", "uniform"):
            raise ValueError("activation_weighting must be coefficient or uniform")
        self.activation_weighting = activation_weighting
        memories = 1 if share_expert_memory else num_item_experts
        self.history_key = nn.Linear(query_embedding_dim, self.attention_dim)
        self.history_value = nn.Linear(query_embedding_dim, self.value_dim)
        self.item_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(item_embedding_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, self.attention_dim)) for _ in range(num_item_experts)])
        self.expert_gate = nn.Linear(item_embedding_dim, num_item_experts)
        self.sparse_coder = NonnegativeSparseCoding(memories, dictionary_size, self.attention_dim,
                                                   max_active_latents, sparse_coding_steps,
                                                   sparse_l1_weight, eps)
        self.memory_projections = nn.ModuleList([
            nn.Linear(self.attention_dim, self.attention_dim) for _ in range(memories)])
        self.value_query = nn.Linear(self.attention_dim, self.value_dim)
        feature_dim = self.attention_dim + 4 * self.value_dim + 2
        self.expert_scorers = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            for _ in range(num_item_experts)])
        self._model_cache = None

    def get_extra_state(self):
        return {"format_version": 1, "step_rule": "frobenius_bound", "num_item_experts": self.num_item_experts,
                "dictionary_size": self.dictionary_size, "max_active_latents": self.max_active_latents,
                "events_per_slot": self.events_per_slot, "num_heads": self.num_heads,
                "attention_dim": self.attention_dim, "value_dim": self.value_dim,
                "use_tail_summaries": self.use_tail_summaries,
                "share_expert_memory": self.share_expert_memory,
                "activation_weighting": self.activation_weighting,
                "sparse_coding_steps": self.sparse_coder.steps,
                "sparse_l1_weight": self.sparse_coder.l1_weight, "eps": self.eps}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Checkpoint ExpertHistory settings differ from the constructed model")
        self._model_cache = None
        self.sparse_coder._step_cache = None

    def version(self):
        return (id(self), self.training, tuple((id(p), p._version, p.dtype, p.device)
                                               for p in self.parameters()))

    def initialize_after_encoder_construction(self):
        # Backbone initialization may have reset every nested scorer parameter.
        self.sparse_coder.reset_parameters()
        self._model_cache = None

    @torch.no_grad()
    def post_optimizer_step(self):
        self.sparse_coder.project_dictionary_()
        self._model_cache = None

    def prepare_model(self):
        version = self.version()
        cacheable = not self.training and not torch.is_grad_enabled()
        if cacheable and self._model_cache is not None and self._model_cache[0] == version:
            return self._model_cache[1]
        self.sparse_coder.step_size()
        queries = torch.stack([proj(self.sparse_coder.dictionary[e])
                               for e, proj in enumerate(self.memory_projections)])
        if cacheable:
            self._model_cache = version, queries
        return queries

    def prepare_items(self, item_embeddings, item_ids=None, **kwargs):
        if item_embeddings.ndim != 3:
            raise ValueError("Item embeddings must be [1/batch, candidates, dimension]")
        x = item_embeddings.to(self.expert_gate.weight.dtype)
        queries = torch.stack([expert(x) for expert in self.item_experts], dim=-2)
        indices, mask, coefficients, weights, fallback, loss = self.sparse_coder(queries, compute_loss=self.training)
        if self.activation_weighting == "uniform":
            weights = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        if item_ids is None:
            item_ids = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        if item_ids.shape != x.shape[:2]:
            raise ValueError("Candidate IDs and embeddings must have matching leading axes")
        value_queries = self.value_query(queries.to(self.value_query.weight.dtype)).float()
        return ItemState(item_ids.clone(), queries, value_queries, indices, mask, coefficients, weights,
                         self.expert_gate(x).float().softmax(-1), fallback,
                         loss if self.training else None, self.version())

    def prepare_user(self, history_states, history_mask, event_payloads=None):
        if history_states.ndim != 3 or history_mask.shape != history_states.shape[:2]:
            raise ValueError("History states/mask must be [batch,length,dim] / [batch,length]")
        event_payloads = dict(event_payloads or {})
        batch, length, _ = history_states.shape
        mask = history_mask.bool()
        if length == 0:
            history_states = history_states.new_zeros(batch, 1, history_states.shape[-1])
            mask = torch.zeros(batch, 1, device=mask.device, dtype=torch.bool)
            event_payloads = {}
        length = history_states.shape[1]
        x = history_states.to(self.history_key.weight.dtype)
        keys = self.history_key(x).float() / math.sqrt(self.attention_dim)
        values = self.history_value(x).float()
        queries = self.prepare_model()
        # FP64 sums and guard events prevent concentrated subtraction from erasing a real tail.
        logits = torch.einsum("emd,bld->beml", queries.double(), keys.double())
        valid = mask[:, None, None, :].expand_as(logits)
        logits = logits.masked_fill(~valid, -torch.inf)
        guard_budget = min(self.max_active_latents * self.events_per_slot, length)
        guard_positions = logits.argsort(dim=-1, descending=True, stable=True)[..., :guard_budget]
        guard_mask = valid.gather(-1, guard_positions)
        selected_budget = min(self.events_per_slot, length)
        selected_mask = guard_mask[..., :selected_budget]

        def summary(presence):
            scale = logits.masked_fill(~presence, -torch.inf).amax(-1)
            scale = torch.where(presence.any(-1), scale, torch.zeros_like(scale))
            weights = (logits - scale.unsqueeze(-1)).masked_fill(~presence, -torch.inf).exp()
            return torch.einsum("beml,blv->bemv", weights, values.double()), weights.sum(-1), scale

        full_u, full_z, full_scale = summary(valid)
        guarded = torch.zeros_like(valid).scatter(-1, guard_positions, guard_mask)
        deep_presence = valid & ~guarded
        deep_u, deep_z, deep_scale = summary(deep_presence)
        deep_count = deep_presence.sum(-1)
        counts = torch.zeros(batch, length, device=mask.device, dtype=torch.long)
        counts.scatter_add_(1, guard_positions.reshape(batch, -1), guard_mask.reshape(batch, -1).long())
        retained = counts > 0
        packed_size = max(1, int(retained.sum(-1).max().item()))
        order = (~retained).long().argsort(dim=-1, stable=True)[:, :packed_size]
        packed_mask = mask.gather(1, order) & retained.gather(1, order)
        reverse = torch.zeros(batch, length, device=mask.device, dtype=torch.long)
        reverse.scatter_(1, order, torch.arange(packed_size, device=mask.device).expand(batch, -1))
        packed_guard = reverse.gather(1, guard_positions.reshape(batch, -1)).reshape_as(guard_positions)
        payloads = {}
        for name, tensor in event_payloads.items():
            if tensor.ndim >= 2 and tensor.shape[:2] == (batch, length):
                index = order.view(batch, packed_size, *([1] * (tensor.ndim - 2))).expand(
                    batch, packed_size, *tensor.shape[2:])
                payloads[name] = tensor.gather(1, index)
        return UserMemory(
            keys.gather(1, order.unsqueeze(-1).expand(-1, -1, keys.shape[-1])),
            values.gather(1, order.unsqueeze(-1).expand(-1, -1, values.shape[-1])),
            order.masked_fill(~packed_mask, -1), packed_mask, mask.sum(-1),
            packed_guard[..., :selected_budget], selected_mask, packed_guard, guard_mask,
            full_u, full_z, full_scale, deep_u, deep_z, deep_scale, deep_count,
            queries, payloads, self.version())

    def _read_expert(self, memory, items, e):
        batch, candidates = memory.batch_size, items.num_candidates
        me = 0 if self.share_expert_memory else e
        indices = items.active_indices[..., e, :].expand(batch, -1, -1)
        active = items.active_mask[..., e, :].expand(batch, -1, -1)
        rows = torch.arange(batch, device=indices.device)[:, None, None]

        def slot(tensor):
            return tensor[:, me][rows, indices]

        selected = slot(memory.selected_indices)
        selected_mask = slot(memory.selected_mask) & active.unsqueeze(-1)
        sentinel = memory.keys.shape[1]
        union = selected.masked_fill(~selected_mask, sentinel).flatten(-2).sort(-1).values
        first = torch.cat([torch.ones_like(union[..., :1], dtype=torch.bool),
                           union[..., 1:] != union[..., :-1]], dim=-1)
        union_mask = (union < sentinel) & first
        union = union.clamp_max(sentinel - 1)
        keys = memory.keys[rows, union]
        values = memory.values[rows, union]
        q = items.queries[..., e, :].expand(batch, -1, -1).float()
        hq = q.reshape(batch, candidates, self.num_heads, -1)
        hk = keys.reshape(batch, candidates, keys.shape[-2], self.num_heads, -1)
        raw_logits = torch.einsum("bnhd,bnuhd->bnhu", hq, hk) * math.sqrt(self.num_heads)
        raw_weights = masked_softmax(raw_logits, union_mask.unsqueeze(-2))
        hv = values.reshape(batch, candidates, values.shape[-2], self.num_heads, -1)
        raw = torch.einsum("bnhu,bnuhv->bnhv", raw_weights, hv).flatten(-2)
        raw_present = union_mask.any(-1)
        tail_present = memory.lengths[:, None] > union_mask.sum(-1)
        tail = torch.zeros_like(raw)
        if self.use_tail_summaries:
            guards = slot(memory.guard_indices)
            guard_valid = slot(memory.guard_mask)
            guard_in_union = ((guards.unsqueeze(-1) == union[:, :, None, None, :])
                              & union_mask[:, :, None, None, :]).any(-1)
            keep_guard = guard_valid & ~guard_in_union & active.unsqueeze(-1)
            guard_keys = memory.keys[rows.unsqueeze(-1), guards]
            guard_values = memory.values[rows.unsqueeze(-1), guards]
            mq = memory.memory_queries[me][indices].double()
            guard_logits = (mq.unsqueeze(-2) * guard_keys.double()).sum(-1)
            union_is_guard = ((union[:, :, None, :, None] == guards.unsqueeze(-2))
                              & guard_valid.unsqueeze(-2)).any(-1)
            subtract = union_mask[:, :, None, :] & ~union_is_guard & active.unsqueeze(-1)
            union_logits = torch.einsum("bnkd,bnud->bnku", mq, keys.double())
            deep_scale = slot(memory.deep_scale)
            subtraction_weights = (union_logits - deep_scale.unsqueeze(-1)).masked_fill(~subtract, -torch.inf).exp()
            deep_z = (slot(memory.deep_z) - subtraction_weights.sum(-1)).clamp_min(0)
            deep_u = slot(memory.deep_u) - torch.einsum("bnku,bnuv->bnkv", subtraction_weights, values.double())
            deep_left = (slot(memory.deep_count) - subtract.sum(-1)) > 0
            deep_valid = deep_left & (deep_z > 0) & active
            deep_z = torch.where(deep_valid, deep_z, torch.zeros_like(deep_z))
            deep_u = torch.where(deep_valid.unsqueeze(-1), deep_u, torch.zeros_like(deep_u))
            guard_scale = guard_logits.masked_fill(~keep_guard, -torch.inf).amax(-1)
            scale = torch.maximum(guard_scale, deep_scale.masked_fill(~deep_valid, -torch.inf))
            scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
            gw = (guard_logits - scale.unsqueeze(-1)).masked_fill(~keep_guard, -torch.inf).exp()
            dw = (deep_scale - scale).masked_fill(~deep_valid, -torch.inf).exp()
            numerator = (gw.unsqueeze(-1) * guard_values.double()).sum(-2) + dw.unsqueeze(-1) * deep_u
            denominator = gw.sum(-1) + dw * deep_z
            summaries = numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny).unsqueeze(-1)
            summaries = summaries.masked_fill(~tail_present[:, :, None, None], 0)
            weights = items.activation_weights[..., e, :].expand(batch, -1, -1)
            tail = (summaries * weights.double().unsqueeze(-1)).sum(-2).float()
        return raw, tail, raw_present, tail_present

    def score_prepared(self, user_memory, item_state, return_auxiliary_losses=True, **kwargs):
        if user_memory.encoder_ref is not None:
            encoder = user_memory.encoder_ref()
            if encoder is None or encoder.history_version() != user_memory.encoder_version:
                raise ValueError("Stale history encoder state; rebuild user_memory")
        if user_memory.version != self.version() or item_state.version != self.version():
            raise ValueError("Stale or foreign prepared state; rebuild after model/mode/device changes")
        if item_state.queries.shape[0] not in (1, user_memory.batch_size):
            raise ValueError("Candidates must be shared or aligned with the user batch")
        logits = []
        for e, scorer in enumerate(self.expert_scorers):
            raw, tail, raw_present, tail_present = self._read_expert(user_memory, item_state, e)
            q = item_state.queries[..., e, :].expand(user_memory.batch_size, -1, -1)
            vq = item_state.value_queries[..., e, :].expand(user_memory.batch_size, -1, -1)
            features = torch.cat([q.float(), raw, tail, raw * vq, tail * vq,
                                  raw_present.unsqueeze(-1), tail_present.unsqueeze(-1)], dim=-1)
            logits.append(scorer(features.to(scorer[0].weight.dtype)).squeeze(-1).float())
        scores = (torch.stack(logits, dim=-1) * item_state.expert_weights).sum(-1)
        aux = item_state.auxiliary_losses() if return_auxiliary_losses and self.training else {}
        return scores, aux

    def forward(self, query_embeddings, item_embeddings, **kwargs):
        memory = kwargs.get("user_memory")
        if memory is None:
            raise ValueError("ExpertHistorySimilarity requires explicit user_memory")
        if query_embeddings.shape[0] != memory.batch_size:
            raise ValueError("User tensor and memory batch sizes differ")
        items = kwargs.get("item_state")
        if items is None:
            items = self.prepare_items(item_embeddings, kwargs.get("item_ids"))
        return self.score_prepared(memory, items)
