"""Original-event attention and full-contextual-history reads for item experts."""

import math
from dataclasses import fields

import torch
from torch import nn

from rails.similarities.module import SimilarityModule
from .sparse_coding import NonnegativeSparseCoding
from .states import ItemState, ModelContext, RawHistoryState, UserMemory


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
                 dictionary_size=512, max_active_latents=16, events_per_slot=16,
                 sparse_coding_steps=8, sparse_l1_weight=0.01, attention_dim=32,
                 value_dim=32, hidden_dim=64, attention_heads_per_expert=1,
                 use_full_history_summaries=True, share_expert_memory=False, eps=1e-8,
                 activation_weighting="coefficient", recent_events=32,
                 latent_conditioned_history=True, dictionary_attention_dim=32,
                 dictionary_attention_heads=1, dictionary_gate_init=0.01,
                 use_selected_attention=True):
        super().__init__()
        if min(num_item_experts, events_per_slot, attention_dim, value_dim, hidden_dim,
               attention_heads_per_expert, dictionary_attention_dim, dictionary_attention_heads) < 1:
            raise ValueError("Expert/history dimensions must be positive")
        if recent_events < 0 or not 0 < dictionary_gate_init < 1:
            raise ValueError("Invalid recent-event budget or dictionary gate")
        self.num_item_experts = num_item_experts
        self.query_embedding_dim = query_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.hidden_dim = hidden_dim
        self.dictionary_size = dictionary_size
        self.max_active_latents = max_active_latents
        self.events_per_slot = events_per_slot
        self.recent_events = recent_events
        self.num_heads = attention_heads_per_expert
        self.attention_dim = attention_dim * self.num_heads
        self.value_dim = value_dim * self.num_heads
        self.use_full_history_summaries = use_full_history_summaries
        self.use_selected_attention = use_selected_attention
        self.latent_conditioned_history = latent_conditioned_history
        self.dictionary_attention_dim = dictionary_attention_dim
        self.dictionary_attention_heads = dictionary_attention_heads
        self.dictionary_gate_init = dictionary_gate_init
        self.share_expert_memory = share_expert_memory
        self.eps = eps
        if activation_weighting not in ("coefficient", "uniform"):
            raise ValueError("activation_weighting must be coefficient or uniform")
        self.activation_weighting = activation_weighting
        memories = 1 if share_expert_memory else num_item_experts
        self.raw_key = nn.Linear(query_embedding_dim, self.attention_dim)
        self.raw_value = nn.Linear(query_embedding_dim, self.value_dim)
        self.context_key = nn.Linear(query_embedding_dim, self.attention_dim)
        self.context_value = nn.Linear(query_embedding_dim, self.value_dim)
        self.item_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(item_embedding_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, self.attention_dim)) for _ in range(num_item_experts)])
        self.expert_gate = nn.Linear(item_embedding_dim, num_item_experts)
        self.sparse_coder = NonnegativeSparseCoding(memories, dictionary_size, self.attention_dim,
                                                   max_active_latents, sparse_coding_steps,
                                                   sparse_l1_weight, eps)
        # Shared query projection learns from differentiable summaries, despite hard selection.
        self.memory_projections = nn.ModuleList([
            nn.Linear(self.attention_dim, self.attention_dim) for _ in range(memories)])
        self.value_query = nn.Linear(self.attention_dim, self.value_dim)
        feature_dim = self.attention_dim + 6 * self.value_dim + 3
        self.expert_scorers = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            for _ in range(num_item_experts)])
        self._model_cache = None

    def get_extra_state(self):
        return {"format_version": 2, "step_rule": "frobenius_bound",
                "query_embedding_dim": self.query_embedding_dim,
                "item_embedding_dim": self.item_embedding_dim, "hidden_dim": self.hidden_dim,
                "num_item_experts": self.num_item_experts, "dictionary_size": self.dictionary_size,
                "max_active_latents": self.max_active_latents, "events_per_slot": self.events_per_slot,
                "recent_events": self.recent_events, "num_heads": self.num_heads,
                "attention_dim": self.attention_dim, "value_dim": self.value_dim,
                "use_full_history_summaries": self.use_full_history_summaries,
                "use_selected_attention": self.use_selected_attention,
                "latent_conditioned_history": self.latent_conditioned_history,
                "dictionary_attention_dim": self.dictionary_attention_dim,
                "dictionary_attention_heads": self.dictionary_attention_heads,
                "dictionary_gate_init": self.dictionary_gate_init,
                "share_expert_memory": self.share_expert_memory,
                "activation_weighting": self.activation_weighting,
                "sparse_coding_steps": self.sparse_coder.steps,
                "sparse_l1_weight": self.sparse_coder.l1_weight, "eps": self.eps}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Checkpoint ExpertHistory settings or architecture version differ")
        self._model_cache = None
        self.sparse_coder._step_cache = None

    def version(self):
        if any(p.is_inference() for p in self.parameters()):
            raise ValueError("Construct and move ExpertHistory parameters outside inference_mode for versioned caches")
        return (id(self), self.training, tuple((id(p), p._version, p.dtype, p.device)
                                               for p in self.parameters()))

    def initialize_after_encoder_construction(self):
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
        atoms = self.sparse_coder.dictionary
        queries = torch.stack([proj(atoms[e]) for e, proj in enumerate(self.memory_projections)])
        context = ModelContext(atoms, queries, version)
        if cacheable:
            self._model_cache = version, context
        return context

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

    @staticmethod
    def _check_history(states, mask):
        if states.ndim != 3 or mask.shape != states.shape[:2]:
            raise ValueError("History states/mask must be [batch,length,dim] / [batch,length]")

    def prepare_raw_history(self, raw_embeddings, history_mask, event_payloads=None, context=None):
        self._check_history(raw_embeddings, history_mask)
        context = self.prepare_model() if context is None else context
        if context.version != self.version():
            raise ValueError("Stale model context")
        event_payloads = dict(event_payloads or {})
        batch, length, _ = raw_embeddings.shape
        mask = history_mask.bool()
        if length == 0:
            raw_embeddings = raw_embeddings.new_zeros(batch, 1, raw_embeddings.shape[-1])
            mask = torch.zeros(batch, 1, device=mask.device, dtype=torch.bool)
            event_payloads = {}
        length = raw_embeddings.shape[1]
        x = raw_embeddings.to(self.raw_key.weight.dtype)
        keys = self.raw_key(x).float() / math.sqrt(self.attention_dim)
        values = self.raw_value(x).float()
        logits = torch.einsum("emd,bld->beml", context.memory_queries.float(), keys)
        valid = mask[:, None, None, :].expand_as(logits)
        selected = logits.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        selected = selected[..., :min(self.events_per_slot, length)]
        selected_mask = valid.gather(-1, selected)

        positions = torch.arange(length, device=mask.device).expand(batch, -1)
        lengths = mask.sum(-1)
        recent_valid = mask & (mask.long().cumsum(-1) > lengths[:, None] - self.recent_events)
        recent = positions.masked_fill(~recent_valid, length).argsort(dim=-1, stable=True)
        recent = recent[:, :min(self.recent_events, length)]
        recent_mask = recent_valid.gather(-1, recent)

        references = torch.cat([selected.reshape(batch, -1), recent], -1)
        reference_masks = torch.cat([selected_mask.reshape(batch, -1), recent_mask], -1)
        counts = torch.zeros(batch, length, device=mask.device, dtype=torch.long)
        counts.scatter_add_(1, references, reference_masks.long())
        retained = counts > 0
        packed_size = max(1, int(retained.sum(-1).max().item()))
        order = (~retained).long().argsort(dim=-1, stable=True)[:, :packed_size]
        packed_mask = mask.gather(1, order) & retained.gather(1, order)
        reverse = torch.zeros(batch, length, device=mask.device, dtype=torch.long)
        reverse.scatter_(1, order, torch.arange(packed_size, device=mask.device).expand(batch, -1))
        packed_selected = reverse.gather(1, selected.reshape(batch, -1)).reshape_as(selected)
        packed_recent = reverse.gather(1, recent)
        payloads = {}
        for name, tensor in event_payloads.items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 and tensor.shape[:2] == (batch, length):
                index = order.view(batch, packed_size, *([1] * (tensor.ndim-2))).expand(
                    batch, packed_size, *tensor.shape[2:])
                payloads[name] = tensor.gather(1, index)
        return RawHistoryState(
            keys.gather(1, order.unsqueeze(-1).expand(-1, -1, keys.shape[-1])),
            values.gather(1, order.unsqueeze(-1).expand(-1, -1, values.shape[-1])),
            order.masked_fill(~packed_mask, -1), packed_mask, lengths,
            packed_selected, selected_mask, packed_recent, recent_mask, payloads, self.version())

    def finish_user_memory(self, raw_state, contextual_states, history_mask, context=None, event_payloads=None):
        self._check_history(contextual_states, history_mask)
        context = self.prepare_model() if context is None else context
        if raw_state.version != self.version() or context.version != self.version():
            raise ValueError("Stale raw history or model context")
        batch, length, _ = contextual_states.shape
        if batch != raw_state.keys.shape[0]:
            raise ValueError("Raw and contextual history batches differ")
        mask = history_mask.bool()
        if not torch.equal(mask.sum(-1), raw_state.lengths):
            raise ValueError("Raw and contextual histories must cover the same valid events")
        if length == 0:
            contextual_states = contextual_states.new_zeros(batch, 1, contextual_states.shape[-1])
            mask = torch.zeros(batch, 1, device=mask.device, dtype=torch.bool)
        queries = context.memory_queries.float()
        presence = (raw_state.lengths > 0)[:, None, None].expand(batch, *queries.shape[:2])
        if self.use_full_history_summaries:
            x = contextual_states.to(self.context_key.weight.dtype)
            keys = self.context_key(x).float() / math.sqrt(self.attention_dim)
            values = self.context_value(x).float()
            logits = torch.einsum("emd,bld->beml", queries, keys)
            weights = masked_softmax(logits, mask[:, None, None, :])
            summaries = torch.einsum("beml,blv->bemv", weights, values)
        else:
            presence = torch.zeros_like(presence)
            summaries = contextual_states.new_zeros(batch, *queries.shape[:2], self.value_dim).float()
        raw_fields = {f.name: getattr(raw_state, f.name) for f in fields(raw_state)}
        return UserMemory(**raw_fields, summaries=summaries, summary_mask=presence)

    def prepare_user(self, raw_embeddings, contextual_states, history_mask, event_payloads=None):
        """Convenience API with explicit pre-block and post-block inputs."""
        context = self.prepare_model()
        raw = self.prepare_raw_history(raw_embeddings, history_mask, event_payloads, context)
        return self.finish_user_memory(raw, contextual_states, history_mask, context)

    def _attend_original(self, memory, q, indices, mask):
        batch, candidates = q.shape[:2]
        if indices.shape[-1] == 0:
            return q.new_zeros(batch, candidates, self.value_dim), mask.any(-1)
        rows = torch.arange(batch, device=q.device)[:, None, None]
        keys, values = memory.keys[rows, indices], memory.values[rows, indices]
        hq = q.float().reshape(batch, candidates, self.num_heads, -1)
        hk = keys.reshape(batch, candidates, keys.shape[-2], self.num_heads, -1)
        logits = torch.einsum("bnhd,bnuhd->bnhu", hq, hk) * math.sqrt(self.num_heads)
        weights = masked_softmax(logits, mask.unsqueeze(-2))
        hv = values.reshape(batch, candidates, values.shape[-2], self.num_heads, -1)
        return torch.einsum("bnhu,bnuhv->bnhv", weights, hv).flatten(-2), mask.any(-1)

    def _read_expert(self, memory, items, e):
        batch = memory.batch_size
        me = 0 if self.share_expert_memory else e
        indices = items.active_indices[..., e, :].expand(batch, -1, -1)
        active = items.active_mask[..., e, :].expand(batch, -1, -1)
        rows = torch.arange(batch, device=indices.device)[:, None, None]

        def slot(tensor):
            return tensor[:, me][rows, indices]

        q = items.queries[..., e, :].expand(batch, -1, -1)
        recent_indices = memory.recent_indices[:, None].expand(batch, items.num_candidates, -1)
        recent_mask = memory.recent_mask[:, None].expand_as(recent_indices)
        recent, recent_present = self._attend_original(memory, q, recent_indices, recent_mask)
        selected = slot(memory.selected_indices)
        selected_mask = slot(memory.selected_mask) & active.unsqueeze(-1)
        sentinel = memory.keys.shape[1]
        union = selected.masked_fill(~selected_mask, sentinel).flatten(-2).sort(-1).values
        first = torch.cat([torch.ones_like(union[..., :1], dtype=torch.bool),
                           union[..., 1:] != union[..., :-1]], dim=-1)
        union_mask = (union < sentinel) & first & self.use_selected_attention
        union = union.clamp_max(sentinel-1)
        selected_read, selected_present = self._attend_original(memory, q, union, union_mask)
        summary_weights = items.activation_weights[..., e, :].expand(batch, -1, -1)
        summary_weights = summary_weights * active
        summary = (slot(memory.summaries) * summary_weights.unsqueeze(-1)).sum(-2)
        summary_present = (slot(memory.summary_mask) & active).any(-1)
        return recent, selected_read, summary, recent_present, selected_present, summary_present

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
            recent, selected, summary, rp, sp, cp = self._read_expert(user_memory, item_state, e)
            q = item_state.queries[..., e, :].expand(user_memory.batch_size, -1, -1).float()
            vq = item_state.value_queries[..., e, :].expand(user_memory.batch_size, -1, -1)
            features = torch.cat([q, recent, selected, summary,
                                  recent*vq, selected*vq, summary*vq,
                                  rp.unsqueeze(-1), sp.unsqueeze(-1), cp.unsqueeze(-1)], dim=-1)
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
