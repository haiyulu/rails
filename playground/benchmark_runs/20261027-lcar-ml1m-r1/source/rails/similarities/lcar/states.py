"""Explicit, gradient-carrying LCAR preparation states."""

from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, Tuple

import torch


def _index(index):
    return slice(index, None if index == -1 else index+1) if isinstance(index, int) else index


def tensor_bytes(tensors):
    return sum(t.numel()*t.element_size() for t in tensors)


@dataclass(frozen=True)
class ModelContext:
    initial_tokens: torch.Tensor
    version: Tuple

    def num_bytes(self):
        return tensor_bytes([self.initial_tokens])


@dataclass(frozen=True)
class HistoryCache:
    original: torch.Tensor
    mask: torch.Tensor
    target_times: Optional[torch.Tensor]
    timestamps: Optional[torch.Tensor]
    layers: Tuple[dict, ...]
    final_states: torch.Tensor
    version: Tuple


@dataclass(frozen=True)
class EncoderOutput:
    original: torch.Tensor
    mask: torch.Tensor
    history_states: torch.Tensor
    latent_states: torch.Tensor
    selected_positions: torch.Tensor
    selected_mask: torch.Tensor
    context: ModelContext
    history_cache: HistoryCache


@dataclass(frozen=True)
class UserMemory:
    tokens: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    retained_mask: torch.Tensor
    lengths: torch.Tensor
    recent_indices: torch.Tensor
    recent_mask: torch.Tensor
    selected_indices: torch.Tensor
    selected_mask: torch.Tensor
    summaries: torch.Tensor
    summary_keys: torch.Tensor
    summary_values: torch.Tensor
    summary_mask: torch.Tensor
    event_payloads: Dict[str, torch.Tensor]
    version: Tuple
    encoder_ref: Optional[object] = None
    encoder_version: Optional[Tuple] = None

    @property
    def batch_size(self):
        return self.tokens.shape[0]

    def index_batch(self, index):
        index = _index(index)
        changes = {f.name: getattr(self, f.name)[index] for f in fields(self)
                   if isinstance(getattr(self, f.name), torch.Tensor)}
        changes['event_payloads'] = {k: v[index] for k,v in self.event_payloads.items()}
        return replace(self, **changes)

    def num_bytes(self):
        return tensor_bytes([getattr(self, f.name) for f in fields(self)
                             if isinstance(getattr(self, f.name), torch.Tensor)]
                            + list(self.event_payloads.values()))


@dataclass(frozen=True)
class ItemState:
    item_ids: torch.Tensor
    queries: torch.Tensor
    attention_queries: torch.Tensor
    value_queries: torch.Tensor
    active_indices: torch.Tensor
    active_mask: torch.Tensor
    coefficients: torch.Tensor
    activation_weights: torch.Tensor  # Diagnostic only; not an attention prior.
    expert_weights: torch.Tensor
    zero_code_fallback: torch.Tensor
    sparse_loss: Optional[torch.Tensor]
    version: Tuple

    @property
    def num_candidates(self):
        return self.queries.shape[1]

    def slice_candidates(self, start, stop):
        return replace(self, **{f.name: getattr(self, f.name)[:,start:stop] for f in fields(self)
                                if isinstance(getattr(self, f.name), torch.Tensor)})

    def index_batch(self, index):
        if self.queries.shape[0] == 1:
            return self
        index = _index(index)
        return replace(self, **{f.name: getattr(self, f.name)[index] for f in fields(self)
                                if isinstance(getattr(self, f.name), torch.Tensor)})

    def auxiliary_losses(self):
        return {} if self.sparse_loss is None else {'sparse_code_loss': self.sparse_loss.mean()}

    @torch.no_grad()
    def diagnostics(self, dictionary_size):
        result = {'sparse_active_atoms': self.active_mask.float().sum(-1).mean(),
                  'sparse_zero_code_rate': self.zero_code_fallback.float().mean(),
                  'expert_gate_entropy': -(self.expert_weights * self.expert_weights.clamp_min(1e-8).log()).sum(-1).mean()}
        usage=[]
        for e in range(self.queries.shape[-2]):
            chosen=self.active_indices[...,e,:][self.active_mask[...,e,:]]
            usage.append((torch.bincount(chosen, minlength=dictionary_size)>0).float().mean())
            result[f'expert_weight_{e}']=self.expert_weights[...,e].mean()
        result['sparse_used_atom_fraction']=torch.stack(usage).mean()
        return result

    def num_bytes(self):
        return tensor_bytes([getattr(self,f.name) for f in fields(self)
                             if isinstance(getattr(self,f.name),torch.Tensor)])


@dataclass(frozen=True)
class JointMemory:
    keys: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor
    selected_positions: torch.Tensor
    selected_mask: torch.Tensor
