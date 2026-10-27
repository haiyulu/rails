"""Explicit gradient-carrying states; inference caches use the same structures."""

from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, Tuple

import torch


def _batch_index(index):
    return slice(index, None if index == -1 else index + 1) if isinstance(index, int) else index


@dataclass(frozen=True)
class UserMemory:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    retained_mask: torch.Tensor
    lengths: torch.Tensor
    selected_indices: torch.Tensor
    selected_mask: torch.Tensor
    guard_indices: torch.Tensor
    guard_mask: torch.Tensor
    summary_u: torch.Tensor
    summary_z: torch.Tensor
    summary_scale: torch.Tensor
    deep_u: torch.Tensor
    deep_z: torch.Tensor
    deep_scale: torch.Tensor
    deep_count: torch.Tensor
    memory_queries: torch.Tensor
    event_payloads: Dict[str, torch.Tensor]
    version: Tuple
    encoder_ref: Optional[object] = None
    encoder_version: Optional[Tuple] = None

    @property
    def batch_size(self):
        return self.keys.shape[0]

    def index_batch(self, index):
        index = _batch_index(index)
        values = {f.name: getattr(self, f.name)[index] for f in fields(self)
                  if isinstance(getattr(self, f.name), torch.Tensor) and f.name != "memory_queries"}
        values["event_payloads"] = {k: v[index] for k, v in self.event_payloads.items()}
        return replace(self, **values)

    def num_bytes(self):
        tensors = [getattr(self, f.name) for f in fields(self)
                   if isinstance(getattr(self, f.name), torch.Tensor) and f.name != "memory_queries"]
        tensors += list(self.event_payloads.values())
        return sum(t.numel() * t.element_size() for t in tensors)


@dataclass(frozen=True)
class ItemState:
    item_ids: torch.Tensor
    queries: torch.Tensor
    value_queries: torch.Tensor
    active_indices: torch.Tensor
    active_mask: torch.Tensor
    coefficients: torch.Tensor
    activation_weights: torch.Tensor
    expert_weights: torch.Tensor
    zero_code_fallback: torch.Tensor
    sparse_loss: Optional[torch.Tensor]
    version: Tuple

    @property
    def num_candidates(self):
        return self.queries.shape[1]

    def slice_candidates(self, start, stop):
        return replace(self, **{f.name: getattr(self, f.name)[:, start:stop]
                                for f in fields(self)
                                if isinstance(getattr(self, f.name), torch.Tensor)})

    def index_batch(self, index):
        if self.queries.shape[0] == 1:
            return self
        index = _batch_index(index)
        return replace(self, **{f.name: getattr(self, f.name)[index]
                                for f in fields(self)
                                if isinstance(getattr(self, f.name), torch.Tensor)})

    def auxiliary_losses(self):
        return {} if self.sparse_loss is None else {"sparse_code_loss": self.sparse_loss.mean()}

    @torch.no_grad()
    def diagnostics(self, dictionary_size):
        metrics = {"sparse_active_atoms": self.active_mask.float().sum(-1).mean(),
                   "sparse_zero_code_rate": self.zero_code_fallback.float().mean(),
                   "expert_gate_entropy": -(self.expert_weights * self.expert_weights.clamp_min(1e-8).log()).sum(-1).mean()}
        usage = []
        for e in range(self.queries.shape[-2]):
            chosen = self.active_indices[..., e, :][self.active_mask[..., e, :]]
            counts = torch.bincount(chosen, minlength=dictionary_size)
            usage.append((counts > 0).float().mean())
            metrics[f"expert_weight_{e}"] = self.expert_weights[..., e].mean()
        metrics["sparse_used_atom_fraction"] = torch.stack(usage).mean()
        return metrics

    def num_bytes(self):
        return sum(t.numel() * t.element_size() for f in fields(self)
                   if isinstance(t := getattr(self, f.name), torch.Tensor))
