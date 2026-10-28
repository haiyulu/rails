"""Pointwise event-to-static-dictionary attention used before sequence blocks."""

import math

import torch
from torch import nn


class LatentDictionaryAttention(nn.Module):
    @torch.inference_mode(False)
    def __init__(self, hidden_dim, atom_dim, attention_dim=32, num_heads=1,
                 gate_init=0.01):
        super().__init__()
        if min(hidden_dim, atom_dim, attention_dim, num_heads) < 1:
            raise ValueError("Dictionary attention dimensions must be positive")
        if not 0 < gate_init < 1:
            raise ValueError("Dictionary gate initialization must be strictly between zero and one")
        self.num_heads, self.head_dim = num_heads, attention_dim
        width = num_heads * attention_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, width)
        self.key = nn.Linear(atom_dim, width)
        self.value = nn.Linear(atom_dim, width)
        self.output = nn.Linear(width, hidden_dim)
        self.gate_logit = nn.Parameter(torch.full((hidden_dim,), math.log(gate_init / (1-gate_init))))
        self._projection_cache = None

    def prepare_dictionary(self, atoms):
        cacheable = not self.training and not torch.is_grad_enabled()
        version = (atoms.data_ptr(), atoms._version, atoms.dtype, atoms.device,
                   tuple((id(p), p._version, p.dtype, p.device) for p in self.parameters()))
        if cacheable and self._projection_cache is not None and self._projection_cache[0] == version:
            return self._projection_cache[1]
        flat = atoms.flatten(0, 1).to(self.key.weight.dtype)
        keys = self.key(flat).reshape(-1, self.num_heads, self.head_dim)
        values = self.value(flat).reshape(-1, self.num_heads, self.head_dim)
        if cacheable:
            self._projection_cache = version, (keys, values)
        return keys, values

    def cache_num_bytes(self):
        return (sum(t.numel() * t.element_size() for t in self._projection_cache[1])
                if self._projection_cache is not None else 0)

    def forward(self, states, atoms, valid_mask=None):
        """Accept dense [B,T,H] or jagged [sum(lengths),H] event states."""
        keys, values = self.prepare_dictionary(atoms)
        q = self.query(self.norm(states)).reshape(*states.shape[:-1], self.num_heads, self.head_dim)
        logits = torch.einsum("...hd,mhd->...hm", q.float(), keys.float()) / math.sqrt(self.head_dim)
        weights = logits.softmax(-1)
        context = torch.einsum("...hm,mhd->...hd", weights, values.float()).flatten(-2)
        update = self.output(context.to(self.output.weight.dtype))
        result = states + self.gate_logit.sigmoid() * update
        if valid_mask is not None:
            result = result * valid_mask.unsqueeze(-1)
        return result


class VersionedHistoryCache(list):
    """HSTU cache with the encoder/dictionary version that produced its tensors."""

    def __init__(self, states, version):
        super().__init__(states)
        self.version = version
