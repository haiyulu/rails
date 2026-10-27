"""Unrolled nonnegative L1 sparse coding, with an explicit inference budget."""

import torch
from torch import nn
from torch.nn import functional as F


class NonnegativeSparseCoding(nn.Module):
    @torch.inference_mode(False)
    def __init__(self, num_dictionaries, dictionary_size, dimension, max_active_latents,
                 steps=8, l1_weight=0.01, eps=1e-8):
        super().__init__()
        if min(num_dictionaries, dictionary_size, dimension, max_active_latents, steps) < 1:
            raise ValueError("Sparse coding dimensions and steps must be positive")
        if max_active_latents > dictionary_size or l1_weight < 0:
            raise ValueError("Invalid support budget or L1 penalty")
        self.dictionary = nn.Parameter(torch.empty(num_dictionaries, dictionary_size, dimension))
        self.max_active_latents = max_active_latents
        self.steps, self.l1_weight, self.eps = steps, l1_weight, eps
        self._step_cache = None
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.dictionary.normal_()
            self.dictionary.copy_(F.normalize(self.dictionary, dim=-1))
        self._step_cache = None

    @torch.no_grad()
    def project_dictionary_(self):
        self.dictionary.div_(self.dictionary.norm(dim=-1, keepdim=True).clamp_min(1.0))
        self._step_cache = None

    def step_size(self):
        key = (self.dictionary._version, self.dictionary.device, self.dictionary.dtype)
        if self._step_cache is None or self._step_cache[0] != key:
            with torch.no_grad():
                # Frobenius squared upper-bounds the spectral Lipschitz constant.
                # It is deterministic, cheap, and has auditable arithmetic costs.
                norm = self.dictionary.float().square().sum(dim=(-2, -1))
                eta = 0.9 / norm.clamp_min(self.eps)
            self._step_cache = key, eta
        return self._step_cache[1]

    def forward(self, queries, compute_loss=True):
        dictionary = self.dictionary.float()
        if dictionary.shape[0] == 1:
            dictionary = dictionary.expand(queries.shape[-2], -1, -1)
        eta = self.step_size().view(*([1] * (queries.ndim - 2)), -1, 1)
        z = F.normalize(queries.float(), dim=-1, eps=self.eps)
        coefficients = z.new_zeros(*z.shape[:-1], dictionary.shape[1])
        for _ in range(self.steps):
            reconstruction = torch.einsum("...em,emd->...ed", coefficients, dictionary)
            gradient = torch.einsum("...ed,emd->...em", z - reconstruction, dictionary)
            coefficients = F.relu(coefficients + eta * (gradient - self.l1_weight))
        # Stable sorting defines ties without changing gradients of retained values.
        indices = coefficients.argsort(dim=-1, descending=True, stable=True)[..., :self.max_active_latents]
        selected = coefficients.gather(-1, indices)
        loss = None
        if compute_loss:
            retained = torch.zeros_like(coefficients).scatter(-1, indices, selected)
            reconstruction = torch.einsum("...em,emd->...ed", retained, dictionary)
            loss = 0.5 * (z - reconstruction).square().sum(-1) + self.l1_weight * retained.sum(-1)
        fallback = selected.sum(-1) == 0
        compatibility = torch.einsum("...ed,emd->...em", z, dictionary)
        fallback_indices = compatibility.argsort(dim=-1, descending=True, stable=True)[..., :self.max_active_latents]
        indices = torch.where(fallback.unsqueeze(-1), fallback_indices, indices)
        mask = selected > 0
        first = torch.arange(self.max_active_latents, device=queries.device) == 0
        mask = torch.where(fallback.unsqueeze(-1), first, mask)
        weights = selected / selected.sum(-1, keepdim=True).clamp_min(torch.finfo(selected.dtype).tiny)
        weights = torch.where(fallback.unsqueeze(-1), first.to(weights.dtype), weights)
        return indices, mask, selected, weights, fallback, loss
