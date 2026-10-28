"""Backbone-independent masked attention primitives."""

import torch


def masked_softmax(logits, mask):
    """Normalize valid keys, returning zeros for an entirely masked memory."""
    if logits.shape[-1] == 0:
        return torch.zeros_like(logits)
    logits = logits.masked_fill(~mask, -torch.inf)
    maximum = logits.amax(-1, keepdim=True)
    maximum = torch.where(mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
    weights = (logits - maximum).exp()
    return weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
