"""Seeded causal cuts, shared by new models and exposure-matched baselines."""

import torch

from .features import SequentialFeatures


def sample_causal_prefix(features, target_ids, generator=None):
    ids = features.past_ids.clone()
    batch, width = ids.shape
    if torch.any(features.past_lengths >= width):
        raise ValueError("Prefix sampling needs an output slot for the training target")
    ids.scatter_(1, features.past_lengths[:, None], target_ids.reshape(batch, 1))
    random = torch.rand(batch, device=ids.device, generator=generator)
    cuts = (random * features.past_lengths.clamp_min(1)).long() + 1
    cuts = torch.minimum(cuts, features.past_lengths)
    targets = ids.gather(1, cuts[:, None]).squeeze(1)
    valid = torch.arange(width, device=ids.device)[None, :] < cuts[:, None]
    payloads = {}
    for key, value in features.past_payloads.items():
        if value.ndim >= 2 and value.shape[:2] == ids.shape:
            mask = valid.reshape(batch, width, *([1] * (value.ndim - 2)))
            payloads[key] = value.masked_fill(~mask, 0)
        else:
            payloads[key] = value.clone()
    return SequentialFeatures(cuts, ids.masked_fill(~valid, 0), None, payloads), targets
