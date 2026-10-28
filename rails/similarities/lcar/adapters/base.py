"""Shared input conventions of the built-in sequential encoders."""

import torch

class BaseAdapter:
    def __init__(self, backbone):
        self.backbone=backbone
        self.num_layers=backbone._num_blocks
        self.activation_checkpoint=getattr(backbone, '_activation_checkpoint', False)

    def preprocess(self, lengths, ids, embeddings, payloads):
        new_lengths, raw, valid=self.backbone._input_features_preproc(lengths,ids,embeddings,payloads)
        if raw.shape[:2]!=ids.shape or not torch.equal(lengths,new_lengths):
            raise ValueError('LCAR requires one original token per event; register an adapter for expanded inputs')
        mask=valid.squeeze(-1).bool() & (torch.arange(ids.shape[1],device=ids.device)[None,:]<lengths[:,None])
        return raw*mask.unsqueeze(-1),mask

    def target_positions(self, lengths):
        positions=getattr(self.backbone._input_features_preproc, '_pos_emb', None)
        if positions is None:
            raise ValueError('Register a backbone adapter with target-position initialization')
        if bool((lengths>=positions.num_embeddings).any()):
            raise ValueError('LCAR requires a reserved next-candidate position')
        return positions(lengths)

    def postprocess(self, states):
        return self.backbone._output_postproc(states)


