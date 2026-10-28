"""Optional model bridge: attach LCAR to an unmodified sequential backbone.

The backbone is registered once under ``backbone`` and LCAR once under
``_ndp_module``. No dependency on the repository's encoder base classes is needed.
"""

import torch

from .encoding import backbone_version


class LCAREncoder(torch.nn.Module):
    def __init__(self, backbone, scorer, *, adapter=None):
        super().__init__()
        if isinstance(backbone, LCAREncoder):
            raise ValueError("LCAR is already attached to this encoder")
        if any(module is scorer for module in backbone.modules()):
            raise ValueError("The host already owns this LCAR module; attach it to a native host instead")
        self.backbone = backbone
        self._ndp_module = scorer
        self.adapter = adapter

    @property
    def _embedding_module(self):
        """Expose the host lookup to existing negative samplers without alias registration."""
        return self.backbone._embedding_module

    def get_item_embeddings(self, item_ids):
        return self.backbone.get_item_embeddings(item_ids)

    def debug_str(self):
        return self.backbone.debug_str()

    def history_version(self):
        return backbone_version(self.backbone)

    def encode_history(self, past_lengths, past_ids, past_embeddings, past_payloads, **kwargs):
        return self._ndp_module.encode_history(
            self.backbone, past_lengths, past_ids, past_embeddings, past_payloads,
            adapter=self.adapter, **kwargs)

    def encode_with_history(self, past_lengths, past_ids, past_embeddings, past_payloads, **kwargs):
        memory = self._ndp_module.prepare_user(
            self.backbone, past_lengths, past_ids, past_embeddings, past_payloads,
            adapter=self.adapter, **kwargs)
        return memory.last_history_state, memory

    def encode(self, past_lengths, past_ids, past_embeddings, past_payloads):
        return self.encode_with_history(
            past_lengths, past_ids, past_embeddings, past_payloads)[0]

    def similarity_fn(self, query_embeddings, item_ids, item_embeddings=None, **kwargs):
        if item_embeddings is None:
            item_embeddings = self.get_item_embeddings(item_ids)
        return self._ndp_module(
            query_embeddings, item_embeddings, item_ids=item_ids, **kwargs)

    def forward(self, past_lengths, past_ids, past_embeddings, past_payloads,
                batch_id=None, prefix_targets=None, negatives_sampler=None, sampled_loss=None):
        if prefix_targets is None:
            return self.encode_history(
                past_lengths, past_ids, past_embeddings, past_payloads).history_states
        current, memory = self.encode_with_history(
            past_lengths, past_ids, past_embeddings, past_payloads)
        return sampled_loss.prefix_forward(
            current, prefix_targets, negatives_sampler, user_memory=memory, **past_payloads)
