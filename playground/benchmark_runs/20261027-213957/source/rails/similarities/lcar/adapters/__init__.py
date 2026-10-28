"""Parameter-free adapters: the only backbone-specific LCAR boundary."""

from typing import Protocol

class BackboneAdapter(Protocol):
    num_layers: int
    activation_checkpoint: bool
    def preprocess(self, lengths, ids, embeddings, payloads): ...
    def target_positions(self, lengths): ...
    def history_step(self, index, history, mask, lengths, payloads): ...
    def latent_step(self, index, latents, context): ...
    def postprocess(self, states): ...


_ADAPTERS={}


def register_backbone_adapter(backbone_type, adapter_factory):
    """Register a host class (or its name) and a parameter-free adapter factory."""
    _ADAPTERS[backbone_type]=adapter_factory


def get_backbone_adapter(backbone):
    for cls in type(backbone).__mro__:
        factory=_ADAPTERS.get(cls, _ADAPTERS.get(cls.__name__))
        if factory is not None:
            return factory(backbone)
    raise ValueError(f'No LCAR adapter registered for {type(backbone).__name__}')



from .sasrec import SASRecAdapter
from .hstu import HSTUAdapter

register_backbone_adapter("SASRec", SASRecAdapter)
register_backbone_adapter("HSTU", HSTUAdapter)
