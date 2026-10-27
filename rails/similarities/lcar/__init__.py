"""LCAR plug-in: dictionary-guided native-block reads and joint retrieval attention."""

from .similarity_fn import LCARSimilarity
from .backbones import BackboneAdapter, register_backbone_adapter, get_backbone_adapter
from .states import EncoderOutput, HistoryCache, ItemState, ModelContext, UserMemory

LCAR=LCARSimilarity

__all__=['LCAR','LCARSimilarity','BackboneAdapter','register_backbone_adapter','get_backbone_adapter',
         'EncoderOutput','HistoryCache','ItemState','ModelContext','UserMemory']
