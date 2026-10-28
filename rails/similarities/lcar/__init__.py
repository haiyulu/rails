"""LCAR plug-in: dictionary-guided native-block reads and joint retrieval attention."""

from .similarity_fn import LCARSimilarity
from .encoder import LCAREncoder
from .adapters import BackboneAdapter, register_backbone_adapter, get_backbone_adapter
from .states import EncoderOutput, HistoryCache, ItemState, ModelContext, UserMemory

LCAR=LCARSimilarity

__all__=['LCAR','LCARSimilarity','LCAREncoder','BackboneAdapter','register_backbone_adapter','get_backbone_adapter',
         'EncoderOutput','HistoryCache','ItemState','ModelContext','UserMemory']
