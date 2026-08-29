from .base import FLMethods, FLMethodsSeg, Combo, SegmentedMethodBase
from .centralized_method import Centralized
from .segment_pulling_method import SegmentPulling
from .dfa_family_base import SegmentedDFAMethod
from .gist_method import Gist
from .dfa_method import DFA
from .sdfa_method import SDFA
from . import segment_ops
from .definitions import FL_METHOD_CLASS_NAMES


METHODS = {
    method_name: globals()[class_name]
    for method_name, class_name in FL_METHOD_CLASS_NAMES.items()
}

__all__ = [
    "METHODS", "Centralized", "SegmentPulling", "Gist", "DFA", "SDFA",
    "SegmentedMethodBase", "SegmentedDFAMethod", "FLMethods", "FLMethodsSeg", "Combo", "segment_ops"
]
