__version__ = "1.0.0"

from .dualpipe import DualPipe
from .comm import (
    set_p2p_tensor_shapes,
    set_p2p_tensor_dtype,
)
from .utils import WeightGradStore

__all__ = [
    "DualPipe",
    "WeightGradStore",
    "set_p2p_tensor_shapes",
    "set_p2p_tensor_dtype",
]
