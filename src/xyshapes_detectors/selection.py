"""Configurable hit-count requirements, independent of field propagation."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class MTSelection:
    """Pass MP alone, FT alone, or the mixed MP+FT requirement."""

    min_mp: int = 3
    min_ft: int = 10
    mixed_min_mp: int = 1
    mixed_min_ft: int = 6

    def __post_init__(self):
        for name in ("min_mp", "min_ft", "mixed_min_mp", "mixed_min_ft"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def evaluate(self, n_mp, n_ft):
        n_mp, n_ft = np.broadcast_arrays(n_mp, n_ft)
        return ((n_mp >= self.min_mp) | (n_ft >= self.min_ft)
                | ((n_mp >= self.mixed_min_mp) & (n_ft >= self.mixed_min_ft)))
