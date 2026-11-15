"""Public result types."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MedoidResult:
    """The exact medoid of a point set.

    ``index`` is the row whose total Euclidean distance to all rows is smallest, and
    ``total_distance`` is that total.
    """

    index: int
    total_distance: float


@dataclass(frozen=True, eq=False)
class KMedoidsResult:
    """A k-medoids clustering.

    ``medoids[c]`` is the row index of cluster ``c``'s medoid, ``labels[i]`` is the cluster
    of row ``i`` (its nearest medoid) and ``loss`` is the total distance from every row to
    its medoid. ``polish_iterations`` counts the exact polish rounds that ran, and
    ``converged`` says whether polishing stopped because no medoid moved; it is ``False``
    when polishing was switched off, because nothing was checked. The arrays are read-only.
    """

    medoids: np.ndarray
    labels: np.ndarray
    loss: float
    polish_iterations: int
    converged: bool

    def __post_init__(self) -> None:
        for name in ("medoids", "labels"):
            frozen = np.array(getattr(self, name), dtype=np.int64, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
