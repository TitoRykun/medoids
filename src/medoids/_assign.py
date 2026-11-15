"""Assigning points to their nearest medoid."""
from __future__ import annotations

import numpy as np

from ._kernels.kmed import nearest

BLOCK_ELEMS = 1 << 22  # distances held at once when assigning new points


def assign_to_medoids(points: np.ndarray, medoids: np.ndarray) -> tuple[np.ndarray, float]:
    """Cluster of every row of ``points`` (index into ``medoids``) and the total distance."""
    centred = points - points.mean(0)
    labels, distance = nearest(centred, np.einsum("ij,ij->i", centred, centred), medoids)
    return labels.astype(np.int64), float(distance.sum())


def assign_to_centres(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Nearest row of ``centres`` for every row of ``points``, computed from coordinates."""
    step = max(1, BLOCK_ELEMS // len(centres))
    blocks = [
        np.stack([((points[start:start + step] - c) ** 2).sum(1) for c in centres], axis=1).argmin(1)
        for start in range(0, len(points), step)
    ]
    return np.concatenate(blocks).astype(np.int64)
