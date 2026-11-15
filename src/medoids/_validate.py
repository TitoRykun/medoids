"""Input checks at the public boundary. Every function returns a new, validated value."""
from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

NUMERIC_KINDS = "biuf"
INTEGER_KINDS = "iu"


def as_points(X: ArrayLike) -> np.ndarray:
    """``X`` as a C-contiguous float64 array of shape (n, d), n >= 1, d >= 1, all finite."""
    arr = np.asarray(X)
    if arr.dtype.kind not in NUMERIC_KINDS:
        raise TypeError(f"X must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D with one row per point, got {arr.ndim}-D")
    if arr.shape[0] == 0:
        raise ValueError("X must contain at least one point")
    if arr.shape[1] == 0:
        raise ValueError("X must have at least one feature")
    points = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("X must be finite: it contains NaN or infinity")
    return points


def as_int(name: str, value: object, minimum: int) -> int:
    """``value`` as a plain int that is at least ``minimum``; booleans are refused."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {number}")
    return number


def as_cluster_count(k: object, n: int) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    count = int(k)
    if not 1 <= count <= n:
        raise ValueError(f"k must be between 1 and the number of points ({n}), got {count}")
    return count


def as_medoids(medoids: ArrayLike, n: int) -> np.ndarray:
    """``medoids`` as a new int64 array of distinct row indices into ``n`` points."""
    arr = np.asarray(medoids)
    if arr.ndim != 1:
        raise ValueError(f"medoids must be 1-D, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValueError("medoids must contain at least one index")
    if arr.dtype.kind not in INTEGER_KINDS:
        raise TypeError(f"medoids must be integer row indices, got dtype {arr.dtype}")
    if arr.min() < 0 or arr.max() >= n:
        raise ValueError(f"medoids must be between 0 and {n - 1}")
    if len(np.unique(arr)) != arr.size:
        raise ValueError("medoids must be distinct")
    return arr.astype(np.int64, copy=True)
