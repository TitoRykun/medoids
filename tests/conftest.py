"""Shared helpers: slow, obviously-correct references and data generators."""
from __future__ import annotations

import os

import numpy as np

# With NUMBA_DISABLE_JIT=1 (used to measure coverage) the kernels run as plain Python,
# so MEDOIDS_TEST_SCALE shrinks every data set, e.g. MEDOIDS_TEST_SCALE=0.1.
SCALE = float(os.environ.get("MEDOIDS_TEST_SCALE", "1"))


def scaled(n: int, floor: int = 70) -> int:
    """Data-set size after applying the test scale, never below `floor`."""
    return max(floor, int(n * SCALE))


def reference_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distances straight from coordinates (no dot-product expansion)."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    return np.stack([np.sqrt(((B - a) ** 2).sum(1)) for a in A])


def reference_sums(X: np.ndarray) -> np.ndarray:
    """Total distance from every point to all points."""
    return reference_distances(X, X).sum(1)


def reference_loss(X: np.ndarray, medoids: np.ndarray) -> float:
    return float(reference_distances(X, np.asarray(X)[medoids]).min(1).sum())


def gaussian(n: int, d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def blobs(n: int, d: int, k: int, seed: int = 0, spread: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """`k` tight, well-separated Gaussian blobs; returns points and their blob index."""
    rng = np.random.default_rng(seed)
    centres = np.zeros((k, d))
    centres[:, 0] = 10.0 * np.arange(k)
    truth = np.arange(n) % k
    return centres[truth] + spread * rng.standard_normal((n, d)), truth
