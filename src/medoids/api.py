"""Public functions: exact medoid, k-medoids, and polishing of existing medoids."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._assign import assign_to_medoids
from ._kernels.brute import brute
from ._kernels.cpv import cpv
from ._kernels.kernels import Oracle
from ._kernels.kmed import alternate, fasterclara
from ._types import KMedoidsResult, MedoidResult
from ._validate import as_cluster_count, as_int, as_medoids, as_points

SMALL_N = 64  # below this, evaluating every point directly is cheaper than bounding
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_N_SAMPLES = 5
DEFAULT_MAX_ITER = 100
POLISH_METHOD = "cpv-warm"


def _exact_medoid(points: np.ndarray) -> MedoidResult:
    oracle = Oracle(points)
    found = brute(oracle) if len(points) <= SMALL_N else cpv(oracle)
    return MedoidResult(index=int(found["medoid"]), total_distance=float(found["S_min"]))


def medoid(X: ArrayLike) -> MedoidResult:
    """Exact medoid of the rows of ``X`` under Euclidean distance.

    The medoid is the row whose total distance to all rows is smallest. The answer is
    exact (ties within a relative 1e-9 may resolve either way), but most rows are ruled
    out by lower bounds instead of being evaluated, so no n x n matrix is ever built.

    Raises ``TypeError`` for non-numeric data and ``ValueError`` for data that is not
    2-D, is empty, or contains NaN or infinity.
    """
    return _exact_medoid(as_points(X))


def _seed_medoids(points: np.ndarray, k: int, sample_size: int, n_samples: int, seed: int) -> np.ndarray:
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            rough = fasterclara(points, k, sample=sample_size, reps=n_samples, seed=seed)
    except ValueError as exc:
        raise ValueError(
            f"could not choose {k} starting medoids: the data, or a sample of it, has fewer than "
            f"{k} distinct points"
        ) from exc
    return np.asarray(rough["medoids"], dtype=np.int64)


def _polish(points: np.ndarray, start: np.ndarray, max_iter: int) -> KMedoidsResult:
    polished = alternate(points, start, POLISH_METHOD, max_iter=max_iter)
    medoids = np.asarray(polished["medoids"], dtype=np.int64)
    labels, loss = assign_to_medoids(points, medoids)
    return KMedoidsResult(medoids=medoids, labels=labels, loss=loss,
                          polish_iterations=int(polished["iters"]), converged=bool(polished["converged"]))


def kmedoids(X: ArrayLike, k: int, *, sample_size: int = DEFAULT_SAMPLE_SIZE,
             n_samples: int = DEFAULT_N_SAMPLES, polish: bool = True,
             max_iter: int = DEFAULT_MAX_ITER, seed: int = 0) -> KMedoidsResult:
    """Cluster the rows of ``X`` around ``k`` medoids under Euclidean distance.

    FasterPAM runs on ``n_samples`` random samples of ``sample_size`` rows and the medoid
    set with the lowest loss on all rows is kept (FasterCLARA). With ``polish=True`` that
    set is then improved on the full data: rows are assigned to their nearest medoid and
    each medoid is replaced by the exact medoid of its cluster, for at most ``max_iter``
    rounds or until nothing moves. Polishing never increases the loss. Memory stays
    linear in the number of rows.

    ``seed`` makes the result reproducible. Raises ``ValueError`` if ``k`` is not between
    1 and the number of rows, if ``sample_size < k``, or if the data has fewer than ``k``
    distinct points.
    """
    points = as_points(X)
    count = as_cluster_count(k, len(points))
    sample_size = as_int("sample_size", sample_size, 1)
    if sample_size < count:
        raise ValueError(f"sample_size must be at least k ({count}), got {sample_size}")
    n_samples = as_int("n_samples", n_samples, 1)
    max_iter = as_int("max_iter", max_iter, 1)
    seed = as_int("seed", seed, 0)
    if count == 1:
        only = _exact_medoid(points)
        return KMedoidsResult(medoids=np.array([only.index]), labels=np.zeros(len(points), np.int64),
                              loss=only.total_distance, polish_iterations=0, converged=True)
    start = _seed_medoids(points, count, sample_size, n_samples, seed)
    if polish:
        return _polish(points, start, max_iter)
    labels, loss = assign_to_medoids(points, start)
    return KMedoidsResult(medoids=start, labels=labels, loss=loss, polish_iterations=0, converged=False)


def refine(X: ArrayLike, medoids: ArrayLike, *, max_iter: int = DEFAULT_MAX_ITER) -> KMedoidsResult:
    """Polish medoids obtained elsewhere, for example from another k-medoids library.

    ``medoids`` are distinct row indices into ``X``. Rows are assigned to their nearest
    medoid and each medoid is replaced by the exact medoid of its cluster, for at most
    ``max_iter`` rounds or until nothing moves. The loss never increases.
    """
    points = as_points(X)
    start = as_medoids(medoids, len(points))
    return _polish(points, start, as_int("max_iter", max_iter, 1))
