"""End-to-end timing of the public API on synthetic data (arrays in, results out).

Usage: python e2e_api.py <n_points> [<n_points> ...]

Times include input validation and everything else the public functions do. The first
call's JIT compilation happens in a warm-up and is reported separately. One JSON object
is logged per measurement.
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

import numba
import numpy as np

import medoids

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("bench")
REPEATS = 3
K = 10
DATASETS = (("gauss", 10), ("clustered", 2), ("clustered", 10), ("clustered", 100))
WARMUP_N = 3000


def gauss(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Isotropic standard Gaussian."""
    return np.random.default_rng(seed).standard_normal((n, d))


def clustered(n: int, d: int, seed: int = 0, k: int = 10, sep: float = 3.0) -> np.ndarray:
    """Mixture of k unit-variance Gaussians with unequal (geometric) weights."""
    rng = np.random.default_rng(seed)
    weights = 0.7 ** np.arange(k)
    labels = rng.choice(k, size=n, p=weights / weights.sum())
    centres = sep * rng.standard_normal((k, d))
    return centres[labels] + rng.standard_normal((n, d))


GENERATORS = {"gauss": gauss, "clustered": clustered}


def best_time(fn: Callable[[], Any]) -> tuple[float, Any]:
    times, result = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), result


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [100_000]
    t0 = time.perf_counter()
    for name, d in DATASETS:
        X = GENERATORS[name](WARMUP_N, d)
        medoids.medoid(X)
        medoids.kmedoids(X, 5)
    log.info(json.dumps({"warmup_s": round(time.perf_counter() - t0, 2), "threads": numba.get_num_threads(),
                         "machine": platform.machine(), "system": platform.system(),
                         "python": platform.python_version(), "numba": numba.__version__,
                         "numpy": np.__version__, "medoids": medoids.__version__}))
    for n in sizes:
        for name, d in DATASETS:
            X = GENERATORS[name](n, d)
            base = {"data": name, "d": d, "n": n}
            seconds, one = best_time(lambda X=X: medoids.medoid(X))
            log.info(json.dumps({**base, "call": "medoid", "seconds": round(seconds, 4), "index": one.index}))
            for polish in (False, True):
                seconds, many = best_time(lambda X=X, polish=polish: medoids.kmedoids(X, K, polish=polish))
                log.info(json.dumps({**base, "call": "kmedoids", "k": K, "polish": polish,
                                     "seconds": round(seconds, 4), "loss": round(many.loss, 2),
                                     "polish_iterations": many.polish_iterations}))


if __name__ == "__main__":
    main()
