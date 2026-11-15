"""A scikit-learn style wrapper around :func:`medoids.kmedoids`."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._assign import assign_to_centres
from ._validate import as_points
from .api import DEFAULT_MAX_ITER, DEFAULT_N_SAMPLES, DEFAULT_SAMPLE_SIZE, kmedoids

PARAMETERS = ("n_clusters", "sample_size", "n_samples", "polish", "max_iter", "random_state")


class KMedoids:
    """k-medoids clustering with ``fit`` / ``predict``, for Euclidean data.

    After ``fit``: ``medoid_indices_`` (rows chosen as medoids), ``cluster_centers_``
    (their coordinates), ``labels_``, ``inertia_`` (total distance to the nearest medoid),
    ``n_iter_`` (polish rounds) and ``n_features_in_``. Parameters are checked when
    ``fit`` is called. It does not depend on scikit-learn.
    """

    def __init__(self, n_clusters: int = 8, *, sample_size: int = DEFAULT_SAMPLE_SIZE,
                 n_samples: int = DEFAULT_N_SAMPLES, polish: bool = True,
                 max_iter: int = DEFAULT_MAX_ITER, random_state: int = 0) -> None:
        self.n_clusters = n_clusters
        self.sample_size = sample_size
        self.n_samples = n_samples
        self.polish = polish
        self.max_iter = max_iter
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Constructor parameters by name (``deep`` is accepted for scikit-learn compatibility)."""
        return {name: getattr(self, name) for name in PARAMETERS}

    def set_params(self, **params: object) -> KMedoids:
        unknown = sorted(set(params) - set(PARAMETERS))
        if unknown:
            raise ValueError(f"unknown parameter(s) {unknown}; valid ones are {list(PARAMETERS)}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def fit(self, X: ArrayLike, y: object = None) -> KMedoids:
        """Cluster ``X``; ``y`` is ignored."""
        points = as_points(X)
        result = kmedoids(points, self.n_clusters, sample_size=self.sample_size, n_samples=self.n_samples,
                          polish=self.polish, max_iter=self.max_iter, seed=self.random_state)
        centres = points[result.medoids]
        centres.setflags(write=False)
        self.medoid_indices_ = result.medoids
        self.cluster_centers_ = centres
        self.labels_ = result.labels
        self.inertia_ = result.loss
        self.n_iter_ = result.polish_iterations
        self.n_features_in_ = points.shape[1]
        return self

    def fit_predict(self, X: ArrayLike, y: object = None) -> np.ndarray:
        """Cluster ``X`` and return the cluster of every row."""
        return self.fit(X).labels_

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Cluster of every row of ``X``: the index of its nearest medoid."""
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("this KMedoids instance is not fitted yet; call fit first")
        points = as_points(X)
        if points.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {points.shape[1]} features, but KMedoids was fitted with "
                             f"{self.n_features_in_} features")
        return assign_to_centres(points, self.cluster_centers_)
