"""The scikit-learn style estimator."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import blobs, gaussian, reference_distances, reference_loss, scaled
from medoids import KMedoids, kmedoids


@pytest.mark.integration
def test_fit_exposes_the_fitted_attributes():
    X, _ = blobs(scaled(600, floor=200), 4, k=3, seed=1)
    model = KMedoids(n_clusters=3, random_state=5)
    assert model.fit(X) is model
    assert model.medoid_indices_.shape == (3,)
    np.testing.assert_array_equal(model.cluster_centers_, X[model.medoid_indices_])
    assert model.labels_.shape == (len(X),)
    assert model.inertia_ == pytest.approx(reference_loss(X, model.medoid_indices_), rel=1e-9)
    assert model.n_features_in_ == 4
    assert model.n_iter_ >= 1


@pytest.mark.integration
def test_matches_the_function():
    X = gaussian(scaled(500), 3, seed=2)
    model = KMedoids(n_clusters=4, sample_size=100, n_samples=2, random_state=7).fit(X)
    direct = kmedoids(X, 4, sample_size=100, n_samples=2, seed=7)
    np.testing.assert_array_equal(model.medoid_indices_, direct.medoids)
    np.testing.assert_array_equal(model.labels_, direct.labels)


@pytest.mark.integration
def test_predict_assigns_the_nearest_medoid():
    X, _ = blobs(scaled(600, floor=200), 3, k=4, seed=3)
    model = KMedoids(n_clusters=4).fit(X)
    np.testing.assert_array_equal(model.predict(X), model.labels_)
    new = gaussian(50, 3, seed=4) * 5
    D = reference_distances(new, model.cluster_centers_)
    np.testing.assert_allclose(D[np.arange(50), model.predict(new)], D.min(1), rtol=1e-9)


@pytest.mark.integration
def test_fit_predict_returns_the_labels():
    X = gaussian(scaled(300), 2, seed=5)
    model = KMedoids(n_clusters=3, random_state=1)
    labels = model.fit_predict(X)
    np.testing.assert_array_equal(labels, model.labels_)


@pytest.mark.unit
def test_predict_before_fit_fails():
    with pytest.raises(RuntimeError, match="not fitted"):
        KMedoids().predict(gaussian(5, 2))


@pytest.mark.unit
def test_predict_checks_the_number_of_features():
    model = KMedoids(n_clusters=2).fit(gaussian(80, 3))
    with pytest.raises(ValueError, match="3 features"):
        model.predict(gaussian(5, 2))


@pytest.mark.unit
def test_parameters_can_be_read_and_replaced():
    model = KMedoids(n_clusters=6, polish=False)
    params = model.get_params()
    assert params == {"n_clusters": 6, "sample_size": 5000, "n_samples": 5, "polish": False,
                      "max_iter": 100, "random_state": 0}
    assert model.set_params(n_clusters=2, random_state=9) is model
    assert model.get_params()["n_clusters"] == 2
    assert model.get_params()["random_state"] == 9
    with pytest.raises(ValueError, match="unknown parameter"):
        model.set_params(colour="red")


@pytest.mark.unit
def test_bad_parameters_surface_at_fit():
    with pytest.raises(ValueError, match="k must be"):
        KMedoids(n_clusters=0).fit(gaussian(20, 2))
