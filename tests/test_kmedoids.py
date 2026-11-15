"""k-medoids through the public function."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import blobs, gaussian, reference_distances, reference_loss, reference_sums, scaled
from medoids import KMedoidsResult, kmedoids, medoid


def assert_consistent(X: np.ndarray, result: KMedoidsResult, k: int) -> None:
    """Medoids are distinct points, labels pick a nearest medoid, loss is their total distance."""
    assert result.medoids.shape == (k,)
    assert result.medoids.dtype == np.int64
    assert len(set(result.medoids.tolist())) == k
    assert result.medoids.min() >= 0
    assert result.medoids.max() < len(X)
    assert result.labels.shape == (len(X),)
    assert result.labels.dtype == np.int64
    D = reference_distances(X, X[result.medoids])
    chosen = D[np.arange(len(X)), result.labels]
    np.testing.assert_allclose(chosen, D.min(1), rtol=1e-9, atol=1e-9)
    assert result.loss == pytest.approx(reference_loss(X, result.medoids), rel=1e-9)


@pytest.mark.integration
@pytest.mark.parametrize(("d", "k"), [(2, 3), (10, 5), (30, 4)])
@pytest.mark.parametrize("polish", [True, False])
def test_result_is_consistent(d, k, polish):
    X = gaussian(scaled(900), d, seed=1)
    assert_consistent(X, kmedoids(X, k, polish=polish), k)


@pytest.mark.integration
def test_sampling_path_when_data_exceeds_the_sample():
    X = gaussian(scaled(1500, floor=300), 4, seed=2)
    assert_consistent(X, kmedoids(X, 6, sample_size=120, n_samples=3), 6)


@pytest.mark.integration
def test_recovers_well_separated_blobs():
    X, truth = blobs(scaled(800, floor=200), 5, k=4, seed=3)
    result = kmedoids(X, 4)
    assert sorted(truth[result.medoids].tolist()) == [0, 1, 2, 3]
    assert len(set(zip(truth.tolist(), result.labels.tolist(), strict=True))) == 4


@pytest.mark.integration
def test_polished_medoids_are_the_exact_medoids_of_their_clusters():
    X = gaussian(scaled(900), 3, seed=4)
    result = kmedoids(X, 5, sample_size=100)
    assert result.converged
    for c, m in enumerate(result.medoids):
        members = np.nonzero(result.labels == c)[0]
        sums = reference_sums(X[members])
        own = sums[np.nonzero(members == m)[0][0]]
        assert own == pytest.approx(sums.min(), rel=1e-9)


@pytest.mark.integration
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_polish_never_increases_the_loss(seed):
    X, _ = blobs(scaled(1000, floor=300), 6, k=7, seed=seed, spread=2.0)
    rough = kmedoids(X, 7, sample_size=100, polish=False, seed=seed)
    polished = kmedoids(X, 7, sample_size=100, polish=True, seed=seed)
    assert polished.loss <= rough.loss * (1 + 1e-12)
    assert rough.polish_iterations == 0
    assert not rough.converged


@pytest.mark.integration
def test_same_seed_gives_the_same_answer():
    X = gaussian(scaled(700), 5, seed=5)
    a, b = kmedoids(X, 4, seed=11), kmedoids(X, 4, seed=11)
    np.testing.assert_array_equal(a.medoids, b.medoids)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.loss == b.loss


@pytest.mark.integration
def test_one_cluster_is_the_medoid():
    X = gaussian(scaled(500), 4, seed=6)
    result = kmedoids(X, 1)
    assert result.medoids.tolist() == [medoid(X).index]
    assert result.labels.tolist() == [0] * len(X)
    assert result.loss == pytest.approx(medoid(X).total_distance, rel=1e-12)


@pytest.mark.integration
def test_as_many_clusters_as_points():
    X = gaussian(12, 3, seed=7)
    result = kmedoids(X, 12)
    assert sorted(result.medoids.tolist()) == list(range(12))
    assert result.loss == pytest.approx(0.0, abs=1e-12)
    assert_consistent(X, result, 12)


@pytest.mark.unit
def test_arrays_are_read_only_and_input_is_untouched():
    X = gaussian(150, 3, seed=8)
    before = X.copy()
    result = kmedoids(X, 3)
    np.testing.assert_array_equal(X, before)
    with pytest.raises(ValueError, match="read-only"):
        result.medoids[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        result.labels[0] = 0


@pytest.mark.integration
def test_duplicated_points_with_enough_distinct_ones():
    X = np.repeat(gaussian(40, 3, seed=12), 6, axis=0)
    result = kmedoids(X, 5)
    assert_consistent(X, result, 5)
    assert len(np.unique(X[result.medoids], axis=0)) == 5


@pytest.mark.integration
def test_one_dimensional_feature():
    X = gaussian(scaled(400), 1, seed=13)
    assert_consistent(X, kmedoids(X, 3), 3)


@pytest.mark.unit
def test_too_few_distinct_points():
    X = np.repeat(gaussian(3, 2, seed=9), 30, axis=0)
    with pytest.raises(ValueError, match="distinct"):
        kmedoids(X, 5)


@pytest.mark.unit
@pytest.mark.parametrize(("kwargs", "error", "match"), [
    ({"k": 0}, ValueError, "k must be"),
    ({"k": 101}, ValueError, "k must be"),
    ({"k": 2.5}, TypeError, "k must be an integer"),
    ({"k": True}, TypeError, "k must be an integer"),
    ({"k": 3, "sample_size": 2}, ValueError, "sample_size"),
    ({"k": 3, "n_samples": 0}, ValueError, "n_samples"),
    ({"k": 3, "max_iter": 0}, ValueError, "max_iter"),
    ({"k": 3, "seed": -1}, ValueError, "seed"),
    ({"k": 3, "seed": 1.5}, TypeError, "seed must be an integer"),
])
def test_rejects_bad_parameters(kwargs, error, match):
    X = gaussian(100, 2, seed=10)
    with pytest.raises(error, match=match):
        kmedoids(X, **kwargs)


@pytest.mark.unit
def test_rejects_bad_data():
    with pytest.raises(ValueError, match="finite"):
        kmedoids(np.array([[0.0, 1.0], [np.nan, 2.0], [1.0, 1.0]]), 2)
