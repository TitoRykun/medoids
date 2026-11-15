"""The exact medoid against a brute-force reference."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import blobs, gaussian, reference_sums, scaled
from medoids import MedoidResult, medoid

SHAPES = [(1, 3), (2, 3), (3, 1), (50, 2), (64, 5), (65, 5), (400, 1), (400, 2), (400, 10), (400, 20),
          (1500, 3), (1500, 40)]


def assert_is_medoid(X: np.ndarray, result: MedoidResult) -> None:
    sums = reference_sums(X)
    assert 0 <= result.index < len(X)
    assert sums[result.index] == pytest.approx(sums.min(), rel=1e-9, abs=1e-9)
    assert result.total_distance == pytest.approx(sums.min(), rel=1e-9, abs=1e-9)


@pytest.mark.integration
@pytest.mark.parametrize(("n", "d"), SHAPES)
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_brute_force_on_gaussian_data(n, d, seed):
    X = gaussian(min(n, scaled(n, floor=min(n, 70))), d, seed)
    assert_is_medoid(X, medoid(X))


@pytest.mark.integration
@pytest.mark.parametrize("d", [2, 10, 30])
def test_matches_brute_force_on_clustered_data(d):
    X, _ = blobs(scaled(1200), d, k=5, seed=3, spread=1.0)
    assert_is_medoid(X, medoid(X))


@pytest.mark.integration
def test_matches_brute_force_on_heavy_tailed_data():
    rng = np.random.default_rng(4)
    X = rng.standard_t(df=2, size=(scaled(1000), 6))
    assert_is_medoid(X, medoid(X))


@pytest.mark.integration
def test_duplicated_points():
    base = gaussian(40, 4, seed=5)
    X = np.repeat(base, 8, axis=0)
    assert_is_medoid(X, medoid(X))


@pytest.mark.unit
@pytest.mark.parametrize("d", [3, 6, 20])
def test_identical_points_have_zero_total_distance(d):
    X = np.full((100, d), 2.5)
    result = medoid(X)
    assert 0 <= result.index < 100
    assert result.total_distance == 0.0


@pytest.mark.integration
def test_far_from_the_origin():
    X = gaussian(scaled(600), 20, seed=6)
    shifted = X + 1e6
    assert medoid(shifted).index == medoid(X).index
    assert_is_medoid(shifted, medoid(shifted))


@pytest.mark.unit
@pytest.mark.parametrize("convert", [
    lambda X: X.tolist(),
    lambda X: X.astype(np.float32),
    lambda X: np.asfortranarray(X),
    lambda X: X[::-1],
])
def test_accepts_array_likes(convert):
    X = np.round(gaussian(120, 3, seed=7) * 8) / 8  # exactly representable in float32
    given = convert(X)
    assert_is_medoid(np.asarray(given, dtype=np.float64), medoid(given))


@pytest.mark.unit
def test_accepts_integer_data():
    X = np.random.default_rng(8).integers(-50, 50, size=(150, 3))
    assert_is_medoid(X.astype(np.float64), medoid(X))


@pytest.mark.unit
def test_does_not_modify_the_input():
    X = gaussian(200, 5, seed=9)
    before = X.copy()
    medoid(X)
    np.testing.assert_array_equal(X, before)


@pytest.mark.unit
def test_result_is_immutable_and_plain():
    result = medoid(gaussian(30, 2))
    assert isinstance(result.index, int)
    assert isinstance(result.total_distance, float)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.index = 0


@pytest.mark.unit
@pytest.mark.parametrize(("bad", "match"), [
    (np.zeros(5), "2-D"),
    (np.zeros((2, 3, 4)), "2-D"),
    (np.zeros((0, 3)), "at least one point"),
    (np.zeros((4, 0)), "at least one feature"),
    (np.array([[1.0, np.nan]]), "finite"),
    (np.array([[1.0, np.inf]]), "finite"),
])
def test_rejects_bad_input(bad, match):
    with pytest.raises(ValueError, match=match):
        medoid(bad)


@pytest.mark.unit
def test_rejects_non_numeric_input():
    with pytest.raises(TypeError, match="numeric"):
        medoid([["a", "b"], ["c", "d"]])
