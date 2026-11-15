"""Polishing medoids that came from somewhere else."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import blobs, gaussian, reference_loss, scaled
from medoids import refine


@pytest.mark.integration
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_never_increases_the_loss(seed):
    X, _ = blobs(scaled(900, floor=200), 4, k=6, seed=seed, spread=2.0)
    start = np.random.default_rng(seed).choice(len(X), 6, replace=False)
    result = refine(X, start)
    assert result.loss <= reference_loss(X, start) * (1 + 1e-12)
    assert result.loss == pytest.approx(reference_loss(X, result.medoids), rel=1e-9)
    assert result.polish_iterations >= 1


@pytest.mark.integration
def test_a_converged_result_is_a_fixed_point():
    X = gaussian(scaled(600), 3, seed=3)
    first = refine(X, [0, 1, 2, 3])
    assert first.converged
    again = refine(X, first.medoids)
    np.testing.assert_array_equal(again.medoids, first.medoids)
    assert again.polish_iterations == 1


@pytest.mark.integration
def test_reports_when_the_iteration_limit_stops_it():
    X, _ = blobs(scaled(900, floor=300), 2, k=5, seed=4, spread=3.0)
    result = refine(X, [0, 1, 2, 3, 4], max_iter=1)
    assert result.polish_iterations == 1
    assert not result.converged


@pytest.mark.unit
def test_accepts_lists_and_leaves_the_input_alone():
    X = gaussian(120, 2, seed=5)
    start = np.array([5, 17, 40])
    before = start.copy()
    result = refine(X, start)
    np.testing.assert_array_equal(start, before)
    assert result.medoids.shape == (3,)
    assert refine(X, [5, 17, 40]).loss == result.loss


@pytest.mark.unit
@pytest.mark.parametrize(("bad", "error", "match"), [
    ([], ValueError, "at least one"),
    ([[0, 1]], ValueError, "1-D"),
    ([0, 120], ValueError, "between 0 and"),
    ([-1, 3], ValueError, "between 0 and"),
    ([4, 4], ValueError, "distinct"),
    ([0.5, 2.0], TypeError, "integer"),
])
def test_rejects_bad_medoids(bad, error, match):
    X = gaussian(120, 2, seed=6)
    with pytest.raises(error, match=match):
        refine(X, bad)


@pytest.mark.unit
def test_rejects_bad_iteration_limit():
    with pytest.raises(ValueError, match="max_iter"):
        refine(gaussian(50, 2), [0, 1], max_iter=0)
