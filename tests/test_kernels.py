"""The ported kernels against brute-force references."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import gaussian, reference_distances, reference_loss, reference_sums, scaled
from medoids._kernels.brute import brute
from medoids._kernels.cpv import cpv
from medoids._kernels.kernels import SMALL_D, Oracle
from medoids._kernels.kmed import fasterpam_blocked, kmedoidspp, loss, nearest

DIMS = [2, SMALL_D, SMALL_D + 1, 40]


@pytest.mark.unit
@pytest.mark.parametrize("d", DIMS)
def test_oracle_rows_and_sums(d):
    X = gaussian(scaled(300), d, seed=1)
    orc = Oracle(X)
    idx = np.array([0, 7, len(X) - 1])
    np.testing.assert_allclose(orc.rows(idx), reference_distances(X[idx], X), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(orc.sums(np.arange(len(X))), reference_sums(X), rtol=1e-10)
    assert orc.sums(np.array([], dtype=np.int64)).shape == (0,)
    assert orc.evals == 3 * len(X) + len(X) ** 2


@pytest.mark.unit
def test_oracle_keeps_near_duplicates_accurate_in_high_dimension():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((40, 30)) * 100
    X = np.concatenate([base, base + 1e-6 * rng.standard_normal((40, 30))])
    rows = Oracle(X).rows(np.arange(40))
    expected = reference_distances(X[:40], X)
    np.testing.assert_allclose(rows[:, 40:], expected[:, 40:], rtol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("flags", [
    {},
    {"use_vps": False},
    {"use_vps": True},
    {"use_planes": False},
    {"use_planes": True},
    {"use_triangle": False},
    {"batch": 2},
    {"init_idx": [3]},
])
def test_cpv_is_exact_under_every_setting(d, flags):
    X = gaussian(scaled(500), d, seed=3)
    sums = reference_sums(X)
    result = cpv(Oracle(X), **flags)
    assert sums[result["medoid"]] == pytest.approx(sums.min(), rel=1e-9)
    assert result["S_min"] == pytest.approx(sums.min(), rel=1e-9)


@pytest.mark.unit
def test_cpv_falls_back_when_nothing_can_be_pruned():
    # Equally spaced points on a circle all have the same total distance, so no bound can
    # rule any of them out and the search has to finish by brute force.
    angle = 2 * np.pi * np.arange(scaled(600, floor=300)) / scaled(600, floor=300)
    X = np.column_stack([np.cos(angle), np.sin(angle)])
    sums = reference_sums(X)
    result = cpv(Oracle(X))
    assert sums[result["medoid"]] == pytest.approx(sums.min(), rel=1e-9)
    assert result["exact_rows"] == len(X)


@pytest.mark.unit
def test_cpv_needs_centred_data():
    with pytest.raises(ValueError, match="centred"):
        cpv(Oracle(gaussian(50, 3), center=False))


@pytest.mark.unit
def test_brute_can_return_every_sum():
    X = gaussian(120, 3, seed=5)
    result = brute(Oracle(X), return_S=True)
    np.testing.assert_allclose(result["S"], reference_sums(X), rtol=1e-10)
    assert result["medoid"] == int(np.argmin(reference_sums(X)))


@pytest.mark.unit
def test_nearest_and_loss():
    X = gaussian(scaled(400), 6, seed=6)
    M = np.array([1, 50, 60])
    Xc = X - X.mean(0)
    labels, dist = nearest(Xc, np.einsum("ij,ij->i", Xc, Xc), M)
    D = reference_distances(X, X[M])
    np.testing.assert_allclose(dist, D.min(1), rtol=1e-9)
    np.testing.assert_allclose(D[np.arange(len(X)), labels], D.min(1), rtol=1e-9)
    assert loss(X, M) == pytest.approx(reference_loss(X, M), rel=1e-9)


@pytest.mark.unit
def test_seeding_picks_distinct_points():
    X = gaussian(scaled(300), 3, seed=7)
    M = kmedoidspp(X, 10, seed=1)
    assert len(set(M.tolist())) == 10
    np.testing.assert_array_equal(M, kmedoidspp(X, 10, seed=1))


@pytest.mark.unit
def test_fasterpam_ends_where_no_single_swap_helps():
    X = gaussian(60, 2, seed=8)
    result = fasterpam_blocked(X, kmedoidspp(X, 3, seed=0), block=16)
    assert result["converged"]
    medoids = result["medoids"]
    best = reference_loss(X, medoids)
    assert result["loss"] == pytest.approx(best, rel=1e-9)
    for slot in range(3):
        for candidate in range(60):
            if candidate in medoids:
                continue
            swapped = medoids.copy()
            swapped[slot] = candidate
            assert reference_loss(X, swapped) >= best * (1 - 1e-9)
