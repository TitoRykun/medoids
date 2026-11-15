# medoids

Exact medoid and k-medoids for large Euclidean data, without a distance matrix.

```python
import numpy as np
from medoids import medoid, kmedoids, refine, KMedoids

X = np.random.default_rng(0).standard_normal((1_000_000, 10))

m = medoid(X)                    # exact: m.index, m.total_distance
c = kmedoids(X, 10)              # c.medoids, c.labels, c.loss
c = refine(X, [3, 14, 15, 92])   # polish medoids that came from another library

model = KMedoids(n_clusters=10).fit(X)   # scikit-learn style
model.predict(X[:5])
```

On a 6-core desktop, the first two calls above take about 0.9 s and 3.3 s.

## Install

Not on PyPI yet. From source:

```bash
pip install git+https://github.com/TitoRykun/medoids
```

Requires Python 3.10+, NumPy and Numba. Pure Python: there is nothing to compile at
install time, and it runs on Linux, macOS (including Apple Silicon) and Windows. The first
call in a process compiles the kernels, which takes a few seconds.

## What it does

| Call | Result | Method |
|---|---|---|
| `medoid(X)` | The row with the smallest total distance to all rows, exactly | Lower bounds rule out most rows, so only a few are evaluated in full |
| `kmedoids(X, k)` | `k` medoids, labels and loss | FasterPAM on random samples (FasterCLARA), then an exact polish on all rows |
| `refine(X, medoids)` | Improved medoids, labels and loss | The polish step alone, starting from medoids you supply |
| `KMedoids` | `fit` / `predict` / `fit_predict` | A thin class around `kmedoids`; no scikit-learn dependency |

Memory is linear in the number of rows everywhere. No n x n matrix is built.

The polish step assigns every row to its nearest medoid and replaces each medoid by the
exact medoid of its cluster, until nothing moves. It never increases the loss.

## When to use something else

- **Any distance other than Euclidean.** This package only handles Euclidean distance on
  numeric arrays. For a precomputed dissimilarity matrix or another metric, use
  [`kmedoids`](https://pypi.org/project/kmedoids/).
- **Small data.** Below roughly 10,000 rows a full-matrix FasterPAM is as fast and gives
  an equal or slightly better loss.
- **You need the best possible loss and the matrix fits in memory.** Full FasterPAM
  searches more thoroughly than sampling plus polish.

## Benchmarks

All numbers were measured on one machine: AMD Ryzen 5 5600X (6 cores, 12 threads),
Windows 11, Python 3.12.1, NumPy 2.5.4, Numba 0.68.1, with all 12 threads available. Data
is synthetic: a standard Gaussian, and a mixture of 10 Gaussians with unequal weights
("clustered"). Compilation time is excluded. Raw results are in
[`benchmarks/results`](benchmarks/results).

### Exact medoid

There is no maintained package for the exact medoid, so the comparison is with what a user
would write: chunked `sklearn.metrics.pairwise_distances_chunked`, row sums, `argmin`. Both
return the same row in every case.

| Data | Rows | scikit-learn row sums | `medoid` | Ratio |
|---|---|---|---|---|
| Gaussian, 10 dims | 100,000 | 65.2 s | 0.094 s | 694x |
| Clustered, 2 dims | 100,000 | 64.5 s | 0.075 s | 864x |
| Clustered, 10 dims | 100,000 | 65.7 s | 0.23 s | 283x |
| Clustered, 100 dims | 100,000 | 67.1 s | 1.32 s | 51x |
| Clustered, 100 dims | 10,000 | 0.64 s | 0.57 s | 1.1x |

At 1,000,000 rows `medoid` takes 0.9 s (Gaussian, 10 dims), 1.0 s (clustered, 2 dims),
2.1 s (clustered, 10 dims) and 10.6 s (clustered, 100 dims). The row-sum baseline was not
run at that size. The advantage shrinks as the dimension grows and the data set gets
smaller: at 10,000 rows in 100 dimensions there is essentially none.

### k-medoids against full-matrix FasterPAM

[`kmedoids`](https://pypi.org/project/kmedoids/) 0.5.5 needs the full distance matrix
(10 GB in float32 at 50,000 rows). Times are medians over 3 starting sets at 50,000 rows;
loss is relative to `kmedoids` FasterPAM on one thread.

| Data | k | `kmedoids`: matrix | `kmedoids`: FasterPAM, all cores | This package | Loss difference |
|---|---|---|---|---|---|
| Clustered, 2 dims | 10 | 15.4 s | 23.2 s | 1.6 s | +0.08% |
| Clustered, 10 dims | 10 | 15.2 s | 12.4 s | 1.1 s | 0.00% |
| Clustered, 10 dims | 50 | 15.2 s | 22.9 s | 1.7 s | +1.08% |
| Clustered, 100 dims | 10 | 17.1 s | 17.0 s | 8.5 s | 0.00% |
| Gaussian, 10 dims | 10 | 15.4 s | 12.6 s | 0.97 s | +0.38% |

Across all 45 runs at 10,000 to 50,000 rows the loss was between 0.96% lower and 1.11%
higher. At 10,000 rows the two are about equally fast, and in 100 dimensions `kmedoids` is
4x faster there (1.5 s against 6.0 s).

### k-medoids against sampling with `kmedoids`

The speed above comes from sampling, which anyone can do: run `kmedoids` FasterPAM on 5
samples of 5,000 rows and keep the best. This is the fair baseline. Both sides used
identical samples and starting medoids, 3 starting sets per configuration, 75 paired runs.

| Condition | `kmedoids` on samples | This package | Loss difference |
|---|---|---|---|
| Up to 10 dims, 10,000–100,000 rows | 1.6–2.3 s | 0.97–2.3 s | −2.81% to +0.02% |
| Up to 10 dims, 1,000,000 rows | 2.3–4.3 s | 3.3–20.1 s | −3.89% to +0.12% |
| 100 dims, 10,000–100,000 rows | 1.7–2.0 s | 6.0–10.3 s | −1.85% to −0.08% |
| 100 dims, 1,000,000 rows | 4.1 s | 23.2 s | −3.77% to −3.46% |

So this package is **not faster** than sampling with `kmedoids`; at 1,000,000 rows and in
100 dimensions it is several times slower. What it adds is quality: the loss was lower in
60 of the 75 runs, equal in 11 and higher in 4 (by at most 0.12%). The gain is about 2–4%
on clustered data and zero on Gaussian data, and it comes entirely from the polish step,
which is also available on its own as `refine`.

### Caveats

- One machine, synthetic data only. Real data may behave differently.
- 3 starting sets per configuration; treat small differences as noise.
- `kmedoids` was given a float32 matrix, which halves its memory use.
- `kmedoids` accepts any dissimilarity; this package only Euclidean distance.

## How it works

**Exact medoid.** Every row keeps a lower bound on its total distance, the largest of: a
Jensen bound from the distance to the mean; a "virtual pivot" bound computed from a few
weighted moments of the data; cutting planes from the subgradient at each fully evaluated
row; and the triangle bound used by trimed (Newling and Fleuret, 2017). A row is dropped
once its bound exceeds the best total found so far. Surviving rows are evaluated in full,
most promising first. If the bounds stop helping, the search finishes by blocked brute
force, so the result is exact either way (ties within a relative 1e-9 may resolve either
way).

**k-medoids.** FasterPAM (Schubert and Rousseeuw, 2021) with k-medoids++ seeding runs on
each sample, with distances computed on the fly. The best medoid set on the full data is
then polished by alternating assignment and exact per-cluster medoids (Park and Jun,
2009), using the exact-medoid search above with the previous medoid as a warm start.

## Details

- **Reproducibility:** `seed` (or `random_state`) fixes the samples and the seeding.
- **Threads:** the kernels use all cores through Numba and BLAS. Limit them with
  `numba.set_num_threads(n)`.
- **Input:** anything `numpy.asarray` turns into a 2-D numeric array. It is converted to
  float64 and never modified. NaN and infinity are rejected.
- **Results:** immutable; the arrays are read-only.
- **Errors:** `kmedoids` raises `ValueError` if the data has fewer than `k` distinct points.

## Limitations and roadmap

- Euclidean distance only; float64 only (float32 input is converted, which doubles its memory).
- No sample weights.
- The exact medoid has no advantage at around 10,000 rows in 100 dimensions.
- The numerical kernels in `src/medoids/_kernels` are ported verbatim from research code
  and are not type-annotated yet.
- Not on PyPI yet.

## Development

```bash
pip install -e ".[dev]"
```

```bash
ruff check . && pytest -q
```

Line coverage is measured with the kernels uncompiled and smaller data sets:

```bash
NUMBA_DISABLE_JIT=1 MEDOIDS_TEST_SCALE=0.12 pytest -q --cov=medoids
```

## Licence

MIT.
