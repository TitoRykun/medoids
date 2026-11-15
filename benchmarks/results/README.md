# Raw benchmark results

- `comparison.jsonl`: one JSON object per run of the comparisons quoted in the top-level
  README. Methods starting with `local-` are this package's kernels as they existed in the
  research code they were ported from (`local-hybrid` is what `medoids.kmedoids` runs,
  `local-cpv` is `medoids.medoid`, `local-fasterpam` is full FasterPAM without sampling,
  which the package does not expose). Methods starting with `pypi-` use `kmedoids` 0.5.5:
  `pypi-fasterpam-*` on the full matrix (`pypi-matrix` is the time to build it) and
  `pypi-clara-*` on 5 samples of 5,000 rows. `sklearn-rowsum` is the hand-written exact
  medoid. `wall` and `cpu` are seconds; every `loss` is recomputed in float64 from the
  medoid indices by one shared function.
- `e2e_api.jsonl`: output of `benchmarks/e2e_api.py 100000 1000000`, run against the
  installed 0.1.0 wheel. It returns the same medoids and losses as the research code.
- Machine: AMD Ryzen 5 5600X, Windows 11, Python 3.12.1, NumPy 2.5.4, Numba 0.68.1.

`python benchmarks/summarise.py benchmarks/results/comparison.jsonl` rebuilds the tables.

The harness that produced `comparison.jsonl` depends on the research repository and is not
included; `benchmarks/e2e_api.py` is the benchmark that runs against this package.
