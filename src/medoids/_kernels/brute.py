# Ported verbatim from the medoids research code (medoid-opt/medoid/baselines.py). Do not edit by hand.
import numpy as np

from .kernels import Oracle


def brute(orc: Oracle, return_S=False):
    e0 = orc.evals
    S = orc.sums(np.arange(orc.n))
    m = int(np.argmin(S))
    out = {"medoid": m, "S_min": float(S[m]), "evals": orc.evals - e0, "exact_rows": orc.n}
    if return_S:
        out["S"] = S
    return out
