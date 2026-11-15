# Ported verbatim from the medoids research code (kmedoids-opt/kmedoids.py). Do not edit by hand.
import time

import numba as nb
import numpy as np

from .brute import brute
from .cpv import cpv
from .kernels import Oracle


@nb.njit(parallel=True, cache=True)
def _dist_to(X, C, lab, out):
    for i in nb.prange(X.shape[0]):
        s = 0.0
        for t in range(X.shape[1]):
            u = X[i, t] - C[lab[i], t]
            s += u * u
        out[i] = np.sqrt(s)


def nearest(X, sq, M):
    """labels, distance of every point to its nearest medoid (indices M)."""
    C = np.ascontiguousarray(X[M])
    D2 = sq[:, None] + sq[M][None, :] - 2.0 * (X @ C.T)
    lab = np.argmin(D2, 1)
    # exact distance to the chosen medoid, computed directly (no GEMM cancellation and no
    # n x d temporary)
    dist = np.empty(len(X))
    _dist_to(X, C, lab, dist)
    return lab, dist


def loss(X, M, centered=False):
    if not centered:
        X = X - X.mean(0)
    sq = np.einsum("ij,ij->i", X, X)
    return float(nearest(X, sq, np.asarray(M))[1].sum())


def kmedoidspp(X, k, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    M = [int(rng.integers(n))]
    dmin = np.sqrt(((X - X[M[0]]) ** 2).sum(1))
    for _ in range(1, k):
        p = dmin / dmin.sum()
        j = int(rng.choice(n, p=p))
        M.append(j)
        dmin = np.minimum(dmin, np.sqrt(((X - X[j]) ** 2).sum(1)))
    return np.array(M)


def _medoid_cpv(Xc, prev):
    o = Oracle(Xc)
    if len(Xc) <= 64:
        r = brute(o)
    else:
        r = cpv(o, init_idx=[prev] if prev is not None else None)
    return r["medoid"], o.evals


def alternate(X, M0, method="cpv-warm", max_iter=100):
    X = np.ascontiguousarray(X, dtype=np.float64)
    X = X - X.mean(0)  # centring keeps the GEMM-based assignment accurate
    sq = np.einsum("ij,ij->i", X, X)
    M = np.array(M0, dtype=np.int64)
    f = MEDOID[method]
    evals, t_med, t_asg, it, converged = 0, 0.0, 0.0, 0, False
    for it in range(1, max_iter + 1):
        t = time.perf_counter()
        lab, _ = nearest(X, sq, M)
        evals += len(X) * len(M)
        t_asg += time.perf_counter() - t
        newM = M.copy()
        t = time.perf_counter()
        for c in range(len(M)):
            idx = np.nonzero(lab == c)[0]
            if len(idx) == 0:
                continue
            pos = np.searchsorted(idx, M[c])
            prev = int(pos) if pos < len(idx) and idx[pos] == M[c] else None
            m, e = f(X[idx], prev)
            evals += e
            newM[c] = idx[m]
        t_med += time.perf_counter() - t
        if np.array_equal(newM, M):
            converged = True
            break
        M = newM
    return {"medoids": M, "loss": loss(X, M, centered=True), "iters": it, "evals": evals,
            "t_medoid": t_med, "t_assign": t_asg,
            "converged": converged}


@nb.njit(parallel=True, cache=True)
def _nearest2(X, M, near, sec, dn, ds):
    n, k = X.shape[0], len(M)
    for o in nb.prange(n):
        b1, b2, i1, i2 = np.inf, np.inf, -1, -1
        for i in range(k):
            s = 0.0
            for t in range(X.shape[1]):
                u = X[o, t] - X[M[i], t]
                s += u * u
            v = np.sqrt(s)
            if v < b1:
                b2, i2, b1, i1 = b1, i1, v, i
            elif v < b2:
                b2, i2 = v, i
        near[o], sec[o], dn[o], ds[o] = i1, i2, b1, b2


@nb.njit(parallel=True, cache=True)
def _after_swap(X, M, i, row, near, sec, dn, ds):
    """Update nearest / second nearest after medoid slot i became the point whose distance
    row is `row`. O(n) except for points whose nearest or second nearest was slot i,
    which are rescanned over the k medoids (as in Schubert & Rousseeuw 2021)."""
    n, k = X.shape[0], len(M)
    rescans = 0
    for o in nb.prange(n):
        doj = row[o]
        if near[o] == i or sec[o] == i:
            b1, b2, i1, i2 = np.inf, np.inf, -1, -1
            for m in range(k):
                if m == i:
                    v = doj
                else:
                    s = 0.0
                    for t in range(X.shape[1]):
                        u = X[o, t] - X[M[m], t]
                        s += u * u
                    v = np.sqrt(s)
                if v < b1:
                    b2, i2, b1, i1 = b1, i1, v, m
                elif v < b2:
                    b2, i2 = v, m
            near[o], sec[o], dn[o], ds[o] = i1, i2, b1, b2
            rescans += 1
        elif doj < dn[o]:
            sec[o], ds[o] = near[o], dn[o]
            near[o], dn[o] = i, doj
        elif doj < ds[o]:
            sec[o], ds[o] = i, doj
    return rescans


@nb.njit(cache=True)
def _fp_block(X, R, start, M, near, sec, dn, ds, rem, is_med, st):
    """Process candidates start .. start+len(R)-1 in order, exactly as _fasterpam does, with
    their distance rows precomputed in R. st = [steps_since, swaps, evals, converged]."""
    n, k = X.shape[0], len(M)
    tol = 1e-12 * dn.sum()
    for b in range(R.shape[0]):
        if st[0] >= n:
            st[3] = 1
            return
        xc = start + b
        if not is_med[xc]:
            row = R[b]
            dT = rem.copy()
            acc = 0.0
            for o in range(n):
                doj = row[o]
                if doj < dn[o]:
                    acc += doj - dn[o]
                    dT[near[o]] += dn[o] - ds[o]
                elif doj < ds[o]:
                    dT[near[o]] += doj - ds[o]
            i = np.argmin(dT)
            if dT[i] + acc < -tol:
                is_med[M[i]] = False
                M[i] = xc
                is_med[xc] = True
                st[1] += 1
                st[2] += _after_swap(X, M, i, row, near, sec, dn, ds) * (k - 1)
                rem[:] = 0.0
                for o in range(n):
                    rem[near[o]] += ds[o] - dn[o]
                tol = 1e-12 * dn.sum()
                st[0] = 0
        st[0] += 1


def fasterpam_blocked(X, M0, max_passes=100, block=256):
    """FasterPAM with candidate distance rows computed in blocks by matrix products (the
    medoid-opt Oracle: GEMM plus direct recomputation of near pairs). Distance rows do not
    depend on the current medoids, so the swap sequence is the same as in `fasterpam`
    (up to floating-point differences in the distances). O(n) memory per block row."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    X = X - X.mean(0)
    n = len(X)
    M = np.array(M0, dtype=np.int64).copy()
    k = len(M)
    orc = Oracle(X, center=False)
    near = np.empty(n, np.int64)
    sec = np.empty(n, np.int64)
    dn = np.empty(n)
    ds = np.empty(n)
    _nearest2(orc.X, M, near, sec, dn, ds)
    is_med = np.zeros(n, np.bool_)
    is_med[M] = True
    rem = np.bincount(near, weights=ds - dn, minlength=k).astype(np.float64)
    st = np.array([0, 0, n * k, 0], dtype=np.int64)
    passes = 0
    while passes < max_passes and not st[3]:
        for start in range(0, n, block):
            # all rows of the block, including current medoids: a medoid swapped out
            # earlier in this block becomes a candidate again later in the same block
            R = orc.rows(np.arange(start, min(start + block, n)))
            _fp_block(orc.X, R, start, M, near, sec, dn, ds, rem, is_med, st)
            if st[3]:
                break
        else:
            passes += 1
    evals = int(st[2]) + orc.evals
    return {"medoids": M, "loss": loss(X, M, centered=True), "swaps": int(st[1]), "evals": evals,
            "passes": passes, "converged": bool(st[3])}


def fasterclara(X, k, sample=5000, reps=5, seed=0, max_passes=100):
    """CLARA with FasterPAM on each sample (Schubert & Rousseeuw 2021, "FasterCLARA"):
    run FasterPAM on `reps` random samples of size `sample` (k-medoids++ seeding inside the
    sample) and keep the medoid set with the lowest loss on the full data set."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    X = X - X.mean(0)
    n = len(X)
    rng = np.random.default_rng(seed)
    best, best_M, evals = np.inf, None, 0
    for r in range(reps):
        S = np.sort(rng.choice(n, min(sample, n), replace=False))
        XS = X[S]
        rS = fasterpam_blocked(XS, kmedoidspp(XS, k, seed=int(rng.integers(1 << 31))), max_passes)
        evals += rS["evals"] + n * k
        M = S[rS["medoids"]]
        L = loss(X, M, centered=True)
        if L < best:
            best, best_M = L, M
    return {"medoids": best_M, "loss": best, "evals": evals}


MEDOID = {"cpv-warm": _medoid_cpv}
