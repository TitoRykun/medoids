# Ported verbatim from the medoids research code (medoid-opt/medoid/kernels.py). Do not edit by hand.
"""Distance kernels with evaluation counting.

Two code paths:
  * small d (d <= SMALL_D): direct Numba loops, sqrt(sum (x_k - y_k)^2), exact to rounding;
  * large d: BLAS GEMM via ||x||^2 + ||y||^2 - 2 x.y, fused clamp/sqrt/sum in Numba.

Every method in this project goes through `Oracle`, so the counter `oracle.evals`
is the number of point-pair Euclidean distances evaluated.
"""
import numba as nb
import numpy as np

SMALL_D = 16
BLOCK_ELEMS = 1 << 24  # ~16M doubles (128 MB) per temporary block


@nb.njit(parallel=True, fastmath=False, cache=True)
def _rows_small(X, idx, out):
    n, d = X.shape
    for a in nb.prange(len(idx)):
        i = idx[a]
        for j in range(n):
            s = 0.0
            for k in range(d):
                t = X[i, k] - X[j, k]
                s += t * t
            out[a, j] = np.sqrt(s)


@nb.njit(parallel=True, fastmath=False, cache=True)
def _sums_small(X, idx, out):
    n, d = X.shape
    for a in nb.prange(len(idx)):
        i = idx[a]
        acc = 0.0
        for j in range(n):
            s = 0.0
            for k in range(d):
                t = X[i, k] - X[j, k]
                s += t * t
            acc += np.sqrt(s)
        out[a] = acc




# The GEMM expansion ||x||^2 + ||y||^2 - 2 x.y has absolute error ~ u (||x||^2 + ||y||^2),
# which is large *relative* to ||x - y||^2 for near-duplicate pairs far from the origin.
# Entries with d^2 < NEAR * (||x||^2 + ||y||^2) are recomputed directly from coordinates,
# which bounds the relative error of every returned distance by ~ 1e-12.
NEAR = 1e-3


@nb.njit(inline="always")
def _direct(Xr, a, Xc, j):
    s = 0.0
    for k in range(Xr.shape[1]):
        t = Xr[a, k] - Xc[j, k]
        s += t * t
    return np.sqrt(s)


@nb.njit(parallel=True, fastmath=False, cache=True)
def _finish_rows(G, sqr, sqc, out, Xr, Xc):
    b, m = G.shape
    for a in nb.prange(b):
        for j in range(m):
            v = sqr[a] + sqc[j] - 2.0 * G[a, j]
            if v < NEAR * (sqr[a] + sqc[j]):
                out[a, j] = _direct(Xr, a, Xc, j)
            else:
                out[a, j] = np.sqrt(v)


@nb.njit(parallel=True, fastmath=False, cache=True)
def _finish_sums(G, sqr, sqc, out, Xr, Xc):
    b, m = G.shape
    for a in nb.prange(b):
        acc = 0.0
        for j in range(m):
            v = sqr[a] + sqc[j] - 2.0 * G[a, j]
            if v < NEAR * (sqr[a] + sqc[j]):
                acc += _direct(Xr, a, Xc, j)
            else:
                acc += np.sqrt(v)
        out[a] += acc


class Oracle:
    """Euclidean distance oracle over a fixed point set X (n x d), counting evaluations."""

    def __init__(self, X, center=True):
        # Centring is distance-preserving; it improves the accuracy of the GEMM
        # expansion ||x||^2 + ||y||^2 - 2 x.y and lets bound code use X directly.
        X = np.asarray(X, dtype=np.float64)
        self.X = np.ascontiguousarray(X - X.mean(0) if center else X)
        self.centered = bool(center)
        self.n, self.d = self.X.shape
        self.sq = np.einsum("ij,ij->i", self.X, self.X)
        self.small = self.d <= SMALL_D
        self.evals = 0

    # ---- full rows -------------------------------------------------------
    def rows(self, idx):
        """Distance matrix between points idx and all n points, shape (len(idx), n)."""
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        out = np.empty((len(idx), self.n))
        self.evals += len(idx) * self.n
        if self.small:
            _rows_small(self.X, idx, out)
        else:
            G = self.X[idx] @ self.X.T
            _finish_rows(G, self.sq[idx], self.sq, out, self.X[idx], self.X)
        return out


    # ---- row sums S(i) = sum_j ||x_i - x_j|| ------------------------------
    def sums(self, idx):
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        out = np.zeros(len(idx))
        if len(idx) == 0:
            return out
        self.evals += len(idx) * self.n
        if self.small:
            _sums_small(self.X, idx, out)
            return out
        b = max(1, min(len(idx), BLOCK_ELEMS // self.n))
        cn = self.n if b * self.n <= BLOCK_ELEMS else max(1, BLOCK_ELEMS // b)
        for s in range(0, len(idx), b):
            ii = idx[s:s + b]
            Xi, sqi = self.X[ii], self.sq[ii]
            acc = np.zeros(len(ii))
            for c in range(0, self.n, cn):
                G = Xi @ self.X[c:c + cn].T
                _finish_sums(G, sqi, self.sq[c:c + cn], acc, Xi, self.X[c:c + cn])
            out[s:s + b] = acc
        return out

    # ---- arbitrary pairs / cross blocks ----------------------------------







