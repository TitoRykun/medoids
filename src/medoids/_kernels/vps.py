# Ported verbatim from the medoids research code (medoid-opt/medoid/vps.py). Do not edit by hand.
"""VPS: Virtual-Pivot Sandwich pruning for the exact medoid (new method, this project).

Core inequality (per term, for any tau > 0 and any Y = r^2 >= 0, r >= 0):

    (Y + tau^2)/(2 tau) - (Y - tau^2)^2/(2 tau^3)   <=   r   <=   (Y + tau^2)/(2 tau)

Summing over reference points p_k with weights chosen INDEPENDENTLY of the candidate x,
both sides are polynomials in x of degree <= 4 whose coefficients are a handful of
weighted moments of the data (scalars, d-vectors, one d x d matrix). Hence, after an
O(n d^2) precomputation, every candidate gets a certified interval [L(x), U(x)] that
contains S(x) at O(d^2) cost instead of O(n d).

Choice of tau ("virtual pivot"): tau_k^2 = rho^2 + ||p_k - c||^2, i.e. the distance from
p_k to a virtual point at distance rho from a centre c in a direction orthogonal to all
data. In high dimension, ||x - p_k||^2 ~= ||x - c||^2 + ||p_k - c||^2 for candidates x
near c, so with rho ~= ||x - c|| the interval is very tight.

Then with z = x - c, s = ||z||^2, t = s - rho^2, q_k = p_k - c, w_k = 1/(2 tau_k),
v_k = 1/(2 tau_k^3):

    U(x) = s * sum w - 2 z . sum(w q) + sum w (2 b_k^2 + rho^2)
    Q(x) = t^2 sum v - 4 t z . sum(v q) + 4 z^T M z,   M = sum v q q^T
    L(x) = U(x) - Q(x)

Grid trick: v_k is decreasing in rho, so M(rho) <= M(rho_A) (PSD order) for rho >= rho_A.
Replacing M(rho) by M(rho_A) in Q only increases Q, so L stays a valid lower bound.
We therefore compute a few "anchor" matrices (O(n d^2) each) and a fine grid of rho
values whose scalar/vector moments cost O(n d) each; each candidate uses the grid rho
closest to its own ||x - c||.

Algorithm:
  round r: centre c_r (round 0: data mean), bounds for all live candidates, exact S for
  the few best-U candidates (incumbent), prune L(x) > best; recentre on the best
  survivors; repeat while it pays; finally evaluate survivors exactly, best-L first,
  pruning as the incumbent improves. Exact (deterministic) by construction.
"""
import math

import numpy as np

from .kernels import Oracle

REL_EPS = 1e-9
CHUNK = 1 << 14


# Rounding allowance: L is reduced by ALLOW * (sum of absolute values of the terms that
# are added/subtracted), so cancellation in the expanded quartic can never make L exceed
# S. ALLOW ~ 4500 unit roundoffs covers the summation error of the O(n)-term moments.
ALLOW = 1e-12
# The VPS moments overflow/underflow for extreme data scales; bounds are disabled there.
SCALE_OK = (1e-50, 1e50)
# rho is floored at RHO_FLOOR * rms(||p_k - c||) so that v_k = 1/(2 tau_k^3) stays bounded
# even when most points sit at the centre (the bounds hold for any rho > 0).
RHO_FLOOR = 0.05


def _weighted_gram(Xc, v, c):
    """M = sum_k v_k (p_k - c)(p_k - c)^T computed directly in row chunks."""
    d = Xc.shape[1]
    P = np.zeros((d, d))
    # small row chunks: large temporaries are dominated by page-fault cost on this VM
    for s in range(0, len(Xc), 4096):
        B = (Xc[s:s + 4096] - c) * np.sqrt(v[s:s + 4096])[:, None]
        P += B.T @ B
    return P


def bounds(Xc, cand, c, gamma=1.1, anchor_ratio=1.5, sqn=None, need_L=True, return_rho=False):
    """Certified [L, U] for S(x), x in cand, using centre c.

    Xc: data (n x d), ideally centred. sqn is accepted for API compatibility (unused:
    all distances to c are computed directly to avoid cancellation).
    Returns L, U and the number of anchor matrices built (each O(n d^2))."""
    n, d = Xc.shape
    c = np.asarray(c, dtype=np.float64)
    b2 = np.empty(n)                                           # ||p_k - c||^2, direct
    for t0 in range(0, n, 4096):
        D = Xc[t0:t0 + 4096] - c
        b2[t0:t0 + 4096] = np.einsum("ij,ij->i", D, D)
    s = b2[cand]
    rms = math.sqrt(float(b2.mean()))
    if not (SCALE_OK[0] < rms < SCALE_OK[1]):
        L = np.full(len(cand), -np.inf)
        U = np.full(len(cand), np.inf)
        return (L, U, 0, np.full(len(cand), np.nan)) if return_rho else (L, U, 0)
    # rho grid (geometric) over the candidates' distances to c
    floor = RHO_FLOOR * rms
    a_eff = np.maximum(np.sqrt(s), floor)
    lo = a_eff.min()
    j = np.rint(np.log(a_eff / lo) / math.log(gamma)).astype(np.int64)
    grid_idx = np.unique(j)
    rhos = lo * gamma ** grid_idx.astype(float)
    G = len(rhos)
    # scalar and vector moments for every grid rho
    tau = np.sqrt(rhos[:, None] ** 2 + b2[None, :])          # G x n
    w = 0.5 / tau
    v = w / (tau * tau)
    W0 = w.sum(1)
    Wc = w @ (2 * b2) + W0 * rhos ** 2
    V0 = v.sum(1)
    WV = np.vstack([w, v])
    mom = np.zeros((2 * G, d))
    for t0 in range(0, n, 4096):
        mom += WV[:, t0:t0 + 4096] @ (Xc[t0:t0 + 4096] - c)
    Bw, Av = mom[:G], mom[G:]                                  # sum w q, sum v q
    del tau, w, v, WV
    # anchors: coarse grid; each fine rho uses the smallest grid rho of its anchor group
    step = max(1, int(round(math.log(anchor_ratio) / math.log(gamma))))
    anchor_of = (grid_idx - grid_idx.min()) // step
    gpos = np.searchsorted(grid_idx, j)                        # grid position per cand
    cand_anchor = anchor_of[gpos]
    if not need_L:
        U = np.empty(len(cand))
        for t0 in range(0, len(cand), 4096):
            sl = slice(t0, t0 + 4096)
            Z = Xc[cand[sl]] - c
            gp = gpos[sl]
            U[sl] = s[sl] * W0[gp] - 2 * np.einsum("ij,ij->i", Z, Bw[gp]) + Wc[gp]
        return (None, U, 0, rhos[gpos]) if return_rho else (None, U, 0)
    U = np.empty(len(cand))
    L = np.empty(len(cand))
    n_anchor = 0
    for A in np.unique(cand_anchor):
        rho_A = rhos[np.nonzero(anchor_of == A)[0][0]]
        M = _weighted_gram(Xc, 0.5 / (rho_A ** 2 + b2) ** 1.5, c)
        n_anchor += 1
        sel_all = np.nonzero(cand_anchor == A)[0]
        for t0 in range(0, len(sel_all), 4096):
            sel = sel_all[t0:t0 + 4096]
            Z = Xc[cand[sel]] - c
            gp = gpos[sel]
            ss = s[sel]
            zB = np.einsum("ij,ij->i", Z, Bw[gp])
            zA = np.einsum("ij,ij->i", Z, Av[gp])
            quad = np.einsum("ij,ij->i", Z @ M, Z)                  # z^T M z
            t = ss - rhos[gp] ** 2
            Uc = ss * W0[gp] - 2 * zB + Wc[gp]
            Q = t * t * V0[gp] - 4 * t * zA + 4 * quad
            err = ALLOW * (np.abs(ss * W0[gp]) + 2 * np.abs(zB) + np.abs(Wc[gp])
                           + t * t * V0[gp] + 4 * np.abs(t * zA) + 4 * np.abs(quad))
            Lc = Uc - np.maximum(Q, 0.0) - err
            # true Q >= 0; a computed Q below -err means the formula broke down numerically
            bad = ~np.isfinite(Lc) | (Q < -err)
            Lc[bad] = -np.inf
            U[sel] = Uc
            L[sel] = Lc
    if return_rho:
        return L, U, n_anchor, rhos[gpos]
    return L, U, n_anchor




