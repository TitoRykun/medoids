# Ported verbatim from the medoids research code (medoid-opt/medoid/cpv.py). Do not edit by hand.
"""CPV: exact medoid by combining certified lower bounds (new method, this project).

Every candidate x keeps LB(x) = max of the following valid lower bounds on S(x):

  * Jensen:          S(x) >= n ||x - mean||                         (norm is convex)
  * VPS sandwich:    S(x) >= L_c,rho(x)  (see vps.py), O(d^2) per candidate
  * cutting planes:  S(x) >= S(p_i) + g_i . (x - p_i) for every exactly evaluated p_i,
                     g_i = sum_{k: p_k != p_i} (p_i - p_k)/||p_i - p_k||  is a subgradient
                     of the convex function S at p_i                    (O(d) per plane)
  * triangle:        S(x) >= |S(p_i) - n ||x - p_i|| |  (trimed's bound)  (O(1) per row)

A candidate is discarded once LB(x) > best * (1 + eps), where best = S of the best point
evaluated exactly so far. Survivors are evaluated exactly, smallest LB first, in batches;
each evaluation adds a plane and a triangle bound for everyone still alive. The returned
point is the exact medoid (up to floating-point ties within eps, see REPORT.md).

Flags allow ablations: use_vps / use_planes / use_triangle.
"""
import math

import numpy as np

from .kernels import Oracle
from .vps import bounds as vps_bounds

REL_EPS = 1e-9


def cpv(orc: Oracle, batch=8, use_vps="auto", use_planes="auto", use_triangle=True,
        vps_max_rounds=6, vps_top_k=32, gamma=1.1, anchor_ratio=1.5, verbose=False,
        init_idx=None):
    X = orc.X
    n, d = X.shape
    if not getattr(orc, "centered", False):
        raise ValueError("cpv requires a centred Oracle (Oracle(X, center=True), the default)")
    e0 = orc.evals
    Xc = X                                      # the oracle stores centred data
    sqn = orc.sq
    nrm = np.sqrt(sqn)
    LB = n * nrm                                   # Jensen
    alive = np.ones(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    st = {"best": np.inf, "m": -1, "rows": 0, "plane_rows": 0, "anchors": 0,
          "vps_rounds": 0, "planes_on": use_planes in (True, "auto"),
          "plane_prunes": 0, "steps": 0}

    def prune():
        alive[:] &= ~(LB > st["best"] * (1 + REL_EPS)) & ~done  # NaN-safe: NaN never prunes

    def exact_step(idx):
        idx = np.asarray([i for i in idx if not done[i]], dtype=np.int64)
        if len(idx) == 0:
            return
        D = orc.rows(idx)
        Si = D.sum(1)
        st["rows"] += len(idx)
        st["steps"] += 1
        done[idx] = True
        LB[idx] = Si
        k = int(np.argmin(Si))
        if Si[k] < st["best"]:
            st["best"], st["m"] = float(Si[k]), int(idx[k])
        alive[idx] = False
        live = np.nonzero(alive)[0]
        if len(live) == 0:
            return
        if use_triangle:
            T = np.abs(Si[:, None] - n * D[:, live]).max(0)
            np.maximum(LB[live], T, out=T)
            LB[live] = T
        if st["planes_on"]:
            with np.errstate(divide="ignore"):
                Winv = np.where(D > 0, 1.0 / D, 0.0)
            G = Xc[idx] * Winv.sum(1)[:, None] - Winv @ Xc       # B x d subgradients
            off = Si - np.einsum("ij,ij->i", Xc[idx], G)          # S_i - g_i . p_i
            if len(live) > n // 4:
                P = (Xc @ G.T)[live]
            else:
                P = Xc[live] @ G.T
            P = (P + off[None, :]).max(1)
            st["plane_rows"] += len(idx)
            before = int((LB[live] <= st["best"] * (1 + REL_EPS)).sum())
            np.maximum(LB[live], P, out=P)
            LB[live] = P
            after = int((LB[live] <= st["best"] * (1 + REL_EPS)).sum())
            st["plane_prunes"] += before - after
            # 'auto': give up on planes if they have pruned nothing after a few steps
            if use_planes == "auto" and st["steps"] >= 4 and st["plane_prunes"] == 0:
                st["planes_on"] = False
        prune()

    # incumbent: the points nearest the mean (plus optional warm-start candidates,
    # e.g. the previous medoid of a cluster in k-medoids)
    first = np.argpartition(nrm, min(batch, n - 1))[:batch]
    if init_idx is not None:
        first = np.unique(np.concatenate([np.asarray(init_idx, dtype=np.int64), first]))
    exact_step(first)
    prune()

    # ---- VPS rounds (high-dimensional regime)
    do_vps = (use_vps is True) or (use_vps == "auto" and d >= 4)
    if do_vps:
        c = np.zeros(d)
        Uall = np.full(n, np.inf)
        prev = alive.sum()
        for r in range(vps_max_rounds):
            cand = np.nonzero(alive)[0]
            if len(cand) <= max(batch, d):
                break
            L, U, na = vps_bounds(Xc, cand, c, gamma=gamma, anchor_ratio=anchor_ratio, sqn=sqn)
            st["anchors"] += na
            st["vps_rounds"] += 1
            LB[cand] = np.maximum(LB[cand], L)
            exact_step(cand[np.argsort(U)[:batch]])
            prune()
            cur = alive.sum()
            # stop when another round would cost more than finishing exactly, or stalls
            if cur <= d * max(na, 1) or cur > 0.7 * prev:
                break
            prev = cur
            Uall[cand] = U
            live = np.nonzero(alive)[0]
            top = live[np.argsort(Uall[live])[:vps_top_k]]
            c = Xc[top].mean(0)

    # ---- best-first exact evaluation with planes + triangle bounds.
    # Batch size doubles whenever a step removes fewer candidates (via bounds) than it
    # evaluated, so on "flat" inputs where nothing prunes the loop degrades gracefully
    # to blocked brute force (GEMM-sized batches) instead of many tiny steps.
    from .kernels import BLOCK_ELEMS
    cap = max(batch, min(128, BLOCK_ELEMS // max(n, 1)))
    bsz = batch
    while alive.any():
        live = np.nonzero(alive)[0]
        if bsz >= cap and len(live) > cap:
            # pruning has failed repeatedly: finish with blocked brute force on the rest
            Sl = orc.sums(live)
            st["rows"] += len(live)
            done[live] = True
            alive[live] = False
            k = int(np.argmin(Sl))
            if Sl[k] < st["best"]:
                st["best"], st["m"] = float(Sl[k]), int(live[k])
            st["fallback"] = True
            break
        k = min(bsz, len(live))
        nxt = live[np.argpartition(LB[live], k - 1)[:k]] if k < len(live) else live
        before = len(live)
        exact_step(nxt)
        pruned = before - k - int(alive.sum())
        if pruned < k:
            bsz = min(cap, 2 * bsz)

    return {"medoid": st["m"], "S_min": st["best"], "evals": orc.evals - e0,
            "exact_rows": st["rows"], "vps_rounds": st["vps_rounds"], "anchors": st["anchors"],
            "plane_rows": st["plane_rows"],
            "moment_flops": st["anchors"] * 2.0 * n * d * d}
