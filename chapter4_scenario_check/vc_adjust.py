"""Variance components of model (4.1) with a cluster bootstrap over personas and wordings,
unadjusted and with the mean-square adjustment for resampling a few levels with replacement.

Why the adjustment. When P personas (and W wordings) are drawn with replacement and every
drawn level keeps its whole block of answers, duplicated levels shrink the spread between levels.
For one item of the balanced P x W x R design, with cells kept whole, the bootstrap expectations
of the ANOVA mean squares are (derivation in RESULT.md, Section 2.1):

  personas only (B-P):            E*MS_p = f_P MS_p          E*MS_w = MS_w + f_P MS_pw
                                  E*MS_pw = f_P MS_pw         E*MS_e = MS_e
  personas and wordings (B-PW):   E*MS_p = f_P MS_p + g MS_pw   E*MS_w = f_W MS_w + g MS_pw
                                  E*MS_pw = g MS_pw              E*MS_e = MS_e

with f_P = (P-1)/P, f_W = (W-1)/W and g = f_P f_W. The adjusted mean squares below invert these
relations, so that their bootstrap expectation equals the observed mean squares. This is the same
idea as the adjusted bootstrap estimates for variance components in generalizability theory.
Components are then estimated from the (adjusted) mean squares exactly as in the vendor code:
per item, truncated at zero, averaged over the items of a block, and turned into shares.
"""
from __future__ import annotations

import numpy as np

COMPS = ("persona", "wording", "persona_x_wording", "run")


def mean_squares(Yc: np.ndarray) -> tuple:
    """Yc: (P, W, R, K) -> mean squares per item (each length K), as smaa_ahp.variance_components."""
    p, w, r, _ = Yc.shape
    gm = Yc.mean(axis=(0, 1, 2))
    mp = Yc.mean(axis=(1, 2))
    mw = Yc.mean(axis=(0, 2))
    mpw = Yc.mean(axis=2)
    ms_p = w * r * ((mp - gm) ** 2).sum(axis=0) / (p - 1)
    ms_w = p * r * ((mw - gm) ** 2).sum(axis=0) / (w - 1)
    ms_pw = r * ((mpw - mp[:, None] - mw[None] + gm) ** 2).sum(axis=(0, 1)) / ((p - 1) * (w - 1))
    ms_e = ((Yc - mpw[:, :, None]) ** 2).sum(axis=(0, 1, 2)) / (p * w * (r - 1))
    return ms_p, ms_w, ms_pw, ms_e


def adjust(ms: tuple, P: int, W: int, scheme: str) -> tuple:
    ms_p, ms_w, ms_pw, ms_e = ms
    fP, fW = (P - 1) / P, (W - 1) / W
    if scheme == "B-P":
        return ms_p / fP, ms_w - ms_pw, ms_pw / fP, ms_e
    if scheme == "B-PW":
        g = fP * fW
        return (ms_p - ms_pw) / fP, (ms_w - ms_pw) / fW, ms_pw / g, ms_e
    raise ValueError(scheme)


def components(ms: tuple, P: int, W: int, R: int, truncate: bool = True) -> np.ndarray:
    ms_p, ms_w, ms_pw, ms_e = ms
    comp = np.stack([(ms_p - ms_pw) / (W * R), (ms_w - ms_pw) / (P * R), (ms_pw - ms_e) / R, ms_e])   # 4 x K
    return np.maximum(comp, 0) if truncate else comp


def block_shares(comp: np.ndarray, blocks: dict) -> dict:
    out = {}
    for b, cols in blocks.items():
        s = comp[:, cols].mean(axis=1)
        out[b] = s / s.sum()
    return out


def n_eff(sh: np.ndarray, N: int = 63, m: int = 9, L: int = 3, R: int = 3) -> float:
    """Equation (4.10): N / (1 + (m-1) s_P + (N/L - 1) s_W + (R-1) s_PW)."""
    return N / (1 + (m - 1) * sh[0] + (N / L - 1) * sh[1] + (R - 1) * sh[2])


def bootstrap(Yc: np.ndarray, blocks: dict, B: int, seed: int, scheme: str) -> dict:
    """Cluster bootstrap with cells kept whole. Returns shares and n_eff per replicate, unadjusted
    and adjusted, plus the mean of the untruncated adjusted components (bias check)."""
    P, W, R, K = Yc.shape
    # n_eff with the design of the data (N = P W R, m = W R, L = W), as the point estimate in analyse.variance; the
    # copy from overhaul_2026-10 used the defaults N 63, R 3 (protocol review of 26.10.2026, optional change 6)
    ne_kw = {"N": P * W * R, "m": W * R, "L": W, "R": R}
    rng = np.random.default_rng(seed)
    out = {v: {b: {"shares": np.zeros((B, 4)), "n_eff": np.zeros(B)} for b in blocks} for v in ("raw", "adj")}
    raw_sum = {b: np.zeros(4) for b in blocks}
    adj_sum = {b: np.zeros(4) for b in blocks}
    for i in range(B):
        ps = rng.integers(0, P, P)
        ws = rng.integers(0, W, W) if scheme == "B-PW" else np.arange(W)
        ms = mean_squares(Yc[ps][:, ws])
        ms_a = adjust(ms, P, W, scheme)
        for v, m in (("raw", ms), ("adj", ms_a)):
            sh = block_shares(components(m, P, W, R), blocks)
            for b in blocks:
                out[v][b]["shares"][i] = sh[b]
                out[v][b]["n_eff"][i] = n_eff(sh[b], **ne_kw)
        cu_raw = components(ms, P, W, R, truncate=False)
        cu_adj = components(ms_a, P, W, R, truncate=False)
        for b, cols in blocks.items():
            raw_sum[b] += cu_raw[:, cols].mean(axis=1)
            adj_sum[b] += cu_adj[:, cols].mean(axis=1)
    point_u = components(mean_squares(Yc), P, W, R, truncate=False)
    out["bias_check"] = {b: {"point_untruncated": point_u[:, cols].mean(axis=1).tolist(),
                             "boot_mean_untruncated_raw": (raw_sum[b] / B).tolist(),
                             "boot_mean_untruncated_adj": (adj_sum[b] / B).tolist()} for b, cols in blocks.items()}
    return out
