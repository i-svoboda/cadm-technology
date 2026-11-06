"""Arrays, priorities and simulated-group summaries for the end-to-end experiment.

A questionnaire k gives
  yc[k]     6 log-judgments of the criteria L, D, F, C (pair (a, b) with a < b alphabetically, value log a_ab,
            as tacs_vendor/elicit.to_log),
  ys[k]     3 x 10 log-judgments of the five variants with respect to L, F and C (same orientation),
  r[k]      3 x 5 ratings of the five variants on L, F and C (anchored 1-9 scales).
D is not asked; every variant has the D score of the chapter 2 rule (case file).

Ratings mode (chapter 2, equation (2.4)):  g_i = sum_c w_c r_ic / 9,   r_iD = D_i.
Relative mode (AHP distributive mode):     g_i = sum_c w_c pi_ic,      pi_ic = eigenvector of the 5 x 5 matrix of
                                           criterion c, pi_iD = D_i / sum_j D_j.
w is the principal eigenvector of the 4 x 4 criteria matrix (equation (1.1)).

A group is aggregated by AIJ (geometric mean of judgments = arithmetic mean of log-judgments, equation (3.1)); ratings
are aggregated by their geometric mean (primary, consistent with the ratio reading of (2.4)) or arithmetic mean
(sensitivity), as fixed in PROTOCOL.md before the batch.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE_FILE = ROOT / "case" / "endtoend_case.json"
CRITERIA = ["L", "D", "F", "C"]           # matrix order of the criteria
RATED = ["L", "F", "C"]                   # criteria asked for the variants
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
TIE = 1e-12


def load_case(path: Path = CASE_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Layout:
    """Fixed analysis order of variants, criteria and pairs."""

    def __init__(self, case: dict):
        self.variants = list(case["variant_order_for_analysis"])
        vd = {v["id"]: v for v in case["variants"]}
        self.D = np.array([vd[v]["D"] for v in self.variants], dtype=float)
        self.steps = np.array([vd[v]["steps"] for v in self.variants])
        self.crit_pairs = list(itertools.combinations(sorted(CRITERIA), 2))          # 6
        self.var_pairs = list(itertools.combinations(sorted(self.variants), 2))      # 10
        self.ci = {c: i for i, c in enumerate(CRITERIA)}
        self.vi = {v: i for i, v in enumerate(self.variants)}

    # ------------------------------------------------------------------ one questionnaire
    def arrays(self, items: list[dict], answer: dict) -> dict:
        """Answer -> yc (6), ys (3, 10), r (3, 5), plus counts of normalized soft inconsistencies."""
        got = {j["pair_id"]: j for j in answer["judgments"]}
        rat = {x["item_id"]: x["rating"] for x in answer["ratings"]}
        yc = np.full(len(self.crit_pairs), np.nan)
        ys = np.full((len(RATED), len(self.var_pairs)), np.nan)
        r = np.full((len(RATED), len(self.variants)), np.nan)
        soft = {"equal_with_intensity_above_1": 0, "intensity_1_not_equal": 0}
        for it in items:
            if "pair_id" in it:
                j = got[it["pair_id"]]
                if j["preferred"] == "equal" and j["intensity"] != 1:
                    soft["equal_with_intensity_above_1"] += 1
                if j["preferred"] != "equal" and j["intensity"] == 1:
                    soft["intensity_1_not_equal"] += 1
                v = 0.0 if j["preferred"] == "equal" or j["intensity"] == 1 else math.log(j["intensity"])
                if j["preferred"] == "second":
                    v = -v
                a, b = it["first"], it["second"]
                if a > b:
                    a, b, v = b, a, -v
                if it["block"] == "criteria":
                    yc[self.crit_pairs.index((a, b))] = v
                else:
                    ys[RATED.index(it["criterion"]), self.var_pairs.index((a, b))] = v
            else:
                r[RATED.index(it["criterion"]), self.vi[it["variant"]]] = rat[it["item_id"]]
        assert not (np.isnan(yc).any() or np.isnan(ys).any() or np.isnan(r).any()), "incomplete answer"
        return {"yc": yc, "ys": ys, "r": r, "soft": soft}

    # ------------------------------------------------------------------ matrices (batched)
    def crit_matrices(self, yc: np.ndarray) -> np.ndarray:
        """(..., 6) -> (..., 4, 4) in CRITERIA order."""
        A = np.ones(yc.shape[:-1] + (4, 4))
        for k, (a, b) in enumerate(self.crit_pairs):
            i, j = self.ci[a], self.ci[b]
            A[..., i, j] = np.exp(yc[..., k])
            A[..., j, i] = np.exp(-yc[..., k])
        return A

    def var_matrices(self, ys: np.ndarray) -> np.ndarray:
        """(..., 3, 10) -> (..., 3, 5, 5) in variant analysis order."""
        n = len(self.variants)
        A = np.ones(ys.shape[:-1] + (n, n))
        for k, (a, b) in enumerate(self.var_pairs):
            i, j = self.vi[a], self.vi[b]
            A[..., i, j] = np.exp(ys[..., k])
            A[..., j, i] = np.exp(-ys[..., k])
        return A

    def weights(self, yc: np.ndarray) -> np.ndarray:
        return eigvec(self.crit_matrices(yc))

    def local(self, ys: np.ndarray) -> np.ndarray:
        return eigvec(self.var_matrices(ys))

    def scores_ratings(self, w: np.ndarray, r: np.ndarray) -> np.ndarray:
        """w (..., 4), r (..., 3, 5) -> g (..., 5), equation (2.4)."""
        g = w[..., [self.ci["D"]]] * self.D / 9.0
        for k, c in enumerate(RATED):
            g = g + w[..., [self.ci[c]]] * r[..., k, :] / 9.0
        return g

    def scores_relative(self, w: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """w (..., 4), pi (..., 3, 5) local priorities -> g (..., 5), distributive AHP."""
        g = w[..., [self.ci["D"]]] * self.D / self.D.sum()
        for k, c in enumerate(RATED):
            g = g + w[..., [self.ci[c]]] * pi[..., k, :]
        return g


def eigvec(A: np.ndarray) -> np.ndarray:
    """Principal eigenvector of every matrix in A (..., n, n), normalized to sum 1 (as smaa_ahp.eigvec)."""
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real, axis=-1)
    v = np.take_along_axis(vecs, k[..., None, None], axis=-1)[..., 0]
    w = np.abs(v.real)
    return w / w.sum(axis=-1, keepdims=True)


def lambda_max(A: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(A).real.max(axis=-1)


def cr(A: np.ndarray) -> np.ndarray:
    n = A.shape[-1]
    return ((lambda_max(A) - n) / (n - 1)) / RI[n]


def agg_ratings(r: np.ndarray, axis: int, how: str = "geo") -> np.ndarray:
    if how == "geo":
        return np.exp(np.log(r).mean(axis=axis))
    if how == "arith":
        return r.mean(axis=axis)
    raise ValueError(how)


# ---------------------------------------------------------------------- simulated-group summaries
def ranks(g: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """0-based ranks (0 = first) per row; exact ties are broken at random (seeded) and counted in tie_share."""
    if rng is None:
        order = np.argsort(-g, axis=-1, kind="stable")
    else:
        jitter = rng.random(g.shape)
        order = np.lexsort((jitter, -np.round(g, 12)), axis=-1)
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.broadcast_to(np.arange(g.shape[-1]), g.shape).copy(), axis=-1)
    return r


def summarise(g: np.ndarray, rng: np.random.Generator) -> dict:
    """Rank probabilities b[i, r] (4.3) and pair probabilities p[i, j] (4.4) over the draws g (D, n).
    A tie counts as half a win for each side."""
    D, n = g.shape
    rk = ranks(g, rng)
    b = np.zeros((n, n))
    for i in range(n):
        b[i] = np.bincount(rk[:, i], minlength=n) / D
    diff = g[:, :, None] - g[:, None, :]
    p = (diff > TIE).mean(axis=0) + 0.5 * (np.abs(diff) <= TIE).mean(axis=0)
    np.fill_diagonal(p, 0.5)
    iu, ju = np.triu_indices(n, 1)
    ties = float((np.abs(diff[:, iu, ju]) <= TIE).any(axis=1).mean())
    return {"b": b, "p": p, "tie_share": ties, "ranks": rk}


def tau_b_to_reference(rk: np.ndarray, ref_rank: np.ndarray) -> np.ndarray:
    """Kendall tau-b between each row's ranking (0 = first) and a strict reference ranking (n,), vectorized."""
    n = rk.shape[-1]
    iu, ju = np.triu_indices(n, 1)
    s_ref = np.sign(ref_rank[iu] - ref_rank[ju])
    s = np.sign(rk[..., iu] - rk[..., ju])
    n0 = len(iu)
    n2 = (s == 0).sum(axis=-1)
    return (s * s_ref).sum(axis=-1) / np.sqrt(n0 * (n0 - n2))


def decision(b: np.ndarray, p: np.ndarray, labels: list[str], tau: float) -> dict:
    """Rule (4.8): pair accepted iff max(p, 1 - p) >= tau; rank r stable iff max_i b_i^r >= tau."""
    n = len(labels)
    acc, unstable = [], []
    for i in range(n):
        for j in range(i + 1, n):
            q = max(p[i, j], 1 - p[i, j])
            hi, lo = (labels[i], labels[j]) if p[i, j] >= 0.5 else (labels[j], labels[i])
            row = {"pair": f"{labels[i]}-{labels[j]}", "higher": hi, "lower": lo, "p": float(q)}
            (acc if q >= tau else unstable).append(row)
    unstable.sort(key=lambda x: abs(x["p"] - 0.5))
    stable_ranks = [{"rank": r + 1, "variant": labels[int(np.argmax(b[:, r]))], "b": float(b[:, r].max())}
                    for r in range(n) if b[:, r].max() >= tau]
    return {"tau": tau, "accepted_pairs": acc, "unstable_pairs": unstable, "stable_ranks": stable_ranks,
            "n_pairs": n * (n - 1) // 2}


def questionnaire_summary(layout: Layout, arr: dict) -> dict:
    """Weights, consistency and both scores of one questionnaire (written to result.json for monitoring)."""
    w = layout.weights(arr["yc"])
    pi = layout.local(arr["ys"])
    g_r = layout.scores_ratings(w, arr["r"])
    g_p = layout.scores_relative(w, pi)
    names = layout.variants
    return {"weights": dict(zip(CRITERIA, np.round(w, 4).tolist())),
            "CR_criteria": round(float(cr(layout.crit_matrices(arr["yc"]))), 4),
            "CR_variants": dict(zip(RATED, np.round(cr(layout.var_matrices(arr["ys"])), 4).tolist())),
            "ratings": {c: dict(zip(names, arr["r"][k].astype(int).tolist())) for k, c in enumerate(RATED)},
            "scores_ratings_mode": dict(zip(names, np.round(g_r, 4).tolist())),
            "scores_relative_mode": dict(zip(names, np.round(g_p, 4).tolist())),
            "order_ratings_mode": [names[i] for i in np.argsort(-g_r, kind="stable")],
            "order_relative_mode": [names[i] for i in np.argsort(-g_p, kind="stable")],
            "soft": arr["soft"]}
