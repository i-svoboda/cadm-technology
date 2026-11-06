"""Decision model, answer arrays and all computations of the DSO controls experiment. Never calls a model.

Network: the chapter 3 ANP network for critical-infrastructure controls (draft/05_rozdil_3.md, 3.3.1-3.3.2; the code of
experiments/replication_2026-10/scripts/anp_model.py, design D1), with the threat cluster H replaced by the five scenario
variants of the distribution system operator (DSO) of chapter 2. Node order of the supermatrix as in chapter 3 (K, R, H, B):

  K  security controls (7)          MON SEG IDM IR SUP BAK PHY
  R  evaluation criteria (5)        E T U C P
  V  scenario variants (5)          PC DS RS MP EE      (in the place of the threat vectors of chapter 3)
  B  implementation constraints (3) BUD WRK LOCK

Elicited matrices (33; chapter 3 has the first 32, the cluster comparison CL|NET comes from chapter 4):
  R|GOAL      criteria with respect to the goal                         1 x 10 pairs
  K|R:c       controls with respect to each criterion                   5 x 21
  K|V:v       controls with respect to each scenario variant            5 x 21   (chapter 3: K|H:t)
  V|R:c       scenario variants with respect to each criterion          5 x 10   (chapter 3: H|R:c)
  B|R:c       constraints with respect to each criterion                5 x 3
  B|K:k       constraints with respect to each control                  7 x 3
  K|B:b       controls with respect to each constraint                  3 x 21
  K|K:MON     SEG against IDM as sources of synergy for MON             1 x 1
  CL|NET      the four clusters compared (group's own cluster weights)  1 x 6
  = 376 pairs; plus 35 coverage shares eta[c, v] and 5 expected annual losses EAL[v] (chapter 3, 3.3.6).

Fixed blocks (no judgment, as chapter 3): synergies SEG->IDM and BAK->IR (single source, weight 1); the criteria vector
R|GOAL fills the criteria rows of every control column (feedback) and of every criterion column (inner dependence).
Synergy reading of the method (3.3.2, "source"): in the column of the enhanced control Y the source X carries the weight.
Sensitivity ("target", the reading of ITS 14(1)): the links are reversed. Weighting (3.6)-(3.7): one cluster vector v for
every column, zeroed for empty blocks and renormalized. Limit (3.9): powers of the weighted supermatrix (L1 change below
1e-13), Cesaro average as the fallback. Cluster weights: the prior v of chapter 3 or the group's own weights (principal
eigenvector of the aggregated CL|NET matrix, as chapter 4).

Portfolio (3.11): f(X) = sum_v EAL_v (1 - prod_{c in X} (1 - eta_cv)), subject to sum_{c in X} k_c <= C_max; exact optimum
by enumeration of all 127 non-empty sets in the order of the chapter 3 code (size, then itertools order), ties to the
first; group inputs: arithmetic mean of the shares, geometric mean of the losses (floor 1e-6), as chapter 3.

Two computation paths. "exact": LAPACK eigenvectors and the power-iteration limit of chapter 3 (pooled group, diagnostics).
"fast": power-iteration eigenvectors and a direct linear solve for the stationary vector (simulated groups); the analysis
checks both paths on random groups and records the largest difference (as overhaul_2026-10/A2_bootstrap/ch4_core.py).
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE_FILE = ROOT / "case" / "dso_controls_case.json"
INPUTS_FILE = ROOT / "case" / "analysis_inputs.json"

K = ["MON", "SEG", "IDM", "IR", "SUP", "BAK", "PHY"]
R = ["E", "T", "U", "C", "P"]
V = ["PC", "DS", "RS", "MP", "EE"]
B = ["BUD", "WRK", "LOCK"]
CLUSTERS = {"K": K, "R": R, "V": V, "B": B}
CLUSTER_ORDER = ["K", "R", "V", "B"]            # supermatrix node order (chapter 3: K, R, H, B)
CL_ELEMENTS = ["K", "R", "V", "B"]              # elements of the cluster comparison CL|NET
V_PRIOR = {"K": 0.4466, "V": 0.2549, "R": 0.2506, "B": 0.0480}     # chapter 3, (3.6); H's weight goes to V
SYNERGY = [("SEG", "MON"), ("IDM", "MON"), ("SEG", "IDM"), ("BAK", "IR")]
COSTS = {"MON": 3.8, "SEG": 5.2, "IDM": 3.0, "IR": 1.4, "SUP": 2.0, "BAK": 1.6, "PHY": 0.8}   # million USD, chapter 3
BUDGET = 10.2
BLOCK_TYPES = ["R|GOAL", "K|R", "K|V", "V|R", "B|R", "B|K", "K|B", "K|K", "CL|NET"]
READINGS = ("source", "target", "source_nofixed", "none")
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
TIE = 1e-12
LOSS_FLOOR = 1e-6
N_PAIRS, N_COVERAGE, N_LOSS = 376, 35, 5


# ============================================================================ network layout
def matrix_specs() -> list[tuple[str, str, str | None]]:
    """(matrix key, row cluster, column node) in the order of anp_model.Case.matrix_specs, plus CL|NET."""
    specs = [("R|GOAL", "R", None)]
    for rows, colc in [("K", "R"), ("K", "V"), ("V", "R"), ("B", "R"), ("B", "K"), ("K", "B")]:
        for col in CLUSTERS[colc]:
            specs.append((f"{rows}|{colc}:{col}", rows, col))
    specs.append(("K|K:MON", "K", "MON"))
    specs.append(("CL|NET", "CL", None))
    return specs


def elements(key: str) -> list[str]:
    if key == "K|K:MON":
        return ["SEG", "IDM"]                       # anp_model.Case.matrix_elements
    if key == "CL|NET":
        return list(CL_ELEMENTS)
    return list(CLUSTERS[key.split("|")[0]])


def block_type(key: str) -> str:
    return key.split(":")[0]


class Layout:
    """Fixed analysis order of matrices, pairs, coverage items and losses (independent of the presentation order)."""

    def __init__(self) -> None:
        self.specs = matrix_specs()
        self.mkeys = [m for m, _, _ in self.specs]
        self.els = {m: elements(m) for m in self.mkeys}
        self.keys = [(m, a, b) for m in self.mkeys for a, b in itertools.combinations(self.els[m], 2)]
        assert len(self.mkeys) == 33 and len(self.keys) == N_PAIRS
        self.kidx = {k: i for i, k in enumerate(self.keys)}
        self.btype = [block_type(m) for m, _, _ in self.keys]
        self.blocks = {b: [i for i, t in enumerate(self.btype) if t == b] for b in BLOCK_TYPES}
        self.entries = {m: [(self.kidx[(m, a, b)], self.els[m].index(a), self.els[m].index(b))
                            for a, b in itertools.combinations(self.els[m], 2)] for m in self.mkeys}
        self.by_size: dict[int, list[str]] = {}
        for m in self.mkeys:
            self.by_size.setdefault(len(self.els[m]), []).append(m)
        self._index = {}
        for n, names in self.by_size.items():
            t_, i_, j_, k_ = [], [], [], []
            for t, m in enumerate(names):
                for k, i, j in self.entries[m]:
                    t_.append(t), i_.append(i), j_.append(j), k_.append(k)
            self._index[n] = tuple(np.array(x) for x in (t_, i_, j_, k_))
        self.cov_keys = [(c, v) for c in K for v in V]          # chapter 3 order (control-major)
        self.nodes = [(c, n) for c in CLUSTER_ORDER for n in CLUSTERS[c]]
        self.pos = {n: i for i, (_, n) in enumerate(self.nodes)}
        self.kpos = [self.pos[k] for k in K]
        self.vpos = [self.pos[v] for v in V]
        self.cluster_pos = {c: [self.pos[n] for n in CLUSTERS[c]] for c in CLUSTER_ORDER}
        self._columns = {r: self._column_plan(r) for r in READINGS}

    # ------------------------------------------------------------------ one answer -> arrays
    def arrays(self, items: list[dict], answer: dict) -> dict:
        """y (376) log-judgments in the analysis orientation (pair (a, b) in element order, y = log a_ab), eta (7, 5),
        eal (5), and counts of the soft notes. Orientation and the reading of 'equal' and of intensity 1 as in
        tacs_vendor/elicit.to_log."""
        got = {j["pair_id"]: j for j in answer["judgments"]}
        cov = {c["item_id"]: c["share"] for c in answer["coverage"]}
        los = {c["item_id"]: c["million_usd"] for c in answer["expected_loss"]}
        y = np.full(N_PAIRS, np.nan)
        eta = np.full((len(K), len(V)), np.nan)
        eal = np.full(len(V), np.nan)
        soft = {"equal_with_intensity_above_1": 0, "intensity_1_not_equal": 0}
        for it in items:
            if "pair_id" in it:
                j = got[it["pair_id"]]
                if j["preferred"] == "equal" and j["intensity"] != 1:
                    soft["equal_with_intensity_above_1"] += 1
                if j["preferred"] != "equal" and j["intensity"] == 1:
                    soft["intensity_1_not_equal"] += 1
                val = 0.0 if j["preferred"] == "equal" or j["intensity"] == 1 else math.log(j["intensity"])
                if j["preferred"] == "second":
                    val = -val
                m, a, b = it["matrix"], it["first"], it["second"]
                if (m, a, b) not in self.kidx:
                    a, b, val = b, a, -val
                y[self.kidx[(m, a, b)]] = val
            elif "coverage_id" in it:
                eta[K.index(it["control"]), V.index(it["variant"])] = float(cov[it["coverage_id"]])
            elif "loss_id" in it:
                eal[V.index(it["variant"])] = float(los[it["loss_id"]])
        assert not (np.isnan(y).any() or np.isnan(eta).any() or np.isnan(eal).any()), "incomplete answer"
        return {"y": y, "eta": eta, "eal": eal, "soft": soft}

    # ------------------------------------------------------------------ matrices and local vectors (batched)
    def matrices_of_size(self, Y: np.ndarray, n: int) -> np.ndarray:
        """Y (D, 376) -> (D, number of matrices of order n, n, n) in self.by_size[n] order."""
        t, i, j, k = self._index[n]
        A = np.ones((Y.shape[0], len(self.by_size[n]), n, n))
        E = np.exp(Y[:, k])
        A[:, t, i, j] = E
        A[:, t, j, i] = 1.0 / E
        return A

    def matrix(self, y: np.ndarray, key: str) -> np.ndarray:
        """One matrix (n, n) of one log-judgment vector y (376)."""
        n = len(self.els[key])
        A = np.ones((n, n))
        for k, i, j in self.entries[key]:
            A[i, j], A[j, i] = math.exp(y[k]), math.exp(-y[k])
        return A

    def local(self, Y: np.ndarray, fast: bool = False) -> dict[str, np.ndarray]:
        """Principal eigenvector of every matrix for every row of Y: key -> (D, n)."""
        out = {}
        for n, names in self.by_size.items():
            A = self.matrices_of_size(Y, n)
            w = eigvec_power(A) if fast else eigvec(A)
            for t, m in enumerate(names):
                out[m] = w[:, t, :]
        return out

    def own_weights(self, loc: dict) -> dict[str, np.ndarray]:
        w = loc["CL|NET"]
        return {c: w[:, i] for i, c in enumerate(CL_ELEMENTS)}

    @staticmethod
    def prior_weights(D: int, v: dict | None = None) -> dict[str, np.ndarray]:
        v = v or V_PRIOR
        return {c: np.full(D, float(v[c])) for c in CL_ELEMENTS}

    # ------------------------------------------------------------------ supermatrix
    def _column_plan(self, reading: str) -> list[list[tuple]]:
        """For every column node: list of (row cluster, kind, payload). kind 'mat' = elicited matrix key; 'goal' =
        R|GOAL; 'syn2' = K|K:MON over SEG, IDM; 'fixed' = equal weights over the listed sources. Readings: 'source'
        (the method), 'target' (ITS 14(1)), and two sensitivity cases of chapter 3 and A12: 'source_nofixed' (the
        method's reading without the fixed blocks SEG->IDM and BAK->IR) and 'none' (no synergy block)."""
        syn = {"source": SYNERGY, "target": [(t, s) for s, t in SYNERGY],
               "source_nofixed": [(s, t) for s, t in SYNERGY if t == "MON"], "none": []}[reading]
        plan = []
        for cj, nj in self.nodes:
            blocks = []
            for m, rows, col in self.specs:
                if m in ("R|GOAL", "K|K:MON", "CL|NET"):
                    continue
                if m.split("|")[1] == f"{cj}:{nj}":
                    blocks.append((rows, "mat", m))
            if cj == "K":
                blocks.append(("R", "goal", "R|GOAL"))                    # criteria feedback into every control
                src = [s for s, t in syn if t == nj]
                if sorted(src) == ["IDM", "SEG"]:
                    blocks.append(("K", "syn2", "K|K:MON"))
                elif src:
                    blocks.append(("K", "fixed", src))
            if cj == "R":
                blocks.append(("R", "goal", "R|GOAL"))                    # inner dependence of the criteria
            assert blocks, f"column {nj} has no active block"
            assert len({b[0] for b in blocks}) == len(blocks), f"two blocks of one cluster in column {nj}"
            plan.append(blocks)
        return plan

    def supermatrix(self, loc: dict, v: dict[str, np.ndarray], reading: str = "source",
                    variant_mult: dict | None = None) -> np.ndarray:
        """Weighted supermatrix (D, 20, 20) by (3.6)-(3.7). v: cluster -> (D,) weights. variant_mult multiplies the rows
        of the variants in the criterion columns (V|R blocks) and renormalizes them (chapter 3's threat scenarios)."""
        D = next(iter(loc.values())).shape[0]
        N = len(self.nodes)
        W = np.zeros((D, N, N))
        for j, blocks in enumerate(self._columns[reading]):
            total = np.zeros(D)
            for c, _, _ in blocks:
                total = total + v[c]
            for c, kind, payload in blocks:
                if kind in ("mat", "goal"):
                    vec, ids = loc[payload], self.els[payload]
                    if variant_mult and kind == "mat" and payload.startswith("V|R"):
                        mult = np.array([variant_mult.get(x, 1.0) for x in ids])
                        vec = vec * mult
                        vec = vec / vec.sum(axis=1, keepdims=True)
                elif kind == "syn2":
                    vec, ids = loc[payload], self.els[payload]
                else:
                    ids = payload
                    vec = np.full((D, len(ids)), 1.0 / len(ids))
                scale = v[c] / total
                rows = [self.pos[x] for x in ids]
                W[:, rows, j] = scale[:, None] * vec
        return W

    def priorities(self, loc: dict, v: dict, reading: str = "source", fast: bool = False,
                   variant_mult: dict | None = None) -> dict:
        """Control priorities g (D, 7), variant priorities (D, 5) and the limit vectors (D, 20)."""
        W = self.supermatrix(loc, v, reading, variant_mult)
        pi = limit_solve(W) if fast else limit_power(W)[0]
        g = pi[:, self.kpos] / pi[:, self.kpos].sum(axis=1, keepdims=True)
        gv = pi[:, self.vpos] / pi[:, self.vpos].sum(axis=1, keepdims=True)
        return {"g": g, "gv": gv, "pi": pi}


# ============================================================================ numerical helpers
def eigvec(A: np.ndarray) -> np.ndarray:
    """Principal eigenvector of every matrix in A (..., n, n), sum 1 (replication common.priorities_eigen)."""
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real, axis=-1)
    v = np.take_along_axis(vecs, k[..., None, None], axis=-1)[..., 0]
    w = np.abs(v.real)
    return w / w.sum(axis=-1, keepdims=True)


def eigvec_power(A: np.ndarray, tol: float = 1e-15, max_iter: int = 2000) -> np.ndarray:
    """Perron vector of positive matrices by batched power iteration (A2_bootstrap/ch4_core.eigvec_power)."""
    n = A.shape[-1]
    w = np.full(A.shape[:-1], 1.0 / n)
    for _ in range(max_iter):
        w2 = np.einsum("...ij,...j->...i", A, w)
        w2 /= w2.sum(axis=-1, keepdims=True)
        done = np.abs(w2 - w).max() < tol
        w = w2
        if done:
            break
    return w


def lambda_max(A: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(A).real.max(axis=-1)


def cr(A: np.ndarray) -> np.ndarray:
    n = A.shape[-1]
    return ((lambda_max(A) - n) / (n - 1)) / RI[n] if RI[n] else np.zeros(A.shape[:-2])


def koczkodaj(A: np.ndarray) -> float:
    """Worst triad of one matrix (replication common.koczkodaj)."""
    n = A.shape[0]
    worst = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = A[i, k], A[i, j], A[j, k]
                worst = max(worst, min(abs(1 - a / (b * c)), abs(1 - (b * c) / a)))
    return worst


def limit_power(W: np.ndarray, tol: float = 1e-13, max_iter: int = 200000) -> tuple[np.ndarray, np.ndarray]:
    """Batched anp_model.Case.limit: powers from the uniform vector until the L1 change is below tol, per group;
    Cesaro average of 5,000 further powers for a group that does not converge. Returns (pi, iterations; -1 = Cesaro)."""
    D, N, _ = W.shape
    x = np.full((D, N), 1.0 / N)
    out = np.full((D, N), np.nan)
    iters = np.full(D, -1)
    active = np.arange(D)
    for it in range(max_iter):
        if active.size == 0:
            break
        y = np.einsum("dij,dj->di", W[active], x[active])
        conv = np.abs(y - x[active]).sum(axis=1) < tol
        done = active[conv]
        out[done] = y[conv] / y[conv].sum(axis=1, keepdims=True)
        iters[done] = it
        x[active] = y
        active = active[~conv]
    if active.size:
        acc = np.zeros((active.size, N))
        xx = x[active]
        for _ in range(5000):
            xx = np.einsum("dij,dj->di", W[active], xx)
            acc += xx
        out[active] = acc / acc.sum(axis=1, keepdims=True)
    return out, iters


def limit_solve(W: np.ndarray) -> np.ndarray:
    """Stationary vector of each column-stochastic supermatrix by a direct solve (A2_bootstrap/ch4_core.limit_solve)."""
    D, n, _ = W.shape
    M = W - np.eye(n)[None]
    M[:, -1, :] = 1.0
    rhs = np.zeros((D, n, 1))
    rhs[:, -1, 0] = 1.0
    pi = np.linalg.solve(M, rhs)[..., 0]
    return pi / pi.sum(axis=1, keepdims=True)


def second_eigenvalue(W: np.ndarray) -> float:
    ev = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]
    return float(ev[1])


# ============================================================================ group aggregation
def aggregate(Yq: np.ndarray, Eq: np.ndarray, Lq: np.ndarray, idx: np.ndarray) -> tuple:
    """idx (D, m) member indices -> AIJ log-judgments (D, 376), mean shares (D, 7, 5), geometric-mean losses (D, 5)."""
    y = Yq[idx].mean(axis=1)
    eta = Eq[idx].mean(axis=1)
    eal = np.exp(np.log(np.maximum(Lq[idx], LOSS_FLOOR)).mean(axis=1))
    return y, eta, eal


# ============================================================================ portfolio
SUBSETS = [c for m in range(1, len(K) + 1) for c in itertools.combinations(range(len(K)), m)]   # 127, chapter 3 order
MASK = np.array([[i in S for i in range(len(K))] for S in SUBSETS])                          # 127 x 7
MASK_E = np.vstack([MASK, np.zeros((1, len(K)), dtype=bool)])                                # index -1 = empty set
COST_VEC = np.array([COSTS[c] for c in K])
SUB_COST = MASK @ COST_VEC


def set_name(i: int) -> str:
    return "{" + ", ".join(K[c] for c in SUBSETS[i]) + "}" if i >= 0 else "{}"


def set_index(controls: list[str] | tuple) -> int:
    t = tuple(sorted(K.index(c) for c in controls))
    return SUBSETS.index(t) if t else -1


def reductions(eta: np.ndarray, eal: np.ndarray) -> np.ndarray:
    """f(S) of (3.11) for every subset and group: eta (D, 7, 5), eal (D, 5) -> (D, 127). Product over controls in K
    order and sum over variants in V order, as the loops of anp_analysis.knapsack."""
    D = eta.shape[0]
    one_minus = 1.0 - eta
    prod = np.ones((D, len(SUBSETS), len(V)))
    for ci in range(len(K)):
        prod = np.where(MASK[None, :, ci:ci + 1], prod * one_minus[:, None, ci, :], prod)
    f = np.zeros((D, len(SUBSETS)))
    for ti in range(len(V)):
        f = f + eal[:, ti:ti + 1] * (1 - prod[:, :, ti])
    return f


def best_sets(f: np.ndarray, budget: float = BUDGET, require: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact optimum per group: the first subset in enumeration order with the largest f among the feasible ones;
    -1 (empty set) when no feasible subset has f > 0 (anp_analysis.knapsack: strict > from best_val = 0)."""
    feas = SUB_COST <= budget + 1e-9
    if require is not None:
        feas = feas & MASK[:, require]
    if not feas.any():
        return np.full(f.shape[0], -1), np.zeros(f.shape[0])
    fv = np.where(feas[None, :], f, -np.inf)
    i = np.argmax(fv, axis=1)
    val = fv[np.arange(f.shape[0]), i]
    i = np.where(val > 0, i, -1)
    return i, np.where(val > 0, val, 0.0)


def greedy(eta: np.ndarray, eal: np.ndarray, budget: float = BUDGET) -> list[str]:
    """Lazy greedy by gain per cost (anp_analysis.knapsack), one group: eta (7, 5), eal (5)."""
    def f(S):
        return sum(eal[t] * (1 - np.prod([1 - eta[K.index(c), t] for c in S])) for t in range(len(V)))
    S, spent = [], 0.0
    while True:
        best, bval = None, 0.0
        for c in K:
            if c in S or spent + COSTS[c] > budget + 1e-9:
                continue
            g = (f(S + [c]) - f(S)) / COSTS[c]
            if g > bval:
                best, bval = c, g
        if best is None:
            break
        S.append(best)
        spent += COSTS[best]
    return S


def portfolio_decision(sets: np.ndarray, ref: int, tau: float) -> dict:
    """Rule (5.1): the set X_B is accepted iff q(X_B) >= tau; when it is not, control c is included iff q_c >= tau,
    excluded iff q_c <= 1 - tau, and sent to human experts otherwise."""
    q = float(np.mean(sets == ref))
    qc = MASK_E[sets].mean(axis=0)
    in_ref = MASK_E[ref]
    rows = {}
    for i, c in enumerate(K):
        if q >= tau:
            d = "in the accepted set" if in_ref[i] else "outside the accepted set"
        else:
            d = "included" if qc[i] >= tau else "excluded" if qc[i] <= 1 - tau else "to human experts"
        rows[c] = {"q_c": round(float(qc[i]), 4), "in_X_B": bool(in_ref[i]), "decision": d}
    return {"tau": tau, "X_B": set_name(ref), "q_X_B": round(q, 4), "set_accepted": q >= tau, "controls": rows,
            "to_human_experts": [c for c in K if rows[c]["decision"] == "to human experts"]}


# ============================================================================ rank and pair probabilities
def ranks(g: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """0-based ranks per row; exact ties broken at random with the given generator (endtoend core.ranks)."""
    if rng is None:
        order = np.argsort(-g, axis=-1, kind="stable")
    else:
        jitter = rng.random(g.shape)
        order = np.lexsort((jitter, -np.round(g, 12)), axis=-1)
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.broadcast_to(np.arange(g.shape[-1]), g.shape).copy(), axis=-1)
    return r


def summarise(g: np.ndarray, rng: np.random.Generator) -> dict:
    """b[i, r] (4.3) and p[i, j] (4.4) over the groups g (D, n); a tie counts as half a win (endtoend core)."""
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
    stable_ranks = [{"rank": r + 1, "control": labels[int(np.argmax(b[:, r]))], "b": float(b[:, r].max())}
                    for r in range(n) if b[:, r].max() >= tau]
    return {"tau": tau, "accepted_pairs": acc, "unstable_pairs": unstable, "stable_ranks": stable_ranks,
            "n_pairs": n * (n - 1) // 2}


def joint_order_share(g: np.ndarray, accepted: list[dict], labels: list[str]) -> float:
    """Share of groups in which every accepted pair keeps its accepted order (algorithm 4.1, step 5)."""
    if not accepted:
        return 1.0
    ok = np.ones(g.shape[0], dtype=bool)
    for row in accepted:
        i, j = labels.index(row["higher"]), labels.index(row["lower"])
        ok &= g[:, i] > g[:, j]
    return float(ok.mean())


def tau_b(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import kendalltau
    t = kendalltau(x, y)[0]
    return float(t) if np.isfinite(t) else float("nan")


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))
