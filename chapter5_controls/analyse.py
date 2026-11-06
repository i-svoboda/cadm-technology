"""Analysis plan of PROTOCOL.md (Section 8) on the stored questionnaires of one batch. Never calls a model.

Steps (numbers follow PROTOCOL.md, Section 8):
  8.1  consistency (CR, Koczkodaj index) of every matrix, soft notes;
  8.2  pooled group S1: local priorities, criteria weights, own cluster weights, limit priorities under the prior v and
       the group's own cluster weights for the method's synergy reading, and the sensitivity cases (other reading,
       chapter 4 form of the own weights, constraint-cluster weight, variant emphasis, equal judgments, leave one persona
       out); diagnostics of the limit; cross-check with the chapter 3 code (replication_vendor/anp_model.py);
  8.3  S1 portfolio at the budget of 10.2 M USD: optimum by enumeration, greedy, best set with the leader, share of the
       expected loss removed; the chapter 3 model of +-20 % coverage changes (secondary);
  8.4  variance components (persona, wording, persona x wording, run) by block, B-PW bootstrap intervals with adjusted
       mean squares (vc_adjust.py), n_eff (4.5), participation ratio (4.6), vendor cross-check;
  8.5  rank and pair probabilities under S1, S2, S3 (D groups each, the same groups for every setting and for the
       portfolio);
  8.6  checkpoint 2 for the priorities: rule (4.8) at tau = 0.8 / 0.9 / 0.95 for each weight set, union of the unstable
       pairs over the two weight sets, B-Q bootstrap intervals (B-P for S3 as a check), joint order share, hand-over list;
  8.7  checkpoint for the portfolio: rule (5.1) under S3 and S2 with B-Q intervals;
  8.8  budget sweep (S1 sets, S3 q and q_c, rule (5.1) at tau = 0.9, price of the leader);
  8.9  descriptive comparison with chapter 3 (GPT-6.1-Sol group) and with the chapter 4 cluster weights;
  8.10 personas (exploratory); 8.11 sensitivity (group size, CR filter, chapter 3 assumed-noise Monte Carlo, chapter 3
       structure checks and cluster grid, chapter 4 weights with B = 0.048, zero-loss floor);
  8.12 exploratory X1 (variant weights against the scenario orders of chapter 2 and 4.7.7, ties kept);
  8.13 cost over every attempt folder, each pilot folder apart (runs/pilot and runs/pilot2, amendment of 2.11.2026).
Changes of the review of 2.11.2026: incomplete cells (A3), the cost count (B3), X1 ties (B4), the comparison note (B1),
and the optional items 2, 3, 4, 5, 7, 8, 9 and 10 (see PROTOCOL.md, Section 13).
Before the analysis, R["freeze_check"] compares PROTOCOL.md, analyse.py, core.py, vc_adjust.py and the case files with
protocol/freeze.json, and a warning is printed on any mismatch.

Usage: python -B scripts/analyse.py --batch runs/main --out results/main [--draws 10000 --B 2000 --inner 2000]
       python -B scripts/analyse.py --batch selftest/SYNTHETIC_runs --out selftest/SYNTHETIC_results --quick
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tacs_vendor"))
sys.path.insert(0, str(HERE / "replication_vendor"))

import numpy as np  # noqa: E402

import core  # noqa: E402
import vc_adjust  # noqa: E402

ROOT = HERE.parent
FREEZE = ROOT / "protocol" / "freeze.json"
INPUTS = ROOT / "case" / "analysis_inputs.json"
COMPARISON = ROOT / "case" / "comparison_reference.json"
TAUS = (0.8, 0.9, 0.95)
SEEDS = {"S2": 2026110201, "S3": 2026110202, "ties": 2026110203, "boot_bq_s2": 2026110204, "boot_bq_s3": 2026110205,
         "boot_bp_s3": 2026110206, "vc_pw": 2026110207, "pm20": 2026110208, "mc_ch3": 2026110209, "size": 2026110210,
         "sens_cr": 2026110211, "validate": 2026110212}
NAME = re.compile(r"^E\d_W\d_r\d$")
PRIMARY = [("prior", "source"), ("own", "source")]                 # the two weight sets of the decision (8.6)
SENS = [("prior", "target"), ("own", "target"), ("own3", "source"), ("ch4", "source")]   # sensitivity (8.2, 8.6, 8.11)
S1_EXTRA = [("prior", "source_nofixed"), ("own", "source_nofixed"), ("prior", "none"), ("own", "none")]   # 8.11
LAB = {("prior", "source"): "prior v, method's reading", ("own", "source"): "own weights, method's reading",
       ("prior", "target"): "prior v, other reading", ("own", "target"): "own weights, other reading",
       ("own3", "source"): "own weights in the chapter 4 form, method's reading",
       ("ch4", "source"): "chapter 4 group's weights with B = 0.048, method's reading",
       ("prior", "source_nofixed"): "prior v, method's reading without the fixed blocks SEG->IDM, BAK->IR",
       ("own", "source_nofixed"): "own weights, method's reading without the fixed blocks SEG->IDM, BAK->IR",
       ("prior", "none"): "prior v, no synergy block", ("own", "none"): "own weights, no synergy block"}
K = core.K
CH4_WEIGHTS: dict = {}                     # chapter 4 own weights scaled to B = 0.048 (set in main from the reference)


def skey(s: tuple) -> str:
    return f"{s[0]}/{s[1]}"


# ============================================================================ data
class Pool:
    """Complete questionnaires of one batch as arrays (analysis order of core.Layout)."""

    def __init__(self, batch: Path):
        self.lay = core.Layout()
        qs = []
        for d in sorted(p for p in batch.iterdir() if p.is_dir() and NAME.match(p.name)):
            res = json.loads((d / "result.json").read_text(encoding="utf-8")) if (d / "result.json").exists() else {}
            if res.get("status") != "complete":
                continue
            items = json.loads((d / "items.json").read_text(encoding="utf-8"))
            answer = json.loads((d / "answer.json").read_text(encoding="utf-8"))
            arr = self.lay.arrays(items, answer)
            mans = [json.loads(m.read_text(encoding="utf-8")) for m in sorted(d.glob("attempt_*/manifest.json"))]
            qs.append({"name": d.name, "persona": res["persona"], "wording": res["wording"], "run": res["replicate"],
                       "arr": arr, "manifests": mans, "attempts": len(mans),
                       "justifications": {j["block_id"]: j["text"] for j in answer.get("justifications", [])}})
        assert qs, f"no complete questionnaires in {batch}"
        self.qs = qs
        self.personas = sorted({q["persona"] for q in qs})
        self.wordings = sorted({q["wording"] for q in qs})
        self.Y = np.array([q["arr"]["y"] for q in qs])                  # N x 376
        self.E = np.array([q["arr"]["eta"] for q in qs])                # N x 7 x 5
        self.L = np.array([q["arr"]["eal"] for q in qs])                # N x 5
        self.p_idx = np.array([self.personas.index(q["persona"]) for q in qs])
        self.by_persona = [np.where(self.p_idx == i)[0] for i in range(len(self.personas))]
        self.N = len(qs)

    def subset(self, keep: np.ndarray) -> "Pool":
        new = object.__new__(Pool)
        new.__dict__ = dict(self.__dict__)
        new.qs = [self.qs[i] for i in keep]
        new.Y, new.E, new.L, new.p_idx = self.Y[keep], self.E[keep], self.L[keep], self.p_idx[keep]
        new.by_persona = [np.where(new.p_idx == i)[0] for i in range(len(self.personas))]
        new.N = len(keep)
        return new

    def vc_personas(self) -> list[str]:
        """Personas whose three persona x wording cells each have at least two complete questionnaires (PROTOCOL.md,
        7.3; review of 2.11.2026, A3). The variance decomposition uses only these; the other analyses use all."""
        cell = defaultdict(int)
        for q in self.qs:
            cell[(q["persona"], q["wording"])] += 1
        return [p for p in self.personas if all(cell[(p, w)] >= 2 for w in self.wordings)]

    def cells(self, values: np.ndarray) -> np.ndarray:
        """values (N, K) -> (P', W, R, K) for the personas of vc_personas (balanced); R = smallest number of runs in
        their cells (two in the plan). The personas left out are in self.vc_dropped."""
        keep = self.vc_personas()
        self.vc_dropped = [p for p in self.personas if p not in keep]
        cell = defaultdict(list)
        for i, q in enumerate(self.qs):
            if q["persona"] in keep:
                cell[(q["persona"], q["wording"])].append((q["run"], i))
        if len(keep) < 2:
            raise ValueError(f"fewer than two personas with complete cells: {keep}")
        r = min(len(v) for v in cell.values())
        Y = np.zeros((len(keep), len(self.wordings), r, values.shape[1]))
        for (p, w), lst in cell.items():
            lst.sort()
            for k, (_, i) in enumerate(lst[:r]):
                Y[keep.index(p), self.wordings.index(w), k] = values[i]
        return Y


def draw(pool: Pool, scheme: str, D: int, rng: np.random.Generator, persona_lists: list | None = None) -> np.ndarray:
    """Member indices (D, m). S1: all questionnaires; S2: one per persona (fixed roles); S3: P personas drawn with
    replacement, one questionnaire each (random roles). As the end-to-end analyse.draw."""
    lists = persona_lists if persona_lists is not None else pool.by_persona
    P = len(lists)
    if scheme == "S1":
        return np.concatenate(lists)[None, :]
    if scheme == "S2":
        return np.stack([L[rng.integers(0, len(L), D)] for L in lists], axis=1)
    if scheme == "S3":
        picks = rng.integers(0, P, (D, P))
        lens = np.array([len(L) for L in lists])
        u = rng.random((D, P))
        pos = np.floor(u * lens[picks]).astype(int)
        M = np.full((P, lens.max()), -1)
        for i, L in enumerate(lists):
            M[i, :len(L)] = L
        return M[picks, pos]
    raise ValueError(scheme)


def own3_weights(lay: core.Layout, y: np.ndarray) -> dict:
    """Chapter 4 form (4.7.5): eigenvector of the K, R, V part of the cluster matrix, scaled so that B keeps 0.048."""
    A = lay.matrices_of_size(y, 4)[:, 0, :3, :3]
    w = core.eigvec(A) * (1 - core.V_PRIOR["B"])
    return {"K": w[:, 0], "R": w[:, 1], "V": w[:, 2], "B": np.full(y.shape[0], core.V_PRIOR["B"])}


def weights_of(lay: core.Layout, loc: dict, y: np.ndarray, kind: str) -> dict:
    D = y.shape[0]
    if kind == "prior":
        return lay.prior_weights(D)
    if kind == "own":
        return lay.own_weights(loc)
    if kind == "own3":
        return own3_weights(lay, y)
    if kind == "ch4":
        assert CH4_WEIGHTS, "chapter 4 weights not loaded"
        return lay.prior_weights(D, CH4_WEIGHTS)
    raise ValueError(kind)


def ch4_weights(comp_ref: dict) -> dict:
    """The chapter 4 group's own weights (K, R, H of the 17-node network) scaled so that B keeps 0.048, as in 4.7.5 and
    5.5.2; H's weight goes to V (review of 2.11.2026, optional 9)."""
    w = comp_ref["chapter4_own_cluster_weights"]["pooled"]
    s = w["K"] + w["R"] + w["H"]
    b = core.V_PRIOR["B"]
    return {"K": w["K"] / s * (1 - b), "R": w["R"] / s * (1 - b), "V": w["H"] / s * (1 - b), "B": b}


def mc_near(p: float, tau: float, D: int) -> bool:
    """Within two Monte Carlo standard errors of tau or 1 - tau (review of 2.11.2026, optional 2; chapter 5, 5.5.2)."""
    se = (tau * (1 - tau) / D) ** 0.5
    return bool(abs(p - tau) <= 2 * se or abs(p - (1 - tau)) <= 2 * se)


def boot_union_share(bt: dict | None, tau: float, point_union: set) -> float | None:
    """Share of bootstrap samples whose union of unstable pairs over the two weight sets equals the point union
    (review of 2.11.2026, optional 3)."""
    if not bt:
        return None
    iu, ju = np.triu_indices(len(K), 1)
    sets = []
    for t in range(bt["B"]):
        u = set()
        for s in PRIMARY:
            P = bt["st"][skey(s)]["p"][t]
            q = np.maximum(P, 1 - P)
            u |= {f"{K[i]}-{K[j]}" for i, j in zip(iu, ju) if q[i, j] < tau}
        sets.append(u)
    return float(np.mean([s == point_union for s in sets]))


def evaluate(pool: Pool, idx: np.ndarray, settings: list, fast: bool = True, chunk: int = 2000,
             budgets: np.ndarray | None = None, keep_f: bool = False) -> dict:
    """Priorities of every setting, own cluster weights and the portfolio of every group in idx (D, m)."""
    lay = pool.lay
    D = idx.shape[0]
    out = {skey(s): np.zeros((D, len(K))) for s in settings}
    out["own_v"] = np.zeros((D, 4))
    out["set"] = np.zeros(D, dtype=int)
    out["f"] = np.zeros(D)
    out["eal_total"] = np.zeros(D)
    if budgets is not None:
        out["sets_b"] = np.zeros((D, len(budgets)), dtype=np.int16)
    if keep_f:
        out["f_all"] = np.zeros((D, len(core.SUBSETS)))
    for a in range(0, D, chunk):
        sl = slice(a, min(D, a + chunk))
        y, eta, eal = core.aggregate(pool.Y, pool.E, pool.L, idx[sl])
        loc = lay.local(y, fast=fast)
        out["own_v"][sl] = loc["CL|NET"]
        for s in settings:
            out[skey(s)][sl] = lay.priorities(loc, weights_of(lay, loc, y, s[0]), s[1], fast=fast)["g"]
        f = core.reductions(eta, eal)
        out["set"][sl], out["f"][sl] = core.best_sets(f)
        out["eal_total"][sl] = eal.sum(axis=1)
        if budgets is not None:
            for k, b in enumerate(budgets):
                out["sets_b"][sl, k] = core.best_sets(f, float(b))[0]
        if keep_f:
            out["f_all"][sl] = f
    return out


def probs(g: np.ndarray, seed: int) -> dict:
    s = core.summarise(g, np.random.default_rng(seed))
    return {"b": s["b"], "p": s["p"], "tie_share": s["tie_share"], "ranks": s["ranks"]}


def rnd(x, d=4):
    return np.round(np.asarray(x, dtype=float), d).tolist()


# ============================================================================ 8.4 variance components
def variance(pool: Pool, B: int) -> dict:
    lay = pool.lay
    blocks_log = {b: lay.blocks[b] for b in core.BLOCK_TYPES}
    blocks_log["all_pairs"] = list(range(core.N_PAIRS))
    out = {}
    try:
        pool.cells(pool.Y[:, :1])
    except ValueError as exc:
        return {"error": str(exc), "personas_left_out": pool.personas}
    out["design"] = {"personas_used": pool.vc_personas(), "personas_left_out": pool.vc_dropped,
                     "note": "personas with a cell of fewer than two complete questionnaires are left out of 8.4 "
                             "(PROTOCOL.md, 7.3)" if pool.vc_dropped else "complete design"}
    for label, vals, blocks in (("log_judgments", pool.Y, blocks_log),
                                ("coverage_shares", pool.E.reshape(pool.N, -1), {"coverage": list(range(35))}),
                                ("log_losses", np.log(np.maximum(pool.L, core.LOSS_FLOOR)), {"losses": list(range(5))})):
        Y = pool.cells(vals)
        P, W, R = Y.shape[:3]
        ms = vc_adjust.mean_squares(Y)
        comp = vc_adjust.components(ms, P, W, R)
        point = vc_adjust.block_shares(comp, blocks)
        res = {"runs_per_cell": R, "blocks": {}}
        boot = vc_adjust.bootstrap(Y, blocks, B, SEEDS["vc_pw"], "B-PW") if B else None
        for bname, cols in blocks.items():
            sh = point[bname]
            tot = comp[:, cols].mean(axis=1).sum()
            row = {"items": len(cols), "total_variance": float(tot),
                   "shares": dict(zip(vc_adjust.COMPS, rnd(sh))) if tot > 0 else None,
                   "wording_plus_interaction": float(sh[1] + sh[2]) if tot > 0 else None,
                   "n_eff": float(vc_adjust.n_eff(sh, N=P * W * R, m=W * R, L=W, R=R)) if tot > 0 else None}
            if boot is not None and tot > 0:
                bs = boot["adj"][bname]["shares"]
                ne = boot["adj"][bname]["n_eff"]
                ok = np.isfinite(bs).all(axis=1)
                row["ci95"] = {c: rnd(np.quantile(bs[ok, k], [0.025, 0.975])) for k, c in enumerate(vc_adjust.COMPS)}
                row["ci95"]["wording_plus_interaction"] = rnd(np.quantile(bs[ok, 1] + bs[ok, 2], [0.025, 0.975]))
                row["ci95"]["n_eff"] = rnd(np.quantile(ne[ok], [0.025, 0.975]), 2)
                row["persona_largest_share_of_boot"] = float(np.mean(bs[ok, 0] >= bs[ok, 1:].max(axis=1)))
            res["blocks"][bname] = row
        out[label] = res
    # run 1 against run 2 (descriptive; review of 2.11.2026, optional 7): mean and mean absolute paired shift per block
    shift = {}
    for label, vals, blocks in (("log_judgments", pool.Y, blocks_log),
                                ("coverage_shares", pool.E.reshape(pool.N, -1), {"coverage": list(range(35))}),
                                ("log_losses", np.log(np.maximum(pool.L, core.LOSS_FLOOR)), {"losses": list(range(5))})):
        Y = pool.cells(vals)
        if Y.shape[2] >= 2:
            d = Y[:, :, 1, :] - Y[:, :, 0, :]
            shift[label] = {b: {"mean_shift_run2_minus_run1": round(float(d[:, :, cols].mean()), 4),
                                "mean_abs_shift": round(float(np.abs(d[:, :, cols]).mean()), 4),
                                "sd_shift": round(float(d[:, :, cols].std()), 4)} for b, cols in blocks.items()}
    out["run_shift"] = shift
    try:                                     # cross-check with the TACS code (diagnostic only), on the same personas
        import smaa_ahp
        keep = set(pool.vc_personas())
        qs = [{"meta": {"persona": q["persona"], "wording": q["wording"]},
               "logj": {k: float(pool.Y[i, j]) for j, k in enumerate(lay.keys)}} for i, q in enumerate(pool.qs)
              if q["persona"] in keep]
        vend = smaa_ahp.variance_components(qs)
        out["vendor_crosscheck"] = {k: {"share": v["share"], "items": v["items"]} for k, v in vend.items()}
    except Exception as exc:  # noqa: BLE001
        out["vendor_crosscheck"] = {"error": str(exc)}
    return out


def participation_ratio(pool: Pool, vals: np.ndarray) -> dict:
    M = np.array([vals[L].mean(axis=0) for L in pool.by_persona])
    Dv = M - M.mean(axis=0)
    C = np.corrcoef(Dv)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    iu = np.triu_indices(len(pool.personas), 1)
    return {"participation_ratio": round(float(lam.sum() ** 2 / (lam ** 2).sum()), 2),
            "raw_profile_corr_mean": round(float(np.corrcoef(M)[iu].mean()), 3), "n_personas": len(pool.personas)}


def kendall_w(ranks_: np.ndarray) -> float:
    """W of m raters x N objects (1 = best), with the tie correction of replication common.kendall_w."""
    m, N = ranks_.shape
    Rs = ranks_.sum(axis=0)
    S = ((Rs - Rs.mean()) ** 2).sum()
    T = sum(((c ** 3) - c).sum() for c in (np.unique(row, return_counts=True)[1] for row in ranks_))
    return float(12 * S / (m ** 2 * (N ** 3 - N) - m * T))


def tied_ranks(w: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Ranks (1 = largest) with values closer than tol treated as ties and given their mean rank (review of
    2.11.2026, optional 4; the end-to-end finding D1)."""
    order = np.argsort(-w, kind="stable")
    r = np.empty(len(w))
    i = 0
    while i < len(w):
        j = i
        while j + 1 < len(w) and abs(w[order[j + 1]] - w[order[i]]) <= tol:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


# ============================================================================ bootstrap of the decisions
def boot(pool: Pool, scheme: str, resample: str, B: int, inner: int, seed: int) -> dict:
    """Percentile intervals of b and p (primary settings) and of q(X_B) and q_c. B-Q: questionnaires resampled within each
    persona (roles fixed); B-P: personas resampled with replacement, each keeping its questionnaires."""
    rng = np.random.default_rng(seed)
    n = len(K)
    st = {skey(s): {"b": np.zeros((B, n, n)), "p": np.zeros((B, n, n))} for s in PRIMARY}
    sets = np.zeros((B, inner), dtype=int)
    for t in range(B):
        if resample == "B-Q":
            lists = [L[rng.integers(0, len(L), len(L))] for L in pool.by_persona]
        else:
            lists = [pool.by_persona[i] for i in rng.integers(0, len(pool.by_persona), len(pool.by_persona))]
        idx = draw(pool, scheme, inner, rng, lists)
        ev = evaluate(pool, idx, PRIMARY)
        for s in PRIMARY:
            pr = probs(ev[skey(s)], seed + t)
            st[skey(s)]["b"][t], st[skey(s)]["p"][t] = pr["b"], pr["p"]
        sets[t] = ev["set"]
    return {"st": st, "sets": sets, "B": B, "inner": inner, "resample": resample, "scheme": scheme}


def add_boot(dec: dict, bt: dict | None, s: str, labels: list[str], point_unstable: set) -> dict:
    if not bt:
        return dec
    P = bt["st"][s]["p"]
    lo, hi = np.quantile(P, [0.025, 0.975], axis=0)
    tau = dec["tau"]
    for row in dec["accepted_pairs"] + dec["unstable_pairs"]:
        i, j = (labels.index(x) for x in row["pair"].split("-"))
        a, b = sorted([lo[i, j], hi[i, j]])
        q_lo, q_hi = (a, b) if row["higher"] == labels[i] else (1 - b, 1 - a)
        row["ci95_of_p"] = [round(float(q_lo), 4), round(float(q_hi), 4)]
        row["near_threshold"] = bool(q_lo <= tau <= q_hi or q_lo <= 1 - tau <= q_hi)
    q = np.maximum(P, 1 - P)
    iu, ju = np.triu_indices(len(labels), 1)
    sets = [frozenset(f"{labels[i]}-{labels[j]}" for i, j in zip(iu, ju) if q[t, i, j] < tau) for t in range(P.shape[0])]
    dec["boot_share_same_unstable_set"] = float(np.mean([x == point_unstable for x in sets]))
    dec["boot_share_no_unstable_pair"] = float(np.mean([len(x) == 0 for x in sets]))
    bb = bt["st"][s]["b"]
    dec["b_ci95"] = {f"{labels[i]}@{r + 1}": rnd(np.quantile(bb[:, i, r], [0.025, 0.975])) for i in range(len(labels))
                     for r in range(len(labels))}
    return dec


def portfolio_boot(bt: dict, ref: int) -> dict:
    sets = bt["sets"]
    q = (sets == ref).mean(axis=1)
    qc = np.stack([core.MASK_E[s].mean(axis=0) for s in sets])
    return {"q_ci95": rnd(np.quantile(q, [0.025, 0.975])),
            "q_c_ci95": {c: rnd(np.quantile(qc[:, i], [0.025, 0.975])) for i, c in enumerate(K)},
            "_q": q, "_qc": qc}


# ============================================================================ 8.13 cost
def cost_of(batch: Path) -> dict:
    """Calls, retries, replacements, time and tokens over every attempt_*/manifest.json under the batch folder
    (complete, failed, replaced and interrupted questionnaires alike). Usage falls back to the session record's
    token_info when the event stream carried none."""
    walls, inp, cached, outt, reas, prim, bins = [], [], [], [], [], [], set()
    n_retry = 0
    for m in sorted(batch.glob("*/attempt_*/manifest.json")):
        man = json.loads(m.read_text(encoding="utf-8"))
        walls.append(man.get("wall_seconds") or 0)
        u = man.get("usage_used") or (man.get("usage") or [None])[-1] or \
            ((man.get("token_info_after") or {}).get("total_token_usage")) or {}
        inp.append(u.get("input_tokens") or 0)
        cached.append(u.get("cached_input_tokens") or 0)
        outt.append(u.get("output_tokens") or 0)
        reas.append(u.get("reasoning_output_tokens") or 0)
        pr_ = ((man.get("rate_limits_after") or {}).get("primary") or {}).get("used_percent")
        if pr_ is not None:
            prim.append((man.get("started_utc"), pr_))
        if m.parent.name != "attempt_1":
            n_retry += 1
        bins.add(((man.get("codex_binary") or {}).get("sha256")))
    folders = [d.name for d in batch.iterdir() if d.is_dir()]
    api = (sum(inp) - sum(cached)) * 2.0 / 1e6 + sum(cached) * 0.1 / 1e6 + sum(outt) * 10.0 / 1e6     # (5.6), GPT-6.1-Sol
    share = [0.0175 * a / 1000 + 0.147 * b / 1000 for a, b in zip(inp, outt)]
    return {"folder": str(batch), "calls": len(walls), "retries": n_retry,
            "replaced": sorted(f for f in folders if "_failed" in f),
            "interrupted": sorted(f for f in folders if "_interrupted" in f),
            "wall_median_s": float(np.median(walls)) if walls else None, "wall_total_min": float(sum(walls) / 60),
            "input_tokens_median": float(np.median(inp)) if inp else None,
            "output_tokens_median": float(np.median(outt)) if outt else None,
            "reasoning_tokens_median": float(np.median(reas)) if reas else None,
            "tokens_total": int(sum(inp) + sum(outt)), "api_equivalent_usd_formula_5_6": round(api, 2),
            "five_hour_share_per_call_token_model_median": round(float(np.median(share)), 3) if share else None,
            "five_hour_share_per_call_token_model_max": round(float(np.max(share)), 3) if share else None,
            "primary_readings_first_last": [sorted(prim)[0], sorted(prim)[-1]] if prim else None,
            "codex_binaries": sorted(b for b in bins if b)}


def versions() -> dict:
    import platform
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}


# ============================================================================ freeze check
def freeze_check() -> dict:
    out = {"freeze_file": str(FREEZE), "freeze_sha256": None, "frozen": None, "files": {}, "all_match": False}
    if not FREEZE.exists():
        out["warning"] = "protocol/freeze.json is missing"
        return out
    fr = json.loads(FREEZE.read_text(encoding="utf-8"))
    out["freeze_sha256"] = hashlib.sha256(FREEZE.read_bytes()).hexdigest()
    out["frozen"] = fr.get("frozen")
    fo, fi = fr.get("other_files") or {}, fr.get("inputs") or {}
    want = {"PROTOCOL.md": fo.get("PROTOCOL.md"), "scripts/analyse.py": fo.get("scripts/analyse.py"),
            "scripts/core.py": fi.get("core.py"), "scripts/vc_adjust.py": fo.get("scripts/vc_adjust.py"),
            "case/analysis_inputs.json": fo.get("case/analysis_inputs.json"),
            "case/comparison_reference.json": fo.get("case/comparison_reference.json"),
            "case/dso_controls_case.json": fi.get("case")}
    for rel, h in want.items():
        p = ROOT / rel
        now = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        out["files"][rel] = "match" if h and now == h else "not frozen" if not h else "MISMATCH"
    out["all_match"] = all(v == "match" for v in out["files"].values())
    return out


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--B", type=int, default=2000, help="bootstrap replicates (variance shares and probabilities)")
    ap.add_argument("--inner", type=int, default=2000, help="simulated groups per bootstrap replicate")
    ap.add_argument("--trials", type=int, default=1000, help="trials of the chapter 3 models (+-20 %%, assumed noise)")
    ap.add_argument("--pilot-batch", action="append",
                    help="pilot folder for the cost report (repeatable; default: every sibling folder pilot*)")
    ap.add_argument("--quick", action="store_true", help="small numbers for tests: draws 2000, B 30, inner 300, trials 200")
    args = ap.parse_args()
    if args.quick:
        args.draws, args.B, args.inner, args.trials = 2000, 30, 300, 200
    t_start = time.time()
    inputs = core.load_json(INPUTS)
    comp_ref = core.load_json(COMPARISON)
    assert inputs["cluster_prior_v"] == core.V_PRIOR and inputs["costs_musd"] == core.COSTS and \
        inputs["budget_musd"] == core.BUDGET, "analysis_inputs.json differs from core.py"
    CH4_WEIGHTS.update(ch4_weights(comp_ref))
    pool = Pool(Path(args.batch))
    lay = pool.lay
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    R: dict = {"batch": str(Path(args.batch).resolve()), "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "n_questionnaires": pool.N, "personas": pool.personas, "wordings": pool.wordings, "controls": K,
               "settings": {"draws": args.draws, "B": args.B, "inner": args.inner, "trials": args.trials, "seeds": SEEDS}}
    bs = Path(args.batch) / "batch_sha256.json"
    R["batch_sha256_file"] = hashlib.sha256(bs.read_bytes()).hexdigest() if bs.exists() else None
    R["freeze_check"] = freeze_check()
    if not R["freeze_check"]["all_match"]:
        print(f"WARNING: protocol or analysis code differs from protocol/freeze.json: {R['freeze_check']['files']}",
              flush=True)
    log = lambda m: print(f"[{time.time() - t_start:7.1f} s] {m}", flush=True)   # noqa: E731

    # ------------------------------------------------------------------ 8.1 consistency
    crs = defaultdict(list)
    kz = defaultdict(list)
    for i in range(pool.N):
        for m in lay.mkeys:
            if len(lay.els[m]) >= 3:
                A = lay.matrix(pool.Y[i], m)
                crs[core.block_type(m)].append(float(core.cr(A)))
                kz[core.block_type(m)].append(core.koczkodaj(A))
    allcr = np.concatenate([np.array(v) for v in crs.values()])
    soft = defaultdict(int)
    for q in pool.qs:
        for k, v in q["arr"]["soft"].items():
            soft[k] += v
    R["consistency"] = {"all": {"matrices": int(allcr.size), "median": float(np.median(allcr)),
                                "p90": float(np.quantile(allcr, .9)), "max": float(allcr.max()),
                                "share_over_0.10": float(np.mean(allcr > 0.1))},
                        "by_block": {b: {"matrices": len(v), "median": float(np.median(v)), "p90": float(np.quantile(v, .9)),
                                         "max": float(np.max(v)), "share_over_0.10": float(np.mean(np.array(v) > 0.1)),
                                         "koczkodaj_mean": float(np.mean(kz[b])), "koczkodaj_max": float(np.max(kz[b]))}
                                     for b, v in crs.items()},
                        "soft_notes": dict(soft)}
    # "less is better" read of criterion C in every questionnaire (review of 2.11.2026, optional 5)
    locq = lay.local(pool.Y)
    tc = np.array([core.tau_b(locq["K|R:C"][i], -core.COST_VEC) for i in range(pool.N)])
    R["consistency"]["tau_b_C_priorities_vs_negative_cost"] = {
        "median": float(np.median(tc)), "share_positive": float(np.mean(tc > 0)),
        "not_positive": [pool.qs[i]["name"] for i in np.flatnonzero(~(tc > 0))]}
    R["consistency"]["zero_losses"] = int((pool.L == 0).sum())
    log("8.1 consistency")

    # ------------------------------------------------------------------ 8.2 pooled group S1 (exact path)
    allidx = np.arange(pool.N)[None, :]
    y1, eta1, eal1 = core.aggregate(pool.Y, pool.E, pool.L, allidx)
    loc1 = lay.local(y1)
    S1 = {"criteria_weights": dict(zip(core.R, rnd(loc1["R|GOAL"][0]))),
          "own_cluster_weights": dict(zip(core.CL_ELEMENTS, rnd(loc1["CL|NET"][0]))),
          "own_cluster_weights_chapter4_form": {c: round(float(v[0]), 4) for c, v in own3_weights(lay, y1).items()},
          "prior_cluster_weights": core.V_PRIOR,
          "CR_aggregated": {m: round(float(core.cr(lay.matrix(y1[0], m))), 4) for m in lay.mkeys if len(lay.els[m]) >= 3},
          "local": {m: dict(zip(lay.els[m], rnd(loc1[m][0]))) for m in lay.mkeys}, "settings": {}}
    g1 = {}
    for s in PRIMARY + SENS + S1_EXTRA:
        v = weights_of(lay, loc1, y1, s[0])
        W = lay.supermatrix(loc1, v, s[1])
        pi, iters = core.limit_power(W)
        g = pi[0, lay.kpos] / pi[0, lay.kpos].sum()
        g1[skey(s)] = g
        Wd = (1 - 1e-6) * W + 1e-6 / W.shape[1]                       # damping check (3.10)
        pid = core.limit_power(Wd)[0]
        order = [K[i] for i in np.argsort(-g, kind="stable")]
        S1["settings"][skey(s)] = {
            "label": LAB[s], "priorities": dict(zip(K, rnd(g))), "order": order, "leader": order[0],
            "margin_leader_second": round(float(np.sort(g)[-1] - np.sort(g)[-2]), 4),
            "variant_priorities": dict(zip(core.V, rnd(pi[0, lay.vpos] / pi[0, lay.vpos].sum()))),
            "cluster_mass": {c: round(float(pi[0, lay.cluster_pos[c]].sum()), 4) for c in core.CLUSTER_ORDER},
            "iterations": int(iters[0]), "lambda2": round(core.second_eigenvalue(W[0]), 4),
            "column_sums_ok": bool(np.allclose(W[0].sum(axis=0), 1.0)),
            "max_diff_direct_solve": float(np.abs(core.limit_solve(W)[0] - pi[0]).max()),
            "max_diff_damping_1e-6": float(np.abs(pid[0] - pi[0]).max())}
    # cross-check with the chapter 3 code: replication_vendor/anp_model.Case on the same local vectors (V as H)
    try:
        import anp_model
        prot = {"goal": "", "clusters": {c2: {"name": c2, "nodes": [{"id": n, "name": n} for n in core.CLUSTERS[c1]]}
                                         for c1, c2 in (("K", "K"), ("R", "R"), ("V", "H"), ("B", "B"))}}
        case3 = anp_model.Case(prot)
        loc3 = {m.replace("V|", "H|").replace("|V:", "|H:"): loc1[m][0] for m in lay.mkeys if m != "CL|NET"}
        xc = {}
        for s in PRIMARY + SENS + S1_EXTRA:
            v = weights_of(lay, loc1, y1, s[0])
            vv = {"K": float(v["K"][0]), "R": float(v["R"][0]), "H": float(v["V"][0]), "B": float(v["B"][0])}
            syn = {"source": anp_model.SYNERGY, "target": [(t, s_) for s_, t in anp_model.SYNERGY],
                   "source_nofixed": [(s_, t) for s_, t in anp_model.SYNERGY if t == "MON"], "none": []}[s[1]]
            pr3, _ = case3.control_priorities(loc3, "D1", v=vv, synergy=syn)
            xc[skey(s)] = float(max(abs(pr3[k] - g1[skey(s)][i]) for i, k in enumerate(K)))
        S1["chapter3_code_crosscheck_max_abs_diff"] = xc
    except Exception as exc:  # noqa: BLE001
        S1["chapter3_code_crosscheck_max_abs_diff"] = {"error": repr(exc)}
    # equal judgments (structure alone), constraint-cluster weight at which the leader changes, variant emphasis
    loc0 = lay.local(np.zeros((1, core.N_PAIRS)))
    S1["equal_judgments"] = {skey(s): dict(zip(K, rnd(lay.priorities(loc0, lay.prior_weights(1), s[1])["g"][0])))
                             for s in (("prior", "source"), ("prior", "target"))}
    sw = {}
    for kind in ("prior", "own"):
        base = weights_of(lay, loc1, y1, kind)
        base = {c: float(base[c][0]) for c in core.CL_ELEMENTS}
        rest = sum(base[c] for c in ("K", "R", "V"))
        lead0 = S1["settings"][f"{kind}/source"]["leader"]
        found = None
        for bw in np.round(np.arange(0.0, 0.5001, 0.005), 3):
            v = {c: np.array([base[c] / rest * (1 - bw)]) for c in ("K", "R", "V")} | {"B": np.array([bw])}
            lead = K[int(np.argmax(lay.priorities(loc1, v, "source")["g"][0]))]
            if bw >= base["B"] and lead != lead0 and found is None:
                found = {"B_weight": float(bw), "new_leader": lead}
        below = []
        for bw in np.round(np.arange(0.0, base["B"], 0.005), 3):
            v = {c: np.array([base[c] / rest * (1 - bw)]) for c in ("K", "R", "V")} | {"B": np.array([bw])}
            lead = K[int(np.argmax(lay.priorities(loc1, v, "source")["g"][0]))]
            if lead != lead0:
                below.append(float(bw))
        sw[kind] = {"B_weight_of_the_set": round(base["B"], 4), "leader": lead0,
                    "first_B_weight_above_with_another_leader": found, "B_weights_below_with_another_leader": below}
    S1["constraint_weight_switch"] = sw
    emph = {}
    for vname in core.V:
        g = lay.priorities(loc1, lay.prior_weights(1), "source", variant_mult={vname: 2.0})["g"][0]
        emph[vname] = {"leader": K[int(np.argmax(g))], "max_abs_change": round(float(np.abs(g - g1["prior/source"]).max()), 4),
                       "order": [K[i] for i in np.argsort(-g, kind="stable")]}
    S1["variant_weight_doubled_prior_v"] = emph
    # chapter 3's cluster grid (3.3.4, appendix Д): proportions of K, V, R in steps of 0.05, each at least 0.15, with
    # B = 0.048; 78 interior points (review of 2.11.2026, optional 9)
    grid_pts = [(a / 20, b / 20, (20 - a - b) / 20) for a in range(3, 21) for b in range(3, 21) if 20 - a - b >= 3]
    assert len(grid_pts) == 78
    cg_out = {}
    for reading in ("source", "target"):
        def lead_at(pk, pv, pr):
            v = {"K": np.array([pk * (1 - core.V_PRIOR["B"])]), "V": np.array([pv * (1 - core.V_PRIOR["B"])]),
                 "R": np.array([pr * (1 - core.V_PRIOR["B"])]), "B": np.array([core.V_PRIOR["B"]])}
            return K[int(np.argmax(lay.priorities(loc1, v, reading)["g"][0]))]
        leaders = [lead_at(*pt) for pt in grid_pts]
        eq = lead_at(1 / 3, 1 / 3, 1 / 3)
        s1_lead = S1["settings"][f"prior/{reading}"]["leader"]
        cg_out[reading] = {"leaders": {k: leaders.count(k) for k in sorted(set(leaders))},
                           "leader_at_equal_proportions": eq,
                           "points_with_leader_at_equal_proportions": leaders.count(eq),
                           "points_with_S1_prior_v_leader": leaders.count(s1_lead), "points": len(grid_pts)}
    S1["chapter3_cluster_grid"] = cg_out
    # leave one persona out (chapter 3, 3.3.5, at the level of the persona's six questionnaires)
    loeo = {}
    for s in PRIMARY:
        rows = {}
        for pi_, p in enumerate(pool.personas):
            keep = np.concatenate([L for k, L in enumerate(pool.by_persona) if k != pi_])[None, :]
            y_, _, _ = core.aggregate(pool.Y, pool.E, pool.L, keep)
            l_ = lay.local(y_)
            rows[p] = lay.priorities(l_, weights_of(lay, l_, y_, s[0]), s[1])["g"][0]
        base = g1[skey(s)]
        loeo[skey(s)] = {"leaders": {p: K[int(np.argmax(g))] for p, g in rows.items()},
                         "mean_abs_change": float(np.mean([np.abs(g - base).mean() for g in rows.values()])),
                         "max_abs_change": float(max(np.abs(g - base).max() for g in rows.values())),
                         "kendall_tau_b": {p: round(core.tau_b(g, base), 3) for p, g in rows.items()}}
    S1["leave_one_persona_out"] = loeo
    # Kendall's W per criterion over the seven personas (descriptive, chapter 3, 3.1.2)
    kw = {}
    for c in core.R:
        rk = []
        for L in pool.by_persona:
            y_, _, _ = core.aggregate(pool.Y, pool.E, pool.L, L[None, :])
            rk.append(tied_ranks(lay.local(y_)[f"K|R:{c}"][0]))
        kw[c] = round(kendall_w(np.array(rk)), 3)
    S1["kendall_w_controls_by_criterion"] = kw
    R["S1"] = S1
    log("8.2 S1")

    # ------------------------------------------------------------------ 8.3 S1 portfolio
    f1 = core.reductions(eta1, eal1)
    i1, v1 = core.best_sets(f1)
    X_B = int(i1[0])
    eal_tot = float(eal1.sum())
    port = {"budget": core.BUDGET, "eal_by_variant": dict(zip(core.V, rnd(eal1[0], 3))), "eal_total": round(eal_tot, 3),
            "coverage_mean": {c: dict(zip(core.V, rnd(eta1[0, i], 3))) for i, c in enumerate(K)},
            "X_B": core.set_name(X_B), "cost": round(float(core.SUB_COST[X_B]) if X_B >= 0 else 0.0, 2),
            "f": round(float(v1[0]), 4), "share_removed": round(float(v1[0]) / eal_tot, 4),
            "greedy": core.greedy(eta1[0], eal1[0]), "feasible_sets": int((core.SUB_COST <= core.BUDGET + 1e-9).sum())}
    port["greedy_equals_optimum"] = sorted(port["greedy"]) == sorted(core.set_name(X_B).strip("{}").split(", "))
    leader_sets = {}
    for s in PRIMARY:
        lead = S1["settings"][skey(s)]["leader"]
        il, vl = core.best_sets(f1, require=K.index(lead))
        leader_sets[skey(s)] = {"leader": lead, "leader_in_X_B": bool(core.MASK_E[X_B][K.index(lead)]),
                                "best_set_with_leader": core.set_name(int(il[0])), "f": round(float(vl[0]), 4),
                                "f_difference": round(float(v1[0] - vl[0]), 4),
                                "relative_difference": round(float((v1[0] - vl[0]) / v1[0]), 4) if v1[0] > 0 else None}
    port["best_set_with_leader"] = leader_sets
    rng = np.random.default_rng(SEEDS["pm20"])                     # chapter 3 / 5 model: shares x U(0.8, 1.2), cap 1
    etas = np.minimum(1.0, eta1[0][None] * rng.uniform(0.8, 1.2, (args.trials,) + eta1[0].shape))
    sets20, _ = core.best_sets(core.reductions(etas, np.repeat(eal1, args.trials, axis=0)))
    port["pm20_coverage_model"] = {str(t): core.portfolio_decision(sets20, X_B, t) for t in TAUS}
    port["pm20_coverage_model"]["trials"] = args.trials
    R["S1_portfolio"] = port
    log("8.3 S1 portfolio")

    # ------------------------------------------------------------------ 8.4 variance components
    R["variance"] = variance(pool, args.B)
    R["participation_ratio"] = {"log_judgments": participation_ratio(pool, pool.Y),
                                "coverage": participation_ratio(pool, pool.E.reshape(pool.N, -1))}
    log("8.4 variance components")

    # ------------------------------------------------------------------ 8.5 S2 and S3 (fast path, checked)
    grid = np.round(np.arange(0.0, core.COST_VEC.sum() + 1e-9, 0.1), 1)
    idx = {"S2": draw(pool, "S2", args.draws, np.random.default_rng(SEEDS["S2"])),
           "S3": draw(pool, "S3", args.draws, np.random.default_rng(SEEDS["S3"]))}
    ev = {s: evaluate(pool, idx[s], PRIMARY + SENS, budgets=grid if s == "S3" else None) for s in ("S2", "S3")}
    rngv = np.random.default_rng(SEEDS["validate"])               # fast path against the exact path
    vi = draw(pool, "S3", 300, rngv)
    fa, ex = evaluate(pool, vi, PRIMARY + SENS, fast=True), evaluate(pool, vi, PRIMARY + SENS, fast=False)
    R["fast_path_check_max_abs_diff"] = {skey(s): float(np.abs(fa[skey(s)] - ex[skey(s)]).max()) for s in PRIMARY + SENS}
    assert max(R["fast_path_check_max_abs_diff"].values()) < 1e-8, R["fast_path_check_max_abs_diff"]
    sm = {}
    for sch in ("S1", "S2", "S3"):
        sm[sch] = {}
        for s in PRIMARY + SENS:
            g = g1[skey(s)][None] if sch == "S1" else ev[sch][skey(s)]
            pr = probs(g, SEEDS["ties"])
            sm[sch][skey(s)] = {"label": LAB[s], "b": rnd(pr["b"]), "p": rnd(pr["p"]), "tie_share": pr["tie_share"],
                                "mean_priority": dict(zip(K, rnd(g.mean(0)))), "sd_priority": dict(zip(K, rnd(g.std(0)))),
                                "_g": g}
        if sch != "S1":
            ow = ev[sch]["own_v"]
            sm[sch]["own_cluster_weights_mean"] = dict(zip(core.CL_ELEMENTS, rnd(ow.mean(0))))
            sm[sch]["own_cluster_weights_q05_q95"] = {c: rnd(np.quantile(ow[:, i], [0.05, 0.95]))
                                                      for i, c in enumerate(core.CL_ELEMENTS)}
            gl = ev[sch]["prior/source"]
            lead_in = core.MASK_E[ev[sch]["set"], np.argmax(gl, axis=1)]
            sm[sch]["share_leader_prior_v_in_own_optimal_portfolio"] = float(lead_in.mean())
    R["smaa"] = {sch: {s: ({k: v for k, v in d.items() if not k.startswith("_")} if isinstance(d, dict) else d)
                       for s, d in res.items()} for sch, res in sm.items()}
    log("8.5 S1-S3")

    # ------------------------------------------------------------------ 8.6 checkpoint 2 for the priorities
    boots = {}
    if args.B:
        boots["S2"] = boot(pool, "S2", "B-Q", args.B, args.inner, SEEDS["boot_bq_s2"])
        log("bootstrap B-Q S2")
        boots["S3"] = boot(pool, "S3", "B-Q", args.B, args.inner, SEEDS["boot_bq_s3"])
        log("bootstrap B-Q S3")
        boots["S3_BP"] = boot(pool, "S3", "B-P", args.B, args.inner, SEEDS["boot_bp_s3"])
        log("bootstrap B-P S3")
    dec = {}
    for sch in ("S2", "S3"):
        for s in PRIMARY + SENS:
            for tau in TAUS:
                b_, p_ = np.array(sm[sch][skey(s)]["b"]), np.array(sm[sch][skey(s)]["p"])
                d = core.decision(b_, p_, K, tau)
                d["joint_order_share"] = core.joint_order_share(sm[sch][skey(s)]["_g"], d["accepted_pairs"], K)
                for row in d["accepted_pairs"] + d["unstable_pairs"]:          # optional 2: two Monte Carlo SEs
                    row["mc_near_threshold"] = mc_near(row["p"], tau, args.draws)
                d["ranks_mc_near_threshold"] = [r + 1 for r in range(len(K)) if mc_near(float(b_[:, r].max()), tau, args.draws)]
                if s in PRIMARY:
                    pu = frozenset(x["pair"] for x in d["unstable_pairs"])
                    d = add_boot(d, boots.get(sch), skey(s), K, pu)
                    if sch == "S3" and "S3_BP" in boots:
                        d["check_B-P_share_same_unstable_set"] = add_boot(core.decision(b_, p_, K, tau), boots["S3_BP"],
                                                                         skey(s), K, pu).get("boot_share_same_unstable_set")
                dec[f"{sch}/{skey(s)}/{tau}"] = d
    R["decision"] = dec

    def combine(sch: str, tau: float, sets_: list) -> dict:
        rows = {s: {r["pair"]: r for r in dec[f"{sch}/{skey(s)}/{tau}"]["accepted_pairs"] +
                    dec[f"{sch}/{skey(s)}/{tau}"]["unstable_pairs"]} for s in sets_}
        unst = {s: {r["pair"] for r in dec[f"{sch}/{skey(s)}/{tau}"]["unstable_pairs"]} for s in sets_}
        acc = {s: {r["pair"]: r["higher"] for r in dec[f"{sch}/{skey(s)}/{tau}"]["accepted_pairs"]} for s in sets_}
        allu = set().union(*unst.values())
        common = set.intersection(*[set(a) for a in acc.values()])
        conflict = sorted(pr for pr in common if len({acc[s][pr] for s in sets_}) > 1)
        same = sorted(pr for pr in common if pr not in conflict)
        return {"tau": tau, "scheme": sch, "weight_sets": [LAB[s] for s in sets_],
                "accepted_same_direction": [{"pair": pr, "higher": acc[sets_[0]][pr],
                                             "p": {skey(s): rows[s][pr]["p"] for s in sets_}} for pr in same],
                "to_human_experts": sorted(allu, key=lambda pr: min(abs(rows[s][pr]["p"] - 0.5) for s in sets_)),
                "accepted_in_opposite_directions": [{"pair": pr, "higher": {skey(s): acc[s][pr] for s in sets_}}
                                                    for pr in conflict],
                "unstable_ranks": {skey(s): [r for r in range(1, 8) if r not in
                                             {x["rank"] for x in dec[f"{sch}/{skey(s)}/{tau}"]["stable_ranks"]}] for s in sets_},
                "rows": {pr: {skey(s): rows[s][pr] for s in sets_} for pr in sorted(allu | set(conflict))}}

    cp2 = {f"{sch}/{tau}": combine(sch, tau, PRIMARY) for sch in ("S2", "S3") for tau in TAUS}
    for sch in ("S2", "S3"):                       # optional 3: bootstrap share of the same union of hand-over pairs
        for tau in TAUS:
            c = cp2[f"{sch}/{tau}"]
            c["boot_share_same_union"] = boot_union_share(boots.get(sch), tau, set(c["to_human_experts"]))
            if sch == "S3":
                c["check_B-P_share_same_union"] = boot_union_share(boots.get("S3_BP"), tau, set(c["to_human_experts"]))
            c["mc_near_threshold_pairs"] = sorted({pr for s in PRIMARY for pr, row in
                                                   ((r["pair"], r) for r in dec[f"{sch}/{skey(s)}/{tau}"]["accepted_pairs"] +
                                                    dec[f"{sch}/{skey(s)}/{tau}"]["unstable_pairs"])
                                                   if row["mc_near_threshold"]})
    # personas whose own six questionnaires put each side higher (hand-over protocol of 4.6.2)
    per_g = {}
    for pi_, p in enumerate(pool.personas):
        L = pool.by_persona[pi_][None, :]
        e_ = evaluate(pool, L, PRIMARY, fast=False)
        per_g[p] = {skey(s): e_[skey(s)][0] for s in PRIMARY}
    for key, c in cp2.items():
        for pr, row in c["rows"].items():
            a, b = pr.split("-")
            row["personas_preferring"] = {skey(s): {a: [p for p in pool.personas if per_g[p][skey(s)][K.index(a)] >=
                                                         per_g[p][skey(s)][K.index(b)]],
                                                    b: [p for p in pool.personas if per_g[p][skey(s)][K.index(a)] <
                                                        per_g[p][skey(s)][K.index(b)]]} for s in PRIMARY}
    R["checkpoint2_priorities"] = cp2
    rd = {}                                                      # order changes between the readings (reported apart)
    for kind in ("prior", "own"):
        a_ = {r["pair"]: r["higher"] for r in dec[f"S3/{kind}/source/0.9"]["accepted_pairs"]}
        b_ = {r["pair"]: r["higher"] for r in dec[f"S3/{kind}/target/0.9"]["accepted_pairs"]}
        rd[kind] = {"accepted_in_both_readings_opposite": sorted(pr for pr in set(a_) & set(b_) if a_[pr] != b_[pr]),
                    "unstable_other_reading": [r["pair"] for r in dec[f"S3/{kind}/target/0.9"]["unstable_pairs"]]}
    R["reading_sensitivity_S3_tau_0.9"] = rd
    log("8.6 checkpoint 2")

    # ------------------------------------------------------------------ 8.7 portfolio checkpoint (5.1)
    pc = {}
    for sch in ("S2", "S3"):
        pb = portfolio_boot(boots[sch], X_B) if sch in boots else None
        for tau in TAUS:
            d = core.portfolio_decision(ev[sch]["set"], X_B, tau)
            d["mc_near_threshold"] = mc_near(d["q_X_B"], tau, args.draws)              # optional 2
            for c in K:
                d["controls"][c]["mc_near_threshold"] = mc_near(d["controls"][c]["q_c"], tau, args.draws)
            if pb:
                d["q_ci95"] = pb["q_ci95"]
                d["near_threshold"] = bool(pb["q_ci95"][0] <= tau <= pb["q_ci95"][1])
                for c in K:
                    d["controls"][c]["q_c_ci95"] = pb["q_c_ci95"][c]
                    lo, hi = pb["q_c_ci95"][c]
                    d["controls"][c]["near_threshold"] = bool(lo <= tau <= hi or lo <= 1 - tau <= hi)
            pc[f"{sch}/{tau}"] = d
        vals, cnt = np.unique(ev[sch]["set"], return_counts=True)
        top = sorted(zip(cnt, vals), reverse=True)[:8]
        pc[f"{sch}/most_frequent_sets"] = [{"set": core.set_name(int(v)), "share": round(float(c) / args.draws, 4)}
                                           for c, v in top]
        pc[f"{sch}/share_removed_q05_q50_q95"] = rnd(np.quantile(ev[sch]["f"] / ev[sch]["eal_total"], [0.05, 0.5, 0.95]))
    if "S3_BP" in boots:
        pb = portfolio_boot(boots["S3_BP"], X_B)
        pc["S3/B-P_check_q_ci95"] = pb["q_ci95"]
    R["checkpoint_portfolio"] = pc
    log("8.7 portfolio checkpoint")

    # ------------------------------------------------------------------ 8.8 budget sweep
    rows = []
    prev = None
    switches = []
    lead_p = S1["settings"]["prior/source"]["leader"]
    for k, b in enumerate(grid):
        ib, vb = core.best_sets(f1, float(b))
        il, vl = core.best_sets(f1, float(b), require=K.index(lead_p))
        ref = int(ib[0])
        d = core.portfolio_decision(ev["S3"]["sets_b"][:, k].astype(int), ref, 0.9)
        name = core.set_name(ref)
        if name != prev:
            switches.append({"budget": float(b), "set": name})
            prev = name
        rows.append({"budget": float(b), "set_S1": name, "f_S1": round(float(vb[0]), 4),
                     "share_removed_S1": round(float(vb[0]) / eal_tot, 4), "q_S3": d["q_X_B"],
                     "set_accepted_tau_0.9": d["set_accepted"], "to_human_experts_tau_0.9": d["to_human_experts"],
                     **{f"q_{c}": d["controls"][c]["q_c"] for c in K},
                     "best_set_with_leader_prior_v": core.set_name(int(il[0])),
                     "price_of_leader": round(float(vb[0] - vl[0]), 4) if int(il[0]) >= 0 else None})
    in_xb = [r["budget"] for r in rows if r["set_S1"] == core.set_name(X_B)]
    R["budget_sweep"] = {"grid_musd": [float(grid[0]), float(grid[-1]), 0.1], "switches_S1": switches,
                         "leader_prior_v": lead_p, "budget_range_of_X_B": [min(in_xb), max(in_xb)] if in_xb else None,
                         "rows": rows}
    with open(out_dir / "budget_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()})
    log("8.8 budget sweep")

    # ------------------------------------------------------------------ 8.9 comparison with chapter 3 and chapter 4
    c3 = comp_ref["chapter3_gpt61"]
    cmpd = {}
    for s, key3 in ((("prior", "source"), "priorities_source_reading"), (("prior", "target"), "priorities_target_reading")):
        a = g1[skey(s)]
        b = np.array([c3[key3][k] for k in K])
        cmpd[skey(s)] = {"dso": dict(zip(K, rnd(a))), "chapter3": dict(zip(K, rnd(b))),
                         "difference": dict(zip(K, rnd(a - b))), "kendall_tau_b": round(core.tau_b(a, b), 3),
                         "leader_dso": K[int(np.argmax(a))], "leader_chapter3": K[int(np.argmax(b))]}
    cw = np.array([S1["criteria_weights"][c] for c in core.R])
    cw3 = np.array([c3["criteria_weights"][c] for c in core.R])
    s3sw = c3["budget_sweep"]
    sw3, prev = [], None
    for r in s3sw:
        if r["set"] != prev:
            sw3.append({"budget": r["budget"], "set": r["set"]})
            prev = r["set"]
    w4 = comp_ref["chapter4_own_cluster_weights"]["pooled"]
    own = S1["own_cluster_weights"]
    R["comparison"] = {
        "priorities": cmpd,
        "criteria_weights": {"dso": S1["criteria_weights"], "chapter3": c3["criteria_weights"],
                             "kendall_tau_b": round(core.tau_b(cw, cw3), 3)},
        "portfolio_10_2": {"dso": port["X_B"], "chapter3": "{" + ", ".join(k for k in K if k in c3["portfolio"]["optimum"]) + "}",
                           "same_set": port["X_B"] == "{" + ", ".join(k for k in K if k in c3["portfolio"]["optimum"]) + "}",
                           "share_removed": {"dso": port["share_removed"], "chapter3": round(c3["portfolio"]["share_removed"], 4)},
                           "eal_total": {"dso": port["eal_total"], "chapter3": round(c3["portfolio"]["eal_total"], 3)}},
        "budget_switches": {"dso": switches, "chapter3": sw3},
        "cluster_weights": {"dso_own_4": own, "prior_v": core.V_PRIOR, "chapter4_own_3": w4,
                            "dso_own_chapter4_form": S1["own_cluster_weights_chapter4_form"]},
        "note": "Descriptive only (PROTOCOL.md, 8.9): the networks (generic threats against the DSO variants), the "
                "persona placement (developer message in chapter 3), the instructions (chapter 3 also asked pair by "
                "pair, but without case material, scope line, justifications or the request to judge each item on its "
                "merits without checking consistency, and with the coverage and loss items in a fixed order), the "
                "number of questionnaires (7 against 42) and the Codex build all differ."}
    log("8.9 comparison")

    # ------------------------------------------------------------------ 8.10 personas (exploratory)
    per = {}
    for pi_, p in enumerate(pool.personas):
        L = pool.by_persona[pi_]
        e_ = evaluate(pool, L[None, :], PRIMARY, fast=False)
        single = evaluate(pool, L[:, None], PRIMARY, fast=False)
        firsts = {skey(s): defaultdict(float) for s in PRIMARY}
        for s in PRIMARY:
            for g in single[skey(s)]:
                top = np.flatnonzero(np.abs(g - g.max()) <= core.TIE)
                for t in top:
                    firsts[skey(s)][K[t]] += 1 / len(top)
        per[p] = {"priorities": {skey(s): dict(zip(K, rnd(e_[skey(s)][0]))) for s in PRIMARY},
                  "first_places": {k: dict(v) for k, v in firsts.items()},
                  "own_cluster_weights": dict(zip(core.CL_ELEMENTS, rnd(e_["own_v"][0]))),
                  "portfolio_10_2": core.set_name(int(e_["set"][0])),
                  "sets_of_single_questionnaires": dict(zip(*np.unique([core.set_name(int(x)) for x in single["set"]],
                                                                        return_counts=True))),
                  "questionnaires": int(len(L))}
        per[p]["sets_of_single_questionnaires"] = {k: int(v) for k, v in per[p]["sets_of_single_questionnaires"].items()}
    R["personas"] = per
    log("8.10 personas")

    # ------------------------------------------------------------------ 8.11 sensitivity
    sens = {}
    rng = np.random.default_rng(SEEDS["size"])
    size = {}
    for k in (3, 7, 15, 21):
        picks = rng.integers(0, len(pool.personas), (4000, k))
        ix = np.stack([pool.by_persona[p][rng.integers(0, len(pool.by_persona[p]))] for p in picks.ravel()]).reshape(4000, k)
        e_ = evaluate(pool, ix, PRIMARY)
        size[k] = {skey(s): dict(zip(K, rnd(np.bincount(e_[skey(s)].argmax(1), minlength=7) / 4000, 3))) for s in PRIMARY}
        size[k]["share_X_B"] = round(float(np.mean(e_["set"] == X_B)), 3)
    sens["first_place_by_group_size_random_roles"] = size
    keep = np.array([i for i in range(pool.N) if max(float(core.cr(lay.matrix(pool.Y[i], m)))
                                                     for m in lay.mkeys if len(lay.els[m]) >= 3) <= 0.1])
    sens["CR_le_0.10"] = {"kept": int(len(keep))}
    if len(keep) >= 7 and all(np.isin(L, keep).any() for L in pool.by_persona):
        sub = pool.subset(keep)
        e_ = evaluate(sub, draw(sub, "S3", args.draws, np.random.default_rng(SEEDS["sens_cr"])), PRIMARY)
        for s in PRIMARY:
            pr = probs(e_[skey(s)], SEEDS["ties"])
            sens["CR_le_0.10"][skey(s)] = core.decision(pr["b"], pr["p"], K, 0.9)
        sens["CR_le_0.10"]["portfolio"] = core.portfolio_decision(e_["set"], X_B, 0.9)
    # chapter 3 assumed-noise Monte Carlo on the pooled judgments (secondary, Table 5.5 "вторинна модель")
    rng = np.random.default_rng(SEEDS["mc_ch3"])

    def crs_of(Yp: np.ndarray) -> np.ndarray:
        out_ = []
        for n, names in lay.by_size.items():
            if n >= 3:
                out_.append(core.cr(lay.matrices_of_size(Yp, n)).ravel())
        return np.concatenate(out_)
    lo_, hi_ = 0.0, 1.5
    for _ in range(16):                                   # bisection as anp_analysis.calibrate_sigma
        mid = (lo_ + hi_) / 2
        Yp = y1 + rng.normal(0.0, mid, (40, core.N_PAIRS))
        if np.median(crs_of(Yp)) < 0.075:
            lo_ = mid
        else:
            hi_ = mid
    sigma = (lo_ + hi_) / 2
    Yp = y1 + rng.normal(0.0, sigma, (args.trials, core.N_PAIRS))
    locp = lay.local(Yp, fast=True)
    mc = {"sigma": sigma, "median_CR_perturbed": float(np.median(crs_of(Yp))), "trials": args.trials}
    for s in PRIMARY:
        g = lay.priorities(locp, weights_of(lay, locp, Yp, s[0]), s[1], fast=True)["g"]
        pr = probs(g, SEEDS["ties"])
        mc[skey(s)] = core.decision(pr["b"], pr["p"], K, 0.9)
    sens["chapter3_assumed_noise_monte_carlo"] = mc
    # chapter 3 structure checks on S1 (review of 2.11.2026, optional 9): without the fixed synergy blocks, without any
    # synergy block; the chapter 4 weights with B = 0.048 also run through S3 above (setting ch4/source, 8.6)
    sens["S1_structure"] = {skey(s): {"label": LAB[s], "order": S1["settings"][skey(s)]["order"],
                                      "leader": S1["settings"][skey(s)]["leader"]} for s in S1_EXTRA + [("ch4", "source")]}
    # zero losses floored at the smallest positive loss instead of 1e-6 (review of 2.11.2026, optional 8)
    pos = pool.L[pool.L > 0]
    floor2 = float(pos.min()) if pos.size else core.LOSS_FLOOR
    zl = {"zero_losses": int((pool.L == 0).sum()), "floor": floor2}
    L2 = np.maximum(pool.L, floor2)
    _, eta2, eal2 = core.aggregate(pool.Y[:, :1], pool.E, L2, allidx)
    i2, v2 = core.best_sets(core.reductions(eta2, eal2))
    zl["S1_set"] = core.set_name(int(i2[0]))
    zl["S1_set_unchanged"] = int(i2[0]) == X_B
    sets2 = np.zeros(args.draws, dtype=int)
    for a in range(0, args.draws, 2000):
        _, e_, l_ = core.aggregate(pool.Y[:, :1], pool.E, L2, idx["S3"][a:a + 2000])
        sets2[a:a + 2000] = core.best_sets(core.reductions(e_, l_))[0]
    zl["S3_rule_5_1_tau_0.9"] = core.portfolio_decision(sets2, int(i2[0]), 0.9)
    sens["zero_loss_floor"] = zl
    R["sensitivity"] = sens
    log("8.11 sensitivity")

    # ------------------------------------------------------------------ 8.12 exploratory X1
    # tau-b on the unrounded values against the rank vectors of the two orders, so that equal losses or priorities
    # stay ties and are not broken in core.V order, which is the analysts' order (review of 2.11.2026, B4)
    ve = comp_ref["endtoend_variant_orders"]
    eal_v = eal1[0]
    lim_v = lay.priorities(loc1, lay.prior_weights(1), "source")["gv"][0]

    def score(order):                      # higher = earlier in the order
        return np.array([len(order) - order.index(v) for v in core.V], dtype=float)

    def tb(values, order):
        return round(core.tau_b(values, score(order)), 3)
    R["exploratory_X1"] = {"eal_by_variant": dict(zip(core.V, rnd(eal_v, 6))),
                           "limit_priority_by_variant_prior_v": dict(zip(core.V, rnd(lim_v, 8))),
                           "analysts_order": ve["analysts_revised"], "endtoend_S1_order": ve["virtual_experts_S1_ratings_mode"],
                           "tau_b": {"eal_vs_analysts": tb(eal_v, ve["analysts_revised"]),
                                     "eal_vs_endtoend": tb(eal_v, ve["virtual_experts_S1_ratings_mode"]),
                                     "limit_vs_analysts": tb(lim_v, ve["analysts_revised"]),
                                     "limit_vs_endtoend": tb(lim_v, ve["virtual_experts_S1_ratings_mode"])},
                           "ties": {"eal": int(len(eal_v) - len(np.unique(eal_v))),
                                    "limit": int(len(lim_v) - len(np.unique(np.round(lim_v, 12))))},
                           "label": "exploratory, not a test"}

    # ------------------------------------------------------------------ 8.13 cost
    # every attempt folder of the batch, including failed and interrupted questionnaires; each pilot apart (B3; pilot 1
    # and the repeated pilot 2 of the amendment of 2.11.2026)
    R["cost"] = {"main": cost_of(Path(args.batch))}
    pilots = ([Path(x) for x in args.pilot_batch] if args.pilot_batch
              else sorted(d for d in Path(args.batch).parent.glob("pilot*") if d.is_dir()))
    for pilot in pilots:
        if pilot.exists() and pilot.resolve() != Path(args.batch).resolve():
            R["cost"][pilot.name] = cost_of(pilot)
    R["versions"] = versions()
    R["run_time_s"] = round(time.time() - t_start, 1)
    (out_dir / "dso_controls_results.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    write_summary(R, out_dir / "summary.md")
    log(f"wrote {out_dir / 'dso_controls_results.json'} and summary.md ({pool.N} questionnaires)")


def write_summary(R: dict, path: Path) -> None:
    S1, port = R["S1"], R["S1_portfolio"]
    L = ["# DSO controls experiment: analysis summary", "",
         f"Batch: `{R['batch']}`, {R['n_questionnaires']} questionnaires, run {R['run_at']}. Settings: {R['settings']['draws']} "
         f"groups per scheme, bootstrap B = {R['settings']['B']} x {R['settings']['inner']}.", ""]
    fc = R.get("freeze_check") or {}
    L += [f"Freeze check: {'all match' if fc.get('all_match') else 'MISMATCH ' + json.dumps(fc.get('files'))}.", ""]
    L += ["## Pooled group (S1)", "", "| Setting | Order | Leader margin |", "|:--|:--|--:|"]
    for k, v in S1["settings"].items():
        order = ", ".join("%s %.3f" % (c, v["priorities"][c]) for c in v["order"])
        L.append(f"| {v['label']} | {order} | {v['margin_leader_second']:.3f} |")
    L += ["", f"Criteria weights: {S1['criteria_weights']}. Own cluster weights: {S1['own_cluster_weights']} "
          f"(prior v: {S1['prior_cluster_weights']}).", ""]
    L += ["## Checkpoint 2, priorities (primary: S3, tau = 0.9, union over the two weight sets)", ""]
    c = R["checkpoint2_priorities"]["S3/0.9"]
    L += [f"- Accepted in the same direction under both weight sets: "
          + (", ".join(f"{x['higher']} above {x['pair'].replace(x['higher'], '').strip('-')}" for x in c["accepted_same_direction"])
             or "none") + ".",
          f"- To human experts: {', '.join(c['to_human_experts']) or 'none'}.",
          f"- Accepted in opposite directions (cluster-weight question for the decision maker): "
          f"{', '.join(x['pair'] for x in c['accepted_in_opposite_directions']) or 'none'}.",
          f"- Unstable ranks: {c['unstable_ranks']}.", ""]
    for tau in TAUS:
        cc = R["checkpoint2_priorities"][f"S3/{tau}"]
        L.append(f"- tau = {tau}: {len(cc['to_human_experts'])} pairs to experts, "
                 f"{len(cc['accepted_same_direction'])} accepted, {len(cc['accepted_in_opposite_directions'])} opposite.")
    L += ["", "## Portfolio (budget 10.2 M USD) and rule (5.1)", "",
          f"S1 set {port['X_B']} (cost {port['cost']}), f = {port['f']:.3f} of {port['eal_total']:.3f} "
          f"({100 * port['share_removed']:.1f} %); greedy {port['greedy']}.", ""]
    for sch in ("S3", "S2"):
        d = R["checkpoint_portfolio"][f"{sch}/0.9"]
        L.append(f"- {sch}, tau = 0.9: q(X_B) = {d['q_X_B']:.3f} -> {'accepted' if d['set_accepted'] else 'not accepted'}; "
                 + ", ".join(f"{k} {v['q_c']:.3f} ({v['decision']})" for k, v in d["controls"].items()))
    L += ["", "## Variance shares (persona / wording / persona x wording / run)", "",
          "| Data | Block | Items | Persona | Wording | P x W | Run | n_eff |", "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for lab in ("log_judgments", "coverage_shares", "log_losses"):
        for b, row in (R["variance"].get(lab) or {}).get("blocks", {}).items():
            if row["shares"]:
                s = row["shares"]
                L.append(f"| {lab} | {b} | {row['items']} | {s['persona']:.3f} | {s['wording']:.3f} | "
                         f"{s['persona_x_wording']:.3f} | {s['run']:.3f} | {row['n_eff']:.1f} |")
    cmpd = R["comparison"]
    L += ["", "## Comparison with chapter 3 (descriptive)", "",
          f"- Priorities, prior v, method's reading: Kendall tau-b {cmpd['priorities']['prior/source']['kendall_tau_b']}, "
          f"leader DSO {cmpd['priorities']['prior/source']['leader_dso']}, chapter 3 {cmpd['priorities']['prior/source']['leader_chapter3']}.",
          f"- Portfolio at 10.2: DSO {cmpd['portfolio_10_2']['dso']}, chapter 3 {cmpd['portfolio_10_2']['chapter3']}.",
          f"- Budget switches DSO: {cmpd['budget_switches']['dso']}", ""]
    cm = R["cost"]["main"]
    L += [f"Cost (main batch, every attempt folder): {cm['calls']} calls, {cm['retries']} retries, replaced "
          f"{cm['replaced'] or 'none'}, median {cm['wall_median_s']} s per call, API equivalent "
          f"{cm['api_equivalent_usd_formula_5_6']} USD."
          + "".join(f" {k}: {v['calls']} calls." for k, v in R["cost"].items() if k != "main")
          + f" Run time of the analysis {R['run_time_s']} s.", ""]
    vd = R["variance"].get("design") or {}
    if vd.get("personas_left_out"):
        L += [f"Variance decomposition without {', '.join(vd['personas_left_out'])} (a cell with fewer than two "
              f"complete questionnaires, PROTOCOL.md 7.3).", ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
