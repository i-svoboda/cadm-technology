"""Analysis plan of PROTOCOL.md (Sections 6-8) on the stored questionnaires of one batch. Never calls a model.

Steps (numbers follow PROTOCOL.md, Section 7):
  7.1  consistency (CR) of every criteria and variant matrix, soft inconsistencies;
  7.2  pooled group S1: criteria weights (eigenvector of the geometric-mean matrix), ratings-mode and relative-mode scores;
  7.3  variance components (persona, wording, persona x wording, run) of log-judgments and ratings, with B-PW
       bootstrap intervals (adjusted mean squares, vc_adjust.py) and n_eff by (4.5); participation ratio (4.6);
  7.4  rank and pair probabilities under S1, S2, S3 (D simulated groups, both modes on the same groups);
  7.5  decision rule (4.8) at tau = 0.8 / 0.9 / 0.95, with B-Q bootstrap intervals (B-P for S3 as a check);
  7.6  comparison with the analysts (Kendall tau-b, rank of their leader, per-rating differences for all 15 ratings and
       the 12 never revised, weights, hybrids; primary measures A1 and A4 in the ratings mode);
  7.7  consistency between the ratings and relative modes, by presentation order (ratings-first, pairs-first);
  7.8  personas: group scores and first places by persona, personas on each side of every unstable pair;
  7.9  hand-over list for human experts and questions for the analysts; this list is the checkpoint 2 decision
       (R["checkpoint2"]), with each mode's own decision beside it;
Before the analysis, R["freeze_check"] compares PROTOCOL.md, analyse.py, core.py, vc_adjust.py and
analysts_reference.json with protocol/freeze.json and a warning is printed on any mismatch.
  7.10 sensitivity: arithmetic mean of ratings, CR > 0.10 excluded, no D, group size;
  7.11 cost: time, tokens, retries, quota readings.

Usage: python -B scripts/analyse.py --batch runs/main --out results/main [--B 2000 --inner 2000 --draws 10000]
       python -B scripts/analyse.py --batch selftest/SYNTHETIC_runs --out selftest/SYNTHETIC_results --quick
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tacs_vendor"))

import numpy as np  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

import core  # noqa: E402
import vc_adjust  # noqa: E402

ROOT = HERE.parent
REF_FILE = ROOT / "case" / "analysts_reference.json"
FREEZE = ROOT / "protocol" / "freeze.json"
TAUS = (0.8, 0.9, 0.95)
SEEDS = {"S1": 2026102701, "S2": 2026102702, "S3": 2026102703, "S3_arith": 2026102704, "vc_pw": 2026102705,
         "boot_bq_s2": 2026102706, "boot_bp": 2026102707, "size": 2026102708, "sens": 2026102709,
         "boot_bq_s3": 2026102710, "sens_cr": 2026102711}
NAME = re.compile(r"^E\d_W\d_r\d$")


# ============================================================================ data
class Pool:
    """Complete questionnaires of one batch as arrays (analysis order of variants and pairs, core.Layout)."""

    def __init__(self, batch: Path, case: dict):
        self.lay = core.Layout(case)
        qs = []
        for d in sorted(p for p in batch.iterdir() if p.is_dir() and NAME.match(p.name)):
            res = json.loads((d / "result.json").read_text(encoding="utf-8")) if (d / "result.json").exists() else {}
            if res.get("status") != "complete":
                continue
            items = json.loads((d / "items.json").read_text(encoding="utf-8"))
            layout = json.loads((d / "layout.json").read_text(encoding="utf-8"))
            answer = json.loads((d / "answer.json").read_text(encoding="utf-8"))
            arr = self.lay.arrays(items, answer)
            man_files = sorted(d.glob("attempt_*/manifest.json"))
            mans = [json.loads(m.read_text(encoding="utf-8")) for m in man_files]
            qs.append({"name": d.name, "persona": res["persona"], "wording": res["wording"], "run": res["replicate"],
                       "part_order": layout["part_order"], "seed": layout["seed"], "arr": arr, "manifests": mans,
                       "attempts": len(mans)})
        assert qs, f"no complete questionnaires in {batch}"
        self.qs = qs
        self.personas = sorted({q["persona"] for q in qs})
        self.wordings = sorted({q["wording"] for q in qs})
        self.runs = sorted({q["run"] for q in qs})
        self.Yc = np.array([q["arr"]["yc"] for q in qs])                 # N x 6
        self.Ys = np.array([q["arr"]["ys"] for q in qs])                 # N x 3 x 10
        self.R = np.array([q["arr"]["r"] for q in qs])                   # N x 3 x 5
        self.p_idx = np.array([self.personas.index(q["persona"]) for q in qs])
        self.by_persona = [np.where(self.p_idx == i)[0] for i in range(len(self.personas))]
        self.N = len(qs)

    def subset(self, keep: np.ndarray) -> "Pool":
        new = object.__new__(Pool)
        new.__dict__ = dict(self.__dict__)
        new.qs = [self.qs[i] for i in keep]
        new.Yc, new.Ys, new.R, new.p_idx = self.Yc[keep], self.Ys[keep], self.R[keep], self.p_idx[keep]
        new.personas = self.personas
        new.by_persona = [np.where(new.p_idx == i)[0] for i in range(len(self.personas))]
        new.N = len(keep)
        return new

    def cells(self, values: np.ndarray) -> np.ndarray:
        """values (N, K) -> (P, W, R, K) for the balanced design; R = smallest complete cell (smaa_ahp rule)."""
        cell = defaultdict(list)
        for i, q in enumerate(self.qs):
            cell[(q["persona"], q["wording"])].append((q["run"], i))
        r = min(len(v) for v in cell.values())
        assert len(cell) == len(self.personas) * len(self.wordings), "a persona x wording cell is empty"
        Y = np.zeros((len(self.personas), len(self.wordings), r, values.shape[1]))
        for (p, w), lst in cell.items():
            lst.sort()
            for k, (_, i) in enumerate(lst[:r]):
                Y[self.personas.index(p), self.wordings.index(w), k] = values[i]
        return Y


def group_scores(pool: Pool, idx: np.ndarray, how: str = "geo", no_d: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """idx (D, m) member indices -> criteria weights (D, 4), ratings-mode and relative-mode scores (D, 5)."""
    lay = pool.lay
    yc = pool.Yc[idx].mean(axis=1)
    ys = pool.Ys[idx].mean(axis=1)
    r = core.agg_ratings(pool.R[idx], axis=1, how=how)
    w = lay.weights(yc)
    if no_d:
        w = w.copy()
        w[:, lay.ci["D"]] = 0
        w /= w.sum(axis=1, keepdims=True)
    return w, lay.scores_ratings(w, r), lay.scores_relative(w, lay.local(ys))


def draw(pool: Pool, scheme: str, D: int, rng: np.random.Generator, persona_lists: list[np.ndarray] | None = None) -> np.ndarray:
    """Member indices (D, m) of D simulated groups. S1: all questionnaires; S2: one per persona (fixed roles);
    S3: P personas with replacement, one questionnaire each (random roles)."""
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


def smaa(pool: Pool, scheme: str, D: int, seed: int, how: str = "geo", no_d: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    idx = draw(pool, scheme, D, rng)
    w, g_r, g_p = group_scores(pool, idx, how, no_d)
    s_r, s_p = core.summarise(g_r, rng), core.summarise(g_p, rng)
    lay = pool.lay
    iu, ju = np.triu_indices(5, 1)
    a = np.sign(s_r["ranks"][:, iu] - s_r["ranks"][:, ju])
    b = np.sign(s_p["ranks"][:, iu] - s_p["ranks"][:, ju])
    tau_modes = (a * b).mean(axis=1)
    out = {"draws": int(idx.shape[0]), "group_size": int(idx.shape[1]),
           "weights_mean": dict(zip(core.CRITERIA, w.mean(0).round(4).tolist())),
           "weights_sd": dict(zip(core.CRITERIA, w.std(0).round(4).tolist())),
           "modes_same_leader": float((s_r["ranks"].argmin(1) == s_p["ranks"].argmin(1)).mean()),
           "modes_same_order": float((s_r["ranks"] == s_p["ranks"]).all(1).mean()),
           "modes_tau_median": float(np.median(tau_modes))}
    for mode, g, s in (("ratings", g_r, s_r), ("relative", g_p, s_p)):
        out[mode] = {"b": s["b"].round(4).tolist(), "p": s["p"].round(4).tolist(), "tie_share": s["tie_share"],
                     "mean_score": g.mean(0).round(4).tolist(), "sd_score": g.std(0).round(4).tolist(),
                     "_ranks": s["ranks"]}
    return out


# ============================================================================ 7.3 variance components
BLOCKS_LOG = {"criteria": list(range(0, 6)), "pairs_L": list(range(6, 16)), "pairs_F": list(range(16, 26)),
              "pairs_C": list(range(26, 36)), "pairs_all": list(range(6, 36))}
BLOCKS_RAT = {"ratings_L": list(range(0, 5)), "ratings_F": list(range(5, 10)), "ratings_C": list(range(10, 15)),
              "ratings_all": list(range(15))}


def variance(pool: Pool, B: int) -> dict:
    P, W = len(pool.personas), len(pool.wordings)
    logj = np.concatenate([pool.Yc, pool.Ys.reshape(pool.N, -1)], axis=1)        # N x 36
    rat = pool.R.reshape(pool.N, -1)                                              # N x 15 (L, F, C blocks)
    out = {}
    for label, vals, blocks in (("log_judgments", logj, BLOCKS_LOG), ("ratings_points", rat, BLOCKS_RAT),
                                ("log_ratings", np.log(rat), BLOCKS_RAT)):
        Y = pool.cells(vals)
        R = Y.shape[2]
        ms = vc_adjust.mean_squares(Y)
        comp = vc_adjust.components(ms, P, W, R)
        point = vc_adjust.block_shares(comp, blocks)
        res = {"runs_per_cell": R, "blocks": {}}
        boot = vc_adjust.bootstrap(Y, blocks, B, SEEDS["vc_pw"], "B-PW") if B else None
        for bname, cols in blocks.items():
            sh = point[bname]
            tot = comp[:, cols].mean(axis=1).sum()
            row = {"items": len(cols), "total_variance": float(tot),
                   "shares": dict(zip(vc_adjust.COMPS, np.round(sh, 4).tolist())) if tot > 0 else None,
                   "wording_plus_interaction": float(sh[1] + sh[2]) if tot > 0 else None,
                   "n_eff": float(vc_adjust.n_eff(sh, N=P * W * R, m=W * R, L=W, R=R)) if tot > 0 else None}
            if boot is not None and tot > 0:
                bs = boot["adj"][bname]["shares"]
                ne = boot["adj"][bname]["n_eff"]
                ok = np.isfinite(bs).all(axis=1)
                row["ci95"] = {c: np.quantile(bs[ok, k], [0.025, 0.975]).round(4).tolist() for k, c in enumerate(vc_adjust.COMPS)}
                row["ci95"]["wording_plus_interaction"] = np.quantile(bs[ok, 1] + bs[ok, 2], [0.025, 0.975]).round(4).tolist()
                row["ci95"]["n_eff"] = np.quantile(ne[ok], [0.025, 0.975]).round(2).tolist()
                row["persona_largest_share_of_boot"] = float(np.mean(bs[ok, 0] >= bs[ok, 1:].max(axis=1)))
            res["blocks"][bname] = row
        out[label] = res
    # cross-check with the TACS code (vendor smaa_ahp.variance_components) for the log-judgment blocks
    try:
        import smaa_ahp
        qs = []
        for i, q in enumerate(pool.qs):
            lj = {("criteria|GOAL", *pool.lay.crit_pairs[k]): float(pool.Yc[i, k]) for k in range(6)}
            for c in range(3):
                for k, pr in enumerate(pool.lay.var_pairs):
                    lj[("alternatives|" + core.RATED[c], *pr)] = float(pool.Ys[i, c, k])
            qs.append({"meta": {"persona": q["persona"], "wording": q["wording"]}, "logj": lj})
        vend = smaa_ahp.variance_components(qs)
        out["vendor_crosscheck"] = {k: {"share": v["share"], "items": v["items"]} for k, v in vend.items()}
    except Exception as exc:  # the cross-check is a diagnostic only
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


# ============================================================================ 7.5 bootstrap of rank and pair probabilities
def boot_probs(pool: Pool, scheme: str, resample: str, B: int, inner: int, seed: int) -> dict:
    """Percentile intervals of b and p. B-Q: questionnaires resampled within each persona (roles fixed);
    B-P: personas resampled with replacement, each keeping its questionnaires."""
    rng = np.random.default_rng(seed)
    n = len(pool.lay.variants)
    store = {m: {"b": np.zeros((B, n, n)), "p": np.zeros((B, n, n))} for m in ("ratings", "relative")}
    for t in range(B):
        if resample == "B-Q":
            lists = [L[rng.integers(0, len(L), len(L))] for L in pool.by_persona]
        elif resample == "B-P":
            lists = [pool.by_persona[i] for i in rng.integers(0, len(pool.by_persona), len(pool.by_persona))]
        else:
            raise ValueError(resample)
        idx = draw(pool, scheme, inner, rng, lists)
        _, g_r, g_p = group_scores(pool, idx)
        for mode, g in (("ratings", g_r), ("relative", g_p)):
            s = core.summarise(g, rng)
            store[mode]["b"][t], store[mode]["p"][t] = s["b"], s["p"]
    out = {}
    for mode, st in store.items():
        out[mode] = {"b_ci95": np.quantile(st["b"], [0.025, 0.975], axis=0).round(4).tolist(),
                     "p_ci95": np.quantile(st["p"], [0.025, 0.975], axis=0).round(4).tolist(),
                     "_p": st["p"], "_b": st["b"]}
    out["B"], out["inner"], out["resample"] = B, inner, resample
    return out


def add_boot_to_decision(dec: dict, bt: dict | None, mode: str, labels: list[str], point_unstable: set) -> dict:
    if not bt:
        return dec
    lo, hi = np.array(bt[mode]["p_ci95"])
    tau = dec["tau"]
    for row in dec["accepted_pairs"] + dec["unstable_pairs"]:
        i, j = (labels.index(x) for x in row["pair"].split("-"))
        a, b = sorted([lo[i, j], hi[i, j]])
        q_lo, q_hi = (a, b) if row["higher"] == labels[i] else (1 - b, 1 - a)
        row["ci95_of_p"] = [round(float(q_lo), 4), round(float(q_hi), 4)]
        row["near_threshold"] = bool(q_lo <= tau <= q_hi or q_lo <= 1 - tau <= q_hi)
    P = bt[mode]["_p"]
    q = np.maximum(P, 1 - P)
    iu, ju = np.triu_indices(len(labels), 1)
    sets = [frozenset(f"{labels[i]}-{labels[j]}" for i, j in zip(iu, ju) if q[t, i, j] < tau) for t in range(P.shape[0])]
    dec["boot_share_same_unstable_set"] = float(np.mean([s == point_unstable for s in sets]))
    dec["boot_share_no_unstable_pair"] = float(np.mean([len(s) == 0 for s in sets]))
    return dec


# ============================================================================ freeze check
def freeze_check() -> dict:
    """Match status of the protocol and the analysis code against protocol/freeze.json (review 26.10.2026, change 8)."""
    out = {"freeze_file": str(FREEZE), "freeze_sha256": None, "frozen": None, "files": {}, "all_match": False}
    if not FREEZE.exists():
        out["warning"] = "protocol/freeze.json is missing"
        return out
    fr = json.loads(FREEZE.read_text(encoding="utf-8"))
    out["freeze_sha256"] = hashlib.sha256(FREEZE.read_bytes()).hexdigest()
    out["frozen"] = fr.get("frozen")
    want = {"PROTOCOL.md": (fr.get("other_files") or {}).get("PROTOCOL.md"),
            "scripts/analyse.py": (fr.get("other_files") or {}).get("scripts/analyse.py"),
            "scripts/core.py": (fr.get("inputs") or {}).get("core.py"),
            "scripts/vc_adjust.py": (fr.get("other_files") or {}).get("scripts/vc_adjust.py"),
            "case/analysts_reference.json": (fr.get("other_files") or {}).get("case/analysts_reference.json")}
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
    ap.add_argument("--case", default=str(core.CASE_FILE))
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--B", type=int, default=2000, help="bootstrap replicates (variance shares and probabilities)")
    ap.add_argument("--inner", type=int, default=2000, help="simulated groups per bootstrap replicate")
    ap.add_argument("--quick", action="store_true", help="small numbers for tests: draws 2000, B 40, inner 300")
    args = ap.parse_args()
    if args.quick:
        args.draws, args.B, args.inner = 2000, 40, 300
    case = core.load_case(Path(args.case))
    ref = json.loads(REF_FILE.read_text(encoding="utf-8"))
    pool = Pool(Path(args.batch), case)
    lay = pool.lay
    V = lay.variants
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    R: dict = {"batch": str(Path(args.batch).resolve()), "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "n_questionnaires": pool.N, "personas": pool.personas, "wordings": pool.wordings, "variants": V,
               "settings": {"draws": args.draws, "B": args.B, "inner": args.inner, "seeds": SEEDS}}
    bs = Path(args.batch) / "batch_sha256.json"
    R["batch_sha256_file"] = hashlib.sha256(bs.read_bytes()).hexdigest() if bs.exists() else None
    R["freeze_check"] = freeze_check()
    if not R["freeze_check"]["all_match"]:
        print(f"WARNING: protocol or analysis code differs from protocol/freeze.json: {R['freeze_check']['files']}"
              + (f" ({R['freeze_check']['warning']})" if "warning" in R["freeze_check"] else ""), flush=True)

    # 7.1 consistency
    crc = core.cr(lay.crit_matrices(pool.Yc))
    crv = core.cr(lay.var_matrices(pool.Ys)).ravel()
    soft = defaultdict(int)
    for q in pool.qs:
        for k, v in q["arr"]["soft"].items():
            soft[k] += v
    R["consistency"] = {
        "criteria": {"matrices": len(crc), "median": float(np.median(crc)), "p90": float(np.quantile(crc, .9)),
                     "max": float(crc.max()), "share_over_0.10": float(np.mean(crc > 0.1))},
        "variants": {"matrices": len(crv), "median": float(np.median(crv)), "p90": float(np.quantile(crv, .9)),
                     "max": float(crv.max()), "share_over_0.10": float(np.mean(crv > 0.1))},
        "soft_inconsistencies": dict(soft)}

    # 7.2 pooled group S1
    allidx = np.arange(pool.N)[None, :]
    w1, gr1, gp1 = group_scores(pool, allidx)
    r_geo = core.agg_ratings(pool.R, axis=0)
    yc_mean = pool.Yc.mean(0)
    R["S1"] = {"weights": dict(zip(core.CRITERIA, w1[0].round(4).tolist())),
               "CR_criteria": float(core.cr(lay.crit_matrices(yc_mean))),
               "ratings_geo": {c: dict(zip(V, r_geo[k].round(3).tolist())) for k, c in enumerate(core.RATED)},
               "ratings_arith": {c: dict(zip(V, pool.R.mean(0)[k].round(3).tolist())) for k, c in enumerate(core.RATED)},
               "scores_ratings": dict(zip(V, gr1[0].round(4).tolist())),
               "scores_relative": dict(zip(V, gp1[0].round(4).tolist())),
               "order_ratings": [V[i] for i in np.argsort(-gr1[0], kind="stable")],
               "order_relative": [V[i] for i in np.argsort(-gp1[0], kind="stable")],
               "local_priorities_relative": {c: dict(zip(V, lay.local(pool.Ys.mean(0))[k].round(4).tolist()))
                                             for k, c in enumerate(core.RATED)}}

    # 7.3 variance components, n_eff, participation ratio
    R["variance"] = variance(pool, args.B)
    R["participation_ratio"] = {"log_judgments": participation_ratio(pool, np.concatenate([pool.Yc, pool.Ys.reshape(pool.N, -1)], 1)),
                                "ratings": participation_ratio(pool, pool.R.reshape(pool.N, -1))}

    # 7.4 SMAA
    sm = {s: smaa(pool, s, 1 if s == "S1" else args.draws, SEEDS[s]) for s in ("S1", "S2", "S3")}
    ref_rank = np.argsort(np.argsort([-ref["scores_revised"][v] for v in V]))
    lead = V[int(np.argmin(ref_rank))]
    for s, res in sm.items():
        for mode in ("ratings", "relative"):
            rk = res[mode]["_ranks"]
            t = core.tau_b_to_reference(rk, ref_rank)
            res[mode]["tau_to_analysts"] = {"median": float(np.median(t)), "q025": float(np.quantile(t, .025)),
                                            "q975": float(np.quantile(t, .975)), "mean": float(t.mean())}
            res[mode]["P_analysts_full_order"] = float((rk == ref_rank).all(1).mean())
            res[mode]["P_analysts_leader_first"] = float(res[mode]["b"][V.index(lead)][0])
    R["smaa"] = {s: {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v)
                     for k, v in res.items()} for s, res in sm.items()}

    # 7.5 decision rule with bootstrap intervals
    boots = {}
    if args.B:
        boots["S2"] = boot_probs(pool, "S2", "B-Q", args.B, args.inner, SEEDS["boot_bq_s2"])
        boots["S3"] = boot_probs(pool, "S3", "B-Q", args.B, args.inner, SEEDS["boot_bq_s3"])
        boots["S3_BP"] = boot_probs(pool, "S3", "B-P", args.B, args.inner, SEEDS["boot_bp"])
    R["decision"] = {}
    for s in ("S2", "S3"):
        for mode in ("ratings", "relative"):
            for tau in TAUS:
                dec = core.decision(np.array(sm[s][mode]["b"]), np.array(sm[s][mode]["p"]), V, tau)
                point_unst = frozenset(x["pair"] for x in dec["unstable_pairs"])
                dec = add_boot_to_decision(dec, boots.get(s), mode, V, point_unst)
                if s == "S3" and "S3_BP" in boots:
                    dec["check_B-P"] = add_boot_to_decision(core.decision(np.array(sm[s][mode]["b"]), np.array(sm[s][mode]["p"]), V, tau),
                                                            boots["S3_BP"], mode, V, point_unst).get("boot_share_same_unstable_set")
                R["decision"][f"{s}/{mode}/{tau}"] = dec
    R["bootstrap_intervals"] = {k: {m: {kk: vv for kk, vv in v[m].items() if not kk.startswith("_")}
                                    for m in ("ratings", "relative")} | {"B": v["B"], "inner": v["inner"], "resample": v["resample"]}
                                for k, v in boots.items()}

    # 7.6 comparison with the analysts
    ar = ref["ratings_revised"]
    ao = ref["ratings_original"]
    aw = np.array([ref["weights"][c] for c in core.CRITERIA])
    comp = {"analysts_order": [V[i] for i in np.argsort(ref_rank)], "analysts_leader": lead}
    for mode, g in (("ratings", gr1[0]), ("relative", gp1[0])):
        rk = np.argsort(np.argsort(-g))
        comp[f"S1_{mode}"] = {"kendall_tau_b": float(kendalltau(rk, ref_rank)[0]),
                              "rank_of_analysts_leader": int(rk[V.index(lead)] + 1),
                              "leader": V[int(np.argmin(rk))]}
        for s in ("S2", "S3"):
            comp[f"{s}_{mode}"] = {"b_leader_first": sm[s][mode]["P_analysts_leader_first"],
                                   "b_of_analysts_leader": sm[s][mode]["b"][V.index(lead)],
                                   "P_full_order": sm[s][mode]["P_analysts_full_order"],
                                   "tau_b": sm[s][mode]["tau_to_analysts"]}
    rows = []
    for k, c in enumerate(core.RATED):
        for v in V:
            x = pool.R[:, k, V.index(v)]
            a = ar[v][c]
            rows.append({"variant": v, "criterion": c, "analysts_revised": a, "analysts_original": ao[v][c],
                         "revised": bool(a != ao[v][c]),
                         "virtual_geo_mean": round(float(np.exp(np.log(x).mean())), 3), "virtual_median": float(np.median(x)),
                         "virtual_q05_q95": np.quantile(x, [0.05, 0.95]).tolist(),
                         "diff_geo_minus_revised": round(float(np.exp(np.log(x).mean()) - a), 3),
                         "share_within_1_point": float(np.mean(np.abs(x - a) <= 1)),
                         "percentile_of_analysts": float(np.mean(x < a) + 0.5 * np.mean(x == a)),
                         "outside_q05_q95": bool(a < np.quantile(x, 0.05) or a > np.quantile(x, 0.95))})
    comp["ratings"] = rows
    # A5 for all 15 ratings and for the 12 never revised; the three revised ratings (RS F, MP F, EE L) were set to the
    # anchors that the questionnaire shows, so they are listed apart (protocol review of 26.10.2026, change 5)
    assert {(r["variant"], r["criterion"]) for r in rows if r["revised"]} == {("RS", "F"), ("MP", "F"), ("EE", "L")}

    def a5_summary(sub: list[dict]) -> dict:
        d_rev = np.array([r["virtual_geo_mean"] - r["analysts_revised"] for r in sub])
        d_org = np.array([r["virtual_geo_mean"] - r["analysts_original"] for r in sub])
        return {"ratings": len(sub), "MAD_to_revised": float(np.abs(d_rev).mean()), "bias_to_revised": float(d_rev.mean()),
                "MAD_to_original": float(np.abs(d_org).mean()), "bias_to_original": float(d_org.mean()),
                "mean_share_within_1_point": float(np.mean([r["share_within_1_point"] for r in sub])),
                "outside_q05_q95": [f"{r['variant']} {r['criterion']}" for r in sub if r["outside_q05_q95"]]}
    comp["ratings_summary"] = {**a5_summary(rows), "unrevised_12": a5_summary([r for r in rows if not r["revised"]]),
                               "revised_3": [r for r in rows if r["revised"]]}
    comp["weights"] = {"virtual_S1": R["S1"]["weights"], "analysts": ref["weights"],
                       "difference": dict(zip(core.CRITERIA, (w1[0] - aw).round(4).tolist())),
                       "virtual_order": [core.CRITERIA[i] for i in np.argsort(-w1[0])],
                       "analysts_order": [core.CRITERIA[i] for i in np.argsort(-aw)]}
    rng = np.random.default_rng(SEEDS["S3"])
    idx = draw(pool, "S3", args.draws, rng)
    w3 = lay.weights(pool.Yc[idx].mean(1))
    comp["weights"]["S3_P_largest"] = dict(zip(core.CRITERIA, np.bincount(w3.argmax(1), minlength=4).astype(float) / len(w3)))
    rv = np.array([[ar[v][c] for v in V] for c in core.RATED], dtype=float)
    hyb = {"virtual_weights_analysts_ratings": lay.scores_ratings(w1[0], rv),
           "analysts_weights_virtual_ratings": lay.scores_ratings(aw, r_geo),
           "analysts_weights_virtual_relative": lay.scores_relative(aw, lay.local(pool.Ys.mean(0)))}
    comp["hybrids"] = {k: {"order": [V[i] for i in np.argsort(-g, kind="stable")],
                           "kendall_tau_b": float(kendalltau(np.argsort(np.argsort(-g)), ref_rank)[0]),
                           "scores": dict(zip(V, np.round(g, 4).tolist()))} for k, g in hyb.items()}
    mp_l = pool.R[:, core.RATED.index("L"), V.index("MP")]
    comp["contested_MP_likelihood"] = {"analysts": ar["MP"]["L"], "distribution": {int(k): int((mp_l == k).sum()) for k in range(1, 10)},
                                       "share_ge_7": float(np.mean(mp_l >= 7)), "geo_mean": float(np.exp(np.log(mp_l).mean()))}
    # primary agreement measures fixed before the batch (PROTOCOL.md, 7.6; review of 26.10.2026, optional change 14)
    comp["primary_measures"] = {"A1_S1_ratings_mode_kendall_tau_b": comp["S1_ratings"]["kendall_tau_b"],
                                "A4_S3_ratings_mode_tau_b": comp["S3_ratings"]["tau_b"]}
    R["analysts_comparison"] = comp

    # 7.7 mode consistency per questionnaire
    mc = []
    for i, q in enumerate(pool.qs):
        pi = lay.local(pool.Ys[i])
        taus = []
        for k in range(3):
            t = kendalltau(pool.R[i, k], pi[k])[0]
            taus.append(float(t) if np.isfinite(t) else None)
        conf = agree = ties = 0
        xs, ysv = [], []
        for k in range(3):
            for m, (a, b) in enumerate(lay.var_pairs):
                ra, rb = pool.R[i, k, lay.vi[a]], pool.R[i, k, lay.vi[b]]
                sj, sr = np.sign(pool.Ys[i, k, m]), np.sign(ra - rb)
                if sj == 0 or sr == 0:
                    ties += 1
                elif sj == sr:
                    agree += 1
                else:
                    conf += 1
                xs.append(np.log(ra / rb))
                ysv.append(pool.Ys[i, k, m])
        corr = float(np.corrcoef(xs, ysv)[0, 1]) if np.std(xs) > 0 and np.std(ysv) > 0 else None
        mc.append({"name": q["name"], "part_order": q["part_order"], "tau_b_by_criterion": dict(zip(core.RATED, taus)),
                   "direction_agree": agree, "direction_conflict": conf, "tie_in_one_mode": ties, "corr_log": corr})
    def mc_sum(sub):
        if not sub:
            return None
        tb = [t for m in sub for t in m["tau_b_by_criterion"].values() if t is not None]
        return {"questionnaires": len(sub), "median_tau_b": float(np.median(tb)) if tb else None,
                "share_tau_b_1": float(np.mean(np.array(tb) == 1)) if tb else None,
                "conflict_share_of_strict_pairs": float(sum(m["direction_conflict"] for m in sub) /
                                                        max(1, sum(m["direction_conflict"] + m["direction_agree"] for m in sub))),
                "median_corr_log": float(np.median([m["corr_log"] for m in sub if m["corr_log"] is not None]))}
    # presentation order only: the answer object is written in schema order in every questionnaire (PROTOCOL.md, 5)
    R["mode_consistency"] = {"all": mc_sum(mc), "ratings_first": mc_sum([m for m in mc if m["part_order"] == "ratings_first"]),
                             "pairs_first": mc_sum([m for m in mc if m["part_order"] == "pairs_first"]),
                             "group_S1_tau_b": float(kendalltau(np.argsort(np.argsort(-gr1[0])), np.argsort(np.argsort(-gp1[0])))[0]),
                             "S3_same_leader": sm["S3"]["modes_same_leader"], "S3_same_order": sm["S3"]["modes_same_order"],
                             "per_questionnaire": mc}

    # 7.8 personas
    per = {}
    for pi_, p in enumerate(pool.personas):
        L = pool.by_persona[pi_]
        _, g_r, g_p = group_scores(pool, L[None, :])
        firsts = {"ratings": defaultdict(float), "relative": defaultdict(float)}
        for i in L:
            _, a, b = group_scores(pool, np.array([[i]]))
            for mode, g in (("ratings", a[0]), ("relative", b[0])):
                top = np.flatnonzero(np.abs(g - g.max()) <= core.TIE)
                for t in top:
                    firsts[mode][V[t]] += 1 / len(top)
        per[p] = {"scores_ratings": dict(zip(V, g_r[0].round(4).tolist())), "scores_relative": dict(zip(V, g_p[0].round(4).tolist())),
                  "first_places": {m: dict(f) for m, f in firsts.items()}, "questionnaires": int(len(L))}
    R["personas"] = per

    # 7.9 hand-over list (S3, tau = 0.9, union of the two modes) and questions for the analysts. Every pair carries its
    # decision row (p, side, bootstrap interval) in both modes.
    rows_by_mode = {m: {r["pair"]: r for r in R["decision"][f"S3/{m}/0.9"]["accepted_pairs"] + R["decision"][f"S3/{m}/0.9"]["unstable_pairs"]}
                    for m in ("ratings", "relative")}
    unst = {m: {r["pair"] for r in R["decision"][f"S3/{m}/0.9"]["unstable_pairs"]} for m in ("ratings", "relative")}
    acc = {m: {r["pair"]: r["higher"] for r in R["decision"][f"S3/{m}/0.9"]["accepted_pairs"]} for m in ("ratings", "relative")}
    conflict = {pr for pr in set(acc["ratings"]) & set(acc["relative"]) if acc["ratings"][pr] != acc["relative"][pr]}   # M5
    hand = {}
    for pair in sorted(unst["ratings"] | unst["relative"] | conflict):
        a, b = pair.split("-")
        row = {"pair": pair, "unstable_in": [m for m in ("ratings", "relative") if pair in unst[m]],
               "mode_conflict": pair in conflict, "ratings": rows_by_mode["ratings"][pair],
               "relative": rows_by_mode["relative"][pair]}
        sides = {"ratings": {a: [], b: []}, "relative": {a: [], b: []}}
        for p, d in per.items():
            for mode, key in (("ratings", "scores_ratings"), ("relative", "scores_relative")):
                sides[mode][a if d[key][a] >= d[key][b] else b].append(p)
        row["personas_preferring"] = sides
        row["closeness"] = min(abs(row[m]["p"] - 0.5) for m in ("ratings", "relative"))
        hand[pair] = row
    unstable_ranks = {m: [r for r in range(1, 6) if r not in {x["rank"] for x in R["decision"][f"S3/{m}/0.9"]["stable_ranks"]}]
                      for m in ("ratings", "relative")}
    R["hand_over"] = {"rule": "S3, tau = 0.9, union of unstable pairs of both modes plus pairs accepted in opposite directions",
                      "pairs": sorted(hand.values(), key=lambda x: x["closeness"]),
                      "unstable_ranks": unstable_ranks,
                      "questions_for_analysts": [r for r in rows if r["outside_q05_q95"]]}
    # Checkpoint 2 decision reported in chapter 5 (PROTOCOL.md, 7.5, fixed on 26.10.2026): the hand-over list of 7.9.
    # A pair is accepted only when both modes accept it in the same direction; each mode's own decision is kept beside it.
    both = sorted(pr for pr in set(acc["ratings"]) & set(acc["relative"]) if pr not in conflict)
    R["checkpoint2"] = {
        "rule": "rule (4.8), S3, tau = 0.9, applied in both modes; union of unstable pairs and M5 pairs go to human experts",
        "accepted_pairs": [{"pair": pr, "higher": acc["ratings"][pr], "p_ratings": rows_by_mode["ratings"][pr]["p"],
                            "p_relative": rows_by_mode["relative"][pr]["p"]} for pr in both],
        "pairs_to_human_experts": [x["pair"] for x in R["hand_over"]["pairs"]],
        "unstable_ranks": unstable_ranks,
        "ratings_mode": {"accepted": sorted(acc["ratings"]), "unstable": sorted(unst["ratings"]),
                         "stable_ranks": R["decision"]["S3/ratings/0.9"]["stable_ranks"]},
        "relative_mode": {"accepted": sorted(acc["relative"]), "unstable": sorted(unst["relative"]),
                          "stable_ranks": R["decision"]["S3/relative/0.9"]["stable_ranks"]}}

    # 7.10 sensitivity
    sens = {}
    s3a = smaa(pool, "S3", args.draws, SEEDS["S3_arith"], how="arith")
    sens["S3_arith_ratings"] = core.decision(np.array(s3a["ratings"]["b"]), np.array(s3a["ratings"]["p"]), V, 0.9)
    s3n = smaa(pool, "S3", args.draws, SEEDS["sens"], no_d=True)
    sens["S3_no_D"] = {m: core.decision(np.array(s3n[m]["b"]), np.array(s3n[m]["p"]), V, 0.9) for m in ("ratings", "relative")}
    keep = np.flatnonzero((crc <= 0.1) & (core.cr(lay.var_matrices(pool.Ys)).max(1) <= 0.1))
    sens["CR_le_0.10"] = {"kept": int(len(keep))}
    if len(keep) >= 7 and all(np.isin(L, keep).any() for L in pool.by_persona):
        sub = pool.subset(keep)
        s3c = smaa(sub, "S3", args.draws, SEEDS["sens_cr"])
        sens["CR_le_0.10"].update({m: core.decision(np.array(s3c[m]["b"]), np.array(s3c[m]["p"]), V, 0.9) for m in ("ratings", "relative")})
    size = {}
    rng = np.random.default_rng(SEEDS["size"])
    for k in (3, 7, 15, 21):
        picks = rng.integers(0, len(pool.personas), (4000, k))
        idx = np.stack([pool.by_persona[p][rng.integers(0, len(pool.by_persona[p]))] for p in picks.ravel()]).reshape(4000, k)
        _, g_r, g_p = group_scores(pool, idx)
        size[k] = {m: dict(zip(V, (np.bincount(g.argmax(1), minlength=5) / 4000).round(3).tolist())) for m, g in (("ratings", g_r), ("relative", g_p))}
    sens["first_place_by_group_size_random_roles"] = size
    R["sensitivity"] = sens

    # 7.11 cost
    walls, toks, out_t, prim = [], [], [], []
    for q in pool.qs:
        for m in q["manifests"]:
            walls.append(m.get("wall_seconds") or 0)
            u = (m.get("usage") or [{}])[-1] or {}
            toks.append(u.get("input_tokens") or 0)
            out_t.append(u.get("output_tokens") or 0)
            pr = ((m.get("rate_limits_after") or {}).get("primary") or {}).get("used_percent")
            if pr is not None:
                prim.append((m.get("started_utc"), pr))
    R["cost"] = {"calls": len(walls), "retries": sum(q["attempts"] - 1 for q in pool.qs),
                 "wall_median_s": float(np.median(walls)) if walls else None, "wall_total_min": float(sum(walls) / 60),
                 "input_tokens_median": float(np.median(toks)) if toks else None,
                 "output_tokens_median": float(np.median(out_t)) if out_t else None,
                 "primary_readings_first_last": [sorted(prim)[0], sorted(prim)[-1]] if prim else None}

    (out_dir / "endtoend_results.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    write_summary(R, out_dir / "summary.md")
    print(f"wrote {out_dir / 'endtoend_results.json'} and summary.md ({pool.N} questionnaires)")


def write_summary(R: dict, path: Path) -> None:
    V = R["variants"]
    L = [f"# End-to-end experiment: analysis summary", "", f"Batch: `{R['batch']}`, {R['n_questionnaires']} questionnaires, "
         f"run {R['run_at']}. Settings: {R['settings']['draws']} groups per scheme, bootstrap B = {R['settings']['B']} "
         f"x {R['settings']['inner']}.", ""]
    L += ["## Pooled group (S1)", "", "| Criterion | Weight (virtual) | Weight (analysts) |", "|:--|--:|--:|"]
    for c in core.CRITERIA:
        L.append(f"| {c} | {R['S1']['weights'][c]:.3f} | {R['analysts_comparison']['weights']['analysts'][c]:.3f} |")
    L += ["", f"Order, ratings mode: {', '.join(R['S1']['order_ratings'])}. Relative mode: {', '.join(R['S1']['order_relative'])}. "
          f"Analysts: {', '.join(R['analysts_comparison']['analysts_order'])}.", ""]
    L += ["## Variance shares (persona / wording / persona x wording / run)", "", "| Data | Block | Items | Persona | Wording | P x W | Run | n_eff |", "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for lab in ("log_judgments", "ratings_points"):
        for b, row in R["variance"][lab]["blocks"].items():
            if row["shares"]:
                s = row["shares"]
                L.append(f"| {lab} | {b} | {row['items']} | {s['persona']:.3f} | {s['wording']:.3f} | {s['persona_x_wording']:.3f} | "
                         f"{s['run']:.3f} | {row['n_eff']:.1f} |")
    fc = R.get("freeze_check") or {}
    L += ["", f"Freeze check: {'all match' if fc.get('all_match') else 'MISMATCH ' + json.dumps(fc.get('files'))}.", ""]
    cp = R["checkpoint2"]
    L += ["## Checkpoint 2 decision (primary: hand-over list of 7.9)", "",
          f"- Accepted in both modes in the same direction: "
          + (", ".join(f"{x['higher']} above {x['pair'].replace(x['higher'], '').strip('-')} ({x['p_ratings']:.3f} / {x['p_relative']:.3f})"
                       for x in cp["accepted_pairs"]) or "none") + ".",
          f"- To human experts: {', '.join(cp['pairs_to_human_experts']) or 'none'}.",
          f"- Unstable ranks: ratings mode {cp['unstable_ranks']['ratings']}, relative mode {cp['unstable_ranks']['relative']}.", ""]
    L += ["## Decision rule of each mode (S3, tau = 0.9)", ""]
    for mode in ("ratings", "relative"):
        d = R["decision"][f"S3/{mode}/0.9"]
        L.append(f"- {mode}: stable ranks {[x['rank'] for x in d['stable_ranks']]}, accepted {len(d['accepted_pairs'])} of "
                 f"{d['n_pairs']} pairs, unstable: " + (", ".join(f"{x['pair']} ({x['p']:.3f})" for x in d["unstable_pairs"]) or "none"))
    L += ["", "## Rank probabilities, S3", "", "| Mode | Variant | " + " | ".join(f"rank {r}" for r in range(1, 6)) + " |",
          "|:--|:--|" + "--:|" * 5]
    for mode in ("ratings", "relative"):
        for i, v in enumerate(V):
            L.append(f"| {mode} | {v} | " + " | ".join(f"{x:.3f}" for x in R["smaa"]["S3"][mode]["b"][i]) + " |")
    ac = R["analysts_comparison"]
    a4 = ac["primary_measures"]["A4_S3_ratings_mode_tau_b"]
    L += ["", "## Comparison with the analysts", "",
          f"- Primary measures (ratings mode): A1 S1 Kendall tau-b {ac['primary_measures']['A1_S1_ratings_mode_kendall_tau_b']:.3f}; "
          f"A4 S3 tau-b median {a4['median']:.3f} (2.5-97.5 %: {a4['q025']:.3f} to {a4['q975']:.3f}).",
          f"- S1 Kendall tau-b: ratings mode {ac['S1_ratings']['kendall_tau_b']:.3f}, relative mode {ac['S1_relative']['kendall_tau_b']:.3f}; "
          f"rank of the analysts' leader {ac['analysts_leader']}: {ac['S1_ratings']['rank_of_analysts_leader']} and {ac['S1_relative']['rank_of_analysts_leader']}.",
          f"- Ratings: mean absolute difference to the revised set {ac['ratings_summary']['MAD_to_revised']:.2f} points, "
          f"to the original set {ac['ratings_summary']['MAD_to_original']:.2f}; for the 12 never revised "
          f"{ac['ratings_summary']['unrevised_12']['MAD_to_revised']:.2f}. Revised ratings (listed apart): "
          + ", ".join(f"{r['variant']} {r['criterion']} {r['analysts_original']}->{r['analysts_revised']} (virtual {r['virtual_geo_mean']:.2f})"
                      for r in ac["ratings_summary"]["revised_3"]) + ".",
          f"- Provider route likelihood (analysts {ac['contested_MP_likelihood']['analysts']}): share of questionnaires with 7 or more "
          f"{ac['contested_MP_likelihood']['share_ge_7']:.2f}.", ""]
    mc = R["mode_consistency"]
    L += ["## Mode consistency", "", f"- Median tau-b between ratings and pairwise priorities per criterion: {mc['all']['median_tau_b']}; "
          f"direction conflicts {mc['all']['conflict_share_of_strict_pairs']:.3f} of strict pairs; S1 tau-b between modes {mc['group_S1_tau_b']:.3f}.", ""]
    L += ["## Hand-over list (S3, tau = 0.9, both modes)", ""]
    for row in R["hand_over"]["pairs"]:
        L.append(f"- {row['pair']}: " + ", ".join(f"{m} p = {row[m]['p']:.3f} ({row[m]['higher']} higher)" for m in ("ratings", "relative"))
                 + (f"; unstable in {', '.join(row['unstable_in'])}" if row["unstable_in"] else "")
                 + (" (modes conflict)" if row["mode_conflict"] else ""))
    if not R["hand_over"]["pairs"]:
        L.append("- none")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
