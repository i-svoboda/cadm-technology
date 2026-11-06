"""SMAA over LLM-elicited AHP questionnaires, plus variance components.

Each run directory written by elicit.py holds one questionnaire (one persona,
one wording, one replicate). This module turns a set of them into:
  - per-questionnaire priorities and consistency,
  - rank acceptability indices b[i, r] and pairwise winning indices P(i > j)
    under four panel-sampling schemes (naive pooling, i.i.d. panels,
    hierarchical panels with fixed roles, hierarchical panels with random roles),
  - variance components of the log-judgments (persona, wording, persona x wording, run).

Usage: python scripts/smaa_ahp.py --case experiments/cases/ahp_datacenter.json \
           --runs experiments/runs/<batch> --draws 10000 --out results/<name>.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def load_questionnaire(run: Path, case: dict) -> dict:
    """Return {'meta': manifest, 'logj': {(matrix, a, b): log a_ab with a < b}}."""
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = {i["pair_id"]: i for i in json.loads((run / "items.json").read_text(encoding="utf-8"))}
    ans = json.loads((run / "last_message.json").read_text(encoding="utf-8"))["judgments"]
    logj = {}
    for j in ans:
        it = items[j["pair_id"]]
        v = 0.0 if j["preferred"] == "equal" or j["intensity"] == 1 else math.log(j["intensity"])
        if j["preferred"] == "second":
            v = -v
        a, b = it["first"], it["second"]
        if a > b:
            a, b, v = b, a, -v
        logj[(it["matrix"], a, b)] = v
    return {"meta": man, "logj": logj}


def matrix_ids(case: dict, matrix: str) -> list[str]:
    return [c["id"] for c in case["criteria"]] if matrix.startswith("criteria") else [a["id"] for a in case["alternatives"]]


def build(case: dict, logj: dict) -> dict[str, np.ndarray]:
    """Log-judgment dict -> reciprocal matrices."""
    mats: dict[str, np.ndarray] = {}
    for (m, a, b), v in logj.items():
        ids = matrix_ids(case, m)
        A = mats.setdefault(m, np.ones((len(ids), len(ids))))
        i, j = ids.index(a), ids.index(b)
        A[i, j], A[j, i] = math.exp(v), math.exp(-v)
    return mats


def eigvec(A: np.ndarray) -> tuple[np.ndarray, float]:
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    return w / w.sum(), float(vals[k].real)


def consistency(A: np.ndarray) -> float:
    n = A.shape[0]
    _, lam = eigvec(A)
    return ((lam - n) / (n - 1)) / RI[n] if RI[n] else 0.0


def ahp_global(case: dict, mats: dict[str, np.ndarray]) -> np.ndarray:
    crit = [c["id"] for c in case["criteria"]]
    cw, _ = eigvec(mats["criteria|GOAL"])
    g = np.zeros(len(case["alternatives"]))
    for k, c in enumerate(crit):
        w, _ = eigvec(mats[f"alternatives|{c}"])
        g += cw[k] * w
    return g


def aggregate(logjs: list[dict]) -> dict:
    """AIJ: geometric mean of judgments = arithmetic mean of log-judgments."""
    keys = logjs[0].keys()
    return {k: float(np.mean([lj[k] for lj in logjs])) for k in keys}


def ranks_of(g: np.ndarray) -> np.ndarray:
    order = np.argsort(-g)
    r = np.empty(len(g), dtype=int)
    r[order] = np.arange(len(g))
    return r


def smaa(case: dict, qs: list[dict], scheme: str, draws: int, panel: int, rng: np.random.Generator) -> dict:
    alts = [a["id"] for a in case["alternatives"]]
    n = len(alts)
    by_persona: dict[str, list[dict]] = defaultdict(list)
    for q in qs:
        by_persona[q["meta"]["persona"]].append(q)
    personas = sorted(by_persona)
    if scheme == "naive":                      # all questionnaires pooled as one panel: no spread
        g = ahp_global(case, build(case, aggregate([q["logj"] for q in qs])))
        b = np.zeros((n, n)); b[np.arange(n), ranks_of(g)] = 1.0
        return {"b": b, "pwin": (g[:, None] > g[None, :]).astype(float), "mean_priority": g}
    b = np.zeros((n, n)); pwin = np.zeros((n, n)); gs = np.zeros(n)
    for _ in range(draws):
        if scheme == "iid":
            pick = [qs[i] for i in rng.integers(0, len(qs), panel)]
        elif scheme == "hier_fixed":
            pick = [by_persona[p][rng.integers(0, len(by_persona[p]))] for p in personas]
        elif scheme == "hier_random":
            pick = []
            for p in rng.choice(personas, size=len(personas), replace=True):
                pick.append(by_persona[p][rng.integers(0, len(by_persona[p]))])
        else:
            raise ValueError(scheme)
        g = ahp_global(case, build(case, aggregate([q["logj"] for q in pick])))
        b[np.arange(n), ranks_of(g)] += 1
        pwin += g[:, None] > g[None, :]
        gs += g
    return {"b": b / draws, "pwin": pwin / draws, "mean_priority": gs / draws}


def variance_components(qs: list[dict]) -> dict:
    """Balanced random-effects ANOVA per item, pooled by matrix type.

    y = mu + persona + wording + persona x wording + run, with p personas,
    w wordings and r runs per cell. Negative estimates are truncated at 0.
    """
    cells: dict = defaultdict(lambda: defaultdict(list))
    for q in qs:
        m = q["meta"]
        for k, v in q["logj"].items():
            cells[k][(m["persona"], m["wording"])].append(v)
    out = defaultdict(lambda: np.zeros(4))
    count = defaultdict(int)
    for k, cell in cells.items():
        P = sorted({c[0] for c in cell}); W = sorted({c[1] for c in cell})
        r = min(len(v) for v in cell.values())
        if len(P) < 2 or len(W) < 2 or r < 2 or len(cell) != len(P) * len(W):
            continue
        y = np.array([[cell[(p, w)][:r] for w in W] for p in P])     # p x w x r
        p, w = len(P), len(W)
        gm = y.mean(); mp = y.mean(axis=(1, 2)); mw = y.mean(axis=(0, 2)); mpw = y.mean(axis=2)
        ms_p = w * r * np.sum((mp - gm) ** 2) / (p - 1)
        ms_w = p * r * np.sum((mw - gm) ** 2) / (w - 1)
        ms_pw = r * np.sum((mpw - mp[:, None] - mw[None, :] + gm) ** 2) / ((p - 1) * (w - 1))
        ms_e = np.sum((y - mpw[:, :, None]) ** 2) / (p * w * (r - 1))
        comp = np.array([max((ms_p - ms_pw) / (w * r), 0), max((ms_w - ms_pw) / (p * r), 0),
                         max((ms_pw - ms_e) / r, 0), ms_e])
        mtype = k[0].split(":")[0] if ":" in k[0] else k[0].split("|")[0]   # AHP: criteria/alternatives; ANP: block
        out[mtype] += comp; count[mtype] += 1
    res = {}
    for mtype, s in out.items():
        s = s / count[mtype]
        res[mtype] = {"items": count[mtype], "persona": s[0], "wording": s[1], "persona_x_wording": s[2], "run": s[3],
                      "share": dict(zip(["persona", "wording", "persona_x_wording", "run"], (s / s.sum()).round(3).tolist())),
                      "icc_persona": s[0] / s.sum()}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", required=True, help="directory whose subdirectories are elicit.py runs")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--out")
    args = ap.parse_args()
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    qs = [load_questionnaire(d, case) for d in sorted(Path(args.runs).iterdir()) if (d / "last_message.json").exists()]
    rng = np.random.default_rng(args.seed)
    alts = [a["id"] for a in case["alternatives"]]
    report = {"n_questionnaires": len(qs), "alternatives": alts, "schemes": {}}
    for q in qs:
        mats = build(case, q["logj"])
        q["cr"] = {m: round(consistency(A), 4) for m, A in mats.items()}
        q["global"] = ahp_global(case, mats).round(4).tolist()
    report["per_questionnaire"] = [{"run": q["meta"].get("session_id"), "persona": q["meta"]["persona"], "wording": q["meta"]["wording"],
                                    "replicate": q["meta"]["replicate"], "max_cr": max(q["cr"].values()), "global": q["global"]} for q in qs]
    personas = {q["meta"]["persona"] for q in qs}
    for scheme in ["naive", "iid", "hier_fixed", "hier_random"]:
        res = smaa(case, qs, scheme, args.draws, len(personas), rng)
        report["schemes"][scheme] = {"rank_acceptability": res["b"].round(4).tolist(), "pairwise_win": res["pwin"].round(4).tolist(),
                                     "mean_priority": np.round(res["mean_priority"], 4).tolist()}
        print(f"\n== {scheme}: rank acceptability (rows {alts}, columns rank 1..{len(alts)})")
        for a, row in zip(alts, res["b"]):
            print(f"  {a:4s}", " ".join(f"{x:5.2f}" for x in row))
    vc = variance_components(qs)
    report["variance_components"] = {k: {kk: (vv if not isinstance(vv, float) else round(vv, 4)) for kk, vv in v.items()} for k, v in vc.items()}
    print("\nvariance components:", json.dumps(report["variance_components"], indent=1, default=float))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
