"""Extra statistics for the AHP batch: consistency, criteria weights, effective panel
size, persona redundancy, panel-size curve and comparison with the published ranking.

Usage: python scripts/ahp_extra.py --case experiments/cases/ahp_datacenter.json \
           --runs experiments/runs/ahp_main --smaa results/ahp_main_smaa.json --out results/ahp_main_extra.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smaa  # noqa: E402
import smaa_ahp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--smaa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    rep = json.loads(Path(args.smaa).read_text(encoding="utf-8"))
    qs = [smaa.load(d) for d in sorted(Path(args.runs).iterdir()) if (d / "last_message.json").exists()]
    qs = [q for q in qs if q["complete"]]
    alts = [a["id"] for a in case["alternatives"]]
    crit = [c["id"] for c in case["criteria"]]
    rng = np.random.default_rng(args.seed)
    out: dict = {"n": len(qs)}

    # Consistency
    crs = []
    for q in qs:
        mats = smaa_ahp.build(case, q["logj"])
        crs += [smaa_ahp.consistency(A) for A in mats.values()]
    crs = np.array(crs)
    out["consistency"] = {"matrices": len(crs), "median_CR": float(np.median(crs)), "p90_CR": float(np.quantile(crs, .9)),
                          "max_CR": float(crs.max()), "share_CR_over_0.1": float(np.mean(crs > 0.1))}

    # Criteria weights: pooled panel and per persona
    def cw(logj):
        w, _ = smaa_ahp.eigvec(smaa_ahp.build(case, logj)["criteria|GOAL"])
        return w
    pooled = smaa_ahp.aggregate([q["logj"] for q in qs])
    out["criteria_weights_pooled"] = dict(zip(crit, np.round(cw(pooled), 4).tolist()))
    by_p: dict[str, list[dict]] = defaultdict(list)
    for q in qs:
        by_p[q["meta"]["persona"]].append(q)
    personas = sorted(by_p)
    out["criteria_weights_by_persona"] = {p: dict(zip(crit, np.round(cw(smaa_ahp.aggregate([q["logj"] for q in by_p[p]])), 4).tolist())) for p in personas}
    out["global_by_persona"] = {p: dict(zip(alts, np.round(smaa_ahp.ahp_global(case, smaa_ahp.build(case, smaa_ahp.aggregate([q["logj"] for q in by_p[p]]))), 4).tolist()))
                                for p in personas}
    tops = {p: {a: 0 for a in alts} for p in personas}
    for q in qs:
        g = smaa_ahp.ahp_global(case, smaa_ahp.build(case, q["logj"]))
        tops[q["meta"]["persona"]][alts[int(np.argmax(g))]] += 1
    out["first_place_counts_by_persona"] = tops

    # Effective sample size of the 63 questionnaires (design effect of persona clusters)
    vc = rep["variance_components"]
    m = len(qs) / len(personas)
    out["design_effect"] = {}
    for t, v in vc.items():
        rho = v["persona"] / (v["persona"] + v["wording"] + v["persona_x_wording"] + v["run"])
        deff = 1 + (m - 1) * rho
        out["design_effect"][t] = {"rho_within_persona": round(rho, 3), "runs_per_persona": m, "deff": round(deff, 2),
                                   "n_eff_of_63": round(len(qs) / deff, 1)}

    # Persona redundancy: participation ratio of the persona correlation matrix
    keys = sorted(qs[0]["logj"])
    M = np.array([[np.mean([q["logj"][k] for q in by_p[p]]) for k in keys] for p in personas])
    D = M - M.mean(axis=0)                       # persona deviations from the panel mean per item
    C = np.corrcoef(D)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    off = C[np.triu_indices(len(personas), 1)]
    out["persona_redundancy"] = {"participation_ratio": round(float(lam.sum() ** 2 / (lam ** 2).sum()), 2),
                                 "mean_offdiag_corr_of_deviations": round(float(off.mean()), 3),
                                 "raw_profile_corr_mean": round(float(np.corrcoef(M)[np.triu_indices(len(personas), 1)].mean()), 3),
                                 "n_personas": len(personas)}

    # Panel-size curve: first-place acceptability when k roles are drawn with replacement
    curve = {}
    for k in (3, 5, 7, 10, 15, 21):
        b1 = np.zeros(len(alts))
        for _ in range(4000):
            pick = [by_p[p][rng.integers(0, len(by_p[p]))] for p in rng.choice(personas, k, replace=True)]
            g = smaa_ahp.ahp_global(case, smaa_ahp.build(case, smaa_ahp.aggregate([q["logj"] for q in pick])))
            b1[int(np.argmax(g))] += 1
        curve[k] = dict(zip(alts, np.round(b1 / 4000, 3).tolist()))
    out["first_place_by_panel_size_random_roles"] = curve

    # Pairwise winning indices of the top three under random roles
    pw = np.array(rep["results"]["hier_random/fixed"]["pairwise_win"])
    out["pairwise_win_random_roles"] = {f"{alts[i]}>{alts[j]}": pw[i, j] for i in range(len(alts)) for j in range(len(alts)) if i != j}
    out["published_reference"] = case["published_reference"]
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
