"""Extra statistics for the ANP batch: consistency, elicited cluster weights and link
probabilities, per-persona priorities under both cluster weightings, effective panel size,
persona redundancy, panel-size curve and the comparison with the published ranking.

Usage: python scripts/anp_extra.py --case experiments/cases/anp_ci_controls.json \
           --runs experiments/runs/anp_main --smaa-its results/anp_main_smaa_its.json \
           --smaa-elicited results/anp_main_smaa_elicited.json --out results/anp_main_extra.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anp_case  # noqa: E402
import smaa  # noqa: E402
import smaa_ahp  # noqa: E402


def prio(case, logj, links, cw):
    on = {k: v >= 0.5 for k, v in links.items()}
    pr = anp_case.control_priorities(case, logj, on, cluster_weights=cw)
    return np.array([pr[n["id"]] for n in case["clusters"]["K"]["nodes"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--smaa-its", required=True)
    ap.add_argument("--smaa-elicited", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    rep = {"its": json.loads(Path(args.smaa_its).read_text(encoding="utf-8")),
           "elicited": json.loads(Path(args.smaa_elicited).read_text(encoding="utf-8"))}
    qs = [q for q in (smaa.load(d) for d in sorted(Path(args.runs).iterdir()) if (d / "last_message.json").exists()) if q["complete"]]
    ctrl = [n["id"] for n in case["clusters"]["K"]["nodes"]]
    rng = np.random.default_rng(args.seed)
    by_p: dict[str, list[dict]] = defaultdict(list)
    for q in qs:
        by_p[q["meta"]["persona"]].append(q)
    personas = sorted(by_p)
    out: dict = {"n": len(qs)}

    crs = []
    for q in qs:
        for m, (ids, A) in anp_case.local_matrices(case, q["logj"]).items():
            if len(ids) > 2:
                crs.append(smaa_ahp.consistency(A))
    crs = np.array(crs)
    out["consistency"] = {"matrices_3plus": len(crs), "median_CR": float(np.median(crs)), "p90_CR": float(np.quantile(crs, .9)),
                          "max_CR": float(crs.max()), "share_CR_over_0.1": float(np.mean(crs > 0.1))}

    def clusters(logj):
        ids, A = anp_case.local_matrices(case, logj)["CL|NET"]
        return dict(zip(ids, np.round(anp_case.eigvec(A), 4).tolist()))
    pooled = smaa_ahp.aggregate([q["logj"] for q in qs])
    out["cluster_weights_pooled"] = clusters(pooled)
    out["cluster_weights_by_persona"] = {p: clusters(smaa_ahp.aggregate([q["logj"] for q in by_p[p]])) for p in personas}
    out["link_probability_mean"] = {f"{s}->{t}": round(float(np.mean([q["links"][(s, t)] for q in qs])), 3) for (s, t) in qs[0]["links"]}
    out["link_probability_min"] = {f"{s}->{t}": round(float(np.min([q["links"][(s, t)] for q in qs])), 3) for (s, t) in qs[0]["links"]}

    for cw in ("its", "elicited"):
        g_p = {}
        for p in personas:
            lj = smaa_ahp.aggregate([q["logj"] for q in by_p[p]])
            g_p[p] = dict(zip(ctrl, np.round(prio(case, lj, smaa.mean_links(by_p[p]), cw), 4).tolist()))
        out[f"global_by_persona_{cw}"] = g_p
        tops = {p: {c: 0 for c in ctrl} for p in personas}
        for q in qs:
            tops[q["meta"]["persona"]][ctrl[int(np.argmax(prio(case, q["logj"], q["links"], cw)))]] += 1
        out[f"first_place_counts_{cw}"] = tops
        curve = {}
        for k in (3, 7, 15, 21):
            b1 = np.zeros(len(ctrl))
            for _ in range(2000):
                pick = [by_p[p][rng.integers(0, len(by_p[p]))] for p in rng.choice(personas, k, replace=True)]
                g = prio(case, smaa_ahp.aggregate([q["logj"] for q in pick]), smaa.mean_links(pick), cw)
                b1[int(np.argmax(g))] += 1
            curve[k] = dict(zip(ctrl, np.round(b1 / 2000, 3).tolist()))
        out[f"first_place_by_panel_size_{cw}"] = curve
        pw = np.array(rep[cw]["results"]["hier_random/fixed"]["pairwise_win"])
        out[f"pairwise_win_random_roles_{cw}"] = {f"{ctrl[i]}>{ctrl[j]}": float(pw[i, j]) for i in range(len(ctrl)) for j in range(len(ctrl)) if i < j}
        res = rep[cw]["results"]
        out[f"structure_effect_{cw}"] = {k: {"sd_priority": dict(zip(ctrl, res[k]["sd_priority"]))} for k in res if k in
                                         ("hier_fixed/fixed", "hier_fixed/sampled", "naive/only", "hier_random/fixed", "hier_random/sampled")}

    vc = rep["its"]["variance_components"]
    m = len(qs) / len(personas)
    out["design_effect"] = {}
    for t, v in vc.items():
        rho = v["persona"] / (v["persona"] + v["wording"] + v["persona_x_wording"] + v["run"])
        out["design_effect"][t] = {"rho_within_persona": round(rho, 3), "deff": round(1 + (m - 1) * rho, 2),
                                   "n_eff": round(len(qs) / (1 + (m - 1) * rho), 1)}
    keys = sorted(k for k in qs[0]["logj"] if k[0] != "CL|NET")
    M = np.array([[np.mean([q["logj"][k] for q in by_p[p]]) for k in keys] for p in personas])
    C = np.corrcoef(M - M.mean(axis=0))
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    out["persona_redundancy"] = {"participation_ratio": round(float(lam.sum() ** 2 / (lam ** 2).sum()), 2),
                                 "raw_profile_corr_mean": round(float(np.corrcoef(M)[np.triu_indices(len(personas), 1)].mean()), 3)}
    out["published_reference"] = case["published_reference"]
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("consistency", "cluster_weights_pooled", "link_probability_mean", "design_effect", "persona_redundancy")}, indent=1))


if __name__ == "__main__":
    main()
