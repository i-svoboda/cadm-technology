"""Analyse the AHP replication batches and compare them with the published 2024 numbers.

For each batch of the published hierarchy: per-persona matrices (29 each), AIJ aggregation by
geometric mean, priorities by the normalized-column method of arXiv 2402.07404 (formulas 3-4)
and by the principal eigenvector, CI/CR, global sub-criteria and alternative priorities,
leave-one-expert-out, and agreement with the published values.

Usage: python ahp_analysis.py ahp_published_s56_r1 [ahp_published_s56_r2 ...] --tag s56
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from common import (RESULTS, RUNS, consistency, geo_mean, kendall_w, load, priorities_approx, priorities_eigen,
                    reciprocal_projection)

PROT = load("ahp_2024_protocol.json")
PUB = PROT["published"]
CRIT = PROT["criteria"]
ALTS = [a["id"] for a in PROT["alternatives"]]
CIDS = [c["id"] for c in CRIT]
SUBS = {c["id"]: [s["id"] for s in c["sub_criteria"]] for c in CRIT}
SUB_ALL = [s for c in CIDS for s in SUBS[c]]
PERSONAS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]


def matrix_keys():
    return ["criteria|GOAL"] + [f"sub|{c}" for c in CIDS] + [f"alt|{s}" for s in SUB_ALL]


def load_persona(batch: Path, pid: str) -> tuple[dict, dict]:
    conv = json.loads((batch / pid / "conversation.json").read_text(encoding="utf-8"))
    mats, meta = {}, {"reminders": 0, "repaired": [], "turns": 0}
    for t in conv["turns"]:
        if "_reminder" in t["folder"] or t.get("attempt", 0) > 0:
            meta["reminders"] += 1
        if not t.get("accepted") or "matrices" not in (t.get("parsed") or {}):
            continue
        meta["turns"] += 1
        for key, m in t["parsed"]["matrices"].items():
            A = np.array(m["matrix"], dtype=float)
            if m["problems"]:
                meta["repaired"].append({"matrix": key, "problems": m["problems"]})
                A = reciprocal_projection(A)
            mats[key] = A
    return mats, meta


def solve(mats: dict, method: str = "approx") -> dict:
    pr = priorities_approx if method == "approx" else (lambda A: priorities_eigen(A)[0])
    wc = dict(zip(CIDS, pr(mats["criteria|GOAL"])))
    sub_local = {c: dict(zip(SUBS[c], pr(mats[f"sub|{c}"]))) for c in CIDS}
    sub_global = {s: wc[c] * sub_local[c][s] for c in CIDS for s in SUBS[c]}
    alt_local = {s: dict(zip(ALTS, pr(mats[f"alt|{s}"]))) for s in SUB_ALL}
    alt = {a: sum(sub_global[s] * alt_local[s][a] for s in SUB_ALL) for a in ALTS}
    return {"criteria": wc, "sub_local": sub_local, "sub_global": sub_global, "alt_local": alt_local,
            "alternatives": alt}


def ranks(d: dict) -> list[str]:
    return [k for k, _ in sorted(d.items(), key=lambda x: -x[1])]


def compare(rep: dict, pub: dict) -> dict:
    keys = list(pub)
    x = np.array([rep[k] for k in keys])
    y = np.array([pub[k] for k in keys])
    diff = {k: rep[k] - pub[k] for k in keys}
    big = sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:3]
    return {"spearman": float(spearmanr(x, y).statistic), "kendall_tau": float(kendalltau(x, y).statistic),
            "max_abs_diff": float(max(abs(v) for v in diff.values())), "largest_diffs": big,
            "rank_replication": ranks(rep), "rank_published": ranks(pub),
            "same_first": ranks(rep)[0] == ranks(pub)[0]}


def published_checks() -> dict:
    A = np.array(PUB["aggregated_criteria_matrix"])
    wa = priorities_approx(A)
    we, lam = priorities_eigen(A)
    c = consistency(A)
    return {"table1_weights_approx": dict(zip(CIDS, wa.round(4).tolist())),
            "table1_weights_eigen": dict(zip(CIDS, we.round(4).tolist())),
            "table1_lambda_max": round(lam, 4), "table1_CI": round(c["CI"], 4), "table1_CR": round(c["CR"], 4),
            "published": {"weights": PUB["criteria_weights"], **PUB["criteria_consistency"]}}


def analyse_batch(batch: Path) -> dict:
    per, meta = {}, {}
    for pid in PERSONAS:
        if (batch / pid / "conversation.json").exists():
            per[pid], meta[pid] = load_persona(batch, pid)
    keys = matrix_keys()
    complete = [p for p in per if all(k in per[p] for k in keys)]
    out = {"batch": batch.name, "personas_complete": complete, "n_matrices": sum(len(per[p]) for p in per),
           "reminders": {p: meta[p]["reminders"] for p in meta}, "repaired": {p: meta[p]["repaired"] for p in meta}}
    # individual consistency and per-persona results
    crs = []
    indiv = {}
    for p in complete:
        cr = {k: consistency(per[p][k])["CR"] for k in keys}
        crs += [v for k, v in cr.items() if per[p][k].shape[0] > 2]
        res = solve(per[p])
        indiv[p] = {"criteria": res["criteria"], "alternatives": res["alternatives"],
                    "top_criterion": ranks(res["criteria"])[0], "top_alternative": ranks(res["alternatives"])[0],
                    "CR_criteria": cr["criteria|GOAL"], "CR_max": max(cr.values())}
    crs = np.array(crs)
    out["individual"] = indiv

    def w_of(field, ids):
        rk = []
        for p in complete:
            v = np.array([indiv[p][field][i] for i in ids])
            r = np.empty(len(v))
            r[np.argsort(-v)] = np.arange(1, len(v) + 1)
            rk.append(r)
        W, pv = kendall_w(np.array(rk))
        return {"W": W, "p": pv}
    out["persona_agreement"] = {"criteria": w_of("criteria", CIDS), "alternatives": w_of("alternatives", ALTS)}
    out["individual_CR"] = {"n": int(crs.size), "median": float(np.median(crs)), "max": float(crs.max()),
                            "share_below_0.01": float((crs < 0.01).mean()), "share_below_0.10": float((crs < 0.10).mean())}
    # AIJ aggregation over the complete personas
    agg = {k: geo_mean([per[p][k] for p in complete]) for k in keys}
    res = solve(agg, "approx")
    res_e = solve(agg, "eigen")
    cons = {k: consistency(agg[k]) for k in keys}
    out["aggregated"] = {
        "criteria": res["criteria"], "criteria_eigen": res_e["criteria"],
        "criteria_consistency": {k: cons["criteria|GOAL"][k] for k in ("lambda_max", "CI", "CR")},
        "criteria_matrix": agg["criteria|GOAL"].round(3).tolist(),
        "sub_global": res["sub_global"], "alternatives": res["alternatives"], "alternatives_eigen": res_e["alternatives"],
        "sub_CR": {c: cons[f"sub|{c}"]["CR"] for c in CIDS},
        "alt_CR": {s: cons[f"alt|{s}"]["CR"] for s in SUB_ALL},
    }
    # agreement with the published values
    pubA = np.array(PUB["aggregated_criteria_matrix"])
    iu = np.triu_indices(7, 1)
    la, lp = np.log(agg["criteria|GOAL"][iu]), np.log(pubA[iu])
    dl = la - lp
    worst = np.argsort(-np.abs(dl))[:3]
    pub_sub_local = {s: PUB["subcriteria_global"][s] / sum(PUB["subcriteria_global"][t] for t in SUBS[c])
                     for c in CIDS for s in SUBS[c]}
    rep_sub_local = {s: res["sub_local"][c][s] for c in CIDS for s in SUBS[c]}
    out["vs_published"] = {
        "criteria_matrix": {"pearson_log_upper": float(pearsonr(la, lp).statistic),
                            "same_direction_share": float(np.mean(np.sign(la) == np.sign(lp))),
                            "largest_log_diffs": [{"pair": f"{CIDS[iu[0][i]]}/{CIDS[iu[1][i]]}",
                                                   "replication": float(np.exp(la[i])), "published": float(np.exp(lp[i]))}
                                                  for i in worst]},
        "criteria": compare(res["criteria"], PUB["criteria_weights_tacs_4dp"]),
        "sub_global": compare(res["sub_global"], PUB["subcriteria_global"]),
        "sub_local": compare(rep_sub_local, pub_sub_local),
        "alternatives": compare(res["alternatives"], PUB["alternatives"]),
        "criteria_CR": {"replication": cons["criteria|GOAL"]["CR"], "published": PUB["criteria_consistency"]["CR"]},
        "sub_CR_range": {"replication": [min(out["aggregated"]["sub_CR"].values()),
                                         max(out["aggregated"]["sub_CR"].values())],
                         "published": [PUB["subcriteria_cr_range"]["min"], PUB["subcriteria_cr_range"]["max"]]},
        "alt_CR_range": {"replication": [min(out["aggregated"]["alt_CR"].values()),
                                         max(out["aggregated"]["alt_CR"].values())],
                         "published": [PUB["alternatives_cr_range"]["min"], PUB["alternatives_cr_range"]["max"]]},
    }
    # leave one expert out
    base_rank = ranks(res["alternatives"])
    loeo = {}
    for p in complete:
        rest = [q for q in complete if q != p]
        r = solve({k: geo_mean([per[q][k] for q in rest]) for k in keys})["alternatives"]
        rk = ranks(r)
        loeo[p] = {"alternatives": r, "top": rk[0], "gap": r[rk[0]] - r[rk[1]],
                   "tau_vs_base": float(kendalltau([r[a] for a in ALTS], [res["alternatives"][a] for a in ALTS]).statistic)}
    out["loeo"] = {"runs": loeo, "top_unchanged": all(v["top"] == base_rank[0] for v in loeo.values()),
                   "base_gap": res["alternatives"][base_rank[0]] - res["alternatives"][base_rank[1]],
                   "mean_tau": float(np.mean([v["tau_vs_base"] for v in loeo.values()]))}
    return out


def write_tables(results: list[dict], tag: str):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"ahp_{tag}_alternatives.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alternative", "published"] + [r["batch"] for r in results])
        for a in ALTS:
            w.writerow([a, PUB["alternatives"][a]] + [round(r["aggregated"]["alternatives"][a], 4) for r in results])
    with open(RESULTS / f"ahp_{tag}_criteria.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "published"] + [r["batch"] for r in results])
        for c in CIDS:
            w.writerow([c, PUB["criteria_weights_tacs_4dp"][c]] + [round(r["aggregated"]["criteria"][c], 4)
                                                                   for r in results])
    with open(RESULTS / f"ahp_{tag}_subcriteria_global.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sub_criterion", "criterion", "published"] + [r["batch"] for r in results])
        for c in CIDS:
            for s in SUBS[c]:
                w.writerow([s, c, PUB["subcriteria_global"][s]] + [round(r["aggregated"]["sub_global"][s], 4)
                                                                    for r in results])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    results = [analyse_batch(RUNS / b) for b in args.batches]
    summary = {"published_checks": published_checks(), "batches": results}
    if len(results) > 1:
        alts = np.array([[r["aggregated"]["alternatives"][a] for a in ALTS] for r in results])
        crit = np.array([[r["aggregated"]["criteria"][c] for c in CIDS] for r in results])
        taus = [kendalltau(alts[i], alts[j]).statistic for i in range(len(results)) for j in range(i + 1, len(results))]
        summary["across_panels"] = {
            "alternatives_mean": dict(zip(ALTS, alts.mean(0).round(4).tolist())),
            "alternatives_min": dict(zip(ALTS, alts.min(0).round(4).tolist())),
            "alternatives_max": dict(zip(ALTS, alts.max(0).round(4).tolist())),
            "criteria_mean": dict(zip(CIDS, crit.mean(0).round(4).tolist())),
            "criteria_min": dict(zip(CIDS, crit.min(0).round(4).tolist())),
            "criteria_max": dict(zip(CIDS, crit.max(0).round(4).tolist())),
            "pairwise_kendall_tau_alternatives": [float(t) for t in taus],
            "first_place": [ranks(r["aggregated"]["alternatives"])[0] for r in results],
            "mean_vs_published": compare(dict(zip(ALTS, alts.mean(0))), PUB["alternatives"]),
        }
    (RESULTS / f"ahp_{args.tag}.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    write_tables(results, args.tag)
    for r in results:
        v = r["vs_published"]
        print(r["batch"], "complete:", len(r["personas_complete"]), "matrices:", r["n_matrices"])
        print("  criteria   ", {k: round(x, 3) for k, x in r["aggregated"]["criteria"].items()},
              "CR", round(r["aggregated"]["criteria_consistency"]["CR"], 4))
        print("  alternatives", {k: round(x, 4) for k, x in r["aggregated"]["alternatives"].items()})
        print("  vs published: criteria rho", round(v["criteria"]["spearman"], 2), "sub rho",
              round(v["sub_global"]["spearman"], 2), "alt rho", round(v["alternatives"]["spearman"], 2),
              "alt max diff", round(v["alternatives"]["max_abs_diff"], 4), "matrix log r",
              round(v["criteria_matrix"]["pearson_log_upper"], 2))
        print("  LOEO top unchanged:", r["loeo"]["top_unchanged"], "reminders:", r["reminders"])
    print(json.dumps(summary["published_checks"], indent=1))


if __name__ == "__main__":
    main()
