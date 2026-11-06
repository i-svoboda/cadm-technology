"""Analyse the ANP replication and compare it with ITS 14(1).

Per panel (seven personas, one questionnaire each): AIJ aggregation of the 32 matrices,
local priorities (principal eigenvector), CR and Koczkodaj index, the weighted supermatrix
with structural-zero renormalization of the published cluster prior, limit priorities,
leave-one-expert-out, Kendall's W per criterion, a 1000-trial Monte Carlo perturbation in the
log-ratio domain, threat scenarios and the knapsack portfolio.

Usage: python anp_analysis.py anp_s56 --tag s56 [--trials 1000]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau, spearmanr

from anp_model import SYNERGY, Case
from common import RESULTS, RUNS, consistency, geo_mean, kendall_w, load, priorities_eigen

PROT = load("anp_2026_protocol.json")
PUB = PROT["published"]
CASE = Case(PROT)
K = CASE.ids["K"]
R = CASE.ids["R"]
H = CASE.ids["H"]
KEYS = [k for k, _, _ in CASE.matrix_specs()]


def load_batch(batch):
    out = defaultdict(dict)                 # replicate -> persona -> record
    for rf in sorted((RUNS / batch).glob("*/run.json")):
        st = json.loads(rf.read_text(encoding="utf-8"))
        if not st.get("accepted"):
            continue
        d = rf.parent
        items = json.loads((d / "items.json").read_text(encoding="utf-8"))
        ans = json.loads((d / st["answer_folder"] / "answer.json").read_text(encoding="utf-8"))
        logj, miss = CASE.to_logs(items, ans)
        alpha, eal, miss2 = CASE.coverage(items, ans)
        man = json.loads((d / st["answer_folder"] / "manifest.json").read_text(encoding="utf-8"))
        out[st["replicate"]][st["persona"]] = {"mats": CASE.matrices(logj), "alpha": alpha, "eal": eal,
                                               "missing": miss + miss2, "name": st["name"],
                                               "wall": man["wall_seconds"], "usage": man["usage"]}
    return out


def local_vectors(mats):
    return {k: priorities_eigen(A)[0] for k, A in mats.items()}


def solve(mats, design="D1", **kw):
    loc = local_vectors(mats)
    pr, diag = CASE.control_priorities(loc, design, **kw)
    return pr, diag, loc


def solve_reversed_synergy(loc):
    """Sensitivity: synergy links read the other way (the target's priority grows)."""
    # MON then has two equally weighted targets of its own; the elicited SEG-vs-IDM comparison is not used
    return CASE.control_priorities(loc, "D1", synergy=[(t, s) for s, t in SYNERGY])[0]


def ranks(d):
    return [k for k, _ in sorted(d.items(), key=lambda x: -x[1])]


def compare(rep: dict, pub: dict):
    keys = list(pub)
    x = np.array([rep[k] for k in keys])
    y = np.array([pub[k] for k in keys])
    diff = {k: float(rep[k] - pub[k]) for k in keys}
    return {"spearman": float(spearmanr(x, y).statistic), "kendall_tau": float(kendalltau(x, y).statistic),
            "max_abs_diff": max(abs(v) for v in diff.values()),
            "largest_diffs": sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:3],
            "rank_replication": ranks(rep), "rank_published": ranks(pub)}


def consistency_stats(mats_list):
    cr, kz = [], []
    for mats in mats_list:
        for k, A in mats.items():
            if A.shape[0] >= 3:
                c = consistency(A)
                cr.append(c["CR"])
                kz.append(c["koczkodaj"])
    cr, kz = np.array(cr), np.array(kz)
    return {"n": int(cr.size), "mean_CR": float(cr.mean()), "sd_CR": float(cr.std(ddof=1)), "max_CR": float(cr.max()),
            "share_CR_below_0.10": float((cr < 0.10).mean()), "mean_K": float(kz.mean()), "max_K": float(kz.max()),
            "share_K_below_0.10": float((kz < 0.10).mean())}


def perturb(mats, sigma, rng):
    out = {}
    for k, A in mats.items():
        n = A.shape[0]
        L = np.log(A)
        E = np.zeros((n, n))
        iu = np.triu_indices(n, 1)
        E[iu] = rng.normal(0.0, sigma, size=len(iu[0]))
        E = E - E.T
        out[k] = np.exp(L + E)
    return out


def perturbed_crs(mats, sigma, rng, reps):
    return [consistency(M)["CR"] for _ in range(reps) for M in perturb(mats, sigma, rng).values() if M.shape[0] >= 3]


def calibrate_sigma(mats, rng, target=0.075):
    """Bisection on sigma so that the median CR of the perturbed matrices is about 0.075 (ITS: CR about 0.05-0.10)."""
    lo, hi = 0.0, 1.5
    for _ in range(16):
        mid = (lo + hi) / 2
        if np.median(perturbed_crs(mats, mid, rng, 40)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def monte_carlo(mats, trials, seed=2026):
    rng = np.random.default_rng(seed)
    sigma = calibrate_sigma(mats, rng)
    base, _, _ = solve(mats)
    brank = ranks(base)
    top2 = set(brank[:2])
    keep_top2, swap_sup_ir, first = 0, 0, defaultdict(int)
    maxchange = 0.0
    crs = []
    for _ in range(trials):
        pm = perturb(mats, sigma, rng)
        crs += [consistency(M)["CR"] for M in pm.values() if M.shape[0] >= 3]
        pr, _, _ = solve(pm)
        rk = ranks(pr)
        keep_top2 += set(rk[:2]) == top2
        first[rk[0]] += 1
        swap_sup_ir += (rk.index("SUP") < rk.index("IR")) != (brank.index("SUP") < brank.index("IR"))
        maxchange = max(maxchange, max(abs(pr[k] - base[k]) for k in K))
    crs = np.array(crs)
    return {"trials": trials, "sigma": sigma, "median_CR_perturbed": float(np.median(crs)),
            "share_CR_0.05_0.10": float(((crs >= 0.05) & (crs <= 0.10)).mean()),
            "top2_kept": keep_top2 / trials, "sup_ir_swap": swap_sup_ir / trials,
            "first_place": {k: v / trials for k, v in sorted(first.items(), key=lambda x: -x[1])},
            "max_weight_change": maxchange, "base_top2": sorted(top2)}


def loeo(panel):
    base, _, _ = solve(geo_agg(panel))
    rows = {}
    for p in panel:
        rest = {q: panel[q] for q in panel if q != p}
        pr, _, _ = solve(geo_agg(rest))
        rows[p] = pr
    tab = {}
    brank = ranks(base)
    for k in K:
        vals = np.array([rows[p][k] for p in rows])
        tab[k] = {"base": base[k], "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "min": float(vals.min()),
                  "max": float(vals.max()),
                  "rank_changes": int(sum(ranks(rows[p]).index(k) != brank.index(k) for p in rows))}
    taus = [kendalltau([rows[p][k] for k in K], [base[k] for k in K]).statistic for p in rows]
    mac = np.mean([abs(rows[p][k] - base[k]) for p in rows for k in K])
    return {"table": tab, "mean_abs_change": float(mac), "mean_kendall_tau": float(np.mean(taus)),
            "top2_invariant": all(set(ranks(rows[p])[:2]) == set(brank[:2]) for p in rows),
            "first_invariant": all(ranks(rows[p])[0] == brank[0] for p in rows)}


def kendall_by_criterion(panel):
    out = {}
    for c in R:
        rk = []
        for p in panel:
            w = priorities_eigen(panel[p]["mats"][f"K|R:{c}"])[0]
            order = np.argsort(-w)
            r = np.empty(len(w))
            r[order] = np.arange(1, len(w) + 1)
            rk.append(r)
        W, pv = kendall_w(np.array(rk))
        out[c] = {"W": W, "p": pv}
    return out


def geo_agg(panel):
    return {k: geo_mean([panel[p]["mats"][k] for p in panel]) for k in KEYS}


def knapsack(panel):
    costs = PUB["knapsack"]["costs_musd"]
    B = PUB["knapsack"]["budget_musd"]
    alpha = {(c, t): float(np.mean([panel[p]["alpha"][(c, t)] for p in panel if (c, t) in panel[p]["alpha"]]))
             for c in K for t in H}
    eal = {t: float(np.exp(np.mean([np.log(max(panel[p]["eal"][t], 1e-6)) for p in panel if t in panel[p]["eal"]])))
           for t in H}

    def f(S):
        return sum(eal[t] * (1 - np.prod([1 - alpha[(c, t)] for c in S])) for t in H)

    # lazy greedy by gain per cost (the "standard lazy greedy" of ITS 14(1)), then the exact optimum
    S, spent = [], 0.0
    while True:
        best, bval = None, 0.0
        for c in K:
            if c in S or spent + costs[c] > B + 1e-9:
                continue
            g = (f(S + [c]) - f(S)) / costs[c]
            if g > bval:
                best, bval = c, g
        if best is None:
            break
        S.append(best)
        spent += costs[best]
    best_set, best_val = (), 0.0
    for r in range(1, len(K) + 1):
        for comb in itertools.combinations(K, r):
            if sum(costs[c] for c in comb) <= B + 1e-9 and f(list(comb)) > best_val:
                best_set, best_val = comb, f(list(comb))
    pub_set = PUB["knapsack"]["selected"]
    return {"greedy": S, "greedy_cost": spent, "greedy_reduction": f(S), "optimum": list(best_set),
            "optimum_reduction": best_val, "published_set_reduction": f(pub_set), "eal_total": sum(eal.values()),
            "eal": eal, "alpha": {f"{c}->{t}": v for (c, t), v in alpha.items()}}


def scenarios(mats, mult=2.0):
    out = {}
    for name, threats in [("SC", ["SC"]), ("EXP", ["EXP"]), ("INS+SAB", ["INS", "SAB"])]:
        pr, _, _ = solve(mats, threat_mult={t: mult for t in threats})
        out[name] = pr
    return out


def published_composition():
    """Hierarchical composition of the published Table 3 vectors (internal check of the published numbers)."""
    t3 = PUB["table3_local"]
    order = PUB["control_order_in_table3"]
    goal = dict(zip(R, t3["R|GOAL"]["vector"]))
    comp = {k: 0.0 for k in order}
    for c in R:
        for k, w in zip(order, t3[f"K|R:{c}"]["vector"]):
            comp[k] += goal[c] * w
    return comp


def analyse_panel(panel, trials):
    agg = geo_agg(panel)
    pr, diag, loc = solve(agg)
    hier = CASE.hierarchical(loc)
    rev = solve_reversed_synergy(loc)
    t3 = PUB["table3_local"]
    order = PUB["control_order_in_table3"]
    local_cmp = {"R|GOAL": compare(dict(zip(R, loc["R|GOAL"])), dict(zip(R, t3["R|GOAL"]["vector"])))}
    for c in R:
        rep = dict(zip(K, loc[f"K|R:{c}"]))
        local_cmp[f"K|R:{c}"] = compare(rep, dict(zip(order, t3[f"K|R:{c}"]["vector"])))
    lam = {k: consistency(agg[k]) for k in ["R|GOAL"] + [f"K|R:{c}" for c in R]}
    return {
        "personas": sorted(panel),
        "global_D1": pr, "global_hierarchical": hier, "global_reversed_synergy": rev,
        "diagnostics": diag,
        "local": {k: dict(zip(CASE.matrix_elements(k), np.round(v, 4).tolist())) for k, v in loc.items()},
        "local_consistency": {k: {"lambda_max": v["lambda_max"], "CI": v["CI"], "CR": v["CR"]} for k, v in lam.items()},
        "consistency_aggregated": consistency_stats([agg]),
        "consistency_individual": consistency_stats([panel[p]["mats"] for p in panel]),
        "vs_published_global": compare(pr, PUB["global"]),
        "vs_published_global_reversed_synergy": compare(rev, PUB["global"]),
        "vs_published_global_hier": compare(hier, PUB["global"]),
        "vs_published_local": local_cmp,
        "loeo": loeo(panel), "kendall_w": kendall_by_criterion(panel),
        "monte_carlo": monte_carlo(agg, trials), "scenarios_x2": scenarios(agg), "knapsack": knapsack(panel),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trials", type=int, default=1000)
    args = ap.parse_args()
    data = load_batch(args.batch)
    res = {"batch": args.batch, "published_composition_check": published_composition(), "panels": {}}
    for r in sorted(data):
        panel = data[r]
        if len(panel) < 7:
            print(f"replicate {r}: only {len(panel)} personas, skipped")
            continue
        res["panels"][f"r{r}"] = analyse_panel(panel, args.trials)
        print(f"panel r{r} done", flush=True)
    complete = [r for r in sorted(data) if len(data[r]) == 7]
    if len(complete) > 1:
        pooled = {f"{p}_r{r}": data[r][p] for r in complete for p in data[r]}
        agg = geo_agg(pooled)
        pr, diag, loc = solve(agg)
        res["pooled"] = {"n_questionnaires": len(pooled), "global_D1": pr, "vs_published_global": compare(pr, PUB["global"]),
                         "consistency_individual": consistency_stats([pooled[p]["mats"] for p in pooled]),
                         "knapsack": knapsack(pooled), "local_R|GOAL": dict(zip(R, loc["R|GOAL"].round(4).tolist()))}
        G = np.array([[res["panels"][f"r{r}"]["global_D1"][k] for k in K] for r in complete])
        res["across_panels"] = {"mean": dict(zip(K, G.mean(0).round(4).tolist())),
                                "min": dict(zip(K, G.min(0).round(4).tolist())),
                                "max": dict(zip(K, G.max(0).round(4).tolist())),
                                "first_place": [ranks(res["panels"][f"r{r}"]["global_D1"])[0] for r in complete],
                                "pairwise_tau": [float(kendalltau(G[i], G[j]).statistic)
                                                 for i in range(len(G)) for j in range(i + 1, len(G))]}
        # per-persona stability across replicates: rank of each control under E
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"anp_{args.tag}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    with open(RESULTS / f"anp_{args.tag}_global.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["control", "published"] + [f"{p}_D1" for p in res["panels"]] + (["pooled_D1"] if "pooled" in res else []))
        for k in K:
            w.writerow([k, PUB["global"][k]] + [round(res["panels"][p]["global_D1"][k], 4) for p in res["panels"]]
                       + ([round(res["pooled"]["global_D1"][k], 4)] if "pooled" in res else []))
    for p, v in res["panels"].items():
        print(p, {k: round(x, 4) for k, x in sorted(v["global_D1"].items(), key=lambda kv: -kv[1])})
        print("   vs published rho", round(v["vs_published_global"]["spearman"], 2), "tau",
              round(v["vs_published_global"]["kendall_tau"], 2), "max diff", round(v["vs_published_global"]["max_abs_diff"], 4))
        print("   CR agg mean", round(v["consistency_aggregated"]["mean_CR"], 4), "indiv mean",
              round(v["consistency_individual"]["mean_CR"], 4), "W", {c: round(x["W"], 2) for c, x in v["kendall_w"].items()})
        print("   LOEO", v["loeo"]["top2_invariant"], round(v["loeo"]["mean_abs_change"], 4), "MC top2",
              v["monte_carlo"]["top2_kept"], "knapsack", v["knapsack"]["greedy"], v["knapsack"]["optimum"])


if __name__ == "__main__":
    main()
