"""Numbers quoted in chapter 2 of the dissertation, computed from the stored results of this replication.

Writes results/dissertation_ch2_numbers.json. Every value in subsections 2.2.2-2.2.4 and 2.3.4-2.3.7 of the
draft comes from this file.

Usage: python dissertation_numbers.py
"""
from __future__ import annotations

import json

import numpy as np

from common import RESULTS, RUNS, consistency, load
from ahp_analysis import load_persona

PROT = load("ahp_2024_protocol.json")
CIDS = [c["id"] for c in PROT["criteria"]]
SUBS = [s["id"] for c in PROT["criteria"] for s in c["sub_criteria"]]
ALTS = [a["id"] for a in PROT["alternatives"]]
K = ["MON", "SEG", "IDM", "IR", "SUP", "BAK", "PHY"]


def rj(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def stats(values):
    v = np.array(values, dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def limit_checks():
    """Power-iteration limit against the eigenvector for eigenvalue 1 and against damping (formula 2.14)."""
    import anp_analysis as aa
    out = {}
    for tag, batch in (("s56", "anp_s56"), ("s61", "anp_s61")):
        data = aa.load_batch(batch)
        for r, panel in sorted(data.items()):
            agg = aa.geo_agg(panel)
            loc = aa.local_vectors(agg)
            idx, W, Ww = aa.CASE.supermatrix(loc, "D1")
            pi, diag = aa.CASE.limit(Ww)
            vals, vecs = np.linalg.eig(Ww)
            k = int(np.argmin(np.abs(vals - 1)))
            v = np.abs(vecs[:, k].real); v = v / v.sum()
            n = Ww.shape[0]
            d = 1e-6
            pd, _ = aa.CASE.limit((1 - d) * Ww + d * np.ones((n, n)) / n)
            out[f"{tag}_r{r}"] = {"eig_diff_L1": float(np.abs(pi - v).sum()), "damping_max_diff": float(np.abs(pi - pd).max()),
                                  "iterations": diag["iterations"], "lambda2": diag["lambda2"]}
    return out


def knapsack_sensitivity(seed=2026, trials=1000):
    """Coverage shares scaled by 0.8 and 1.2 (capped at 1) and by independent factors from U(0.8, 1.2)."""
    import itertools
    import anp_analysis as aa
    costs = aa.PUB["knapsack"]["costs_musd"]
    B = aa.PUB["knapsack"]["budget_musd"]
    rng = np.random.default_rng(seed)
    out = {}
    for tag, batch in (("s56", "anp_s56"), ("s61", "anp_s61")):
        data = aa.load_batch(batch)
        for r, panel in sorted(data.items()):
            base = aa.knapsack(panel)
            alpha = {tuple(k.split("->")): v for k, v in base["alpha"].items()}
            eal = base["eal"]

            def best(al):
                def f(S):
                    return sum(eal[t] * (1 - np.prod([1 - al[(c, t)] for c in S])) for t in aa.H)
                bs, bv = (), 0.0
                for m in range(1, 8):
                    for comb in itertools.combinations(aa.K, m):
                        if sum(costs[c] for c in comb) <= B + 1e-9 and f(comb) > bv:
                            bs, bv = comb, f(comb)
                return set(bs), bv
            s0, v0 = best(alpha)
            res = {"base_set": sorted(s0), "base_value": v0}
            for fct in (0.8, 1.2):
                s1, v1 = best({k: min(1.0, v * fct) for k, v in alpha.items()})
                res[f"x{fct}"] = {"same_set": s1 == s0, "value_change": (v1 - v0) / v0}
            same, chg = 0, []
            for _ in range(trials):
                s1, v1 = best({k: min(1.0, v * rng.uniform(0.8, 1.2)) for k, v in alpha.items()})
                same += s1 == s0
                chg.append(abs(v1 - v0) / v0)
            res["random_pm20"] = {"same_set_share": same / trials, "max_value_change": float(max(chg)),
                                  "mean_value_change": float(np.mean(chg))}
            out[f"{tag}_r{r}"] = res
    return out


def main():
    a56, a61 = rj("ahp_s56.json")["batches"], rj("ahp_s61.json")["batches"][0]
    st56, st61 = rj("ahp_structure_s56.json"), rj("ahp_structure_s61.json")
    n56, n61 = rj("anp_s56.json"), rj("anp_s61.json")
    cost = rj("cost_summary.json")["batches"]
    out = {"ahp": {}, "anp": {}}
    A = out["ahp"]
    A["criteria_s56"] = {c: stats([b["aggregated"]["criteria"][c] for b in a56]) for c in CIDS}
    A["criteria_s61"] = a61["aggregated"]["criteria"]
    A["sub_global_s56"] = {s: stats([b["aggregated"]["sub_global"][s] for b in a56]) for s in SUBS}
    A["sub_global_s61"] = a61["aggregated"]["sub_global"]
    A["alternatives_s56"] = {a: stats([b["aggregated"]["alternatives"][a] for b in a56]) for a in ALTS}
    A["alternatives_s61"] = a61["aggregated"]["alternatives"]
    A["criteria_CR"] = {"s56": [b["aggregated"]["criteria_consistency"]["CR"] for b in a56],
                        "s61": a61["aggregated"]["criteria_consistency"]["CR"],
                        "lambda_s56": [b["aggregated"]["criteria_consistency"]["lambda_max"] for b in a56]}
    A["sub_CR_max"] = {"s56": max(max(b["aggregated"]["sub_CR"].values()) for b in a56),
                       "s61": max(a61["aggregated"]["sub_CR"].values())}
    A["alt_CR_max"] = {"s56": max(max(b["aggregated"]["alt_CR"].values()) for b in a56),
                       "s61": max(a61["aggregated"]["alt_CR"].values())}
    A["individual_CR"] = {"s56": [b["individual_CR"] for b in a56], "s61": a61["individual_CR"]}
    A["persona_W_criteria"] = {"s56": [b["persona_agreement"]["criteria"]["W"] for b in a56],
                               "s61": a61["persona_agreement"]["criteria"]["W"]}
    A["persona_tops"] = {"s56": [{p: (v["top_criterion"], v["top_alternative"]) for p, v in b["individual"].items()}
                                 for b in a56],
                         "s61": {p: (v["top_criterion"], v["top_alternative"]) for p, v in a61["individual"].items()}}
    A["loeo_first_changes"] = {
        "s56": [sum(v["top"] != max(b["aggregated"]["alternatives"], key=b["aggregated"]["alternatives"].get)
                    for v in b["loeo"]["runs"].values()) for b in a56],
        "s61": sum(v["top"] != max(a61["aggregated"]["alternatives"], key=a61["aggregated"]["alternatives"].get)
                   for v in a61["loeo"]["runs"].values())}
    A["reminders"] = {"s56": [sum(b["reminders"].values()) for b in a56], "s61": sum(a61["reminders"].values())}
    # example matrix: Dr. Ava Chen, sub-criteria of social engineering awareness, panel 1
    mats, _ = load_persona(RUNS / "ahp_published_s56_r1", "A1")
    M = np.array(mats["sub|SEA"])
    conv = json.loads((RUNS / "ahp_published_s56_r1" / "A1" / "conversation.json").read_text(encoding="utf-8"))
    rat = [t for t in conv["turns"] if t["name"] == "sub_matrices"][0]["parsed"]["matrices"]["sub|SEA"]["rationale"]
    A["example_A1_SEA"] = {"matrix": M.round(4).tolist(), "CR": consistency(M)["CR"], "rationale": rat}
    A["pub45"] = {b: {"top7": v["top7"], "rank_of_published": v["rank_of_published_criteria"]}
                  for b, v in {**st56["pub45"], **st61["pub45"]}.items()}
    A["fresh"] = {b: {"criteria_proposed": v["criteria_proposed"],
                      "criteria_selected": [c["name"] for c in v["criteria_selected"]],
                      "alternatives_selected": [x["name"] for x in v["alternatives_selected"]],
                      "result": v.get("result"), "individual_CR": v.get("individual_CR")}
                  for b, v in st56["fresh"].items()}
    A["guide_first_answers"] = {b: v["guide_G1"][:200] for b, v in {**st56["guide"], **st61["guide"]}.items()}
    A["resources"] = {b: {k: cost[b][k] for k in ("calls", "wall_s", "input", "cached", "output", "reasoning")}
                      for b in cost if b.startswith("ahp_published")}

    N = out["anp"]
    panels = list(n56["panels"].values())
    p61 = n61["panels"]["r1"]
    N["global_s56"] = {k: stats([p["global_D1"][k] for p in panels]) for k in K}
    N["global_s61"] = p61["global_D1"]
    N["reversed_s56"] = {k: stats([p["global_reversed_synergy"][k] for p in panels]) for k in K}
    N["reversed_s61"] = p61["global_reversed_synergy"]
    N["hierarchical_s56"] = {k: stats([p["global_hierarchical"][k] for p in panels]) for k in K}
    N["goal_s56"] = {c: stats([p["local"]["R|GOAL"][c] for p in panels]) for c in ["E", "T", "U", "C", "P"]}
    N["goal_s61"] = p61["local"]["R|GOAL"]
    N["E_s56"] = {k: stats([p["local"]["K|R:E"][k] for p in panels]) for k in K}
    N["consistency"] = {"s56": [p["consistency_aggregated"] for p in panels], "s61": p61["consistency_aggregated"],
                        "s56_individual": [p["consistency_individual"] for p in panels],
                        "s61_individual": p61["consistency_individual"]}
    N["lambda2"] = {"s56": [p["diagnostics"]["lambda2"] for p in panels], "s61": p61["diagnostics"]["lambda2"]}
    N["loeo_r1"] = n56["panels"]["r1"]["loeo"]
    N["loeo_summary"] = {"s56": [{k: p["loeo"][k] for k in ("mean_abs_change", "mean_kendall_tau", "top2_invariant",
                                                           "first_invariant")} for p in panels],
                         "s61": {k: p61["loeo"][k] for k in ("mean_abs_change", "mean_kendall_tau", "top2_invariant",
                                                             "first_invariant")}}
    N["kendall_w"] = {"s56": [p["kendall_w"] for p in panels], "s61": p61["kendall_w"]}
    N["monte_carlo"] = {"s56": [p["monte_carlo"] for p in panels], "s61": p61["monte_carlo"]}
    N["scenarios_s56"] = {s: {k: stats([p["scenarios_x2"][s][k] for p in panels]) for k in K}
                          for s in ["SC", "EXP", "INS+SAB"]}
    N["scenarios_s61"] = p61["scenarios_x2"]
    N["knapsack"] = {"s56": [{k: p["knapsack"][k] for k in ("greedy", "greedy_cost", "greedy_reduction", "optimum",
                                                            "optimum_reduction", "published_set_reduction",
                                                            "eal_total", "eal")} for p in panels],
                     "s61": {k: p61["knapsack"][k] for k in ("greedy", "greedy_cost", "greedy_reduction", "optimum",
                                                             "optimum_reduction", "published_set_reduction",
                                                             "eal_total", "eal")}}
    N["resources"] = {b: {k: cost[b][k] for k in ("calls", "wall_s", "input", "cached", "output", "reasoning")}
                      for b in ("anp_s56", "anp_s61")}
    N["limit_checks"] = limit_checks()
    N["knapsack_alpha_pm20"] = knapsack_sensitivity()
    (RESULTS / "dissertation_ch2_numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float),
                                                           encoding="utf-8")
    print("written results/dissertation_ch2_numbers.json")


if __name__ == "__main__":
    main()
