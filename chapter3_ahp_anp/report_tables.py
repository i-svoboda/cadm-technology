"""Build the Markdown tables of REPORT.md from the results files (results/report_tables.md).

Usage: python report_tables.py
"""
from __future__ import annotations

import json
import re

from common import RESULTS, load

AHP = load("ahp_2024_protocol.json")
ANP = load("anp_2026_protocol.json")
CNAME = {c["id"]: c["name"] for c in AHP["criteria"]}
ANAME = {a["id"]: a["name"] for a in AHP["alternatives"]}
KNAME = {n["id"]: n["name"] for n in ANP["clusters"]["K"]["nodes"]}


def rj(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def is_num(s) -> bool:
    t = str(s).split(" ")[0].replace("–", "").replace("%", "").replace("≈", "").replace("+", "").replace("-", "")
    return t.replace(".", "", 1).isdigit()


def table(head, rows):
    num = [i > 0 and all(is_num(r[i]) or str(r[i]) in ("", "–") for r in rows) for i in range(len(head))]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("--:" if n else ":--" for n in num) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def rank_of(d):
    order = sorted(d, key=lambda k: -d[k])
    return {k: order.index(k) + 1 for k in d}


def ahp_tables(s56, s61):
    panels = [(f"5.6 {b['batch'].split('_')[-1]}", b) for b in (s56 or {}).get("batches", [])]
    panels += [(f"6.1 {b['batch'].split('_')[-1]}", b) for b in (s61 or {}).get("batches", [])]
    pub = AHP["published"]
    out = []
    # criteria
    head = ["Criterion", "Published"] + [p for p, _ in panels]
    pr = rank_of(pub["criteria_weights_tacs_4dp"])
    rows = []
    for c in CNAME:
        row = [CNAME[c], f"{pub['criteria_weights_tacs_4dp'][c]:.3f} ({pr[c]})"]
        for _, b in panels:
            w = b["aggregated"]["criteria"]
            row.append(f"{w[c]:.3f} ({rank_of(w)[c]})")
        rows.append(row)
    out.append("**Table A1. Top-level criteria weights (rank in brackets)**\n\n" + table(head, rows))
    # alternatives
    pa = rank_of(pub["alternatives"])
    rows = []
    for a in ANAME:
        row = [ANAME[a], f"{pub['alternatives'][a]:.3f} ({pa[a]})"]
        for _, b in panels:
            w = b["aggregated"]["alternatives"]
            row.append(f"{w[a]:.3f} ({rank_of(w)[a]})")
        rows.append(row)
    out.append("**Table A2. Global priorities of the alternatives (rank in brackets)**\n\n" + table(
        ["Alternative", "Published"] + [p for p, _ in panels], rows))
    # agreement and checks
    rows = []
    for p, b in panels:
        v = b["vs_published"]
        a = b["aggregated"]
        rows.append([p, f(v["criteria"]["spearman"], 2), f(v["sub_global"]["spearman"], 2),
                     f(v["alternatives"]["spearman"], 2), f(v["alternatives"]["max_abs_diff"]),
                     "yes" if v["alternatives"]["same_first"] else "no",
                     f(v["criteria_matrix"]["pearson_log_upper"], 2),
                     f(a["criteria_consistency"]["CR"]),
                     f"{max(0.0, min(a['sub_CR'].values())):.3f}–{max(a['sub_CR'].values()):.3f}",
                     f"{max(0.0, min(a['alt_CR'].values())):.3f}–{max(a['alt_CR'].values()):.3f}",
                     f"{b['individual_CR']['share_below_0.01']:.0%}",
                     "yes" if b["loeo"]["top_unchanged"] else "no",
                     f"{b['persona_agreement']['criteria']['W']:.2f} / {b['persona_agreement']['alternatives']['W']:.2f}",
                     sum(b["reminders"].values())])
    rows.append(["Published", "", "", "", "", "", "", "0.016", "0.002–0.02", "0.002–0.03", "most", "yes", "–", "occasional"])
    out.append("**Table A3. Agreement with the published values and consistency**\n\n"
               "Spearman's ρ is computed against the published criteria weights (7), global sub-criteria weights "
               "(21) and alternative priorities (5). \"Matrix r\" is the Pearson correlation of the log entries above "
               "the diagonal of the aggregated criteria matrix and Table 1. The CR columns refer to aggregated matrices, and the "
               "individual share counts matrices of 3×3 and larger. W is Kendall's coefficient of concordance of "
               "the seven personas' rankings (criteria / alternatives).\n\n" + table(
                   ["Panel", "ρ criteria", "ρ sub-criteria", "ρ alternatives", "Largest gap", "Same first",
                    "Matrix r", "CR criteria", "CR sub-criteria", "CR alternatives", "Individual CR < 0.01",
                    "LOEO first unchanged", "Persona W", "Reminders"], rows))
    if s56 and "across_panels" in s56:
        ap = s56["across_panels"]
        rows = [[ANAME[a], f"{ap['alternatives_min'][a]:.3f}–{ap['alternatives_max'][a]:.3f}"] for a in ANAME]
        out.append("**Table A4. Range of alternative priorities over the gpt-5.6-sol panels**\n\n" + table(
            ["Alternative", "Min–max"], rows) + f"\n\nFirst place by panel: {', '.join(ap['first_place'])}. "
            f"Pairwise Kendall's τ between panels: {', '.join(f'{t:.2f}' for t in ap['pairwise_kendall_tau_alternatives'])}.")
    return "\n\n".join(out)


def structure_tables(st56, st61):
    out = []
    rows = []
    for tag, st in (("5.6", st56), ("6.1", st61)):
        for b, v in ((st or {}).get("pub45") or {}).items():
            rows.append([f"{tag} {b.split('_')[-1]}", v["n_overlap"], "; ".join(v["top7"]),
                         ", ".join(f"{CNAME[c] if c in CNAME else c} {r}" for c, r in
                                   sorted(((c, r) for c, r in v["rank_of_published_criteria"].items()),
                                          key=lambda x: x[1]))])
    if rows:
        out.append("**Table A5. Criteria selected from the published 45 candidates (sum of 1–9 scores, top 7)**\n\n"
                   + table(["Run", "Published criteria among the 7", "Top 7", "Rank of each published criterion"],
                           [[r[0], r[1], r[2], r[3].replace("Social Engineering Awareness", "SEA")] for r in rows]))
    rows = []
    for tag, st in (("5.6", st56), ("6.1", st61)):
        for b, v in ((st or {}).get("fresh") or {}).items():
            cp = v.get("criteria_proposed", {})
            res = v.get("result", {})
            rows.append([f"{tag} {b.split('_')[-1]}", f"{cp.get('n')} → {cp.get('unique')}",
                         cp.get("exact_in_appendix_c"), "; ".join(c["name"] for c in v.get("criteria_selected", [])),
                         "; ".join(a["name"] for a in v.get("alternatives_selected", [])),
                         "; ".join(f"{k} {x:.3f}" for k, x in list(res.get("alternatives", {}).items())[:5]),
                         f(res.get("criteria_CR"))])
    if rows:
        out.append("**Table A6. End-to-end runs: structure chosen by the personas and the result**\n\n" + table(
            ["Run", "Criteria proposed → unique", "Same name as in Appendix C", "Criteria selected",
             "Alternatives selected", "Global priorities", "CR criteria"], rows) +
            "\n\nPublished: 49 criteria proposed, 45 unique. Criteria selected: SEA, PAC, Audit Trails, Behavior Analysis, ORC, "
            "Psychological Profiling, SLAs. Alternatives as in Table A2.")
    return "\n\n".join(out)


def anp_tables(a56, a61):
    pub = ANP["published"]
    out = []
    panels = [(f"5.6 {k}", v) for k, v in (a56 or {}).get("panels", {}).items()]
    panels += [(f"6.1 {k}", v) for k, v in (a61 or {}).get("panels", {}).items()]
    pr = rank_of(pub["global"])
    rows = []
    for k in KNAME:
        row = [f"{KNAME[k]} ({k})", f"{pub['global'][k]:.3f} ({pr[k]})"]
        for _, v in panels:
            row.append(f"{v['global_D1'][k]:.3f} ({rank_of(v['global_D1'])[k]})")
        for _, v in panels:
            row.append(f"{v['global_reversed_synergy'][k]:.3f} ({rank_of(v['global_reversed_synergy'])[k]})")
        rows.append(row)
    head = ["Control", "Published"] + [f"{p} D1" for p, _ in panels] + [f"{p} reversed" for p, _ in panels]
    out.append("**Table N1. Global priorities of the controls (rank in brackets)**\n\nD1 reads the synergy links as "
               "the paper defines blocks (row cluster influences column cluster). The \"reversed\" columns read them the other "
               "way.\n\n" + table(head, rows))
    rows = []
    for p, v in panels:
        lc = v["vs_published_local"]
        ca, ci = v["consistency_aggregated"], v["consistency_individual"]
        W = v["kendall_w"]
        mc = v["monte_carlo"]
        rows.append([p, f(v["vs_published_global"]["spearman"], 2), f(v["vs_published_global_reversed_synergy"]["spearman"], 2),
                     f(v["vs_published_global_hier"]["spearman"], 2), f(lc["R|GOAL"]["spearman"], 2),
                     f(lc["K|R:E"]["spearman"], 2),
                     f"{ca['mean_CR']:.3f} ({ca['sd_CR']:.3f})", f"{ci['mean_CR']:.3f}", f"{ca['mean_K']:.2f}",
                     f"{min(x['W'] for x in W.values()):.2f}–{max(x['W'] for x in W.values()):.2f}",
                     f"{v['loeo']['mean_abs_change']:.4f}", f"{v['loeo']['mean_kendall_tau']:.2f}",
                     f"{mc['top2_kept']:.0%}", f"{mc['max_weight_change']:.3f}", f"{v['diagnostics']['lambda2']:.2f}"])
    rows.append(["Published", "", "", "", "", "", "0.006 (0.004)", "–", "0.034", "0.62–0.77", "0.0039", "0.90",
                 "100%", "0.015", "0.947"])
    out.append("**Table N2. Agreement with ITS 14(1) and the robustness checks**\n\nSpearman's ρ against the published "
               "global priorities (D1, reversed synergies, hierarchical composition) and against the Table 3 vectors "
               "(criteria with respect to the goal, and controls with respect to E). CR and the Koczkodaj index K are "
               "means over the 31 aggregated matrices of size 3 and larger. The LOEO columns give the mean absolute "
               "change of the global weights and the mean Kendall's τ with the base ranking. The MC columns give the "
               "share of 1000 trials that keep the base top two and the largest weight change.\n\n" + table(
                   ["Panel", "ρ D1", "ρ reversed", "ρ hierarchical", "ρ criteria (goal)", "ρ controls under E",
                    "CR aggregated (SD)", "CR individual", "K", "Kendall's W", "LOEO change", "LOEO τ",
                    "MC top two kept", "MC largest change", "λ2"], rows))
    rows = []
    t3 = pub["table3_local"]["R|GOAL"]["vector"]
    rows.append(["Published"] + [f"{x:.3f}" for x in t3])
    for p, v in panels:
        rows.append([p] + [f"{v['local']['R|GOAL'][c]:.3f}" for c in ["E", "T", "U", "C", "P"]])
    out.append("**Table N3. Criteria weights with respect to the goal (Table 3, first row)**\n\n" + table(
        ["Panel", "E", "T", "U", "C", "P"], rows))
    rows = []
    for p, v in panels:
        ks = v["knapsack"]
        sc = v["scenarios_x2"]
        rows.append([p, ", ".join(ks["greedy"]), f"{ks['greedy_cost']:.1f}", f"{ks['greedy_reduction']:.0f}",
                     ", ".join(ks["optimum"]), f"{ks['published_set_reduction']:.0f}", f"{ks['eal_total']:.0f}",
                     "; ".join(f"{s}: " + ", ".join(sorted(x, key=lambda k: -x[k])[:3]) for s, x in sc.items())])
    rows.append(["Published", "MON, IDM, IR, SUP", "10.2", "≈27", "–", "≈27", "not given",
                 "SC: MON, SEG, SUP; EXP: SEG, MON, IDM; INS+SAB: MON, IDM, SEG"])
    out.append("**Table N4. Budget portfolio (B = 10.2 M USD, published costs) and threat scenarios (top three)**\n\n"
               "The replication uses the panel's own coverage shares and expected annual losses (EAL, M USD), since the paper "
               "did not publish its values. Scenario multiplier 2 for the dominant threats (the paper's multipliers "
               "are not given).\n\n" + table(["Panel", "Greedy set", "Cost", "Risk reduction", "Optimal set",
                                              "Reduction of the published set", "Total EAL", "Scenarios"], rows))
    if a56 and "published_composition_check" in a56:
        pc = a56["published_composition_check"]
        out.append("**Table N5. Hierarchical composition of the published Table 3 vectors against the published global "
                   "priorities**\n\n" + table(["Control", "Composition of Table 3", "Published Table 4", "Difference"],
                                              [[k, f"{pc[k]:.4f}", f"{pub['global'][k]:.4f}",
                                                f"{pc[k] - pub['global'][k]:+.4f}"] for k in KNAME]))
    return "\n\n".join(out)


def cost_table(cs):
    rows = []
    for b, v in cs["batches"].items():
        if b.startswith("_"):
            continue
        rows.append([b, v["calls"], f"{v['wall_s'] / 60:.0f}", f"{v['input'] / 1e6:.2f}", f"{v['cached'] / 1e6:.2f}",
                     f"{v['output'] / 1e3:.0f}", f"{v['reasoning'] / 1e3:.0f}", "/".join(v["models"]),
                     "/".join(v["tiers"]), v["agents_md_loaded"]])
    shown = [v for k, v in cs["batches"].items() if not k.startswith("_")]
    t = {k: sum(v[k] for v in shown) for k in ("calls", "wall_s", "input", "cached", "output", "reasoning")}
    rows.append(["Total", t["calls"], f"{t['wall_s'] / 60:.0f}", f"{t['input'] / 1e6:.2f}", f"{t['cached'] / 1e6:.2f}",
                 f"{t['output'] / 1e3:.0f}", f"{t['reasoning'] / 1e3:.0f}", "", "", ""])
    return ("**Table C1. Calls, model time and tokens per batch**\n\n" + table(
        ["Batch", "Calls", "Model time, min", "Input, M tokens", "of which cached", "Output, k tokens",
         "of which reasoning", "Model", "Tier", "AGENTS.md loaded"], rows) +
        f"\n\nCredit balance in every session record: {', '.join(cs['credit_balances_seen'])}.")


def main():
    parts = ["# Report tables (generated by scripts/report_tables.py)"]
    parts.append(ahp_tables(rj("ahp_s56.json"), rj("ahp_s61.json")))
    parts.append(structure_tables(rj("ahp_structure_s56.json"), rj("ahp_structure_s61.json")))
    parts.append(anp_tables(rj("anp_s56.json"), rj("anp_s61.json")))
    cs = rj("cost_summary.json")
    if cs:
        parts.append(cost_table(cs))
    text = "\n\n".join(p for p in parts if p) + "\n"
    (RESULTS / "report_tables.md").write_text(text, encoding="utf-8")
    # fill {{TABLE:A1}} style placeholders of the report template
    blocks = {}
    for blk in text.split("\n\n**Table ")[1:]:
        key = blk.split(".", 1)[0]
        blocks[key] = "**Table " + blk.strip()
    tpl = RESULTS.parent / "scripts" / "report_template.md"
    if tpl.exists():
        rep = tpl.read_text(encoding="utf-8")
        for key, blk in blocks.items():
            rep = rep.replace("{{TABLE:" + key + "}}", blk)
        missing = sorted(set(re.findall(r"{{TABLE:([A-Z0-9]+)}}", rep)))
        (RESULTS.parent / "REPORT.md").write_text(rep, encoding="utf-8")
        print("REPORT.md written; tables without data:", missing)
    print("tables:", sorted(blocks))


if __name__ == "__main__":
    main()
