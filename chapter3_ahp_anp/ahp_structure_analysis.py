"""Structure-building arms of the AHP replication: the 45-criteria scoring, the AHP Guide and the
end-to-end ("fresh") pipeline, compared with arXiv 2402.07404.

Usage:
  python ahp_structure_analysis.py --pub45 ahp_pub45_s56_r1 ahp_pub45_s56_r2 --guide ahp_guide_s56_r1
      --fresh ahp_fresh_s56_r1 --tag s56   (each option takes any number of batches)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ahp_analysis import load_persona
from common import RESULTS, RUNS, consistency, geo_mean, load, name_similarity, norm_name, priorities_approx

PROT = load("ahp_2024_protocol.json")
PUB_CRIT = [c["name"] for c in PROT["criteria"]]
PUB_SUBS = [s["name"] for c in PROT["criteria"] for s in c["sub_criteria"]]
PUB_ALTS = [a["name"] for a in PROT["alternatives"]]
APPC = [x for v in PROT["appendix_c_initial_criteria"].values() for x in v]
PERSONAS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]


def best_match(x: str, pool: list[str]) -> tuple[str, float]:
    s = [(name_similarity(x, p), p) for p in pool]
    v, p = max(s)
    return p, v


def pub45(batch: str) -> dict:
    items = PROT["published_45_criteria"]
    scores = {}
    for pid in PERSONAS:
        f = RUNS / batch / pid / "conversation.json"
        if f.exists():
            t = json.loads(f.read_text(encoding="utf-8"))["turns"]
            acc = [x for x in t if x.get("accepted")]
            if acc:
                scores[pid] = acc[-1]["parsed"]["scores"]
    totals = {}
    for i, x in enumerate(items, 1):
        s = [scores[p][str(i)] for p in scores if str(i) in scores[p]]
        totals[x] = {"total": sum(s), "nines": sum(v == 9 for v in s), "n": len(s), "order": i}
    ranked = sorted(totals, key=lambda x: (-totals[x]["total"], -totals[x]["nines"], totals[x]["order"]))
    top7 = ranked[:7]
    pub_rank = {c: ranked.index(c) + 1 for c in PUB_CRIT}
    return {"personas": sorted(scores), "top7": top7, "overlap_with_published": sorted(set(top7) & set(PUB_CRIT)),
            "n_overlap": len(set(top7) & set(PUB_CRIT)), "rank_of_published_criteria": pub_rank,
            "totals": {x: totals[x]["total"] for x in ranked}}


def guide(batch: str) -> dict:
    conv = json.loads((RUNS / batch / "GUIDE" / "conversation.json").read_text(encoding="utf-8"))
    out = {}
    for t in conv["turns"]:
        if t.get("accepted"):
            out[t["name"]] = (RUNS / batch / "GUIDE" / t["folder"] / "last_message.txt").read_text(encoding="utf-8")
    return out


def solve_h(mats: dict, hier: dict) -> dict:
    crit = hier["criteria"]
    alts = [a["id"] for a in hier["alternatives"]]
    wc = dict(zip([c["id"] for c in crit], priorities_approx(mats["criteria|GOAL"])))
    sub_g, alt = {}, {a: 0.0 for a in alts}
    for c in crit:
        ids = [s["id"] for s in c["sub_criteria"]]
        wl = dict(zip(ids, priorities_approx(mats[f"sub|{c['id']}"])))
        for s in ids:
            sub_g[s] = wc[c["id"]] * wl[s]
            for a, w in zip(alts, priorities_approx(mats[f"alt|{s}"])):
                alt[a] += sub_g[s] * w
    return {"criteria": wc, "sub_global": sub_g, "alternatives": alt}


def fresh(batch: str) -> dict:
    st = RUNS / batch / "stages"
    S = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in st.glob("*.json")}
    out = {}
    if "01_criteria_proposed" in S:
        props = [x for _, x in S["01_criteria_proposed"]["proposals"]]
        exact = [x for x in props if any(norm_name(x) == norm_name(y) for y in APPC)]
        close = [x for x in props if best_match(x, APPC)[1] >= 0.5]
        out["criteria_proposed"] = {"n": len(props), "unique": len(S["01_criteria_proposed"]["unique"]),
                                    "exact_in_appendix_c": len(exact), "close_to_appendix_c": len(close),
                                    "appendix_c_unique_after_dedup": 45}
    if "02_criteria_scored" in S:
        sel = S["02_criteria_scored"]["selected"]
        out["criteria_selected"] = [{"name": c, "closest_published": best_match(c, PUB_CRIT)[0],
                                     "similarity": round(best_match(c, PUB_CRIT)[1], 2)} for c in sel]
        out["n_selected_matching_published"] = sum(best_match(c, PUB_CRIT)[1] >= 0.5 for c in sel)
    if "04_subcriteria_scored" in S:
        subs = [s for v in S["04_subcriteria_scored"]["selected"].values() for s in v]
        out["subcriteria_selected"] = S["04_subcriteria_scored"]["selected"]
        out["n_sub_matching_published"] = sum(best_match(s, PUB_SUBS)[1] >= 0.5 for s in subs)
    if "05_alternatives_proposed" in S:
        out["alternatives_proposed"] = {"n": len(S["05_alternatives_proposed"]["proposals"]),
                                        "unique": len(S["05_alternatives_proposed"]["unique"])}
    if "06_alternatives_scored" in S:
        sel = S["06_alternatives_scored"]["selected"]
        out["alternatives_selected"] = [{"name": a, "closest_published": best_match(a, PUB_ALTS)[0],
                                         "similarity": round(best_match(a, PUB_ALTS)[1], 2)} for a in sel]
    if "07_hierarchy" in S:
        hier = S["07_hierarchy"]
        keys = ["criteria|GOAL"] + [f"sub|{c['id']}" for c in hier["criteria"]] + \
               [f"alt|{s['id']}" for c in hier["criteria"] for s in c["sub_criteria"]]
        per = {}
        for pid in PERSONAS:
            try:
                m, _ = load_persona(RUNS / batch, pid)
            except FileNotFoundError:
                continue
            if all(k in m for k in keys):
                per[pid] = m
        if len(per) == 7:
            agg = {k: geo_mean([per[p][k] for p in per]) for k in keys}
            res = solve_h(agg, hier)
            names = {a["id"]: a["name"] for a in hier["alternatives"]}
            cn = {c["id"]: c["name"] for c in hier["criteria"]}
            out["result"] = {"criteria": {cn[k]: round(v, 4) for k, v in sorted(res["criteria"].items(), key=lambda x: -x[1])},
                             "alternatives": {names[k]: round(v, 4) for k, v in sorted(res["alternatives"].items(),
                                                                                       key=lambda x: -x[1])},
                             "criteria_CR": consistency(agg["criteria|GOAL"])["CR"],
                             "max_CR": max(consistency(agg[k])["CR"] for k in keys)}
            crs = [consistency(per[p][k])["CR"] for p in per for k in keys if per[p][k].shape[0] > 2]
            out["individual_CR"] = {"median": float(np.median(crs)), "max": float(max(crs)),
                                    "share_below_0.01": float(np.mean(np.array(crs) < 0.01))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pub45", nargs="*", default=[])
    ap.add_argument("--guide", nargs="*", default=[])
    ap.add_argument("--fresh", nargs="*", default=[])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    res = {}
    res["pub45"] = {b: pub45(b) for b in a.pub45}
    res["guide"] = {b: guide(b) for b in a.guide}
    res["fresh"] = {b: fresh(b) for b in a.fresh}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"ahp_structure_{a.tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float),
                                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "guide"}, indent=1, ensure_ascii=False, default=float)[:6000])


if __name__ == "__main__":
    main()
