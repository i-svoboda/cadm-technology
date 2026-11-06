"""SMAA over LLM-elicited questionnaires for AHP and ANP cases.

Panel-sampling schemes (each draw builds one 7-member panel and aggregates it by AIJ):
  naive        all questionnaires pooled into one panel (no spread, the usual practice)
  iid          7 questionnaires drawn at random from the pool
  hier_fixed   one questionnaire per persona, drawn from that persona's runs
  hier_random  personas resampled with replacement, then one run per drawn persona
ANP structure modes (optional synergy links):
  fixed        links included when the pooled mean probability is >= 0.5
  sampled      each link included with the panel's mean stated probability
  only         judgments fixed at the pooled aggregate, links sampled

Usage: python scripts/smaa.py --case <case.json> --runs <batch dir> [--draws 10000] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anp_case  # noqa: E402
import smaa_ahp  # noqa: E402
from elicit import to_log  # noqa: E402


def load(run: Path) -> dict:
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = json.loads((run / "items.json").read_text(encoding="utf-8"))
    ans = json.loads((run / "last_message.json").read_text(encoding="utf-8"))
    logj, missing, _ = to_log(items, ans)
    links = {}
    if "links" in ans:
        p = {l["link_id"]: l["probability"] for l in ans["links"]}
        links = {(it["source"], it["target"]): float(np.clip(p.get(it["link_id"], 0.0), 0, 1)) for it in items if "link_id" in it}
    return {"meta": man, "logj": logj, "links": links, "complete": not missing}


CLUSTER_WEIGHTS = "its"     # ANP only: "its" (ITS 14(1) prior vector) or "elicited" (CL|NET block)


def priorities(case: dict, logj: dict, links_on: dict | None) -> np.ndarray:
    if case["method"] == "ANP":
        pr = anp_case.control_priorities(case, logj, links_on or {}, cluster_weights=CLUSTER_WEIGHTS)
        return np.array([pr[n["id"]] for n in case["clusters"]["K"]["nodes"]])
    return smaa_ahp.ahp_global(case, smaa_ahp.build(case, logj))


def labels(case: dict) -> list[str]:
    if case["method"] == "ANP":
        return [n["id"] for n in case["clusters"]["K"]["nodes"]]
    return [a["id"] for a in case["alternatives"]]


def mean_links(qs: list[dict]) -> dict:
    keys = qs[0]["links"].keys()
    return {k: float(np.mean([q["links"][k] for q in qs])) for k in keys}


def run_smaa(case: dict, qs: list[dict], scheme: str, structure: str, draws: int, rng: np.random.Generator) -> dict:
    alts = labels(case)
    n = len(alts)
    by_p: dict[str, list[dict]] = defaultdict(list)
    for q in qs:
        by_p[q["meta"]["persona"]].append(q)
    personas = sorted(by_p)
    pooled = smaa_ahp.aggregate([q["logj"] for q in qs])
    pooled_links = mean_links(qs) if qs[0]["links"] else {}
    b = np.zeros((n, n)); pwin = np.zeros((n, n)); g_all = np.zeros((draws, n))
    for d in range(draws if scheme != "naive" or structure in ("sampled", "only") else 1):
        if scheme == "naive" or structure == "only":
            panel = qs
            logj = pooled
        else:
            if scheme == "iid":
                panel = [qs[i] for i in rng.integers(0, len(qs), len(personas))]
            elif scheme == "hier_fixed":
                panel = [by_p[p][rng.integers(0, len(by_p[p]))] for p in personas]
            elif scheme == "hier_random":
                panel = [by_p[p][rng.integers(0, len(by_p[p]))] for p in rng.choice(personas, len(personas), replace=True)]
            else:
                raise ValueError(scheme)
            logj = smaa_ahp.aggregate([q["logj"] for q in panel])
        links_on = None
        if pooled_links:
            if structure == "fixed":
                links_on = {k: v >= 0.5 for k, v in pooled_links.items()}
            else:
                pl = mean_links(panel) if structure == "sampled" else pooled_links
                links_on = {k: bool(rng.random() < v) for k, v in pl.items()}
        g = priorities(case, logj, links_on)
        g_all[d] = g
        b[np.arange(n), smaa_ahp.ranks_of(g)] += 1
        pwin += g[:, None] > g[None, :]
    k = d + 1
    g_all = g_all[:k]
    return {"rank_acceptability": (b / k).round(4).tolist(), "pairwise_win": (pwin / k).round(4).tolist(),
            "mean_priority": g_all.mean(axis=0).round(4).tolist(), "sd_priority": g_all.std(axis=0).round(4).tolist(),
            "draws": k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--out")
    ap.add_argument("--cluster-weights", choices=["its", "elicited"], default="its")
    args = ap.parse_args()
    global CLUSTER_WEIGHTS
    CLUSTER_WEIGHTS = args.cluster_weights
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    qs = [load(d) for d in sorted(Path(args.runs).iterdir()) if (d / "last_message.json").exists()]
    dropped = [q["meta"]["session_id"] for q in qs if not q["complete"]]
    qs = [q for q in qs if q["complete"]]
    rng = np.random.default_rng(args.seed)
    alts = labels(case)
    report = {"case": case["case_id"], "n_questionnaires": len(qs), "dropped_incomplete": dropped, "labels": alts,
              "cluster_weights": args.cluster_weights if case["method"] == "ANP" else None, "results": {}}
    structures = ["fixed", "sampled", "only"] if case["method"] == "ANP" else ["fixed"]
    for scheme in ["naive", "iid", "hier_fixed", "hier_random"]:
        for st in structures:
            if (st == "only") != (scheme == "naive") and st != "fixed":
                continue                       # naive: fixed and only; panels: fixed and sampled
            res = run_smaa(case, qs, scheme, st, args.draws, rng)
            report["results"][f"{scheme}/{st}"] = res
            print(f"\n== {scheme} / structure {st}  (draws {res['draws']})")
            for a, row in zip(alts, res["rank_acceptability"]):
                print(f"  {a:5s}", " ".join(f"{x:5.2f}" for x in row))
    report["variance_components"] = smaa_ahp.variance_components(qs)
    print("\nvariance components:", json.dumps(report["variance_components"], indent=1, default=float))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
