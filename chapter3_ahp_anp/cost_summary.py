"""Calls, time, tokens and quota readings per batch, from the turn manifests.

Usage: python cost_summary.py   (writes results/cost_summary.json and .csv)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

from common import RESULTS, RUNS


def main():
    rows = defaultdict(lambda: {"calls": 0, "wall_s": 0.0, "input": 0, "cached": 0, "output": 0, "reasoning": 0,
                                "models": set(), "efforts": set(), "tiers": set(), "agents_md_loaded": 0,
                                "first": None, "last": None, "credits": set(), "readings": []})
    for m in RUNS.rglob("manifest.json"):
        d = json.loads(m.read_text(encoding="utf-8"))
        batch = m.relative_to(RUNS).parts[0]
        r = rows[batch]
        r["calls"] += 1
        r["wall_s"] += d.get("wall_seconds") or 0
        for u in d.get("usage") or []:
            r["input"] += u.get("input_tokens", 0)
            r["cached"] += u.get("cached_input_tokens", 0)
            r["output"] += u.get("output_tokens", 0)
            r["reasoning"] += u.get("reasoning_output_tokens", 0)
        r["models"].add(d.get("model_recorded"))
        r["efforts"].add(d.get("effort_recorded"))
        r["tiers"].add(d.get("service_tier_recorded"))
        r["agents_md_loaded"] += bool(d.get("global_agents_md_loaded"))
        st = d.get("started_utc")
        r["first"] = min(filter(None, [r["first"], st]))
        r["last"] = max(filter(None, [r["last"], st]))
        rl = d.get("rate_limits_after") or {}
        if rl:
            r["credits"].add((rl.get("credits") or {}).get("balance"))
            r["readings"].append((st, (rl.get("primary") or {}).get("used_percent"),
                                  (rl.get("secondary") or {}).get("used_percent")))
    out = {}
    for b, r in sorted(rows.items(), key=lambda kv: kv[1]["first"] or ""):
        rd = sorted(r["readings"])
        out[b] = {k: (sorted(x for x in v if x is not None) if isinstance(v, set) else v)
                  for k, v in r.items() if k != "readings"}
        out[b]["wall_s"] = round(r["wall_s"], 1)
        out[b]["five_hour_first_last"] = [rd[0][1], rd[-1][1]] if rd else None
        out[b]["weekly_first_last"] = [rd[0][2], rd[-1][2]] if rd else None
    tot = {k: sum(v[k] for v in out.values()) for k in ("calls", "wall_s", "input", "cached", "output", "reasoning")}
    allcred = sorted({c for v in out.values() for c in v["credits"]})
    summary = {"batches": out, "total": tot, "credit_balances_seen": allcred}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cost_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(RESULTS / "cost_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["batch", "calls", "wall_s", "input_tokens", "cached_input", "output_tokens", "reasoning_tokens",
                    "model", "effort", "tier", "agents_md_loaded", "first_utc", "five_hour_first_last",
                    "weekly_first_last", "credits"])
        for b, v in out.items():
            w.writerow([b, v["calls"], v["wall_s"], v["input"], v["cached"], v["output"], v["reasoning"],
                        "/".join(v["models"]), "/".join(v["efforts"]), "/".join(v["tiers"]), v["agents_md_loaded"],
                        v["first"], v["five_hour_first_last"], v["weekly_first_last"], "/".join(v["credits"])])
    for b, v in out.items():
        print(f"{b:28s} {v['calls']:4d} calls {v['wall_s'] / 60:6.1f} min  in {v['input']:>9,} (cached {v['cached']:>9,})"
              f"  out {v['output']:>7,} (reasoning {v['reasoning']:>7,})  {'/'.join(v['models'])} {'/'.join(v['tiers'])}"
              f"  AGENTS.md {v['agents_md_loaded']}  5h {v['five_hour_first_last']} week {v['weekly_first_last']}")
    print("total", tot, "credits seen", allcred)


if __name__ == "__main__":
    main()
