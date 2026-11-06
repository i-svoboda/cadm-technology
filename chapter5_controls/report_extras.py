"""Report extras for RESULT.md. Added on 3 November 2026 after the analysis; not part of the frozen plan of
PROTOCOL.md, Section 8, and not in protocol/freeze.json. Never calls a model and makes no random draw.

1. Archive counts: every attempt folder of runs/main, runs/pilot and runs/pilot2 (validity, errors, return code,
   recorded model, effort and tier, tool calls, declared tool list, global AGENTS.md, files in the working folder);
   the validator's soft notes of the accepted attempts (validation.json); justification lengths; coverage shares of
   exactly 0 or 1; the range of the losses; prompt lengths; the quota readings and the time span of each block.
2. Matrices with CR above 0.10, by questionnaire and block type (recounted from the stored answers with core.py).
3. Budget sweep (results/main/budget_sweep.csv): the budget intervals in which rule (5.1) at tau = 0.9 under S3
   accepts the S1 set and those in which it does not.
4. Exploratory: the feasible sets of the pooled group at 10.2 M USD ranked by f (the runner-up and its gap).

Usage: python -B scripts/report_extras.py --batch runs/main --results results/main
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import analyse  # noqa: E402
import core  # noqa: E402


def attempts(batch: Path) -> list[dict]:
    rows = []
    for m in sorted(batch.glob("*/attempt_*/manifest.json")):
        man = json.loads(m.read_text(encoding="utf-8"))
        val_p = m.parent / "validation.json"
        val = json.loads(val_p.read_text(encoding="utf-8")) if val_p.exists() else {}
        start = datetime.fromisoformat(man["started_utc"]) if man.get("started_utc") else None
        rows.append({"questionnaire": m.parent.parent.name, "attempt": m.parent.name, "valid": bool(man.get("valid")),
                     "errors": val.get("errors"), "returncode": man.get("returncode"),
                     "event_errors": [e.get("message") or (e.get("error") or {}).get("message")
                                      for e in man.get("event_errors") or []],
                     "model": man.get("model_recorded"), "effort": man.get("effort_recorded"),
                     "tier": man.get("service_tier_recorded"), "tool_calls": len(man.get("tool_calls") or []),
                     "tool_list_sha256": man.get("tool_list_sha256") or [],
                     "agents_md_loaded": man.get("global_agents_md_loaded"),
                     "workdir_files": len(man.get("workdir_files") or []), "settings_problem": man.get("settings_problem"),
                     "start": start, "end": start + timedelta(seconds=man.get("wall_seconds") or 0) if start else None,
                     "wall_seconds": man.get("wall_seconds"),
                     "primary": ((man.get("rate_limits_after") or {}).get("primary") or {}).get("used_percent"),
                     "secondary": ((man.get("rate_limits_after") or {}).get("secondary") or {}).get("used_percent"),
                     "balance": (((man.get("rate_limits_after") or {}).get("credits") or {}).get("balance")),
                     "soft": val.get("soft")})
    return rows


def archive(batch: Path) -> dict:
    rows = attempts(batch)
    out = {"attempts": len(rows), "valid_attempts": sum(r["valid"] for r in rows),
           "invalid_attempts": [{k: r[k] for k in ("questionnaire", "attempt", "errors", "returncode", "event_errors",
                                                   "wall_seconds")} for r in rows if not r["valid"]],
           "models": dict(Counter(r["model"] for r in rows)), "efforts": dict(Counter(r["effort"] for r in rows)),
           "tiers": dict(Counter(r["tier"] for r in rows)), "tool_calls": sum(r["tool_calls"] for r in rows),
           "tool_list_sha256": dict(Counter(h for r in rows for h in r["tool_list_sha256"])),
           "agents_md_loaded": sum(bool(r["agents_md_loaded"]) for r in rows),
           "workdir_files": sum(r["workdir_files"] for r in rows),
           "settings_problems": sum(r["settings_problem"] is not None for r in rows),
           "credit_balances": sorted({r["balance"] for r in rows if r["balance"]})}
    # questionnaires, accepted attempts and their soft notes
    qs, soft, jlen, shares, losses, prompts = [], Counter(), [], [], [], []
    for d in sorted(p for p in batch.iterdir() if p.is_dir() and (p / "result.json").exists()):
        res = json.loads((d / "result.json").read_text(encoding="utf-8"))
        qs.append({"name": d.name, "status": res.get("status"), "accepted_attempt": res.get("accepted_attempt"),
                   "attempts": len(res.get("attempts") or [])})
        prompts.append(len((d / "prompt.txt").read_text(encoding="utf-8")))
        if res.get("status") != "complete":
            continue
        acc = d / f"attempt_{res['accepted_attempt']}"
        val = json.loads((acc / "validation.json").read_text(encoding="utf-8"))
        for k, v in (val.get("soft") or {}).items():
            soft[k] += v if isinstance(v, int) else len(v)
        ans = json.loads((d / "answer.json").read_text(encoding="utf-8"))
        jlen += [len(j["text"]) for j in ans.get("justifications", [])]
        shares += [c["share"] for c in ans["coverage"]]
        losses += [c["million_usd"] for c in ans["expected_loss"]]
    out["questionnaires"] = {"folders": len(qs), "status": dict(Counter(q["status"] for q in qs)),
                             "accepted_at_attempt": dict(Counter(q["accepted_attempt"] for q in qs))}
    out["soft_notes_accepted"] = dict(soft)
    if jlen:
        out["justifications"] = {"texts": len(jlen), "chars_min": min(jlen), "chars_median": statistics.median(jlen),
                                 "chars_max": max(jlen), "empty": sum(x == 0 for x in jlen)}
        out["coverage_shares"] = {"items": len(shares), "exactly_0": sum(s == 0 for s in shares),
                                  "exactly_1": sum(s == 1 for s in shares), "min": min(shares), "max": max(shares)}
        out["losses_musd"] = {"items": len(losses), "min": min(losses), "median": statistics.median(losses),
                              "max": max(losses), "zero": sum(x == 0 for x in losses),
                              "above_1000": sum(x > 1000 for x in losses)}
    out["prompt_chars"] = {"min": min(prompts), "max": max(prompts)} if prompts else None
    # blocks: replicate r of the main batch (pilots: one block)
    by_block = defaultdict(list)
    for r in rows:
        name = r["questionnaire"]
        by_block[name.rsplit("_r", 1)[1] if "_r" in name else "pilot"].append(r)
    out["blocks"] = {}
    for b, rr in sorted(by_block.items()):
        rr = sorted((r for r in rr if r["start"]), key=lambda r: r["start"])
        out["blocks"][b] = {"calls": len(rr), "first_start_utc": rr[0]["start"].isoformat(timespec="seconds"),
                            "last_end_utc": max(r["end"] for r in rr).isoformat(timespec="seconds"),
                            "span_min": round((max(r["end"] for r in rr) - rr[0]["start"]).total_seconds() / 60, 1),
                            "five_hour_readings_first_last": [rr[0]["primary"], rr[-1]["primary"]],
                            "weekly_readings_first_last": [rr[0]["secondary"], rr[-1]["secondary"]]}
    return out


def cr_counts(pool: analyse.Pool) -> dict:
    lay = pool.lay
    per_q, per_block = {}, Counter()
    for i, q in enumerate(pool.qs):
        n = 0
        for m in lay.mkeys:
            if len(lay.els[m]) >= 3 and float(core.cr(lay.matrix(pool.Y[i], m))) > 0.1:
                n += 1
                per_block[core.block_type(m)] += 1
        per_q[q["name"]] = n
    return {"matrices_above_0.10": sum(per_q.values()), "by_block_type": dict(per_block),
            "questionnaires_with_any": sum(v > 0 for v in per_q.values()),
            "questionnaires_by_count": dict(sorted(Counter(per_q.values()).items())),
            "by_persona": {p: sum(v for k, v in per_q.items() if k.startswith(p + "_")) for p in pool.personas}}


def sweep_intervals(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    acc, nacc = [], []
    for r in rows:
        (acc if r["set_accepted_tau_0.9"] == "True" else nacc).append(r)

    def runs(sel: list[dict]) -> list[dict]:
        out, cur = [], None
        for r in sel:
            b = float(r["budget"])
            ex = [c for c in r["to_human_experts_tau_0.9"].split(";") if c]
            if cur and abs(b - cur["to"] - 0.1) < 1e-9 and r["set_S1"] == cur["set"]:
                cur["to"] = b
                cur["q_min"] = min(cur["q_min"], float(r["q_S3"]))
                cur["q_max"] = max(cur["q_max"], float(r["q_S3"]))
                cur["to_human_experts_union"] = [c for c in core.K if c in set(cur["to_human_experts_union"]) | set(ex)]
            else:
                cur = {"from": b, "to": b, "set": r["set_S1"], "q_min": float(r["q_S3"]), "q_max": float(r["q_S3"]),
                       "to_human_experts_union": [c for c in core.K if c in ex]}
                out.append(cur)
        return out
    return {"grid_points": len(rows), "accepted_points": len(acc), "not_accepted_points": len(nacc),
            "not_accepted_intervals": runs(nacc)}


def ranked_sets(pool: analyse.Pool, budget: float = core.BUDGET, top: int = 5) -> list[dict]:
    y1, eta1, eal1 = core.aggregate(pool.Y, pool.E, pool.L, np.arange(pool.N)[None, :])
    f = core.reductions(eta1, eal1)[0]
    feas = [i for i in range(len(core.SUBSETS)) if core.SUB_COST[i] <= budget + 1e-9]
    feas.sort(key=lambda i: -f[i])
    tot = float(eal1.sum())
    return [{"set": core.set_name(i), "cost": round(float(core.SUB_COST[i]), 2), "f": round(float(f[i]), 4),
             "share_removed": round(float(f[i]) / tot, 4)} for i in feas[:top]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    batch = Path(args.batch)
    out = {"label": "extras outside the frozen plan (PROTOCOL.md, Section 8); counts of the archive and summaries of "
                    "stored results; item 4 is exploratory",
           "archive": {batch.name: archive(batch)}}
    for pilot in sorted(d for d in batch.parent.glob("pilot*") if d.is_dir()):
        out["archive"][pilot.name] = archive(pilot)
    pool = analyse.Pool(batch)
    out["cr_above_0.10"] = cr_counts(pool)
    out["budget_sweep_rule_5_1_tau_0.9"] = sweep_intervals(Path(args.results) / "budget_sweep.csv")
    out["ranked_feasible_sets_10_2_exploratory"] = ranked_sets(pool)
    p = Path(args.results) / "extras.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
