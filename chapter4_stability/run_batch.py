"""Run the full elicitation design for one case, one questionnaire at a time.

The plan (persona x wording x replicate) is shuffled once from --seed and saved, so
time trends in the service are not confounded with the design factors. Runs that
already have a result are skipped, which makes the batch resumable. The batch stops
when the paid credit balance changes or the 5-hour quota passes --max-primary.

Usage:
  python scripts/run_batch.py --case experiments/cases/ahp_datacenter.json \
      --personas experiments/personas/ahp_datacenter_personas.json \
      --batch experiments/runs/ahp_main --replicates 3 --seed 4401 [--limit 10] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def plan(personas: list[str], wordings: list[str], replicates: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    runs = [{"persona": p, "wording": w, "replicate": r} for p in personas for w in wordings for r in range(1, replicates + 1)]
    rng.shuffle(runs)
    for i, run in enumerate(runs):
        run["order"] = i + 1
        run["seed"] = rng.randrange(1, 2**31)          # pair-order seed for this questionnaire
        run["name"] = f"{run['persona']}_{run['wording']}_r{run['replicate']}"
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--personas", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--wordings", default="W1,W2,W3")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--with-constraints", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new runs (0 = all)")
    ap.add_argument("--max-primary", type=float, default=95.0, help="stop when 5-hour quota use reaches this percent")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    batch = Path(args.batch)
    batch.mkdir(parents=True, exist_ok=True)
    personas = [p["id"] for p in json.loads(Path(args.personas).read_text(encoding="utf-8"))["personas"]]
    plan_file = batch / "plan.json"
    if plan_file.exists():
        runs = json.loads(plan_file.read_text(encoding="utf-8"))
    else:
        runs = plan(personas, args.wordings.split(","), args.replicates, args.seed)
        plan_file.write_text(json.dumps(runs, indent=1), encoding="utf-8")
    todo = [r for r in runs if not (batch / r["name"] / "result.json").exists()]
    print(f"{len(runs)} planned, {len(runs) - len(todo)} done, {len(todo)} to do")
    if args.dry_run:
        for r in todo[:10]:
            print(r)
        return
    credits0 = None
    done = 0
    for r in todo:
        out = batch / r["name"]
        if out.exists():                                   # half-finished run: keep it for inspection
            out.rename(out.with_name(out.name + "_failed"))
        cmd = [sys.executable, str(HERE / "elicit.py"), "--case", args.case, "--personas", args.personas,
               "--persona", r["persona"], "--wording", r["wording"], "--replicate", str(r["replicate"]),
               "--seed", str(r["seed"]), "--out", str(out)]
        if args.with_constraints:
            cmd.append("--with-constraints")
        print(f"[{r['order']}/{len(runs)}] {r['name']} ...", flush=True)
        subprocess.run(cmd, check=False)
        man = out / "manifest.json"
        if not man.exists():
            print("  no manifest; stopping"); break
        m = json.loads(man.read_text(encoding="utf-8"))
        res = json.loads((out / "result.json").read_text(encoding="utf-8"))
        rl = m.get("rate_limits_after") or {}
        credits = (rl.get("credits") or {}).get("balance")
        primary = (rl.get("primary") or {}).get("used_percent")
        print(f"  {m['wall_seconds']} s, missing {len(res.get('missing', []))}, 5h quota {primary}%, "
              f"week {(rl.get('secondary') or {}).get('used_percent')}%, credits {credits}", flush=True)
        if credits0 is None:
            credits0 = credits
        elif credits != credits0:
            print("  credit balance changed; stopping"); break
        if primary is not None and primary >= args.max_primary:
            print("  5-hour quota threshold reached; stopping"); break
        done += 1
        if args.limit and done >= args.limit:
            break


if __name__ == "__main__":
    main()
