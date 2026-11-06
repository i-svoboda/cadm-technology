"""Run the remaining batches in order, waiting for the 5-hour window to reset when it is nearly used.

Before each step the newest session record decides: if the 5-hour window is at or above --wait-at
and its reset time lies ahead, the orchestrator sleeps until the reset. A step that stops on the
5-hour quota is run again after the reset (batches resume where they stopped). Any other stop
(credit balance changed, weekly quota, failed call) ends the orchestration.

Usage: python orchestrate.py --plan ../protocol/run_plan_2026-10-25.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from common import RUNS

BASE_CREDITS = "62437.5049350000"          # balance in every session record since 21 October 2026


def latest_rate() -> dict | None:
    best, t = None, 0.0
    for m in RUNS.rglob("manifest.json"):
        mt = m.stat().st_mtime
        if mt > t:
            rl = json.loads(m.read_text(encoding="utf-8")).get("rate_limits_after")
            if rl:
                best, t = rl, mt
    return best


def log(msg: str, f: Path):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--wait-at", type=float, default=85.0)
    ap.add_argument("--max-rounds", type=int, default=4)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    logf = RUNS / "orchestrate.log"
    here = Path(__file__).resolve().parent
    for step in plan["steps"]:
        for rnd in range(args.max_rounds):
            rl = latest_rate() or {}
            prim = (rl.get("primary") or {}).get("used_percent", 0.0)
            reset = (rl.get("primary") or {}).get("resets_at", 0)
            sec = (rl.get("secondary") or {}).get("used_percent", 0.0)
            cred = (rl.get("credits") or {}).get("balance")
            if cred is not None and cred != BASE_CREDITS:
                log(f"credit balance {cred} differs from {BASE_CREDITS}; stopping", logf)
                return
            if sec >= 90:
                log(f"weekly quota at {sec}%; stopping", logf)
                return
            if prim >= args.wait_at and reset > time.time():
                wait = reset - time.time() + 180
                log(f"5-hour window at {prim}%; waiting {wait / 60:.0f} min until the reset", logf)
                time.sleep(wait)
            batch = step["args"][step["args"].index("--batch") + 1]
            (RUNS / batch).mkdir(parents=True, exist_ok=True)
            out = RUNS / batch / f"stdout_orchestrated_{int(time.time())}.txt"
            log(f"start {step['name']} ({batch})", logf)
            with open(out, "w", encoding="utf-8") as fh:
                rc = subprocess.run([sys.executable, str(here / step["script"]), *step["args"]], stdout=fh,
                                    stderr=subprocess.STDOUT, cwd=str(here)).returncode
            text = out.read_text(encoding="utf-8")
            if "batch complete" in text:
                log(f"done {step['name']}", logf)
                break
            last = [l for l in text.splitlines() if "batch stopped" in l]
            reason = last[-1] if last else f"no completion line (rc {rc})"
            log(f"{step['name']} stopped: {reason}", logf)
            if "5-hour quota" not in reason:
                return
        else:
            log(f"{step['name']} not finished after {args.max_rounds} rounds; stopping", logf)
            return
    log("plan complete", logf)


if __name__ == "__main__":
    main()
