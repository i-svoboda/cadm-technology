"""Run the end-to-end questionnaire design, one questionnaire at a time (adapted from tacs_vendor/run_batch.py).

The plan (7 personas x 3 wordings x 3 runs = 63) is drawn from --seed and saved with the SHA-256 of every input.
By default it is blocked by run: block r holds run r of all 21 persona x wording cells in shuffled order, so service
drift over time is not confounded with persona or wording (--plan-type shuffle gives the single shuffle of the TACS code). Each questionnaire gets its own order seed.
The batch is resumable:
  - a questionnaire with result.json status "complete" is skipped;
  - a folder without result.json, or stopped by the guard ("invalid_settings", "retry_blocked_by_quota"), is renamed
    <name>_interrupted<n> and asked again;
  - a questionnaire that failed both attempts ("failed") is asked once more at the end of the batch as a replacement
    (the failed folder becomes <name>_failed1); there is no second replacement.
The guard (codex_guard.QuotaGuard) reads the session record of every call and stops on a credit change, a projected
5-hour reading of 95 % or more, a weekly reading of 90 % or more, a reached rate limit or spend control, a change of
has_credits, a settings problem, or a questionnaire that ends without the status "complete" or "failed".
Before the first call the newest session record gives the credit baseline and the current readings (preflight); the
batch does not start when the balance differs from --expected-credits or the 5-hour projection is already at 95 %.
The batch also refuses to start when any frozen file (protocol/freeze.json: inputs, other files except selftest.py,
vendor copies) differs (protocol review of 26.10.2026, required changes 8 and 9).

Usage (always through scripts/run_batch_clean.sh, which pauses ~/.codex/AGENTS.md):
  bash scripts/run_batch_clean.sh --batch runs/pilot --pilot
  bash scripts/run_batch_clean.sh --batch runs/main --seed 4426
  python -B scripts/run_batch.py --batch runs/main --seed 4426 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codex_guard as cg  # noqa: E402

ROOT = HERE.parent
INPUTS = {"case": ROOT / "case" / "endtoend_case.json", "personas": ROOT / "personas" / "anp_ci_controls_personas.json",
          "schema": ROOT / "schema" / "answer.schema.json", "elicit_scenarios.py": HERE / "elicit_scenarios.py",
          "codex_guard.py": HERE / "codex_guard.py", "core.py": HERE / "core.py"}
# Files hashed in protocol/freeze.json besides INPUTS (freeze_inputs.py writes them, main() checks them).
OTHER_FILES = [ROOT / "PROTOCOL.md", ROOT / "case" / "build_case.py", ROOT / "case" / "analysts_reference.json",
               ROOT / "case" / "build_log.json", HERE / "run_batch.py", HERE / "run_batch_clean.sh", HERE / "analyse.py",
               HERE / "vc_adjust.py", HERE / "selftest.py"]
NOT_ENFORCED = {"scripts/selftest.py"}          # may change without touching the batch (review 26.10.2026, change 8)
FREEZE = ROOT / "protocol" / "freeze.json"
# Pilot: E1 with W1 ratings-first, and E6 (field operations manager, the persona furthest from the case) with W3
# pairs-first (protocol review of 26.10.2026, optional change 3).
PILOT = [{"persona": "E1", "wording": "W1", "replicate": 1, "seed": 910001, "part_order": "ratings_first", "name": "E1_W1_p1"},
         {"persona": "E6", "wording": "W3", "replicate": 1, "seed": 910002, "part_order": "pairs_first", "name": "E6_W3_p1"}]


def part_order(persona: str, wording: str, replicate: int) -> str:
    """Ratings part first when persona, wording and run numbers add up to an odd number: 32 of 63 questionnaires,
    4 or 5 of the 9 of every persona and 10 or 11 of the 21 of every wording."""
    return "ratings_first" if (int(persona[1:]) + int(wording[1:]) + replicate) % 2 else "pairs_first"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def input_hashes() -> dict:
    return {k: sha(p) for k, p in INPUTS.items()}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def other_hashes() -> dict:
    return {rel(p): sha(p) for p in OTHER_FILES if p.exists()}


def vendor_hashes() -> dict:
    return {p.name: sha(p) for p in sorted((HERE / "tacs_vendor").iterdir()) if p.is_file()}


def freeze_differences() -> list[str]:
    """Every frozen file whose current SHA-256 differs from protocol/freeze.json (selftest.py excepted)."""
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    diff = [k for k, h in input_hashes().items() if frozen["inputs"].get(k) != h]
    fo = frozen.get("other_files") or {}
    now = other_hashes()
    for p in OTHER_FILES:
        k = rel(p)
        if k not in NOT_ENFORCED and (fo.get(k) is None or fo.get(k) != now.get(k)):
            diff.append(k)
    fv = frozen.get("vendor_copies") or {}
    for k, h in vendor_hashes().items():
        if fv.get(k) != h:
            diff.append(f"tacs_vendor/{k}")
    return sorted(diff)


def plan(personas: list[str], wordings: list[str], replicates: int, seed: int) -> list[dict]:
    """As tacs_vendor/run_batch.plan: one shuffle, then one order seed per questionnaire."""
    rng = random.Random(seed)
    runs = [{"persona": p, "wording": w, "replicate": r} for p in personas for w in wordings for r in range(1, replicates + 1)]
    rng.shuffle(runs)
    for i, run in enumerate(runs):
        run["order"] = i + 1
        run["seed"] = rng.randrange(1, 2**31)
        run["name"] = f"{run['persona']}_{run['wording']}_r{run['replicate']}"
        run["part_order"] = part_order(run["persona"], run["wording"], run["replicate"])
    return runs


def plan_blocked(personas: list[str], wordings: list[str], replicates: int, seed: int) -> list[dict]:
    """Default plan of this experiment: run r of every persona x wording cell forms block r (21 questionnaires),
    shuffled inside the block; then one order seed per questionnaire. Service drift over time then falls on the run
    factor, which is the error component of model (4.1), and a batch stopped by the quota after a whole block is still
    balanced."""
    rng = random.Random(seed)
    runs = []
    for r in range(1, replicates + 1):
        block = [{"persona": p, "wording": w, "replicate": r} for p in personas for w in wordings]
        rng.shuffle(block)
        runs += block
    for i, run in enumerate(runs):
        run["order"] = i + 1
        run["seed"] = rng.randrange(1, 2**31)
        run["name"] = f"{run['persona']}_{run['wording']}_r{run['replicate']}"
        run["part_order"] = part_order(run["persona"], run["wording"], run["replicate"])
    return runs


def status_of(qdir: Path) -> str | None:
    f = qdir / "result.json"
    if not qdir.exists():
        return None
    if not f.exists():
        return "no_result"
    return json.loads(f.read_text(encoding="utf-8")).get("status")


def set_aside(qdir: Path, tag: str) -> Path:
    n = 1
    while qdir.with_name(f"{qdir.name}_{tag}{n}").exists():
        n += 1
    new = qdir.with_name(f"{qdir.name}_{tag}{n}")
    qdir.rename(new)
    return new


def run_one(batch: Path, r: dict, guard: cg.QuotaGuard, args) -> str:
    out = batch / r["name"]
    cmd = [sys.executable, "-B", str(HERE / "elicit_scenarios.py"), "--persona", r["persona"], "--wording", r["wording"],
           "--replicate", str(r["replicate"]), "--seed", str(r["seed"]), "--part-order", r["part_order"], "--out", str(out),
           "--cmdline", args.cmdline, "--primary-limit", str(args.max_primary), "--primary-step", str(guard.step())]
    assert guard.credits0 is not None, "credit baseline missing: the preflight reading must come before the first call"
    cmd += ["--credits0", str(guard.credits0)]
    print(f"[{r.get('order', '-')}] {r['name']} (seed {r['seed']}) ...", flush=True)
    subprocess.run(cmd, check=False)
    for m in sorted(out.glob("attempt_*/manifest.json")):
        guard.record(f"{r['name']}/{m.parent.name}", json.loads(m.read_text(encoding="utf-8")))
    st = status_of(out) or "no_folder"
    if st not in ("complete", "failed"):          # e.g. a crash after the call: never go on without a quota check
        guard.stop_reason = guard.stop_reason or f"{r['name']}: status {st}"
    with open(batch / "batch.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {r['name']} status {st}\n")
    return st


def summary(batch: Path, runs: list[dict]) -> dict:
    tot = {"complete": 0, "failed": 0, "other": 0, "calls": 0, "wall_seconds": 0.0, "input_tokens": 0,
           "cached_input_tokens": 0, "output_tokens": 0, "reasoning_output_tokens": 0, "retries": 0}
    readings, credits = [], set()
    for d in sorted(p for p in batch.iterdir() if p.is_dir()):
        for m in sorted(d.glob("attempt_*/manifest.json")):
            man = json.loads(m.read_text(encoding="utf-8"))
            tot["calls"] += 1
            tot["wall_seconds"] += man.get("wall_seconds") or 0
            u = (man.get("usage") or [{}])[-1] or {}
            for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
                tot[k] += u.get(k) or 0
            rl = man.get("rate_limits_after") or {}
            if rl:
                readings.append((man["started_utc"], (rl.get("primary") or {}).get("used_percent"),
                                 (rl.get("secondary") or {}).get("used_percent")))
                credits.add((rl.get("credits") or {}).get("balance"))
            if m.parent.name != "attempt_1":
                tot["retries"] += 1
    for r in runs:
        st = status_of(batch / r["name"])
        tot["complete" if st == "complete" else "failed" if st == "failed" else "other"] += 1
    readings.sort()
    return {**tot, "planned": len(runs), "credit_balances_seen": sorted(c for c in credits if c is not None),
            "first_reading": readings[0] if readings else None, "last_reading": readings[-1] if readings else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default=str(INPUTS["personas"]))
    ap.add_argument("--batch", required=True)
    ap.add_argument("--wordings", default="W1,W2,W3")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, help="plan seed (main batch: 4426, PROTOCOL.md)")
    ap.add_argument("--pilot", action="store_true", help="the two fixed pilot questionnaires of PROTOCOL.md")
    ap.add_argument("--plan-type", default="blocked", choices=["blocked", "shuffle"],
                    help="blocked by run (default) or one shuffle of all 63 as in tacs_vendor/run_batch.py")
    ap.add_argument("--cmdline", default="isolated", choices=["isolated", "tacs"])
    ap.add_argument("--limit", type=int, default=0, help="stop after this many questionnaires (0 = all)")
    ap.add_argument("--max-primary", type=float, default=95.0)
    ap.add_argument("--max-secondary", type=float, default=90.0)
    ap.add_argument("--allow-unfrozen", action="store_true", help="run although inputs differ from protocol/freeze.json")
    ap.add_argument("--expected-credits", default=cg.EXPECTED_CREDITS,
                    help="credit balance the newest session record must show before the first call (author decision)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    batch = Path(args.batch)
    hashes = input_hashes()
    if FREEZE.exists():
        diff = freeze_differences()
        if diff and not args.allow_unfrozen:
            sys.exit(f"files differ from protocol/freeze.json: {diff} (re-freeze after an amendment, or --allow-unfrozen)")
    elif not args.dry_run and not args.allow_unfrozen:
        sys.exit("protocol/freeze.json is missing: run scripts/freeze_inputs.py before the batch")
    binary = cg.binary_info() if not args.dry_run else None
    plan_file = batch / "plan.json"
    if plan_file.exists():
        saved = json.loads(plan_file.read_text(encoding="utf-8"))
        runs = saved["runs"]
        if saved["inputs_sha256"] != hashes and not args.allow_unfrozen:
            sys.exit("inputs changed since this batch was planned; refusing to mix questionnaires")
        if binary and saved.get("codex_binary") and saved["codex_binary"].get("sha256") != binary["sha256"] \
                and not args.allow_unfrozen:
            sys.exit("the Codex binary changed since this batch was planned; refusing to mix questionnaires")
    else:
        if args.pilot:
            runs = [dict(r, order=i + 1) for i, r in enumerate(PILOT)]
        else:
            if args.seed is None:
                sys.exit("--seed is required for a new plan")
            personas = [p["id"] for p in json.loads(Path(args.personas).read_text(encoding="utf-8"))["personas"]]
            maker = plan_blocked if args.plan_type == "blocked" else plan
            runs = maker(personas, args.wordings.split(","), args.replicates, args.seed)
        saved = {"created": datetime.now().astimezone().isoformat(timespec="seconds"), "plan_seed": args.seed,
                 "plan_type": None if args.pilot else args.plan_type,
                 "pilot": args.pilot, "cmdline": args.cmdline, "inputs_sha256": hashes,
                 "freeze_sha256": sha(FREEZE) if FREEZE.exists() else None, "codex_binary": binary, "runs": runs}
        if not args.dry_run:
            batch.mkdir(parents=True, exist_ok=True)
            plan_file.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    done = [r for r in runs if status_of(batch / r["name"]) == "complete"]
    failed = [r for r in runs if status_of(batch / r["name"]) == "failed"]
    todo = [r for r in runs if status_of(batch / r["name"]) not in ("complete", "failed")]
    print(f"{len(runs)} planned, {len(done)} complete, {len(failed)} failed (replacement at the end), {len(todo)} to do")
    if args.dry_run:
        for r in todo[:10]:
            print(r)
        print("agents.md:", cg.agents_md_state())
        return
    st = cg.agents_md_state()
    if st["global_present"]:
        sys.exit("~/.codex/AGENTS.md is present: start the batch with scripts/run_batch_clean.sh")
    guard = cg.QuotaGuard(args.max_primary, args.max_secondary, log=batch / "guard.log")
    # Reading before the first call (review 26.10.2026, required change 9b): credit baseline and current windows.
    reading = cg.latest_reading()
    why = cg.preflight(reading, args.expected_credits, args.max_primary, args.max_secondary, guard.step())
    if why:
        with open(batch / "guard.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} NOT STARTED: {why}\n")
        sys.exit(f"not started: {why}")
    guard.set_baseline(reading)
    n_new = 0
    for r in todo:
        out = batch / r["name"]
        if out.exists():
            print(f"  set aside {set_aside(out, 'interrupted').name}")
        run_one(batch, r, guard, args)
        n_new += 1
        if not guard.ok() or (args.limit and n_new >= args.limit):
            break
    else:
        for r in [r for r in runs if status_of(batch / r["name"]) == "failed"]:
            if not guard.ok():
                break
            if (batch / f"{r['name']}_failed1").exists():
                continue                                    # one replacement only
            print(f"  replacement for {set_aside(batch / r['name'], 'failed').name}")
            run_one(batch, r, guard, args)
    s = summary(batch, runs)
    s["stop_reason"] = guard.stop_reason
    s["baseline_before_first_call"] = guard.baseline
    (batch / "batch_summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    files = sorted(p for p in batch.rglob("*") if p.is_file() and p.name != "batch_sha256.json")
    (batch / "batch_sha256.json").write_text(json.dumps({str(p.relative_to(batch)).replace("\\", "/"): sha(p) for p in files},
                                                        indent=0), encoding="utf-8")
    print(json.dumps(s, indent=1))
    if guard.stop_reason:
        print(f"STOPPED: {guard.stop_reason}. Resume with the same command after the reason is cleared.")


if __name__ == "__main__":
    main()
