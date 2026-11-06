"""Run the DSO controls questionnaire design, one questionnaire at a time (adapted from the end-to-end run_batch.py).

The plan (7 personas x 3 wordings x 2 runs = 42; author decision of 2.11.2026) is drawn from --seed and saved with the
SHA-256 of every input and the number of runs. It is blocked by run: block r holds run r of all 21 persona x wording
cells in shuffled order, so service drift over time falls on the run factor (--plan-type shuffle gives the single
shuffle of the TACS code). Each questionnaire gets its own order seed. A new main plan with another number of runs is
refused unless --allow-unfrozen is given. The batch is resumable:
  - a questionnaire with result.json status "complete" is skipped;
  - a folder without result.json, or stopped by the guard ("invalid_settings", "retry_blocked_by_quota",
    "tool_call_stop"), is renamed <name>_interrupted<n> and asked again;
  - when every planned questionnaire is "complete" or "failed" (failed both attempts) and the guard is ok, each failed
    one is asked once more as a replacement with the same seed (the failed folder becomes <name>_failed1), and, if that
    fails too, a second time with a seed derived from it (<name>_failed2; review of 2.11.2026, A2 and optional 15).
    This happens in whichever invocation completes the plan, also one started with --limit.
A main-batch invocation starts only after the 5-hour window has reset, so that each block stays in one window
(review of 2.11.2026, optional 6; --allow-running-window overrides it and is logged).
The guard (codex_guard.QuotaGuard) reads the session record of every call and stops on a credit change, a projected
5-hour reading of 95 % or more, a weekly reading of 90 % or more, a reached rate limit or spend control, a change of
has_credits, a settings problem, a tool call, a change of the declared tool list against the batch's first call (kept
in <batch>/tool_list.json; --expected-tool-list-sha256 pins it to the pilot's), or a questionnaire that ends without the
status "complete" or "failed".
Before the first call the newest session record gives the credit baseline and the current readings (preflight). The
batch does not start when the balance differs from --expected-credits, when the 5-hour projection is already at 95 %,
when the Codex binary differs from --expected-codex-sha256, or when any frozen file (protocol/freeze.json: inputs, other
files except selftest.py, vendor copies) differs.

Usage (always through scripts/run_batch_clean.sh, which pauses ~/.codex/AGENTS.md):
  bash scripts/run_batch_clean.sh --batch runs/pilot2 --pilot --pilot-round 2
  bash scripts/run_batch_clean.sh --batch runs/main --seed 1102 --replicates 2 --min-step <step>
      --expected-tool-list-sha256 <the pilot's> [--limit 21]        (one command line)
  python -B scripts/run_batch.py --batch runs/main --seed 1102 --replicates 2 --dry-run
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
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codex_guard as cg  # noqa: E402

ROOT = HERE.parent
INPUTS = {"case": ROOT / "case" / "dso_controls_case.json", "personas": ROOT / "personas" / "anp_ci_controls_personas.json",
          "schema": ROOT / "schema" / "answer.schema.json", "elicit_controls.py": HERE / "elicit_controls.py",
          "codex_guard.py": HERE / "codex_guard.py", "core.py": HERE / "core.py"}
# Files hashed in protocol/freeze.json besides INPUTS (freeze_inputs.py writes them, main() checks them).
OTHER_FILES = [ROOT / "PROTOCOL.md", ROOT / "case" / "build_case.py", ROOT / "case" / "analysis_inputs.json",
               ROOT / "case" / "comparison_reference.json", ROOT / "case" / "build_log.json", HERE / "run_batch.py",
               HERE / "run_batch_clean.sh", HERE / "analyse.py", HERE / "vc_adjust.py", HERE / "make_schema.py",
               HERE / "freeze_inputs.py", HERE / "isolation_check.py", HERE / "dual_use_scan.py", HERE / "selftest.py"]
NOT_ENFORCED = {"scripts/selftest.py"}          # may change without touching the batch
VENDOR_DIRS = [HERE / "tacs_vendor", HERE / "replication_vendor"]
FREEZE = ROOT / "protocol" / "freeze.json"
MAIN_SEED = 1102
REPLICATES = 2                                  # author decision of 2.11.2026 (R = 2, 42 questionnaires)
# Pilot: E1 (ICS engineer) with W1 and E6 (field operations manager) with W3, the two most different wordings.
# Round 1 (runs/pilot) failed on logging only (PILOT_REPORT.md); round 2 repeats it with new seeds in runs/pilot2
# (amendment of 2.11.2026).
PILOT_ROUNDS = {1: [{"persona": "E1", "wording": "W1", "replicate": 1, "seed": 911001, "name": "E1_W1_p1"},
                    {"persona": "E6", "wording": "W3", "replicate": 1, "seed": 911002, "name": "E6_W3_p1"}],
                2: [{"persona": "E1", "wording": "W1", "replicate": 1, "seed": 911003, "name": "E1_W1_p2"},
                    {"persona": "E6", "wording": "W3", "replicate": 1, "seed": 911004, "name": "E6_W3_p2"}]}
PILOT = PILOT_ROUNDS[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def input_hashes() -> dict:
    return {k: sha(p) for k, p in INPUTS.items()}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def other_hashes() -> dict:
    return {rel(p): sha(p) for p in OTHER_FILES if p.exists()}


def vendor_hashes() -> dict:
    return {f"{d.name}/{p.name}": sha(p) for d in VENDOR_DIRS for p in sorted(d.iterdir()) if p.is_file()}


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
    vh = vendor_hashes()
    for k in sorted(set(fv) | set(vh)):
        if fv.get(k) != vh.get(k):
            diff.append(k)
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
    return runs


def plan_blocked(personas: list[str], wordings: list[str], replicates: int, seed: int) -> list[dict]:
    """Default plan: run r of every persona x wording cell forms block r (21 questionnaires), shuffled inside the
    block; then one order seed per questionnaire (end-to-end run_batch.plan_blocked)."""
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
    return runs


def replacement_seed(seed: int) -> int:
    """Seed of the second replacement, derived from the planned seed (review of 2.11.2026, optional 15)."""
    return int(hashlib.sha256(f"{seed}-replacement-2".encode()).hexdigest()[:8], 16) % (2**31 - 1) + 1


def pending_replacements(batch: Path, runs: list[dict]) -> list[tuple[dict, int]]:
    """(run, replacement number) for every failed questionnaire, once every planned questionnaire is complete or
    failed (review of 2.11.2026, A2): number 1 with the planned seed, number 2 with the derived seed; none after that."""
    if not all(status_of(batch / r["name"]) in ("complete", "failed") for r in runs):
        return []
    out = []
    for r in runs:
        if status_of(batch / r["name"]) != "failed":
            continue
        k = 1
        while (batch / f"{r['name']}_failed{k}").exists():
            k += 1
        if k <= 2:
            out.append((dict(r, seed=r["seed"] if k == 1 else replacement_seed(r["seed"]), replacement=k), k))
    return out


def window_has_reset(reading: dict | None, now: float | None = None) -> bool:
    """True when the newest session record's 5-hour window has reset (or there is no record)."""
    import time as _t
    if not reading:
        return True
    prim = (reading.get("rate_limits") or {}).get("primary") or {}
    return (prim.get("resets_at") or 0) <= (_t.time() if now is None else now)


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
    cmd = [sys.executable, "-B", str(HERE / "elicit_controls.py"), "--persona", r["persona"], "--wording", r["wording"],
           "--replicate", str(r["replicate"]), "--seed", str(r["seed"]), "--out", str(out),
           "--cmdline", args.cmdline, "--primary-limit", str(args.max_primary), "--primary-step", str(guard.step()),
           "--secondary-limit", str(args.max_secondary)]
    assert guard.credits0 is not None, "credit baseline missing: the preflight reading must come before the first call"
    cmd += ["--credits0", str(guard.credits0)]
    if guard.has_credits0 is not None:
        cmd += ["--has-credits0", str(guard.has_credits0)]
    if guard.tool_list0 is not None:               # the declared tool list of the batch's first call
        cmd += ["--tool-list0=" + ",".join(guard.tool_list0)]
    print(f"[{r.get('order', '-')}] {r['name']} (seed {r['seed']}) ...", flush=True)
    subprocess.run(cmd, check=False)
    for m in sorted(out.glob("attempt_*/manifest.json")):
        guard.record(f"{r['name']}/{m.parent.name}", json.loads(m.read_text(encoding="utf-8")))
    st = status_of(out) or "no_folder"
    if st not in ("complete", "failed"):          # e.g. a crash after the call: never go on without a quota check
        guard.stop_reason = guard.stop_reason or f"{r['name']}: status {st}"
    with open(batch / "batch.log", "a", encoding="utf-8") as f:
        rep = f" (replacement {r['replacement']}, seed {r['seed']})" if r.get("replacement") else ""
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {r['name']}{rep} status {st}\n")
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
            u = man.get("usage_used") or (man.get("usage") or [{}])[-1] or {}
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
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--seed", type=int, help=f"plan seed (main batch: {MAIN_SEED}, PROTOCOL.md)")
    ap.add_argument("--pilot", action="store_true", help="the two fixed pilot questionnaires of PROTOCOL.md")
    ap.add_argument("--pilot-round", type=int, default=2, choices=sorted(PILOT_ROUNDS),
                    help="pilot round (1: seeds 911001-2, failed on logging; 2: seeds 911003-4)")
    ap.add_argument("--expected-tool-list-sha256",
                    help="SHA-256 of the declared tool list that the first call must show (the pilot's, PROTOCOL.md 11.3)")
    ap.add_argument("--plan-type", default="blocked", choices=["blocked", "shuffle"],
                    help="blocked by run (default) or one shuffle of all 42 as in tacs_vendor/run_batch.py")
    ap.add_argument("--allow-running-window", action="store_true",
                    help="start a main-batch invocation although the 5-hour window has not reset (logged)")
    ap.add_argument("--cmdline", default="isolated", choices=["isolated", "tacs"])
    ap.add_argument("--limit", type=int, default=0, help="stop after this many questionnaires (0 = all); --limit 21 "
                                                          "runs one block per 5-hour window (PROTOCOL.md, 11.3)")
    ap.add_argument("--max-primary", type=float, default=95.0)
    ap.add_argument("--max-secondary", type=float, default=90.0)
    ap.add_argument("--min-step", type=float, default=3.0, help="smallest step per call of the 5-hour projection, %%")
    ap.add_argument("--allow-unfrozen", action="store_true", help="run although inputs differ from protocol/freeze.json")
    ap.add_argument("--expected-credits", default=cg.EXPECTED_CREDITS,
                    help="credit balance the newest session record must show before the first call (author decision)")
    ap.add_argument("--expected-codex-sha256", default=cg.EXPECTED_CODEX_SHA256,
                    help="SHA-256 the Codex binary must have (author decision after an update of the Codex app)")
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
    if binary is not None and binary.get("sha256") != args.expected_codex_sha256:
        sys.exit(f"Codex binary {cg.CODEX} has SHA-256 {binary.get('sha256')}, expected {args.expected_codex_sha256}; "
                 f"ask the author (record the new binary as an amendment and pass --expected-codex-sha256)")
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
            runs = [dict(r, order=i + 1) for i, r in enumerate(PILOT_ROUNDS[args.pilot_round])]
        else:
            if args.seed is None:
                sys.exit("--seed is required for a new plan")
            if args.replicates != REPLICATES and not args.allow_unfrozen:
                sys.exit(f"the frozen plan has {REPLICATES} runs per persona and wording; --replicates "
                         f"{args.replicates} needs an amendment (or --allow-unfrozen)")
            personas = [p["id"] for p in json.loads(Path(args.personas).read_text(encoding="utf-8"))["personas"]]
            maker = plan_blocked if args.plan_type == "blocked" else plan
            runs = maker(personas, args.wordings.split(","), args.replicates, args.seed)
        saved = {"created": datetime.now().astimezone().isoformat(timespec="seconds"), "plan_seed": args.seed,
                 "plan_type": None if args.pilot else args.plan_type, "pilot": args.pilot,
                 "pilot_round": args.pilot_round if args.pilot else None,
                 "expected_tool_list_sha256": args.expected_tool_list_sha256,
                 "replicates": None if args.pilot else args.replicates, "cmdline": args.cmdline,
                 "min_step": args.min_step, "inputs_sha256": hashes,
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
    guard = cg.QuotaGuard(args.max_primary, args.max_secondary, min_step=args.min_step, log=batch / "guard.log",
                          state_file=batch / "tool_list.json")
    exp = (args.expected_tool_list_sha256,) if args.expected_tool_list_sha256 else None
    if exp and guard.tool_list0 is not None and guard.tool_list0 != exp:
        sys.exit(f"the declared tool list of this batch {guard.tool_list0} differs from --expected-tool-list-sha256")
    reading = cg.latest_reading()
    why = cg.preflight(reading, args.expected_credits, args.max_primary, args.max_secondary, guard.step())
    if not why and not args.pilot and not window_has_reset(reading):
        if args.allow_running_window:
            with open(batch / "guard.log", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} started in a running 5-hour window "
                        f"(--allow-running-window)\n")
        else:
            prim = reading["rate_limits"].get("primary") or {}
            why = (f"the 5-hour window has not reset (reading {prim.get('used_percent')}%, reset at "
                   f"{datetime.fromtimestamp(prim.get('resets_at') or 0).astimezone().isoformat(timespec='minutes')}); "
                   f"each block starts in a fresh window (PROTOCOL.md, 11.3)")
    if why:
        with open(batch / "guard.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} NOT STARTED: {why}\n")
        sys.exit(f"not started: {why}")
    if exp and guard.tool_list0 is None:             # the first call must show the pilot's tool list
        guard.set_tool_list(exp, "--expected-tool-list-sha256")
    guard.set_baseline(reading)
    n_new = 0
    for r in todo:
        out = batch / r["name"]
        if out.exists():
            print(f"  set aside {set_aside(out, 'interrupted').name}")
        k = 1
        while (batch / f"{r['name']}_failed{k}").exists():
            k += 1
        if k >= 3:                                  # an interrupted second replacement keeps its derived seed
            r = dict(r, seed=replacement_seed(r["seed"]), replacement=2)
        elif k == 2:
            r = dict(r, replacement=1)
        run_one(batch, r, guard, args)
        n_new += 1
        if not guard.ok() or (args.limit and n_new >= args.limit):
            break
    # replacements whenever every planned questionnaire is complete or failed and the guard is ok (A2, optional 15)
    while guard.ok():
        pend = pending_replacements(batch, runs)
        if not pend:
            break
        r, k = pend[0]
        print(f"  replacement {k} for {set_aside(batch / r['name'], 'failed').name} (seed {r['seed']})")
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
