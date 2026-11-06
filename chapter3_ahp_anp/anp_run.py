"""Replicate the ITS 14(1) ANP elicitation: one Codex session per persona questionnaire.

The plan (persona x replicate) is shuffled once from --seed and saved, as in the TACS SMAA
run_batch.py, and each questionnaire gets its own pair-order seed. Runs that already have an
accepted answer are skipped, so the batch can be resumed.

Usage:
  python anp_run.py --batch anp_s56 --model gpt-5.6-sol --replicates 3 --seed 5601 [--limit 1]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

from anp_model import Case
from common import RUNS, PausedGlobalAgentsMd, codex_turn, load, schema_path

PROT = load("anp_2026_protocol.json")
PERS = {p["id"]: p for p in load("anp_2026_personas.json")["personas"]}
CASE = Case(PROT)
REMINDER = ("Your last answer is incomplete. These items are missing: {missing}. Please return the complete JSON "
            "object again, with every pair, every coverage item and every loss item.")


def plan(personas, replicates, seed):
    rng = random.Random(seed)
    runs = [{"persona": p, "replicate": r} for p in personas for r in range(1, replicates + 1)]
    rng.shuffle(runs)
    for i, r in enumerate(runs):
        r["order"] = i + 1
        r["seed"] = rng.randrange(1, 2 ** 31)
        r["name"] = f"{r['persona']}_r{r['replicate']}"
    return runs


def run_one(batch, r, model, effort, guard):
    out = batch / r["name"]
    state_f = out / "run.json"
    if state_f.exists() and json.loads(state_f.read_text(encoding="utf-8")).get("accepted"):
        return
    if out.exists():
        out.rename(out.with_name(out.name + f"_failed_{int(time.time())}"))
    out.mkdir(parents=True)
    prompt, items = CASE.build_prompt(r["seed"])
    (out / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    p = PERS[r["persona"]]
    session, text, attempt, state = None, prompt, 0, {**r, "turns": []}
    while True:
        if not guard.ok():
            raise RuntimeError(guard.stop_reason)
        tname = f"{len(state['turns']) + 1:02d}_questionnaire" + (f"_reminder{attempt}" if attempt else "")
        res = codex_turn.run_turn(out / tname, text, model=model, effort=effort, workdir=out / "workdir",
                                  schema=schema_path("anp"), resume_id=session,
                                  developer_instructions=p["developer_message"],
                                  meta={"persona": r["persona"], "replicate": r["replicate"], "seed": r["seed"]})
        guard.record(f"{batch.name}/{r['name']}/{tname}", res["manifest"])
        session = session or res["session_id"]
        ans = res["answer"]
        if res["manifest"]["returncode"] != 0 or not isinstance(ans, dict) or "judgments" not in ans:
            state["turns"].append({"folder": tname, "error": "call failed"})
            state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
            raise RuntimeError(f"{r['name']}/{tname}: call failed")
        logj, miss = CASE.to_logs(items, ans)
        _, _, miss2 = CASE.coverage(items, ans)
        missing = miss + miss2
        state["turns"].append({"folder": tname, "missing": missing})
        if not missing or attempt >= 1:
            state.update({"accepted": True, "answer_folder": tname, "missing_final": missing, "session_id": session})
            state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
            return
        attempt += 1
        text = REMINDER.format(missing=", ".join(missing))
        state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-primary", type=float, default=95.0)
    ap.add_argument("--max-secondary", type=float, default=90.0)
    args = ap.parse_args()
    batch = RUNS / args.batch
    batch.mkdir(parents=True, exist_ok=True)
    pf = batch / "plan.json"
    if pf.exists():
        runs = json.loads(pf.read_text(encoding="utf-8"))
    else:
        runs = plan(list(PERS), args.replicates, args.seed)
        pf.write_text(json.dumps(runs, indent=1), encoding="utf-8")
        (batch / "batch.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    todo = [r for r in runs if not ((batch / r["name"] / "run.json").exists() and
                                    json.loads((batch / r["name"] / "run.json").read_text()).get("accepted"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(runs)} planned, {len(todo)} to run", flush=True)
    guard = codex_turn.QuotaGuard(args.max_primary, args.max_secondary, log=batch / "batch.log")
    errs = []
    with PausedGlobalAgentsMd():
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, batch, r, args.model, args.effort, guard): r["name"] for r in todo}
            for f, name in futs.items():
                try:
                    f.result()
                except Exception as e:
                    errs.append(f"{name}: {e}")
    print("batch complete" if not errs else "batch stopped: " + "; ".join(errs), flush=True)


if __name__ == "__main__":
    main()
