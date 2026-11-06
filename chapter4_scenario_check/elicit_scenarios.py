"""One virtual expert fills in the end-to-end scenario questionnaire once, through the local Codex CLI.

Adapted from tacs_vendor/elicit.py (TACS SMAA project, 21.10.2026). One call = one persona filling in one questionnaire
in an empty read-only working folder. The questionnaire has three parts:
  B1        6 pairwise comparisons of the criteria L, D, F, C (Saaty scale), always first (as the AHP case of elicit.py);
  ratings   15 ratings of the five variants on L, F and C (anchored 1-9 scales of chapter 2), one block per criterion;
  pairs     30 pairwise comparisons of the five variants with respect to L, F and C, one block per criterion.
The order of the two scenario parts is fixed by the plan (balanced over personas and wordings, run_batch.part_order);
the order of the blocks inside each part, of the items inside each block, of the two
elements of a pair and of the variants in the case material are drawn from --seed, so order effects enter the run
component and can be audited (layout.json). Each block gets one short justification. The part order changes only the
order in which the blocks are presented: the answer object always lists the judgments, then the ratings, then the
justifications (schema order). The persona line is followed by one fixed line (PERSONA_SCOPE) that is the same for
every persona and wording.

The answer is checked against schema/answer.schema.json (JSON Schema Draft 2020-12) and for complete, unique item ids.
On a failed check, a non-zero exit, a timeout or a tool call recorded in the session, the same questionnaire is asked
once more (attempt_2); both attempts are kept. Everything is archived with SHA-256 (sha256.json).

Usage (a batch normally calls this through scripts/run_batch.py):
  python -B scripts/elicit_scenarios.py --persona E1 --wording W1 --replicate 1 --seed 910001
      --part-order ratings_first --out runs/pilot/E1_W1_p1
  python -B scripts/elicit_scenarios.py --persona E1 --wording W1 --seed 910001 --part-order ratings_first
      --dry-run --preview pilot_prompt_preview.txt
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import codex_guard as cg  # noqa: E402
import core  # noqa: E402

ROOT = HERE.parent
SCHEMA = ROOT / "schema" / "answer.schema.json"
CASE = ROOT / "case" / "endtoend_case.json"
PERSONAS = ROOT / "personas" / "anp_ci_controls_personas.json"

SCALE = (  # verbatim from tacs_vendor/elicit.py
    "Use Saaty's 1-9 scale: 1 = equal, 3 = moderately more, 5 = strongly more, "
    "7 = very strongly more, 9 = extremely more; 2, 4, 6 and 8 are intermediate values."
)

# Near-synonym wordings. Intro and criteria question of W1 and W2 are verbatim from the AHP wordings of chapter 4
# (tacs_vendor/elicit.py, Table 4.2); W3's intro adds "and rate" because this questionnaire also has ratings.
# The variant questions adapt the chapter 4 alternative questions to "stands higher on the stated criterion".
WORDINGS = {
    "W1": {
        "intro": ("You are taking part in a structured expert elicitation for the Analytic Hierarchy Process (AHP). "
                  "Answer as the expert described below, from that expert's own professional experience and priorities."),
        "criteria_q": "Which criterion is more important for the goal, and by how much?",
        "rating_q": "Which point of the anchored scale does each scenario variant deserve on this criterion?",
        "pair_q": "With respect to the stated criterion, which scenario variant stands higher, and by how much?",
    },
    "W2": {
        "intro": ("This is a structured expert questionnaire for the Analytic Hierarchy Process (AHP). Please answer "
                  "in the role of the expert described below, using that expert's experience and priorities."),
        "criteria_q": "Considering the goal, how much more does one criterion matter than the other?",
        "rating_q": "Considering only this criterion, which anchored rating fits each scenario variant?",
        "pair_q": "Considering only the stated criterion, how much higher does one scenario variant stand than the other?",
    },
    "W3": {
        "intro": ("You are one of several experts asked to compare and rate decision elements for an Analytic "
                  "Hierarchy Process (AHP) study. Give the judgments that the expert described below would give."),
        "criteria_q": "Rate the relative importance of the two criteria for achieving the goal.",
        "rating_q": "Rate each scenario variant on this criterion with the anchored scale.",
        "pair_q": "Rate how high each of the two scenario variants stands on the stated criterion.",
    },
}

# Fixed line after the persona (protocol review of 26.10.2026, PROTOCOL.md, Sections 3 and 9.1). The persona texts were
# written for choosing security controls in the TACS ANP case; they stay verbatim, and this line, the same for every
# persona and wording, says how they apply here. Setting it to "" is a recorded amendment (Section 9.1).
PERSONA_SCOPE = ("This expert profile was written for choosing security controls. Use the same professional experience "
                 "and viewpoint to judge the scenario variants below on the criteria defined here; parts of the profile "
                 "about controls or about criteria not defined here do not apply.")

HOW_TO_ANSWER = (
    "How to answer. For every numbered pair, state which element is preferred ('first', 'second' or 'equal') and the "
    "intensity of the preference. For two criteria, the preferred one is the more important one for the goal; for two "
    "scenario variants, it is the variant that stands higher on the stated criterion. " + SCALE + " Use intensity 1 "
    "only with 'equal'. For every rating item, give one whole number from 1 to 9 on the anchored scale of the stated "
    "criterion. Judge each item on its merits; do not compute matrices, scores or rankings and do not check the "
    "answers for consistency. For every block, give one short justification of one or two sentences at the level of "
    "the scenario descriptions, without device commands, exploit details or other technical attack instructions. Do "
    "not search the web, read files or run commands. Return one entry for every pair, every rating item and every "
    "block, and nothing else, as the JSON object required by the output schema."
)

OPERATIONAL_MARKERS = re.compile(r"(\bcmd\b|powershell|\bsudo\b|\bnmap\b|mimikatz|metasploit|\bCVE-\d|\bpayload\b|"
                                 r"\bshellcode\b|\bexploit code\b|\bport \d+|\b\d{1,3}(\.\d{1,3}){3}\b)", re.I)


def lower_first(s: str) -> str:
    """Lower-case the first letter unless the first word is an acronym (OT, IT)."""
    return s if len(s) > 1 and s[1].isupper() else s[:1].lower() + s[1:]


def anchors_text(c: dict) -> str:
    a = c["anchors"]
    return f"1-3 {a['1-3']}; 4-6 {a['4-6']}; 7-9 {a['7-9']}"


def build_prompt(case: dict, persona: dict, wording: str, seed: int,
                 part_order: str | None = None) -> tuple[str, list[dict], dict]:
    """Prompt text, item list and layout of one questionnaire. Random draws, in this order: variant order in the
    material, part order, criteria pairs (order, sides), criterion order of the first part and its items, criterion
    order of the second part and its items. A part order given by the plan replaces the drawn one (the draw is still
    made, so the rest of the stream does not change)."""
    w = WORDINGS[wording]
    rng = random.Random(seed)
    crit = {c["id"]: c for c in case["criteria"]}
    var = {v["id"]: v for v in case["variants"]}
    vorder = [v["id"] for v in case["variants"]]
    rng.shuffle(vorder)
    ratings_first = rng.random() < 0.5
    if part_order is not None:
        assert part_order in ("ratings_first", "pairs_first"), part_order
        ratings_first = part_order == "ratings_first"

    items: list[dict] = []
    blocks: list[dict] = []
    q = itertools.count(1)
    rnum = itertools.count(1)

    def pair_items(block_id: str, kind: str, elements: list[str], criterion: str | None) -> None:
        pairs = list(itertools.combinations(elements, 2))
        rng.shuffle(pairs)
        for a, b in pairs:
            first, second = (a, b) if rng.random() < 0.5 else (b, a)
            items.append({"pair_id": f"Q{next(q):02d}", "block_id": block_id, "block": kind,
                          "first": first, "second": second, "criterion": criterion})

    blocks.append({"block_id": "B1", "kind": "criteria", "criterion": None})
    pair_items("B1", "criteria", ["L", "D", "F", "C"], None)
    parts = ["ratings", "pairs"] if ratings_first else ["pairs", "ratings"]
    bnum = 2
    for part in parts:
        order = core.RATED[:]
        rng.shuffle(order)
        for c in order:
            bid = f"B{bnum}"
            bnum += 1
            blocks.append({"block_id": bid, "kind": part, "criterion": c})
            if part == "ratings":
                vs = [v["id"] for v in case["variants"]]
                rng.shuffle(vs)
                for v in vs:
                    items.append({"item_id": f"R{next(rnum):02d}", "block_id": bid, "block": "rating",
                                  "variant": v, "criterion": c})
            else:
                pair_items(bid, "pairs", [v["id"] for v in case["variants"]], c)

    org = case["organization"]
    lines = [w["intro"], "", f"Expert: {persona['role']}. {persona['conditioning']}"]
    if PERSONA_SCOPE:
        lines.append(PERSONA_SCOPE)
    lines += ["", f"Goal: {case['goal']}.", "", "Organization. " + " ".join(org["description"]), "",
             f"Initial state ({org['initial_state']['id']}): {org['initial_state']['label'].lower()}."]
    lines += [f"- {f}" for f in org["initial_state"]["facts"]]
    lines += ["", f"Target incident ({org['target_incident']['id']}): {org['target_incident']['label'].lower()}."]
    lines += [f"- {f}" for f in org["target_incident"]["facts"]]
    lines += ["", "Incident reports used for the case:"]
    for ev in case["incident_evidence"]:
        lines.append(f"- {ev['period']}, {ev['title']}. Source: {ev['sources']}. " + " ".join(ev["shown"]))
    lines += ["", "Scenario variants. Each variant is a complete chain of events from its entry event to the target "
                  "incident B. Each event is linked to the supporting object of the organization that it needs (the "
                  "object's role in brackets). " + case["non_operational_note"]]
    for vid in vorder:
        v = var[vid]
        lines += ["", f"{v['id']}: {v['name']} ({v['steps']} steps between events)"]
        for e in v["events"]:
            lines.append(f"  {e['id']}  {e['text']}. Supporting object: {lower_first(e['object'])} ({e['role']}).")
    lines += ["", "Criteria. A higher value on any criterion means a higher priority for review."]
    for cid in ["L", "D", "F", "C"]:
        c = crit[cid]
        if cid == "D":
            lines.append(f"- D, {c['name']}: higher means {c['higher_means']}. It is rated from the path length: "
                         f"{c['rule']}. It follows from the chains above and is not asked below.")
        else:
            note = f" {c['note']}" if c.get("note") else ""
            lines.append(f"- {cid}, {c['name']}: higher means {c['higher_means']}.{note} Anchored scale: {anchors_text(c)}.")
    lines += [case["same_consequence_note"], "", HOW_TO_ANSWER]

    names = {c["id"]: c["name"] for c in case["criteria"]} | {v["id"]: v["name"] for v in case["variants"]}
    for blk in blocks:
        bid, kind, cid = blk["block_id"], blk["kind"], blk["criterion"]
        lines.append("")
        if kind == "criteria":
            lines.append(f"Block {bid}. Criteria compared with respect to the goal. {w['criteria_q']}")
        elif kind == "ratings":
            lines.append(f"Block {bid}. Ratings on criterion {cid} ({names[cid]}; anchored scale {anchors_text(crit[cid])}). "
                         f"{w['rating_q']}")
        else:
            lines.append(f"Block {bid}. Scenario variants compared with respect to criterion {cid} ({names[cid]}; higher "
                         f"means {crit[cid]['higher_means']}). {w['pair_q']}")
        for it in items:
            if it["block_id"] != bid:
                continue
            if "pair_id" in it:
                lines.append(f"{it['pair_id']}: first = {it['first']} ({names[it['first']]}); "
                             f"second = {it['second']} ({names[it['second']]})")
            else:
                lines.append(f"{it['item_id']}: {it['variant']} ({names[it['variant']]})")
    prompt = "\n".join(lines) + "\n"
    layout = {"seed": seed, "wording": wording, "persona": persona["id"], "variant_order_in_material": vorder,
              "part_order": "ratings_first" if ratings_first else "pairs_first",
              "part_order_source": "plan" if part_order is not None else "seed", "blocks": blocks,
              "n_pairs": sum("pair_id" in it for it in items), "n_ratings": sum("item_id" in it for it in items)}
    assert layout["n_pairs"] == 36 and layout["n_ratings"] == 15 and len(blocks) == 7
    return prompt, items, layout


def validate(text: str, items: list[dict], layout: dict, schema: dict) -> dict:
    """Hard errors (trigger the one retry) and soft notes (logged, normalized as in elicit.to_log)."""
    import jsonschema
    errors, soft = [], {}
    if not text.strip():
        return {"ok": False, "errors": ["empty answer"], "soft": soft}
    try:
        ans = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"ok": False, "errors": [f"parse error: {exc}"], "soft": soft}
    v = jsonschema.Draft202012Validator(schema)
    errors += [f"schema: {e.message} at {list(e.absolute_path)}" for e in v.iter_errors(ans)]
    if errors:
        return {"ok": False, "errors": errors[:20], "soft": soft}

    def check(kind: str, got: list[str], want: list[str]) -> None:
        dup = sorted({x for x in got if got.count(x) > 1})
        miss = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        if dup or miss or extra:
            errors.append(f"{kind}: duplicates {dup}, missing {miss}, unknown {extra}")
    check("pairs", [j["pair_id"] for j in ans["judgments"]], [it["pair_id"] for it in items if "pair_id" in it])
    check("ratings", [r["item_id"] for r in ans["ratings"]], [it["item_id"] for it in items if "item_id" in it])
    check("justifications", [j["block_id"] for j in ans["justifications"]], [b["block_id"] for b in layout["blocks"]])
    soft["equal_with_intensity_above_1"] = sum(j["preferred"] == "equal" and j["intensity"] != 1 for j in ans["judgments"])
    soft["intensity_1_not_equal"] = sum(j["preferred"] != "equal" and j["intensity"] == 1 for j in ans["judgments"])
    soft["empty_justifications"] = [j["block_id"] for j in ans["justifications"] if not j["text"].strip()]
    soft["long_justifications"] = [j["block_id"] for j in ans["justifications"] if len(j["text"]) > 400]
    soft["operational_markers"] = sorted({m.group(0) for j in ans["justifications"] for m in OPERATIONAL_MARKERS.finditer(j["text"])})
    return {"ok": not errors, "errors": errors, "soft": soft}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_hashes(qdir: Path) -> None:
    files = sorted(p for p in qdir.rglob("*") if p.is_file() and p.name != "sha256.json")
    (qdir / "sha256.json").write_text(json.dumps({str(p.relative_to(qdir)).replace("\\", "/"): sha(p) for p in files},
                                                 indent=1), encoding="utf-8")


def run_attempt(qdir: Path, k: int, prompt: str, items: list[dict], layout: dict, schema: dict, meta: dict,
                cmdline: str, timeout: int) -> tuple[dict, dict]:
    adir = qdir / f"attempt_{k}"
    adir.mkdir()
    work = adir / "workdir"                   # empty working root for the agent
    work.mkdir()
    out_file = adir / "last_message.json"
    cmd = cg.build_cmd(SCHEMA, out_file, work, cmdline)
    (adir / "command.json").write_text(json.dumps(cmd[1:], indent=1), encoding="utf-8")
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    rc, err = None, None
    with open(adir / "events.jsonl", "w", encoding="utf-8") as ev, open(adir / "stderr.txt", "w", encoding="utf-8") as er:
        try:
            proc = subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", stdout=ev, stderr=er, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            err = "timeout"
    wall = time.time() - t0
    events = []
    for line in (adir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    session_id = next((e.get("thread_id") or e.get("session_id") for e in events
                       if e.get("thread_id") or e.get("session_id")), None)
    # manifest.json must be written whatever happens below, so that the batch guard always sees the call
    # (protocol review of 26.10.2026, required change 9c)
    internal = []
    try:
        info = cg.copy_rollout(session_id, adir / "session_rollout.jsonl")
    except Exception as exc:  # noqa: BLE001
        info = {}
        internal.append(f"copy_rollout crashed: {exc!r}")
    try:
        text = out_file.read_text(encoding="utf-8") if out_file.exists() else ""
        val = validate(text, items, layout, schema)
    except Exception as exc:  # noqa: BLE001
        val = {"ok": False, "errors": [f"validator crashed: {exc!r}"], "soft": {}}
        internal.append(f"validate crashed: {exc!r}")
    (adir / "validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
    manifest = {
        **meta, "attempt": k, "cmdline": cmdline,
        "model_requested": cg.MODEL, "effort_requested": cg.EFFORT, "service_tier_requested": cg.TIER,
        "model_recorded": info.get("model"), "effort_recorded": info.get("effort"),
        "service_tier_recorded": info.get("service_tier"), "personality_recorded": info.get("personality"),
        "global_agents_md_loaded": info.get("agents_md_loaded"), "tool_calls": info.get("tool_calls", []),
        "settings_problem": cg.settings_problem(info),
        "session_id": session_id, "started_utc": started, "wall_seconds": round(wall, 1),
        "returncode": rc, "error": err, "event_errors": [e for e in events if e.get("type") in ("error", "turn.failed")],
        "usage": [e.get("usage") for e in events if "usage" in e], "rate_limits_after": info.get("rate_limits"),
        "token_info_after": info.get("token_info"), "valid": val["ok"], "internal_errors": internal,
    }
    (adir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=str(CASE))
    ap.add_argument("--personas", default=str(PERSONAS))
    ap.add_argument("--persona", required=True)
    ap.add_argument("--wording", default="W1", choices=sorted(WORDINGS))
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--part-order", choices=["ratings_first", "pairs_first"],
                    help="order of the two scenario parts fixed by the plan (default: drawn from the seed)")
    ap.add_argument("--out", help="questionnaire folder (must not exist)")
    ap.add_argument("--cmdline", default="isolated", choices=["isolated", "tacs"])
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--credits0", help="credit balance at the start of the batch (stop retries if it changes)")
    ap.add_argument("--primary-limit", type=float, default=95.0)
    ap.add_argument("--primary-step", type=float, default=1.0)
    ap.add_argument("--allow-global-agents-md", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build the prompt only; never calls Codex")
    ap.add_argument("--preview", help="dry run: write the exact prompt to this file")
    args = ap.parse_args()

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    persona = next(p for p in json.loads(Path(args.personas).read_text(encoding="utf-8"))["personas"] if p["id"] == args.persona)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    prompt, items, layout = build_prompt(case, persona, args.wording, args.seed, args.part_order)
    if args.dry_run:
        if args.preview:
            Path(args.preview).write_text(prompt, encoding="utf-8")
            meta = {"persona": args.persona, "wording": args.wording, "seed": args.seed, "part_order": args.part_order,
                    "layout": layout,
                    "items": items, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_chars": len(prompt), "prompt_words": len(prompt.split()),
                    "codex_command_template": cg.build_cmd(SCHEMA, Path("<out>/attempt_1/last_message.json"),
                                                          Path("<out>/attempt_1/workdir"), args.cmdline)[1:]}
            Path(args.preview).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(prompt)
        print(f"[dry run] {len(prompt)} characters, {len(prompt.split())} words, {layout['n_pairs']} pairs, "
              f"{layout['n_ratings']} ratings, {len(layout['blocks'])} blocks, {layout['part_order']}; Codex not called")
        return

    if (cg.MODEL, cg.EFFORT, cg.TIER) != ("gpt-6.1-sol", "xhigh", "default"):
        sys.exit("model settings differ from the author's rule")
    st = cg.agents_md_state()
    if st["global_present"] and not args.allow_global_agents_md:
        sys.exit("~/.codex/AGENTS.md is present: run the batch through scripts/run_batch_clean.sh")
    if not Path(cg.CODEX).exists():
        sys.exit(f"Codex binary not found: {cg.CODEX}")
    qdir = Path(args.out)
    qdir.mkdir(parents=True, exist_ok=False)
    (qdir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (qdir / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    (qdir / "layout.json").write_text(json.dumps(layout, indent=1), encoding="utf-8")
    meta = {"case": case["case_id"], "persona": args.persona, "wording": args.wording, "replicate": args.replicate,
            "seed": args.seed, "n_items": len(items), "agents_md_state_before": st}
    attempts, accepted, status = [], None, "failed"
    for k in range(1, args.max_attempts + 1):
        man, val = run_attempt(qdir, k, prompt, items, layout, schema, meta, args.cmdline, args.timeout)
        attempts.append({"attempt": k, "valid": val["ok"], "errors": val["errors"], "returncode": man["returncode"],
                         "error": man["error"], "tool_calls": man["tool_calls"], "wall_seconds": man["wall_seconds"],
                         "settings_problem": man["settings_problem"]})
        print(f"attempt {k}: valid {val['ok']}, rc {man['returncode']}, {man['wall_seconds']} s, "
              f"tools {len(man['tool_calls'])}, errors {val['errors'][:3]}", flush=True)
        if man["settings_problem"]:
            status = "invalid_settings"
            break
        if val["ok"] and man["returncode"] == 0 and not man["tool_calls"]:
            accepted, status = k, "complete"
            break
        rl = man.get("rate_limits_after") or {}
        credits = (rl.get("credits") or {}).get("balance")
        prim = (rl.get("primary") or {}).get("used_percent")
        if not rl or (args.credits0 is not None and credits != args.credits0) or \
                (prim is not None and prim + 2 * args.primary_step >= args.primary_limit) or \
                rl.get("rate_limit_reached_type") is not None or rl.get("spend_control_reached") is not None:
            status = "retry_blocked_by_quota"
            break
    result = {"status": status, "accepted_attempt": accepted, "attempts": attempts,
              "persona": args.persona, "wording": args.wording, "replicate": args.replicate, "seed": args.seed}
    if accepted:
        text = (qdir / f"attempt_{accepted}" / "last_message.json").read_text(encoding="utf-8")
        answer = json.loads(text)
        (qdir / "answer.json").write_text(json.dumps(answer, indent=1), encoding="utf-8")
        lay = core.Layout(case)
        result["summary"] = core.questionnaire_summary(lay, lay.arrays(items, answer))
        result["soft"] = json.loads((qdir / f"attempt_{accepted}" / "validation.json").read_text(encoding="utf-8"))["soft"]
    (qdir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_hashes(qdir)
    print(json.dumps({k: result[k] for k in ("status", "accepted_attempt")}), flush=True)
    if "summary" in result:
        s = result["summary"]
        print("weights", s["weights"], "CR", s["CR_criteria"], s["CR_variants"])
        print("ratings mode", s["order_ratings_mode"], "relative mode", s["order_relative_mode"])
    sys.exit(0 if status == "complete" else 3 if status in ("invalid_settings", "retry_blocked_by_quota") else 2)


if __name__ == "__main__":
    main()
