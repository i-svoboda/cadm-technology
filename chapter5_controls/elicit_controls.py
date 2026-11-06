"""One virtual expert fills in the DSO controls questionnaire once, through the local Codex CLI.

Adapted from experiments/endtoend_2026-10/scripts/elicit_scenarios.py (which was adapted from tacs_vendor/elicit.py).
One call = one persona filling in one questionnaire in an empty read-only working folder. The questionnaire is the
chapter 3 ANP questionnaire of 32 matrices for the network of core.py, plus the cluster comparison of chapter 4, plus the
portfolio inputs of chapter 3:
  33 matrix blocks   376 pairwise comparisons (Saaty scale), the order of the blocks, of the pairs inside each block and
                     of the two elements of every pair drawn from --seed;
  coverage block     35 shares eta[c, v] (0-1), order drawn from --seed;
  loss block         5 expected annual losses EAL[v] (million USD), order drawn from --seed;
  one short justification per block (35). The variant order in the material is drawn from --seed too.
The answer object lists judgments, coverage, losses and then the justifications (schema order), so the justifications
are written after all numbers.

The persona line is followed by one fixed line (PERSONA_SCOPE) that is the same for every persona and wording.

Checks (PROTOCOL.md, 7.3): JSON, JSON Schema Draft 2020-12, every id exactly once, shares in [0, 1], losses >= 0,
exit code 0, no tool call. A failed check triggers one retry (attempt_2); both attempts are kept. A tool call or a
settings problem stops the questionnaire without a retry, and the batch guard stops the batch. A tool call is a
function or tool item in the session record or the JSON stream, or a file written in the working folder; the
additional_tools record of the new Codex build is the declared tool list, not a call. Its SHA-256 goes into the
manifest, and a list that differs from the batch's first call (--tool-list0) stops the questionnaire the same way
(amendment of 2.11.2026). Everything is archived
with SHA-256 (sha256.json). Every manifest holds the SHA-256 of the Codex binary and the Python, NumPy and SciPy
versions; the monitoring summary of result.json cannot discard a valid answer (review of 2.11.2026, optional 10).

Usage (a batch normally calls this through scripts/run_batch.py):
  python -B scripts/elicit_controls.py --persona E1 --wording W1 --replicate 1 --seed 911003 --out runs/pilot2/E1_W1_p2
  python -B scripts/elicit_controls.py --persona E1 --wording W1 --seed 911001 --dry-run --preview pilot_prompt_preview.txt
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import codex_guard as cg  # noqa: E402
import core  # noqa: E402

ROOT = HERE.parent
SCHEMA = ROOT / "schema" / "answer.schema.json"
CASE = ROOT / "case" / "dso_controls_case.json"
PERSONAS = ROOT / "personas" / "anp_ci_controls_personas.json"

SCALE = (  # verbatim from tacs_vendor/elicit.py and replication anp_model.py
    "Use Saaty's 1-9 scale: 1 = equal, 3 = moderately more, 5 = strongly more, "
    "7 = very strongly more, 9 = extremely more; 2, 4, 6 and 8 are intermediate values."
)

# Near-synonym wordings (PROTOCOL.md, Section 6.2). Intros: the chapter 4 ANP intros verbatim (tacs_vendor/elicit.py,
# ANP_WORDINGS). W1 questions: the chapter 3 questions (replication_vendor/anp_model.py, QUESTIONS, COVERAGE_Q, LOSS_Q)
# with "threat vector" -> "scenario variant"; for K|R, K|V, B|K and K|B they equal the chapter 4 W1 questions. W2 and
# W3: the chapter 4 W2 and W3 questions where chapter 4 has the block (K|R, K|V, B|K, K|B, CL|NET; K|K without "if/
# assuming both links exist", since the links are fixed here); for R|GOAL, V|R, B|R, coverage and loss, near-synonyms of
# W1 in the style of chapter 4 ("Considering only ...", "Rate ..."). CL|NET: the chapter 4 questions verbatim.
WORDINGS = {
    "W1": {
        "intro": ("You are taking part in a structured expert elicitation for the Analytic Network Process (ANP). "
                  "Answer as the expert described below, from that expert's own professional experience and priorities."),
        "R|GOAL": ("With respect to the overall decision (which security controls this operator should fund first), "
                   "which evaluation criterion is more important?"),
        "K|R": "With respect to criterion {col} ({col_name}), which control performs better?",
        "K|V": "With respect to scenario variant {col} ({col_name}), which control reduces this risk more?",
        "V|R": ("When controls are judged on criterion {col} ({col_name}), which scenario variant should carry more "
                "weight for this operator?"),
        "B|R": "When controls are judged on criterion {col} ({col_name}), which implementation constraint should carry more weight?",
        "B|K": "For control {col} ({col_name}), which constraint binds more?",
        "K|B": "Under constraint {col} ({col_name}), which control is easier to implement?",
        "K|K": "Which of the two contributes more to the value of ICS network monitoring and anomaly detection (MON)?",
        "CL|NET": "Which group of elements should have more influence on the priorities in this network?",
        "coverage": ("For each control and scenario variant below, estimate the share of the expected annual loss from "
                     "that scenario variant that the control alone would remove at this operator, as a number from 0 to 1."),
        "loss": ("Estimate the expected annual loss, in million US dollars, that each scenario variant would cause this "
                 "operator if none of the seven controls above were added."),
    },
    "W2": {
        "intro": ("This is a structured expert questionnaire for the Analytic Network Process (ANP). Please answer in "
                  "the role of the expert described below, using that expert's experience and priorities."),
        "R|GOAL": ("Considering the overall decision (which security controls this operator should fund first), how much "
                   "more does one evaluation criterion matter than the other?"),
        "K|R": "Considering only criterion {col} ({col_name}), how much better is one control than the other?",
        "K|V": "Considering only scenario variant {col} ({col_name}), how much more does one control reduce this risk?",
        "V|R": ("When controls are judged only on criterion {col} ({col_name}), how much more weight should one scenario "
                "variant carry for this operator?"),
        "B|R": ("When controls are judged only on criterion {col} ({col_name}), how much more weight should one "
                "implementation constraint carry?"),
        "B|K": "Considering only control {col} ({col_name}), how much more does one constraint bind it?",
        "K|B": "Considering only constraint {col} ({col_name}), how much easier is one control to implement?",
        "K|K": "How much more does one control add to the value of monitoring (MON)?",
        "CL|NET": "Considering the whole network, how much more influence should one group of elements have than the other?",
        "coverage": ("For each control and scenario variant below, what share of the expected annual loss from that "
                     "scenario variant would the control alone remove at this operator? Answer with a number from 0 to 1."),
        "loss": ("If none of the seven controls above were added, what expected annual loss, in million US dollars, would "
                 "each scenario variant cause this operator?"),
    },
    "W3": {
        "intro": ("You are one of several experts asked to compare decision elements in pairs for an Analytic Network "
                  "Process (ANP) study. Give the judgments that the expert described below would give."),
        "R|GOAL": ("Rate the relative importance of the two evaluation criteria for the overall decision (which security "
                   "controls this operator should fund first)."),
        "K|R": "Rate the two controls on criterion {col} ({col_name}).",
        "K|V": "Rate how much each of the two controls reduces the risk from scenario variant {col} ({col_name}).",
        "V|R": ("Rate the weight that each of the two scenario variants should carry for this operator when controls are "
                "judged on criterion {col} ({col_name})."),
        "B|R": ("Rate the weight that each of the two implementation constraints should carry when controls are judged "
                "on criterion {col} ({col_name})."),
        "B|K": "Rate how strongly each of the two constraints binds control {col} ({col_name}).",
        "K|B": "Rate how easy each of the two controls is to implement under constraint {col} ({col_name}).",
        "K|K": "Rate the contribution of the two controls to the value of monitoring (MON).",
        "CL|NET": "Rate the relative influence that the two groups of elements should have on the priorities in this network.",
        "coverage": ("Estimate, as a number between 0 and 1, the share of the expected annual loss from the named scenario "
                     "variant that the named control alone would remove at this operator."),
        "loss": ("Estimate, in million US dollars, the expected annual loss that each scenario variant would cause this "
                 "operator if none of the seven controls above were added."),
    },
}
# Fixed first sentence of the K|K:MON block in every wording (chapter 3, replication anp_model.QUESTIONS["K|K"]).
SYNERGY_PREFACE = ("Network segmentation (SEG) improves observability for monitoring, and identity, MFA and privileged "
                   "access management (IDM) enriches its detection context.")

# Fixed line after the persona (PROTOCOL.md, Sections 5 and 10.1). The first sentence is that of the end-to-end
# experiment's line; its second sentence there ("Use the same professional experience and viewpoint to judge the
# scenario variants below on the criteria defined here; parts of the profile about controls or about criteria not
# defined here do not apply.") does not fit a questionnaire about controls, so it is adapted. "" removes the line
# (recorded amendment).
PERSONA_SCOPE = ("This expert profile was written for choosing security controls. Use the same professional experience "
                 "and viewpoint to judge the controls, criteria, scenario variants and constraints defined below for the "
                 "regional electricity distribution operator described here.")

HOW_TO_ANSWER = (
    "How to answer. Each block below asks one question. For every numbered pair, state which element is preferred "
    "('first', 'second' or 'equal') and the intensity of the preference; the preferred element is the one that the "
    "question of the block favours. " + SCALE + " Use intensity 1 only with 'equal'. For every coverage item, give a "
    "share as a number from 0 to 1. For every loss item, give an amount in million US dollars. Judge each item on its "
    "merits; do not compute the matrices or check them for consistency. For every "
    "block, give one short justification of one or two sentences at the level of the descriptions above, without device "
    "commands, exploit details or other technical attack instructions. Do not search the web, read files or run "
    "commands. Return one entry for every pair, every coverage item, every loss item and every block, and nothing else, "
    "as the JSON object required by the output schema."
)

# Ids shown for the four clusters in the cluster comparison (B is the target incident in the material).
CLUSTER_SHOWN = {"K": "CTL", "R": "CRI", "V": "VAR", "B": "CON"}

OPERATIONAL_MARKERS = re.compile(r"(\bcmd\b|powershell|\bsudo\b|\bnmap\b|mimikatz|metasploit|\bCVE-\d|\bpayload\b|"
                                 r"\bshellcode\b|\bexploit code\b|\bport \d+|\b\d{1,3}(\.\d{1,3}){3}\b)", re.I)
N_BLOCKS = 35


def lower_first(s: str) -> str:
    """Lower-case the first letter unless the first word is an acronym (OT, IT)."""
    return s if len(s) > 1 and s[1].isupper() else s[:1].lower() + s[1:]


def build_prompt(case: dict, persona: dict, wording: str, seed: int) -> tuple[str, list[dict], dict]:
    """Prompt text, item list and layout of one questionnaire. Random draws, in this order: variant order in the
    material, order of the 33 matrix blocks, then for every block in that order the pair order and the side of every
    pair, then the order of the coverage items and of the loss items."""
    w = WORDINGS[wording]
    rng = random.Random(seed)
    lay = core.Layout()
    var = {v["id"]: v for v in case["variants"]}
    vorder = [v["id"] for v in case["variants"]]
    rng.shuffle(vorder)
    net = case["network"]
    names = {n["id"]: n["name"] for c in core.CLUSTER_ORDER for n in net[c]["nodes"]}
    cl_names = {c: net[c]["name"] for c in core.CLUSTER_ORDER}

    order = list(lay.mkeys)
    rng.shuffle(order)
    items: list[dict] = []
    blocks: list[dict] = []
    q = itertools.count(1)
    for bnum, m in enumerate(order, 1):
        bid = f"B{bnum:02d}"
        blocks.append({"block_id": bid, "kind": "matrix", "matrix": m, "block_type": core.block_type(m)})
        pairs = list(itertools.combinations(lay.els[m], 2))
        rng.shuffle(pairs)
        for a, b in pairs:
            first, second = (a, b) if rng.random() < 0.5 else (b, a)
            items.append({"pair_id": f"Q{next(q):03d}", "block_id": bid, "matrix": m, "block_type": core.block_type(m),
                          "first": first, "second": second})
    cov = list(lay.cov_keys)
    rng.shuffle(cov)
    blocks.append({"block_id": f"B{len(order) + 1:02d}", "kind": "coverage", "matrix": None, "block_type": "coverage"})
    for i, (c, v) in enumerate(cov, 1):
        items.append({"coverage_id": f"V{i:02d}", "block_id": blocks[-1]["block_id"], "control": c, "variant": v})
    loss = list(core.V)
    rng.shuffle(loss)
    blocks.append({"block_id": f"B{len(order) + 2:02d}", "kind": "loss", "matrix": None, "block_type": "loss"})
    for i, v in enumerate(loss, 1):
        items.append({"loss_id": f"L{i}", "block_id": blocks[-1]["block_id"], "variant": v})

    org = case["organization"]
    lines = [w["intro"], "", f"Expert: {persona['role']}. {persona['conditioning']}"]
    if PERSONA_SCOPE:
        lines.append(PERSONA_SCOPE)
    lines += ["", f"Decision problem: {case['goal']}.", "", "Organization. " + " ".join(org["description"]), "",
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
    lines += ["", case["same_consequence_note"], "", "The network has four groups of elements."]
    for c in core.CLUSTER_ORDER:
        ids = vorder if c == "V" else [n["id"] for n in net[c]["nodes"]]
        lines += ["", f"{cl_names[c]}:"] + [f"- {i}: {names[i]}" for i in ids]
    lines += ["", HOW_TO_ANSWER]

    def shown(x: str, m: str) -> str:
        return f"{CLUSTER_SHOWN[x]} ({cl_names[x]})" if m == "CL|NET" else f"{x} ({names[x]})"

    for blk in blocks:
        bid = blk["block_id"]
        lines.append("")
        if blk["kind"] == "matrix":
            m, bt = blk["matrix"], blk["block_type"]
            col = m.split(":")[1] if ":" in m else ""
            text = w[bt].format(col=col, col_name=names.get(col, ""))
            if bt == "K|K":
                text = SYNERGY_PREFACE + " " + text
            lines.append(f"Block {bid}. {text}")
            for it in items:
                if it.get("block_id") == bid:
                    lines.append(f"{it['pair_id']}: first = {shown(it['first'], m)}; second = {shown(it['second'], m)}")
        elif blk["kind"] == "coverage":
            lines.append(f"Block {bid}. Coverage. {w['coverage']}")
            for it in items:
                if "coverage_id" in it:
                    lines.append(f"{it['coverage_id']}: control {it['control']} ({names[it['control']]}) against "
                                 f"scenario variant {it['variant']} ({names[it['variant']]})")
        else:
            lines.append(f"Block {bid}. Expected annual loss. {w['loss']}")
            for it in items:
                if "loss_id" in it:
                    lines.append(f"{it['loss_id']}: {it['variant']} ({names[it['variant']]})")
    prompt = "\n".join(lines) + "\n"
    layout = {"seed": seed, "wording": wording, "persona": persona["id"], "variant_order_in_material": vorder,
              "blocks": blocks, "n_pairs": sum("pair_id" in it for it in items),
              "n_coverage": sum("coverage_id" in it for it in items), "n_loss": sum("loss_id" in it for it in items)}
    assert (layout["n_pairs"], layout["n_coverage"], layout["n_loss"], len(blocks)) == \
        (core.N_PAIRS, core.N_COVERAGE, core.N_LOSS, N_BLOCKS)
    return prompt, items, layout


def validate(text: str, items: list[dict], layout: dict, schema: dict) -> dict:
    """Hard errors (trigger the one retry) and soft notes (logged)."""
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
            errors.append(f"{kind}: duplicates {dup[:10]}, missing {miss[:10]}, unknown {extra[:10]}")
    check("pairs", [j["pair_id"] for j in ans["judgments"]], [it["pair_id"] for it in items if "pair_id" in it])
    check("coverage", [c["item_id"] for c in ans["coverage"]], [it["coverage_id"] for it in items if "coverage_id" in it])
    check("losses", [c["item_id"] for c in ans["expected_loss"]], [it["loss_id"] for it in items if "loss_id" in it])
    check("justifications", [j["block_id"] for j in ans["justifications"]], [b["block_id"] for b in layout["blocks"]])
    bad_share = [c["item_id"] for c in ans["coverage"] if not (math.isfinite(c["share"]) and 0 <= c["share"] <= 1)]
    bad_loss = [c["item_id"] for c in ans["expected_loss"] if not (math.isfinite(c["million_usd"]) and c["million_usd"] >= 0)]
    if bad_share:
        errors.append(f"coverage share outside [0, 1]: {bad_share[:10]}")
    if bad_loss:
        errors.append(f"expected loss negative or not finite: {bad_loss}")
    soft["equal_with_intensity_above_1"] = sum(j["preferred"] == "equal" and j["intensity"] != 1 for j in ans["judgments"])
    soft["intensity_1_not_equal"] = sum(j["preferred"] != "equal" and j["intensity"] == 1 for j in ans["judgments"])
    soft["shares_0"] = sum(c["share"] == 0 for c in ans["coverage"])
    soft["shares_1"] = sum(c["share"] == 1 for c in ans["coverage"])
    soft["losses_0"] = [c["item_id"] for c in ans["expected_loss"] if c["million_usd"] == 0]
    soft["losses_above_1000"] = [c["item_id"] for c in ans["expected_loss"] if c["million_usd"] > 1000]
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


def versions() -> dict:
    import platform
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}


def usage_of(manifest: dict) -> dict:
    """Token usage of one attempt: the last usage event, or the session record's token_info when the event stream
    carries none (review of 2.11.2026, optional 10)."""
    u = (manifest.get("usage") or [None])[-1]
    if not u:
        u = ((manifest.get("token_info_after") or {}).get("total_token_usage")) or {}
    return u


STREAM_NON_CALL = ("agent_message", "reasoning", "error")


def detect_calls(events: list[dict], work: Path) -> tuple[list[str], list[str]]:
    """Tool calls outside the session record (amendment of 2.11.2026): any item of the --json event stream other than an
    agent message, reasoning or an error, and any file written in the attempt's working folder."""
    calls = sorted({f"stream item: {(e.get('item') or {}).get('type')}" for e in events
                    if str(e.get("type", "")).startswith("item.") and (e.get("item") or {}).get("type") not in STREAM_NON_CALL})
    files = sorted(str(p.relative_to(work)).replace("\\", "/") for p in work.rglob("*")) if work.exists() else []
    return calls, files


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
    # manifest.json is written whatever happens below, so that the batch guard always sees the call
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
    stream_calls, work_files = detect_calls(events, work)
    manifest = {
        **meta, "attempt": k, "cmdline": cmdline,
        "model_requested": cg.MODEL, "effort_requested": cg.EFFORT, "service_tier_requested": cg.TIER,
        "model_recorded": info.get("model"), "effort_recorded": info.get("effort"),
        "service_tier_recorded": info.get("service_tier"), "personality_recorded": info.get("personality"),
        "global_agents_md_loaded": info.get("agents_md_loaded"),
        "tool_calls": info.get("tool_calls", []) + stream_calls + [f"file written: {f}" for f in work_files],
        "tool_list_sha256": [t["sha256"] for t in info.get("tool_lists", [])] if info else None,
        "tool_lists": [{k_: t[k_] for k_ in ("sha256", "chars", "names")} for t in info.get("tool_lists", [])],
        "workdir_files": work_files,
        "settings_problem": cg.settings_problem(info),
        "session_id": session_id, "started_utc": started, "wall_seconds": round(wall, 1),
        "returncode": rc, "error": err, "event_errors": [e for e in events if e.get("type") in ("error", "turn.failed")],
        "usage": [e.get("usage") for e in events if "usage" in e], "rate_limits_after": info.get("rate_limits"),
        "token_info_after": info.get("token_info"), "valid": val["ok"], "internal_errors": internal,
    }
    manifest["usage_used"] = usage_of(manifest)
    (adir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest, val


def stop_status(man: dict, tl0: tuple | None) -> str | None:
    """Status that ends a questionnaire without a retry, or None: a settings problem, a tool call, or a declared tool
    list that differs from the batch's first call (amendment of 2.11.2026). A valid answer is then not accepted."""
    if man.get("settings_problem"):
        return "invalid_settings"
    if man.get("tool_calls"):
        return "tool_call_stop"
    if tl0 is not None and man.get("tool_list_sha256") is not None and tuple(man["tool_list_sha256"]) != tuple(tl0):
        return "tool_list_changed"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=str(CASE))
    ap.add_argument("--personas", default=str(PERSONAS))
    ap.add_argument("--persona", required=True)
    ap.add_argument("--wording", default="W1", choices=sorted(WORDINGS))
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", help="questionnaire folder (must not exist)")
    ap.add_argument("--cmdline", default="isolated", choices=["isolated", "tacs"])
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--credits0", help="credit balance at the start of the batch (stop retries if it changes)")
    ap.add_argument("--has-credits0", choices=["True", "False"],
                    help="has_credits at the start of the batch (stop retries if it changes)")
    ap.add_argument("--primary-limit", type=float, default=95.0)
    ap.add_argument("--primary-step", type=float, default=3.0)
    ap.add_argument("--secondary-limit", type=float, default=90.0)
    ap.add_argument("--tool-list0", help="comma-separated SHA-256 of the declared tool list of the batch's first call "
                                         "(empty: none declared); a different list stops the questionnaire")
    ap.add_argument("--allow-global-agents-md", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build the prompt only; never calls Codex")
    ap.add_argument("--preview", help="dry run: write the exact prompt to this file")
    args = ap.parse_args()

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    persona = next(p for p in json.loads(Path(args.personas).read_text(encoding="utf-8"))["personas"] if p["id"] == args.persona)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    prompt, items, layout = build_prompt(case, persona, args.wording, args.seed)
    if args.dry_run:
        if args.preview:
            Path(args.preview).write_text(prompt, encoding="utf-8")
            meta = {"persona": args.persona, "wording": args.wording, "seed": args.seed, "layout": layout,
                    "items": items, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_chars": len(prompt), "prompt_words": len(prompt.split()),
                    "codex_command_template": cg.build_cmd(SCHEMA, Path("<out>/attempt_1/last_message.json"),
                                                          Path("<out>/attempt_1/workdir"), args.cmdline)[1:]}
            Path(args.preview).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(prompt)
        print(f"[dry run] {len(prompt)} characters, {len(prompt.split())} words, {layout['n_pairs']} pairs, "
              f"{layout['n_coverage']} coverage items, {layout['n_loss']} loss items, {len(layout['blocks'])} blocks; "
              f"Codex not called")
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
            "seed": args.seed, "n_items": len(items), "agents_md_state_before": st,
            "codex_binary": {"path": cg.CODEX, "sha256": cg.binary_sha256()}, "versions": versions()}
    tl0 = tuple(h for h in args.tool_list0.split(",") if h) if args.tool_list0 is not None else None
    attempts, accepted, status = [], None, "failed"
    for k in range(1, args.max_attempts + 1):
        man, val = run_attempt(qdir, k, prompt, items, layout, schema, meta, args.cmdline, args.timeout)
        attempts.append({"attempt": k, "valid": val["ok"], "errors": val["errors"], "returncode": man["returncode"],
                         "error": man["error"], "tool_calls": man["tool_calls"], "wall_seconds": man["wall_seconds"],
                         "settings_problem": man["settings_problem"]})
        print(f"attempt {k}: valid {val['ok']}, rc {man['returncode']}, {man['wall_seconds']} s, "
              f"tools {len(man['tool_calls'])}, errors {val['errors'][:3]}", flush=True)
        stop = stop_status(man, tl0)
        if stop:                              # no retry; the batch guard stops the batch
            status = stop
            break
        if val["ok"] and man["returncode"] == 0:
            accepted, status = k, "complete"
            break
        rl = man.get("rate_limits_after") or {}
        credits = (rl.get("credits") or {}).get("balance")
        has = (rl.get("credits") or {}).get("has_credits")
        prim = (rl.get("primary") or {}).get("used_percent")
        sec = (rl.get("secondary") or {}).get("used_percent")
        if not rl or (args.credits0 is not None and credits != args.credits0) or \
                (args.has_credits0 is not None and str(has) != args.has_credits0) or \
                (prim is not None and prim + 2 * args.primary_step >= args.primary_limit) or \
                (sec is not None and sec >= args.secondary_limit) or \
                rl.get("rate_limit_reached_type") is not None or rl.get("spend_control_reached") is not None:
            status = "retry_blocked_by_quota"
            break
    result = {"status": status, "accepted_attempt": accepted, "attempts": attempts,
              "persona": args.persona, "wording": args.wording, "replicate": args.replicate, "seed": args.seed}
    if accepted:
        text = (qdir / f"attempt_{accepted}" / "last_message.json").read_text(encoding="utf-8")
        answer = json.loads(text)
        (qdir / "answer.json").write_text(json.dumps(answer, indent=1), encoding="utf-8")
        try:                                   # monitoring only: a crash here must not discard a valid answer
            lay = core.Layout()
            result["summary"] = core_summary(lay, lay.arrays(items, answer))
        except Exception as exc:  # noqa: BLE001
            result["summary_error"] = repr(exc)
        result["soft"] = json.loads((qdir / f"attempt_{accepted}" / "validation.json").read_text(encoding="utf-8"))["soft"]
    (qdir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_hashes(qdir)
    print(json.dumps({k: result[k] for k in ("status", "accepted_attempt")}), flush=True)
    if "summary" in result:
        s = result["summary"]
        print("priorities (prior v)", s["priorities_prior_v"], "own cluster weights", s["own_cluster_weights"])
        print("portfolio", s["portfolio_10_2"], "max CR", s["max_CR"], "tau_b C vs -cost",
              s["tau_b_C_priorities_vs_negative_cost"])
    sys.exit(0 if status == "complete" else 3 if status in ("invalid_settings", "retry_blocked_by_quota",
                                                           "tool_call_stop", "tool_list_changed") else 2)


def core_summary(lay: core.Layout, arr: dict) -> dict:
    """Priorities, consistency and portfolio of one questionnaire (written to result.json for monitoring only)."""
    loc = lay.local(arr["y"][None])
    g_v = lay.priorities(loc, lay.prior_weights(1))["g"][0]
    g_o = lay.priorities(loc, lay.own_weights(loc))["g"][0]
    crs = {m: float(core.cr(lay.matrix(arr["y"], m))) for m in lay.mkeys if len(lay.els[m]) >= 3}
    f = core.reductions(arr["eta"][None], np.maximum(arr["eal"], core.LOSS_FLOOR)[None])
    i, val = core.best_sets(f)
    # "less is better" read of criterion C (review of 2.11.2026, optional 5): tau-b of the controls' local priorities
    # with respect to C against the negative chapter 3 costs; positive when cheaper controls are judged better on C
    tau_c = core.tau_b(loc["K|R:C"][0], -core.COST_VEC)
    return {"priorities_prior_v": dict(zip(core.K, np.round(g_v, 4).tolist())),
            "tau_b_C_priorities_vs_negative_cost": round(tau_c, 3),
            "local_K_wrt_C": dict(zip(core.K, np.round(loc["K|R:C"][0], 4).tolist())),
            "priorities_own_weights": dict(zip(core.K, np.round(g_o, 4).tolist())),
            "own_cluster_weights": dict(zip(core.CL_ELEMENTS, np.round(loc["CL|NET"][0], 4).tolist())),
            "criteria_weights": dict(zip(core.R, np.round(loc["R|GOAL"][0], 4).tolist())),
            "max_CR": round(max(crs.values()), 4), "n_CR_above_0.10": sum(v > 0.1 for v in crs.values()),
            "portfolio_10_2": core.set_name(int(i[0])), "f_10_2": round(float(val[0]), 3),
            "eal_total": round(float(arr["eal"].sum()), 3), "soft": arr["soft"]}


if __name__ == "__main__":
    main()
