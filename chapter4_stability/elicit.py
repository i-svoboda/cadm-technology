"""Elicit one persona's pairwise judgments for a decision case through Codex CLI.

One call = one virtual expert filling in the whole questionnaire once.
The block order, the pair order and the order inside each pair are randomized
from --seed, so order effects become part of the run noise and can be audited.

Usage (example):
  python scripts/elicit.py --case experiments/cases/ahp_datacenter.json \
      --personas experiments/personas/ahp_datacenter_personas.json --persona P3 \
      --wording W1 --replicate 1 --seed 101 --out experiments/runs/pilot/P3_W1_r1
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anp_case  # noqa: E402

CODEX = "codex"  # path to the Codex command line tool
ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = {"AHP": ROOT / "experiments" / "schema" / "judgments.schema.json",
           "ANP": ROOT / "experiments" / "schema" / "judgments_links.schema.json"}
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

SCALE = (
    "Use Saaty's 1-9 scale: 1 = equal, 3 = moderately more, 5 = strongly more, "
    "7 = very strongly more, 9 = extremely more; 2, 4, 6 and 8 are intermediate values."
)

# AHP wordings. W1 is the exact text of the two pilot runs of 21 October 2026.
WORDINGS = {
    "W1": {
        "intro": (
            "You are taking part in a structured expert elicitation for the Analytic "
            "Hierarchy Process (AHP). Answer as the expert described below, from that "
            "expert's own professional experience and priorities."
        ),
        "criteria_q": "Which criterion is more important for the goal, and by how much?",
        "alt_q": "With respect to the stated criterion, which alternative contributes more, and by how much?",
    },
    "W2": {
        "intro": (
            "This is a structured expert questionnaire for the Analytic Hierarchy Process "
            "(AHP). Please answer in the role of the expert described below, using that "
            "expert's experience and priorities."
        ),
        "criteria_q": "Considering the goal, how much more does one criterion matter than the other?",
        "alt_q": "Considering only the stated criterion, how much better is one alternative than the other?",
    },
    "W3": {
        "intro": (
            "You are one of several experts asked to compare decision elements in pairs for "
            "an Analytic Hierarchy Process (AHP) study. Give the judgments that the expert "
            "described below would give."
        ),
        "criteria_q": "Rate the relative importance of the two criteria for achieving the goal.",
        "alt_q": "Rate how much each of the two alternatives contributes to the stated criterion.",
    },
}

# ANP wordings: intro, one question template per block (matrix prefix), link question.
ANP_WORDINGS = {
    "W1": {
        "intro": ("You are taking part in a structured expert elicitation for the Analytic Network "
                  "Process (ANP). Answer as the expert described below, from that expert's own "
                  "professional experience and priorities."),
        "K|R": "With respect to criterion {col} ({col_name}), which control performs better?",
        "K|H": "With respect to threat vector {col} ({col_name}), which control reduces this risk more?",
        "R|K": "For control {col} ({col_name}), which criterion matters more when judging the value of this control?",
        "H|R": "For criterion {col} ({col_name}), which threat vector matters more for this operator?",
        "K|B": "Under constraint {col} ({col_name}), which control is easier to implement?",
        "B|K": "For control {col} ({col_name}), which constraint binds more?",
        "K|K": "If both links exist, which control contributes more to the value of monitoring (MON)?",
        "CL|NET": "Which group of elements should have more influence on the priorities in this network?",
        "links": "For each link below, give the probability (from 0 to 1) that it holds for this operator.",
    },
    "W2": {
        "intro": ("This is a structured expert questionnaire for the Analytic Network Process (ANP). "
                  "Please answer in the role of the expert described below, using that expert's "
                  "experience and priorities."),
        "K|R": "Considering only criterion {col} ({col_name}), how much better is one control than the other?",
        "K|H": "Considering only threat vector {col} ({col_name}), how much more does one control reduce this risk?",
        "R|K": "Considering only control {col} ({col_name}), how much more does one criterion matter for its value?",
        "H|R": "Considering only criterion {col} ({col_name}), how much more does one threat vector matter for this operator?",
        "K|B": "Considering only constraint {col} ({col_name}), how much easier is one control to implement?",
        "B|K": "Considering only control {col} ({col_name}), how much more does one constraint bind it?",
        "K|K": "Assuming both links exist, how much more does one control add to the value of monitoring (MON)?",
        "CL|NET": "Considering the whole network, how much more influence should one group of elements have than the other?",
        "links": "For each statement below, how likely is it to be true for this operator? Answer with a probability from 0 to 1.",
    },
    "W3": {
        "intro": ("You are one of several experts asked to compare decision elements in pairs for an "
                  "Analytic Network Process (ANP) study. Give the judgments that the expert described "
                  "below would give."),
        "K|R": "Rate the two controls on criterion {col} ({col_name}).",
        "K|H": "Rate how much each of the two controls reduces the risk from threat vector {col} ({col_name}).",
        "R|K": "Rate the relative weight of the two criteria in judging control {col} ({col_name}).",
        "H|R": "Rate the relative importance of the two threat vectors for criterion {col} ({col_name}) at this operator.",
        "K|B": "Rate how easy each of the two controls is to implement under constraint {col} ({col_name}).",
        "B|K": "Rate how strongly each of the two constraints binds control {col} ({col_name}).",
        "K|K": "Rate the contribution of the two controls to the value of monitoring (MON), assuming both links exist.",
        "CL|NET": "Rate the relative influence that the two groups of elements should have on the priorities in this network.",
        "links": "Estimate, as a probability between 0 and 1, whether each of the following dependencies exists at this operator.",
    },
}


def ahp_questions(case: dict) -> list[dict]:
    """All pairwise questions of a two-level AHP case, grouped into matrices."""
    qs = []
    crit = case["criteria"]
    for a, b in itertools.combinations(crit, 2):
        qs.append({"matrix": "criteria|GOAL", "a": a, "b": b, "context": None})
    for c in crit:
        for a, b in itertools.combinations(case["alternatives"], 2):
            qs.append({"matrix": f"alternatives|{c['id']}", "a": a, "b": b, "context": c})
    return qs


def build_prompt_ahp(case: dict, persona: dict, wording: str, seed: int) -> tuple[str, list[dict]]:
    w = WORDINGS[wording]
    rng = random.Random(seed)
    qs = ahp_questions(case)
    blocks: dict[str, list[dict]] = {}
    for q in qs:
        blocks.setdefault(q["matrix"], []).append(q)
    alt_blocks = [k for k in blocks if k.startswith("alternatives|")]
    rng.shuffle(alt_blocks)
    order = ["criteria|GOAL"] + alt_blocks
    items = []
    n = 0
    for k in order:
        block = blocks[k][:]
        rng.shuffle(block)
        for q in block:
            first, second = (q["a"], q["b"]) if rng.random() < 0.5 else (q["b"], q["a"])
            n += 1
            items.append({"pair_id": f"Q{n:03d}", "matrix": k, "first": first["id"], "second": second["id"],
                          "context": q["context"]["id"] if q["context"] else None})
    crit_lines = "\n".join(f"- {c['id']}: {c['name']}" for c in case["criteria"])
    alt_lines = "\n".join(f"- {a['id']}: {a['name']}" for a in case["alternatives"])
    names = {x["id"]: x["name"] for x in case["criteria"] + case["alternatives"]}
    lines = []
    current = None
    for it in items:
        if it["matrix"] != current:
            current = it["matrix"]
            if current == "criteria|GOAL":
                lines.append(f"\nPart 1. Criteria compared with respect to the goal. {w['criteria_q']}")
            else:
                c = it["context"]
                lines.append(f"\nAlternatives compared with respect to criterion {c} ({names[c]}). {w['alt_q']}")
        lines.append(f"{it['pair_id']}: first = {it['first']} ({names[it['first']]}); "
                     f"second = {it['second']} ({names[it['second']]})")
    prompt = (
        f"{w['intro']}\n\n"
        f"Expert: {persona['role']}. {persona['conditioning']}\n\n"
        f"Goal: {case['goal']}.\n\nCriteria:\n{crit_lines}\n\nAlternatives:\n{alt_lines}\n\n"
        f"For every numbered pair below, state which element is preferred ('first', 'second' "
        f"or 'equal') and the intensity of the preference. {SCALE} Use intensity 1 only with "
        f"'equal'. Judge each pair on its merits; do not compute the matrices or check them "
        f"for consistency. Do not search the web, read files or run commands. Return one "
        f"entry for every pair and nothing else, as the JSON object required by the output schema.\n"
        + "\n".join(lines) + "\n"
    )
    return prompt, items


def build_prompt_anp(case: dict, persona: dict, wording: str, seed: int, with_b: bool) -> tuple[str, list[dict]]:
    w = ANP_WORDINGS[wording]
    rng = random.Random(seed)
    qs = anp_case.anp_questions(case, with_b)
    names = {n["id"]: n["name"] for c in case["clusters"].values() for n in c["nodes"]}
    blocks: dict[str, list[dict]] = {}
    for q in qs:
        blocks.setdefault(q["matrix"], []).append(q)
    order = list(blocks)
    rng.shuffle(order)
    items, lines, n = [], [], 0
    for k in order:
        prefix, _, col = k.partition(":")             # e.g. "K|R", "E"; "CL|NET" has no column node
        lines.append("\n" + w[prefix].format(col=col, col_name=names.get(col, "")))
        block = blocks[k][:]
        rng.shuffle(block)
        for q in block:
            first, second = (q["a"], q["b"]) if rng.random() < 0.5 else (q["b"], q["a"])
            n += 1
            items.append({"pair_id": f"Q{n:03d}", "matrix": k, "first": first["id"], "second": second["id"]})
            lines.append(f"Q{n:03d}: first = {first['id']} ({first['name']}); second = {second['id']} ({second['name']})")
    links = anp_case.anp_link_items(case)
    rng.shuffle(links)
    lines.append("\n" + w["links"])
    for lk in links:
        items.append({"link_id": lk["link_id"], "source": lk["source"], "target": lk["target"]})
        lines.append(f"{lk['link_id']}: {lk['text']}")
    clusters = ["K", "R", "H"] + (["B"] if with_b else [])
    cl_lines = "\n\n".join(
        f"{case['clusters'][c]['name']}:\n" + "\n".join(f"- {x['id']}: {x['name']}" for x in case["clusters"][c]["nodes"])
        for c in clusters)
    prompt = (
        f"{w['intro']}\n\n"
        f"Expert: {persona['role']}. {persona['conditioning']}\n\n"
        f"Decision problem: {case['goal']}.\n\n{cl_lines}\n\n"
        f"For every numbered pair below, state which element is preferred ('first', 'second' "
        f"or 'equal') and the intensity of the preference. {SCALE} Use intensity 1 only with "
        f"'equal'. Judge each pair on its merits; do not compute the matrices or check them "
        f"for consistency. Do not search the web, read files or run commands. Return one entry "
        f"for every pair and one probability for every link, and nothing else, as the JSON object "
        f"required by the output schema.\n" + "\n".join(lines) + "\n"
    )
    return prompt, items


def principal(A: np.ndarray) -> tuple[np.ndarray, float]:
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    return w / w.sum(), float(vals[k].real)


def to_log(items: list[dict], answer: dict) -> tuple[dict, list[str], list[str]]:
    got = {j["pair_id"]: j for j in answer.get("judgments", [])}
    pairs = [it for it in items if "pair_id" in it]
    missing = [it["pair_id"] for it in pairs if it["pair_id"] not in got]
    extra = sorted(set(got) - {it["pair_id"] for it in pairs})
    logj = {}
    for it in pairs:
        j = got.get(it["pair_id"])
        if j is None:
            continue
        v = 0.0 if j["preferred"] == "equal" or j["intensity"] == 1 else float(np.log(j["intensity"]))
        if j["preferred"] == "second":
            v = -v
        a, b = it["first"], it["second"]
        if a > b:
            a, b, v = b, a, -v
        logj[(it["matrix"], a, b)] = v
    return logj, missing, extra


def analyse(case: dict, items: list[dict], answer: dict) -> dict:
    logj, missing, extra = to_log(items, answer)
    out = {"missing": missing, "extra": extra, "matrices": {}}
    if case["method"] == "ANP":
        for m, (ids, A) in anp_case.local_matrices(case, logj).items():
            w, lam = principal(A)
            n = len(ids)
            out["matrices"][m] = {"priorities": dict(zip(ids, w.round(4).tolist())),
                                  "CR": round(((lam - n) / (n - 1)) / RI[n], 4) if RI.get(n) else 0.0}
        links = {l["link_id"]: l["probability"] for l in answer.get("links", [])}
        link_items = [it for it in items if "link_id" in it]
        out["links"] = {f"{it['source']}->{it['target']}": links.get(it["link_id"]) for it in link_items}
        if not missing:
            on = {(it["source"], it["target"]): (links.get(it["link_id"]) or 0) >= 0.5 for it in link_items}
            wb = any(m.startswith(("K|B", "B|K")) for m in out["matrices"])
            pr = anp_case.control_priorities(case, logj, on, with_b=wb)
            out["global_priorities"] = {k: round(v, 4) for k, v in sorted(pr.items(), key=lambda x: -x[1])}
            if "CL|NET" in out["matrices"]:
                pr = anp_case.control_priorities(case, logj, on, with_b=wb, cluster_weights="elicited")
                out["global_priorities_elicited_v"] = {k: round(v, 4) for k, v in sorted(pr.items(), key=lambda x: -x[1])}
        return out
    mats: dict[str, dict] = {}
    for (m, a, b), v in logj.items():
        mats.setdefault(m, {})[(a, b)] = v
    for k, pairs in mats.items():
        ids = [c["id"] for c in case["criteria"]] if k.startswith("criteria") else [a["id"] for a in case["alternatives"]]
        n = len(ids)
        A = np.ones((n, n))
        for (f, s), v in pairs.items():
            i, j = ids.index(f), ids.index(s)
            A[i, j], A[j, i] = np.exp(v), np.exp(-v)
        w, lam = principal(A)
        ci = (lam - n) / (n - 1)
        out["matrices"][k] = {"ids": ids, "matrix": A.round(4).tolist(), "priorities": dict(zip(ids, w.round(4).tolist())),
                              "lambda_max": round(lam, 4), "CR": round(ci / RI[n], 4) if RI[n] else 0.0}
    if not missing and "criteria|GOAL" in out["matrices"]:
        cw = out["matrices"]["criteria|GOAL"]["priorities"]
        alts = [a["id"] for a in case["alternatives"]]
        g = {a: sum(cw[c] * out["matrices"][f"alternatives|{c}"]["priorities"][a] for c in cw) for a in alts}
        out["global_priorities"] = {a: round(v, 4) for a, v in sorted(g.items(), key=lambda x: -x[1])}
    return out


def find_rollout(session_id: str) -> Path | None:
    base = Path.home() / ".codex" / "sessions"
    hits = sorted(base.rglob(f"*{session_id}*.jsonl"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--personas", required=True)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--wording", default="W1")
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-constraints", action="store_true", help="ANP only: 20-node version")
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    persona = next(p for p in json.loads(Path(args.personas).read_text(encoding="utf-8"))["personas"] if p["id"] == args.persona)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    work = out / "workdir"          # empty working root for the agent
    work.mkdir()
    if case["method"] == "ANP":
        prompt, items = build_prompt_anp(case, persona, args.wording, args.seed, args.with_constraints)
    else:
        prompt, items = build_prompt_ahp(case, persona, args.wording, args.seed)
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")

    cmd = [CODEX, "exec", "-m", args.model,
           "-c", f'model_reasoning_effort="{args.effort}"', "-c", 'service_tier="default"',
           "-c", 'web_search="disabled"', "-c", "notify=[]",
           "--disable", "memories", "--disable", "hooks", "--disable", "multi_agent",
           "--disable", "multi_agent_v2", "--disable", "plugins", "--disable", "remote_plugin",
           "--disable", "goals",
           "-s", "read-only", "--skip-git-repo-check", "-C", str(work),
           "--output-schema", str(SCHEMAS[case["method"]]), "-o", str(out / "last_message.json"), "--json", "-"]
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    with open(out / "events.jsonl", "w", encoding="utf-8") as ev, open(out / "stderr.txt", "w", encoding="utf-8") as er:
        proc = subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", stdout=ev, stderr=er, timeout=args.timeout)
    wall = time.time() - t0

    events = [json.loads(l) for l in (out / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip().startswith("{")]
    session_id = next((e.get("thread_id") or e.get("session_id") for e in events if e.get("thread_id") or e.get("session_id")), None)
    usage = [e for e in events if "usage" in e]
    rollout = find_rollout(session_id) if session_id else None
    rate = None
    if rollout:
        shutil.copy2(rollout, out / "session_rollout.jsonl")
        for line in rollout.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            p = r.get("payload") or {}
            if p.get("type") == "token_count" and p.get("rate_limits"):
                rate = p["rate_limits"]
    try:
        answer = json.loads((out / "last_message.json").read_text(encoding="utf-8"))
    except Exception as exc:  # keep the run for inspection
        answer = {"parse_error": str(exc)}
    result = analyse(case, items, answer) if "judgments" in answer else {"error": answer}
    manifest = {
        "case": case["case_id"], "persona": args.persona, "wording": args.wording, "replicate": args.replicate,
        "seed": args.seed, "model": args.model, "effort": args.effort, "service_tier": "default",
        "with_constraints": args.with_constraints,
        "started_utc": started, "wall_seconds": round(wall, 1), "returncode": proc.returncode,
        "session_id": session_id, "usage_events": [e.get("usage") for e in usage],
        "rate_limits_after": rate, "n_items": len(items), "command": cmd[1:],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("wall_seconds", "returncode", "session_id", "usage_events", "rate_limits_after")}, indent=1))
    print("missing:", result.get("missing"), "extra:", result.get("extra"))
    for k, m in result.get("matrices", {}).items():
        print(f"{k:28s} CR={m['CR']:.3f}  {m['priorities']}")
    if "links" in result:
        print("links:", result["links"])
    print("global:", result.get("global_priorities"))


if __name__ == "__main__":
    main()
