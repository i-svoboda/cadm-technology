"""Replicate the 2024 AHP study (arXiv 2402.07404) with persona conversations in Codex.

Arms (each arm is one batch folder under runs/):
  guide      the "AHP Guide" conversation (number of experts, persona list, number of levels)
  published  each persona fills the 29 matrices of the published hierarchy in one conversation
             (criteria matrix, sub-criteria matrices, then alternatives three sub-criteria at a time)
  pub45      each persona scores the 45 criteria reconstructed from Appendix C (fresh session)
  fresh      end to end: personas propose and score criteria, sub-criteria and alternatives, then
             fill the 29 matrices of the hierarchy they selected, all in one conversation per persona

Usage:
  python ahp_run.py --arm published --batch ahp_published_s56_r1 --model gpt-5.6-sol [--personas A1,A2] [--max-turns 2]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import (RUNS, PausedGlobalAgentsMd, check_matrix, codex_turn, load, name_similarity, norm_name,
                    schema_path)

PROT = load("ahp_2024_protocol.json")
PERS = {p["id"]: p for p in load("ahp_2024_personas.json")["personas"]}
P = PROT["prompts"]
GOAL = PROT["goal"]
MAX_REMINDERS = 2


class Stop(Exception):
    pass


class Conversation:
    """One persona's (or the guide's) session; turns are stored as numbered folders."""

    def __init__(self, batch: Path, key: str, developer: str, model: str, effort: str, guard):
        self.dir = batch / key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.key, self.developer, self.model, self.effort, self.guard = key, developer, model, effort, guard
        self.state_file = self.dir / "conversation.json"
        self.state = json.loads(self.state_file.read_text(encoding="utf-8")) if self.state_file.exists() else \
            {"key": key, "session_id": None, "turns": []}
        self.workdir = self.dir / "workdir"

    def save(self):
        self.state_file.write_text(json.dumps(self.state, indent=1, ensure_ascii=False), encoding="utf-8")

    def done(self, name: str):
        for t in self.state["turns"]:
            if t["name"] == name and t.get("accepted"):
                return t
        return None

    def send(self, name: str, prompt: str, schema: str | None, validate=None) -> dict:
        """Send a turn (with reminder turns if validate() reports problems); return the accepted record."""
        prev = self.done(name)
        if prev:
            return prev
        if not self.guard.ok():
            raise Stop(self.guard.stop_reason)
        attempt = 0
        text = prompt
        while True:
            idx = len(self.state["turns"]) + 1
            tname = f"{idx:02d}_{name}" + (f"_reminder{attempt}" if attempt else "")
            tdir = self.dir / tname
            if tdir.exists():
                tdir.rename(tdir.with_name(tdir.name + f"_failed_{int(time.time())}"))
            r = codex_turn.run_turn(tdir, text, model=self.model, effort=self.effort, workdir=self.workdir,
                                    schema=schema_path(schema) if schema else None,
                                    resume_id=self.state["session_id"], developer_instructions=self.developer,
                                    meta={"conversation": self.key, "turn": tname})
            self.guard.record(f"{self.dir.parent.name}/{self.key}/{tname}", r["manifest"])
            man = r["manifest"]
            if man["returncode"] != 0 or not r["session_id"] or (schema and r["answer"] is None):
                rec = {"name": name, "folder": tname, "accepted": False, "error": "call failed",
                       "returncode": man["returncode"]}
                self.state["turns"].append(rec)
                self.save()
                raise Stop(f"{self.key}/{tname}: call failed (rc {man['returncode']}, {man.get('error')})")
            if self.state["session_id"] is None:
                self.state["session_id"] = r["session_id"]
            parsed, problems = (validate(r["answer"]) if validate else (r["answer"] if schema else r["text"], []))
            rec = {"name": name, "folder": tname, "attempt": attempt, "problems": problems,
                   "accepted": not problems or attempt >= MAX_REMINDERS, "repaired": bool(problems)}
            if rec["accepted"]:
                rec["parsed"] = parsed
            self.state["turns"].append(rec)
            self.save()
            if rec["accepted"]:
                return rec
            attempt += 1
            found = "Problems found: " + "; ".join(problems) + "."
            if schema == "matrices":
                text = P["reminder"]["text"].format(problems=found, n=parsed.get("n_expected", "the"))
            else:
                text = P["reminder_generic"]["text"].format(problems=found)
            if not self.guard.ok():
                raise Stop(self.guard.stop_reason)


# ------------------------------------------------------------------ validators
def matrices_validator(expected: list[tuple[str, list[str], list[str]]]):
    """expected: (matrix key, element ids, element names) in prompt order."""

    def validate(answer: dict):
        got = answer.get("matrices") or []
        problems, out = [], {"n_expected": len(expected), "matrices": {}}
        if len(got) != len(expected):
            problems.append(f"{len(got)} matrices were returned instead of {len(expected)}")
        # match answered matrices to expected ones by name, else by position
        used = set()
        for pos, (key, ids, names) in enumerate(expected):
            label = key.split("|", 1)[1]
            best, bi = 0.0, None
            for i, m in enumerate(got):
                if i in used:
                    continue
                s = max(name_similarity(m.get("name", ""), label_name) for label_name in [label, *expected_names(key)])
                if s > best:
                    best, bi = s, i
            if bi is None or best < 0.5:
                bi = pos if pos < len(got) and pos not in used else None
            if bi is None:
                problems.append(f"the matrix for {label_of(key)} is missing")
                continue
            used.add(bi)
            A, probs = check_matrix(got[bi], names)
            if A is not None:
                out["matrices"][key] = {"ids": ids, "matrix": A.round(6).tolist(), "problems": probs,
                                        "name_given": got[bi].get("name"), "rationale": got[bi].get("rationale")}
            for pr in probs:
                problems.append(f"the matrix for {label_of(key)}: {pr}")
        return out, problems

    names_by_key = {k: [n for n in ([k.split("|", 1)[1]] + LABELS.get(k, []))] for k, _, _ in expected}

    def expected_names(key):
        return names_by_key.get(key, [])

    return validate


LABELS: dict[str, list[str]] = {}


def label_of(key: str) -> str:
    return (LABELS.get(key) or [key.split("|", 1)[1]])[0]


def list_validator(field: str, n: int):
    def validate(answer: dict):
        items = [x.strip() for x in (answer.get(field) or []) if x and x.strip()]
        probs = [] if len(items) == n else [f"{len(items)} items were given instead of {n}"]
        return {"items": items}, probs
    return validate


def sets_validator(criteria: list[str]):
    def validate(answer: dict):
        sets = answer.get("sets") or []
        out, probs = {}, []
        for c in criteria:
            best = max(sets, key=lambda s: name_similarity(s.get("criterion", ""), c), default=None)
            if best is None or name_similarity(best.get("criterion", ""), c) < 0.5:
                probs.append(f"no sub-criteria for {c}")
                continue
            subs = [x.strip() for x in best.get("sub_criteria") or [] if x.strip()]
            if len(subs) != 3:
                probs.append(f"{len(subs)} sub-criteria for {c} instead of 3")
            out[c] = subs
        return {"sets": out}, probs
    return validate


def scores_validator(numbered: list[tuple[str, str]]):
    """numbered: (number, item text) as listed in the prompt."""
    def validate(answer: dict):
        got = answer.get("scores") or []
        by_no = {str(g.get("number", "")).strip().rstrip("."): g for g in got}
        out, probs = {}, []
        for no, item in numbered:
            g = by_no.get(no)
            if g is None or name_similarity(g.get("item", ""), item) < 0.5:
                cands = [x for x in got if norm_name(x.get("item", "")) == norm_name(item)]
                g = cands[0] if cands else g
            if g is None:
                probs.append(f"no score for item {no} ({item})")
                continue
            out[no] = int(g["score"])
        return {"scores": out}, probs
    return validate


# ------------------------------------------------------------------ published hierarchy
def pub_structure():
    crit = PROT["criteria"]
    alts = PROT["alternatives"]
    return crit, alts


def matrices_turns(crit: list[dict], alts: list[dict]) -> list[tuple[str, str, list]]:
    """(turn name, prompt, expected) for the 1 + 1 + 7 matrix turns of a hierarchy."""
    cnames = ", ".join(c["name"] for c in crit)
    turns = []
    exp = [("criteria|GOAL", [c["id"] for c in crit], [c["name"] for c in crit])]
    LABELS["criteria|GOAL"] = ["top-level criteria", "criteria"]
    turns.append(("criteria_matrix", P["criteria_matrix"]["text"].format(criteria=cnames, goal=GOAL), exp))
    tree = "\n".join(f"- {c['name']}:\n" + "\n".join(s["name"] for s in c["sub_criteria"]) for c in crit)
    exp = []
    for c in crit:
        key = f"sub|{c['id']}"
        LABELS[key] = [c["name"]]
        exp.append((key, [s["id"] for s in c["sub_criteria"]], [s["name"] for s in c["sub_criteria"]]))
    turns.append(("sub_matrices", P["sub_matrices"]["text"].format(tree=tree), exp))
    anames = ", ".join(a["name"] for a in alts)
    for c in crit:
        subs = c["sub_criteria"]
        exp = []
        for s in subs:
            key = f"alt|{s['id']}"
            LABELS[key] = [s["name"]]
            exp.append((key, [a["id"] for a in alts], [a["name"] for a in alts]))
        turns.append((f"alt_matrices_{c['id']}", P["alt_matrices"]["text"].format(
            subs=", ".join(s["name"] for s in subs), alts=anames), exp))
    return turns


def run_published(conv: Conversation, max_turns: int | None, opener: str = "hello_only"):
    crit, alts = pub_structure()
    turns = matrices_turns(crit, alts)
    for i, (name, prompt, exp) in enumerate(turns):
        if max_turns is not None and i >= max_turns:
            return
        if i == 0 and opener:
            prompt = P[opener]["text"] + "\n\n" + prompt
        conv.send(name, prompt + "\n\n" + PROT["matrix_format_note"], "matrices", matrices_validator(exp))


def run_pub45(conv: Conversation):
    items = PROT["published_45_criteria"]
    numbered = [(str(i + 1), x) for i, x in enumerate(items)]
    lst = "\n".join(f"{no}. {x}" for no, x in numbered)
    prompt = (P["hello_only"]["text"] + "\n\n" + P["score_criteria"]["text"].format(n=len(items), items=lst)
              + "\n\n" + PROT["format_note"])
    conv.send("score_published_45", prompt, "scores", scores_validator(numbered))


def run_guide(conv: Conversation):
    for g in PROT["guide_prompts"]:
        conv.send(f"guide_{g['id']}", g["text"], None, None)


# ------------------------------------------------------------------ fresh end-to-end arm
def dedupe(items: list[tuple[str, str]]) -> tuple[list[str], dict]:
    """items: (persona, text). Merge exact matches after normalization; keep first spelling."""
    seen, out, merged = {}, [], defaultdict(list)
    for pid, x in items:
        k = norm_name(x)
        if k not in seen:
            seen[k] = x
            out.append(x)
        merged[seen[k]].append(pid)
    return out, dict(merged)


def top_n(numbered: list[tuple[str, str]], all_scores: dict[str, dict], n: int) -> list[dict]:
    rows = []
    for idx, (no, item) in enumerate(numbered):
        s = [all_scores[p][no] for p in all_scores if no in all_scores[p]]
        rows.append({"number": no, "item": item, "total": sum(s), "nines": sum(1 for v in s if v == 9),
                     "n_scores": len(s), "order": idx})
    rows.sort(key=lambda r: (-r["total"], -r["nines"], r["order"]))
    for r in rows:
        r["selected"] = False
    for r in rows[:n]:
        r["selected"] = True
    return rows


def parallel(fn, keys, workers):
    errs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, k): k for k in keys}
        for f, k in futs.items():
            try:
                f.result()
            except Stop as e:
                errs.append(f"{k}: stopped ({e})")
            except Exception as e:  # keep the other conversations going, report at the end
                errs.append(f"{k}: {type(e).__name__}: {e}")
    if errs:
        raise Stop("; ".join(errs))


def run_fresh(batch: Path, convs: dict[str, Conversation], workers: int):
    stage = batch / "stages"
    stage.mkdir(exist_ok=True)

    def save(name, obj):
        (stage / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")

    # 1. criteria proposals
    parallel(lambda k: convs[k].send("propose_criteria", P["hello_criteria"]["text"] + "\n\n" + PROT["format_note"],
                                     "criteria_list", list_validator("criteria", 7)), convs, workers)
    props = [(k, x) for k in convs for x in convs[k].done("propose_criteria")["parsed"]["items"]]
    crit_list, merged = dedupe(props)
    save("01_criteria_proposed", {"proposals": props, "unique": crit_list, "merged": merged})
    numbered = [(str(i + 1), x) for i, x in enumerate(crit_list)]
    lst = "\n".join(f"{no}. {x}" for no, x in numbered)
    # 2. criteria scores
    prompt = P["score_criteria"]["text"].format(n=len(crit_list), items=lst) + "\n\n" + PROT["format_note"]
    parallel(lambda k: convs[k].send("score_criteria", prompt, "scores", scores_validator(numbered)), convs, workers)
    scores = {k: convs[k].done("score_criteria")["parsed"]["scores"] for k in convs}
    table = top_n(numbered, scores, 7)
    crit = [r["item"] for r in table if r["selected"]]
    crit = sorted(crit, key=lambda x: crit_list.index(x))
    save("02_criteria_scored", {"scores": scores, "table": table, "selected": crit})
    # 3. sub-criteria proposals
    prompt = P["subcriteria"]["text"].format(criteria=", ".join(crit)) + "\n\n" + PROT["format_note"]
    parallel(lambda k: convs[k].send("propose_subcriteria", prompt, "sub_sets", sets_validator(crit)), convs, workers)
    per_c = {}
    for c in crit:
        items = [(k, x) for k in convs for x in convs[k].done("propose_subcriteria")["parsed"]["sets"].get(c, [])]
        per_c[c] = dedupe(items)
    save("03_subcriteria_proposed", {c: {"unique": v[0], "merged": v[1]} for c, v in per_c.items()})
    numbered_s, blocks = [], []
    for ci, c in enumerate(crit, 1):
        lines = []
        for si, s in enumerate(per_c[c][0], 1):
            no = f"{ci}.{si}"
            numbered_s.append((no, s))
            lines.append(f"{no}. {s}")
        blocks.append(f"{c}:\n" + "\n".join(lines))
    # 4. sub-criteria scores
    prompt = P["score_subcriteria"]["text"].format(items="\n".join(blocks)) + "\n\n" + PROT["format_note"]
    parallel(lambda k: convs[k].send("score_subcriteria", prompt, "scores", scores_validator(numbered_s)), convs,
             workers)
    sscores = {k: convs[k].done("score_subcriteria")["parsed"]["scores"] for k in convs}
    subs = {}
    tables = {}
    for ci, c in enumerate(crit, 1):
        nb = [(no, s) for no, s in numbered_s if no.split(".")[0] == str(ci)]
        t = top_n(nb, sscores, 3)
        tables[c] = t
        subs[c] = sorted([r["item"] for r in t if r["selected"]], key=lambda x: per_c[c][0].index(x))
    save("04_subcriteria_scored", {"scores": sscores, "tables": tables, "selected": subs})
    # 5. alternatives proposals
    prompt = P["alternatives"]["text"].format(goal=GOAL) + "\n\n" + PROT["format_note"]
    parallel(lambda k: convs[k].send("propose_alternatives", prompt, "alternatives_list",
                                     list_validator("alternatives", 5)), convs, workers)
    aprops = [(k, x) for k in convs for x in convs[k].done("propose_alternatives")["parsed"]["items"]]
    alt_list, amerged = dedupe(aprops)
    save("05_alternatives_proposed", {"proposals": aprops, "unique": alt_list, "merged": amerged})
    numbered_a = [(str(i + 1), x) for i, x in enumerate(alt_list)]
    lst = "\n".join(f"{no}. {x}" for no, x in numbered_a)
    # 6. alternatives scores
    prompt = P["score_alternatives"]["text"].format(n=len(alt_list), items=lst, goal=GOAL) + "\n\n" + PROT["format_note"]
    parallel(lambda k: convs[k].send("score_alternatives", prompt, "scores", scores_validator(numbered_a)), convs,
             workers)
    ascores = {k: convs[k].done("score_alternatives")["parsed"]["scores"] for k in convs}
    atable = top_n(numbered_a, ascores, 5)
    alts = sorted([r["item"] for r in atable if r["selected"]], key=lambda x: alt_list.index(x))
    save("06_alternatives_scored", {"scores": ascores, "table": atable, "selected": alts})
    # 7. the hierarchy built by the panel, then the 29 matrices in each conversation
    hier = {"criteria": [{"id": f"C{i}", "name": c, "sub_criteria": [{"id": f"C{i}.{j}", "name": s}
                                                                      for j, s in enumerate(subs[c], 1)]}
                         for i, c in enumerate(crit, 1)],
            "alternatives": [{"id": f"X{i}", "name": a} for i, a in enumerate(alts, 1)]}
    save("07_hierarchy", hier)
    turns = matrices_turns(hier["criteria"], hier["alternatives"])

    def matrices(k):
        for name, prompt, exp in turns:
            convs[k].send(name, prompt + "\n\n" + PROT["matrix_format_note"], "matrices", matrices_validator(exp))
    parallel(matrices, convs, workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["guide", "published", "pub45", "fresh"])
    ap.add_argument("--batch", required=True)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--personas", default="A1,A2,A3,A4,A5,A6,A7")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--max-primary", type=float, default=95.0)
    ap.add_argument("--max-secondary", type=float, default=90.0)
    args = ap.parse_args()
    batch = RUNS / args.batch
    batch.mkdir(parents=True, exist_ok=True)
    cfg = batch / "batch.json"
    if not cfg.exists():
        cfg.write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    guard = codex_turn.QuotaGuard(args.max_primary, args.max_secondary, log=batch / "batch.log")
    with PausedGlobalAgentsMd():
        try:
            if args.arm == "guide":
                from protocol_data import guide_developer_message
                run_guide(Conversation(batch, "GUIDE", guide_developer_message(), args.model, args.effort, guard))
            else:
                keys = args.personas.split(",")
                convs = {k: Conversation(batch, k, PERS[k]["developer_message"], args.model, args.effort, guard)
                         for k in keys}
                if args.arm == "published":
                    parallel(lambda k: run_published(convs[k], args.max_turns), keys, args.workers)
                elif args.arm == "pub45":
                    parallel(lambda k: run_pub45(convs[k]), keys, args.workers)
                else:
                    run_fresh(batch, convs, args.workers)
            print("batch complete", flush=True)
        except Stop as e:
            print(f"batch stopped: {e}", flush=True)


if __name__ == "__main__":
    main()
