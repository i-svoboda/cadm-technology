"""Supplementary numbers for RESULT.md that the frozen scripts/analyse.py does not write.

Written on 27 October 2026, after the main batch and after the frozen analysis had run. This script is not part of
the frozen plan (PROTOCOL.md, Section 7) and is not listed in protocol/freeze.json. It calls no model. It only
(a) counts what the stored questionnaires and manifests already hold (data quality, cost), (b) applies the union rule
of PROTOCOL.md 7.5 to the per-mode decisions that analyse.py wrote for tau = 0.8 and 0.95 (analyse.py writes the
union only for tau = 0.9), and (c) adds exploratory views chosen after seeing the data (marked "exploratory").

Usage: python -B scripts/report_extras.py --batch runs/main --results results/main
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tacs_vendor"))

import numpy as np  # noqa: E402

import analyse  # noqa: E402
import core  # noqa: E402

NAME = re.compile(r"^E\d_W\d_r\d$")


def load_questionnaires(batch: Path) -> list[dict]:
    out = []
    for d in sorted(p for p in batch.iterdir() if p.is_dir() and NAME.match(p.name)):
        res = json.loads((d / "result.json").read_text(encoding="utf-8"))
        mans = [json.loads(m.read_text(encoding="utf-8")) for m in sorted(d.glob("attempt_*/manifest.json"))]
        vals = [json.loads(v.read_text(encoding="utf-8")) for v in sorted(d.glob("attempt_*/validation.json"))]
        ans = json.loads((d / "answer.json").read_text(encoding="utf-8")) if (d / "answer.json").exists() else None
        lay = json.loads((d / "layout.json").read_text(encoding="utf-8"))
        out.append({"name": d.name, "res": res, "mans": mans, "vals": vals, "answer": ans, "layout": lay})
    return out


def quality(qs: list[dict]) -> dict:
    soft = Counter()
    soft_lists = defaultdict(list)
    for q in qs:
        for v in q["vals"]:
            for k, x in v["soft"].items():
                if isinstance(x, list):
                    soft[k] += len(x)
                    if x:
                        soft_lists[k].append({"questionnaire": q["name"], "items": x})
                else:
                    soft[k] += int(x)
    settings = Counter((m.get("model_recorded"), m.get("effort_recorded"), m.get("service_tier_recorded")) for q in qs for m in q["mans"])
    return {
        "questionnaires": len(qs),
        "status": dict(Counter(q["res"]["status"] for q in qs)),
        "attempts_total": sum(len(q["mans"]) for q in qs),
        "retries": sum(len(q["mans"]) - 1 for q in qs),
        "valid_first_attempt": sum(1 for q in qs if q["vals"] and q["vals"][0]["ok"]),
        "schema_errors": sum(len(v["errors"]) for q in qs for v in q["vals"]),
        "tool_calls": sum(len(m.get("tool_calls") or []) for q in qs for m in q["mans"]),
        "agents_md_loaded": sum(1 for q in qs for m in q["mans"] if m.get("global_agents_md_loaded")),
        "settings_problems": sum(1 for q in qs for m in q["mans"] if m.get("settings_problem")),
        "internal_errors": sum(len(m.get("internal_errors") or []) for q in qs for m in q["mans"]),
        "recorded_settings": {" / ".join(map(str, k)): v for k, v in settings.items()},
        "part_order": dict(Counter(q["layout"]["part_order"] for q in qs)),
        "soft_notes": dict(soft),
        "soft_note_details": dict(soft_lists),
    }


def cr_by_criterion(pool: analyse.Pool) -> dict:
    lay = pool.lay
    crc = core.cr(lay.crit_matrices(pool.Yc))
    crv = core.cr(lay.var_matrices(pool.Ys))           # N x 3
    out = {"criteria": {"n": int(len(crc)), "over_0.10": int((crc > 0.1).sum())}}
    for k, c in enumerate(core.RATED):
        x = crv[:, k]
        out[f"variants_{c}"] = {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)),
                                "max": float(x.max()), "over_0.10": int((x > 0.1).sum())}
    allmax = np.maximum(crc, crv.max(1))
    out["questionnaires_all_four_le_0.10"] = int((allmax <= 0.1).sum())
    out["questionnaires_with_some_matrix_over_0.10"] = [pool.qs[i]["name"] for i in np.flatnonzero(allmax > 0.1)]
    return out


def cost(qs: list[dict]) -> dict:
    walls, inp, cached, outp, reas, readings = [], [], [], [], [], []
    for q in qs:
        for m in q["mans"]:
            walls.append(m.get("wall_seconds") or 0)
            u = (m.get("usage") or [{}])[-1] or {}
            inp.append(u.get("input_tokens", 0)); cached.append(u.get("cached_input_tokens", 0))
            outp.append(u.get("output_tokens", 0)); reas.append(u.get("reasoning_output_tokens", 0))
            rl = m.get("rate_limits_after") or {}
            readings.append((m.get("started_utc") or "", (rl.get("primary") or {}).get("used_percent"),
                             (rl.get("secondary") or {}).get("used_percent"), (rl.get("credits") or {}).get("balance")))
    readings.sort()
    st = lambda a: {"median": float(np.median(a)), "min": float(min(a)), "max": float(max(a)), "mean": float(np.mean(a)),
                    "total": float(sum(a))}
    return {"calls": len(walls), "wall_seconds": st(walls), "input_tokens": st(inp), "cached_input_tokens": st(cached),
            "output_tokens": st(outp), "reasoning_output_tokens": st(reas),
            "first_call_started_utc": readings[0][0], "last_call_started_utc": readings[-1][0],
            "readings_after_first_and_last_call": [readings[0][1:], readings[-1][1:]],
            "credit_balances": sorted({r[3] for r in readings})}


def union_rule(R: dict, scheme: str, tau: float) -> dict:
    """PROTOCOL.md 7.5/7.9 union rule applied to the stored per-mode decisions of analyse.py."""
    dec = {m: R["decision"][f"{scheme}/{m}/{tau}"] for m in ("ratings", "relative")}
    acc = {m: {r["pair"]: r for r in dec[m]["accepted_pairs"]} for m in dec}
    unst = {m: {r["pair"]: r for r in dec[m]["unstable_pairs"]} for m in dec}
    m5 = sorted(p for p in set(acc["ratings"]) & set(acc["relative"]) if acc["ratings"][p]["higher"] != acc["relative"][p]["higher"])
    hand = sorted(set(unst["ratings"]) | set(unst["relative"]) | set(m5))
    both = sorted(p for p in set(acc["ratings"]) & set(acc["relative"]) if p not in m5)
    row = lambda m, p: (acc[m].get(p) or unst[m].get(p))
    return {"scheme": scheme, "tau": tau,
            "accepted_both_same_direction": [{"pair": p, "higher": acc["ratings"][p]["higher"],
                                              "p_ratings": acc["ratings"][p]["p"], "p_relative": acc["relative"][p]["p"]} for p in both],
            "to_human_experts": [{"pair": p, "unstable_in": [m for m in dec if p in unst[m]], "mode_conflict": p in m5,
                                  "ratings": {"higher": row("ratings", p)["higher"], "p": row("ratings", p)["p"]},
                                  "relative": {"higher": row("relative", p)["higher"], "p": row("relative", p)["p"]}} for p in hand],
            "stable_ranks": {m: dec[m]["stable_ranks"] for m in dec},
            "unstable_ranks": {m: [r for r in range(1, 6) if r not in {x["rank"] for x in dec[m]["stable_ranks"]}] for m in dec}}


def justification_scan(qs: list[dict]) -> dict:
    """Exploratory: words that would show the persona texts leaking past the scope line, and any value given for D."""
    pats = {"control_word": r"\bcontrols?\b", "security_control": r"security controls?",
            "criterion_P_or_U": r"criterion [PU]\b|\((?:P|U)\)", "implementation_cost": r"implementation|\bcost\b",
            "D_value": r"\bD\s*(?:=|of|is|at)\s*\d", "path_directness": r"directness|event count|number of (?:steps|events)"}
    out = {k: {"questionnaires": 0, "texts": 0, "examples": []} for k in pats}
    for q in qs:
        hit_q = Counter()
        for j in q["answer"]["justifications"]:
            for k, p in pats.items():
                m = re.search(p, j["text"], flags=re.I)
                if m:
                    out[k]["texts"] += 1
                    hit_q[k] += 1
                    if len(out[k]["examples"]) < 6:
                        s = max(0, m.start() - 60)
                        out[k]["examples"].append({"q": q["name"], "block": j["block_id"], "context": j["text"][s:m.end() + 60]})
        for k in hit_q:
            out[k]["questionnaires"] += 1
    lens = [len(j["text"].split()) for q in qs for j in q["answer"]["justifications"]]
    out["justification_words"] = {"median": float(np.median(lens)), "min": int(min(lens)), "max": int(max(lens))}
    return out


def persona_views(pool: analyse.Pool, R: dict) -> dict:
    """Exploratory: criteria weights and geometric-mean ratings of each persona's nine questionnaires."""
    lay, V = pool.lay, pool.lay.variants
    out = {}
    for i, p in enumerate(pool.personas):
        L = pool.by_persona[i]
        w = lay.weights(pool.Yc[L].mean(0))
        r = core.agg_ratings(pool.R[L], axis=0)
        mp_l = pool.R[L, core.RATED.index("L"), V.index("MP")]
        out[p] = {"weights": dict(zip(core.CRITERIA, np.round(w, 3).tolist())),
                  "ratings_geo": {c: dict(zip(V, np.round(r[k], 2).tolist())) for k, c in enumerate(core.RATED)},
                  "MP_L_answers": sorted(mp_l.astype(int).tolist())}
    return out


def by_wording_run(pool: analyse.Pool) -> dict:
    """Exploratory: S1-style group of each wording and of each run (21 questionnaires each), ratings and relative mode."""
    V = pool.lay.variants
    out = {}
    for key in ("wording", "run"):
        vals = sorted({q[key] for q in pool.qs})
        for v in vals:
            idx = np.array([i for i, q in enumerate(pool.qs) if q[key] == v])
            w, g_r, g_p = analyse.group_scores(pool, idx[None, :])
            out[f"{key}={v}"] = {"weights": dict(zip(core.CRITERIA, np.round(w[0], 3).tolist())),
                                 "order_ratings": [V[k] for k in np.argsort(-g_r[0], kind="stable")],
                                 "order_relative": [V[k] for k in np.argsort(-g_p[0], kind="stable")],
                                 "scores_ratings": dict(zip(V, np.round(g_r[0], 4).tolist())),
                                 "scores_relative": dict(zip(V, np.round(g_p[0], 4).tolist()))}
    return out


EXTRA_SEED = 2026102790   # B-Q bootstrap of the pooled weights and scores (not in the frozen plan)


def weight_intervals(pool: analyse.Pool, draws: int, B: int = 2000) -> dict:
    """Intervals of the criteria weights and S1 scores that analyse.py does not write.
    S2/S3: 2.5-97.5 % quantiles over the same simulated groups as analyse.smaa (same seeds, same draw order).
    S1: B-Q bootstrap (questionnaires resampled within each persona), seed EXTRA_SEED."""
    lay, V = pool.lay, pool.lay.variants
    out = {}
    for s in ("S2", "S3"):
        rng = np.random.default_rng(analyse.SEEDS[s])
        idx = analyse.draw(pool, s, draws, rng)
        w, g_r, g_p = analyse.group_scores(pool, idx)
        out[s] = {"weights_q025_q975": {c: np.quantile(w[:, k], [0.025, 0.975]).round(4).tolist() for k, c in enumerate(core.CRITERIA)},
                  "scores_ratings_q025_q975": {v: np.quantile(g_r[:, i], [0.025, 0.975]).round(4).tolist() for i, v in enumerate(V)},
                  "scores_relative_q025_q975": {v: np.quantile(g_p[:, i], [0.025, 0.975]).round(4).tolist() for i, v in enumerate(V)}}
    rng = np.random.default_rng(EXTRA_SEED)
    ws, grs, gps, lead_r, lead_p = [], [], [], [], []
    for _ in range(B):
        idx = np.concatenate([L[rng.integers(0, len(L), len(L))] for L in pool.by_persona])[None, :]
        w, g_r, g_p = analyse.group_scores(pool, idx)
        ws.append(w[0]); grs.append(g_r[0]); gps.append(g_p[0])
    ws, grs, gps = np.array(ws), np.array(grs), np.array(gps)
    out["S1_BQ"] = {"B": B, "seed": EXTRA_SEED,
                    "weights_q025_q975": {c: np.quantile(ws[:, k], [0.025, 0.975]).round(4).tolist() for k, c in enumerate(core.CRITERIA)},
                    "scores_ratings_q025_q975": {v: np.quantile(grs[:, i], [0.025, 0.975]).round(4).tolist() for i, v in enumerate(V)},
                    "scores_relative_q025_q975": {v: np.quantile(gps[:, i], [0.025, 0.975]).round(4).tolist() for i, v in enumerate(V)},
                    "leader_share_ratings": {v: float(np.mean(grs.argmax(1) == i)) for i, v in enumerate(V)},
                    "leader_share_relative": {v: float(np.mean(gps.argmax(1) == i)) for i, v in enumerate(V)},
                    "weight_order_share": dict(Counter("".join(core.CRITERIA[k] for k in np.argsort(-w)) for w in ws).most_common(5))}
    return out


def single_questionnaire_leaders(pool: analyse.Pool) -> dict:
    """First places over the 63 single questionnaires (ties split), both modes."""
    V = pool.lay.variants
    res = {"ratings": Counter(), "relative": Counter()}
    for i in range(pool.N):
        _, a, b = analyse.group_scores(pool, np.array([[i]]))
        for mode, g in (("ratings", a[0]), ("relative", b[0])):
            top = np.flatnonzero(np.abs(g - g.max()) <= core.TIE)
            for t in top:
                res[mode][V[t]] += 1 / len(top)
    return {m: {v: round(c[v], 2) for v in V} for m, c in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="runs/main")
    ap.add_argument("--results", default="results/main")
    args = ap.parse_args()
    root = HERE.parent
    batch, res_dir = root / args.batch, root / args.results
    R = json.loads((res_dir / "endtoend_results.json").read_text(encoding="utf-8"))
    qs = load_questionnaires(batch)
    pool = analyse.Pool(batch, core.load_case())
    X = {"note": "Not part of the frozen plan. Counts, the union rule at tau 0.8 and 0.95 derived from the stored "
                 "per-mode decisions, and exploratory views (marked). Written after the frozen analysis.",
         "quality": quality(qs), "cr_by_criterion": cr_by_criterion(pool), "cost": cost(qs),
         "union_rule": {f"{s}/{t}": union_rule(R, s, t) for s in ("S3", "S2") for t in (0.8, 0.9, 0.95)},
         "intervals_not_in_plan": weight_intervals(pool, R["settings"]["draws"]),
         "monte_carlo_se_at_tau": {str(t): round(float(np.sqrt(t * (1 - t) / R["settings"]["draws"])), 4) for t in (0.8, 0.9, 0.95)},
         "exploratory": {"justification_scan": justification_scan(qs), "personas": persona_views(pool, R),
                         "by_wording_and_run": by_wording_run(pool),
                         "single_questionnaire_first_places": single_questionnaire_leaders(pool)}}
    # consistency check: the derived union at 0.9 must equal analyse.py's checkpoint2
    u9 = X["union_rule"]["S3/0.9"]
    assert sorted(x["pair"] for x in u9["to_human_experts"]) == sorted(R["checkpoint2"]["pairs_to_human_experts"])
    assert sorted(x["pair"] for x in u9["accepted_both_same_direction"]) == sorted(x["pair"] for x in R["checkpoint2"]["accepted_pairs"])
    (res_dir / "extras.json").write_text(json.dumps(X, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"wrote {res_dir / 'extras.json'}")


if __name__ == "__main__":
    main()
