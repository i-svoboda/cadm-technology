"""ANP model of ITS 14(1): questionnaire blocks, prompt builder and supermatrix solver.

Adapted from tacs_smaa_llm_elicitation/scripts/anp_case.py and elicit.py (read only, not imported).

Supermatrix convention (ITS 14(1), Stage 3): block W_ij holds the influence of the nodes of
cluster i (rows) on the nodes of cluster j (columns); every non-zero sub-column sums to 1.
The weighted supermatrix scales block (i, j) by the cluster weight C_ij, built from the
published prior v = (K 0.4466, H 0.2549, R 0.2506, B 0.0480) by setting C_ij = 0 for
structurally empty blocks and renormalizing the rest of the column (formula (1)).

Blocks elicited (32 matrices):
  R|GOAL   criteria with respect to the goal                      (Table 3, row 1)
  K|R:*    controls with respect to each criterion  (5)           ("Controls -> Criteria"; Table 3)
  H|R:*    threats with respect to each criterion   (5)           ("Threats -> Criteria")
  B|R:*    constraints with respect to each criterion (5)         ("Constraints -> Criteria")
  B|K:*    constraints with respect to each control (7)           ("Constraints -> Controls")
  K|B:*    controls with respect to each constraint (3)           (feedback "Controls -> Constraints")
  K|K:MON  SEG against IDM as sources of synergy for MON (1)      (synergies SEG->MON, IDM->MON)
  K|H:*    controls with respect to each threat     (5)           (not named in the text; added, see DESIGN.md)
Fixed blocks: synergies SEG->IDM and BAK->IR (single sources, weight 1); the criteria
vector R|GOAL fills the criteria rows of every control column (feedback that links the
criteria back to the controls) and of every criterion column (inner dependence "Criteria <->
Criteria"). DESIGN.md explains why these choices are needed for a strongly connected network.
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np

V_ITS = {"K": 0.4466, "H": 0.2549, "R": 0.2506, "B": 0.0480}
SYNERGY = [("SEG", "MON"), ("IDM", "MON"), ("SEG", "IDM"), ("BAK", "IR")]
CLUSTER_ORDER = ["K", "R", "H", "B"]

SCALE = ("Use Saaty's 1-9 scale: 1 = equal, 3 = moderately more, 5 = strongly more, 7 = very strongly more, "
         "9 = extremely more; 2, 4, 6 and 8 are intermediate values.")

INTRO = ("You are taking part in a structured expert elicitation for the Analytic Network Process (ANP). Answer as "
         "the expert described in your instructions, from that expert's own professional experience and priorities.")

QUESTIONS = {
    "R|GOAL": "With respect to the overall decision (which security controls this operator should fund first), which "
              "evaluation criterion is more important?",
    "K|R": "With respect to criterion {col} ({col_name}), which control performs better?",
    "K|H": "With respect to threat vector {col} ({col_name}), which control reduces this risk more?",
    "H|R": "When controls are judged on criterion {col} ({col_name}), which threat vector should carry more weight for "
           "this operator?",
    "B|R": "When controls are judged on criterion {col} ({col_name}), which implementation constraint should carry "
           "more weight?",
    "B|K": "For control {col} ({col_name}), which constraint binds more?",
    "K|B": "Under constraint {col} ({col_name}), which control is easier to implement?",
    "K|K": "Network segmentation (SEG) improves observability for monitoring, and identity, MFA and privileged access "
           "management (IDM) enriches its detection context. Which of the two contributes more to the value of ICS "
           "network monitoring and anomaly detection (MON)?",
}
COVERAGE_Q = ("Coverage. For each control and threat vector below, estimate the share of the expected annual loss from "
              "that threat that the control alone would remove at this operator, as a number from 0 to 1.")
LOSS_Q = ("Expected annual loss. Estimate the expected annual loss, in million US dollars, that each threat vector "
          "would cause this operator if none of the seven controls above were added.")


class Case:
    def __init__(self, protocol: dict):
        self.p = protocol
        self.goal = protocol["goal"]
        self.clusters = protocol["clusters"]
        self.ids = {c: [n["id"] for n in self.clusters[c]["nodes"]] for c in self.clusters}
        self.names = {n["id"]: n["name"] for c in self.clusters.values() for n in c["nodes"]}
        self.cluster_of = {n: c for c, ns in self.ids.items() for n in ns}

    # ---------------------------------------------------------- questionnaire
    def matrix_specs(self) -> list[tuple[str, str, str | None]]:
        """(matrix key, row cluster, column node or None)."""
        specs = [("R|GOAL", "R", None)]
        for rows, colc in [("K", "R"), ("K", "H"), ("H", "R"), ("B", "R"), ("B", "K"), ("K", "B")]:
            for col in self.ids[colc]:
                specs.append((f"{rows}|{colc}:{col}", rows, col))
        specs.append(("K|K:MON", "K", "MON"))
        return specs

    def matrix_elements(self, key: str) -> list[str]:
        if key == "K|K:MON":
            return ["SEG", "IDM"]
        return self.ids[key.split("|")[0]]

    def questions(self) -> list[dict]:
        qs = []
        for key, rows, col in self.matrix_specs():
            for a, b in itertools.combinations(self.matrix_elements(key), 2):
                qs.append({"matrix": key, "a": a, "b": b})
        return qs

    def coverage_items(self) -> list[dict]:
        out = []
        for i, (c, t) in enumerate(itertools.product(self.ids["K"], self.ids["H"]), 1):
            out.append({"item_id": f"V{i:02d}", "control": c, "threat": t})
        return out

    def loss_items(self) -> list[dict]:
        return [{"item_id": f"L{i}", "threat": t} for i, t in enumerate(self.ids["H"], 1)]

    def build_prompt(self, seed: int) -> tuple[str, list[dict]]:
        rng = random.Random(seed)
        blocks: dict[str, list[dict]] = {}
        for q in self.questions():
            blocks.setdefault(q["matrix"], []).append(q)
        order = list(blocks)
        rng.shuffle(order)
        lines, items, n = [], [], 0
        for key in order:
            prefix = key.split(":")[0] if key != "K|K:MON" else "K|K"
            col = key.split(":")[1] if ":" in key else ""
            lines.append("\n" + QUESTIONS[prefix].format(col=col, col_name=self.names.get(col, "")))
            block = blocks[key][:]
            rng.shuffle(block)
            for q in block:
                first, second = (q["a"], q["b"]) if rng.random() < 0.5 else (q["b"], q["a"])
                n += 1
                pid = f"Q{n:03d}"
                items.append({"pair_id": pid, "matrix": key, "first": first, "second": second})
                lines.append(f"{pid}: first = {first} ({self.names[first]}); second = {second} ({self.names[second]})")
        cov = self.coverage_items()
        lines.append("\n" + COVERAGE_Q)
        for it in cov:
            items.append({"coverage_id": it["item_id"], "control": it["control"], "threat": it["threat"]})
            lines.append(f"{it['item_id']}: control {it['control']} ({self.names[it['control']]}) against threat "
                         f"{it['threat']} ({self.names[it['threat']]})")
        lines.append("\n" + LOSS_Q)
        for it in self.loss_items():
            items.append({"loss_id": it["item_id"], "threat": it["threat"]})
            lines.append(f"{it['item_id']}: {it['threat']} ({self.names[it['threat']]})")
        cl = "\n\n".join(f"{self.clusters[c]['name']}:\n" + "\n".join(f"- {i}: {self.names[i]}" for i in self.ids[c])
                         for c in CLUSTER_ORDER)
        prompt = (
            f"{INTRO}\n\nDecision problem: {self.goal}.\n\nThe network has four groups of elements.\n\n{cl}\n\n"
            f"For every numbered pair below, state which element is preferred ('first', 'second' or 'equal') and the "
            f"intensity of the preference. {SCALE} Use intensity 1 only with 'equal'. Do not search the web, read "
            f"files or run commands. Return one entry for every pair, one share for every coverage item and one value "
            f"for every loss item, and nothing else, as the JSON object required by the output schema.\n"
            + "\n".join(lines) + "\n")
        return prompt, items

    # ---------------------------------------------------------- answers -> matrices
    def to_logs(self, items: list[dict], answer: dict) -> tuple[dict, list[str]]:
        got = {j["pair_id"]: j for j in answer.get("judgments", [])}
        missing, logj = [], {}
        for it in items:
            if "pair_id" not in it:
                continue
            j = got.get(it["pair_id"])
            if j is None:
                missing.append(it["pair_id"])
                continue
            v = 0.0 if j["preferred"] == "equal" or j["intensity"] == 1 else math.log(j["intensity"])
            if j["preferred"] == "second":
                v = -v
            logj[(it["matrix"], it["first"], it["second"])] = v
        return logj, missing

    def matrices(self, logj: dict) -> dict[str, np.ndarray]:
        out = {}
        for key, rows, col in self.matrix_specs():
            ids = self.matrix_elements(key)
            out[key] = np.ones((len(ids), len(ids)))
        for (key, a, b), v in logj.items():
            ids = self.matrix_elements(key)
            i, j = ids.index(a), ids.index(b)
            out[key][i, j], out[key][j, i] = math.exp(v), math.exp(-v)
        return out

    def coverage(self, items: list[dict], answer: dict) -> tuple[dict, dict, list[str]]:
        cov = {c["item_id"]: c.get("share") for c in answer.get("coverage", [])}
        loss = {c["item_id"]: c.get("million_usd") for c in answer.get("expected_loss", [])}
        alpha, eal, missing = {}, {}, []
        for it in items:
            if "coverage_id" in it:
                v = cov.get(it["coverage_id"])
                if v is None:
                    missing.append(it["coverage_id"])
                else:
                    alpha[(it["control"], it["threat"])] = float(min(1.0, max(0.0, v)))
            if "loss_id" in it:
                v = loss.get(it["loss_id"])
                if v is None:
                    missing.append(it["loss_id"])
                else:
                    eal[it["threat"]] = float(v)
        return alpha, eal, missing

    # ---------------------------------------------------------- supermatrix
    def node_index(self) -> list[tuple[str, str]]:
        return [(c, n) for c in CLUSTER_ORDER for n in self.ids[c]]

    def supermatrix(self, local: dict[str, np.ndarray], design: str = "D1", v: dict | None = None,
                    threat_mult: dict | None = None, synergy: list | None = None) -> tuple[list, np.ndarray, np.ndarray]:
        """local: matrix key -> local priority vector (aligned with matrix_elements). Returns idx, W, Ww."""
        v = v or V_ITS
        synergy = SYNERGY if synergy is None else synergy
        idx = self.node_index()
        pos = {n: i for i, (_, n) in enumerate(idx)}
        N = len(idx)
        W = np.zeros((N, N))
        Ww = np.zeros((N, N))
        goal = dict(zip(self.ids["R"], local["R|GOAL"]))
        for j, (cj, nj) in enumerate(idx):
            blocks: dict[str, dict[str, float]] = {}
            for key in local:
                if key in ("R|GOAL", "K|K:MON") or ":" not in key:
                    continue
                rows, col = key.split("|")
                if col.split(":")[1] == nj and col.split(":")[0] == cj:
                    blocks[rows] = dict(zip(self.matrix_elements(key), local[key]))
            if cj == "K":
                if design in ("D1", "D2"):
                    blocks["R"] = goal                           # criteria feedback into every control
                src = [s for s, t in synergy if t == nj]
                if sorted(src) == ["IDM", "SEG"]:
                    blocks["K"] = dict(zip(self.matrix_elements("K|K:MON"), local["K|K:MON"]))
                elif src:
                    blocks["K"] = {s: 1.0 / len(src) for s in src}
            if cj == "R" and design == "D1":
                blocks["R"] = goal                               # inner dependence of the criteria
            if cj == "R" and threat_mult and "H" in blocks:
                h = {t: w * threat_mult.get(t, 1.0) for t, w in blocks["H"].items()}
                s = sum(h.values())
                blocks["H"] = {t: w / s for t, w in h.items()}
            if not blocks:
                raise ValueError(f"column {nj} has no active block")
            total = sum(v[c] for c in blocks)
            for c, vec in blocks.items():
                for nid, w in vec.items():
                    W[pos[nid], j] = w
                    Ww[pos[nid], j] = v[c] / total * w
        return idx, W, Ww

    def limit(self, Ww: np.ndarray, tol: float = 1e-13, max_iter: int = 200000) -> tuple[np.ndarray, dict]:
        x = np.full(Ww.shape[0], 1.0 / Ww.shape[0])
        for it in range(max_iter):
            y = Ww @ x
            if np.abs(y - x).sum() < tol:
                y = y / y.sum()
                break
            x = y
        else:
            acc = np.zeros_like(x)                              # Cesaro average for periodic chains
            for _ in range(5000):
                x = Ww @ x
                acc += x
            y = acc / acc.sum()
            it = -1
        ev = np.sort(np.abs(np.linalg.eigvals(Ww)))[::-1]
        return y, {"iterations": it, "lambda2": float(ev[1]) if len(ev) > 1 else None,
                   "column_sums_ok": bool(np.allclose(Ww.sum(axis=0), 1.0))}

    def control_priorities(self, local: dict, design: str = "D1", **kw) -> tuple[dict, dict]:
        idx, W, Ww = self.supermatrix(local, design, **kw)
        pi, diag = self.limit(Ww)
        k = [(i, n) for i, (c, n) in enumerate(idx) if c == "K"]
        s = sum(pi[i] for i, _ in k)
        diag["cluster_mass"] = {c: float(sum(pi[i] for i, (cc, _) in enumerate(idx) if cc == c)) for c in CLUSTER_ORDER}
        return {n: float(pi[i] / s) for i, n in k}, diag

    def hierarchical(self, local: dict) -> dict:
        """AHP-style composition: sum over criteria of goal weight x control priority under the criterion."""
        goal = dict(zip(self.ids["R"], local["R|GOAL"]))
        out = {k: 0.0 for k in self.ids["K"]}
        for c, wc in goal.items():
            for k, w in zip(self.ids["K"], local[f"K|R:{c}"]):
                out[k] += wc * w
        return out
