"""Shared helpers: paths, protocol loading, the AGENTS.md pause, schemas and matrix parsing."""
from __future__ import annotations

import atexit
import json
import math
import os
import re
import signal
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
SCHEMAS = PROTOCOL / "schemas"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codex_turn  # noqa: E402

RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def load(name: str) -> dict:
    return json.loads((PROTOCOL / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ global AGENTS.md pause
class PausedGlobalAgentsMd:
    """Rename ~/.codex/AGENTS.md for the duration of a batch and always put it back.

    Same protocol as scripts/run_batch_clean.sh of the TACS SMAA project (author decision
    of 21 October 2026), so that the user's coding working agreement does not enter the
    virtual experts' context.
    """
    G = codex_turn.GLOBAL_AGENTS
    P = codex_turn.GLOBAL_AGENTS.with_name("AGENTS.md.paused-by-claude-replication")
    H = codex_turn.GLOBAL_AGENTS.with_name("AGENTS.md.paused-by-claude-replication.holders")

    @classmethod
    def _live_holders(cls) -> list[Path]:
        import psutil
        cls.H.mkdir(exist_ok=True)
        live = []
        for f in cls.H.glob("*.pid"):
            try:
                pid = int(f.stem)
            except ValueError:
                continue
            if pid != os.getpid() and psutil.pid_exists(pid):
                live.append(f)
            elif pid != os.getpid():
                f.unlink(missing_ok=True)                  # holder of a crashed batch
        return live

    @classmethod
    def restore(cls) -> None:
        """Put the file back unless another running batch still holds the pause."""
        (cls.H / f"{os.getpid()}.pid").unlink(missing_ok=True)
        if cls._live_holders():
            return
        if cls.P.exists() and not cls.G.exists():
            cls.P.rename(cls.G)
            print(f"[restored {cls.G}]", flush=True)

    def __enter__(self):
        if not self._live_holders() and self.P.exists() and not self.G.exists():
            self.P.rename(self.G)                          # recover from an earlier crash
        (self.H / f"{os.getpid()}.pid").write_text(str(os.getpid()), encoding="utf-8")
        if self.G.exists():
            if self.P.exists():
                raise RuntimeError(f"both {self.G} and {self.P} exist; resolve by hand")
            self.G.rename(self.P)
            print(f"[paused {self.G}]", flush=True)
        atexit.register(self.restore)
        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
            if sig is not None:
                try:
                    signal.signal(sig, self._on_signal)
                except (ValueError, OSError):
                    pass
        return self

    def _on_signal(self, signum, frame):
        self.restore()
        raise SystemExit(f"signal {signum}")

    def __exit__(self, *exc):
        self.restore()
        return False


# ------------------------------------------------------------------ schemas
def obj(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


SCHEMA_DEFS = {
    "matrices": obj({"matrices": {"type": "array", "items": obj({
        "name": {"type": "string"},
        "elements": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "rationale": {"type": "string"}})}}),
    "criteria_list": obj({"criteria": {"type": "array", "items": {"type": "string"}}}),
    "sub_sets": obj({"sets": {"type": "array", "items": obj({
        "criterion": {"type": "string"}, "sub_criteria": {"type": "array", "items": {"type": "string"}}})}}),
    "alternatives_list": obj({"alternatives": {"type": "array", "items": {"type": "string"}}}),
    "scores": obj({"scores": {"type": "array", "items": obj({
        "number": {"type": "string"}, "item": {"type": "string"},
        "score": {"type": "integer", "enum": list(range(1, 10))}})}}),
    "anp": obj({
        "judgments": {"type": "array", "items": obj({
            "pair_id": {"type": "string"},
            "preferred": {"type": "string", "enum": ["first", "second", "equal"]},
            "intensity": {"type": "integer", "enum": list(range(1, 10))}})},
        "coverage": {"type": "array", "items": obj({"item_id": {"type": "string"}, "share": {"type": "number"}})},
        "expected_loss": {"type": "array", "items": obj({"item_id": {"type": "string"},
                                                         "million_usd": {"type": "number"}})}}),
}


def schema_path(name: str) -> Path:
    SCHEMAS.mkdir(parents=True, exist_ok=True)
    p = SCHEMAS / f"{name}.schema.json"
    text = json.dumps(SCHEMA_DEFS[name], indent=1)
    if not p.exists() or p.read_text(encoding="utf-8") != text:
        p.write_text(text, encoding="utf-8")
    return p


# ------------------------------------------------------------------ names and values
def norm_name(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    words = []
    for w in s.split():
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


def name_similarity(a: str, b: str) -> float:
    x, y = set(norm_name(a).split()), set(norm_name(b).split())
    return len(x & y) / max(1, len(x | y))


def parse_value(s) -> float | None:
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip().replace(" ", "")
    try:
        if "/" in t:
            a, b = t.split("/", 1)
            return float(a) / float(b)
        return float(t)
    except (ValueError, ZeroDivisionError):
        return None


def match_order(answer_names: list[str], expected_names: list[str]) -> list[int] | None:
    """Index in answer_names of each expected name (exact normalized match, then best overlap)."""
    n = len(expected_names)
    if len(answer_names) != n:
        return None
    na = [norm_name(x) for x in answer_names]
    ne = [norm_name(x) for x in expected_names]
    if na == ne:
        return list(range(n))
    out, used = [], set()
    for e in expected_names:
        scores = [(name_similarity(a, e), i) for i, a in enumerate(answer_names) if i not in used]
        best = max(scores) if scores else (0, None)
        if best[0] < 0.34:
            return None
        out.append(best[1])
        used.add(best[1])
    return out


def check_matrix(m: dict, expected_names: list[str], tol: float = 0.05) -> tuple[np.ndarray | None, list[str]]:
    """Parse one matrix; return it in the expected element order and a list of problems."""
    probs = []
    n = len(expected_names)
    rows = m.get("rows") or []
    if len(rows) != n or any(len(r) != n for r in rows):
        return None, [f"it is not a {n}x{n} matrix"]
    A = np.array([[parse_value(v) if parse_value(v) is not None else np.nan for v in r] for r in rows], dtype=float)
    if np.isnan(A).any():
        return None, ["some cells are not numbers"]
    order = match_order(m.get("elements") or [], expected_names)
    if order is None:
        order = list(range(n))
        if m.get("elements") and [norm_name(x) for x in m["elements"]] != [norm_name(x) for x in expected_names]:
            probs.append("its elements do not match the listed elements")
    A = A[np.ix_(order, order)]
    if (A <= 0).any():
        return A, ["it has zero or negative values"]
    if (A < 1 / 9 - 1e-3).any() or (A > 9 + 1e-3).any():
        probs.append("some values are outside 1/9 to 9")
    if not np.allclose(np.diag(A), 1.0, atol=1e-3):
        probs.append("the diagonal is not 1")
    L = np.log(A)
    bad = np.abs(L + L.T) > tol
    np.fill_diagonal(bad, False)
    if bad.any():
        probs.append(f"{int(bad.sum() // 2)} pairs of mirrored values are not reciprocal")
    return A, probs


def reciprocal_projection(A: np.ndarray) -> np.ndarray:
    """Closest reciprocal matrix in the log least-squares sense: b_ij = sqrt(a_ij / a_ji)."""
    L = np.log(A)
    S = (L - L.T) / 2
    return np.exp(S)


# ------------------------------------------------------------------ AHP maths
def priorities_eigen(A: np.ndarray) -> tuple[np.ndarray, float]:
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    return w / w.sum(), float(vals[k].real)


def priorities_approx(A: np.ndarray) -> np.ndarray:
    """Normalized-column row averages, formulas (3)-(4) of arXiv 2402.07404."""
    N = A / A.sum(axis=0, keepdims=True)
    return N.mean(axis=1)


def consistency(A: np.ndarray) -> dict:
    n = A.shape[0]
    w, lam = priorities_eigen(A)
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    cr = ci / RI[n] if RI.get(n) else 0.0
    return {"lambda_max": lam, "CI": ci, "CR": cr, "koczkodaj": koczkodaj(A)}


def koczkodaj(A: np.ndarray) -> float:
    n = A.shape[0]
    worst = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = A[i, k], A[i, j], A[j, k]           # triad (a_ik, a_ij, a_jk): consistent if a_ik = a_ij a_jk
                worst = max(worst, min(abs(1 - a / (b * c)), abs(1 - (b * c) / a)))
    return worst


def geo_mean(mats: list[np.ndarray]) -> np.ndarray:
    return np.exp(np.mean([np.log(M) for M in mats], axis=0))


def kendall_w(ranks: np.ndarray) -> tuple[float, float]:
    """ranks: m raters x N objects (1 = best). Returns W and the chi-square p-value."""
    from scipy.stats import chi2
    m, N = ranks.shape
    R = ranks.sum(axis=0)
    S = ((R - R.mean()) ** 2).sum()
    # tie correction
    T = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        T += ((counts ** 3) - counts).sum()
    W = 12 * S / (m ** 2 * (N ** 3 - N) - m * T)
    p = float(chi2.sf(m * (N - 1) * W, N - 1))
    return float(W), p


def rank_desc(v: dict | list) -> list:
    if isinstance(v, dict):
        return [k for k, _ in sorted(v.items(), key=lambda x: -x[1])]
    return list(np.argsort(-np.asarray(v)))


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"
