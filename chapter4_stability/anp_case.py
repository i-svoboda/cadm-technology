"""ANP questionnaire and solver for the critical-infrastructure controls case.

Supermatrix convention: column j is a node; the rows of block (X, cluster(j)) hold the
priorities of the nodes of cluster X with respect to node j. Columns are made
stochastic with the ITS 14(1) cluster prior v, renormalized over the blocks that are
active in each column (structural zeros removed). The limit is found by powering the
weighted supermatrix, with a Cesaro average as a fallback for periodic chains.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

# Blocks of the 17-node design: (row cluster, column nodes, question template)
BLOCKS = [
    ("K", ("R", None), "With respect to criterion {col} ({col_name}), which control performs better?"),
    ("K", ("H", None), "With respect to threat vector {col} ({col_name}), which control reduces this risk more?"),
    ("R", ("K", None), "For control {col} ({col_name}), which criterion matters more when judging the value of this control?"),
    ("H", ("R", "E"), "For criterion {col} ({col_name}), which threat vector matters more for this operator?"),
]
# Blocks added by the 20-node option
BLOCKS_B = [
    ("K", ("B", None), "Under constraint {col} ({col_name}), which control is easier to implement?"),
    ("B", ("K", None), "For control {col} ({col_name}), which constraint binds more?"),
]
# Control synergies (ITS 14(1)): source -> target. Each is an optional link.
LINKS = [("SEG", "MON"), ("IDM", "MON"), ("SEG", "IDM"), ("BAK", "IR")]
CLUSTERS_17 = ("K", "R", "H")


def nodes(case: dict, cluster: str) -> list[dict]:
    return case["clusters"][cluster]["nodes"]


def node_index(case: dict, with_b: bool) -> list[tuple[str, str]]:
    clusters = ["K", "R", "H"] + (["B"] if with_b else [])
    return [(c, n["id"]) for c in clusters for n in nodes(case, c)]


def anp_questions(case: dict, with_b: bool = False) -> list[dict]:
    qs = []
    for rows, (col_cluster, only), template in BLOCKS + (BLOCKS_B if with_b else []):
        for col in nodes(case, col_cluster):
            if only and col["id"] != only:
                continue
            matrix = f"{rows}|{col_cluster}:{col['id']}"
            text = template.format(col=col["id"], col_name=col["name"])
            for a, b in itertools.combinations(nodes(case, rows), 2):
                qs.append({"matrix": matrix, "a": a, "b": b, "question": text})
    # Synergy comparison for the only column with two optional sources
    k = {n["id"]: n for n in nodes(case, "K")}
    qs.append({"matrix": "K|K:MON", "a": k["SEG"], "b": k["IDM"],
               "question": "If both links exist, which control contributes more to the value of monitoring (MON)?"})
    # Cluster comparison (alternative to the ITS prior vector v)
    cl = [{"id": c, "name": case["clusters"][c]["name"]} for c in CLUSTERS_17]
    for a, b in itertools.combinations(cl, 2):
        qs.append({"matrix": "CL|NET", "a": a, "b": b,
                   "question": "Which group of elements should have more influence on the priorities in this network?"})
    return qs


def anp_link_items(case: dict) -> list[dict]:
    k = {n["id"]: n["name"] for n in nodes(case, "K")}
    return [{"link_id": f"L{i}", "source": s, "target": t,
             "text": f"{s} ({k[s]}) materially increases the value of {t} ({k[t]})"} for i, (s, t) in enumerate(LINKS, 1)]


def eigvec(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(A)
    w = np.abs(vecs[:, int(np.argmax(vals.real))].real)
    return w / w.sum()


def local_matrices(case: dict, logj: dict) -> dict[str, tuple[list[str], np.ndarray]]:
    """Log-judgments keyed (matrix, a, b) with a < b -> reciprocal matrices."""
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for (m, a, b), v in logj.items():
        rows = m.split("|")[0]
        if m == "CL|NET":
            ids = list(CLUSTERS_17)
        elif m == "K|K:MON":
            ids = ["IDM", "SEG"]
        else:
            ids = [n["id"] for n in nodes(case, rows)]
        if m not in out:
            out[m] = (ids, np.ones((len(ids), len(ids))))
        i, j = ids.index(a), ids.index(b)
        out[m][1][i, j], out[m][1][j, i] = math.exp(v), math.exp(-v)
    return out


def cluster_vector(case: dict, mats: dict, source: str) -> dict[str, float]:
    """ITS 14(1) prior vector, or the eigenvector of the elicited cluster comparison."""
    if source == "its":
        return {c: case["cluster_prior_v"][c] for c in ("K", "R", "H", "B")}
    ids, A = mats["CL|NET"]
    return {**dict(zip(ids, eigvec(A))), "B": 0.0}


def supermatrix(case: dict, logj: dict, links_on: dict[str, bool], with_b: bool = False,
                cluster_weights: str = "its") -> tuple[list, np.ndarray]:
    idx = node_index(case, with_b)
    pos = {nid: i for i, (_, nid) in enumerate(idx)}
    n = len(idx)
    mats = local_matrices(case, logj)
    v = cluster_vector(case, mats, cluster_weights)
    W = np.zeros((n, n))
    for j, (cj, nj) in enumerate(idx):
        blocks: dict[str, dict[str, float]] = {}
        for m, (ids, A) in mats.items():
            rows, col = m.split("|")
            if m in ("K|K:MON", "CL|NET") or col != f"{cj}:{nj}":
                continue
            blocks[rows] = dict(zip(ids, eigvec(A)))
        if cj == "K":                                   # optional synergy links into this control
            src = [s for (s, t), on in links_on.items() if on and t == nj]
            if src:
                if len(src) == 2 and "K|K:MON" in mats:
                    ids, A = mats["K|K:MON"]
                    blocks["K"] = dict(zip(ids, eigvec(A)))
                else:
                    blocks["K"] = {s: 1.0 / len(src) for s in src}
        if not blocks:
            raise ValueError(f"column {nj} has no active block")
        total = sum(v[c] for c in blocks)
        for c, vec in blocks.items():
            for nid, w in vec.items():
                W[pos[nid], j] = v[c] / total * w
    return idx, W


def limit_priorities(W: np.ndarray, tol: float = 1e-12, max_iter: int = 100000) -> np.ndarray:
    x = np.full(W.shape[0], 1.0 / W.shape[0])
    for _ in range(max_iter):
        y = W @ x
        if np.abs(y - x).max() < tol:
            return y / y.sum()
        x = y
    # Periodic or slowly mixing chain: Cesaro average of the last iterates
    acc = np.zeros_like(x)
    for _ in range(1000):
        x = W @ x
        acc += x
    return acc / acc.sum()


def control_priorities(case: dict, logj: dict, links_on: dict, with_b: bool = False,
                       cluster_weights: str = "its") -> dict[str, float]:
    idx, W = supermatrix(case, logj, links_on, with_b, cluster_weights)
    pi = limit_priorities(W)
    k = [(i, nid) for i, (c, nid) in enumerate(idx) if c == "K"]
    s = sum(pi[i] for i, _ in k)
    return {nid: float(pi[i] / s) for i, nid in k}
