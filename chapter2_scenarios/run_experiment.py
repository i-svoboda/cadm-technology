#!/usr/bin/env python3
"""Run the asymmetric object/event decomposition and scenario-ranking study.

Object edges describe what the organization contains. Causal edges describe
what happens. The two graphs meet only through typed grounding links.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator

import jsonschema
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = (
    "identity_personnel",
    "enterprise_it",
    "it_ot_reachability",
    "ot_process",
    "continuity_recovery",
    "external_dependencies",
)
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}


@dataclass(frozen=True)
class DirectBaselineSpec:
    condition_id: str
    label: str
    run_folder: str
    mapping_filename: str
    case_brief_filename: str = "direct_joint_case_brief.json"
    prompt_filename: str = "direct_joint_prompt.txt"
    generation_filename: str = "direct_joint_generation.json"
    task_sessions: int = 1
    stored_json_outputs: int = 1


@dataclass(frozen=True)
class DirectBaselineFiles:
    case_brief: Path
    mapping: Path
    prompt: Path
    generation: Path
    source_key: str
    mapping_source_file: str


ULTRA_DIRECT_BASELINE = DirectBaselineSpec(
    condition_id="direct_joint_ultra_generation",
    label="Direct joint generation (ultra reasoning effort)",
    run_folder="direct_joint_ultra",
    mapping_filename="direct_joint_ultra_mapping.json",
)


def resolve_direct_baseline_files(
    root: Path,
    runs_root: Path,
    spec: DirectBaselineSpec = ULTRA_DIRECT_BASELINE,
) -> DirectBaselineFiles:
    """Resolve one frozen direct baseline without reading its eventual output."""
    root, runs_root = root.resolve(), runs_root.resolve()
    try:
        runs_prefix = runs_root.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("The runs directory must be inside the experiment root") from exc
    source_key = f"{spec.run_folder}/{spec.generation_filename}"
    return DirectBaselineFiles(
        case_brief=root / "scenario" / spec.case_brief_filename,
        mapping=root / "scenario" / spec.mapping_filename,
        prompt=runs_root / spec.run_folder / spec.prompt_filename,
        generation=runs_root / spec.run_folder / spec.generation_filename,
        source_key=source_key,
        mapping_source_file=f"{runs_prefix}/{source_key}",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def normalized(value: object) -> str:
    text = re.sub(r"\bot\b", "operational technology", str(value or "").lower())
    return " ".join(re.findall(r"[a-z0-9]+", text))


def similarity(left: object, right: object) -> float:
    a, b = normalized(left), normalized(right)
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    jaccard = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return 0.65 * ratio + 0.35 * jaccard


def node_domain(node: dict) -> str:
    return str(node.get("domain") or node.get("dimension") or "")


def object_edge_endpoints(edge: dict) -> tuple[str, str]:
    if "parent" not in edge or "child" not in edge:
        raise ValueError(f"Object edge must use parent/child: {edge}")
    return str(edge["parent"]), str(edge["child"])


def causal_edge_endpoints(edge: dict) -> tuple[str, str]:
    """Return chronological cause->effect; backward refers only to generation."""
    if "cause" not in edge or "effect" not in edge:
        raise ValueError(f"Causal edge must use cause/effect: {edge}")
    return str(edge["cause"]), str(edge["effect"])


def grounding_endpoints(link: dict) -> tuple[str, str]:
    object_id = link.get("object_id", link.get("object"))
    event_id = link.get("event_id", link.get("event"))
    if object_id is None or event_id is None:
        raise ValueError(f"Grounding link must identify object_id and event_id: {link}")
    return str(object_id), str(event_id)


def object_graph_from_edges(edges: Iterable[dict]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for edge in edges:
        if edge.get("edge_type") == "causal":
            raise TypeError("A causal edge cannot enter the object graph")
        parent, child = (
            (edge["source"], edge["target"])
            if "source" in edge and "target" in edge
            else object_edge_endpoints(edge)
        )
        graph.add_edge(parent, child, relation=edge.get("relation", "contains"))
    return graph


def causal_graph_from_edges(edges: Iterable[dict]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for edge in edges:
        if edge.get("edge_type") == "object" or "parent" in edge or "child" in edge:
            raise TypeError("A structural edge cannot enter the causal graph")
        cause, effect = (
            (edge["source"], edge["target"])
            if "source" in edge and "target" in edge
            else causal_edge_endpoints(edge)
        )
        graph.add_edge(cause, effect, relation=edge.get("relation", "causes"))
    return graph


def _assert_dag(graph: nx.DiGraph, label: str) -> None:
    if graph and not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"{label} must be a directed acyclic graph")


def validate_reference_model(reference: dict) -> None:
    object_ids = {row["id"] for row in reference["object_nodes"]}
    event_ids = {row["id"] for row in reference["event_nodes"]}
    if object_ids & event_ids:
        raise ValueError("Reference object and event identifiers must be disjoint")
    root = reference["endpoints"]["object_root"]
    target = reference["endpoints"]["target_event"]
    if root not in object_ids or target not in event_ids:
        raise ValueError("Reference endpoints have the wrong semantic type")
    for edge in reference["object_edges"]:
        parent, child = object_edge_endpoints(edge)
        if parent not in object_ids or child not in object_ids:
            raise ValueError(f"Structural edge crosses the type boundary: {edge}")
    for edge in reference["causal_edges"]:
        cause, effect = causal_edge_endpoints(edge)
        if cause not in event_ids or effect not in event_ids:
            raise ValueError(f"Causal edge crosses the type boundary: {edge}")
    _assert_dag(object_graph_from_edges(reference["object_edges"]), "Reference object graph")
    _assert_dag(causal_graph_from_edges(reference["causal_edges"]), "Reference causal graph")
    for link in reference["grounding_links"]:
        object_id, event_id = grounding_endpoints(link)
        if object_id not in object_ids or event_id not in event_ids:
            raise ValueError(f"Grounding link has an incorrectly typed endpoint: {link}")
        if link.get("role") not in {"actor", "target", "channel", "resource"}:
            raise ValueError(f"Grounding link has an unknown semantic role: {link}")
    valid_causal = {causal_edge_endpoints(edge) for edge in reference["causal_edges"]}
    for family in reference.get("reference_scenario_families", []):
        event_path = family["event_path"]
        if not event_path or event_path[-1] != target:
            raise ValueError(f"Family {family['id']} does not end at the target event")
        if any((a, b) not in valid_causal for a, b in zip(event_path, event_path[1:])):
            raise ValueError(f"Family {family['id']} contains an unknown causal edge")
        if any(anchor not in object_ids for anchor in family.get("object_anchors", [])):
            raise ValueError(f"Family {family['id']} contains an unknown object anchor")


def validate_tree(tree: dict, schema: dict, path: Path) -> None:
    try:
        jsonschema.validate(tree, schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"Invalid role tree {path}: {exc.message}") from exc
    object_ids = [row["id"] for row in tree["object_nodes"]]
    event_ids = [row["id"] for row in tree["event_nodes"]]
    if len(object_ids) != len(set(object_ids)) or len(event_ids) != len(set(event_ids)):
        raise ValueError(f"Duplicate node identifiers in {path}")
    if set(object_ids) & set(event_ids):
        raise ValueError(f"Object and event identifiers overlap in {path}")
    for edge in tree["object_edges"]:
        parent, child = object_edge_endpoints(edge)
        if parent not in object_ids or child not in object_ids:
            raise ValueError(f"Object edge has a non-object endpoint in {path}: {edge}")
    for edge in tree["causal_edges"]:
        cause, effect = causal_edge_endpoints(edge)
        if cause not in event_ids or effect not in event_ids:
            raise ValueError(f"Causal edge has a non-event endpoint in {path}: {edge}")
    _assert_dag(object_graph_from_edges(tree["object_edges"]), f"Object hierarchy {path}")
    _assert_dag(causal_graph_from_edges(tree["causal_edges"]), f"Causal-event graph {path}")


def validate_direct_baseline(tree: dict, schema: dict, path: Path) -> tuple[dict, dict]:
    """Validate a frozen joint output and exclude cross-type edges from scoring."""
    try:
        jsonschema.validate(tree, schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"Invalid direct baseline {path}: {exc.message}") from exc

    object_ids = [row["id"] for row in tree["object_nodes"]]
    event_ids = [row["id"] for row in tree["event_nodes"]]
    if len(object_ids) != len(set(object_ids)) or len(event_ids) != len(set(event_ids)):
        raise ValueError(f"Duplicate node identifiers in direct baseline {path}")
    object_id_set, event_id_set = set(object_ids), set(event_ids)
    if object_id_set & event_id_set:
        raise ValueError(f"Object and event identifiers overlap in direct baseline {path}")
    if "A" not in object_id_set or "B" not in event_id_set:
        raise ValueError(f"Direct baseline {path} must contain object root A and target event B")

    def endpoint_type(node_id: str) -> str:
        if node_id in object_id_set:
            return "object"
        if node_id in event_id_set:
            return "event"
        return "unknown"

    valid_object_edges, dropped_object_edges = [], []
    for edge in tree["object_edges"]:
        parent, child = object_edge_endpoints(edge)
        if parent in object_id_set and child in object_id_set:
            valid_object_edges.append(edge)
        else:
            dropped_object_edges.append({
                "edge": dict(edge),
                "parent": parent, "child": child,
                "parent_type": endpoint_type(parent), "child_type": endpoint_type(child),
                "reason": "structural edge requires two object endpoints",
            })

    valid_causal_edges, dropped_causal_edges = [], []
    for edge in tree["causal_edges"]:
        cause, effect = causal_edge_endpoints(edge)
        if cause in event_id_set and effect in event_id_set:
            valid_causal_edges.append(edge)
        else:
            dropped_causal_edges.append({
                "edge": dict(edge),
                "cause": cause, "effect": effect,
                "cause_type": endpoint_type(cause), "effect_type": endpoint_type(effect),
                "reason": "causal edge requires two event endpoints",
            })

    _assert_dag(object_graph_from_edges(valid_object_edges), f"Direct baseline object graph {path}")
    _assert_dag(causal_graph_from_edges(valid_causal_edges), f"Direct baseline causal graph {path}")
    scored_tree = {
        **tree,
        "object_edges": valid_object_edges,
        "causal_edges": valid_causal_edges,
    }
    diagnostics = {
        "raw_object_edges": len(tree["object_edges"]),
        "scored_object_edges": len(valid_object_edges),
        "dropped_structural_edges": dropped_object_edges,
        "raw_causal_edges": len(tree["causal_edges"]),
        "scored_causal_edges": len(valid_causal_edges),
        "dropped_causal_edges": dropped_causal_edges,
    }
    return scored_tree, diagnostics


@dataclass(frozen=True)
class CanonicalMatch:
    node_id: str | None
    score: float


def _node_text(node: dict) -> str:
    fields = []
    for key, value in node.items():
        if key not in {"id", "initial_attributes", "required_roles"} and isinstance(value, (str, int, float)):
            fields.append(str(value))
    for key in ("initial_attributes", "required_roles"):
        if node.get(key):
            fields.append(json.dumps(node[key], ensure_ascii=False, sort_keys=True))
    return " ".join(fields)


class ReferenceMatcher:
    """Match generated nodes only against reference nodes of the same type."""

    def __init__(self, reference: dict, overrides: dict | None = None):
        validate_reference_model(reference)
        self.reference = reference
        self.object_nodes = {row["id"]: row for row in reference["object_nodes"]}
        self.event_nodes = {row["id"]: row for row in reference["event_nodes"]}
        self.valid_object_edges = {object_edge_endpoints(row) for row in reference["object_edges"]}
        self.valid_causal_edges = {causal_edge_endpoints(row) for row in reference["causal_edges"]}
        raw = overrides or {}
        self.mapping_artifact = raw
        self.overrides = raw.get("files") or {
            key: value for key, value in raw.items() if key.endswith(".json") and isinstance(value, dict)
        }
        self.global_overrides = {
            "object": raw.get("object_node_mappings", {}),
            "event": raw.get("event_node_mappings", {}),
        }

    def _file_overrides(self, source_file: str, kind: str) -> dict:
        row = self.overrides.get(source_file) or self.overrides.get(Path(source_file).name, {})
        if not isinstance(row, dict):
            row = {}
        nested = row.get(f"{kind}_nodes") or row.get(kind)
        result = nested if isinstance(nested, dict) else row
        global_rows = self.global_overrides.get(kind, {})
        if source_file in global_rows and isinstance(global_rows[source_file], dict):
            return {**result, **global_rows[source_file]}
        if Path(source_file).name in global_rows and isinstance(global_rows[Path(source_file).name], dict):
            return {**result, **global_rows[Path(source_file).name]}
        # Flat maps are accepted for hand-checked fixtures with globally unique ids.
        flat = {key: value for key, value in global_rows.items() if not isinstance(value, dict)}
        return {**flat, **result}

    @staticmethod
    def _match_node(node: dict, candidates: dict[str, dict]) -> CanonicalMatch:
        # A and B are contract endpoints. All other agent identifiers are local
        # to a role file and may coincidentally resemble canonical ids.
        if node.get("id") in {"A", "B"} and node.get("id") in candidates:
            return CanonicalMatch(str(node["id"]), 1.0)
        best_id, best_score = None, 0.0
        for node_id, candidate in candidates.items():
            targets = [candidate.get("label", ""), *candidate.get("aliases", [])]
            label_score = max(similarity(node.get("label", ""), value) for value in targets)
            semantic_score = similarity(_node_text(node), _node_text(candidate))
            domain_score = 1.0 if node_domain(node) == node_domain(candidate) else 0.0
            score = 0.72 * label_score + 0.23 * semantic_score + 0.05 * domain_score
            if score > best_score:
                best_id, best_score = node_id, score
        return CanonicalMatch(best_id if best_score >= 0.53 else None, best_score)

    def _match_nodes(self, nodes: list[dict], candidates: dict[str, dict], overrides: dict) -> tuple[dict, list[dict]]:
        mapping, rows = {}, []
        for node in nodes:
            if node["id"] in overrides:
                canonical = overrides[node["id"]]
                if canonical is not None and canonical not in candidates:
                    raise ValueError(f"Override {canonical} has the wrong type for {node['id']}")
                match = CanonicalMatch(canonical, 1.0 if canonical else 0.0)
            else:
                match = self._match_node(node, candidates)
            mapping[node["id"]] = match.node_id
            rows.append({
                "agent_node": node["id"], "agent_label": node["label"],
                "canonical_node": match.node_id, "score": round(match.score, 4),
            })
        return mapping, rows

    @staticmethod
    def _normalize_edges(
        raw_edges: list[dict], mapping: dict[str, str | None], valid_edges: set[tuple[str, str]],
        endpoint_reader, reference_nodes: dict[str, dict], edge_type: str,
        source_file: str, tree: dict,
    ) -> list[dict]:
        rows = []
        for edge in raw_edges:
            raw_source, raw_target = endpoint_reader(edge)
            source, target = mapping.get(raw_source), mapping.get(raw_target)
            accepted = source is not None and target is not None and (source, target) in valid_edges
            rows.append({
                "source": source, "target": target,
                "agent_source": raw_source, "agent_target": raw_target,
                "dimension": node_domain(reference_nodes[target]) if target in reference_nodes else tree["primary_dimension"],
                "edge_type": edge_type, "match_kind": "direct_edge",
                "agent_subpath": [raw_source, raw_target], "accepted": accepted,
                "source_file": source_file, "role_id": tree["role_id"],
                "primary_dimension": tree["primary_dimension"],
            })
        raw_graph = nx.DiGraph(endpoint_reader(edge) for edge in raw_edges)
        accepted_pairs = {(row["source"], row["target"]) for row in rows if row["accepted"]}
        mapped = [node_id for node_id, canonical in mapping.items() if canonical]
        for raw_source in mapped:
            for raw_target in mapped:
                source, target = mapping[raw_source], mapping[raw_target]
                if (raw_source == raw_target or source == target or (source, target) not in valid_edges
                        or (source, target) in accepted_pairs or raw_source not in raw_graph
                        or raw_target not in raw_graph):
                    continue
                try:
                    subpath = nx.shortest_path(raw_graph, raw_source, raw_target)
                except nx.NetworkXNoPath:
                    continue
                if len(subpath) <= 2 or any(mapping.get(node_id) for node_id in subpath[1:-1]):
                    continue
                rows.append({
                    "source": source, "target": target,
                    "agent_source": raw_source, "agent_target": raw_target,
                    "dimension": node_domain(reference_nodes[target]), "edge_type": edge_type,
                    "match_kind": "abstracted_local_subpath", "agent_subpath": subpath,
                    "accepted": True, "source_file": source_file, "role_id": tree["role_id"],
                    "primary_dimension": tree["primary_dimension"],
                })
                accepted_pairs.add((source, target))
        return rows

    def normalize_tree(self, tree: dict, source_file: str) -> dict:
        object_map, object_matches = self._match_nodes(
            tree["object_nodes"], self.object_nodes, self._file_overrides(source_file, "object")
        )
        event_map, event_matches = self._match_nodes(
            tree["event_nodes"], self.event_nodes, self._file_overrides(source_file, "event")
        )
        return {
            "tree": tree,
            "object_matches": object_matches,
            "event_matches": event_matches,
            "object_edges": self._normalize_edges(
                tree["object_edges"], object_map, self.valid_object_edges, object_edge_endpoints,
                self.object_nodes, "object", source_file, tree,
            ),
            "causal_edges": self._normalize_edges(
                tree["causal_edges"], event_map, self.valid_causal_edges, causal_edge_endpoints,
                self.event_nodes, "causal", source_file, tree,
            ),
        }


def validate_explicit_mapping(
    tree: dict, matcher: ReferenceMatcher, reference: dict, source_file: str,
) -> dict:
    """Require a frozen baseline adjudication for every generated node."""
    reference_objects = {row["id"] for row in reference["object_nodes"]}
    reference_events = {row["id"] for row in reference["event_nodes"]}
    generated_objects = {row["id"] for row in tree["object_nodes"]}
    generated_events = {row["id"] for row in tree["event_nodes"]}

    def source_mapping(kind: str) -> dict:
        global_rows = matcher.mapping_artifact.get(f"{kind}_node_mappings", {})
        if isinstance(global_rows, dict) and isinstance(global_rows.get(source_file), dict):
            return global_rows[source_file]
        files = matcher.mapping_artifact.get("files", {})
        row = files.get(source_file) if isinstance(files, dict) else None
        if isinstance(row, dict):
            nested = row.get(f"{kind}_nodes") or row.get(kind)
            if isinstance(nested, dict):
                return nested
        raise ValueError(
            f"Direct baseline mapping must contain an exact {kind} namespace for {source_file}"
        )

    object_mapping = source_mapping("object")
    event_mapping = source_mapping("event")

    def check(kind: str, generated: set[str], mapping: dict, canonical: set[str]) -> None:
        missing, extra = generated - set(mapping), set(mapping) - generated
        if missing or extra:
            raise ValueError(
                f"Explicit {kind} mapping for {source_file} is stale: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        wrong = {
            node_id: target for node_id, target in mapping.items()
            if target is not None and target not in canonical
        }
        if wrong:
            raise ValueError(f"Explicit {kind} mapping has wrong-type canonical ids: {wrong}")

    check("object", generated_objects, object_mapping, reference_objects)
    check("event", generated_events, event_mapping, reference_events)
    return {
        "source_file": source_file,
        "object_entries": len(object_mapping),
        "object_matches": sum(value is not None for value in object_mapping.values()),
        "event_entries": len(event_mapping),
        "event_matches": sum(value is not None for value in event_mapping.values()),
    }


def load_trees(
    folder: Path, schema: dict, matcher: ReferenceMatcher, source_namespace: str | None = None
) -> list[dict]:
    rows = []
    if not folder.exists():
        return rows
    for path in sorted(folder.glob("*.json")):
        tree = load_json(path)
        validate_tree(tree, schema, path)
        source_key = f"{source_namespace}/{path.name}" if source_namespace else path.name
        row = matcher.normalize_tree(tree, source_key)
        row["path"] = str(path)
        rows.append(row)
    return rows


def validate_asymmetric_mapping_artifact(
    mapping_artifact: dict,
    reference: dict,
    reference_path: Path,
    schema: dict,
    source_paths: dict[str, Path],
) -> dict:
    """Bind every asymmetric adjudication to the current reference and raw source bytes."""
    metadata = mapping_artifact.get("metadata", {})
    expected_reference_hash = sha256(reference_path)
    if metadata.get("reference_sha256") != expected_reference_hash:
        raise ValueError("Asymmetric mapping artifact is not bound to the current reference model")
    declared_hashes = metadata.get("source_sha256_by_file")
    if not isinstance(declared_hashes, dict) or set(declared_hashes) != set(source_paths):
        raise ValueError("Asymmetric mapping source-hash namespace is incomplete or stale")

    reference_objects = {row["id"] for row in reference["object_nodes"]}
    reference_events = {row["id"] for row in reference["event_nodes"]}
    object_namespaces = mapping_artifact.get("object_node_mappings", {})
    event_namespaces = mapping_artifact.get("event_node_mappings", {})

    def namespace(rows: dict, source_key: str) -> dict:
        value = rows.get(source_key)
        if value is None and source_key.startswith("initial/"):
            value = rows.get(Path(source_key).name)
        if not isinstance(value, dict):
            raise ValueError(f"Asymmetric mapping lacks an exact namespace for {source_key}")
        return value

    mapped_objects = mapped_events = 0
    for source_key, path in sorted(source_paths.items()):
        actual_hash = sha256(path)
        if str(declared_hashes[source_key]).upper() != actual_hash:
            raise ValueError(f"Asymmetric mapping source hash is stale for {source_key}")
        tree = load_json(path)
        validate_tree(tree, schema, path)
        generated_objects = {row["id"] for row in tree["object_nodes"]}
        generated_events = {row["id"] for row in tree["event_nodes"]}
        object_mapping = namespace(object_namespaces, source_key)
        event_mapping = namespace(event_namespaces, source_key)
        if set(object_mapping) != generated_objects or set(event_mapping) != generated_events:
            raise ValueError(f"Asymmetric mapping node coverage is stale for {source_key}")
        wrong_objects = {value for value in object_mapping.values() if value and value not in reference_objects}
        wrong_events = {value for value in event_mapping.values() if value and value not in reference_events}
        if wrong_objects or wrong_events:
            raise ValueError(
                f"Asymmetric mapping contains wrong-type ids for {source_key}: "
                f"objects={sorted(wrong_objects)}, events={sorted(wrong_events)}"
            )
        mapped_objects += sum(value is not None for value in object_mapping.values())
        mapped_events += sum(value is not None for value in event_mapping.values())
    return {
        "reference_sha256": expected_reference_hash,
        "source_files": len(source_paths),
        "mapped_object_nodes": mapped_objects,
        "mapped_event_nodes": mapped_events,
    }


def load_direct_baseline(
    path: Path,
    schema: dict,
    matcher: ReferenceMatcher,
    source_file: str,
    *,
    mapping_source_file: str,
    reference_sha256: str | None = None,
) -> tuple[dict, dict]:
    declared_source = matcher.mapping_artifact.get("source_file")
    normalized_declared = str(declared_source or "").replace("\\", "/")
    normalized_expected = str(mapping_source_file).replace("\\", "/")
    if not declared_source:
        raise ValueError("Direct baseline mapping must record source_file")
    if normalized_declared != normalized_expected:
        raise ValueError(
            f"Direct baseline mapping declares source_file {declared_source}, "
            f"expected {mapping_source_file}"
        )
    if reference_sha256 is not None:
        declared_reference = matcher.mapping_artifact.get("reference_sha256")
        if str(declared_reference or "").upper() != reference_sha256.upper():
            raise ValueError("Direct baseline mapping is not bound to the current reference model")
    expected_hash = matcher.mapping_artifact.get("source_sha256")
    actual_hash = sha256(path)
    if not expected_hash:
        raise ValueError("Direct baseline mapping must record source_sha256")
    if str(expected_hash).upper() != actual_hash:
        raise ValueError(
            f"Direct baseline mapping targets {expected_hash}, but {path} is {actual_hash}"
        )
    raw_tree = load_json(path)
    scored_tree, validation = validate_direct_baseline(raw_tree, schema, path)
    mapping = validate_explicit_mapping(scored_tree, matcher, matcher.reference, source_file)
    mapping["source_sha256"] = actual_hash
    mapping["mapping_source_file"] = normalized_expected
    row = matcher.normalize_tree(scored_tree, source_file)
    row["path"] = str(path)
    return row, {"validation": validation, "mapping": mapping}


def _accepted_edges(trees: Iterable[dict], kind: str) -> list[dict]:
    return [edge for tree in trees for edge in tree[f"{kind}_edges"] if edge["accepted"]]


def _recovered_nodes(trees: Iterable[dict], kind: str) -> set[str]:
    return {
        row["canonical_node"]
        for tree in trees for row in tree[f"{kind}_matches"]
        if row["canonical_node"] is not None
    }


def tree_structure(tree: dict) -> dict:
    object_graph = object_graph_from_edges(tree["tree"]["object_edges"])
    causal_graph = causal_graph_from_edges(tree["tree"]["causal_edges"])
    _assert_dag(object_graph, f"Object hierarchy {tree.get('path', '')}")
    _assert_dag(causal_graph, f"Causal-event graph {tree.get('path', '')}")
    causal_decomposition = causal_graph.reverse(copy=True)
    return {
        "object_depth": nx.dag_longest_path_length(object_graph) if object_graph else 0,
        "object_branch_nodes": sum(object_graph.out_degree(node) > 1 for node in object_graph),
        "causal_depth": nx.dag_longest_path_length(causal_graph) if causal_graph else 0,
        "causal_branch_nodes": sum(causal_decomposition.out_degree(node) > 1 for node in causal_decomposition),
    }


def validate_targeted_structure(trees: Iterable[dict], minimum_depth: int = 3) -> None:
    for tree in trees:
        metrics = tree_structure(tree)
        for kind in ("object", "causal"):
            if metrics[f"{kind}_depth"] < minimum_depth:
                raise ValueError(
                    f"{tree['path']} has {kind} depth {metrics[f'{kind}_depth']}; expected {minimum_depth}"
                )
            if metrics[f"{kind}_branch_nodes"] < 1:
                raise ValueError(f"{tree['path']} has no {kind} branching node")


def recovered_grounding_links(
    reference: dict,
    recovered_objects: set[str],
    recovered_events: set[str],
    object_graph: nx.DiGraph | None = None,
) -> list[dict]:
    root = reference["endpoints"]["object_root"]
    rows = []
    for index, link in enumerate(reference["grounding_links"], start=1):
        object_id, event_id = grounding_endpoints(link)
        reachable = (
            object_graph is None
            or (root in object_graph and object_id in object_graph and nx.has_path(object_graph, root, object_id))
        )
        if (link.get("status", "direct") == "direct" and object_id in recovered_objects
                and event_id in recovered_events and reachable):
            row = dict(link)
            row.setdefault("link_id", f"G{index:02d}")
            rows.append(row)
    return rows


def _all_paths(graph: nx.DiGraph, source: str, target: str, cutoff: int = 30) -> Iterator[list[str]]:
    if source not in graph or target not in graph:
        return iter(())
    return nx.all_simple_paths(graph, source, target, cutoff=cutoff)


def complete_scenarios(
    object_graph: nx.DiGraph,
    causal_graph: nx.DiGraph,
    grounding_links: list[dict],
    reference: dict,
) -> list[dict]:
    """Construct families only when every mandatory organizational role is bound."""
    root = reference["endpoints"]["object_root"]
    target = reference["endpoints"]["target_event"]
    object_nodes = {row["id"]: row for row in reference["object_nodes"]}
    event_nodes = {row["id"]: row for row in reference["event_nodes"]}
    links_by_event_role: dict[tuple[str, str], list[dict]] = {}
    for link in grounding_links:
        object_id, event_id = grounding_endpoints(link)
        if object_id not in object_nodes or event_id not in event_nodes:
            raise ValueError(f"Grounding link has the wrong endpoint type: {link}")
        links_by_event_role.setdefault((event_id, link["role"]), []).append(link)

    rows, seen = [], set()
    for family in reference.get("reference_scenario_families", []):
        event_path = list(family["event_path"])
        if event_path[-1] != target or any(
            not causal_graph.has_edge(a, b) for a, b in zip(event_path, event_path[1:])
        ):
            continue

        requirement_candidates: list[list[dict]] = []
        complete = True
        for event_id in event_path:
            for requirement in event_nodes[event_id]["required_roles"]:
                if not requirement.get("mandatory", True) or requirement.get("scope") != "initial_state":
                    continue
                candidates = []
                for link in links_by_event_role.get((event_id, requirement["role"]), []):
                    object_id, _ = grounding_endpoints(link)
                    if object_id not in family.get("object_anchors", []):
                        continue
                    if object_nodes[object_id]["object_class"] != requirement["object_class"]:
                        continue
                    for object_path in _all_paths(object_graph, root, object_id):
                        candidates.append({
                            "object_id": object_id,
                            "event_id": event_id,
                            "role": requirement["role"],
                            "status": link.get("status", "direct"),
                            "object_path": list(object_path),
                            "structural_depth": len(object_path) - 1,
                        })
                if not candidates:
                    complete = False
                    break
                requirement_candidates.append(candidates)
            if not complete:
                break
        if not complete or not requirement_candidates:
            continue

        for bindings_tuple in product(*requirement_candidates):
            bindings = list(bindings_tuple)
            entry = next((row for row in bindings if row["event_id"] == event_path[0]), None)
            if entry is None:
                continue
            key = (
                family["id"], tuple(event_path),
                tuple((row["object_id"], row["event_id"], row["role"], tuple(row["object_path"])) for row in bindings),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "family": family["id"],
                "dimension": node_domain(event_nodes[entry["event_id"]]),
                "object_anchor": entry["object_id"],
                "event_anchor": entry["event_id"],
                "grounding_role": entry["role"],
                "grounding_link": {
                    "object_id": entry["object_id"], "event_id": entry["event_id"],
                    "role": entry["role"], "status": entry["status"],
                },
                "object_path": entry["object_path"],
                "structural_depth": entry["structural_depth"],
                "grounding_bindings": [
                    {key: row[key] for key in ("object_id", "event_id", "role", "status")}
                    for row in bindings
                ],
                "object_context_paths": [
                    {key: row[key] for key in ("object_id", "event_id", "role", "object_path", "structural_depth")}
                    for row in bindings
                ],
                "event_path": event_path,
                "causal_steps": len(event_path) - 1,
            })
    rows.sort(key=lambda row: (row["causal_steps"], row["family"], row["object_path"]))
    for index, row in enumerate(rows, start=1):
        row["scenario_id"] = f"S{index:03d}"
    return rows


def ahp_weights(matrix: list[list[float]]) -> dict:
    values = np.asarray(matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(values)
    index = int(np.argmax(eigenvalues.real))
    vector = np.abs(eigenvectors[:, index].real)
    vector /= vector.sum()
    lambda_max = float(eigenvalues[index].real)
    n = values.shape[0]
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    cr = ci / RI[n] if RI.get(n, 0.0) else 0.0
    return {"weights": vector.tolist(), "lambda_max": lambda_max, "ci": ci, "cr": cr}


def _directness_rating(causal_steps: int) -> float:
    return 9.0 if causal_steps == 5 else 7.0 if causal_steps <= 7 else 5.0 if causal_steps == 8 else 3.0


def scenario_ratings(scenario: dict, reference: dict) -> dict[str, float]:
    ratings = dict(reference["ahp"]["family_ratings"][scenario["family"]])
    ratings["causal_directness"] = _directness_rating(scenario["causal_steps"])
    return ratings


def rank_scenarios(scenarios: list[dict], reference: dict) -> dict:
    ahp = ahp_weights(reference["ahp"]["pairwise_matrix"])
    criteria = [row["id"] for row in reference["ahp"]["criteria"]]
    rows = []
    for scenario in scenarios:
        ratings = scenario_ratings(scenario, reference)
        score = sum(
            weight * ratings[criterion] / 9.0
            for criterion, weight in zip(criteria, ahp["weights"])
        )
        rows.append({**scenario, "ratings": ratings, "ahp_score": score})
    rows.sort(key=lambda row: (-row["ahp_score"], row["causal_steps"], row["scenario_id"]))
    distinct_scores = []
    for row in rows:
        if not distinct_scores or not math.isclose(row["ahp_score"], distinct_scores[-1], abs_tol=1e-12):
            distinct_scores.append(row["ahp_score"])
        row["rank"] = len(distinct_scores)
        row["ahp_score"] = round(row["ahp_score"], 6)
    families = []
    for family in sorted({row["family"] for row in rows}):
        members = [row for row in rows if row["family"] == family]
        top = members[0]
        families.append({
            "family": family,
            "complete_scenario_variants": len(members),
            "minimum_causal_steps": min(row["causal_steps"] for row in members),
            "ahp_score": top["ahp_score"], "rank": top["rank"], "ratings": top["ratings"],
        })
    families.sort(key=lambda row: (row["rank"], row["family"]))
    return {"criteria": criteria, "ahp": ahp, "scenarios": rows, "families": families}


def weight_sensitivity(ranking: dict, repetitions: int = 2000, seed: int = 20260818) -> dict:
    if not ranking["scenarios"]:
        return {"repetitions": 0, "top_family_counts": {}, "top_family_shares": {}}
    rng = np.random.default_rng(seed)
    base = np.asarray(ranking["ahp"]["weights"], dtype=float)
    criteria, scenarios = ranking["criteria"], ranking["scenarios"]
    counts: Counter[str] = Counter()
    for _ in range(repetitions):
        weights = base * rng.uniform(0.8, 1.2, len(base))
        weights /= weights.sum()
        scores = [
            sum(weights[i] * row["ratings"][criterion] / 9.0 for i, criterion in enumerate(criteria))
            for row in scenarios
        ]
        counts[scenarios[int(np.argmax(scores))]["family"]] += 1
    return {
        "repetitions": repetitions, "seed": seed,
        "perturbation": "independent uniform multipliers in [0.8, 1.2], renormalized",
        "top_family_counts": dict(counts),
        "top_family_shares": {family: count / repetitions for family, count in counts.items()},
    }


def minimum_event_comparison(ranking: dict) -> dict:
    if not ranking["scenarios"]:
        return {"minimum_causal_steps": None, "minimum_event_families": [],
                "ahp_top_family": None, "ahp_top_causal_steps": None, "same_family": None}
    minimum = min(row["causal_steps"] for row in ranking["scenarios"])
    shortest = sorted({row["family"] for row in ranking["scenarios"] if row["causal_steps"] == minimum})
    top = ranking["scenarios"][0]
    return {
        "minimum_causal_steps": minimum, "minimum_event_families": shortest,
        "ahp_top_family": top["family"], "ahp_top_causal_steps": top["causal_steps"],
        "same_family": top["family"] in shortest,
    }


def _edge_dimension(edge: dict, nodes: dict[str, dict], endpoint_reader) -> str:
    _, target = endpoint_reader(edge)
    return str(edge.get("dimension") or edge.get("domain") or node_domain(nodes[target]))


def dimension_coverage(
    reference: dict,
    recovered_objects: set[str],
    recovered_events: set[str],
    recovered_object_edges: set[tuple[str, str]],
    recovered_causal_edges: set[tuple[str, str]],
    groundings: list[dict],
    weights: dict[str, float],
) -> list[dict]:
    object_nodes = {row["id"]: row for row in reference["object_nodes"]}
    event_nodes = {row["id"]: row for row in reference["event_nodes"]}
    direct_links = [row for row in reference["grounding_links"] if row.get("status", "direct") == "direct"]
    recovered_link_keys = {(*grounding_endpoints(row), row["role"]) for row in groundings}
    rows = []
    for dimension in DIMENSIONS:
        ref_object_nodes = {node_id for node_id, node in object_nodes.items() if node_domain(node) == dimension}
        ref_object_edges = {
            object_edge_endpoints(edge) for edge in reference["object_edges"]
            if _edge_dimension(edge, object_nodes, object_edge_endpoints) == dimension
        }
        ref_event_nodes = {node_id for node_id, node in event_nodes.items() if node_domain(node) == dimension}
        ref_causal_edges = {
            causal_edge_endpoints(edge) for edge in reference["causal_edges"]
            if _edge_dimension(edge, event_nodes, causal_edge_endpoints) == dimension
        }
        ref_links = {
            (*grounding_endpoints(link), link["role"]) for link in direct_links
            if node_domain(event_nodes[grounding_endpoints(link)[1]]) == dimension
        }
        object_denominator = len(ref_object_nodes) + len(ref_object_edges)
        event_denominator = len(ref_event_nodes) + len(ref_causal_edges)
        object_recovered = len(ref_object_nodes & recovered_objects) + len(ref_object_edges & recovered_object_edges)
        event_recovered = len(ref_event_nodes & recovered_events) + len(ref_causal_edges & recovered_causal_edges)
        grounding_recovered = len(ref_links & recovered_link_keys)
        c_object = object_recovered / object_denominator if object_denominator else 1.0
        c_event = event_recovered / event_denominator if event_denominator else 1.0
        c_grounding = grounding_recovered / len(ref_links) if ref_links else 1.0
        rows.append({
            "dimension": dimension, "weight": weights.get(dimension, 0.0),
            "object_elements_recovered": object_recovered,
            "object_elements_reference": object_denominator,
            "event_elements_recovered": event_recovered,
            "event_elements_reference": event_denominator,
            "grounding_links_recovered": grounding_recovered,
            "grounding_links_reference": len(ref_links),
            "c_object": round(c_object, 6), "c_event": round(c_event, 6),
            "c_grounding": round(c_grounding, 6),
            "gap_priority": round(weights.get(dimension, 0.0) * (1.0 - min(c_object, c_event, c_grounding)), 6),
        })
    return sorted(rows, key=lambda row: (-row["gap_priority"], row["dimension"]))


def summarize_condition(trees: list[dict], reference: dict, weights: dict[str, float]) -> dict:
    recovered_objects = _recovered_nodes(trees, "object")
    recovered_events = _recovered_nodes(trees, "event")
    object_edges = {(row["source"], row["target"]) for row in _accepted_edges(trees, "object")}
    causal_edges = {(row["source"], row["target"]) for row in _accepted_edges(trees, "causal")}
    object_graph = object_graph_from_edges([
        {"source": a, "target": b, "edge_type": "object"} for a, b in object_edges
    ])
    causal_graph = causal_graph_from_edges([
        {"source": a, "target": b, "edge_type": "causal"} for a, b in causal_edges
    ])
    object_graph.add_nodes_from(recovered_objects)
    causal_graph.add_nodes_from(recovered_events)
    grounding_pairs = recovered_grounding_links(reference, recovered_objects, recovered_events)
    groundings = recovered_grounding_links(
        reference, recovered_objects, recovered_events, object_graph=object_graph
    )
    scenarios = complete_scenarios(object_graph, causal_graph, groundings, reference)
    ranking = rank_scenarios(scenarios, reference)
    reference_object_edges = {object_edge_endpoints(row) for row in reference["object_edges"]}
    reference_causal_edges = {causal_edge_endpoints(row) for row in reference["causal_edges"]}
    direct_groundings = [row for row in reference["grounding_links"] if row.get("status", "direct") == "direct"]
    return {
        "tree_count": len(trees),
        "tree_structure": [{
            "source_file": Path(tree["path"]).name,
            "role_id": tree["tree"]["role_id"],
            "primary_dimension": tree["tree"]["primary_dimension"],
            **tree_structure(tree),
        } for tree in trees],
        "object_nodes_recovered": len(recovered_objects),
        "object_nodes_reference": len(reference["object_nodes"]),
        "object_node_recall": len(recovered_objects) / len(reference["object_nodes"]),
        "object_edges_recovered": len(object_edges),
        "object_edges_reference": len(reference_object_edges),
        "object_edge_recall": len(object_edges) / len(reference_object_edges),
        "causal_nodes_recovered": len(recovered_events),
        "causal_nodes_reference": len(reference["event_nodes"]),
        "causal_node_recall": len(recovered_events) / len(reference["event_nodes"]),
        "causal_edges_recovered": len(causal_edges),
        "causal_edges_reference": len(reference_causal_edges),
        "causal_edge_recall": len(causal_edges) / len(reference_causal_edges),
        "grounding_pairs_available": len(grounding_pairs),
        "grounding_pair_recall": len(grounding_pairs) / len(direct_groundings),
        "grounding_links_recovered": len(groundings),
        "grounding_links_reference": len(direct_groundings),
        "grounding_link_recall": len(groundings) / len(direct_groundings),
        "recovered_grounding_links": groundings,
        "coverage": dimension_coverage(
            reference, recovered_objects, recovered_events, object_edges, causal_edges, groundings, weights
        ),
        "complete_scenario_count": len(scenarios),
        "scenario_family_count": len({row["family"] for row in scenarios}),
        "complete_scenarios": scenarios,
        "ranking": ranking,
    }


def _scenario_counts(
    object_edges: set[tuple[str, str]], causal_edges: set[tuple[str, str]],
    groundings: list[dict], reference: dict,
) -> tuple[int, int]:
    object_graph = object_graph_from_edges([
        {"source": a, "target": b, "edge_type": "object"} for a, b in object_edges
    ])
    causal_graph = causal_graph_from_edges([
        {"source": a, "target": b, "edge_type": "causal"} for a, b in causal_edges
    ])
    rows = complete_scenarios(object_graph, causal_graph, groundings, reference)
    return len(rows), len({row["family"] for row in rows})


def connection_ablation(trees: list[dict], reference: dict, baseline: dict) -> dict:
    object_edges = {(row["source"], row["target"]) for row in _accepted_edges(trees, "object")}
    causal_edges = {(row["source"], row["target"]) for row in _accepted_edges(trees, "causal")}
    recovered_groundings = baseline["recovered_grounding_links"]
    causal_rows = []
    used_causal_edges = {
        (cause, effect)
        for scenario in baseline["complete_scenarios"]
        for cause, effect in zip(scenario["event_path"], scenario["event_path"][1:])
    }
    for cause, effect in sorted(causal_edges & used_causal_edges):
        count, families = _scenario_counts(
            object_edges, causal_edges - {(cause, effect)}, recovered_groundings, reference
        )
        causal_rows.append({
            "removed_cause": cause, "removed_effect": effect,
            "remaining_scenarios": count,
            "lost_scenarios": baseline["complete_scenario_count"] - count,
            "remaining_families": families,
        })
    used_groundings = {
        (row["object_anchor"], row["event_anchor"], row["grounding_role"])
        for row in baseline["complete_scenarios"]
    }
    scenario_entry_links = [
        link for link in recovered_groundings
        if (*grounding_endpoints(link), link["role"]) in used_groundings
    ]
    grounding_rows = []
    for index, link in enumerate(scenario_entry_links):
        remaining_entry_links = scenario_entry_links[:index] + scenario_entry_links[index + 1 :]
        non_entry_links = [link for link in recovered_groundings if link not in scenario_entry_links]
        count, families = _scenario_counts(
            object_edges, causal_edges, non_entry_links + remaining_entry_links, reference
        )
        object_id, event_id = grounding_endpoints(link)
        family = next(
            row["family"] for row in baseline["complete_scenarios"]
            if row["object_anchor"] == object_id
            and row["event_anchor"] == event_id
            and row["grounding_role"] == link["role"]
        )
        grounding_rows.append({
            "removed_family": family, "removed_object": object_id,
            "removed_event": event_id, "removed_role": link["role"],
            "remaining_scenarios": count,
            "lost_scenarios": baseline["complete_scenario_count"] - count,
            "remaining_families": families,
        })
    return {"causal_edge_leave_one_out": causal_rows, "grounding_leave_one_out": grounding_rows}


def make_figures(output: Path, phases: list[dict], ablation: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10})
    coverage = phases[-1]["condition"]["coverage"]
    labels, y = [row["dimension"].replace("_", " ") for row in coverage], np.arange(len(coverage))
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    height = 0.23
    ax.barh(y - height, [row["c_object"] for row in coverage], height, color="white", edgecolor="black", label="Object")
    ax.barh(y, [row["c_event"] for row in coverage], height, color="0.68", edgecolor="black", label="Event")
    ax.barh(y + height, [row["c_grounding"] for row in coverage], height, color="0.32", edgecolor="black", label="Grounding")
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 1.02)
    ax.set_xlabel("Recovered reference share")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="x", color="0.88", linewidth=0.6); fig.tight_layout()
    fig.savefig(output / "coverage_gap.png", dpi=400, bbox_inches="tight"); plt.close(fig)

    phase_labels = [row["phase"].replace("targeted_round", "round ") for row in phases]
    x, width = np.arange(len(phases)), 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - width, [row["condition"]["grounding_pairs_available"] for row in phases],
           width, color="white", edgecolor="black", label="Matched pairs")
    ax.bar(x, [row["condition"]["grounding_links_recovered"] for row in phases],
           width, color="0.68", edgecolor="black", label="Reachable bindings")
    ax.bar(x + width, [row["condition"]["scenario_family_count"] for row in phases],
           width, color="0.32", edgecolor="black", label="Complete families")
    ax.set_xticks(x, phase_labels); ax.set_ylabel("Count")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="y", color="0.88", linewidth=0.6); fig.tight_layout()
    fig.savefig(output / "phase_progression.png", dpi=400, bbox_inches="tight"); plt.close(fig)

    rows = ablation["grounding_leave_one_out"]
    if rows:
        labels = [row["removed_family"].replace("_", " ") for row in rows]
        y = np.arange(len(rows)); fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.32 * len(rows))))
        ax.barh(y, [row["lost_scenarios"] for row in rows], color="0.65", edgecolor="black")
        ax.set_yticks(y, labels); ax.invert_yaxis()
        ax.set_xlabel("Complete scenarios lost after removing one grounding link")
        ax.grid(axis="x", color="0.88", linewidth=0.6); fig.tight_layout()
        fig.savefig(output / "connection_ablation.png", dpi=400, bbox_inches="tight"); plt.close(fig)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def build_generation_comparison(
    direct: dict,
    initial: dict,
    refined: dict,
    direct_spec: DirectBaselineSpec = ULTRA_DIRECT_BASELINE,
) -> list[dict]:
    conditions = (
        (
            direct_spec.condition_id,
            direct_spec.label,
            direct,
            direct_spec.task_sessions,
            direct_spec.stored_json_outputs,
        ),
        ("initial_asymmetric_generation", "Initial asymmetric generation", initial, 3, 6),
        (
            "asymmetric_after_targeted_refinement",
            "Asymmetric generation after targeted refinement",
            refined,
            11,
            18,
        ),
    )
    return [{
        "condition_id": condition_id,
        "condition": label,
        "task_sessions": invocations,
        "stored_json_outputs": outputs,
        "object_nodes_recovered": metrics["object_nodes_recovered"],
        "ordered_structural_endpoint_pairs_recovered": metrics["object_edges_recovered"],
        "event_nodes_recovered": metrics["causal_nodes_recovered"],
        "ordered_causal_endpoint_pairs_recovered": metrics["causal_edges_recovered"],
        "usable_object_event_connections": metrics["grounding_links_recovered"],
        "complete_reference_paths": metrics["complete_scenario_count"],
        "complete_reference_families": metrics["scenario_family_count"],
    } for condition_id, label, metrics, invocations, outputs in conditions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--runs-dir", default="runs_v2")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    root = args.root.resolve()
    reference_path = root / "scenario" / "reference_model.json"
    brief_path = root / "scenario" / "scenario_brief.json"
    schema_path = root / "schema" / "role_tree.schema.json"
    overrides_path = root / "scenario" / "node_mapping_overrides.json"
    generation_record_path = root / "generation_record.json"
    prompt_paths = [root / "prompts" / "initial_tree_prompt.md", root / "prompts" / "targeted_gap_prompt.md"]
    runs_root = root / args.runs_dir
    direct_spec = ULTRA_DIRECT_BASELINE
    direct_files = resolve_direct_baseline_files(root, runs_root, direct_spec)

    reference, brief, schema = load_json(reference_path), load_json(brief_path), load_json(schema_path)
    overrides = load_json(overrides_path) if overrides_path.exists() else {}
    asymmetric_source_paths = {
        f"{path.parent.name}/{path.name}": path
        for folder in [runs_root / "initial", *sorted(runs_root.glob("targeted*"))]
        if folder.is_dir()
        for path in sorted(folder.glob("*.json"))
    }
    asymmetric_mapping_validation = validate_asymmetric_mapping_artifact(
        overrides, reference, reference_path, schema, asymmetric_source_paths,
    )
    matcher = ReferenceMatcher(reference, overrides)
    if not direct_files.mapping.exists():
        raise SystemExit(f"Direct baseline mapping not found: {direct_files.mapping}")
    direct_matcher = ReferenceMatcher(reference, load_json(direct_files.mapping))
    direct_tree, direct_diagnostics = load_direct_baseline(
        direct_files.generation,
        schema,
        direct_matcher,
        direct_files.source_key,
        mapping_source_file=direct_files.mapping_source_file,
        reference_sha256=sha256(reference_path),
    )
    initial = load_trees(runs_root / "initial", schema, matcher, "initial")
    if not initial:
        raise SystemExit(f"No initial role trees found under {runs_root / 'initial'}")
    validate_targeted_structure(initial)
    targeted_rounds = []
    for folder in sorted(runs_root.glob("targeted*")):
        if folder.is_dir():
            round_trees = load_trees(folder, schema, matcher, folder.name)
            if round_trees:
                targeted_rounds.append({"round": folder.name, "trees": round_trees})
    targeted = [tree for row in targeted_rounds for tree in row["trees"]]
    if targeted:
        validate_targeted_structure(targeted)

    weights = {row["id"]: float(row["weight"]) for row in brief["coverage_dimensions"]}
    cumulative = list(initial)
    phases = [{"phase": "initial", "condition": summarize_condition(cumulative, reference, weights)}]
    for round_row in targeted_rounds:
        cumulative.extend(round_row["trees"])
        phases.append({
            "phase": round_row["round"],
            "condition": summarize_condition(cumulative, reference, weights),
        })
    final = phases[-1]["condition"]
    direct_condition = summarize_condition([direct_tree], reference, weights)
    generation_comparison = build_generation_comparison(
        direct_condition, phases[0]["condition"], final, direct_spec,
    )
    sensitivity = weight_sensitivity(final["ranking"], args.repetitions, args.seed)
    minimum_comparison = minimum_event_comparison(final["ranking"])
    ablation = connection_ablation(cumulative, reference, final)

    def file_record(path: Path) -> dict | None:
        return {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        } if path.exists() else None

    manifest = {
        "case_id": reference["case_id"],
        "method": "asymmetric object-event synthesis",
        "semantic_invariant": "object edges are structural context; only causal event edges contribute to causal_steps",
        "inputs": {
            "scenario_brief": file_record(brief_path), "reference_model": file_record(reference_path),
            "schema": file_record(schema_path), "runner": file_record(Path(__file__).resolve()),
            "node_mapping_overrides": file_record(overrides_path),
            "direct_baseline_case_brief": file_record(direct_files.case_brief),
            "direct_baseline_mapping": file_record(direct_files.mapping),
            "direct_baseline_prompt": file_record(direct_files.prompt),
            "direct_baseline_generation": file_record(direct_files.generation),
            "generation_record": file_record(generation_record_path),
            "prompts": [file_record(path) for path in prompt_paths],
            "initial_files": [file_record(Path(tree["path"])) for tree in initial],
            "targeted_rounds": [{
                "round": row["round"],
                "files": [file_record(Path(tree["path"])) for tree in row["trees"]],
            } for row in targeted_rounds],
        },
        "phases": phases,
        "initial_condition": phases[0]["condition"],
        "refinement_conditions": phases[1:],
        "refined_condition": final,
        "generation_baseline": {
            "condition_id": direct_spec.condition_id,
            "condition_label": direct_spec.label,
            "source_key": direct_files.source_key,
            "mapping_source_file": direct_files.mapping_source_file,
            "task_sessions": direct_spec.task_sessions,
            "stored_json_outputs": direct_spec.stored_json_outputs,
            "diagnostics": direct_diagnostics,
            "condition": direct_condition,
        },
        "asymmetric_mapping_validation": asymmetric_mapping_validation,
        "generation_comparison": generation_comparison,
        "ahp_weight_sensitivity": sensitivity,
        "minimum_event_count_comparison": minimum_comparison,
        "connection_ablation": ablation,
    }
    output = root / "results"
    # Do not leave artifacts from the superseded symmetric experiment beside
    # the asymmetric results; supplement builders may collect files by glob.
    for stale in (
        output / "dimension_ablation.csv",
        output / "figures" / "dimension_ablation.png",
    ):
        if stale.exists():
            stale.unlink()
    write_json(output / "experiment_results.json", manifest)
    make_figures(output / "figures", phases, ablation)
    phase_rows = [{
        "phase": row["phase"], "trees": row["condition"]["tree_count"],
        "object_nodes": row["condition"]["object_nodes_recovered"],
        "object_edges": row["condition"]["object_edges_recovered"],
        "causal_nodes": row["condition"]["causal_nodes_recovered"],
        "causal_edges": row["condition"]["causal_edges_recovered"],
        "grounding_pairs_available": row["condition"]["grounding_pairs_available"],
        "grounding_links": row["condition"]["grounding_links_recovered"],
        "complete_scenarios": row["condition"]["complete_scenario_count"],
        "scenario_families": row["condition"]["scenario_family_count"],
    } for row in phases]
    _write_csv(output / "phase_metrics.csv", phase_rows)
    _write_csv(output / "generation_comparison.csv", generation_comparison)
    _write_csv(output / "coverage_matrix.csv", final["coverage"])
    _write_csv(output / "causal_edge_ablation.csv", ablation["causal_edge_leave_one_out"])
    _write_csv(output / "grounding_ablation.csv", ablation["grounding_leave_one_out"])

    summary = {
        "direct_condition_id": direct_spec.condition_id,
        "direct_object_nodes": direct_condition["object_nodes_recovered"],
        "direct_object_edges": direct_condition["object_edges_recovered"],
        "direct_causal_nodes": direct_condition["causal_nodes_recovered"],
        "direct_causal_edges": direct_condition["causal_edges_recovered"],
        "direct_grounding_links": direct_condition["grounding_links_recovered"],
        "direct_complete_paths": direct_condition["complete_scenario_count"],
        "direct_scenario_families": direct_condition["scenario_family_count"],
        "initial_role_trees": len(initial), "targeted_role_trees": len(targeted),
        "targeted_rounds": len(targeted_rounds),
        "initial_object_nodes": phases[0]["condition"]["object_nodes_recovered"],
        "refined_object_nodes": final["object_nodes_recovered"],
        "initial_causal_nodes": phases[0]["condition"]["causal_nodes_recovered"],
        "refined_causal_nodes": final["causal_nodes_recovered"],
        "initial_grounding_pairs": phases[0]["condition"]["grounding_pairs_available"],
        "refined_grounding_pairs": final["grounding_pairs_available"],
        "initial_grounding_links": phases[0]["condition"]["grounding_links_recovered"],
        "refined_grounding_links": final["grounding_links_recovered"],
        "initial_complete_scenarios": phases[0]["condition"]["complete_scenario_count"],
        "refined_complete_scenarios": final["complete_scenario_count"],
        "refined_scenario_families": final["scenario_family_count"],
        "ahp_cr": final["ranking"]["ahp"]["cr"],
        "top_family": final["ranking"]["families"][0] if final["ranking"]["families"] else None,
        "minimum_event_count_comparison": minimum_comparison,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
