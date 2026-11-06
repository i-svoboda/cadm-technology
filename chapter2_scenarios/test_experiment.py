import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import networkx as nx
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("experiment", ROOT / "src" / "run_experiment.py")
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def reference():
    return json.loads((ROOT / "scenario" / "reference_model.json").read_text(encoding="utf-8"))


def direct_tree():
    return json.loads(
        (ROOT / "runs_v2" / "direct_joint" / "direct_joint_generation.json").read_text(encoding="utf-8")
    )


def test_ultra_direct_baseline_paths_resolve_without_reading_generation_output():
    files = experiment.resolve_direct_baseline_files(ROOT, ROOT / "runs_v2")
    assert files.case_brief == ROOT / "scenario" / "direct_joint_case_brief.json"
    assert files.mapping == ROOT / "scenario" / "direct_joint_ultra_mapping.json"
    assert files.prompt == ROOT / "runs_v2" / "direct_joint_ultra" / "direct_joint_prompt.txt"
    assert files.generation == ROOT / "runs_v2" / "direct_joint_ultra" / "direct_joint_generation.json"
    assert files.source_key == "direct_joint_ultra/direct_joint_generation.json"
    assert files.mapping_source_file == "runs_v2/direct_joint_ultra/direct_joint_generation.json"
    assert experiment.ULTRA_DIRECT_BASELINE.condition_id == "direct_joint_ultra_generation"


def test_reference_model_is_typed_and_all_declared_families_are_valid():
    data = reference()
    experiment.validate_reference_model(data)
    assert data["case_id"] == "UA-EDO-02"
    assert not ({row["id"] for row in data["object_nodes"]} & {row["id"] for row in data["event_nodes"]})
    assert len(data["reference_scenario_families"]) == 5


def test_ahp_matrix_is_reciprocal_and_consistent():
    matrix = reference()["ahp"]["pairwise_matrix"]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            assert abs(value * matrix[j][i] - 1.0) < 1e-8
    result = experiment.ahp_weights(matrix)
    assert abs(sum(result["weights"]) - 1.0) < 1e-10
    assert result["cr"] < 0.10


def test_cause_effect_is_compiled_in_chronological_direction():
    graph = experiment.causal_graph_from_edges([
        {"cause": "earlier", "effect": "later", "relation": "causes"}
    ])
    assert graph.has_edge("earlier", "later")
    assert not graph.has_edge("later", "earlier")
    assert nx.shortest_path(graph, "earlier", "later") == ["earlier", "later"]


def test_local_event_id_cannot_force_a_false_canonical_match():
    canonical = next(row for row in reference()["event_nodes"] if row["id"] == "E10")
    unrelated = {
        "id": "E10",
        "label": "Field crews acknowledge a restoration assignment",
        "node_type": "causal_event",
        "event_class": "response_requirement",
        "domain": "continuity_recovery",
        "required_roles": [],
        "predecessor_logic": "single",
        "observable_result": "A field response is acknowledged.",
    }
    result = experiment.ReferenceMatcher._match_node(unrelated, {"E10": canonical})
    assert result.node_id is None
    assert result.score < 1.0


def test_namespaced_mapping_takes_priority_over_reused_basename():
    matcher = experiment.ReferenceMatcher(reference(), {
        "object_node_mappings": {
            "fixture.json": {"local": "D_IT"},
            "targeted_round1/fixture.json": {"local": "D_OT"},
        },
        "event_node_mappings": {},
    })
    assert matcher._file_overrides("targeted_round1/fixture.json", "object")["local"] == "D_OT"


def test_asymmetric_mapping_is_bound_to_current_reference_and_every_raw_file():
    reference_path = ROOT / "scenario" / "reference_model.json"
    schema = json.loads((ROOT / "schema" / "role_tree.schema.json").read_text(encoding="utf-8"))
    mapping = json.loads((ROOT / "scenario" / "node_mapping_overrides.json").read_text(encoding="utf-8"))
    source_paths = {
        f"{path.parent.name}/{path.name}": path
        for folder in [ROOT / "runs_v2" / "initial", *sorted((ROOT / "runs_v2").glob("targeted*"))]
        if folder.is_dir()
        for path in sorted(folder.glob("*.json"))
    }
    result = experiment.validate_asymmetric_mapping_artifact(
        mapping, reference(), reference_path, schema, source_paths,
    )
    assert result["reference_sha256"] == experiment.sha256(reference_path)
    assert result["source_files"] == 18

    wrong_reference = deepcopy(mapping)
    wrong_reference["metadata"]["reference_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current reference"):
        experiment.validate_asymmetric_mapping_artifact(
            wrong_reference, reference(), reference_path, schema, source_paths,
        )


def test_structural_edges_cannot_enter_causal_graph():
    with pytest.raises(TypeError, match="structural edge"):
        experiment.causal_graph_from_edges([
            {"parent": "organization", "child": "identity_service", "relation": "contains"}
        ])


def test_causal_edges_cannot_enter_object_graph():
    with pytest.raises(TypeError, match="causal edge"):
        experiment.object_graph_from_edges([
            {"source": "event_one", "target": "event_two", "edge_type": "causal"}
        ])


def test_grounding_link_must_join_an_object_to_an_event():
    data = reference()
    broken = deepcopy(data)
    broken["grounding_links"][0]["object_id"] = "E01"
    with pytest.raises(ValueError, match="incorrectly typed endpoint"):
        experiment.validate_reference_model(broken)


def test_grounding_recovery_requires_both_typed_endpoints():
    data = reference()
    all_objects = {row["id"] for row in data["object_nodes"]}
    all_events = {row["id"] for row in data["event_nodes"]}
    assert len(experiment.recovered_grounding_links(data, all_objects, all_events)) == 16
    without_event = all_events - {"E01"}
    links = experiment.recovered_grounding_links(data, all_objects, without_event)
    assert not any(link["event_id"] == "E01" for link in links)
    assert len(links) == 15
    disconnected = experiment.object_graph_from_edges(data["object_edges"])
    disconnected.remove_edge("O_DISPATCH_CONTROL", "O_CONTROL_SERVICE")
    connected_links = experiment.recovered_grounding_links(
        data, all_objects, all_events, object_graph=disconnected
    )
    assert not any(link["object_id"] == "O_CONTROL_SERVICE" for link in connected_links)
    assert len(connected_links) == 13


def test_object_depth_never_changes_causal_distance():
    data = reference()
    causal_graph = experiment.causal_graph_from_edges(data["causal_edges"])
    shallow = experiment.object_graph_from_edges(data["object_edges"])
    deep = shallow.copy()
    deep.remove_edge("D_ID", "O_WORKFORCE")
    deep.add_edges_from([
        ("D_ID", "noun_level_1"),
        ("noun_level_1", "noun_level_2"),
        ("noun_level_2", "O_WORKFORCE"),
    ])
    shallow_row = next(row for row in experiment.complete_scenarios(
        shallow, causal_graph, data["grounding_links"], data
    ) if row["family"] == "workforce_identity")
    deep_row = next(row for row in experiment.complete_scenarios(
        deep, causal_graph, data["grounding_links"], data
    ) if row["family"] == "workforce_identity")
    assert shallow_row["structural_depth"] == 3
    assert deep_row["structural_depth"] == 5
    assert shallow_row["causal_steps"] == deep_row["causal_steps"] == 9


def test_disconnected_later_event_object_blocks_complete_scenario():
    data = reference()
    object_graph = experiment.object_graph_from_edges(data["object_edges"])
    object_graph.remove_edge("O_DISPATCH_CONTROL", "O_CONTROL_SERVICE")
    causal_graph = experiment.causal_graph_from_edges(data["causal_edges"])
    rows = experiment.complete_scenarios(object_graph, causal_graph, data["grounding_links"], data)
    assert "workforce_identity" not in {row["family"] for row in rows}


def test_operational_control_relation_is_shared_validation_bottleneck():
    data = reference()
    all_objects = {row["id"] for row in data["object_nodes"]}
    all_events = {row["id"] for row in data["event_nodes"]}
    causal_graph = experiment.causal_graph_from_edges(data["causal_edges"])
    connected = experiment.object_graph_from_edges(data["object_edges"])

    connected_links = experiment.recovered_grounding_links(
        data, all_objects, all_events, object_graph=connected
    )
    connected_families = experiment.complete_scenarios(
        connected, causal_graph, data["grounding_links"], data
    )
    assert len(connected_links) == 16
    assert {row["family"] for row in connected_families} == {
        "workforce_identity",
        "remote_service",
        "enterprise_endpoint",
        "provider_identity",
        "dispatcher_session",
    }

    disconnected = connected.copy()
    disconnected.remove_edge("O_DISPATCH_CONTROL", "O_CONTROL_SERVICE")
    disconnected_links = experiment.recovered_grounding_links(
        data, all_objects, all_events, object_graph=disconnected
    )
    disconnected_families = experiment.complete_scenarios(
        disconnected, causal_graph, data["grounding_links"], data
    )
    assert len(disconnected_links) == 13
    assert {
        row["link_id"] for row in connected_links
    } - {
        row["link_id"] for row in disconnected_links
    } == {"G10", "G11", "G13"}
    assert disconnected_families == []


def test_full_reference_builds_five_distinct_grounded_families():
    data = reference()
    object_graph = experiment.object_graph_from_edges(data["object_edges"])
    causal_graph = experiment.causal_graph_from_edges(data["causal_edges"])
    rows = experiment.complete_scenarios(object_graph, causal_graph, data["grounding_links"], data)
    assert len(rows) == 5
    assert {row["family"] for row in rows} == {
        "workforce_identity",
        "remote_service",
        "enterprise_endpoint",
        "provider_identity",
        "dispatcher_session",
    }
    assert all(row["event_path"][-1] == "B" for row in rows)
    assert all(row["grounding_link"]["object_id"] == row["object_anchor"] for row in rows)
    assert all(len(row["grounding_bindings"]) == len(row["event_path"]) for row in rows)


def test_directness_uses_only_causal_steps():
    assert experiment._directness_rating(5) == 9
    assert experiment._directness_rating(6) == 7
    assert experiment._directness_rating(7) == 7
    assert experiment._directness_rating(8) == 5
    assert experiment._directness_rating(9) == 3


def test_minimum_event_baseline_can_disagree_with_ahp():
    ranking = {
        "scenarios": [
            {"family": "workforce_identity", "causal_steps": 9, "ahp_score": 0.83},
            {"family": "dispatcher_session", "causal_steps": 5, "ahp_score": 0.76},
        ]
    }
    result = experiment.minimum_event_comparison(ranking)
    assert result["minimum_causal_steps"] == 5
    assert result["minimum_event_families"] == ["dispatcher_session"]
    assert result["ahp_top_family"] == "workforce_identity"
    assert not result["same_family"]


def test_existing_v2_role_outputs_validate_against_asymmetric_schema():
    schema = json.loads((ROOT / "schema" / "role_tree.schema.json").read_text(encoding="utf-8"))
    paths = sorted((ROOT / "runs_v2" / "initial").glob("*.json"))
    assert paths
    for path in paths:
        tree = json.loads(path.read_text(encoding="utf-8"))
        experiment.validate_tree(tree, schema, path)
        metrics = experiment.tree_structure({"tree": tree, "path": str(path)})
        assert metrics["object_depth"] >= 3
        assert metrics["causal_depth"] >= 3
        assert metrics["object_branch_nodes"] >= 1
        assert metrics["causal_branch_nodes"] >= 1


def test_direct_joint_output_is_frozen_and_cross_type_edges_are_excluded_from_scoring():
    path = ROOT / "runs_v2" / "direct_joint" / "direct_joint_generation.json"
    assert experiment.sha256(path) == "5E540782FD5FFA688417968568E51D1DC650E1FFF86474BAD2C57818E4754305"
    schema = json.loads((ROOT / "schema" / "role_tree.schema.json").read_text(encoding="utf-8"))
    raw = direct_tree()
    scored, diagnostics = experiment.validate_direct_baseline(raw, schema, path)
    assert len(raw["causal_edges"]) == 21
    assert len(scored["causal_edges"]) == 17
    assert diagnostics["raw_causal_edges"] == 21
    assert diagnostics["scored_causal_edges"] == 17
    assert len(diagnostics["dropped_causal_edges"]) == 4
    assert {row["cause_type"] for row in diagnostics["dropped_causal_edges"]} == {"object"}
    assert diagnostics["dropped_structural_edges"] == []


def test_direct_mapping_must_adjudicate_every_node_and_preserve_types():
    data = direct_tree()
    object_mapping = {row["id"]: None for row in data["object_nodes"]}
    event_mapping = {row["id"]: None for row in data["event_nodes"]}
    object_mapping["A"], event_mapping["B"] = "A", "B"
    matcher = experiment.ReferenceMatcher(reference(), {
        "object_node_mappings": {"direct.json": object_mapping},
        "event_node_mappings": {"direct.json": event_mapping},
    })
    result = experiment.validate_explicit_mapping(data, matcher, reference(), "direct.json")
    assert result["object_entries"] == len(data["object_nodes"])
    assert result["event_entries"] == len(data["event_nodes"])

    missing = deepcopy(object_mapping)
    del missing["A"]
    incomplete = experiment.ReferenceMatcher(reference(), {
        "object_node_mappings": {"direct.json": missing},
        "event_node_mappings": {"direct.json": event_mapping},
    })
    with pytest.raises(ValueError, match="missing=.*A"):
        experiment.validate_explicit_mapping(data, incomplete, reference(), "direct.json")

    wrong_type = deepcopy(event_mapping)
    wrong_type["B"] = "A"
    invalid = experiment.ReferenceMatcher(reference(), {
        "object_node_mappings": {"direct.json": object_mapping},
        "event_node_mappings": {"direct.json": wrong_type},
    })
    with pytest.raises(ValueError, match="wrong-type canonical ids"):
        experiment.validate_explicit_mapping(data, invalid, reference(), "direct.json")

    flat = experiment.ReferenceMatcher(reference(), {
        "object_node_mappings": object_mapping,
        "event_node_mappings": event_mapping,
    })
    with pytest.raises(ValueError, match="exact object namespace"):
        experiment.validate_explicit_mapping(data, flat, reference(), "direct.json")


def test_direct_baseline_uses_reference_recovery_and_complete_family_rules():
    path = ROOT / "runs_v2" / "direct_joint" / "direct_joint_generation.json"
    mapping_path = ROOT / "scenario" / "direct_joint_mapping.json"
    schema = json.loads((ROOT / "schema" / "role_tree.schema.json").read_text(encoding="utf-8"))
    matcher = experiment.ReferenceMatcher(
        reference(), json.loads(mapping_path.read_text(encoding="utf-8"))
    )
    tree, diagnostics = experiment.load_direct_baseline(
        path,
        schema,
        matcher,
        "direct_joint/direct_joint_generation.json",
        mapping_source_file="runs_v2/direct_joint/direct_joint_generation.json",
    )
    weights = {dimension: 1 / 6 for dimension in experiment.DIMENSIONS}
    condition = experiment.summarize_condition([tree], reference(), weights)
    assert diagnostics["validation"]["scored_causal_edges"] == 17
    assert diagnostics["mapping"]["source_sha256"] == experiment.sha256(path)
    assert condition["object_nodes_reference"] == 40
    assert condition["object_edges_reference"] == 39
    assert condition["causal_nodes_reference"] == 16
    assert condition["causal_edges_reference"] == 16
    assert condition["grounding_links_reference"] == 16
    assert {
        "object_nodes": condition["object_nodes_recovered"],
        "object_relations": condition["object_edges_recovered"],
        "event_nodes": condition["causal_nodes_recovered"],
        "causal_relations": condition["causal_edges_recovered"],
        "usable_connections": condition["grounding_links_recovered"],
        "complete_families": condition["scenario_family_count"],
    } == {
        "object_nodes": 16,
        "object_relations": 14,
        "event_nodes": 6,
        "causal_relations": 2,
        "usable_connections": 1,
        "complete_families": 0,
    }
    declared = {row["id"] for row in reference()["reference_scenario_families"]}
    assert {row["family"] for row in condition["complete_scenarios"]} <= declared


def test_direct_baseline_mapping_is_bound_to_declared_source_and_hash():
    path = ROOT / "runs_v2" / "direct_joint" / "direct_joint_generation.json"
    mapping_path = ROOT / "scenario" / "direct_joint_mapping.json"
    schema = json.loads((ROOT / "schema" / "role_tree.schema.json").read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    wrong_source = deepcopy(mapping)
    wrong_source["source_file"] = "runs_v2/direct_joint_ultra/direct_joint_generation.json"
    with pytest.raises(ValueError, match="declares source_file"):
        experiment.load_direct_baseline(
            path,
            schema,
            experiment.ReferenceMatcher(reference(), wrong_source),
            "direct_joint/direct_joint_generation.json",
            mapping_source_file="runs_v2/direct_joint/direct_joint_generation.json",
        )

    wrong_hash = deepcopy(mapping)
    wrong_hash["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mapping targets"):
        experiment.load_direct_baseline(
            path,
            schema,
            experiment.ReferenceMatcher(reference(), wrong_hash),
            "direct_joint/direct_joint_generation.json",
            mapping_source_file="runs_v2/direct_joint/direct_joint_generation.json",
        )


def test_ultra_direct_mapping_is_bound_to_current_reference():
    path = ROOT / "runs_v2" / "direct_joint_ultra" / "direct_joint_generation.json"
    mapping_path = ROOT / "scenario" / "direct_joint_ultra_mapping.json"
    reference_path = ROOT / "scenario" / "reference_model.json"
    schema = json.loads((ROOT / "schema" / "role_tree.schema.json").read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    expected_reference_hash = experiment.sha256(reference_path)
    assert mapping["reference_sha256"] == expected_reference_hash
    experiment.load_direct_baseline(
        path,
        schema,
        experiment.ReferenceMatcher(reference(), mapping),
        "direct_joint_ultra/direct_joint_generation.json",
        mapping_source_file="runs_v2/direct_joint_ultra/direct_joint_generation.json",
        reference_sha256=expected_reference_hash,
    )
    wrong_reference = deepcopy(mapping)
    wrong_reference["reference_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current reference"):
        experiment.load_direct_baseline(
            path,
            schema,
            experiment.ReferenceMatcher(reference(), wrong_reference),
            "direct_joint_ultra/direct_joint_generation.json",
            mapping_source_file="runs_v2/direct_joint_ultra/direct_joint_generation.json",
            reference_sha256=expected_reference_hash,
        )

def test_generation_comparison_has_three_conditions_and_fixed_execution_counts():
    metrics = {
        "object_nodes_recovered": 1, "object_edges_recovered": 2,
        "causal_nodes_recovered": 3, "causal_edges_recovered": 4,
        "grounding_links_recovered": 5, "complete_scenario_count": 6,
        "scenario_family_count": 7,
    }
    rows = experiment.build_generation_comparison(metrics, metrics, metrics)
    assert [row["condition_id"] for row in rows] == [
        "direct_joint_ultra_generation",
        "initial_asymmetric_generation",
        "asymmetric_after_targeted_refinement",
    ]
    assert [row["task_sessions"] for row in rows] == [1, 3, 11]
    assert [row["stored_json_outputs"] for row in rows] == [1, 6, 18]
    assert [row["complete_reference_paths"] for row in rows] == [6, 6, 6]
    assert [row["complete_reference_families"] for row in rows] == [7, 7, 7]
    assert [row["ordered_structural_endpoint_pairs_recovered"] for row in rows] == [2, 2, 2]
    assert [row["ordered_causal_endpoint_pairs_recovered"] for row in rows] == [4, 4, 4]
    assert all("object_relations_recovered" not in row for row in rows)
    assert all("causal_relations_recovered" not in row for row in rows)
    assert all("approximate_output_tokens" not in row for row in rows)
