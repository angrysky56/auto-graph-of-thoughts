# Copyright (c) 2026 angrysky56 (Ty).
#
# Tests for the graph-theoretic diagnostics over a Graph of Operations.

from __future__ import annotations

import pytest

from graph_of_thoughts.diagnostics import compute_metrics, metrics_from_graph_def


def test_line_graph_metrics():
    gd = {
        "nodes": [
            {"id": "a", "type": "generate"},
            {"id": "b", "type": "score", "predecessors": ["a"]},
            {"id": "c", "type": "keep_best_n", "predecessors": ["b"]},
        ]
    }
    m = metrics_from_graph_def(gd)
    assert m["num_nodes"] == 3
    assert m["num_edges"] == 2
    assert m["num_roots"] == 1  # a
    assert m["num_leaves"] == 1  # c
    assert m["depth"] == 2  # a -> b -> c
    assert m["diameter"] == 2
    assert 0.0 < m["global_efficiency"] <= 1.0
    assert m["density"] == pytest.approx(2 / 3, abs=1e-3)


def test_hub_reliance_detected_in_star():
    # one central node every cross-pair routes through -> high betweenness
    gd = {
        "nodes": [
            {"id": "hub", "type": "aggregate"},
            {"id": "l1", "type": "x", "predecessors": ["hub"]},
            {"id": "l2", "type": "x", "predecessors": ["hub"]},
            {"id": "l3", "type": "x", "predecessors": ["hub"]},
        ]
    }
    m = metrics_from_graph_def(gd)
    assert "hub" in (m["betweenness_top_node"] or "")
    assert m["hub_reliance"] > 1.5
    assert any("hub reliance" in note for note in m["notes"])


def test_diamond_graph_is_more_efficient_than_line():
    # diamond: a->b, a->c, b->d, c->d  (parallel paths)
    diamond = {
        "nodes": [
            {"id": "a", "type": "generate"},
            {"id": "b", "type": "score", "predecessors": ["a"]},
            {"id": "c", "type": "score", "predecessors": ["a"]},
            {"id": "d", "type": "aggregate", "predecessors": ["b", "c"]},
        ]
    }
    line = {
        "nodes": [
            {"id": "a", "type": "x"},
            {"id": "b", "type": "x", "predecessors": ["a"]},
            {"id": "c", "type": "x", "predecessors": ["b"]},
            {"id": "d", "type": "x", "predecessors": ["c"]},
        ]
    }
    assert (
        metrics_from_graph_def(diamond)["global_efficiency"]
        > metrics_from_graph_def(line)["global_efficiency"]
    )


def test_single_node_is_trivial():
    m = compute_metrics(["only"], [], {"only": "generate"})
    assert m["num_nodes"] == 1
    assert m["global_efficiency"] == 0.0
    assert "trivial" in m["notes"][0]
