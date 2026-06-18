# Copyright (c) 2026 angrysky56 (Ty).
#                    All rights reserved.
#
# Use of this source code is governed by the BSD-style license that governs
# this repository (see LICENSE). Original addition; pure-Python, no dependencies.

"""Graph-theoretic health diagnostics for a Graph of Operations.

Points the same toolkit that connectomics uses on brain networks at GoT's own
reasoning DAG: density, global efficiency, characteristic path length, hub
(betweenness) concentration, clustering, and DAG depth. The motivating analogy
is that dysfunction in a network often lives in its *connectivity* (long routes,
over-reliance on a few hubs) rather than in any single node — so these metrics
flag when a reasoning graph is inefficient or hub-fragile.

All metrics are computed with small, dependency-free implementations:
* shortest paths via BFS (unweighted),
* betweenness via Brandes' algorithm,
* clustering as the local clustering coefficient,
* depth as the longest path in the DAG.

Topology metrics (efficiency, path length, betweenness, clustering) treat edges
as **undirected**, matching how brain connectomes are analysed. Structure
metrics (roots, leaves, depth) respect edge **direction** (information flow).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

Node = Hashable
Edge = Tuple[Node, Node]


# --------------------------------------------------------------------------- #
# Core graph computations
# --------------------------------------------------------------------------- #


def _undirected_adj(nodes: Sequence[Node], edges: Sequence[Edge]) -> Dict[Node, set]:
    adj: Dict[Node, set] = {n: set() for n in nodes}
    for u, v in edges:
        if u == v:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _directed_adj(nodes: Sequence[Node], edges: Sequence[Edge]) -> Dict[Node, set]:
    adj: Dict[Node, set] = {n: set() for n in nodes}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
    return adj


def _bfs_distances(adj: Dict[Node, set], source: Node) -> Dict[Node, int]:
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


def global_efficiency(adj: Dict[Node, set]) -> float:
    """Average inverse shortest-path length over all ordered node pairs.

    1.0 = fully connected (every node one hop away); 0.0 = no paths.
    """
    nodes = list(adj)
    n = len(nodes)
    if n < 2:
        return 0.0
    total = 0.0
    for src in nodes:
        dist = _bfs_distances(adj, src)
        for tgt, d in dist.items():
            if tgt != src and d > 0:
                total += 1.0 / d
    return total / (n * (n - 1))


def characteristic_path_length(adj: Dict[Node, set]) -> Tuple[float, int]:
    """Mean and max shortest-path length over reachable pairs (diameter)."""
    nodes = list(adj)
    total, count, diameter = 0.0, 0, 0
    for src in nodes:
        dist = _bfs_distances(adj, src)
        for tgt, d in dist.items():
            if tgt != src and d > 0:
                total += d
                count += 1
                diameter = max(diameter, d)
    return (total / count if count else 0.0, diameter)


def betweenness_centrality(adj: Dict[Node, set]) -> Dict[Node, float]:
    """Brandes' betweenness centrality (unweighted, undirected), normalised."""
    nodes = list(adj)
    bc: Dict[Node, float] = {n: 0.0 for n in nodes}
    for s in nodes:
        stack: List[Node] = []
        preds: Dict[Node, List[Node]] = {w: [] for w in nodes}
        sigma = {w: 0.0 for w in nodes}
        sigma[s] = 1.0
        dist = {s: 0}
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = {w: 0.0 for w in nodes}
        while stack:
            w = stack.pop()
            for v in preds[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]
    # undirected: each pair counted twice
    n = len(nodes)
    scale = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
    for w in bc:
        bc[w] *= scale / 2.0 if n > 2 else 0.0
    return bc


def average_clustering(adj: Dict[Node, set]) -> float:
    """Mean local clustering coefficient (small-world indicator)."""
    nodes = list(adj)
    if not nodes:
        return 0.0
    total = 0.0
    for v in nodes:
        nbrs = adj[v]
        k = len(nbrs)
        if k < 2:
            continue
        links = 0
        nbr_list = list(nbrs)
        for i in range(len(nbr_list)):
            for j in range(i + 1, len(nbr_list)):
                if nbr_list[j] in adj[nbr_list[i]]:
                    links += 1
        total += (2.0 * links) / (k * (k - 1))
    return total / len(nodes)


def longest_path_depth(nodes: Sequence[Node], edges: Sequence[Edge]) -> int:
    """Longest directed path length (edges) in the DAG = reasoning depth."""
    dadj = _directed_adj(nodes, edges)
    memo: Dict[Node, int] = {}

    def dfs(u: Node, seen: frozenset) -> int:
        if u in memo:
            return memo[u]
        if u in seen:  # cycle guard (GoT graphs are DAGs, but be safe)
            return 0
        best = 0
        for w in dadj[u]:
            best = max(best, 1 + dfs(w, seen | {u}))
        memo[u] = best
        return best

    return max((dfs(n, frozenset()) for n in nodes), default=0)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def compute_metrics(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    labels: Optional[Dict[Node, str]] = None,
) -> dict:
    """Compute the full diagnostic report for a node/edge graph."""
    labels = labels or {}
    n = len(nodes)
    udir = _undirected_adj(nodes, edges)
    dir_adj = _directed_adj(nodes, edges)
    undirected_edges = sum(len(v) for v in udir.values()) // 2

    max_possible = n * (n - 1) / 2 if n > 1 else 0
    density = (undirected_edges / max_possible) if max_possible else 0.0

    eff = global_efficiency(udir)
    avg_path, diameter = characteristic_path_length(udir)
    bc = betweenness_centrality(udir)
    clustering = average_clustering(udir)

    degrees = {node: len(udir[node]) for node in nodes}
    hub_node = max(degrees, key=degrees.get) if degrees else None
    max_degree = degrees.get(hub_node, 0) if hub_node is not None else 0

    bc_values = list(bc.values())
    mean_bc = (sum(bc_values) / len(bc_values)) if bc_values else 0.0
    max_bc = max(bc_values) if bc_values else 0.0
    top_bc_node = max(bc, key=bc.get) if bc else None
    # hub reliance: how concentrated routing is on the busiest node (1.0 = even)
    hub_reliance = (max_bc / mean_bc) if mean_bc > 0 else 0.0

    roots = [node for node in nodes if not dir_adj_has_incoming(node, edges)]
    leaves = [node for node in nodes if len(dir_adj[node]) == 0]
    depth = longest_path_depth(nodes, edges)

    report = {
        "num_nodes": n,
        "num_edges": undirected_edges,
        "density": round(density, 4),
        "global_efficiency": round(eff, 4),
        "avg_path_length": round(avg_path, 4),
        "diameter": diameter,
        "avg_clustering": round(clustering, 4),
        "max_degree": max_degree,
        "hub_node": _label(hub_node, labels),
        "betweenness_top_node": _label(top_bc_node, labels),
        "max_betweenness": round(max_bc, 4),
        "hub_reliance": round(hub_reliance, 4),
        "num_roots": len(roots),
        "num_leaves": len(leaves),
        "depth": depth,
        "notes": _interpret(n, eff, hub_reliance, avg_path, depth),
    }
    return report


def dir_adj_has_incoming(node: Node, edges: Sequence[Edge]) -> bool:
    return any(v == node for _, v in edges)


def _label(node: Optional[Node], labels: Dict[Node, str]) -> Optional[str]:
    if node is None:
        return None
    return f"{labels[node]}#{node}" if node in labels else str(node)


def _interpret(
    n: int, eff: float, hub_reliance: float, avg_path: float, depth: int
) -> List[str]:
    """Plain-language flags mirroring the connectome framing."""
    notes: List[str] = []
    if n < 2:
        notes.append("trivial graph (single node)")
        return notes
    if eff >= 0.6:
        notes.append("efficient routing (information reaches nodes directly)")
    elif eff < 0.35:
        notes.append("inefficient routing (long, indirect paths between steps)")
    if hub_reliance >= 2.5:
        notes.append(
            "high hub reliance: routing concentrates on one operation "
            "(fragile single point of failure)"
        )
    elif hub_reliance and hub_reliance < 1.5:
        notes.append("evenly distributed routing (no dominant hub)")
    if avg_path > 2.5:
        notes.append("long characteristic path length (deep/serial reasoning)")
    return notes


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


def metrics_from_graph_def(graph_def: dict) -> dict:
    """Structural metrics for a proposed graph_def (no execution needed)."""
    nodes_spec = graph_def.get("nodes", [])
    nodes = [node["id"] for node in nodes_spec]
    labels = {node["id"]: node.get("type", "op") for node in nodes_spec}
    edges: List[Edge] = []
    for node in nodes_spec:
        for pred in node.get("predecessors", []) or []:
            edges.append((pred, node["id"]))
    return compute_metrics(nodes, edges, labels)


def metrics_from_operations(graph) -> dict:
    """Metrics for an executed GraphOfOperations (uses op ids + successors)."""
    ops = list(graph.operations)
    nodes = [op.id for op in ops]
    labels = {
        op.id: getattr(op.operation_type, "name", str(op.operation_type))
        for op in ops
    }
    edges: List[Edge] = []
    for op in ops:
        for succ in op.successors:
            edges.append((op.id, succ.id))
    return compute_metrics(nodes, edges, labels)
