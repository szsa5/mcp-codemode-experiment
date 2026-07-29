"""Load the committed node-graph dataset and provide the deterministic oracle.

This module is the single source of truth shared by the server, the smoke test,
and the sweep runner. The oracle here defines the correct answer for every task;
nothing about it is exposed to the model as a tool.
"""

import json
from pathlib import Path

GRAPH_PATH = Path(__file__).with_name("graph.json")


def load_graph(path: Path | str = GRAPH_PATH) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {node["id"]: node for node in data["nodes"]}
    data["by_id"] = by_id
    return data


def get_node(graph: dict, node_id: str) -> dict:
    node = graph["by_id"].get(node_id)
    if node is None:
        raise KeyError(node_id)
    return node


def walk(graph: dict, start_id: str, depth: int) -> list[dict]:
    """Return the nodes visited by following next_id `depth` times from start.

    The returned list has depth+1 entries: the start node plus one per hop.
    Raises KeyError if the chain runs out before `depth` hops are taken.
    """
    visited = [get_node(graph, start_id)]
    current = visited[0]
    for _ in range(depth):
        next_id = current["next_id"]
        if next_id is None:
            raise ValueError(f"chain ended at {current['id']} before {depth} hops")
        current = get_node(graph, next_id)
        visited.append(current)
    return visited


def oracle_final_value(graph: dict, start_id: str, depth: int) -> int:
    """RQ1 task answer: the value at the node reached after `depth` next-hops."""
    return walk(graph, start_id, depth)[-1]["value"]


def oracle_value_sum(graph: dict, start_id: str, depth: int) -> int:
    """RQ2 task answer: sum of the value of every node visited (start + hops).

    This task needs each visited node's value AND its next pointer, so the fine
    surface pays two calls per hop (get_value + get_next) where the coarse
    surface pays one (get_node). It is the granularity-amplifying task.
    """
    return sum(node["value"] for node in walk(graph, start_id, depth))
