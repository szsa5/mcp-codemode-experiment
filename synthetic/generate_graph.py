"""Deterministically generate the synthetic node-graph dataset (graph.json).

The dataset is the fixed, committed ground truth for the synthetic experiments.
Generation is fully deterministic (closed-form integer formulas, no RNG), so
re-running this script reproduces byte-identical data on any machine and any
Python version. Commit graph.json; do not edit it by hand.

Graph shape: a single linear chain n0 -> n1 -> ... -> n{N-1}. Each node is
{ id, value, attrs{...}, next_id }. The chain lets a task follow `next_id` D
times from a start node; the destination is start_index + D. With N=64 and the
sweep using starts n0..n7 and depths up to 8, the deepest destination is index
15, well inside the chain.
"""

import json
from pathlib import Path

NODE_COUNT = 64

COLORS = ["red", "green", "blue", "amber", "violet", "teal", "coral", "slate"]
REGIONS = ["north", "south", "east", "west"]
MATERIALS = ["oak", "steel", "glass", "linen", "basalt"]


def make_node(i: int) -> dict:
    """Build node i with closed-form deterministic value and attributes."""
    return {
        "id": f"n{i}",
        "value": (i * 37 + 11) % 997,
        "attrs": {
            "color": COLORS[i % len(COLORS)],
            "size": (i * 13 + 5) % 50,
            "weight": (i * 29 + 7) % 100,
            "region": REGIONS[(i // 4) % len(REGIONS)],
            "material": MATERIALS[(i * 3) % len(MATERIALS)],
        },
        "next_id": f"n{i + 1}" if i < NODE_COUNT - 1 else None,
    }


def build_graph() -> dict:
    nodes = [make_node(i) for i in range(NODE_COUNT)]
    return {
        "version": 1,
        "node_count": NODE_COUNT,
        "start_id": "n0",
        "nodes": nodes,
    }


def main() -> None:
    graph = build_graph()
    out = Path(__file__).with_name("graph.json")
    out.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({graph['node_count']} nodes)")


if __name__ == "__main__":
    main()
