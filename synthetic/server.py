"""Synthetic node-graph MCP server (FastMCP over SSE).

The controlled ground for RQ1 (depth/K scaling) and the controlled half of RQ2
(coarse vs fine granularity). The same fixed dataset is exposed through one of
two real tool surfaces, plus an adjustable number of inert decoy tools to dial
the advertised tool count K. The reusable Bedrock+SSE+MCP clients reach it by
pointing WESLEY_MCP_URL at http://<host>:<port>/sse.

Environment knobs (all optional):
  GRAPH_SURFACE   coarse | fine | both   (default: coarse)
                  coarse -> get_node (1 real tool)
                  fine   -> get_value, get_attr, get_next (3 real tools)
                  both   -> all four (used only for the offline smoke test)
  DECOY_TOOLS     integer N >= 0         (default: 0)
                  adds N inert decoy tools so K = real_tools + N
  GRAPH_HOST      bind host              (default: 127.0.0.1)
  GRAPH_PORT      bind port              (default: 8200)
  GRAPH_DATA      path to graph.json     (default: ./graph.json next to this file)

The dataset and the oracle live in graph_data.py; nothing here computes or
exposes a task's answer.
"""

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import graph_data

GRAPH_PATH = os.environ.get("GRAPH_DATA") or str(graph_data.GRAPH_PATH)
GRAPH = graph_data.load_graph(GRAPH_PATH)

SURFACE = os.environ.get("GRAPH_SURFACE", "coarse").strip().lower()
DECOY_TOOLS = int(os.environ.get("DECOY_TOOLS", "0"))
HOST = os.environ.get("GRAPH_HOST", "127.0.0.1")
PORT = int(os.environ.get("GRAPH_PORT", "8200"))

if SURFACE not in {"coarse", "fine", "both"}:
    raise SystemExit(f"GRAPH_SURFACE must be coarse|fine|both, got {SURFACE!r}")
if DECOY_TOOLS < 0:
    raise SystemExit(f"DECOY_TOOLS must be >= 0, got {DECOY_TOOLS}")


# --- Real tools: coarse surface ------------------------------------------------

def get_node(node_id: str) -> str:
    """Return the full node as JSON: its id, integer value, attrs object, and
    next_id pointer. This is the coarse, one-call-per-node tool."""
    node = GRAPH["by_id"].get(node_id)
    if node is None:
        return json.dumps({"error": f"unknown node_id: {node_id}"})
    return json.dumps(
        {
            "id": node["id"],
            "value": node["value"],
            "attrs": node["attrs"],
            "next_id": node["next_id"],
        }
    )


# --- Real tools: fine surface --------------------------------------------------

def get_value(node_id: str) -> str:
    """Return only the integer value stored at the given node, as a string."""
    node = GRAPH["by_id"].get(node_id)
    if node is None:
        return json.dumps({"error": f"unknown node_id: {node_id}"})
    return str(node["value"])


def get_attr(node_id: str, key: str) -> str:
    """Return only the named attribute (e.g. color, size, weight, region,
    material) of the given node, as a string."""
    node = GRAPH["by_id"].get(node_id)
    if node is None:
        return json.dumps({"error": f"unknown node_id: {node_id}"})
    if key not in node["attrs"]:
        return json.dumps({"error": f"unknown attr key: {key}"})
    return str(node["attrs"][key])


def get_next(node_id: str) -> str:
    """Return only the next_id pointer of the given node (the string 'null' if
    this node ends the chain)."""
    node = GRAPH["by_id"].get(node_id)
    if node is None:
        return json.dumps({"error": f"unknown node_id: {node_id}"})
    next_id = node["next_id"]
    return "null" if next_id is None else str(next_id)


# --- Decoy tools ---------------------------------------------------------------

_DECOY_NOUNS = [
    "invoice", "shipment", "ticket", "sensor", "contract", "playlist",
    "warehouse", "satellite", "recipe", "patent", "ledger", "vineyard",
]


def _make_decoy(index: int):
    noun = _DECOY_NOUNS[index % len(_DECOY_NOUNS)]

    def decoy(query: str, limit: int = 10) -> str:
        """placeholder; replaced per-instance below"""
        return json.dumps({"matches": [], "query": query, "limit": limit})

    decoy.__doc__ = (
        f"Search the {noun} catalogue for records matching a free-text query and "
        f"return up to `limit` results. Unrelated to the node graph; provided to "
        f"populate the tool surface."
    )
    return decoy


# --- Registration --------------------------------------------------------------

mcp = FastMCP("synthetic-node-graph", host=HOST, port=PORT)

if SURFACE in {"coarse", "both"}:
    mcp.add_tool(get_node, name="get_node")
if SURFACE in {"fine", "both"}:
    mcp.add_tool(get_value, name="get_value")
    mcp.add_tool(get_attr, name="get_attr")
    mcp.add_tool(get_next, name="get_next")

for i in range(DECOY_TOOLS):
    fn = _make_decoy(i)
    mcp.add_tool(fn, name=f"decoy_search_{i}", description=fn.__doc__)


if __name__ == "__main__":
    real = {"coarse": 1, "fine": 3, "both": 4}[SURFACE]
    print(
        f"[server] surface={SURFACE} real_tools={real} decoys={DECOY_TOOLS} "
        f"K={real + DECOY_TOOLS} sse=http://{HOST}:{PORT}/sse",
        flush=True,
    )
    mcp.run(transport="sse")
