"""Offline end-to-end smoke test for the synthetic node-graph server.

Launches the FastMCP server over SSE, connects with the MCP client, exercises
both the coarse and fine surfaces the way the model would, and checks every
answer against the deterministic oracle. No Bedrock and no AWS credentials are
involved, so this validates the SSE + MCP pipeline before any paid run.

Run:  python smoke_test.py
Exit code 0 on success, 1 on any mismatch.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

import graph_data

HERE = Path(__file__).parent
HOST = "127.0.0.1"
PORT = 8265  # off the default 8200 so a running sweep server does not clash
URL = f"http://{HOST}:{PORT}/sse"


def wait_for_port(host: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"server did not open {host}:{port} within {timeout}s")


def _text(result) -> str:
    return "".join(block.text for block in result.content if getattr(block, "text", None))


async def run_checks() -> int:
    graph = graph_data.load_graph()
    failures: list[str] = []

    async with sse_client(URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = {t.name for t in (await session.list_tools()).tools}
            expected_tools = {"get_node", "get_value", "get_attr", "get_next",
                              "decoy_search_0", "decoy_search_1", "decoy_search_2"}
            if tools != expected_tools:
                failures.append(f"tool set mismatch: {sorted(tools)}")

            # Coarse: get_node returns value matching the oracle.
            import json
            node = json.loads(_text(await session.call_tool("get_node", {"node_id": "n5"})))
            if node["value"] != graph["by_id"]["n5"]["value"]:
                failures.append(f"get_node value wrong: {node}")

            # Fine traversal: follow get_next 4 hops from n0, read final get_value.
            current = "n0"
            for _ in range(4):
                current = _text(await session.call_tool("get_next", {"node_id": current}))
            final_value = int(_text(await session.call_tool("get_value", {"node_id": current})))
            expected_final = graph_data.oracle_final_value(graph, "n0", 4)
            if final_value != expected_final:
                failures.append(f"fine final_value {final_value} != oracle {expected_final}")

            # value_sum via fine surface: sum get_value over the 5 visited nodes.
            node_id = "n0"
            total = int(_text(await session.call_tool("get_value", {"node_id": node_id})))
            for _ in range(4):
                node_id = _text(await session.call_tool("get_next", {"node_id": node_id}))
                total += int(_text(await session.call_tool("get_value", {"node_id": node_id})))
            expected_sum = graph_data.oracle_value_sum(graph, "n0", 4)
            if total != expected_sum:
                failures.append(f"fine value_sum {total} != oracle {expected_sum}")

            # get_attr spot check.
            color = _text(await session.call_tool("get_attr", {"node_id": "n3", "key": "color"}))
            if color != graph["by_id"]["n3"]["attrs"]["color"]:
                failures.append(f"get_attr color wrong: {color}")

    if failures:
        print("SMOKE TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("SMOKE TEST PASSED: coarse + fine surfaces agree with the oracle over SSE.")
    return 0


def main() -> int:
    env = dict(os.environ)
    env.update({"GRAPH_SURFACE": "both", "DECOY_TOOLS": "3",
                "GRAPH_HOST": HOST, "GRAPH_PORT": str(PORT)})
    proc = subprocess.Popen([sys.executable, str(HERE / "server.py")],
                            cwd=str(HERE), env=env)
    try:
        wait_for_port(HOST, PORT)
        return asyncio.run(run_checks())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
