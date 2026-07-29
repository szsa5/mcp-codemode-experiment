"""Sweep runner for the synthetic ground.

Orchestrates the RQ1 depth/K sweep and the RQ2 coarse-vs-fine contrast over the
two arms (standard MCP, Code Mode). For each server configuration it launches
the FastMCP SSE server once, runs every (question, arm) cell against it by
invoking the reusable TypeScript client, parses the client's RUN_RESULT_JSON,
scores the final answer against the deterministic oracle, and appends one JSONL
row per run to results/.

This is the only part of Phase A that costs money: each cell is one Bedrock
Converse loop on Haiku. It needs the user's AWS credentials in the environment
(AWS_BEDROCK_ID, AWS_BEDROCK_SECRET, and optionally AWS_BEDROCK_REGION). Use
--dry-run to print the planned cells without spending anything, and --limit to
run a small pilot first.

Examples:
  python run_sweep.py --dry-run                 # show the full plan, spend nothing
  python run_sweep.py --rq rq1 --depths 1 --ks 3 --repeats 1   # 1-cell viability
  python run_sweep.py --rq both                 # the full sweep
"""

import argparse
import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import graph_data
import tasks

HERE = Path(__file__).parent
REPO = HERE.parent
RESULTS_DIR = REPO / "results"

# Synthetic ground serves Haiku 4.5 via the Anthropic API (Option 1: provider per
# ground). Wesley keeps Bedrock. Override the model/provider via env if needed.
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL_ID = "claude-haiku-4-5-20251001"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the user's question by calling the "
    "provided tools to read data; never guess or invent values. Follow the "
    "output instructions exactly and give a final textual answer."
)
DEFAULT_MAX_ROUNDS = "20"
# Decoding params held equal across both providers (Bedrock + Anthropic).
DEFAULT_TEMPERATURE = "0"

REAL_TOOLS = {"coarse": 1, "fine": 3, "both": 4}

CLIENT_PATH = {
    "standard": REPO / "src" / "clients" / "standard_mcp_client.ts",
    "codemode": REPO / "src" / "clients" / "codemode_mcp_client.ts",
}


@dataclass
class Row:
    rq: str
    arm: str
    surface: str
    depth: int
    tool_count: int
    cache: str
    repeat: int
    task_type: str
    task_id: str
    start_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    reasoning_turns: int
    tool_calls: int
    latency_total_ms: int
    setup_ms: int
    parsed_answer: int | None
    expected_answer: int
    success: bool
    code_executed: bool | None
    answer_correct: bool
    error: str | None
    notes: str


# --- server lifecycle ----------------------------------------------------------

def _wait_for_port(host: str, port: int, timeout: float = 25.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"server did not open {host}:{port} within {timeout}s")


@contextlib.contextmanager
def server(surface: str, decoys: int, host: str, port: int):
    env = dict(os.environ)
    env.update({
        "GRAPH_SURFACE": surface,
        "DECOY_TOOLS": str(decoys),
        "GRAPH_HOST": host,
        "GRAPH_PORT": str(port),
    })
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "server.py")],
        cwd=str(HERE), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(host, port)
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# --- client invocation ---------------------------------------------------------

def _ts_node_cmd(client: Path) -> list[str]:
    binname = "tsx.cmd" if os.name == "nt" else "tsx"
    bin_path = REPO / "node_modules" / ".bin" / binname
    return [str(bin_path), str(client)]


def _parse_run_result(stdout: str) -> dict | None:
    result = None
    for line in stdout.splitlines():
        if line.startswith("RUN_RESULT_JSON:"):
            with contextlib.suppress(json.JSONDecodeError):
                result = json.loads(line[len("RUN_RESULT_JSON:"):])
    return result


def _extract_int(text: str) -> int | None:
    matches = re.findall(r"-?\d+", text.replace(",", ""))
    return int(matches[-1]) if matches else None


def _answer_correct(text: str, expected: int) -> bool:
    """A numeric-answer run is correct if the oracle integer appears as a standalone
    token in the final answer. This is robust to prose answers that state the answer
    and then list component values: the older 'last integer' heuristic mis-scored one
    value_sum case where the model answered correctly (sum) but then listed the visited
    nodes, so the trailing node value was extracted instead of the sum. The oracle
    values here are specific (a single value or a large sum), so coincidental presence
    is not a realistic false-positive risk for these tasks."""
    ints = [int(m) for m in re.findall(r"-?\d+", (text or "").replace(",", ""))]
    return expected in ints


def run_client(arm: str, prompt: str, mcp_url: str, env_extra: dict) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    env["WESLEY_MCP_URL"] = mcp_url
    # Pass the prompt via the environment, not argv: cmd.exe mangles a multi-line
    # argument on Windows, but the process environment block preserves newlines.
    env["WESLEY_EXPERIMENT_PROMPT"] = prompt
    proc = subprocess.run(
        _ts_node_cmd(CLIENT_PATH[arm]),
        cwd=str(REPO), env=env,
        capture_output=True, text=True,
    )
    result = _parse_run_result(proc.stdout)
    if result is None:
        return {"error": f"no RUN_RESULT_JSON (exit {proc.returncode}); "
                         f"stderr tail: {proc.stderr[-300:]}"}
    return result


# --- scoring -------------------------------------------------------------------

def score(arm: str, rq: str, surface: str, tool_count: int, cache: str,
          q: tasks.Question, result: dict) -> Row:
    totals = result.get("totals", {})
    traces = result.get("tool_traces", [])
    final_answer = result.get("final_answer", "") or ""
    error = result.get("error")
    answer_correct = _answer_correct(final_answer, q.expected)
    # parsed_answer reports the matched oracle value when correct, else the best-effort
    # extraction so a wrong answer still shows what the model actually said (e.g. the
    # standard-MCP off-by-one at depth 8).
    parsed = q.expected if answer_correct else _extract_int(final_answer)
    code_blocks = result.get("generated_code_blocks") or []
    code_executed = None
    if arm == "codemode":
        code_executed = error is None and len(code_blocks) > 0
    return Row(
        rq=rq, arm=arm, surface=surface, depth=q.depth, tool_count=tool_count,
        cache=cache,
        repeat=int(q.task_id.rsplit("_r", 1)[1]),
        task_type=q.task_type, task_id=q.task_id, start_id=q.start_id,
        prompt_tokens=totals.get("prompt_tokens", 0),
        completion_tokens=totals.get("completion_tokens", 0),
        total_tokens=totals.get("total_tokens", 0),
        cache_creation_tokens=totals.get("cache_creation_tokens", 0),
        cache_read_tokens=totals.get("cache_read_tokens", 0),
        reasoning_turns=result.get("reasoning_turns", 0),
        tool_calls=len(traces),
        latency_total_ms=result.get("duration_ms", 0),
        setup_ms=result.get("setup_duration_ms", 0),
        parsed_answer=parsed, expected_answer=q.expected,
        success=answer_correct, code_executed=code_executed,
        answer_correct=answer_correct, error=error, notes="",
    )


# --- plan ----------------------------------------------------------------------

@dataclass
class Cell:
    rq: str
    surface: str
    decoys: int
    tool_count: int
    questions: list
    arms: list


def build_plan(args, graph) -> list[Cell]:
    cells: list[Cell] = []
    arms = args.arms
    if args.rq in ("rq1", "both"):
        q = tasks.rq1_questions(graph, args.depths, args.repeats)
        if args.limit:
            q = q[: args.limit]
        for k in args.ks:
            decoys = k - REAL_TOOLS["coarse"]
            if decoys < 0:
                raise SystemExit(f"--ks {k} is below the coarse real-tool count")
            cells.append(Cell("rq1", "coarse", decoys, k, q, arms))
    if args.rq in ("rq2", "both"):
        q = tasks.rq2_questions(graph, args.rq2_depth, args.repeats)
        if args.limit:
            q = q[: args.limit]
        for surface in ("coarse", "fine"):
            decoys = args.rq2_k - REAL_TOOLS[surface]
            if decoys < 0:
                raise SystemExit(f"--rq2-k {args.rq2_k} below {surface} real-tool count")
            cells.append(Cell("rq2", surface, decoys, args.rq2_k, q, arms))
    return cells


def load_env_file(path: Path) -> dict:
    """Load KEY=VALUE / `export KEY=VALUE` lines from a secrets file into the
    process environment without overriding vars already set in the shell.

    Returns the names of the keys it set (never values). Comments (#) and blank
    lines are ignored; surrounding single or double quotes are stripped. The file
    is gitignored; keep real keys out of version control.
    """
    set_keys: list[str] = []
    if not path.exists():
        return {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            set_keys.append(key)
    return {k: True for k in set_keys}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rq", choices=["rq1", "rq2", "both"], default="both")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--ks", type=int, nargs="+", default=[3, 25])
    p.add_argument("--rq2-depth", type=int, default=4)
    p.add_argument("--rq2-k", type=int, default=3)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--arms", nargs="+", default=["standard", "codemode"],
                   choices=["standard", "codemode"])
    p.add_argument("--limit", type=int, default=0,
                   help="cap questions per cell (quick pilot); 0 = no cap")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--base-port", type=int, default=8200)
    p.add_argument("--out", default=str(RESULTS_DIR / "synthetic_sweep.jsonl"))
    p.add_argument("--dry-run", action="store_true",
                   help="print the planned cells and exit without spending")
    p.add_argument("--append", action="store_true",
                   help="append to --out instead of overwriting (for staged runs)")
    p.add_argument("--cache", choices=["on", "off"], default="off",
                   help="prompt caching: 'on' adds cache_control breakpoints "
                        "(Anthropic path); recorded in the cache column")
    p.add_argument("--env-file", default=str(REPO / ".env"),
                   help="path to a KEY=VALUE secrets file loaded into the client "
                        "env (gitignored; default: repo-root .env)")
    args = p.parse_args()

    loaded = load_env_file(Path(args.env_file))
    if loaded:
        print(f"loaded {len(loaded)} var(s) from {args.env_file}: "
              f"{', '.join(sorted(loaded))}")

    graph = graph_data.load_graph()
    cells = build_plan(args, graph)

    total_runs = sum(len(c.questions) * len(c.arms) for c in cells)
    print(f"plan: {len(cells)} server config(s), {total_runs} client runs")
    for c in cells:
        print(f"  [{c.rq}] surface={c.surface} K={c.tool_count} "
              f"(decoys={c.decoys}) questions={len(c.questions)} arms={c.arms}")

    if args.dry_run:
        return 0

    provider = os.environ.get("MODEL_PROVIDER", DEFAULT_PROVIDER)
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nANTHROPIC_API_KEY is not set; the client cannot reach the "
              "Anthropic API.\nExport ANTHROPIC_API_KEY before a real run, or "
              "pass --dry-run.", file=sys.stderr)
        return 2
    if provider == "bedrock" and not os.environ.get("AWS_BEDROCK_ID"):
        print("\nAWS_BEDROCK_ID is not set; the client cannot reach Bedrock.\n"
              "Set AWS_BEDROCK_ID / AWS_BEDROCK_SECRET (and optionally\n"
              "AWS_BEDROCK_REGION) before a real run, or pass --dry-run.",
              file=sys.stderr)
        return 2

    env_extra = {
        "MODEL_PROVIDER": provider,
        "CACHE": args.cache,
        "WESLEY_EXPERIMENT_MODEL_ID": os.environ.get(
            "WESLEY_EXPERIMENT_MODEL_ID", DEFAULT_MODEL_ID),
        "WESLEY_EXPERIMENT_SYSTEM_PROMPT": os.environ.get(
            "WESLEY_EXPERIMENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        "WESLEY_EXPERIMENT_MAX_ROUNDS": os.environ.get(
            "WESLEY_EXPERIMENT_MAX_ROUNDS", DEFAULT_MAX_ROUNDS),
        "WESLEY_EXPERIMENT_TEMPERATURE": os.environ.get(
            "WESLEY_EXPERIMENT_TEMPERATURE", DEFAULT_TEMPERATURE),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Raw, verbatim RUN_RESULT per run (final_answer, generated code, tool traces,
    # usage) goes in a parallel file: the integrity-grade evidence behind every
    # scored row, and the source for RQ3 qualitative categorization.
    raw_path = out_path.with_name(out_path.stem + "_raw" + out_path.suffix)
    mode = "a" if args.append else "w"
    written = 0
    started = time.monotonic()
    with out_path.open(mode, encoding="utf-8") as fh, raw_path.open(mode, encoding="utf-8") as rawfh:
        for ci, c in enumerate(cells):
            port = args.base_port + ci
            mcp_url = f"http://{args.host}:{port}/sse"
            print(f"\n=== cell {ci + 1}/{len(cells)} [{c.rq}] "
                  f"surface={c.surface} K={c.tool_count} -> {mcp_url} ===")
            with server(c.surface, c.decoys, args.host, port):
                for q in c.questions:
                    for arm in c.arms:
                        result = run_client(arm, q.prompt, mcp_url, env_extra)
                        row = score(arm, c.rq, c.surface, c.tool_count,
                                    args.cache, q, result)
                        fh.write(json.dumps(asdict(row)) + "\n")
                        fh.flush()
                        rawfh.write(json.dumps({
                            "rq": c.rq, "arm": arm, "surface": c.surface,
                            "tool_count": c.tool_count, "cache": args.cache,
                            "depth": q.depth,
                            "repeat": int(q.task_id.rsplit("_r", 1)[1]),
                            "task_id": q.task_id, "start_id": q.start_id,
                            "expected_answer": q.expected, "prompt": q.prompt,
                            "result": result,
                        }) + "\n")
                        rawfh.flush()
                        written += 1
                        flag = "ok " if row.success else "MISS"
                        print(f"  {flag} {arm:8s} {q.task_id} cache={args.cache} "
                              f"parsed={row.parsed_answer} expected={row.expected_answer} "
                              f"tok={row.total_tokens} cw={row.cache_creation_tokens} "
                              f"cr={row.cache_read_tokens} turns={row.reasoning_turns} "
                              f"calls={row.tool_calls}"
                              + (f" err={row.error[:50]}" if row.error else ""))

    elapsed = time.monotonic() - started
    print(f"\nwrote {written} rows to {out_path} in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
