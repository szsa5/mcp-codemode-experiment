# Synthetic node-graph ground

The controlled internal-validity ground for the comparative study. A purpose-built
FastMCP server exposes a fixed node-graph dataset through one of two tool surfaces,
with an adjustable number of decoy tools, so the two arms (standard MCP, Code Mode)
can be measured as orchestration depth and tool count grow. Covers RQ1 (depth/K
scaling) and the controlled half of RQ2 (coarse vs fine granularity).

## Files

- `generate_graph.py` writes `graph.json` deterministically (closed-form, no RNG).
  Re-running it reproduces byte-identical data.
- `graph.json` the committed dataset: a 64-node chain, each node
  `{ id, value, attrs{color,size,weight,region,material}, next_id }`.
- `graph_data.py` loads the dataset and holds the oracle (correct answers). The
  oracle is never exposed to the model.
- `tasks.py` builds the task instances and their prompts:
  - `final_value` (RQ1): follow `next_id` D times, report the destination value.
  - `value_sum` (RQ2): follow `next_id` D times and sum the value of every visited
    node, so the fine surface pays two calls per hop where coarse pays one.
- `server.py` the FastMCP SSE server (the tool backend).
- `smoke_test.py` offline end-to-end check over SSE, no Bedrock.
- `run_sweep.py` the sweep runner: launches the server per cell, invokes the
  reusable TS client, scores against the oracle, writes rows to `../results/`.

## Tool surfaces and the K knob

| Surface | Real tools                          | Count |
|---------|-------------------------------------|-------|
| coarse  | `get_node`                          | 1     |
| fine    | `get_value`, `get_attr`, `get_next` | 3     |

`DECOY_TOOLS=N` adds N inert decoy tools, so the advertised tool count is
`K = real_tools + N`. Decoys carry comparable schema weight to real tools.

Server environment knobs: `GRAPH_SURFACE` (coarse|fine|both), `DECOY_TOOLS`,
`GRAPH_HOST` (default 127.0.0.1), `GRAPH_PORT` (default 8200), `GRAPH_DATA`.

## Setup

The TS clients live one level up in `../src`. Node dependencies install at the repo
root; the server uses a local Python venv here.

```
# from the repo root: install node deps (includes tsx, the TS runner)
npm install

# from this synthetic/ folder: create the python venv and install the MCP SDK
python -m venv .venv
./.venv/Scripts/python.exe -m pip install "mcp[cli]"      # Windows
# .venv/bin/python   -m pip install "mcp[cli]"            # POSIX
```

## Verify offline (no AWS needed)

```
./.venv/Scripts/python.exe smoke_test.py
```

Launches the server, drives both surfaces over SSE, and checks every answer against
the oracle. This proves the server, SSE transport, and MCP wiring before any paid run.

## Run the sweep (needs an Anthropic API key)

The synthetic ground serves Claude Haiku 4.5 through the **Anthropic API**
(Option 1: provider per ground; Wesley keeps Bedrock). Provide `ANTHROPIC_API_KEY`
either as a shell environment variable, or in a gitignored `.env` at the repo root
(copy `.env.example` to `.env` and fill it in). `run_sweep.py` loads that `.env`
automatically (override the path with `--env-file`); a shell variable, if set, wins
over the file. The model id, system prompt, round cap, and temperature default
to the values in `run_sweep.py` (`claude-haiku-4-5-20251001`, temperature 0) and can
be overridden with `WESLEY_EXPERIMENT_MODEL_ID`, `WESLEY_EXPERIMENT_SYSTEM_PROMPT`,
`WESLEY_EXPERIMENT_MAX_ROUNDS`, `WESLEY_EXPERIMENT_TEMPERATURE`.

The provider is selected by `MODEL_PROVIDER` (`anthropic` | `bedrock`); the runner
sets `anthropic` for this ground. To point the same clients at Bedrock instead, set
`MODEL_PROVIDER=bedrock` with `AWS_BEDROCK_ID` / `AWS_BEDROCK_SECRET` and a Bedrock
model id. Decoding params (max_tokens 2048, temperature) are applied equally on both
providers; only temperature is set (not top_p), since Haiku 4.5 rejects both together.

```
# see the plan, spend nothing
./.venv/Scripts/python.exe run_sweep.py --dry-run

# one-cell viability check (the first real Anthropic-API run)
./.venv/Scripts/python.exe run_sweep.py --rq rq1 --depths 1 --ks 3 --repeats 1

# the full sweep: RQ1 (D in {1,2,4,8} x K in {3,25}) + RQ2 (coarse vs fine at D=4)
./.venv/Scripts/python.exe run_sweep.py --rq both
```

Each run writes one JSONL row (see `../results/README.md` for the column
dictionary). The runner groups cells by server configuration so the server starts
once per surface/K, not once per task.

## Notes

- The round cap is env-configurable (`WESLEY_EXPERIMENT_MAX_ROUNDS`, default 20),
  comfortably above the deepest task (D=8), so deep standard-MCP runs do not
  truncate artifactually.
- The TS clients run under `tsx` (ts-node does not resolve the explicit `.ts`
  import extensions on Node 22).
- The model-call layer is provider-routed in `src/common/model.ts`: it keeps the
  Bedrock Converse request/response shape as the internal canonical format and
  translates to/from the Anthropic Messages API for `MODEL_PROVIDER=anthropic`.
  The SSE/MCP transport, tool-spec translation, and Code Mode executor are shared.
