# mcp-codemode-experiment

Experiment harness and replication artifact for the thesis *Evaluating
Scalability of Model Context Protocol in LLM Tool Integration*: standard MCP
vs.\ Code Mode orchestration. This repo is the committable, reproducible
artifact for the synthetic ground; the Wesley case study uses the same clients
against a private server.

**Scope of this public artifact:** the synthetic ground in full (server, clients,
raw runs, aggregation and plotting scripts, figures). The Wesley case study's data
cannot be redistributed and is not included; the thesis reports its aggregate
results, and the record is retained privately by the author.

## Layout

- `src/clients/` — the reusable Bedrock + SSE + MCP clients (`standard_mcp_client.ts`,
  `codemode_mcp_client.ts`), env-parameterized via `WESLEY_MCP_URL`,
  `WESLEY_EXPERIMENT_MODEL_ID`, `WESLEY_EXPERIMENT_SYSTEM_PROMPT`. Copied from the
  Wesley repo; canonical here.
- `src/common/` — Bedrock `Converse`, MCP SSE connection, config, logging, types.
- `synthetic/` — the synthetic node-graph SSE server (Python FastMCP): coarse
  `get_node` vs fine `get_value`/`get_attr`/`get_next`, depth via `next_id`, a
  `DECOY_TOOLS=N` knob, a deterministic oracle, and the sweep runner.
- `results/` — committed raw runs, data dictionary, aggregation/plotting scripts,
  figures. Ground truth for the Evaluation chapter.

## Model

Claude Haiku 4.5 on both grounds, the same snapshot, served per ground: the
synthetic ground on the Anthropic API (`claude-haiku-4-5-20251001`), Wesley on
Bedrock (its production inference profile). The model id is set via
`WESLEY_EXPERIMENT_MODEL_ID`; the provider via `MODEL_PROVIDER`.

## Grounds

The synthetic node-graph server lives here and is fully reproducible. The Wesley
case study runs these same clients against the Wesley MCP server (a private
repository) by pointing `WESLEY_MCP_URL` at it.

Full design and rationale: the accompanying MSc thesis (Vrije Universiteit Amsterdam, 2026).
