# Results (committed experiment data)

Ground truth for the Evaluation chapter's synthetic results. Every synthetic number
in the thesis traces to a file here; the Wesley record is private (see the top-level
README). The build/run phase fills this in; the writing phase reads it and writes
nothing that is not sourced here.

## Reproducibility record (fill on first run)

- Model (Claude Haiku 4.5, same snapshot both grounds; serving platform differs per
  Option 1 = provider per ground):
  - Synthetic ground: Anthropic API, model id `claude-haiku-4-5-20251001`
    (author's personal key). Provider path built (`src/common/model.ts`,
    `MODEL_PROVIDER=anthropic`); synthetic sweep RAN on this path (see below).
  - Wesley ground: Claude Haiku 4.5 via a Bedrock application inference profile,
    Wesley's production serving path (the profile ARN is withheld with the private
    Wesley record). The Wesley sweeps completed in July 2026; their rows are not
    redistributable and are not part of this public artifact.
- Decoding params (held equal across both providers): `max_tokens=2048`,
  `temperature=0`. `top_p` is NOT set (left at the provider default) on either side;
  Haiku 4.5 is a Claude 4+ model and rejects temperature and top_p together. Applied
  centrally in `sendModel` (`src/common/model.ts`).
- Synthetic server: `synthetic/server.py` + committed `synthetic/graph.json`
  (version 1, 64-node chain, closed-form deterministic). The state published here is
  the recorded state of the artifact at release.
- Client round cap: `WESLEY_EXPERIMENT_MAX_ROUNDS`, default 20 (max task depth is 8).
- TS runner: `tsx` (ts-node does not resolve the clients' explicit `.ts` import
  extensions on Node 22). Prompt passed via env (`WESLEY_EXPERIMENT_PROMPT`), not argv.
- Runs per cell: 8 repeats.
- Synthetic sweep (`synthetic_sweep.jsonl` + `synthetic_sweep_raw.jsonl`): 320 rows,
  the full cache 2x2; the cache=off half is the canonical uncached sweep of 160 runs,
  155/160 oracle success, est cost ~$1.85. RQ1 = D{1,2,4,8} x K{3,25} x
  {standard, codemode} x 8; RQ2 = value_sum at D=4, coarse vs fine, K=3, x8. All 5
  failures are standard MCP at D=8 (off by exactly one hop's value, 37); Code Mode 80/80.
  Anthropic client `maxRetries=6` so transient 5xx do not fail a run (zero API errors
  this run). The raw file carries each run's final_answer, generated code, and tool
  traces (RQ3 evidence).
- Date of run: 2026-06-19.

## Data dictionary (fill on first run)

The synthetic sweep writes one JSONL file. One row per client run, produced by
`synthetic/run_sweep.py` (`Row` dataclass). RQ1 and RQ2 rows share the schema and
are distinguished by the `rq` column; the default run writes both to
`synthetic_sweep.jsonl`.

- `synthetic_sweep.jsonl`: one row per (rq, arm, surface, depth, tool_count, cache,
  repeat). Columns:
  - `rq` (rq1 | rq2), `arm` (standard | codemode), `surface` (coarse | fine),
  - `depth` (D, dependent hops), `tool_count` (K, advertised tools),
  - `cache` (on | off), `repeat`,
  - `task_type` (final_value | value_sum), `task_id`, `start_id`,
  - `prompt_tokens` (UNCACHED input tokens; with cache=on, cached tokens move out of
    this into the two fields below), `completion_tokens`, `total_tokens` (all processed
    tokens = uncached input + cache write + cache read + output),
  - `cache_creation_tokens`, `cache_read_tokens` (prompt-caching buckets; 0 when
    cache=off). See effective-cost note below.
  - `reasoning_turns` (model rounds), `tool_calls` (MCP tool invocations),
  - `latency_total_ms` (model loop, excludes setup), `setup_ms` (connect + list),
  - `parsed_answer` (the oracle value when correct, else the best-effort extracted
    integer so a wrong answer still shows what the model said), `expected_answer` (oracle),
  - `success` / `answer_correct` (the oracle integer appears as a standalone token in
    the final answer; this presence rule is robust to prose answers that state the
    answer then list component values, which the older last-integer rule mis-scored),
    `code_executed` (codemode only:
    code ran without error; null for standard),
  - `error` (client error message or null), `notes`.

### Caching condition and effective cost

The sweep is a full 2x2: `cache` in {off, on} x {standard, codemode}, both joined on
(rq, arm, surface, depth, tool_count, repeat). cache=off is the canonical uncached
sweep (155/160); cache=on adds manual `cache_control` breakpoints on the Anthropic
path (one on the last system block, caching tools+system; one incremental on the
latest message). Caching is applied SYMMETRICALLY to both arms.

- **Effective cost is the comparison metric, not raw `total_tokens`.** With caching,
  `total_tokens` barely changes (the same tokens are still sent; they are re-priced).
  Compute effective input units per run as
  `prompt_tokens*1.0 + cache_creation_tokens*1.25 + cache_read_tokens*0.1`
  (5-minute-TTL multipliers), then dollars = effective_input/1e6*$1 + completion/1e6*$5.
- **Haiku 4.5's minimum cacheable prefix is 4096 tokens**, so caching is a NO-OP where
  the prefix is smaller: at K=3 (surface ~360 tok) both arms show cache_*=0 at every
  depth; at K=25 standard, caching engages only once conversation history pushes the
  prefix past 4096 (deep cells), not at low depth. This is reported behavior, not a bug.
- **Cold vs warm within a cell:** repeats of a cell run within the 5-min TTL, so the
  first repeat (repeat 0) pays the cache write (cold) and later repeats may read the
  shared prefix (warm). Report cold-start as repeat 0 and amortized as the mean over
  repeats; do not relabel measured reads as writes. The per-run buckets are recorded
  as measured.
- `synthetic_sweep_raw.jsonl`: the verbatim per-run evidence behind every scored row
  (same coordinates plus `prompt` and the full client `result`: `final_answer`,
  `generated_code_blocks` (codemode), `tool_traces`, `reasoning_turns`, token totals,
  `error`, and the cache buckets in `result.totals`). This is the integrity-grade raw
  output and the source for RQ3 qualitative categorization (e.g. quoting the generated
  code that produced a wrong result). Join to the scored rows on
  (rq, arm, surface, depth, tool_count, cache, repeat).
- `figures/`: plots, each regenerable by `scripts/plot.py` (needs matplotlib; see
  `synthetic/requirements.txt`). Current figures: `fig_turns_vs_depth.png` (model
  round trips vs depth, per arm, K=3 and K=25), `fig_effinput_vs_depth.png` (effective
  token cost vs depth, cache off), `fig_latency_vs_depth.png` (latency vs depth),
  `fig_caching_effect_k25.png` (effective input off vs on, all four arm/cache lines),
  `fig_success_k25.png` (success rate vs depth).
- `scripts/`: aggregation and plotting scripts (the only way figures/tables are made).
  `scripts/aggregate.py` is the sanctioned raw-to-reported path: it reads
  `synthetic_sweep.jsonl` and writes `aggregates.txt` (committed, regenerable) with
  the RQ1/RQ2/RQ3 tables, the caching effect, and the within-conversation cache
  decomposition. `scripts/plot.py` reads the same file and writes the `figures/`. Every
  reported number must trace to a table aggregate.py produces.

## Findings and caveats (see `aggregates.txt` for the numbers)

- **RQ1 (cache=off):** Code Mode is flat across depth on turns (2), effective input,
  and latency (~4s); standard scales on all three (D8/K25: 9.5 turns, 41.5k effective
  input, 12.1s). Both arms make the same number of MCP calls (~9 at D8), so Code Mode
  removes model round-trips, not tool calls.
- **Caching (RQ1 K=25):** floor-bounded. Standard reclaim only at depth (D4 +26%,
  D8 +51% effective input); no-op at D1 and at all of K=3 (surface under Haiku's 4096
  floor). Caching mildly PENALIZES Code Mode at D4/D8 (-13%: it pays the 1.25x write
  premium on the ~4.3k inlined API but ~2 rounds cannot recoup it). The D1/D2 codemode
  effects are within noise (CV 21-31%) and are not claimed. The +51% at D8/K25 is a
  per-request (cold-start) figure, not a warm-cache artifact (cold ~ amortized; the
  caching is within-conversation, not cross-repeat).
- **RQ2 (D4, coarse vs fine, K=3):** supports Code Mode granularity-invariance (2
  turns either way); does NOT support a standard-side granularity cost at K=3 (fine
  raises tool calls 5->9 without raising standard's turns/tokens). One point only.
- **RQ3:** under-powered here, and now ZERO Code Mode failures. Code Mode is 160/160
  (code_executed true 160/160, executed-but-wrong 0/160). The single apparent
  executed-but-wrong case was traced to a scoring bug, not a Code Mode failure: the
  model summed correctly (answered 1535) then listed the visited node values, and the
  old "last integer" extractor grabbed the trailing node value (381). The scorer now
  credits a run when the oracle integer appears as a standalone token in the final
  answer; re-grading from the saved raw outputs flipped exactly that one row. Standard
  MCP's only failure is a consistent off-by-one at D8 (all 10 wrong by -37, one hop
  short, identical off vs on). So the synthetic ground gives NO Code Mode reliability
  signal at all; RQ3 reliability is entirely on Wesley.
- **Bias direction:** localhost MCP makes in-code tool calls nearly free (inflates
  Code Mode's latency lead); the trivial chain hides Code Mode code-gen failure; the
  synthetic chain is the maximally amortizable shape. Synthetic magnitudes are an
  UPPER BOUND on Code Mode's advantage; Wesley carries realistic magnitude.

## Rules

- Raw outputs are committed, not summarized away. Aggregations are scripted.
- Figures regenerate from committed scripts if data changes.
- No file here is edited to make a result look better. Anomalies get a note.
