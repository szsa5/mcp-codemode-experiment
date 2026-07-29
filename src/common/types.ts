export type TokenTotals = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  // Prompt-caching buckets (0 when caching is off). prompt_tokens is the uncached
  // input remainder; effective cost = prompt_tokens + cache_creation*1.25 +
  // cache_read*0.1 (5-min TTL multipliers).
  cache_creation_tokens: number;
  cache_read_tokens: number;
};

export type ToolTrace = {
  round_index: number;
  tool_name: string;
  arguments: Record<string, unknown>;
  duration_ms: number;
  result_preview: string;
  result_chars: number;
};

export type ExperimentRunResult = {
  arm: "standard_mcp" | "codemode_mcp";
  model: string;
  final_answer: string;
  tool_traces: ToolTrace[];
  generated_code_blocks?: string[];
  // Code Mode only: how many generated-code blocks failed to execute (syntax/runtime
  // error) and were fed back to the model for a retry. A direct RQ3 reliability signal.
  code_exec_errors?: number;
  reasoning_turns: number;
  totals: TokenTotals;
  duration_ms: number;
  setup_duration_ms: number;
  error: string | null;
};
