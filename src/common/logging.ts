import type { ExperimentRunResult, ToolTrace } from "./types.ts";

export function emitJsonMarker(label: string, payload: unknown) {
  console.log(`${label}:${JSON.stringify(payload)}`);
}

// One-shot CLI clients: the MCP SSE connection and the model SDK's keep-alive
// sockets keep the Node event loop alive after main() resolves, so the process
// never exits on its own. Force exit once stdout has flushed the result marker
// (an empty write callback fires after all prior queued writes drain).
export function flushAndExit(code: number): void {
  process.stdout.write("", () => process.exit(code));
  // Backstop in case the drain callback does not fire.
  setTimeout(() => process.exit(code), 2000);
}

export function previewText(value: unknown, max = 200): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length <= max ? text : `${text.slice(0, max)}...`;
}

export function createRunResult(params: {
  arm: "standard_mcp" | "codemode_mcp";
  model: string;
  final_answer: string;
  tool_traces: ToolTrace[];
  generated_code_blocks?: string[];
  code_exec_errors?: number;
  reasoning_turns: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
  duration_ms: number;
  setup_duration_ms: number;
  error: string | null;
}): ExperimentRunResult {
  return {
    arm: params.arm,
    model: params.model,
    final_answer: params.final_answer,
    tool_traces: params.tool_traces,
    generated_code_blocks: params.generated_code_blocks,
    code_exec_errors: params.code_exec_errors,
    reasoning_turns: params.reasoning_turns,
    totals: {
      prompt_tokens: params.prompt_tokens,
      completion_tokens: params.completion_tokens,
      total_tokens: params.total_tokens,
      cache_creation_tokens: params.cache_creation_tokens,
      cache_read_tokens: params.cache_read_tokens,
    },
    duration_ms: params.duration_ms,
    setup_duration_ms: params.setup_duration_ms,
    error: params.error,
  };
}
