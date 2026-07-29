import { performance } from "node:perf_hooks";

import { createModelClient, sendModel } from "../common/model.ts";
import { getMaxRounds, getModelId, getSystemPrompt } from "../common/config.ts";
import { emitJsonMarker, flushAndExit, previewText, createRunResult } from "../common/logging.ts";
import { callMcpTool, connectMcpClient, listMcpTools } from "../common/mcp.ts";
import { buildFinalAnswerCue, shouldAppendFinalAnswerCue } from "../common/standard_mcp_policy.ts";
import type { ToolTrace } from "../common/types.ts";

function toBedrockTools(mcpTools: Awaited<ReturnType<typeof listMcpTools>>) {
  return mcpTools.map((tool) => ({
    toolSpec: {
      name: tool.name,
      description: tool.description,
      inputSchema: {
        json: tool.inputSchema,
      },
    },
  })) as any;
}

function getTextFromContent(content: any[] | undefined): string {
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .filter((block) => typeof block?.text === "string")
    .map((block) => block.text)
    .join("\n")
    .trim();
}

async function main() {
  // Prefer the prompt from the environment (WESLEY_EXPERIMENT_PROMPT): a
  // multi-line prompt passed as a command-line argument is mangled by cmd.exe on
  // Windows, whereas the environment block preserves newlines. Fall back to argv
  // for manual `npm run` invocations.
  const prompt = (process.env.WESLEY_EXPERIMENT_PROMPT || process.argv.slice(2).join(" ")).trim();
  if (!prompt) {
    throw new Error("Prompt is required (set WESLEY_EXPERIMENT_PROMPT or pass an argument)");
  }

  let finalAnswer = "";
  let promptTokens = 0;
  let completionTokens = 0;
  let totalTokens = 0;
  let cacheCreationTokens = 0;
  let cacheReadTokens = 0;
  let setupDurationMs = 0;
  let durationMs = 0;
  let modelId = "unknown";
  const toolTraces: ToolTrace[] = [];

  const setupStartedAt = performance.now();
  console.error("[standard_mcp] connecting to MCP server");
  const mcpClient = await connectMcpClient();
  console.error("[standard_mcp] connected to MCP server");
  console.error("[standard_mcp] listing MCP tools");
  const mcpTools = await listMcpTools(mcpClient);
  console.error(`[standard_mcp] listed ${mcpTools.length} MCP tools`);
  const bedrockTools = toBedrockTools(mcpTools);
  console.error("[standard_mcp] creating model client");
  const modelClient = createModelClient();
  setupDurationMs = Math.round(performance.now() - setupStartedAt);
  modelId = getModelId();

  const messages: any[] = [{ role: "user", content: [{ text: prompt }] }];
  let roundIndex = 0;
  let reasoningTurns = 0;
  let executedToolCount = 0;

  emitJsonMarker("STANDARD_MCP_TRANSLATION_JSON", {
    translated_tool_count: bedrockTools.length,
    executed_tool_count: 0,
  });

  const startedAt = performance.now();
  const maxRounds = getMaxRounds();

  while (roundIndex < maxRounds) {
    reasoningTurns += 1;
    const bedrockStartedAt = performance.now();
    console.error(`[standard_mcp] round ${roundIndex} sending model request`);
    const response = await sendModel(modelClient, {
      system: [{ text: getSystemPrompt() }],
      messages,
      toolConfig: { tools: bedrockTools },
      inferenceConfig: { maxTokens: 2048 },
    });
    const bedrockDurationMs = Math.round(performance.now() - bedrockStartedAt);
    console.error(`[standard_mcp] round ${roundIndex} received model response in ${bedrockDurationMs} ms`);

    const outputMessage = response.output?.message;
    const content = outputMessage?.content || [];
    const usage: any = response.usage;
    promptTokens += usage?.inputTokens || 0;
    completionTokens += usage?.outputTokens || 0;
    totalTokens += usage?.totalTokens || 0;
    // Cache buckets: anthropic-normalized names, falling back to Bedrock's
    // cacheWrite/cacheReadInputTokens for the (future) Bedrock caching path.
    cacheCreationTokens += usage?.cacheCreationTokens ?? usage?.cacheWriteInputTokens ?? usage?.cacheCreationInputTokens ?? 0;
    cacheReadTokens += usage?.cacheReadTokens ?? usage?.cacheReadInputTokens ?? 0;

    const toolBlocks = content.filter((block: any) => block?.toolUse);
    if (toolBlocks.length === 0) {
      console.error(`[standard_mcp] round ${roundIndex} no tool calls, finishing`);
      finalAnswer = getTextFromContent(content);
      if (!finalAnswer && executedToolCount > 0) {
        const error = new Error("Standard MCP client executed tools but did not receive a final textual answer");
        (error as any).runState = {
          finalAnswer,
          toolTraces,
          reasoningTurns,
          promptTokens,
          completionTokens,
          totalTokens,
          cacheCreationTokens,
          cacheReadTokens,
          durationMs: Math.round(performance.now() - startedAt),
          setupDurationMs,
          modelId,
        };
        throw error;
      }
      break;
    }

    messages.push({ role: "assistant", content });
    console.error(`[standard_mcp] round ${roundIndex} executing ${toolBlocks.length} tool calls`);

    const toolResults = [];
    const roundToolNames: string[] = [];
    for (const block of toolBlocks) {
      const toolUse = block.toolUse as {
        name?: string;
        input?: Record<string, unknown>;
        toolUseId?: string;
      };
      const toolName = toolUse.name || "";
      const toolInput = toolUse.input || {};
      const toolUseId = toolUse.toolUseId || "";
      const toolStartedAt = performance.now();
      console.error(`[standard_mcp] calling MCP tool ${toolName}`);
      roundToolNames.push(toolName);
      const resultText = await callMcpTool(
        mcpClient,
        toolName,
        toolInput,
      );
      console.error(`[standard_mcp] MCP tool ${toolName} returned ${resultText.length} chars`);
      const durationMs = Math.round(performance.now() - toolStartedAt);
      executedToolCount += 1;
      toolTraces.push({
        round_index: roundIndex,
        tool_name: toolName,
        arguments: toolInput,
        duration_ms: durationMs,
        result_preview: previewText(resultText),
        result_chars: resultText.length,
      });
      toolResults.push({
        toolResult: {
          toolUseId,
          content: [{ text: resultText }],
          status: "success",
        },
      });
    }

    emitJsonMarker("STANDARD_MCP_TRANSLATION_JSON", {
      translated_tool_count: bedrockTools.length,
      executed_tool_count: executedToolCount,
    });

    const nextUserContent: any[] = [...toolResults];
    if (shouldAppendFinalAnswerCue(roundToolNames)) {
      console.error(`[standard_mcp] round ${roundIndex} appending final-answer cue after UI-only tools`);
      nextUserContent.push({ text: buildFinalAnswerCue() });
    }
    messages.push({ role: "user", content: nextUserContent });
    roundIndex += 1;
  }

  durationMs = Math.round(performance.now() - startedAt);
  console.error("[standard_mcp] emitting RUN_RESULT_JSON");
  const runResult = createRunResult({
    arm: "standard_mcp",
    model: modelId,
    final_answer: finalAnswer,
    tool_traces: toolTraces,
    reasoning_turns: reasoningTurns,
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: totalTokens,
    cache_creation_tokens: cacheCreationTokens,
    cache_read_tokens: cacheReadTokens,
    duration_ms: durationMs,
    setup_duration_ms: setupDurationMs,
    error: null,
  });
  emitJsonMarker("RUN_RESULT_JSON", runResult);
}

main()
  .then(() => flushAndExit(0))
  .catch((error) => {
  const details = (error as any)?.runState as
    | {
        finalAnswer?: string;
        toolTraces?: ToolTrace[];
        reasoningTurns?: number;
        promptTokens?: number;
        completionTokens?: number;
        totalTokens?: number;
        cacheCreationTokens?: number;
        cacheReadTokens?: number;
        durationMs?: number;
        setupDurationMs?: number;
        modelId?: string;
      }
    | undefined;
  const runResult = createRunResult({
    arm: "standard_mcp",
    model: details?.modelId || process.env.WESLEY_EXPERIMENT_MODEL_ID || "unknown",
    final_answer: details?.finalAnswer || "",
    tool_traces: details?.toolTraces || [],
    reasoning_turns: details?.reasoningTurns || 0,
    prompt_tokens: details?.promptTokens || 0,
    completion_tokens: details?.completionTokens || 0,
    total_tokens: details?.totalTokens || 0,
    cache_creation_tokens: details?.cacheCreationTokens || 0,
    cache_read_tokens: details?.cacheReadTokens || 0,
    duration_ms: details?.durationMs || 0,
    setup_duration_ms: details?.setupDurationMs || 0,
    error: error instanceof Error ? error.message : String(error),
  });
  emitJsonMarker("RUN_RESULT_JSON", runResult);
  flushAndExit(1);
  });
