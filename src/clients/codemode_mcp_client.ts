import { performance } from "node:perf_hooks";

import { createCodeTool } from "@cloudflare/codemode/ai";
import type { Executor, ExecuteResult, ToolDescriptors } from "@cloudflare/codemode";
import { z, type ZodType } from "zod";

import { createModelClient, sendModel } from "../common/model.ts";
import { getMaxRounds, getModelId, getSystemPrompt } from "../common/config.ts";
import { emitJsonMarker, flushAndExit, previewText, createRunResult } from "../common/logging.ts";
import { callMcpTool, connectMcpClient, listMcpTools } from "../common/mcp.ts";
import { buildFinalAnswerCue, shouldAppendFinalAnswerCue } from "../common/standard_mcp_policy.ts";
import type { ToolTrace } from "../common/types.ts";

class LocalExecutor implements Executor {
  async execute(
    code: string,
    fns: Record<string, (...args: unknown[]) => Promise<unknown>>,
  ): Promise<ExecuteResult> {
    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor as new (
      ...args: string[]
    ) => (...innerArgs: unknown[]) => Promise<unknown>;
    const logs: string[] = [];
    const consoleProxy = {
      log: (...args: unknown[]) => logs.push(args.map(String).join(" ")),
      warn: (...args: unknown[]) => logs.push(`[warn] ${args.map(String).join(" ")}`),
      error: (...args: unknown[]) => logs.push(`[error] ${args.map(String).join(" ")}`),
    };
    const executedTools: string[] = [];
    const codemode = new Proxy(
      {},
      {
        get(_target, property) {
          return async (args: unknown) => {
            const fn = fns[String(property)];
            if (!fn) {
              throw new Error(`Tool "${String(property)}" not found`);
            }
            executedTools.push(String(property));
            return fn(args ?? {});
          };
        },
      },
    );

    try {
      const fn = new AsyncFunction("codemode", "console", `return await (${code})()`);
      const result = await fn(codemode, consoleProxy);
      return { result: { value: result, executedTools }, logs };
    } catch (error) {
      return {
        result: undefined,
        error: error instanceof Error ? error.message : String(error),
        logs,
      };
    }
  }
}

function jsonSchemaToZod(schema: Record<string, unknown> | undefined): ZodType {
  if (!schema || typeof schema !== "object") {
    return z.any();
  }

  // Preserve the param description and (for strings) the enum. `@cloudflare/codemode`
  // generates the TypeScript API the model writes against from these Zod types, rendering
  // `.describe()` as a JSDoc comment and `z.enum` as a string-literal union. Dropping them
  // (the earlier behavior) was a HARNESS ASYMMETRY: standard MCP sends the model the full
  // JSON schema (param descriptions + enums), so Code Mode must expose the same, or the two
  // arms see different tool documentation and the comparison is confounded.
  const description = typeof schema.description === "string" ? schema.description : undefined;
  const withDesc = (zt: ZodType): ZodType => (description ? zt.describe(description) : zt);

  const schemaType = schema.type;

  if (
    Array.isArray(schema.enum) &&
    schema.enum.length > 0 &&
    schema.enum.every((v) => typeof v === "string")
  ) {
    return withDesc(z.enum(schema.enum as [string, ...string[]]));
  }
  if (schemaType === "string") {
    return withDesc(z.string());
  }
  if (schemaType === "number") {
    return withDesc(z.number());
  }
  if (schemaType === "integer") {
    return withDesc(z.number().int());
  }
  if (schemaType === "boolean") {
    return withDesc(z.boolean());
  }
  if (schemaType === "array") {
    const itemSchema = jsonSchemaToZod((schema.items as Record<string, unknown>) || undefined);
    return withDesc(z.array(itemSchema));
  }
  if (schemaType === "object" || schema.properties) {
    const properties = (schema.properties as Record<string, Record<string, unknown>>) || {};
    const required = new Set(Array.isArray(schema.required) ? schema.required : []);
    const shape: Record<string, ZodType> = {};
    for (const [key, value] of Object.entries(properties)) {
      const propertySchema = jsonSchemaToZod(value);
      shape[key] = required.has(key) ? propertySchema : propertySchema.optional();
    }
    return withDesc(z.object(shape).passthrough());
  }
  return z.any();
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

function toToolResultText(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value);
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
  const generatedCodeBlocks: string[] = [];
  const setupStartedAt = performance.now();
  console.error("[codemode_mcp] connecting to MCP server");
  const mcpClient = await connectMcpClient();
  console.error("[codemode_mcp] connected to MCP server");
  console.error("[codemode_mcp] listing MCP tools");
  const mcpTools = await listMcpTools(mcpClient);
  console.error(`[codemode_mcp] listed ${mcpTools.length} MCP tools`);
  const toolTraces: ToolTrace[] = [];
  let currentRoundIndex = 0;

  const toolDescriptors: ToolDescriptors = Object.fromEntries(
    mcpTools.map((tool) => [
      tool.name,
      {
        description: tool.description,
        inputSchema: jsonSchemaToZod(tool.inputSchema),
        execute: async (args: unknown) => {
          const startedAt = performance.now();
          const resultText = await callMcpTool(
            mcpClient,
            tool.name,
            typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {},
          );
          const durationMs = Math.round(performance.now() - startedAt);
          toolTraces.push({
            round_index: currentRoundIndex,
            tool_name: tool.name,
            arguments:
              typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {},
            duration_ms: durationMs,
            result_preview: previewText(resultText),
            result_chars: resultText.length,
          });
          // Code Mode's contract is that tools return structured values to the
          // generated code (so it can navigate fields like result.next_id).
          // MCP tool output is text; deserialize JSON payloads so the code sees
          // an object/number, falling back to the raw string for non-JSON output.
          try {
            return JSON.parse(resultText);
          } catch {
            return resultText;
          }
        },
      },
    ]),
  );

  const codemodeTool = createCodeTool({
    tools: toolDescriptors,
    executor: new LocalExecutor(),
  }) as any;

  const codemodeExecute = codemodeTool.execute;
  if (typeof codemodeExecute !== "function") {
    throw new Error("createCodeTool() did not expose an executable tool");
  }

  console.error("[codemode_mcp] creating model client");
  const modelClient = createModelClient();
  setupDurationMs = Math.round(performance.now() - setupStartedAt);
  modelId = getModelId();

  const messages: any[] = [{ role: "user", content: [{ text: prompt }] }];
  let roundIndex = 0;
  let reasoningTurns = 0;
  let executedCodeModeCount = 0;
  let codeExecErrors = 0;

  const startedAt = performance.now();
  const maxRounds = getMaxRounds();

  while (roundIndex < maxRounds) {
    currentRoundIndex = roundIndex;
    reasoningTurns += 1;
    const bedrockStartedAt = performance.now();
    console.error(`[codemode_mcp] round ${roundIndex} sending model request`);
    const response = await sendModel(modelClient, {
      system: [
        {
          text:
            `${getSystemPrompt()}\n\n[Experiment harness rule]\nYou are running the Wesley Code Mode experiment client. Use the codemode tool when multi-step orchestration helps. After any tool use, you must always provide a final textual answer to the user.`,
        },
      ],
      messages,
      toolConfig: {
        tools: [
          {
            toolSpec: {
              name: "codemode",
              description: codemodeTool.description || "Execute code to orchestrate Wesley MCP tools",
              inputSchema: {
                json: {
                  type: "object",
                  properties: {
                    code: { type: "string" },
                  },
                  required: ["code"],
                },
              },
            },
          },
        ],
      },
      inferenceConfig: { maxTokens: 2048 },
    });
    const bedrockDurationMs = Math.round(performance.now() - bedrockStartedAt);
    console.error(`[codemode_mcp] round ${roundIndex} received model response in ${bedrockDurationMs} ms`);

    const outputMessage = response.output?.message;
    const content = outputMessage?.content || [];
    const usage: any = response.usage;
    promptTokens += usage?.inputTokens || 0;
    completionTokens += usage?.outputTokens || 0;
    totalTokens += usage?.totalTokens || 0;
    cacheCreationTokens += usage?.cacheCreationTokens ?? usage?.cacheWriteInputTokens ?? usage?.cacheCreationInputTokens ?? 0;
    cacheReadTokens += usage?.cacheReadTokens ?? usage?.cacheReadInputTokens ?? 0;

    const toolBlocks = content.filter((block: any) => block?.toolUse);
    if (toolBlocks.length === 0) {
      console.error(`[codemode_mcp] round ${roundIndex} no tool calls, finishing`);
      finalAnswer = getTextFromContent(content);
      if (!finalAnswer && executedCodeModeCount > 0) {
        const error = new Error("Code Mode MCP client executed tools but did not receive a final textual answer");
        (error as any).runState = {
          finalAnswer,
          toolTraces,
          generatedCodeBlocks,
          codeExecErrors,
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
    console.error(`[codemode_mcp] round ${roundIndex} executing ${toolBlocks.length} codemode calls`);

    const toolResults = [];
    const roundToolNames: string[] = [];
    for (const block of toolBlocks) {
      const toolUse = block.toolUse as {
        input?: Record<string, unknown>;
        toolUseId?: string;
      };
      const toolInput = toolUse.input || {};
      const code = typeof toolInput.code === "string" ? toolInput.code : "";
      generatedCodeBlocks.push(code);
      console.error(`[codemode_mcp] executing codemode tool with ${code.length} chars of code`);
      console.error(`[codemode_mcp] generated code:\n${code}`);
      const execution = await codemodeExecute({ code });
      executedCodeModeCount += 1;
      if (execution?.error) {
        // Generated code failed to execute (syntax/runtime error). Feed the error back
        // to the model as an ERROR tool-result so it can fix the code and retry, exactly
        // as the standard-MCP arm feeds a failed tool call back for a retry. Aborting the
        // run here (the earlier behavior) held Code Mode to a harsher standard than
        // standard MCP and discarded the run's instrumentation (zeroed tokens/turns). A
        // genuine "cannot get it right within the round cap" failure is still captured
        // (the loop runs to maxRounds with no final answer), which is real RQ3 signal;
        // codeExecErrors counts the failed attempts for the RQ3 reliability analysis.
        codeExecErrors += 1;
        console.error(`[codemode_mcp] round ${roundIndex} code execution error: ${execution.error}`);
        toolResults.push({
          toolResult: {
            toolUseId: toolUse.toolUseId || "",
            content: [{
              text: `Code execution failed: ${execution.error}\n` +
                `Fix the code and try again. Return an async arrow function with valid JavaScript.`,
            }],
            status: "error",
          },
        });
        continue;
      }
      const executedTools = Array.isArray(execution?.result?.executedTools)
        ? execution.result.executedTools
        : [];
      for (const toolName of executedTools) {
        if (typeof toolName === "string") {
          roundToolNames.push(toolName);
        }
      }
      const resultText = toToolResultText(execution?.result?.value);
      toolResults.push({
        toolResult: {
          toolUseId: toolUse.toolUseId || "",
          content: [{ text: resultText }],
          status: "success",
        },
      });
    }

    const nextUserContent: any[] = [...toolResults];
    if (shouldAppendFinalAnswerCue(roundToolNames)) {
      console.error(`[codemode_mcp] round ${roundIndex} appending final-answer cue after UI-only tools`);
      nextUserContent.push({ text: buildFinalAnswerCue() });
    }
    messages.push({ role: "user", content: nextUserContent });
    roundIndex += 1;
  }

  durationMs = Math.round(performance.now() - startedAt);
  console.error("[codemode_mcp] emitting RUN_RESULT_JSON");
  const runResult = createRunResult({
    arm: "codemode_mcp",
    model: modelId,
    final_answer: finalAnswer,
    tool_traces: toolTraces,
    generated_code_blocks: generatedCodeBlocks,
    code_exec_errors: codeExecErrors,
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
        generatedCodeBlocks?: string[];
        codeExecErrors?: number;
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
    arm: "codemode_mcp",
    model: details?.modelId || process.env.WESLEY_EXPERIMENT_MODEL_ID || "unknown",
    final_answer: details?.finalAnswer || "",
    tool_traces: details?.toolTraces || [],
    generated_code_blocks: details?.generatedCodeBlocks || [],
    code_exec_errors: details?.codeExecErrors || 0,
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
