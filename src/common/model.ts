import Anthropic from "@anthropic-ai/sdk";

import { createBedrockClient, sendConverse } from "./bedrock.ts";
import { getCacheMode, getModelId, getModelProvider, getTemperature, requireEnv } from "./config.ts";

// Provider abstraction over the model-call layer.
//
// Both grounds run the same Claude Haiku 4.5 snapshot, but on different serving
// platforms (Option 1, provider per ground): the synthetic ground on the
// Anthropic Messages API, Wesley on Bedrock Converse. The two clients
// (standard_mcp, codemode) speak one canonical request/response shape, the
// Bedrock Converse shape, and this module routes it to the selected provider.
// For the Anthropic path it translates that canonical shape to/from the Messages
// API. Only the model-call layer differs across providers; the SSE/MCP transport,
// tool-spec translation, Code Mode executor, and instrumentation are shared.

const DEFAULT_MAX_TOKENS = 2048;

type CanonicalBlock = {
  text?: string;
  toolUse?: { name?: string; input?: unknown; toolUseId?: string };
  toolResult?: {
    toolUseId?: string;
    content?: Array<{ text?: string }>;
    status?: string;
  };
};

type CanonicalMessage = { role: "user" | "assistant"; content: CanonicalBlock[] };

type CanonicalRequest = {
  system?: Array<{ text?: string }>;
  messages: CanonicalMessage[];
  toolConfig?: { tools?: Array<{ toolSpec?: { name?: string; description?: string; inputSchema?: { json?: unknown } } }> };
  inferenceConfig?: { maxTokens?: number };
};

// Normalized response in the Bedrock Converse shape the clients already read.
// cacheCreationTokens / cacheReadTokens are the prompt-caching buckets (0 when
// caching is off). totalTokens counts every processed token (uncached input +
// cache write + cache read + output) so it stays meaningful under caching.
type CanonicalResponse = {
  output: { message: { role: "assistant"; content: CanonicalBlock[] } };
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
    cacheCreationTokens: number;
    cacheReadTokens: number;
  };
  stopReason?: string | null;
};

export type ModelClient =
  | { provider: "bedrock"; bedrock: ReturnType<typeof createBedrockClient> }
  | { provider: "anthropic"; anthropic: Anthropic };

export function createModelClient(): ModelClient {
  const provider = getModelProvider();
  if (provider === "anthropic") {
    const apiKey = requireEnv("ANTHROPIC_API_KEY");
    // Raise maxRetries above the default 2 so transient 5xx/429 from the API are
    // ridden out instead of failing a run (one run in the first sweep died on a
    // 500). The SDK only retries on retryable errors, so this costs nothing on the
    // happy path.
    return { provider, anthropic: new Anthropic({ apiKey, maxRetries: 6 }) };
  }
  return { provider: "bedrock", bedrock: createBedrockClient() };
}

// --- Canonical (Bedrock) -> Anthropic Messages translation ---------------------

function toAnthropicTools(request: CanonicalRequest): any[] {
  return (request.toolConfig?.tools || []).map((tool) => ({
    name: tool.toolSpec?.name,
    description: tool.toolSpec?.description,
    input_schema: tool.toolSpec?.inputSchema?.json,
  }));
}

function toAnthropicSystem(request: CanonicalRequest): string {
  return (request.system || [])
    .map((block) => block.text || "")
    .filter((text) => text.length > 0)
    .join("\n\n");
}

function toAnthropicMessages(messages: CanonicalMessage[]): any[] {
  return messages.map((message) => {
    const content: any[] = [];
    for (const block of message.content) {
      if (typeof block.text === "string") {
        // Drop empty text blocks; the Anthropic API rejects them and Bedrock
        // sometimes emits an empty assistant text block alongside a tool call.
        if (block.text.trim().length > 0) {
          content.push({ type: "text", text: block.text });
        }
      } else if (block.toolUse) {
        content.push({
          type: "tool_use",
          id: block.toolUse.toolUseId,
          name: block.toolUse.name,
          input: block.toolUse.input ?? {},
        });
      } else if (block.toolResult) {
        content.push({
          type: "tool_result",
          tool_use_id: block.toolResult.toolUseId,
          content: (block.toolResult.content || []).map((item) => ({
            type: "text",
            text: item.text ?? "",
          })),
          is_error: block.toolResult.status === "error",
        });
      }
    }
    return { role: message.role, content };
  });
}

// --- Anthropic Messages response -> canonical (Bedrock) ------------------------

function fromAnthropicResponse(response: any): CanonicalResponse {
  const blocks: CanonicalBlock[] = [];
  for (const block of response?.content || []) {
    if (block?.type === "text") {
      blocks.push({ text: block.text });
    } else if (block?.type === "tool_use") {
      blocks.push({
        toolUse: { toolUseId: block.id, name: block.name, input: block.input },
      });
    }
    // Other block types (e.g. thinking) are not used by this experiment.
  }
  const inputTokens = response?.usage?.input_tokens || 0;
  const outputTokens = response?.usage?.output_tokens || 0;
  const cacheCreationTokens = response?.usage?.cache_creation_input_tokens || 0;
  const cacheReadTokens = response?.usage?.cache_read_input_tokens || 0;
  return {
    output: { message: { role: "assistant", content: blocks } },
    usage: {
      inputTokens,
      outputTokens,
      totalTokens: inputTokens + cacheCreationTokens + cacheReadTokens + outputTokens,
      cacheCreationTokens,
      cacheReadTokens,
    },
    stopReason: response?.stop_reason ?? null,
  };
}

// --- Unified send --------------------------------------------------------------

export async function sendModel(client: ModelClient, request: CanonicalRequest) {
  const maxTokens = request.inferenceConfig?.maxTokens ?? DEFAULT_MAX_TOKENS;
  const temperature = getTemperature();

  if (client.provider === "bedrock") {
    // Decoding params are injected here so both providers use identical values.
    // Prompt caching (Bedrock cachePoint). Converse caches in the order tools ->
    // system -> messages, so one cachePoint at the end of `system` caches the re-sent
    // tools + system prefix, and one at the end of the latest message caches the
    // growing conversation, mirroring the Anthropic path's two breakpoints. Haiku
    // 4.5's minimum cacheable prefix (4096 tokens) still applies, so a small
    // tools+system prefix silently does not cache and the benefit appears once the
    // conversation grows past the floor (the floor-bounded finding, Bedrock side).
    const cacheOn = getCacheMode() === "on";
    let system = request.system;
    let messages = request.messages;
    if (cacheOn) {
      if (system && system.length > 0) {
        system = [...system, { cachePoint: { type: "default" } } as any];
      }
      if (messages.length > 0) {
        const last = messages[messages.length - 1];
        messages = [
          ...messages.slice(0, -1),
          { ...last, content: [...last.content, { cachePoint: { type: "default" } } as any] },
        ];
      }
    }
    return await sendConverse(client.bedrock, {
      system,
      messages,
      toolConfig: request.toolConfig,
      inferenceConfig: { maxTokens, temperature },
    } as any);
  }

  // Haiku 4.5 is a Claude 4+ model: temperature and top_p cannot both be set, so
  // only temperature is sent (top_p left at the provider default on both sides).
  const cacheOn = getCacheMode() === "on";
  const systemText = toAnthropicSystem(request);
  const messages = toAnthropicMessages(request.messages);

  // Manual prompt caching (Anthropic path only). Render order is tools -> system
  // -> messages, so one cache_control breakpoint on the last system block caches
  // tools + system together (the re-sent tool surface). A second breakpoint on the
  // last block of the latest message caches the growing conversation incrementally.
  // NOTE: Haiku 4.5's minimum cacheable prefix is 4096 tokens, so caching only
  // engages once the prefix exceeds that (e.g. high K and/or deep conversations);
  // smaller prefixes silently do not cache.
  const system: any = cacheOn
    ? [{ type: "text", text: systemText, cache_control: { type: "ephemeral" } }]
    : systemText;
  if (cacheOn && messages.length > 0) {
    const lastContent = messages[messages.length - 1].content;
    if (Array.isArray(lastContent) && lastContent.length > 0) {
      lastContent[lastContent.length - 1].cache_control = { type: "ephemeral" };
    }
  }

  const response = await client.anthropic.messages.create({
    model: getModelId(),
    max_tokens: maxTokens,
    temperature,
    system,
    tools: toAnthropicTools(request),
    messages,
  } as any);

  return fromAnthropicResponse(response);
}
