import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

import { getMcpRequestTimeoutMs, requireEnv } from "./config.ts";

type MappedTool = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
};

export async function connectMcpClient() {
  const url = requireEnv("WESLEY_MCP_URL");
  const transport = new SSEClientTransport(new URL(url));
  const client = new Client(
    { name: "wesley-experiment-client", version: "1.0.0" },
    { capabilities: {} },
  );
  await client.connect(transport);
  return client;
}

export async function listMcpTools(client: any): Promise<MappedTool[]> {
  const result = await client.listTools(undefined, { timeout: getMcpRequestTimeoutMs() });
  const tools = result?.tools || [];
  return tools.map((tool: any) => ({
    name: tool.name,
    description: tool.description || "",
    inputSchema: tool.inputSchema || { type: "object", properties: {}, required: [] },
  }));
}

export async function callMcpTool(
  client: any,
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  const result = await client.callTool({
    name,
    arguments: args,
  }, undefined, { timeout: getMcpRequestTimeoutMs() });
  const content = result?.content || [];
  const textParts = content
    .map((item: any) => (typeof item?.text === "string" ? item.text : ""))
    .filter(Boolean);
  return textParts.join("\n");
}
