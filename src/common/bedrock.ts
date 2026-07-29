import {
  BedrockRuntimeClient,
  ConverseCommand,
  type ConverseCommandInput,
} from "@aws-sdk/client-bedrock-runtime";

import { getBedrockRoundTimeoutMs, getModelId, requireEnv } from "./config.ts";

export function createBedrockClient(): BedrockRuntimeClient {
  return new BedrockRuntimeClient({
    region: process.env.AWS_BEDROCK_REGION || "eu-central-1",
    credentials: {
      accessKeyId: requireEnv("AWS_BEDROCK_ID"),
      secretAccessKey: requireEnv("AWS_BEDROCK_SECRET"),
    },
  });
}

export async function sendConverse(
  client: BedrockRuntimeClient,
  input: Omit<ConverseCommandInput, "modelId">,
) {
  const timeoutMs = getBedrockRoundTimeoutMs();
  return await Promise.race([
    client.send(
      new ConverseCommand({
        ...input,
        modelId: getModelId(),
      }),
    ),
    new Promise<never>((_, reject) => {
      const handle = setTimeout(() => {
        reject(new Error(`Bedrock Converse timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      handle.unref?.();
    }),
  ]);
}
