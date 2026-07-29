export function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

export function getModelId(): string {
  return requireEnv("WESLEY_EXPERIMENT_MODEL_ID");
}

export function getModelProvider(): "bedrock" | "anthropic" {
  const raw = (process.env.MODEL_PROVIDER || "bedrock").trim().toLowerCase();
  if (raw !== "bedrock" && raw !== "anthropic") {
    throw new Error(`Invalid MODEL_PROVIDER (expected bedrock|anthropic): ${raw}`);
  }
  return raw;
}

export function getTemperature(): number {
  const raw = process.env.WESLEY_EXPERIMENT_TEMPERATURE || "0";
  const value = Number.parseFloat(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Invalid WESLEY_EXPERIMENT_TEMPERATURE: ${raw}`);
  }
  return value;
}

export function getSystemPrompt(): string {
  return requireEnv("WESLEY_EXPERIMENT_SYSTEM_PROMPT");
}

export function getCacheMode(): "on" | "off" {
  const raw = (process.env.CACHE || "off").trim().toLowerCase();
  if (raw !== "on" && raw !== "off") {
    throw new Error(`Invalid CACHE (expected on|off): ${raw}`);
  }
  return raw;
}

export function getLanguage(): string {
  return process.env.WESLEY_EXPERIMENT_LANGUAGE || "en";
}

export function getMcpRequestTimeoutMs(): number {
  const raw = process.env.WESLEY_MCP_REQUEST_TIMEOUT_MS || "180000";
  const value = Number.parseInt(raw, 10);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Invalid WESLEY_MCP_REQUEST_TIMEOUT_MS: ${raw}`);
  }
  return value;
}

export function getMaxRounds(): number {
  const raw = process.env.WESLEY_EXPERIMENT_MAX_ROUNDS || "20";
  const value = Number.parseInt(raw, 10);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Invalid WESLEY_EXPERIMENT_MAX_ROUNDS: ${raw}`);
  }
  return value;
}

export function getBedrockRoundTimeoutMs(): number {
  const raw = process.env.WESLEY_BEDROCK_ROUND_TIMEOUT_MS || "120000";
  const value = Number.parseInt(raw, 10);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Invalid WESLEY_BEDROCK_ROUND_TIMEOUT_MS: ${raw}`);
  }
  return value;
}
