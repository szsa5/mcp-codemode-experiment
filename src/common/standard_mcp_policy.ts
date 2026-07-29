const UI_ONLY_TOOL_NAMES = new Set(["attach_buttons"]);

export function shouldAppendFinalAnswerCue(toolNames: string[]): boolean {
  return toolNames.length > 0 && toolNames.every((toolName) => UI_ONLY_TOOL_NAMES.has(toolName));
}

export function buildFinalAnswerCue(): string {
  return "Now provide the final textual answer to the user.";
}
