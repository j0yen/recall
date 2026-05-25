#!/usr/bin/env bash
# recall PostToolUseFailure hook — feed tool errors to `recall observe`.
#
# Claude Code delivers the hook payload as one JSON object on stdin. The
# observer expects `{tool_name, tool_input, tool_response, status,
# user_prompt_after}`. We translate from the harness's wire format, mark
# `status: error` (this hook only fires on failure), and leave
# `user_prompt_after` empty.
#
# CAVEAT: with `user_prompt_after` empty the v0.4 heuristic catalog will
# never propose. Wiring this in is the structural half; the functional
# half is a UserPromptSubmit correlator that pairs the most recent error
# with the next user prompt. Tracked as v0.4.1.
#
# Silent on any failure — never block a Claude turn on telemetry.

set -uo pipefail
exec 2>/dev/null

RECALL_BIN="${RECALL_BIN:-$HOME/.local/bin/recall}"
JQ="${JQ:-/usr/sbin/jq}"
[ -x "$RECALL_BIN" ] || exit 0
[ -x "$JQ" ] || exit 0

raw="$(cat -)"
[ -z "$raw" ] && exit 0

event="$("$JQ" -cn \
  --arg name "$("$JQ" -r '.tool_name // empty' <<<"$raw")" \
  --argjson input "$("$JQ" '.tool_input // {}' <<<"$raw")" \
  --argjson resp  "$("$JQ" '.tool_response // {}' <<<"$raw")" \
  '{tool_name:$name, tool_input:$input, tool_response:$resp, status:"error"}')"
[ -z "$event" ] && exit 0

printf '%s\n' "$event" | "$RECALL_BIN" observe --format text >/dev/null || true
exit 0
