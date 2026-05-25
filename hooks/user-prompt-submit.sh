#!/usr/bin/env bash
# recall UserPromptSubmit hook (v0.4.2, *braid* correlator) — pair the
# most recent error state from PostToolUseFailure with the prompt the
# user just typed, and feed the joined event to `recall observe`. The
# heuristic catalog needs `user_prompt_after` to fire; this is where
# we finally populate it.
#
# State source:
#   ~/.cache/recall-braid/<session-id>/last-error.json
#
# Read-then-delete pattern: the same error never pairs with two
# subsequent prompts. Freshness gate: errors older than 60s are
# treated as expired and discarded.
#
# session_id comes from the input JSON's `.session_id` field (what the
# harness actually passes); v0.4.1 read it from $CLAUDE_SESSION_ID
# which the harness does not export — silent total no-op. v0.4.2 reads
# JSON first, env as fallback.
#
# This hook runs on the latency-sensitive UserPromptSubmit path.
# Total wall-budget target: < 200ms. Failure here must never block
# the prompt — silent exit on every error path.

set -uo pipefail
exec 2>/dev/null

RECALL_BIN="${RECALL_BIN:-$HOME/.local/bin/recall}"
JQ="${JQ:-/usr/sbin/jq}"
MAX_AGE_SEC="${RECALL_BRAID_MAX_AGE:-60}"

[ -x "$RECALL_BIN" ] || exit 0
[ -x "$JQ" ] || exit 0

# Read the harness payload (the prompt) before we touch state, so a
# parse error doesn't leave stale state behind.
raw="$(cat -)"

SESSION_ID="$("$JQ" -r '.session_id // empty' <<<"$raw" 2>/dev/null)"
SESSION_ID="${SESSION_ID:-${CLAUDE_SESSION_ID:-}}"
[ -n "$SESSION_ID" ] || exit 0

state_file="$HOME/.cache/recall-braid/$SESSION_ID/last-error.json"
[ -f "$state_file" ] || exit 0

prompt="$("$JQ" -r '.prompt // .user_prompt // empty' <<<"$raw" 2>/dev/null)"

# Always clear state — read-then-delete, even if the prompt was empty
# or we decide to drop the event below. Same error pairs at most once.
err_json="$(cat "$state_file" 2>/dev/null)"
rm -f "$state_file"

[ -z "$err_json" ] && exit 0
[ -z "$prompt" ] && exit 0

# Freshness gate: drop if older than MAX_AGE_SEC.
now_unix="$(date +%s)"
err_ts="$("$JQ" -r '.ts_unix // 0' <<<"$err_json")"
[ -z "$err_ts" ] && exit 0
age=$(( now_unix - err_ts ))
[ "$age" -lt 0 ] && exit 0
[ "$age" -gt "$MAX_AGE_SEC" ] && exit 0

# Build the joined observer event.
event="$("$JQ" -cn \
  --argjson e "$err_json" \
  --arg prompt "$prompt" \
  '{tool_name: $e.tool_name,
    tool_input: $e.tool_input,
    tool_response: $e.tool_response,
    status: "error",
    user_prompt_after: $prompt}')"
[ -z "$event" ] && exit 0

printf '%s\n' "$event" | "$RECALL_BIN" observe --format text >/dev/null || true
exit 0
