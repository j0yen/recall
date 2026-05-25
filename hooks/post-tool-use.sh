#!/usr/bin/env bash
# recall PostToolUseFailure hook (v0.4.1, *braid* correlator) — write the
# error to a per-session state file so the next UserPromptSubmit can pair
# it with the user's prompt. This hook does NOT call `recall observe`
# anymore — the observe call moves to recall-user-prompt.sh, where the
# joined event has a non-empty `user_prompt_after` and the heuristic
# catalog can actually fire.
#
# State layout:
#   ~/.cache/recall-braid/<session-id>/last-error.json
#     { ts_unix, ts_mono, tool_name, tool_input, tool_response }
#
# Atomic write via tempfile + rename. Per-session subdir keeps concurrent
# Claude sessions from cross-pollinating. If $CLAUDE_SESSION_ID is unset,
# no-op (better to drop the event than to merge it with another session).
#
# Silent on any failure — never block a Claude turn on telemetry.

set -uo pipefail
exec 2>/dev/null

JQ="${JQ:-/usr/sbin/jq}"
SESSION_ID="${CLAUDE_SESSION_ID:-}"
[ -x "$JQ" ] || exit 0
[ -n "$SESSION_ID" ] || exit 0

raw="$(cat -)"
[ -z "$raw" ] && exit 0

ts_unix="$(date +%s)"
ts_mono="$(awk '{print $1}' /proc/uptime 2>/dev/null || echo 0)"

event="$("$JQ" -cn \
  --argjson ts_unix "$ts_unix" \
  --arg ts_mono "$ts_mono" \
  --arg name "$("$JQ" -r '.tool_name // empty' <<<"$raw")" \
  --argjson input "$("$JQ" '.tool_input // {}' <<<"$raw")" \
  --argjson resp  "$("$JQ" '.tool_response // {}' <<<"$raw")" \
  '{ts_unix:$ts_unix, ts_mono:($ts_mono|tonumber), tool_name:$name, tool_input:$input, tool_response:$resp}')"
[ -z "$event" ] && exit 0

dir="$HOME/.cache/recall-braid/$SESSION_ID"
mkdir -p "$dir" 2>/dev/null || exit 0

tmp="$(mktemp "$dir/.last-error.XXXXXX.json" 2>/dev/null)" || exit 0
printf '%s\n' "$event" > "$tmp" || { rm -f "$tmp"; exit 0; }
mv -f "$tmp" "$dir/last-error.json" 2>/dev/null || rm -f "$tmp"
exit 0
