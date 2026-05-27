#!/usr/bin/env bash
# recall stop hook — promote within-session scratch to long-term memory,
# and auto-accept any memory IDs that were surfaced this session (v0.6.1,
# PRD-recall-outcome-feedback AC2).
#
# Stop fires when a Claude Code session ends. Two best-effort steps:
#   1. `recall scratch write` entries graduate to indexed long-term memory.
#   2. The "weather" auto-accept walk: any memory id listed in
#      $RECALL_WEATHER_DIR/<sid>/recalled.json (default ~/.cache/recall-weather/)
#      gets `recall feedback --accept`'d once — a small implicit-accept
#      bump for memories that surfaced and weren't contradicted.
# Silent if recall is not installed, no session id is set, or nothing
# was scratched/recalled.

set -uo pipefail

RECALL_BIN="${RECALL_BIN:-$HOME/.local/bin/recall}"
[ -x "$RECALL_BIN" ] || exit 0

# v0.5.1: read session_id from JSON payload (harness passes it as .session_id;
# $CLAUDE_SESSION_ID is not exported — same fix as braid v0.4.2 post-tool-use.sh
# and user-prompt-submit.sh). Silent on any failure preserved.
JQ="${JQ:-/usr/sbin/jq}"
sid=""
if [ -x "$JQ" ] && [ ! -t 0 ]; then
    raw="$(cat -)"
    if [ -n "$raw" ]; then
        sid="$("$JQ" -r '.session_id // empty' <<<"$raw" 2>/dev/null || true)"
    fi
fi
sid="${sid:-${CLAUDE_SESSION_ID:-}}"
[ -z "$sid" ] && exit 0

# `recall promote` returns 0 even on (nothing to promote), so this is a
# best-effort attempt; any error goes to stderr and the session still ends.
"$RECALL_BIN" promote --session "$sid" --format text 2>&1 | sed 's/^/[recall] /' || true

# v0.6.1 weather: auto-accept memories the session-start hook surfaced.
weather_dir="${RECALL_WEATHER_DIR:-$HOME/.cache/recall-weather}/$sid"
recalled_file="$weather_dir/recalled.json"
if [ -x "$JQ" ] && [ -f "$recalled_file" ]; then
    ids="$("$JQ" -r '.[]?' "$recalled_file" 2>/dev/null | tr '\n' ' ')"
    if [ -n "${ids// /}" ]; then
        # shellcheck disable=SC2086 -- intentional word-splitting on $ids.
        "$RECALL_BIN" feedback --accept $ids --format text 2>&1 \
            | sed 's/^/[recall] /' || true
    fi
    rm -rf "$weather_dir" 2>/dev/null || true
fi
