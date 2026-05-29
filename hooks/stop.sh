#!/usr/bin/env bash
# recall stop hook — promote within-session scratch to long-term memory,
# record hook-surfaced events, and auto-accept recalled memories.
#
# Stop fires when a Claude Code session ends. Three best-effort steps:
#   1. `recall scratch write` entries graduate to indexed long-term memory.
#   2. Surfaced increment (PRD-recall-surfaced-tracking AC6): any memory id
#      listed in $RECALL_WEATHER_DIR/<sid>/surfaced.json gets
#      `recall feedback --surfaced` — increments surfaced_count only, no
#      confidence change. Runs BEFORE the accept step so the two events
#      are distinct in the data.
#   3. The "weather" auto-accept walk: any memory id listed in
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

weather_dir="${RECALL_WEATHER_DIR:-$HOME/.cache/recall-weather}/$sid"

# Step 2 — surfaced increment (PRD-recall-surfaced-tracking AC6).
# Runs BEFORE the accept step; surfaced_count and feedback_count are distinct.
surfaced_file="$weather_dir/surfaced.json"
if [ -x "$JQ" ] && [ -f "$surfaced_file" ]; then
    surfaced_ids="$("$JQ" -r '.[]?' "$surfaced_file" 2>/dev/null | tr '\n' ' ')"
    if [ -n "${surfaced_ids// /}" ]; then
        # shellcheck disable=SC2086 -- intentional word-splitting on $surfaced_ids.
        "$RECALL_BIN" feedback --surfaced $surfaced_ids --format text 2>&1 \
            | sed 's/^/[recall] /' || true
    fi
fi

# Step 3 — v0.6.1 weather: auto-accept memories the session-start hook recalled.
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
