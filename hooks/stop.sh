#!/usr/bin/env bash
# recall stop hook — promote within-session scratch to long-term memory,
# record hook-surfaced events, discriminate used vs unused memories, and
# collect use-evidence for surfaced memories.
#
# Stop fires when a Claude Code session ends. Four best-effort steps:
#   1. `recall scratch write` entries graduate to indexed long-term memory.
#   2. Surfaced increment (PRD-recall-surfaced-tracking AC6): any memory id
#      listed in $RECALL_WEATHER_DIR/<sid>/surfaced.json gets
#      `recall feedback --surfaced` — increments surfaced_count only, no
#      confidence change. Runs BEFORE the feedback step so the two events
#      are distinct in the data.
#   3. Use-detect (PRD-recall-use-evidence): scan the session transcript for
#      n-gram and API-recall evidence; write used.json to the weather dir.
#      Must run before Step 4 so used.json exists when we compute set diff.
#   4. Discriminating feedback (PRD-recall-stop-hook-discriminate v0.7.3):
#      - If surfaced.json AND used.json both exist:
#          * used ids → `recall feedback --accept-used` (bumps conf + used_count)
#          * (surfaced - used) ids → `recall feedback --abstain` (no conf change)
#      - If surfaced.json missing but recalled.json exists: legacy blanket-accept.
#      - If surfaced.json exists but used.json missing (use-detect failed):
#          all surfaced ids treated as abstain (conservative — don't reward without evidence).
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
# Runs BEFORE the feedback step; surfaced_count and feedback_count are distinct.
surfaced_file="$weather_dir/surfaced.json"
if [ -x "$JQ" ] && [ -f "$surfaced_file" ]; then
    surfaced_ids="$("$JQ" -r '.[]?' "$surfaced_file" 2>/dev/null | tr '\n' ' ')"
    if [ -n "${surfaced_ids// /}" ]; then
        # shellcheck disable=SC2086 -- intentional word-splitting on $surfaced_ids.
        "$RECALL_BIN" feedback --surfaced $surfaced_ids --format text 2>&1 \
            | sed 's/^/[recall] /' || true
    fi
fi

# Step 3 — use-detect (PRD-recall-use-evidence AC8): scan the session transcript
# and write used.json to the weather dir alongside surfaced.json.
# Must run before Step 4 so that used.json is available for set-difference.
# Best-effort: silent on failure.
"$RECALL_BIN" use-detect --session "$sid" --format text 2>&1 \
    | sed 's/^/[recall] /' || true

# Step 4 — discriminating feedback (PRD-recall-stop-hook-discriminate v0.7.3).
used_file="$weather_dir/used.json"
recalled_file="$weather_dir/recalled.json"

if [ -x "$JQ" ] && [ -f "$surfaced_file" ]; then
    # New path: surfaced.json exists → discriminate used vs unused.
    if [ -f "$used_file" ]; then
        # Both surfaced.json and used.json present — apply --accept-used to used
        # ids and --abstain to (surfaced - used) ids.
        used_ids="$("$JQ" -r '.[]?' "$used_file" 2>/dev/null | tr '\n' ' ')"
        abstain_ids="$("$JQ" -rn \
            --slurpfile s "$surfaced_file" \
            --slurpfile u "$used_file" \
            '(($s[0] // []) - ($u[0] // []))[]?' 2>/dev/null | tr '\n' ' ')"

        if [ -n "${used_ids// /}" ]; then
            # shellcheck disable=SC2086 -- intentional word-splitting on $used_ids.
            "$RECALL_BIN" feedback --accept-used $used_ids --format text 2>&1 \
                | sed 's/^/[recall] /' || true
        fi
        if [ -n "${abstain_ids// /}" ]; then
            # shellcheck disable=SC2086 -- intentional word-splitting on $abstain_ids.
            "$RECALL_BIN" feedback --abstain $abstain_ids --format text 2>&1 \
                | sed 's/^/[recall] /' || true
        fi
    else
        # surfaced.json exists but used.json missing (use-detect failed).
        # AC6: conservative default — abstain on all surfaced ids.
        surfaced_ids_abs="$("$JQ" -r '.[]?' "$surfaced_file" 2>/dev/null | tr '\n' ' ')"
        if [ -n "${surfaced_ids_abs// /}" ]; then
            # shellcheck disable=SC2086 -- intentional word-splitting on $surfaced_ids_abs.
            "$RECALL_BIN" feedback --abstain $surfaced_ids_abs --format text 2>&1 \
                | sed 's/^/[recall] /' || true
        fi
    fi
elif [ -x "$JQ" ] && [ -f "$recalled_file" ]; then
    # AC5 legacy fallback: no surfaced.json but recalled.json exists.
    # Pre-PRD-#1 weather dir: apply old blanket-accept on recalled.json ids.
    ids="$("$JQ" -r '.[]?' "$recalled_file" 2>/dev/null | tr '\n' ' ')"
    if [ -n "${ids// /}" ]; then
        # shellcheck disable=SC2086 -- intentional word-splitting on $ids.
        "$RECALL_BIN" feedback --accept $ids --format text 2>&1 \
            | sed 's/^/[recall] /' || true
    fi
fi

rm -rf "$weather_dir" 2>/dev/null || true
