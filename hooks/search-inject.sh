#!/usr/bin/env bash
# recall-search-inject.sh — UserPromptSubmit hook (sync, runs before
# Claude processes the prompt). Runs `recall query "$prompt"` and emits
# the top matches as a context block on stdout so the orchestrator sees
# them alongside the user's message.
#
# Closes the "static memory load at SessionStart" gap. recall already
# loads relevant memories on session start; this hook re-queries
# per-prompt so that mid-session topic shifts surface fresh memories.
#
# v0.10.0 surfaced-tracking (PRD-recall-surfaced-tracking AC5): appends
# surfaced memory ids to $RECALL_WEATHER_DIR/<sid>/surfaced.json.
# Does NOT touch recalled.json (these are mid-session surfacings, not
# start-of-session loads). Degrades silently if session_id is unavailable.
#
# Wall-budget target: <300ms. Skips short prompts and slash commands.

set -uo pipefail

RECALL_BIN="${RECALL_BIN:-$HOME/.local/bin/recall}"
[ -x "$RECALL_BIN" ] || exit 0

JQ="${JQ:-/usr/sbin/jq}"
prompt=""
sid=""
if [ -x "$JQ" ] && [ ! -t 0 ]; then
    raw="$(cat -)"
    if [ -n "$raw" ]; then
        prompt="$("$JQ" -r '.prompt // empty' <<<"$raw" 2>/dev/null || true)"
        sid="$("$JQ" -r '.session_id // empty' <<<"$raw" 2>/dev/null || true)"
    fi
fi

# Fallbacks
[ -n "$prompt" ] || prompt="${CLAUDE_USER_PROMPT:-}"
[ -n "$prompt" ] || exit 0
[ -n "$sid" ] || sid="${CLAUDE_SESSION_ID:-}"

# Skip very short prompts — not enough signal
[ "${#prompt}" -ge 12 ] || exit 0

# Skip slash-command invocations — skill arg context handles its own
case "$prompt" in
    /*) exit 0 ;;
esac

# FTS5-only query (no --hybrid) to keep cold-start fast.
# --limit 5 by default. Suppress stderr to keep the hook silent on errors.
result=$("$RECALL_BIN" query "$prompt" --limit 5 --format text 2>/dev/null || true)
[ -n "$result" ] || exit 0

# Filter to "real" matches — `recall query` lines start with a ULID
# (26 chars, 0-9A-Z) followed by metadata including "score=".
match_count=$(printf '%s\n' "$result" | grep -cE 'score=[0-9]' || true)
[ "${match_count:-0}" -ge 1 ] || exit 0

printf '\n=== recall: relevant memories for this prompt ===\n%s\n=== /recall ===\n' "$result"

# AC5 (PRD-recall-surfaced-tracking): append newly surfaced ids to
# surfaced.json. Only fires when session_id is known and jq is available.
# Does NOT touch recalled.json. Atomic: write tmp then mv.
if [ -n "$sid" ] && [ -x "$JQ" ]; then
    weather_dir="${RECALL_WEATHER_DIR:-$HOME/.cache/recall-weather}/$sid"
    if mkdir -p "$weather_dir" 2>/dev/null; then
        surfaced_file="$weather_dir/surfaced.json"
        # Extract ULIDs from the result (first token on each score= line).
        new_ids="$(printf '%s\n' "$result" \
            | grep -oE '^[0-9A-Z]{26}' \
            | "$JQ" -R . \
            | "$JQ" -s . 2>/dev/null || true)"
        [ -n "$new_ids" ] || exit 0
        # Merge with any existing surfaced.json, dedup, write atomically.
        existing="$("$JQ" '. // []' "$surfaced_file" 2>/dev/null || printf '[]')"
        merged="$(printf '%s\n%s' "$existing" "$new_ids" \
            | "$JQ" -s 'add | unique' 2>/dev/null || true)"
        [ -n "$merged" ] || exit 0
        printf '%s\n' "$merged" > "$surfaced_file.tmp" 2>/dev/null \
            && mv "$surfaced_file.tmp" "$surfaced_file" 2>/dev/null || true
    fi
fi

exit 0
