#!/usr/bin/env bash
# recall stop hook — promote within-session scratch to long-term memory.
#
# Stop fires when a Claude Code session ends. Any `recall scratch write`
# entries made during the session graduate to indexed long-term memory.
# Silent if recall is not installed, no session id is set, or nothing was
# scratched.

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
