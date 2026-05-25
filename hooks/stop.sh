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

sid="${CLAUDE_SESSION_ID:-}"
[ -z "$sid" ] && exit 0

# `recall promote` returns 0 even on (nothing to promote), so this is a
# best-effort attempt; any error goes to stderr and the session still ends.
"$RECALL_BIN" promote --session "$sid" --format text 2>&1 | sed 's/^/[recall] /' || true
