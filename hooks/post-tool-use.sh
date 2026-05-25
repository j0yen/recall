#!/usr/bin/env bash
# recall PostToolUse hook — pipe the event JSON into `recall observe`.
#
# Claude Code delivers the tool-use payload on stdin as one JSON object.
# `recall observe` reads it, applies the v0.4 heuristic catalog, and parks
# any proposals under `<root>/proposals/`. Review them with:
#     recall proposals
# Apply or discard with:
#     recall proposals --apply  <id>
#     recall proposals --discard <id>
#
# Silent if recall is not installed. Never exits non-zero — failure here
# must not block the agent loop.

set -uo pipefail

RECALL_BIN="${RECALL_BIN:-$HOME/.local/bin/recall}"
[ -x "$RECALL_BIN" ] || exit 0

# `recall observe` expects one event per line; the harness gives us one
# object, so we re-emit it as a single line and pipe.
payload="$(cat -)"
[ -z "$payload" ] && exit 0

printf '%s\n' "$payload" | "$RECALL_BIN" observe --format text >/dev/null 2>&1 || true
exit 0
