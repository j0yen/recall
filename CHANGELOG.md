# Changelog

## v0.4.2 — 2026-05-25

Fix the braid correlator hooks: v0.4.1 read `$CLAUDE_SESSION_ID` from
env, but the Claude Code harness passes session id in the input JSON's
`.session_id` field, not as an env var — so both hooks silently no-op'd
on every fire and zero proposals ever landed. v0.4.2 reads `.session_id`
from the JSON payload first, with `$CLAUDE_SESSION_ID` as a fallback for
forward-compat. No binary code changes; pure hook-script fix.

Verified end-to-end this session: synthetic invocation against the hooks
with real harness-shaped JSON now writes the state file and produces a
proposal when paired with corrective language in the next prompt.

## v0.4.1 — 2026-05-25


The v0.4 `recall observe` heuristic catalog needs `user_prompt_after` to
propose anything — but a single `PostToolUseFailure` event can't carry the
next user prompt, because that prompt hasn't been written yet. As a result
the wire-up shipped in v0.4 is structurally correct and functionally
inert: zero proposals get parked, ever. `braid` adds the missing piece —
a session-scoped state file that pairs the most recent error with the
next `UserPromptSubmit` and feeds the joined event to `recall observe`
synchronously, in the prompt-submit hook's milliseconds-budget window.

The whole change is a state file under `~/.cache/recall-braid/` plus two
hook scripts (replacing `recall-post-tool-use.sh` and adding a
`recall-user-prompt.sh`). No recall binary change required. The observer
already accepts the right input shape; we just need to assemble it.