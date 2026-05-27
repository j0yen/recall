# Changelog

## v0.5.3 — 2026-05-27

Braid proposals from Bash failures used to render as `Tool error: {}`
because the Claude Code harness ships `tool_response: {}` for failing
Bash calls — neither stderr nor the failing command surfaced. The
proposal still fired (heuristic gates on `status:error` + corrective
language), but lost the most informative half of the context.

`recall observe` now falls back to `tool_input.command` (Bash),
`tool_input.file_path` (Edit/Read/Write), or `tool_input.description`
when `tool_response` is null/empty/`{}`. Truncated at 400 chars under
an `Input:` line in the proposal body. Non-empty `tool_response`
keeps the prior shape (no `Input:` line, avoids duplication).

Pure observer-side change; no hooks touched, no schema bump. Three new
unit tests cover empty-Bash, empty-Edit, and unknown-tool-input.

## v0.5.2 — 2026-05-26

`recall daemon start`, `daemon stop`, and `daemon restart` ship as
first-class subcommands. `start` forks `recalld` into the background by
default (writing a pidfile under `$XDG_RUNTIME_DIR/recall.pid`, or
`~/.cache/recall/recall.pid` fallback) and supports `--foreground` for
systemd `Type=simple` integration. `stop` sends `SIGTERM` and gates
on socket disappearance via `--wait-secs`; `restart` chains the two.

`recall doctor --format json` now reports `daemon_active` (bool) and
`daemon_uptime_s` (u64), probed via `ping_socket_sync` against the live
daemon when present. Closes the last gap on PRD-recall-daemon §5 (AC1
cold-start subcommand path, AC4 doctor liveness, AC6 SIGKILL+restart).

No protocol or storage changes. Pidfile handling is atomic — `recalld`
removes the pidfile on graceful exit, and `daemon start` refuses to
launch if a live pid is already listening on the socket.

## v0.5.1 — 2026-05-25

Fix the recall Stop-hook session_id source: like the braid hooks before
the v0.4.2 fix, the Stop hook was reading `$CLAUDE_SESSION_ID` from env,
but the harness passes session id in the JSON payload's `.session_id`
field. Hook now reads from JSON first with the env var as fallback.
No binary change; hook-script-only fix.

## v0.5.0 — 2026-05-25

`recalld` long-lived daemon ships. Loads the fastembed BGE-small ONNX
model once at startup, listens on `$XDG_RUNTIME_DIR/recall.sock` (or
`~/.cache/recall/recall.sock` fallback), and answers length-prefixed
JSON requests over UDS. Ops in v1: `ping`, `query`, `embed`, `touch`.

The CLI auto-forwards filter-free `recall query` invocations to the
socket when present, silently falling back to in-process retrieval when
the daemon is down or the request bears filters the v0.5.0 daemon `query`
op doesn't yet expose. `recall daemon status [--format text|json]`
reports model_id / uptime / version / root over the same protocol;
`recall where` adds a `daemon_active` liveness line.

Includes a `contrib/systemd/recalld.service` user unit and
`contrib/systemd/README.md` install notes. SIGINT/SIGTERM trigger
graceful shutdown. Subsumes `PRD-recall-observer-correlation.md`
state-file correlator for the warm path; cold path is unchanged.

## v0.4.3 — 2026-05-25

Raise the braid freshness gate default from 60s to 300s so human-paced
turns (read assistant message + type corrective reply) don't silently
drop. `$RECALL_BRAID_MAX_AGE` env override is preserved and now
documented in `README.md` under Configuration. No Rust code change —
literal-value bump in `hooks/user-prompt-submit.sh` plus docs.

Motivated by live verification 2026-05-25: a ~120s human read+type gap
tripped the 60s gate and dropped a real correction event. 300s comfortably
covers human-paced cycles while still well short of session drift
(coincides with Anthropic prompt-cache TTL).

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