# Changelog

## v0.15.0 — 2026-06-15

Adds `recall dedup` subcommand for finding near-duplicate memories by cosine
similarity. Loads all memories with stored embeddings from the SQLite index,
computes pairwise cosine similarity (dot product over L2-normalized vectors),
groups pairs above a configurable threshold (default 0.92) into clusters using
union-find, and reports the clusters with IDs, subjects, kinds, similarity
score, and a recommended action (`merge-into-newest`, `keep-highest-confidence`,
or `review-manually`). `--threshold`, `--min-cluster`, and `--json` flags
supported. Dry-run only — never writes to the database. Adds
`Index::get_all_embeddings()` to the library API. AC1-AC7 all green.

---

## v0.14.0 — 2026-06-03

The doctor exposes high-surface-low-use memories; this PRD acts on
them. `recall vacuum` is a sweep that, by default, lists candidate
ids matching `surfaced_count >= 20 AND used_count == 0` (the
pure-noise corpus). With `--apply` it executes one of three
configurable actions: aggressive decay (`confidence -= 0.10`),
supersede-proposal (writes under `~/.claude/recall/proposals/` for
user review, same surface braid uses), or archive (moves the file
to `memories-archive/`). Default action: decay. Plus a self-review
playbook entry that runs `recall vacuum --dry-run` weekly and
surfaces the count. Last PRD of the fidelity vision; closes the
loop from "measure utility" to "act on it."
---

## v0.13.0 — 2026-06-03

Discriminate accepted-used vs surfaced-but-unused memories in the Stop hook. Adds `used_count` column to `memories_meta`, a new `recall feedback --accept-used` flag that increments both `used_count` and `feedback_count`, and rewrites `hooks/stop.sh` to apply `--accept-used` on ids in `used.json` and `--abstain` on surfaced-but-unused ids. Legacy fallback to blanket-accept when only `recalled.json` exists. AC1–AC8 all green.

## v0.12.0 — 2026-06-03

Add surfaced_count column to track hook-injected memory surfacings separately from API-driven recall_count. Adds recall feedback --surfaced flag, SQLite migration, hook scripts (search-inject.sh, stop.sh), and full AC1-AC3 test coverage.

## v0.11.1 — 2026-05-30

Adds a `utility` section to `recall doctor` (JSON + text) reporting
`total_memories`, `with_surface_data` (surfaced_count >= 5), and two
ranked buckets: `low_utility_high_surface` (ratio < 0.2, top 10 by
surfaced_count) and `high_utility_validated` (ratio >= 0.7, top 10 by
used_count). Each record includes `calibration_drift = confidence -
(0.5 + ratio * 0.5)`. Pure diagnostic — no writes, no confidence
mutation. Sets the table for `recall vacuum` (PRD #5).

## v0.11.0 — 2026-05-30

Discriminate accepted-used vs surfaced-but-unused memories in the Stop hook. Adds `used_count` column to `memories_meta`, a new `recall feedback --accept-used` flag that increments both `used_count` and `feedback_count`, and rewrites `hooks/stop.sh` to apply `--accept-used` on ids in `used.json` and `--abstain` on surfaced-but-unused ids. Legacy fallback to blanket-accept when only `recalled.json` exists. AC1–AC8 all green.

## v0.10.0 — 2026-05-29

Separate `surfaced_count` from `recall_count` — the Stop hook now records hook-injected surface events (`recall feedback --surfaced`) before the accept step, enabling downstream use-evidence discrimination. Adds `surfaced.json` tracking alongside `recalled.json` in the weather dir.

## v0.9.0 — 2026-05-29

Promotes temporal decay to a first-class `recall temporal-decay` subcommand with dry-run
support, per-memory reporting, configurable thresholds, and a `temporal_decay` pure-function
module. Dry-run (default) shows what would decay; `--apply` writes the changes. Accepts
`--half-life-d`, `--min-interval-d`, `--min-delta`, `--subject`, and `--format text|json`.
The new `Index::temporal_decay_report` method is the preferred path over the legacy
`--decay-sweep` flag (kept for backward compat). AC1-AC6 all green.

## v0.8.0 — 2026-05-29

feat: add `recall vacuum` subcommand (PRD-recall-corpus-vacuum)

New `recall vacuum` command sweeps low-utility-high-surface memories:

- Candidates: `surfaced_count >= min_surfaced AND recall_count <= max_used` (defaults: 20, 0)
- Default mode: dry-run (lists candidates, no writes)
- `--apply` executes the configured action:
  - `decay` (default): reduces confidence by `decay_amount` (0.10), floored at 0.05; recoverable
  - `supersede`: writes a proposal file under `proposals/` for user review; memory unchanged
  - `archive`: moves the file to `memories-archive/<kind>/` and removes the index row
- `--min-surfaced` / `--max-used` flags override `recall.toml` thresholds
- `recall.toml` `[vacuum]` section for defaults
- Self-review playbook: `~/.claude/skills/self-review/playbooks/recall_corpus_vacuum.md`
- JSON and text output formats

feat: add `recall doctor --check-claims` command

New `doctor_claims` module spots-checks memory body assertions against live
filesystem state and installed binary versions:

- Kind A: filesystem-path assertions (fenced code blocks, `path:` prefixes)
  — each asserted path is stat'd; missing paths become proposals
- Kind B: binary-version assertions for a curated whitelist of binaries
  — forks `<binary> --version`, extracts semver, compares; mismatches become proposals
- Proposals written to `<root>/proposals/` as Markdown files for user review
- `--dry-run` flag to report without writing proposal files
- `--skip-version-checks` flag to skip binary fork-execs
- `--subject-prefix` and `--since` filters to scope the scan
- JSON and plain-text output formats supported

## v0.7.0 — 2026-05-29

`recall doctor --check-claims` spot-checks memory body assertions against live filesystem state
and binary versions. Two Fleet-1 assertion kinds: (A) filesystem-path assertions (in fenced code
blocks or prose with trigger phrases like "see ", "lives at ", etc.), (B) version-number claims
for a 19-binary whitelist. Disconfirmed assertions produce proposal files under
`~/.claude/recall/proposals/` (same review surface as `recall observe`) — no auto-edits.

Also lands: temporal-decay report via `recall doctor --temporal-decay`; use-detect evidence
tracking via `recall use-detect`; session-stamp query filters (`--session`, `--no-session`);
`recall sessions` subcommand listing distinct session ids with memory counts.

Exit codes: 0 clean scan, 1 at least one disconfirmed assertion, 2 scan error.

## v0.6.0 — 2026-05-27

PRD-recall-outcome-feedback §6a (codename *weather*) lands the first half
of outcome feedback for memory confidence.

New subcommand `recall feedback` accepts `--accept <id>...` /
`--reject <id>...` / `--abstain <id>` / `--decay-sweep`. Accept bumps
confidence by `accept_delta` (default 0.02, ceiling 0.95). Reject
decays by `reject_delta` (default 0.10, floor 0.05). Decay sweep moves
confidence toward 0.5 along the formula
`confidence' = 0.5 + (confidence - 0.5) * 2^(-days/half_life_d)`
with a default 90-day half-life, idempotent within a day.

Schema gains a `feedback_count` column on the memory index. New
`[feedback]` config block exposes `accept_delta`, `reject_delta`,
`floor`, `ceiling`, `half_life_d` knobs.

ACs 1, 3, 5 pair against new `feedback.rs` unit tests (42/42 green).
AC4 idempotency lives in `index::apply_decay_sweep`. ACs 2 (Stop-hook
auto-accept), 6 (ranking diff smoke), 7 (doctor `confidence_drift`)
phase to v0.6.1 / v0.6.2.

Pure rust-extend; no hooks rewired this version.

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
