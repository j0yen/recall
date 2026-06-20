# recall

Local-first memory for Claude Code: plain `.md` files on disk, indexed for keyword and semantic search, with hooks that surface the right memory into a live session and park new ones from what went wrong.

An agent that can't remember rediscovers the same lesson every session. The fix people reach for is a vector database and a service to run it — which means a daemon, a network hop, and your notes living somewhere you can't read them. `recall` takes the opposite position: a memory is a Markdown file with YAML frontmatter, in a directory you own. You can `grep` it, edit it in your editor, and put it in git. The index — SQLite with FTS5 for keywords and a vector column for meaning — is derived state, rebuildable from the files at any time. No daemon required, no network, no external service. The files are the source of truth.

Built in Rust (edition 2024, `rustc 1.85+`). SQLite is bundled; the embedding model is the only thing fetched, and only on first semantic use.

## Install

### One-liner

```sh
curl -fsSL https://raw.githubusercontent.com/j0yen/recall/main/install.sh | bash
```

It self-clones into `~/.local/share/recall/`, builds the binary with `cargo install --path . --locked` (lands `recall` in `~/.cargo/bin/`), symlinks the braid hook scripts into `~/.claude/scripts/`, and prints the JSON snippet you paste into `~/.claude/settings.json`. It never edits `settings.json` for you — the snippet is yours to review and merge.

### Manual

```sh
git clone --depth 1 https://github.com/j0yen/recall.git
cd recall
./install.sh
```

### Prerequisites

- `cargo` / `rustc 1.85+`
- `git`, `jq`, `bash`
- Claude Code — for the hooks. The `recall` CLI works standalone without it.
- ~130 MB free in `~/.cache/fastembed/` for the embedding model (BGE-small-en-v1.5, lazy-fetched on first `--embedder fastembed` use).

## Quickstart

```sh
# Write a memory:
recall write --kind reflective --subject self \
  "Prefer pnpm over npm for TypeScript projects on this laptop."

# Query — hybrid is FTS5 keyword + vector semantic, ranked together:
recall query --hybrid "package manager"

# See what's stored:
recall list --subject self

# Rebuild the index from the files (e.g. after switching embedders):
recall reindex
```

The CLI is the API. `recall --help` lists every subcommand; the ones you'll reach for most are `write`, `query`, `list`, `show`, `similar`, and `reindex`.

## The model

A **memory** is one `.md` file: YAML frontmatter (id, kind, subject, confidence, timestamps, recall/surface/use counts) plus a free-text body. **Kind** is what sort of thing it is — `reflective`, `semantic`, and so on. **Subject** is who or what it's about (`self`, `user`, a project), and it doubles as the directory the file lives in.

Two embedders ship. `fastembed` (the default) loads BGE-small-en-v1.5 in-process for real semantic similarity. `hash` is a deterministic, dependency-free, non-semantic fallback — useful for tests and for environments where you don't want the model download. Switch with the global `--embedder` flag; `recall reindex` after switching, since the vectors change.

## How it wires into a session — the braid

The hooks turn a passive store into something that participates in the session. Add them to `~/.claude/settings.json` under `.hooks` (or paste what `install.sh` prints):

| Hook | Script | What it does |
|---|---|---|
| `UserPromptSubmit` | `recall-user-prompt.sh` | Surfaces relevant memories for the prompt; joins the prompt with a recent tool failure and feeds the observer's heuristics. |
| `PostToolUseFailure` | `recall-post-tool-use.sh` | Records the error to per-session state so the next prompt can pair with it. |
| `Stop` | `recall-stop.sh` | Promotes scratch entries to long-term memory, and records which surfaced memories the session actually used. |

These three implement the **braid** correlator: a tool error paired with the corrective prompt that follows it gets parked as a `reflective` draft under `~/.claude/recall/proposals/`. Drafts are proposals, not memories — review them with `recall proposals` and accept with `recall promote <id>`. A freshness gate (`RECALL_BRAID_MAX_AGE`, default 300 s) keeps the pairing relevant: an error too old when the next prompt arrives is dropped.

The repo also ships `hooks/search-inject.sh` for injecting search results at the prompt step; the installer symlinks the three braid hooks above by default.

## Closing the loop — does a memory earn its place?

Surfacing a memory isn't the same as it being useful. `recall` tracks the difference and acts on it.

- The Stop hook records `surfaced_count` (the memory was injected into context) separately from `used_count` (the session actually referenced it). `recall feedback --accept-used` rewards the latter; `--abstain` marks surfaced-but-ignored.
- `recall doctor` reports memories with high surface and no use — the ones quietly adding noise to every session.
- `recall vacuum` acts on them: by default it lists the pure-noise corpus (`surfaced_count >= 20 AND used_count == 0`); `--apply` decays their confidence, files a supersede-proposal, or archives the file.
- `recall dedup` finds near-duplicate memories by cosine similarity, clusters them, and recommends a merge — read-only, never writes.
- `recall temporal-decay` ages confidence over time so stale memories fade rather than linger.

Together these answer a question most memory systems never ask: which memories are worth keeping?

## The daemon (optional)

Cold-starting the embedding model on every `query` is the slow part. `recalld` keeps the index and embedder warm and answers read-only requests (`query`, `embed`, `touch`, `ping`) over a Unix socket; writes stay on the CLI by design.

```sh
recall daemon start      # detached
recall daemon status
recall daemon stop
```

For an always-warm daemon in your user session, see [`contrib/systemd/`](contrib/systemd/) for a `recalld.service` user unit.

## Storage layout

```
~/.claude/recall/
├── memories/          # one .md per memory, grouped by subject; YAML frontmatter
│   ├── self/
│   ├── user/
│   └── ...
├── index/             # SQLite (FTS5 + vector column) — derived, rebuildable
├── proposals/         # draft memories from the braid; reviewable
├── scratch/           # per-session scratch; promoted on Stop
└── session/           # ephemeral per-session state
```

The memories are plain text. Read, edit, or `grep` them directly; `recall reindex` picks up changes. Override the root with `--root` or `$RECALL_HOME`.

## The recall family

`recall` is the flagship. Two companion tools work the same on-disk store standalone:

- **[recall-doctor](https://github.com/j0yen/recall-doctor)** — `fsck` for the store. Reports divergence between the `.md` files and the SQLite index (orphans, missing rows, embedder ids) and exits non-zero when they disagree. `recall doctor` is the in-binary version; the standalone tool predates it and runs against any store.
- **[recall-io](https://github.com/j0yen/recall-io)** — backup and migration. Exports the whole store to NDJSON (one memory per line, deterministic) and imports it back, round-trip-faithful.

## Status

Daily-driver on the laptop where it was built; see [CHANGELOG.md](CHANGELOG.md) for the version history. The braid correlator and the surface/use fidelity loop are verified end-to-end. External use is welcome — expect rough edges, and note the embedding-model download is the slowest part of a cold install.

## Provenance

Built via the [`autobuilder`](https://github.com/j0yen/autobuilder) pipeline (PRD intake → intent-card → scaffold → iterate-and-prove). Originally a subdirectory of the [`wintermute`](https://github.com/j0yen/wintermute) monorepo; this standalone repo is the canonical distribution.

## License

Dual-licensed under MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

Copyright (c) 2026 Joe Yen.
