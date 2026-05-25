# recall

> Local-first agentic memory for Claude Code. File-backed memories
> with a keyword/FTS5 index, an in-process semantic embedder
> (BGE-small-en-v1.5 via fastembed), and four hook scripts that wire
> recall into a live Claude Code session so memories surface
> automatically and tool errors plus corrective prompts get parked
> as reflective draft memories.

Memories are plain `.md` files with YAML frontmatter on disk, indexed
into SQLite (FTS5 + a vector column). No daemon, no network, no
external service. Built with Rust 2024 / `rustc 1.85`.

## Install

### One-liner

```sh
curl -fsSL https://raw.githubusercontent.com/j0yen/recall/main/install.sh | bash
```

Self-clones into `~/.local/share/recall/`, runs
`cargo install --path .` to put `recall` in `~/.cargo/bin/`, then
symlinks the four hook scripts into `~/.claude/scripts/`. It prints
the JSON snippets you paste into `~/.claude/settings.json` to
activate the hooks (it does not auto-edit settings.json — see
[Wiring the hooks](#wiring-the-hooks) below).

### Manual install

```sh
git clone --depth 1 https://github.com/j0yen/recall.git
cd recall
./install.sh
```

### Prerequisites

- `cargo` / `rustc 1.85+`
- `git`, `jq`, `bash`
- Claude Code (to use the hooks; the `recall` CLI works standalone)
- ~130MB free in `~/.cache/fastembed/` for the embedding model
  (lazy-fetched on first `--embedder fastembed` use)

## Quick start

```sh
# Write a memory:
recall write --kind reflective --subject self \
  "Always prefer pnpm over npm for TypeScript projects on this laptop."

# Query (hybrid = FTS5 + vector):
recall query --hybrid "package manager"

# Inspect what's stored:
recall list --subject self

# Reindex after switching embedders:
recall reindex
```

Full help: `recall --help`.

## Wiring the hooks

Once the binary is installed and the hook scripts are symlinked,
add these entries to `~/.claude/settings.json` under `.hooks` (or
paste from the snippet `install.sh` prints):

```jsonc
{
  "hooks": {
    "SessionStart": [
      { "matcher": "*", "hooks": [
        { "type": "command", "command": "/home/<you>/.claude/scripts/recall-session-start.sh" }
      ]}
    ],
    "UserPromptSubmit": [
      { "matcher": "", "hooks": [
        { "type": "command", "command": "/home/<you>/.claude/scripts/recall-user-prompt.sh",
          "timeout": 1, "async": true }
      ]}
    ],
    "PostToolUseFailure": [
      { "matcher": "*", "hooks": [
        { "type": "command", "command": "/home/<you>/.claude/scripts/recall-post-tool-use.sh",
          "timeout": 5, "async": true }
      ]}
    ],
    "Stop": [
      { "matcher": "", "hooks": [
        { "type": "command", "command": "/home/<you>/.claude/scripts/recall-stop.sh",
          "timeout": 30, "async": true }
      ]}
    ]
  }
}
```

What each hook does:

| Hook | Script | Behavior |
|---|---|---|
| `SessionStart` | `recall-session-start.sh` | Emits relevant memories into the session opening context |
| `UserPromptSubmit` | `recall-user-prompt.sh` | Joins the prompt with the last tool-failure (if recent) and feeds the observer's heuristics |
| `PostToolUseFailure` | `recall-post-tool-use.sh` | Writes the error to per-session state so the next prompt can pair with it |
| `Stop` | `recall-stop.sh` | Promotes any `recall scratch` entries to long-term memory at session end |

Together these implement the *braid* correlator — tool errors paired
with corrective user prompts get parked as `reflective` draft
memories under `~/.claude/recall/proposals/`. Review with
`recall proposals` and promote with `recall promote <id>`.

## Configuration

Environment variables read by the hooks:

| Var | Default | Effect |
|---|---|---|
| `RECALL_BIN` | `~/.local/bin/recall` | Path to the recall binary |
| `JQ` | `/usr/sbin/jq` | Path to jq; hooks no-op if missing |
| `RECALL_BRAID_MAX_AGE` | `300` (seconds) | Freshness gate on the braid correlator. Tool errors older than this when the next user prompt arrives are dropped. Default raised from 60s in v0.4.3 to cover interactive read+type latency without sacrificing relevance. |

## Storage layout

```
~/.claude/recall/
├── memories/          # one .md per memory, YAML frontmatter
│   ├── self/
│   ├── user/
│   └── ...
├── index/             # SQLite (FTS5 + vector column)
├── proposals/         # draft memories from the observer; reviewable
├── scratch/           # per-session scratch; promoted on Stop
└── session/           # ephemeral per-session state
```

Memories are plain text. You can read, edit, or `grep` them
directly; recall picks up changes on next index refresh.

## Status

`v0.4.2` — daily-driver on the laptop where it was built. The
braid correlator (PostToolUseFailure → UserPromptSubmit) verified
end-to-end live as of 2026-05-25. External use is welcome but
expect rough edges; the embedding model is the slowest part of
cold install. See [CHANGELOG.md](CHANGELOG.md).

Note: the SessionStart hook script (`recall-session-start.sh`) is
not yet shipped from this repo — it currently lives in the author's
dotfiles tree. The one-liner installer symlinks the three braid
hooks (`post-tool-use`, `user-prompt-submit`, `stop`) which all
ship here. SessionStart will move in when it lands as a recall
subcommand (`recall session-start --emit`).

## Provenance

Built via the [`autobuilder`](https://github.com/j0yen/autobuilder)
pipeline (PRD intake → intent-card → scaffold → iterate-and-prove).
Originally consolidated as a subdir of the
[`wintermute`](https://github.com/j0yen/wintermute) monorepo; this
standalone repo is the canonical distribution.

## License

Dual-licensed under MIT OR Apache-2.0. See
[LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

Copyright (c) 2026 Joe Yen.
