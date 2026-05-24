# recall

> Local-first agentic memory: file-backed memories with a keyword/FTS5 index and an in-process semantic embedder.

## Why

v0.1's HashEmbedder is non-semantic by design — it catches morphological variation but misses true synonyms and paraphrases. Two live consumers (SessionStart hook + /self-review Phase 0) use --hybrid retrieval, and reflective-memory compounding across sessions depends on semantic similarity surfacing prior notes when this session's wording differs. With a placeholder embedder the hybrid leaderboard collapses to pure FTS5 for cross-vocabulary matches and the compounding loop silently degrades. The load-bearing v0.2 change — separable from the 4b.1-4b.5 CLI sand-the-edges items — is graduating embeddings to a real in-process model (fastembed-rs + BGE-small-en-v1.5), no daemon, no Ollama, no breakage of the v0.1 file/frontmatter format. CLI gaps go to a follow-up slice.

## Build

```sh
cargo build --release
```

Produces `target/release/recall`. Symlink into `~/.local/bin/` if you want it on `$PATH`.

## Usage

```sh
recall --help
```

## Audience

Every Claude Code session running on the author's laptop, plus the author when inspecting memory. The binary is invoked from shells, from SessionStart hooks, and from the /self-review skill. The user wants memories about the same concept written in different words to surface for each other.

## Acceptance criteria

This project was scaffolded from a PRD via the `autobuilder` pipeline. The MUST-level acceptance criteria are:

- **AC1**: Cargo.toml depends on `fastembed` (any 4.x or 5.x), pinned with a tilde or caret range, and the workspace builds clean.
- **AC2**: A `FastembedEmbedder` type implements the existing `Embedder` trait. Its constructor lazy-fetches BGE-small-en-v1.5 into `~/.cache/fastembed/` on first use; subsequent constructions reuse the cache.
- **AC3**: FastembedEmbedder reports `embedding_id = "fastembed:bge-small-en-v1.5"` and `embedding_dim = 384` (BGE-small's actual dimensionality). HashEmbedder continues to report its own id and 256-dim.
- **AC4**: CLI accepts `--embedder hash|fastembed`. When the flag is absent, the default is `fastembed`. `hash` remains selectable for offline/test contexts.
- **AC5**: Semantic discrimination: given two memories — one about "package manager preferences" and one about "unrelated text" — a hybrid query for the synonym phrase "dependency tool" ranks the package-manager memory above the unrelated one with ...
- **AC6**: `recall reindex` regenerates the vector column using whichever embedder is active. After switching `--embedder hash` → `--embedder fastembed` and running reindex, the vector column reflects 384-dim FastembedEmbedder embeddings (verified ...
- **AC7**: File layout and YAML frontmatter remain byte-compatible with v0.1. A v0.1 memory file (provided as a fixture) round-trips through v0.2 read → write → read with zero byte differences in the frontmatter.

Each AC has a matching integration test under `tests/acceptance_ac<n>.rs`.

## Provenance

Built via the [`autobuilder`](https://github.com/j0yen/autobuilder) pipeline (PRD intake -> intent-card -> scaffold -> iterate-and-prove). Originally consolidated as a subdir of the [`wintermute`](https://github.com/j0yen/wintermute) monorepo; this standalone repo is a fresh-init snapshot for easier consumption and distribution.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
