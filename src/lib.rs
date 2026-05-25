//! recall — local-first agentic memory.
//!
//! See `PRD-agentic-memory.md` for motivation. This v0.2 crate ships the
//! v0.1 modules (file store, FTS5 index, retrieval, paths, memory schema)
//! unchanged, swaps the placeholder hashed-feature `HashEmbedder` for an
//! in-process semantic `FastembedEmbedder` (BGE-small-en-v1.5 via
//! `fastembed-rs`), and keeps `HashEmbedder` available behind
//! `--embedder hash` for offline / test contexts.

#![cfg_attr(not(test), forbid(unsafe_code))]
#![allow(missing_docs)]

pub mod config;
pub mod daemon;
pub mod embeddings;
pub mod index;
pub mod memory;
pub mod observer;
pub mod paths;
pub mod retrieval;
pub mod scratch;
pub mod store;
