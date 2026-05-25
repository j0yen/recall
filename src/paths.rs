//! `~/.claude/recall/` path resolution. Honors `RECALL_HOME` for tests.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Root of the recall data dir. `$RECALL_HOME` if set, else `~/.claude/recall`.
pub fn root() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("RECALL_HOME") {
        return Ok(PathBuf::from(custom));
    }
    let home = directories::BaseDirs::new()
        .context("could not resolve user home directory")?;
    Ok(home.home_dir().join(".claude").join("recall"))
}

pub fn memories_dir(root: &Path) -> PathBuf {
    root.join("memories")
}

pub fn index_dir(root: &Path) -> PathBuf {
    root.join("index")
}

pub fn index_db(root: &Path) -> PathBuf {
    index_dir(root).join("recall.sqlite")
}

pub fn session_dir(root: &Path) -> PathBuf {
    root.join("session")
}

/// Scratch dir for a single session — `session/<sid>/`. Phase 3.
pub fn scratch_session_dir(root: &Path, session_id: &str) -> PathBuf {
    session_dir(root).join(session_id)
}

/// Where Phase 4 `recall propose` parks draft memories before the user
/// promotes or discards them. `proposals/<id>.md`.
pub fn proposals_dir(root: &Path) -> PathBuf {
    root.join("proposals")
}

/// Where on disk a memory's Markdown file lives, given its subject and id.
/// Layout: `memories/<namespace>/[<slug>/]<id>.md`.
pub fn memory_path(root: &Path, subject: &crate::memory::Subject, id: &str) -> PathBuf {
    let mut p = memories_dir(root);
    let ns = subject.namespace();
    p.push(ns);
    if let Some(slug) = subject.as_str().strip_prefix(&format!("{ns}:")) {
        p.push(slug);
    }
    p.push(format!("{id}.md"));
    p
}
