//! Phase 3 within-session scratch storage.
//!
//! Scratch memories live at `<root>/session/<sid>/<id>.md`. They use the
//! same `Memory` schema as long-term memories so `promote` is a plain file
//! move + index insert.
//!
//! Scratch is NOT indexed in the FTS5 store. `recall query`/`list` never
//! see scratch entries. `recall scratch list` walks the per-session dir.

use crate::memory::Memory;
use crate::paths;
use anyhow::{Context, Result, anyhow};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Resolve a usable session id. Honors `--session`, falls back to
/// `$CLAUDE_SESSION_ID`, then errors. Caller decides the precedence.
pub fn resolve_session_id(explicit: Option<&str>) -> Result<String> {
    if let Some(s) = explicit {
        if !s.trim().is_empty() {
            return Ok(s.to_string());
        }
    }
    if let Ok(s) = std::env::var("CLAUDE_SESSION_ID") {
        if !s.trim().is_empty() {
            return Ok(s);
        }
    }
    Err(anyhow!(
        "no session id (pass --session or set CLAUDE_SESSION_ID)"
    ))
}

pub fn write(root: &Path, session_id: &str, mem: &Memory) -> Result<PathBuf> {
    let dir = paths::scratch_session_dir(root, session_id);
    fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let path = dir.join(format!("{}.md", mem.front.id));
    let tmp = dir.join(format!(".{}.md.tmp", mem.front.id));
    fs::write(&tmp, mem.to_markdown()?)
        .with_context(|| format!("write tempfile {}", tmp.display()))?;
    fs::rename(&tmp, &path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(path)
}

pub fn list(root: &Path, session_id: Option<&str>) -> Result<Vec<(Memory, PathBuf)>> {
    let base = match session_id {
        Some(s) => paths::scratch_session_dir(root, s),
        None => paths::session_dir(root),
    };
    if !base.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in WalkDir::new(&base)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_type().is_file()
                && e.path().extension().is_some_and(|x| x == "md")
        })
    {
        let text = fs::read_to_string(entry.path())
            .with_context(|| format!("read {}", entry.path().display()))?;
        let mem = Memory::from_markdown(&text)
            .with_context(|| format!("parse {}", entry.path().display()))?;
        out.push((mem, entry.path().to_path_buf()));
    }
    out.sort_by(|a, b| b.0.front.created_at.cmp(&a.0.front.created_at));
    Ok(out)
}

pub fn show(root: &Path, session_id: &str, id: &str) -> Result<(Memory, PathBuf)> {
    let path = paths::scratch_session_dir(root, session_id).join(format!("{id}.md"));
    if !path.exists() {
        return Err(anyhow!("scratch {id} not found in session {session_id}"));
    }
    let text = fs::read_to_string(&path)?;
    let mem = Memory::from_markdown(&text)?;
    Ok((mem, path))
}

pub fn clear(root: &Path, session_id: &str) -> Result<usize> {
    let dir = paths::scratch_session_dir(root, session_id);
    if !dir.exists() {
        return Ok(0);
    }
    let count = list(root, Some(session_id))?.len();
    fs::remove_dir_all(&dir).with_context(|| format!("rmdir {}", dir.display()))?;
    Ok(count)
}

pub fn remove(path: &Path) -> Result<()> {
    fs::remove_file(path).with_context(|| format!("remove {}", path.display()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memory::{Kind, Subject};

    #[test]
    fn write_then_list_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sid = "sess-abc";
        let mem = Memory::new(Kind::Semantic, Subject::user(), "scratch note");
        let path = write(root, sid, &mem).unwrap();
        assert!(path.exists());
        let listed = list(root, Some(sid)).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].0.front.id, mem.front.id);
    }

    #[test]
    fn clear_removes_session_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sid = "sess-x";
        for i in 0..3 {
            let m = Memory::new(Kind::Episodic, Subject::self_(), format!("note-{i}"));
            write(root, sid, &m).unwrap();
        }
        let n = clear(root, sid).unwrap();
        assert_eq!(n, 3);
        assert!(!paths::scratch_session_dir(root, sid).exists());
    }

    #[test]
    fn missing_session_lists_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let listed = list(tmp.path(), Some("nope")).unwrap();
        assert!(listed.is_empty());
    }
}
