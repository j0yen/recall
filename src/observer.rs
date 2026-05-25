//! Phase 4 PostToolUse observer (MVP).
//!
//! Reads one or more PostToolUse hook events on stdin (one JSON object per
//! line). Applies a tiny heuristic catalog to decide whether the event
//! sequence is "memory-worthy," and parks any matches as draft memories
//! under `<root>/proposals/` for the user to review with `recall promote`
//! or discard with `recall delete`.
//!
//! This is intentionally conservative: false positives waste the user's
//! attention more than false negatives waste recall's memory.

use crate::memory::{Kind, Memory, Subject};
use crate::paths;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct Event {
    /// Tool name (e.g. "Edit", "Bash"). Required.
    pub tool_name: String,
    /// Tool input as JSON. May be empty.
    #[serde(default)]
    pub tool_input: serde_json::Value,
    /// Tool output as JSON or string. May be empty.
    #[serde(default)]
    pub tool_response: serde_json::Value,
    /// Result of the tool call. "ok" | "error" | other.
    #[serde(default)]
    pub status: String,
    /// Optional surrounding context.
    #[serde(default)]
    pub user_prompt_after: Option<String>,
}

#[derive(Debug)]
pub struct Proposal {
    pub body: String,
    pub kind: Kind,
    pub subject: Subject,
    pub reason: String,
}

/// Apply heuristics to one event. Returns at most one proposal.
pub fn classify(ev: &Event) -> Option<Proposal> {
    // Heuristic 1: tool errored, then the user re-prompted with corrective
    // language. The error is worth remembering if we can phrase it well.
    if ev.status.eq_ignore_ascii_case("error") {
        if let Some(p) = ev.user_prompt_after.as_deref() {
            if has_corrective_language(p) {
                return Some(Proposal {
                    body: format!(
                        "{} call failed and the user corrected with: {:?}\n\nTool error: {}",
                        ev.tool_name,
                        first_n_chars(p, 200),
                        first_n_chars(&ev.tool_response.to_string(), 400),
                    ),
                    kind: Kind::Reflective,
                    subject: Subject::self_(),
                    reason: "tool error followed by corrective user prompt".into(),
                });
            }
        }
    }

    // Heuristic 2: Edit immediately reverted (Edit then Edit on same file
    // with the old_string / new_string swapped). The hook delivers one event
    // at a time, so a *sequence* heuristic would need history. v0.4 leaves
    // that for the caller; this single-event observer only handles the two
    // single-event signals above.

    None
}

fn has_corrective_language(prompt: &str) -> bool {
    let lower = prompt.to_lowercase();
    const NEEDLES: &[&str] = &[
        "no wait", "undo", "revert", "that's wrong", "thats wrong", "don't",
        "stop", "actually", "no, ", "rollback", "no, i meant",
    ];
    NEEDLES.iter().any(|n| lower.contains(n))
}

fn first_n_chars(s: &str, n: usize) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i >= n {
            out.push('…');
            break;
        }
        out.push(c);
    }
    out
}

/// Stream events from stdin (or any `BufRead`). For each event, classify and
/// (if a proposal results) park it under `proposals/`. Returns the proposal
/// paths written.
pub fn run<R: BufRead>(root: &Path, reader: R) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(paths::proposals_dir(root))?;
    let mut written = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let ev: Event = serde_json::from_str(&line)
            .with_context(|| format!("parse event line {}", lineno + 1))?;
        if let Some(p) = classify(&ev) {
            let mut mem = Memory::new(p.kind, p.subject, &p.body);
            mem.front.confidence = 0.4;
            let path = park_proposal(root, &mem, &p.reason)?;
            written.push(path);
        }
    }
    Ok(written)
}

fn park_proposal(root: &Path, mem: &Memory, reason: &str) -> Result<PathBuf> {
    let dir = paths::proposals_dir(root);
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.md", mem.front.id));
    let mut text = mem.to_markdown()?;
    text.push_str(&format!("\n\n<!-- observer reason: {reason} -->\n"));
    let tmp = dir.join(format!(".{}.md.tmp", mem.front.id));
    fs::write(&tmp, &text)?;
    fs::rename(&tmp, &path)?;
    Ok(path)
}

/// List parked proposals (newest first).
pub fn list_proposals(root: &Path) -> Result<Vec<(Memory, PathBuf)>> {
    let dir = paths::proposals_dir(root);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() && entry.path().extension().is_some_and(|x| x == "md") {
            let text = fs::read_to_string(entry.path())?;
            let mem = Memory::from_markdown(&text)?;
            out.push((mem, entry.path()));
        }
    }
    out.sort_by(|a, b| b.0.front.created_at.cmp(&a.0.front.created_at));
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn corrective_language_detected() {
        assert!(has_corrective_language("no wait, run the other one"));
        assert!(has_corrective_language("Actually, that's wrong"));
        assert!(has_corrective_language("undo that please"));
        assert!(!has_corrective_language("looks great, keep going"));
    }

    #[test]
    fn error_plus_correction_yields_proposal() {
        let ev = Event {
            tool_name: "Bash".into(),
            tool_input: serde_json::json!({"command":"rm -rf /"}),
            tool_response: serde_json::json!("permission denied"),
            status: "error".into(),
            user_prompt_after: Some("no wait — never run that".into()),
        };
        let p = classify(&ev).expect("should propose");
        assert_eq!(p.kind, Kind::Reflective);
    }

    #[test]
    fn ok_call_yields_nothing() {
        let ev = Event {
            tool_name: "Read".into(),
            tool_input: serde_json::Value::Null,
            tool_response: serde_json::Value::Null,
            status: "ok".into(),
            user_prompt_after: None,
        };
        assert!(classify(&ev).is_none());
    }

    #[test]
    fn run_writes_proposal_file() {
        let tmp = tempfile::tempdir().unwrap();
        let input = r#"{"tool_name":"Edit","status":"error","tool_input":{},"tool_response":"file not found","user_prompt_after":"no wait, the path was different"}"#;
        let written = run(tmp.path(), Cursor::new(input)).unwrap();
        assert_eq!(written.len(), 1);
        assert!(written[0].exists());
    }
}
