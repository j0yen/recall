//! Memory schema and Markdown frontmatter (de)serialization.
//!
//! Memories are plain Markdown files with a YAML frontmatter header.
//! That keeps every memory `grep`-able and human-editable; the `SQLite`
//! index is purely auxiliary.

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use ulid::Ulid;

/// Memory kind. See PRD §4.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// "On <date> I tried X; outcome Y." Append-only.
    Episodic,
    /// Distilled facts: "user prefers integration tests over mocks."
    Semantic,
    /// "To run the harness for this repo: …"
    Procedural,
    /// "When I asked too many clarifying questions, the user said 'just do it.'"
    Reflective,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
            Self::Reflective => "reflective",
        }
    }
}

impl FromStr for Kind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "episodic" => Ok(Self::Episodic),
            "semantic" => Ok(Self::Semantic),
            "procedural" => Ok(Self::Procedural),
            "reflective" => Ok(Self::Reflective),
            other => Err(anyhow!("unknown memory kind: {other}")),
        }
    }
}

/// Subject namespace. `user`, `self`, `project:<slug>`, `tool:<name>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Subject(pub String);

impl Subject {
    pub fn user() -> Self {
        Self("user".into())
    }
    pub fn self_() -> Self {
        Self("self".into())
    }
    pub fn project(slug: &str) -> Self {
        Self(format!("project:{slug}"))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn namespace(&self) -> &str {
        self.0.split(':').next().unwrap_or(&self.0)
    }
}

/// Pointer back to the source of a memory — a session turn excerpt, a file/line, etc.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Evidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub excerpt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
}

/// YAML frontmatter for a memory file. The body of the file follows after `---\n`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frontmatter {
    pub id: String,
    pub kind: Kind,
    pub subject: Subject,
    #[serde(default)]
    pub evidence: Vec<Evidence>,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_recalled_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub recall_count: u32,
    #[serde(default)]
    pub feedback_count: u32,
    #[serde(default)]
    pub surfaced_count: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supersedes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decays_after: Option<String>,
    /// Session id that originally wrote this memory (PRD-recall-session-stamp
    /// §2.1). Optional & schema-additive: files written by recall < v0.7.0
    /// parse back as `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub written_by_session: Option<String>,
    /// Intent of the writing session (e.g. `/build`). Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub written_by_intent: Option<String>,
    /// Session id of the most recent `touch` / `update`. Updated on touch;
    /// never fabricated for memories of unknown origin (PRD §2.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_touched_by_session: Option<String>,
}

fn default_confidence() -> f64 {
    0.5
}

/// A memory in memory (heh). Body is the Markdown content following the frontmatter.
#[derive(Debug, Clone)]
pub struct Memory {
    pub front: Frontmatter,
    pub body: String,
}

impl Memory {
    /// Create a new memory with a fresh ULID id and `created_at = now`.
    pub fn new(kind: Kind, subject: Subject, body: impl Into<String>) -> Self {
        Self {
            front: Frontmatter {
                id: Ulid::new().to_string(),
                kind,
                subject,
                evidence: Vec::new(),
                confidence: default_confidence(),
                created_at: Utc::now(),
                updated_at: None,
                last_recalled_at: None,
                recall_count: 0,
                feedback_count: 0,
                surfaced_count: 0,
                supersedes: Vec::new(),
                decays_after: None,
                written_by_session: None,
                written_by_intent: None,
                last_touched_by_session: None,
            },
            body: body.into(),
        }
    }

    /// Render to the on-disk format: YAML frontmatter + `---\n` + body.
    pub fn to_markdown(&self) -> Result<String> {
        let yaml = serde_yaml::to_string(&self.front)
            .context("serialize frontmatter")?;
        Ok(format!("---\n{yaml}---\n\n{}", self.body.trim_end_matches('\n')))
    }

    /// Parse from the on-disk format.
    pub fn from_markdown(text: &str) -> Result<Self> {
        let stripped = text.strip_prefix("---\n").ok_or_else(|| {
            anyhow!("memory file missing leading `---` frontmatter delimiter")
        })?;
        let end_idx = stripped
            .find("\n---")
            .ok_or_else(|| anyhow!("memory file missing closing `---` delimiter"))?;
        let (yaml, rest) = stripped.split_at(end_idx);
        let front: Frontmatter =
            serde_yaml::from_str(yaml).context("parse frontmatter YAML")?;
        // skip the closing `\n---` and any following newline
        let body = rest
            .strip_prefix("\n---")
            .unwrap_or(rest)
            .trim_start_matches('\n')
            .to_string();
        Ok(Self { front, body })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_to_and_from_markdown() {
        let m = Memory::new(
            Kind::Semantic,
            Subject::user(),
            "user prefers integration tests over mocks",
        );
        let md = m.to_markdown().unwrap();
        let back = Memory::from_markdown(&md).unwrap();
        assert_eq!(back.front.id, m.front.id);
        assert_eq!(back.front.kind, Kind::Semantic);
        assert_eq!(back.front.subject.as_str(), "user");
        assert_eq!(back.body.trim(), "user prefers integration tests over mocks");
    }

    #[test]
    fn subject_namespace_extraction() {
        assert_eq!(Subject::project("autobuilder").namespace(), "project");
        assert_eq!(Subject::user().namespace(), "user");
        assert_eq!(Subject::self_().namespace(), "self");
    }

    #[test]
    fn kind_roundtrip() {
        for k in [Kind::Episodic, Kind::Semantic, Kind::Procedural, Kind::Reflective] {
            assert_eq!(Kind::from_str(k.as_str()).unwrap(), k);
        }
    }

    /// AC3 (PRD-recall-surfaced-tracking) — `surfaced_count` survives a
    /// frontmatter round-trip. Verifies the field serializes and the
    /// `#[serde(default)]` reads back the explicit value (not the default).
    #[test]
    fn surfaced_count_roundtrip_markdown() {
        let mut m = Memory::new(
            Kind::Semantic,
            Subject::user(),
            "surface tracking smoke",
        );
        m.front.surfaced_count = 7;
        let md = m.to_markdown().unwrap();
        let back = Memory::from_markdown(&md).unwrap();
        assert_eq!(back.front.surfaced_count, 7);
        // Independent of the other counters.
        assert_eq!(back.front.feedback_count, 0);
        assert_eq!(back.front.recall_count, 0);
    }

    /// AC3 (companion) — older markdown without `surfaced_count:` reads
    /// back as 0. Covers the `#[serde(default)]` migration path for files
    /// written by recall < v0.7.x.
    #[test]
    fn surfaced_count_defaults_when_absent_from_markdown() {
        let mut m = Memory::new(Kind::Semantic, Subject::user(), "legacy body");
        m.front.surfaced_count = 9;
        let md = m.to_markdown().unwrap();
        // Strip the field from the serialized YAML to simulate a file that
        // predates the column. Match the whole line including trailing
        // newline so we don't leave a blank line in the frontmatter.
        let mut stripped = String::with_capacity(md.len());
        for line in md.lines() {
            if line.trim_start().starts_with("surfaced_count:") {
                continue;
            }
            stripped.push_str(line);
            stripped.push('\n');
        }
        let back = Memory::from_markdown(&stripped).unwrap();
        assert_eq!(back.front.surfaced_count, 0);
    }

    /// AC2/AC3 (PRD-recall-session-stamp) — the three session fields survive a
    /// frontmatter round-trip, and a memory written without them parses back
    /// as `None` (schema-additive, no migration needed).
    #[test]
    fn session_stamp_fields_roundtrip_and_default_none() {
        // Default: no stamp → fields are None and omitted from the YAML.
        let m = Memory::new(Kind::Reflective, Subject::self_(), "no stamp body");
        assert!(m.front.written_by_session.is_none());
        let md = m.to_markdown().unwrap();
        assert!(
            !md.contains("written_by_session"),
            "unstamped memory must not emit the field; got:\n{md}"
        );
        let back = Memory::from_markdown(&md).unwrap();
        assert!(back.front.written_by_session.is_none());
        assert!(back.front.written_by_intent.is_none());
        assert!(back.front.last_touched_by_session.is_none());

        // Stamped: all three round-trip.
        let mut s = Memory::new(Kind::Reflective, Subject::self_(), "stamped body");
        s.front.written_by_session = Some("deadbeef0123".into());
        s.front.written_by_intent = Some("/build".into());
        s.front.last_touched_by_session = Some("deadbeef0123".into());
        let md = s.to_markdown().unwrap();
        let back = Memory::from_markdown(&md).unwrap();
        assert_eq!(back.front.written_by_session.as_deref(), Some("deadbeef0123"));
        assert_eq!(back.front.written_by_intent.as_deref(), Some("/build"));
        assert_eq!(
            back.front.last_touched_by_session.as_deref(),
            Some("deadbeef0123")
        );
    }
}
