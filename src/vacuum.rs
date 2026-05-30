//! Corpus-vacuum sweep: identify and optionally act on high-surface,
//! zero-use memories (PRD-recall-corpus-vacuum).
//!
//! `recall vacuum` lists candidates matching
//! `surfaced_count >= min_surfaced AND recall_count <= max_used`,
//! then optionally applies one of three actions:
//! - `decay`     — reduces confidence by `decay_amount` (default, recoverable)
//! - `supersede` — writes a proposal file for user review
//! - `archive`   — moves the file to `memories-archive/` and removes the DB row

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use std::path::Path;

// ──────────────────────────────────────────────────────────────
// Candidate representation
// ──────────────────────────────────────────────────────────────

/// One candidate memory returned by the vacuum query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VacuumCandidate {
    pub id: String,
    pub kind: String,
    pub subject: String,
    pub surfaced: u32,
    pub used: u32,
    pub confidence_before: f64,
    /// Projected confidence after decay (dry-run) or actual after apply.
    /// For `supersede` and `archive` actions, this equals `confidence_before`.
    pub confidence_after: f64,
    /// `null` in dry-run, set to the action string when applied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_applied: Option<String>,
}

// ──────────────────────────────────────────────────────────────
// Config for the vacuum sweep (from recall.toml `[vacuum]`)
// ──────────────────────────────────────────────────────────────

/// `[vacuum]` section of `recall.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VacuumConfig {
    /// Default action when `--apply` is passed without `--action`.
    pub default_action: String,
    /// `surfaced_count` must be >= this to be a candidate (default 20).
    pub min_surfaced: u32,
    /// `recall_count` (used) must be <= this to be a candidate (default 0).
    pub max_used: u32,
    /// How much to subtract from confidence per `decay` sweep (default 0.10).
    pub decay_amount: f64,
}

impl Default for VacuumConfig {
    fn default() -> Self {
        Self {
            default_action: "decay".to_string(),
            min_surfaced: 20,
            max_used: 0,
            decay_amount: 0.10,
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Candidate query
// ──────────────────────────────────────────────────────────────

/// Query `memories_meta` for vacuum candidates.
///
/// Returns rows where `surfaced_count >= min_surfaced AND
/// recall_count <= max_used`, ordered by `surfaced_count DESC`.
pub fn query_candidates(
    conn: &rusqlite::Connection,
    min_surfaced: u32,
    max_used: u32,
) -> Result<Vec<VacuumCandidate>> {
    let mut stmt = conn.prepare(
        "SELECT id, kind, subject, confidence, surfaced_count, recall_count
         FROM memories_meta
         WHERE surfaced_count >= ?1 AND recall_count <= ?2
         ORDER BY surfaced_count DESC",
    )?;
    // Safety note: i64::from(u32) is always safe — u32::MAX < i64::MAX.
    let rows = stmt
        .query_map(params![i64::from(min_surfaced), i64::from(max_used)], |row| {
            let conf: f64 = row.get(3)?;
            let surfaced: i64 = row.get(4)?;
            let used: i64 = row.get(5)?;
            Ok(VacuumCandidate {
                id: row.get(0)?,
                kind: row.get(1)?,
                subject: row.get(2)?,
                surfaced: u32::try_from(surfaced).unwrap_or(u32::MAX),
                used: u32::try_from(used).unwrap_or(u32::MAX),
                confidence_before: conf,
                confidence_after: conf,
                action_applied: None,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()
        .context("query vacuum candidates")?;
    Ok(rows)
}

// ──────────────────────────────────────────────────────────────
// Action: decay
// ──────────────────────────────────────────────────────────────

/// Apply confidence decay to a single candidate.
///
/// Subtracts `decay_amount` from the candidate's confidence, floored at 0.05.
/// Updates both `memories_meta` (via `idx`) and the markdown file (via `store`).
/// Returns the new confidence.
pub fn apply_decay(
    idx: &crate::index::Index,
    store: &crate::store::FileStore,
    candidate: &VacuumCandidate,
    decay_amount: f64,
) -> Result<f64> {
    let new_conf = idx.vacuum_apply_decay_sql(
        &candidate.id,
        candidate.confidence_before,
        decay_amount,
    )?;
    // Mirror to markdown file so it stays the source of truth.
    let (mut mem, path) = store.find_by_id(&candidate.id)?;
    mem.front.confidence = new_conf;
    mem.front.updated_at = Some(Utc::now());
    store.overwrite(&path, &mem)?;
    Ok(new_conf)
}

// ──────────────────────────────────────────────────────────────
// Action: archive
// ──────────────────────────────────────────────────────────────

/// Archive a candidate memory: move its file to `memories-archive/<kind>/<id>.md`
/// and remove its row from `memories_meta`.
///
/// `src_path` is the on-disk path of the memory file (from `store.find_by_id`
/// or the `path` column in `memories_meta`). The caller resolves it before
/// calling this function.
///
/// Atomicity caveat: `rename` succeeds before SQLite delete. On crash between
/// the two, `recall doctor` will report a file-vs-index divergence which the
/// user can hand-resolve.
pub fn apply_archive(
    idx: &crate::index::Index,
    root: &Path,
    candidate: &VacuumCandidate,
    src_path: &Path,
) -> Result<()> {
    let dst = root
        .join("memories-archive")
        .join(&candidate.kind)
        .join(format!("{}.md", candidate.id));

    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create archive dir {}", parent.display()))?;
    }
    if src_path.exists() {
        std::fs::rename(src_path, &dst)
            .with_context(|| format!("rename {} to {}", src_path.display(), dst.display()))?;
    }
    idx.vacuum_remove(&candidate.id)
        .context("remove row after archive")?;
    Ok(())
}

// ──────────────────────────────────────────────────────────────
// Action: supersede-proposal
// ──────────────────────────────────────────────────────────────

/// Write a vacuum-proposal file under `<root>/proposals/<ULID>.md`.
/// Does NOT modify the memory — the user reviews and applies later.
pub fn apply_supersede_proposal(
    root: &Path,
    candidate: &VacuumCandidate,
    body_preview: &str,
) -> Result<String> {
    use ulid::Ulid;
    let proposal_id = Ulid::new().to_string();
    let now = Utc::now();
    let proposals_dir = root.join("proposals");
    std::fs::create_dir_all(&proposals_dir)
        .with_context(|| format!("create proposals dir {}", proposals_dir.display()))?;
    let path = proposals_dir.join(format!("{proposal_id}.md"));
    let preview = if body_preview.len() > 300 {
        &body_preview[..300]
    } else {
        body_preview
    };
    let content = format!(
        "---\n\
         id: {proposal_id}\n\
         proposal_type: vacuum\n\
         target_id: {target_id}\n\
         target_subject: {target_subject}\n\
         surfaced: {surfaced}\n\
         used: {used}\n\
         created_at: {created_at}\n\
         pending: true\n\
         ---\n\
         \n\
         # Vacuum candidate: {target_id}\n\
         \n\
         Surfaced {surfaced} times across recent sessions with {used} evidence of use.\n\
         Suggested action: delete or supersede with a more accurate variant.\n\
         \n\
         Original body:\n\
         > {preview}\n\
         \n\
         Apply with:\n\
           recall update {target_id} --supersedes {proposal_id}\n\
           # OR\n\
           recall feedback --reject {target_id}\n\
         \n\
         Discard this proposal by deleting this file.\n",
        proposal_id = proposal_id,
        target_id = candidate.id,
        target_subject = candidate.subject,
        surfaced = candidate.surfaced,
        used = candidate.used,
        created_at = now.to_rfc3339(),
        preview = preview,
    );
    std::fs::write(&path, content.as_bytes())
        .with_context(|| format!("write proposal {}", path.display()))?;
    Ok(proposal_id)
}

// ──────────────────────────────────────────────────────────────
// Result summary
// ──────────────────────────────────────────────────────────────

/// Top-level result returned by a vacuum run.
#[derive(Debug, Serialize)]
pub struct VacuumResult {
    pub candidates: usize,
    pub would_apply: String,
    pub dry_run: bool,
    pub memories: Vec<VacuumCandidate>,
}
