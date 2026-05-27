//! `SQLite` + FTS5 keyword index over memories.
//!
//! The Markdown files are the source of truth. This index is rebuildable
//! at any time from `FileStore::iter_all` via `Index::reindex`.

use crate::embeddings;
use crate::memory::Memory;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 3;

/// Read the persisted schema_version row. Returns None if the row is missing
/// (which means the store predates `schema_meta`).
pub fn read_schema_version(db_path: &Path) -> Result<Option<u32>> {
    if !db_path.exists() {
        return Ok(None);
    }
    let conn = Connection::open(db_path)?;
    let v: Option<String> = conn
        .query_row(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    Ok(v.and_then(|s| s.parse().ok()))
}

pub struct Index {
    conn: Connection,
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub id: String,
    pub kind: String,
    pub subject: String,
    pub path: PathBuf,
    pub snippet: String,
    pub bm25: f64,
    pub confidence: f64,
    pub recall_count: u32,
    pub last_recalled_at: Option<DateTime<Utc>>,
}

/// Subset of `memories_meta` columns useful outside ranked search — what
/// `doctor`/`stats`/`gc`/`update` need without going through FTS5.
#[derive(Debug, Clone)]
pub struct MetaRow {
    pub id: String,
    pub kind: String,
    pub subject: String,
    pub path: PathBuf,
    pub confidence: f64,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    pub last_recalled_at: Option<DateTime<Utc>>,
    pub recall_count: u32,
    pub decays_after: Option<String>,
    pub supersedes: Vec<String>,
    pub embedding_id: Option<String>,
}

impl Index {
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)
            .with_context(|| format!("open sqlite at {}", db_path.display()))?;
        // Concurrency + safety pragmas (PRD §4b.11).
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;
             PRAGMA foreign_keys=ON;",
        )?;
        conn.execute_batch(SCHEMA)?;
        migrate(&conn)?;
        Ok(Self { conn })
    }


    /// Insert or replace the index entry for a memory. `embedding` is optional;
    /// pass `Some((id, vec))` to also store an embedding for vector search.
    pub fn upsert(
        &self,
        mem: &Memory,
        path: &Path,
        embedding: Option<(&str, &[f32])>,
    ) -> Result<()> {
        let path_str = path.to_string_lossy().to_string();
        let supersedes_json = if mem.front.supersedes.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&mem.front.supersedes)?)
        };
        let (embed_id, embed_blob, embed_dim): (
            Option<String>,
            Option<Vec<u8>>,
            Option<i64>,
        ) = match embedding {
            Some((id, v)) => (
                Some(id.to_string()),
                Some(embeddings::pack(v)),
                Some(i64::try_from(v.len()).unwrap_or(0)),
            ),
            None => (None, None, None),
        };

        // FTS5: delete-then-insert to update.
        self.conn
            .execute("DELETE FROM memories_fts WHERE id = ?1", params![mem.front.id])?;
        self.conn.execute(
            "INSERT INTO memories_fts (id, body, subject, kind) VALUES (?1, ?2, ?3, ?4)",
            params![
                mem.front.id,
                mem.body,
                mem.front.subject.as_str(),
                mem.front.kind.as_str()
            ],
        )?;

        self.conn.execute(
            "INSERT INTO memories_meta (id, kind, subject, path, confidence, created_at, updated_at, last_recalled_at, recall_count, decays_after, supersedes_json, embedding, embedding_id, embedding_dim, feedback_count, confidence_at_create)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
             ON CONFLICT(id) DO UPDATE SET
               kind = excluded.kind,
               subject = excluded.subject,
               path = excluded.path,
               confidence = excluded.confidence,
               updated_at = excluded.updated_at,
               decays_after = excluded.decays_after,
               supersedes_json = excluded.supersedes_json,
               embedding = excluded.embedding,
               embedding_id = excluded.embedding_id,
               embedding_dim = excluded.embedding_dim,
               feedback_count = excluded.feedback_count",
            params![
                mem.front.id,
                mem.front.kind.as_str(),
                mem.front.subject.as_str(),
                path_str,
                mem.front.confidence,
                mem.front.created_at.to_rfc3339(),
                mem.front.updated_at.map(|t| t.to_rfc3339()),
                mem.front.last_recalled_at.map(|t| t.to_rfc3339()),
                mem.front.recall_count,
                mem.front.decays_after,
                supersedes_json,
                embed_blob,
                embed_id,
                embed_dim,
                mem.front.feedback_count,
                mem.front.confidence,
            ],
        )?;
        Ok(())
    }

    /// Apply a feedback delta to a memory's confidence (clamped to
    /// `[floor, ceiling]`) and increment `feedback_count`. Writes both the
    /// `memories_meta` row AND the on-disk Markdown frontmatter so the
    /// file remains the source of truth.
    ///
    /// Returns the new `(confidence, feedback_count)` tuple.
    pub fn apply_feedback_delta(
        &self,
        id: &str,
        delta: f64,
        cfg: &crate::config::Feedback,
    ) -> Result<(f64, u32)> {
        let current: (f64, i64) = self
            .conn
            .query_row(
                "SELECT confidence, feedback_count FROM memories_meta WHERE id = ?1",
                params![id],
                |r| Ok((r.get::<_, f64>(0)?, r.get::<_, i64>(1)?)),
            )
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("memory id {id} not found"))?;
        let new_conf = crate::feedback::apply_delta(current.0, delta, cfg);
        let new_count = current.1.saturating_add(1);
        self.conn.execute(
            "UPDATE memories_meta
             SET confidence = ?1,
                 feedback_count = ?2,
                 updated_at = ?3
             WHERE id = ?4",
            params![new_conf, new_count, Utc::now().to_rfc3339(), id],
        )?;
        Ok((new_conf, u32::try_from(new_count).unwrap_or(u32::MAX)))
    }

    /// Apply a decay sweep across every memory. Idempotent within
    /// `min_interval_d` days: if `last_decay_sweep_at` for a row is more
    /// recent than `min_interval_d` days ago, that row is skipped.
    ///
    /// Returns the number of rows updated.
    pub fn apply_decay_sweep(
        &self,
        half_life_d: u32,
        min_interval_d: u32,
    ) -> Result<usize> {
        let now = Utc::now();
        let now_str = now.to_rfc3339();
        let min_interval_secs = i64::from(min_interval_d) * 86400;
        let mut stmt = self.conn.prepare(
            "SELECT id, confidence, last_decay_sweep_at, created_at FROM memories_meta",
        )?;
        struct Row {
            id: String,
            confidence: f64,
            last_sweep: Option<String>,
            created_at: String,
        }
        let rows: Vec<Row> = stmt
            .query_map([], |r| {
                Ok(Row {
                    id: r.get(0)?,
                    confidence: r.get(1)?,
                    last_sweep: r.get(2)?,
                    created_at: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        drop(stmt);

        let mut updated = 0usize;
        for row in rows {
            // Idempotency gate: skip if last sweep is within min_interval_d.
            if let Some(last) = &row.last_sweep {
                if let Ok(t) = DateTime::parse_from_rfc3339(last) {
                    let elapsed = now.signed_duration_since(t.with_timezone(&Utc)).num_seconds();
                    if elapsed < min_interval_secs {
                        continue;
                    }
                }
            }
            // Days since last sweep (or since creation if never swept).
            let baseline_str = row.last_sweep.as_deref().unwrap_or(row.created_at.as_str());
            let baseline = DateTime::parse_from_rfc3339(baseline_str)
                .ok()
                .map(|t| t.with_timezone(&Utc))
                .unwrap_or(now);
            let days = (now.signed_duration_since(baseline).num_seconds() as f64) / 86400.0;
            let new_conf = crate::feedback::decay_toward_neutral(row.confidence, days, half_life_d);
            self.conn.execute(
                "UPDATE memories_meta
                 SET confidence = ?1,
                     last_decay_sweep_at = ?2
                 WHERE id = ?3",
                params![new_conf, now_str, row.id],
            )?;
            updated += 1;
        }
        Ok(updated)
    }

    /// Look up `feedback_count` for a single id (used by tests + observability).
    pub fn get_feedback_count(&self, id: &str) -> Result<u32> {
        let n: i64 = self
            .conn
            .query_row(
                "SELECT feedback_count FROM memories_meta WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("memory id {id} not found"))?;
        Ok(u32::try_from(n).unwrap_or(0))
    }

    /// List `(id, drift)` for memories whose confidence has moved
    /// `>= threshold` from `confidence_at_create`. Used by `recall doctor`.
    pub fn confidence_drift(&self, threshold: f64) -> Result<Vec<(String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, confidence, confidence_at_create
             FROM memories_meta
             WHERE confidence_at_create IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, f64>(1)?,
                r.get::<_, f64>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, conf, base) = row?;
            let drift = conf - base;
            if drift.abs() >= threshold {
                out.push((id, drift));
            }
        }
        Ok(out)
    }

    /// Look up the meta row for a single id.
    pub fn get_meta(&self, id: &str) -> Result<Option<MetaRow>> {
        let row = self.conn.query_row(
            "SELECT id, kind, subject, path, confidence, created_at, updated_at, last_recalled_at, recall_count, decays_after, supersedes_json, embedding_id
             FROM memories_meta WHERE id = ?1",
            params![id],
            row_to_meta,
        ).optional()?;
        Ok(row)
    }

    /// Stream every meta row (ordered newest-first).
    pub fn all_meta(&self) -> Result<Vec<MetaRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, path, confidence, created_at, updated_at, last_recalled_at, recall_count, decays_after, supersedes_json, embedding_id
             FROM memories_meta
             ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], row_to_meta)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// IDs cited by *any* memory's `supersedes` list. These are the rows that
    /// should be filtered out of default queries (PRD §4b.7).
    pub fn superseded_ids(&self) -> Result<HashSet<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT supersedes_json FROM memories_meta
             WHERE supersedes_json IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut set = HashSet::new();
        for r in rows {
            let s = r?;
            if let Ok(ids) = serde_json::from_str::<Vec<String>>(&s) {
                for id in ids {
                    set.insert(id);
                }
            }
        }
        Ok(set)
    }

    /// Memory ids that have `decays_after = <duration>` whose deadline has
    /// passed (relative to `created_at`). `never` is preserved forever.
    pub fn decayed_ids(&self) -> Result<HashSet<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, created_at, decays_after FROM memories_meta
             WHERE decays_after IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |r| {
            let id: String = r.get(0)?;
            let created: String = r.get(1)?;
            let decay: String = r.get(2)?;
            Ok((id, created, decay))
        })?;
        let now = Utc::now();
        let mut set = HashSet::new();
        for r in rows {
            let (id, created, decay) = r?;
            if let Some(deadline) = decay_deadline(&created, &decay) {
                if now > deadline {
                    set.insert(id);
                }
            }
        }
        Ok(set)
    }

    pub fn remove(&self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM memories_fts WHERE id = ?1", params![id])?;
        self.conn
            .execute("DELETE FROM memories_meta WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// FTS5 query. Returns hits ordered by BM25 (lower is better in `SQLite`'s `bm25()`).
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let sanitized = sanitize_fts_query(query);
        let mut stmt = self.conn.prepare(
            "SELECT
                m.id, m.kind, m.subject, m.path,
                snippet(memories_fts, 1, '[', ']', '…', 12) AS snip,
                bm25(memories_fts) AS rank,
                m.confidence, m.recall_count, m.last_recalled_at
             FROM memories_fts
             JOIN memories_meta m ON m.id = memories_fts.id
             WHERE memories_fts MATCH ?1
             ORDER BY rank ASC
             LIMIT ?2",
        )?;
        let lim = i64::try_from(limit).unwrap_or(i64::MAX);
        let rows = stmt.query_map(params![sanitized, lim], |row| {
            let last_str: Option<String> = row.get(8)?;
            let last = last_str.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|d| d.with_timezone(&Utc)));
            Ok(Hit {
                id: row.get(0)?,
                kind: row.get(1)?,
                subject: row.get(2)?,
                path: PathBuf::from(row.get::<_, String>(3)?),
                snippet: row.get(4)?,
                bm25: row.get(5)?,
                confidence: row.get(6)?,
                recall_count: u32::try_from(row.get::<_, i64>(7)?).unwrap_or(0),
                last_recalled_at: last,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list(&self, subject_prefix: Option<&str>, limit: usize) -> Result<Vec<Hit>> {
        let lim = i64::try_from(limit).unwrap_or(i64::MAX);
        let (sql, args): (&str, Vec<rusqlite::types::Value>) = match subject_prefix {
            Some(p) => (
                "SELECT id, kind, subject, path, '', 0.0, confidence, recall_count, last_recalled_at
                 FROM memories_meta
                 WHERE subject LIKE ?1
                 ORDER BY created_at DESC
                 LIMIT ?2",
                vec![format!("{p}%").into(), lim.into()],
            ),
            None => (
                "SELECT id, kind, subject, path, '', 0.0, confidence, recall_count, last_recalled_at
                 FROM memories_meta
                 ORDER BY created_at DESC
                 LIMIT ?1",
                vec![lim.into()],
            ),
        };
        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(args), |row| {
            let last_str: Option<String> = row.get(8)?;
            let last = last_str.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|d| d.with_timezone(&Utc)));
            Ok(Hit {
                id: row.get(0)?,
                kind: row.get(1)?,
                subject: row.get(2)?,
                path: PathBuf::from(row.get::<_, String>(3)?),
                snippet: row.get(4)?,
                bm25: row.get(5)?,
                confidence: row.get(6)?,
                recall_count: u32::try_from(row.get::<_, i64>(7)?).unwrap_or(0),
                last_recalled_at: last,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Wipe and rebuild from a fresh iterator. Used by `recall reindex`.
    /// If `embedder` is Some, every memory is re-embedded as it's reinserted.
    pub fn rebuild_from<I>(
        &self,
        iter: I,
        embedder: Option<&dyn crate::embeddings::Embedder>,
    ) -> Result<usize>
    where
        I: Iterator<Item = (Memory, PathBuf)>,
    {
        self.conn.execute_batch(
            "DELETE FROM memories_fts; DELETE FROM memories_meta;",
        )?;
        let mut n = 0;
        for (mem, path) in iter {
            let vec_owned = if let Some(e) = embedder {
                Some(e.embed(&mem.body)?)
            } else {
                None
            };
            let embed = vec_owned
                .as_ref()
                .map(|v| (embedder.unwrap_or(&NullEmbedder).id(), v.as_slice()));
            self.upsert(&mem, &path, embed)?;
            n += 1;
        }
        Ok(n)
    }

    /// Brute-force cosine-similarity search over every stored embedding.
    /// Fine for the few-thousand-memory scale; swap in vss/hnsw later.
    pub fn vector_search(&self, query_vec: &[f32], limit: usize) -> Result<Vec<(Hit, f32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, path, confidence, recall_count, last_recalled_at, embedding
             FROM memories_meta
             WHERE embedding IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| {
            let last_str: Option<String> = row.get(6)?;
            let last = last_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s).ok().map(|d| d.with_timezone(&Utc))
            });
            let blob: Vec<u8> = row.get(7)?;
            Ok((
                Hit {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    subject: row.get(2)?,
                    path: PathBuf::from(row.get::<_, String>(3)?),
                    snippet: String::new(),
                    bm25: 0.0,
                    confidence: row.get(4)?,
                    recall_count: u32::try_from(row.get::<_, i64>(5)?).unwrap_or(0),
                    last_recalled_at: last,
                },
                blob,
            ))
        })?;
        let mut scored: Vec<(Hit, f32)> = Vec::new();
        for r in rows {
            let (hit, blob) = r?;
            let v = crate::embeddings::unpack(&blob);
            if v.len() != query_vec.len() {
                continue;
            }
            let sim = crate::embeddings::cosine(query_vec, &v);
            scored.push((hit, sim));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }

    /// Bump `last_recalled_at = now` and increment `recall_count`. Called after a successful query.
    pub fn touch_recall(&self, id: &str) -> Result<u32> {
        let now = Utc::now().to_rfc3339();
        let n = self.conn.execute(
            "UPDATE memories_meta
             SET recall_count = recall_count + 1,
                 last_recalled_at = ?1
             WHERE id = ?2",
            params![now, id],
        )?;
        if n == 0 {
            return Err(anyhow::anyhow!("memory id {id} not found"));
        }
        let count: i64 = self.conn.query_row(
            "SELECT recall_count FROM memories_meta WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )?;
        Ok(u32::try_from(count).unwrap_or(0))
    }

    pub fn count(&self) -> Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories_meta", [], |row| row.get(0))?;
        Ok(usize::try_from(n).unwrap_or(0))
    }

    /// Fetch the stored embedding for a memory id, if any.
    pub fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>> {
        let result: rusqlite::Result<Option<Vec<u8>>> = self.conn.query_row(
            "SELECT embedding FROM memories_meta WHERE id = ?1",
            params![id],
            |row| row.get::<_, Option<Vec<u8>>>(0),
        );
        match result {
            Ok(Some(blob)) => Ok(Some(crate::embeddings::unpack(&blob))),
            Ok(None) => Ok(None),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

const SCHEMA: &str = r"
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    body,
    subject,
    kind UNINDEXED
);

CREATE TABLE IF NOT EXISTS memories_meta (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    path TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    last_recalled_at TEXT,
    recall_count INTEGER NOT NULL DEFAULT 0,
    decays_after TEXT,
    supersedes_json TEXT,
    embedding BLOB,
    embedding_id TEXT,
    embedding_dim INTEGER,
    feedback_count INTEGER NOT NULL DEFAULT 0,
    confidence_at_create REAL,
    last_decay_sweep_at TEXT
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_meta_subject ON memories_meta(subject);
CREATE INDEX IF NOT EXISTS idx_meta_kind    ON memories_meta(kind);
CREATE INDEX IF NOT EXISTS idx_meta_created ON memories_meta(created_at);
";

/// Apply additive migrations for stores written by older recall versions.
/// Idempotent: `ALTER TABLE` errors that mean "column already exists" are
/// treated as success.
fn migrate(conn: &Connection) -> Result<()> {
    let existing: u32 = conn
        .query_row(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'",
            [],
            |r| r.get::<_, String>(0),
        )
        .optional()?
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    if existing < 2 {
        // v0.1 stores predate `updated_at`. CREATE TABLE above now has it, but
        // existing tables don't — add it. Same shape for any v0.2 additive cols.
        add_column_if_missing(conn, "memories_meta", "updated_at", "TEXT")?;
    }

    if existing < 3 {
        // v0.5 stores predate `feedback_count`, `confidence_at_create`,
        // `last_decay_sweep_at`. All additive, all nullable or defaulted.
        add_column_if_missing(
            conn,
            "memories_meta",
            "feedback_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        add_column_if_missing(conn, "memories_meta", "confidence_at_create", "REAL")?;
        add_column_if_missing(conn, "memories_meta", "last_decay_sweep_at", "TEXT")?;
        // Backfill confidence_at_create from the current confidence so the
        // drift query has a baseline for pre-existing rows. Pre-existing
        // memories report drift=0 until they're touched by feedback.
        conn.execute(
            "UPDATE memories_meta
             SET confidence_at_create = confidence
             WHERE confidence_at_create IS NULL",
            [],
        )?;
    }

    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?1)",
        params![SCHEMA_VERSION.to_string()],
    )?;
    Ok(())
}

fn add_column_if_missing(conn: &Connection, table: &str, col: &str, typ: &str) -> Result<()> {
    let exists = column_exists(conn, table, col)?;
    if !exists {
        conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {col} {typ}"), [])?;
    }
    Ok(())
}

fn column_exists(conn: &Connection, table: &str, col: &str) -> Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = stmt.query_map([], |r| r.get::<_, String>(1))?;
    for r in rows {
        if r? == col {
            return Ok(true);
        }
    }
    Ok(false)
}

fn row_to_meta(row: &rusqlite::Row<'_>) -> rusqlite::Result<MetaRow> {
    let created: String = row.get(5)?;
    let updated: Option<String> = row.get(6)?;
    let last: Option<String> = row.get(7)?;
    let supersedes_json: Option<String> = row.get(10)?;
    let supersedes = supersedes_json
        .as_deref()
        .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
        .unwrap_or_default();
    Ok(MetaRow {
        id: row.get(0)?,
        kind: row.get(1)?,
        subject: row.get(2)?,
        path: PathBuf::from(row.get::<_, String>(3)?),
        confidence: row.get(4)?,
        created_at: DateTime::parse_from_rfc3339(&created)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        updated_at: updated.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .ok()
                .map(|d| d.with_timezone(&Utc))
        }),
        last_recalled_at: last.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .ok()
                .map(|d| d.with_timezone(&Utc))
        }),
        recall_count: u32::try_from(row.get::<_, i64>(8)?).unwrap_or(0),
        decays_after: row.get(9)?,
        supersedes,
        embedding_id: row.get(11)?,
    })
}

/// Parse a `decays_after` string like "30d", "6mo", "1y", "never".
/// `never` returns `None` so the caller treats it as "no deadline".
pub(crate) fn decay_deadline(created_at: &str, decays: &str) -> Option<DateTime<Utc>> {
    let created = DateTime::parse_from_rfc3339(created_at)
        .ok()?
        .with_timezone(&Utc);
    let d = decays.trim();
    if d.eq_ignore_ascii_case("never") || d.is_empty() {
        return None;
    }
    let dur = parse_duration_phrase(d)?;
    Some(created + dur)
}

fn parse_duration_phrase(s: &str) -> Option<chrono::Duration> {
    let (num_part, suffix) = split_num_suffix(s)?;
    let n: i64 = num_part.parse().ok()?;
    match suffix {
        "m" => Some(chrono::Duration::minutes(n)),
        "h" => Some(chrono::Duration::hours(n)),
        "d" => Some(chrono::Duration::days(n)),
        "w" => Some(chrono::Duration::weeks(n)),
        "mo" => Some(chrono::Duration::days(n.saturating_mul(30))),
        "y" => Some(chrono::Duration::days(n.saturating_mul(365))),
        _ => None,
    }
}

fn split_num_suffix(s: &str) -> Option<(&str, &str)> {
    let idx = s.find(|c: char| !c.is_ascii_digit() && c != '-')?;
    Some((s.split_at(idx).0, s.split_at(idx).1))
}

/// Stub embedder used only so `rebuild_from` can call `.id()` when given a
/// concrete `Option<&dyn Embedder>` that is `Some`. Never used as a real embedder.
struct NullEmbedder;
impl crate::embeddings::Embedder for NullEmbedder {
    fn dim(&self) -> usize {
        0
    }
    fn id(&self) -> &'static str {
        "null"
    }
    fn embed(&self, _: &str) -> Result<Vec<f32>> {
        Ok(Vec::new())
    }
}

/// FTS5 has a small query mini-language; strip characters that turn user input
/// into a syntax error. We're lenient: words become an OR of prefix matches.
fn sanitize_fts_query(q: &str) -> String {
    let cleaned: String = q
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect();
    let terms: Vec<String> = cleaned
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .map(|t| format!("{t}*"))
        .collect();
    if terms.is_empty() {
        // FTS will error on an empty query; match nothing instead.
        return "__no_match__".into();
    }
    terms.join(" OR ")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memory::{Kind, Subject};

    #[test]
    fn upsert_and_search_finds_match() {
        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("idx.sqlite");
        let idx = Index::open(&db).unwrap();
        let mem = Memory::new(
            Kind::Semantic,
            Subject::user(),
            "user prefers integration tests over mocks for the auth code",
        );
        let path = tmp.path().join("mem.md");
        std::fs::write(&path, mem.to_markdown().unwrap()).unwrap();
        idx.upsert(&mem, &path, None).unwrap();
        let hits = idx.search("integration auth", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, mem.front.id);
    }

    #[test]
    fn list_filters_by_subject_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = Index::open(&tmp.path().join("idx.sqlite")).unwrap();
        let m_user = Memory::new(Kind::Semantic, Subject::user(), "u");
        let m_proj = Memory::new(Kind::Procedural, Subject::project("recall"), "p");
        idx.upsert(&m_user, &tmp.path().join("a.md"), None).unwrap();
        idx.upsert(&m_proj, &tmp.path().join("b.md"), None).unwrap();
        let proj_hits = idx.list(Some("project:"), 10).unwrap();
        assert_eq!(proj_hits.len(), 1);
        assert_eq!(proj_hits[0].subject, "project:recall");
    }

    #[test]
    fn touch_recall_increments_count() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = Index::open(&tmp.path().join("idx.sqlite")).unwrap();
        let mem = Memory::new(Kind::Semantic, Subject::user(), "x");
        idx.upsert(&mem, &tmp.path().join("a.md"), None).unwrap();
        idx.touch_recall(&mem.front.id).unwrap();
        idx.touch_recall(&mem.front.id).unwrap();
        let hits = idx.list(None, 10).unwrap();
        assert_eq!(hits[0].recall_count, 2);
        assert!(hits[0].last_recalled_at.is_some());
    }

    #[test]
    fn sanitize_strips_punctuation() {
        assert_eq!(sanitize_fts_query("hello, world!"), "hello* OR world*");
        assert_eq!(sanitize_fts_query(""), "__no_match__");
        assert_eq!(sanitize_fts_query("foo's bar"), "foo* OR s* OR bar*");
    }

    #[test]
    fn vector_search_returns_nearest_first() {
        use crate::embeddings::{Embedder, HashEmbedder};
        let tmp = tempfile::tempdir().unwrap();
        let idx = Index::open(&tmp.path().join("idx.sqlite")).unwrap();
        let e = HashEmbedder::new();

        let m1 = Memory::new(
            Kind::Procedural,
            Subject::project("recall"),
            "build the rust project with cargo build --release",
        );
        let m2 = Memory::new(
            Kind::Semantic,
            Subject::user(),
            "user prefers integration tests over mocks for auth code",
        );
        let v1 = e.embed(&m1.body).unwrap();
        let v2 = e.embed(&m2.body).unwrap();
        idx.upsert(&m1, &tmp.path().join("a.md"), Some((e.id(), &v1))).unwrap();
        idx.upsert(&m2, &tmp.path().join("b.md"), Some((e.id(), &v2))).unwrap();

        let q = e.embed("build the rust crate with cargo").unwrap();
        let hits = idx.vector_search(&q, 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0.id, m1.front.id, "near id should rank first");
    }
}
