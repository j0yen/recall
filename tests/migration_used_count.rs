//! AC1 (PRD-recall-stop-hook-discriminate) — schema migration adds `used_count`.
//!
//! Open a v0.7.2-shaped DB (schema_version=4, no `used_count` column),
//! verify the column appears after `Index::open`. Re-opening the migrated DB
//! is a no-op: `used_count` stays at the seeded value.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::index::Index;
use rusqlite::Connection;
use tempfile::tempdir;

fn columns(conn: &Connection, table: &str) -> Vec<(String, String, String)> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .unwrap();
    let rows = stmt
        .query_map([], |r| {
            let name: String = r.get(1)?;
            let typ: String = r.get(2)?;
            let dflt: Option<String> = r.get(4)?;
            Ok((name, typ, dflt.unwrap_or_default()))
        })
        .unwrap();
    rows.collect::<rusqlite::Result<_>>().unwrap()
}

/// Fresh DB must have `used_count` with default 0.
#[test]
fn ac1_fresh_db_has_used_count_column() {
    let tmp = tempdir().unwrap();
    let db_path = tmp.path().join("index.db");
    let _ = Index::open(&db_path).unwrap();

    let conn = Connection::open(&db_path).unwrap();
    let cols = columns(&conn, "memories_meta");
    let uc = cols
        .iter()
        .find(|(name, _, _)| name == "used_count")
        .expect("used_count column must be present after Index::open on a fresh DB");
    assert_eq!(uc.1, "INTEGER", "used_count column type");
    assert_eq!(uc.2.trim(), "0", "used_count default literal");
}

/// Migrate from a v4 (schema_version=4, no used_count column) shaped store.
#[test]
fn ac1_migrate_from_v4_schema_adds_used_count() {
    let tmp = tempdir().unwrap();
    let db_path = tmp.path().join("index.db");

    // Build a v4-shaped store by hand: has surfaced_count but not used_count.
    {
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE memories_fts USING fts5(
                id UNINDEXED, body, subject, kind UNINDEXED
            );
            CREATE TABLE memories_meta (
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
                last_decay_sweep_at TEXT,
                surfaced_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta(key, value) VALUES ('schema_version', '4');",
        )
        .unwrap();
        // Seed a row to check the default backfill.
        conn.execute(
            "INSERT INTO memories_meta (id, kind, subject, path, created_at)
             VALUES ('01HUSEDMIG0000000000000000', 'semantic', 'user', '/tmp/x.md', '2026-01-01T00:00:00Z')",
            [],
        )
        .unwrap();
    }

    // First open: migration must add used_count.
    let _ = Index::open(&db_path).unwrap();
    let conn = Connection::open(&db_path).unwrap();
    let cols = columns(&conn, "memories_meta");
    assert!(
        cols.iter().any(|(name, _, _)| name == "used_count"),
        "migration must add used_count column to a v4-shaped store"
    );
    let seeded: i64 = conn
        .query_row(
            "SELECT used_count FROM memories_meta WHERE id = '01HUSEDMIG0000000000000000'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(seeded, 0, "pre-existing row backfills to default 0");

    // Schema version bumped to current.
    let v: String = conn
        .query_row(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(v, recall::index::SCHEMA_VERSION.to_string());
}

/// Re-opening a migrated DB must not clobber a seeded `used_count`.
#[test]
fn ac1_reopen_on_migrated_db_preserves_used_count() {
    let tmp = tempdir().unwrap();
    let db_path = tmp.path().join("index.db");

    // First open: creates fresh schema with used_count.
    let _ = Index::open(&db_path).unwrap();
    {
        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            "INSERT INTO memories_meta (id, kind, subject, path, created_at, used_count)
             VALUES ('01HUSEDMIG0000000000000001', 'semantic', 'user', '/tmp/y.md', '2026-01-01T00:00:00Z', 7)",
            [],
        )
        .unwrap();
    }

    // Second open: must be idempotent.
    let _ = Index::open(&db_path).unwrap();
    let conn = Connection::open(&db_path).unwrap();
    let uc: i64 = conn
        .query_row(
            "SELECT used_count FROM memories_meta WHERE id = '01HUSEDMIG0000000000000001'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(uc, 7, "re-open must not clobber used_count");
}
