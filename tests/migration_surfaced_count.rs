//! AC1 (PRD-recall-surfaced-tracking) — schema migration is idempotent.
//!
//! Fresh DB and a v0.6.0-shaped DB both end with the `surfaced_count`
//! column at default 0. Re-running `Index::open` on a migrated DB is a
//! no-op: same `surfaced_count`, no errors, no spurious writes.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::index::Index;
use rusqlite::Connection;
use tempfile::tempdir;

/// PRAGMA table_info returns one row per column. Used to assert column
/// presence + DEFAULT clause.
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

#[test]
fn ac1_fresh_db_has_surfaced_count_column() {
    let tmp = tempdir().unwrap();
    let db_path = tmp.path().join("index.db");
    let _ = Index::open(&db_path).unwrap();

    let conn = Connection::open(&db_path).unwrap();
    let cols = columns(&conn, "memories_meta");
    let sc = cols
        .iter()
        .find(|(name, _, _)| name == "surfaced_count")
        .expect("surfaced_count column must be present after Index::open on a fresh DB");
    assert_eq!(sc.1, "INTEGER", "surfaced_count column type");
    // DEFAULT clause is "0" — SQLite stores it as the literal text.
    assert_eq!(sc.2.trim(), "0", "surfaced_count default literal");
}

#[test]
fn ac1_migrate_from_v3_schema_adds_surfaced_count() {
    // Hand-construct a v0.6.0-shaped store (schema_version=3, no
    // surfaced_count column). Then Index::open and assert the migration
    // ran.
    let tmp = tempdir().unwrap();
    let db_path = tmp.path().join("index.db");

    // Build the v3 schema by hand. Mirrors the SCHEMA constant in
    // src/index.rs WITHOUT the surfaced_count column.
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
                last_decay_sweep_at TEXT
            );
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta(key, value) VALUES ('schema_version', '3');",
        )
        .unwrap();
        // Seed one row so we can assert the default-0 backfill applies.
        conn.execute(
            "INSERT INTO memories_meta (id, kind, subject, path, created_at)
             VALUES ('01HSEED0000000000000000000', 'semantic', 'user', '/tmp/x.md', '2026-01-01T00:00:00Z')",
            [],
        )
        .unwrap();
    }

    // First open: migration runs.
    let _ = Index::open(&db_path).unwrap();
    let conn = Connection::open(&db_path).unwrap();
    let cols = columns(&conn, "memories_meta");
    assert!(
        cols.iter().any(|(name, _, _)| name == "surfaced_count"),
        "migration must add surfaced_count column"
    );
    let seeded: i64 = conn
        .query_row(
            "SELECT surfaced_count FROM memories_meta WHERE id = '01HSEED0000000000000000000'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(seeded, 0, "pre-existing row backfills to default 0");

    // schema_version row was bumped to current SCHEMA_VERSION.
    let v: String = conn
        .query_row(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(v, recall::index::SCHEMA_VERSION.to_string());
}

#[test]
fn ac1_reopen_on_migrated_db_is_noop() {
    // Idempotency: open twice. The second open must NOT error and must
    // NOT change surfaced_count on any pre-seeded row.
    let tmp = tempdir().unwrap();
    let db_path = tmp.path().join("index.db");

    // First open creates the schema fresh.
    let _ = Index::open(&db_path).unwrap();
    {
        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            "INSERT INTO memories_meta (id, kind, subject, path, created_at, surfaced_count)
             VALUES ('01HSEED0000000000000000001', 'semantic', 'user', '/tmp/y.md', '2026-01-01T00:00:00Z', 5)",
            [],
        )
        .unwrap();
    }

    // Second open: migration must be a no-op for our seed row.
    let _ = Index::open(&db_path).unwrap();
    let conn = Connection::open(&db_path).unwrap();
    let sc: i64 = conn
        .query_row(
            "SELECT surfaced_count FROM memories_meta WHERE id = '01HSEED0000000000000000001'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(sc, 5, "re-open must not clobber surfaced_count");
}
