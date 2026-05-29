//! AC2 (PRD-recall-surfaced-tracking) — surfaced feedback increments
//! `surfaced_count` only. Confidence and `feedback_count` are unchanged.
//!
//! Exercises `Index::apply_surfaced_increment` directly: it is the storage
//! contract the `recall feedback --surfaced` CLI path leans on. The CLI
//! path mirrors the same write into the markdown file
//! (`apply_surfaced_one` in main.rs) — covered manually here by writing,
//! incrementing, and re-reading the file via FileStore.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;
use tempfile::tempdir;

fn setup() -> (tempfile::TempDir, FileStore, Index, Memory) {
    let tmp = tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let store = FileStore::open(root.clone()).unwrap();
    let idx = Index::open(&paths::index_db(&root)).unwrap();
    let mut mem = Memory::new(Kind::Semantic, Subject::user(), "surface AC2 body");
    mem.front.confidence = 0.42;
    mem.front.feedback_count = 3;
    let path = store.write(&mem).unwrap();
    idx.upsert(&mem, &path, None).unwrap();
    (tmp, store, idx, mem)
}

/// AC2 — single increment changes only `surfaced_count`.
#[test]
fn ac2_surfaced_increments_only_surface_count() {
    let (_tmp, _store, idx, mem) = setup();
    let id = mem.front.id.clone();

    // Pre-state.
    assert_eq!(idx.get_surfaced_count(&id).unwrap(), 0);
    assert_eq!(idx.get_feedback_count(&id).unwrap(), 3);

    let new = idx.apply_surfaced_increment(&id).unwrap();
    assert_eq!(new, 1);

    // Post-state: surface bumped, the other two are untouched.
    assert_eq!(idx.get_surfaced_count(&id).unwrap(), 1);
    assert_eq!(
        idx.get_feedback_count(&id).unwrap(),
        3,
        "feedback_count must not move on a surface event"
    );
    let conf: f64 = rusqlite_query_conf(&paths::index_db(_tmp.path()), &id);
    assert!(
        (conf - 0.42).abs() < 1e-9,
        "confidence must not move on a surface event; got {conf}"
    );
}

/// AC2 follow-up — incrementing several times is monotonically linear
/// and never accidentally bumps feedback_count.
#[test]
fn ac2_surfaced_increments_monotonic() {
    let (_tmp, _store, idx, mem) = setup();
    let id = mem.front.id.clone();

    for expected in 1..=5u32 {
        let n = idx.apply_surfaced_increment(&id).unwrap();
        assert_eq!(n, expected);
    }
    assert_eq!(idx.get_surfaced_count(&id).unwrap(), 5);
    assert_eq!(idx.get_feedback_count(&id).unwrap(), 3);
}

/// Mirror path: incrementing surfaced_count and writing the memory back
/// through FileStore preserves the round-trip. Mirrors what the CLI's
/// `apply_surfaced_one` does end-to-end.
#[test]
fn ac2_mirror_to_markdown_roundtrip() {
    let (_tmp, store, idx, mut mem) = setup();
    let id = mem.front.id.clone();

    let new = idx.apply_surfaced_increment(&id).unwrap();
    mem.front.surfaced_count = new;
    let (_, path) = store.find_by_id(&id).unwrap();
    store.overwrite(&path, &mem).unwrap();

    let back = store.read(&path).unwrap();
    assert_eq!(back.front.surfaced_count, 1);
    assert_eq!(back.front.feedback_count, 3);
    assert!((back.front.confidence - 0.42).abs() < 1e-9);
}

/// Read confidence directly from the SQLite file so the test does not
/// depend on extra Index public methods.
fn rusqlite_query_conf(db: &std::path::Path, id: &str) -> f64 {
    let conn = rusqlite::Connection::open(db).unwrap();
    conn.query_row(
        "SELECT confidence FROM memories_meta WHERE id = ?1",
        [id],
        |r| r.get(0),
    )
    .unwrap()
}
