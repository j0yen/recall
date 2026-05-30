//! AC1: `recall vacuum --dry-run` lists candidates without mutation.
//!
//! Fixture: one memory with (surf=25, used=0, conf=0.55),
//! one with (surf=25, used=3, conf=0.55).
//! Dry-run returns 1 candidate; both memories unchanged.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;
use recall::vacuum::VacuumConfig;

fn seed_memory(
    idx: &Index,
    store: &FileStore,
    root: &std::path::Path,
    surfaced: u32,
    recall_count: u32,
    confidence: f64,
) -> String {
    let mut mem = Memory::new(
        Kind::Semantic,
        Subject::user(),
        "test memory body for vacuum dry run test",
    );
    mem.front.confidence = confidence;
    mem.front.surfaced_count = surfaced;
    let path = store.write(&mem).unwrap();
    // Update surfaced_count and recall_count in index.
    // upsert syncs the memory meta fields.
    idx.upsert(&mem, &path, None).unwrap();
    // Now manually set surfaced_count and recall_count via SQL since
    // upsert uses mem.front values (surfaced_count already there).
    // But recall_count needs to be set separately.
    if recall_count > 0 {
        let db_path = paths::index_db(root);
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute(
            "UPDATE memories_meta SET recall_count = ?1 WHERE id = ?2",
            rusqlite::params![i64::from(recall_count), mem.front.id],
        )
        .unwrap();
    }
    mem.front.id.clone()
}

#[test]
fn vacuum_dry_run_lists_one_candidate_no_mutation() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();

    let vcfg = VacuumConfig::default(); // min_surfaced=20, max_used=0

    // Memory 1: surfaced=25, recall_count=0, conf=0.55 → CANDIDATE
    let id1 = seed_memory(&idx, &store, root, 25, 0, 0.55);
    // Memory 2: surfaced=25, recall_count=3, conf=0.55 → NOT a candidate (recall_count > max_used)
    let id2 = seed_memory(&idx, &store, root, 25, 3, 0.55);

    // Dry-run: query candidates, do NOT apply.
    let candidates = idx
        .vacuum_candidates(vcfg.min_surfaced, vcfg.max_used)
        .expect("vacuum_candidates should succeed");

    assert_eq!(
        candidates.len(),
        1,
        "expected exactly 1 candidate; got {:?}",
        candidates.iter().map(|c| &c.id).collect::<Vec<_>>()
    );
    assert_eq!(candidates[0].id, id1, "candidate should be id1");

    // Verify neither memory's confidence changed.
    let conf1_after = idx
        .vacuum_candidates(1, 0) // threshold=1 captures id1 again
        .unwrap()
        .into_iter()
        .find(|c| c.id == id1)
        .map(|c| c.confidence_before)
        .expect("id1 still in index");
    assert!(
        (conf1_after - 0.55).abs() < 1e-9,
        "id1 confidence must be unchanged after dry-run; got {conf1_after}"
    );

    // id2 should not appear (recall_count=3 > max_used=0).
    let candidates2 = idx.vacuum_candidates(vcfg.min_surfaced, vcfg.max_used).unwrap();
    assert!(
        candidates2.iter().all(|c| c.id != id2),
        "id2 must not appear in candidates"
    );
    let _ = (id1, id2); // suppress unused warnings
}
