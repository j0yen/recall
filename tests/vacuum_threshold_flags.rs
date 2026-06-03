//! AC5: `--min-surfaced` and `--max-used` flags override defaults.
//!
//! Seed 5 memories with varying surfaced/used counts.
//! Run vacuum_candidates with custom thresholds and assert only
//! matching candidates are returned.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;

fn seed(
    idx: &Index,
    store: &FileStore,
    root: &std::path::Path,
    surfaced: u32,
    recall_count: u32,
) -> String {
    let mut mem = Memory::new(Kind::Semantic, Subject::user(), "threshold test memory");
    mem.front.confidence = 0.60;
    mem.front.surfaced_count = surfaced;
    let path = store.write(&mem).unwrap();
    idx.upsert(&mem, &path, None).unwrap();
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
fn vacuum_min_surfaced_and_max_used_override() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();

    // Seed memories with various (surfaced, recall_count) combos:
    // (10, 0)  — surfaced too low for threshold 50
    // (60, 0)  — CANDIDATE at (min_s=50, max_u=5)
    // (70, 3)  — CANDIDATE at (min_s=50, max_u=5)
    // (80, 6)  — recall_count too high
    // (55, 5)  — CANDIDATE at (min_s=50, max_u=5) — boundary
    let id_low_surf = seed(&idx, &store, root, 10, 0);
    let id_match_a = seed(&idx, &store, root, 60, 0);
    let id_match_b = seed(&idx, &store, root, 70, 3);
    let id_high_used = seed(&idx, &store, root, 80, 6);
    let id_boundary = seed(&idx, &store, root, 55, 5);

    // Custom thresholds: min_surfaced=50, max_used=5
    let candidates = idx.vacuum_candidates(50, 5).unwrap();

    let ids: Vec<&str> = candidates.iter().map(|c| c.id.as_str()).collect();

    // Must include matches.
    assert!(
        ids.contains(&id_match_a.as_str()),
        "id_match_a (surf=60, used=0) must be a candidate"
    );
    assert!(
        ids.contains(&id_match_b.as_str()),
        "id_match_b (surf=70, used=3) must be a candidate"
    );
    assert!(
        ids.contains(&id_boundary.as_str()),
        "id_boundary (surf=55, used=5) must be a candidate at max_used=5"
    );

    // Must exclude non-matches.
    assert!(
        !ids.contains(&id_low_surf.as_str()),
        "id_low_surf (surf=10) must NOT be a candidate at min_surfaced=50"
    );
    assert!(
        !ids.contains(&id_high_used.as_str()),
        "id_high_used (used=6) must NOT be a candidate at max_used=5"
    );

    // Default thresholds (min_s=20, max_u=0) should match differently.
    let default_candidates = idx.vacuum_candidates(20, 0).unwrap();
    let default_ids_set: std::collections::HashSet<&str> =
        default_candidates.iter().map(|c| c.id.as_str()).collect();

    // id_low_surf has surfaced=10 < min_surfaced=20 so must NOT appear.
    assert!(
        !default_ids_set.contains(id_low_surf.as_str()),
        "id_low_surf (surf=10) must not appear under min_surfaced=20"
    );
    // id_match_a (surf=60, recall=0) DOES meet default thresholds.
    assert!(
        default_ids_set.contains(id_match_a.as_str()),
        "id_match_a must appear under default thresholds"
    );
    let _ = (id_high_used, id_match_b, id_boundary); // suppress warnings
}
