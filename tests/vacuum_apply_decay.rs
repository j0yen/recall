//! AC2: `--apply --action decay` decreases confidence by `decay_amount`, clamped at floor.
//!
//! Fixture: one memory with (surf=25, recall_count=0, conf=0.55).
//! After apply, confidence == 0.55 - 0.10 == 0.45.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;
use recall::vacuum::{VacuumConfig, apply_decay};

fn seed_memory_with_conf(
    idx: &Index,
    store: &FileStore,
    confidence: f64,
    surfaced: u32,
) -> (String, std::path::PathBuf) {
    let mut mem = Memory::new(Kind::Semantic, Subject::user(), "test decay candidate body");
    mem.front.confidence = confidence;
    mem.front.surfaced_count = surfaced;
    let path = store.write(&mem).unwrap();
    idx.upsert(&mem, &path, None).unwrap();
    (mem.front.id.clone(), path)
}

#[test]
fn vacuum_apply_decay_reduces_confidence() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();
    let vcfg = VacuumConfig::default();

    let (id, _path) = seed_memory_with_conf(&idx, &store, 0.55, 25);

    let candidates = idx
        .vacuum_candidates(vcfg.min_surfaced, vcfg.max_used)
        .unwrap();
    assert_eq!(candidates.len(), 1, "should have one candidate");
    let c = &candidates[0];

    let new_conf = apply_decay(&idx, &store, c, vcfg.decay_amount).unwrap();

    let expected = 0.55 - 0.10; // 0.45
    assert!(
        (new_conf - expected).abs() < 1e-9,
        "confidence should be {expected:.3} after decay; got {new_conf:.6}"
    );

    // Verify via index re-query.
    let after = idx
        .vacuum_candidates(1, 0)
        .unwrap()
        .into_iter()
        .find(|c| c.id == id);
    // After decay to 0.45, conf is now below any threshold that would filter it out.
    // vacuum_candidates uses surfaced_count and recall_count, not confidence.
    // The candidate should still match (surfaced=25 >= 1, recall=0 <= 0).
    let after = after.expect("id should still be in index after decay");
    assert!(
        (after.confidence_before - expected).abs() < 1e-9,
        "index should show new confidence {expected:.3}; got {:.3}",
        after.confidence_before
    );
}

#[test]
fn vacuum_apply_decay_floor_at_005() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();
    let vcfg = VacuumConfig::default();

    // Set confidence very low so decay would go below floor.
    let (_, _path) = seed_memory_with_conf(&idx, &store, 0.08, 25);
    let candidates = idx.vacuum_candidates(vcfg.min_surfaced, vcfg.max_used).unwrap();
    let c = &candidates[0];
    let new_conf = apply_decay(&idx, &store, c, vcfg.decay_amount).unwrap();
    assert!(
        (new_conf - 0.05).abs() < 1e-9,
        "floor should be 0.05; got {new_conf:.6}"
    );
}
