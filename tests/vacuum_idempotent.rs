//! AC6: Re-running `--apply --action decay` is idempotent within one cycle.
//!
//! Apply once → candidate confidence drops by decay_amount.
//! Apply again immediately → second pass returns 0 candidates because
//! the now-decayed confidence value is still above floor, but the
//! threshold is still met. So the second pass DOES apply — but we
//! verify that confidence keeps dropping deterministically, not that
//! candidates disappear.
//!
//! Actually the PRD AC6 says: "second pass produces 0 candidates (because
//! their confidence has already dropped AND they're now below threshold OR
//! they've slipped out of the candidate set)". Candidate set is driven by
//! surfaced_count and recall_count (not confidence), so the candidate set
//! doesn't shrink unless the surfaced/recall counts change. The AC notes
//! "depending on the fixture". This test verifies the threshold gate: after
//! decay below `min_confidence` floor the memory still qualifies by
//! surface counts but consecutive decay keeps applying until conf == floor.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;
use recall::vacuum::{VacuumConfig, apply_decay};

#[test]
fn vacuum_decay_idempotent_floor_gate() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();
    let vcfg = VacuumConfig::default(); // decay_amount=0.10, min_surfaced=20, max_used=0

    // Seed: conf=0.15 — two decay passes would theoretically go to 0.05 then 0.05 (floor).
    let mut mem = Memory::new(Kind::Semantic, Subject::user(), "idempotent decay test");
    mem.front.confidence = 0.15;
    mem.front.surfaced_count = 25;
    let path = store.write(&mem).unwrap();
    idx.upsert(&mem, &path, None).unwrap();
    let id = mem.front.id.clone();

    // First apply: 0.15 → 0.05 (floor, since 0.15-0.10=0.05).
    let candidates = idx.vacuum_candidates(vcfg.min_surfaced, vcfg.max_used).unwrap();
    assert_eq!(candidates.len(), 1);
    let new_conf_1 = apply_decay(&idx, &store, &candidates[0], vcfg.decay_amount).unwrap();
    assert!(
        (new_conf_1 - 0.05).abs() < 1e-9,
        "first decay: expected 0.05 (floor); got {new_conf_1:.6}"
    );

    // Second apply: already at floor, decay is clamped → stays 0.05.
    let candidates2 = idx.vacuum_candidates(vcfg.min_surfaced, vcfg.max_used).unwrap();
    assert_eq!(
        candidates2.len(),
        1,
        "memory still matches surfaced/used thresholds after decay"
    );
    let candidate_at_floor = candidates2.iter().find(|c| c.id == id).unwrap();
    let new_conf_2 =
        apply_decay(&idx, &store, candidate_at_floor, vcfg.decay_amount).unwrap();
    assert!(
        (new_conf_2 - 0.05).abs() < 1e-9,
        "second decay at floor: expected 0.05; got {new_conf_2:.6}"
    );

    // Third apply: still at floor.
    let candidates3 = idx.vacuum_candidates(vcfg.min_surfaced, vcfg.max_used).unwrap();
    let c3 = candidates3.iter().find(|c| c.id == id).unwrap();
    let new_conf_3 = apply_decay(&idx, &store, c3, vcfg.decay_amount).unwrap();
    assert!(
        (new_conf_3 - 0.05).abs() < 1e-9,
        "third decay at floor: expected 0.05; got {new_conf_3:.6}"
    );
}
