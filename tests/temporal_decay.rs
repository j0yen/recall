//! Integration tests for PRD-recall-temporal-decay.
//!
//! Covers AC1–AC6 from the PRD.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

use chrono::{Duration, Utc};
use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;
use recall::temporal_decay::{is_worth_updating, project_decay};
use tempfile::tempdir;

// ── helpers ────────────────────────────────────────────────────────────────

fn setup_store() -> (tempfile::TempDir, FileStore, Index) {
    let tmp = tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let store = FileStore::open(root.clone()).unwrap();
    let idx = Index::open(&paths::index_db(&root)).unwrap();
    (tmp, store, idx)
}

/// Insert a memory with the given confidence and a `created_at` backdated
/// by `days_ago`. Returns the memory id.
fn insert_memory_aged(
    store: &FileStore,
    idx: &Index,
    confidence: f64,
    subject: Subject,
    days_ago: i64,
) -> String {
    let mut mem = Memory::new(Kind::Semantic, subject, "temporal decay test body");
    mem.front.confidence = confidence;
    // Backdate created_at so the decay formula sees elapsed days.
    mem.front.created_at = Utc::now() - Duration::days(days_ago);
    let path = store.write(&mem).unwrap();
    idx.upsert(&mem, &path, None).unwrap();
    mem.front.id
}

fn get_confidence(idx: &Index, id: &str) -> f64 {
    idx.get_meta(id)
        .unwrap()
        .expect("memory must exist")
        .confidence
}

// ── AC1: dry-run reports candidates without mutation ───────────────────────

/// AC1 — dry-run returns 1 candidate for an old high-confidence memory;
/// the confidence in the store is UNCHANGED after the call.
#[test]
fn dry_run_no_mutation() {
    let (_tmp, store, idx) = setup_store();

    // 0.9 confidence, created 30 days ago → should decay toward 0.5
    let id = insert_memory_aged(&store, &idx, 0.9, Subject::user(), 30);

    let conf_before = get_confidence(&idx, &id);

    let candidates = idx
        .temporal_decay_report(
            90,    // half_life_d
            0,     // min_interval_d (0 = no gate for tests)
            0.001, // min_delta
            None,  // subject_prefix
            false, // apply = false → dry run
        )
        .unwrap();

    assert_eq!(candidates.len(), 1, "expected exactly 1 candidate");
    assert_eq!(candidates[0].id, id);
    assert!(!candidates[0].applied, "dry-run candidates must not be marked applied");

    // Confidence in store must be unchanged.
    let conf_after = get_confidence(&idx, &id);
    assert!(
        (conf_after - conf_before).abs() < 1e-9,
        "dry-run must not mutate store; before={conf_before}, after={conf_after}"
    );
}

// ── AC2: apply decreases confidence ────────────────────────────────────────

/// AC2 — apply=true writes the decayed confidence. Projected value matches
/// the half-life formula: 0.5 + (0.9-0.5) * 2^(-30/90) ≈ 0.817.
#[test]
fn apply_decreases_confidence() {
    let (_tmp, store, idx) = setup_store();
    let id = insert_memory_aged(&store, &idx, 0.9, Subject::user(), 30);

    let conf_before = get_confidence(&idx, &id);

    let candidates = idx
        .temporal_decay_report(90, 0, 0.001, None, true)
        .unwrap();

    assert_eq!(candidates.len(), 1);
    assert!(candidates[0].applied, "candidate must be marked applied");

    let conf_after = get_confidence(&idx, &id);

    // Confidence must have decreased.
    assert!(
        conf_after < conf_before,
        "confidence must decrease after apply; before={conf_before}, after={conf_after}"
    );

    // Must match the formula (within floating-point tolerance).
    let expected = project_decay(0.9, 30.0, 90);
    assert!(
        (conf_after - expected).abs() < 0.01,
        "applied confidence {conf_after} must be close to projected {expected}"
    );
}

// ── AC3: neutral memory excluded by min_delta ──────────────────────────────

/// AC3 — a memory at exactly confidence=0.5 projects to 0.5, so its delta
/// is 0. With min_delta=0.001, it must be excluded from candidates.
#[test]
fn neutral_memory_excluded() {
    let (_tmp, store, idx) = setup_store();
    // Neutral memory, old enough to be considered.
    let _id = insert_memory_aged(&store, &idx, 0.5, Subject::user(), 365);

    let candidates = idx
        .temporal_decay_report(90, 0, 0.001, None, false)
        .unwrap();

    assert!(
        candidates.is_empty(),
        "neutral memory (confidence=0.5) must not appear in candidates; got {candidates:?}"
    );
}

// ── AC4: min_interval_d idempotency gate ───────────────────────────────────

/// AC4 — after applying once, re-running with min_interval_d=1 returns 0
/// candidates because last_decay_sweep_at is now recent.
#[test]
fn idempotency_gate() {
    let (_tmp, store, idx) = setup_store();
    let _id = insert_memory_aged(&store, &idx, 0.9, Subject::user(), 30);

    // First sweep: apply.
    let first = idx
        .temporal_decay_report(90, 0, 0.001, None, true)
        .unwrap();
    assert_eq!(first.len(), 1, "first sweep should find 1 candidate");

    // Second sweep with min_interval_d=1 (86400 seconds must elapse): should
    // find 0 candidates because we just swept.
    let second = idx
        .temporal_decay_report(90, 1, 0.001, None, true)
        .unwrap();
    assert!(
        second.is_empty(),
        "idempotency gate should suppress re-sweep within min_interval_d=1; got {second:?}"
    );
}

// ── AC5: subject_prefix filter ─────────────────────────────────────────────

/// AC5 — two memories: one user, one self. With subject_prefix="user",
/// only the user memory appears as a candidate.
#[test]
fn subject_prefix_filter() {
    let (_tmp, store, idx) = setup_store();

    let user_id = insert_memory_aged(&store, &idx, 0.9, Subject::user(), 60);
    let _self_id = insert_memory_aged(&store, &idx, 0.9, Subject::self_(), 60);

    let candidates = idx
        .temporal_decay_report(90, 0, 0.001, Some("user"), false)
        .unwrap();

    assert_eq!(candidates.len(), 1, "only user memory should match prefix 'user'");
    assert_eq!(candidates[0].id, user_id);
    assert_eq!(candidates[0].subject, "user");
}

// ── AC6: project_decay formula matches feedback::decay_toward_neutral ──────

/// AC6 — project_decay must be identical to the underlying formula.
/// This test is the "canary" that catches any drift between the module
/// and the existing feedback implementation.
#[test]
fn project_decay_matches_feedback_formula() {
    use recall::feedback::decay_toward_neutral;

    let cases: &[(f64, f64, u32)] = &[
        (0.9, 90.0, 90),
        (0.7, 30.0, 60),
        (0.5, 365.0, 90),
        (0.95, 1.0, 90),
        (0.1, 45.0, 30),
    ];
    for (conf, days, hl) in cases {
        let via_module = project_decay(*conf, *days, *hl);
        let via_feedback = decay_toward_neutral(*conf, *days, *hl);
        assert!(
            (via_module - via_feedback).abs() < 1e-15,
            "project_decay({conf}, {days}, {hl}) diverges: {via_module} != {via_feedback}"
        );
    }
}

// ── is_worth_updating unit tests ────────────────────────────────────────────

#[test]
fn worth_updating_threshold() {
    assert!(is_worth_updating(0.9, 0.85, 0.001));
    assert!(!is_worth_updating(0.9, 0.8999, 0.001)); // delta = 0.0001 < 0.001
    assert!(is_worth_updating(0.5, 0.499, 0.001)); // below neutral? still applies if large enough
}
