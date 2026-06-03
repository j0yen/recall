//! AC8 (PRD-recall-stop-hook-discriminate) — compound utility tracking over 10
//! simulated Stop-hook sessions.
//!
//! Scenario:
//!   - Memory A ("used every time"): receives `--accept-used` each session.
//!   - Memory B ("never used"): receives `--abstain` each session.
//!
//! After 10 sessions:
//!   A: used_count=10, surfaced_count=10, confidence ≈ min(0.5 + 10*accept_delta, ceiling).
//!   B: used_count=0,  surfaced_count=10, confidence=0.5 (abstain is a no-op on confidence).
//!
//! Confirms the "used_ratio of a never-used id stays at 0 while a used-every-time
//! id reaches 1.0" invariant from PRD §3 AC8.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::config::Feedback as FeedbackCfg;
use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;
use tempfile::tempdir;

fn default_cfg() -> FeedbackCfg {
    FeedbackCfg::default()
}

#[test]
fn compound_10_sessions_used_vs_never_used() {
    let tmp = tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let store = FileStore::open(root.clone()).unwrap();
    let idx = Index::open(&paths::index_db(&root)).unwrap();
    let cfg = default_cfg();

    // Create memory A (used every session) and B (surfaced, never used).
    let mut mem_a = Memory::new(Kind::Semantic, Subject::user(), "always used memory");
    let mut mem_b = Memory::new(Kind::Semantic, Subject::user(), "never used memory");
    mem_a.front.confidence = 0.5;
    mem_b.front.confidence = 0.5;

    let path_a = store.write(&mem_a).unwrap();
    let path_b = store.write(&mem_b).unwrap();
    idx.upsert(&mem_a, &path_a, None).unwrap();
    idx.upsert(&mem_b, &path_b, None).unwrap();

    let id_a = mem_a.front.id.clone();
    let id_b = mem_b.front.id.clone();

    // Simulate 10 Stop-hook fires.
    for _ in 0..10 {
        // Surfaced increment for both.
        idx.apply_surfaced_increment(&id_a).unwrap();
        idx.apply_surfaced_increment(&id_b).unwrap();

        // A → accept-used; B → abstain.
        idx.apply_used_feedback(&id_a, &cfg).unwrap();
        idx.apply_feedback_delta(&id_b, 0.0, &cfg).unwrap();
    }

    // Verify A.
    let uc_a = idx.get_used_count(&id_a).unwrap();
    let sc_a = idx.get_surfaced_count(&id_a).unwrap();
    assert_eq!(uc_a, 10, "A: used_count must be 10 after 10 accept-used sessions");
    assert_eq!(sc_a, 10, "A: surfaced_count must be 10");

    // Confidence of A: 0.5 + 10 * accept_delta, clamped to ceiling.
    let expected_a = (0.5 + 10.0 * cfg.accept_delta).min(cfg.ceiling);
    let (_, _, conf_a) = query_meta(&idx, &id_a);
    assert!(
        (conf_a - expected_a).abs() < 1e-9,
        "A: confidence expected {expected_a:.4} got {conf_a:.4}"
    );

    // used_ratio for A: uc / sc = 1.0.
    let ratio_a = f64::from(uc_a) / f64::from(sc_a);
    assert!(
        (ratio_a - 1.0).abs() < 1e-9,
        "A used_ratio must be 1.0; got {ratio_a}"
    );

    // Verify B.
    let uc_b = idx.get_used_count(&id_b).unwrap();
    let sc_b = idx.get_surfaced_count(&id_b).unwrap();
    assert_eq!(uc_b, 0, "B: used_count must stay 0 (never used)");
    assert_eq!(sc_b, 10, "B: surfaced_count must be 10");

    let (_, _, conf_b) = query_meta(&idx, &id_b);
    assert!(
        (conf_b - 0.5).abs() < 1e-9,
        "B: confidence must remain 0.5 after 10 abstain sessions; got {conf_b}"
    );

    // used_ratio for B: 0/10 = 0.0.
    let ratio_b = f64::from(uc_b) / f64::from(sc_b);
    assert!(
        ratio_b.abs() < 1e-9,
        "B used_ratio must be 0.0; got {ratio_b}"
    );
}

/// Session 1 snapshot from PRD §4.2: A used_count=1, surfaced=1, conf=0.52.
#[test]
fn after_session_1_a_has_conf_0_52_and_used_1() {
    let tmp = tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let store = FileStore::open(root.clone()).unwrap();
    let idx = Index::open(&paths::index_db(&root)).unwrap();
    let cfg = default_cfg();

    let mut mem = Memory::new(Kind::Semantic, Subject::user(), "session 1 snapshot");
    mem.front.confidence = 0.5;
    let path = store.write(&mem).unwrap();
    idx.upsert(&mem, &path, None).unwrap();
    let id = mem.front.id.clone();

    // One Stop-hook fire.
    idx.apply_surfaced_increment(&id).unwrap();
    let (conf, _, uc) = idx.apply_used_feedback(&id, &cfg).unwrap();

    assert_eq!(uc, 1, "used_count must be 1 after session 1");
    assert_eq!(idx.get_surfaced_count(&id).unwrap(), 1);
    assert!(
        (conf - 0.52).abs() < 1e-9,
        "confidence must be 0.52 (0.5 + accept_delta=0.02); got {conf}"
    );
}

/// Helper: pull (feedback_count, used_count, confidence) from the index.
fn query_meta(idx: &Index, id: &str) -> (u32, u32, f64) {
    let fc = idx.get_feedback_count(id).unwrap();
    let uc = idx.get_used_count(id).unwrap();
    // Read confidence via a raw SQLite query via the Index's public API.
    // We use get_surfaced_count as a proxy to confirm the id exists, then
    // reconstruct confidence from our recorded expected value — but since
    // we need the actual confidence here, use a helper that exercises
    // the index's all_meta or a known good method.
    // Actually: Index exposes `apply_used_feedback` which returns conf.
    // For a read-only path, use the fact that we can call get_feedback_count
    // and accept that confidence is embedded in the apply calls above.
    // Simplest: just return (fc, uc, 0.0) and check conf separately.
    // But we need conf for the compound test above. Let's read it through
    // the `all_meta()` call.
    let metas = idx.all_meta().unwrap();
    let row = metas.iter().find(|m| m.id == id).unwrap();
    (fc, uc, row.confidence)
}
