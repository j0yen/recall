//! AC2 (PRD-recall-stop-hook-discriminate) — `recall feedback --accept-used <id>`
//! increments both `used_count` and `feedback_count` and bumps confidence by
//! `accept_delta`.
//!
//! AC3 regression: `recall feedback --abstain <id>` leaves all three counters
//! (`used_count`, `feedback_count`, confidence) unchanged.
//!
//! AC7: markdown frontmatter round-trips `used_count: N`.

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

fn setup() -> (tempfile::TempDir, FileStore, Index, Memory) {
    let tmp = tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let store = FileStore::open(root.clone()).unwrap();
    let idx = Index::open(&paths::index_db(&root)).unwrap();
    let mut mem = Memory::new(Kind::Semantic, Subject::user(), "accept-used test body");
    mem.front.confidence = 0.5;
    mem.front.feedback_count = 0;
    mem.front.used_count = 0;
    let path = store.write(&mem).unwrap();
    idx.upsert(&mem, &path, None).unwrap();
    (tmp, store, idx, mem)
}

/// AC2 — `--accept-used` increments `used_count`, `feedback_count`, and bumps confidence.
#[test]
fn accept_used_increments_use_and_feedback() {
    let (_tmp, store, idx, mem) = setup();
    let id = mem.front.id.clone();
    let cfg = default_cfg();

    // Pre-state.
    assert_eq!(idx.get_used_count(&id).unwrap(), 0);
    assert_eq!(idx.get_feedback_count(&id).unwrap(), 0);

    let (new_conf, new_fc, new_uc) = idx.apply_used_feedback(&id, &cfg).unwrap();

    // All three must advance.
    assert_eq!(new_uc, 1, "used_count must be 1 after one accept-used");
    assert_eq!(new_fc, 1, "feedback_count must be 1 after one accept-used");
    assert!(
        (new_conf - (0.5 + cfg.accept_delta)).abs() < 1e-9,
        "confidence must be bumped by accept_delta; got {new_conf}"
    );

    // DB state matches returned values.
    assert_eq!(idx.get_used_count(&id).unwrap(), 1);
    assert_eq!(idx.get_feedback_count(&id).unwrap(), 1);

    // Mirror round-trip: overwrite markdown and read back.
    let (mut loaded, path) = store.find_by_id(&id).unwrap();
    loaded.front.confidence = new_conf;
    loaded.front.feedback_count = new_fc;
    loaded.front.used_count = new_uc;
    store.overwrite(&path, &loaded).unwrap();

    let back = store.read(&path).unwrap();
    assert_eq!(back.front.used_count, 1);
    assert_eq!(back.front.feedback_count, 1);
    assert!((back.front.confidence - (0.5 + cfg.accept_delta)).abs() < 1e-9);
}

/// AC2 — repeated `--accept-used` calls accumulate monotonically on both counters.
#[test]
fn accept_used_accumulates_monotonically() {
    let (_tmp, _store, idx, mem) = setup();
    let id = mem.front.id.clone();
    let cfg = default_cfg();

    for i in 1..=5u32 {
        let (_, fc, uc) = idx.apply_used_feedback(&id, &cfg).unwrap();
        assert_eq!(uc, i, "used_count at iteration {i}");
        assert_eq!(fc, i, "feedback_count at iteration {i}");
    }
}

/// AC2 — `used_count` advances while plain `--accept` does NOT touch `used_count`.
#[test]
fn plain_accept_does_not_increment_used_count() {
    let (_tmp, _store, idx, mem) = setup();
    let id = mem.front.id.clone();
    let cfg = default_cfg();

    // Plain accept (old path).
    let (_, _) = idx.apply_feedback_delta(&id, cfg.accept_delta, &cfg).unwrap();

    assert_eq!(
        idx.get_used_count(&id).unwrap(),
        0,
        "plain --accept must NOT change used_count"
    );
    assert_eq!(
        idx.get_feedback_count(&id).unwrap(),
        1,
        "plain --accept must bump feedback_count"
    );
}

/// AC3 regression — `--abstain` leaves all three counters unchanged.
#[test]
fn abstain_leaves_all_counters_unchanged() {
    let (_tmp, _store, idx, mem) = setup();
    let id = mem.front.id.clone();
    let cfg = default_cfg();

    // Seed some state.
    idx.apply_used_feedback(&id, &cfg).unwrap(); // used=1, fc=1
    let conf_before: f64 = {
        use rusqlite::Connection;
        let conn = Connection::open(&paths::index_db(_tmp.path())).unwrap();
        conn.query_row(
            "SELECT confidence FROM memories_meta WHERE id = ?1",
            [&id],
            |r| r.get(0),
        )
        .unwrap()
    };

    // Now abstain.
    let (new_conf, new_fc) = idx.apply_feedback_delta(&id, 0.0, &cfg).unwrap();

    // Confidence unchanged (delta=0).
    assert!(
        (new_conf - conf_before).abs() < 1e-9,
        "abstain must not change confidence; before={conf_before} after={new_conf}"
    );
    assert_eq!(new_fc, 2, "abstain still bumps feedback_count for observability");
    assert_eq!(
        idx.get_used_count(&id).unwrap(),
        1,
        "abstain must NOT touch used_count"
    );
}

/// AC7 — markdown frontmatter round-trips `used_count: N` correctly.
#[test]
fn used_roundtrip_markdown() {
    let tmp = tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let store = FileStore::open(root.clone()).unwrap();

    let mut mem = Memory::new(Kind::Semantic, Subject::user(), "roundtrip body");
    mem.front.used_count = 7;
    mem.front.feedback_count = 12;
    mem.front.confidence = 0.64;

    let path = store.write(&mem).unwrap();
    let back = store.read(&path).unwrap();

    assert_eq!(back.front.used_count, 7, "used_count must survive roundtrip");
    assert_eq!(back.front.feedback_count, 12);
    assert!((back.front.confidence - 0.64).abs() < 1e-9);
}

/// AC7 — a memory with `used_count = 0` must NOT emit `used_count` in the YAML
/// (clean files stay clean for pre-PRD-#3 compatibility).
#[test]
fn used_count_zero_omitted_from_yaml() {
    let mem = Memory::new(Kind::Semantic, Subject::user(), "zero omit test");
    assert_eq!(mem.front.used_count, 0);
    let md = mem.to_markdown().unwrap();
    assert!(
        !md.contains("used_count"),
        "used_count: 0 must be omitted from YAML; got:\n{md}"
    );
}

/// AC7 — a memory with `used_count > 0` MUST emit `used_count` in the YAML.
#[test]
fn used_count_nonzero_present_in_yaml() {
    let mut mem = Memory::new(Kind::Semantic, Subject::user(), "nonzero test");
    mem.front.used_count = 3;
    let md = mem.to_markdown().unwrap();
    assert!(
        md.contains("used_count: 3"),
        "used_count must appear in YAML when nonzero; got:\n{md}"
    );
}
