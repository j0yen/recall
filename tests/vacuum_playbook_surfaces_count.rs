//! AC8: the self-review playbook surfaces a non-zero candidate count.
//!
//! The playbook (`~/.claude/skills/self-review/playbooks/recall_corpus_vacuum.md`)
//! runs `recall vacuum --dry-run --format json | jq '.candidates'` and, when
//! the count is > 0, surfaces a "Pending your call" line. This test exercises
//! that exact JSON path: it seeds a store with candidate memories
//! (surfaced >= 20, unused), invokes the real `recall` binary in dry-run JSON
//! mode against that store, and asserts `.candidates` is the positive count
//! the playbook would surface. Driving the binary (not the library) is the
//! point — it proves the playbook's command emits a count the playbook can act
//! on.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;
use std::process::Command;

/// Seed one candidate memory (surfaced=`surfaced`, recall_count=0) into the
/// store rooted at `root`. Mirrors the AC1 dry-run fixture so candidate
/// detection is identical to the rest of the vacuum suite.
fn seed_candidate(idx: &Index, store: &FileStore, surfaced: u32, confidence: f64) -> String {
    let mut mem = Memory::new(
        Kind::Semantic,
        Subject::user(),
        "test memory body for vacuum playbook count test",
    );
    mem.front.confidence = confidence;
    mem.front.surfaced_count = surfaced;
    let path = store.write(&mem).unwrap();
    idx.upsert(&mem, &path, None).unwrap();
    mem.front.id.clone()
}

#[test]
fn vacuum_playbook_surfaces_nonzero_count() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();

    // Three candidates (surfaced=25, recall_count=0) → all match the default
    // bar (min_surfaced=20, max_used=0). The playbook would surface "3".
    let expected = 3usize;
    for _ in 0..expected {
        let _ = seed_candidate(&idx, &store, 25, 0.55);
    }
    // One non-candidate (surfaced below threshold) — must NOT inflate the count.
    let _ = seed_candidate(&idx, &store, 5, 0.55);
    drop(idx);

    // Run the exact command the playbook runs.
    let out = Command::new(env!("CARGO_BIN_EXE_recall"))
        .args(["vacuum", "--root"])
        .arg(root)
        .args(["--dry-run", "--format", "json"])
        .output()
        .expect("run recall vacuum --dry-run --format json");

    assert!(
        out.status.success(),
        "vacuum dry-run exited non-zero: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("vacuum dry-run must emit valid JSON");
    let count = v
        .get("candidates")
        .and_then(serde_json::Value::as_u64)
        .expect("JSON must contain a numeric `candidates` field (playbook reads `.candidates`)");

    // The playbook's surfacing gate is `count > 0`; assert that AND the exact
    // candidate count so the test catches both "never surfaces" and
    // "miscounts" regressions.
    assert!(count > 0, "playbook would surface nothing; expected > 0");
    assert_eq!(
        count as usize, expected,
        "playbook should surface exactly {expected} candidates; got {count}"
    );
    // Dry-run must not have mutated anything.
    assert!(
        v.get("dry_run").and_then(serde_json::Value::as_bool) == Some(true),
        "dry_run flag must be true in the playbook's command"
    );
}
