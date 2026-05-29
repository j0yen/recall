//! Integration tests for `recall::session` — precedence chain (iter-2)
//! and `SessionFilter` / `parse_filter_arg` (iter-4).
//! PRD-recall-session-stamp.md §2.2 + §2.3 + AC4 + AC6-AC10.
//!
//! Tests use a process-wide mutex to serialize env-var mutation (Rust
//! integration tests run in parallel within one binary).

#![allow(clippy::unwrap_used, clippy::expect_used, unsafe_code)]

use std::sync::Mutex;

use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::session::{
    SessionFilter, parse_filter_arg, resolve, Source, AGENTNS_ENV_VAR, RECALL_ENV_VAR,
};
use recall::store::FileStore;

static ENV_GUARD: Mutex<()> = Mutex::new(());

fn with_env<F: FnOnce()>(pairs: &[(&str, Option<&str>)], f: F) {
    let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let prior: Vec<(String, Option<String>)> = pairs
        .iter()
        .map(|(k, _)| ((*k).to_string(), std::env::var(k).ok()))
        .collect();
    for (k, v) in pairs {
        // SAFETY: env::set_var/remove_var require sole control of the
        // environment; the ENV_GUARD mutex provides that across these
        // integration tests.
        unsafe {
            match v {
                Some(val) => std::env::set_var(k, val),
                None => std::env::remove_var(k),
            }
        }
    }
    f();
    for (k, v) in prior {
        unsafe {
            match v {
                Some(val) => std::env::set_var(&k, val),
                None => std::env::remove_var(&k),
            }
        }
    }
}

#[test]
fn ac4_recall_env_wins_over_agentns_env() {
    with_env(
        &[
            (RECALL_ENV_VAR, Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")),
            (AGENTNS_ENV_VAR, Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")),
        ],
        || {
            let r = resolve();
            assert_eq!(r.source, Source::RecallEnv);
            assert_eq!(r.id, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        },
    );
}

#[test]
fn agentns_env_used_when_recall_env_absent() {
    with_env(
        &[
            (RECALL_ENV_VAR, None),
            (AGENTNS_ENV_VAR, Some("cccccccccccccccccccccccccccccccc")),
        ],
        || {
            let r = resolve();
            assert_eq!(r.source, Source::AgentnsEnv);
            assert_eq!(r.id, "cccccccccccccccccccccccccccccccc");
        },
    );
}

#[test]
fn whitespace_recall_env_treated_as_absent() {
    with_env(
        &[
            (RECALL_ENV_VAR, Some("   ")),
            (AGENTNS_ENV_VAR, Some("dddddddddddddddddddddddddddddddd")),
        ],
        || {
            let r = resolve();
            assert_eq!(r.source, Source::AgentnsEnv);
            assert_eq!(r.id, "dddddddddddddddddddddddddddddddd");
        },
    );
}

#[test]
fn agentns_value_trimmed() {
    with_env(
        &[
            (RECALL_ENV_VAR, None),
            (AGENTNS_ENV_VAR, Some("  eeeeeeee  \n")),
        ],
        || {
            let r = resolve();
            assert_eq!(r.source, Source::AgentnsEnv);
            assert_eq!(r.id, "eeeeeeee");
        },
    );
}

// ---------------------------------------------------------------------------
// AC6 — SessionFilter: exact full-id match
// ---------------------------------------------------------------------------
#[test]
fn ac6_session_filter_exact_match() {
    // Exact 32-char hex matches the specific memory.
    let sid = "deadbeef0123456789abcdef01234567";
    let other = "aabbccdd0011223344556677aabbccdd";
    let sf = parse_filter_arg(sid).unwrap();
    assert!(
        matches!(sf, SessionFilter::Exact(_)),
        "32 hex chars must produce Exact variant"
    );
    assert!(sf.matches(Some(sid)), "Exact should match self");
    assert!(!sf.matches(Some(other)), "Exact should not match other");
    assert!(!sf.matches(None), "Exact should not match unstamped memory");
}

// ---------------------------------------------------------------------------
// AC7 — SessionFilter: prefix (≥8 hex chars) matches; <8 chars errors
// ---------------------------------------------------------------------------
#[test]
fn ac7_session_filter_prefix_match() {
    let sid = "deadbeef0123456789abcdef01234567";
    // 8-char prefix should produce Prefix and match the full id.
    let sf8 = parse_filter_arg("deadbeef").unwrap();
    assert!(
        matches!(sf8, SessionFilter::Prefix(_)),
        "8 hex chars must produce Prefix variant"
    );
    assert!(sf8.matches(Some(sid)), "prefix should match sid starting with 'deadbeef'");
    assert!(
        !sf8.matches(Some("aabbccdd0123456789abcdef01234567")),
        "prefix should not match a different id"
    );

    // <8 chars must error.
    let err = parse_filter_arg("deadbee");
    assert!(err.is_err(), "7 hex chars should error with 'at least 8 hex chars'");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("8"),
        "error message should mention 8-char minimum; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// AC8 — SessionFilter `current` resolves to the env-chain id at call time
// ---------------------------------------------------------------------------
#[test]
fn ac8_session_filter_current_resolves_via_env() {
    let sid = "f0f0f0f0a1a1a1a1b2b2b2b2c3c3c3c3";
    with_env(
        &[
            (RECALL_ENV_VAR, Some(sid)),
            (AGENTNS_ENV_VAR, None),
        ],
        || {
            let sf = parse_filter_arg("current").unwrap();
            match &sf {
                SessionFilter::Current(id) => {
                    assert_eq!(id, sid, "Current should resolve to RECALL_SESSION_ID env value");
                }
                other => panic!("expected Current variant, got {other:?}"),
            }
            // Should match a memory whose written_by_session == sid.
            assert!(sf.matches(Some(sid)), "current filter should match the resolved id");
            assert!(
                !sf.matches(Some("ffffffffffffffffffffffffffffffff")),
                "current filter should not match a different id"
            );
        },
    );
}

// ---------------------------------------------------------------------------
// AC9 — `recall sessions` lists distinct session ids with counts
// ---------------------------------------------------------------------------
#[test]
fn ac9_sessions_listing_counts_by_session() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let store = FileStore::open(root.clone()).unwrap();
    let idx = Index::open(&paths::index_db(&root)).unwrap();
    let embedder = recall::embeddings::EmbedderKind::Hash.build().unwrap();

    let sid_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1";
    let sid_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb02";

    // Write 2 memories for session A and 1 for session B.
    for i in 0..2_u32 {
        let mut m = Memory::new(Kind::Semantic, Subject::user(), format!("body-a-{i}"));
        m.front.written_by_session = Some(sid_a.to_string());
        let path = store.write(&m).unwrap();
        let vec = embedder.embed(&m.body).unwrap();
        idx.upsert(&m, &path, Some((embedder.id(), &vec))).unwrap();
    }
    {
        let mut m = Memory::new(Kind::Reflective, Subject::self_(), "body-b-0".to_string());
        m.front.written_by_session = Some(sid_b.to_string());
        let path = store.write(&m).unwrap();
        let vec = embedder.embed(&m.body).unwrap();
        idx.upsert(&m, &path, Some((embedder.id(), &vec))).unwrap();
    }

    // Replicate cmd_sessions counting logic (walks store, groups by session).
    let mut counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for item in store.iter_all() {
        let (mem, _) = item.unwrap();
        if let Some(sid) = mem.front.written_by_session {
            *counts.entry(sid).or_insert(0) += 1;
        }
    }

    assert_eq!(counts.get(sid_a).copied(), Some(2), "session A should have 2 memories");
    assert_eq!(counts.get(sid_b).copied(), Some(1), "session B should have 1 memory");

    // Sorting descending by count (mirrors cmd_sessions ordering).
    let mut rows: Vec<(String, u64)> = counts.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    assert_eq!(rows[0].0, sid_a, "session A (count=2) should sort first");
    assert_eq!(rows[1].0, sid_b, "session B (count=1) should sort second");
}

// ---------------------------------------------------------------------------
// AC10 — recall touch updates last_touched_by_session; written_by_session unchanged
// ---------------------------------------------------------------------------
#[test]
fn ac10_touch_updates_last_touched_by_session() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let store = FileStore::open(root.clone()).unwrap();
    let idx = Index::open(&paths::index_db(&root)).unwrap();
    let embedder = recall::embeddings::EmbedderKind::Hash.build().unwrap();

    let write_sid = "cccccccccccccccccccccccccccccccc";
    let touch_sid = "dddddddddddddddddddddddddddddddd";

    // Write a memory stamped with write_sid.
    let mut m = Memory::new(Kind::Semantic, Subject::user(), "touch-test body".to_string());
    m.front.written_by_session = Some(write_sid.to_string());
    m.front.last_touched_by_session = Some(write_sid.to_string());
    let path = store.write(&m).unwrap();
    let vec = embedder.embed(&m.body).unwrap();
    idx.upsert(&m, &path, Some((embedder.id(), &vec))).unwrap();

    // Simulate cmd_touch: bump recall_count and update last_touched_by_session.
    idx.touch_recall(&m.front.id).unwrap();
    let (mut mem_on_disk, disk_path) = store.find_by_id(&m.front.id).unwrap();
    mem_on_disk.front.last_touched_by_session = Some(touch_sid.to_string());
    store.overwrite(&disk_path, &mem_on_disk).unwrap();

    // Verify: last_touched_by_session updated; written_by_session unchanged.
    let (reread, _) = store.find_by_id(&m.front.id).unwrap();
    assert_eq!(
        reread.front.last_touched_by_session.as_deref(),
        Some(touch_sid),
        "last_touched_by_session should reflect the touch session"
    );
    assert_eq!(
        reread.front.written_by_session.as_deref(),
        Some(write_sid),
        "written_by_session must remain unchanged after touch"
    );
}
