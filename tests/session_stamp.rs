//! Integration tests for `recall::session::resolve()` precedence chain.
//! PRD-recall-session-stamp.md §2.2 + AC4. Iter-2 deliverable.
//!
//! The frontmatter round-trip half of the shape diagram lands in a later
//! iter — that requires editing `src/memory.rs`, which is currently dirty
//! from sibling PRD-recall-surfaced-tracking work.
//!
//! Tests use a process-wide mutex to serialize env-var mutation (Rust
//! integration tests run in parallel within one binary).

#![allow(clippy::unwrap_used, clippy::expect_used, unsafe_code)]

use std::sync::Mutex;

use recall::session::{resolve, Source, AGENTNS_ENV_VAR, RECALL_ENV_VAR};

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
