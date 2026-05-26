//! PRD-recall-stop-hook-session-id: verify hooks/stop.sh reads session_id
//! from JSON stdin (harness convention) and falls back to $CLAUDE_SESSION_ID.
//! Mirrors the v0.4.2 braid hook fix that landed for post-tool-use.sh and
//! user-prompt-submit.sh.
//!
//! Strategy: stub RECALL_BIN with a shell script that records its argv to a
//! file, then invoke the hook with various stdin/env combinations and assert
//! the recorded argv.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn hook_path() -> PathBuf {
    repo_root().join("hooks").join("stop.sh")
}

fn write_fake_recall(dir: &Path) -> PathBuf {
    let log = dir.join("recall.log");
    let bin = dir.join("recall-stub");
    let script = format!(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> {log}\n",
        log = log.display()
    );
    fs::write(&bin, script).unwrap();
    let mut perms = fs::metadata(&bin).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&bin, perms).unwrap();
    bin
}

fn invoke(stub: &Path, stdin: Option<&str>, env_sid: Option<&str>) -> std::process::Output {
    let mut cmd = Command::new("bash");
    cmd.arg(hook_path())
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        .env("RECALL_BIN", stub)
        .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(v) = env_sid {
        cmd.env("CLAUDE_SESSION_ID", v);
    }
    let mut child = cmd.spawn().unwrap();
    if let Some(payload) = stdin {
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(payload.as_bytes())
            .unwrap();
    }
    child.wait_with_output().unwrap()
}

fn log_lines(dir: &Path) -> Vec<String> {
    let p = dir.join("recall.log");
    match fs::read_to_string(&p) {
        Ok(s) => s.lines().map(|s| s.to_string()).collect(),
        Err(_) => vec![],
    }
}

#[test]
fn ac1_json_session_id_triggers_promote() {
    let dir = tempfile::tempdir().unwrap();
    let stub = write_fake_recall(dir.path());

    let payload = r#"{"session_id":"abc-123","other":"ignored"}"#;
    let out = invoke(&stub, Some(payload), None);
    assert!(out.status.success(), "hook should exit 0, got {:?}", out);

    let lines = log_lines(dir.path());
    assert_eq!(lines.len(), 1, "expected one promote call, got {lines:?}");
    assert!(
        lines[0].contains("promote --session abc-123 --format text"),
        "argv was {:?}",
        lines[0]
    );
}

#[test]
fn ac2_env_fallback_when_json_missing_session_id() {
    let dir = tempfile::tempdir().unwrap();
    let stub = write_fake_recall(dir.path());

    let payload = r#"{"other":"only"}"#;
    let out = invoke(&stub, Some(payload), Some("env-sid-456"));
    assert!(out.status.success(), "hook should exit 0, got {:?}", out);

    let lines = log_lines(dir.path());
    assert_eq!(lines.len(), 1, "expected one promote call, got {lines:?}");
    assert!(
        lines[0].contains("promote --session env-sid-456 --format text"),
        "argv was {:?}",
        lines[0]
    );
}

#[test]
fn ac3_no_session_id_anywhere_is_silent_noop() {
    let dir = tempfile::tempdir().unwrap();
    let stub = write_fake_recall(dir.path());

    let payload = r#"{"unrelated":"x"}"#;
    let out = invoke(&stub, Some(payload), None);
    assert!(out.status.success(), "hook should exit 0, got {:?}", out);

    let lines = log_lines(dir.path());
    assert!(
        lines.is_empty(),
        "expected zero promote calls, got {lines:?}"
    );
}

#[test]
fn ac4_no_stdin_no_env_is_silent_noop() {
    let dir = tempfile::tempdir().unwrap();
    let stub = write_fake_recall(dir.path());

    let out = invoke(&stub, None, None);
    assert!(out.status.success(), "hook should exit 0, got {:?}", out);

    let lines = log_lines(dir.path());
    assert!(
        lines.is_empty(),
        "expected zero promote calls, got {lines:?}"
    );
}

#[test]
fn ac5_json_session_id_wins_over_env() {
    let dir = tempfile::tempdir().unwrap();
    let stub = write_fake_recall(dir.path());

    let payload = r#"{"session_id":"json-wins"}"#;
    let out = invoke(&stub, Some(payload), Some("env-loses"));
    assert!(out.status.success(), "hook should exit 0, got {:?}", out);

    let lines = log_lines(dir.path());
    assert_eq!(lines.len(), 1, "expected one promote call, got {lines:?}");
    assert!(
        lines[0].contains("promote --session json-wins --format text"),
        "argv was {:?}",
        lines[0]
    );
}
