//! Session-id resolution per `PRD-recall-session-stamp.md` §2.2.
//!
//! Precedence chain (first non-empty wins):
//!
//! 1. `RECALL_SESSION_ID` env var (explicit override).
//! 2. `AGENTNS_SESSION_ID` env var (hook / agentns-claude mock).
//! 3. `/proc/self/agent_session` (kernel primary source under
//!    `linux-wintermute`).
//! 4. `~/.claude/agentns-session-id` (hook-written side channel).
//! 5. Fallback string `comm:<argv0>:pid:<pid>:uid:<uid>` — same shape
//!    provfs uses when the agent_session id is unavailable.
//!
//! Iter-1 of recall-session-stamp: this module only exposes the
//! resolver. Wiring into `recall save` / query / sessions subcommand
//! lands in subsequent iters.

use std::env;
use std::fs;

pub const RECALL_ENV_VAR: &str = "RECALL_SESSION_ID";
pub const AGENTNS_ENV_VAR: &str = "AGENTNS_SESSION_ID";
pub const PROC_AGENT_SESSION: &str = "/proc/self/agent_session";
pub const HOOK_SIDE_CHANNEL: &str = ".claude/agentns-session-id";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    RecallEnv,
    AgentnsEnv,
    ProcSelf,
    HookFile,
    Fallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub id: String,
    pub source: Source,
}

/// Resolve a session id using the PRD §2.2 precedence chain. Never
/// returns `None`; the fallback form is always available.
pub fn resolve() -> Resolved {
    if let Some(v) = read_env(RECALL_ENV_VAR) {
        return Resolved { id: v, source: Source::RecallEnv };
    }
    if let Some(v) = read_env(AGENTNS_ENV_VAR) {
        return Resolved { id: v, source: Source::AgentnsEnv };
    }
    if let Some(v) = read_proc_self_agent_session() {
        return Resolved { id: v, source: Source::ProcSelf };
    }
    if let Some(v) = read_hook_side_channel() {
        return Resolved { id: v, source: Source::HookFile };
    }
    Resolved { id: fallback_id(), source: Source::Fallback }
}

fn read_env(name: &str) -> Option<String> {
    let raw = env::var(name).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
}

fn read_proc_self_agent_session() -> Option<String> {
    read_session_file(PROC_AGENT_SESSION)
}

fn read_hook_side_channel() -> Option<String> {
    let home = env::var("HOME").ok()?;
    let path = format!("{}/{}", home.trim_end_matches('/'), HOOK_SIDE_CHANNEL);
    read_session_file(&path)
}

fn read_session_file(path: &str) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.chars().all(|c| c == '0') {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn fallback_id() -> String {
    let argv0 = env::args().next().unwrap_or_else(|| "unknown".to_string());
    let comm = argv0.rsplit('/').next().unwrap_or("unknown").to_string();
    let pid = std::process::id();
    let uid = read_uid_from_status();
    format!("comm:{}:pid:{}:uid:{}", comm, pid, uid)
}

fn read_uid_from_status() -> u32 {
    let Ok(s) = fs::read_to_string("/proc/self/status") else { return 0 };
    for line in s.lines() {
        // Nested if-let rather than a let-chain: let-chains are unstable
        // until rustc 1.88; this repo's MSRV is 1.85 (single-channel toolchain).
        if let Some(rest) = line.strip_prefix("Uid:") {
            if let Some(field) = rest.split_whitespace().next() {
                if let Ok(v) = field.parse::<u32>() {
                    return v;
                }
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_shape_contains_comm_pid_uid() {
        let id = fallback_id();
        assert!(id.starts_with("comm:"), "got: {id}");
        assert!(id.contains(":pid:"), "got: {id}");
        assert!(id.contains(":uid:"), "got: {id}");
    }

    #[test]
    fn empty_or_zero_session_file_treated_as_absent() {
        // /proc/self/agent_session on stock kernels returns 32 ASCII '0'
        // chars; resolver must treat that as "not stamped" so the chain
        // can fall through. We test the helper indirectly by feeding
        // strings to a hidden parser; the public surface is `resolve()`.
        // Here we just exercise the path through `read_session_file` via
        // a tempfile.
        let tmp = std::env::temp_dir().join("recall-session-stamp-iter1.txt");
        std::fs::write(&tmp, "0".repeat(32)).unwrap();
        let s = tmp.to_str().unwrap();
        assert!(read_session_file(s).is_none());
        std::fs::write(&tmp, "   \n").unwrap();
        assert!(read_session_file(s).is_none());
        std::fs::write(&tmp, "deadbeef\n").unwrap();
        assert_eq!(read_session_file(s).as_deref(), Some("deadbeef"));
        let _ = std::fs::remove_file(&tmp);
    }
}
