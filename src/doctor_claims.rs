//! `recall doctor --check-claims` — spot-check memory body assertions
//! against live filesystem state and binary versions.
//!
//! Fleet 1 scope: two assertion kinds.
//!   - Kind A: filesystem-path assertions (in fenced code blocks or specific
//!     prose prefixes).
//!   - Kind B: version-number assertions for a known binary whitelist.
//!
//! No auto-edits; disconfirmed assertions produce proposals under
//! `<root>/proposals/` for the user to review.

use crate::memory::Memory;
use crate::paths;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

// ── Whitelist of binaries for Kind B version checks ─────────────────────────

const VERSION_BINARY_WHITELIST: &[&str] = &[
    "recall",
    "cargo",
    "rustc",
    "agorabus",
    "episodic-observer",
    "daily-receipt",
    "confidant",
    "letter-curate",
    "zine",
    "reliquary",
    "cadence",
    "peon-ping",
    "bpolicy",
    "ctrace",
    "wchg",
    "procstat",
    "txn-edit",
    "tcap",
    "sbx",
    "pevent",
];

// ── Public types ──────────────────────────────────────────────────────────────

/// One disconfirmed assertion found in a memory body.
#[derive(Debug, Clone)]
pub struct DisconfirmedAssertion {
    /// Line number (1-based) in the memory body where the assertion appears.
    pub line_no: usize,
    /// The raw line text.
    pub line_text: String,
    /// What we were checking.
    pub assertion: AssertionKind,
    /// Human-readable evidence of why it's disconfirmed.
    pub evidence: String,
    /// Advisory action: "update" (partially wrong) or "supersede" (fully wrong).
    pub suggested_action: &'static str,
}

#[derive(Debug, Clone)]
pub enum AssertionKind {
    /// A filesystem path that was stated to exist.
    FilesystemPath { path: String },
    /// A version claim for a known binary.
    BinaryVersion {
        binary: String,
        claimed_version: String,
    },
}

/// Per-memory result from the claims scan.
#[derive(Debug, Clone)]
pub struct MemoryClaimsResult {
    pub memory_id: String,
    pub file_path: PathBuf,
    pub disconfirmed: Vec<DisconfirmedAssertion>,
}

/// Options for `check_claims`.
#[derive(Debug, Clone, Default)]
pub struct CheckClaimsOpts {
    /// Filter to memories whose subject starts with this string.
    pub subject_filter: Option<String>,
    /// Only check memories created/updated within this duration.
    pub since: Option<chrono::Duration>,
    /// If true, report but do not write proposal files.
    pub dry_run: bool,
    /// If true, skip binary version checks (escape hatch for hangy binaries).
    pub no_binary_checks: bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Walk all memories under `root`, extract assertions, verify them, and
/// (unless `--dry-run`) park proposals for disconfirmed ones.
///
/// Returns the list of per-memory results (all scanned memories, including
/// those with zero disconfirmed assertions).
pub fn check_claims(
    root: &Path,
    opts: &CheckClaimsOpts,
    memories: &[(Memory, PathBuf)],
) -> Result<Vec<MemoryClaimsResult>> {
    let mut results = Vec::new();
    let cutoff: Option<DateTime<Utc>> = opts.since.map(|d| Utc::now() - d);

    for (mem, path) in memories {
        // Apply subject filter.
        if let Some(ref subj) = opts.subject_filter {
            if !mem.front.subject.as_str().starts_with(subj.as_str()) {
                continue;
            }
        }

        // Apply since filter: skip memories older than the cutoff.
        if let Some(cut) = cutoff {
            let ts = mem.front.updated_at.unwrap_or(mem.front.created_at);
            if ts < cut {
                continue;
            }
        }

        // Extract and verify assertions.
        let disconfirmed = verify_memory(mem, opts)?;

        // Write a proposal unless dry-run or no disconfirmed assertions.
        if !disconfirmed.is_empty() && !opts.dry_run {
            park_drift_proposal(root, mem, path, &disconfirmed)?;
        }

        results.push(MemoryClaimsResult {
            memory_id: mem.front.id.clone(),
            file_path: path.clone(),
            disconfirmed,
        });
    }

    Ok(results)
}

// ── Assertion extraction & verification ─────────────────────────────────────

fn verify_memory(
    mem: &Memory,
    opts: &CheckClaimsOpts,
) -> Result<Vec<DisconfirmedAssertion>> {
    let mut out = Vec::new();

    // Track which paths/binaries we've already checked to avoid duplicates.
    let mut seen_paths: HashSet<String> = HashSet::new();
    let mut seen_versions: HashSet<(String, String)> = HashSet::new();

    let mut in_code_block = false;

    for (idx, raw_line) in mem.body.lines().enumerate() {
        let line_no = idx + 1;

        // Toggle fenced code block state.
        if raw_line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
        }

        // ── Kind A: filesystem-path assertions ──────────────────────────────
        let path_candidates = extract_path_candidates(raw_line, in_code_block);
        for candidate in path_candidates {
            if seen_paths.contains(&candidate) {
                continue;
            }
            seen_paths.insert(candidate.clone());

            let expanded = tilde_expand(&candidate);
            if !std::path::Path::new(&expanded).exists() {
                let suggested_action = "supersede";
                out.push(DisconfirmedAssertion {
                    line_no,
                    line_text: raw_line.to_string(),
                    assertion: AssertionKind::FilesystemPath {
                        path: candidate,
                    },
                    evidence: format!("path `{expanded}` does not exist on this filesystem"),
                    suggested_action,
                });
            }
        }

        // ── Kind B: version-number assertions ───────────────────────────────
        if !opts.no_binary_checks {
            let version_claims = extract_version_claims(raw_line);
            for (binary, claimed_ver) in version_claims {
                let key = (binary.clone(), claimed_ver.clone());
                if seen_versions.contains(&key) {
                    continue;
                }
                seen_versions.insert(key);

                if let Some(evidence) = check_binary_version(&binary, &claimed_ver)? {
                    out.push(DisconfirmedAssertion {
                        line_no,
                        line_text: raw_line.to_string(),
                        assertion: AssertionKind::BinaryVersion {
                            binary: binary.clone(),
                            claimed_version: claimed_ver,
                        },
                        evidence,
                        suggested_action: "update",
                    });
                }
            }
        }
    }

    Ok(out)
}

// ── Path extraction (Kind A) ─────────────────────────────────────────────────

/// Extract path candidates from a single line.
/// In code blocks: any token matching `(?:/|~/)[A-Za-z0-9_.\-/]+`.
/// In prose: only paths preceded by recognized trigger phrases.
fn extract_path_candidates(line: &str, in_code_block: bool) -> Vec<String> {
    let mut out = Vec::new();

    if in_code_block {
        // Every token that looks like an absolute or tilde path.
        for token in line.split_whitespace() {
            let token = strip_trailing_punctuation(token);
            if looks_like_path(token) {
                out.push(token.to_string());
            }
        }
    } else {
        // Prose: only paths introduced by trigger phrases.
        const TRIGGERS: &[&str] = &[
            "see ",
            "at ",
            "path: ",
            "(see ",
            "lives in ",
            "lives at ",
            "is at ",
        ];
        let lower = line.to_lowercase();
        for trigger in TRIGGERS {
            let mut search_start = 0;
            while let Some(pos) = lower[search_start..].find(trigger) {
                let abs_pos = search_start + pos + trigger.len();
                let remainder = &line[abs_pos..];
                // Grab the next whitespace-delimited token.
                let token = remainder.split_whitespace().next().unwrap_or("");
                let token = strip_trailing_punctuation(token);
                if looks_like_path(token) {
                    out.push(token.to_string());
                }
                search_start = abs_pos;
            }
        }
    }

    out
}

fn looks_like_path(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    // Must start with `/` or `~/` and contain at least one more `/`.
    let core = if let Some(rest) = s.strip_prefix('~') {
        rest
    } else {
        s
    };
    if !core.starts_with('/') {
        return false;
    }
    // No spaces (we already split on whitespace, but guard against embedded ones).
    if s.contains(' ') {
        return false;
    }
    // Must look like a path component, not just `/` or `~/`.
    core.len() > 1
}

fn strip_trailing_punctuation(s: &str) -> &str {
    s.trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | ':' | ')' | '\'' | '"'))
}

fn tilde_expand(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    if s == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return home;
        }
    }
    s.to_string()
}

// ── Version extraction (Kind B) ──────────────────────────────────────────────

/// Return (binary, claimed_version) pairs found in a line.
/// Only matches binary names from the whitelist.
fn extract_version_claims(line: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();

    for binary in VERSION_BINARY_WHITELIST {
        // Look for "<binary> v<version>" or "<binary> <version>" patterns.
        // Case-insensitive match on the binary name.
        let lower_line = line.to_lowercase();
        let lower_bin = binary.to_lowercase();

        let mut search_start = 0;
        while let Some(pos) = lower_line[search_start..].find(lower_bin.as_str()) {
            let abs_pos = search_start + pos;
            // Ensure the match is a word boundary (not part of a larger token).
            let before_ok = abs_pos == 0
                || !lower_line
                    .as_bytes()
                    .get(abs_pos - 1)
                    .map(|b| b.is_ascii_alphanumeric() || *b == b'-' || *b == b'_')
                    .unwrap_or(false);
            let after_pos = abs_pos + binary.len();
            let after_ok = lower_line
                .as_bytes()
                .get(after_pos)
                .map(|b| !b.is_ascii_alphanumeric() && *b != b'-' && *b != b'_')
                .unwrap_or(true);

            if before_ok && after_ok {
                // Scan forward (up to 5 tokens) for a version token.
                let remainder = &line[after_pos..];
                let tokens: Vec<&str> = remainder.split_whitespace().take(5).collect();
                for token in &tokens {
                    let token = strip_trailing_punctuation(token);
                    // Strip leading 'v' for matching.
                    let ver_part = token.strip_prefix('v').unwrap_or(token);
                    if is_version_token(ver_part) {
                        out.push((binary.to_string(), ver_part.to_string()));
                        break;
                    }
                }
            }

            search_start = abs_pos + 1;
            if search_start >= lower_line.len() {
                break;
            }
        }
    }

    out
}

fn is_version_token(s: &str) -> bool {
    // Match `\d+\.\d+(\.\d+)?`
    let mut parts = s.splitn(4, '.');
    let major = parts.next().unwrap_or("");
    let minor = parts.next().unwrap_or("");
    if major.is_empty() || minor.is_empty() {
        return false;
    }
    if !major.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    if !minor.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    if let Some(patch) = parts.next() {
        if !patch.chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
    }
    true
}

/// Check `binary --version` output against `claimed_version`.
/// Returns `Some(evidence)` if disconfirmed, `None` if confirmed or binary absent-but-skip.
/// Binary absence counts as disconfirmed (the memory references something that doesn't exist).
fn check_binary_version(
    binary: &str,
    claimed: &str,
) -> Result<Option<String>> {
    // Try to locate the binary.
    let bin_path = find_binary(binary);

    match bin_path {
        None => {
            // Binary not on PATH — disconfirmed.
            Ok(Some(format!(
                "`{binary}` is not available on $PATH; the claim references a binary that doesn't exist"
            )))
        }
        Some(bin) => {
            // Run with a 2-second timeout.
            let output = run_with_timeout(
                &bin,
                &["--version"],
                Duration::from_secs(2),
            );
            match output {
                Err(e) => {
                    // Couldn't run — skip silently (avoids false positives for
                    // binaries that don't support --version).
                    let _ = e;
                    Ok(None)
                }
                Ok(out) => {
                    let text = String::from_utf8_lossy(&out.stdout).to_string()
                        + &String::from_utf8_lossy(&out.stderr);
                    // Parse any `\d+\.\d+(\.\d+)?` token from the output.
                    if let Some(live_ver) = extract_first_version_token(&text) {
                        if live_ver != claimed && !versions_compatible(&live_ver, claimed) {
                            return Ok(Some(format!(
                                "`{binary} --version` reports `{live_ver}`; memory claims `{claimed}`"
                            )));
                        }
                    }
                    Ok(None)
                }
            }
        }
    }
}

/// Look up a binary in `~/.local/bin`, `~/.cargo/bin`, then `which`.
fn find_binary(binary: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.local/bin/{binary}"),
        format!("{home}/.cargo/bin/{binary}"),
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    // Fall back to PATH via `which`.
    if let Ok(out) = Command::new("which").arg(binary).output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return Some(PathBuf::from(s));
            }
        }
    }
    None
}

/// Run a command with a timeout. Returns the output if it completes in time.
fn run_with_timeout(
    bin: &Path,
    args: &[&str],
    timeout: Duration,
) -> Result<std::process::Output> {
    // Use `timeout(1)` for portability, wrapping the actual binary.
    let secs = timeout.as_secs().max(1).to_string();
    let mut cmd = Command::new("timeout");
    cmd.arg(&secs);
    cmd.arg(bin);
    for a in args {
        cmd.arg(a);
    }
    cmd.output().context("run binary --version")
}

/// Find the first `\d+\.\d+(\.\d+)?` token in a string.
fn extract_first_version_token(s: &str) -> Option<String> {
    for token in s.split_whitespace() {
        let token = strip_trailing_punctuation(token);
        let ver_part = token.strip_prefix('v').unwrap_or(token);
        if is_version_token(ver_part) {
            return Some(ver_part.to_string());
        }
    }
    None
}

/// True if two version strings are "close enough" (same major.minor,
/// possibly different patch). Used to suppress spurious drift on patch bumps
/// when the memory only mentions major.minor.
fn versions_compatible(live: &str, claimed: &str) -> bool {
    // If claimed has no patch component and live shares major.minor: compatible.
    let live_parts: Vec<&str> = live.splitn(3, '.').collect();
    let claimed_parts: Vec<&str> = claimed.splitn(3, '.').collect();
    if claimed_parts.len() == 2 && live_parts.len() >= 2 {
        return live_parts[0] == claimed_parts[0] && live_parts[1] == claimed_parts[1];
    }
    false
}

// ── Proposal writing ─────────────────────────────────────────────────────────

fn park_drift_proposal(
    root: &Path,
    mem: &Memory,
    mem_path: &Path,
    disconfirmed: &[DisconfirmedAssertion],
) -> Result<PathBuf> {
    use ulid::Ulid;

    let proposals_dir = paths::proposals_dir(root);
    std::fs::create_dir_all(&proposals_dir)?;

    let ulid = Ulid::new().to_string();
    let filename = format!("{ulid}.md");
    let proposal_path = proposals_dir.join(&filename);
    let tmp_path = proposals_dir.join(format!(".{filename}.tmp"));

    let detected_at = Utc::now().to_rfc3339();
    let about_memory = mem_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&mem.front.id)
        .to_string();
    let mem_path_str = mem_path.to_string_lossy();

    // Compute suggested_action: if any assertion is "supersede", use that;
    // otherwise "update".
    let top_action = if disconfirmed.iter().any(|d| d.suggested_action == "supersede") {
        "supersede"
    } else {
        "update"
    };

    // Build the proposal content.
    // Uses a simplified frontmatter + body (NOT a full recall Memory so we
    // don't accidentally add it to the index with wrong metadata).
    let mut body = format!(
        "---\nname: drift-proposal-{about_memory}\ndescription: >-\n  Memory `{about_memory}` contains claims that no longer match live state.\nmetadata:\n  source: doctor-claims\n  type: drift-proposal\n  about_memory: {about_memory}\n  about_memory_path: {mem_path_str}\n  detected_at: {detected_at}\n  suggested_action: {top_action}\n---\n\n"
    );

    body.push_str(&format!(
        "# Drift candidates in `{about_memory}`\n\n"
    ));

    for (i, d) in disconfirmed.iter().enumerate() {
        let kind_label = match &d.assertion {
            AssertionKind::FilesystemPath { .. } => "path",
            AssertionKind::BinaryVersion { .. } => "version",
        };
        body.push_str(&format!(
            "## Assertion {} (disconfirmed — {})\n\n",
            i + 1,
            kind_label
        ));
        body.push_str(&format!("> Line {}: `{}`\n\n", d.line_no, d.line_text.trim()));
        body.push_str(&format!("**Live evidence:** {}\n\n", d.evidence));
        body.push_str(&format!(
            "**Suggested action:** `{}`\n\n",
            d.suggested_action
        ));
    }

    body.push_str(&format!(
        "\n---\n\n(Review with `recall proposals list`, promote with `recall proposals --apply {ulid}`, or discard with `recall proposals --discard {ulid}`.)\n"
    ));

    let mut f = std::fs::File::create(&tmp_path)
        .with_context(|| format!("create tmp proposal file {}", tmp_path.display()))?;
    f.write_all(body.as_bytes())
        .with_context(|| format!("write proposal body to {}", tmp_path.display()))?;
    drop(f);

    std::fs::rename(&tmp_path, &proposal_path).with_context(|| {
        format!(
            "rename {} -> {}",
            tmp_path.display(),
            proposal_path.display()
        )
    })?;

    Ok(proposal_path)
}

// ── Summary helpers ───────────────────────────────────────────────────────────

/// Print a human-readable summary of the check-claims run.
pub fn print_summary(
    results: &[MemoryClaimsResult],
    dry_run: bool,
    format: &str,
) -> Result<()> {
    let scanned = results.len();
    let total_assertions: usize = results.iter().map(|r| r.disconfirmed.len()).sum();
    let memories_with_drift: usize = results.iter().filter(|r| !r.disconfirmed.is_empty()).count();

    if format == "json" {
        let arr: Vec<serde_json::Value> = results
            .iter()
            .filter(|r| !r.disconfirmed.is_empty())
            .map(|r| {
                let assertions: Vec<serde_json::Value> = r
                    .disconfirmed
                    .iter()
                    .map(|d| {
                        let (kind, detail) = match &d.assertion {
                            AssertionKind::FilesystemPath { path } => {
                                ("path", serde_json::json!({ "path": path }))
                            }
                            AssertionKind::BinaryVersion {
                                binary,
                                claimed_version,
                            } => (
                                "version",
                                serde_json::json!({ "binary": binary, "claimed_version": claimed_version }),
                            ),
                        };
                        serde_json::json!({
                            "line_no": d.line_no,
                            "line_text": d.line_text.trim(),
                            "assertion_kind": kind,
                            "assertion_detail": detail,
                            "evidence": d.evidence,
                            "suggested_action": d.suggested_action,
                        })
                    })
                    .collect();
                serde_json::json!({
                    "memory_id": r.memory_id,
                    "file_path": r.file_path.to_string_lossy(),
                    "disconfirmed_assertions": assertions,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    } else {
        println!("check-claims  : scanned {scanned} memories");
        println!(
            "disconfirmed  : {total_assertions} assertions across {memories_with_drift} memories"
        );
        if dry_run {
            println!("mode          : dry-run (no proposals written)");
        } else {
            println!(
                "proposals     : {} written to proposals/",
                memories_with_drift
            );
        }
        if total_assertions > 0 {
            println!();
            for r in results {
                if r.disconfirmed.is_empty() {
                    continue;
                }
                println!(
                    "  {} ({})",
                    r.memory_id,
                    r.file_path.file_name().and_then(|s| s.to_str()).unwrap_or("")
                );
                for d in &r.disconfirmed {
                    let kind_label = match &d.assertion {
                        AssertionKind::FilesystemPath { path } => {
                            format!("path:{path}")
                        }
                        AssertionKind::BinaryVersion {
                            binary,
                            claimed_version,
                        } => format!("version:{binary}@{claimed_version}"),
                    };
                    println!(
                        "    line {:>4}  [{}]  {}",
                        d.line_no,
                        kind_label,
                        d.suggested_action
                    );
                    println!("              {}", d.evidence);
                }
            }
        }
    }
    Ok(())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ── Path extraction ──────────────────────────────────────────────────────

    #[test]
    fn path_in_code_block_extracted() {
        let candidates = extract_path_candidates("/home/jsy/.claude/recall", true);
        assert!(
            candidates.contains(&"/home/jsy/.claude/recall".to_string()),
            "expected path in candidates: {candidates:?}"
        );
    }

    #[test]
    fn tilde_path_in_code_block_extracted() {
        let candidates = extract_path_candidates("~/.claude/recall/memories", true);
        assert!(
            candidates.contains(&"~/.claude/recall/memories".to_string()),
            "expected tilde path: {candidates:?}"
        );
    }

    #[test]
    fn bare_path_in_prose_not_extracted() {
        // Without a trigger phrase, paths in prose are NOT extracted.
        let candidates =
            extract_path_candidates("the file /tmp/foo.txt should be checked", false);
        assert!(
            candidates.is_empty(),
            "bare prose path should not be extracted: {candidates:?}"
        );
    }

    #[test]
    fn triggered_prose_path_extracted() {
        let candidates =
            extract_path_candidates("see /home/jsy/wintermute/recall for details", false);
        assert!(
            !candidates.is_empty(),
            "triggered prose path should be extracted"
        );
        assert!(candidates.contains(&"/home/jsy/wintermute/recall".to_string()));
    }

    #[test]
    fn tilde_expand_works() {
        // Use a fixed home derived from the actual env rather than set_var
        // (set_var is unsafe in Rust 2024 edition; avoid it here).
        // Test the absolute-path passthrough (no env dependency).
        assert_eq!(tilde_expand("/absolute/path"), "/absolute/path");
        // Test tilde expansion with the real HOME value.
        if let Ok(home) = std::env::var("HOME") {
            assert_eq!(tilde_expand("~/foo/bar"), format!("{home}/foo/bar"));
        }
    }

    // ── Version extraction ───────────────────────────────────────────────────

    #[test]
    fn version_token_recognised() {
        assert!(is_version_token("0.6.0"));
        assert!(is_version_token("1.85"));
        assert!(is_version_token("4.5.1"));
        assert!(!is_version_token("abc"));
        assert!(!is_version_token("v0.6.0")); // leading v is stripped by caller
        assert!(!is_version_token("1"));
    }

    #[test]
    fn recall_version_claim_extracted() {
        let claims = extract_version_claims("recall v0.6.0 ships today");
        assert_eq!(claims.len(), 1, "should find one claim: {claims:?}");
        let (bin, ver) = &claims[0];
        assert_eq!(bin, "recall");
        assert_eq!(ver, "0.6.0");
    }

    #[test]
    fn rustc_version_claim_extracted() {
        let claims = extract_version_claims("requires rustc 1.85 or later");
        let found = claims.iter().any(|(b, v)| b == "rustc" && v == "1.85");
        assert!(found, "rustc 1.85 claim not found: {claims:?}");
    }

    #[test]
    fn non_whitelisted_binary_not_extracted() {
        let claims = extract_version_claims("foo-bar 1.2.3 is used");
        assert!(
            claims.is_empty(),
            "non-whitelisted binary should not be extracted"
        );
    }

    #[test]
    fn versions_compatible_patch_suppression() {
        // memory says "1.85" (no patch), live reports "1.85.0" — should be compatible.
        assert!(versions_compatible("1.85.0", "1.85"));
        // Different minor — not compatible.
        assert!(!versions_compatible("1.86.0", "1.85"));
    }

    // ── Integration: verify_memory on synthetic body ─────────────────────────

    #[test]
    fn nonexistent_path_disconfirmed() {
        use crate::memory::{Kind, Memory, Subject};
        let body = "```\n/this/path/does/not/exist/at/all\n```";
        let mem = Memory::new(Kind::Semantic, Subject::self_(), body);
        let opts = CheckClaimsOpts {
            no_binary_checks: true,
            ..Default::default()
        };
        let disconfirmed = verify_memory(&mem, &opts).unwrap();
        assert!(
            !disconfirmed.is_empty(),
            "non-existent path should be disconfirmed"
        );
        assert!(matches!(
            &disconfirmed[0].assertion,
            AssertionKind::FilesystemPath { path } if path.contains("this/path/does/not/exist")
        ));
    }

    #[test]
    fn existing_path_confirmed() {
        use crate::memory::{Kind, Memory, Subject};
        let body = "```\n/tmp\n```";
        let mem = Memory::new(Kind::Semantic, Subject::self_(), body);
        let opts = CheckClaimsOpts {
            no_binary_checks: true,
            ..Default::default()
        };
        let disconfirmed = verify_memory(&mem, &opts).unwrap();
        // /tmp always exists; should be confirmed (no disconfirmed entries for /tmp).
        let tmp_disconfirmed: Vec<_> = disconfirmed
            .iter()
            .filter(|d| matches!(&d.assertion, AssertionKind::FilesystemPath { path } if path == "/tmp"))
            .collect();
        assert!(tmp_disconfirmed.is_empty(), "/tmp should be confirmed");
    }

    #[test]
    fn dry_run_writes_no_proposals() {
        use crate::memory::{Kind, Memory, Subject};
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let body = "```\n/absolutely/nonexistent/path/xyz\n```";
        let mem = Memory::new(Kind::Semantic, Subject::self_(), body);
        let mem_path = root.join("memories/self/test.md");
        std::fs::create_dir_all(root.join("memories/self")).unwrap();
        std::fs::write(&mem_path, "body").unwrap();

        let opts = CheckClaimsOpts {
            dry_run: true,
            no_binary_checks: true,
            ..Default::default()
        };
        let results = check_claims(root, &opts, &[(mem, mem_path)]).unwrap();
        assert!(!results[0].disconfirmed.is_empty());
        // No proposal files should exist.
        let proposals_dir = root.join("proposals");
        if proposals_dir.exists() {
            let count = std::fs::read_dir(&proposals_dir)
                .unwrap()
                .filter_map(Result::ok)
                .count();
            assert_eq!(count, 0, "dry-run should write no proposals");
        }
    }

    #[test]
    fn proposal_written_when_not_dry_run() {
        use crate::memory::{Kind, Memory, Subject};
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let body = "```\n/absolutely/nonexistent/path/xyz\n```";
        let mem = Memory::new(Kind::Semantic, Subject::self_(), body);
        let mem_path = root.join("memories/self/test.md");
        std::fs::create_dir_all(root.join("memories/self")).unwrap();
        std::fs::write(&mem_path, "body").unwrap();

        let opts = CheckClaimsOpts {
            dry_run: false,
            no_binary_checks: true,
            ..Default::default()
        };
        let results = check_claims(root, &opts, &[(mem, mem_path)]).unwrap();
        assert!(!results[0].disconfirmed.is_empty());
        // A proposal file should exist.
        let proposals_dir = root.join("proposals");
        assert!(proposals_dir.exists(), "proposals dir should be created");
        let count = std::fs::read_dir(&proposals_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
            .count();
        assert_eq!(count, 1, "one proposal should be written");
    }

    #[test]
    fn subject_filter_applied() {
        use crate::memory::{Kind, Memory, Subject};
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let body = "```\n/absolutely/nonexistent/path/xyz\n```";
        let mem = Memory::new(Kind::Semantic, Subject::user(), body);
        let mem_path = root.join("memories/user/test.md");
        std::fs::create_dir_all(root.join("memories/user")).unwrap();
        std::fs::write(&mem_path, "body").unwrap();

        // Filter to "self" — should skip the "user" memory.
        let opts = CheckClaimsOpts {
            subject_filter: Some("self".to_string()),
            dry_run: true,
            no_binary_checks: true,
            ..Default::default()
        };
        let results = check_claims(root, &opts, &[(mem, mem_path)]).unwrap();
        // Result list should be empty (memory was filtered out).
        assert!(results.is_empty(), "user memory should be filtered by --subject self");
    }
}
