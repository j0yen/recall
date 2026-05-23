//! recall-memory-linter — walk a memory-store directory, report stale, duplicate,
//! and broken-wiki-link findings.

#![cfg_attr(not(test), forbid(unsafe_code))]
#![allow(
    clippy::trivially_copy_pass_by_ref,
    clippy::option_if_let_else,
    clippy::single_match_else,
    clippy::type_complexity,
    clippy::module_name_repetitions
)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One memory entry on disk.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Absolute or root-relative path to the .md file.
    pub path: PathBuf,
    /// Parsed YAML frontmatter (None if absent or unparseable).
    pub frontmatter: Option<Frontmatter>,
    /// Body bytes after the closing `---` (or whole file when no frontmatter).
    pub body: String,
    /// File mtime.
    pub mtime: DateTime<Utc>,
}

/// Frontmatter fields we recognize.
#[derive(Debug, Clone, Deserialize)]
pub struct Frontmatter {
    /// RFC3339 last-recalled timestamp (optional).
    pub last_recalled_at: Option<DateTime<Utc>>,
    /// RFC3339 created timestamp (optional).
    pub created_at: Option<DateTime<Utc>>,
}

/// Output shape of one lint finding.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Finding {
    /// Memory hasn't been recalled in a while.
    Stale {
        /// File path.
        path: PathBuf,
        /// Effective freshness timestamp considered.
        last_seen: DateTime<Utc>,
        /// Age in days at lint time.
        age_days: i64,
    },
    /// Two or more memories with byte-identical (trimmed) body.
    Duplicate {
        /// Sorted set of duplicate paths.
        paths: Vec<PathBuf>,
    },
    /// A `[[wiki-link]]` whose target couldn't be resolved.
    BrokenLink {
        /// File containing the broken link.
        path: PathBuf,
        /// Target text inside the brackets.
        target: String,
        /// 1-indexed line number.
        line: usize,
    },
}

/// Lint configuration.
#[derive(Debug, Clone)]
pub struct LintConfig {
    /// Threshold for `stale` finding kind.
    pub stale_days: u32,
    /// Glob patterns to ignore.
    pub ignore: Vec<String>,
    /// Frozen "now" (for deterministic testing); defaults to `Utc::now`.
    pub now: DateTime<Utc>,
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            stale_days: 30,
            ignore: vec![],
            now: Utc::now(),
        }
    }
}

/// Walk a root directory and emit every Markdown memory entry.
///
/// # Errors
/// Returns `io::Error` if `root` doesn't exist or can't be read.
pub fn walk_memories(root: &Path, ignore_patterns: &[String]) -> std::io::Result<Vec<MemoryEntry>> {
    if !root.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("root does not exist: {}", root.display()),
        ));
    }
    let compiled: Vec<glob::Pattern> = ignore_patterns
        .iter()
        .filter_map(|p| glob::Pattern::new(p).ok())
        .collect();

    let mut entries = Vec::new();
    let walker = walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(e.depth() > 0 && e.file_type().is_dir() && name.starts_with('.'))
        });

    for entry in walker {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "md") {
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(path);
        let rel_str = rel.to_string_lossy();
        if compiled.iter().any(|p| p.matches(&rel_str)) {
            continue;
        }
        if let Ok(parsed) = parse_memory(path) {
            entries.push(parsed);
        }
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(entries)
}

fn parse_memory(path: &Path) -> std::io::Result<MemoryEntry> {
    let content = std::fs::read_to_string(path)?;
    let mtime: DateTime<Utc> = std::fs::metadata(path)?
        .modified()
        .ok()
        .map_or_else(Utc::now, DateTime::<Utc>::from);
    let (frontmatter, body) = split_frontmatter(&content);
    Ok(MemoryEntry {
        path: path.to_path_buf(),
        frontmatter,
        body: body.to_string(),
        mtime,
    })
}

fn split_frontmatter(content: &str) -> (Option<Frontmatter>, &str) {
    let after_open = if let Some(rest) = content.strip_prefix("---\n") {
        rest
    } else if let Some(rest) = content.strip_prefix("---\r\n") {
        rest
    } else {
        return (None, content);
    };
    // Empty-frontmatter case: closing --- immediately follows the opener.
    if let Some(body) = after_open.strip_prefix("---\n") {
        return (serde_yml::from_str::<Frontmatter>("").ok(), body);
    }
    if let Some(body) = after_open.strip_prefix("---\r\n") {
        return (serde_yml::from_str::<Frontmatter>("").ok(), body);
    }
    let close = "\n---\n";
    let close_crlf = "\n---\r\n";
    let (yaml_text, body) = if let Some(idx) = after_open.find(close) {
        (&after_open[..idx], &after_open[idx + close.len()..])
    } else if let Some(idx) = after_open.find(close_crlf) {
        (&after_open[..idx], &after_open[idx + close_crlf.len()..])
    } else {
        return (None, content);
    };
    let fm: Option<Frontmatter> = serde_yml::from_str::<Frontmatter>(yaml_text).ok();
    (fm, body)
}

/// Run all three lint kinds against the parsed memory entries.
#[must_use]
pub fn lint(entries: &[MemoryEntry], cfg: &LintConfig) -> Vec<Finding> {
    let mut findings = Vec::new();
    findings.extend(stale_findings(entries, cfg));
    findings.extend(duplicate_findings(entries));
    findings.extend(broken_link_findings(entries));
    findings
}

fn stale_findings(entries: &[MemoryEntry], cfg: &LintConfig) -> Vec<Finding> {
    let cutoff = cfg.now - chrono::Duration::days(i64::from(cfg.stale_days));
    let mut out = Vec::new();
    for e in entries {
        let last_seen = freshness(e);
        if last_seen < cutoff {
            let age = (cfg.now - last_seen).num_days();
            out.push(Finding::Stale {
                path: e.path.clone(),
                last_seen,
                age_days: age,
            });
        }
    }
    out
}

fn freshness(e: &MemoryEntry) -> DateTime<Utc> {
    let mut newest = e.mtime;
    if let Some(fm) = &e.frontmatter {
        if let Some(t) = fm.last_recalled_at {
            if t > newest {
                newest = t;
            }
        }
        if let Some(t) = fm.created_at {
            if t > newest {
                newest = t;
            }
        }
    }
    newest
}

fn duplicate_findings(entries: &[MemoryEntry]) -> Vec<Finding> {
    let mut buckets: HashMap<[u8; 32], Vec<PathBuf>> = HashMap::new();
    for e in entries {
        let trimmed = e.body.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut hasher = Sha256::new();
        hasher.update(trimmed.as_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        buckets.entry(digest).or_default().push(e.path.clone());
    }
    let mut groups: Vec<Vec<PathBuf>> = buckets
        .into_values()
        .filter(|v| v.len() >= 2)
        .collect();
    for g in &mut groups {
        g.sort();
    }
    groups.sort_by(|a, b| a.first().cmp(&b.first()));
    groups.into_iter().map(|paths| Finding::Duplicate { paths }).collect()
}

fn broken_link_findings(entries: &[MemoryEntry]) -> Vec<Finding> {
    let mut stems: std::collections::HashSet<String> = std::collections::HashSet::new();
    for e in entries {
        if let Some(stem) = e.path.file_stem() {
            stems.insert(stem.to_string_lossy().to_string());
        }
    }
    let mut out = Vec::new();
    for e in entries {
        for (line_idx, line) in e.body.lines().enumerate() {
            let mut search_from = 0usize;
            while let Some(open) = line[search_from..].find("[[") {
                let abs_open = search_from + open + 2;
                let Some(close_rel) = line[abs_open..].find("]]") else {
                    break;
                };
                let target = &line[abs_open..abs_open + close_rel];
                if !target.is_empty() && !target.contains('\n') {
                    let target_stem = target.strip_suffix(".md").unwrap_or(target);
                    if !stems.contains(target_stem) {
                        out.push(Finding::BrokenLink {
                            path: e.path.clone(),
                            target: target.to_string(),
                            line: line_idx + 1,
                        });
                    }
                }
                search_from = abs_open + close_rel + 2;
            }
        }
    }
    out
}

/// Render findings as deterministic JSON.
///
/// # Errors
/// Returns serde error if serialization fails.
pub fn render_json(findings: &[Finding]) -> serde_json::Result<String> {
    let v = serde_json::json!({ "findings": findings });
    serde_json::to_string(&v)
}

/// Render findings as human-readable text, grouped by kind.
#[must_use]
pub fn render_text(findings: &[Finding]) -> String {
    if findings.is_empty() {
        return String::from("(no findings)\n");
    }
    let mut s = String::new();
    for f in findings {
        match f {
            Finding::Stale { path, last_seen, age_days } => {
                s.push_str(&format!(
                    "stale: {} (last_seen={}, age={}d)\n",
                    path.display(),
                    last_seen.to_rfc3339(),
                    age_days
                ));
            }
            Finding::Duplicate { paths } => {
                s.push_str("duplicate: ");
                for (i, p) in paths.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&p.display().to_string());
                }
                s.push('\n');
            }
            Finding::BrokenLink { path, target, line } => {
                s.push_str(&format!(
                    "broken_link: {}:{} → [[{}]]\n",
                    path.display(),
                    line,
                    target
                ));
            }
        }
    }
    s
}
