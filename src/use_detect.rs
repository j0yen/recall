//! Use-evidence detection for surfaced memories (PRD-recall-use-evidence).
//!
//! `scan_transcript` walks a Claude Code session JSONL file and detects which
//! of the supplied surfaced memory ids were actually used by the assistant,
//! via two heuristics:
//!
//! 1. **N-gram match**: the memory body's most-distinctive 5-word n-gram
//!    appears (case-folded) in any assistant text turn.
//! 2. **API recall**: the memory id appears as a literal substring in any
//!    Bash tool result that looks like a `recall query` or `recall show`
//!    invocation.
//!
//! Both heuristics are conservative (prefer false-negatives over
//! false-positives). The caller writes the result as `used.json`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::Path;
use std::time::Instant;

// ──────────────────────────────────────────────────────────────
// Stopword list (top 100 English words, embedded)
// ──────────────────────────────────────────────────────────────

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "it", "its", "he", "she", "they", "we",
    "you", "i", "my", "our", "your", "his", "her", "their", "its", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "each", "few",
    "more", "most", "other", "some", "such", "than", "too", "very", "just",
    "also", "there", "here", "when", "where", "how", "what", "which",
    "who", "all", "any", "only", "same", "s", "t", "d", "m", "re", "ve",
    "ll", "about", "after", "before", "into", "through", "during", "up",
    "down", "out", "then", "if", "while", "since", "because", "although",
    "new", "one", "two", "three", "get", "set", "use", "make",
];

fn is_stopword(w: &str) -> bool {
    STOPWORDS.contains(&w)
}

// ──────────────────────────────────────────────────────────────
// N-gram extraction
// ──────────────────────────────────────────────────────────────

/// Tokenize text into lowercase ASCII-word tokens (Unicode boundary aware:
/// consecutive alphabetic/numeric characters, everything else is a boundary).
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Score an n-gram window by the sum of inverse stopword indicators:
/// each non-stopword word contributes 1, each stopword contributes 0.
/// Ties broken by first occurrence in the token stream (deterministic).
fn score_ngram(window: &[String]) -> u32 {
    window
        .iter()
        .map(|w| if is_stopword(w.as_str()) { 0_u32 } else { 1_u32 })
        .sum()
}

/// Extract the most distinctive n-gram (n = `n_len`) from `body`.
///
/// Returns a space-joined lowercase string, or `None` if the body has fewer
/// than `n_len` tokens.
///
/// "Distinctive" = fewest stopwords in the window. Among ties the
/// first-occurring window wins. This gives stable, reproducible results
/// without any external frequency table.
pub fn distinctive_ngram(body: &str, n_len: usize) -> Option<String> {
    let tokens = tokenize(body);
    if tokens.len() < n_len {
        return None;
    }
    let windows: Vec<&[String]> = tokens.windows(n_len).collect();
    let best = windows
        .iter()
        .enumerate()
        .max_by_key(|&(i, w)| {
            // higher score = more non-stopwords = more distinctive
            // secondary key: prefer later windows only as a tiebreak we
            // DON'T want; so use (score, reversed index) → first-occurring wins
            let score = score_ngram(w);
            let inv_idx = usize::MAX - i;
            (score, inv_idx)
        })
        .map(|(_, w)| *w)?;
    // Only emit if at least 2 non-stopword words in the window
    if score_ngram(best) < 2 {
        return None;
    }
    Some(best.join(" "))
}

// ──────────────────────────────────────────────────────────────
// Minimal JSONL schema (Claude Code session format)
// ──────────────────────────────────────────────────────────────
//
// Real JSONL schema (from live transcripts):
//   assistant turns: { "type": "assistant", "message": { "role": "assistant",
//                       "content": [{"type":"text","text":"..."},...] } }
//   user/tool turns: { "type": "user", "message": { "role": "user",
//                       "content": [{"type":"tool_result","content":"..."}, ...] } }
// The "type" at top level and "role" inside message are both used for routing.

#[derive(Debug, Deserialize)]
struct TranscriptLine {
    #[serde(rename = "type")]
    line_type: Option<String>,
    message: Option<Message>,
}

#[derive(Debug, Deserialize)]
struct Message {
    role: Option<String>,
    content: Option<serde_json::Value>,
}

/// Collect all text blocks from an assistant content array.
/// Content may be a plain string or an array of typed blocks.
fn extract_assistant_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|block| {
                // { "type": "text", "text": "..." }
                if block.get("type").and_then(|v| v.as_str()) == Some("text") {
                    block.get("text").and_then(|v| v.as_str()).map(str::to_string)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Extract the text of all tool_result blocks inside a user-role content array.
/// Returns each tool result's text as a separate entry.
///
/// Real format: content is an array containing `{"type": "tool_result", "content": "<text>"}`.
/// The "content" inside tool_result may be a string or an array of text blocks.
fn extract_tool_result_texts(content: &serde_json::Value) -> Vec<String> {
    let arr = match content {
        serde_json::Value::Array(a) => a,
        _ => return vec![],
    };
    arr.iter()
        .filter_map(|block| {
            if block.get("type").and_then(|v| v.as_str()) != Some("tool_result") {
                return None;
            }
            let inner = block.get("content")?;
            let text = match inner {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Array(sub) => sub
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|v| v.as_str()).map(str::to_string))
                    .collect::<Vec<_>>()
                    .join(" "),
                _ => return None,
            };
            Some(text)
        })
        .collect()
}

/// Return true if `text` looks like a `recall query` or `recall show` output.
/// Heuristic: if any whitespace-delimited token is a 26-char uppercase ULID,
/// the result might be recall output. Conservative — false negatives are fine.
fn text_contains_recall_id_pattern(text: &str) -> bool {
    text.split_whitespace().any(|tok| {
        tok.len() == 26
            && tok
                .chars()
                .all(|c| c.is_ascii_alphanumeric() && (c.is_ascii_digit() || c.is_ascii_uppercase()))
    })
}

// ──────────────────────────────────────────────────────────────
// Surfaced memory — minimal shape (matches recalled.json / surfaced.json)
// ──────────────────────────────────────────────────────────────

/// A surfaced memory entry: its id and its body text (loaded on demand by
/// the caller from disk via `store::FileStore`).
#[derive(Debug, Clone)]
pub struct SurfacedMemory {
    pub id: String,
    pub body: String,
}

// ──────────────────────────────────────────────────────────────
// Scan result
// ──────────────────────────────────────────────────────────────

/// Summary returned by `scan_transcript`.
#[derive(Debug, Clone, Serialize)]
pub struct ScanResult {
    pub session_id: String,
    pub surfaced: usize,
    pub used: usize,
    pub ngram_hits: usize,
    pub id_hits: usize,
    pub transcript_bytes: u64,
    pub scan_ms: u128,
    pub used_ids: Vec<String>,
}

// ──────────────────────────────────────────────────────────────
// Main scan entry point
// ──────────────────────────────────────────────────────────────

/// Scan the Claude Code session JSONL at `transcript_path` and return
/// which surfaced memory ids appear to have been used, plus a summary.
///
/// For each surfaced memory the function:
/// 1. Extracts its most-distinctive 5-word n-gram.
/// 2. Scans every assistant-role text turn for the n-gram (substring match,
///    case-insensitive).
/// 3. Scans every tool-result block for the memory id as a literal substring.
///
/// Returns `Ok(ScanResult)` even when the result set is empty.
/// Returns `Err` only on IO or JSONL parse failure at the file level.
pub fn scan_transcript(
    transcript_path: &Path,
    surfaced: &[SurfacedMemory],
    session_id: &str,
    ngram_len: usize,
) -> Result<ScanResult> {
    let start = Instant::now();

    // Pre-compute n-grams once per surfaced memory (avoid recomputing per line).
    let mut ngrams: HashMap<&str, Option<String>> = HashMap::new();
    for sm in surfaced {
        let ng = distinctive_ngram(&sm.body, ngram_len);
        ngrams.insert(&sm.id, ng);
    }

    let file = std::fs::File::open(transcript_path)
        .with_context(|| format!("open transcript {}", transcript_path.display()))?;
    let transcript_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
    let reader = std::io::BufReader::new(file);

    let mut used_ids: HashSet<String> = HashSet::new();
    let mut ngram_hits = 0_usize;
    let mut id_hits = 0_usize;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue, // skip IO errors on individual lines
        };
        if line.trim().is_empty() {
            continue;
        }

        // Parse minimal structure; skip lines that don't parse
        let parsed: TranscriptLine = match serde_json::from_str(&line) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let line_type = parsed.line_type.as_deref().unwrap_or("");

        if let Some(msg) = &parsed.message {
            let role = msg.role.as_deref().unwrap_or("");
            let content = msg.content.as_ref();

            if role == "assistant" || line_type == "assistant" {
                // Heuristic 1: n-gram match in assistant text turns.
                if let Some(content_val) = content {
                    let text = extract_assistant_text(content_val).to_ascii_lowercase();
                    if !text.is_empty() {
                        for sm in surfaced {
                            if used_ids.contains(&sm.id) {
                                continue;
                            }
                            if let Some(Some(ng)) = ngrams.get(sm.id.as_str()) {
                                if text.contains(ng.as_str()) {
                                    used_ids.insert(sm.id.clone());
                                    ngram_hits += 1;
                                }
                            }
                        }
                    }
                }
            } else if role == "user" || line_type == "user" {
                // Tool results are embedded in user messages as:
                // content: [{"type": "tool_result", "content": "..."}]
                // Heuristic 2: look for recall id patterns in tool result text.
                if let Some(content_val) = content {
                    let tool_texts = extract_tool_result_texts(content_val);
                    for tr_text in &tool_texts {
                        if text_contains_recall_id_pattern(tr_text) {
                            for sm in surfaced {
                                if used_ids.contains(&sm.id) {
                                    continue;
                                }
                                if tr_text.contains(&sm.id) {
                                    used_ids.insert(sm.id.clone());
                                    id_hits += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let scan_ms = start.elapsed().as_millis();
    let mut used_ids_sorted: Vec<String> = used_ids.into_iter().collect();
    used_ids_sorted.sort();

    Ok(ScanResult {
        session_id: session_id.to_string(),
        surfaced: surfaced.len(),
        used: used_ids_sorted.len(),
        ngram_hits,
        id_hits,
        transcript_bytes,
        scan_ms,
        used_ids: used_ids_sorted,
    })
}

// ──────────────────────────────────────────────────────────────
// Weather dir helpers
// ──────────────────────────────────────────────────────────────

/// Resolve the recall weather dir for a session.
/// Checks `$RECALL_WEATHER_DIR/<sid>/`, falls back to `~/.cache/recall-weather/<sid>/`.
pub fn weather_dir_for_session(sid: &str) -> Result<std::path::PathBuf> {
    let base = if let Ok(custom) = std::env::var("RECALL_WEATHER_DIR") {
        std::path::PathBuf::from(custom)
    } else {
        let home = directories::BaseDirs::new()
            .context("could not resolve user home directory")?;
        home.cache_dir().join("recall-weather")
    };
    Ok(base.join(sid))
}

/// Load surfaced ids from `surfaced.json` (falls back to `recalled.json`
/// for backward compat with sessions pre-dating the surfaced.json hook write).
/// Returns an empty vec if neither file exists.
pub fn load_surfaced_ids(weather_dir: &Path) -> Result<Vec<String>> {
    // Try surfaced.json first (PRD-recall-surfaced-tracking)
    let surfaced_path = weather_dir.join("surfaced.json");
    if surfaced_path.exists() {
        let content = std::fs::read_to_string(&surfaced_path)
            .with_context(|| format!("read {}", surfaced_path.display()))?;
        let ids: Vec<String> = serde_json::from_str(&content)
            .with_context(|| format!("parse {}", surfaced_path.display()))?;
        return Ok(ids);
    }
    // Fallback: recalled.json (pre-surfaced-tracking sessions)
    let recalled_path = weather_dir.join("recalled.json");
    if recalled_path.exists() {
        let content = std::fs::read_to_string(&recalled_path)
            .with_context(|| format!("read {}", recalled_path.display()))?;
        let ids: Vec<String> = serde_json::from_str(&content)
            .with_context(|| format!("parse {}", recalled_path.display()))?;
        return Ok(ids);
    }
    Ok(vec![])
}

/// Write `used.json` to the weather dir. Creates the dir if it does not exist.
pub fn write_used_json(weather_dir: &Path, used_ids: &[String]) -> Result<()> {
    std::fs::create_dir_all(weather_dir)
        .with_context(|| format!("create weather dir {}", weather_dir.display()))?;
    let path = weather_dir.join("used.json");
    let json = serde_json::to_string_pretty(used_ids)?;
    std::fs::write(&path, json.as_bytes())
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

// ──────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── AC2: distinctive n-gram skips stopwords ──────────────
    #[test]
    fn ngram_skips_stopwords() {
        let body = "the user has been working on the recall feedback weather module since 2026-05-25";
        let ng = distinctive_ngram(body, 5).expect("should produce a 5-gram");
        // Must NOT start with the all-stopword prefix
        assert!(
            !ng.starts_with("the user has been"),
            "stopword-heavy n-gram selected: {ng:?}"
        );
        // Should contain "recall" or "feedback" or "weather" (distinctive words)
        let has_distinctive = ng.contains("recall")
            || ng.contains("feedback")
            || ng.contains("weather")
            || ng.contains("module")
            || ng.contains("2026");
        assert!(
            has_distinctive,
            "expected distinctive n-gram from body, got: {ng:?}"
        );
    }

    #[test]
    fn ngram_short_body_returns_none() {
        assert!(distinctive_ngram("too short", 5).is_none());
    }

    #[test]
    fn ngram_exact_length() {
        let body = "recall feedback weather module since";
        let ng = distinctive_ngram(body, 5).expect("5 tokens → should produce 5-gram");
        assert_eq!(ng, "recall feedback weather module since");
    }

    // ── AC3: n-gram hit in assistant text ───────────────────
    #[test]
    fn scan_detects_ngram_in_assistant_text() {
        // The body n-gram extracted from "the user has been working on the recall feedback
        // weather module since 2026-05-25" is "recall feedback weather module since"
        // (score 4: recall, feedback, weather, module non-stopwords; since is a stopword
        //  but included in the window; first window wins among equal-score candidates).
        // The assistant text must contain that exact lowercased substring.
        let transcript_jsonl = r#"{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"The recall feedback weather module since 2026 was tracking session data effectively."}]}}
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp.as_file(), "{}", transcript_jsonl).unwrap();

        let surfaced = vec![SurfacedMemory {
            id: "01ABC123DEF456GHI789JKL012".to_string(),
            body: "the user has been working on the recall feedback weather module since 2026-05-25"
                .to_string(),
        }];

        let result = scan_transcript(tmp.path(), &surfaced, "test-session", 5).unwrap();
        assert_eq!(
            result.used_ids,
            vec!["01ABC123DEF456GHI789JKL012"],
            "expected n-gram hit; result: {result:?}"
        );
        assert_eq!(result.ngram_hits, 1);
        assert_eq!(result.id_hits, 0);
    }

    // ── AC4: id hit in recall tool result ───────────────────
    #[test]
    fn scan_detects_id_in_tool_result() {
        // A tool_result message containing the memory id (simulating recall query output).
        // The id "01KSRM7KFR0TGAFHAK69Y5YHAA" is 26 chars, all uppercase alphanumeric.
        // Real Claude Code JSONL format: tool results are embedded in a "user" role
        // message as {"type":"tool_result","content":[{"type":"text","text":"..."}]}.
        let id = "01KSRM7KFR0TGAFHAK69Y5YHAA";
        let transcript_jsonl = format!(
            r#"{{"type":"user","message":{{"role":"user","content":[{{"type":"tool_result","content":[{{"type":"text","text":"id: {id}\nkind: semantic\nsubject: self\nbody: some content"}}]}}]}}}}
"#,
            id = id
        );
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp.as_file(), "{}", transcript_jsonl).unwrap();

        let surfaced = vec![SurfacedMemory {
            id: id.to_string(),
            body: "some recall body text for testing purposes here".to_string(),
        }];

        let result = scan_transcript(tmp.path(), &surfaced, "test-session", 5).unwrap();
        assert!(
            result.used_ids.contains(&id.to_string()),
            "expected id hit; result: {result:?}"
        );
        assert_eq!(result.id_hits, 1);
    }

    // ── AC5: empty used.json when nothing matches ───────────
    #[test]
    fn scan_empty_result_when_no_match() {
        let transcript_jsonl =
            r#"{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"completely unrelated assistant message about something else entirely"}]}}
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp.as_file(), "{}", transcript_jsonl).unwrap();

        let surfaced = vec![SurfacedMemory {
            id: "01AAABBBCCC444555666777888".to_string(),
            body: "recall feedback weather module since 2026-05-25 tracking data".to_string(),
        }];

        let result = scan_transcript(tmp.path(), &surfaced, "test-session", 5).unwrap();
        assert!(result.used_ids.is_empty(), "expected empty result; got: {result:?}");
        assert_eq!(result.used, 0);

        // Verify write_used_json writes "[]" for empty results
        let tmp_dir = tempfile::TempDir::new().unwrap();
        write_used_json(tmp_dir.path(), &result.used_ids).unwrap();
        let written = std::fs::read_to_string(tmp_dir.path().join("used.json")).unwrap();
        assert_eq!(written.trim(), "[]");
    }

    // ── AC6: graceful on missing transcript (tested at cmd level) ──
    // The scan_transcript function only handles the file-exists path;
    // the caller (cmd_use_detect in main.rs) handles the missing-file case.

    // ── AC7: performance — 5MB JSONL scans under 1000ms ─────
    #[test]
    fn scan_performance_5mb() {
        // Generate a ~5MB JSONL with 1000 assistant turns of ~5KB each.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut f = tmp.as_file();
        let filler = "a ".repeat(2500); // ~5000 chars per line
        for _ in 0..1000 {
            let line = format!(
                r#"{{"type":"message","message":{{"role":"assistant","content":[{{"type":"text","text":"{}"}}]}}}}"#,
                filler.trim()
            );
            writeln!(f, "{}", line).unwrap();
        }
        f.flush().unwrap();

        let surfaced = vec![SurfacedMemory {
            id: "01KSRM7KFR0TGAFHAK69Y5YHAA".to_string(),
            body: "recall feedback weather module tracking session data".to_string(),
        }];

        let start = std::time::Instant::now();
        let result = scan_transcript(tmp.path(), &surfaced, "perf-test", 5).unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 1000,
            "scan took {}ms, budget is 1000ms; transcript_bytes={}",
            elapsed.as_millis(),
            result.transcript_bytes
        );
    }
}
