//! recall — local-first agentic memory CLI.

#![allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Duration, Utc};
use clap::{Parser, Subcommand};
use recall::config::Config;
use recall::daemon;
use recall::embeddings::EmbedderKind;
use recall::index::{Index, MetaRow};
use recall::memory::{Evidence, Kind, Memory, Subject};
use recall::observer;
use recall::paths;
use recall::retrieval::{self, Weights};
use recall::scratch;
use recall::store::FileStore;
use std::collections::HashSet;
use std::io::{self, BufRead, Read};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "recall",
    version,
    about = "Local-first agentic memory: file-backed memories with a keyword/FTS5 index and an in-process semantic embedder."
)]
struct Cli {
    /// Override the recall data root (default: `$RECALL_HOME` or `~/.claude/recall`).
    #[arg(long, global = true)]
    root: Option<PathBuf>,

    /// Embedder to use for write/query --hybrid/reindex.
    /// `fastembed` (default) loads BGE-small-en-v1.5 in-process (~130MB lazy
    /// fetch to ~/.cache/fastembed/). `hash` is the v0.1 hashed-feature
    /// embedder — deterministic, dependency-free, non-semantic.
    #[arg(long, global = true, default_value = "fastembed", value_parser = ["hash", "fastembed"])]
    embedder: String,

    /// Active project directory for project-boosted ranking. Defaults to
    /// `$CLAUDE_PROJECT_DIR` or `$PWD`. Pass `--project-dir=""` to disable.
    #[arg(long, global = true)]
    project_dir: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Create the data dir and initialize the `SQLite` index.
    Init,

    /// Write a new memory. Body is read from --body, --file, or stdin.
    Write {
        #[arg(long, default_value = "semantic")]
        kind: String,
        #[arg(long, default_value = "user")]
        subject: String,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long)]
        confidence: Option<f64>,
        #[arg(long)]
        supersedes: Vec<String>,
        /// `path=foo.rs:42` style evidence. Repeatable.
        #[arg(long)]
        evidence: Vec<String>,
        /// e.g. `30d`, `6mo`, `1y`, `never`.
        #[arg(long)]
        decays_after: Option<String>,
    },

    /// Search memories by keyword (and optionally vector). Prints ranked results.
    Query {
        query: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long, default_value = "text")]
        format: String,
        #[arg(long)]
        touch: bool,
        /// Use hybrid retrieval (FTS5 + vector cosine). Off by default.
        #[arg(long)]
        hybrid: bool,
        #[arg(long)]
        subject: Option<String>,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        min_confidence: Option<f64>,
        #[arg(long, default_value_t = false)]
        include_superseded: bool,
        #[arg(long, default_value_t = false)]
        include_decayed: bool,
        /// Caller-side token budget on returned bodies (0 = unlimited).
        #[arg(long)]
        max_tokens: Option<usize>,
    },

    /// List memories (newest first). Optionally filter by subject prefix.
    List {
        #[arg(long)]
        subject: Option<String>,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        since: Option<String>,
        #[arg(long, default_value = "text")]
        format: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long, default_value_t = false)]
        include_superseded: bool,
        #[arg(long, default_value_t = false)]
        include_decayed: bool,
        #[arg(long)]
        max_tokens: Option<usize>,
    },

    /// Show a single memory's Markdown content by id.
    Show {
        id: String,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Find memories whose stored embedding is closest to <id>'s embedding.
    Similar {
        id: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Bump `recall_count` for one or more memories. Returns the new counts.
    Touch {
        /// One or more memory ids.
        ids: Vec<String>,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Edit a memory in place. Preserves id; bumps `updated_at`.
    Update {
        id: String,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
        /// Read body from stdin (mutually exclusive with --body / --file).
        #[arg(long, default_value_t = false)]
        stdin: bool,
        #[arg(long)]
        confidence: Option<f64>,
        #[arg(long)]
        add_evidence: Vec<String>,
        #[arg(long)]
        add_supersedes: Vec<String>,
        #[arg(long)]
        decays_after: Option<String>,
    },

    /// Walk the supersedes chain forward and backward from <id>.
    Lineage {
        id: String,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Delete a memory by id.
    Delete { id: String },

    /// Wipe the `SQLite` index and rebuild it from the files on disk
    /// using the active embedder.
    Reindex,

    /// Print where recall is reading from, including the active embedder id.
    Where,

    /// Audit the store: disk vs index drift, supersedes integrity, embedder mix.
    Doctor {
        /// Run `reindex` for index/disk drift before reporting.
        #[arg(long, default_value_t = false)]
        fix: bool,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// List candidate memories for pruning. Never deletes unless `--apply` is set.
    Gc {
        /// Only consider memories created longer ago than this (e.g. `30d`, `6mo`).
        #[arg(long)]
        older_than: Option<String>,
        /// Only consider memories never recalled (`recall_count == 0`).
        #[arg(long, default_value_t = false)]
        never_recalled: bool,
        /// Actually delete (default is dry-run).
        #[arg(long, default_value_t = false)]
        apply: bool,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Human-readable summary of the store.
    Stats {
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Dump every memory as JSONL on stdout.
    Export {
        #[arg(long, default_value = "jsonl")]
        format: String,
    },

    /// Restore memories from a JSONL file (or stdin). Existing ids are skipped
    /// unless `--overwrite` is passed.
    Import {
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        overwrite: bool,
    },

    /// Phase 3 within-session scratch storage. Scratch entries are excluded
    /// from default `query` / `list` and are promoted to long-term memory
    /// via `recall promote`.
    Scratch {
        #[command(subcommand)]
        op: ScratchOp,
    },

    /// Promote scratch entries from `session/<sid>/` to long-term memory.
    /// Equivalent to: read each scratch file, write it via `store.write`,
    /// index it, then delete the scratch file. Stop-hook wires this.
    Promote {
        #[arg(long)]
        session: Option<String>,
        /// Promote only this id (omit to promote every entry in the session).
        #[arg(long)]
        id: Option<String>,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Phase 4 PostToolUse observer. Reads one JSON event per line on stdin
    /// and parks proposals under `proposals/` for the user to review.
    Observe {
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// List proposed memories awaiting promote/discard.
    Proposals {
        /// Promote this proposal id to long-term memory; deletes the proposal file.
        #[arg(long)]
        apply: Option<String>,
        /// Delete this proposal id without promoting.
        #[arg(long)]
        discard: Option<String>,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Render what changed in the recall store during a session (writes,
    /// updates, touches, promotions). Best-effort: relies on `created_at`
    /// and `updated_at` falling within `--since`.
    SessionDiff {
        #[arg(long)]
        session: Option<String>,
        /// Only include entries created/updated since this duration ago
        /// (e.g. `2h`, `1d`). Defaults to `8h`.
        #[arg(long)]
        since: Option<String>,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Recall daemon lifecycle.
    Daemon {
        #[command(subcommand)]
        op: DaemonOp,
    },

    /// Outcome feedback — bump or decay a memory's confidence based on
    /// whether it helped or misled. See PRD-recall-outcome-feedback.
    Feedback {
        /// Memories to mark as helpful (raises confidence by `accept_delta`).
        #[arg(long, value_delimiter = ' ', num_args = 0..)]
        accept: Vec<String>,
        /// Memories to mark as wrong (lowers confidence by `reject_delta`).
        #[arg(long, value_delimiter = ' ', num_args = 0..)]
        reject: Vec<String>,
        /// Memories to explicitly abstain on — recorded as no-op feedback
        /// (still increments `feedback_count`, leaves confidence unchanged).
        #[arg(long, value_delimiter = ' ', num_args = 0..)]
        abstain: Vec<String>,
        /// Run the decay sweep across every memory after applying ids.
        #[arg(long, default_value_t = false)]
        decay_sweep: bool,
        /// Minimum days between two decay sweeps of the same row
        /// (idempotency gate). Default 1 — once-per-day.
        #[arg(long, default_value_t = 1)]
        min_interval_d: u32,
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[derive(Debug, Subcommand)]
enum DaemonOp {
    /// Ping the daemon's UDS and print `model_id`, `uptime_s`, version.
    /// Exit code 0 if a live daemon answered, 1 if the socket is absent
    /// or unresponsive (so shell scripts can branch on it cheaply).
    Status {
        /// Override the socket path. Defaults to
        /// `$XDG_RUNTIME_DIR/recall.sock` (or `~/.cache/recall/recall.sock`).
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Start a `recalld` instance. By default detaches into the
    /// background and writes the PID file alongside the socket; with
    /// `--foreground` blocks until shutdown (suitable for systemd Type=simple).
    Start {
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        embedder: Option<String>,
        /// Run in the foreground (do not fork). The CLI execs into
        /// `recalld`; SIGTERM/SIGINT shut it down cleanly.
        #[arg(long)]
        foreground: bool,
        /// Override the pidfile path. Defaults to `recall.pid` in the
        /// socket's parent directory.
        #[arg(long)]
        pidfile: Option<PathBuf>,
    },
    /// Stop a running `recalld` by reading its pidfile and sending
    /// SIGTERM, then waiting up to `--wait-secs` for the socket to
    /// disappear. Exit 0 if the daemon stopped cleanly, 1 if it could
    /// not be reached, 2 if it did not exit in time.
    Stop {
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long)]
        pidfile: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        wait_secs: u64,
    },
    /// Stop the daemon (if running) and start a fresh one in the
    /// background. Forwards `--socket`/`--root`/`--embedder` to the
    /// new instance.
    Restart {
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        embedder: Option<String>,
        #[arg(long)]
        pidfile: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        wait_secs: u64,
    },
}

#[derive(Debug, Subcommand)]
enum ScratchOp {
    /// Write a scratch memory under `session/<sid>/<id>.md`.
    Write {
        #[arg(long, default_value = "episodic")]
        kind: String,
        #[arg(long, default_value = "self")]
        subject: String,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long)]
        session: Option<String>,
    },
    /// List scratch entries. Without --session, walks every session dir.
    List {
        #[arg(long)]
        session: Option<String>,
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Show one scratch memory.
    Show {
        id: String,
        #[arg(long)]
        session: Option<String>,
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Wipe a session's scratch dir.
    Clear {
        #[arg(long)]
        session: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = match cli.root {
        Some(r) => r,
        None => paths::root()?,
    };
    let config = Config::load(&root)?;
    let embedder_kind = EmbedderKind::parse(&cli.embedder)?;
    let project_subject = resolve_project_subject(cli.project_dir.as_deref());

    match cli.command {
        Command::Init => cmd_init(&root),
        Command::Write {
            kind,
            subject,
            body,
            file,
            confidence,
            supersedes,
            evidence,
            decays_after,
        } => cmd_write(
            &root,
            &kind,
            &subject,
            body,
            file,
            confidence,
            supersedes,
            evidence,
            decays_after,
            embedder_kind,
        ),
        Command::Query {
            query,
            limit,
            format,
            touch,
            hybrid,
            subject,
            kind,
            since,
            min_confidence,
            include_superseded,
            include_decayed,
            max_tokens,
        } => cmd_query(
            &root,
            &config,
            &query,
            limit,
            &format,
            touch,
            hybrid,
            subject.as_deref(),
            kind.as_deref(),
            since.as_deref(),
            min_confidence,
            include_superseded,
            include_decayed,
            max_tokens.unwrap_or(config.retrieval.max_tokens),
            project_subject.as_deref(),
            embedder_kind,
        ),
        Command::List {
            subject,
            kind,
            since,
            format,
            limit,
            include_superseded,
            include_decayed,
            max_tokens,
        } => cmd_list(
            &root,
            subject.as_deref(),
            kind.as_deref(),
            since.as_deref(),
            &format,
            limit,
            include_superseded,
            include_decayed,
            max_tokens.unwrap_or(config.retrieval.max_tokens),
        ),
        Command::Show { id, format } => cmd_show(&root, &id, &format),
        Command::Delete { id } => cmd_delete(&root, &id),
        Command::Reindex => cmd_reindex(&root, embedder_kind),
        Command::Similar { id, limit, format } => cmd_similar(&root, &id, limit, &format),
        Command::Touch { ids, format } => cmd_touch(&root, ids, &format),
        Command::Update {
            id,
            body,
            file,
            stdin,
            confidence,
            add_evidence,
            add_supersedes,
            decays_after,
        } => cmd_update(
            &root,
            &id,
            body,
            file,
            stdin,
            confidence,
            add_evidence,
            add_supersedes,
            decays_after,
            embedder_kind,
        ),
        Command::Lineage { id, format } => cmd_lineage(&root, &id, &format),
        Command::Doctor { fix, format } => cmd_doctor(&root, fix, &format, embedder_kind),
        Command::Gc {
            older_than,
            never_recalled,
            apply,
            format,
        } => cmd_gc(
            &root,
            older_than.as_deref(),
            never_recalled,
            apply,
            &format,
        ),
        Command::Stats { format } => cmd_stats(&root, &format),
        Command::Export { format } => cmd_export(&root, &format),
        Command::Import { file, overwrite } => cmd_import(&root, file, overwrite, embedder_kind),
        Command::Scratch { op } => cmd_scratch(&root, op),
        Command::Promote {
            session,
            id,
            format,
        } => cmd_promote(&root, session.as_deref(), id.as_deref(), &format, embedder_kind),
        Command::Observe { file, format } => cmd_observe(&root, file, &format),
        Command::Proposals {
            apply,
            discard,
            format,
        } => cmd_proposals(
            &root,
            apply.as_deref(),
            discard.as_deref(),
            &format,
            embedder_kind,
        ),
        Command::SessionDiff {
            session,
            since,
            format,
        } => cmd_session_diff(&root, session.as_deref(), since.as_deref(), &format),
        Command::Where => {
            println!("{}", root.display());
            println!("embedder: {}", cli.embedder);
            if let Some(p) = &project_subject {
                println!("project_subject: {p}");
            }
            if let Ok(sock) = daemon::default_socket_path() {
                let alive = sock.exists() && ping_socket_sync(&sock).is_ok();
                println!(
                    "socket: {} ({})",
                    sock.display(),
                    if alive { "alive" } else { "absent" }
                );
            }
            Ok(())
        }
        Command::Daemon { op } => cmd_daemon(op),
        Command::Feedback {
            accept,
            reject,
            abstain,
            decay_sweep,
            min_interval_d,
            format,
        } => cmd_feedback(
            &root,
            &config,
            accept,
            reject,
            abstain,
            decay_sweep,
            min_interval_d,
            &format,
            embedder_kind,
        ),
    }
}

/// Blocking helper: spin a single-threaded tokio runtime and send a
/// `ping` op over UDS. Returned `serde_json::Value` is the daemon's
/// `{ok: {...}}` body on success, or an `Err` if the socket is dead
/// or the response was malformed.
fn ping_socket_sync(socket: &std::path::Path) -> Result<serde_json::Value> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build current-thread tokio runtime for daemon ping")?;
    let req = serde_json::json!({ "op": "ping", "args": {} });
    let resp = rt.block_on(daemon::client_roundtrip(socket, &req))?;
    if let Some(ok) = resp.get("ok") {
        Ok(ok.clone())
    } else if let Some(err) = resp.get("error") {
        Err(anyhow!("daemon ping returned error: {err}"))
    } else {
        Err(anyhow!("daemon ping returned malformed response: {resp}"))
    }
}

/// Blocking helper: send a `query` op over UDS and return the
/// `ranked_hits` array from the daemon's `{ok: {...}}` body. Returns
/// `Err` if the socket is dead, the daemon returned an `{error: ...}`
/// frame, or the response shape is malformed.
fn query_socket_sync(
    socket: &std::path::Path,
    text: &str,
    limit: usize,
    hybrid: bool,
    project_subject: Option<&str>,
) -> Result<Vec<serde_json::Value>> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build current-thread tokio runtime for daemon query")?;
    let mut args = serde_json::json!({
        "text": text,
        "limit": limit,
        "hybrid": hybrid,
    });
    if let Some(p) = project_subject {
        args["project_subject"] = serde_json::Value::String(p.to_string());
    }
    let req = serde_json::json!({ "op": "query", "args": args });
    let resp = rt.block_on(daemon::client_roundtrip(socket, &req))?;
    if let Some(ok) = resp.get("ok") {
        Ok(ok
            .get("ranked_hits")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default())
    } else if let Some(err) = resp.get("error") {
        Err(anyhow!("daemon query returned error: {err}"))
    } else {
        Err(anyhow!("daemon query returned malformed response: {resp}"))
    }
}

/// Print the daemon's `ranked_hits` payload using the same text/json
/// shape as the in-process `cmd_query` output. `max_tokens` applies the
/// same char-budget truncation as the in-process path.
fn render_daemon_query_hits(
    hits: &[serde_json::Value],
    max_tokens: usize,
    format: &str,
) -> Result<()> {
    let mut hits = hits.to_vec();
    if max_tokens > 0 {
        let budget_bytes = max_tokens.saturating_mul(4);
        let mut used = 0_usize;
        let mut cut_at = hits.len();
        for (i, h) in hits.iter().enumerate() {
            let n = h
                .get("snippet")
                .and_then(|s| s.as_str())
                .map_or(0, str::len);
            if used.saturating_add(n) > budget_bytes && i > 0 {
                cut_at = i;
                break;
            }
            used = used.saturating_add(n);
        }
        hits.truncate(cut_at);
    }
    if format == "json" {
        let arr: Vec<serde_json::Value> = hits
            .iter()
            .map(|h| {
                serde_json::json!({
                    "id": h.get("id").cloned().unwrap_or(serde_json::Value::Null),
                    "kind": h.get("kind").cloned().unwrap_or(serde_json::Value::Null),
                    "subject": h.get("subject").cloned().unwrap_or(serde_json::Value::Null),
                    "path": h.get("path").cloned().unwrap_or(serde_json::Value::Null),
                    "snippet": h.get("snippet").cloned().unwrap_or(serde_json::Value::Null),
                    "score": h.get("score").cloned().unwrap_or(serde_json::Value::Null),
                    "vector_sim": h.get("vector_sim").cloned().unwrap_or(serde_json::Value::Null),
                    "confidence": h.get("confidence").cloned().unwrap_or(serde_json::Value::Null),
                    "recall_count": h.get("recall_count").cloned().unwrap_or(serde_json::Value::Null),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    } else {
        if hits.is_empty() {
            println!("(no matches)");
        }
        for h in &hits {
            let id = h.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let kind = h.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            let subj = h.get("subject").and_then(|v| v.as_str()).unwrap_or("");
            let score = h.get("score").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            let vec_sim = h
                .get("vector_sim")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            let recalls = h
                .get("recall_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let snippet = h.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
            println!(
                "{}  [{}/{}]  score={:.3} vec_sim={:.3} recalls={}",
                id, kind, subj, score, vec_sim, recalls
            );
            println!("  {snippet}");
        }
    }
    Ok(())
}

fn cmd_daemon(op: DaemonOp) -> Result<()> {
    match op {
        DaemonOp::Status { socket, format } => {
            let sock = match socket {
                Some(s) => s,
                None => daemon::default_socket_path()?,
            };
            match ping_socket_sync(&sock) {
                Ok(body) => {
                    if format == "json" {
                        let out = serde_json::json!({
                            "daemon_active": true,
                            "socket": sock.display().to_string(),
                            "model_id": body.get("model_id"),
                            "uptime_s": body.get("uptime_s"),
                            "query_count": body.get("query_count"),
                            "version": body.get("version"),
                            "root": body.get("root"),
                        });
                        println!("{}", serde_json::to_string_pretty(&out)?);
                    } else {
                        println!("daemon_active: true");
                        println!("socket: {}", sock.display());
                        if let Some(v) = body.get("model_id").and_then(|x| x.as_str()) {
                            println!("model_id: {v}");
                        }
                        if let Some(v) = body.get("uptime_s").and_then(|x| x.as_u64()) {
                            println!("uptime_s: {v}");
                        }
                        if let Some(v) = body.get("query_count").and_then(|x| x.as_u64()) {
                            println!("query_count: {v}");
                        }
                        if let Some(v) = body.get("version").and_then(|x| x.as_str()) {
                            println!("version: {v}");
                        }
                        if let Some(v) = body.get("root").and_then(|x| x.as_str()) {
                            println!("root: {v}");
                        }
                    }
                    Ok(())
                }
                Err(_e) => {
                    if format == "json" {
                        let out = serde_json::json!({
                            "daemon_active": false,
                            "socket": sock.display().to_string(),
                        });
                        println!("{}", serde_json::to_string_pretty(&out)?);
                    } else {
                        println!("daemon_active: false");
                        println!("socket: {}", sock.display());
                    }
                    // Exit non-zero so shell scripts can branch on the absence.
                    std::process::exit(1);
                }
            }
        }
        DaemonOp::Start {
            socket,
            root,
            embedder,
            foreground,
            pidfile,
        } => {
            let sock = match socket {
                Some(s) => s,
                None => daemon::default_socket_path()?,
            };
            let pid = match pidfile {
                Some(p) => p,
                None => daemon::pid_path_for_socket(&sock)?,
            };
            if pid_alive(&pid) && sock.exists() && ping_socket_sync(&sock).is_ok() {
                eprintln!("recalld already running (pidfile {})", pid.display());
                return Ok(());
            }
            cmd_daemon_start(&sock, root.as_deref(), embedder.as_deref(), foreground, &pid)
        }
        DaemonOp::Stop {
            socket,
            pidfile,
            wait_secs,
        } => {
            let sock = match socket {
                Some(s) => s,
                None => daemon::default_socket_path()?,
            };
            let pid = match pidfile {
                Some(p) => p,
                None => daemon::pid_path_for_socket(&sock)?,
            };
            cmd_daemon_stop(&sock, &pid, wait_secs)
        }
        DaemonOp::Restart {
            socket,
            root,
            embedder,
            pidfile,
            wait_secs,
        } => {
            let sock = match socket {
                Some(s) => s,
                None => daemon::default_socket_path()?,
            };
            let pid = match pidfile {
                Some(p) => p,
                None => daemon::pid_path_for_socket(&sock)?,
            };
            // Stop best-effort; ignore "not running" so restart works
            // when nothing is up yet.
            let _ = cmd_daemon_stop(&sock, &pid, wait_secs);
            cmd_daemon_start(&sock, root.as_deref(), embedder.as_deref(), false, &pid)
        }
    }
}

/// Resolve the `recalld` binary path. Honors `$RECALLD_BIN`, then looks
/// next to the current `recall` executable, then falls back to `recalld`
/// on `$PATH`.
fn locate_recalld() -> PathBuf {
    if let Ok(v) = std::env::var("RECALLD_BIN") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let cand = dir.join("recalld");
            if cand.exists() {
                return cand;
            }
        }
    }
    PathBuf::from("recalld")
}

/// Return true iff the pidfile exists, parses as a PID, and `/proc/<pid>`
/// indicates a live process owned by this user.
fn pid_alive(pidfile: &std::path::Path) -> bool {
    let Ok(raw) = std::fs::read_to_string(pidfile) else {
        return false;
    };
    let Ok(pid) = raw.trim().parse::<u32>() else {
        return false;
    };
    pid_alive_raw(pid)
}

fn pid_alive_raw(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

/// Send a signal by spawning `/bin/kill`. Returns the exit status:
/// 0 → delivered, 1 → no such process / no permission.
fn send_signal(pid: u32, signal: &str) -> std::io::Result<std::process::ExitStatus> {
    std::process::Command::new("/bin/kill")
        .arg(signal)
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
}

fn cmd_daemon_start(
    socket: &std::path::Path,
    root: Option<&std::path::Path>,
    embedder: Option<&str>,
    foreground: bool,
    pidfile: &std::path::Path,
) -> Result<()> {
    let bin = locate_recalld();
    let mut cmd = std::process::Command::new(&bin);
    cmd.arg("--socket").arg(socket);
    cmd.arg("--pidfile").arg(pidfile);
    if let Some(r) = root {
        cmd.arg("--root").arg(r);
    }
    if let Some(e) = embedder {
        cmd.arg("--embedder").arg(e);
    }

    if foreground {
        // Block in the foreground so signals reach this process and
        // recalld's child stdio inherits our terminal. Useful for
        // systemd Type=simple or interactive debugging.
        let status = cmd
            .status()
            .with_context(|| format!("spawn {}", bin.display()))?;
        if status.success() {
            return Ok(());
        }
        return Err(anyhow!("recalld exited with status {status}"));
    }

    // Background: detach via setsid so the daemon survives the
    // launching shell. stdin from /dev/null; stdout/stderr to a log file
    // next to the socket. Use the `setsid(1)` utility for portability
    // (avoids needing libc / unsafe blocks).
    let log_path = socket.with_extension("log");
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("open daemon log {}", log_path.display()))?;
    let log_err = log
        .try_clone()
        .context("clone log fd for stderr")?;
    let mut detach = std::process::Command::new("setsid");
    detach.arg("--fork").arg(&bin);
    detach.arg("--socket").arg(socket);
    detach.arg("--pidfile").arg(pidfile);
    if let Some(r) = root {
        detach.arg("--root").arg(r);
    }
    if let Some(e) = embedder {
        detach.arg("--embedder").arg(e);
    }
    detach
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(log_err));
    let status = detach
        .status()
        .with_context(|| format!("spawn setsid {}", bin.display()))?;
    if !status.success() {
        return Err(anyhow!(
            "setsid recalld exited with status {status}; see {}",
            log_path.display()
        ));
    }
    // Wait briefly for the daemon to become responsive so the CLI exits
    // with a useful status instead of returning before the socket is up.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if socket.exists() && ping_socket_sync(socket).is_ok() {
            let pid_str = std::fs::read_to_string(pidfile)
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| "?".to_string());
            println!("recalld started (pid {pid_str}, socket {})", socket.display());
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(anyhow!(
                "recalld did not respond on {} within 5s; see {}",
                socket.display(),
                log_path.display()
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn cmd_daemon_stop(
    socket: &std::path::Path,
    pidfile: &std::path::Path,
    wait_secs: u64,
) -> Result<()> {
    let raw = match std::fs::read_to_string(pidfile) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("recalld not running (no pidfile at {})", pidfile.display());
            std::process::exit(1);
        }
    };
    let pid: u32 = raw
        .trim()
        .parse()
        .with_context(|| format!("malformed pid in {}", pidfile.display()))?;
    if !pid_alive_raw(pid) {
        eprintln!("recalld pid {pid} not running; cleaning stale pidfile");
        let _ = std::fs::remove_file(pidfile);
        std::process::exit(1);
    }
    let status = send_signal(pid, "-TERM").context("invoke /bin/kill")?;
    if !status.success() {
        return Err(anyhow!("kill -TERM {pid}: exit {status}"));
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(wait_secs);
    loop {
        if !pid_alive_raw(pid) && !socket.exists() {
            println!("recalld stopped (pid {pid})");
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            std::process::exit(2);
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn resolve_project_subject(explicit: Option<&str>) -> Option<String> {
    let raw = match explicit {
        Some(s) if s.is_empty() => return None,
        Some(s) => s.to_string(),
        None => match std::env::var("CLAUDE_PROJECT_DIR") {
            Ok(v) if !v.is_empty() => v,
            _ => std::env::var("PWD").unwrap_or_default(),
        },
    };
    if raw.is_empty() {
        return None;
    }
    let base = std::path::Path::new(&raw)
        .file_name()
        .and_then(|s| s.to_str())?
        .to_string();
    if base.is_empty() {
        None
    } else {
        Some(format!("project:{base}"))
    }
}

fn parse_since(s: &str) -> Result<Duration> {
    let t = s.trim();
    if t.len() < 2 {
        return Err(anyhow!("bad duration: {s}"));
    }
    let (num_part, suffix) = t.split_at(t.len() - 1);
    let n: i64 = num_part
        .parse()
        .map_err(|_| anyhow!("bad duration number in: {s}"))?;
    match suffix {
        "d" => Ok(Duration::days(n)),
        "h" => Ok(Duration::hours(n)),
        "m" => Ok(Duration::minutes(n)),
        _ => Err(anyhow!("unknown duration suffix in: {s} (use d|h|m)")),
    }
}

fn cmd_init(root: &std::path::Path) -> Result<()> {
    let _store = FileStore::open(root.to_path_buf())?;
    let _idx = Index::open(&paths::index_db(root))?;
    println!("initialized {}", root.display());
    Ok(())
}

/// Apply outcome feedback to memories. Writes confidence + feedback_count
/// to both SQLite and the on-disk markdown frontmatter so the file stays
/// canonical.
#[allow(clippy::too_many_arguments)]
fn cmd_feedback(
    root: &std::path::Path,
    config: &Config,
    accept: Vec<String>,
    reject: Vec<String>,
    abstain: Vec<String>,
    decay_sweep: bool,
    min_interval_d: u32,
    format: &str,
    _embedder_kind: EmbedderKind,
) -> Result<()> {
    if accept.is_empty() && reject.is_empty() && abstain.is_empty() && !decay_sweep {
        return Err(anyhow!(
            "feedback requires at least one of --accept, --reject, --abstain, or --decay-sweep"
        ));
    }
    let store = FileStore::open(root.to_path_buf())?;
    let idx = Index::open(&paths::index_db(root))?;
    let cfg = &config.feedback;

    #[derive(Debug)]
    struct Outcome {
        id: String,
        op: &'static str,
        new_confidence: Option<f64>,
        feedback_count: Option<u32>,
        error: Option<String>,
    }

    let mut results: Vec<Outcome> = Vec::new();

    // accept = +accept_delta, reject = -reject_delta, abstain = 0.0 delta
    // (still bumps feedback_count for observability).
    let work: Vec<(String, &'static str, f64)> = accept
        .into_iter()
        .map(|id| (id, "accept", cfg.accept_delta))
        .chain(reject.into_iter().map(|id| (id, "reject", -cfg.reject_delta)))
        .chain(abstain.into_iter().map(|id| (id, "abstain", 0.0)))
        .collect();

    for (id, op, delta) in work {
        match apply_feedback_one(&store, &idx, &id, delta, cfg) {
            Ok((conf, n)) => results.push(Outcome {
                id,
                op,
                new_confidence: Some(conf),
                feedback_count: Some(n),
                error: None,
            }),
            Err(e) => results.push(Outcome {
                id,
                op,
                new_confidence: None,
                feedback_count: None,
                error: Some(format!("{e:#}")),
            }),
        }
    }

    let sweep_count = if decay_sweep {
        Some(idx.apply_decay_sweep(cfg.half_life_d, min_interval_d)?)
    } else {
        None
    };

    if format == "json" {
        let arr: Vec<serde_json::Value> = results
            .iter()
            .map(|o| {
                serde_json::json!({
                    "id": o.id,
                    "op": o.op,
                    "confidence": o.new_confidence,
                    "feedback_count": o.feedback_count,
                    "error": o.error,
                })
            })
            .collect();
        let obj = serde_json::json!({
            "results": arr,
            "decay_sweep_updated": sweep_count,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        for o in &results {
            match (&o.new_confidence, &o.error) {
                (Some(c), _) => println!(
                    "{}  {}  conf={:.3}  feedback_count={}",
                    o.id,
                    o.op,
                    c,
                    o.feedback_count.unwrap_or(0)
                ),
                (None, Some(e)) => println!("{}  {}  ERROR: {}", o.id, o.op, e),
                (None, None) => println!("{}  {}  (no change)", o.id, o.op),
            }
        }
        if let Some(n) = sweep_count {
            println!("decay sweep   : {n} rows updated");
        }
    }
    Ok(())
}

fn apply_feedback_one(
    store: &FileStore,
    idx: &Index,
    id: &str,
    delta: f64,
    cfg: &recall::config::Feedback,
) -> Result<(f64, u32)> {
    let (new_conf, new_count) = idx.apply_feedback_delta(id, delta, cfg)?;
    // Mirror to the markdown file so the file stays the source of truth.
    let (mut mem, path) = store.find_by_id(id)?;
    mem.front.confidence = new_conf;
    mem.front.feedback_count = new_count;
    mem.front.updated_at = Some(Utc::now());
    store.overwrite(&path, &mem)?;
    Ok((new_conf, new_count))
}

fn cmd_write(
    root: &std::path::Path,
    kind: &str,
    subject: &str,
    body: Option<String>,
    file: Option<PathBuf>,
    confidence: Option<f64>,
    supersedes: Vec<String>,
    evidence: Vec<String>,
    decays_after: Option<String>,
    embedder_kind: EmbedderKind,
) -> Result<()> {
    let body_text = read_body(body, file)?;
    if body_text.trim().is_empty() {
        return Err(anyhow!("body is empty"));
    }
    let store = FileStore::open(root.to_path_buf())?;
    let idx = Index::open(&paths::index_db(root))?;
    let kind: Kind = kind.parse()?;
    let mut mem = Memory::new(kind, Subject(subject.to_string()), body_text);
    if let Some(c) = confidence {
        mem.front.confidence = c.clamp(0.0, 1.0);
    }
    mem.front.supersedes = supersedes;
    mem.front.evidence = evidence
        .into_iter()
        .map(|e| parse_evidence(&e))
        .collect::<Result<Vec<_>>>()?;
    mem.front.decays_after = decays_after;
    let path = store.write(&mem)?;
    let embedder = embedder_kind.build()?;
    let vec = embedder.embed(&mem.body)?;
    idx.upsert(&mem, &path, Some((embedder.id(), &vec)))?;
    println!("{}", mem.front.id);
    Ok(())
}

fn parse_evidence(s: &str) -> Result<Evidence> {
    // Supported forms: `path=foo.rs:42`, `session=ID`, `turn=12`, `excerpt=...`.
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| anyhow!("--evidence expects KEY=VALUE, got: {s}"))?;
    let mut ev = Evidence::default();
    match k {
        "path" | "source_path" => ev.source_path = Some(v.to_string()),
        "session" => ev.session = Some(v.to_string()),
        "turn" => ev.turn = Some(v.parse().context("turn must be a number")?),
        "excerpt" => ev.excerpt = Some(v.to_string()),
        _ => return Err(anyhow!("unknown evidence key: {k}")),
    }
    Ok(ev)
}

fn read_body(body: Option<String>, file: Option<PathBuf>) -> Result<String> {
    if let Some(b) = body {
        return Ok(b);
    }
    if let Some(p) = file {
        return std::fs::read_to_string(&p)
            .with_context(|| format!("read body file {}", p.display()));
    }
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf).context("read stdin")?;
    Ok(buf)
}

fn cmd_query(
    root: &std::path::Path,
    config: &Config,
    query: &str,
    limit: usize,
    format: &str,
    touch: bool,
    hybrid: bool,
    subject: Option<&str>,
    kind: Option<&str>,
    since: Option<&str>,
    min_confidence: Option<f64>,
    include_superseded: bool,
    include_decayed: bool,
    max_tokens: usize,
    project_subject: Option<&str>,
    embedder_kind: EmbedderKind,
) -> Result<()> {
    // Auto-forward to the daemon when the call is filter-free: the v0.5.0
    // `query` op has no filter surface yet, so any subject/kind/since/
    // min_confidence selector or `--touch` flag stays on the in-process
    // path. `include_superseded`/`include_decayed` are *true*-means-"no
    // filter"; default is false → in-process. Any forward failure (dead
    // socket, malformed response) silently falls back — PRD AC3.
    let can_forward = subject.is_none()
        && kind.is_none()
        && since.is_none()
        && min_confidence.is_none()
        && include_superseded
        && include_decayed
        && !touch;
    if can_forward {
        if let Ok(sock) = daemon::default_socket_path() {
            if sock.exists() {
                if let Ok(hits) =
                    query_socket_sync(&sock, query, limit, hybrid, project_subject)
                {
                    return render_daemon_query_hits(&hits, max_tokens, format);
                }
            }
        }
    }
    let idx = Index::open(&paths::index_db(root))?;
    let weights: Weights = config.into();
    let need_overfetch = subject.is_some()
        || kind.is_some()
        || since.is_some()
        || min_confidence.is_some()
        || !include_superseded
        || !include_decayed;
    let inner_limit = if need_overfetch {
        limit.saturating_mul(4).max(20)
    } else {
        limit
    };
    let mut hits = if hybrid {
        let embedder = embedder_kind.build()?;
        retrieval::hybrid_with(&idx, embedder.as_ref(), query, inner_limit, weights, project_subject)?
    } else {
        retrieval::search_with(&idx, query, inner_limit, weights, project_subject)?
    };
    let store = if since.is_some() {
        Some(FileStore::open(root.to_path_buf())?)
    } else {
        None
    };
    let cutoff: Option<DateTime<Utc>> = match since {
        Some(s) => Some(Utc::now() - parse_since(s)?),
        None => None,
    };
    let superseded = if include_superseded {
        HashSet::new()
    } else {
        idx.superseded_ids()?
    };
    let decayed = if include_decayed {
        HashSet::new()
    } else {
        idx.decayed_ids()?
    };
    hits.retain(|r| {
        if !include_superseded && superseded.contains(&r.hit.id) {
            return false;
        }
        if !include_decayed && decayed.contains(&r.hit.id) {
            return false;
        }
        if let Some(p) = subject {
            if !r.hit.subject.starts_with(p) {
                return false;
            }
        }
        if let Some(k) = kind {
            if r.hit.kind != k {
                return false;
            }
        }
        if let Some(m) = min_confidence {
            if r.hit.confidence < m {
                return false;
            }
        }
        if let (Some(c), Some(st)) = (cutoff, &store) {
            if let Ok((mem, _)) = st.find_by_id(&r.hit.id) {
                if mem.front.created_at < c {
                    return false;
                }
            }
        }
        true
    });
    hits.truncate(limit);
    if max_tokens > 0 {
        truncate_to_token_budget(&mut hits, max_tokens, |r| r.hit.snippet.len());
    }
    if touch {
        for r in &hits {
            let _ = idx.touch_recall(&r.hit.id);
        }
    }
    if format == "json" {
        let arr: Vec<serde_json::Value> = hits
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.hit.id,
                    "kind": r.hit.kind,
                    "subject": r.hit.subject,
                    "path": r.hit.path.to_string_lossy(),
                    "snippet": r.hit.snippet,
                    "score": r.score,
                    "vector_sim": r.vector_sim,
                    "confidence": r.hit.confidence,
                    "recall_count": r.hit.recall_count,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    } else {
        if hits.is_empty() {
            println!("(no matches)");
        }
        for r in hits {
            println!(
                "{}  [{}/{}]  score={:.3} vec_sim={:.3} recalls={}",
                &r.hit.id, r.hit.kind, r.hit.subject, r.score, r.vector_sim, r.hit.recall_count
            );
            println!("  {}", r.hit.snippet);
        }
    }
    Ok(())
}

/// Truncate a result list so the cumulative body bytes stay below
/// `max_tokens * 4` (rough char→token approximation). Items are kept in
/// rank order; the first item that overflows the budget stops the include.
fn truncate_to_token_budget<T, F: Fn(&T) -> usize>(items: &mut Vec<T>, max_tokens: usize, body_size: F) {
    let budget_bytes = max_tokens.saturating_mul(4);
    if budget_bytes == 0 {
        return;
    }
    let mut used = 0_usize;
    let mut cut_at = items.len();
    for (i, item) in items.iter().enumerate() {
        let n = body_size(item);
        if used.saturating_add(n) > budget_bytes && i > 0 {
            cut_at = i;
            break;
        }
        used = used.saturating_add(n);
    }
    items.truncate(cut_at);
}

fn cmd_list(
    root: &std::path::Path,
    subject: Option<&str>,
    kind: Option<&str>,
    since: Option<&str>,
    format: &str,
    limit: usize,
    include_superseded: bool,
    include_decayed: bool,
    max_tokens: usize,
) -> Result<()> {
    let idx = Index::open(&paths::index_db(root))?;
    let need_overfetch = kind.is_some()
        || since.is_some()
        || !include_superseded
        || !include_decayed;
    let inner_limit = if need_overfetch {
        limit.saturating_mul(4).max(40)
    } else {
        limit
    };
    let mut hits = idx.list(subject, inner_limit)?;
    let store = if since.is_some() {
        Some(FileStore::open(root.to_path_buf())?)
    } else {
        None
    };
    let cutoff: Option<DateTime<Utc>> = match since {
        Some(s) => Some(Utc::now() - parse_since(s)?),
        None => None,
    };
    let superseded = if include_superseded {
        HashSet::new()
    } else {
        idx.superseded_ids()?
    };
    let decayed = if include_decayed {
        HashSet::new()
    } else {
        idx.decayed_ids()?
    };
    hits.retain(|h| {
        if !include_superseded && superseded.contains(&h.id) {
            return false;
        }
        if !include_decayed && decayed.contains(&h.id) {
            return false;
        }
        if let Some(k) = kind {
            if h.kind != k {
                return false;
            }
        }
        if let (Some(c), Some(st)) = (cutoff, &store) {
            if let Ok((mem, _)) = st.find_by_id(&h.id) {
                if mem.front.created_at < c {
                    return false;
                }
            }
        }
        true
    });
    hits.truncate(limit);
    if max_tokens > 0 {
        // For list, body sizes aren't loaded into Hit; approximate per-entry
        // as the id+kind+subject overhead (~64 bytes) so the cap is at least
        // monotone in `limit`. Better fidelity than ignoring max_tokens.
        truncate_to_token_budget(&mut hits, max_tokens, |_| 64);
    }
    if format == "json" {
        let arr: Vec<serde_json::Value> = hits
            .iter()
            .map(|h| {
                serde_json::json!({
                    "id": h.id,
                    "kind": h.kind,
                    "subject": h.subject,
                    "path": h.path.to_string_lossy(),
                    "confidence": h.confidence,
                    "recall_count": h.recall_count,
                    "last_recalled_at": h.last_recalled_at.map(|t| t.to_rfc3339()),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
        return Ok(());
    }
    if hits.is_empty() {
        println!("(no memories)");
    }
    for h in hits {
        println!("{}  [{}/{}]  recalls={}", h.id, h.kind, h.subject, h.recall_count);
    }
    Ok(())
}

fn cmd_show(root: &std::path::Path, id: &str, format: &str) -> Result<()> {
    let store = FileStore::open(root.to_path_buf())?;
    let (mem, path) = store.find_by_id(id)?;
    if format == "json" {
        let fm_yaml = serde_yaml::to_string(&mem.front)?;
        let fm_value: serde_yaml::Value = serde_yaml::from_str(&fm_yaml)?;
        let obj = serde_json::json!({
            "path": path.to_string_lossy(),
            "frontmatter": serde_json::to_value(&fm_value)?,
            "body": mem.body,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }
    println!("# {}", path.display());
    println!("{}", mem.to_markdown()?);
    Ok(())
}

fn cmd_similar(root: &std::path::Path, id: &str, limit: usize, format: &str) -> Result<()> {
    let idx = Index::open(&paths::index_db(root))?;
    let vec = idx
        .get_embedding(id)?
        .ok_or_else(|| anyhow!("memory {id} has no stored embedding"))?;
    let mut hits = idx.vector_search(&vec, limit + 1)?;
    hits.retain(|(h, _)| h.id != id);
    hits.truncate(limit);
    if format == "json" {
        let arr: Vec<serde_json::Value> = hits
            .iter()
            .map(|(h, sim)| {
                serde_json::json!({
                    "id": h.id,
                    "kind": h.kind,
                    "subject": h.subject,
                    "path": h.path.to_string_lossy(),
                    "vector_sim": sim,
                    "confidence": h.confidence,
                    "recall_count": h.recall_count,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    } else {
        if hits.is_empty() {
            println!("(no similar memories)");
        }
        for (h, sim) in hits {
            println!("{}  [{}/{}]  sim={:.3}", h.id, h.kind, h.subject, sim);
        }
    }
    Ok(())
}

fn cmd_delete(root: &std::path::Path, id: &str) -> Result<()> {
    let store = FileStore::open(root.to_path_buf())?;
    let idx = Index::open(&paths::index_db(root))?;
    let (_, path) = store.find_by_id(id)?;
    store.delete(&path)?;
    idx.remove(id)?;
    println!("deleted {id}");
    Ok(())
}

fn cmd_reindex(root: &std::path::Path, embedder_kind: EmbedderKind) -> Result<()> {
    let store = FileStore::open(root.to_path_buf())?;
    let idx = Index::open(&paths::index_db(root))?;
    let it = store.iter_all().filter_map(|r| match r {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("skip: {e:#}");
            None
        }
    });
    let embedder = embedder_kind.build()?;
    let id_for_msg = embedder.id().to_string();
    let n = idx.rebuild_from(it, Some(embedder.as_ref()))?;
    println!("indexed {n} memories (with {id_for_msg} embeddings)");
    Ok(())
}

fn cmd_touch(root: &std::path::Path, ids: Vec<String>, format: &str) -> Result<()> {
    if ids.is_empty() {
        return Err(anyhow!("touch requires at least one id"));
    }
    let idx = Index::open(&paths::index_db(root))?;
    let mut results: Vec<(String, Option<u32>, Option<String>)> = Vec::new();
    for id in ids {
        match idx.touch_recall(&id) {
            Ok(n) => results.push((id, Some(n), None)),
            Err(e) => results.push((id, None, Some(format!("{e:#}")))),
        }
    }
    if format == "json" {
        let arr: Vec<serde_json::Value> = results
            .iter()
            .map(|(id, n, err)| {
                serde_json::json!({
                    "id": id,
                    "recall_count": n,
                    "error": err,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    } else {
        for (id, n, err) in results {
            match (n, err) {
                (Some(c), _) => println!("{id}  recalls={c}"),
                (None, Some(e)) => println!("{id}  ERROR: {e}"),
                (None, None) => println!("{id}  (no change)"),
            }
        }
    }
    Ok(())
}

fn cmd_update(
    root: &std::path::Path,
    id: &str,
    body: Option<String>,
    file: Option<PathBuf>,
    use_stdin: bool,
    confidence: Option<f64>,
    add_evidence: Vec<String>,
    add_supersedes: Vec<String>,
    decays_after: Option<String>,
    embedder_kind: EmbedderKind,
) -> Result<()> {
    let exclusive = [body.is_some(), file.is_some(), use_stdin]
        .iter()
        .filter(|b| **b)
        .count();
    if exclusive > 1 {
        return Err(anyhow!("--body, --file, and --stdin are mutually exclusive"));
    }
    let store = FileStore::open(root.to_path_buf())?;
    let idx = Index::open(&paths::index_db(root))?;
    let (mut mem, path) = store.find_by_id(id)?;

    let body_changed = body.is_some() || file.is_some() || use_stdin;
    if body_changed {
        let new_body = if use_stdin {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf).context("read stdin")?;
            buf
        } else {
            read_body(body, file)?
        };
        if new_body.trim().is_empty() {
            return Err(anyhow!("body is empty"));
        }
        mem.body = new_body;
    }
    if let Some(c) = confidence {
        mem.front.confidence = c.clamp(0.0, 1.0);
    }
    for ev in add_evidence {
        mem.front.evidence.push(parse_evidence(&ev)?);
    }
    mem.front.supersedes.extend(add_supersedes);
    if let Some(d) = decays_after {
        if d.is_empty() {
            mem.front.decays_after = None;
        } else {
            mem.front.decays_after = Some(d);
        }
    }
    mem.front.updated_at = Some(Utc::now());

    store.overwrite(&path, &mem)?;
    if body_changed {
        let embedder = embedder_kind.build()?;
        let vec = embedder.embed(&mem.body)?;
        idx.upsert(&mem, &path, Some((embedder.id(), &vec)))?;
    } else {
        idx.upsert(&mem, &path, None)?;
    }
    println!("updated {id}");
    Ok(())
}

fn cmd_lineage(root: &std::path::Path, id: &str, format: &str) -> Result<()> {
    let idx = Index::open(&paths::index_db(root))?;
    let center = idx
        .get_meta(id)?
        .ok_or_else(|| anyhow!("memory {id} not found in index"))?;

    let mut older: Vec<MetaRow> = Vec::new();
    let mut stack: Vec<String> = center.supersedes.clone();
    while let Some(curr) = stack.pop() {
        if let Some(row) = idx.get_meta(&curr)? {
            stack.extend(row.supersedes.clone());
            older.push(row);
        }
    }

    let all = idx.all_meta()?;
    let mut newer: Vec<MetaRow> = Vec::new();
    let mut frontier: Vec<String> = vec![id.to_string()];
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(id.to_string());
    while let Some(curr) = frontier.pop() {
        for row in &all {
            if row.supersedes.iter().any(|s| s == &curr) && seen.insert(row.id.clone()) {
                frontier.push(row.id.clone());
                newer.push(row.clone());
            }
        }
    }

    if format == "json" {
        let obj = serde_json::json!({
            "id": center.id,
            "older": older.iter().map(meta_to_json).collect::<Vec<_>>(),
            "newer": newer.iter().map(meta_to_json).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("# lineage of {id}");
        if older.is_empty() {
            println!("(no older — this is the chain's root)");
        } else {
            println!("older (this id supersedes):");
            for r in &older {
                println!("  {}  [{}/{}]", r.id, r.kind, r.subject);
            }
        }
        if newer.is_empty() {
            println!("newer (none — this is the current tip)");
        } else {
            println!("newer (supersedes this id):");
            for r in &newer {
                println!("  {}  [{}/{}]", r.id, r.kind, r.subject);
            }
        }
    }
    Ok(())
}

fn cmd_doctor(
    root: &std::path::Path,
    fix: bool,
    format: &str,
    embedder_kind: EmbedderKind,
) -> Result<()> {
    if fix {
        cmd_reindex(root, embedder_kind)?;
    }
    let store = FileStore::open(root.to_path_buf())?;
    let idx = Index::open(&paths::index_db(root))?;

    let on_disk: HashSet<String> = store.iter_ids().map(|(id, _)| id).collect();
    let all_meta = idx.all_meta()?;
    let in_index: HashSet<String> = all_meta.iter().map(|m| m.id.clone()).collect();

    let orphans: Vec<String> = on_disk.difference(&in_index).cloned().collect();
    let missing: Vec<String> = in_index.difference(&on_disk).cloned().collect();

    let supersedes_targets = idx.superseded_ids()?;
    let dangling: Vec<String> = supersedes_targets
        .iter()
        .filter(|t| !in_index.contains(*t))
        .cloned()
        .collect();

    let mut embedder_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for m in &all_meta {
        let id = m.embedding_id.clone().unwrap_or_else(|| "(none)".into());
        *embedder_counts.entry(id).or_insert(0) += 1;
    }

    let oldest = all_meta.iter().map(|m| m.created_at).min();
    let newest = all_meta.iter().map(|m| m.created_at).max();
    let total_recall: u64 = all_meta.iter().map(|m| u64::from(m.recall_count)).sum();
    let decayed = idx.decayed_ids()?;
    let schema_version = recall::index::read_schema_version(&paths::index_db(root))?;
    let fs_warning = wal_fs_warning(&paths::index_db(root));
    let active_embedder_id = match embedder_kind {
        EmbedderKind::Hash => "hash-v1-d256".to_string(),
        EmbedderKind::Fastembed => "fastembed:bge-small-en-v1.5".to_string(),
    };
    let embedder_mismatch: Vec<String> = embedder_counts
        .keys()
        .filter(|k| *k != &active_embedder_id && *k != "(none)")
        .cloned()
        .collect();
    let model_path = fastembed_model_dir();

    let (daemon_active, daemon_uptime_s) = match daemon::default_socket_path() {
        Ok(sock) if sock.exists() => match ping_socket_sync(&sock) {
            Ok(body) => (true, body.get("uptime_s").and_then(|v| v.as_u64())),
            Err(_) => (false, None),
        },
        _ => (false, None),
    };

    let drift = idx.confidence_drift(0.3)?;
    let drift_json: Vec<serde_json::Value> = drift
        .iter()
        .map(|(id, d)| serde_json::json!({ "id": id, "drift": d }))
        .collect();

    if format == "json" {
        let obj = serde_json::json!({
            "files_on_disk": on_disk.len(),
            "index_count": in_index.len(),
            "orphans_on_disk": orphans,
            "missing_files": missing,
            "supersedes_dangling": dangling,
            "decayed_count": decayed.len(),
            "embedder_histogram": embedder_counts,
            "active_embedder_id": active_embedder_id,
            "embedder_id_mismatches": embedder_mismatch,
            "fastembed_cache_path": model_path,
            "schema_version": schema_version,
            "schema_version_expected": recall::index::SCHEMA_VERSION,
            "filesystem_warning": fs_warning,
            "oldest_created_at": oldest.map(|t| t.to_rfc3339()),
            "newest_created_at": newest.map(|t| t.to_rfc3339()),
            "total_recall_count": total_recall,
            "daemon_active": daemon_active,
            "daemon_uptime_s": daemon_uptime_s,
            "confidence_drift": drift_json,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("files on disk : {}", on_disk.len());
        println!("rows in index : {}", in_index.len());
        println!("orphans       : {}  (md on disk, not in index)", orphans.len());
        println!("missing       : {}  (in index, no md on disk)", missing.len());
        println!("dangling sup. : {}  (supersedes a vanished id)", dangling.len());
        println!("decayed       : {}", decayed.len());
        println!(
            "schema_version: {}  (expected {})",
            schema_version.map_or("(missing)".into(), |v| v.to_string()),
            recall::index::SCHEMA_VERSION
        );
        println!("active embedd : {active_embedder_id}");
        if let Some(p) = &model_path {
            println!("fastembed dir : {p}");
        }
        println!("embedders     :");
        for (k, v) in &embedder_counts {
            let flag = if k == &active_embedder_id || k == "(none)" {
                ""
            } else {
                "  (mismatch — reindex)"
            };
            println!("  {k}: {v}{flag}");
        }
        if let (Some(o), Some(n)) = (oldest, newest) {
            println!("oldest        : {o}");
            println!("newest        : {n}");
        }
        println!("total recalls : {total_recall}");
        println!("drift >=0.3   : {}  (confidence moved from creation)", drift.len());
        if daemon_active {
            match daemon_uptime_s {
                Some(s) => println!("daemon        : active (uptime {s}s)"),
                None => println!("daemon        : active"),
            }
        } else {
            println!("daemon        : inactive");
        }
        if let Some(w) = &fs_warning {
            println!("fs warning    : {w}");
        }
        if !orphans.is_empty() && !fix {
            println!("hint: run `recall doctor --fix` to reindex.");
        }
        if !embedder_mismatch.is_empty() {
            println!(
                "hint: {} memories have a different embedder id; run `recall reindex` to rebuild.",
                embedder_mismatch.iter().map(|_| ()).count()
            );
        }
    }
    Ok(())
}

/// `~/.cache/fastembed` (or `$XDG_CACHE_HOME/fastembed`), if it exists.
fn fastembed_model_dir() -> Option<String> {
    let base = if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        if xdg.is_empty() { None } else { Some(xdg) }
    } else {
        std::env::var("HOME").ok().map(|h| format!("{h}/.cache"))
    }?;
    let p = std::path::Path::new(&base).join("fastembed");
    if p.exists() {
        Some(p.to_string_lossy().to_string())
    } else {
        None
    }
}

/// Warn if the SQLite file lives on a filesystem where WAL is known to be
/// unsafe (NFS, SMB/CIFS, fuse-overlay). Best-effort: returns None if we
/// can't classify the filesystem.
fn wal_fs_warning(db_path: &std::path::Path) -> Option<String> {
    let canon = std::fs::canonicalize(db_path).ok()?;
    let mounts = std::fs::read_to_string("/proc/mounts").ok()?;
    let mut best: Option<(usize, String, String)> = None;
    for line in mounts.lines() {
        let mut parts = line.split_whitespace();
        let _dev = parts.next()?;
        let mount_point = parts.next()?;
        let fs_type = parts.next()?;
        if canon.starts_with(mount_point) && mount_point.len() > best.as_ref().map_or(0, |t| t.0) {
            best = Some((mount_point.len(), mount_point.into(), fs_type.into()));
        }
    }
    let (_, mount, fs_type) = best?;
    const UNSAFE_FS: &[&str] = &["nfs", "nfs4", "cifs", "smbfs", "fuse.sshfs"];
    if UNSAFE_FS.iter().any(|u| fs_type.starts_with(u)) {
        return Some(format!(
            "{} is on {fs_type} at {mount} — WAL is unsafe here. Consider PRAGMA journal_mode=DELETE.",
            db_path.display(),
        ));
    }
    None
}

fn cmd_gc(
    root: &std::path::Path,
    older_than: Option<&str>,
    never_recalled: bool,
    apply: bool,
    format: &str,
) -> Result<()> {
    let store = FileStore::open(root.to_path_buf())?;
    let idx = Index::open(&paths::index_db(root))?;
    let all = idx.all_meta()?;
    let cutoff: Option<DateTime<Utc>> = match older_than {
        Some(s) => Some(Utc::now() - parse_since_long(s)?),
        None => None,
    };
    let candidates: Vec<&MetaRow> = all
        .iter()
        .filter(|m| {
            if never_recalled && m.recall_count > 0 {
                return false;
            }
            if let Some(c) = cutoff {
                if m.created_at > c {
                    return false;
                }
            }
            true
        })
        .collect();

    let mut deleted = Vec::new();
    if apply {
        for m in &candidates {
            if let Ok((_, p)) = store.find_by_id(&m.id) {
                store.delete(&p)?;
            }
            idx.remove(&m.id)?;
            deleted.push(m.id.clone());
        }
    }

    if format == "json" {
        let obj = serde_json::json!({
            "applied": apply,
            "older_than": older_than,
            "never_recalled": never_recalled,
            "candidate_count": candidates.len(),
            "candidates": candidates.iter().map(|m| meta_to_json(m)).collect::<Vec<_>>(),
            "deleted": deleted,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else if candidates.is_empty() {
        println!("(no candidates)");
    } else {
        let mode = if apply { "DELETING" } else { "dry-run" };
        println!("{mode} ({} candidates):", candidates.len());
        for m in &candidates {
            println!(
                "  {}  [{}/{}]  recalls={}  created={}",
                m.id, m.kind, m.subject, m.recall_count, m.created_at
            );
        }
        if !apply {
            println!("(pass --apply to delete)");
        }
    }
    Ok(())
}

fn cmd_stats(root: &std::path::Path, format: &str) -> Result<()> {
    let idx = Index::open(&paths::index_db(root))?;
    let all = idx.all_meta()?;
    let mut by_subject: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut by_kind: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut by_embedder: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut total_recall: u64 = 0;
    for m in &all {
        *by_subject.entry(m.subject.clone()).or_insert(0) += 1;
        *by_kind.entry(m.kind.clone()).or_insert(0) += 1;
        let eid = m.embedding_id.clone().unwrap_or_else(|| "(none)".into());
        *by_embedder.entry(eid).or_insert(0) += 1;
        total_recall += u64::from(m.recall_count);
    }
    let avg_recall = if all.is_empty() {
        0.0
    } else {
        total_recall as f64 / all.len() as f64
    };
    let oldest = all.iter().map(|m| m.created_at).min();
    let db_size = std::fs::metadata(paths::index_db(root))
        .map(|md| md.len())
        .unwrap_or(0);

    if format == "json" {
        let obj = serde_json::json!({
            "total": all.len(),
            "by_subject": by_subject,
            "by_kind": by_kind,
            "by_embedder": by_embedder,
            "average_recall_count": avg_recall,
            "oldest_created_at": oldest.map(|t| t.to_rfc3339()),
            "index_size_bytes": db_size,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("total memories : {}", all.len());
        println!("average recalls: {avg_recall:.2}");
        if let Some(o) = oldest {
            println!("oldest         : {o}");
        }
        println!("index size     : {db_size} bytes");
        println!("by subject:");
        for (k, v) in &by_subject {
            println!("  {k}: {v}");
        }
        println!("by kind:");
        for (k, v) in &by_kind {
            println!("  {k}: {v}");
        }
        println!("by embedder:");
        for (k, v) in &by_embedder {
            println!("  {k}: {v}");
        }
    }
    Ok(())
}

fn cmd_export(root: &std::path::Path, format: &str) -> Result<()> {
    if format != "jsonl" {
        return Err(anyhow!("only --format jsonl is supported"));
    }
    let store = FileStore::open(root.to_path_buf())?;
    let stdout = io::stdout();
    let mut out = stdout.lock();
    use std::io::Write;
    for r in store.iter_all() {
        match r {
            Ok((mem, path)) => {
                let fm_yaml = serde_yaml::to_string(&mem.front)?;
                let fm_value: serde_yaml::Value = serde_yaml::from_str(&fm_yaml)?;
                let obj = serde_json::json!({
                    "path": path.to_string_lossy(),
                    "frontmatter": serde_json::to_value(&fm_value)?,
                    "body": mem.body,
                });
                if let Err(e) = writeln!(out, "{}", serde_json::to_string(&obj)?) {
                    if e.kind() == io::ErrorKind::BrokenPipe {
                        return Ok(());
                    }
                    return Err(e.into());
                }
            }
            Err(e) => eprintln!("skip: {e:#}"),
        }
    }
    Ok(())
}

fn cmd_import(
    root: &std::path::Path,
    file: Option<PathBuf>,
    overwrite: bool,
    embedder_kind: EmbedderKind,
) -> Result<()> {
    let store = FileStore::open(root.to_path_buf())?;
    let idx = Index::open(&paths::index_db(root))?;
    let reader: Box<dyn BufRead> = match file {
        Some(p) => Box::new(io::BufReader::new(
            std::fs::File::open(&p).with_context(|| format!("open {}", p.display()))?,
        )),
        None => Box::new(io::stdin().lock()),
    };
    let embedder = embedder_kind.build()?;
    let mut imported = 0_usize;
    let mut skipped = 0_usize;
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("parse jsonl line {}", lineno + 1))?;
        let fm = v
            .get("frontmatter")
            .ok_or_else(|| anyhow!("line {} missing `frontmatter`", lineno + 1))?;
        let body = v
            .get("body")
            .and_then(|b| b.as_str())
            .ok_or_else(|| anyhow!("line {} missing `body`", lineno + 1))?;
        let front: recall::memory::Frontmatter =
            serde_json::from_value(fm.clone()).context("decode frontmatter")?;
        let mem = Memory {
            front,
            body: body.to_string(),
        };
        if !overwrite && idx.get_meta(&mem.front.id)?.is_some() {
            skipped += 1;
            continue;
        }
        let path = store.write(&mem)?;
        let vec = embedder.embed(&mem.body)?;
        idx.upsert(&mem, &path, Some((embedder.id(), &vec)))?;
        imported += 1;
    }
    println!("imported {imported}, skipped {skipped}");
    Ok(())
}

fn meta_to_json(m: &MetaRow) -> serde_json::Value {
    serde_json::json!({
        "id": m.id,
        "kind": m.kind,
        "subject": m.subject,
        "path": m.path.to_string_lossy(),
        "confidence": m.confidence,
        "created_at": m.created_at.to_rfc3339(),
        "updated_at": m.updated_at.map(|t| t.to_rfc3339()),
        "last_recalled_at": m.last_recalled_at.map(|t| t.to_rfc3339()),
        "recall_count": m.recall_count,
        "decays_after": m.decays_after,
        "supersedes": m.supersedes,
        "embedding_id": m.embedding_id,
    })
}

fn cmd_scratch(root: &std::path::Path, op: ScratchOp) -> Result<()> {
    match op {
        ScratchOp::Write {
            kind,
            subject,
            body,
            file,
            session,
        } => {
            let sid = scratch::resolve_session_id(session.as_deref())?;
            let body_text = read_body(body, file)?;
            if body_text.trim().is_empty() {
                return Err(anyhow!("body is empty"));
            }
            let kind: Kind = kind.parse()?;
            let mem = Memory::new(kind, Subject(subject), body_text);
            scratch::write(root, &sid, &mem)?;
            println!("{}", mem.front.id);
            Ok(())
        }
        ScratchOp::List { session, format } => {
            let entries = scratch::list(root, session.as_deref())?;
            if format == "json" {
                let arr: Vec<serde_json::Value> = entries
                    .iter()
                    .map(|(m, p)| {
                        serde_json::json!({
                            "id": m.front.id,
                            "kind": m.front.kind.as_str(),
                            "subject": m.front.subject.as_str(),
                            "path": p.to_string_lossy(),
                            "created_at": m.front.created_at.to_rfc3339(),
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&arr)?);
            } else {
                if entries.is_empty() {
                    println!("(no scratch)");
                }
                for (m, p) in entries {
                    println!(
                        "{}  [{}/{}]  {}",
                        m.front.id,
                        m.front.kind.as_str(),
                        m.front.subject.as_str(),
                        p.display()
                    );
                }
            }
            Ok(())
        }
        ScratchOp::Show {
            id,
            session,
            format,
        } => {
            let sid = scratch::resolve_session_id(session.as_deref())?;
            let (mem, path) = scratch::show(root, &sid, &id)?;
            if format == "json" {
                let fm_yaml = serde_yaml::to_string(&mem.front)?;
                let fm_value: serde_yaml::Value = serde_yaml::from_str(&fm_yaml)?;
                let obj = serde_json::json!({
                    "path": path.to_string_lossy(),
                    "frontmatter": serde_json::to_value(&fm_value)?,
                    "body": mem.body,
                });
                println!("{}", serde_json::to_string_pretty(&obj)?);
            } else {
                println!("# {}", path.display());
                println!("{}", mem.to_markdown()?);
            }
            Ok(())
        }
        ScratchOp::Clear { session } => {
            let sid = scratch::resolve_session_id(session.as_deref())?;
            let n = scratch::clear(root, &sid)?;
            println!("cleared {n} scratch from session {sid}");
            Ok(())
        }
    }
}

fn cmd_promote(
    root: &std::path::Path,
    session: Option<&str>,
    only_id: Option<&str>,
    format: &str,
    embedder_kind: EmbedderKind,
) -> Result<()> {
    let sid = scratch::resolve_session_id(session)?;
    let entries = scratch::list(root, Some(&sid))?;
    let filtered: Vec<_> = entries
        .into_iter()
        .filter(|(m, _)| only_id.map_or(true, |x| m.front.id == x))
        .collect();
    if filtered.is_empty() {
        if format == "json" {
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({"promoted": []}))?);
        } else {
            println!("(nothing to promote in session {sid})");
        }
        return Ok(());
    }
    let store = FileStore::open(root.to_path_buf())?;
    let idx = Index::open(&paths::index_db(root))?;
    let embedder = embedder_kind.build()?;
    let mut promoted: Vec<String> = Vec::new();
    for (mem, path) in filtered {
        let new_path = store.write(&mem)?;
        let vec = embedder.embed(&mem.body)?;
        idx.upsert(&mem, &new_path, Some((embedder.id(), &vec)))?;
        scratch::remove(&path)?;
        promoted.push(mem.front.id);
    }
    if format == "json" {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "session": sid,
                "promoted": promoted,
            }))?
        );
    } else {
        for id in &promoted {
            println!("promoted {id}");
        }
        println!("({} promoted from session {sid})", promoted.len());
    }
    Ok(())
}

fn cmd_observe(root: &std::path::Path, file: Option<PathBuf>, format: &str) -> Result<()> {
    let reader: Box<dyn BufRead> = match file {
        Some(p) => Box::new(io::BufReader::new(
            std::fs::File::open(&p).with_context(|| format!("open {}", p.display()))?,
        )),
        None => Box::new(io::stdin().lock()),
    };
    let written = observer::run(root, reader)?;
    if format == "json" {
        let arr: Vec<String> = written
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"proposed": arr}))?
        );
    } else if written.is_empty() {
        println!("(no proposals)");
    } else {
        for p in written {
            println!("proposed {}", p.display());
        }
    }
    Ok(())
}

fn cmd_proposals(
    root: &std::path::Path,
    apply: Option<&str>,
    discard: Option<&str>,
    format: &str,
    embedder_kind: EmbedderKind,
) -> Result<()> {
    if apply.is_some() && discard.is_some() {
        return Err(anyhow!("--apply and --discard are mutually exclusive"));
    }
    if let Some(id) = apply {
        let entries = observer::list_proposals(root)?;
        let (mem, path) = entries
            .into_iter()
            .find(|(m, _)| m.front.id == id)
            .ok_or_else(|| anyhow!("proposal {id} not found"))?;
        let store = FileStore::open(root.to_path_buf())?;
        let idx = Index::open(&paths::index_db(root))?;
        let new_path = store.write(&mem)?;
        let embedder = embedder_kind.build()?;
        let vec = embedder.embed(&mem.body)?;
        idx.upsert(&mem, &new_path, Some((embedder.id(), &vec)))?;
        std::fs::remove_file(&path)?;
        println!("applied {id}");
        return Ok(());
    }
    if let Some(id) = discard {
        let entries = observer::list_proposals(root)?;
        let (_, path) = entries
            .into_iter()
            .find(|(m, _)| m.front.id == id)
            .ok_or_else(|| anyhow!("proposal {id} not found"))?;
        std::fs::remove_file(&path)?;
        println!("discarded {id}");
        return Ok(());
    }
    let entries = observer::list_proposals(root)?;
    if format == "json" {
        let arr: Vec<serde_json::Value> = entries
            .iter()
            .map(|(m, p)| {
                serde_json::json!({
                    "id": m.front.id,
                    "kind": m.front.kind.as_str(),
                    "subject": m.front.subject.as_str(),
                    "path": p.to_string_lossy(),
                    "created_at": m.front.created_at.to_rfc3339(),
                    "body_preview": m.body.chars().take(160).collect::<String>(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    } else {
        if entries.is_empty() {
            println!("(no proposals)");
        }
        for (m, p) in entries {
            println!("{}  [{}/{}]  {}", m.front.id, m.front.kind.as_str(), m.front.subject.as_str(), p.display());
            let preview: String = m.body.chars().take(120).collect();
            println!("  {preview}");
        }
        if entries_present(root) {
            println!("(apply with `recall proposals --apply <id>`; discard with `--discard <id>`)");
        }
    }
    Ok(())
}

fn entries_present(root: &std::path::Path) -> bool {
    observer::list_proposals(root).map(|v| !v.is_empty()).unwrap_or(false)
}

fn cmd_session_diff(
    root: &std::path::Path,
    session: Option<&str>,
    since: Option<&str>,
    format: &str,
) -> Result<()> {
    let cutoff = Utc::now() - parse_since(since.unwrap_or("8h"))?;
    let idx = Index::open(&paths::index_db(root))?;
    let all = idx.all_meta()?;

    let mut new_memories: Vec<&MetaRow> = Vec::new();
    let mut updated: Vec<&MetaRow> = Vec::new();
    let mut touched: Vec<&MetaRow> = Vec::new();
    for m in &all {
        if m.created_at >= cutoff {
            new_memories.push(m);
        } else if let Some(u) = m.updated_at {
            if u >= cutoff {
                updated.push(m);
            }
        }
        if let Some(t) = m.last_recalled_at {
            if t >= cutoff && m.created_at < cutoff {
                touched.push(m);
            }
        }
    }

    let scratch_entries = match session {
        Some(s) => scratch::list(root, Some(s))?,
        None => Vec::new(),
    };

    if format == "json" {
        let obj = serde_json::json!({
            "session": session,
            "since": since.unwrap_or("8h"),
            "new":     new_memories.iter().map(|m| meta_to_json(m)).collect::<Vec<_>>(),
            "updated": updated.iter().map(|m| meta_to_json(m)).collect::<Vec<_>>(),
            "touched": touched.iter().map(|m| meta_to_json(m)).collect::<Vec<_>>(),
            "scratch_pending": scratch_entries.iter().map(|(m, p)| serde_json::json!({
                "id": m.front.id,
                "subject": m.front.subject.as_str(),
                "path": p.to_string_lossy(),
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("# session diff — since {}", since.unwrap_or("8h"));
        println!("new     ({}):", new_memories.len());
        for m in &new_memories {
            println!("  + {}  [{}/{}]", m.id, m.kind, m.subject);
        }
        println!("updated ({}):", updated.len());
        for m in &updated {
            println!("  ~ {}  [{}/{}]", m.id, m.kind, m.subject);
        }
        println!("touched ({}):", touched.len());
        for m in &touched {
            println!("  · {}  recalls={}", m.id, m.recall_count);
        }
        if !scratch_entries.is_empty() {
            println!("scratch pending ({}):", scratch_entries.len());
            for (m, _) in &scratch_entries {
                println!("  ? {}  [{}/{}]", m.front.id, m.front.kind.as_str(), m.front.subject.as_str());
            }
        }
    }
    Ok(())
}

/// Extended duration parser used by `gc --older-than`: accepts the `m/h/d`
/// suffixes of `parse_since` plus `w`, `mo`, `y`.
fn parse_since_long(s: &str) -> Result<Duration> {
    let t = s.trim();
    if t.len() < 2 {
        return Err(anyhow!("bad duration: {s}"));
    }
    if let Some(stripped) = t.strip_suffix("mo") {
        let n: i64 = stripped
            .parse()
            .map_err(|_| anyhow!("bad duration number in: {s}"))?;
        return Ok(Duration::days(n.saturating_mul(30)));
    }
    let (num_part, suffix) = t.split_at(t.len() - 1);
    let n: i64 = num_part
        .parse()
        .map_err(|_| anyhow!("bad duration number in: {s}"))?;
    match suffix {
        "m" => Ok(Duration::minutes(n)),
        "h" => Ok(Duration::hours(n)),
        "d" => Ok(Duration::days(n)),
        "w" => Ok(Duration::weeks(n)),
        "y" => Ok(Duration::days(n.saturating_mul(365))),
        _ => Err(anyhow!("unknown duration suffix in: {s} (use m|h|d|w|mo|y)")),
    }
}
