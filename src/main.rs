//! recall — local-first agentic memory CLI.

#![allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Duration, Utc};
use clap::{Parser, Subcommand};
use recall::embeddings::EmbedderKind;
use recall::index::{Index, MetaRow};
use recall::memory::{Evidence, Kind, Memory, Subject};
use recall::paths;
use recall::retrieval;
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = match cli.root {
        Some(r) => r,
        None => paths::root()?,
    };
    let embedder_kind = EmbedderKind::parse(&cli.embedder)?;

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
        } => cmd_query(
            &root,
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
        } => cmd_list(
            &root,
            subject.as_deref(),
            kind.as_deref(),
            since.as_deref(),
            &format,
            limit,
            include_superseded,
            include_decayed,
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
        Command::Where => {
            println!("{}", root.display());
            println!("embedder: {}", cli.embedder);
            Ok(())
        }
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
    embedder_kind: EmbedderKind,
) -> Result<()> {
    let idx = Index::open(&paths::index_db(root))?;
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
        retrieval::hybrid_search(&idx, embedder.as_ref(), query, inner_limit)?
    } else {
        retrieval::search(&idx, query, inner_limit)?
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

fn cmd_list(
    root: &std::path::Path,
    subject: Option<&str>,
    kind: Option<&str>,
    since: Option<&str>,
    format: &str,
    limit: usize,
    include_superseded: bool,
    include_decayed: bool,
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

    if format == "json" {
        let obj = serde_json::json!({
            "files_on_disk": on_disk.len(),
            "index_count": in_index.len(),
            "orphans_on_disk": orphans,
            "missing_files": missing,
            "supersedes_dangling": dangling,
            "decayed_count": decayed.len(),
            "embedder_histogram": embedder_counts,
            "oldest_created_at": oldest.map(|t| t.to_rfc3339()),
            "newest_created_at": newest.map(|t| t.to_rfc3339()),
            "total_recall_count": total_recall,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("files on disk : {}", on_disk.len());
        println!("rows in index : {}", in_index.len());
        println!("orphans       : {}  (md on disk, not in index)", orphans.len());
        println!("missing       : {}  (in index, no md on disk)", missing.len());
        println!("dangling sup. : {}  (supersedes a vanished id)", dangling.len());
        println!("decayed       : {}", decayed.len());
        println!("embedders     :");
        for (k, v) in &embedder_counts {
            println!("  {k}: {v}");
        }
        if let (Some(o), Some(n)) = (oldest, newest) {
            println!("oldest        : {o}");
            println!("newest        : {n}");
        }
        println!("total recalls : {total_recall}");
        if !orphans.is_empty() && !fix {
            println!("hint: run `recall doctor --fix` to reindex.");
        }
    }
    Ok(())
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
