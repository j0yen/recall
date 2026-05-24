//! recall — local-first agentic memory CLI.

#![allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Duration, Utc};
use clap::{Parser, Subcommand};
use recall::embeddings::EmbedderKind;
use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::retrieval;
use recall::store::FileStore;
use std::io::{self, Read};
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

    /// Delete a memory by id.
    Delete { id: String },

    /// Wipe the `SQLite` index and rebuild it from the files on disk
    /// using the active embedder.
    Reindex,

    /// Print where recall is reading from, including the active embedder id.
    Where,
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
        } => cmd_write(
            &root,
            &kind,
            &subject,
            body,
            file,
            confidence,
            supersedes,
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
        } => cmd_list(
            &root,
            subject.as_deref(),
            kind.as_deref(),
            since.as_deref(),
            &format,
            limit,
        ),
        Command::Show { id, format } => cmd_show(&root, &id, &format),
        Command::Delete { id } => cmd_delete(&root, &id),
        Command::Reindex => cmd_reindex(&root, embedder_kind),
        Command::Similar { id, limit, format } => cmd_similar(&root, &id, limit, &format),
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
    let path = store.write(&mem)?;
    let embedder = embedder_kind.build()?;
    let vec = embedder.embed(&mem.body)?;
    idx.upsert(&mem, &path, Some((embedder.id(), &vec)))?;
    println!("{}", mem.front.id);
    Ok(())
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
    _include_superseded: bool,
    _include_decayed: bool,
    embedder_kind: EmbedderKind,
) -> Result<()> {
    let idx = Index::open(&paths::index_db(root))?;
    let inner_limit =
        if subject.is_some() || kind.is_some() || since.is_some() || min_confidence.is_some() {
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
    hits.retain(|r| {
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
            idx.touch_recall(&r.hit.id)?;
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
) -> Result<()> {
    let idx = Index::open(&paths::index_db(root))?;
    let inner_limit = if kind.is_some() || since.is_some() {
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
    hits.retain(|h| {
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
