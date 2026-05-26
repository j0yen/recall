//! Daemon-mode for `recalld` (PRD: recall-daemon, codename *current*).
//!
//! iter-3 wires the four wire ops (`ping`, `query`, `embed`, `touch`) end
//! to end. The daemon now owns an open `Index` plus a built `Embedder`,
//! and dispatches read-only requests against them. Writes
//! (`write`/`update`/`delete`/`reindex`) stay CLI-only per PRD §2.1. The
//! CLI auto-forward path and the systemd-user unit land separately in
//! iter-4 toward v0.5.0.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex as AsyncMutex;

use crate::embeddings::{Embedder, EmbedderKind};
use crate::index::Index;
use crate::paths;
use crate::retrieval::{self, Weights};

/// Wire-protocol op names. Stable across v0.5.x.
pub const OPS: &[&str] = &["query", "embed", "touch", "ping"];

/// Fallback identifier used only when `DaemonState` is built without
/// resources (test helpers that never invoke `Ping`). Production paths
/// report `state.embedder_id`, which is the embedder's actual id.
pub const DEFAULT_MODEL_ID: &str = "bge-small-en-v1.5";

/// Upper bound on hits returned by a single `query` op. Mirrors the
/// CLI's implicit ceiling so a malicious caller cannot ask for an
/// arbitrarily large page that fans out into vector scans.
pub const MAX_QUERY_LIMIT: usize = 200;

/// Maximum bytes accepted for a single framed message. Generous enough
/// for any plausible query payload; cheap insurance against a malformed
/// length prefix exhausting memory.
pub const MAX_FRAME_BYTES: u32 = 4 * 1024 * 1024;

/// Default UDS path. Prefers `$XDG_RUNTIME_DIR/recall.sock`; falls back
/// to `~/.cache/recall/recall.sock` if no runtime dir is set.
///
/// # Errors
/// Returns an error if neither `$XDG_RUNTIME_DIR` nor the user's home
/// directory can be resolved.
pub fn default_socket_path() -> Result<PathBuf> {
    if let Ok(rt) = std::env::var("XDG_RUNTIME_DIR") {
        if !rt.is_empty() {
            return Ok(PathBuf::from(rt).join("recall.sock"));
        }
    }
    let home = directories::BaseDirs::new()
        .context("could not resolve user home directory")?;
    Ok(home
        .home_dir()
        .join(".cache")
        .join("recall")
        .join("recall.sock"))
}

/// Default PID file path. Same directory as [`default_socket_path`], with
/// a `recall.pid` filename so `recall daemon start/stop` can find the
/// running `recalld` to signal.
///
/// # Errors
/// Returns an error if the default socket directory cannot be resolved.
pub fn default_pid_path() -> Result<PathBuf> {
    let sock = default_socket_path()?;
    let dir = sock
        .parent()
        .context("socket path has no parent dir")?
        .to_path_buf();
    Ok(dir.join("recall.pid"))
}

/// PID file location implied by a given socket path: same directory,
/// `recall.pid` filename. Used so `--socket /tmp/x.sock` writes its PID
/// to `/tmp/recall.pid` rather than colliding with the default daemon.
///
/// # Errors
/// Returns an error if the socket path has no parent directory.
pub fn pid_path_for_socket(socket: &std::path::Path) -> Result<PathBuf> {
    let dir = socket
        .parent()
        .context("socket path has no parent dir")?
        .to_path_buf();
    Ok(dir.join("recall.pid"))
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", content = "args", rename_all = "snake_case")]
pub enum Request {
    Query(QueryArgs),
    Embed(EmbedArgs),
    Touch(TouchArgs),
    Ping(PingArgs),
}

#[derive(Debug, Deserialize, Default)]
pub struct QueryArgs {
    pub text: String,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub hybrid: Option<bool>,
    #[serde(default)]
    pub filters: serde_json::Value,
    #[serde(default)]
    pub project_subject: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EmbedArgs {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct TouchArgs {
    pub id: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct PingArgs {}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Response {
    Ok { ok: serde_json::Value },
    Err { error: ErrorBody },
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub code: &'static str,
    pub message: String,
}

impl Response {
    pub fn ok<T: Serialize>(body: &T) -> Self {
        Self::Ok {
            ok: serde_json::to_value(body).unwrap_or(serde_json::Value::Null),
        }
    }

    pub fn err(code: &'static str, message: impl Into<String>) -> Self {
        Self::Err {
            error: ErrorBody {
                code,
                message: message.into(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct PingResponse {
    pub model_id: String,
    pub uptime_s: u64,
    pub query_count: u64,
    pub version: &'static str,
    pub root: String,
}

/// Per-hit wire shape. Mirrors `retrieval::RankedHit` flattened so
/// callers don't have to know about `Hit`/`RankedHit` internals.
#[derive(Debug, Serialize)]
pub struct QueryHit {
    pub id: String,
    pub kind: String,
    pub subject: String,
    pub path: String,
    pub snippet: String,
    pub score: f64,
    pub bm25: f64,
    pub vector_sim: f64,
    pub confidence: f64,
    pub recall_count: u32,
    pub last_recalled_at: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub ranked_hits: Vec<QueryHit>,
    pub query_count: u64,
    pub limit: usize,
    pub hybrid: bool,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub vector: Vec<f32>,
    pub dim: usize,
    pub embedder_id: String,
}

#[derive(Debug, Serialize)]
pub struct TouchResponse {
    pub recall_count: u32,
}

/// Per-process daemon state. Owns the open `Index` (behind a sync
/// `Mutex` because `rusqlite::Connection` is `Send + !Sync`) and the
/// built `Embedder`. Handlers run sync — `serve_connection` wraps the
/// dispatch in `spawn_blocking` so a slow query never stalls the
/// async accept loop.
pub struct DaemonState {
    pub started: Instant,
    pub query_count: std::sync::atomic::AtomicU64,
    pub root: PathBuf,
    pub embedder_id: String,
    pub embedder_dim: usize,
    index: std::sync::Mutex<Index>,
    embedder: Box<dyn Embedder>,
}

impl std::fmt::Debug for DaemonState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonState")
            .field("started", &self.started)
            .field("query_count", &self.query_count)
            .field("root", &self.root)
            .field("embedder_id", &self.embedder_id)
            .field("embedder_dim", &self.embedder_dim)
            .field("index", &"<Mutex<Index>>")
            .field("embedder", &"<Box<dyn Embedder>>")
            .finish()
    }
}

impl DaemonState {
    /// Open the recall index at `paths::index_db(root)` and build the
    /// embedder. This is the production constructor; tests that need a
    /// no-op state use [`open_in_dir`] against a `tempdir`.
    ///
    /// # Errors
    /// Returns an error if the index cannot be opened or the embedder
    /// fails to load (e.g. fastembed model download timeout).
    pub fn open(root: PathBuf, embedder_kind: EmbedderKind) -> Result<Self> {
        let idx_path = paths::index_db(&root);
        let index = Index::open(&idx_path)
            .with_context(|| format!("open index at {}", idx_path.display()))?;
        let embedder = embedder_kind.build()?;
        let embedder_id = embedder.id().to_string();
        let embedder_dim = embedder.dim();
        Ok(Self {
            started: Instant::now(),
            query_count: std::sync::atomic::AtomicU64::new(0),
            root,
            embedder_id,
            embedder_dim,
            index: std::sync::Mutex::new(index),
            embedder,
        })
    }

    pub fn uptime_s(&self) -> u64 {
        self.started.elapsed().as_secs()
    }
}

async fn read_frame(stream: &mut UnixStream) -> Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        anyhow::bail!(
            "frame exceeds MAX_FRAME_BYTES ({} > {})",
            len,
            MAX_FRAME_BYTES
        );
    }
    let mut body = vec![0u8; len as usize];
    stream.read_exact(&mut body).await?;
    Ok(Some(body))
}

async fn write_frame(stream: &mut UnixStream, body: &[u8]) -> Result<()> {
    let len = u32::try_from(body.len()).context("response body exceeds u32::MAX")?;
    if len > MAX_FRAME_BYTES {
        anyhow::bail!(
            "response exceeds MAX_FRAME_BYTES ({} > {})",
            len,
            MAX_FRAME_BYTES
        );
    }
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    Ok(())
}

/// Dispatch one request to its response. Pulled out of the connection
/// loop so unit tests can drive it directly. Synchronous; callers in
/// async contexts should run this on `spawn_blocking` because index
/// queries acquire a sync `Mutex` and call into rusqlite.
pub fn handle_request(state: &DaemonState, req: Request) -> Response {
    match req {
        Request::Ping(_) => handle_ping(state),
        Request::Query(args) => handle_query(state, args),
        Request::Embed(args) => handle_embed(state, args),
        Request::Touch(args) => handle_touch(state, args),
    }
}

fn handle_ping(state: &DaemonState) -> Response {
    let body = PingResponse {
        model_id: state.embedder_id.clone(),
        uptime_s: state.uptime_s(),
        query_count: state.query_count.load(Ordering::Relaxed),
        version: env!("CARGO_PKG_VERSION"),
        root: state.root.display().to_string(),
    };
    Response::ok(&body)
}

fn handle_query(state: &DaemonState, args: QueryArgs) -> Response {
    if args.text.trim().is_empty() {
        return Response::err("bad_request", "query.text must not be empty");
    }
    let limit = args.limit.unwrap_or(10).clamp(1, MAX_QUERY_LIMIT);
    let hybrid = args.hybrid.unwrap_or(true);
    let project_subject = args.project_subject.as_deref();

    let idx = match state.index.lock() {
        Ok(g) => g,
        Err(_) => return Response::err("internal_error", "index mutex poisoned"),
    };

    let weights = Weights::default();
    let ranked = if hybrid {
        retrieval::hybrid_with(
            &idx,
            state.embedder.as_ref(),
            &args.text,
            limit,
            weights,
            project_subject,
        )
    } else {
        retrieval::search_with(&idx, &args.text, limit, weights, project_subject)
    };
    drop(idx);

    let count = state.query_count.fetch_add(1, Ordering::Relaxed) + 1;
    match ranked {
        Ok(hits) => {
            let ranked_hits = hits
                .into_iter()
                .map(|r| QueryHit {
                    id: r.hit.id,
                    kind: r.hit.kind,
                    subject: r.hit.subject,
                    path: r.hit.path.display().to_string(),
                    snippet: r.hit.snippet,
                    score: r.score,
                    bm25: r.hit.bm25,
                    vector_sim: r.vector_sim,
                    confidence: r.hit.confidence,
                    recall_count: r.hit.recall_count,
                    last_recalled_at: r.hit.last_recalled_at.map(|t| t.to_rfc3339()),
                })
                .collect();
            Response::ok(&QueryResponse {
                ranked_hits,
                query_count: count,
                limit,
                hybrid,
            })
        }
        Err(e) => Response::err("query_failed", format!("{e:#}")),
    }
}

fn handle_embed(state: &DaemonState, args: EmbedArgs) -> Response {
    if args.text.is_empty() {
        return Response::err("bad_request", "embed.text must not be empty");
    }
    match state.embedder.embed(&args.text) {
        Ok(vector) => {
            let dim = vector.len();
            Response::ok(&EmbedResponse {
                vector,
                dim,
                embedder_id: state.embedder_id.clone(),
            })
        }
        Err(e) => Response::err("embed_failed", format!("{e:#}")),
    }
}

fn handle_touch(state: &DaemonState, args: TouchArgs) -> Response {
    if args.id.trim().is_empty() {
        return Response::err("bad_request", "touch.id must not be empty");
    }
    let idx = match state.index.lock() {
        Ok(g) => g,
        Err(_) => return Response::err("internal_error", "index mutex poisoned"),
    };
    match idx.touch_recall(&args.id) {
        Ok(n) => Response::ok(&TouchResponse { recall_count: n }),
        Err(e) => Response::err("touch_failed", format!("{e:#}")),
    }
}

async fn serve_connection(state: Arc<DaemonState>, mut stream: UnixStream) -> Result<()> {
    loop {
        let frame = match read_frame(&mut stream).await? {
            Some(f) => f,
            None => return Ok(()),
        };
        let resp = match serde_json::from_slice::<Request>(&frame) {
            Ok(req) => {
                let st = state.clone();
                // Run the sync handler on the blocking pool so a slow
                // SQLite query or fastembed call doesn't stall the
                // accept loop's worker thread.
                tokio::task::spawn_blocking(move || handle_request(&st, req))
                    .await
                    .unwrap_or_else(|e| {
                        Response::err("internal_error", format!("handler panicked: {e}"))
                    })
            }
            Err(e) => Response::err("bad_request", format!("invalid JSON request: {e}")),
        };
        let body = serde_json::to_vec(&resp).context("serialize response")?;
        write_frame(&mut stream, &body).await?;
    }
}

fn ensure_parent_dir(socket_path: &std::path::Path) -> Result<()> {
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create socket parent dir {}", parent.display()))?;
    }
    Ok(())
}

/// If a socket already exists at `socket_path`, attempt to connect; if
/// the connection refuses (no live listener), unlink the stale file.
/// Returns an error if a live daemon is detected.
async fn handle_existing_socket(socket_path: &std::path::Path) -> Result<()> {
    if !socket_path.exists() {
        return Ok(());
    }
    match UnixStream::connect(socket_path).await {
        Ok(_) => anyhow::bail!(
            "live recalld already listening on {} (recall daemon stop first)",
            socket_path.display()
        ),
        Err(_) => {
            std::fs::remove_file(socket_path).with_context(|| {
                format!("remove stale socket {}", socket_path.display())
            })?;
            Ok(())
        }
    }
}

/// Bind the UDS and serve until `shutdown` resolves. Caller owns the
/// shutdown signal (e.g. SIGTERM/Ctrl-C in `recalld`'s main) and owns
/// the `DaemonState` (so iter-4's `recall daemon status` over the
/// socket can share the same state struct).
///
/// # Errors
/// Returns an error if the socket path cannot be prepared, a live
/// daemon is already bound there, or the listener cannot be created.
pub async fn run_server<F>(
    state: Arc<DaemonState>,
    socket_path: PathBuf,
    shutdown: F,
) -> Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    ensure_parent_dir(&socket_path)?;
    handle_existing_socket(&socket_path).await?;

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("bind UDS at {}", socket_path.display()))?;
    let conns: Arc<AsyncMutex<Vec<tokio::task::JoinHandle<()>>>> =
        Arc::new(AsyncMutex::new(Vec::new()));

    let accept_state = state.clone();
    let accept_conns = conns.clone();
    let accept_loop = async move {
        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let st = accept_state.clone();
                    let h = tokio::spawn(async move {
                        if let Err(e) = serve_connection(st, stream).await {
                            eprintln!("recalld: connection error: {e}");
                        }
                    });
                    accept_conns.lock().await.push(h);
                }
                Err(e) => {
                    eprintln!("recalld: accept error: {e}");
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
    };

    tokio::select! {
        () = accept_loop => {}
        () = shutdown => {}
    }

    // The socket file is the daemon's liveness signal; unlink on
    // graceful shutdown so the next start sees a clean slate.
    let _ = std::fs::remove_file(&socket_path);
    Ok(())
}

/// Test/dev helper: spawn `run_server` on `socket_path` against the
/// provided `state` and return a shutdown sender plus the task handle.
/// Public so integration tests in `tests/` can use it without
/// re-implementing framing.
#[doc(hidden)]
pub async fn spawn_for_test(
    state: Arc<DaemonState>,
    socket_path: PathBuf,
) -> Result<(
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<Result<()>>,
)> {
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server_path = socket_path.clone();
    let handle = tokio::spawn(async move {
        run_server(state, server_path, async move {
            let _ = rx.await;
        })
        .await
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while std::time::Instant::now() < deadline {
        if socket_path.exists() {
            return Ok((tx, handle));
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    anyhow::bail!("test daemon failed to bind socket within 1s")
}

/// Client-side helper: connect, send one request frame, read one
/// response frame. Public so the CLI auto-forward path (iter-3) and
/// integration tests share the same framing.
///
/// # Errors
/// Returns an error if the socket cannot be reached, the request cannot
/// be serialized, or the response is malformed.
pub async fn client_roundtrip(
    socket_path: &std::path::Path,
    req: &serde_json::Value,
) -> Result<serde_json::Value> {
    let mut stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("connect to {}", socket_path.display()))?;
    let body = serde_json::to_vec(req).context("serialize request")?;
    write_frame(&mut stream, &body).await?;
    let resp_bytes = read_frame(&mut stream)
        .await?
        .context("daemon closed connection before responding")?;
    let resp: serde_json::Value =
        serde_json::from_slice(&resp_bytes).context("parse response JSON")?;
    Ok(resp)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memory::{Kind, Memory, Subject};
    use crate::store::FileStore;
    use tempfile::TempDir;

    fn test_state() -> (DaemonState, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let state = DaemonState::open(dir.path().to_path_buf(), EmbedderKind::Hash).unwrap();
        (state, dir)
    }

    fn write_memory(root: &std::path::Path, subject: &str, body: &str) -> String {
        let store = FileStore::open(root.to_path_buf()).unwrap();
        let idx = Index::open(&paths::index_db(root)).unwrap();
        let mem = Memory::new(Kind::Semantic, Subject(subject.to_string()), body.to_string());
        let path = store.write(&mem).unwrap();
        let embedder = EmbedderKind::Hash.build().unwrap();
        let v = embedder.embed(&mem.body).unwrap();
        idx.upsert(&mem, &path, Some((embedder.id(), &v))).unwrap();
        mem.front.id
    }

    #[test]
    fn ping_request_handler_reports_embedder_id_and_uptime() {
        let (state, _dir) = test_state();
        let resp = handle_request(&state, Request::Ping(PingArgs {}));
        let v = match resp {
            Response::Ok { ok } => ok,
            Response::Err { error } => panic!("expected ok, got {error:?}"),
        };
        // HashEmbedder reports an id like "hash-v1-d256".
        assert!(
            v["model_id"].as_str().unwrap().starts_with("hash"),
            "model_id was {}",
            v["model_id"]
        );
        assert!(v["uptime_s"].as_u64().is_some());
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
        assert!(v["root"].as_str().is_some());
    }

    #[test]
    fn query_rejects_empty_text() {
        let (state, _dir) = test_state();
        let resp = handle_request(
            &state,
            Request::Query(QueryArgs {
                text: "   ".into(),
                ..Default::default()
            }),
        );
        match resp {
            Response::Err { error } => assert_eq!(error.code, "bad_request"),
            Response::Ok { .. } => panic!("expected bad_request"),
        }
    }

    #[test]
    fn query_returns_ranked_hits_against_seeded_index() {
        let (state, _dir) = test_state();
        write_memory(
            &state.root,
            "self",
            "the rust toolchain at ~/.cargo/bin is on PATH in all zsh shells",
        );
        write_memory(&state.root, "user", "unrelated note about gardening");
        let resp = handle_request(
            &state,
            Request::Query(QueryArgs {
                text: "rust toolchain PATH".into(),
                limit: Some(5),
                hybrid: Some(true),
                ..Default::default()
            }),
        );
        let v = match resp {
            Response::Ok { ok } => ok,
            Response::Err { error } => panic!("expected ok, got {error:?}"),
        };
        let hits = v["ranked_hits"].as_array().expect("ranked_hits array");
        assert!(!hits.is_empty(), "expected at least one hit");
        assert_eq!(
            hits[0]["subject"], "self",
            "rust-toolchain memory should rank first; ranking={hits:#?}"
        );
        assert_eq!(v["limit"], 5);
        assert_eq!(v["hybrid"], true);
        assert_eq!(v["query_count"].as_u64(), Some(1));
    }

    #[test]
    fn embed_returns_unit_length_vector_with_embedder_id() {
        let (state, _dir) = test_state();
        let resp = handle_request(
            &state,
            Request::Embed(EmbedArgs {
                text: "hello daemon".into(),
            }),
        );
        let v = match resp {
            Response::Ok { ok } => ok,
            Response::Err { error } => panic!("expected ok, got {error:?}"),
        };
        let dim = v["dim"].as_u64().expect("dim is u64");
        let vec: Vec<f32> = v["vector"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(vec.len() as u64, dim);
        assert_eq!(dim as usize, state.embedder_dim);
        assert!(v["embedder_id"].as_str().unwrap().starts_with("hash"));
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm was {norm}");
    }

    #[test]
    fn touch_increments_recall_count() {
        let (state, _dir) = test_state();
        let id = write_memory(&state.root, "self", "memory to be touched");
        let r1 = handle_request(&state, Request::Touch(TouchArgs { id: id.clone() }));
        let r2 = handle_request(&state, Request::Touch(TouchArgs { id: id.clone() }));
        let n1 = match r1 {
            Response::Ok { ok } => ok["recall_count"].as_u64().unwrap(),
            Response::Err { error } => panic!("first touch failed: {error:?}"),
        };
        let n2 = match r2 {
            Response::Ok { ok } => ok["recall_count"].as_u64().unwrap(),
            Response::Err { error } => panic!("second touch failed: {error:?}"),
        };
        assert_eq!(n1, 1);
        assert_eq!(n2, 2);
    }

    #[test]
    fn touch_rejects_empty_id() {
        let (state, _dir) = test_state();
        let resp = handle_request(&state, Request::Touch(TouchArgs { id: " ".into() }));
        match resp {
            Response::Err { error } => assert_eq!(error.code, "bad_request"),
            Response::Ok { .. } => panic!("expected bad_request"),
        }
    }

    #[test]
    fn ops_constant_lists_the_four_v1_ops() {
        assert_eq!(OPS, &["query", "embed", "touch", "ping"]);
    }
}
