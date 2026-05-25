//! Daemon-mode for `recalld` (PRD: recall-daemon, codename *current*).
//!
//! iter-2 wires the UDS transport, length-prefixed JSON framing, and the
//! `ping` op end-to-end. `query` / `embed` / `touch` return a structured
//! `not_implemented` error and land in iter-3+ when retrieval is plumbed
//! into the daemon. The CLI auto-forward path lands separately.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;

/// Wire-protocol op names. Stable across v0.5.x.
pub const OPS: &[&str] = &["query", "embed", "touch", "ping"];

/// Identifier the daemon reports for its loaded embedder. Until the
/// embedder is plumbed in (iter-3), this is the static name fastembed
/// has been using since v0.2.
pub const DEFAULT_MODEL_ID: &str = "bge-small-en-v1.5";

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
    pub model_id: &'static str,
    pub uptime_s: u64,
    pub query_count: u64,
    pub version: &'static str,
}

/// Per-process daemon state. The counter is atomic so handlers can
/// update it without holding the lock for the full request.
#[derive(Debug)]
pub struct DaemonState {
    pub started: Instant,
    pub query_count: std::sync::atomic::AtomicU64,
}

impl DaemonState {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            query_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn uptime_s(&self) -> u64 {
        self.started.elapsed().as_secs()
    }
}

impl Default for DaemonState {
    fn default() -> Self {
        Self::new()
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
/// loop so unit tests can drive it directly.
pub fn handle_request(state: &DaemonState, req: Request) -> Response {
    match req {
        Request::Ping(_) => {
            let body = PingResponse {
                model_id: DEFAULT_MODEL_ID,
                uptime_s: state.uptime_s(),
                query_count: state
                    .query_count
                    .load(std::sync::atomic::Ordering::Relaxed),
                version: env!("CARGO_PKG_VERSION"),
            };
            Response::ok(&body)
        }
        Request::Query(_) | Request::Embed(_) | Request::Touch(_) => Response::err(
            "not_implemented",
            "query/embed/touch land in recalld iter-3; only `ping` is wired in iter-2",
        ),
    }
}

async fn serve_connection(state: Arc<DaemonState>, mut stream: UnixStream) -> Result<()> {
    loop {
        let frame = match read_frame(&mut stream).await? {
            Some(f) => f,
            None => return Ok(()),
        };
        let resp = match serde_json::from_slice::<Request>(&frame) {
            Ok(req) => handle_request(&state, req),
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
/// shutdown signal (e.g. SIGTERM/Ctrl-C in `recalld`'s main).
///
/// # Errors
/// Returns an error if the socket path cannot be prepared, a live
/// daemon is already bound there, or the listener cannot be created.
pub async fn run_server<F>(socket_path: PathBuf, shutdown: F) -> Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    ensure_parent_dir(&socket_path)?;
    handle_existing_socket(&socket_path).await?;

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("bind UDS at {}", socket_path.display()))?;
    let state = Arc::new(DaemonState::new());
    let conns: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));

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

/// Test/dev helper: spawn `run_server` on `socket_path` and return a
/// shutdown sender plus the task handle. Public so integration tests
/// in `tests/` can use it without re-implementing framing.
#[doc(hidden)]
pub async fn spawn_for_test(
    socket_path: PathBuf,
) -> Result<(
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<Result<()>>,
)> {
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server_path = socket_path.clone();
    let handle = tokio::spawn(async move {
        run_server(server_path, async move {
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

    #[test]
    fn ping_request_handler_returns_uptime_and_model_id() {
        let state = DaemonState::new();
        let resp = handle_request(&state, Request::Ping(PingArgs {}));
        let v = match resp {
            Response::Ok { ok } => ok,
            Response::Err { .. } => panic!("expected ok response"),
        };
        assert_eq!(v["model_id"], DEFAULT_MODEL_ID);
        assert!(v["uptime_s"].as_u64().is_some());
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn query_returns_not_implemented_error_in_iter2() {
        let state = DaemonState::new();
        let resp = handle_request(
            &state,
            Request::Query(QueryArgs {
                text: "hi".into(),
                ..Default::default()
            }),
        );
        match resp {
            Response::Err { error } => assert_eq!(error.code, "not_implemented"),
            Response::Ok { .. } => panic!("expected not_implemented error"),
        }
    }

    #[test]
    fn ops_constant_lists_the_four_v1_ops() {
        assert_eq!(OPS, &["query", "embed", "touch", "ping"]);
    }
}
