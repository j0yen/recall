//! `recalld` — the recall daemon binary (iter-3: all four ops wired
//! against an open `Index` + built `Embedder`). Foreground server
//! only; the systemd-user unit and the `recall daemon start/stop/
//! status` CLI glue land in iter-4.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use recall::daemon::{self, DaemonState};
use recall::embeddings::EmbedderKind;
use recall::paths;

fn parse_value_flag(args: &[String], flag: &str) -> Option<String> {
    let prefix = format!("{flag}=");
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == flag {
            return it.next().cloned();
        }
        if let Some(rest) = a.strip_prefix(&prefix) {
            return Some(rest.to_string());
        }
    }
    None
}

fn parse_socket_override(args: &[String]) -> Option<PathBuf> {
    parse_value_flag(args, "--socket").map(PathBuf::from)
}

fn parse_root_override(args: &[String]) -> Option<PathBuf> {
    parse_value_flag(args, "--root").map(PathBuf::from)
}

fn parse_embedder_override(args: &[String]) -> Option<String> {
    parse_value_flag(args, "--embedder")
}

fn print_help() {
    eprintln!("recalld v{} — recall daemon", env!("CARGO_PKG_VERSION"));
    eprintln!(
        "Usage: recalld [--socket PATH] [--root PATH] [--embedder hash|fastembed] [--help]"
    );
    eprintln!();
    eprintln!("Listens on a Unix-domain socket and serves recall ops to");
    eprintln!("hook-cadence clients. SIGINT or SIGTERM trigger a graceful");
    eprintln!("shutdown that removes the socket file.");
    eprintln!();
    eprintln!("Default socket:   $XDG_RUNTIME_DIR/recall.sock");
    eprintln!("Default root:     ~/.claude/recall  (per `recall where`)");
    eprintln!("Default embedder: fastembed");
    eprintln!("Wire ops:         {}", daemon::OPS.join(", "));
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return ExitCode::SUCCESS;
    }

    let sock = match parse_socket_override(&args) {
        Some(p) => p,
        None => match daemon::default_socket_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("recalld: cannot resolve default socket path: {e}");
                return ExitCode::from(2);
            }
        },
    };

    let root = match parse_root_override(&args) {
        Some(p) => p,
        None => match paths::root() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("recalld: cannot resolve default recall root: {e}");
                return ExitCode::from(2);
            }
        },
    };

    let embedder_kind = match parse_embedder_override(&args).as_deref() {
        Some(name) => match EmbedderKind::parse(name) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("recalld: {e}");
                return ExitCode::from(2);
            }
        },
        None => EmbedderKind::Fastembed,
    };

    let state = match DaemonState::open(root.clone(), embedder_kind) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("recalld: cannot open daemon state: {e:#}");
            return ExitCode::from(2);
        }
    };

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("recalld: cannot build tokio runtime: {e}");
            return ExitCode::from(2);
        }
    };

    let sock_for_log = sock.clone();
    let result = runtime.block_on(async move {
        eprintln!(
            "recalld v{} listening on {} (root={}, embedder={})",
            env!("CARGO_PKG_VERSION"),
            sock_for_log.display(),
            state.root.display(),
            state.embedder_id,
        );
        let shutdown = async {
            let mut sigterm = match tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::terminate(),
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("recalld: cannot install SIGTERM handler: {e}");
                    return;
                }
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    eprintln!("recalld: SIGINT received, shutting down");
                }
                _ = sigterm.recv() => {
                    eprintln!("recalld: SIGTERM received, shutting down");
                }
            }
        };
        daemon::run_server(state, sock_for_log, shutdown).await
    });

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("recalld: {e:#}");
            ExitCode::from(1)
        }
    }
}
