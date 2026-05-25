//! `recalld` — the recall daemon binary (iter-2: ping op wired,
//! query/embed/touch return `not_implemented`). Foreground server only;
//! the systemd-user unit and the `recall daemon start/stop/status` CLI
//! glue land in iter-3.

use std::path::PathBuf;
use std::process::ExitCode;

use recall::daemon;

fn parse_socket_override(args: &[String]) -> Option<PathBuf> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--socket" {
            return it.next().map(PathBuf::from);
        }
        if let Some(rest) = a.strip_prefix("--socket=") {
            return Some(PathBuf::from(rest));
        }
    }
    None
}

fn print_help() {
    eprintln!("recalld v{} — recall daemon", env!("CARGO_PKG_VERSION"));
    eprintln!("Usage: recalld [--socket PATH] [--help]");
    eprintln!();
    eprintln!("Listens on a Unix-domain socket and serves recall ops to");
    eprintln!("hook-cadence clients. SIGINT or SIGTERM trigger a graceful");
    eprintln!("shutdown that removes the socket file.");
    eprintln!();
    eprintln!("Default socket: $XDG_RUNTIME_DIR/recall.sock");
    eprintln!("Wire ops:       {}", daemon::OPS.join(", "));
    eprintln!("  (iter-2 implements `ping` only; others return not_implemented)");
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
            "recalld v{} listening on {}",
            env!("CARGO_PKG_VERSION"),
            sock_for_log.display()
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
        daemon::run_server(sock_for_log, shutdown).await
    });

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("recalld: {e:#}");
            ExitCode::from(1)
        }
    }
}
