//! recall-lint — CLI binary for recall-memory-linter.

#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::doc_markdown,
    clippy::redundant_closure_for_method_calls
)]

use clap::{Parser, ValueEnum};
use recall_memory_linter::{LintConfig, lint, render_json, render_text, walk_memories};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "recall-lint", about = "Lint a recall memory store")]
struct Cli {
    /// Path to the memory root directory.
    root: PathBuf,
    /// Output format
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
    /// Flag entries older than this many days as `stale`.
    #[arg(long, default_value_t = 30)]
    stale_days: u32,
    /// Glob pattern (root-relative) to ignore. Repeatable.
    #[arg(long, action = clap::ArgAction::Append)]
    ignore: Vec<String>,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Text,
    Json,
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
            let _ = e.print();
            return ExitCode::from(2);
        }
    };

    let entries = match walk_memories(&cli.root, &cli.ignore) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("recall-lint: {err}");
            return ExitCode::from(2);
        }
    };

    let cfg = LintConfig {
        stale_days: cli.stale_days,
        ignore: cli.ignore.clone(),
        ..LintConfig::default()
    };
    let findings = lint(&entries, &cfg);

    match cli.format {
        Format::Json => match render_json(&findings) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("recall-lint: render: {e}");
                return ExitCode::from(2);
            }
        },
        Format::Text => print!("{}", render_text(&findings)),
    }

    if findings.is_empty() {
        ExitCode::from(0)
    } else {
        ExitCode::from(1)
    }
}
