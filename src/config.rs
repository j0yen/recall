//! `recall.toml` — optional config that overrides hard-coded defaults.
//!
//! Lives at `<root>/recall.toml` (so `~/.claude/recall/recall.toml` by
//! default; honors `$RECALL_HOME`). Missing-file is fine: every field has
//! a default that matches the v0.1/v0.2 hard-coded behavior.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default)]
pub struct Config {
    pub weights: Weights,
    pub retrieval: Retrieval,
    pub embedder: EmbedderCfg,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct Weights {
    pub bm25: f64,
    pub vector: f64,
    pub recency: f64,
    pub recall_count: f64,
    pub confidence: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            bm25: 1.0,
            vector: 1.5,
            recency: 0.3,
            recall_count: 0.2,
            confidence: 0.5,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct Retrieval {
    pub include_superseded: bool,
    pub include_decayed: bool,
    /// Extra additive weight when a hit's subject matches the active project.
    pub project_boost: f64,
    /// 0 = unlimited. Caller-side budget on list/query body bytes.
    pub max_tokens: usize,
}

impl Default for Retrieval {
    fn default() -> Self {
        Self {
            include_superseded: false,
            include_decayed: false,
            project_boost: 0.5,
            max_tokens: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct EmbedderCfg {
    /// `fastembed` (default) or `hash` for offline contexts.
    pub model: String,
}

impl Default for EmbedderCfg {
    fn default() -> Self {
        Self {
            model: "fastembed".to_string(),
        }
    }
}

impl Config {
    /// Load from `<root>/recall.toml`. Missing file → defaults; malformed file
    /// → error so the user sees the problem rather than silently inheriting
    /// every default.
    pub fn load(root: &Path) -> Result<Self> {
        let path = root.join("recall.toml");
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let cfg: Self = toml::from_str(&text)
            .with_context(|| format!("parse {}", path.display()))?;
        Ok(cfg)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn missing_file_yields_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = Config::load(tmp.path()).unwrap();
        let d = Weights::default();
        assert!((cfg.weights.vector - d.vector).abs() < 1e-9);
        assert_eq!(cfg.embedder.model, "fastembed");
        assert_eq!(cfg.retrieval.max_tokens, 0);
    }

    #[test]
    fn partial_overrides_keep_other_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("recall.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "[weights]\nvector = 2.0\n\n[retrieval]\nmax_tokens = 500").unwrap();
        drop(f);
        let cfg = Config::load(tmp.path()).unwrap();
        assert!((cfg.weights.vector - 2.0).abs() < 1e-9);
        assert!((cfg.weights.bm25 - 1.0).abs() < 1e-9); // unchanged
        assert_eq!(cfg.retrieval.max_tokens, 500);
        assert_eq!(cfg.embedder.model, "fastembed"); // unchanged
    }
}
