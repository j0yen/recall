//! Embeddings: trait + two implementations.
//!
//! - `HashEmbedder`: zero-dependency hashed-feature default (v0.1). Catches
//!   morphological variation but misses synonyms. Stays available behind
//!   `--embedder hash` for tests and offline contexts.
//! - `FastembedEmbedder`: in-process BGE-small-en-v1.5 via `fastembed-rs`.
//!   Lazy-fetches the ONNX model to `~/.cache/fastembed/` on first
//!   construction; produces 384-dim L2-normalized embeddings. This is the
//!   v0.2 default when `--embedder` is not specified.

use anyhow::{Context, Result};
use std::hash::{DefaultHasher, Hash, Hasher};

pub const DIM: usize = 256;

pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    /// Returns a unit-length vector of length `self.dim()`.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    /// Stable identifier written into the index so a model swap can trigger reindex.
    fn id(&self) -> &str;
}

/// Hashed-feature default. No model, no deps. Deterministic per (id, text).
pub struct HashEmbedder {
    dim: usize,
    id: String,
}

impl HashEmbedder {
    pub fn new() -> Self {
        Self {
            dim: DIM,
            id: format!("hash-v1-d{DIM}"),
        }
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0.0_f32; self.dim];
        let lower = text.to_lowercase();

        // Word tokens (alphanumeric runs).
        for tok in lower.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()) {
            add_feature(&mut v, &format!("w:{tok}"), 1.0);
        }

        // Character bigrams over the raw lowercased text (skipping multi-whitespace runs).
        let chars: Vec<char> = lower.chars().filter(|c| !c.is_whitespace()).collect();
        for w in chars.windows(2) {
            let bigram = format!("b:{}{}", w[0], w[1]);
            add_feature(&mut v, &bigram, 0.5);
        }

        // Character trigrams — a touch of context.
        for w in chars.windows(3) {
            let trigram: String = w.iter().collect();
            add_feature(&mut v, &format!("t:{trigram}"), 0.3);
        }

        l2_normalize(&mut v);
        Ok(v)
    }
}

fn add_feature(v: &mut [f32], key: &str, weight: f32) {
    let mut h = DefaultHasher::new();
    key.hash(&mut h);
    let raw = h.finish();
    let idx = usize::try_from(raw % (v.len() as u64)).unwrap_or(0);
    // Sign bit from the next hash so collisions partially cancel.
    let sign = if (raw >> 32) & 1 == 0 { 1.0 } else { -1.0 };
    v[idx] += sign * weight;
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    // Both expected unit-length; this is just a dot product then.
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Pack a `[f32]` into a little-endian byte BLOB for `SQLite` storage.
pub fn pack(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Unpack a byte BLOB back into a `Vec<f32>`. Returns an empty vec on length mismatch.
pub fn unpack(bytes: &[u8]) -> Vec<f32> {
    if bytes.len() % 4 != 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(arr));
    }
    out
}

/// In-process semantic embedder backed by BGE-small-en-v1.5 via `fastembed-rs`.
///
/// The model (~130MB ONNX) is fetched lazily on first construction into
/// `~/.cache/fastembed/`. Subsequent constructions reuse the cache. Each
/// embedding call is `&self` (fastembed-rs's `TextEmbedding::embed` is
/// non-mutating), so this fits the `Embedder` trait directly.
///
/// `embedding_id` is the fixed string `"fastembed:bge-small-en-v1.5"` so
/// the index can detect a model swap and trigger reindex.
pub struct FastembedEmbedder {
    model: fastembed::TextEmbedding,
    dim: usize,
    id: String,
}

impl FastembedEmbedder {
    /// Construct a new `FastembedEmbedder`. Triggers the model fetch on first
    /// call in a fresh `~/.cache/fastembed/` (~500ms–1s warm cache,
    /// longer if download is required).
    ///
    /// Cache path is anchored at `$XDG_CACHE_HOME/fastembed/`, falling
    /// back to `~/.cache/fastembed/`. Without this, fastembed-rs defaults
    /// to `./fastembed_cache/` (relative to PWD), which (a) leaks a
    /// 130MB blob into whichever directory you happen to invoke from
    /// and (b) refuses to share the model between sessions.
    pub fn new() -> Result<Self> {
        let cache_dir = Self::cache_dir()?;
        std::fs::create_dir_all(&cache_dir)
            .with_context(|| format!("create fastembed cache dir at {}", cache_dir.display()))?;
        let opts = fastembed::InitOptions::new(fastembed::EmbeddingModel::BGESmallENV15)
            .with_cache_dir(cache_dir);
        let model = fastembed::TextEmbedding::try_new(opts)
            .context("initialize fastembed BGE-small-en-v1.5")?;
        Ok(Self {
            model,
            dim: 384,
            id: "fastembed:bge-small-en-v1.5".to_string(),
        })
    }

    fn cache_dir() -> Result<std::path::PathBuf> {
        if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            if !xdg.is_empty() {
                return Ok(std::path::PathBuf::from(xdg).join("fastembed"));
            }
        }
        let home = std::env::var("HOME").context("HOME not set; cannot resolve fastembed cache")?;
        Ok(std::path::PathBuf::from(home)
            .join(".cache")
            .join("fastembed"))
    }
}

impl Embedder for FastembedEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self
            .model
            .embed(vec![text], None)
            .context("fastembed embed")?;
        let mut v = out.pop().context("fastembed returned no embeddings")?;
        // fastembed-rs's BGE-small output is already L2-normalized, but
        // we re-normalize defensively so the `cosine()` dot-product
        // assumption is upheld regardless of upstream changes.
        l2_normalize(&mut v);
        Ok(v)
    }
}

/// Named selector for the CLI `--embedder` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderKind {
    Hash,
    Fastembed,
}

impl EmbedderKind {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "hash" => Ok(Self::Hash),
            "fastembed" => Ok(Self::Fastembed),
            other => Err(anyhow::anyhow!(
                "unknown embedder {other:?}; expected hash|fastembed"
            )),
        }
    }

    /// Build the boxed embedder. `Fastembed` triggers the lazy model load.
    pub fn build(self) -> Result<Box<dyn Embedder>> {
        match self {
            Self::Hash => Ok(Box::new(HashEmbedder::new())),
            Self::Fastembed => Ok(Box::new(FastembedEmbedder::new()?)),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn embed_is_unit_length() {
        let e = HashEmbedder::new();
        let v = e.embed("user prefers integration tests over mocks").unwrap();
        assert_eq!(v.len(), DIM);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn embed_is_deterministic() {
        let e = HashEmbedder::new();
        let a = e.embed("hello world").unwrap();
        let b = e.embed("hello world").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn related_strings_score_higher_than_unrelated() {
        let e = HashEmbedder::new();
        let q = e.embed("build the rust project with cargo").unwrap();
        let near = e
            .embed("build the rust project using cargo build --release")
            .unwrap();
        let far = e
            .embed("user prefers integration tests over mocks for auth")
            .unwrap();
        let s_near = cosine(&q, &near);
        let s_far = cosine(&q, &far);
        assert!(s_near > s_far, "near={s_near} far={s_far}");
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let v = vec![0.1_f32, -0.2, 1e-3, 12.5];
        let bytes = pack(&v);
        let back = unpack(&bytes);
        assert_eq!(back.len(), v.len());
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
