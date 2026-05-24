//! Composite retrieval: keyword + vector + recency + recall-count.
//!
//! `search` does FTS5-only ranking (Phase 1 behavior).
//! `hybrid_search` adds a vector-cosine column from any `Embedder` and merges
//! both leaderboards by additive score.

use crate::embeddings::Embedder;
use crate::index::{Hit, Index};
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RankedHit {
    pub hit: Hit,
    pub score: f64,
    pub vector_sim: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Weights {
    /// Negated BM25 contribution. `SQLite`'s `bm25()` is lower-is-better; we negate.
    pub bm25: f64,
    /// Cosine similarity from the embedder, in [-1, 1].
    pub vector: f64,
    /// Recency contribution. Decays with days since `last_recalled_at`.
    pub recency: f64,
    /// `tanh`-squashed `recall_count`.
    pub recall_count: f64,
    /// Confidence in [0,1].
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

pub fn search(idx: &Index, query: &str, limit: usize) -> Result<Vec<RankedHit>> {
    search_with(idx, query, limit, Weights::default())
}

pub fn search_with(
    idx: &Index,
    query: &str,
    limit: usize,
    weights: Weights,
) -> Result<Vec<RankedHit>> {
    let raw = idx.search(query, limit * 4)?;
    let mut ranked: Vec<RankedHit> = raw
        .into_iter()
        .map(|h| score(h, 0.0, weights))
        .collect();
    ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    Ok(ranked)
}

/// Hybrid retrieval: union of FTS5 keyword hits and vector-cosine hits.
pub fn hybrid_search(
    idx: &Index,
    embedder: &dyn Embedder,
    query: &str,
    limit: usize,
) -> Result<Vec<RankedHit>> {
    hybrid_with(idx, embedder, query, limit, Weights::default())
}

pub fn hybrid_with(
    idx: &Index,
    embedder: &dyn Embedder,
    query: &str,
    limit: usize,
    weights: Weights,
) -> Result<Vec<RankedHit>> {
    let overfetch = limit * 4;
    let fts = idx.search(query, overfetch)?;
    let qvec = embedder.embed(query)?;
    let vec_hits = idx.vector_search(&qvec, overfetch)?;

    // Merge by id. FTS5 hits carry a snippet + bm25; vector hits carry sim.
    let mut by_id: HashMap<String, (Hit, f64, f32, bool)> = HashMap::new();
    for h in fts {
        let id = h.id.clone();
        let bm25 = h.bm25;
        by_id.insert(id, (h, bm25, 0.0, true));
    }
    for (h, sim) in vec_hits {
        let id = h.id.clone();
        by_id
            .entry(id)
            .and_modify(|(_, _, s, _)| *s = sim)
            .or_insert((h, 0.0, sim, false));
    }

    let mut ranked: Vec<RankedHit> = by_id
        .into_values()
        .map(|(mut h, bm25, sim, has_bm25)| {
            h.bm25 = if has_bm25 { bm25 } else { 0.0 };
            score(h, f64::from(sim), weights)
        })
        .collect();
    ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    Ok(ranked)
}

fn score(hit: Hit, vector_sim: f64, w: Weights) -> RankedHit {
    let bm25_score = if hit.bm25 == 0.0 { 0.0 } else { -hit.bm25 };
    let recency_score = match hit.last_recalled_at {
        Some(t) => {
            let days = (Utc::now() - t).num_seconds() as f64 / 86_400.0;
            (-days / 30.0).exp()
        }
        None => 0.0,
    };
    let recall_score = (f64::from(hit.recall_count) / 5.0).tanh();
    let total = w.bm25 * bm25_score
        + w.vector * vector_sim
        + w.recency * recency_score
        + w.recall_count * recall_score
        + w.confidence * hit.confidence;
    RankedHit {
        hit,
        score: total,
        vector_sim,
    }
}
