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
    /// Extra weight added when the hit's subject equals the active project.
    pub project_boost: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            bm25: 1.0,
            vector: 1.5,
            recency: 0.3,
            recall_count: 0.2,
            confidence: 0.5,
            project_boost: 0.5,
        }
    }
}

impl From<&crate::config::Config> for Weights {
    fn from(c: &crate::config::Config) -> Self {
        Self {
            bm25: c.weights.bm25,
            vector: c.weights.vector,
            recency: c.weights.recency,
            recall_count: c.weights.recall_count,
            confidence: c.weights.confidence,
            project_boost: c.retrieval.project_boost,
        }
    }
}

pub fn search(idx: &Index, query: &str, limit: usize) -> Result<Vec<RankedHit>> {
    search_with(idx, query, limit, Weights::default(), None)
}

pub fn search_with(
    idx: &Index,
    query: &str,
    limit: usize,
    weights: Weights,
    project_subject: Option<&str>,
) -> Result<Vec<RankedHit>> {
    let raw = idx.search(query, limit * 4)?;
    let mut ranked: Vec<RankedHit> = raw
        .into_iter()
        .map(|h| score(h, 0.0, weights, project_subject))
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
    hybrid_with(idx, embedder, query, limit, Weights::default(), None)
}

pub fn hybrid_with(
    idx: &Index,
    embedder: &dyn Embedder,
    query: &str,
    limit: usize,
    weights: Weights,
    project_subject: Option<&str>,
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
            score(h, f64::from(sim), weights, project_subject)
        })
        .collect();
    ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    Ok(ranked)
}

fn score(hit: Hit, vector_sim: f64, w: Weights, project_subject: Option<&str>) -> RankedHit {
    let bm25_score = if hit.bm25 == 0.0 { 0.0 } else { -hit.bm25 };
    let recency_score = match hit.last_recalled_at {
        Some(t) => {
            let days = (Utc::now() - t).num_seconds() as f64 / 86_400.0;
            (-days / 30.0).exp()
        }
        None => 0.0,
    };
    let recall_score = (f64::from(hit.recall_count) / 5.0).tanh();
    let project_bonus = match project_subject {
        Some(p) if hit.subject == p => w.project_boost,
        _ => 0.0,
    };
    let total = w.bm25 * bm25_score
        + w.vector * vector_sim
        + w.recency * recency_score
        + w.recall_count * recall_score
        + w.confidence * hit.confidence
        + project_bonus;
    RankedHit {
        hit,
        score: total,
        vector_sim,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::index::Index;
    use crate::memory::{Kind, Memory, Subject};

    /// PRD-recall-outcome-feedback AC6: two otherwise-identical memories at
    /// confidence 0.9 and 0.5 rank the 0.9 higher by exactly
    /// `w.confidence * 0.4` in score. Same body so bm25 ties; no recall
    /// touches so recency/recall_count contributions are zero for both.
    #[test]
    fn ac6_confidence_drives_ranking_delta() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = Index::open(&tmp.path().join("idx.sqlite")).unwrap();

        let body = "shared body about deployment runbook for the payments cluster";
        let mut m_high = Memory::new(Kind::Semantic, Subject::user(), body);
        m_high.front.confidence = 0.9;
        let mut m_low = Memory::new(Kind::Semantic, Subject::user(), body);
        m_low.front.confidence = 0.5;

        idx.upsert(&m_high, &tmp.path().join("h.md"), None).unwrap();
        idx.upsert(&m_low, &tmp.path().join("l.md"), None).unwrap();

        let weights = Weights::default();
        let ranked = search_with(&idx, "deployment runbook payments", 5, weights, None).unwrap();

        assert_eq!(ranked.len(), 2, "both memories should match the shared body");
        assert_eq!(
            ranked[0].hit.id, m_high.front.id,
            "0.9-confidence memory should rank first"
        );
        assert_eq!(ranked[1].hit.id, m_low.front.id);

        let delta = ranked[0].score - ranked[1].score;
        let expected = weights.confidence * 0.4;
        assert!(
            (delta - expected).abs() < 1e-9,
            "score delta should equal w.confidence * 0.4 (= {expected}); got {delta}"
        );
    }
}
