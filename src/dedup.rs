//! Memory deduplication: cluster near-duplicate memories by cosine similarity.
//!
//! `recall dedup` scans all memories with stored embeddings, computes pairwise
//! cosine similarity (dot product over L2-normalized vectors), groups pairs
//! above a configurable threshold using union-find, then reports the clusters.
//!
//! This is a **dry-run only** command: it never writes to the database or
//! modifies any memory file. All results are printed to stdout.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Options for a dedup run.
#[derive(Debug, Clone)]
pub struct DedupOpts {
    /// Cosine-similarity threshold; pairs above this are considered duplicates.
    pub threshold: f64,
    /// Only report clusters with at least this many members.
    pub min_cluster: usize,
    /// Emit JSON instead of human-readable text.
    pub json: bool,
}

impl Default for DedupOpts {
    fn default() -> Self {
        Self {
            threshold: 0.92,
            min_cluster: 2,
            json: false,
        }
    }
}

/// One near-duplicate cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupCluster {
    /// IDs of the clustered memories.
    pub ids: Vec<String>,
    /// Subjects of the clustered memories (parallel to `ids`).
    pub subjects: Vec<String>,
    /// Kind of each memory (parallel to `ids`).
    pub kinds: Vec<String>,
    /// Maximum pairwise similarity observed within the cluster.
    pub similarity: f64,
    /// Recommended action for the user.
    pub recommended_action: String,
}

/// Top-level result from a dedup run.
#[derive(Debug, Serialize, Deserialize)]
pub struct DedupResult {
    /// Clusters of near-duplicate memories (already filtered by `min_cluster`).
    pub clusters: Vec<DedupCluster>,
    /// How many memories had embeddings and were examined.
    pub memories_scanned: usize,
    /// Similarity threshold that was used.
    pub threshold: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Union-find
// ──────────────────────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core logic
// ──────────────────────────────────────────────────────────────────────────────

/// Row fetched from the index for dedup purposes.
#[derive(Debug)]
struct EmbedRow {
    id: String,
    subject: String,
    kind: String,
    vec: Vec<f32>,
}

/// Run the dedup algorithm against the recall index at `root`.
pub fn run_dedup(root: &std::path::Path, opts: &DedupOpts) -> Result<DedupResult> {
    let idx = crate::index::Index::open(&crate::paths::index_db(root))
        .context("open recall index")?;

    let rows = fetch_embeddings(&idx)?;
    let n = rows.len();

    if n < 2 {
        return Ok(DedupResult {
            clusters: Vec::new(),
            memories_scanned: n,
            threshold: opts.threshold,
        });
    }

    // Pairwise cosine similarity + union-find clustering.
    let mut uf = UnionFind::new(n);
    // Track the maximum similarity seen for each root (union representative).
    let mut max_sim: Vec<f64> = vec![0.0; n];

    let threshold = opts.threshold as f32;

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = crate::embeddings::cosine(&rows[i].vec, &rows[j].vec);
            if sim >= threshold {
                let ri = uf.find(i);
                let rj = uf.find(j);
                let sim_f64 = f64::from(sim);
                // Before merging, update the max for both current roots.
                if sim_f64 > max_sim[ri] {
                    max_sim[ri] = sim_f64;
                }
                if sim_f64 > max_sim[rj] {
                    max_sim[rj] = sim_f64;
                }
                uf.union(i, j);
                // Propagate max to the new root.
                let new_root = uf.find(i);
                let other_root = if new_root == ri { rj } else { ri };
                if max_sim[other_root] > max_sim[new_root] {
                    max_sim[new_root] = max_sim[other_root];
                }
            }
        }
    }

    // Collect cluster members by root.
    let mut cluster_map: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let root_idx = uf.find(i);
        cluster_map.entry(root_idx).or_default().push(i);
    }

    let threshold_f64 = opts.threshold;
    let mut clusters: Vec<DedupCluster> = cluster_map
        .into_iter()
        .filter_map(|(root_idx, members)| {
            if members.len() < opts.min_cluster {
                return None;
            }
            let ids: Vec<String> = members.iter().map(|&i| rows[i].id.clone()).collect();
            let subjects: Vec<String> = members.iter().map(|&i| rows[i].subject.clone()).collect();
            let kinds: Vec<String> = members.iter().map(|&i| rows[i].kind.clone()).collect();
            let similarity = max_sim[root_idx].max(threshold_f64);
            let recommended_action = pick_action(&kinds);
            Some(DedupCluster {
                ids,
                subjects,
                kinds,
                similarity,
                recommended_action,
            })
        })
        .collect();

    // Stable sort: largest clusters first, then by descending similarity.
    clusters.sort_by(|a, b| {
        b.ids
            .len()
            .cmp(&a.ids.len())
            .then(b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal))
    });

    Ok(DedupResult {
        clusters,
        memories_scanned: n,
        threshold: opts.threshold,
    })
}

/// Fetch all memories that have a stored embedding.
fn fetch_embeddings(idx: &crate::index::Index) -> Result<Vec<EmbedRow>> {
    let raw = idx.get_all_embeddings().context("fetch all embeddings from index")?;
    Ok(raw
        .into_iter()
        .map(|(id, kind, subject, vec)| EmbedRow { id, kind, subject, vec })
        .collect())
}

/// Choose a recommended action based on the kinds present in the cluster.
fn pick_action(kinds: &[String]) -> String {
    // All episodic → merge into newest (lowest knowledge debt).
    // Any semantic → keep highest confidence.
    // Otherwise → manual review.
    let all_episodic = kinds.iter().all(|k| k == "episodic");
    let any_semantic = kinds.iter().any(|k| k == "semantic");
    if all_episodic {
        "merge-into-newest".to_string()
    } else if any_semantic {
        "keep-highest-confidence".to_string()
    } else {
        "review-manually".to_string()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Rendering
// ──────────────────────────────────────────────────────────────────────────────

/// Print the dedup result to stdout according to `opts.json`.
pub fn render_result(result: &DedupResult, opts: &DedupOpts) -> Result<()> {
    if opts.json {
        let out = serde_json::json!({
            "clusters": result.clusters,
            "memories_scanned": result.memories_scanned,
            "threshold": result.threshold,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "dedup: scanned {} memories at threshold {:.2}",
            result.memories_scanned, result.threshold
        );
        if result.clusters.is_empty() {
            println!("No duplicate clusters found.");
        } else {
            println!("{} cluster(s) found:\n", result.clusters.len());
            for (ci, cluster) in result.clusters.iter().enumerate() {
                println!(
                    "cluster {}  ({} members)  sim={:.4}  action={}",
                    ci + 1,
                    cluster.ids.len(),
                    cluster.similarity,
                    cluster.recommended_action
                );
                for (id, subj, kind) in cluster.ids.iter().zip(cluster.subjects.iter()).zip(cluster.kinds.iter()).map(|((id, s), k)| (id, s, k)) {
                    println!("  [{kind}] {id}  {subj}");
                }
                println!();
            }
        }
    }
    Ok(())
}
