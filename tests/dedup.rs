//! Integration tests for `recall dedup` (PRD-recall-memdedup).
//!
//! ACs tested:
//! AC1: exits 0 on store with zero embeddings
//! AC2: exits 0 on store with < 2 memories
//! AC3: two identical-body memories appear in the same cluster
//! AC4: two clearly different memories are NOT clustered at default threshold
//! AC5: --threshold 1.0 clusters exact copies; --threshold 0.5 produces more clusters
//! AC6: --json output is valid JSON with `clusters` array; fields present
//! AC7: --min-cluster 3 omits clusters smaller than 3

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::dedup::{DedupOpts, run_dedup};
use recall::embeddings::HashEmbedder;
use recall::embeddings::Embedder;
use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use recall::paths;
use recall::store::FileStore;

/// Write a memory with a stored embedding using the hash embedder.
fn seed_with_embedding(
    idx: &Index,
    store: &FileStore,
    body: &str,
    kind: Kind,
) -> String {
    let mem = Memory::new(kind, Subject::user(), body);
    let path = store.write(&mem).unwrap();
    let embedder = HashEmbedder::new();
    let vec = embedder.embed(body).unwrap();
    idx.upsert(&mem, &path, Some((embedder.id(), &vec))).unwrap();
    mem.front.id.clone()
}

// ──────────────────────────────────────────────────────────────────────────────
// AC1: zero embeddings → exits cleanly with empty clusters
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn ac1_zero_embeddings_returns_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Empty store (no memories, no embeddings).
    let _store = FileStore::open(root.to_path_buf()).unwrap();
    let _idx = Index::open(&paths::index_db(root)).unwrap();

    let opts = DedupOpts::default();
    let result = run_dedup(root, &opts).expect("dedup should succeed on empty store");

    assert_eq!(result.clusters.len(), 0);
    assert_eq!(result.memories_scanned, 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// AC2: fewer than 2 embeddings → exits cleanly with empty clusters
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn ac2_single_memory_returns_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();

    seed_with_embedding(&idx, &store, "only one memory in the store", Kind::Semantic);

    let opts = DedupOpts::default();
    let result = run_dedup(root, &opts).expect("dedup should succeed with one memory");

    assert_eq!(result.clusters.len(), 0);
    assert_eq!(result.memories_scanned, 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// AC3: identical body → same cluster
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn ac3_identical_bodies_clustered_together() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();

    let body = "The Rust programming language is memory safe without GC";
    let id1 = seed_with_embedding(&idx, &store, body, Kind::Semantic);
    let id2 = seed_with_embedding(&idx, &store, body, Kind::Semantic);

    // Identical embeddings → cosine = 1.0, well above default threshold 0.92.
    let opts = DedupOpts::default();
    let result = run_dedup(root, &opts).unwrap();

    assert!(!result.clusters.is_empty(), "expected at least one cluster");
    let cluster = result
        .clusters
        .iter()
        .find(|c| c.ids.contains(&id1) && c.ids.contains(&id2));
    assert!(
        cluster.is_some(),
        "id1 and id2 should be in the same cluster; clusters={:?}",
        result.clusters.iter().map(|c| &c.ids).collect::<Vec<_>>()
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// AC4: clearly different memories → NOT clustered at default threshold
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn ac4_different_memories_not_clustered() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();

    // Very different bodies → low cosine.
    let id1 = seed_with_embedding(
        &idx,
        &store,
        "Rust ownership system prevents data races at compile time through borrow checker",
        Kind::Semantic,
    );
    let id2 = seed_with_embedding(
        &idx,
        &store,
        "French cooking uses butter cream wine herbs garlic onions tomatoes",
        Kind::Semantic,
    );

    let opts = DedupOpts::default(); // threshold 0.92
    let result = run_dedup(root, &opts).unwrap();

    // Verify id1 and id2 are not in the same cluster.
    for cluster in &result.clusters {
        let has1 = cluster.ids.contains(&id1);
        let has2 = cluster.ids.contains(&id2);
        assert!(
            !(has1 && has2),
            "unrelated memories should not be in the same cluster"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AC5a: --threshold 1.0 only clusters exact copies (identical vectors)
// AC5b: --threshold 0.5 produces more (or equal) clusters than default
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn ac5a_threshold_1_0_clusters_only_exact_copies() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();

    let body = "exact duplicate body for threshold test";
    // Two identical → they will cluster at threshold 1.0.
    seed_with_embedding(&idx, &store, body, Kind::Semantic);
    seed_with_embedding(&idx, &store, body, Kind::Semantic);
    // Slightly different → should NOT cluster at threshold 1.0.
    seed_with_embedding(
        &idx,
        &store,
        "slightly different body for threshold test with extra words",
        Kind::Semantic,
    );

    let opts = DedupOpts { threshold: 1.0, min_cluster: 2, json: false };
    let result = run_dedup(root, &opts).unwrap();

    // At threshold 1.0, only the two identical bodies should cluster.
    assert_eq!(
        result.clusters.len(),
        1,
        "at threshold 1.0, exactly one cluster expected; got {:?}",
        result.clusters.iter().map(|c| &c.ids).collect::<Vec<_>>()
    );
    assert_eq!(result.clusters[0].ids.len(), 2);
}

#[test]
fn ac5b_lower_threshold_produces_more_or_equal_clusters() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();

    // Use a pair of identical bodies so default threshold can find them.
    let body = "cluster candidate body text same text";
    seed_with_embedding(&idx, &store, body, Kind::Semantic);
    seed_with_embedding(&idx, &store, body, Kind::Semantic);
    // A pair with moderate similarity (same topic, different phrasing).
    seed_with_embedding(&idx, &store, "cargo build compiles Rust code", Kind::Semantic);
    seed_with_embedding(&idx, &store, "cargo build compiles your Rust project", Kind::Semantic);

    let default_result = run_dedup(
        root,
        &DedupOpts { threshold: 0.92, min_cluster: 2, json: false },
    )
    .unwrap();
    let low_result = run_dedup(
        root,
        &DedupOpts { threshold: 0.5, min_cluster: 2, json: false },
    )
    .unwrap();

    assert!(
        low_result.clusters.len() >= default_result.clusters.len(),
        "lower threshold should produce >= clusters than default; low={} default={}",
        low_result.clusters.len(),
        default_result.clusters.len(),
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// AC6: --json emits valid JSON with `clusters` array and required fields
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn ac6_json_output_valid_structure() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();

    let body = "json output test body text identical copy";
    seed_with_embedding(&idx, &store, body, Kind::Semantic);
    seed_with_embedding(&idx, &store, body, Kind::Semantic);

    let opts = DedupOpts { threshold: 0.92, min_cluster: 2, json: true };
    let result = run_dedup(root, &opts).unwrap();

    // Serialize the same way render_result would.
    let json_val = serde_json::json!({
        "clusters": result.clusters,
        "memories_scanned": result.memories_scanned,
        "threshold": result.threshold,
    });

    // `clusters` must be an array.
    let clusters = json_val["clusters"].as_array().expect("clusters must be array");
    assert!(!clusters.is_empty(), "expected at least one cluster in JSON");

    for cluster in clusters {
        assert!(cluster.get("ids").is_some(), "cluster must have ids");
        assert!(cluster.get("subjects").is_some(), "cluster must have subjects");
        assert!(cluster.get("similarity").is_some(), "cluster must have similarity");
        assert!(
            cluster.get("recommended_action").is_some(),
            "cluster must have recommended_action"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AC7: --min-cluster 3 omits clusters smaller than 3 members
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn ac7_min_cluster_filters_small_clusters() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let store = FileStore::open(root.to_path_buf()).unwrap();
    let idx = Index::open(&paths::index_db(root)).unwrap();

    // Two identical → 2-member cluster.
    let body = "two member cluster body";
    seed_with_embedding(&idx, &store, body, Kind::Semantic);
    seed_with_embedding(&idx, &store, body, Kind::Semantic);

    // Three identical → 3-member cluster.
    let body3 = "three member cluster body text content";
    seed_with_embedding(&idx, &store, body3, Kind::Semantic);
    seed_with_embedding(&idx, &store, body3, Kind::Semantic);
    seed_with_embedding(&idx, &store, body3, Kind::Semantic);

    // With min_cluster=2 we expect 2 clusters (one pair, one triple).
    let result_2 = run_dedup(
        root,
        &DedupOpts { threshold: 0.92, min_cluster: 2, json: false },
    )
    .unwrap();
    assert!(result_2.clusters.len() >= 2, "min_cluster=2 should show both clusters");

    // With min_cluster=3, the 2-member cluster should be filtered out.
    let result_3 = run_dedup(
        root,
        &DedupOpts { threshold: 0.92, min_cluster: 3, json: false },
    )
    .unwrap();
    for cluster in &result_3.clusters {
        assert!(
            cluster.ids.len() >= 3,
            "with min_cluster=3, all reported clusters must have >= 3 members; got {}",
            cluster.ids.len()
        );
    }
}
