//! End-to-end daemon test: spawn the UDS server, send a `ping`, assert
//! the response shape. Covers part of PRD-recall-daemon AC-1 (cold start
//! → responsive) and the framing/protocol contract. Updated for iter-3,
//! which dispatches all four wire ops against a real `DaemonState`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use recall::daemon::{self, DaemonState};
use recall::embeddings::EmbedderKind;
use serde_json::json;

fn build_state() -> (Arc<DaemonState>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let state = DaemonState::open(dir.path().to_path_buf(), EmbedderKind::Hash).unwrap();
    (Arc::new(state), dir)
}

#[tokio::test]
async fn ping_roundtrip_returns_uptime_and_model_id() {
    let (state, dir) = build_state();
    let sock = dir.path().join("recall.sock");

    let (shutdown, handle) = daemon::spawn_for_test(state, sock.clone()).await.unwrap();

    let resp = daemon::client_roundtrip(&sock, &json!({"op": "ping", "args": {}}))
        .await
        .unwrap();

    assert!(resp.get("ok").is_some(), "expected ok response, got {resp}");
    let ok = &resp["ok"];
    // HashEmbedder reports an id like "hash-v1-d256"; production paths
    // report the fastembed model id. Either way it should be non-empty.
    let model_id = ok["model_id"].as_str().expect("model_id is a string");
    assert!(model_id.starts_with("hash"), "model_id was {model_id}");
    assert!(ok["uptime_s"].as_u64().is_some());
    assert!(ok["version"].as_str().is_some());
    assert!(ok["root"].as_str().is_some());

    drop(shutdown);
    let _ = handle.await;
    assert!(!sock.exists(), "socket should be unlinked after shutdown");
}

#[tokio::test]
async fn query_op_returns_ok_with_empty_hits_against_fresh_index() {
    let (state, dir) = build_state();
    let sock = dir.path().join("recall.sock");

    let (shutdown, handle) = daemon::spawn_for_test(state, sock.clone()).await.unwrap();

    let resp = daemon::client_roundtrip(
        &sock,
        &json!({"op": "query", "args": {"text": "anything goes here"}}),
    )
    .await
    .unwrap();

    assert!(resp.get("ok").is_some(), "expected ok, got {resp}");
    let ok = &resp["ok"];
    let hits = ok["ranked_hits"].as_array().expect("ranked_hits array");
    assert!(hits.is_empty(), "fresh index should produce no hits, got {hits:?}");
    assert_eq!(ok["query_count"].as_u64(), Some(1));
    assert_eq!(ok["hybrid"], true);

    drop(shutdown);
    let _ = handle.await;
}

#[tokio::test]
async fn malformed_request_returns_bad_request_error() {
    let (state, dir) = build_state();
    let sock = dir.path().join("recall.sock");

    let (shutdown, handle) = daemon::spawn_for_test(state, sock.clone()).await.unwrap();

    let resp = daemon::client_roundtrip(&sock, &json!({"not_a_request": true}))
        .await
        .unwrap();

    let err = &resp["error"];
    assert_eq!(err["code"], "bad_request", "got {resp}");

    drop(shutdown);
    let _ = handle.await;
}

#[tokio::test]
async fn embed_op_returns_unit_vector() {
    let (state, dir) = build_state();
    let sock = dir.path().join("recall.sock");

    let (shutdown, handle) = daemon::spawn_for_test(state, sock.clone()).await.unwrap();

    let resp = daemon::client_roundtrip(
        &sock,
        &json!({"op": "embed", "args": {"text": "hello daemon"}}),
    )
    .await
    .unwrap();

    assert!(resp.get("ok").is_some(), "expected ok, got {resp}");
    let ok = &resp["ok"];
    let dim = ok["dim"].as_u64().expect("dim is u64");
    let vec: Vec<f32> = ok["vector"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(vec.len() as u64, dim);
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "norm was {norm}");

    drop(shutdown);
    let _ = handle.await;
}
