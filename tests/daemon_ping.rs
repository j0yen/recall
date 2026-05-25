//! End-to-end daemon test: spawn the UDS server, send a `ping`, assert
//! the response shape. Covers part of PRD-recall-daemon AC-1 (cold start
//! → responsive) and the framing/protocol contract.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use recall::daemon;
use serde_json::json;

#[tokio::test]
async fn ping_roundtrip_returns_uptime_and_model_id() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("recall.sock");

    let (shutdown, handle) = daemon::spawn_for_test(sock.clone()).await.unwrap();

    let resp = daemon::client_roundtrip(&sock, &json!({"op": "ping", "args": {}}))
        .await
        .unwrap();

    assert!(resp.get("ok").is_some(), "expected ok response, got {resp}");
    let ok = &resp["ok"];
    assert_eq!(ok["model_id"], daemon::DEFAULT_MODEL_ID);
    assert!(ok["uptime_s"].as_u64().is_some());
    assert!(ok["version"].as_str().is_some());

    drop(shutdown);
    let _ = handle.await;
    assert!(!sock.exists(), "socket should be unlinked after shutdown");
}

#[tokio::test]
async fn query_op_returns_not_implemented_in_iter2() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("recall.sock");

    let (shutdown, handle) = daemon::spawn_for_test(sock.clone()).await.unwrap();

    let resp = daemon::client_roundtrip(&sock, &json!({"op": "query", "args": {"text": "x"}}))
        .await
        .unwrap();

    let err = &resp["error"];
    assert_eq!(err["code"], "not_implemented", "got {resp}");

    drop(shutdown);
    let _ = handle.await;
}

#[tokio::test]
async fn malformed_request_returns_bad_request_error() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("recall.sock");

    let (shutdown, handle) = daemon::spawn_for_test(sock.clone()).await.unwrap();

    let resp = daemon::client_roundtrip(&sock, &json!({"not_a_request": true}))
        .await
        .unwrap();

    let err = &resp["error"];
    assert_eq!(err["code"], "bad_request", "got {resp}");

    drop(shutdown);
    let _ = handle.await;
}
