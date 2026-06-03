//! Integration tests for PRD-recall-doctor-utility.
//!
//! AC1: `recall doctor --format json` includes `utility` section.
//! AC2: low_utility_high_surface membership.
//! AC3: high_utility_validated membership.
//! AC4: calibration_drift formula.
//! AC5: Sorting.
//! AC6: Text format prints both blocks.
//! AC7: Empty/cold-start store returns empty buckets.
//! AC8: JSON section serializes with `low_utility_high_surface | length` queryable.

use recall::doctor_utility::{BUCKET_LIMIT, HIGH_RATIO, LOW_RATIO, SURFACE_MIN, compute_utility_report};
use recall::index::Index;
use recall::memory::{Kind, Memory, Subject};
use rusqlite::Connection;

/// Helper: open test index.
fn open_index(dir: &std::path::Path) -> (Index, std::path::PathBuf) {
    let db = dir.join("idx.sqlite");
    let idx = Index::open(&db).unwrap();
    (idx, db)
}

/// Helper: insert a memory and set surfaced/used counts directly.
fn insert_with_counts(
    idx: &Index,
    db: &std::path::Path,
    label: &str,
    surfaced: u32,
    used: u32,
    confidence: f64,
) -> String {
    let mem = Memory::new(Kind::Semantic, Subject::self_(), label);
    let path = db.parent().unwrap().join(format!("{label}.md"));
    idx.upsert(&mem, &path, None).unwrap();
    let conn = Connection::open(db).unwrap();
    conn.execute(
        "UPDATE memories_meta SET surfaced_count = ?1, used_count = ?2, confidence = ?3 WHERE id = ?4",
        rusqlite::params![surfaced, used, confidence, mem.front.id],
    )
    .unwrap();
    mem.front.id
}

// ── AC1: utility section present on non-empty store ──────────────────────────

#[test]
fn ac1_utility_section_present() {
    let tmp = tempfile::tempdir().unwrap();
    let (idx, db) = open_index(tmp.path());
    insert_with_counts(&idx, &db, "x", 8, 1, 0.65);

    let report = compute_utility_report(&db).unwrap();
    // total_memories and with_surface_data must be populated.
    assert!(report.total_memories >= 1, "AC1: total_memories present");
    assert!(report.with_surface_data >= 1, "AC1: with_surface_data present (surf>=5)");
}

// ── AC2: low bucket membership ────────────────────────────────────────────────

#[test]
fn ac2_low_bucket_membership() {
    let tmp = tempfile::tempdir().unwrap();
    let (idx, db) = open_index(tmp.path());

    // surf=10 used=0  ratio=0.00 => low
    let id_a = insert_with_counts(&idx, &db, "a", 10, 0, 0.7);
    // surf=10 used=1  ratio=0.10 => low
    let id_b = insert_with_counts(&idx, &db, "b", 10, 1, 0.7);
    // surf=4  used=0  => excluded (cold-start)
    let _id_c = insert_with_counts(&idx, &db, "c", 4, 0, 0.7);
    // surf=10 used=8  ratio=0.80 => high, NOT low
    let id_d = insert_with_counts(&idx, &db, "d", 10, 8, 0.7);

    let report = compute_utility_report(&db).unwrap();
    let low_ids: Vec<&str> = report.low_utility_high_surface.iter().map(|m| m.id.as_str()).collect();

    assert!(low_ids.contains(&id_a.as_str()), "AC2: ratio=0.0 should be low");
    assert!(low_ids.contains(&id_b.as_str()), "AC2: ratio=0.1 should be low");
    assert!(!low_ids.contains(&id_d.as_str()), "AC2: ratio=0.8 should NOT be low");
}

// ── AC3: high bucket membership ───────────────────────────────────────────────

#[test]
fn ac3_high_bucket_membership() {
    let tmp = tempfile::tempdir().unwrap();
    let (idx, db) = open_index(tmp.path());

    let _id_a = insert_with_counts(&idx, &db, "a", 10, 0, 0.7);
    let id_d = insert_with_counts(&idx, &db, "d", 10, 8, 0.7);  // ratio=0.8 => high

    let report = compute_utility_report(&db).unwrap();
    let high_ids: Vec<&str> = report.high_utility_validated.iter().map(|m| m.id.as_str()).collect();

    assert!(high_ids.contains(&id_d.as_str()), "AC3: ratio=0.8 should be high");
    assert!(!high_ids.contains(&_id_a.as_str()), "AC3: ratio=0.0 should NOT be high");
}

// ── AC4: calibration_drift formula ────────────────────────────────────────────

#[test]
fn ac4_calibration_drift_formula() {
    // drift = confidence - (0.5 + ratio * 0.5)
    // surf=10, used=0 => ratio=0.0, conf=0.75 => drift=0.75-(0.5+0.0)=0.25
    let tmp = tempfile::tempdir().unwrap();
    let (idx, db) = open_index(tmp.path());

    let id = insert_with_counts(&idx, &db, "z", 10, 0, 0.75);

    let report = compute_utility_report(&db).unwrap();
    let mem = report.low_utility_high_surface.iter().find(|m| m.id == id).unwrap();

    let expected_drift = 0.75_f32 - (0.5_f32 + 0.0_f32 * 0.5_f32); // 0.25
    assert!(
        (mem.calibration_drift - expected_drift).abs() < 1e-4,
        "AC4: drift={:.4} expected={:.4}",
        mem.calibration_drift,
        expected_drift
    );
}

// ── AC5: low bucket sorted descending by surfaced_count ───────────────────────

#[test]
fn ac5_low_bucket_sorted_desc_by_surfaced() {
    let tmp = tempfile::tempdir().unwrap();
    let (idx, db) = open_index(tmp.path());

    // 12 memories all low-utility, distinct surfaced counts 10..21.
    for i in 0..12u32 {
        insert_with_counts(&idx, &db, &format!("m{i:02}"), 10 + i, 0, 0.7);
    }

    let report = compute_utility_report(&db).unwrap();
    assert_eq!(report.low_utility_high_surface.len(), BUCKET_LIMIT, "AC5: capped at {BUCKET_LIMIT}");

    let surfs: Vec<u32> = report.low_utility_high_surface.iter().map(|m| m.surfaced).collect();
    let mut sorted_desc = surfs.clone();
    sorted_desc.sort_by(|a, b| b.cmp(a));
    assert_eq!(surfs, sorted_desc, "AC5: low bucket must be sorted descending by surfaced_count");
}

// ── AC7: empty/cold-start store ───────────────────────────────────────────────

#[test]
fn ac7_cold_start_store() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, db) = open_index(tmp.path());

    let report = compute_utility_report(&db).unwrap();
    assert!(report.low_utility_high_surface.is_empty(), "AC7: low bucket empty on fresh store");
    assert!(report.high_utility_validated.is_empty(), "AC7: high bucket empty on fresh store");
}

// ── AC8: JSON section queryable (simulates `jq '.utility.low_utility_high_surface | length'`) ──

#[test]
fn ac8_json_low_utility_length_queryable() {
    let tmp = tempfile::tempdir().unwrap();
    let (idx, db) = open_index(tmp.path());
    insert_with_counts(&idx, &db, "p", 8, 1, 0.65);

    let report = compute_utility_report(&db).unwrap();
    let json = serde_json::to_string(&report).unwrap();
    let val: serde_json::Value = serde_json::from_str(&json).unwrap();

    // Simulate `jq '.low_utility_high_surface | length'`
    let arr = val
        .get("low_utility_high_surface")
        .and_then(|v| v.as_array())
        .expect("AC8: low_utility_high_surface must be present and an array");
    let _length: usize = arr.len(); // returns an integer without panicking
}

// ── Thresholds are exposed as constants ──────────────────────────────────────

#[test]
fn constants_have_expected_values() {
    assert_eq!(SURFACE_MIN, 5, "cold-start cutoff");
    assert!((LOW_RATIO - 0.2_f32).abs() < 1e-6, "low threshold");
    assert!((HIGH_RATIO - 0.7_f32).abs() < 1e-6, "high threshold");
    assert_eq!(BUCKET_LIMIT, 10, "bucket size");
}
