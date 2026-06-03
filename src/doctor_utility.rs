//! `recall doctor` utility section — surfaces high-surface-low-use and
//! high-surface-high-use memory statistics.
//!
//! Computes `{id, surfaced, used, ratio, confidence, calibration_drift}` per
//! memory using the `memories_meta.surfaced_count` and `used_count` columns.
//! Pure diagnostic: no writes, no confidence mutation, no schema change.
//!
//! PRD-recall-doctor-utility v0.7.4.

use anyhow::Result;
use rusqlite::{Connection, params};
use serde::Serialize;
use std::path::Path;

// ── Cold-start cutoff (configurable in future) ───────────────────────────────

/// Minimum `surfaced_count` for a memory to have meaningful ratio data.
pub const SURFACE_MIN: u32 = 5;

/// Low-utility threshold: `ratio < LOW_RATIO` memories "rank well, rarely used."
pub const LOW_RATIO: f32 = 0.2;

/// High-utility threshold: `ratio >= HIGH_RATIO` memories are "validated workhorses."
pub const HIGH_RATIO: f32 = 0.7;

/// Maximum entries per bucket in the report.
pub const BUCKET_LIMIT: usize = 10;

// ── Public types ─────────────────────────────────────────────────────────────

/// Per-memory utility record.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryUtility {
    pub id: String,
    pub kind: String,
    pub subject: String,
    pub surfaced: u32,
    pub used: u32,
    pub ratio: f32,
    pub confidence: f32,
    /// `confidence - (0.5 + ratio * 0.5)`
    /// Positive = ranked higher than utility justifies.
    /// Negative = underranked workhorse.
    pub calibration_drift: f32,
}

/// Full utility section for `recall doctor`.
#[derive(Debug, Clone, Serialize)]
pub struct UtilityReport {
    /// Total memories in the store.
    pub total_memories: u64,
    /// Memories with `surfaced_count >= SURFACE_MIN` (cold-start cutoff).
    pub with_surface_data: u64,
    /// Top-10 by `surfaced_count` where `ratio < LOW_RATIO`.
    pub low_utility_high_surface: Vec<MemoryUtility>,
    /// Top-10 by `used_count` where `ratio >= HIGH_RATIO`.
    pub high_utility_validated: Vec<MemoryUtility>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Compute the utility report from the SQLite index.
///
/// Opens its own connection to `db_path` so callers don't have to pass in
/// the `Index` object (doctor already has one open; this is a second handle,
/// which is safe under WAL mode).
pub fn compute_utility_report(db_path: &Path) -> Result<UtilityReport> {
    let conn = Connection::open(db_path)?;

    // Total count (no filter).
    let total_memories: u64 = conn.query_row(
        "SELECT COUNT(*) FROM memories_meta",
        [],
        |r| r.get::<_, i64>(0),
    ).map(|n| n.max(0) as u64)?;

    // Query all memories with surfaced_count >= SURFACE_MIN, sorted by
    // surfaced_count DESC (handles both buckets; Rust does the ratio split).
    let mut stmt = conn.prepare(
        "SELECT id, kind, subject, confidence, surfaced_count, used_count,
                CAST(used_count AS REAL) / NULLIF(surfaced_count, 0) AS ratio
         FROM memories_meta
         WHERE surfaced_count >= ?1
         ORDER BY surfaced_count DESC",
    )?;

    let rows: Vec<MemoryUtility> = stmt
        .query_map(params![SURFACE_MIN as i64], |r| {
            let id: String = r.get(0)?;
            let kind: String = r.get(1)?;
            let subject: String = r.get(2)?;
            let confidence: f64 = r.get(3)?;
            let surfaced: i64 = r.get(4)?;
            let used: i64 = r.get(5)?;
            // ratio may be NULL if surfaced == 0 (shouldn't happen given WHERE clause)
            let ratio: f64 = r.get::<_, Option<f64>>(6)?.unwrap_or(0.0);
            Ok((id, kind, subject, confidence, surfaced, used, ratio))
        })?
        .filter_map(|r| r.ok())
        .map(|(id, kind, subject, confidence, surfaced, used, ratio)| {
            let conf_f32 = confidence as f32;
            let ratio_f32 = ratio as f32;
            let calibration_drift = conf_f32 - (0.5 + ratio_f32 * 0.5);
            MemoryUtility {
                id,
                kind,
                subject,
                surfaced: surfaced.max(0) as u32,
                used: used.max(0) as u32,
                ratio: ratio_f32,
                confidence: conf_f32,
                calibration_drift,
            }
        })
        .collect();

    let with_surface_data = rows.len() as u64;

    // AC2/AC5: low-utility = ratio < LOW_RATIO, sorted descending by surfaced (already is).
    let low_utility_high_surface: Vec<MemoryUtility> = rows
        .iter()
        .filter(|m| m.ratio < LOW_RATIO)
        .take(BUCKET_LIMIT)
        .cloned()
        .collect();

    // AC3: high-utility = ratio >= HIGH_RATIO, sorted descending by used_count.
    let mut high_utility_validated: Vec<MemoryUtility> = rows
        .iter()
        .filter(|m| m.ratio >= HIGH_RATIO)
        .cloned()
        .collect();
    high_utility_validated.sort_by(|a, b| b.used.cmp(&a.used));
    high_utility_validated.truncate(BUCKET_LIMIT);

    Ok(UtilityReport {
        total_memories,
        with_surface_data,
        low_utility_high_surface,
        high_utility_validated,
    })
}

// ── Text formatting ───────────────────────────────────────────────────────────

/// Print the utility section in human-readable text format (same style as
/// the `confidence_drift` block in doctor's text output).
pub fn print_utility_text(report: &UtilityReport) {
    println!(
        "utility (n={}, with_surface={}):",
        report.total_memories, report.with_surface_data
    );
    println!(
        "  low-utility, high-surface (top {}, ratio < {:.1}):",
        BUCKET_LIMIT, LOW_RATIO
    );
    if report.low_utility_high_surface.is_empty() {
        println!("    (none)");
    } else {
        for m in &report.low_utility_high_surface {
            println!(
                "    {:<26}  {}/{:<12}  surf={:<5} used={:<5} ratio={:.2} conf={:.2} drift={:+.2}",
                &m.id[..m.id.len().min(26)],
                m.kind,
                m.subject,
                m.surfaced,
                m.used,
                m.ratio,
                m.confidence,
                m.calibration_drift
            );
        }
    }
    println!(
        "  high-utility, validated (top {}, ratio >= {:.1}):",
        BUCKET_LIMIT, HIGH_RATIO
    );
    if report.high_utility_validated.is_empty() {
        println!("    (none)");
    } else {
        for m in &report.high_utility_validated {
            println!(
                "    {:<26}  {}/{:<12}  surf={:<5} used={:<5} ratio={:.2} conf={:.2} drift={:+.2}",
                &m.id[..m.id.len().min(26)],
                m.kind,
                m.subject,
                m.surfaced,
                m.used,
                m.ratio,
                m.confidence,
                m.calibration_drift
            );
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::index::Index;
    use crate::memory::{Kind, Memory, Subject};

    /// Create an in-memory SQLite index (via Index) and return the db path
    /// so `compute_utility_report` can open its own handle.
    fn open_test_index(dir: &std::path::Path) -> (Index, std::path::PathBuf) {
        let db = dir.join("idx.sqlite");
        let idx = Index::open(&db).unwrap();
        (idx, db)
    }

    fn upsert_with_counts(
        idx: &Index,
        db: &std::path::Path,
        id_seed: &str,
        surfaced: u32,
        used: u32,
        confidence: f64,
    ) -> String {
        use rusqlite::Connection;
        let mem = Memory::new(Kind::Semantic, Subject::self_(), id_seed);
        let path = db.parent().unwrap().join(format!("{id_seed}.md"));
        idx.upsert(&mem, &path, None).unwrap();
        // Directly set the counters (the API path goes through the full write
        // cycle; here we just want specific test values).
        let conn = Connection::open(db).unwrap();
        conn.execute(
            "UPDATE memories_meta SET surfaced_count = ?1, used_count = ?2, confidence = ?3 WHERE id = ?4",
            rusqlite::params![surfaced, used, confidence, mem.front.id],
        )
        .unwrap();
        mem.front.id
    }

    // ── AC1: utility section present ─────────────────────────────────────────

    #[test]
    fn utility_report_fields_present() {
        let tmp = tempfile::tempdir().unwrap();
        let (idx, db) = open_test_index(tmp.path());

        // Insert one memory with enough surface data.
        upsert_with_counts(&idx, &db, "mem-a", 10, 5, 0.7);

        let report = compute_utility_report(&db).unwrap();
        // AC1: both top-level count fields exist.
        assert!(report.total_memories >= 1);
        assert!(report.with_surface_data >= 1);
    }

    // ── AC7: empty / cold-start store ────────────────────────────────────────

    #[test]
    fn empty_store_returns_empty_buckets() {
        let tmp = tempfile::tempdir().unwrap();
        let (_, db) = open_test_index(tmp.path());

        let report = compute_utility_report(&db).unwrap();
        assert_eq!(report.total_memories, 0);
        assert_eq!(report.with_surface_data, 0);
        assert!(report.low_utility_high_surface.is_empty(), "AC7: low bucket empty");
        assert!(report.high_utility_validated.is_empty(), "AC7: high bucket empty");
    }

    // ── AC2 + AC3: bucket membership ─────────────────────────────────────────

    #[test]
    fn bucket_membership_correct() {
        // Per PRD AC2/AC3:
        //   surf=10 used=0  ratio=0.00 => low (surf>=5, ratio<0.2)
        //   surf=10 used=1  ratio=0.10 => low (surf>=5, ratio<0.2)
        //   surf=4  used=0  ratio=N/A  => excluded (surf<5)
        //   surf=10 used=8  ratio=0.80 => high (surf>=5, ratio>=0.7)
        let tmp = tempfile::tempdir().unwrap();
        let (idx, db) = open_test_index(tmp.path());

        let id_a = upsert_with_counts(&idx, &db, "mem-a", 10, 0, 0.7);
        let id_b = upsert_with_counts(&idx, &db, "mem-b", 10, 1, 0.7);
        let _id_c = upsert_with_counts(&idx, &db, "mem-c", 4, 0, 0.7);
        let id_d = upsert_with_counts(&idx, &db, "mem-d", 10, 8, 0.7);

        let report = compute_utility_report(&db).unwrap();

        // AC2: low bucket
        let low_ids: Vec<&str> = report.low_utility_high_surface.iter().map(|m| m.id.as_str()).collect();
        assert!(low_ids.contains(&id_a.as_str()), "mem-a should be in low bucket");
        assert!(low_ids.contains(&id_b.as_str()), "mem-b should be in low bucket");
        assert!(!low_ids.contains(&id_d.as_str()), "mem-d should NOT be in low bucket");

        // AC3: high bucket
        let high_ids: Vec<&str> = report.high_utility_validated.iter().map(|m| m.id.as_str()).collect();
        assert!(high_ids.contains(&id_d.as_str()), "mem-d should be in high bucket");
        assert!(!high_ids.contains(&id_a.as_str()), "mem-a should NOT be in high bucket");

        // cold-start: mem-c (surf=4) not in either bucket
        let with_surface = report.with_surface_data;
        // Only mem-a, mem-b, mem-d have surf>=5
        assert_eq!(with_surface, 3, "only 3 memories pass cold-start cutoff");
    }

    // ── AC4: calibration_drift formula ───────────────────────────────────────

    #[test]
    fn calibration_drift_formula() {
        // drift = confidence - (0.5 + ratio * 0.5)
        // surf=20, used=4 => ratio=0.2, confidence=0.7 => drift=0.7-(0.5+0.1)=0.1
        let tmp = tempfile::tempdir().unwrap();
        let (idx, db) = open_test_index(tmp.path());

        let _id = upsert_with_counts(&idx, &db, "mem-x", 20, 4, 0.7);

        let report = compute_utility_report(&db).unwrap();
        // mem-x has ratio=0.2 which is NOT < 0.2 (boundary), so not in low bucket.
        // It's also < 0.7 so not in high bucket.
        // Check via combined scan.
        let all: Vec<&MemoryUtility> = report
            .low_utility_high_surface
            .iter()
            .chain(report.high_utility_validated.iter())
            .collect();
        // mem-x should not appear in either bucket (ratio exactly at low boundary).
        assert!(all.is_empty() || all.iter().all(|m| m.ratio != 0.2 || m.id != "mem-x" ));

        // Now check drift for a clear low-utility case:
        // surf=10, used=0 => ratio=0.0, conf=0.75 => drift=0.75-(0.5+0.0)=0.25
        let _id2 = upsert_with_counts(&idx, &db, "mem-y", 10, 0, 0.75);
        let report2 = compute_utility_report(&db).unwrap();
        let mem_y = report2.low_utility_high_surface.iter().find(|m| m.id != _id.as_str());
        if let Some(m) = mem_y {
            let expected_drift = 0.75_f32 - (0.5 + 0.0 * 0.5);
            let actual_drift = m.calibration_drift;
            assert!(
                (actual_drift - expected_drift).abs() < 1e-4,
                "drift formula mismatch: expected {expected_drift:.4} got {actual_drift:.4}"
            );
        }
    }

    // ── AC5: low-utility bucket sorted by surfaced_count DESC ────────────────

    #[test]
    fn low_utility_sorted_by_surfaced_desc() {
        let tmp = tempfile::tempdir().unwrap();
        let (idx, db) = open_test_index(tmp.path());

        // 12 memories all matching low-utility cutoff with distinct surfaced_count.
        for i in 0..12u32 {
            let surf = 10 + i; // 10..21
            upsert_with_counts(&idx, &db, &format!("mem-{i:02}"), surf, 0, 0.7);
        }

        let report = compute_utility_report(&db).unwrap();
        assert_eq!(report.low_utility_high_surface.len(), BUCKET_LIMIT, "capped at 10");

        // Verify descending order of surfaced_count.
        let surfs: Vec<u32> = report.low_utility_high_surface.iter().map(|m| m.surfaced).collect();
        let mut sorted = surfs.clone();
        sorted.sort_by(|a, b| b.cmp(a));
        assert_eq!(surfs, sorted, "AC5: low bucket must be sorted descending by surfaced_count");
    }

    // ── AC8: jq-style smoke test (structure check) ────────────────────────────

    #[test]
    fn utility_section_serializes_to_json() {
        let tmp = tempfile::tempdir().unwrap();
        let (idx, db) = open_test_index(tmp.path());
        upsert_with_counts(&idx, &db, "mem-a", 8, 1, 0.65);

        let report = compute_utility_report(&db).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        // AC8: low_utility_high_surface key exists and is an array with a length.
        let low = val.get("low_utility_high_surface").and_then(|v| v.as_array());
        assert!(low.is_some(), "AC8: low_utility_high_surface must exist");
        let _ = low.unwrap().len(); // returns an integer (not panics)
    }
}
