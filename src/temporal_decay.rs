//! Temporal-decay business logic (PRD-recall-temporal-decay).
//!
//! Pure functions and types for time-based confidence decay. No I/O — fully
//! testable without a store. The index integration lives in
//! `index::temporal_decay_report`.

use crate::feedback::decay_toward_neutral;

/// One memory's decay projection or applied result.
#[derive(Debug, Clone)]
pub struct DecayCandidate {
    /// Memory id (ULID string).
    pub id: String,
    /// Memory kind (e.g. `"semantic"`).
    pub kind: String,
    /// Memory subject (e.g. `"user"`, `"self"`).
    pub subject: String,
    /// Confidence before decay.
    pub confidence_before: f64,
    /// Confidence after decay (projected or applied).
    pub confidence_after: f64,
    /// Days elapsed since the decay baseline (last sweep or creation).
    pub days_since_baseline: f64,
    /// Half-life in days used for this projection.
    pub half_life_d: u32,
    /// Whether the decay was actually written to the store.
    pub applied: bool,
}

impl DecayCandidate {
    /// Signed difference: `confidence_after - confidence_before`.
    pub fn delta(&self) -> f64 {
        self.confidence_after - self.confidence_before
    }
}

/// Project the post-decay confidence using the half-life formula.
///
/// Delegates to [`crate::feedback::decay_toward_neutral`] so that the formula
/// stays in exactly one place. AC6 verifies the two are identical.
pub fn project_decay(confidence: f64, days_since_baseline: f64, half_life_d: u32) -> f64 {
    decay_toward_neutral(confidence, days_since_baseline, half_life_d)
}

/// Return true if the decay delta is large enough to be worth writing.
///
/// `min_delta` is an absolute threshold on `|before - after|`. Values below it
/// (e.g. a memory already at 0.5 whose projected confidence is 0.5000001) are
/// skipped to avoid pointless I/O.
pub fn is_worth_updating(before: f64, after: f64, min_delta: f64) -> bool {
    (before - after).abs() >= min_delta
}

/// Summary statistics for a decay sweep run.
#[derive(Debug, Clone)]
pub struct SweepSummary {
    /// Number of memories examined.
    pub examined: usize,
    /// Number of memories that met the candidate criteria.
    pub candidates: usize,
    /// Number of memories actually written (0 in dry-run).
    pub applied: usize,
    /// The half-life used for this sweep.
    pub half_life_d: u32,
    /// The min-interval used for this sweep.
    pub min_interval_d: u32,
    /// Whether this was a dry run.
    pub dry_run: bool,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::feedback::decay_toward_neutral;

    /// AC6 — project_decay formula matches feedback::decay_toward_neutral exactly.
    #[test]
    fn project_decay_matches_feedback_formula() {
        let cases: &[(f64, f64, u32)] = &[
            (0.9, 90.0, 90),
            (0.7, 30.0, 60),
            (0.5, 365.0, 90), // neutral stays neutral
            (0.95, 1.0, 90),
            (0.1, 45.0, 30),
        ];
        for (conf, days, hl) in cases {
            let via_module = project_decay(*conf, *days, *hl);
            let via_feedback = decay_toward_neutral(*conf, *days, *hl);
            assert!(
                (via_module - via_feedback).abs() < 1e-15,
                "mismatch: project_decay({conf}, {days}, {hl}) = {via_module} != {via_feedback}"
            );
        }
    }

    /// AC3 — neutral memory (confidence=0.5) projects to exactly 0.5.
    #[test]
    fn neutral_memory_excluded_by_min_delta() {
        let after = project_decay(0.5, 100.0, 90);
        assert!(
            (after - 0.5_f64).abs() < 1e-12,
            "neutral memory should not decay; got {after}"
        );
        // With any min_delta > 0, a neutral memory is not worth updating.
        assert!(
            !is_worth_updating(0.5, after, 0.001),
            "neutral memory should not pass is_worth_updating with min_delta=0.001"
        );
    }

    /// is_worth_updating passes when delta exceeds threshold.
    #[test]
    fn worth_updating_passes_above_threshold() {
        assert!(is_worth_updating(0.9, 0.85, 0.001));
        assert!(is_worth_updating(0.9, 0.85, 0.04));
        assert!(!is_worth_updating(0.9, 0.8991, 0.001)); // delta=0.0009 < 0.001
    }

    /// DecayCandidate::delta returns signed difference.
    #[test]
    fn decay_candidate_delta() {
        let c = DecayCandidate {
            id: "test".into(),
            kind: "semantic".into(),
            subject: "user".into(),
            confidence_before: 0.9,
            confidence_after: 0.817,
            days_since_baseline: 30.0,
            half_life_d: 90,
            applied: false,
        };
        assert!((c.delta() - (0.817 - 0.9)).abs() < 1e-9);
    }
}
