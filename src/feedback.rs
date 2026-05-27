//! Outcome-feedback math (PRD-recall-outcome-feedback, codename *weather*).
//!
//! Pure functions: clamp/decay logic lives here so it's testable without
//! touching SQLite or the filesystem. Storage-side updates live in
//! `index::apply_feedback_delta` and `index::apply_decay_sweep`.

use crate::config::Feedback;

/// Apply `delta` to `confidence`, clamped to `[floor, ceiling]`.
///
/// Accept = positive delta, reject = negative delta. The caller knows
/// which it is; this function is sign-agnostic.
pub fn apply_delta(confidence: f64, delta: f64, cfg: &Feedback) -> f64 {
    (confidence + delta).clamp(cfg.floor, cfg.ceiling)
}

/// Decay `confidence` toward 0.5 by `days` of elapsed time, with the
/// configured half-life. `confidence' = 0.5 + (confidence - 0.5) * 2^(-days/H)`.
///
/// Memories already at 0.5 stay at 0.5. A memory at 0.95 with a 90-day
/// half-life decays to ~0.725 after 90 days, ~0.6125 after 180 days.
pub fn decay_toward_neutral(confidence: f64, days: f64, half_life_d: u32) -> f64 {
    if half_life_d == 0 || days <= 0.0 {
        return confidence;
    }
    let h = f64::from(half_life_d);
    let factor = 2f64.powf(-days / h);
    0.5 + (confidence - 0.5) * factor
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn defaults() -> Feedback {
        Feedback::default()
    }

    #[test]
    fn accept_bumps_by_delta() {
        let cfg = defaults();
        let after = apply_delta(0.5, cfg.accept_delta, &cfg);
        assert!((after - 0.52).abs() < 1e-9);
    }

    #[test]
    fn accept_caps_at_ceiling() {
        let cfg = defaults();
        let after = apply_delta(0.95, cfg.accept_delta, &cfg);
        assert!((after - cfg.ceiling).abs() < 1e-9);
    }

    #[test]
    fn reject_decays_by_delta() {
        let cfg = defaults();
        let after = apply_delta(0.5, -cfg.reject_delta, &cfg);
        assert!((after - 0.4).abs() < 1e-9);
    }

    #[test]
    fn reject_floors_at_floor() {
        let cfg = defaults();
        let after = apply_delta(0.10, -cfg.reject_delta, &cfg);
        assert!((after - cfg.floor).abs() < 1e-9);
    }

    #[test]
    fn decay_half_life_matches_formula() {
        // 0.95 with H=90, days=90 → 0.5 + 0.45 * 0.5 = 0.725
        let out = decay_toward_neutral(0.95, 90.0, 90);
        assert!((out - 0.725).abs() < 0.01, "expected ~0.725, got {out}");
    }

    #[test]
    fn decay_two_half_lives() {
        // 0.95, H=90, days=180 → 0.5 + 0.45 * 0.25 = 0.6125
        let out = decay_toward_neutral(0.95, 180.0, 90);
        assert!((out - 0.6125).abs() < 0.01, "expected ~0.6125, got {out}");
    }

    #[test]
    fn decay_neutral_stays_neutral() {
        let out = decay_toward_neutral(0.5, 365.0, 90);
        assert!((out - 0.5).abs() < 1e-9);
    }

    #[test]
    fn decay_zero_days_noop() {
        let out = decay_toward_neutral(0.9, 0.0, 90);
        assert!((out - 0.9).abs() < 1e-9);
    }
}
