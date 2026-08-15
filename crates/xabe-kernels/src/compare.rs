//! Differential-testing harness: compare a candidate tensor against a
//! reference tensor and report *why* they differ, not just *whether*.
//!
//! Cosine similarity alone hides magnitude errors (a tensor scaled by 0.5 in
//! every element is "perfectly similar" by cosine and badly wrong). Max-abs
//! error alone hides direction errors on small-magnitude tensors. Reporting
//! both, plus max-relative error and a NaN/Inf count, plus the *location* of
//! the worst element, is what makes a failing test actionable instead of a
//! bare "assertion failed".

use core::fmt;

/// Per-tensor comparison metrics between a candidate and a reference.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonResult {
    /// Largest `|candidate - reference|` over all elements.
    pub max_abs_error: f32,
    /// Flat index of the element with the largest absolute error.
    pub max_abs_error_index: usize,
    /// Largest `|candidate - reference| / max(|reference|, eps)` over all
    /// elements. `eps` guards near-zero reference values from producing a
    /// meaningless blown-up ratio.
    pub max_rel_error: f32,
    /// Flat index of the element with the largest relative error.
    pub max_rel_error_index: usize,
    /// Cosine similarity between the two tensors treated as flat vectors.
    /// `1.0` means identical direction, independent of magnitude.
    pub cosine_similarity: f32,
    /// Count of NaN or Inf elements in the candidate tensor.
    pub non_finite_count: usize,
    /// Number of elements compared.
    pub len: usize,
}

impl ComparisonResult {
    /// The reference value at the worst-absolute-error element, for
    /// building an actionable message. Returns `None` if `reference` is
    /// shorter than the recorded index (should not happen if the caller
    /// passed the same slices used to build this result).
    pub fn worst_abs_pair(&self, candidate: &[f32], reference: &[f32]) -> Option<(f32, f32)> {
        let i = self.max_abs_error_index;
        Some((*candidate.get(i)?, *reference.get(i)?))
    }
}

impl fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cosine={:.6} max_abs={:.6e} @[{}] max_rel={:.6e} @[{}] non_finite={}/{}",
            self.cosine_similarity,
            self.max_abs_error,
            self.max_abs_error_index,
            self.max_rel_error,
            self.max_rel_error_index,
            self.non_finite_count,
            self.len,
        )
    }
}

/// Computes [`ComparisonResult`] for two equal-length flat tensors.
///
/// # Panics
/// If `candidate.len() != reference.len()`, or either is empty — comparing
/// zero elements is a test-construction bug, not a "pass".
pub fn compare(candidate: &[f32], reference: &[f32]) -> ComparisonResult {
    assert_eq!(
        candidate.len(),
        reference.len(),
        "compare() requires equal-length tensors"
    );
    assert!(
        !candidate.is_empty(),
        "compare() requires non-empty tensors"
    );

    const REL_EPS: f32 = 1e-6;

    let mut max_abs_error = 0.0f32;
    let mut max_abs_error_index = 0usize;
    let mut max_rel_error = 0.0f32;
    let mut max_rel_error_index = 0usize;
    let mut non_finite_count = 0usize;

    let mut dot = 0.0f64;
    let mut norm_c = 0.0f64;
    let mut norm_r = 0.0f64;

    for (i, (&c, &r)) in candidate.iter().zip(reference.iter()).enumerate() {
        if !c.is_finite() {
            non_finite_count += 1;
            continue;
        }

        let abs_err = (c - r).abs();
        if abs_err > max_abs_error {
            max_abs_error = abs_err;
            max_abs_error_index = i;
        }

        let rel_err = abs_err / r.abs().max(REL_EPS);
        if rel_err > max_rel_error {
            max_rel_error = rel_err;
            max_rel_error_index = i;
        }

        dot += f64::from(c) * f64::from(r);
        norm_c += f64::from(c) * f64::from(c);
        norm_r += f64::from(r) * f64::from(r);
    }

    let denom = norm_c.sqrt() * norm_r.sqrt();
    let cosine_similarity = if denom > 0.0 {
        (dot / denom) as f32
    } else {
        // Both vectors are exactly zero (or the candidate is all non-finite):
        // direction is undefined. Report 1.0 only when both are truly zero.
        if norm_c == 0.0 && norm_r == 0.0 {
            1.0
        } else {
            0.0
        }
    };

    ComparisonResult {
        max_abs_error,
        max_abs_error_index,
        max_rel_error,
        max_rel_error_index,
        cosine_similarity,
        non_finite_count,
        len: candidate.len(),
    }
}

/// A named error-tolerance preset with a documented rationale.
///
/// Every threshold here is a claim about what "close enough" means for a
/// specific numeric situation, not a universal constant — a threshold
/// copied into the wrong context (e.g. using [`Self::exact`] to check a
/// GPU fp16 kernel) is a silent way to make a broken kernel look correct.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    pub max_abs_error: f32,
    pub max_rel_error: f32,
    pub min_cosine_similarity: f32,
    pub allow_non_finite: bool,
}

impl Tolerance {
    /// Two fp32 scalar implementations of the *same* formula, differing
    /// only in operation order (e.g. a naive loop vs. a chunked
    /// reformulation). fp32 summation is not associative, so exact equality
    /// is the wrong bar; this tolerance is tight enough to catch a real
    /// formulation bug while absorbing float reassociation noise.
    pub const fn tight_fp32() -> Self {
        Self {
            max_abs_error: 1e-3,
            max_rel_error: 1e-3,
            min_cosine_similarity: 1.0 - 1e-5,
            allow_non_finite: false,
        }
    }

    /// The project's stated gate for Gated DeltaNet recurrent-vs-chunked
    /// equivalence: cosine within `1e-3` over long sequences. See
    /// `AGENTS.md` / the milestone plan. Kept separate from
    /// [`Self::tight_fp32`] so the two call sites can diverge later without
    /// one silently loosening the other.
    pub const fn gdn_chunk_vs_recurrent() -> Self {
        Self {
            max_abs_error: 5e-2,
            max_rel_error: 5e-2,
            min_cosine_similarity: 1.0 - 1e-3,
            allow_non_finite: false,
        }
    }

    /// A future GPU (fp16/bf16 accumulation) kernel checked against an fp32
    /// CPU reference. Loose enough to absorb reduced-precision arithmetic;
    /// still tight enough to catch a wrong kernel outright. Not exercised by
    /// this crate today — no GPU kernel exists yet — but declared here so
    /// the threshold is decided once, in one place, before anyone is
    /// tempted to invent a number under deadline pressure.
    pub const fn reduced_precision_gpu() -> Self {
        Self {
            max_abs_error: 5e-2,
            max_rel_error: 5e-2,
            min_cosine_similarity: 1.0 - 1e-3,
            allow_non_finite: false,
        }
    }

    /// Bit-for-bit: used for operations with no floating-point rounding
    /// freedom, e.g. the untouched tail of a partial-rotary embedding, which
    /// must pass through unmodified rather than merely "close".
    pub const fn exact() -> Self {
        Self {
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            min_cosine_similarity: 1.0,
            allow_non_finite: false,
        }
    }
}

/// The outcome of checking a [`ComparisonResult`] against a [`Tolerance`].
#[derive(Debug, Clone, PartialEq)]
pub enum ToleranceCheck {
    Pass,
    Fail {
        reason: String,
        result: ComparisonResult,
    },
}

impl ToleranceCheck {
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass)
    }
}

/// Checks `result` against `tolerance`, returning a description of the
/// first violated bound rather than a bare boolean.
pub fn check(result: &ComparisonResult, tolerance: &Tolerance) -> ToleranceCheck {
    if !tolerance.allow_non_finite && result.non_finite_count > 0 {
        return ToleranceCheck::Fail {
            reason: format!(
                "{} of {} elements are NaN/Inf",
                result.non_finite_count, result.len
            ),
            result: result.clone(),
        };
    }
    if result.max_abs_error > tolerance.max_abs_error {
        return ToleranceCheck::Fail {
            reason: format!(
                "max_abs_error {:.6e} exceeds tolerance {:.6e} at index {}",
                result.max_abs_error, tolerance.max_abs_error, result.max_abs_error_index
            ),
            result: result.clone(),
        };
    }
    if result.max_rel_error > tolerance.max_rel_error {
        return ToleranceCheck::Fail {
            reason: format!(
                "max_rel_error {:.6e} exceeds tolerance {:.6e} at index {}",
                result.max_rel_error, tolerance.max_rel_error, result.max_rel_error_index
            ),
            result: result.clone(),
        };
    }
    if result.cosine_similarity < tolerance.min_cosine_similarity {
        return ToleranceCheck::Fail {
            reason: format!(
                "cosine_similarity {:.6} below minimum {:.6}",
                result.cosine_similarity, tolerance.min_cosine_similarity
            ),
            result: result.clone(),
        };
    }
    ToleranceCheck::Pass
}

/// Runs a reference implementation and a candidate implementation on the
/// same input, compares them, and panics with an actionable message if they
/// disagree beyond `tolerance`. This is the API a future GPU differential
/// test calls: `assert_matches(reference_fn(x), candidate_fn(x),
/// &Tolerance::reduced_precision_gpu())`.
pub fn assert_matches(candidate: &[f32], reference: &[f32], tolerance: &Tolerance) {
    let result = compare(candidate, reference);
    if let ToleranceCheck::Fail { reason, result } = check(&result, tolerance) {
        let worst = result.worst_abs_pair(candidate, reference);
        panic!(
            "differential comparison failed: {reason}\n  full metrics: {result}\n  worst element: {worst:?} (candidate, reference)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_tensors_compare_perfectly() {
        let v = [1.0f32, 2.0, -3.0, 0.5];
        let result = compare(&v, &v);
        assert_eq!(result.max_abs_error, 0.0);
        assert_eq!(result.max_rel_error, 0.0);
        assert!((result.cosine_similarity - 1.0).abs() < 1e-6);
        assert_eq!(result.non_finite_count, 0);
    }

    #[test]
    fn cosine_alone_hides_a_magnitude_error_but_max_abs_catches_it() {
        let reference = [1.0f32, 2.0, 3.0, 4.0];
        let candidate: Vec<f32> = reference.iter().map(|x| x * 0.5).collect();
        let result = compare(&candidate, &reference);
        assert!(
            result.cosine_similarity > 0.999,
            "same direction should score high on cosine"
        );
        assert!(
            result.max_abs_error > 1.0,
            "but the magnitude is badly wrong: {result}"
        );
    }

    #[test]
    fn max_abs_alone_hides_a_direction_error_but_cosine_catches_it() {
        let reference = [1.0f32, 0.0, 0.0];
        let candidate = [0.0f32, 1.0, 0.0];
        let result = compare(&candidate, &reference);
        assert!(result.max_abs_error <= 1.0);
        assert!(
            result.cosine_similarity < 0.5,
            "orthogonal vectors must score low on cosine: {result}"
        );
    }

    #[test]
    fn worst_element_is_identified_by_index() {
        let reference = [0.0f32, 0.0, 0.42, 0.0];
        let candidate = [0.0f32, 0.0, 0.31, 0.0];
        let result = compare(&candidate, &reference);
        assert_eq!(result.max_abs_error_index, 2);
        let (c, r) = result.worst_abs_pair(&candidate, &reference).unwrap();
        assert!((c - 0.31).abs() < 1e-6);
        assert!((r - 0.42).abs() < 1e-6);
    }

    #[test]
    fn nan_in_candidate_is_counted_and_excluded_from_error_stats() {
        let reference = [1.0f32, 2.0, 3.0];
        let candidate = [1.0f32, f32::NAN, 3.0];
        let result = compare(&candidate, &reference);
        assert_eq!(result.non_finite_count, 1);
        // The finite elements still compare exactly.
        assert_eq!(result.max_abs_error, 0.0);
    }

    #[test]
    fn tolerance_check_reports_the_first_violated_bound() {
        let result = compare(&[1.5f32], &[1.0f32]);
        let tol = Tolerance {
            max_abs_error: 1.0,
            max_rel_error: 0.1,
            min_cosine_similarity: 0.9,
            allow_non_finite: false,
        };
        let outcome = check(&result, &tol);
        match outcome {
            ToleranceCheck::Fail { reason, .. } => assert!(reason.contains("max_rel_error")),
            ToleranceCheck::Pass => panic!("expected a tolerance failure"),
        }
    }

    #[test]
    fn non_finite_fails_unless_explicitly_allowed() {
        let result = compare(&[f32::NAN], &[1.0f32]);
        let strict = Tolerance {
            allow_non_finite: false,
            ..Tolerance::tight_fp32()
        };
        assert!(!check(&result, &strict).is_pass());

        let lenient = Tolerance {
            allow_non_finite: true,
            ..Tolerance::tight_fp32()
        };
        assert!(check(&result, &lenient).is_pass());
    }

    #[test]
    #[should_panic(expected = "differential comparison failed")]
    fn assert_matches_panics_with_an_actionable_message() {
        assert_matches(&[10.0], &[0.0], &Tolerance::exact());
    }

    #[test]
    fn assert_matches_passes_within_tolerance() {
        assert_matches(&[1.0000001], &[1.0], &Tolerance::tight_fp32());
    }

    #[test]
    #[should_panic]
    fn compare_panics_on_length_mismatch() {
        compare(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn two_zero_vectors_are_perfectly_similar_by_convention() {
        let result = compare(&[0.0f32, 0.0], &[0.0f32, 0.0]);
        assert_eq!(result.cosine_similarity, 1.0);
    }
}
