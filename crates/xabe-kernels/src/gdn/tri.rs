//! Explicit inverse of a unit lower-triangular matrix, via forward
//! substitution.
//!
//! The chunked (prefill) form of the delta rule needs `(I - A)^{-1}` (or,
//! in the sign convention this module derives independently in
//! `chunked.rs`'s doc comment, `(I + A)^{-1}`) where `A` is strictly lower
//! triangular — see `crates/xabe-kernels/src/gdn/chunked.rs` for the
//! derivation. `I ± A` is always unit lower triangular (ones on the
//! diagonal, since `A`'s diagonal is zero), which is exactly the case
//! forward substitution solves in closed form: row `i` of the inverse only
//! depends on rows `< i`, so a single top-to-bottom sweep suffices — no
//! pivoting, no iteration, and no possibility of the matrix being singular
//! (a unit lower-triangular matrix always has determinant 1).

/// Inverts an `n x n` unit lower-triangular matrix `m` (row-major, `m[i *
/// n + j]` for `j < i` is the strictly-lower part; the diagonal is assumed
/// to be exactly `1.0` and the strictly-upper part is assumed to be `0.0` —
/// both are ignored on input, not read).
///
/// Returns the inverse, also unit lower-triangular, row-major.
///
/// # Panics
/// If `m.len() != n * n`.
pub fn invert_unit_lower_triangular(m: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(
        m.len(),
        n * n,
        "invert_unit_lower_triangular: matrix size mismatch"
    );

    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
        for j in 0..i {
            // X[i][j] = -sum_{k=j}^{i-1} M[i][k] * X[k][j]
            // (X[i][i] = 1 handled above; X[i][j] = 0 for j > i, left as
            // the vec's zero-initialized default and never written.)
            let mut acc = 0.0f32;
            for k in j..i {
                acc += m[i * n + k] * inv[k * n + j];
            }
            inv[i * n + j] = -acc;
        }
    }
    inv
}

/// Multiplies two row-major matrices: `a` is `rows x inner`, `b` is `inner
/// x cols`, result is `rows x cols`. A small, obviously-correct triple loop
/// — this crate optimizes for readability, not throughput (see module docs
/// at the crate root).
pub fn matmul(a: &[f32], b: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(b.len(), inner * cols);
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for k in 0..inner {
            let a_ik = a[i * inner + k];
            if a_ik == 0.0 {
                continue;
            }
            for j in 0..cols {
                out[i * cols + j] += a_ik * b[k * cols + j];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{Tolerance, assert_matches};
    use crate::rng::Xorshift64Star;

    fn identity(n: usize) -> Vec<f32> {
        let mut id = vec![0.0f32; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        id
    }

    #[test]
    fn inverse_of_identity_is_identity() {
        let n = 5;
        let m = identity(n);
        let inv = invert_unit_lower_triangular(&m, n);
        assert_matches(&inv, &m, &Tolerance::tight_fp32());
    }

    #[test]
    fn product_of_matrix_and_its_inverse_is_identity() {
        let mut rng = Xorshift64Star::new(1);
        let n = 8;
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
            for j in 0..i {
                m[i * n + j] = rng.next_f32_range(-1.0, 1.0);
            }
        }
        let inv = invert_unit_lower_triangular(&m, n);
        let product = matmul(&m, &inv, n, n, n);
        assert_matches(&product, &identity(n), &Tolerance::tight_fp32());
    }

    #[test]
    fn inverse_of_a_2x2_matches_hand_computation() {
        // M = [[1, 0], [3, 1]]  ->  M^-1 = [[1, 0], [-3, 1]]
        let n = 2;
        let m = [1.0f32, 0.0, 3.0, 1.0];
        let inv = invert_unit_lower_triangular(&m, n);
        assert_matches(&inv, &[1.0, 0.0, -3.0, 1.0], &Tolerance::tight_fp32());
    }

    #[test]
    fn inverse_is_itself_lower_triangular() {
        let mut rng = Xorshift64Star::new(2);
        let n = 6;
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
            for j in 0..i {
                m[i * n + j] = rng.next_f32_range(-1.0, 1.0);
            }
        }
        let inv = invert_unit_lower_triangular(&m, n);
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(
                    inv[i * n + j],
                    0.0,
                    "strictly-upper entry [{i}][{j}] must be zero"
                );
            }
        }
    }

    #[test]
    fn matmul_matches_naive_triple_loop_on_random_matrices() {
        let mut rng = Xorshift64Star::new(3);
        let (rows, inner, cols) = (4, 5, 3);
        let a = rng.vec_f32(rows * inner, -2.0, 2.0);
        let b = rng.vec_f32(inner * cols, -2.0, 2.0);
        let out = matmul(&a, &b, rows, inner, cols);

        let mut naive = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut acc = 0.0f32;
                for k in 0..inner {
                    acc += a[i * inner + k] * b[k * cols + j];
                }
                naive[i * cols + j] = acc;
            }
        }
        assert_matches(&out, &naive, &Tolerance::tight_fp32());
    }
}
