//! Reference for the integer tensor-core matmul.
//!
//! Turing's `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` multiplies an
//! 8x16 int8 tile by a 16x8 int8 tile into an 8x8 int32 accumulator. It is the
//! only tensor-core shape that matters for this project, because the model's
//! weights are already Q6_K/Q8_0 — int8 is the *native* format here rather
//! than a precision downgrade imposed on fp32 weights.
//!
//! # Why the accumulator is exact
//!
//! Every input is an integer in `[-128, 127]` and every product fits in 15
//! bits, so a `K`-length dot product needs `15 + ceil(log2(K))` bits. At
//! `K = 2048` that is 26 bits, comfortably inside int32. **There is no
//! rounding anywhere in this function**, which is what makes it a usable
//! oracle: the GPU must match it *bit for bit*, not to a tolerance. A
//! differential test that accepted a tolerance here would be unable to
//! distinguish a wrong fragment layout from arithmetic noise, and a wrong
//! fragment layout is the characteristic bug of hand-written MMA.
//!
//! Contrast that with the fp32 kernels elsewhere in this crate, where the
//! reference and the device legitimately disagree in the last bits because
//! they sum in different orders. Integer addition is associative, so that
//! entire class of excuse is unavailable and the gate can be exact.

/// `d[m][n] = sum_k a[m][k] * b[n][k]`, in exact int32 arithmetic.
///
/// `a` is `[m][k]` row-major and `b` is `[n][k]` row-major — that is, `b` is
/// the **transpose** of the mathematical right-hand operand, which is what the
/// `.col` in the PTX mnemonic means and what every quantized weight layout in
/// this project already stores (an output row's contraction run is
/// contiguous).
///
/// # Panics
///
/// If `a` or `b` is not the length its declared shape implies. A silently
/// short operand would read as a zero-padded matrix and produce a plausible
/// wrong answer.
pub fn int8_gemm(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
    assert_eq!(a.len(), m * k, "a must be [m][k] row-major");
    assert_eq!(b.len(), n * k, "b must be [n][k] row-major");

    let mut d = vec![0i32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for i in 0..k {
                acc += i32::from(a[row * k + i]) * i32::from(b[col * k + i]);
            }
            d[row * n + col] = acc;
        }
    }
    d
}

/// Where lane `lane` of a warp holds its piece of an `m8n8k16` **A** fragment.
///
/// Returns `(row, first_column)`; the lane holds four consecutive columns
/// starting there. From the PTX ISA's fragment layout for `m8n8k16`: the lane
/// index splits as `row = lane >> 2` and `k-group = lane & 3`.
///
/// Spelled in Rust as well as in the kernel so a test can check the two agree.
/// Getting this wrong produces a finite, plausible, completely wrong product —
/// it is the single most likely defect in an MMA port, and it is invisible
/// without an exact oracle.
pub const fn a_fragment_slot(lane: usize) -> (usize, usize) {
    (lane >> 2, (lane & 3) * 4)
}

/// Where lane `lane` holds its piece of an `m8n8k16` **B** fragment.
///
/// Returns `(column, first_contraction_index)`. B is the `.col` operand, so a
/// lane owns one output column and four consecutive contraction elements —
/// the mirror image of [`a_fragment_slot`].
pub const fn b_fragment_slot(lane: usize) -> (usize, usize) {
    (lane >> 2, (lane & 3) * 4)
}

/// Which two accumulator entries lane `lane` holds in an `m8n8k16` **C/D**
/// fragment.
///
/// Returns `(row, first_column)`; the lane holds columns `first_column` and
/// `first_column + 1`. Note this is *not* the same split as the operands: the
/// accumulator is 8x8 with two int32 per lane, so the column stride is 2 and
/// not 4.
pub const fn cd_fragment_slot(lane: usize) -> (usize, usize) {
    (lane >> 2, (lane & 3) * 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_agrees_with_a_hand_computed_product() {
        // a = [[1,2],[3,4]] as [2][2]; b = [[5,6],[7,8]] as [2][2] meaning
        // b[n][k], so the mathematical B is [[5,7],[6,8]].
        // d[0][0] = 1*5 + 2*6 = 17;  d[0][1] = 1*7 + 2*8 = 23
        // d[1][0] = 3*5 + 4*6 = 39;  d[1][1] = 3*7 + 4*8 = 53
        let a = [1i8, 2, 3, 4];
        let b = [5i8, 6, 7, 8];
        assert_eq!(int8_gemm(&a, &b, 2, 2, 2), vec![17, 23, 39, 53]);
    }

    #[test]
    fn the_accumulator_does_not_overflow_at_the_worst_case() {
        // Every product at its most negative, over the longest contraction the
        // MoE uses. -128 * -128 = 16384, times 2048 = 33,554,432 — well inside
        // int32, which is the claim the module docs make.
        let k = 2048;
        let a = vec![-128i8; k];
        let b = vec![-128i8; k];
        let d = int8_gemm(&a, &b, 1, 1, k);
        assert_eq!(d[0], 16_384 * k as i32);
        assert!(d[0] < i32::MAX / 2);
    }

    #[test]
    fn every_fragment_slot_is_covered_exactly_once() {
        // 32 lanes x 4 elements must tile an 8x16 operand with no gap and no
        // overlap. A layout that double-covers one element and skips another
        // still produces finite output, so this is checked rather than
        // assumed.
        let mut seen = [0u8; 8 * 16];
        for lane in 0..32 {
            let (row, col0) = a_fragment_slot(lane);
            for c in col0..col0 + 4 {
                seen[row * 16 + c] += 1;
            }
        }
        assert!(
            seen.iter().all(|&n| n == 1),
            "A fragment does not tile 8x16"
        );

        // The accumulator is 8x8 with two entries per lane.
        let mut acc = [0u8; 8 * 8];
        for lane in 0..32 {
            let (row, col0) = cd_fragment_slot(lane);
            for c in col0..col0 + 2 {
                acc[row * 8 + c] += 1;
            }
        }
        assert!(
            acc.iter().all(|&n| n == 1),
            "C/D fragment does not tile 8x8"
        );
    }
}
