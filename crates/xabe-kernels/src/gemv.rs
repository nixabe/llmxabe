//! Scalar fp32 GEMV reference: a row-major matrix times one or more vectors.
//!
//! The oracle for the LM head (`xabe_cuda::kernels::lm_head`), which is a
//! single 248,320 x 2,048 matrix-vector product per token and nothing else.
//!
//! # Why this is not `moe::gemm`'s `matvec`
//!
//! [`crate::moe::gemm`] has a private `matvec` with the same body, reached
//! only through `expert_mlp`. It is private on purpose — it is an
//! implementation detail of the expert forward pass, not an interface — and
//! the LM head needs two things it does not offer: a public entry point, and
//! a batched form that shares one pass over the weights across several
//! vectors so the caller can check the device kernel's batch tiling against
//! it. Rather than widen `moe::gemm`'s surface and give the MoE reference a
//! caller it does not serve, the shared arithmetic is written once here.
//!
//! # Summation order is part of the contract
//!
//! Each output element is the sequential fp32 sum of `in_dim` products, in
//! ascending index order and with the multiply and the add rounded
//! separately. That is deliberately the *least* accurate reasonable order —
//! a device kernel reducing in a shuffle tree, or contracting into an FMA,
//! will disagree in the last few ulps, and the differential test's job is to
//! measure that disagreement rather than to be spared it. Accumulating in
//! f64 here would make the reference better than the thing it is checking
//! and turn a rounding difference into an apparent kernel error.
//!
//! # Chunking
//!
//! [`gemv`] takes `out_dim` rows at a time rather than requiring the whole
//! matrix. At the LM head's real shape the full fp32 weight is 2.03 GiB, so
//! a caller that dequantizes the GGUF tensor in row chunks and calls this
//! once per chunk stays within a few tens of megabytes. The output index is
//! chunk-local; the caller offsets it.

/// `out[o] = sum_i weight[o * in_dim + i] * x[i]`.
///
/// `weight` is row-major `[out_dim x in_dim]` — the layout GGUF stores the
/// LM head in, with `dims[0] = in_dim` fastest-varying.
///
/// # Panics
/// If `weight.len() != out_dim * in_dim` or `x.len() != in_dim`. A silent
/// truncation here would make every comparison downstream meaningless.
pub fn gemv(weight: &[f32], out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(
        weight.len(),
        out_dim * in_dim,
        "gemv: weight is not {out_dim} x {in_dim}",
    );
    assert_eq!(x.len(), in_dim, "gemv: x is not {in_dim} long");

    (0..out_dim)
        .map(|o| {
            let row = &weight[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x.iter()).map(|(&w, &v)| w * v).sum()
        })
        .collect()
}

/// [`gemv`] for several vectors sharing one pass over `weight`.
///
/// Returns `[xs.len()][out_dim]`. Each output is bit-identical to calling
/// [`gemv`] per vector — the shared pass changes only the order the *rows*
/// are visited in, never the order any single dot product is summed in — so
/// this is a convenience and a statement of intent, not a second algorithm.
///
/// # Panics
/// If any vector is not `in_dim` long, or `xs` is empty.
pub fn gemv_batch(weight: &[f32], out_dim: usize, in_dim: usize, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    assert!(!xs.is_empty(), "gemv_batch: nothing to multiply");
    assert_eq!(
        weight.len(),
        out_dim * in_dim,
        "gemv_batch: weight is not {out_dim} x {in_dim}",
    );

    let mut out = vec![vec![0.0f32; out_dim]; xs.len()];
    for o in 0..out_dim {
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        for (t, x) in xs.iter().enumerate() {
            assert_eq!(x.len(), in_dim, "gemv_batch: x[{t}] is not {in_dim} long");
            out[t][o] = row.iter().zip(x.iter()).map(|(&w, &v)| w * v).sum();
        }
    }
    out
}

/// Index of the largest element, ties going to the lower index.
///
/// The LM head exists to be argmaxed, and the sampled token is the only part
/// of a 248,320-element logit vector that reaches the user. A kernel can be
/// well within any sensible tolerance and still flip an argmax between two
/// near-tied candidates, so this is a separate assertion in the differential
/// test rather than something a cosine similarity is trusted to cover.
///
/// The tie-break matters for the same reason it does in
/// [`crate::moe::router`]: without it, two implementations that agree on
/// every value can still disagree on the answer.
///
/// # Panics
/// If `v` is empty — there is no argmax of nothing, and returning 0 would
/// hide a caller bug.
pub fn argmax(v: &[f32]) -> usize {
    assert!(!v.is_empty(), "argmax: empty logits");
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xorshift64Star;

    #[test]
    fn gemv_multiplies_row_major() {
        // [[1,2,3],[4,5,6]] @ [1,10,100] = [321, 654]. Reading the weight
        // column-major instead would give [1*1+4*10, ...] and is the error
        // this catches.
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = gemv(&w, 2, 3, &[1.0, 10.0, 100.0]);
        assert_eq!(out, vec![321.0, 654.0]);
    }

    #[test]
    fn the_batched_form_is_bit_identical_to_the_per_vector_form() {
        // The property that lets the differential test use either: sharing a
        // pass over the rows must not change any dot product's summation
        // order. Exact equality, not a tolerance — anything else means the
        // two paths are not the same arithmetic.
        let mut rng = Xorshift64Star::new(0x_6E11_0001);
        let (out_dim, in_dim) = (17usize, 96usize);
        let w = rng.vec_f32(out_dim * in_dim, -1.0, 1.0);
        let xs: Vec<Vec<f32>> = (0..5).map(|_| rng.vec_f32(in_dim, -1.0, 1.0)).collect();

        let batched = gemv_batch(&w, out_dim, in_dim, &xs);
        for (t, x) in xs.iter().enumerate() {
            assert_eq!(
                batched[t],
                gemv(&w, out_dim, in_dim, x),
                "vector {t} differs between the batched and per-vector forms",
            );
        }
    }

    #[test]
    fn chunking_the_rows_reproduces_the_whole_product_exactly() {
        // What the LM head differential test relies on to avoid holding a
        // 2.03 GiB fp32 weight: rows are independent, so the product of a
        // row chunk is the corresponding slice of the whole product.
        let mut rng = Xorshift64Star::new(0x_6E11_0002);
        let (out_dim, in_dim) = (64usize, 32usize);
        let w = rng.vec_f32(out_dim * in_dim, -2.0, 2.0);
        let x = rng.vec_f32(in_dim, -2.0, 2.0);

        let whole = gemv(&w, out_dim, in_dim, &x);
        let mut chunked = Vec::with_capacity(out_dim);
        for start in (0..out_dim).step_by(7) {
            let rows = (out_dim - start).min(7);
            chunked.extend(gemv(
                &w[start * in_dim..(start + rows) * in_dim],
                rows,
                in_dim,
                &x,
            ));
        }
        assert_eq!(whole, chunked);
    }

    #[test]
    fn argmax_breaks_ties_towards_the_lower_index() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[3.0, 3.0, 3.0]), 0);
        assert_eq!(argmax(&[-5.0]), 0);
        // Negative logits are the normal case for a language model; an
        // argmax seeded with 0.0 instead of the first element would return
        // the wrong token for every one of them.
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }

    #[test]
    fn the_reference_sums_sequentially_in_f32_not_in_f64() {
        // The documented contract. Once the running sum reaches 2^24 a
        // further +1 rounds back to it, so this sums to 2^24 in f32 and to
        // 2^24 + 7 in f64. An f64 accumulator would make this reference more
        // accurate than the kernel it is supposed to be checking, and the
        // resulting disagreement would be read as a kernel error.
        let w = vec![1.0f32; 8];
        let mut x = vec![1.0f32; 8];
        x[0] = 16_777_216.0;
        let out = gemv(&w, 1, 8, &x);
        assert_eq!(
            out[0], 16_777_216.0,
            "the accumulator stopped being f32 (an f64 sum would give 16777223)",
        );
    }
}
