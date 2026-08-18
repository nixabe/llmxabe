//! A host-side, no-GPU-required measurement, not a kernel: simulates what
//! switching `attn_flash_causal_mma`'s P*V accumulator from
//! fp32 to fp16 (matching llama.cpp's `fattn-mma-f16.cuh` on Turing, where
//! `T_C_VKQ = tile<16, 8, half2>`) would do to the error our own kernel
//! already measures against `MMA_GATE` (3.9e-3, defined in
//! `crates/xabe-engine/tests/attention_differential.rs`).
//!
//! O(seq_len) per query row, not O(seq_len^2): only the deepest row at each
//! depth is evaluated (the worst case a real chunk stresses), following the
//! same reasoning `attention_differential.rs`'s own docstring gives for why
//! a dense 131,072-key reference is evaluated at specific rows rather than
//! materializing every row.
//!
//! Mirrors `xabe_kernels::attention::causal_attention_streaming`'s math
//! (verified to agree with it, see the `sanity` row below) but:
//! - Rounds Q, K, P, V operands through fp16 before use, exactly like the
//!   MMA_GATE kernel already does (this part is NOT the change under test;
//!   it establishes the same baseline the real gate measures from).
//! - Processes keys in tiles (either llama.cpp's `nbatch_fa` width or our
//!   own kernel's narrower `MMA_KEY_TILE`, both swept below), one
//!   online-softmax rescale per tile rather than per key -- matching the
//!   real kernel's granularity (already priced into the current 1.88e-3
//!   measured error, so this is not new).
//! - NEW: within a tile, the P*V accumulator itself is rounded to fp16
//!   after every `MMA_K` (8)-key sub-group (one HMMA.1688.F16 call each,
//!   the K=8 reduction computed at full precision internally and only the
//!   *output* rounded -- matching real tensor-core semantics) and again
//!   after every tile's rescale multiply. That is the change: every other
//!   rounding point already exists in the shipped, gated kernel.
//!
//! Run: `cargo run --release -p xabe-kernels --example fp16_accum_simulation`
//!
//! ## Result (2026-08-18)
//!
//! Holds with real margin at every depth from 512 to 131,072, at both
//! rescale cadences, at the deepest row (three seeds at 131,072):
//! fp16-accumulate max_abs error is ~1.2e-4-1.5e-4 against `MMA_GATE`'s
//! 3.906e-3 -- about 26-30x headroom, not trending toward the gate as depth
//! grows, and not materially different between an 8x-more-frequent rescale
//! (our own kernel's tiling) and llama.cpp's own, wider one. The
//! self-normalizing division at the end bounds it: the accumulator's
//! absolute magnitude grows with the running softmax denominator, and both
//! are divided out together, so a fp16 rounding's *relative* contribution
//! to the final normalized output does not compound with depth or rescale
//! frequency. This does not mean adopting fp16 accumulation is free of risk
//! the model itself would not also need re-validating against
//! (`forward_pass`'s golden logits, not just this differential gate) -- it
//! means the differential gate alone is not the blocker. See "Two
//! accumulators, one instruction set" in docs/BENCHMARKS.md for the
//! decision this was reported for.

use xabe_kernels::compare::compare;
use xabe_kernels::f16::round_through_f16;
use xabe_kernels::rng::Xorshift64Star;

const HEAD_DIM: usize = 256; // Qwen3.6 attention head_dim.
const MMA_K: usize = 8; // m16n8k8's K: one HMMA call's reduction depth.

const MMA_GATE_MAX_ABS: f32 = 8.0 / 2048.0; // 3.90625e-3, from attention_differential.rs.

fn f16(x: f32) -> f32 {
    round_through_f16(x)
}

/// True fp32 streaming reference for exactly ONE query row against keys
/// `0..=t`, no fp16 rounding anywhere. Same math as
/// `causal_attention_streaming`, restricted to a single row so it is
/// O(seq_len) rather than O(seq_len^2).
fn reference_row(q_t: &[f32], k: &[Vec<f32>], v: &[Vec<f32>], t: usize) -> Vec<f32> {
    let head_dim = q_t.len();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;
    let mut acc = vec![0.0f32; head_dim];
    for i in 0..=t {
        let dot: f32 = q_t.iter().zip(k[i].iter()).map(|(&a, &b)| a * b).sum();
        let score = dot * scale;
        let new_max = running_max.max(score);
        let correction = if running_max == f32::NEG_INFINITY {
            0.0
        } else {
            (running_max - new_max).exp()
        };
        let weight = (score - new_max).exp();
        running_sum = running_sum * correction + weight;
        for (a, vi) in acc.iter_mut().zip(v[i].iter()) {
            *a = *a * correction + weight * vi;
        }
        running_max = new_max;
    }
    acc.iter().map(|&a| a / running_sum).collect()
}

/// Tile-batched form for exactly ONE query row, either fp32- or
/// fp16-accumulating the P*V running sum. `q_t`, `k`, `v` are already
/// fp16-rounded operands (both variants use fp16 operands, matching the
/// shipped kernel -- only the accumulator's own rounding differs).
fn tiled_row(
    qh_t: &[f32],
    kh: &[Vec<f32>],
    vh: &[Vec<f32>],
    t: usize,
    fp16_accum: bool,
    chunk: usize,
) -> Vec<f32> {
    let head_dim = qh_t.len();
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32; // KQ_rowsum: a separate fp32 scalar, not part of T_C_VKQ.
    let mut acc = vec![0.0f32; head_dim]; // T_C_VKQ.

    let mut chunk_start = 0usize;
    while chunk_start <= t {
        let chunk_end = (chunk_start + chunk - 1).min(t);

        let scores: Vec<f32> = (chunk_start..=chunk_end)
            .map(|i| {
                let dot: f32 = qh_t.iter().zip(kh[i].iter()).map(|(&a, &b)| a * b).sum();
                dot * scale
            })
            .collect();
        let chunk_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let new_max = running_max.max(chunk_max);
        let correction = if running_max == f32::NEG_INFINITY {
            0.0
        } else {
            (running_max - new_max).exp()
        };

        running_sum *= correction;
        for a in acc.iter_mut() {
            *a *= correction;
            if fp16_accum {
                *a = f16(*a);
            }
        }

        let mut i = chunk_start;
        while i <= chunk_end {
            let group_end = (i + MMA_K - 1).min(chunk_end);
            let mut partial = vec![0.0f32; head_dim];
            let mut partial_sum = 0.0f32;
            for key in i..=group_end {
                let score = scores[key - chunk_start];
                let p = f16((score - new_max).exp());
                partial_sum += p;
                for (d, pv) in vh[key].iter().enumerate() {
                    partial[d] += p * pv;
                }
            }
            running_sum += partial_sum;
            for (a, p) in acc.iter_mut().zip(partial.iter()) {
                *a += p;
                if fp16_accum {
                    *a = f16(*a);
                }
            }
            i = group_end + 1;
        }

        running_max = new_max;
        chunk_start = chunk_end + 1;
    }

    acc.iter().map(|&a| a / running_sum).collect()
}

fn run_at_depth(seq_len: usize, seed: u64, chunk: usize, label: &str) {
    let mut rng = Xorshift64Star::new(seed);
    let q: Vec<Vec<f32>> = (0..seq_len)
        .map(|_| rng.vec_f32(HEAD_DIM, -1.0, 1.0))
        .collect();
    let k: Vec<Vec<f32>> = (0..seq_len)
        .map(|_| rng.vec_f32(HEAD_DIM, -1.0, 1.0))
        .collect();
    let v: Vec<Vec<f32>> = (0..seq_len)
        .map(|_| rng.vec_f32(HEAD_DIM, -1.0, 1.0))
        .collect();

    let t = seq_len - 1; // deepest row: the worst case a real chunk stresses.
    let qh: Vec<Vec<f32>> = q
        .iter()
        .map(|r| r.iter().map(|&x| f16(x)).collect())
        .collect();
    let kh: Vec<Vec<f32>> = k
        .iter()
        .map(|r| r.iter().map(|&x| f16(x)).collect())
        .collect();
    let vh: Vec<Vec<f32>> = v
        .iter()
        .map(|r| r.iter().map(|&x| f16(x)).collect())
        .collect();

    let reference = reference_row(&q[t], &k, &v, t);
    let baseline = tiled_row(&qh[t], &kh, &vh, t, false, chunk);
    let fp16_accum = tiled_row(&qh[t], &kh, &vh, t, true, chunk);

    let base_cmp = compare(&baseline, &reference);
    let fp16_cmp = compare(&fp16_accum, &reference);

    println!(
        "seq_len={seq_len:>7} chunk={chunk:>3} ({label:<24}) baseline: max_abs={:.3e} cosine={:.9}  |  \
         fp16-accum candidate: max_abs={:.3e} cosine={:.9}  |  MMA_GATE max_abs={:.3e}  {}",
        base_cmp.max_abs_error,
        base_cmp.cosine_similarity,
        fp16_cmp.max_abs_error,
        fp16_cmp.cosine_similarity,
        MMA_GATE_MAX_ABS,
        if fp16_cmp.max_abs_error <= MMA_GATE_MAX_ABS {
            "HOLDS"
        } else {
            "BREACHES"
        },
    );
}

/// llama.cpp's `nbatch_fa` at Qwen3.6's geometry (DKQ=DV=256, ncols=32) on
/// Turing -- see `ggml_cuda_fattn_mma_get_config_turing` in fattn-mma-f16.cuh.
const LLAMA_CPP_CHUNK: usize = 64;

/// Our own kernel's key-tile width (`MMA_KEY_TILE` in attention.rs), which
/// is what a fp16-accumulate port of *this* kernel would rescale at if it
/// kept its current tiling rather than adopting llama.cpp's wider one. 8x
/// more rescales than `LLAMA_CPP_CHUNK` over the same depth -- the more
/// pessimistic of the two real cadences to check.
const OUR_KERNEL_CHUNK: usize = 8;

fn main() {
    println!(
        "Simulating attn_flash_causal_mma's P*V accumulator at fp16 (llama.cpp's Turing arithmetic)"
    );
    println!("against the current fp32-accumulate baseline MMA_GATE (3.90625e-3) already gates.");
    println!(
        "Only the deepest query row is evaluated at each depth (O(seq_len), matches the worst case)."
    );
    println!(
        "Two rescale cadences: llama.cpp's own nbatch_fa=64, and our kernel's narrower MMA_KEY_TILE=8.\n"
    );

    for &(seq_len, seed) in &[
        (512usize, 0x5EED_A001u64),
        (2_048, 0x5EED_A002),
        (8_192, 0x5EED_A003),
        (32_768, 0x5EED_A004),
        (65_536, 0x5EED_A005),
        (131_072, 0x5EED_A006),
        (131_072, 0x5EED_A007),
        (131_072, 0x5EED_A008),
    ] {
        run_at_depth(seq_len, seed, LLAMA_CPP_CHUNK, "llama.cpp nbatch_fa=64");
        run_at_depth(seq_len, seed, OUR_KERNEL_CHUNK, "our MMA_KEY_TILE=8");
    }
}

#[cfg(test)]
mod sanity {
    use super::*;
    use xabe_kernels::attention::causal_attention_streaming;

    #[test]
    fn reference_row_matches_the_librarys_streaming_form() {
        let mut rng = Xorshift64Star::new(0xABCD);
        let seq_len = 200;
        let q: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| rng.vec_f32(HEAD_DIM, -1.0, 1.0))
            .collect();
        let k: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| rng.vec_f32(HEAD_DIM, -1.0, 1.0))
            .collect();
        let v: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| rng.vec_f32(HEAD_DIM, -1.0, 1.0))
            .collect();

        let full = causal_attention_streaming(&q, &k, &v);
        let t = seq_len - 1;
        let mine = reference_row(&q[t], &k, &v, t);

        for d in 0..HEAD_DIM {
            let a = full[t][d];
            let b = mine[d];
            assert!(
                (a - b).abs() < 1e-6,
                "row reimplementation diverges at d={d}: library={a} mine={b}"
            );
        }
    }
}
