//! Differential test: device Gated Attention against the scalar reference, at
//! the real Qwen3.6 geometry — 16 query heads, 2 KV heads (GQA 8:1), head
//! dimension 256, partial rotary over 64 of those 256 dimensions.
//!
//! This is the G004 gate. Gated Attention covers 10 of Qwen3.6's 40 layers
//! plus the MTP head at block 40, and unlike Gated DeltaNet it has a
//! Turing-validated codebase to port the *algorithm* from
//! (`fattn-vec.cuh`) — which means a wrong answer here has no excuse.
//!
//! ## What is compared, and against what
//!
//! `xabe_kernels::attention::causal_attention_streaming` — the online-softmax
//! form the kernel implements — run on the host, one query head at a time
//! through `kv_head_for_query_head`. That reference is itself cross-checked
//! here against `causal_attention_naive`, which materializes the full score
//! row and shares no code with it, so the oracle is not one implementation
//! vouching for itself.
//!
//! ## Why the two cannot be bit-identical
//!
//! Two differences are inherent:
//!
//! - The reference sums each `q . k` dot product sequentially over 256 terms;
//!   the kernel reduces it in a warp-shuffle tree. fp32 addition is not
//!   associative.
//! - The kernel applies the online-softmax rescale once per 8-key tile rather
//!   than once per key, and `expf` on the device and `f32::exp` on the host
//!   agree to within an ulp rather than exactly.
//!
//! Both are unbiased and bounded, so the gate is a tolerance — but a tight
//! one. The kernel accumulates in fp32 throughout, so anything near
//! `Tolerance::reduced_precision_gpu()` (5e-2, declared for fp16 accumulation)
//! would be a formulation error hiding behind a loose threshold.
//!
//! Three things here are gated at exact equality instead, because they have no
//! floating-point freedom at all: the partial-rotary tail, the packed
//! query/gate deinterleave, and the causal mask (perturbing a future key must
//! leave earlier rows bit-identical, not merely close).
//!
//! ## The 128K claim, stated precisely
//!
//! The G004 objective asks for agreement "at 128K context". The device is run
//! at a genuine 131,072-key window here, twice. The scalar reference is
//! **not** — it is O(n^2 * head_dim), about 4.4e12 FLOP per head at that
//! length, which is tens of minutes per head single-threaded, and it computes
//! every query row when only the deep ones are interesting.
//!
//! So the deep coverage is split:
//!
//! - `device_attention_matches_the_reference_at_a_128k_window` runs the device
//!   over 131,072 keys and compares against the real CPU reference, by
//!   arranging for all but 1,280 of those keys to carry a zero key vector.
//!   Zero keys score exactly 0 while the live keys score around 64, so their
//!   softmax weight is `exp(-64)` and their contribution to the numerator is
//!   exactly zero (their value vector is zero too). The residual is bounded
//!   and the bound is asserted, not asserted-in-prose. The compacted problem
//!   is causally order-isomorphic to the full one, so the reference sees the
//!   identical attention problem at 1,280 rows.
//! - `device_attention_holds_the_softmax_normalizer_over_128k_dense_keys` runs
//!   the device over 131,072 *dense* random keys and checks it against an
//!   analytic oracle that is exact at any depth: when every value vector is
//!   the same, the output is that vector, because softmax weights are a
//!   partition of unity. That exercises 16,384 online-softmax rescales over
//!   real scores, which is the thing that drifts at depth.
//!
//! What is therefore **not** checked: a dense 131,072-key softmax against the
//! scalar reference element by element. Nothing in this file claims it is.
//!
//! SKIPS — reporting that it skipped — without a driver or a supported device.
//! It needs no model file: the geometry comes from `ModelConfig` and the
//! activations are synthetic, because there is no captured Qwen3.6 activation
//! to compare against.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::attention::{
    AttentionKernels, attn_packed_gate_offset, attn_packed_query_offset,
};
use xabe_kernels::attention::{
    causal_attention_naive, causal_attention_streaming, kv_head_for_query_head,
};
use xabe_kernels::compare::{Tolerance, assert_matches, compare};
use xabe_kernels::rng::Xorshift64Star;
use xabe_kernels::rope::apply_rope;
use xabe_model::config::ModelConfig;

/// Tolerance for an fp32 device kernel against an fp32 scalar reference
/// differing only in reduction order, tile-wise rather than key-wise softmax
/// rescaling, and one `exp`.
///
/// **`max_abs_error` and `min_cosine_similarity` are the gate;
/// `max_rel_error` is not, and deliberately so.**
///
/// `compare()` computes relative error as `|c - r| / max(|r|, 1e-6)`.
/// Attention output is a convex combination of value vectors, so at depth the
/// components are averages of hundreds of zero-mean values and land close to
/// zero — far below that 1e-6 floor for many elements. For those the
/// denominator *is* the floor, and the ratio reports `abs_error / 1e-6`
/// instead of anything about accuracy. `max_rel_error` is therefore bounded
/// above by `max_abs_error / 1e-6` no matter how correct the kernel is.
///
/// Gating on `max_abs_error` loses nothing: any element large enough for a
/// relative error to mean something is also large enough that a relative error
/// implies an absolute one. An output component of magnitude 0.1 wrong by 1%
/// is off by 1e-3, a hundred times the bound below.
///
/// Every test that gates on this also asserts that the element driving
/// `max_rel_error` really is near zero, so the justification fails loudly if
/// it stops holding.
///
/// `max_rel_error` is therefore set to exactly `max_abs_error / REL_EPS` —
/// the largest value the absolute gate can possibly permit. It is not an
/// independent constraint and is not pretending to be one; writing a smaller
/// number there would be a bound on `compare()`'s floor rather than on the
/// kernel. That is not a hypothetical: the 128K window test measures
/// `max_rel = 7.4e-1` on an element whose reference value is below 1e-6 and
/// whose absolute error is 7.4e-7.
///
/// `max_abs_error` at 1e-5 is about 6x the worst case measured anywhere it is
/// applied (1.60e-6, on the 128K window test) — enough for driver and
/// hardware variation, and two to three orders of magnitude below what any
/// real formulation error in an fp32 kernel would produce.
const GATE: Tolerance = Tolerance {
    max_abs_error: 1e-5,
    max_rel_error: 1e-5 / 1e-6,
    min_cosine_similarity: 1.0 - 1e-6,
    allow_non_finite: false,
};

fn setup() -> Option<Arc<CudaContext>> {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return None;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            return None;
        }
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return None;
    }
    Some(ctx)
}

/// The real geometry, in the shape the tests want it.
struct Geometry {
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
}

fn geometry() -> Geometry {
    let a = ModelConfig::qwen3_6_35b_a3b().attention;
    Geometry {
        q_heads: a.q_heads as usize,
        kv_heads: a.kv_heads as usize,
        head_dim: a.head_dim as usize,
        rope_dim: a.rope_dim as usize,
    }
}

/// Extract one head's `[rows][head_dim]` view out of a flat
/// `[rows][heads][head_dim]` tensor, in the `Vec<Vec<f32>>` shape the
/// reference takes.
fn head_rows(
    flat: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    head: usize,
) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|t| {
            let base = (t * heads + head) * head_dim;
            flat[base..base + head_dim].to_vec()
        })
        .collect()
}

/// Run the device kernel over a whole window and bring the output back.
#[allow(clippy::too_many_arguments)]
fn run_device(
    stream: &Arc<CudaStream>,
    kernels: &AttentionKernels,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_query: usize,
    n_keys: usize,
    key_offset: usize,
    q_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let d_q = stream.clone_htod(q).expect("upload q");
    let d_k = stream.clone_htod(k).expect("upload k");
    let d_v = stream.clone_htod(v).expect("upload v");
    let mut d_out = stream
        .alloc_zeros::<f32>(n_query * q_heads * head_dim)
        .expect("allocate output");
    kernels
        .forward(
            stream, &d_q, &d_k, &d_v, &mut d_out, n_query, n_keys, key_offset,
        )
        .expect("attention launches");
    let out = stream.clone_dtoh(&d_out).expect("read output");
    stream.synchronize().expect("sync");
    out
}

/// The `max_rel_error` justification from [`GATE`]'s doc comment, asserted.
///
/// A relative error at fp32 rounding scale needs no explanation — 1e-4 is
/// three orders of magnitude above fp32 epsilon, so anything below it is
/// consistent with a correct kernel on a well-scaled element. Above it, the
/// only acceptable explanation is `compare()`'s 1e-6 relative floor, which
/// requires the driving reference element to be near zero. If it is not, the
/// relative error is a real measurement and gating on `max_abs_error` alone
/// has stopped being sufficient.
fn assert_relative_error_is_a_floor_artefact(
    result: &xabe_kernels::compare::ComparisonResult,
    reference: &[f32],
    context: &str,
) {
    if result.max_rel_error <= 1e-4 {
        return;
    }
    let driving = reference[result.max_rel_error_index].abs();
    assert!(
        driving < 1e-3,
        "{context}: max_rel_error {:.3e} is on a reference value of {driving:.3e}, \
         which is large enough for the ratio to be meaningful — max_abs_error \
         alone is no longer a sufficient gate",
        result.max_rel_error,
    );
}

// --------------------------------------------------------------------------
// (a) + (b) + (c): device vs CPU reference at several depths, one of them
// deliberately not a multiple of the kernel's key tile.
// --------------------------------------------------------------------------

#[test]
fn device_attention_matches_the_reference_across_sequence_depths() {
    let Some(ctx) = setup() else { return };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = AttentionKernels::new(&ctx, g.q_heads, g.kv_heads, g.head_dim)
        .expect("kernels must compile for the real geometry");

    let tile = kernels.keys_per_tile();
    assert_eq!(tile, 8, "head_dim 256 gives 8 warps and so an 8-key tile");
    println!(
        "geometry: q_heads={}, kv_heads={} (GQA {}:1), head_dim={}, key tile={tile}, \
         shared={} B/block",
        g.q_heads,
        g.kv_heads,
        kernels.gqa_ratio(),
        g.head_dim,
        kernels.shared_bytes(),
    );

    // 1 is the degenerate single-token case; 8 is exactly one tile; 9, 129,
    // 511 and 1003 are all ragged against the 8-key tile, so the partial
    // final tile is exercised at four different remainders (1, 1, 7, 3).
    const DEPTHS: [usize; 6] = [1, 8, 9, 129, 511, 1003];
    for d in DEPTHS {
        assert!(
            d == 8 || !d.is_multiple_of(tile),
            "depth {d} was meant to be ragged against the {tile}-key tile",
        );
    }

    let mut rng = Xorshift64Star::new(0x0A77_3E47);
    for seq in DEPTHS {
        let q: Vec<f32> = rng.vec_f32(seq * g.q_heads * g.head_dim, -1.0, 1.0);
        let k: Vec<f32> = rng.vec_f32(seq * g.kv_heads * g.head_dim, -1.0, 1.0);
        let v: Vec<f32> = rng.vec_f32(seq * g.kv_heads * g.head_dim, -1.0, 1.0);

        let device = run_device(
            &stream, &kernels, &q, &k, &v, seq, seq, 0, g.q_heads, g.head_dim,
        );

        let mut worst_abs = 0.0f32;
        let mut worst_cos = 1.0f32;
        let mut worst_rel = 0.0f32;
        for h in 0..g.q_heads {
            let kvh =
                kv_head_for_query_head(h as u32, g.q_heads as u32, g.kv_heads as u32) as usize;
            let q_h = head_rows(&q, seq, g.q_heads, g.head_dim, h);
            let k_h = head_rows(&k, seq, g.kv_heads, g.head_dim, kvh);
            let v_h = head_rows(&v, seq, g.kv_heads, g.head_dim, kvh);

            let reference: Vec<f32> = causal_attention_streaming(&q_h, &k_h, &v_h)
                .into_iter()
                .flatten()
                .collect();
            let candidate: Vec<f32> = (0..seq)
                .flat_map(|t| {
                    let base = (t * g.q_heads + h) * g.head_dim;
                    device[base..base + g.head_dim].to_vec()
                })
                .collect();

            let result = compare(&candidate, &reference);
            worst_abs = worst_abs.max(result.max_abs_error);
            worst_cos = worst_cos.min(result.cosine_similarity);
            worst_rel = worst_rel.max(result.max_rel_error);
            assert_relative_error_is_a_floor_artefact(
                &result,
                &reference,
                &format!("seq {seq} head {h}"),
            );
            assert_matches(&candidate, &reference, &GATE);
        }
        println!(
            "seq={seq:5} x {} heads: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} \
             cosine={worst_cos:.9}",
            g.q_heads,
        );
    }
}

#[test]
fn the_streaming_oracle_agrees_with_the_naive_one_at_the_deepest_swept_depth() {
    // The reference the kernel is checked against is the online-softmax form.
    // If that form were itself wrong, every comparison above would agree on a
    // wrong answer. The naive form materializes the full score row and shares
    // no code with it. This needs no device.
    let g = geometry();
    let seq = 1003;
    let mut rng = Xorshift64Star::new(0x11_2233);
    let q: Vec<Vec<f32>> = (0..seq)
        .map(|_| rng.vec_f32(g.head_dim, -1.0, 1.0))
        .collect();
    let k: Vec<Vec<f32>> = (0..seq)
        .map(|_| rng.vec_f32(g.head_dim, -1.0, 1.0))
        .collect();
    let v: Vec<Vec<f32>> = (0..seq)
        .map(|_| rng.vec_f32(g.head_dim, -1.0, 1.0))
        .collect();

    let naive: Vec<f32> = causal_attention_naive(&q, &k, &v)
        .into_iter()
        .flatten()
        .collect();
    let streaming: Vec<f32> = causal_attention_streaming(&q, &k, &v)
        .into_iter()
        .flatten()
        .collect();
    let result = compare(&streaming, &naive);
    println!("oracle cross-check at seq={seq}: {result}");
    assert!(
        result.max_abs_error < 1e-5 && result.cosine_similarity > 1.0 - 1e-7,
        "the two reference forms disagree beyond fp32 rounding: {result}",
    );
}

// --------------------------------------------------------------------------
// Chunked prefill / decode: a query block that does not start at position 0.
// --------------------------------------------------------------------------

#[test]
fn device_attention_matches_the_reference_for_a_mid_sequence_query_block() {
    // `key_offset` is what makes chunked prefill and decode the same kernel.
    // Getting it wrong shifts every query's causal window and is invisible in
    // a test that always starts at position 0.
    let Some(ctx) = setup() else { return };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = AttentionKernels::new(&ctx, g.q_heads, g.kv_heads, g.head_dim).expect("compiles");

    const N_KEYS: usize = 1003;
    const KEY_OFFSET: usize = 501;
    let n_query = N_KEYS - KEY_OFFSET;

    let mut rng = Xorshift64Star::new(0x0FF5_E700);
    let q_full: Vec<f32> = rng.vec_f32(N_KEYS * g.q_heads * g.head_dim, -1.0, 1.0);
    let k: Vec<f32> = rng.vec_f32(N_KEYS * g.kv_heads * g.head_dim, -1.0, 1.0);
    let v: Vec<f32> = rng.vec_f32(N_KEYS * g.kv_heads * g.head_dim, -1.0, 1.0);
    // Only the tail rows are launched, but the reference needs the whole
    // sequence, so the query block is a slice of the same buffer.
    let q_block = &q_full[KEY_OFFSET * g.q_heads * g.head_dim..];

    let device = run_device(
        &stream, &kernels, q_block, &k, &v, n_query, N_KEYS, KEY_OFFSET, g.q_heads, g.head_dim,
    );

    let mut worst_abs = 0.0f32;
    let mut worst_cos = 1.0f32;
    for h in 0..g.q_heads {
        let kvh = kv_head_for_query_head(h as u32, g.q_heads as u32, g.kv_heads as u32) as usize;
        let q_h = head_rows(&q_full, N_KEYS, g.q_heads, g.head_dim, h);
        let k_h = head_rows(&k, N_KEYS, g.kv_heads, g.head_dim, kvh);
        let v_h = head_rows(&v, N_KEYS, g.kv_heads, g.head_dim, kvh);
        let full = causal_attention_streaming(&q_h, &k_h, &v_h);

        let reference: Vec<f32> = full[KEY_OFFSET..].iter().flatten().copied().collect();
        let candidate: Vec<f32> = (0..n_query)
            .flat_map(|i| {
                let base = (i * g.q_heads + h) * g.head_dim;
                device[base..base + g.head_dim].to_vec()
            })
            .collect();
        let result = compare(&candidate, &reference);
        worst_abs = worst_abs.max(result.max_abs_error);
        worst_cos = worst_cos.min(result.cosine_similarity);
        assert_relative_error_is_a_floor_artefact(&result, &reference, &format!("head {h}"));
        assert_matches(&candidate, &reference, &GATE);
    }
    println!(
        "key_offset={KEY_OFFSET} n_query={n_query} n_keys={N_KEYS} x {} heads: \
         max_abs={worst_abs:.3e} cosine={worst_cos:.9}",
        g.q_heads,
    );
}

// --------------------------------------------------------------------------
// (d) The causal mask, tested directly and structurally.
// --------------------------------------------------------------------------

#[test]
fn a_future_key_cannot_change_an_earlier_output_by_a_single_bit() {
    // An off-by-one in the causal bound leaks exactly one future token per
    // row. That moves max_abs_error by a few parts in a thousand at short
    // depth and by almost nothing at long depth — it would pass every
    // tolerance in this file. So it is tested by construction instead:
    // perturb one key/value pair enormously, and require every output row
    // that must not see it to come back *bit-identical*, not merely close.
    let Some(ctx) = setup() else { return };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = AttentionKernels::new(&ctx, g.q_heads, g.kv_heads, g.head_dim).expect("compiles");

    // A mid-sequence query block, so the mask is tested against `key_offset`
    // arithmetic rather than only against `qi`.
    const N_KEYS: usize = 1000;
    const KEY_OFFSET: usize = 900;
    const N_QUERY: usize = 100;
    // Absolute position of the key that gets perturbed. Query row i sits at
    // 900 + i, so rows 0..=49 must not see it and rows 50.. must.
    const PERTURBED: usize = 950;
    let first_affected = PERTURBED - KEY_OFFSET;

    let mut rng = Xorshift64Star::new(0xC0DE_1A5C);
    let q: Vec<f32> = rng.vec_f32(N_QUERY * g.q_heads * g.head_dim, -1.0, 1.0);
    let k: Vec<f32> = rng.vec_f32(N_KEYS * g.kv_heads * g.head_dim, -1.0, 1.0);
    let v: Vec<f32> = rng.vec_f32(N_KEYS * g.kv_heads * g.head_dim, -1.0, 1.0);

    let base = run_device(
        &stream, &kernels, &q, &k, &v, N_QUERY, N_KEYS, KEY_OFFSET, g.q_heads, g.head_dim,
    );

    let mut k2 = k.clone();
    let mut v2 = v.clone();
    for kvh in 0..g.kv_heads {
        for d in 0..g.head_dim {
            let i = (PERTURBED * g.kv_heads + kvh) * g.head_dim + d;
            // Large enough to dominate the softmax outright wherever it is
            // visible, so "unchanged" cannot be luck.
            k2[i] = 20.0;
            v2[i] = 500.0;
        }
    }
    let perturbed = run_device(
        &stream, &kernels, &q, &k2, &v2, N_QUERY, N_KEYS, KEY_OFFSET, g.q_heads, g.head_dim,
    );

    let row = |buf: &[f32], i: usize| {
        let w = g.q_heads * g.head_dim;
        buf[i * w..(i + 1) * w].to_vec()
    };
    for i in 0..first_affected {
        let a = row(&base, i);
        let b = row(&perturbed, i);
        assert_eq!(
            a,
            b,
            "query row {i} (absolute position {}) changed when key {PERTURBED} was \
             perturbed — the causal mask leaks a future token",
            KEY_OFFSET + i,
        );
    }
    // And the perturbation was real: the first row that *should* see the key
    // must have moved, or the test above proves nothing.
    let a = row(&base, first_affected);
    let b = row(&perturbed, first_affected);
    let moved = compare(&b, &a).max_abs_error;
    assert!(
        moved > 1.0,
        "query row {first_affected} (absolute position {PERTURBED}) did not move when \
         the key it attends to was perturbed: max_abs={moved:.3e}",
    );
    println!(
        "causal mask: rows 0..{first_affected} bit-identical under a perturbation of key \
         {PERTURBED}; row {first_affected} moved by {moved:.3e}",
    );
}

// --------------------------------------------------------------------------
// The 128K gate.
// --------------------------------------------------------------------------

/// Keys in the full window.
const DEEP_KEYS: usize = 131_072;
/// Live keys scattered below the query block in the sparse-support test.
const DEEP_SPREAD: usize = 1024;
/// Contiguous query rows at the deep end.
const DEEP_QUERY: usize = 256;

#[test]
fn device_attention_matches_the_reference_at_a_128k_window() {
    let Some(ctx) = setup() else { return };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = AttentionKernels::new(&ctx, g.q_heads, g.kv_heads, g.head_dim).expect("compiles");

    let key_offset = DEEP_KEYS - DEEP_QUERY;

    // Live key positions: DEEP_SPREAD scattered across [0, key_offset), then
    // the DEEP_QUERY positions the queries themselves sit at. Every other key
    // carries a zero key vector and a zero value vector.
    //
    // Stride 127 is deliberately coprime with the 8-key tile, so the live keys
    // land at every remainder rather than always at tile boundaries.
    let mut live: Vec<usize> = (0..DEEP_SPREAD).map(|j| j * 127 + 3).collect();
    assert!(
        *live.last().unwrap() < key_offset,
        "the scattered keys must all sit below the query block",
    );
    live.extend(key_offset..DEEP_KEYS);
    let n_live = live.len();

    // Scale so that a live key scores around 64 while a zero key scores
    // exactly 0. exp(-64) is 1.6e-28, so the ~130k zero keys move the softmax
    // normalizer by ~2e-23 — bounded and asserted below.
    const K_SCALE: f32 = 4.0;
    const SPREAD: f32 = 0.3;

    let mut rng = Xorshift64Star::new(0x1280_0001);
    // Queries: 1 + noise, so every live key's dot product is dominated by the
    // constant component and lands in a tight band well above zero.
    let q: Vec<f32> = (0..DEEP_QUERY * g.q_heads * g.head_dim)
        .map(|_| 1.0 + SPREAD * rng.next_f32_range(-1.0, 1.0))
        .collect();

    let mut k = vec![0.0f32; DEEP_KEYS * g.kv_heads * g.head_dim];
    let mut v = vec![0.0f32; DEEP_KEYS * g.kv_heads * g.head_dim];
    for &p in &live {
        for kvh in 0..g.kv_heads {
            let base = (p * g.kv_heads + kvh) * g.head_dim;
            for d in 0..g.head_dim {
                k[base + d] = K_SCALE * (1.0 + SPREAD * rng.next_f32_range(-1.0, 1.0));
                v[base + d] = rng.next_f32_range(-1.0, 1.0);
            }
        }
    }

    let device = run_device(
        &stream, &kernels, &q, &k, &v, DEEP_QUERY, DEEP_KEYS, key_offset, g.q_heads, g.head_dim,
    );

    // --- the compacted problem the reference actually solves ---------------
    //
    // Compact row j holds absolute position live[j]. The live positions are
    // ascending and the query block is the tail of them, so query row i is
    // compact row n_live - DEEP_QUERY + i, and its compact causal window
    // [0, that row] is exactly the set of live keys visible at absolute
    // position key_offset + i. The two problems are causally isomorphic.
    let first_query_row = n_live - DEEP_QUERY;
    let mut worst_abs = 0.0f32;
    let mut worst_cos = 1.0f32;
    let mut worst_rel = 0.0f32;
    let mut worst_leak = 0.0f64;

    for h in 0..g.q_heads {
        let kvh = kv_head_for_query_head(h as u32, g.q_heads as u32, g.kv_heads as u32) as usize;
        let k_c: Vec<Vec<f32>> = live
            .iter()
            .map(|&p| {
                let b = (p * g.kv_heads + kvh) * g.head_dim;
                k[b..b + g.head_dim].to_vec()
            })
            .collect();
        let v_c: Vec<Vec<f32>> = live
            .iter()
            .map(|&p| {
                let b = (p * g.kv_heads + kvh) * g.head_dim;
                v[b..b + g.head_dim].to_vec()
            })
            .collect();
        // Rows below the query block are never compared; zeros keep them
        // finite and cheap.
        let mut q_c: Vec<Vec<f32>> = vec![vec![0.0f32; g.head_dim]; first_query_row];
        for i in 0..DEEP_QUERY {
            let b = (i * g.q_heads + h) * g.head_dim;
            q_c.push(q[b..b + g.head_dim].to_vec());
        }

        // Bound the residual the dropped zero keys leave behind, using a
        // lower bound on each query's maximum score: the score of compact key
        // 0, which every query row can see. Their value vectors are exactly
        // zero, so they perturb only the normalizer, by at most
        // n_zero * exp(-m_lower) relative.
        let scale = 1.0f64 / (g.head_dim as f64).sqrt();
        for i in 0..DEEP_QUERY {
            let dot: f64 = q_c[first_query_row + i]
                .iter()
                .zip(k_c[0].iter())
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let m_lower = dot * scale;
            let n_zero = (key_offset + i + 1 - n_live) as f64;
            worst_leak = worst_leak.max(n_zero * (-m_lower).exp());
        }

        let full = causal_attention_streaming(&q_c, &k_c, &v_c);
        let reference: Vec<f32> = full[first_query_row..].iter().flatten().copied().collect();
        let candidate: Vec<f32> = (0..DEEP_QUERY)
            .flat_map(|i| {
                let b = (i * g.q_heads + h) * g.head_dim;
                device[b..b + g.head_dim].to_vec()
            })
            .collect();

        let result = compare(&candidate, &reference);
        worst_abs = worst_abs.max(result.max_abs_error);
        worst_cos = worst_cos.min(result.cosine_similarity);
        worst_rel = worst_rel.max(result.max_rel_error);
        assert_relative_error_is_a_floor_artefact(&result, &reference, &format!("head {h}"));
        assert_matches(&candidate, &reference, &GATE);
    }

    // The justification for comparing against a 1,280-row problem, asserted
    // rather than argued: if the score separation ever stops holding, this
    // fails instead of quietly turning into a loose comparison.
    assert!(
        worst_leak < 1e-12,
        "the {} zero-key rows perturb the softmax normalizer by up to {worst_leak:.3e} \
         relative — too much for the compacted reference to be equivalent",
        DEEP_KEYS - n_live,
    );

    println!(
        "128K window: {DEEP_KEYS} keys ({n_live} live, {} zero), {DEEP_QUERY} queries at \
         positions {key_offset}..{DEEP_KEYS}, {} heads: max_abs={worst_abs:.3e} \
         max_rel={worst_rel:.3e} cosine={worst_cos:.9}; zero-key normalizer residual \
         <= {worst_leak:.3e}",
        DEEP_KEYS - n_live,
        g.q_heads,
    );
}

#[test]
fn device_attention_holds_the_softmax_normalizer_over_128k_dense_keys() {
    // 131,072 dense random keys, so all 16,384 online-softmax tiles carry real
    // scores and every one of them can trigger a running-max rescale. The
    // oracle is analytic and exact at any depth: softmax weights are
    // non-negative and sum to one, so if every value vector in the window is
    // the same vector c, the output is c. A kernel that rescales the
    // accumulator but not the normalizer (or the reverse) fails this outright
    // while still producing plausible attention on random values.
    let Some(ctx) = setup() else { return };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = AttentionKernels::new(&ctx, g.q_heads, g.kv_heads, g.head_dim).expect("compiles");

    const N_QUERY: usize = 64;
    let key_offset = DEEP_KEYS - N_QUERY;

    let mut rng = Xorshift64Star::new(0x0DE5_5E00);
    let q: Vec<f32> = rng.vec_f32(N_QUERY * g.q_heads * g.head_dim, -1.0, 1.0);
    let k: Vec<f32> = rng.vec_f32(DEEP_KEYS * g.kv_heads * g.head_dim, -1.0, 1.0);

    let c: Vec<f32> = rng.vec_f32(g.kv_heads * g.head_dim, -1.0, 1.0);
    let mut v = vec![0.0f32; DEEP_KEYS * g.kv_heads * g.head_dim];
    for chunk in v.chunks_mut(g.kv_heads * g.head_dim) {
        chunk.copy_from_slice(&c);
    }

    let device = run_device(
        &stream, &kernels, &q, &k, &v, N_QUERY, DEEP_KEYS, key_offset, g.q_heads, g.head_dim,
    );

    let reference: Vec<f32> = (0..N_QUERY)
        .flat_map(|_| {
            (0..g.q_heads).flat_map(|h| {
                let kvh =
                    kv_head_for_query_head(h as u32, g.q_heads as u32, g.kv_heads as u32) as usize;
                c[kvh * g.head_dim..(kvh + 1) * g.head_dim].to_vec()
            })
        })
        .collect();

    let result = compare(&device, &reference);

    // This test does NOT gate on [`GATE`], and the reason is arithmetic, not
    // convenience.
    //
    // Everywhere else in this file the candidate and the reference perform the
    // *same* accumulation and differ only in its order, so their errors
    // largely cancel. Here the reference is exact closed-form arithmetic — the
    // vector `c` itself — while the kernel reaches it by summing 131,072
    // rescaled fp32 terms into the accumulator and the normalizer separately.
    // The disagreement is therefore the kernel's own accumulation noise
    // against exact math, and its expected scale is the standard random-walk
    // model `sqrt(n) * eps`:
    //
    //     sqrt(131072) * f32::EPSILON = 362 * 1.19e-7 = 4.3e-5
    //
    // which is above GATE's 1e-5. Gating this comparison at 1e-5 would be
    // gating below the arithmetic floor of the computation being tested.
    //
    // The bound is that model with 4x headroom, so it is derived rather than
    // invented, and it is still two orders of magnitude below the worst-case
    // *linear* error bound `n * eps = 1.6e-2` that a systematically biased
    // accumulation would approach. A kernel that rescaled the accumulator but
    // not the normalizer misses by O(1) and fails this by five orders.
    let random_walk_bound = (DEEP_KEYS as f32).sqrt() * f32::EPSILON;
    let deep_gate = Tolerance {
        max_abs_error: 4.0 * random_walk_bound,
        max_rel_error: 4.0 * random_walk_bound / 1e-6,
        min_cosine_similarity: 1.0 - 1e-6,
        allow_non_finite: false,
    };
    println!(
        "128K dense: {DEEP_KEYS} keys, {N_QUERY} queries at positions {key_offset}.., \
         {} heads, {} online-softmax tiles per block: {result}",
        g.q_heads,
        DEEP_KEYS.div_ceil(kernels.keys_per_tile()),
    );
    println!(
        "  sqrt(n)*eps accumulation model = {random_walk_bound:.3e}, measured/model = {:.3}, \
         gate = {:.3e}",
        result.max_abs_error / random_walk_bound,
        deep_gate.max_abs_error,
    );
    assert_eq!(
        result.non_finite_count, 0,
        "the deep softmax produced NaN/Inf"
    );
    // The justification above, asserted: if the measured error ever climbs
    // meaningfully past the random-walk model, the model has stopped
    // describing the kernel and the loosening is no longer justified.
    assert!(
        result.max_abs_error < 4.0 * random_walk_bound,
        "measured error {:.3e} is {:.1}x the sqrt(n)*eps accumulation model \
         ({random_walk_bound:.3e}) — that is no longer accumulation noise",
        result.max_abs_error,
        result.max_abs_error / random_walk_bound,
    );
    assert_matches(&device, &reference, &deep_gate);
}

// --------------------------------------------------------------------------
// (e) Partial rotary: the tail passes through bit-exactly.
// --------------------------------------------------------------------------

#[test]
fn partial_rotary_rotates_64_dimensions_and_copies_the_other_192_bit_exactly() {
    let Some(ctx) = setup() else { return };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = AttentionKernels::new(&ctx, g.q_heads, g.kv_heads, g.head_dim).expect("compiles");

    const THETA_BASE: f32 = 10_000.0;
    const N_TOKENS: usize = 32;
    assert_eq!(g.rope_dim, 64);
    assert_eq!(g.head_dim, 256);

    // Position 0 (the identity rotation), a short-context position, and the
    // top of the natively trained 262,144-token context — where a float angle
    // would have lost about five significant digits.
    let native_context = ModelConfig::qwen3_6_35b_a3b().native_context as usize;
    for pos_offset in [0usize, 1_337, native_context - N_TOKENS] {
        // Both streams: q is 16 heads wide, k is 2. RoPE runs before the GQA
        // broadcast, so the kernel must handle both head counts.
        for n_heads in [g.q_heads, g.kv_heads] {
            let mut rng = Xorshift64Star::new(0x5200_E000u64 ^ pos_offset as u64);
            let input: Vec<f32> = rng.vec_f32(N_TOKENS * n_heads * g.head_dim, -1.0, 1.0);
            let d_in = stream.clone_htod(&input).expect("upload");
            let mut d_out = stream
                .alloc_zeros::<f32>(input.len())
                .expect("allocate output");
            kernels
                .rope(
                    &stream, &d_in, &mut d_out, N_TOKENS, n_heads, g.rope_dim, pos_offset,
                    THETA_BASE,
                )
                .expect("rope launches");
            let device = stream.clone_dtoh(&d_out).expect("read back");
            stream.synchronize().expect("sync");

            let mut rotated_c = Vec::new();
            let mut rotated_r = Vec::new();
            let mut tail_c = Vec::new();
            let mut tail_r = Vec::new();
            for t in 0..N_TOKENS {
                for h in 0..n_heads {
                    let b = (t * n_heads + h) * g.head_dim;
                    let head = &input[b..b + g.head_dim];
                    let expected =
                        apply_rope(head, (pos_offset + t) as u32, g.rope_dim as u32, THETA_BASE);
                    rotated_c.extend_from_slice(&device[b..b + g.rope_dim]);
                    rotated_r.extend_from_slice(&expected[..g.rope_dim]);
                    tail_c.extend_from_slice(&device[b + g.rope_dim..b + g.head_dim]);
                    tail_r.extend_from_slice(&expected[g.rope_dim..]);
                }
            }

            // The tail is a copy on both sides, with no arithmetic anywhere in
            // it, so "close" is the wrong bar — it must be identical.
            assert_matches(&tail_c, &tail_r, &Tolerance::exact());
            let rot = compare(&rotated_c, &rotated_r);
            println!(
                "rope pos_offset={pos_offset:6} n_heads={n_heads:2}: \
                 tail dims {}..{} ({} elements) bit-identical; rotated span {rot}",
                g.rope_dim,
                g.head_dim,
                tail_c.len(),
            );
            assert!(
                rot.max_abs_error < 1e-5 && rot.cosine_similarity > 1.0 - 1e-7,
                "rotated span diverged beyond fp32 rounding: {rot}",
            );
        }
    }
}

// --------------------------------------------------------------------------
// The packed query/gate tensor.
// --------------------------------------------------------------------------

#[test]
fn the_packed_query_tensor_is_deinterleaved_per_head_not_split_in_half() {
    // `attn_q.weight` is [2048, 8192] and packs the query and its output gate
    // interleaved per head: [q_h0, gate_h0, q_h1, gate_h1, ...]. Confirmed
    // against llama.cpp's src/models/qwen35moe.cpp, which views it with
    // stride = head_dim*2 — not inferred from the dimensions.
    //
    // A halves split is arithmetically valid, finite, plausible, and a
    // different model. So this test does two things: it checks the kernel
    // against the interleaved layout at exact equality, and it separately
    // builds what a halves split would have produced and requires the kernel
    // *not* to have produced that.
    let Some(ctx) = setup() else { return };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = AttentionKernels::new(&ctx, g.q_heads, g.kv_heads, g.head_dim).expect("compiles");

    const N_TOKENS: usize = 4;
    let packed_width = g.q_heads * 2 * g.head_dim;
    assert_eq!(packed_width, 8192, "the real attn_q output width");

    // Values that identify their own head and slice, so a misread is visible
    // in the value rather than only in an aggregate metric.
    let mut packed = vec![0.0f32; N_TOKENS * packed_width];
    for t in 0..N_TOKENS {
        for h in 0..g.q_heads {
            for d in 0..g.head_dim {
                let row = t * packed_width;
                packed[row + attn_packed_query_offset(h, g.head_dim) + d] =
                    1000.0 * t as f32 + h as f32 + 0.001 * d as f32;
                packed[row + attn_packed_gate_offset(h, g.head_dim) + d] =
                    -(1000.0 * t as f32 + h as f32 + 0.001 * d as f32);
            }
        }
    }

    let d_packed = stream.clone_htod(&packed).expect("upload");
    let n = N_TOKENS * g.q_heads * g.head_dim;
    let mut d_q = stream.alloc_zeros::<f32>(n).expect("allocate q");
    let mut d_gate = stream.alloc_zeros::<f32>(n).expect("allocate gate");
    kernels
        .split_query_and_gate(&stream, &d_packed, &mut d_q, &mut d_gate, N_TOKENS)
        .expect("split launches");
    let device_q = stream.clone_dtoh(&d_q).expect("read q");
    let device_gate = stream.clone_dtoh(&d_gate).expect("read gate");
    stream.synchronize().expect("sync");

    let mut expect_q = vec![0.0f32; n];
    let mut expect_gate = vec![0.0f32; n];
    // What a loader that split the tensor down the middle would have read.
    let mut halves_q = vec![0.0f32; n];
    for t in 0..N_TOKENS {
        for h in 0..g.q_heads {
            for d in 0..g.head_dim {
                let dst = (t * g.q_heads + h) * g.head_dim + d;
                let row = t * packed_width;
                expect_q[dst] = packed[row + attn_packed_query_offset(h, g.head_dim) + d];
                expect_gate[dst] = packed[row + attn_packed_gate_offset(h, g.head_dim) + d];
                halves_q[dst] = packed[row + h * g.head_dim + d];
            }
        }
    }

    assert_matches(&device_q, &expect_q, &Tolerance::exact());
    assert_matches(&device_gate, &expect_gate, &Tolerance::exact());

    // The two layouts agree on head 0 and on nothing else, which is exactly
    // why a spot check on head 0 passes while the model is broken.
    let wrong = compare(&halves_q, &expect_q);
    assert!(
        wrong.max_abs_error > 1.0,
        "the halves-split layout is indistinguishable from the interleaved one \
         on this input, so this test proves nothing: {wrong}",
    );
    let against_wrong = compare(&device_q, &halves_q);
    assert!(
        against_wrong.max_abs_error > 1.0,
        "the kernel reproduced the halves-split layout: {against_wrong}",
    );
    println!(
        "packed attn_q [{}, {packed_width}]: interleaved deinterleave is bit-exact; \
         a halves split would differ by max_abs={:.3e}",
        ModelConfig::qwen3_6_35b_a3b().hidden_size,
        wrong.max_abs_error,
    );
}
