//! Differential test: device Gated DeltaNet against the scalar reference, at
//! the real Qwen3.6 geometry.
//!
//! This is the milestone-01 gate. GDN covers 30 of 40 layers and has no
//! flash-attention codebase to lean on, so if it does not come together
//! nothing else in the project matters — which is why it is checked before
//! anything is built on top of it.
//!
//! ## What is compared
//!
//! `xabe_kernels::gdn::recurrent::recurrent_forward` run on the host, against
//! the same inputs pushed through `GdnKernels::step` token by token. The
//! comparison covers both the per-token outputs and the final recurrent
//! state, because a kernel can produce plausible outputs from a state that is
//! drifting — and the state is what a later token, or a resumed prefix cache
//! hit, actually depends on.
//!
//! ## Why the two cannot be bit-identical
//!
//! Two differences are inherent and neither is a defect:
//!
//! - The reference sums sequentially; the kernel reduces in a warp-shuffle
//!   tree. fp32 addition is not associative.
//! - `expf` on the device and `f32::exp` on the host agree to within an ulp,
//!   not exactly.
//!
//! Both are unbiased and bounded, so the gate is a tolerance. It is
//! deliberately much tighter than `Tolerance::reduced_precision_gpu()` (5e-2,
//! declared for fp16 accumulation): this kernel accumulates in fp32
//! throughout, so anything near that bound would indicate a real formulation
//! error hiding behind a loose threshold rather than reassociation noise.
//!
//! SKIPS — reporting that it skipped — without a driver or a supported device.
//! It needs no model file: the geometry comes from `ModelConfig`, and the
//! activations are synthetic because there is no captured Qwen3.6 activation
//! to compare against.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::gdn::GdnKernels;
use xabe_kernels::compare::{Tolerance, assert_matches, compare};
use xabe_kernels::gdn::recurrent::recurrent_forward;
use xabe_kernels::rng::Xorshift64Star;
use xabe_model::config::ModelConfig;

/// Tolerance for an fp32 device kernel against an fp32 scalar reference
/// differing only in reduction order and one `exp`.
///
/// **`max_abs_error` and `min_cosine_similarity` are the gate here;
/// `max_rel_error` is not, and deliberately so.**
///
/// `compare()` computes relative error as `|c - r| / max(|r|, 1e-6)`. GDN
/// outputs contain many elements far below that floor — the delta rule drives
/// components toward zero — so for those the denominator is the floor rather
/// than the value, and the ratio reports `abs_error / 1e-6` instead of
/// anything about accuracy. That makes `max_rel_error` bounded above by
/// `max_abs_error / 1e-6` no matter how correct the kernel is: at the
/// measured `max_abs = 2.6e-8` it can reach 2.6e-2 on an element whose true
/// error is four billionths.
///
/// Gating on `max_abs_error` instead loses nothing. Any element large enough
/// for a relative error to be meaningful is also large enough that a relative
/// error implies an absolute one: an element of magnitude 1e-2 wrong by 1%
/// has an absolute error of 1e-4, a hundred times the bound below. So the
/// absolute bound subsumes the relative check over exactly the range where
/// the relative check means anything.
///
/// 1e-6 is roughly 38x the measured worst case, which leaves room for
/// hardware and driver variation without leaving room for a formulation bug.
const GATE: Tolerance = Tolerance {
    max_abs_error: 1e-6,
    max_rel_error: 5e-2,
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

/// Synthetic activations with realistic scale.
///
/// `log_decay` is negative — it is the log of a decay in (0, 1] — and `beta`
/// is in (0, 1). Feeding positive `log_decay` would make the state grow
/// without bound and turn the comparison into a test of overflow behaviour.
struct Inputs {
    q: Vec<Vec<Vec<f32>>>,
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
    log_decay: Vec<Vec<f32>>,
    beta: Vec<Vec<f32>>,
}

impl Inputs {
    fn generate(seq_len: usize, head_dim: usize, value_heads: usize, qk_heads: usize) -> Self {
        let mut rng = Xorshift64Star::new(0x5EED_1234);
        let mut unit = |n: usize| {
            (0..n)
                .map(|_| rng.next_f32() * 2.0 - 1.0)
                .collect::<Vec<_>>()
        };

        Self {
            q: (0..seq_len)
                .map(|_| (0..qk_heads).map(|_| unit(head_dim)).collect())
                .collect(),
            k: (0..seq_len)
                .map(|_| (0..qk_heads).map(|_| unit(head_dim)).collect())
                .collect(),
            v: (0..seq_len)
                .map(|_| (0..value_heads).map(|_| unit(head_dim)).collect())
                .collect(),
            log_decay: (0..seq_len)
                .map(|_| {
                    (0..value_heads)
                        .map(|_| -(rng.next_f32() * 0.1 + 0.001))
                        .collect()
                })
                .collect(),
            beta: (0..seq_len)
                .map(|_| {
                    (0..value_heads)
                        .map(|_| rng.next_f32() * 0.9 + 0.05)
                        .collect()
                })
                .collect(),
        }
    }
}

#[test]
fn device_gdn_matches_the_reference_over_a_long_sequence() {
    let Some(ctx) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let g = config.gdn;
    let head_dim = g.head_dim as usize;
    let value_heads = g.value_heads as usize;
    let qk_heads = g.qk_heads as usize;
    let heads_per_kv = value_heads / qk_heads;

    // Long enough that reassociation error has room to accumulate through the
    // recurrence — a 4-token test would pass on a kernel that drifts.
    const SEQ_LEN: usize = 512;

    println!(
        "geometry: head_dim={head_dim}, value_heads={value_heads}, qk_heads={qk_heads} \
         ({heads_per_kv} value heads per kv head), {SEQ_LEN} tokens",
    );

    let inputs = Inputs::generate(SEQ_LEN, head_dim, value_heads, qk_heads);
    let stream = ctx.default_stream();
    let kernels = GdnKernels::new(&ctx, head_dim, value_heads, qk_heads)
        .expect("kernels must compile for the real geometry");
    let mut scratch = kernels.scratch(&stream).expect("scratch allocates");

    // --- device: token by token, state carried forward -------------------

    let mut d_state = stream
        .alloc_zeros::<f32>(value_heads * head_dim * head_dim)
        .expect("state allocates");
    let mut d_out = stream
        .alloc_zeros::<f32>(value_heads * head_dim)
        .expect("output allocates");

    let mut device_outputs: Vec<Vec<f32>> = Vec::with_capacity(SEQ_LEN);
    for t in 0..SEQ_LEN {
        let q_flat: Vec<f32> = inputs.q[t].concat();
        let k_flat: Vec<f32> = inputs.k[t].concat();
        let v_flat: Vec<f32> = inputs.v[t].concat();

        let d_q = stream.clone_htod(&q_flat).expect("upload q");
        let d_k = stream.clone_htod(&k_flat).expect("upload k");
        let d_v = stream.clone_htod(&v_flat).expect("upload v");
        let d_g = stream.clone_htod(&inputs.log_decay[t]).expect("upload g");
        let d_b = stream.clone_htod(&inputs.beta[t]).expect("upload beta");

        kernels
            .step(
                &stream,
                &mut scratch,
                &mut d_state,
                &d_q,
                &d_k,
                &d_v,
                &d_g,
                &d_b,
                &mut d_out,
            )
            .expect("step launches");
        device_outputs.push(stream.clone_dtoh(&d_out).expect("read output"));
    }
    let device_state = stream.clone_dtoh(&d_state).expect("read state");
    stream.synchronize().expect("sync");

    // --- host reference: one head at a time ------------------------------

    let mut worst_out = 0.0f32;
    let mut worst_cos = 1.0f32;
    for h in 0..value_heads {
        let qk = h / heads_per_kv;
        let q: Vec<Vec<f32>> = (0..SEQ_LEN).map(|t| inputs.q[t][qk].clone()).collect();
        let k: Vec<Vec<f32>> = (0..SEQ_LEN).map(|t| inputs.k[t][qk].clone()).collect();
        let v: Vec<Vec<f32>> = (0..SEQ_LEN).map(|t| inputs.v[t][h].clone()).collect();
        let decay: Vec<f32> = (0..SEQ_LEN).map(|t| inputs.log_decay[t][h]).collect();
        let beta: Vec<f32> = (0..SEQ_LEN).map(|t| inputs.beta[t][h]).collect();

        let (ref_out, ref_state) = recurrent_forward(head_dim, &q, &k, &v, &decay, &beta, None);

        // Per-token outputs, flattened across the sequence for this head.
        let reference: Vec<f32> = ref_out.concat();
        let candidate: Vec<f32> = (0..SEQ_LEN)
            .flat_map(|t| device_outputs[t][h * head_dim..(h + 1) * head_dim].to_vec())
            .collect();
        let result = compare(&candidate, &reference);
        worst_out = worst_out.max(result.max_abs_error);
        worst_cos = worst_cos.min(result.cosine_similarity);

        // Evidence for the claim in GATE's doc comment: the element driving
        // `max_rel_error` is below the 1e-6 relative-error floor, so its
        // ratio is an artefact of the floor rather than a measurement.
        // Asserted, not asserted-in-prose, so it fails if that stops being
        // true and the relative error starts meaning something.
        let worst_rel_reference = reference[result.max_rel_error_index].abs();
        assert!(
            worst_rel_reference < 1e-3,
            "head {h}: max_rel_error {:.3e} is on a reference value of {:.3e}, \
             which is large enough for the ratio to be meaningful — \
             max_abs_error alone is no longer a sufficient gate",
            result.max_rel_error,
            worst_rel_reference,
        );

        assert_matches(&candidate, &reference, &GATE);

        // The final state matters independently: a prefix-cache hit resumes
        // from it, so a state that drifts while outputs look fine would
        // corrupt every continuation.
        let state_slice = &device_state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim];
        assert_matches(state_slice, &ref_state, &GATE);
    }

    println!(
        "worst over {value_heads} heads x {SEQ_LEN} tokens: max_abs={worst_out:.3e}, \
         cosine={worst_cos:.9}",
    );
    println!(
        "gate: max_abs<{:.0e}, max_rel<{:.0e}, cosine>{:.9}",
        GATE.max_abs_error, GATE.max_rel_error, GATE.min_cosine_similarity,
    );
}

#[test]
fn a_zeroed_state_and_zero_beta_leaves_the_state_untouched() {
    // The degenerate case the recurrence must not corrupt: with beta = 0 no
    // correction is written, so the state stays zero and every output is
    // zero. A kernel that leaked `v` into the state — an easy slip when the
    // correction term is mis-signed — passes the long-sequence comparison on
    // cosine similarity but fails this outright.
    let Some(ctx) = setup() else { return };
    let head_dim = 128usize;
    let value_heads = 4usize;
    let qk_heads = 2usize;

    let stream = ctx.default_stream();
    let kernels = GdnKernels::new(&ctx, head_dim, value_heads, qk_heads).expect("compiles");
    let mut scratch = kernels.scratch(&stream).expect("scratch");

    let mut d_state = stream
        .alloc_zeros::<f32>(value_heads * head_dim * head_dim)
        .expect("state");
    let mut d_out = stream
        .alloc_zeros::<f32>(value_heads * head_dim)
        .expect("out");

    let mut rng = Xorshift64Star::new(7);
    let q: Vec<f32> = (0..qk_heads * head_dim).map(|_| rng.next_f32()).collect();
    let k: Vec<f32> = (0..qk_heads * head_dim).map(|_| rng.next_f32()).collect();
    let v: Vec<f32> = (0..value_heads * head_dim)
        .map(|_| rng.next_f32() * 10.0)
        .collect();

    let d_q = stream.clone_htod(&q).expect("q");
    let d_k = stream.clone_htod(&k).expect("k");
    let d_v = stream.clone_htod(&v).expect("v");
    let d_g = stream.clone_htod(&vec![-0.05f32; value_heads]).expect("g");
    let d_b = stream.clone_htod(&vec![0.0f32; value_heads]).expect("beta");

    kernels
        .step(
            &stream,
            &mut scratch,
            &mut d_state,
            &d_q,
            &d_k,
            &d_v,
            &d_g,
            &d_b,
            &mut d_out,
        )
        .expect("step");

    let state = stream.clone_dtoh(&d_state).expect("state back");
    let out = stream.clone_dtoh(&d_out).expect("out back");
    stream.synchronize().expect("sync");

    assert!(
        state.iter().all(|&s| s == 0.0),
        "beta=0 wrote {} non-zero state elements",
        state.iter().filter(|&&s| s != 0.0).count(),
    );
    assert!(
        out.iter().all(|&o| o == 0.0),
        "beta=0 produced non-zero output"
    );
    println!("beta=0 leaves state and output at exactly zero");
}
