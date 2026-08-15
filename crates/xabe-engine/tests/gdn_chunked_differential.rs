//! Differential test: the device chunked-parallel (prefill) Gated DeltaNet
//! against both scalar reference forms, at the real Qwen3.6 geometry.
//!
//! This is the prefill half of the milestone-01 gate. Its sibling,
//! `gdn_differential.rs`, checks the recurrent (decode) form. GDN covers 30 of
//! Qwen3.6's 40 layers, and the two forms have to agree with each other as well
//! as with the reference: prefill fills a prefix cache that decode then resumes
//! from, so a chunked kernel that is merely self-consistent would produce a
//! model that changes its mind at the prefill/decode boundary.
//!
//! ## What is compared
//!
//! Three things, per value head, for every case:
//!
//! 1. Device chunked vs `xabe_kernels::gdn::chunked::chunked_forward` — the
//!    same algorithm on both sides, so this isolates device arithmetic.
//! 2. Device chunked vs `xabe_kernels::gdn::recurrent::recurrent_forward` — a
//!    *different* algorithm, so this is the one that would catch a wrong
//!    derivation rather than a wrong transcription.
//! 3. The final recurrent state, separately from the per-token outputs. A
//!    kernel can emit plausible outputs from a state that is drifting, and the
//!    state is exactly what a prefix-cache hit resumes from.
//!
//! ## Cases
//!
//! - 512 tokens, 8 whole chunks of 64, zero initial state.
//! - 581 tokens, 9 whole chunks plus a 5-token tail, non-zero initial state.
//!   The ragged tail exercises the partial-chunk path (a shorter Gram block, a
//!   shorter triangular system, a smaller shared-memory allocation), and the
//!   non-zero initial state is what makes chunk-to-chunk threading observable:
//!   with a zero start, a bug that dropped the carried-in state entirely would
//!   still pass the first chunk and could hide in the reassociation noise.
//! - 197 tokens carrying **this model's real decay rates**, including the
//!   per-token log-decays measured on Qwen3.6 blocks 0 and 20. See below.
//!
//! ## The two blind spots this test used to have
//!
//! **Decay range.** The synthetic `log_decay` above is drawn from
//! `[-0.101, -0.001]`, so the cumulative in-chunk decay never falls below
//! `exp(-6.5)` and `1/lambda_t` never exceeds 660. The real model is nowhere
//! near that: block 0's smallest per-token log-decay is **-91.578** and block
//! 20's is **-12.436**, so `lambda` reaches `2.46e-42` by the second token of
//! a chunk. The kernel formed `v_t / lambda_t`, which overflows fp32 at
//! `|v_t| > 1.2e-4` against a measured `max|v_t|` of 6.95: block 0's chunked
//! prefill came out 38912/38912 `NaN` and block 20's 20480/38912, while block
//! 4 — worst per-token log-decay only -5.99 — passed at `5.96e-8`. A test
//! whose inputs cannot reach the failure is not a test of it, so
//! [`device_chunked_gdn_survives_this_models_real_decay_rates`] injects those
//! exact measured values and gates on the result being finite.
//!
//! That case originally compared against the **recurrent** reference only,
//! because the host `chunked_forward` in `xabe-kernels` divided by the
//! cumulative decay too and overflowed on precisely these inputs, so it could
//! not serve as an oracle for them. The case detected that per head, said so,
//! and fell back to the recurrent form, which never accumulates a decay and is
//! immune.
//!
//! The host form has since been reformulated the same way. The mechanism
//! worked as designed: with zero heads now excluded, this case gates against
//! the host chunked oracle again with **no edit to the gating logic** — and at
//! `2.235e-8`, some 450x tighter than the recurrent-form comparison. The
//! per-head exclusion is deliberately kept rather than deleted, so a future
//! regression degrades this test to a weaker oracle loudly instead of
//! silently comparing against `NaN`.
//!
//! **Broadcast convention.** The CPU reference runs one head, so this file
//! performs the query/key head broadcast, and it used to do so with
//! `qk = h / heads_per_kv` — the same mapping the kernel had hard-coded.
//! Device and reference agreed with each other while both disagreed with the
//! model, which indexes the query/key head as `fastmodulo(h_idx, n_k_heads)`
//! (`ggml/src/ggml-cuda/gated_delta_net.cu:37`) and broadcasts with
//! `ggml_repeat_4d`, which tiles. Both sides now use `h % qk_heads`, and
//! [`the_query_key_head_broadcast_is_modulo_not_division`] scores one device
//! run against both candidate mappings so the convention is measured rather
//! than shared.
//!
//! ## Why the two cannot be bit-identical
//!
//! Three differences are inherent:
//!
//! - The reference sums sequentially; parts of the kernel reduce in a
//!   warp-shuffle tree. fp32 addition is not associative.
//! - `expf` on the device and `f32::exp` on the host agree to within an ulp.
//! - The kernel solves the per-chunk unit lower-triangular system by forward
//!   substitution while `chunked_forward` forms the inverse explicitly and
//!   multiplies. Same answer in exact arithmetic, different rounding — the
//!   reasoning behind that choice is in `xabe_cuda::kernels::gdn_chunked`'s
//!   module docs, and the numbers this test prints are what checks it.
//!
//! SKIPS — reporting that it skipped — without a driver or a supported device.
//! It needs no model file: the geometry comes from `ModelConfig` and the
//! activations are synthetic, because there is no captured Qwen3.6 activation
//! to compare against.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::gdn_chunked::GdnChunkedKernels;
use xabe_kernels::compare::{ComparisonResult, Tolerance, assert_matches, compare};
use xabe_kernels::gdn::chunked::chunked_forward;
use xabe_kernels::gdn::recurrent::recurrent_forward;
use xabe_kernels::rng::Xorshift64Star;
use xabe_model::config::ModelConfig;

/// The project's stated gate for Gated DeltaNet chunked-vs-recurrent
/// equivalence: `max_abs 5e-2`, `max_rel 5e-2`, `cosine 1 - 1e-3`.
///
/// Asserted on every comparison, in addition to [`GATE`], so that the goal's
/// literal wording is checked and not merely implied by a tighter bound.
const STATED_GATE: Tolerance = Tolerance::gdn_chunk_vs_recurrent();

/// The gate actually enforced: [`STATED_GATE`] with `max_abs` and `cosine`
/// tightened by several orders of magnitude, and `max_rel` demoted to a
/// tripwire.
///
/// **`max_abs_error` and `min_cosine_similarity` are the gate here;
/// `max_rel_error` is not.** `compare()` computes relative error as
/// `|c - r| / max(|r|, 1e-6)`. GDN drives many output and state components
/// toward zero, so for those the denominator is that floor rather than the
/// value, and the ratio reports `abs_error / 1e-6` instead of anything about
/// accuracy. `max_rel_error` is therefore bounded above by
/// `max_abs_error / 1e-6` no matter how correct the kernel is.
///
/// That is not hypothetical here. The measured worst ratio, on the ragged-tail
/// case, is `6.40e-2` — over `STATED_GATE`'s `5e-2` — on a reference element
/// whose magnitude is below the floor: the implied absolute error is
/// `6.40e-2 * 1e-6 = 6.4e-8`, *smaller* than that tensor's measured
/// `max_abs_error` of `9.5e-8`. The ratio is an artefact of the denominator,
/// and [`check`] asserts that rather than asserting it in prose.
///
/// Demoting `max_rel` loses nothing, because the replacement is strictly
/// stronger where the relative check means anything: `max_abs_error` is
/// tightened from `5e-2` to `1e-5` — 5000x — and `min_cosine_similarity` from
/// `1 - 1e-3` to `1 - 1e-6`. Any element large enough for a relative error to
/// be meaningful is large enough that a relative error implies an absolute
/// one: an element of magnitude `1e-2` wrong by 1% has an absolute error of
/// `1e-4`, ten times this bound.
///
/// `1e-5` is roughly 40x the worst absolute error measured over both cases and
/// all four comparisons (`2.4e-7`), which leaves room for hardware and driver
/// variation without leaving room for a formulation bug. `max_rel_error` is
/// left at `2e-1` — three times the measured worst, and well under the `10`
/// the floor alone would permit at this `max_abs` — so a relative error that
/// stops being a floor artefact still trips something.
const GATE: Tolerance = Tolerance {
    max_abs_error: 1e-5,
    max_rel_error: 2e-1,
    min_cosine_similarity: 1.0 - 1e-6,
    allow_non_finite: false,
};

/// Above this reference magnitude, a relative error is a measurement rather
/// than an artefact of `compare()`'s `1e-6` denominator floor, and [`GATE`]'s
/// demotion of `max_rel_error` would no longer be justified.
const REL_ARTEFACT_CEILING: f32 = 1e-3;

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

/// Synthetic activations with realistic scale, laid out the way the kernel
/// wants them.
///
/// `log_decay` is negative — it is the log of a decay in (0, 1] — and `beta` is
/// in (0, 1). Positive `log_decay` would make the state grow without bound and
/// turn the comparison into a test of overflow behaviour.
struct Inputs {
    /// `[seq_len][qk_heads][head_dim]`.
    q: Vec<Vec<Vec<f32>>>,
    /// `[seq_len][qk_heads][head_dim]`.
    k: Vec<Vec<Vec<f32>>>,
    /// `[seq_len][value_heads][head_dim]`.
    v: Vec<Vec<Vec<f32>>>,
    /// `[seq_len][value_heads]`.
    log_decay: Vec<Vec<f32>>,
    /// `[seq_len][value_heads]`.
    beta: Vec<Vec<f32>>,
    /// `[value_heads][head_dim * head_dim]`, `S[v][k]` per head.
    initial_state: Vec<Vec<f32>>,
}

impl Inputs {
    fn generate(
        seed: u64,
        seq_len: usize,
        head_dim: usize,
        value_heads: usize,
        qk_heads: usize,
        nonzero_initial_state: bool,
    ) -> Self {
        let mut rng = Xorshift64Star::new(seed);
        let mut unit = |n: usize| {
            (0..n)
                .map(|_| rng.next_f32() * 2.0 - 1.0)
                .collect::<Vec<_>>()
        };

        let q = (0..seq_len)
            .map(|_| (0..qk_heads).map(|_| unit(head_dim)).collect())
            .collect();
        let k = (0..seq_len)
            .map(|_| (0..qk_heads).map(|_| unit(head_dim)).collect())
            .collect();
        let v = (0..seq_len)
            .map(|_| (0..value_heads).map(|_| unit(head_dim)).collect())
            .collect();
        let log_decay = (0..seq_len)
            .map(|_| {
                (0..value_heads)
                    .map(|_| -(rng.next_f32() * 0.1 + 0.001))
                    .collect()
            })
            .collect();
        let beta = (0..seq_len)
            .map(|_| {
                (0..value_heads)
                    .map(|_| rng.next_f32() * 0.9 + 0.05)
                    .collect()
            })
            .collect();
        // A state of the magnitude a few hundred tokens of this input would
        // actually leave behind, not an arbitrary one: too large and the
        // carried-in term swamps the intra-chunk terms, which would make the
        // triangular solve untested rather than tested harder.
        let initial_state = (0..value_heads)
            .map(|_| {
                if nonzero_initial_state {
                    (0..head_dim * head_dim)
                        .map(|_| (rng.next_f32() * 2.0 - 1.0) * 0.25)
                        .collect()
                } else {
                    vec![0.0f32; head_dim * head_dim]
                }
            })
            .collect();

        Self {
            q,
            k,
            v,
            log_decay,
            beta,
            initial_state,
        }
    }

    fn flat_q(&self) -> Vec<f32> {
        self.q.iter().flat_map(|t| t.concat()).collect()
    }
    fn flat_k(&self) -> Vec<f32> {
        self.k.iter().flat_map(|t| t.concat()).collect()
    }
    fn flat_v(&self) -> Vec<f32> {
        self.v.iter().flat_map(|t| t.concat()).collect()
    }
    fn flat_log_decay(&self) -> Vec<f32> {
        self.log_decay.concat()
    }
    fn flat_beta(&self) -> Vec<f32> {
        self.beta.concat()
    }
    fn flat_state(&self) -> Vec<f32> {
        self.initial_state.concat()
    }
}

/// Worst-case metrics accumulated over every head of one comparison.
#[derive(Default)]
struct Worst {
    max_abs: f32,
    max_rel: f32,
    min_cosine: f32,
    /// Magnitude of the reference element that drove `max_rel`.
    rel_driver_magnitude: f32,
}

impl Worst {
    fn new() -> Self {
        Self {
            max_abs: 0.0,
            max_rel: 0.0,
            min_cosine: 1.0,
            rel_driver_magnitude: 0.0,
        }
    }

    fn absorb(&mut self, result: &ComparisonResult, reference: &[f32]) {
        self.max_abs = self.max_abs.max(result.max_abs_error);
        self.min_cosine = self.min_cosine.min(result.cosine_similarity);
        if result.max_rel_error > self.max_rel {
            self.max_rel = result.max_rel_error;
            self.rel_driver_magnitude = reference[result.max_rel_error_index].abs();
        }
    }
}

/// Print one comparison's worst case, with the caveat that makes `max_rel`
/// readable.
///
/// `compare()` divides by `max(|reference|, 1e-6)`, so on an element whose true
/// value sits below that floor the ratio reports `abs_error / 1e-6` and says
/// nothing about accuracy. The reference magnitude that drove the worst ratio
/// is printed alongside it so the number can be judged rather than trusted.
fn report(label: &str, w: &Worst) {
    println!(
        "  {label:<34} max_abs={:.3e}  max_rel={:.3e} (on |ref|={:.3e})  cosine={:.9}",
        w.max_abs, w.max_rel, w.rel_driver_magnitude, w.min_cosine,
    );
}

/// Compare one head's tensor against one reference, assert both gates, and
/// fold the result into the running worst case.
///
/// The relative-error justification is asserted here, not described in a
/// comment: if the element driving `max_rel_error` ever climbs above
/// [`REL_ARTEFACT_CEILING`], the ratio has stopped being an artefact of
/// `compare()`'s denominator floor, [`GATE`]'s demotion of `max_rel_error`
/// stops being defensible, and this fails loudly instead of quietly passing.
fn check(candidate: &[f32], reference: &[f32], label: &str, head: usize, worst: &mut Worst) {
    let result = compare(candidate, reference);
    worst.absorb(&result, reference);

    let rel_driver = reference[result.max_rel_error_index].abs();
    assert!(
        rel_driver < REL_ARTEFACT_CEILING,
        "{label}, head {head}: max_rel_error {:.3e} is on a reference value of {rel_driver:.3e}, \
         which is large enough for the ratio to be a measurement rather than an artefact of \
         compare()'s 1e-6 floor — max_abs_error and cosine alone are no longer a sufficient gate",
        result.max_rel_error,
    );

    // The stated project gate, on the two bounds the floor does not distort.
    assert!(
        result.max_abs_error <= STATED_GATE.max_abs_error
            && result.cosine_similarity >= STATED_GATE.min_cosine_similarity
            && result.non_finite_count == 0,
        "{label}, head {head}: fails the stated Tolerance::gdn_chunk_vs_recurrent gate: {result}",
    );
    // And the much tighter bound this kernel actually holds to.
    assert_matches(candidate, reference, &GATE);
}

/// Run the device chunked kernel over `seq_len` tokens and check it against
/// both host forms, per head, outputs and final state separately.
fn run_case(ctx: &Arc<CudaContext>, seed: u64, seq_len: usize, nonzero_initial_state: bool) {
    let config = ModelConfig::qwen3_6_35b_a3b();
    let g = config.gdn;
    let head_dim = g.head_dim as usize;
    let value_heads = g.value_heads as usize;
    let qk_heads = g.qk_heads as usize;
    let chunk_len = g.chunk_len as usize;
    let heads_per_kv = value_heads / qk_heads;

    let whole_chunks = seq_len / chunk_len;
    let tail = seq_len % chunk_len;
    println!(
        "case: {seq_len} tokens = {whole_chunks} chunk(s) of {chunk_len} + tail {tail}, \
         initial_state={}, head_dim={head_dim}, value_heads={value_heads}, qk_heads={qk_heads} \
         ({heads_per_kv} value heads per kv head)",
        if nonzero_initial_state {
            "non-zero"
        } else {
            "zero"
        },
    );

    let inputs = Inputs::generate(
        seed,
        seq_len,
        head_dim,
        value_heads,
        qk_heads,
        nonzero_initial_state,
    );

    // --- device ----------------------------------------------------------

    let stream = ctx.default_stream();
    let kernels = GdnChunkedKernels::new(ctx, head_dim, value_heads, qk_heads, chunk_len)
        .expect("kernels must compile for the real geometry");
    let mut scratch = kernels
        .scratch(&stream, seq_len)
        .expect("scratch allocates");

    let mut d_state = stream
        .clone_htod(&inputs.flat_state())
        .expect("state uploads");
    let mut d_out = stream
        .alloc_zeros::<f32>(seq_len * value_heads * head_dim)
        .expect("output allocates");
    let d_q = stream.clone_htod(&inputs.flat_q()).expect("upload q");
    let d_k = stream.clone_htod(&inputs.flat_k()).expect("upload k");
    let d_v = stream.clone_htod(&inputs.flat_v()).expect("upload v");
    let d_g = stream
        .clone_htod(&inputs.flat_log_decay())
        .expect("upload log_decay");
    let d_b = stream.clone_htod(&inputs.flat_beta()).expect("upload beta");

    kernels
        .prefill(
            &stream,
            &mut scratch,
            &mut d_state,
            &d_q,
            &d_k,
            &d_v,
            &d_g,
            &d_b,
            &mut d_out,
            seq_len,
        )
        .expect("prefill launches");

    let device_out = stream.clone_dtoh(&d_out).expect("read outputs");
    let device_state = stream.clone_dtoh(&d_state).expect("read state");
    stream.synchronize().expect("sync");

    // --- host, one head at a time ----------------------------------------

    let mut out_vs_chunked = Worst::new();
    let mut out_vs_recurrent = Worst::new();
    let mut state_vs_chunked = Worst::new();
    let mut state_vs_recurrent = Worst::new();

    for h in 0..value_heads {
        // Modulo, not division — see the module docs.
        let qk = h % qk_heads;
        let q: Vec<Vec<f32>> = (0..seq_len).map(|t| inputs.q[t][qk].clone()).collect();
        let k: Vec<Vec<f32>> = (0..seq_len).map(|t| inputs.k[t][qk].clone()).collect();
        let v: Vec<Vec<f32>> = (0..seq_len).map(|t| inputs.v[t][h].clone()).collect();
        let decay: Vec<f32> = (0..seq_len).map(|t| inputs.log_decay[t][h]).collect();
        let beta: Vec<f32> = (0..seq_len).map(|t| inputs.beta[t][h]).collect();
        let s0 = &inputs.initial_state[h];

        let (chunk_out, chunk_state) =
            chunked_forward(head_dim, chunk_len, &q, &k, &v, &decay, &beta, Some(s0));
        let (rec_out, rec_state) = recurrent_forward(head_dim, &q, &k, &v, &decay, &beta, Some(s0));

        // This head's per-token outputs, gathered out of the interleaved
        // [token][head][dim] device buffer.
        let candidate: Vec<f32> = (0..seq_len)
            .flat_map(|t| {
                let base = (t * value_heads + h) * head_dim;
                device_out[base..base + head_dim].to_vec()
            })
            .collect();
        let state_slice = &device_state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim];

        let ref_chunk_out: Vec<f32> = chunk_out.concat();
        let ref_rec_out: Vec<f32> = rec_out.concat();

        // (a) the same algorithm on both sides, (b) a different algorithm on
        // the host, (c) outputs and state checked separately.
        check(
            &candidate,
            &ref_chunk_out,
            "out vs host chunked",
            h,
            &mut out_vs_chunked,
        );
        check(
            &candidate,
            &ref_rec_out,
            "out vs host recurrent",
            h,
            &mut out_vs_recurrent,
        );
        check(
            state_slice,
            &chunk_state,
            "state vs host chunked",
            h,
            &mut state_vs_chunked,
        );
        check(
            state_slice,
            &rec_state,
            "state vs host recurrent",
            h,
            &mut state_vs_recurrent,
        );
    }

    report("device vs host chunked (out)", &out_vs_chunked);
    report("device vs host recurrent (out)", &out_vs_recurrent);
    report("device vs host chunked (state)", &state_vs_chunked);
    report("device vs host recurrent (state)", &state_vs_recurrent);
    println!(
        "  enforced gate: max_abs<{:.0e}, cosine>{:.9} (max_rel tripwire {:.0e}, \
         floor artefact ceiling |ref|<{:.0e})",
        GATE.max_abs_error, GATE.min_cosine_similarity, GATE.max_rel_error, REL_ARTEFACT_CEILING,
    );
    println!(
        "  stated gate (Tolerance::gdn_chunk_vs_recurrent): max_abs<{:.0e}, cosine>{:.9} \
         — met on both, by 5000x and better",
        STATED_GATE.max_abs_error, STATED_GATE.min_cosine_similarity,
    );
}

#[test]
fn device_chunked_gdn_matches_both_reference_forms_over_eight_whole_chunks() {
    let Some(ctx) = setup() else { return };
    // 512 tokens is 8 whole chunks of 64: long enough that a per-chunk state
    // handoff bug has to show, and the same length the recurrent differential
    // test uses so the two are directly comparable.
    run_case(&ctx, 0x5EED_1234, 512, false);
}

#[test]
fn device_chunked_gdn_handles_a_ragged_tail_from_a_non_zero_initial_state() {
    let Some(ctx) = setup() else { return };
    // 581 = 9 * 64 + 5. The tail chunk is 5 tokens: a 5x5 triangular system, a
    // 5-thread Gram block, and a shared-memory allocation an order of
    // magnitude smaller than the whole-chunk one.
    run_case(&ctx, 0x1234_5EED, 581, true);
}

/// The per-token log-decays measured on the real model, as
/// `(token, value head, log_decay)`.
///
/// The first two are verbatim from Qwen3.6: block 0's smallest per-token
/// log-decay is -91.578 at token 1, head 9, and block 20's is -12.436 at token
/// 9, head 7. The other two put the same magnitudes in the second and third
/// chunks, so the failure has to survive a state handoff rather than only the
/// opening chunk.
const REAL_DECAY_SPIKES: [(usize, usize, f32); 4] = [
    (1, 9, -91.578),
    (9, 7, -12.436),
    (70, 0, -91.578),
    (150, 31, -45.0),
];

#[test]
fn device_chunked_gdn_survives_this_models_real_decay_rates() {
    // The defect-2 gate. The synthetic decays the two cases above use bottom
    // out at exp(-6.5) per chunk; this model reaches exp(-91.578) in a single
    // token, and the kernel's original `v_t / lambda_t` overflowed fp32 there
    // and filled the chunk with NaN. The reformulated kernel never divides by
    // a cumulative decay — see `xabe_cuda::kernels::gdn_chunked`'s module docs
    // — so the worst that can happen is an underflow to +0, which is the
    // correct limit.
    //
    // Measured on this exact input with the `v_t / lambda_t` form and the
    // query/key broadcast already corrected, so the number is attributable to
    // the division alone: **41330 of 806912 output elements and 32768 of
    // 524288 state elements non-finite**. With the reformulation: 0 and 0,
    // agreeing with the recurrent form at max_abs 5.18e-7 (outputs) and
    // 9.86e-7 (state), cosine 1.000000000.
    let Some(ctx) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let g = config.gdn;
    let head_dim = g.head_dim as usize;
    let value_heads = g.value_heads as usize;
    let qk_heads = g.qk_heads as usize;
    let chunk_len = g.chunk_len as usize;
    // 197 = 3 * 64 + 5: three whole chunks and a ragged tail, so the spikes
    // land in three different chunks and one state handoff follows each.
    let seq_len = 197usize;

    let mut inputs = Inputs::generate(0xDECA_1234, seq_len, head_dim, value_heads, qk_heads, true);
    for (t, h, value) in REAL_DECAY_SPIKES {
        assert!(t < seq_len && h < value_heads);
        inputs.log_decay[t][h] = value;
    }
    let worst_decay = inputs
        .log_decay
        .iter()
        .flatten()
        .copied()
        .fold(f32::INFINITY, f32::min);
    println!(
        "case: {seq_len} tokens with the real model's decay rates, \
         min per-token log-decay {worst_decay:.3} (lambda {:.3e}), \
         initial_state=non-zero",
        worst_decay.exp(),
    );

    let stream = ctx.default_stream();
    let kernels = GdnChunkedKernels::new(&ctx, head_dim, value_heads, qk_heads, chunk_len)
        .expect("kernels must compile for the real geometry");
    let mut scratch = kernels
        .scratch(&stream, seq_len)
        .expect("scratch allocates");

    let mut d_state = stream
        .clone_htod(&inputs.flat_state())
        .expect("state uploads");
    let mut d_out = stream
        .alloc_zeros::<f32>(seq_len * value_heads * head_dim)
        .expect("output allocates");
    let d_q = stream.clone_htod(&inputs.flat_q()).expect("upload q");
    let d_k = stream.clone_htod(&inputs.flat_k()).expect("upload k");
    let d_v = stream.clone_htod(&inputs.flat_v()).expect("upload v");
    let d_g = stream
        .clone_htod(&inputs.flat_log_decay())
        .expect("upload log_decay");
    let d_b = stream.clone_htod(&inputs.flat_beta()).expect("upload beta");

    kernels
        .prefill(
            &stream,
            &mut scratch,
            &mut d_state,
            &d_q,
            &d_k,
            &d_v,
            &d_g,
            &d_b,
            &mut d_out,
            seq_len,
        )
        .expect("prefill launches");

    let device_out = stream.clone_dtoh(&d_out).expect("read outputs");
    let device_state = stream.clone_dtoh(&d_state).expect("read state");
    stream.synchronize().expect("sync");

    // Stated before any comparison, because `compare()` *skips* non-finite
    // candidate elements when accumulating max_abs_error: a kernel that
    // produced all NaN would otherwise report max_abs_error = 0.
    let out_nonfinite = device_out.iter().filter(|x| !x.is_finite()).count();
    let state_nonfinite = device_state.iter().filter(|x| !x.is_finite()).count();
    println!(
        "  device non-finite: {out_nonfinite}/{} outputs, {state_nonfinite}/{} state elements",
        device_out.len(),
        device_state.len(),
    );
    assert_eq!(
        out_nonfinite, 0,
        "the chunked kernel produced non-finite outputs on this model's real decay rates",
    );
    assert_eq!(state_nonfinite, 0, "the chunk-end state is non-finite");

    let mut out_vs_recurrent = Worst::new();
    let mut state_vs_recurrent = Worst::new();
    let mut host_chunked_overflowed = 0usize;
    let mut out_vs_chunked = Worst::new();

    for h in 0..value_heads {
        let qk = h % qk_heads;
        let q: Vec<Vec<f32>> = (0..seq_len).map(|t| inputs.q[t][qk].clone()).collect();
        let k: Vec<Vec<f32>> = (0..seq_len).map(|t| inputs.k[t][qk].clone()).collect();
        let v: Vec<Vec<f32>> = (0..seq_len).map(|t| inputs.v[t][h].clone()).collect();
        let decay: Vec<f32> = (0..seq_len).map(|t| inputs.log_decay[t][h]).collect();
        let beta: Vec<f32> = (0..seq_len).map(|t| inputs.beta[t][h]).collect();
        let s0 = &inputs.initial_state[h];

        let candidate: Vec<f32> = (0..seq_len)
            .flat_map(|t| {
                let base = (t * value_heads + h) * head_dim;
                device_out[base..base + head_dim].to_vec()
            })
            .collect();
        let state_slice = &device_state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim];

        // The recurrent form is the oracle here: it decays the state one token
        // at a time and never accumulates, so exp(-91.578) simply zeroes the
        // state instead of being inverted.
        let (rec_out, rec_state) = recurrent_forward(head_dim, &q, &k, &v, &decay, &beta, Some(s0));
        let ref_rec_out: Vec<f32> = rec_out.concat();
        assert!(
            ref_rec_out.iter().all(|x| x.is_finite()),
            "head {h}: the recurrent reference is itself non-finite, so it cannot be the oracle",
        );
        check(
            &candidate,
            &ref_rec_out,
            "out vs host recurrent",
            h,
            &mut out_vs_recurrent,
        );
        check(
            state_slice,
            &rec_state,
            "state vs host recurrent",
            h,
            &mut state_vs_recurrent,
        );

        // The host chunked form divides by the cumulative decay and therefore
        // has the very defect this case exists to catch. Gate on it only where
        // it survived, so this starts checking it automatically once it is
        // fixed rather than needing an edit here.
        let (chunk_out, chunk_state) =
            chunked_forward(head_dim, chunk_len, &q, &k, &v, &decay, &beta, Some(s0));
        let ref_chunk_out: Vec<f32> = chunk_out.concat();
        if ref_chunk_out.iter().all(|x| x.is_finite()) && chunk_state.iter().all(|x| x.is_finite())
        {
            check(
                &candidate,
                &ref_chunk_out,
                "out vs host chunked",
                h,
                &mut out_vs_chunked,
            );
        } else {
            host_chunked_overflowed += 1;
        }
    }

    report("device vs host recurrent (out)", &out_vs_recurrent);
    report("device vs host recurrent (state)", &state_vs_recurrent);
    if host_chunked_overflowed < value_heads {
        report("device vs host chunked (out)", &out_vs_chunked);
    }
    println!(
        "  xabe-kernels' host chunked_forward overflowed on {host_chunked_overflowed}/{value_heads} \
         heads and was excluded as an oracle for those \
         (0 is expected: the host form no longer divides by the cumulative decay)",
    );
}

#[test]
fn the_query_key_head_broadcast_is_modulo_not_division() {
    // The test `run_case` cannot be: it uses one broadcast convention on both
    // sides, so it measures the kernel against a reference that shares the
    // kernel's assumption. This one takes a single device run and scores it
    // against *both* candidate mappings.
    //
    // With 4 value heads and 2 query/key heads:
    //   modulo:   0->0, 1->1, 2->0, 3->1
    //   division: 0->0, 1->0, 2->1, 3->1
    // so heads 1 and 2 discriminate and heads 0 and 3 are the control.
    //
    // Measured against the mapping the kernel shipped with: head 1 scores
    // max_abs 5.304e-2 at cosine 0.0016 and head 2 max_abs 5.729e-2 at cosine
    // 0.0074. With the kernel on modulo every head lands at max_abs ~2e-8,
    // cosine 1.000000.
    let Some(ctx) = setup() else { return };
    let head_dim = 128usize;
    let value_heads = 4usize;
    let qk_heads = 2usize;
    let heads_per_kv = value_heads / qk_heads;
    let chunk_len = 8usize;
    let seq_len = 20usize; // two whole chunks of 8 and a 4-token tail

    let inputs = Inputs::generate(0xB0AD_CA57, seq_len, head_dim, value_heads, qk_heads, true);

    let stream = ctx.default_stream();
    let kernels =
        GdnChunkedKernels::new(&ctx, head_dim, value_heads, qk_heads, chunk_len).expect("compiles");
    let mut scratch = kernels.scratch(&stream, seq_len).expect("scratch");

    let mut d_state = stream.clone_htod(&inputs.flat_state()).expect("state");
    let mut d_out = stream
        .alloc_zeros::<f32>(seq_len * value_heads * head_dim)
        .expect("out");
    let d_q = stream.clone_htod(&inputs.flat_q()).expect("q");
    let d_k = stream.clone_htod(&inputs.flat_k()).expect("k");
    let d_v = stream.clone_htod(&inputs.flat_v()).expect("v");
    let d_g = stream.clone_htod(&inputs.flat_log_decay()).expect("g");
    let d_b = stream.clone_htod(&inputs.flat_beta()).expect("beta");

    kernels
        .prefill(
            &stream,
            &mut scratch,
            &mut d_state,
            &d_q,
            &d_k,
            &d_v,
            &d_g,
            &d_b,
            &mut d_out,
            seq_len,
        )
        .expect("prefill");

    let device_out = stream.clone_dtoh(&d_out).expect("out back");
    stream.synchronize().expect("sync");

    let reference_for = |h: usize, qk: usize| -> Vec<f32> {
        let q: Vec<Vec<f32>> = (0..seq_len).map(|t| inputs.q[t][qk].clone()).collect();
        let k: Vec<Vec<f32>> = (0..seq_len).map(|t| inputs.k[t][qk].clone()).collect();
        let v: Vec<Vec<f32>> = (0..seq_len).map(|t| inputs.v[t][h].clone()).collect();
        let decay: Vec<f32> = (0..seq_len).map(|t| inputs.log_decay[t][h]).collect();
        let beta: Vec<f32> = (0..seq_len).map(|t| inputs.beta[t][h]).collect();
        recurrent_forward(
            head_dim,
            &q,
            &k,
            &v,
            &decay,
            &beta,
            Some(&inputs.initial_state[h]),
        )
        .0
        .concat()
    };

    let mut discriminating = 0usize;
    for h in 0..value_heads {
        let candidate: Vec<f32> = (0..seq_len)
            .flat_map(|t| {
                let base = (t * value_heads + h) * head_dim;
                device_out[base..base + head_dim].to_vec()
            })
            .collect();
        let modulo = h % qk_heads;
        let division = h / heads_per_kv;

        let m = compare(&candidate, &reference_for(h, modulo));
        println!(
            "value head {h}: h % qk_heads -> qk {modulo}: max_abs={:.3e} cosine={:.6}",
            m.max_abs_error, m.cosine_similarity,
        );
        assert!(
            m.max_abs_error <= GATE.max_abs_error
                && m.cosine_similarity >= GATE.min_cosine_similarity
                && m.non_finite_count == 0,
            "value head {h} does not read query/key head {modulo}: {m}",
        );

        if division == modulo {
            continue;
        }
        discriminating += 1;
        let d = compare(&candidate, &reference_for(h, division));
        println!(
            "value head {h}: h / heads_per_kv -> qk {division}: max_abs={:.3e} cosine={:.6}",
            d.max_abs_error, d.cosine_similarity,
        );
        assert!(
            d.max_abs_error > 1e-3 && d.cosine_similarity < 0.9,
            "value head {h} is indistinguishable under the two broadcast conventions, \
             so this test proves nothing: {d}",
        );
    }
    assert_eq!(
        discriminating, 2,
        "the geometry must contain heads on which the two mappings disagree",
    );
}

#[test]
fn one_chunk_of_prefill_lands_on_the_same_state_as_one_step_of_decode() {
    // The narrowest possible statement of the prefill/decode agreement that
    // matters at the cache boundary: a single token pushed through the chunked
    // path must leave exactly the state the recurrent path would. If this
    // fails, nothing about a resumed prefix cache is trustworthy, and the
    // failure is one token deep instead of five hundred.
    let Some(ctx) = setup() else { return };
    let head_dim = 128usize;
    let value_heads = 4usize;
    let qk_heads = 2usize;
    let chunk_len = 64usize;

    let inputs = Inputs::generate(11, 1, head_dim, value_heads, qk_heads, true);

    let stream = ctx.default_stream();
    let kernels =
        GdnChunkedKernels::new(&ctx, head_dim, value_heads, qk_heads, chunk_len).expect("compiles");
    let mut scratch = kernels.scratch(&stream, 1).expect("scratch");

    let mut d_state = stream.clone_htod(&inputs.flat_state()).expect("state");
    let mut d_out = stream
        .alloc_zeros::<f32>(value_heads * head_dim)
        .expect("out");
    let d_q = stream.clone_htod(&inputs.flat_q()).expect("q");
    let d_k = stream.clone_htod(&inputs.flat_k()).expect("k");
    let d_v = stream.clone_htod(&inputs.flat_v()).expect("v");
    let d_g = stream.clone_htod(&inputs.flat_log_decay()).expect("g");
    let d_b = stream.clone_htod(&inputs.flat_beta()).expect("beta");

    kernels
        .prefill(
            &stream,
            &mut scratch,
            &mut d_state,
            &d_q,
            &d_k,
            &d_v,
            &d_g,
            &d_b,
            &mut d_out,
            1,
        )
        .expect("prefill");

    let device_out = stream.clone_dtoh(&d_out).expect("out back");
    let device_state = stream.clone_dtoh(&d_state).expect("state back");
    stream.synchronize().expect("sync");

    for h in 0..value_heads {
        // Modulo, not division — see the module docs.
        let qk = h % qk_heads;
        let (rec_out, rec_state) = recurrent_forward(
            head_dim,
            &[inputs.q[0][qk].clone()],
            &[inputs.k[0][qk].clone()],
            &[inputs.v[0][h].clone()],
            &[inputs.log_decay[0][h]],
            &[inputs.beta[0][h]],
            Some(&inputs.initial_state[h]),
        );
        assert_matches(
            &device_out[h * head_dim..(h + 1) * head_dim],
            &rec_out[0],
            &GATE,
        );
        assert_matches(
            &device_state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim],
            &rec_state,
            &GATE,
        );
    }
    println!("single-token chunk matches one recurrent step, outputs and state");
}

#[test]
fn a_sequence_longer_than_the_scratch_is_rejected_rather_than_truncated() {
    // Silently prefilling the first N tokens of a longer prompt would produce a
    // fluent continuation of the wrong text, which is the failure mode that
    // costs the most to notice.
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let kernels = GdnChunkedKernels::new(&ctx, 128, 4, 2, 64).expect("compiles");
    let mut scratch = kernels.scratch(&stream, 64).expect("scratch");

    let mut d_state = stream.alloc_zeros::<f32>(4 * 128 * 128).expect("state");
    let mut d_out = stream.alloc_zeros::<f32>(128 * 4 * 128).expect("out");
    let d_q = stream.alloc_zeros::<f32>(128 * 2 * 128).expect("q");
    let d_k = stream.alloc_zeros::<f32>(128 * 2 * 128).expect("k");
    let d_v = stream.alloc_zeros::<f32>(128 * 4 * 128).expect("v");
    let d_g = stream.alloc_zeros::<f32>(128 * 4).expect("g");
    let d_b = stream.alloc_zeros::<f32>(128 * 4).expect("beta");

    let err = kernels
        .prefill(
            &stream,
            &mut scratch,
            &mut d_state,
            &d_q,
            &d_k,
            &d_v,
            &d_g,
            &d_b,
            &mut d_out,
            128,
        )
        .expect_err("128 tokens must not fit scratch sized for 64");
    println!("rejected as expected: {err}");
}
