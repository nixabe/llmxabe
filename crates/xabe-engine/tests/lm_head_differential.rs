//! Differential test: the device LM head GEMV against the `xabe-kernels`
//! scalar fp32 reference, over the **whole** 248,320-entry vocabulary, on
//! the real Q8_0 `output.weight` taken from the model file.
//!
//! This is the last kernel G006 needs before a forward pass can produce
//! logits, and it is the one whose output a user sees directly: everything
//! upstream is checked against a reference, but the LM head's argmax *is*
//! the sampled token. A kernel that is well inside any sensible tolerance
//! and still flips one argmax has produced a different model, and no mean
//! error catches that. So argmax agreement is a separate, exact assertion
//! here, alongside the usual max-abs and cosine gates.
//!
//! ## What is compared, and how strictly
//!
//! | Property | Reference | Gate |
//! |---|---|---|
//! | logits, one token, all 248,320 entries | `gemv` over chunk-dequantized real weights | tolerance |
//! | logits, a 5-token batch, all entries | `gemv_batch` over the same | tolerance |
//! | argmax, every token | `argmax` | **exact** |
//! | batch tile vs single-token path | the device against itself | **bit-identical** |
//!
//! Nothing here can be bit-identical to the host: the reference sums 2,048
//! products sequentially, while the kernel sums 64 per lane in four-element
//! groups and then combines 32 lanes in a shuffle tree, plus nvcc is free to
//! contract the accumulation into an FMA. The tile-versus-single-token
//! comparison *is* exact, because
//! both device paths do the identical arithmetic in the identical order —
//! only the number of accumulators in flight differs — so anything less than
//! equality there means one of the eight template instantiations is not the
//! kernel the others are.
//!
//! ## Why the full vocabulary and not a sample
//!
//! 248,320 x 2,048 is 508 M multiply-accumulates on the host — slow, but it
//! runs, and sampling would undercut the argmax assertion specifically: the
//! maximum is a property of the whole vector, and a sampled comparison
//! cannot see a wrong row unless it happens to sample it. The weights are
//! dequantized in row chunks so the full fp32 tensor (2.03 GiB) is never
//! resident.
//!
//! ## Why real weights
//!
//! Synthetic weights would exercise none of Q8_0's real per-block delta
//! distribution or sign pattern, and none of the subnormal deltas the
//! quantizer emits for near-zero blocks. They would also not confirm what
//! this test asserts against the file's own tensor directory: `output.weight`
//! is Q8_0, `[2048, 248320]`, 540,344,320 bytes, and is a *separate* tensor
//! from `token_embd.weight` — the head is untied, so a forward pass cannot
//! reuse the embedding.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::lm_head::{
    ARGMAX_BLOCKS, HeadTensor, LmHeadGeometry, LmHeadKernels, MAX_BATCH_TILE,
};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_kernels::compare::{Tolerance, assert_matches, compare};
use xabe_kernels::gemv::{argmax, gemv_batch};
use xabe_kernels::quant::dequantize_row_q8_0;
use xabe_kernels::rng::Xorshift64Star;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Role, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Tokens in the batched test.
///
/// Five is deliberately not a power of two. The kernel compiles an entry
/// point for every tile in `1..=8` precisely so an awkward batch does not
/// pay an extra 540 MB pass, and a power-of-two batch would never reach the
/// `b5` instantiation that decision exists for.
const BATCH: usize = 5;

/// Buffer capacity, in tokens. Larger than [`BATCH`] on purpose: it is what
/// proves the launch shape comes from the geometry rather than from the live
/// batch, and it leaves slots that must come back untouched.
const MAX_TOKENS: usize = 8;

/// Vocabulary rows dequantized per host chunk.
///
/// 8,192 rows is 17.8 MiB of Q8_0 in and 64 MiB of fp32 out — enough to
/// amortize the per-call overhead, small enough that the 2.03 GiB full
/// tensor is never resident.
const CHUNK_ROWS: usize = 8192;

/// Timed iterations for the bandwidth figure, after [`WARMUP`] untimed ones.
const ITERS: usize = 20;
const WARMUP: usize = 3;

/// Fixed seeds, so a failure is reproducible from the test name alone.
const SINGLE_SEED: u64 = 0x_5EED_1A01;
const BATCH_SEED: u64 = 0x_5EED_1B02;

/// Tolerance for the device GEMV against the fp32 scalar reference.
///
/// The dequantized *weights* are bit-identical to the reference — the
/// milestone-04 gate proved that for `q * d`, and this kernel reuses the
/// operand order verbatim. What is left is a 2,048-term dot product summed
/// sequentially on the host against 64 terms per lane, in four-element
/// groups, plus a 32-lane shuffle tree on the device, with the device
/// additionally free to contract into an
/// FMA. Both differences favour the *device*; the reference is deliberately
/// the least accurate reasonable order (see `xabe_kernels::gemv`).
///
/// **`max_abs_error` and `min_cosine_similarity` are the gate;
/// `max_rel_error` is not.** `compare()` divides by `max(|reference|, 1e-6)`,
/// and a 248,320-entry logit vector centred near zero has thousands of
/// entries below that floor, for which the ratio reports `abs_error / 1e-6`
/// rather than anything about accuracy. The tests assert below that the
/// element driving `max_rel_error` really is near zero, so the justification
/// fails loudly if it stops holding.
///
/// Measured worst case over the six full-vocabulary comparisons these tests
/// run on the real 248,320 x 2,048 Q8_0 `output.weight`: `max_abs` between
/// 3.70e-6 and 4.41e-6, `cosine` 1.000000 to six places, `max_rel` up to
/// 1.58e-1. The bounds below are ~11x the measured `max_abs` — room for
/// driver and hardware variation, not room for a formulation bug, which
/// would land orders of magnitude away on a logit vector whose own max
/// magnitude is about 2.4 and whose mean magnitude is 0.38.
///
/// `max_rel_error` is set above the measured 1.58e-1 only because it is not
/// the gate; every test asserts separately that the element driving it has a
/// reference magnitude below 1e-3, which on this tensor is under 0.05% of
/// the maximum. The largest such driver seen was 1.28e-5.
const GATE: Tolerance = Tolerance {
    max_abs_error: 5e-5,
    max_rel_error: 1.0,
    min_cosine_similarity: 1.0 - 1e-9,
    allow_non_finite: false,
};

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

/// A context on the only visible device, or `None` with a printed reason.
fn device() -> Option<(Arc<CudaContext>, DeviceInfo)> {
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
    Some((ctx, info))
}

fn device_and_model() -> Option<(Arc<CudaContext>, DeviceInfo, GgufFile)> {
    let (ctx, info) = device()?;
    let path = model_path();
    if !path.exists() {
        println!("SKIPPED: model file not found at {}", path.display());
        return None;
    }
    Some((ctx, info, GgufFile::open(&path).expect("valid GGUF v3")))
}

fn geometry() -> LmHeadGeometry {
    let config = ModelConfig::qwen3_6_35b_a3b();
    LmHeadGeometry {
        hidden: config.hidden_size as usize,
        vocab: config.vocab_size as usize,
        max_tokens: MAX_TOKENS,
    }
}

/// Guard against a tensor that would compare perfectly while proving nothing.
fn assert_carries_signal(name: &str, v: &[f32]) {
    let nonzero = v.iter().filter(|x| **x != 0.0).count();
    let frac = nonzero as f64 / v.len() as f64;
    assert!(
        frac > 0.25,
        "{name}: only {:.1}% of {} sampled values are non-zero — an all-zero \
         tensor would pass every comparison below vacuously",
        frac * 100.0,
        v.len(),
    );
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{name}: contains a non-finite value",
    );
}

/// The real `output.weight` bytes, with the file's own claims about it
/// asserted rather than assumed.
fn lm_head_bytes<'a>(file: &'a GgufFile, g: &LmHeadGeometry) -> &'a [u8] {
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(file).expect("schema must resolve");
    let entry = directory
        .find(Role::LmHead, None)
        .expect("output.weight present — the LM head is untied");
    println!(
        "{}: {} {:?}",
        entry.spec.name,
        entry.info.ggml_type.name(),
        entry.spec.dims,
    );
    assert_eq!(
        entry.info.ggml_type,
        GgmlType::Q8_0,
        "the LM head is not Q8_0; docs/MODEL.md's 540 MB/token figure and this \
         kernel's only dequant prologue both assume it is",
    );
    let bytes = file
        .tensor_bytes(&entry.spec.name)
        .expect("tensor readable through the mmap");
    assert_eq!(
        bytes.len(),
        g.weight_bytes(),
        "output.weight is {} bytes, not the {} that {} x {} at Q8_0 implies",
        bytes.len(),
        g.weight_bytes(),
        g.vocab,
        g.hidden,
    );
    bytes
}

/// The scalar reference over the whole vocabulary, dequantizing the weight in
/// row chunks so the 2.03 GiB fp32 tensor is never resident.
///
/// Returns `[xs.len()][vocab]` plus a sample of the dequantized weights, so
/// the caller can prove the weights it multiplied by carried signal.
fn host_logits(bytes: &[u8], g: &LmHeadGeometry, xs: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut out = vec![vec![0.0f32; g.vocab]; xs.len()];
    let mut sample: Vec<f32> = Vec::new();
    let mut row = 0usize;
    while row < g.vocab {
        let rows = CHUNK_ROWS.min(g.vocab - row);
        let span = row * g.row_bytes()..(row + rows) * g.row_bytes();
        let w = dequantize_row_q8_0(&bytes[span]).expect("reference q8_0 unpacking");
        if sample.is_empty() {
            sample.extend_from_slice(&w[..8192]);
        }
        for (t, part) in gemv_batch(&w, rows, g.hidden, xs).into_iter().enumerate() {
            out[t][row..row + rows].copy_from_slice(&part);
        }
        row += rows;
    }
    (out, sample)
}

/// Random hidden states, plus the same rows padded to the buffer capacity.
fn hidden_states(g: &LmHeadGeometry, seed: u64, tokens: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let live: Vec<Vec<f32>> = (0..tokens)
        .map(|_| rng.vec_f32(g.hidden, -1.0, 1.0))
        .collect();
    let mut flat: Vec<f32> = live.concat();
    flat.resize(g.max_tokens * g.hidden, 0.0);
    (live, flat)
}

/// Upload the head once and report how long it took; 540 MB over PCIe is the
/// dominant cost of this test file and worth saying out loud.
fn upload_head(stream: &Arc<CudaStream>, bytes: &[u8]) -> CudaSlice<u8> {
    let t = Instant::now();
    let d = stream.clone_htod(bytes).expect("upload output.weight");
    stream.synchronize().expect("sync");
    let secs = t.elapsed().as_secs_f64();
    println!(
        "uploaded {:.1} MiB of Q8_0 LM head in {:.2?} ({:.1} GB/s over PCIe)",
        bytes.len() as f64 / (1024.0 * 1024.0),
        t.elapsed(),
        bytes.len() as f64 / secs / 1.0e9,
    );
    d
}

/// Report both metrics and the evidence that `max_rel_error` is a floor
/// artefact rather than a measurement, per `.omc/handoffs/team-plan.md`.
fn report(label: &str, candidate: &[f32], reference: &[f32]) {
    let result = compare(candidate, reference);
    println!("{label}: {result}");
    println!(
        "  |ref|: max {:.4e}, mean {:.4e}",
        reference.iter().fold(0.0f32, |m, v| m.max(v.abs())),
        reference.iter().map(|v| v.abs()).sum::<f32>() / reference.len() as f32,
    );
    let driver = reference[result.max_rel_error_index].abs();
    println!("  max_rel driven by reference value {driver:.3e} (compare()'s floor is 1e-6)",);
    assert!(
        driver < 1e-3,
        "{label}: max_rel_error {:.3e} sits on a reference value of {driver:.3e}, \
         large enough for the ratio to mean something — max_abs_error and cosine \
         alone are no longer a sufficient gate",
        result.max_rel_error,
    );
    assert_matches(candidate, reference, &GATE);
}

/// Exact argmax agreement, with the margin printed so a knife-edge case is
/// visible rather than silently lucky.
fn assert_argmax_agrees(label: &str, candidate: &[f32], reference: &[f32]) {
    let want = argmax(reference);
    let got = argmax(candidate);
    let mut sorted: Vec<f32> = reference.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).expect("no NaNs in the reference"));
    let margin = sorted[0] - sorted[1];
    println!(
        "{label}: argmax {got} (device) vs {want} (reference); top-1 {:.6}, \
         top-2 {:.6}, margin {margin:.3e}",
        sorted[0], sorted[1],
    );
    assert_eq!(
        got, want,
        "{label}: the sampled token differs. This is the one disagreement a \
         tolerance cannot absorb — it is a different model output, not a \
         rounding difference",
    );
}

// ---------------------------------------------------------------------------
// one token, the whole vocabulary
// ---------------------------------------------------------------------------

#[test]
fn device_lm_head_matches_the_scalar_reference_over_the_whole_vocabulary() {
    let Some((ctx, info, file)) = device_and_model() else {
        return;
    };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = LmHeadKernels::new(&ctx, g).expect("kernels must compile for sm_75");

    println!(
        "geometry: hidden {}, vocab {} (untied), {} B/row, {} B total; \
         one warp per row is a {}x oversubscription of {} resident warps",
        g.hidden,
        g.vocab,
        g.row_bytes(),
        g.weight_bytes(),
        g.occupancy_surplus(),
        72 * 32,
    );
    println!(
        "device: {} ({}), {:.0} GB/s peak",
        info.name,
        info.compute_capability.sm_arch(),
        info.peak_bandwidth_gb_s(),
    );

    let bytes = lm_head_bytes(&file, &g);
    let d_weight = upload_head(&stream, bytes);
    let weight = HeadTensor::q8_0(&d_weight);

    let (live, flat) = hidden_states(&g, SINGLE_SEED, 1);
    assert_carries_signal("hidden state", &live[0]);
    let d_hidden = stream.clone_htod(&flat).expect("upload hidden");
    let mut d_logits = stream
        .alloc_zeros::<f32>(g.max_tokens * g.vocab)
        .expect("logits allocate");

    kernels
        .forward(&stream, weight, &d_hidden, 1, &mut d_logits)
        .expect("lm head forward");
    let full = stream.clone_dtoh(&d_logits).expect("logits back");
    stream.synchronize().expect("sync");
    let device_logits = &full[..g.vocab];

    // --- bandwidth --------------------------------------------------------
    for _ in 0..WARMUP {
        kernels
            .forward(&stream, weight, &d_hidden, 1, &mut d_logits)
            .expect("warmup");
    }
    stream.synchronize().expect("sync");
    let t = Instant::now();
    for _ in 0..ITERS {
        kernels
            .forward(&stream, weight, &d_hidden, 1, &mut d_logits)
            .expect("timed");
    }
    stream.synchronize().expect("sync");
    let per_call = t.elapsed().as_secs_f64() / ITERS as f64;
    let achieved = g.weight_bytes() as f64 / per_call / 1.0e9;
    let peak = info.peak_bandwidth_gb_s();
    println!(
        "decode (1 token): {:.3} ms/call, {:.1} GB/s of weight traffic, \
         {:.1}% of the card's {:.0} GB/s spec bandwidth",
        per_call * 1.0e3,
        achieved,
        achieved / peak * 100.0,
        peak,
    );
    println!(
        "  roofline floor for one pass: {:.3} ms; activations and logits add \
         {:.2}% on top of the {} weight bytes",
        g.weight_bytes() as f64 / (peak * 1.0e9) * 1.0e3,
        ((g.hidden + g.vocab) * 4) as f64 / g.weight_bytes() as f64 * 100.0,
        g.weight_bytes(),
    );
    assert!(
        achieved > 0.0 && per_call.is_finite(),
        "the bandwidth measurement did not produce a number",
    );

    // --- host reference over all 248,320 rows -----------------------------
    let t = Instant::now();
    let (reference, weight_sample) = host_logits(bytes, &g, &live);
    println!(
        "host reference: {} rows x {} in {:.2?} ({:.0} M MAC/s)",
        g.vocab,
        g.hidden,
        t.elapsed(),
        g.elements() as f64 / t.elapsed().as_secs_f64() / 1.0e6,
    );
    assert_carries_signal("dequantized Q8_0 weights", &weight_sample);
    assert_carries_signal("reference logits", &reference[0]);

    report(
        "logits, 1 token, all 248,320 entries",
        device_logits,
        &reference[0],
    );
    assert_argmax_agrees("1 token", device_logits, &reference[0]);

    // Slots the live step never used must be untouched.
    assert!(
        full[g.vocab..].iter().all(|&v| v == 0.0),
        "the kernel wrote past the one live token",
    );
}

// ---------------------------------------------------------------------------
// a batch, and the tile that serves it in one pass
// ---------------------------------------------------------------------------

#[test]
fn a_batch_of_tokens_matches_the_reference_and_costs_one_pass_over_the_weights() {
    let Some((ctx, info, file)) = device_and_model() else {
        return;
    };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = LmHeadKernels::new(&ctx, g).expect("compiles");

    let bytes = lm_head_bytes(&file, &g);
    let d_weight = upload_head(&stream, bytes);
    let weight = HeadTensor::q8_0(&d_weight);

    let (live, flat) = hidden_states(&g, BATCH_SEED, BATCH);
    assert_carries_signal("hidden states", &live.concat());
    let d_hidden = stream.clone_htod(&flat).expect("upload hidden");
    let mut d_logits = stream
        .alloc_zeros::<f32>(g.max_tokens * g.vocab)
        .expect("logits allocate");

    // A compile error rather than a test failure if BATCH ever grows past
    // what one pass over the weights can serve — the assertion below about
    // pass counts would otherwise start testing a different claim.
    const { assert!(BATCH <= MAX_BATCH_TILE) };
    assert_eq!(
        g.weight_passes(BATCH),
        1,
        "{BATCH} tokens must be one pass over the weights; a power-of-two-only \
         tiling would need two",
    );

    kernels
        .forward(&stream, weight, &d_hidden, BATCH, &mut d_logits)
        .expect("lm head forward");
    let full = stream.clone_dtoh(&d_logits).expect("logits back");
    stream.synchronize().expect("sync");

    // --- host reference for every token, whole vocabulary ------------------
    let t = Instant::now();
    let (reference, weight_sample) = host_logits(bytes, &g, &live);
    println!(
        "host reference: {BATCH} tokens x {} rows in {:.2?}",
        g.vocab,
        t.elapsed(),
    );
    assert_carries_signal("dequantized Q8_0 weights", &weight_sample);

    for t in 0..BATCH {
        let candidate = &full[t * g.vocab..(t + 1) * g.vocab];
        assert_carries_signal(&format!("reference logits, token {t}"), &reference[t]);
        report(
            &format!("logits, token {t} of {BATCH}"),
            candidate,
            &reference[t],
        );
        assert_argmax_agrees(&format!("token {t}"), candidate, &reference[t]);
    }

    // Distinct hidden states must give distinct argmaxes; if they did not,
    // every argmax assertion above could be passing on the same row.
    let picks: Vec<usize> = (0..BATCH)
        .map(|t| argmax(&full[t * g.vocab..(t + 1) * g.vocab]))
        .collect();
    println!("argmax per token: {picks:?}");
    assert!(
        picks
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 1,
        "every token picked the same row — the batch is not being indexed",
    );

    assert!(
        full[BATCH * g.vocab..].iter().all(|&v| v == 0.0),
        "the kernel wrote past the {BATCH} live tokens",
    );

    // --- the tile is the same arithmetic as the single-token path ----------
    //
    // Exact, not tolerance: the `b5` instantiation sums each dot product in
    // the identical order `b1` does — only the number of accumulators in
    // flight differs — so any disagreement means one of the eight template
    // instantiations is not the kernel the others are. This is what makes
    // the tiling safe to use for prefill without re-verifying every width
    // against the host.
    for t in 0..BATCH {
        let mut one = live[t].clone();
        one.resize(g.max_tokens * g.hidden, 0.0);
        let d_one = stream.clone_htod(&one).expect("upload single");
        let mut d_single = stream
            .alloc_zeros::<f32>(g.max_tokens * g.vocab)
            .expect("logits allocate");
        kernels
            .forward(&stream, weight, &d_one, 1, &mut d_single)
            .expect("single-token forward");
        let single = stream.clone_dtoh(&d_single).expect("back");
        stream.synchronize().expect("sync");
        assert_eq!(
            &single[..g.vocab],
            &full[t * g.vocab..(t + 1) * g.vocab],
            "token {t}: the b{BATCH} tile disagrees with the b1 path, which is \
             the same arithmetic in the same order",
        );
    }
    println!(
        "the b{BATCH} tile is bit-identical to {BATCH} b1 launches on all {} entries",
        g.vocab,
    );

    // --- what the tiling is worth -----------------------------------------
    let peak = info.peak_bandwidth_gb_s();
    let mut time = |tokens: usize| {
        for _ in 0..WARMUP {
            kernels
                .forward(&stream, weight, &d_hidden, tokens, &mut d_logits)
                .expect("warmup");
        }
        stream.synchronize().expect("sync");
        let t = Instant::now();
        for _ in 0..ITERS {
            kernels
                .forward(&stream, weight, &d_hidden, tokens, &mut d_logits)
                .expect("timed");
        }
        stream.synchronize().expect("sync");
        t.elapsed().as_secs_f64() / ITERS as f64
    };

    let one = time(1);
    println!(
        "{:>6}  {:>10}  {:>10}  {:>12}  {:>10}",
        "tokens", "passes", "ms/call", "GB/s (wgt)", "% of peak",
    );
    for tokens in [1usize, 2, BATCH, MAX_BATCH_TILE] {
        let secs = time(tokens);
        let read = g.weight_bytes_read(tokens) as f64;
        println!(
            "{tokens:>6}  {:>10}  {:>10.3}  {:>12.1}  {:>9.1}%",
            g.weight_passes(tokens),
            secs * 1.0e3,
            read / secs / 1.0e9,
            read / secs / 1.0e9 / peak * 100.0,
        );
        assert_eq!(g.weight_passes(tokens), tokens.div_ceil(MAX_BATCH_TILE));
    }
    let eight = time(MAX_BATCH_TILE);
    println!(
        "batch tiling: {} tokens in {:.3} ms against {:.3} ms for {} separate \
         single-token calls — {:.2}x, against the {:.2}x that reading the \
         weights once instead of {} times allows",
        MAX_BATCH_TILE,
        eight * 1.0e3,
        one * MAX_BATCH_TILE as f64 * 1.0e3,
        MAX_BATCH_TILE,
        one * MAX_BATCH_TILE as f64 / eight,
        MAX_BATCH_TILE as f64,
        MAX_BATCH_TILE,
    );
}

// ---------------------------------------------------------------------------
// the three-token row tiles
// ---------------------------------------------------------------------------

/// The row-tiled three-token entry points (`b3r2`, `b3r4`) against the
/// untiled path and against three `b1` launches — **bit-identical**, all
/// 248,320 entries, on the real head.
///
/// The row tile changes which warp owns a row and how many accumulators are
/// in flight; it does not change any row's arithmetic or its order. So this
/// is the same exactness claim the `b5`-versus-`b1` check makes, extended to
/// the N=3 decode path that actually serves the three-sequence goal shape.
/// `b1` is itself gated against the scalar host reference above, which is
/// what lets this test be exact instead of re-deriving a 3 x 508 M MAC host
/// reference.
///
/// SKIPS — reporting that it skipped — without a driver, a supported device,
/// or the model file.
#[test]
fn the_row_tiled_three_token_paths_are_bit_identical_to_the_untiled_kernel() {
    let Some((ctx, _info, file)) = device_and_model() else {
        return;
    };
    let g = geometry();
    let stream = ctx.default_stream();
    let untiled = LmHeadKernels::with_row_tile(&ctx, g, None).expect("compiles untiled");
    let rt2 = LmHeadKernels::with_row_tile(&ctx, g, Some(2)).expect("compiles rt2");
    let rt4 = LmHeadKernels::with_row_tile(&ctx, g, Some(4)).expect("compiles rt4");

    let bytes = lm_head_bytes(&file, &g);
    let d_weight = upload_head(&stream, bytes);
    let weight = HeadTensor::q8_0(&d_weight);

    const TOKENS: usize = 3;
    let (live, flat) = hidden_states(&g, BATCH_SEED ^ 0x33, TOKENS);
    assert_carries_signal("hidden states", &live.concat());
    let d_hidden = stream.clone_htod(&flat).expect("upload hidden");

    let run = |kernels: &LmHeadKernels, label: &str| -> Vec<f32> {
        let mut d_logits = stream
            .alloc_zeros::<f32>(g.max_tokens * g.vocab)
            .expect("logits allocate");
        kernels
            .forward(&stream, weight, &d_hidden, TOKENS, &mut d_logits)
            .expect(label);
        let full = stream.clone_dtoh(&d_logits).expect("logits back");
        stream.synchronize().expect("sync");
        assert!(
            full[TOKENS * g.vocab..].iter().all(|&v| v == 0.0),
            "{label}: wrote past the {TOKENS} live tokens",
        );
        full
    };

    let base = run(&untiled, "untiled b3");
    let two = run(&rt2, "b3r2");
    let four = run(&rt4, "b3r4");
    assert_carries_signal("untiled b3 logits", &base[..TOKENS * g.vocab]);

    for (label, candidate) in [("b3r2", &two), ("b3r4", &four)] {
        for t in 0..TOKENS {
            assert_eq!(
                &candidate[t * g.vocab..(t + 1) * g.vocab],
                &base[t * g.vocab..(t + 1) * g.vocab],
                "token {t}: {label} disagrees with the untiled b3 path, which \
                 is the same per-row arithmetic in the same order",
            );
        }
    }

    // And against b1, which is the instantiation the host reference gates.
    for t in 0..TOKENS {
        let mut one = live[t].clone();
        one.resize(g.max_tokens * g.hidden, 0.0);
        let d_one = stream.clone_htod(&one).expect("upload single");
        let mut d_single = stream
            .alloc_zeros::<f32>(g.max_tokens * g.vocab)
            .expect("logits allocate");
        untiled
            .forward(&stream, weight, &d_one, 1, &mut d_single)
            .expect("single-token forward");
        let single = stream.clone_dtoh(&d_single).expect("back");
        stream.synchronize().expect("sync");
        assert_eq!(
            &single[..g.vocab],
            &base[t * g.vocab..(t + 1) * g.vocab],
            "token {t}: the b3 tile disagrees with the b1 path",
        );
    }

    // Informal timing so the retained tile's win is visible where the gate
    // runs; the whole-pass A/B in docs/BENCHMARKS.md is the real evidence.
    let time = |kernels: &LmHeadKernels, label: &str| {
        let mut d_logits = stream
            .alloc_zeros::<f32>(g.max_tokens * g.vocab)
            .expect("logits allocate");
        for _ in 0..WARMUP {
            kernels
                .forward(&stream, weight, &d_hidden, TOKENS, &mut d_logits)
                .expect("warmup");
        }
        stream.synchronize().expect("sync");
        let t = Instant::now();
        for _ in 0..ITERS {
            kernels
                .forward(&stream, weight, &d_hidden, TOKENS, &mut d_logits)
                .expect("timed");
        }
        stream.synchronize().expect("sync");
        let per_call = t.elapsed().as_secs_f64() / ITERS as f64;
        println!("{label}: {:.3} ms/call at 3 tokens", per_call * 1.0e3);
        per_call
    };
    time(&untiled, "untiled b3");
    time(&rt2, "b3r2");
    time(&rt4, "b3r4");

    println!(
        "b3r2 and b3r4 are bit-identical to the untiled b3 and to three b1 \
         launches on all {} entries x {TOKENS} tokens",
        g.vocab,
    );
}

// ---------------------------------------------------------------------------
// the device argmax
// ---------------------------------------------------------------------------

/// The device reduction against `xabe_kernels::gemv::argmax`, on inputs
/// chosen for the ways a two-pass reduction can disagree with a sequential
/// scan.
///
/// The GEMV tests above compare the device's *logits* and then argmax them on
/// the host. This tests the other half: the kernel that turns those logits
/// into a token id without moving 993 KiB across PCIe first. It needs no
/// model file — the reduction does not care where the floats came from —
/// so it runs anywhere a device does.
///
/// The cases are the ones a partitioned reduction gets wrong:
///
/// - **ties**, which is the whole reason the reference documents a tie-break.
///   A constant vector must answer 0; a vector whose maximum appears in two
///   partitions must answer the lower index regardless of which block found
///   it first.
/// - **a maximum in the last partition**, and one in the first, so a
///   grid-stride loop that drops its tail or seeds from the wrong element is
///   caught in both directions.
/// - **all-NaN**, where `NaN > NaN` is false and the reference never moves
///   off index 0. A kernel seeded from a `-INFINITY` sentinel answers
///   something else, and nothing in a forward pass would show it.
/// - **lengths that are not multiples of the block or the grid**, including
///   1, so the bounds are exercised rather than assumed.
///
/// SKIPS — reporting that it skipped — without a driver or a supported device.
#[test]
fn the_device_argmax_agrees_with_the_scalar_reference() {
    let Some((ctx, _info)) = device() else {
        return;
    };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = LmHeadKernels::new(&ctx, g).expect("compiles");

    let mut rng = Xorshift64Star::new(0x5EED_A46E_A700_0001);
    let mut cases: Vec<(String, Vec<f32>)> = Vec::new();

    for n in [1usize, 2, 31, 255, 256, 257, 1023, 4096, 131_073, g.vocab] {
        cases.push((format!("random n={n}"), rng.vec_f32(n, -8.0, 8.0)));
    }
    cases.push(("all equal".into(), vec![0.5f32; g.vocab]));
    cases.push(("all zero".into(), vec![0.0f32; g.vocab]));
    cases.push(("all NaN".into(), vec![f32::NAN; 4096]));

    // A tie between the first and last partitions. `ARGMAX_BLOCKS * 256` is
    // the widest the first pass strides, so these two indices are guaranteed
    // to land in different blocks at this length.
    let mut tied = rng.vec_f32(g.vocab, -1.0, 1.0);
    tied[7] = 9.0;
    tied[g.vocab - 3] = 9.0;
    cases.push(("tie across partitions".into(), tied));

    // The maximum alone in the very last element, and alone in the very
    // first: a dropped tail and a mis-seeded accumulator fail one each.
    let mut last = rng.vec_f32(g.vocab, -1.0, 1.0);
    last[g.vocab - 1] = 100.0;
    cases.push(("max at the end".into(), last));
    let mut first = rng.vec_f32(g.vocab, -1.0, 1.0);
    first[0] = 100.0;
    cases.push(("max at index 0".into(), first));

    // Infinities, which a reduction that sums or averages anything would
    // turn into NaN and lose.
    let mut infs = rng.vec_f32(g.vocab, -1.0, 1.0);
    infs[11] = f32::NEG_INFINITY;
    infs[g.vocab / 2] = f32::INFINITY;
    cases.push(("with infinities".into(), infs));

    let mut pv = stream.alloc_zeros::<f32>(ARGMAX_BLOCKS).expect("alloc pv");
    let mut pi = stream.alloc_zeros::<i32>(ARGMAX_BLOCKS).expect("alloc pi");
    let mut out = stream.alloc_zeros::<i32>(1).expect("alloc out");

    for (label, values) in &cases {
        let d = stream.clone_htod(values).expect("upload");
        kernels
            .argmax(&stream, &d, values.len(), &mut pv, &mut pi, &mut out)
            .expect("argmax launches");
        let got = stream.clone_dtoh(&out).expect("read back")[0];
        stream.synchronize().expect("sync");
        let want = argmax(values) as i32;
        assert_eq!(
            got,
            want,
            "{label} (len {}): device argmax {got}, reference {want}",
            values.len(),
        );
    }
    println!(
        "device argmax agrees with the reference on all {} cases, including \
         ties across partitions, an all-NaN vector, and the full {}-entry \
         vocabulary",
        cases.len(),
        g.vocab,
    );
}
