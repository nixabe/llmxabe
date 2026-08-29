//! Differential test: the MoE grouped GEMM's **community-quant** prologues
//! against the `xabe-kernels` scalar reference.
//!
//! ## Why this is separate from `moe_differential.rs`
//!
//! That file tests the shipped model's own expert stacks — Q6_K gate/up, Q8_0
//! down — on the real geometry, on real weights, through both the integer
//! tensor-core path and the fp32 one. It is the test that guards the numbers
//! in `docs/BENCHMARKS.md`.
//!
//! This one guards a path those files never take. `Q4_K_M`, the most common
//! community build of either architecture, stores `ffn_*_exps` as a *mixture*
//! of Q4_K, Q5_K and Q6_K depending on the layer, so an engine reading only
//! Q6_K and Q8_0 cannot load one at all. These three formats have no int8
//! tensor-core body — `ExpertQuant::has_mma_body()` is false for each, and
//! the MMA pair selection returns `None` for any pair involving them — so
//! what is under test here is specifically the fp32 `dequant_tile` prologue.
//!
//! ## Why a small synthetic geometry and synthetic weights
//!
//! Both are forced. No file on this host holds a Q4_K expert stack, so the
//! weights have to be built; and requantizing one real layer's stacks (268 M
//! elements per projection) into three formats would dominate the suite's
//! runtime to exercise arithmetic that is identical at any width.
//!
//! What the geometry does have to keep is every property the prologue's
//! correctness rests on: `hidden` and `intermediate` are multiples of the
//! 128-element tile *and* of the 256-element k-quant superblock, so a lane's
//! four consecutive elements never straddle a scale group — which is the
//! assumption that lets these prologues read a scale once per tile.
//!
//! ## What is compared
//!
//! Device output against `naive_forward` over the **dequantized** weights —
//! not the pre-quantization floats. Comparing against those would fold each
//! format's own quantization error into the gate and measure the format
//! rather than the kernel. Routing is asserted to agree first, because a
//! divergence there would make the output comparison meaningless.
//!
//! SKIPS — reporting that it skipped — without a driver or a supported device.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::moe::{
    ExpertQuant, MoeGeometry, MoeKernels, QuantTensor, to_device_layout,
};
use xabe_kernels::compare::{Tolerance, assert_matches, compare};
use xabe_kernels::moe::gemm::{ExpertWeights, naive_forward};
use xabe_kernels::moe::router::route_batch;
use xabe_kernels::quant::{
    QK_K, QK4_0, dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k,
    dequantize_q8_0, quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_0,
};
use xabe_kernels::rng::Xorshift64Star;

/// Small, but with every divisibility the real geometry has.
///
/// `hidden` and `intermediate` are multiples of 256, so they are multiples of
/// the 128-element tile and of a k-quant superblock at once. Shrinking either
/// below 256 would silently stop testing the k-quant path's group-alignment
/// assumption, which is the thing most likely to be wrong.
const HIDDEN: usize = 512;
const INTERMEDIATE: usize = 256;
const EXPERTS: usize = 8;
const TOP_K: usize = 2;
const BLOCK_SIZE: usize = 32;

fn geometry_at(max_tokens: usize) -> MoeGeometry {
    MoeGeometry {
        num_experts: EXPERTS,
        experts_per_token: TOP_K,
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        block_size: BLOCK_SIZE,
        max_tokens,
    }
}

fn device() -> Option<Arc<CudaContext>> {
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

/// Quantize `src` into `quant` and return the serialized bytes plus the fp32
/// the scalar reference gets back out of them.
///
/// The second is what the kernel is claimed to compute with, so it is what
/// the reference must be run over.
fn pack(quant: ExpertQuant, src: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::new();
    let mut back = Vec::with_capacity(src.len());
    match quant {
        ExpertQuant::Q6K => {
            for x in src.as_chunks::<QK_K>().0 {
                let b = quantize_q6_k(x);
                bytes.extend_from_slice(&b.to_bytes());
                back.extend_from_slice(&dequantize_q6_k(&b));
            }
        }
        ExpertQuant::Q8_0 => {
            for x in src.as_chunks::<32>().0 {
                let b = quantize_q8_0(x);
                bytes.extend_from_slice(&b.to_bytes());
                back.extend_from_slice(&dequantize_q8_0(&b));
            }
        }
        ExpertQuant::Q4_0 => {
            for x in src.as_chunks::<QK4_0>().0 {
                let b = quantize_q4_0(x);
                bytes.extend_from_slice(&b.to_bytes());
                back.extend_from_slice(&dequantize_q4_0(&b));
            }
        }
        ExpertQuant::Q4K => {
            for x in src.as_chunks::<QK_K>().0 {
                let b = quantize_q4_k(x);
                bytes.extend_from_slice(&b.to_bytes());
                back.extend_from_slice(&dequantize_q4_k(&b));
            }
        }
        ExpertQuant::Q5K => {
            for x in src.as_chunks::<QK_K>().0 {
                let b = quantize_q5_k(x);
                bytes.extend_from_slice(&b.to_bytes());
                back.extend_from_slice(&dequantize_q5_k(&b));
            }
        }
    }
    (bytes, back)
}

/// Weight-like values: centred on zero, order 0.05, and *different per
/// expert* so a kernel that indexed the wrong expert stack cannot pass.
fn stack(seed: u64, experts: usize, out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    let mut v = Vec::with_capacity(experts * out_dim * in_dim);
    for e in 0..experts {
        let scale = 0.02 * (1.0 + e as f32);
        for x in rng.vec_f32(out_dim * in_dim, -1.0, 1.0) {
            v.push(x * scale);
        }
    }
    v
}

/// The tolerance is looser than the LM head's because this is a two-layer
/// composition — a gate/up GEMM, a SwiGLU, then a down GEMM — summed in a
/// different order on each side and then accumulated over `TOP_K` experts.
/// Cosine is the gate that matters; `max_abs` is scaled to the output
/// magnitude rather than to the weights'.
const GATE: Tolerance = Tolerance {
    max_abs_error: 2e-4,
    max_rel_error: 1.0,
    min_cosine_similarity: 1.0 - 1e-7,
    allow_non_finite: false,
};

/// Every width the routing branches on.
///
/// The community path is selected per projection and then has to *not* be
/// launched twice, and which tuned branch it has to suppress depends on the
/// token count: one token takes the GEMV pair, `<= MOE_NARROW_DECODE_MAX`
/// takes the flat direct kernels, and above that the tiled ones. A bug in any
/// one of those is invisible at the other widths -- the first version of this
/// routing double-dispatched the down projection at *every* width and passed
/// a Q6_K control anyway, because the second kernel overwrote the first with
/// the right answer. Only Q8_0, and only through the community control,
/// showed it.
fn check_widths(quant: ExpertQuant, label: &str, force_community: bool) {
    for (max_tokens, num_tokens) in [(1usize, 1usize), (2, 2), (4, 3), (8, 6)] {
        check_at(
            quant,
            &format!("{label} n={num_tokens}/{max_tokens}"),
            force_community,
            max_tokens,
            num_tokens,
        );
    }
}

fn check_at(
    quant: ExpertQuant,
    label: &str,
    force_community: bool,
    max_tokens: usize,
    num_tokens: usize,
) {
    let Some(ctx) = device() else { return };
    let g = geometry_at(max_tokens);
    let stream = ctx.default_stream();
    let mut kernels = MoeKernels::new(&ctx, g).expect("kernels compile for sm_75");
    // Every format here is compared on the **fp32 path**, which is the one
    // under test: the three community-quant formats have no int8 body at all,
    // and forcing Q6_K and Q8_0 down the same path is what makes them a
    // control for this harness rather than a second measurement of the
    // tensor-core kernels that `moe_differential.rs` already covers.
    //
    // Without this the control fails — and instructively. Q6_K through the
    // int8 path lands at cosine 0.999989, max_abs 5.5e-4 against a reference
    // the fp32 path matches to 2e-4, because that path quantizes the
    // *activations* to int8 as well. That is a property of the integer
    // kernels, not of the Q6_K unpacking, and it is not what this file is
    // asking about.
    kernels.disable_tensor_cores();
    // The community formats have no tuned instantiation, so they always take
    // the `*_community` kernels while Q6_K and Q8_0 take the tuned ones. A
    // control that runs a *different kernel* from the cases is not a control
    // -- see `the_shipped_formats_pass_through_the_community_kernels_too`.
    kernels.force_community(force_community);
    let mut buffers = kernels.buffers(&stream).expect("buffers allocate");

    // --- weights ----------------------------------------------------------
    let gate_src = stack(0x6A7E_0001, EXPERTS, INTERMEDIATE, HIDDEN);
    let up_src = stack(0x0BE0_0002, EXPERTS, INTERMEDIATE, HIDDEN);
    let down_src = stack(0xD0E0_0003, EXPERTS, HIDDEN, INTERMEDIATE);

    let (gate_bytes, gate_ref) = pack(quant, &gate_src);
    let (up_bytes, up_ref) = pack(quant, &up_src);
    let (down_bytes, down_ref) = pack(quant, &down_src);

    let amax = gate_ref.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let nonzero = gate_ref.iter().filter(|v| **v != 0.0).count();
    println!(
        "{label}: {} experts, gate |w| max {amax:.4e}, {:.1}% non-zero",
        EXPERTS,
        100.0 * nonzero as f64 / gate_ref.len() as f64,
    );
    assert!(amax > 0.0, "{label}: requantized gate stack is all zeros");
    assert!(
        nonzero > gate_ref.len() / 2,
        "{label}: over half the requantized stack is zero; the quantizer or \
         the serializer is wrong and every comparison below is vacuous",
    );

    let d_gate = stream
        .clone_htod(&*to_device_layout(quant, &gate_bytes))
        .expect("upload gate");
    let d_up = stream
        .clone_htod(&*to_device_layout(quant, &up_bytes))
        .expect("upload up");
    let d_down = stream
        .clone_htod(&*to_device_layout(quant, &down_bytes))
        .expect("upload down");

    // --- activations and routing -----------------------------------------
    let mut rng = Xorshift64Star::new(0xAC17_0004);
    let hidden_states: Vec<Vec<f32>> = (0..num_tokens)
        .map(|_| rng.vec_f32(HIDDEN, -1.0, 1.0))
        .collect();
    let mut flat_hidden: Vec<f32> = hidden_states.concat();
    flat_hidden.resize(max_tokens * HIDDEN, 0.0);
    let d_hidden = stream.clone_htod(&flat_hidden).expect("upload hidden");

    let mut rng = Xorshift64Star::new(0x0F17_0005);
    let logits: Vec<Vec<f32>> = (0..num_tokens)
        .map(|_| rng.vec_f32(EXPERTS, -1.0, 1.0))
        .collect();
    let mut flat_logits: Vec<f32> = logits.concat();
    flat_logits.resize(max_tokens * EXPERTS, 0.0);
    let d_logits = stream.clone_htod(&flat_logits).expect("upload logits");

    kernels
        .set_valid_tokens(&stream, &mut buffers, num_tokens)
        .expect("valid_tokens");
    kernels
        .route(&stream, &mut buffers, &d_logits)
        .expect("route");
    kernels
        .build_dispatch(&stream, &mut buffers)
        .expect("dispatch");

    // --- device -----------------------------------------------------------
    let mut d_out = stream
        .alloc_zeros::<f32>(max_tokens * HIDDEN)
        .expect("out allocates");
    kernels
        .grouped_forward(
            &stream,
            &mut buffers,
            QuantTensor {
                bytes: &d_gate,
                quant,
            },
            QuantTensor {
                bytes: &d_up,
                quant,
            },
            QuantTensor {
                bytes: &d_down,
                quant,
            },
            &d_hidden,
            &mut d_out,
        )
        .expect("grouped forward");
    stream.synchronize().expect("sync");
    let got = stream.clone_dtoh(&d_out).expect("out back");

    // --- reference over the dequantized weights ---------------------------
    let routing = route_batch(&logits, TOP_K);
    let experts: Vec<ExpertWeights> = (0..EXPERTS)
        .map(|e| {
            let gu = INTERMEDIATE * HIDDEN;
            let dn = HIDDEN * INTERMEDIATE;
            ExpertWeights {
                gate: gate_ref[e * gu..(e + 1) * gu].to_vec(),
                up: up_ref[e * gu..(e + 1) * gu].to_vec(),
                down: down_ref[e * dn..(e + 1) * dn].to_vec(),
            }
        })
        .collect();
    let reference = naive_forward(&hidden_states, &routing, &experts, HIDDEN, INTERMEDIATE);

    for (t, want) in reference.iter().enumerate() {
        let have = &got[t * HIDDEN..(t + 1) * HIDDEN];
        println!("{label}, token {t}: {}", compare(have, want));
        assert_matches(have, want, &GATE);
    }
}

#[test]
fn the_q4_0_expert_prologue_matches_the_scalar_reference() {
    check_widths(ExpertQuant::Q4_0, "q4_0", false);
}

#[test]
fn the_q4_k_expert_prologue_matches_the_scalar_reference() {
    check_widths(ExpertQuant::Q4K, "q4_K", false);
}

#[test]
fn the_q5_k_expert_prologue_matches_the_scalar_reference() {
    check_widths(ExpertQuant::Q5K, "q5_K", false);
}

/// The two formats the shipped files use, on the fp32 path, through the same
/// synthetic harness.
///
/// Not redundant with `moe_differential.rs`: this runs them at a geometry and
/// a weight distribution that file never exercises, so a prologue that was
/// accidentally depending on the real shape shows up here. It is also the
/// control — if these fail, the harness is wrong rather than the new
/// prologues, and the tolerance below is not evidence about anything.
#[test]
fn the_shipped_formats_pass_the_same_harness() {
    check_widths(ExpertQuant::Q6K, "q6_K", false);
    check_widths(ExpertQuant::Q8_0, "q8_0", false);
}

/// The control that actually covers the code under test.
///
/// `the_shipped_formats_pass_the_same_harness` above runs Q6_K and Q8_0
/// through the *tuned* kernels, because those are the only formats with a
/// tuned instantiation. So it proves the harness is sound and proves nothing
/// about the `*_community` kernels the three cases above run on -- a control
/// on a different kernel from the cases, which is the exact trap
/// `docs/TESTING.md` records from the first attempt at these formats.
///
/// This one pins `force_community`, so Q6_K and Q8_0 go through the same four
/// kernels the community formats do. If the community kernels are wrong, this
/// fails on a format whose unpacking is independently known-good, which
/// separates "the kernel is wrong" from "the unpacker is wrong" in one run.
#[test]
fn the_shipped_formats_pass_through_the_community_kernels_too() {
    check_widths(ExpertQuant::Q6K, "q6_K via community", true);
    check_widths(ExpertQuant::Q8_0, "q8_0 via community", true);
}

/// The shared expert's community kernels, against a host SwiGLU MLP.
///
/// `grouped_forward` above drives the *routed* experts only, so without this
/// `moe_shared_ffn_community` and `moe_shared_down_community` would ship with
/// no differential at all -- which `AGENTS.md` says is not done, however
/// carefully the routed pair was checked.
///
/// Same control discipline: Q6_K and Q8_0 are run through the community
/// kernels too, so a green result on the three new formats is not resting on
/// a control that took a different path.
fn check_shared(quant: ExpertQuant, label: &str, force_community: bool, max_tokens: usize) {
    let Some(ctx) = device() else { return };
    let g = geometry_at(max_tokens);
    let stream = ctx.default_stream();
    let mut kernels = MoeKernels::new(&ctx, g).expect("kernels compile for sm_75");
    kernels.disable_tensor_cores();
    kernels.force_community(force_community);
    let mut buffers = kernels.buffers(&stream).expect("buffers allocate");

    let gate_src = stack(0x5EED_0011, 1, INTERMEDIATE, HIDDEN);
    let up_src = stack(0x5EED_0012, 1, INTERMEDIATE, HIDDEN);
    let down_src = stack(0x5EED_0013, 1, HIDDEN, INTERMEDIATE);
    let (gate_bytes, gate_ref) = pack(quant, &gate_src);
    let (up_bytes, up_ref) = pack(quant, &up_src);
    let (down_bytes, down_ref) = pack(quant, &down_src);
    // Through `to_device_layout` like every other upload: Q6_K is re-strided
    // from the file's 210 bytes per superblock to the device's 224, and
    // skipping it makes the tensor look 480 superblocks long instead of 512.
    let d_gate = stream
        .clone_htod(&*to_device_layout(quant, &gate_bytes))
        .expect("gate up");
    let d_up = stream
        .clone_htod(&*to_device_layout(quant, &up_bytes))
        .expect("up up");
    let d_down = stream
        .clone_htod(&*to_device_layout(quant, &down_bytes))
        .expect("down up");

    let mut rng = Xorshift64Star::new(0x_5A5E_0001);
    let mut x = rng.vec_f32(max_tokens * HIDDEN, -1.0, 1.0);
    x.resize(max_tokens * HIDDEN, 0.0);
    let d_x = stream.clone_htod(&x).expect("x up");
    let mut d_out = stream
        .alloc_zeros::<f32>(max_tokens * HIDDEN)
        .expect("out alloc");
    kernels
        .set_valid_tokens(&stream, &mut buffers, max_tokens)
        .expect("valid tokens");
    kernels
        .shared_expert(
            &stream,
            &mut buffers,
            QuantTensor {
                bytes: &d_gate,
                quant,
            },
            QuantTensor {
                bytes: &d_up,
                quant,
            },
            QuantTensor {
                bytes: &d_down,
                quant,
            },
            &d_x,
            &mut d_out,
        )
        .expect("shared expert");
    stream.synchronize().expect("sync");
    let got = stream.clone_dtoh(&d_out).expect("out back");

    // Host reference over the *dequantized* weights, so quantization error is
    // on both sides and what is left is the unpacking and the contraction.
    for t in 0..max_tokens {
        let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];
        let mut want = vec![0.0f32; HIDDEN];
        let mut act = vec![0.0f32; INTERMEDIATE];
        for r in 0..INTERMEDIATE {
            let (mut gsum, mut usum) = (0.0f32, 0.0f32);
            for k in 0..HIDDEN {
                gsum += gate_ref[r * HIDDEN + k] * xt[k];
                usum += up_ref[r * HIDDEN + k] * xt[k];
            }
            act[r] = (gsum / (1.0 + (-gsum).exp())) * usum;
        }
        for h in 0..HIDDEN {
            let mut acc = 0.0f32;
            for j in 0..INTERMEDIATE {
                acc += down_ref[h * INTERMEDIATE + j] * act[j];
            }
            want[h] = acc;
        }
        let have = &got[t * HIDDEN..(t + 1) * HIDDEN];
        println!("{label} shared, token {t}: {}", compare(have, &want));
        assert_matches(have, &want, &GATE);
    }
}

#[test]
fn the_shared_expert_reads_the_community_formats() {
    for (q, l) in [
        (ExpertQuant::Q4_0, "q4_0"),
        (ExpertQuant::Q4K, "q4_K"),
        (ExpertQuant::Q5K, "q5_K"),
    ] {
        for width in [1usize, 4, 8] {
            check_shared(q, l, false, width);
        }
    }
}

#[test]
fn the_shared_expert_control_goes_through_the_community_kernels() {
    for (q, l) in [(ExpertQuant::Q6K, "q6_K"), (ExpertQuant::Q8_0, "q8_0")] {
        for width in [1usize, 4, 8] {
            check_shared(q, l, true, width);
            check_shared(q, l, false, width);
        }
    }
}
