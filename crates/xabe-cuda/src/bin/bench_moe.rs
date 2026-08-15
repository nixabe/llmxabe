//! Wall-clock harness for the MoE path, in isolation from the forward pass.
//!
//! ```sh
//! CUDA_VISIBLE_DEVICES=0 cargo run --release -p xabe-cuda --bin bench_moe
//! ```
//!
//! Times the four public entry points of [`xabe_cuda::kernels::moe`] at the
//! real Qwen3.6 MoE geometry (256 experts, top-8, hidden 2048, intermediate
//! 512, `block_size` 16) across the batch sizes the engine actually presents:
//! 1 (decode), 19, 128 and 512 (prefill). Each batch size gets its own
//! `MoeKernels`, because `forward.rs` builds the geometry with
//! `max_tokens == tokens` — so the launch shapes under test are the ones
//! production uses.
//!
//! ## What the reported bandwidth means
//!
//! Expert weights dominate, so the roofline number is quantized weight bytes
//! moved per second. Two denominators are reported:
//!
//! - **unique** — every *active* expert's stack read exactly once. This is
//!   the floor a tiled grouped GEMM can reach, and what llama.cpp's
//!   `mul_mat_id` reaches.
//! - **naive** — one full weight read per `(token, expert)` pair. This is
//!   what a kernel with no reuse across the tokens sharing an expert must
//!   move, and at 512 tokens it is ~16x the floor.
//!
//! The card is a Quadro RTX 8000: 672 GB/s, 16.3 TFLOP/s fp32.
//!
//! Weights here are synthetic but *shaped* exactly like the file's: Q6_K
//! gate/up, Q8_0 down, whole superblocks, a normal fp16 scale. Timing does
//! not depend on the bit patterns; correctness is gated by
//! `crates/xabe-engine/tests/moe_differential.rs` against the real file.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaStream};
use xabe_cuda::kernels::moe::{ExpertQuant, MoeBuffers, MoeGeometry, MoeKernels, QuantTensor};

/// Card peak, for the fraction-of-roofline column.
const PEAK_GB_S: f64 = 672.0;
const PEAK_TFLOP_S: f64 = 16.3;

/// Batch sizes the report covers.
const BATCHES: [usize; 4] = [1, 19, 128, 512];

/// Grouped-GEMM tile width, matching `forward.rs`'s `MOE_BLOCK_SIZE`.
const BLOCK_SIZE: usize = 16;

/// fp16 bits for 2^-8, an exactly representable, unremarkable scale.
const HALF_SCALE: u16 = 0x1C00;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 33) as u8
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

/// A Q6_K stack of `elements` values, laid out as the GGUF file lays it out.
fn q6_k_stack(elements: usize, seed: u64) -> Vec<u8> {
    let blocks = elements / 256;
    let mut rng = Rng(seed);
    let mut out = vec![0u8; blocks * 210];
    for b in 0..blocks {
        let base = b * 210;
        for i in 0..192 {
            out[base + i] = rng.byte();
        }
        for i in 0..16 {
            // int8 scales in a plausible range; the sign pattern matters to
            // the kernel's signed read, not to its timing.
            out[base + 192 + i] = (rng.byte() % 64).wrapping_sub(32);
        }
        out[base + 208] = (HALF_SCALE & 0xff) as u8;
        out[base + 209] = (HALF_SCALE >> 8) as u8;
    }
    out
}

/// A Q8_0 stack of `elements` values.
fn q8_0_stack(elements: usize, seed: u64) -> Vec<u8> {
    let blocks = elements / 32;
    let mut rng = Rng(seed);
    let mut out = vec![0u8; blocks * 34];
    for b in 0..blocks {
        let base = b * 34;
        out[base] = (HALF_SCALE & 0xff) as u8;
        out[base + 1] = (HALF_SCALE >> 8) as u8;
        for i in 0..32 {
            out[base + 2 + i] = rng.byte();
        }
    }
    out
}

/// Run `f` until it has taken at least `budget`, and report the mean.
fn timed<F: FnMut()>(stream: &Arc<CudaStream>, budget: Duration, mut f: F) -> Duration {
    // One warm-up outside the measurement: the first launch of a freshly
    // loaded module pays JIT and page-in costs that no steady-state step does.
    f();
    stream.synchronize().expect("warm-up sync");

    let mut iters = 0u32;
    let start = Instant::now();
    loop {
        f();
        stream.synchronize().expect("sync");
        iters += 1;
        if start.elapsed() >= budget || iters >= 200 {
            break;
        }
    }
    start.elapsed() / iters
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn main() {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIPPED: no context on device 0: {e}");
            return;
        }
    };
    let stream = ctx.default_stream();

    let probe = MoeGeometry::qwen3_6(BLOCK_SIZE, 1);
    let stack = probe.stack_elements();
    let one_expert = probe.intermediate * probe.hidden;

    println!(
        "uploading synthetic expert stacks ({} experts)...",
        probe.num_experts
    );
    let t0 = Instant::now();
    let d_gate = stream.clone_htod(&q6_k_stack(stack, 1)).expect("gate");
    let d_up = stream.clone_htod(&q6_k_stack(stack, 2)).expect("up");
    let d_down = stream.clone_htod(&q8_0_stack(stack, 3)).expect("down");
    let d_sgate = stream
        .clone_htod(&q6_k_stack(one_expert, 4))
        .expect("sgate");
    let d_sup = stream.clone_htod(&q6_k_stack(one_expert, 5)).expect("sup");
    let d_sdown = stream
        .clone_htod(&q8_0_stack(one_expert, 6))
        .expect("sdown");
    stream.synchronize().expect("sync");
    let routed_bytes = d_gate.len() + d_up.len() + d_down.len();
    println!(
        "  {:.1} MiB routed + {:.1} MiB shared in {:.2?}\n",
        routed_bytes as f64 / (1024.0 * 1024.0),
        (d_sgate.len() + d_sup.len() + d_sdown.len()) as f64 / (1024.0 * 1024.0),
        t0.elapsed(),
    );

    let gate = QuantTensor {
        bytes: &d_gate,
        quant: ExpertQuant::Q6K,
    };
    let up = QuantTensor {
        bytes: &d_up,
        quant: ExpertQuant::Q6K,
    };
    let down = QuantTensor {
        bytes: &d_down,
        quant: ExpertQuant::Q8_0,
    };
    let sgate = QuantTensor {
        bytes: &d_sgate,
        quant: ExpertQuant::Q6K,
    };
    let sup = QuantTensor {
        bytes: &d_sup,
        quant: ExpertQuant::Q6K,
    };
    let sdown = QuantTensor {
        bytes: &d_sdown,
        quant: ExpertQuant::Q8_0,
    };

    println!(
        "{:>7}  {:>9}  {:>10}  {:>10}  {:>10}  {:>8}  {:>10}  {:>9}",
        "tokens", "route", "dispatch", "grouped", "shared", "active", "GB/s uniq", "% peak",
    );

    // `bench_moe 512` restricts the sweep to one batch size, which is what a
    // profiler run wants.
    let only: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let batches: Vec<usize> = if only.is_empty() {
        BATCHES.to_vec()
    } else {
        only
    };

    for tokens in batches {
        let g = MoeGeometry::qwen3_6(BLOCK_SIZE, tokens);
        let kernels = MoeKernels::new(&ctx, g).expect("kernels compile");
        let mut buffers = kernels.buffers(&stream).expect("buffers");

        let mut rng = Rng(0x5EED_B01C ^ tokens as u64);
        let logits: Vec<f32> = (0..g.max_tokens * g.num_experts)
            .map(|_| rng.unit() * 8.0)
            .collect();
        let hidden: Vec<f32> = (0..g.max_tokens * g.hidden).map(|_| rng.unit()).collect();
        let d_logits = stream.clone_htod(&logits).expect("logits");
        let d_hidden = stream.clone_htod(&hidden).expect("hidden");
        let mut d_out = stream
            .alloc_zeros::<f32>(g.max_tokens * g.hidden)
            .expect("out");

        kernels
            .set_valid_tokens(&stream, &mut buffers, tokens)
            .expect("valid_tokens");
        kernels
            .route(&stream, &mut buffers, &d_logits)
            .expect("route");
        kernels
            .build_dispatch(&stream, &mut buffers)
            .expect("dispatch");
        stream.synchronize().expect("sync");

        let active = distinct_experts(&stream, &buffers, tokens * g.experts_per_token);
        let budget = Duration::from_millis(400);

        let t_route = timed(&stream, budget, || {
            kernels
                .route(&stream, &mut buffers, &d_logits)
                .expect("route");
        });
        let t_dispatch = timed(&stream, budget, || {
            kernels
                .build_dispatch(&stream, &mut buffers)
                .expect("dispatch");
        });
        let t_grouped = timed(&stream, budget, || {
            kernels
                .grouped_forward(&stream, &mut buffers, gate, up, down, &d_hidden, &mut d_out)
                .expect("grouped");
        });
        let t_shared = timed(&stream, budget, || {
            kernels
                .shared_expert(
                    &stream,
                    &mut buffers,
                    sgate,
                    sup,
                    sdown,
                    &d_hidden,
                    &mut d_out,
                )
                .expect("shared");
        });

        // Quantized weight bytes an ideal (fully reusing) kernel must move.
        let per_expert = (routed_bytes / g.num_experts) as f64;
        let unique = active as f64 * per_expert;
        let naive = (tokens * g.experts_per_token) as f64 * per_expert;
        let secs = t_grouped.as_secs_f64();
        let gb_uniq = unique / secs / 1e9;
        let gb_naive = naive / secs / 1e9;
        // 2 flops per MAC; gate + up over hidden, down over intermediate.
        let flops = (tokens * g.experts_per_token) as f64
            * g.intermediate as f64
            * g.hidden as f64
            * 2.0
            * 3.0;
        let tflops = flops / secs / 1e12;

        println!(
            "{tokens:>7}  {:>8.3}m  {:>9.3}m  {:>9.3}m  {:>9.3}m  {active:>8}  {gb_uniq:>10.1}  {:>8.1}%",
            ms(t_route),
            ms(t_dispatch),
            ms(t_grouped),
            ms(t_shared),
            100.0 * gb_uniq / PEAK_GB_S,
        );
        println!(
            "         (grouped: {gb_naive:.1} GB/s if every (token,expert) pair \
             re-reads its expert = {:.1}% peak; {tflops:.2} TFLOP/s = {:.1}% of {PEAK_TFLOP_S})",
            100.0 * gb_naive / PEAK_GB_S,
            100.0 * tflops / PEAK_TFLOP_S,
        );
    }
}

/// How many distinct experts this batch routes to, from the device tables.
fn distinct_experts(stream: &Arc<CudaStream>, buffers: &MoeBuffers, pairs: usize) -> usize {
    let ids: Vec<i32> = stream.clone_dtoh(buffers.topk_ids()).expect("ids back");
    stream.synchronize().expect("sync");
    let mut seen = [false; 4096];
    let mut n = 0;
    for &e in &ids[..pairs] {
        let e = e as usize;
        if e < seen.len() && !seen[e] {
            seen[e] = true;
            n += 1;
        }
    }
    n
}
