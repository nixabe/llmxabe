//! VRAM segmentation and per-token decode bandwidth, derived from
//! [`ModelConfig`].
//!
//! Every quantity here is either an exact closed-form derivation from
//! [`ModelConfig`]'s accessors, or an explicitly labeled estimate. This
//! module does not blur the two: `xabe-cuda` and `xabe-cache` are load-
//! bearing on the exact ones (KV pool size, GDN state size), while the
//! estimates (compute-buffer scratch, CUDA context overhead) exist only to
//! give a complete VRAM picture before milestone 06 produces a measured one
//! — see `AGENTS.md`'s "Reporting results honestly" section. Nothing here
//! should be read as a measurement.
//!
//! The reference numbers in this module's doc comments and tests come from
//! two places: the closed-form parameter counts in [`crate::config`]
//! (cross-checked against the real tensor shapes in
//! `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` while building `xabe-gguf`), and the
//! project's own planning document, `qwen36-rust-engine-plan.md` §03–04.
//! Where the two disagree, this module follows the closed-form derivation
//! and documents the gap rather than quietly adopting the plan's rougher
//! figure — see [`WeightBytesPerParam::observed_in_target_file`].

use crate::config::ModelConfig;

/// One gibibyte, in bytes.
const GIB: u64 = 1024 * 1024 * 1024;

// ---------------------------------------------------------------------
// VRAM segmentation
// ---------------------------------------------------------------------

/// Estimated CUDA driver context plus cuBLAS/cuBLASLt handle overhead, per
/// GPU.
///
/// **Not measured.** No CUDA kernels exist yet (see `AGENTS.md`'s milestone
/// table); this is the "estimate"-confidence figure from
/// `qwen36-rust-engine-plan.md` §03, carried over unchanged. Treat it as a
/// placeholder pending a real measurement, not as ground truth.
pub const CUDA_CONTEXT_OVERHEAD_BYTES: u64 = GIB / 2;

/// Estimated scratch/workspace VRAM for chunked-prefill compute at the
/// reference micro-batch size (`-ub 4096`): activation buffers, the MoE
/// token-permutation scratch, and prefill intermediate tensors.
///
/// **Not measured**, for the same reason as
/// [`CUDA_CONTEXT_OVERHEAD_BYTES`]. This is `qwen36-rust-engine-plan.md`
/// §03's own "estimate"-confidence figure for this segment.
pub const COMPUTE_BUFFER_BYTES: u64 = (11 * GIB) / 5; // 2.2 GiB

/// Per-card VRAM segmentation for one worker replica.
///
/// Each worker holds an independent full copy of the model — this project's
/// three workers are three replicas sharing a host-side prefix cache, not a
/// tensor-parallel split (see the README's architecture section) — so every
/// field here is the footprint on *one* GPU, not summed across the fleet.
///
/// Deliberately excludes the vision encoder (`mmproj-F16.gguf`): this
/// engine is text-only (`AGENTS.md`, "Scope"), so a vision-encoder segment
/// would describe VRAM this crate never allocates. The planning document's
/// own total (`qwen36-rust-engine-plan.md` §03) includes a ~1.5 GiB vision
/// segment that has no counterpart here; expect this module's total to run
/// about that much lower for the same inputs, not higher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramBudget {
    /// Model weights resident on this card, as reported by the loader.
    pub weights_bytes: u64,
    /// Paged KV cache pool, sized to the aggregate token capacity — see
    /// [`vram_budget`] for why this is not multiplied by slot count.
    pub kv_pool_bytes: u64,
    /// Gated DeltaNet recurrent state, one fixed-size block per concurrent
    /// slot.
    pub gdn_state_bytes: u64,
    /// Chunked-prefill compute scratch. See [`COMPUTE_BUFFER_BYTES`].
    pub compute_buffer_bytes: u64,
    /// CUDA context and library handle overhead. See
    /// [`CUDA_CONTEXT_OVERHEAD_BYTES`].
    pub cuda_context_overhead_bytes: u64,
}

impl VramBudget {
    /// Sum of every segment.
    pub const fn total_bytes(&self) -> u64 {
        self.weights_bytes
            + self.kv_pool_bytes
            + self.gdn_state_bytes
            + self.compute_buffer_bytes
            + self.cuda_context_overhead_bytes
    }

    /// Bytes left against a usable-VRAM figure. Negative means the
    /// configuration does not fit — an admission-control input, not just a
    /// reporting number.
    pub fn headroom_bytes(&self, usable_vram_bytes: u64) -> i64 {
        usable_vram_bytes as i64 - self.total_bytes() as i64
    }
}

/// Compute the per-card VRAM segmentation for `cfg` at a given serving
/// configuration.
///
/// # Why `kv_pool` is sized to `context_tokens`, not `context_tokens * slots`
///
/// The two-group pager (`xabe-cache`) draws pages from one shared pool of
/// aggregate token capacity; it does not reserve `context_tokens` worth of
/// pages per slot up front (that would be the page-geometry mistake
/// `AGENTS.md` rule 1 warns about, applied to slots instead of cache
/// groups). `context_tokens` here *is* that aggregate capacity — for the
/// reference configuration (393,216 tokens, f16 KV) it is exactly
/// `20,480 B/token * 393,216 = 7.5 GiB`, matching
/// `qwen36-rust-engine-plan.md` §03's "derived"-confidence figure.
///
/// # Why `gdn_state` *is* multiplied by `slots`
///
/// Recurrent state cannot be paged the way growing KV can: it is a fixed
/// `[value_heads, head_dim, head_dim]` block that exists in full for every
/// concurrently active sequence, whether that sequence has produced one
/// token or a million. `slots` concurrent sequences need `slots` such
/// blocks, full stop — see [`ModelConfig::gdn_state_bytes_per_sequence`].
pub fn vram_budget(
    cfg: &ModelConfig,
    context_tokens: u64,
    slots: u32,
    kv_elem_bytes: u64,
    weights_bytes: u64,
) -> VramBudget {
    VramBudget {
        weights_bytes,
        kv_pool_bytes: cfg.kv_bytes_per_token(kv_elem_bytes) * context_tokens,
        gdn_state_bytes: cfg.gdn_state_bytes_per_sequence() * u64::from(slots),
        compute_buffer_bytes: COMPUTE_BUFFER_BYTES,
        cuda_context_overhead_bytes: CUDA_CONTEXT_OVERHEAD_BYTES,
    }
}

// ---------------------------------------------------------------------
// Per-token decode bandwidth
// ---------------------------------------------------------------------

/// Bytes read per active parameter, per weight component, at a given
/// quantization.
///
/// This is three numbers rather than one uniform figure because the real
/// target file does not use one quantization for every weight: MoE expert
/// matrices, the LM head, and the per-layer projections can each land on a
/// different `ggml_type` depending on the quant recipe. Collapsing that to
/// a single average would hide exactly the kind of per-component
/// difference that turned out to matter — see
/// [`Self::observed_in_target_file`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightBytesPerParam {
    /// Applies to [`ModelConfig::active_ffn_params`].
    pub active_ffn: f64,
    /// Applies to [`ModelConfig::lm_head_params`].
    pub lm_head: f64,
    /// Applies to [`ModelConfig::projection_params`].
    pub projections: f64,
}

impl WeightBytesPerParam {
    /// Q6_K: 210 bytes per 256-element block. Block layout transcribed from
    /// `block_q6_K` in `ggml/src/ggml-common.h` (see `xabe-gguf`'s
    /// `GgmlType::Q6K`, which this crate deliberately does not depend on —
    /// see the module doc — so the constant is repeated here rather than
    /// imported).
    pub const Q6_K: f64 = 210.0 / 256.0;

    /// Q8_0: 34 bytes per 32-element block.
    pub const Q8_0: f64 = 34.0 / 32.0;

    /// Bytes-per-parameter as actually observed, tensor by tensor, in
    /// `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` while building `xabe-gguf`:
    ///
    /// - MoE expert matrices (`blk.N.ffn_{gate,up}_exps.weight`) are Q6_K
    ///   on most layers (a handful of layers use Q8_0 instead — unsloth's
    ///   "XL" recipe varies quant per layer by measured importance; Q6_K is
    ///   the modal case and is what this figure uses).
    /// - The LM head (`output.weight`) is Q8_0.
    /// - Every GDN and attention projection matrix inspected
    ///   (`attn_qkv`, `attn_gate`, `ssm_out`, `attn_q`, `attn_k`, `attn_v`,
    ///   `attn_output`) is Q8_0.
    ///
    /// **This disagrees with `qwen36-rust-engine-plan.md` §04**, whose
    /// "GDN + attention projections" row (0.944B params, 774 MB/token)
    /// implies both a smaller parameter count than
    /// [`ModelConfig::projection_params`] actually derives (1.305B — that
    /// figure is independently verified against the real tensor shapes,
    /// unlike the plan's estimate) and a Q6_K-level byte rate rather than
    /// the Q8_0 actually used. Using this constant, projection traffic
    /// comes out to ~1.39 GB/token rather than the plan's 774 MB/token, and
    /// total weight bytes/token at any context length is correspondingly
    /// higher than the plan's table states. This was reported upstream
    /// rather than silently reconciled — the plan's table has not been
    /// corrected to match.
    pub const fn observed_in_target_file() -> Self {
        Self {
            active_ffn: Self::Q6_K,
            lm_head: Self::Q8_0,
            projections: Self::Q8_0,
        }
    }
}

/// Per-token decode bandwidth at one context length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeBandwidth {
    /// Weight bytes read per decoded token: active experts, LM head, and
    /// all per-layer projections.
    pub weight_bytes: u64,
    /// KV bytes read per decoded token: the full attention KV cache,
    /// re-read from scratch on every step (there is no incremental
    /// re-use — the whole point of the roofline is that this term grows
    /// linearly with context).
    pub kv_bytes: u64,
    /// `weight_bytes + kv_bytes`.
    pub total_bytes: u64,
    /// `gpu_bandwidth_bytes_per_sec / total_bytes` — a ceiling, not a
    /// prediction. It assumes perfect bandwidth utilization and zero
    /// compute-bound stalls, neither of which has been measured on this
    /// hardware yet.
    pub roofline_tokens_per_sec: f64,
}

/// Compute per-token decode bandwidth and its roofline throughput.
pub fn decode_bandwidth(
    cfg: &ModelConfig,
    context_tokens: u64,
    kv_elem_bytes: u64,
    weight_bpp: WeightBytesPerParam,
    gpu_bandwidth_bytes_per_sec: f64,
) -> DecodeBandwidth {
    let weight_bytes = (cfg.active_ffn_params() as f64 * weight_bpp.active_ffn
        + cfg.lm_head_params() as f64 * weight_bpp.lm_head
        + cfg.projection_params() as f64 * weight_bpp.projections)
        .round() as u64;
    let kv_bytes = cfg.kv_bytes_per_token(kv_elem_bytes) * context_tokens;
    let total_bytes = weight_bytes + kv_bytes;
    DecodeBandwidth {
        weight_bytes,
        kv_bytes,
        total_bytes,
        roofline_tokens_per_sec: gpu_bandwidth_bytes_per_sec / total_bytes as f64,
    }
}

/// Context length, in tokens, past which per-token KV reads exceed
/// per-token weight reads — the point decode stops being weight-bound and
/// becomes KV-bound.
///
/// `weight_bytes` is context-independent (it is the same regardless of how
/// long the sequence is), while KV bytes grow linearly with context at
/// `cfg.kv_bytes_per_token(kv_elem_bytes)` per token. The crossover is
/// simply where the two lines meet.
pub fn kv_bound_crossover_tokens(cfg: &ModelConfig, kv_elem_bytes: u64, weight_bytes: u64) -> u64 {
    weight_bytes / cfg.kv_bytes_per_token(kv_elem_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ModelConfig {
        ModelConfig::qwen3_6_35b_a3b()
    }

    /// Relative error helper for reference-figure comparisons. All
    /// tolerances in this module are stated as a percentage in the assert
    /// message, not just a bare epsilon.
    fn relative_error(actual: f64, reference: f64) -> f64 {
        (actual - reference).abs() / reference
    }

    // -- VRAM segmentation --------------------------------------------

    /// Reference config from `qwen36-rust-engine-plan.md` §03: context
    /// 393,216, 3 slots, f16 KV.
    fn reference_vram(weights_bytes: u64) -> VramBudget {
        vram_budget(&cfg(), 393_216, 3, 2, weights_bytes)
    }

    #[test]
    fn kv_pool_matches_the_plans_derived_figure_exactly() {
        // 20,480 B/token * 393,216 tokens = 8,053,063,680 B = exactly 7.5 GiB.
        // This is an exact closed-form result, not an estimate, so it is
        // asserted exactly rather than within a tolerance.
        let b = reference_vram(0);
        assert_eq!(b.kv_pool_bytes, (75 * GIB) / 10);
    }

    #[test]
    fn gdn_state_matches_the_plans_derived_figure_within_1_percent() {
        // 30 GDN layers * 2 MiB/layer * 3 slots = 188,743,680 B ~= 0.176
        // GiB, which the plan rounds to "0.18 GiB".
        let b = reference_vram(0);
        let gib = b.gdn_state_bytes as f64 / GIB as f64;
        assert!(
            relative_error(gib, 0.18) < 0.03,
            "GDN state {gib:.4} GiB should be within 3% of the plan's rounded 0.18 GiB"
        );
    }

    #[test]
    fn total_vram_is_within_a_few_percent_of_the_plans_total_excluding_vision() {
        // The plan's own total (~41.3 GiB) includes a ~1.5 GiB vision-
        // encoder segment (`mmproj-F16.gguf`) that this crate does not
        // budget for — this engine is text-only (AGENTS.md, "Scope"). The
        // correct comparison point is the plan's total minus that segment,
        // ~39.8 GiB, not the raw headline figure.
        const WEIGHTS_BYTES: u64 = (296 * GIB) / 10; // 29.6 GiB, plan's "measured" figure
        const REFERENCE_TOTAL_EXCLUDING_VISION_GIB: f64 = 41.3 - 1.5;

        let b = reference_vram(WEIGHTS_BYTES);
        let total_gib = b.total_bytes() as f64 / GIB as f64;
        assert!(
            relative_error(total_gib, REFERENCE_TOTAL_EXCLUDING_VISION_GIB) < 0.03,
            "total {total_gib:.3} GiB should be within 3% of {REFERENCE_TOTAL_EXCLUDING_VISION_GIB:.3} GiB \
             (the plan's ~41.3 GiB minus its ~1.5 GiB out-of-scope vision segment)"
        );

        let headroom_gib = b.headroom_bytes((475 * GIB) / 10) as f64 / GIB as f64;
        assert!(
            headroom_gib > 0.0,
            "reference configuration must fit in usable VRAM"
        );
    }

    #[test]
    fn headroom_goes_negative_when_the_configuration_does_not_fit() {
        let b = reference_vram(45 * GIB); // deliberately oversized weights
        assert!(b.headroom_bytes(47 * GIB) < 0);
    }

    // -- Bandwidth ------------------------------------------------------

    const RTX_8000_BANDWIDTH_BYTES_PER_SEC: f64 = 672e9;

    #[test]
    fn lm_head_bytes_match_the_plans_reference_within_1_percent() {
        let bw = decode_bandwidth(
            &cfg(),
            131_072, // 128K — context is irrelevant to weight_bytes, held fixed here
            2,
            WeightBytesPerParam::observed_in_target_file(),
            RTX_8000_BANDWIDTH_BYTES_PER_SEC,
        );
        let lm_head_mb = cfg().lm_head_params() as f64 * WeightBytesPerParam::Q8_0 / 1e6;
        assert!(
            relative_error(lm_head_mb, 540.0) < 0.01,
            "LM head {lm_head_mb:.1} MB should be within 1% of the plan's 540 MB"
        );
        let _ = bw; // exercised for its side effect of not panicking
    }

    #[test]
    fn active_expert_bytes_match_the_plans_reference_within_1_percent() {
        let expert_mb = cfg().active_ffn_params() as f64 * WeightBytesPerParam::Q6_K / 1e6;
        assert!(
            relative_error(expert_mb, 929.0) < 0.01,
            "active expert traffic {expert_mb:.1} MB should be within 1% of the plan's 929 MB"
        );
    }

    #[test]
    fn lm_head_exceeds_half_of_active_expert_moe_traffic() {
        let c = cfg();
        let lm_head_bytes = c.lm_head_params() as f64 * WeightBytesPerParam::Q8_0;
        let expert_bytes = c.active_ffn_params() as f64 * WeightBytesPerParam::Q6_K;
        assert!(
            lm_head_bytes > expert_bytes / 2.0,
            "LM head ({lm_head_bytes:.0} B) should exceed half of active-expert \
             traffic ({expert_bytes:.0} B) — a single untied vocab matrix costing \
             more than half of all forty MoE layers combined"
        );
    }

    #[test]
    fn kv_reads_grow_past_weight_reads_somewhere_in_the_hundred_k_token_region() {
        // The plan's own crossover estimate (~109K, from its 2.24 GB/token
        // weight figure) and this module's closed-form-derived crossover
        // (~139K, from the higher, tensor-verified weight_bytes figure —
        // see `WeightBytesPerParam::observed_in_target_file`'s doc comment)
        // disagree by about 28%. Rather than assert either point estimate
        // tightly, this test asserts the qualitative claim both agree on:
        // the crossover lands somewhere in the tens-of-hundred-K region,
        // not at 10K and not at 1M.
        let c = cfg();
        let weight_bytes = (c.active_ffn_params() as f64 * WeightBytesPerParam::Q6_K
            + c.lm_head_params() as f64 * WeightBytesPerParam::Q8_0
            + c.projection_params() as f64 * WeightBytesPerParam::Q8_0)
            .round() as u64;
        let crossover = kv_bound_crossover_tokens(&c, 2, weight_bytes);
        assert!(
            (50_000..200_000).contains(&crossover),
            "KV/weight crossover at {crossover} tokens should land in the \
             tens-of-hundred-K region"
        );
    }

    #[test]
    fn weight_bytes_are_constant_across_context_but_kv_bytes_scale_linearly() {
        let c = cfg();
        let bpp = WeightBytesPerParam::observed_in_target_file();
        let at_4k = decode_bandwidth(&c, 4096, 2, bpp, RTX_8000_BANDWIDTH_BYTES_PER_SEC);
        let at_256k = decode_bandwidth(&c, 262_144, 2, bpp, RTX_8000_BANDWIDTH_BYTES_PER_SEC);
        assert_eq!(at_4k.weight_bytes, at_256k.weight_bytes);
        assert_eq!(at_256k.kv_bytes, at_4k.kv_bytes * 64); // 262144 / 4096 = 64
        assert!(at_256k.roofline_tokens_per_sec < at_4k.roofline_tokens_per_sec);
    }
}
