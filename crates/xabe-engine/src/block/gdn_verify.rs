//! The GDN recurrent-state rollback that makes speculative verify correct.
//!
//! A verify step folds
//! `1 + d` drafted positions through all thirty Gated DeltaNet layers in one
//! pass, but only `k <= 1 + d` of them may turn out to be accepted. The
//! recurrent state ([`crate::block::gdn::GdnState`]) has no notion of
//! "undo" — it is a `[value_heads][head_dim][head_dim]` matrix that the
//! delta rule has already folded every rejected position into — so getting
//! `k` right requires the state to have been snapshotted at every position
//! boundary *during* the same pass that computed it, not recomputed after.
//!
//! # Why this needs no change to `xabe-cuda` or `gdn.rs`
//!
//! [`GdnBlock::forward`]'s single call to [`GdnBlock::mix`] is what makes
//! this hard: at `tokens = 1 + d` it takes the chunked path and only the
//! *final* state is ever materialized. But `mix` is public, and at
//! `tokens == 1` it dispatches to the recurrent step — the exact kernel an
//! ordinary one-token decode already uses, "no weight traffic at all" (see
//! `GdnBlock::forward_batch_decode`'s own docs, which decompose the same
//! block the same way for a different reason). So this file does not touch
//! [`GdnBlock`] at all: it replicates `GdnBlock::run`'s ten-step sequence
//! (the reference is `run`'s own doc comments, transcribed step for step)
//! using only its `pub` methods, but splits step 7 — the one step that
//! touches `state.recurrent` — and step 3/4's convolution — the one step
//! that touches `state.conv` — into `1 + d` single-token calls, snapshotting
//! both pieces of state after each one. Every other step (the norm, the two
//! big projections, the gates, the output projection) stays a single call
//! over the whole window, so the weight-bound steps still read every matrix
//! **once** — R6's whole claimed win — and only cheap, weight-free
//! per-position launches and small `CudaSlice` copies are duplicated `1 + d`
//! times instead of one.
//!
//! `mix` at `tokens == 1` and the chunked path at `tokens == 1 + d` are
//! proven equivalent already: `tests/gdn_chunked_differential.rs`'s
//! `device_scan_gdn_matches_both_reference_forms_over_512_tokens` and
//! `one_chunk_of_prefill_lands_on_the_same_state_as_one_step_of_decode`
//! establish it at the kernel level. `tests/gdn_verify_differential.rs`
//! extends that one level up, at exactly this file's composition: running
//! [`run_layer_with_snapshots`] over a window and comparing its committed
//! final state against a plain [`GdnBlock::forward`] call on the same input.
//!
//! # Why the projections take the split-tile path
//!
//! A verify window is `1 + d` tokens of **one** sequence, not a batch of
//! independent sequences — but the weight-bound qkv/gate/out projections
//! still have a real token axis, and they must land on the same floats a
//! chain of one-token [`GdnBlock::forward`] calls would. Those one-token
//! calls take `gdn_proj_split_gemv` (the GEMV's 4-per-lane grouping). The
//! standard-layout `gdn_proj_q8_0_t*` pairing disagrees with that GEMV at
//! the first FMA; `split_tiled`
//! is the GEMV's own grouping amortized across the window. Without that,
//! `tests/speculative_identity.rs` would reject a draft the target's own
//! one-token argmax would have emitted. The MMA path is never taken: a
//! verify window does not cross [`GdnBlock::uses_tensor_cores`].
//!
//! # Cost
//!
//! `2 + d` snapshots of one layer's `conv` (tiny — `conv_dim * (conv_kernel -
//! 1)` floats) and `recurrent` (2 MiB at this model's geometry) state — one
//! per possible commit point, `0..=(1 + d)` positions accepted inclusive. At
//! `d = 3` and 30 Gated DeltaNet layers that is a little over 300 MiB per
//! in-flight sequence, and a "commit" is a choice of which snapshot becomes
//! the live state — one `cuMemcpyDtoD` per layer per sequence, not a
//! recompute.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DriverError};

use xabe_cuda::kernels::layer_ops::LayerOpsError;

use crate::block::gdn::{
    GdnBlock, GdnBlockError, GdnGeometry, GdnLayerInt8, GdnLayerWeights, GdnState, Projection,
};
use crate::viewslice::subslice;

/// Something went wrong running a snapshotted Gated DeltaNet layer.
#[derive(Debug)]
pub enum GdnVerifyError {
    /// One of `GdnBlock`'s public steps rejected a launch.
    Gdn(GdnBlockError),
    /// The norm, convolution or SwiGLU rejected a launch.
    LayerOps(LayerOpsError),
    /// The driver failed on a snapshot or commit copy.
    Driver(DriverError),
    /// `tokens` did not match the geometry a scratch or ring was built for.
    WrongTokenCount { expected: usize, got: usize },
}

impl std::fmt::Display for GdnVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gdn(e) => write!(f, "{e}"),
            Self::LayerOps(e) => write!(f, "{e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::WrongTokenCount { expected, got } => write!(
                f,
                "this snapshot ring/scratch was built for {expected} tokens, got {got}",
            ),
        }
    }
}

impl std::error::Error for GdnVerifyError {}

impl From<GdnBlockError> for GdnVerifyError {
    fn from(e: GdnBlockError) -> Self {
        Self::Gdn(e)
    }
}
impl From<LayerOpsError> for GdnVerifyError {
    fn from(e: LayerOpsError) -> Self {
        Self::LayerOps(e)
    }
}
impl From<DriverError> for GdnVerifyError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Per-position snapshots of one Gated DeltaNet layer's recurrent state, for
/// one sequence.
///
/// Slot `i` holds the state as it stood after exactly `i` of the window's
/// tokens had been folded in — slot 0 is the state the window *started*
/// from, slot `depth - 1` is what a plain [`GdnBlock::forward`] over the
/// whole window would have left.
pub struct GdnSnapshotRing {
    conv: CudaSlice<f32>,
    recurrent: CudaSlice<f32>,
    depth: usize,
    conv_len: usize,
    recurrent_len: usize,
}

impl GdnSnapshotRing {
    /// Allocate `depth` slots.
    ///
    /// A window of `tokens = 1 + d` positions needs `depth = tokens + 1`:
    /// slot 0 is the state before anything in the window is folded in and
    /// slot `tokens` is the state after all of it is, so the number of
    /// genuinely-committed positions — which ranges `0..=tokens` — has one
    /// slot per possible value.
    pub fn new(
        stream: &Arc<CudaStream>,
        geometry: &GdnGeometry,
        depth: usize,
    ) -> Result<Self, DriverError> {
        let conv_len = geometry.conv_state_len();
        let recurrent_len = geometry.recurrent_state_len();
        Ok(Self {
            conv: stream.alloc_zeros::<f32>(depth * conv_len)?,
            recurrent: stream.alloc_zeros::<f32>(depth * recurrent_len)?,
            depth,
            conv_len,
            recurrent_len,
        })
    }

    /// How many slots this ring holds.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Device bytes held.
    pub fn bytes(&self) -> u64 {
        ((self.conv.len() + self.recurrent.len()) * size_of::<f32>()) as u64
    }

    /// Copy `state`'s current conv/recurrent buffers into slot `i`.
    fn snapshot(
        &mut self,
        stream: &Arc<CudaStream>,
        i: usize,
        state: &GdnState,
    ) -> Result<(), DriverError> {
        let mut conv_slot =
            unsafe { subslice(stream, &self.conv, i * self.conv_len, self.conv_len) };
        stream.memcpy_dtod(&state.conv, &mut *conv_slot)?;
        let mut rec_slot = unsafe {
            subslice(
                stream,
                &self.recurrent,
                i * self.recurrent_len,
                self.recurrent_len,
            )
        };
        stream.memcpy_dtod(&state.recurrent, &mut *rec_slot)?;
        Ok(())
    }

    /// Overwrite `state`'s live conv/recurrent buffers with slot `i` — the
    /// commit that makes an accepted-count of `i` positions the sequence's
    /// real, going-forward state.
    ///
    /// `i == 0` is "reject the whole draft"; `i == depth - 1` is "accept
    /// everything", which restores exactly what a plain
    /// [`GdnBlock::forward`] over the window would have left, and is the
    /// invariant `tests/gdn_verify_differential.rs` checks.
    pub fn commit(
        &self,
        stream: &Arc<CudaStream>,
        i: usize,
        state: &mut GdnState,
    ) -> Result<(), DriverError> {
        let conv_slot = unsafe { subslice(stream, &self.conv, i * self.conv_len, self.conv_len) };
        stream.memcpy_dtod(&*conv_slot, &mut state.conv)?;
        let rec_slot = unsafe {
            subslice(
                stream,
                &self.recurrent,
                i * self.recurrent_len,
                self.recurrent_len,
            )
        };
        stream.memcpy_dtod(&*rec_slot, &mut state.recurrent)?;
        Ok(())
    }
}

/// Scratch for one Gated DeltaNet layer's snapshotted verify pass, sized for
/// exactly `tokens` positions.
///
/// Field names and shapes mirror `GdnBlock::run`'s private `Scratch`
/// exactly, because this **is** `run`'s sequence — just with steps 3/4 and 7
/// unrolled to one token at a time. Kept in this file rather than reused
/// from `gdn.rs` because `Scratch` there is private, and the fields it needs
/// are a strict subset of what `run` allocates (no tensor-core staging: `1 +
/// d` tokens never crosses `GdnBlock::uses_tensor_cores`'s threshold). The
/// qkv/gate/out projections take `project_split_tiled` when the split-layout
/// repack is present, matching one-token decode's GEMV — see the module docs.
pub struct GdnVerifyScratch {
    tokens: usize,
    normed: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    z: CudaSlice<f32>,
    conv_raw: CudaSlice<f32>,
    conv_silu: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    alpha: CudaSlice<f32>,
    beta_raw: CudaSlice<f32>,
    a_softplus: CudaSlice<f32>,
    log_decay: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    core: CudaSlice<f32>,
    core_norm: CudaSlice<f32>,
    final_output: CudaSlice<f32>,
    projected: CudaSlice<f32>,
}

impl GdnVerifyScratch {
    /// Allocate for exactly `tokens` positions — the verify window's
    /// `1 + d`.
    pub fn new(
        stream: &Arc<CudaStream>,
        geometry: &GdnGeometry,
        tokens: usize,
    ) -> Result<Self, DriverError> {
        let g = geometry;
        let conv_dim = g.conv_dim();
        let key_dim = g.key_dim();
        let value_dim = g.value_dim();
        let heads = g.value_heads;
        Ok(Self {
            tokens,
            normed: stream.alloc_zeros::<f32>(tokens * g.hidden)?,
            qkv: stream.alloc_zeros::<f32>(tokens * conv_dim)?,
            z: stream.alloc_zeros::<f32>(tokens * value_dim)?,
            conv_raw: stream.alloc_zeros::<f32>(tokens * conv_dim)?,
            conv_silu: stream.alloc_zeros::<f32>(tokens * conv_dim)?,
            q: stream.alloc_zeros::<f32>(tokens * key_dim)?,
            k: stream.alloc_zeros::<f32>(tokens * key_dim)?,
            v: stream.alloc_zeros::<f32>(tokens * value_dim)?,
            alpha: stream.alloc_zeros::<f32>(tokens * heads)?,
            beta_raw: stream.alloc_zeros::<f32>(tokens * heads)?,
            a_softplus: stream.alloc_zeros::<f32>(tokens * heads)?,
            log_decay: stream.alloc_zeros::<f32>(tokens * heads)?,
            beta: stream.alloc_zeros::<f32>(tokens * heads)?,
            core: stream.alloc_zeros::<f32>(tokens * value_dim)?,
            core_norm: stream.alloc_zeros::<f32>(tokens * value_dim)?,
            final_output: stream.alloc_zeros::<f32>(tokens * value_dim)?,
            projected: stream.alloc_zeros::<f32>(tokens * g.hidden)?,
        })
    }
}

/// Run one Gated DeltaNet layer over `hidden` (`[tokens][hidden]`),
/// snapshotting `state` into `ring` after every position, and leave `state`
/// exactly where a plain [`GdnBlock::forward`] over the same window would.
///
/// `int8` is the split-layout Q8_0 repack a [`crate::forward::Forward`]
/// already holds for this layer. Pass `Some` so the window's projections
/// match one-token decode; pass `None` to stay on the standard-layout
/// path (what a cold [`GdnBlock::forward`] with `int8 = None` takes).
///
/// `ring.depth()` must equal `tokens + 1` and `scratch` must have been built
/// for `tokens`. `out` (`[tokens][hidden]`) is the residual-added mixer
/// output, same contract as [`GdnBlock::forward`]'s own `out`.
///
/// Committing an accepted count `k` afterward is [`GdnSnapshotRing::commit`]
/// with `i = k` — the caller's job, once every layer in the window has run
/// and the verify pass's argmax comparison has decided `k`.
#[allow(clippy::too_many_arguments)]
pub fn run_layer_with_snapshots(
    stream: &Arc<CudaStream>,
    gdn: &mut GdnBlock,
    weights: &GdnLayerWeights,
    int8: Option<&GdnLayerInt8>,
    state: &mut GdnState,
    hidden: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    scratch: &mut GdnVerifyScratch,
    ring: &mut GdnSnapshotRing,
    tokens: usize,
) -> Result<(), GdnVerifyError> {
    if scratch.tokens != tokens || ring.depth() != tokens + 1 {
        return Err(GdnVerifyError::WrongTokenCount {
            expected: scratch.tokens,
            got: tokens,
        });
    }
    let g = gdn.geometry();
    let conv_dim = g.conv_dim();
    let key_dim = g.key_dim();
    let value_dim = g.value_dim();
    let heads = g.value_heads;

    // 1. `attn_norm-N`.
    gdn.layer_ops().rms_norm(
        stream,
        hidden,
        &weights.input_norm,
        &mut scratch.normed,
        tokens,
        g.hidden,
        g.rms_eps,
    )?;

    // 2. `linear_attn_qkv_mixed-N` and `z-N` — the two big Q8_0 matrices,
    //    read once over the whole window. This is the step that carries
    //    R6's arithmetic; nothing below re-reads either of them.
    //
    //    When the split-layout repack is present (it is, on every Forward
    //    this engine builds), take `project_split_tiled` so the window's
    //    first FMA matches one-token decode's GEMV. Standard-layout
    //    `project` is the fallback for callers that did not upload a
    //    repack — the differential test against a cold `GdnBlock`.
    let split = int8.filter(|_| tokens > 1 && !GdnBlock::uses_tensor_cores(tokens));
    if let Some(i8w) = split {
        let (qkv_q, qkv_s) = i8w.qkv();
        gdn.project_split_tiled(
            stream,
            qkv_q,
            qkv_s,
            &scratch.normed,
            &mut scratch.qkv,
            g.hidden,
            conv_dim,
            tokens,
        )?;
        let (gate_q, gate_s) = i8w.gate();
        gdn.project_split_tiled(
            stream,
            gate_q,
            gate_s,
            &scratch.normed,
            &mut scratch.z,
            g.hidden,
            value_dim,
            tokens,
        )?;
    } else {
        gdn.project(
            stream,
            Projection::Q8_0(&weights.qkv),
            &scratch.normed,
            &mut scratch.qkv,
            g.hidden,
            conv_dim,
            tokens,
        )?;
        gdn.project(
            stream,
            Projection::Q8_0(&weights.gate),
            &scratch.normed,
            &mut scratch.z,
            g.hidden,
            value_dim,
            tokens,
        )?;
    }

    // 6. alpha/beta gates. Reads `w.alpha`/`w.beta` — `[hidden, value_heads]`,
    //    a few hundred KiB — so batching or not costs nothing either way;
    //    batched here to match `run`'s own order.
    gdn.alpha_beta_gates(
        stream,
        &weights.alpha,
        &weights.beta,
        &scratch.normed,
        &weights.dt_bias,
        &weights.a,
        &mut scratch.alpha,
        &mut scratch.beta_raw,
        &mut scratch.a_softplus,
        &mut scratch.log_decay,
        &mut scratch.beta,
        tokens,
    )?;

    // Slot 0: the state the window started from — "0 of `tokens` positions
    // committed". Every later slot is taken after processing one more
    // position, so slot `i` always means "positions `0..i` committed".
    ring.snapshot(stream, 0, state)?;

    // Steps 3/4/5/7, one token at a time: `conv1d` and `mix` are the two
    // steps that read and write `state`'s own buffers, so — unlike 1, 2 and
    // 6 above — they cannot be batched without losing the per-position
    // snapshot this whole file exists to take. Both are weight-free or
    // effectively so (`conv1d`'s filter is `conv_kernel * conv_dim` floats,
    // a few hundred KiB; `mix` at `tokens == 1` reads no weight at all), so
    // `tokens` small launches cost nothing next to the two big projections
    // above.
    for i in 0..tokens {
        let qkv_i = unsafe { subslice(stream, &scratch.qkv, i * conv_dim, conv_dim) };
        let mut conv_raw_i = unsafe { subslice(stream, &scratch.conv_raw, i * conv_dim, conv_dim) };
        gdn.layer_ops().conv1d(
            stream,
            &qkv_i,
            &weights.conv1d,
            &mut state.conv,
            &mut conv_raw_i,
            1,
            conv_dim,
            g.conv_kernel,
        )?;

        let mut conv_silu_i =
            unsafe { subslice(stream, &scratch.conv_silu, i * conv_dim, conv_dim) };
        let mut q_i = unsafe { subslice(stream, &scratch.q, i * key_dim, key_dim) };
        let mut k_i = unsafe { subslice(stream, &scratch.k, i * key_dim, key_dim) };
        let mut v_i = unsafe { subslice(stream, &scratch.v, i * value_dim, value_dim) };
        gdn.silu_split_qkv(
            stream,
            &conv_raw_i,
            &mut conv_silu_i,
            &mut q_i,
            &mut k_i,
            &mut v_i,
            1,
        )?;

        let log_decay_i = unsafe { subslice(stream, &scratch.log_decay, i * heads, heads) };
        let beta_i = unsafe { subslice(stream, &scratch.beta, i * heads, heads) };
        let mut core_i = unsafe { subslice(stream, &scratch.core, i * value_dim, value_dim) };
        gdn.mix(
            stream,
            state,
            &q_i,
            &k_i,
            &v_i,
            &log_decay_i,
            &beta_i,
            &mut core_i,
            1,
        )?;

        // Position `i` is now folded in, so this is slot `i + 1` — "positions
        // `0..=i` committed". At `i == tokens - 1` this is slot `tokens`,
        // the ring's last: everything in the window accepted.
        ring.snapshot(stream, i + 1, state)?;
    }

    // 8. `final_output-N = ssm_norm(core) * silu(z)`. No state, batches over
    //    the whole window.
    gdn.layer_ops().rms_norm_swiglu(
        stream,
        &scratch.core,
        &weights.ssm_norm,
        &scratch.z,
        &mut scratch.core_norm,
        &mut scratch.final_output,
        tokens * heads,
        g.head_dim,
        g.rms_eps,
    )?;

    // 9/10. `linear_attn_out-N`, then the residual add. No state, batches
    //       over the whole window, and is the pass's other big matrix.
    //       Same split-vs-standard pairing as step 2: the out projection
    //       has no fused token-axis residual form, so this is always
    //       project-then-add, matching `run_batch_decode` at this width.
    if let Some(i8w) = split {
        let (out_q, out_s) = i8w.out();
        gdn.project_split_tiled(
            stream,
            out_q,
            out_s,
            &scratch.final_output,
            &mut scratch.projected,
            value_dim,
            g.hidden,
            tokens,
        )?;
    } else {
        gdn.project(
            stream,
            Projection::Q8_0(&weights.out),
            &scratch.final_output,
            &mut scratch.projected,
            value_dim,
            g.hidden,
            tokens,
        )?;
    }
    gdn.layer_ops()
        .add(stream, &scratch.projected, hidden, out, tokens * g.hidden)?;

    Ok(())
}
