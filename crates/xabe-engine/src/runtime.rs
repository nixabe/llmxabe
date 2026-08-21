//! CUDA execution owned by one scheduler worker.
//!
//! This is deliberately separate from the host-only construction path used
//! by routing tests. [`crate::worker::Worker::bind_device`] installs one of
//! these after admission/cache geometry has been validated.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::JoinHandle;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream, DriverError};
use smallvec::SmallVec;
use tracing::debug;
use xabe_gguf::{GgufError, GgufFile};
use xabe_model::ModelConfig;
use xabe_model::weights::WeightSchema;
use xabe_sched::ngram::{NgramConfig, NgramConfigError, NgramSpeculator};
use xabe_sched::request::{BatchDescription, NewRequest, RequestId};

use crate::block::attention::KvCache;
use crate::block::gdn_verify::GdnSnapshotRing;
use crate::block::mtp::{MtpBlock, MtpBlockError};
use crate::forward::{BatchStepGraph, Forward, ForwardError, arena_holds};
use crate::image::{
    ImagePlacement, SequenceImage, chunk_overlaps_images, fill_mrope_triples, rope_delta_at,
    validate_placements,
};
use crate::sampling::{Sampler, SamplingParams};
use crate::state::{SequenceSnapshot, SnapshotArena, SnapshotSlots};
use crate::vision::VisionForward;
use crate::{DeviceWeights, LoadError, SequenceState, StateError};

/// Pinned snapshot capacity per worker when no budget is given. One
/// default-retention slot is 102.8125 MiB, so 24 slots consume 2.41 GiB per
/// worker and 7.23 GiB process-wide.
/// This stays below the host's locked-memory limit while covering eight
/// simultaneous retention points for each of the three serving sequences.
/// `--cache-ram` overrides it; see `docs/CLI.md`.
pub const DEFAULT_SNAPSHOT_SLOTS_PER_WORKER: usize = 24;

/// Default patch budget for the vision tower: 4096 patches = 1024
/// language-model tokens ≈ a one-megapixel image. Bounds the score
/// workspace at 4 heads × 4096² f16 = 128 MiB; `--image-max-tokens`
/// raises it.
pub const DEFAULT_MAX_IMAGE_PATCHES: usize = 4096;

/// Everything a device runtime needs beyond the model itself.
///
/// A struct rather than eight positional arguments because two of these are
/// `usize` counts that mean entirely different things, and swapping them
/// would compile.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Tokens per chunked-prefill step.
    pub prefill_chunk: usize,
    /// Widest decode batch this worker will be asked to run.
    pub max_batch: usize,
    /// Speculative drafting, or `None` for no drafting at all.
    pub ngram: Option<NgramConfig>,
    /// Tokens the trained MTP head drafts per step, or zero for no MTP.
    /// Mutually exclusive with `ngram`; `crate::worker` enforces that by
    /// construction from the single `Speculation` enum.
    pub mtp_drafts: usize,
    /// Tokens between retained GDN snapshots.
    pub retention_interval: usize,
    /// Pinned host snapshot slots. Zero disables retention, and with it
    /// prefix sharing: a sequence that cannot check one out simply stops
    /// snapshotting and serves normally.
    pub snapshot_slots: usize,
    /// Whether the end-of-turn token ends a sequence. A fixed-width
    /// throughput benchmark sets this false so its measurement cannot
    /// silently shrink.
    pub stop_on_eos: bool,
    /// Vision tower (mmproj) GGUF to load, or `None` for text-only serving —
    /// in which case nothing vision-related is allocated and every request
    /// carrying images is refused.
    pub mmproj: Option<PathBuf>,
    /// Patch budget the vision tower is pre-allocated for (4 patches per
    /// language-model token). Ignored without `mmproj`.
    pub max_image_patches: usize,
}

#[derive(Debug)]
pub enum RuntimeError {
    Driver(DriverError),
    Gguf(GgufError),
    Schema(String),
    Load(LoadError),
    Forward(ForwardError),
    State(StateError),
    ZeroPrefillChunk,
    ZeroBatchWidth,
    DuplicateRequest(RequestId),
    UnknownRequest(RequestId),
    PromptLength {
        declared: u32,
        actual: usize,
    },
    TokenOutOfRange {
        token: i32,
        vocab: u32,
    },
    BatchTooWide {
        requested: usize,
        maximum: usize,
    },
    SnapshotPrefix {
        snapshot: usize,
        prompt: usize,
    },
    Speculation(NgramConfigError),
    /// Building or running the MTP draft head failed.
    Mtp(MtpBlockError),
    RuntimeStopped,
    /// The request carries images but this runtime loaded no mmproj.
    VisionNotEnabled,
    /// The vision tower failed to load or encode.
    Vision(String),
    /// A request's image placements are inconsistent with its prompt.
    ImagePlacement(String),
}

impl core::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::VisionNotEnabled => {
                write!(f, "request carries images but no --mmproj was loaded")
            }
            Self::Vision(e) => write!(f, "vision tower: {e}"),
            Self::ImagePlacement(e) => write!(f, "image placement: {e}"),
            Self::Gguf(e) => write!(f, "GGUF error: {e}"),
            Self::Schema(e) => write!(f, "model schema mismatch: {e}"),
            Self::Load(e) => write!(f, "weight load failed: {e}"),
            Self::Forward(e) => write!(f, "forward pass failed: {e}"),
            Self::State(e) => write!(f, "sequence state failed: {e}"),
            Self::Speculation(e) => write!(f, "speculative decoding: {e}"),
            Self::Mtp(e) => write!(f, "MTP draft head: {e}"),
            Self::ZeroPrefillChunk => write!(f, "prefill chunk must be non-zero"),
            Self::ZeroBatchWidth => write!(f, "maximum decode batch must be non-zero"),
            Self::DuplicateRequest(id) => write!(f, "request {} is already resident", id.0),
            Self::UnknownRequest(id) => write!(f, "request {} has no device state", id.0),
            Self::PromptLength { declared, actual } => write!(
                f,
                "request declares {declared} prompt tokens but supplied {actual} ids"
            ),
            Self::TokenOutOfRange { token, vocab } => {
                write!(f, "token id {token} is outside vocabulary 0..{vocab}")
            }
            Self::BatchTooWide { requested, maximum } => write!(
                f,
                "scheduled decode batch {requested} exceeds prebuilt maximum {maximum}"
            ),
            Self::SnapshotPrefix { snapshot, prompt } => write!(
                f,
                "snapshot covers {snapshot} tokens but prompt contains {prompt}"
            ),
            Self::RuntimeStopped => write!(f, "device runtime thread stopped"),
        }
    }
}

impl core::error::Error for RuntimeError {}

impl From<DriverError> for RuntimeError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<MtpBlockError> for RuntimeError {
    fn from(value: MtpBlockError) -> Self {
        Self::Mtp(value)
    }
}

impl From<GgufError> for RuntimeError {
    fn from(value: GgufError) -> Self {
        Self::Gguf(value)
    }
}

impl From<LoadError> for RuntimeError {
    fn from(value: LoadError) -> Self {
        Self::Load(value)
    }
}

impl From<ForwardError> for RuntimeError {
    fn from(value: ForwardError) -> Self {
        Self::Forward(value)
    }
}

impl From<StateError> for RuntimeError {
    fn from(value: StateError) -> Self {
        Self::State(value)
    }
}

/// One sequence's half of the MTP draft head: its own key/value cache and
/// the seed hidden state for the next draft chain.
struct MtpSequence {
    /// The draft head's own KV cache — entirely separate from the target's.
    /// Sized `full_seq_len + drafts` so a chain drafted right at the output
    /// cap still has slots to write into (its tail is truncated, not run).
    cache: KvCache,
    /// The target's `final_norm` row at the last computed position — the
    /// producer of `next_token`. During prefill this doubles as the
    /// cross-chunk carry for the shifted-right catch-up input; after a
    /// verify step it is refreshed from the row that produced the last
    /// emitted token. Zero before any position is computed, which is the
    /// defined seed for position 0.
    pending_h: cudarc::driver::CudaSlice<f32>,
}

/// The runtime's half of the MTP draft head: one set of weights, one block
/// per prefill/draft shape, and the step scratch the drafting chain reuses
/// (AGENTS.md rule 6: nothing here allocates per step).
struct MtpRuntime {
    /// `(tokens, block)` per shape. Prefill shapes (chunk, retention
    /// interval, tails) are built without the LM head; widths
    /// `1..=max_batch` carry it, because they both draft and serve as the
    /// narrow catch-up remainder shapes. All share one upload of the
    /// block-40 MoE expert weights.
    shapes: Vec<(usize, MtpBlock)>,
    /// Shifted-`h` staging for catch-up and the chained draft `h`:
    /// `[largest shape][hidden]`.
    h: cudarc::driver::CudaSlice<f32>,
    /// Per-sequence absolute-position scalars, `[max_batch]`; catch-up
    /// borrows element 0.
    positions: cudarc::driver::CudaSlice<i32>,
    /// Host staging for `positions`.
    pos_host: Vec<i32>,
    /// Tokens drafted per step (the verify window is this plus one).
    drafts: usize,
    hidden: usize,
    /// For sizing per-sequence draft caches at admission.
    config: ModelConfig,
}

struct RuntimeSequence {
    state: Option<SequenceState>,
    prompt: Vec<i32>,
    prefilled: usize,
    next_token: Option<i32>,
    ngram: Option<NgramSpeculator>,
    /// The MTP draft head's per-sequence state, when MTP serving is on and
    /// the sequence is eligible (text-only, not restored from a snapshot —
    /// a restored prefix was never caught up in the draft cache, and
    /// drafting over a hole would propose from garbage attention).
    mtp: Option<MtpSequence>,
    draft: Vec<i32>,
    emitted: u32,
    max_output: u32,
    last_snapshot_position: usize,
    last_snapshot: Option<Arc<SequenceSnapshot>>,
    retention_disabled: bool,
    /// `None` is greedy argmax, decided entirely on the device. `Some` pays a
    /// host round-trip per emitted token, only for this sequence.
    sampler: Option<Sampler>,
    /// The prompt's image spans, sorted. Empty for text-only requests.
    images: Vec<ImagePlacement>,
    /// Projected image embeddings, all images concatenated in placement
    /// order, `sum(tokens) * hidden` f32. Encoded once at admission.
    image_embeds: Option<cudarc::driver::CudaSlice<f32>>,
}

fn choose_prefill_width(
    position: usize,
    remaining: usize,
    prefill_chunk: usize,
    retention_interval: usize,
    tail_widths: impl DoubleEndedIterator<Item = usize>,
    max_batch: usize,
) -> usize {
    let to_boundary = if retention_interval == 0 {
        usize::MAX
    } else {
        retention_interval - position % retention_interval
    };
    if remaining >= prefill_chunk && prefill_chunk <= to_boundary {
        return prefill_chunk;
    }
    if retention_interval > 0
        && remaining >= retention_interval
        && position.is_multiple_of(retention_interval)
    {
        return retention_interval;
    }
    tail_widths
        .rev()
        .find(|&width| width <= remaining && width <= to_boundary)
        .unwrap_or_else(|| remaining.min(to_boundary).min(max_batch).max(1))
}

/// The prebuilt pass for a `width`-token prefill piece — over split borrows
/// so a caller can hold other runtime fields (the MTP scratch, a sequence's
/// draft cache) at the same time. [`DeviceRuntime::prefill_pass`] is the
/// whole-`self` convenience over this.
fn prefill_pass_in<'a>(
    prefill: &'a mut Forward,
    retention_prefill: &'a mut Option<Forward>,
    prefill_tails: &'a mut [(usize, Forward)],
    decode: &'a mut [Option<Forward>],
    prefill_chunk: usize,
    retention_interval: usize,
    width: usize,
) -> &'a mut Forward {
    if width == prefill_chunk {
        prefill
    } else if width == retention_interval && width != 1 {
        retention_prefill
            .as_mut()
            .expect("distinct retention shape was prebuilt")
    } else if let Some(index) = prefill_tails
        .iter()
        .position(|(tail_width, _)| *tail_width == width)
    {
        &mut prefill_tails[index].1
    } else {
        decode[width]
            .as_mut()
            .expect("narrow prefill width was prebuilt")
    }
}

/// Tokens emitted by one scheduler/device step.
pub struct DeviceStep {
    pub decode_items: usize,
    pub prefill_items: usize,
    pub generated: SmallVec<[(RequestId, i32); 16]>,
    pub completed: SmallVec<[RequestId; 3]>,
    pub stopped: SmallVec<[RequestId; 3]>,
    pub retained: SmallVec<[(RequestId, Arc<SequenceSnapshot>); 8]>,
}

pub(crate) struct RuntimeStep {
    pub scheduled: BatchDescription,
    pub generated: SmallVec<[(RequestId, i32); 16]>,
    pub completed: SmallVec<[RequestId; 3]>,
    pub stopped: SmallVec<[RequestId; 3]>,
    pub retained: SmallVec<[(RequestId, Arc<SequenceSnapshot>); 8]>,
}

/// One card's context, stream, weights, prebuilt shapes, and sequence states.
pub struct DeviceRuntime {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    prefill: Forward,
    retention_prefill: Option<Forward>,
    prefill_tails: Vec<(usize, Forward)>,
    decode: Vec<Option<Forward>>,
    graphs: Vec<Option<(SmallVec<[RequestId; 3]>, BatchStepGraph)>>,
    /// Speculative verify passes, indexed by decode width like `decode`.
    /// Empty `None`s when drafting is off (or `LLMXABE_NGRAM_GATED` forces
    /// the round-gated fallback), so the non-speculative baseline allocates
    /// and runs nothing new.
    verify: Vec<Option<Forward>>,
    /// One reusable set of per-layer GDN snapshot rings per decode slot.
    /// Rings are step-scratch, not sequence state: a verify step borrows
    /// `[..width]` and every ring is dead again once the step commits.
    verify_rings: Vec<Vec<GdnSnapshotRing>>,
    /// Verify window width: `1 + draft_tokens`. Zero when drafting is off.
    window: usize,
    /// The MTP draft head, present iff `mtp_drafts > 0` was configured — the
    /// non-speculative baseline allocates and runs none of this.
    mtp: Option<MtpRuntime>,
    weights: DeviceWeights,
    sequences: HashMap<RequestId, RuntimeSequence>,
    prefill_chunk: usize,
    max_batch: usize,
    vocab: u32,
    ngram: Option<NgramConfig>,
    retention_interval: usize,
    snapshot_arena: SnapshotArena,
    sampled: Vec<i32>,
    /// One logits row copied back for host-side sampling, and the candidate
    /// scratch the sampler filters in. Both are pre-sized at load so the
    /// sampling path allocates nothing per token (AGENTS.md rule 6).
    host_logits: Vec<f32>,
    sample_scratch: Vec<(f32, u32)>,
    eos_token: Option<i32>,
    /// The vision tower, present iff an mmproj was configured.
    vision: Option<VisionForward>,
    /// Reused host buffer for per-chunk `(t, h, w)` rotary triples
    /// (AGENTS.md rule 6: no per-chunk allocation).
    mrope_host: Vec<i32>,
}

enum RuntimeCommand {
    Admit {
        req: NewRequest,
        prompt: Vec<i32>,
        images: Vec<SequenceImage>,
        snapshot: Option<Arc<SequenceSnapshot>>,
        sampling: SamplingParams,
        reply: SyncSender<Result<(), RuntimeError>>,
    },
    Remove {
        id: RequestId,
        reply: SyncSender<bool>,
    },
    Execute {
        batch: BatchDescription,
        reply: SyncSender<Result<RuntimeStep, RuntimeError>>,
    },
    Snapshot {
        id: RequestId,
        reply: SyncSender<Result<Arc<SequenceSnapshot>, RuntimeError>>,
    },
    Stop,
}

/// Bounded command handle for a permanently thread-owned CUDA runtime.
pub struct DeviceRuntimeHandle {
    commands: SyncSender<RuntimeCommand>,
    thread: Option<JoinHandle<()>>,
    slots: SnapshotSlots,
}

impl DeviceRuntimeHandle {
    pub fn spawn(
        device_ordinal: usize,
        model_path: PathBuf,
        config: ModelConfig,
        runtime: RuntimeConfig,
    ) -> Result<Self, RuntimeError> {
        let (commands, receiver) = sync_channel::<RuntimeCommand>(1);
        let (ready_tx, ready_rx) = sync_channel::<Result<SnapshotSlots, RuntimeError>>(0);
        let thread = std::thread::Builder::new()
            .name(format!("xabe-gpu-{device_ordinal}"))
            .spawn(move || {
                let runtime = DeviceRuntime::load(device_ordinal, &model_path, config, runtime);
                match runtime {
                    Ok(mut runtime) => {
                        if ready_tx.send(Ok(runtime.snapshot_arena.slots())).is_err() {
                            return;
                        }
                        while let Ok(command) = receiver.recv() {
                            match command {
                                RuntimeCommand::Admit {
                                    req,
                                    prompt,
                                    images,
                                    snapshot,
                                    sampling,
                                    reply,
                                } => {
                                    let result = match snapshot {
                                        Some(snapshot) => runtime.admit_restored(
                                            req, prompt, images, snapshot, sampling,
                                        ),
                                        None => runtime.admit(req, prompt, images, sampling),
                                    };
                                    let _ = reply.send(result);
                                }
                                RuntimeCommand::Remove { id, reply } => {
                                    let _ = reply.send(runtime.remove(id));
                                }
                                RuntimeCommand::Execute { batch, reply } => {
                                    let _ = reply.send(runtime.execute(batch));
                                }
                                RuntimeCommand::Snapshot { id, reply } => {
                                    let _ = reply.send(runtime.snapshot(id));
                                }
                                RuntimeCommand::Stop => break,
                            }
                        }
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                }
            })
            .map_err(|_| RuntimeError::RuntimeStopped)?;
        let slots = ready_rx
            .recv()
            .map_err(|_| RuntimeError::RuntimeStopped)??;
        Ok(Self {
            commands,
            thread: Some(thread),
            slots,
        })
    }

    /// How much room this device's pinned snapshot arena has left.
    pub fn snapshot_slots(&self) -> &SnapshotSlots {
        &self.slots
    }

    fn request<T>(
        &self,
        build: impl FnOnce(SyncSender<T>) -> RuntimeCommand,
    ) -> Result<T, RuntimeError> {
        let (reply, receive) = sync_channel(0);
        self.commands
            .send(build(reply))
            .map_err(|_| RuntimeError::RuntimeStopped)?;
        receive.recv().map_err(|_| RuntimeError::RuntimeStopped)
    }

    pub fn admit(
        &self,
        req: NewRequest,
        prompt: Vec<i32>,
        images: Vec<SequenceImage>,
        sampling: SamplingParams,
    ) -> Result<(), RuntimeError> {
        self.request(|reply| RuntimeCommand::Admit {
            req,
            prompt,
            images,
            snapshot: None,
            sampling,
            reply,
        })?
    }

    pub fn admit_restored(
        &self,
        req: NewRequest,
        prompt: Vec<i32>,
        images: Vec<SequenceImage>,
        snapshot: Arc<SequenceSnapshot>,
        sampling: SamplingParams,
    ) -> Result<(), RuntimeError> {
        self.request(|reply| RuntimeCommand::Admit {
            req,
            prompt,
            images,
            snapshot: Some(snapshot),
            sampling,
            reply,
        })?
    }

    pub fn remove(&self, id: RequestId) -> Result<bool, RuntimeError> {
        self.request(|reply| RuntimeCommand::Remove { id, reply })
    }

    pub(crate) fn execute(&self, batch: BatchDescription) -> Result<RuntimeStep, RuntimeError> {
        self.request(|reply| RuntimeCommand::Execute { batch, reply })?
    }

    pub fn snapshot(&self, id: RequestId) -> Result<Arc<SequenceSnapshot>, RuntimeError> {
        self.request(|reply| RuntimeCommand::Snapshot { id, reply })?
    }
}

impl Drop for DeviceRuntimeHandle {
    fn drop(&mut self) {
        let _ = self.commands.send(RuntimeCommand::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl DeviceRuntime {
    pub fn load(
        device_ordinal: usize,
        model_path: &Path,
        config: ModelConfig,
        runtime: RuntimeConfig,
    ) -> Result<Self, RuntimeError> {
        let RuntimeConfig {
            prefill_chunk,
            max_batch,
            ngram,
            mtp_drafts,
            retention_interval,
            snapshot_slots,
            stop_on_eos,
            mmproj,
            max_image_patches,
        } = runtime;
        let load_started = Instant::now();
        if prefill_chunk == 0 {
            return Err(RuntimeError::ZeroPrefillChunk);
        }
        if max_batch == 0 {
            return Err(RuntimeError::ZeroBatchWidth);
        }
        let ctx = CudaContext::new(device_ordinal)?;
        let stream = ctx.new_stream()?;
        // SAFETY: this runtime owns the context's serving stream and all
        // allocations. CUDA graph capture cannot coexist with cudarc's
        // per-allocation event tracking; disable it before the first upload.
        unsafe { ctx.disable_event_tracking() };

        let snapshot_arena =
            SnapshotArena::new(&ctx, &config, retention_interval.max(1), snapshot_slots)?;
        debug!(
            device = device_ordinal,
            slots = snapshot_arena.capacity(),
            bytes_per_slot = snapshot_arena.bytes_per_slot(),
            "worker pinned snapshot arena ready"
        );

        let file = GgufFile::open(model_path)?;
        let eos_token = stop_on_eos
            .then(|| file.get_u32("tokenizer.ggml.eos_token_id"))
            .flatten()
            .map(|token| token as i32);
        // MTP serving needs block 40's tensors in the directory; its expert
        // weights stay out of the arena either way (`arena_holds`) and are
        // uploaded exactly once by the first `MtpBlock` below.
        let schema = if mtp_drafts > 0 {
            WeightSchema::with_mtp(&config)
        } else {
            WeightSchema::new(&config)
        };
        let directory = schema
            .resolve(&file)
            .map_err(|errors| RuntimeError::Schema(format!("{errors:?}")))?;
        let (weights, _) =
            DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)?;
        debug!(
            device = device_ordinal,
            elapsed_ms = load_started.elapsed().as_secs_f64() * 1e3,
            "worker weights resident"
        );
        let prefill = Forward::new(
            &ctx,
            &stream,
            &file,
            &directory,
            &weights,
            config.clone(),
            prefill_chunk,
        )?;
        debug!(
            device = device_ordinal,
            elapsed_ms = load_started.elapsed().as_secs_f64() * 1e3,
            tokens = prefill_chunk,
            "worker prefill shape ready"
        );
        let retention_prefill = if retention_interval > 0 && retention_interval != prefill_chunk {
            Some(prefill.reshape(
                &ctx,
                &stream,
                &file,
                &directory,
                &weights,
                retention_interval,
            )?)
        } else {
            None
        };
        let tail_ceiling = prefill_chunk.min(retention_interval.max(1)).min(256);
        let mut prefill_tails = Vec::new();
        let mut tail = 2usize;
        while tail <= tail_ceiling {
            if tail > max_batch && tail != prefill_chunk && tail != retention_interval {
                prefill_tails.push((
                    tail,
                    prefill.reshape(&ctx, &stream, &file, &directory, &weights, tail)?,
                ));
            }
            tail *= 2;
        }
        debug!(
            device = device_ordinal,
            elapsed_ms = load_started.elapsed().as_secs_f64() * 1e3,
            tail_shapes = prefill_tails.len(),
            "worker prefill tail shapes ready"
        );

        // Index by batch width. Every serving width is built at startup, so
        // switching N=1/N=2/N=3 never allocates on the decode hot path.
        let mut decode = Vec::with_capacity(max_batch + 1);
        decode.push(None);
        for width in 1..=max_batch {
            let mut pass = prefill.reshape(&ctx, &stream, &file, &directory, &weights, width)?;
            pass.enable_batch_decode(&ctx, &stream)?;
            decode.push(Some(pass));
        }
        debug!(
            device = device_ordinal,
            elapsed_ms = load_started.elapsed().as_secs_f64() * 1e3,
            decode_shapes = max_batch,
            "worker decode shapes ready"
        );
        let graphs = (0..=max_batch).map(|_| None).collect();

        // Speculative verify shapes: one pass per decode width at
        // `width * (1 + draft_tokens)` tokens, plus one reusable ring set
        // per decode slot. Only when drafting is configured — the
        // non-speculative baseline must not pay a byte or a branch for
        // this — and `LLMXABE_NGRAM_GATED` keeps the old round-gated
        // behavior for A/B measurement.
        let window = if mtp_drafts > 0 {
            mtp_drafts + 1
        } else {
            ngram.map_or(0, |config| config.max_draft_tokens + 1)
        };
        // The gating env is the n-gram A/B lever only: MTP has no
        // round-gated fallback drafter, so it always verifies batched.
        let gated = mtp_drafts == 0 && std::env::var_os("LLMXABE_NGRAM_GATED").is_some();
        let mut verify: Vec<Option<Forward>> = Vec::with_capacity(max_batch + 1);
        verify.push(None);
        let mut verify_rings = Vec::new();
        if window >= 2 && !gated {
            for width in 1..=max_batch {
                let mut pass =
                    prefill.reshape(&ctx, &stream, &file, &directory, &weights, width * window)?;
                pass.enable_batch_decode(&ctx, &stream)?;
                pass.enable_verify(&stream)?;
                verify.push(Some(pass));
            }
            for _ in 0..max_batch {
                verify_rings.push(prefill.new_verify_rings(&stream, window)?);
            }
            debug!(
                device = device_ordinal,
                window,
                ring_bytes = verify_rings
                    .iter()
                    .flatten()
                    .map(GdnSnapshotRing::bytes)
                    .sum::<u64>(),
                "speculative verify shapes ready"
            );
        } else {
            verify.extend((1..=max_batch).map(|_| None));
        }

        // The MTP draft head: one block per shape the serving loop can ask
        // for. Prefill shapes catch the draft cache up chunk by chunk;
        // widths `1..=max_batch` run the chained batch draft (and the
        // narrow catch-up remainders, which is why they exist even though
        // drafting itself only ever uses the scheduled decode width).
        let mtp = if mtp_drafts > 0 {
            let rms_eps = file
                .get_f32(crate::forward::RMS_EPS_KEY)
                .ok_or_else(|| RuntimeError::Schema(crate::forward::RMS_EPS_KEY.into()))?;
            let rope_theta = file
                .get_f32(crate::forward::ROPE_FREQ_BASE_KEY)
                .ok_or_else(|| RuntimeError::Schema(crate::forward::ROPE_FREQ_BASE_KEY.into()))?;
            // Width shapes first, with the LM head — drafting needs it.
            // Prefill shapes follow without one; a duplicate token count
            // (e.g. a prefill remainder equal to a draft width) keeps the
            // LM-head variant, whose extra argmax on a catch-up call is
            // harmless. All reshapes share the first block's one upload of
            // the block-40 expert weights.
            let first = MtpBlock::new(
                &ctx, &stream, &file, &directory, &weights, &config, 1, rms_eps, rope_theta, true,
            )?;
            let mut shapes = vec![(1usize, first)];
            let want = |tokens: usize,
                        lm_head: bool,
                        shapes: &mut Vec<(usize, MtpBlock)>|
             -> Result<(), RuntimeError> {
                if shapes.iter().all(|(t, _)| *t != tokens) {
                    let block = shapes[0].1.reshape(
                        &ctx, &stream, &weights, &config, tokens, rms_eps, rope_theta, lm_head,
                    )?;
                    shapes.push((tokens, block));
                }
                Ok(())
            };
            for width in 2..=max_batch {
                want(width, true, &mut shapes)?;
            }
            want(prefill_chunk, false, &mut shapes)?;
            if retention_interval > 0 {
                want(retention_interval, false, &mut shapes)?;
            }
            let tail_ceiling = prefill_chunk.min(retention_interval.max(1)).min(256);
            let mut tail = 2usize;
            while tail <= tail_ceiling {
                if tail > max_batch {
                    want(tail, false, &mut shapes)?;
                }
                tail *= 2;
            }
            let hidden = config.hidden_size as usize;
            let max_shape = shapes
                .iter()
                .map(|(t, _)| *t)
                .max()
                .expect("at least the chunk shape exists");
            debug!(
                device = device_ordinal,
                elapsed_ms = load_started.elapsed().as_secs_f64() * 1e3,
                shapes = shapes.len(),
                drafts = mtp_drafts,
                "MTP draft head resident"
            );
            Some(MtpRuntime {
                shapes,
                h: stream.alloc_zeros::<f32>(max_shape * hidden)?,
                positions: stream.alloc_zeros::<i32>(max_batch)?,
                pos_host: vec![0; max_batch],
                drafts: mtp_drafts,
                hidden,
                config: config.clone(),
            })
        } else {
            None
        };

        // Vision tower: opt-in via --mmproj. Loading enables image-row
        // staging on every prefill-capable shape; text-only serving skips
        // all of it, which is what keeps the baseline untouched by
        // construction.
        let vision = match &mmproj {
            Some(path) => {
                let vision_file = GgufFile::open(path)?;
                let vision_cfg = xabe_model::VisionConfig::qwen3_6_35b_a3b();
                let vision_weights = crate::vision::load_vision_weights(&vision_file, &vision_cfg)
                    .map_err(RuntimeError::Vision)?;
                let tower = VisionForward::new(
                    &ctx,
                    Arc::clone(&stream),
                    &vision_cfg,
                    &vision_weights,
                    max_image_patches.max(4),
                )
                .map_err(|e| RuntimeError::Vision(e.to_string()))?;
                debug!(
                    device = device_ordinal,
                    elapsed_ms = load_started.elapsed().as_secs_f64() * 1e3,
                    max_image_patches,
                    "vision tower resident"
                );
                Some(tower)
            }
            None => None,
        };

        let mut this = Self {
            ctx,
            stream,
            prefill,
            retention_prefill,
            prefill_tails,
            decode,
            graphs,
            verify,
            verify_rings,
            window,
            mtp,
            weights,
            sequences: HashMap::with_capacity(max_batch),
            prefill_chunk,
            max_batch,
            vocab: config.vocab_size,
            ngram,
            retention_interval,
            snapshot_arena,
            sampled: Vec::with_capacity(max_batch),
            host_logits: Vec::with_capacity(config.vocab_size as usize),
            sample_scratch: Vec::with_capacity(config.vocab_size as usize),
            eos_token,
            vision,
            mrope_host: Vec::new(),
        };
        if this.vision.is_some() {
            let stream = Arc::clone(&this.stream);
            this.prefill.enable_image_injection(&stream)?;
            if let Some(fwd) = this.retention_prefill.as_mut() {
                fwd.enable_image_injection(&stream)?;
            }
            for (_, fwd) in this.prefill_tails.iter_mut() {
                fwd.enable_image_injection(&stream)?;
            }
            for fwd in this.decode.iter_mut().flatten() {
                fwd.enable_image_injection(&stream)?;
            }
        }
        Ok(this)
    }

    pub fn device_ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    pub fn resident_weight_bytes(&self) -> usize {
        self.weights.arena().used()
    }

    pub fn admit(
        &mut self,
        req: NewRequest,
        prompt: Vec<i32>,
        images: Vec<SequenceImage>,
        sampling: SamplingParams,
    ) -> Result<(), RuntimeError> {
        if self.sequences.contains_key(&req.id) {
            return Err(RuntimeError::DuplicateRequest(req.id));
        }
        if prompt.len() != req.prompt_tokens as usize {
            return Err(RuntimeError::PromptLength {
                declared: req.prompt_tokens,
                actual: prompt.len(),
            });
        }
        if let Some(&token) = prompt
            .iter()
            .find(|&&token| token < 0 || token as u32 >= self.vocab)
        {
            return Err(RuntimeError::TokenOutOfRange {
                token,
                vocab: self.vocab,
            });
        }
        let placements: Vec<ImagePlacement> = images.iter().map(|i| i.placement).collect();
        validate_placements(&placements, prompt.len()).map_err(RuntimeError::ImagePlacement)?;
        let image_embeds = self.encode_images(&images)?;
        let state = self
            .prefill
            .new_state(&self.stream, req.full_seq_len() as usize)?;
        let draft = Vec::with_capacity(self.window.saturating_sub(1));
        let ngram = self.ngram.map(NgramSpeculator::new);
        // The draft head's per-sequence state: its own KV cache and seed
        // hidden state. Admission-time allocation, like image embeddings —
        // never on the per-step path rule 6 governs. Image-bearing
        // sequences are excluded: the draft head embeds prompt token ids,
        // and image spans have no token ids to embed, only injected rows.
        let mtp = match &self.mtp {
            Some(rt) if images.is_empty() => Some(MtpSequence {
                cache: KvCache::new(
                    &self.stream,
                    &rt.config,
                    req.full_seq_len() as usize + rt.drafts,
                )
                .map_err(|e| RuntimeError::Mtp(MtpBlockError::Attention(e)))?,
                pending_h: self.stream.alloc_zeros::<f32>(rt.hidden)?,
            }),
            _ => None,
        };
        self.sequences.insert(
            req.id,
            RuntimeSequence {
                state: Some(state),
                prompt,
                prefilled: 0,
                next_token: None,
                ngram,
                mtp,
                draft,
                emitted: 0,
                max_output: req.max_output_tokens,
                last_snapshot_position: 0,
                last_snapshot: None,
                retention_disabled: false,
                sampler: (!sampling.is_greedy()).then(|| Sampler::new(sampling)),
                images: placements,
                image_embeds,
            },
        );
        Ok(())
    }

    /// Encode a request's images through the vision tower into one device
    /// buffer, embeddings concatenated in placement order.
    ///
    /// This is the only per-request device allocation in the engine, and it
    /// happens at admission — never on the per-step path rule 6 governs.
    fn encode_images(
        &mut self,
        images: &[SequenceImage],
    ) -> Result<Option<cudarc::driver::CudaSlice<f32>>, RuntimeError> {
        if images.is_empty() {
            return Ok(None);
        }
        let vision = self.vision.as_mut().ok_or(RuntimeError::VisionNotEnabled)?;
        let merge = vision.config().spatial_merge;
        let hidden = vision.config().projection_dim as usize;
        let total: usize = images.iter().map(|i| i.placement.tokens()).sum();
        let embeds = self
            .stream
            .alloc_zeros::<f32>(total * hidden)
            .map_err(RuntimeError::Driver)?;
        let mut row = 0usize;
        for img in images {
            let (mh, mw) = (img.image.grid_h / merge, img.image.grid_w / merge);
            if (img.placement.grid_h, img.placement.grid_w) != (mh, mw) {
                return Err(RuntimeError::ImagePlacement(format!(
                    "placement grid {}x{} does not match the preprocessed {}x{}",
                    img.placement.grid_h, img.placement.grid_w, mh, mw
                )));
            }
            let out = vision
                .encode(&img.image)
                .map_err(|e| RuntimeError::Vision(e.to_string()))?;
            let n = img.placement.tokens() * hidden;
            // SAFETY: `out` holds at least `n` floats (the tower checked the
            // patch budget) and `embeds` was sized to the placement total.
            let src = unsafe { crate::viewslice::subslice(&self.stream, out, 0, n) };
            let mut dst =
                unsafe { crate::viewslice::subslice(&self.stream, &embeds, row * hidden, n) };
            self.stream
                .memcpy_dtod(&*src, &mut *dst)
                .map_err(RuntimeError::Driver)?;
            row += img.placement.tokens();
        }
        Ok(Some(embeds))
    }

    /// Admit with a prefix restored from pinned host memory.
    ///
    /// A snapshot covering the *entire* prompt hands the sequence its first
    /// output token via [`SequenceSnapshot::next_token`], which is only valid
    /// for a greedy consumer inheriting from a greedy producer; the engine's
    /// placement path is what guarantees a sampling request never gets here
    /// with such a snapshot.
    pub fn admit_restored(
        &mut self,
        req: NewRequest,
        prompt: Vec<i32>,
        images: Vec<SequenceImage>,
        snapshot: Arc<SequenceSnapshot>,
        sampling: SamplingParams,
    ) -> Result<(), RuntimeError> {
        let prefix = snapshot.position();
        if prefix > prompt.len() {
            return Err(RuntimeError::SnapshotPrefix {
                snapshot: prefix,
                prompt: prompt.len(),
            });
        }
        self.admit(req, prompt, images, sampling)?;
        let seq = self
            .sequences
            .get_mut(&req.id)
            .expect("admit inserted the request");
        seq.state
            .as_mut()
            .expect("resident request has state")
            .restore(&self.stream, &snapshot)?;
        seq.prefilled = prefix;
        // The restored prefix may cover image spans; the rotary base for
        // whatever runs next lags the slot position by their accumulated
        // delta. (The prefill loop re-derives this per chunk; this covers
        // the full-restore case that goes straight to decode.)
        let delta = rope_delta_at(&seq.images, prefix);
        seq.state
            .as_mut()
            .expect("resident request has state")
            .set_rope_delta(delta);
        seq.last_snapshot_position = prefix;
        seq.last_snapshot = Some(Arc::clone(&snapshot));
        seq.retention_disabled = false;
        // Snapshots restore the target's state only; the draft head's cache
        // over the restored prefix was computed by whoever produced the
        // snapshot and is gone. Drafting over that hole would propose from
        // zeroed attention slots, so this sequence decodes plain instead.
        seq.mtp = None;
        if prefix == seq.prompt.len() {
            seq.next_token = snapshot.next_token();
        }
        if let Some(ngram) = &mut seq.ngram {
            ngram.observe_all(&seq.prompt[..prefix]);
        }
        Ok(())
    }

    /// Capture a currently resident request at its exact computed position.
    pub fn snapshot(&self, id: RequestId) -> Result<Arc<SequenceSnapshot>, RuntimeError> {
        let seq = self
            .sequences
            .get(&id)
            .ok_or(RuntimeError::UnknownRequest(id))?;
        let mut snapshot = seq
            .state
            .as_ref()
            .ok_or(RuntimeError::UnknownRequest(id))?
            .snapshot(
                &self.stream,
                &self.snapshot_arena,
                seq.last_snapshot.clone(),
            )?;
        // A sampled sequence's pending token was drawn from *its* RNG; naming
        // it on the snapshot would hand that draw to whichever request resumes
        // here. Leave it unset — a consumer whose prompt ends exactly at this
        // snapshot is then refused the restore instead of inheriting it.
        if seq.sampler.is_none()
            && let Some(token) = seq.next_token
        {
            snapshot.set_next_token(token);
        }
        Ok(Arc::new(snapshot))
    }

    pub fn remove(&mut self, id: RequestId) -> bool {
        self.sequences.remove(&id).is_some()
    }

    pub(crate) fn execute(
        &mut self,
        scheduled: BatchDescription,
    ) -> Result<RuntimeStep, RuntimeError> {
        let mut generated = SmallVec::new();
        let mut retained = SmallVec::new();
        let mut stopped = SmallVec::new();
        self.execute_decodes(&scheduled, &mut generated, &mut retained, &mut stopped)?;
        for item in &scheduled.prefills {
            self.execute_prefill(
                item.id,
                item.tokens as usize,
                &mut generated,
                &mut retained,
                &mut stopped,
            )?;
        }
        let completed: SmallVec<[RequestId; 3]> = self
            .sequences
            .iter()
            .filter_map(|(&id, seq)| (seq.emitted >= seq.max_output).then_some(id))
            .collect();
        for id in &completed {
            self.sequences.remove(id);
        }
        Ok(RuntimeStep {
            scheduled,
            generated,
            completed,
            stopped,
            retained,
        })
    }

    fn execute_decodes(
        &mut self,
        scheduled: &BatchDescription,
        generated: &mut SmallVec<[(RequestId, i32); 16]>,
        retained: &mut SmallVec<[(RequestId, Arc<SequenceSnapshot>); 8]>,
        stopped: &mut SmallVec<[RequestId; 3]>,
    ) -> Result<(), RuntimeError> {
        let width = scheduled.decodes.len();
        if width == 0 {
            return Ok(());
        }
        if width > self.max_batch {
            return Err(RuntimeError::BatchTooWide {
                requested: width,
                maximum: self.max_batch,
            });
        }

        // Moving the states out avoids aliasing the pass and the request map.
        // Device allocations keep stable addresses when their owning Rust
        // values move, which is the property Forward's state split relies on.
        let mut owned: SmallVec<[(RequestId, RuntimeSequence); 3]> = SmallVec::new();
        for item in &scheduled.decodes {
            let seq = self
                .sequences
                .remove(&item.id)
                .ok_or(RuntimeError::UnknownRequest(item.id))?;
            owned.push((item.id, seq));
        }

        for (_, seq) in &mut owned {
            seq.draft.clear();
        }
        // The MTP head drafts for every eligible sequence in one chained
        // batch; the n-gram speculator proposes per sequence. The two are
        // mutually exclusive by configuration, so exactly one fills drafts.
        if self.mtp.is_some() {
            self.mtp_draft_chain(&mut owned)?;
        }
        for (_, seq) in &mut owned {
            if let Some(ngram) = &seq.ngram {
                ngram.propose_into(&mut seq.draft);
            }
            let remaining_after_plain = seq.max_output.saturating_sub(seq.emitted + 1) as usize;
            seq.draft.truncate(remaining_after_plain);
            // A verify step advances up to `draft + 1` positions at once;
            // capping the draft at the next retention boundary means the
            // step can land exactly on it but never skip it, so retained
            // snapshots keep their every-`retention_interval` cadence.
            if self.retention_interval > 0 && !seq.retention_disabled {
                let position = seq
                    .state
                    .as_ref()
                    .expect("resident sequence has state")
                    .position();
                let to_boundary = self.retention_interval - position % self.retention_interval;
                seq.draft.truncate(to_boundary.saturating_sub(1));
            }
        }

        // Drafted tokens are fed through a single batched verify pass when
        // one was built: every sequence's whole window shares one weight
        // read, which is the speculative saving the round-gated loop below
        // never had. Empty drafts (or no verify pass) fall through to the
        // plain captured batch-decode round — except under MTP, where a
        // sequence with a draft head always takes the verify step even with
        // an empty (boundary-truncated) draft, because only the verify
        // pass's `final_norm` can refresh its seed hidden state exactly.
        if self.verify[width].is_some()
            && owned
                .iter()
                .any(|(_, seq)| !seq.draft.is_empty() || seq.mtp.is_some())
        {
            let result = self.verify_decode_step(&mut owned, generated, retained, stopped);
            for (id, seq) in owned {
                self.sequences.insert(id, seq);
            }
            return result;
        }

        // Round zero is the normal N-wide decode. Sequences whose draft token
        // matches stay active for another target-model round; mismatches stop
        // immediately with the target token. Full acceptance runs one final
        // round to obtain the standard speculative bonus token.
        let mut active: SmallVec<[usize; 3]> = (0..width).collect();
        let mut outputs = std::mem::take(&mut self.sampled);
        let mut round = 0usize;
        while !active.is_empty() {
            let mut inputs: SmallVec<[i32; 3]> = SmallVec::new();
            let mut states: SmallVec<[SequenceState; 3]> = SmallVec::new();
            for &index in &active {
                let id = owned[index].0;
                let seq = &mut owned[index].1;
                inputs.push(seq.next_token.ok_or(RuntimeError::UnknownRequest(id))?);
                states.push(seq.state.take().ok_or(RuntimeError::UnknownRequest(id))?);
            }
            let active_ids: SmallVec<[RequestId; 3]> =
                active.iter().map(|&index| owned[index].0).collect();
            let width = active.len();
            let pass = self.decode[width]
                .as_mut()
                .expect("every serving width was prebuilt");
            let graph_matches = self.graphs[width]
                .as_ref()
                .is_some_and(|(ids, _)| ids == &active_ids);
            if !graph_matches {
                let graph = pass.capture_batch_step(&self.stream, &mut states)?;
                self.graphs[width] = Some((active_ids, graph));
            }
            let replay = pass.replay_batch_step_into(
                &self.stream,
                &mut states,
                &self.graphs[width]
                    .as_ref()
                    .expect("graph installed above")
                    .1,
                &inputs,
                &mut outputs,
            );
            if let Err(error) = replay {
                outputs.clear();
                self.sampled = outputs;
                return Err(error.into());
            }

            // Host sampling overrides the device argmax verdict, per sequence
            // that asked for it, before anything downstream reads `outputs`.
            // The replayed step has already synchronized for the argmax
            // read-back, so the logits rows are final. Draft acceptance below
            // stays exact under sampling: the emitted token *is* the target
            // model's draw, and a draft is accepted only by equaling it.
            if active.iter().any(|&index| owned[index].1.sampler.is_some()) {
                let stream = Arc::clone(&self.stream);
                let mut host = std::mem::take(&mut self.host_logits);
                let mut scratch = std::mem::take(&mut self.sample_scratch);
                for (row, &index) in active.iter().enumerate() {
                    if owned[index].1.sampler.is_none() {
                        continue;
                    }
                    let pass = self.decode[width]
                        .as_ref()
                        .expect("every serving width was prebuilt");
                    if let Err(error) = pass.read_batch_logits_row_into(&stream, row, &mut host) {
                        outputs.clear();
                        self.sampled = outputs;
                        self.host_logits = host;
                        self.sample_scratch = scratch;
                        return Err(error.into());
                    }
                    let sampler = owned[index].1.sampler.as_mut().expect("checked above");
                    outputs[row] = sampler.sample(&host, &mut scratch);
                }
                self.host_logits = host;
                self.sample_scratch = scratch;
            }

            let mut next_active: SmallVec<[usize; 3]> = SmallVec::new();
            for (((index, state), output), _) in active
                .into_iter()
                .zip(states)
                .zip(outputs.iter().copied())
                .zip(inputs)
            {
                let (id, seq) = &mut owned[index];
                seq.state = Some(state);
                let position = seq
                    .state
                    .as_ref()
                    .expect("state was restored above")
                    .position();
                if self.retention_interval > 0
                    && !seq.retention_disabled
                    && position > seq.last_snapshot_position
                    && position.is_multiple_of(self.retention_interval)
                {
                    let snapshot = seq
                        .state
                        .as_ref()
                        .expect("state was restored above")
                        .snapshot(
                            &self.stream,
                            &self.snapshot_arena,
                            seq.last_snapshot.clone(),
                        );
                    match snapshot {
                        Ok(mut snapshot) => {
                            // A sampled token is this sequence's own draw;
                            // recording it would let another request inherit
                            // it as a first output token. See `snapshot()`.
                            if seq.sampler.is_none() {
                                snapshot.set_next_token(output);
                            }
                            let snapshot = Arc::new(snapshot);
                            seq.last_snapshot = Some(Arc::clone(&snapshot));
                            retained.push((*id, snapshot));
                            seq.last_snapshot_position = position;
                        }
                        Err(StateError::SnapshotArenaExhausted) => {
                            seq.retention_disabled = true;
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
                seq.next_token = Some(output);
                if self.eos_token == Some(output) {
                    seq.emitted = seq.max_output;
                    stopped.push(*id);
                    continue;
                }
                if let Some(ngram) = &mut seq.ngram {
                    ngram.observe(output);
                }
                generated.push((*id, output));
                seq.emitted += 1;
                if seq.emitted < seq.max_output
                    && owned[index]
                        .1
                        .draft
                        .get(round)
                        .is_some_and(|&draft| draft == output)
                {
                    next_active.push(index);
                }
            }
            active = next_active;
            outputs.clear();
            round += 1;
        }
        self.sampled = outputs;

        for (id, seq) in owned {
            self.sequences.insert(id, seq);
        }
        Ok(())
    }

    /// Draft up to `drafts` tokens for every MTP-eligible scheduled
    /// sequence, greedily, in one chained batch: each round runs the draft
    /// head once at the participant width (weight-bound projections batch
    /// across sequences) and feeds its own `h_nextn` back as the next
    /// round's hidden input — the draft chain, never the target's `h`,
    /// which has not run these positions yet.
    ///
    /// Drafted ids only *propose*; the verify pass recomputes everything,
    /// so nothing here can change what is emitted — only how often drafts
    /// are accepted.
    fn mtp_draft_chain(
        &mut self,
        owned: &mut SmallVec<[(RequestId, RuntimeSequence); 3]>,
    ) -> Result<(), RuntimeError> {
        let stream = Arc::clone(&self.stream);
        let rt = self.mtp.as_mut().expect("caller checked MTP is on");
        let MtpRuntime {
            shapes,
            h,
            positions,
            pos_host,
            drafts,
            hidden,
            ..
        } = rt;
        let (drafts, hidden) = (*drafts, *hidden);
        let parts: SmallVec<[usize; 3]> = owned
            .iter()
            .enumerate()
            .filter(|(_, (_, seq))| seq.mtp.is_some() && seq.next_token.is_some())
            .map(|(index, _)| index)
            .collect();
        let w = parts.len();
        if w == 0 || drafts == 0 {
            return Ok(());
        }
        let block_index = shapes
            .iter()
            .position(|(t, _)| *t == w)
            .expect("every draft width was prebuilt");
        let block = &mut shapes[block_index].1;

        // Seed the chain with each participant's pending target hidden row.
        for (r, &i) in parts.iter().enumerate() {
            let sm = owned[i].1.mtp.as_ref().expect("participant has a head");
            let mut dst = h.slice_mut(r * hidden..(r + 1) * hidden);
            stream.memcpy_dtod(&sm.pending_h, &mut dst)?;
        }
        let mut toks: SmallVec<[i32; 3]> = parts
            .iter()
            .map(|&i| owned[i].1.next_token.expect("participant has a token"))
            .collect();
        let base: SmallVec<[usize; 3]> = parts
            .iter()
            .map(|&i| {
                owned[i]
                    .1
                    .state
                    .as_ref()
                    .expect("resident sequence has state")
                    .position()
            })
            .collect();

        for j in 0..drafts {
            for (r, &b) in base.iter().enumerate() {
                pos_host[r] = (b + j) as i32;
            }
            {
                let mut dst = positions.slice_mut(0..w);
                stream.memcpy_htod(&pos_host[..w], &mut dst)?;
            }
            let pos_offsets: SmallVec<[usize; 3]> = base.iter().map(|&b| b + j).collect();
            // SAFETY: `r < w <= max_batch`, the length `positions` was
            // allocated with; `w * hidden` is within `h`'s largest-shape
            // allocation.
            let pos_views: SmallVec<[_; 3]> = (0..w)
                .map(|r| unsafe { crate::viewslice::subslice(&stream, positions, r, 1) })
                .collect();
            let pos_refs: SmallVec<[&cudarc::driver::CudaSlice<i32>; 3]> =
                pos_views.iter().map(|view| &**view).collect();
            let mut caches: SmallVec<[&mut KvCache; 3]> = owned
                .iter_mut()
                .filter_map(|(_, seq)| {
                    if seq.next_token.is_some() {
                        seq.mtp.as_mut().map(|sm| &mut sm.cache)
                    } else {
                        None
                    }
                })
                .collect();
            let sampled = {
                let h_view = unsafe { crate::viewslice::subslice(&stream, h, 0, w * hidden) };
                block.forward_batch_decode(
                    &stream,
                    &toks,
                    &h_view,
                    &mut caches,
                    &pos_offsets,
                    &pos_refs,
                )?
            }
            .expect("draft width shapes are built with the LM head");
            drop(caches);
            for (r, &i) in parts.iter().enumerate() {
                owned[i].1.draft.push(sampled[r]);
                toks[r] = sampled[r];
            }
            if j + 1 < drafts {
                let mut dst = h.slice_mut(0..w * hidden);
                stream.memcpy_dtod(block.h_nextn(), &mut dst)?;
            }
        }
        Ok(())
    }

    /// One true speculative decode step: every scheduled sequence's window
    /// (`[id_last, draft...]`, padded with `id_last` where a draft came up
    /// short) through one batched verify pass, then per-sequence
    /// acceptance, rollback and bookkeeping.
    ///
    /// Emitted tokens are exactly the target model's own choices at every
    /// position — a draft is accepted only by equaling the target's
    /// argmax (or, for a sampling sequence, its own draw from the same
    /// logits row the plain path would have produced), and a rejected
    /// position emits the target's token in its place. What changes is the
    /// number of weight-read passes per emitted token, never the tokens.
    fn verify_decode_step(
        &mut self,
        owned: &mut SmallVec<[(RequestId, RuntimeSequence); 3]>,
        generated: &mut SmallVec<[(RequestId, i32); 16]>,
        retained: &mut SmallVec<[(RequestId, Arc<SequenceSnapshot>); 8]>,
        stopped: &mut SmallVec<[RequestId; 3]>,
    ) -> Result<(), RuntimeError> {
        let width = owned.len();
        let window = self.window;

        let mut ids: SmallVec<[i32; 16]> = SmallVec::new();
        let mut states: SmallVec<[SequenceState; 3]> = SmallVec::new();
        for (id, seq) in owned.iter_mut() {
            let last = seq.next_token.ok_or(RuntimeError::UnknownRequest(*id))?;
            ids.push(last);
            for j in 0..window - 1 {
                // Padding a short draft with `id_last` keeps the pass shape
                // fixed; the padded rows are never compared, never emitted,
                // and their state is rolled back by the commit below.
                ids.push(seq.draft.get(j).copied().unwrap_or(last));
            }
            states.push(seq.state.take().ok_or(RuntimeError::UnknownRequest(*id))?);
        }

        let pass = self.verify[width]
            .as_mut()
            .expect("caller checked the verify pass exists");
        let rows = match pass.run_batch_verify(
            &self.stream,
            &mut states,
            &mut self.verify_rings[..width],
            &ids,
        ) {
            Ok(rows) => rows,
            Err(error) => {
                for ((_, seq), state) in owned.iter_mut().zip(states) {
                    seq.state = Some(state);
                }
                return Err(error.into());
            }
        };

        let stream = Arc::clone(&self.stream);
        let mut host = std::mem::take(&mut self.host_logits);
        let mut scratch = std::mem::take(&mut self.sample_scratch);
        let mut result = Ok(());
        for (s, mut state) in states.into_iter().enumerate() {
            let (id, seq) = &mut owned[s];
            if result.is_err() {
                // A failure on an earlier sequence: restore and skip, so
                // every sequence keeps its state even on the error path.
                seq.state = Some(state);
                continue;
            }
            let seq_rows = &rows[s * window..(s + 1) * window];
            let d_real = seq.draft.len();

            // The emissions, in order: accepted drafts then the bonus (or
            // the first mismatch's target token). For a sampling sequence
            // each row is drawn with its own sampler — the acceptance rule
            // is the same equality the round-gated path used, so sampling
            // stays exact: row `j`'s logits are conditioned on the drafted
            // prefix, which the loop has already proven equal to the
            // emitted prefix.
            let mut emit: SmallVec<[i32; 8]> = SmallVec::new();
            if let Some(sampler) = seq.sampler.as_mut() {
                let pass = self.verify[width].as_ref().expect("checked above");
                for j in 0..=d_real {
                    if let Err(error) =
                        pass.read_batch_logits_row_into(&stream, s * window + j, &mut host)
                    {
                        result = Err(error.into());
                        break;
                    }
                    let token = sampler.sample(&host, &mut scratch);
                    emit.push(token);
                    if j >= d_real || token != seq.draft[j] {
                        break;
                    }
                }
                if result.is_err() {
                    seq.state = Some(state);
                    continue;
                }
            } else {
                let mut accepted = 0usize;
                while accepted < d_real && seq_rows[accepted] == seq.draft[accepted] {
                    accepted += 1;
                }
                emit.extend_from_slice(&seq.draft[..accepted]);
                emit.push(seq_rows[accepted]);
            }

            // A mid-window end-of-turn truncates the step at the eos: the
            // tokens before it are real inputs, the eos itself ends the
            // sequence exactly as it does on the plain decode path.
            let mut hit_eos = false;
            if let Some(eos) = self.eos_token
                && let Some(at) = emit.iter().position(|&token| token == eos)
            {
                emit.truncate(at + 1);
                hit_eos = true;
            }

            // Inputs folded for real: `id_last` plus every emitted token
            // except the last (which has not been fed back yet).
            let commit_positions = emit.len();
            {
                let pass = self.verify[width].as_ref().expect("checked above");
                if let Err(error) = pass.commit_verify_window(
                    &stream,
                    &mut state,
                    &self.verify_rings[s],
                    commit_positions,
                ) {
                    result = Err(error.into());
                    seq.state = Some(state);
                    continue;
                }
            }

            let last = *emit.last().expect("a verify step always emits");
            seq.next_token = Some(last);

            // The next draft chain's seed: the target's own hidden state at
            // the window row that produced `last`. Exactness never depends
            // on this — a stale seed can only cost acceptance rate — but
            // this row is exact, which is the point of always routing MTP
            // sequences through the verify step.
            if let Some(sm) = seq.mtp.as_mut() {
                let hidden = self
                    .mtp
                    .as_ref()
                    .expect("a sequence has a draft head only when MTP is on")
                    .hidden;
                let pass = self.verify[width].as_ref().expect("checked above");
                let row = (s * window + emit.len() - 1) * hidden;
                let src = pass.final_norm().slice(row..row + hidden);
                if let Err(error) = stream.memcpy_dtod(&src, &mut sm.pending_h) {
                    result = Err(error.into());
                    seq.state = Some(state);
                    continue;
                }
            }

            // Retention: the draft cap in `execute_decodes` means the step
            // can land exactly on a boundary but never cross it, so the
            // every-interval cadence holds. Same rules as the plain path:
            // sampled sequences record no next token on the snapshot.
            let position = state.position();
            if self.retention_interval > 0
                && !seq.retention_disabled
                && position > seq.last_snapshot_position
                && position.is_multiple_of(self.retention_interval)
            {
                match state.snapshot(&stream, &self.snapshot_arena, seq.last_snapshot.clone()) {
                    Ok(mut snapshot) => {
                        if seq.sampler.is_none() {
                            snapshot.set_next_token(last);
                        }
                        let snapshot = Arc::new(snapshot);
                        seq.last_snapshot = Some(Arc::clone(&snapshot));
                        retained.push((*id, snapshot));
                        seq.last_snapshot_position = position;
                    }
                    Err(StateError::SnapshotArenaExhausted) => {
                        seq.retention_disabled = true;
                    }
                    Err(error) => {
                        result = Err(error.into());
                        seq.state = Some(state);
                        continue;
                    }
                }
            }

            let real = if hit_eos {
                &emit[..emit.len() - 1]
            } else {
                &emit[..]
            };
            for &token in real {
                if let Some(ngram) = &mut seq.ngram {
                    ngram.observe(token);
                }
                generated.push((*id, token));
                seq.emitted += 1;
            }
            if hit_eos {
                seq.emitted = seq.max_output;
                stopped.push(*id);
            }
            seq.state = Some(state);
        }
        self.host_logits = host;
        self.sample_scratch = scratch;
        result
    }

    /// The prebuilt pass that just ran a prefill piece of `width` tokens —
    /// the same four-way choice the launch site makes, so the logits being
    /// read are the ones that pass produced.
    fn prefill_pass(&mut self, width: usize) -> &mut Forward {
        prefill_pass_in(
            &mut self.prefill,
            &mut self.retention_prefill,
            &mut self.prefill_tails,
            &mut self.decode,
            self.prefill_chunk,
            self.retention_interval,
            width,
        )
    }

    /// Catch the draft head's own KV cache up over the prefill piece that
    /// just ran: pair token `p` with the target's `final_norm` row at
    /// `p - 1` (`pending_h` carries the last row across chunk boundaries,
    /// and is zero before position 0), run the same-shaped MTP block into
    /// the sequence's draft cache, and advance the carry.
    fn mtp_catchup(
        &mut self,
        width: usize,
        piece: &[i32],
        start: usize,
        sm: &mut MtpSequence,
    ) -> Result<(), RuntimeError> {
        let Self {
            prefill,
            retention_prefill,
            prefill_tails,
            decode,
            prefill_chunk,
            retention_interval,
            mtp,
            stream,
            ..
        } = self;
        let Some(rt) = mtp.as_mut() else {
            return Ok(());
        };
        let MtpRuntime {
            shapes,
            h,
            positions,
            hidden,
            ..
        } = rt;
        let hidden = *hidden;
        let pass = prefill_pass_in(
            prefill,
            retention_prefill,
            prefill_tails,
            decode,
            *prefill_chunk,
            *retention_interval,
            width,
        );
        let fnorm = pass.final_norm();
        {
            let mut dst = h.slice_mut(0..hidden);
            stream.memcpy_dtod(&sm.pending_h, &mut dst)?;
        }
        if width > 1 {
            let src = fnorm.slice(0..(width - 1) * hidden);
            let mut dst = h.slice_mut(hidden..width * hidden);
            stream.memcpy_dtod(&src, &mut dst)?;
        }
        {
            let src = fnorm.slice((width - 1) * hidden..width * hidden);
            stream.memcpy_dtod(&src, &mut sm.pending_h)?;
        }
        // SAFETY: element 0 of `positions` (len `max_batch >= 1`) and the
        // first `width * hidden` elements of `h` (sized for the largest
        // shape) — both in bounds by construction.
        let mut pos0 = unsafe { crate::viewslice::subslice(stream, positions, 0, 1) };
        stream.memcpy_htod(&[start as i32], &mut *pos0)?;
        let h_view = unsafe { crate::viewslice::subslice(stream, h, 0, width * hidden) };
        let index = shapes
            .iter()
            .position(|(t, _)| *t == width)
            .expect("every prefill shape was prebuilt for the draft head");
        shapes[index]
            .1
            .forward(stream, piece, &h_view, &mut sm.cache, start, &pos0)?;
        Ok(())
    }

    /// Draw the next token from the logits `prefill_pass(width)` just
    /// produced, with `sampler`.
    fn sample_prefill_output(
        &mut self,
        sampler: &mut Sampler,
        width: usize,
    ) -> Result<i32, RuntimeError> {
        let stream = Arc::clone(&self.stream);
        let mut host = std::mem::take(&mut self.host_logits);
        let mut scratch = std::mem::take(&mut self.sample_scratch);
        let drawn = self
            .prefill_pass(width)
            .read_logits_into(&stream, &mut host)
            .map(|()| sampler.sample(&host, &mut scratch));
        self.host_logits = host;
        self.sample_scratch = scratch;
        Ok(drawn?)
    }

    fn execute_prefill(
        &mut self,
        id: RequestId,
        tokens: usize,
        generated: &mut SmallVec<[(RequestId, i32); 16]>,
        retained: &mut SmallVec<[(RequestId, Arc<SequenceSnapshot>); 8]>,
        stopped: &mut SmallVec<[RequestId; 3]>,
    ) -> Result<(), RuntimeError> {
        let mut seq = self
            .sequences
            .remove(&id)
            .ok_or(RuntimeError::UnknownRequest(id))?;
        let end = seq.prefilled + tokens;
        let work = &seq.prompt[seq.prefilled..end];
        let mut offset = 0usize;
        let mut last_shape = 1usize;
        while offset < work.len() {
            let position = seq
                .state
                .as_ref()
                .expect("resident sequence has state")
                .position();
            let remaining = work.len() - offset;
            let width = choose_prefill_width(
                position,
                remaining,
                self.prefill_chunk,
                self.retention_interval,
                self.prefill_tails.iter().map(|(width, _)| *width),
                self.max_batch,
            );
            let piece = &work[offset..offset + width];

            // Image-bearing sequences: fix the rotary base for this chunk,
            // and when the chunk overlaps a span, switch that one pass to
            // per-token rotary positions and stage the projector rows that
            // replace the placeholder embeddings. Text-only sequences set a
            // delta of 0 — the value the rotary scalar always carried — and
            // take neither branch.
            let stream = Arc::clone(&self.stream);
            seq.state
                .as_mut()
                .expect("resident sequence has state")
                .set_rope_delta(rope_delta_at(&seq.images, position));
            if chunk_overlaps_images(&seq.images, position, width) {
                let mut triples = std::mem::take(&mut self.mrope_host);
                fill_mrope_triples(&seq.images, position, width, &mut triples);
                let embeds = seq
                    .image_embeds
                    .as_ref()
                    .expect("overlapping spans imply encoded images");
                let fwd = self.prefill_pass(width);
                fwd.publish_mrope(&stream, &triples)?;
                let mut row_base = 0usize;
                for img in &seq.images {
                    let begin = img.start.max(position);
                    let end = img.end().min(position + width);
                    if begin < end {
                        fwd.stage_image_rows(
                            &stream,
                            embeds,
                            row_base + (begin - img.start),
                            begin - position,
                            end - begin,
                        )?;
                    }
                    row_base += img.tokens();
                }
                self.mrope_host = triples;
            }
            let run_result = self.prefill_pass(width).run(
                &stream,
                seq.state.as_mut().expect("resident sequence has state"),
                piece,
                |_, _| {},
            );
            {
                let fwd = self.prefill_pass(width);
                fwd.clear_mrope();
                fwd.clear_staged_image_rows();
            }
            run_result?;
            // Draft-head catch-up rides each chunk: the target's hidden
            // states for exactly these positions are sitting in the pass's
            // `final_norm` right now, and they are gone once the next chunk
            // overwrites it. `piece` borrows `seq.prompt` and the catch-up
            // state is `seq.mtp` — disjoint fields.
            if let Some(sm) = seq.mtp.as_mut() {
                self.mtp_catchup(width, piece, position, sm)?;
            }
            offset += width;
            last_shape = width;
            let position = seq
                .state
                .as_ref()
                .expect("resident sequence has state")
                .position();
            if self.retention_interval > 0
                && !seq.retention_disabled
                && position > seq.last_snapshot_position
                && position.is_multiple_of(self.retention_interval)
            {
                // Inside the prompt this token is a greedy *prediction*,
                // recorded so a full-prefix resume knows what followed. At a
                // boundary that is also the end of the prompt it is the first
                // emitted token, so a sampling sequence draws it instead —
                // and then it must not be recorded, for the same reason
                // decode-time snapshots of sampled sequences record nothing.
                let sample_here = position == seq.prompt.len() && seq.sampler.is_some();
                let next = if sample_here {
                    let mut sampler = seq.sampler.take().expect("sample_here checked it");
                    let drawn = self.sample_prefill_output(&mut sampler, width);
                    seq.sampler = Some(sampler);
                    drawn?
                } else {
                    let stream = Arc::clone(&self.stream);
                    self.prefill_pass(width).sample_argmax(&stream)?
                };
                let snapshot = seq
                    .state
                    .as_ref()
                    .expect("resident sequence has state")
                    .snapshot(
                        &self.stream,
                        &self.snapshot_arena,
                        seq.last_snapshot.clone(),
                    );
                seq.next_token = Some(next);
                match snapshot {
                    Ok(mut snapshot) => {
                        if !sample_here {
                            snapshot.set_next_token(next);
                        }
                        let snapshot = Arc::new(snapshot);
                        seq.last_snapshot = Some(Arc::clone(&snapshot));
                        retained.push((id, snapshot));
                        seq.last_snapshot_position = position;
                    }
                    Err(StateError::SnapshotArenaExhausted) => {
                        seq.retention_disabled = true;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }
        seq.prefilled = end;
        // The delta that decode inherits is the one past the last span; a
        // final chunk that *contained* a span set the base for that chunk's
        // start, which is stale now.
        seq.state
            .as_mut()
            .expect("resident sequence has state")
            .set_rope_delta(rope_delta_at(&seq.images, seq.prefilled));
        if let Some(ngram) = &mut seq.ngram {
            ngram.observe_all(work);
        }
        if seq.prefilled == seq.prompt.len() && seq.max_output > 0 {
            let at_retained_boundary = seq
                .state
                .as_ref()
                .is_some_and(|state| state.position() == seq.last_snapshot_position);
            let next = if work.is_empty() || at_retained_boundary {
                // For a sampled sequence this token was drawn at the boundary
                // above; the work-is-empty case cannot be a sampled sequence,
                // because placement refuses it a snapshot covering the whole
                // prompt.
                seq.next_token.ok_or(RuntimeError::SnapshotPrefix {
                    snapshot: seq.prefilled,
                    prompt: seq.prompt.len(),
                })?
            } else if seq.sampler.is_some() {
                let mut sampler = seq.sampler.take().expect("checked just above");
                let drawn = self.sample_prefill_output(&mut sampler, last_shape);
                seq.sampler = Some(sampler);
                drawn?
            } else {
                let stream = Arc::clone(&self.stream);
                self.prefill_pass(last_shape).sample_argmax(&stream)?
            };
            seq.next_token = Some(next);
            if self.eos_token == Some(next) {
                seq.emitted = seq.max_output;
                stopped.push(id);
            } else {
                if let Some(ngram) = &mut seq.ngram {
                    ngram.observe(next);
                }
                generated.push((id, next));
                seq.emitted += 1;
            }
        }
        self.sequences.insert(id, seq);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::choose_prefill_width;

    #[test]
    fn ragged_prefill_uses_prebuilt_tail_shapes_not_token_at_a_time() {
        let tails = [4, 8, 16, 32, 64, 128, 256];
        let mut position = 0usize;
        let target = 3000usize;
        let mut widths = Vec::new();
        while position < target {
            let width = choose_prefill_width(
                position,
                target - position,
                4096,
                2048,
                tails.into_iter(),
                3,
            );
            assert!(position / 2048 == (position + width - 1) / 2048);
            widths.push(width);
            position += width;
        }
        assert_eq!(position, target);
        assert_eq!(widths[0], 2048);
        assert!(
            widths.len() < 10,
            "3K prefill should take a handful of passes"
        );
        assert_eq!(widths.iter().filter(|&&width| width == 1).count(), 0);
    }

    #[test]
    fn narrow_remainder_uses_the_prebuilt_decode_widths() {
        let tails = [4, 8, 16, 32, 64, 128, 256];
        assert_eq!(
            choose_prefill_width(2048, 3, 4096, 2048, tails.into_iter(), 3),
            3
        );
        assert_eq!(
            choose_prefill_width(2051, 2, 4096, 2048, tails.into_iter(), 3),
            2
        );
    }
}
