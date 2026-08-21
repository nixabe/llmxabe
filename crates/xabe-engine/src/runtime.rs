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
use xabe_sched::ngram::{NgramConfig, NgramSpeculator};
use xabe_sched::request::{BatchDescription, NewRequest, RequestId};

use crate::forward::{BatchStepGraph, Forward, ForwardError, arena_holds};
use crate::state::{SequenceSnapshot, SnapshotArena, SnapshotSlots};
use crate::{DeviceWeights, LoadError, SequenceState, StateError};

/// Pinned snapshot capacity per worker when no budget is given. One
/// default-retention slot is 102.8125 MiB, so 24 slots consume 2.41 GiB per
/// worker and 7.23 GiB process-wide.
/// This stays below the host's locked-memory limit while covering eight
/// simultaneous retention points for each of the three serving sequences.
/// `--cache-ram` overrides it; see `docs/CLI.md`.
pub const DEFAULT_SNAPSHOT_SLOTS_PER_WORKER: usize = 24;

/// Everything a device runtime needs beyond the model itself.
///
/// A struct rather than eight positional arguments because two of these are
/// `usize` counts that mean entirely different things, and swapping them
/// would compile.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    /// Tokens per chunked-prefill step.
    pub prefill_chunk: usize,
    /// Widest decode batch this worker will be asked to run.
    pub max_batch: usize,
    /// Speculative drafting, or `None` for no drafting at all.
    pub ngram: Option<NgramConfig>,
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
    PromptLength { declared: u32, actual: usize },
    TokenOutOfRange { token: i32, vocab: u32 },
    BatchTooWide { requested: usize, maximum: usize },
    SnapshotPrefix { snapshot: usize, prompt: usize },
    RuntimeStopped,
}

impl core::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Gguf(e) => write!(f, "GGUF error: {e}"),
            Self::Schema(e) => write!(f, "model schema mismatch: {e}"),
            Self::Load(e) => write!(f, "weight load failed: {e}"),
            Self::Forward(e) => write!(f, "forward pass failed: {e}"),
            Self::State(e) => write!(f, "sequence state failed: {e}"),
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

struct RuntimeSequence {
    state: Option<SequenceState>,
    prompt: Vec<i32>,
    prefilled: usize,
    next_token: Option<i32>,
    ngram: Option<NgramSpeculator>,
    draft: Vec<i32>,
    emitted: u32,
    max_output: u32,
    last_snapshot_position: usize,
    last_snapshot: Option<Arc<SequenceSnapshot>>,
    retention_disabled: bool,
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
    weights: DeviceWeights,
    sequences: HashMap<RequestId, RuntimeSequence>,
    prefill_chunk: usize,
    max_batch: usize,
    vocab: u32,
    ngram: Option<NgramConfig>,
    retention_interval: usize,
    snapshot_arena: SnapshotArena,
    sampled: Vec<i32>,
    eos_token: Option<i32>,
}

enum RuntimeCommand {
    Admit {
        req: NewRequest,
        prompt: Vec<i32>,
        snapshot: Option<Arc<SequenceSnapshot>>,
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
                                    snapshot,
                                    reply,
                                } => {
                                    let result = match snapshot {
                                        Some(snapshot) => {
                                            runtime.admit_restored(req, prompt, snapshot)
                                        }
                                        None => runtime.admit(req, prompt),
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

    pub fn admit(&self, req: NewRequest, prompt: Vec<i32>) -> Result<(), RuntimeError> {
        self.request(|reply| RuntimeCommand::Admit {
            req,
            prompt,
            snapshot: None,
            reply,
        })?
    }

    pub fn admit_restored(
        &self,
        req: NewRequest,
        prompt: Vec<i32>,
        snapshot: Arc<SequenceSnapshot>,
    ) -> Result<(), RuntimeError> {
        self.request(|reply| RuntimeCommand::Admit {
            req,
            prompt,
            snapshot: Some(snapshot),
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
            retention_interval,
            snapshot_slots,
            stop_on_eos,
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
        let schema = WeightSchema::new(&config);
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

        Ok(Self {
            ctx,
            stream,
            prefill,
            retention_prefill,
            prefill_tails,
            decode,
            graphs,
            weights,
            sequences: HashMap::with_capacity(max_batch),
            prefill_chunk,
            max_batch,
            vocab: config.vocab_size,
            ngram,
            retention_interval,
            snapshot_arena,
            sampled: Vec::with_capacity(max_batch),
            eos_token,
        })
    }

    pub fn device_ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    pub fn resident_weight_bytes(&self) -> usize {
        self.weights.arena().used()
    }

    pub fn admit(&mut self, req: NewRequest, prompt: Vec<i32>) -> Result<(), RuntimeError> {
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
        let state = self
            .prefill
            .new_state(&self.stream, req.full_seq_len() as usize)?;
        let draft = Vec::with_capacity(self.ngram.map_or(0, |config| config.max_draft_tokens));
        let ngram = self.ngram.map(NgramSpeculator::new);
        self.sequences.insert(
            req.id,
            RuntimeSequence {
                state: Some(state),
                prompt,
                prefilled: 0,
                next_token: None,
                ngram,
                draft,
                emitted: 0,
                max_output: req.max_output_tokens,
                last_snapshot_position: 0,
                last_snapshot: None,
                retention_disabled: false,
            },
        );
        Ok(())
    }

    /// Admit with a prefix restored from pinned host memory.
    pub fn admit_restored(
        &mut self,
        req: NewRequest,
        prompt: Vec<i32>,
        snapshot: Arc<SequenceSnapshot>,
    ) -> Result<(), RuntimeError> {
        let prefix = snapshot.position();
        if prefix > prompt.len() {
            return Err(RuntimeError::SnapshotPrefix {
                snapshot: prefix,
                prompt: prompt.len(),
            });
        }
        self.admit(req, prompt)?;
        let seq = self
            .sequences
            .get_mut(&req.id)
            .expect("admit inserted the request");
        seq.state
            .as_mut()
            .expect("resident request has state")
            .restore(&self.stream, &snapshot)?;
        seq.prefilled = prefix;
        seq.last_snapshot_position = prefix;
        seq.last_snapshot = Some(Arc::clone(&snapshot));
        seq.retention_disabled = false;
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
        if let Some(token) = seq.next_token {
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
            if let Some(ngram) = &seq.ngram {
                ngram.propose_into(&mut seq.draft);
            }
            let remaining_after_plain = seq.max_output.saturating_sub(seq.emitted + 1) as usize;
            seq.draft.truncate(remaining_after_plain);
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
                            snapshot.set_next_token(output);
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
            if width == self.prefill_chunk {
                self.prefill.run(
                    &self.stream,
                    seq.state.as_mut().expect("resident sequence has state"),
                    piece,
                    |_, _| {},
                )?;
            } else if width == self.retention_interval && width != 1 {
                self.retention_prefill
                    .as_mut()
                    .expect("distinct retention shape was prebuilt")
                    .run(
                        &self.stream,
                        seq.state.as_mut().expect("resident sequence has state"),
                        piece,
                        |_, _| {},
                    )?;
            } else if let Some((_, tail)) = self
                .prefill_tails
                .iter_mut()
                .find(|(tail_width, _)| *tail_width == width)
            {
                tail.run(
                    &self.stream,
                    seq.state.as_mut().expect("resident sequence has state"),
                    piece,
                    |_, _| {},
                )?;
            } else {
                self.decode[width]
                    .as_mut()
                    .expect("narrow prefill width was prebuilt")
                    .run(
                        &self.stream,
                        seq.state.as_mut().expect("resident sequence has state"),
                        piece,
                        |_, _| {},
                    )?;
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
                let next = if width == self.prefill_chunk {
                    self.prefill.sample_argmax(&self.stream)?
                } else if width == self.retention_interval && width != 1 {
                    self.retention_prefill
                        .as_mut()
                        .expect("distinct retention shape was prebuilt")
                        .sample_argmax(&self.stream)?
                } else if let Some((_, tail)) = self
                    .prefill_tails
                    .iter_mut()
                    .find(|(tail_width, _)| *tail_width == width)
                {
                    tail.sample_argmax(&self.stream)?
                } else {
                    self.decode[width]
                        .as_mut()
                        .expect("narrow prefill width was prebuilt")
                        .sample_argmax(&self.stream)?
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
                        snapshot.set_next_token(next);
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
        if let Some(ngram) = &mut seq.ngram {
            ngram.observe_all(work);
        }
        if seq.prefilled == seq.prompt.len() && seq.max_output > 0 {
            let at_retained_boundary = seq
                .state
                .as_ref()
                .is_some_and(|state| state.position() == seq.last_snapshot_position);
            let next = if work.is_empty() || at_retained_boundary {
                seq.next_token.ok_or(RuntimeError::SnapshotPrefix {
                    snapshot: seq.prefilled,
                    prompt: seq.prompt.len(),
                })?
            } else if last_shape == self.prefill_chunk {
                self.prefill.sample_argmax(&self.stream)?
            } else if last_shape == self.retention_interval && last_shape != 1 {
                self.retention_prefill
                    .as_mut()
                    .expect("distinct retention shape was prebuilt")
                    .sample_argmax(&self.stream)?
            } else if let Some((_, tail)) = self
                .prefill_tails
                .iter_mut()
                .find(|(tail_width, _)| *tail_width == last_shape)
            {
                tail.sample_argmax(&self.stream)?
            } else {
                self.decode[last_shape]
                    .as_mut()
                    .expect("narrow prefill width was prebuilt")
                    .sample_argmax(&self.stream)?
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
