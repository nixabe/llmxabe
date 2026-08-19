//! Everything about one sequence that outlives a single forward pass.
//!
//! A forward pass is built for a fixed number of positions — see the module
//! docs on [`crate::block::attention`] for why — so prefill and decode are two
//! differently-shaped [`Forward`](crate::forward::Forward) objects. What makes
//! the second a *continuation* of the first rather than a new sequence is that
//! they share one of these.
//!
//! # What is carried, and what is not
//!
//! | Carried | Where | Size |
//! | --- | --- | --- |
//! | 30 Gated DeltaNet recurrent matrices and convolution windows | [`GdnState`] | fixed, ~4 MiB total |
//! | 10 attention key/value caches | [`KvCache`] | 40 KiB per position |
//! | the write position | [`SequenceState::position`] | — |
//!
//! Activations are *not* carried. They are scratch inside the pass and are
//! overwritten every call; nothing downstream of the residual stream survives
//! a step, which is what makes it safe for two `Forward` objects of different
//! shapes to alternate over one state.
//!
//! # The asymmetry between the two halves
//!
//! Gated DeltaNet state is **constant in sequence length**: the delta rule
//! folds each token into a `[value_heads][head_dim][head_dim]` matrix, so
//! position 100,000 costs exactly what position 1 costs. That is the entire
//! architectural point of the hybrid, and it is why this model can hold a
//! very long context in a cache that a forty-layer dense model would need
//! four times over.
//!
//! Attention state is **linear** in sequence length, and only ten of forty
//! layers have it. See [`KvCache`] for the arithmetic.
//!
//! # Why a reset does not clear the key/value cache
//!
//! [`SequenceState::reset`] zeroes the Gated DeltaNet states and rewinds the
//! position, but leaves the key/value caches alone. That is not an oversight
//! and not a shortcut: the attention kernel reads keys `[0, position]` only,
//! so every slot at or above a rewound position is unreachable until it has
//! been overwritten by the pass that advances back past it. Zeroing 1.25 GiB
//! to hide data that cannot be read would make a reset cost more than the
//! prefill that follows it.
//!
//! The Gated DeltaNet halves are different, and *must* be cleared: they are
//! read unconditionally on the very first token, with no position to bound
//! them. Leaving them would resume the previous prompt's recurrent matrix and
//! its last three convolution taps, which is finite, plausible, and a
//! different sequence.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, PinnedHostSlice};

use xabe_model::config::ModelConfig;

use crate::block::attention::{AttentionBlockError, HostKvPrefix, KvCache};
use crate::block::gdn::{GdnBlock, GdnBlockError, GdnState};

/// Something went wrong allocating or resetting sequence state.
#[derive(Debug)]
pub enum StateError {
    /// A Gated DeltaNet state could not be allocated or cleared.
    Gdn(GdnBlockError),
    /// A key/value cache could not be allocated.
    Attention(AttentionBlockError),
    /// The driver rejected a clear.
    Driver(cudarc::driver::DriverError),
    /// A host snapshot belongs to a different model geometry or is too long.
    SnapshotShape,
    /// Every preallocated host snapshot slot is still retained.
    SnapshotArenaExhausted,
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gdn(e) => write!(f, "Gated DeltaNet state: {e}"),
            Self::Attention(e) => write!(f, "key/value cache: {e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::SnapshotShape => write!(f, "prefix snapshot does not match sequence geometry"),
            Self::SnapshotArenaExhausted => write!(f, "pinned snapshot arena is exhausted"),
        }
    }
}

impl std::error::Error for StateError {}

impl From<GdnBlockError> for StateError {
    fn from(e: GdnBlockError) -> Self {
        Self::Gdn(e)
    }
}

impl From<AttentionBlockError> for StateError {
    fn from(e: AttentionBlockError) -> Self {
        Self::Attention(e)
    }
}

impl From<cudarc::driver::DriverError> for StateError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Driver(e)
    }
}

/// One sequence's carried state: 30 recurrent states, 10 key/value caches, and
/// the position they are all filled to.
pub struct SequenceState {
    gdn: Vec<GdnState>,
    kv: Vec<KvCache>,
    position: usize,
    /// `position`, on the device, as one `i32`.
    ///
    /// The rotary embedding, the causal bound and the cache append all need
    /// the position, and all three used to take it as a host argument. That
    /// is exactly what stops a decode step from being captured once as a CUDA
    /// graph and replayed: a recorded launch keeps the argument it was
    /// recorded with, so the replay would rotate by, and append at, the
    /// position of the step that was captured — forever. Holding it here and
    /// pushing four bytes before each step makes every step the same launch
    /// sequence. It is also `AGENTS.md` rule 5.
    ///
    /// The host copy stays because the bounds checks are still the host's job
    /// (`position + tokens <= max_seq`); [`Self::publish_position`] is what
    /// keeps the two in step, and it is called on the forward path rather
    /// than by [`Self::advance`] so that a failed pass cannot leave the device
    /// claiming a position no cache was written for.
    d_position: CudaSlice<i32>,
    max_seq: usize,
}

struct HostGdnState {
    conv: PinnedHostSlice<f32>,
    recurrent: PinnedHostSlice<f32>,
}

struct SnapshotBuffers {
    attention: Vec<HostKvPrefix>,
    gdn: Vec<HostGdnState>,
}

struct LeasePool<T> {
    free: Mutex<Vec<T>>,
    capacity: usize,
}

impl<T> LeasePool<T> {
    fn new(free: Vec<T>) -> Arc<Self> {
        let capacity = free.len();
        Arc::new(Self {
            free: Mutex::new(free),
            capacity,
        })
    }

    fn checkout(self: &Arc<Self>) -> Option<Lease<T>> {
        self.free
            .lock()
            .expect("snapshot arena poisoned")
            .pop()
            .map(|value| Lease {
                value: Some(value),
                pool: Arc::clone(self),
            })
    }
}

struct Lease<T> {
    value: Option<T>,
    pool: Arc<LeasePool<T>>,
}

impl<T> Drop for Lease<T> {
    fn drop(&mut self) {
        self.pool
            .free
            .lock()
            .expect("snapshot arena poisoned")
            .push(self.value.take().expect("live lease owns its value"));
    }
}

impl<T> std::ops::Deref for Lease<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().expect("live lease owns its value")
    }
}

impl<T> std::ops::DerefMut for Lease<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().expect("live lease owns its value")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SnapshotLayout {
    attention_layers: usize,
    kv_dim: usize,
    attention_elements_per_half: usize,
    gdn_layers: usize,
    conv_elements: usize,
    recurrent_elements: usize,
}

impl SnapshotLayout {
    fn new(config: &ModelConfig, retention_interval: usize) -> Self {
        let attention_layers = (0..config.num_layers)
            .filter(|&layer| {
                config.layer_kind(layer) == xabe_model::config::LayerKind::GatedAttention
            })
            .count();
        let gdn_layers = config.num_layers as usize - attention_layers;
        let kv_dim = config.attention.kv_heads as usize * config.attention.head_dim as usize;
        let conv_dim = (2 * config.gdn.qk_heads as usize + config.gdn.value_heads as usize)
            * config.gdn.head_dim as usize;
        Self {
            attention_layers,
            kv_dim,
            attention_elements_per_half: retention_interval * kv_dim,
            gdn_layers,
            conv_elements: conv_dim * (config.gdn.conv_kernel as usize - 1),
            recurrent_elements: config.gdn.value_heads as usize
                * config.gdn.head_dim as usize
                * config.gdn.head_dim as usize,
        }
    }

    fn attention_bytes(self) -> usize {
        self.attention_layers * 2 * self.attention_elements_per_half * size_of::<u16>()
    }

    fn gdn_bytes(self) -> usize {
        self.gdn_layers * (self.conv_elements + self.recurrent_elements) * size_of::<f32>()
    }
}

/// Fixed-capacity page-locked storage for retained sequence snapshots.
pub(crate) struct SnapshotArena {
    slots: Arc<LeasePool<SnapshotBuffers>>,
    layout: SnapshotLayout,
}

impl SnapshotArena {
    pub(crate) fn new(
        ctx: &Arc<CudaContext>,
        config: &ModelConfig,
        retention_interval: usize,
        slots: usize,
    ) -> Result<Self, StateError> {
        let layout = SnapshotLayout::new(config, retention_interval);
        let mut buffers = Vec::with_capacity(slots);
        for _ in 0..slots {
            let mut attention = Vec::with_capacity(layout.attention_layers);
            for _ in 0..layout.attention_layers {
                // SAFETY: capture initializes the used prefix before publication.
                let k = unsafe { ctx.alloc_pinned(layout.attention_elements_per_half)? };
                let v = unsafe { ctx.alloc_pinned(layout.attention_elements_per_half)? };
                attention.push(HostKvPrefix { k, v });
            }
            let mut gdn = Vec::with_capacity(layout.gdn_layers);
            for _ in 0..layout.gdn_layers {
                // SAFETY: capture initializes both buffers before publication.
                let conv = unsafe { ctx.alloc_pinned(layout.conv_elements)? };
                let recurrent = unsafe { ctx.alloc_pinned(layout.recurrent_elements)? };
                gdn.push(HostGdnState { conv, recurrent });
            }
            buffers.push(SnapshotBuffers { attention, gdn });
        }
        Ok(Self {
            slots: LeasePool::new(buffers),
            layout,
        })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.capacity
    }

    pub(crate) fn bytes_per_slot(&self) -> usize {
        self.layout.attention_bytes() + self.layout.gdn_bytes()
    }
}

/// A complete resumable prefix, stored in page-locked host memory.
///
/// Attention and GDN retain their natural geometries: attention contains only
/// `position` prefix positions, while each GDN layer contains its fixed-size
/// convolution and recurrent state. No group is padded to the other's size.
pub struct SequenceSnapshot {
    position: usize,
    start: usize,
    kv_dim: usize,
    buffers: Lease<SnapshotBuffers>,
    next_token: Option<i32>,
    parent: Option<Arc<SequenceSnapshot>>,
}

impl SequenceSnapshot {
    pub fn position(&self) -> usize {
        self.position
    }

    /// Target-model prediction produced after computing this exact prefix.
    pub fn next_token(&self) -> Option<i32> {
        self.next_token
    }

    pub(crate) fn set_next_token(&mut self, token: i32) {
        self.next_token = Some(token);
    }

    pub fn attention_bytes(&self) -> usize {
        let own = self.buffers.attention.len()
            * 2
            * (self.position - self.start)
            * self.kv_dim
            * size_of::<u16>();
        own + self
            .parent
            .as_ref()
            .map_or(0, |parent| parent.attention_bytes())
    }

    pub fn gdn_bytes(&self) -> usize {
        self.buffers
            .gdn
            .iter()
            .map(|state| state.conv.num_bytes() + state.recurrent.num_bytes())
            .sum()
    }
}

impl SequenceState {
    /// Allocate state for one sequence of up to `max_seq` positions.
    ///
    /// `gdn_layers` and `attention_layers` are counted from `config` rather
    /// than passed, so a state can never be built with the wrong number of
    /// either for the model it will be run against.
    pub fn new(
        stream: &Arc<CudaStream>,
        gdn: &GdnBlock,
        config: &ModelConfig,
        max_seq: usize,
    ) -> Result<Self, StateError> {
        let (mut gdn_states, mut kv) = (Vec::new(), Vec::new());
        for layer in 0..config.num_layers {
            match config.layer_kind(layer) {
                xabe_model::config::LayerKind::GatedDeltaNet => {
                    gdn_states.push(gdn.state(stream)?);
                }
                xabe_model::config::LayerKind::GatedAttention => {
                    kv.push(KvCache::new(stream, config, max_seq)?);
                }
            }
        }
        Ok(Self {
            gdn: gdn_states,
            kv,
            position: 0,
            d_position: stream.alloc_zeros::<i32>(1)?,
            max_seq,
        })
    }

    /// How many positions have been written.
    ///
    /// This is where the next pass appends, and it is the `pos_offset` the
    /// rotary embedding rotates by — so a state at position 0 and a state at
    /// position 5,000 produce genuinely different attention for the same
    /// token, which is the point.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The longest sequence this state was allocated for.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Device bytes held by everything in here.
    pub fn bytes(&self) -> u64 {
        let kv: u64 = self.kv.iter().map(KvCache::bytes).sum();
        let gdn: u64 = self
            .gdn
            .iter()
            .map(|s| ((s.conv.len() + s.recurrent.len()) * size_of::<f32>()) as u64)
            .sum();
        kv + gdn
    }

    /// Return to a cold start: zero the recurrent state and rewind to 0.
    ///
    /// See the module docs on why the key/value caches are left as they are.
    /// The zeroing is a `cuMemsetD8Async` per state, not a reallocation.
    pub fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), StateError> {
        for state in &mut self.gdn {
            stream.memset_zeros(&mut state.conv)?;
            stream.memset_zeros(&mut state.recurrent)?;
        }
        self.position = 0;
        stream.memset_zeros(&mut self.d_position)?;
        Ok(())
    }

    /// Copy the filled prefix and recurrent state into page-locked host RAM.
    pub(crate) fn snapshot(
        &self,
        stream: &Arc<CudaStream>,
        arena: &SnapshotArena,
        parent: Option<Arc<SequenceSnapshot>>,
    ) -> Result<SequenceSnapshot, StateError> {
        let start = parent.as_ref().map_or(0, |snapshot| snapshot.position);
        if start > self.position {
            return Err(StateError::SnapshotShape);
        }
        if self.position - start
            > arena.layout.attention_elements_per_half
                / (self
                    .kv
                    .first()
                    .map_or(1, |cache| cache.keys().len() / cache.max_seq()))
        {
            return Err(StateError::SnapshotShape);
        }
        let mut buffers = arena
            .slots
            .checkout()
            .ok_or(StateError::SnapshotArenaExhausted)?;
        if buffers.attention.len() != self.kv.len() || buffers.gdn.len() != self.gdn.len() {
            return Err(StateError::SnapshotShape);
        }
        for (cache, prefix) in self.kv.iter().zip(&mut buffers.attention) {
            cache.snapshot_range_into(stream, start, self.position, prefix)?;
        }
        for (state, host) in self.gdn.iter().zip(&mut buffers.gdn) {
            stream.memcpy_dtoh(&state.conv, &mut host.conv)?;
            stream.memcpy_dtoh(&state.recurrent, &mut host.recurrent)?;
        }
        stream.synchronize()?;
        Ok(SequenceSnapshot {
            position: self.position,
            start,
            kv_dim: arena.layout.kv_dim,
            buffers,
            next_token: None,
            parent,
        })
    }

    /// Restore a snapshot created on this or another CUDA context.
    pub fn restore(
        &mut self,
        stream: &Arc<CudaStream>,
        snapshot: &SequenceSnapshot,
    ) -> Result<(), StateError> {
        if snapshot.position > self.max_seq
            || snapshot.buffers.attention.len() != self.kv.len()
            || snapshot.buffers.gdn.len() != self.gdn.len()
        {
            return Err(StateError::SnapshotShape);
        }
        self.restore_attention_chain(stream, snapshot)?;
        for (state, host) in self.gdn.iter_mut().zip(&snapshot.buffers.gdn) {
            if state.conv.len() != host.conv.len() || state.recurrent.len() != host.recurrent.len()
            {
                return Err(StateError::SnapshotShape);
            }
            stream.memcpy_htod(&host.conv, &mut state.conv)?;
            stream.memcpy_htod(&host.recurrent, &mut state.recurrent)?;
        }
        self.position = snapshot.position;
        stream.memcpy_htod(&[self.position as i32], &mut self.d_position)?;
        stream.synchronize()?;
        Ok(())
    }

    fn restore_attention_chain(
        &mut self,
        stream: &Arc<CudaStream>,
        snapshot: &SequenceSnapshot,
    ) -> Result<(), StateError> {
        if let Some(parent) = snapshot.parent.as_deref() {
            self.restore_attention_chain(stream, parent)?;
        }
        for (cache, prefix) in self.kv.iter_mut().zip(&snapshot.buffers.attention) {
            cache.restore_prefix(stream, prefix, snapshot.start, snapshot.position)?;
        }
        Ok(())
    }

    /// Number of Gated DeltaNet layers this state carries.
    pub fn gdn_layers(&self) -> usize {
        self.gdn.len()
    }

    /// Number of attention layers this state carries.
    pub fn attention_layers(&self) -> usize {
        self.kv.len()
    }

    /// The `slot`-th Gated DeltaNet state, counting only GDN layers.
    pub(crate) fn gdn_mut(&mut self, slot: usize) -> &mut GdnState {
        &mut self.gdn[slot]
    }

    /// The `slot`-th key/value cache and the device position together.
    ///
    /// Two accessors would be two borrows of `self`, one of them mutable, in
    /// one expression. They are disjoint fields, so one call that splits them
    /// is the borrow checker's answer rather than a workaround.
    pub(crate) fn kv_and_position_mut(&mut self, slot: usize) -> (&mut KvCache, &CudaSlice<i32>) {
        (&mut self.kv[slot], &self.d_position)
    }

    /// Copy the host position to the device.
    ///
    /// Four bytes, once per pass, before anything reads it.
    pub(crate) fn publish_position(&mut self, stream: &Arc<CudaStream>) -> Result<(), StateError> {
        stream.memcpy_htod(&[self.position as i32], &mut self.d_position)?;
        Ok(())
    }

    /// Record that `tokens` more positions have been written.
    ///
    /// Called by the forward pass after every block has appended, never by a
    /// caller: advancing early would make a mid-pass failure leave the state
    /// claiming positions that no cache actually holds.
    pub(crate) fn advance(&mut self, tokens: usize) {
        self.position += tokens;
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;

    #[test]
    fn default_retention_slot_keeps_natural_groups_separate() {
        let layout = SnapshotLayout::new(&ModelConfig::qwen3_6_35b_a3b(), 2048);
        assert_eq!(layout.attention_bytes(), 40 * 1024 * 1024);
        assert_eq!(layout.gdn_bytes(), 65_863_680);
        assert_eq!(layout.attention_bytes() + layout.gdn_bytes(), 107_806_720);
    }

    #[test]
    fn a_lease_returns_its_slot_only_after_drop() {
        let pool = LeasePool::new(vec![7u32]);
        let lease = pool.checkout().expect("one slot available");
        assert!(pool.checkout().is_none());
        assert_eq!(*lease, 7);
        drop(lease);
        assert_eq!(*pool.checkout().expect("slot returned"), 7);
    }
}
