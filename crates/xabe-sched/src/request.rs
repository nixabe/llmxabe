//! Request-facing types: what a caller submits and what a scheduled step
//! reports back.

/// Opaque request identifier, assigned by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

/// A request as submitted for admission.
///
/// `max_output_tokens` is required, not optional, because AGENTS.md rule 4
/// depends on it: admission checks capacity against the request's *full*
/// potential lifetime (`prompt_tokens + max_output_tokens`), not just the
/// prompt or the first chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NewRequest {
    pub id: RequestId,
    pub prompt_tokens: u32,
    pub max_output_tokens: u32,
}

impl NewRequest {
    /// Tokens this request could occupy over its entire life: prompt plus
    /// the worst-case generated output. The quantity AGENTS.md rule 4 says
    /// admission must check, in full, up front.
    pub const fn full_seq_len(&self) -> u32 {
        self.prompt_tokens + self.max_output_tokens
    }
}

/// One decoding request's contribution to a scheduled step.
///
/// `tokens` is [`crate::config::SchedulerConfig::tokens_per_decode_step`]:
/// one real token plus every MTP draft token, since draft tokens consume
/// step budget and KV capacity whether or not they're later accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeItem {
    pub id: RequestId,
    pub tokens: u32,
}

/// One prefilling request's contribution to a scheduled step: the number of
/// new prompt tokens computed this step, which may be less than the
/// request's remaining prompt if it was the chunked, budget-exhausting
/// request for this step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillItem {
    pub id: RequestId,
    pub tokens: u32,
}

/// The result of one scheduling step: a description of the batch to run,
/// deterministic from scheduler state alone. Contains no device handles, so
/// it is fully testable without a GPU.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchDescription {
    /// Decode-ready requests, batched first per AGENTS.md's scheduling
    /// policy: batch all pending decodes, then spend what's left on
    /// prefill.
    pub decodes: Vec<DecodeItem>,
    /// Prefill work this step, in schedule order. The last entry may be a
    /// partial chunk of a longer prompt if the token budget ran out.
    pub prefills: Vec<PrefillItem>,
}

impl BatchDescription {
    /// Whether the running batch contains any prefill work at all.
    pub fn has_prefill(&self) -> bool {
        !self.prefills.is_empty()
    }

    /// Total tokens scheduled this step, decode and prefill combined.
    pub fn total_tokens(&self) -> u32 {
        self.decodes.iter().map(|d| d.tokens).sum::<u32>()
            + self.prefills.iter().map(|p| p.tokens).sum::<u32>()
    }
}
