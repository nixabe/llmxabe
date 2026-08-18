//! Error types for scheduler construction and admission control.

use thiserror::Error;

/// A structurally invalid [`crate::config::SchedulerConfig`].
#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum SchedulerConfigError {
    /// AGENTS.md rule 3: at `token_budget <= block_size + max_concurrent_decodes`,
    /// a single decoding request can consume the entire per-step budget on
    /// its own, and no new request can ever begin prefill — execution
    /// serializes to batch 1 (~7x throughput loss, per the LMCache Qwen3.6
    /// recipe this rule cites). This is rejected at construction, not
    /// logged and allowed to run degraded.
    #[error(
        "token_budget ({token_budget}) must be strictly greater than \
         block_size ({block_size}) + max_concurrent_decodes ({max_concurrent_decodes}); \
         at or below that, a single decoding request can starve all prefill \
         admission and execution serializes to batch 1"
    )]
    BudgetTooSmall {
        token_budget: u32,
        block_size: u32,
        max_concurrent_decodes: u32,
    },
    /// `block_size` was zero.
    #[error("block_size must be non-zero")]
    ZeroBlockSize,
    /// `watermark_fraction` was outside `[0, 1)`.
    #[error("watermark_fraction ({0}) must be in [0.0, 1.0)")]
    WatermarkOutOfRange(f64),
}

/// A request rejected at admission time.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    /// A cache restore cannot claim tokens beyond the submitted prompt.
    #[error("reusable prefix {prefix_tokens} exceeds prompt length {prompt_tokens}")]
    InvalidReusablePrefix {
        prefix_tokens: u32,
        prompt_tokens: u32,
    },
    /// AGENTS.md rule 4: chunked prefill splits *compute*, not *memory*. A
    /// request that will never fit in KV capacity across its full lifetime
    /// (prompt + max output) must be rejected up front, even though its
    /// first chunk would fit within one step's token budget — admitting it
    /// anyway over-commits capacity and causes thrashing once later chunks
    /// (or decode) can't find room.
    #[error(
        "request needs {needed_blocks} attention blocks over its full lifetime \
         ({full_seq_len} tokens), but only {total_blocks} exist in the pool"
    )]
    ExceedsTotalCapacity {
        full_seq_len: u32,
        needed_blocks: u32,
        total_blocks: u32,
    },
    /// Enough total capacity exists, but not enough is currently free after
    /// reserving the watermark headroom for future admissions.
    #[error(
        "admitting this request would leave fewer than the {watermark_blocks}-block \
         watermark free ({free_blocks} free, {needed_blocks} needed)"
    )]
    BelowWatermark {
        free_blocks: u32,
        needed_blocks: u32,
        watermark_blocks: u32,
    },
    /// The bounded waiting queue is full. The caller can retry after a later
    /// scheduler step rather than allowing host memory to grow indefinitely.
    #[error("waiting queue is full ({capacity} requests)")]
    WaitingQueueFull { capacity: u32 },
}
