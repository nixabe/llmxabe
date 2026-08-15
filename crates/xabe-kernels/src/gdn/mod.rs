//! Gated DeltaNet reference kernels: the recurrent (decode) form and the
//! chunked parallel (prefill) form, plus the equivalence test between them.
//!
//! Start with [`recurrent`] for the derivation and upstream citations; the
//! chunked form in [`chunked`] is derived *from* the recurrent form, not
//! independently ported, and is graded entirely by whether the two agree.

pub mod chunked;
pub mod recurrent;
pub mod tri;

pub use chunked::chunked_forward;
pub use recurrent::{GdnState, recurrent_forward, recurrent_step};
