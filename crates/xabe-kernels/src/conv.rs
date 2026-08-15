//! The Gated DeltaNet short causal depthwise convolution.
//!
//! Every one of Qwen3.6's 30 GDN layers carries an `ssm_conv1d.weight` of
//! shape `[4, 8192]` (`qwen35moe.ssm.conv_kernel = 4`) and applies it to the
//! fused q/k/v stream *before* the delta rule. This kernel was absent from
//! `docs/KERNELS.md` until the weight schema made it visible; see
//! `docs/MODEL.md`. It is not optional and it is not folded into the delta
//! rule.
//!
//! # Where these semantics come from
//!
//! Derived from llama.cpp (`/home/nixabe/llama.cpp`), not from the tensor
//! shape:
//!
//! - **The op.** `ggml_compute_forward_ssm_conv_f32` in
//!   `ggml/src/ggml-cpu/ops.cpp` (line 9557). Its inner loop is
//!   `sumf += s[i0 + i1*ncs] * c[i0 + i1*nc]` over `i0 in [0, d_conv)`, where
//!   `s` has already been advanced by `i2` (the token index) elements. So
//!   output element `(token t, channel ch)` reads window positions `t .. t+3`
//!   of a buffer whose first `d_conv - 1` entries are the carried state and
//!   whose remaining entries are this batch's tokens. That is a *causal*
//!   convolution: window position `t + (d_conv - 1)` is token `t` itself, and
//!   the earlier positions are tokens `t-1 .. t-3`. There is no bias term and
//!   no cross-channel mixing — `c` is indexed by the same `i1` as `s`, which
//!   is what makes it depthwise.
//!
//! - **The window and the state.** `llm_build_delta_net_base::build_conv_state`
//!   in `src/models/delta-net-base.cpp` (line 449) forms the input by
//!   `ggml_concat(conv_states, transpose(qkv_mixed), 0)` where `conv_states`
//!   is `[d_conv - 1, channels]`, then copies the *last* `d_conv - 1` columns
//!   of that concatenation back out as the new state (`s_idx =
//!   conv_input->ne[0] - conv_states->ne[0]`). The cache therefore holds the
//!   last `conv_kernel - 1 = 3` inputs, and a batch shorter than 3 tokens
//!   correctly keeps part of the previous state.
//!
//! - **The call site.** `src/models/qwen35moe.cpp` line 413-425: the
//!   convolution runs on `qkv_mixed` straight out of the input projection,
//!   and `ggml_silu` is applied to its output as a **separate** op
//!   (`conv_output_silu`) before q/k/v are sliced apart and L2-normalized.
//!   This module implements the convolution alone, matching `ggml_ssm_conv`'s
//!   boundary; the activation is [`crate::norm::silu`] applied by the caller.
//!   Folding it in here would make the reference disagree with the op it is
//!   supposed to be an oracle for.
//!
//! - **The state size.** `llama_hparams::n_embd_r` in `src/llama-hparams.cpp`
//!   (line 204) returns `(ssm_d_conv - 1) * (ssm_d_inner + 2*ssm_n_group*
//!   ssm_d_state)` — 3 x 8192 floats = 96 KiB per layer, which is the figure
//!   `docs/MODEL.md` quotes.
//!
//! # Layouts
//!
//! Chosen to match the GGUF tensor and the engine's activation layout rather
//! than ggml's internal transposes:
//!
//! - `weight` is `[channels][conv_kernel]`, tap index contiguous. This is the
//!   file layout verbatim: ggml reports `ne = [4, 8192]` with `ne[0]`
//!   contiguous, so tap `i` of channel `ch` is at `ch * conv_kernel + i`.
//! - `x` and the output are `[seq_len][channels]`, channel contiguous —
//!   token-major, which is what every other activation in this crate uses and
//!   what lets the device kernel coalesce across channels.
//! - `state` is `[channels][conv_kernel - 1]`, **oldest first**: `state[ch]
//!   [j]` is the input at relative position `j - (conv_kernel - 1)`, so
//!   `state[ch][conv_kernel - 2]` is the immediately preceding token.
//!
//! The tap ordering is load-bearing and easy to get backwards: tap
//! `conv_kernel - 1` multiplies the *current* token, tap `0` multiplies the
//! oldest. A reversed kernel produces a finite, plausible, anti-causal model.

/// Runs the causal depthwise convolution over `seq_len` tokens and returns
/// the output together with the updated cache.
///
/// `x` is `[seq_len][channels]`, `weight` is `[channels][conv_kernel]`, and
/// `state` is `[channels][conv_kernel - 1]` holding the `conv_kernel - 1`
/// tokens that preceded `x[0]` (all zeros at the start of a sequence).
///
/// Returns `(out, new_state)` with `out` shaped like `x` and `new_state`
/// shaped like `state`. The state is returned rather than mutated in place so
/// that a caller can compare against it without having to clone first, and so
/// that the aliasing hazard the device kernel has to handle explicitly cannot
/// exist here at all.
///
/// Output at token `t` depends only on tokens `t - (conv_kernel - 1) ..= t`;
/// anything before token 0 comes from `state`.
///
/// # Panics
/// If `conv_kernel` is zero, if any slice length disagrees with the declared
/// geometry, or if `channels` is zero.
pub fn causal_depthwise_conv1d(
    x: &[f32],
    weight: &[f32],
    state: &[f32],
    seq_len: usize,
    channels: usize,
    conv_kernel: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        conv_kernel > 0,
        "causal_depthwise_conv1d: conv_kernel must be non-zero"
    );
    assert!(
        channels > 0,
        "causal_depthwise_conv1d: channels must be non-zero"
    );
    assert_eq!(
        x.len(),
        seq_len * channels,
        "causal_depthwise_conv1d: x must be [seq_len][channels]"
    );
    assert_eq!(
        weight.len(),
        channels * conv_kernel,
        "causal_depthwise_conv1d: weight must be [channels][conv_kernel]"
    );
    assert_eq!(
        state.len(),
        channels * (conv_kernel - 1),
        "causal_depthwise_conv1d: state must be [channels][conv_kernel - 1]"
    );

    let carry = conv_kernel - 1;

    // The window value at signed position `u` for channel `ch`: this batch's
    // token when `u >= 0`, otherwise the carried state, whose last entry is
    // position -1.
    let window = |u: isize, ch: usize| -> f32 {
        if u >= 0 {
            x[u as usize * channels + ch]
        } else {
            state[ch * carry + (u + carry as isize) as usize]
        }
    };

    let mut out = vec![0.0f32; seq_len * channels];
    for t in 0..seq_len {
        for ch in 0..channels {
            // Ascending tap order, sequential fp32 accumulation — the same
            // order as `ggml_compute_forward_ssm_conv_f32`'s inner loop, so
            // that a disagreement with it is a formulation difference rather
            // than a reassociation artefact.
            let mut acc = 0.0f32;
            for i in 0..conv_kernel {
                let u = t as isize - carry as isize + i as isize;
                acc += window(u, ch) * weight[ch * conv_kernel + i];
            }
            out[t * channels + ch] = acc;
        }
    }

    // The new cache is the last `carry` window positions, which slides back
    // into the old state when the batch is shorter than the kernel.
    let mut new_state = vec![0.0f32; channels * carry];
    for ch in 0..channels {
        for j in 0..carry {
            let u = seq_len as isize - carry as isize + j as isize;
            new_state[ch * carry + j] = window(u, ch);
        }
    }

    (out, new_state)
}

/// Convolution channel count for the fused GDN q/k/v stream.
///
/// `2 * qk_heads * head_dim + value_heads * head_dim`, matching
/// `conv_dim = key_dim * 2 + value_dim` in `src/models/qwen35moe.cpp`
/// (line 71). At Qwen3.6's geometry that is `2*16*128 + 32*128 = 8192`, which
/// is the second dimension of `ssm_conv1d.weight`.
pub const fn gdn_conv_channels(qk_heads: usize, value_heads: usize, head_dim: usize) -> usize {
    2 * qk_heads * head_dim + value_heads * head_dim
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{Tolerance, assert_matches};
    use crate::rng::Xorshift64Star;

    const K: usize = 4;

    #[test]
    fn the_channel_count_matches_the_ssm_conv1d_weight_shape() {
        // ssm_conv1d.weight is [4, 8192]; 8192 is what the geometry must give.
        let g = xabe_model::ModelConfig::qwen3_6_35b_a3b().gdn;
        assert_eq!(
            gdn_conv_channels(
                g.qk_heads as usize,
                g.value_heads as usize,
                g.head_dim as usize
            ),
            8192,
        );
        assert_eq!(g.conv_kernel as usize, K);
    }

    #[test]
    fn a_unit_impulse_on_the_last_tap_is_the_identity() {
        // Tap `conv_kernel - 1` multiplies the current token. If the taps were
        // reversed this would instead delay the signal by three tokens, which
        // is the single most likely way to get this kernel wrong.
        let channels = 3;
        let seq_len = 5;
        let mut rng = Xorshift64Star::new(101);
        let x = rng.vec_f32(seq_len * channels, -1.0, 1.0);
        let mut weight = vec![0.0f32; channels * K];
        for ch in 0..channels {
            weight[ch * K + (K - 1)] = 1.0;
        }
        let state = vec![0.0f32; channels * (K - 1)];
        let (out, _) = causal_depthwise_conv1d(&x, &weight, &state, seq_len, channels, K);
        assert_eq!(out, x, "the last tap must select the current token");
    }

    #[test]
    fn a_unit_impulse_on_the_first_tap_delays_by_conv_kernel_minus_one() {
        let channels = 2;
        let seq_len = 6;
        let mut rng = Xorshift64Star::new(102);
        let x = rng.vec_f32(seq_len * channels, -1.0, 1.0);
        let mut weight = vec![0.0f32; channels * K];
        for ch in 0..channels {
            weight[ch * K] = 1.0;
        }
        let state = vec![0.0f32; channels * (K - 1)];
        let (out, _) = causal_depthwise_conv1d(&x, &weight, &state, seq_len, channels, K);

        // The first K-1 outputs read the zero state; the rest are x delayed.
        for ch in 0..channels {
            for t in 0..K - 1 {
                assert_eq!(out[t * channels + ch], 0.0);
            }
            for t in K - 1..seq_len {
                assert_eq!(out[t * channels + ch], x[(t - (K - 1)) * channels + ch]);
            }
        }
    }

    #[test]
    fn the_output_is_causal_a_later_token_cannot_change_an_earlier_output() {
        // The property the whole module exists to guarantee, checked by
        // perturbation rather than by inspection: rewrite token `t` and every
        // output before `t` must be bit-identical.
        let channels = 4;
        let seq_len = 12;
        let mut rng = Xorshift64Star::new(103);
        let weight = rng.vec_f32(channels * K, -0.5, 0.5);
        let state = rng.vec_f32(channels * (K - 1), -1.0, 1.0);
        let x = rng.vec_f32(seq_len * channels, -1.0, 1.0);
        let (base, _) = causal_depthwise_conv1d(&x, &weight, &state, seq_len, channels, K);

        const PERTURBED: usize = 7;
        let mut y = x.clone();
        for ch in 0..channels {
            y[PERTURBED * channels + ch] += 100.0;
        }
        let (perturbed, _) = causal_depthwise_conv1d(&y, &weight, &state, seq_len, channels, K);

        assert_eq!(
            &base[..PERTURBED * channels],
            &perturbed[..PERTURBED * channels],
            "a token changed an output that precedes it — the conv is not causal",
        );
        assert_ne!(
            base[PERTURBED * channels],
            perturbed[PERTURBED * channels],
            "the perturbed token did not reach its own output",
        );
    }

    #[test]
    fn the_conv_is_depthwise_channels_do_not_mix() {
        let channels = 5;
        let seq_len = 8;
        let mut rng = Xorshift64Star::new(104);
        let weight = rng.vec_f32(channels * K, -1.0, 1.0);
        let state = rng.vec_f32(channels * (K - 1), -1.0, 1.0);
        let mut x = vec![0.0f32; seq_len * channels];
        // Only channel 2 carries a signal.
        for t in 0..seq_len {
            x[t * channels + 2] = 1.0 + t as f32;
        }
        let mut zeroed_state = state.clone();
        for ch in 0..channels {
            if ch != 2 {
                for j in 0..K - 1 {
                    zeroed_state[ch * (K - 1) + j] = 0.0;
                }
            }
        }
        let (out, _) = causal_depthwise_conv1d(&x, &weight, &zeroed_state, seq_len, channels, K);
        for t in 0..seq_len {
            for ch in 0..channels {
                if ch != 2 {
                    assert_eq!(
                        out[t * channels + ch],
                        0.0,
                        "channel {ch} picked up signal from channel 2",
                    );
                }
            }
        }
    }

    #[test]
    fn streaming_one_token_at_a_time_reproduces_the_batched_result() {
        // The decode path is the batched path with seq_len = 1 and the cache
        // threaded through. If these disagree, prefill and decode are two
        // different models — exactly the failure `AGENTS.md` warns about.
        let channels = 6;
        let seq_len = 17;
        let mut rng = Xorshift64Star::new(105);
        let weight = rng.vec_f32(channels * K, -0.8, 0.8);
        let state0 = rng.vec_f32(channels * (K - 1), -1.0, 1.0);
        let x = rng.vec_f32(seq_len * channels, -2.0, 2.0);

        let (batched, batched_state) =
            causal_depthwise_conv1d(&x, &weight, &state0, seq_len, channels, K);

        let mut state = state0.clone();
        let mut streamed = Vec::with_capacity(seq_len * channels);
        for t in 0..seq_len {
            let token = &x[t * channels..(t + 1) * channels];
            let (out, next) = causal_depthwise_conv1d(token, &weight, &state, 1, channels, K);
            streamed.extend_from_slice(&out);
            state = next;
        }

        // Both accumulate the same taps in the same ascending order over the
        // same operands, so this is exact, not merely close.
        assert_eq!(streamed, batched);
        assert_eq!(state, batched_state);
    }

    #[test]
    fn the_cache_holds_the_last_conv_kernel_minus_one_inputs() {
        let channels = 3;
        let seq_len = 9;
        let mut rng = Xorshift64Star::new(106);
        let weight = rng.vec_f32(channels * K, -1.0, 1.0);
        let state = rng.vec_f32(channels * (K - 1), -1.0, 1.0);
        let x = rng.vec_f32(seq_len * channels, -1.0, 1.0);
        let (_, new_state) = causal_depthwise_conv1d(&x, &weight, &state, seq_len, channels, K);

        for ch in 0..channels {
            for j in 0..K - 1 {
                // oldest first: j = 0 is token seq_len - 3.
                let t = seq_len - (K - 1) + j;
                assert_eq!(new_state[ch * (K - 1) + j], x[t * channels + ch]);
            }
        }
    }

    #[test]
    fn a_batch_shorter_than_the_kernel_keeps_part_of_the_old_state() {
        // ggml's `s_idx = conv_input->ne[0] - conv_states->ne[0]` slides into
        // the previous state when n_t < d_conv - 1. Dropping that would reset
        // the convolution's history on any 1- or 2-token step, i.e. on every
        // decode step.
        let channels = 2;
        let mut rng = Xorshift64Star::new(107);
        let weight = rng.vec_f32(channels * K, -1.0, 1.0);
        let state = rng.vec_f32(channels * (K - 1), -1.0, 1.0);
        let x = rng.vec_f32(channels, -1.0, 1.0);

        let (_, new_state) = causal_depthwise_conv1d(&x, &weight, &state, 1, channels, K);
        for ch in 0..channels {
            // Shifted left by one: the two most recent old entries, then x.
            assert_eq!(new_state[ch * (K - 1)], state[ch * (K - 1) + 1]);
            assert_eq!(new_state[ch * (K - 1) + 1], state[ch * (K - 1) + 2]);
            assert_eq!(new_state[ch * (K - 1) + 2], x[ch]);
        }
    }

    #[test]
    fn a_hand_computed_three_tap_window_matches() {
        // One channel, K = 4, weights [1, 2, 3, 4], state [10, 20, 30],
        // x = [1, 2].
        //   out[0] = 1*10 + 2*20 + 3*30 + 4*1  = 10 + 40 + 90 + 4  = 144
        //   out[1] = 1*20 + 2*30 + 3*1  + 4*2  = 20 + 60 + 3  + 8  = 91
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let state = [10.0f32, 20.0, 30.0];
        let x = [1.0f32, 2.0];
        let (out, new_state) = causal_depthwise_conv1d(&x, &weight, &state, 2, 1, 4);
        assert_matches(&out, &[144.0, 91.0], &Tolerance::tight_fp32());
        assert_eq!(new_state, vec![30.0, 1.0, 2.0]);
    }

    #[test]
    fn a_zero_state_and_zero_input_produce_exactly_zero() {
        let channels = 4;
        let seq_len = 3;
        let mut rng = Xorshift64Star::new(108);
        let weight = rng.vec_f32(channels * K, -5.0, 5.0);
        let x = vec![0.0f32; seq_len * channels];
        let state = vec![0.0f32; channels * (K - 1)];
        let (out, new_state) = causal_depthwise_conv1d(&x, &weight, &state, seq_len, channels, K);
        assert!(out.iter().all(|&v| v == 0.0));
        assert!(new_state.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn conv_kernel_one_is_a_per_channel_scale_with_no_state() {
        let channels = 3;
        let seq_len = 4;
        let mut rng = Xorshift64Star::new(109);
        let weight = rng.vec_f32(channels, -2.0, 2.0);
        let x = rng.vec_f32(seq_len * channels, -1.0, 1.0);
        let (out, new_state) = causal_depthwise_conv1d(&x, &weight, &[], seq_len, channels, 1);
        assert!(new_state.is_empty());
        for t in 0..seq_len {
            for ch in 0..channels {
                assert_eq!(out[t * channels + ch], x[t * channels + ch] * weight[ch]);
            }
        }
    }

    #[test]
    #[should_panic(expected = "weight must be")]
    fn a_mis_shaped_weight_is_rejected_rather_than_silently_reinterpreted() {
        causal_depthwise_conv1d(&[1.0, 2.0], &[1.0, 2.0, 3.0], &[0.0; 3], 2, 1, 4);
    }
}
