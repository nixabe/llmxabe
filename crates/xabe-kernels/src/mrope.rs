//! Multimodal RoPE position assignment for mixed text + image sequences.
//!
//! Qwen3.6's attention layers are trained with M-RoPE: every token carries a
//! three-component position `(t, h, w)` and each rotary channel reads one of
//! the three, per the GGUF's `qwen35moe.rope.dimension_sections`. For pure
//! text the three components are equal, and every section layout collapses
//! to ordinary RoPE — which is why the text path can keep its scalar
//! positions untouched. Image spans are where the components diverge, and
//! this module is the reference for how.
//!
//! The assignment rule, identical in vLLM
//! (`Qwen3VLForConditionalGeneration._get_mrope_input_positions`,
//! `vllm/model_executor/models/qwen3_vl.py`) and exllamav3
//! (`gen_mrope_pos_ids`, `exllamav3/exllamav3_ext/rope.cu`):
//!
//! - a text token at running base `P` gets `(P, P, P)` and advances the
//!   base by 1;
//! - the `k`-th token of an image whose merged grid is `gh x gw`, entering
//!   at base `B`, gets `(B, B + k / gw, B + k % gw)`;
//! - after the image the base advances to `B + max(1, gh, gw)` — the
//!   maximum component seen, plus one. An image occupies `max(gh, gw)`
//!   positions of sequence "time", not `gh * gw`.
//!
//! Decode continues from the final base with all components equal, so a
//! sequence's rope state after its last image is fully described by one
//! scalar — vLLM stores it as `mrope_position_delta`.

/// Three-component rope position for one token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MropePos {
    /// Temporal component.
    pub t: u32,
    /// Height component.
    pub h: u32,
    /// Width component.
    pub w: u32,
}

impl MropePos {
    /// The all-equal position a pure-text token gets.
    pub const fn text(p: u32) -> Self {
        Self { t: p, h: p, w: p }
    }

    /// Whether all three components agree — true for every token of a
    /// text-only sequence, and the condition under which M-RoPE equals
    /// ordinary RoPE.
    pub const fn is_scalar(self) -> bool {
        self.t == self.h && self.h == self.w
    }
}

/// One image's placement inside a token sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    /// Index of the image's first embedding token in the sequence.
    pub start: usize,
    /// Merged grid height (`grid_h / spatial_merge`).
    pub grid_h: u32,
    /// Merged grid width (`grid_w / spatial_merge`).
    pub grid_w: u32,
}

impl ImageSpan {
    /// Tokens this span occupies in the sequence.
    pub const fn len(&self) -> usize {
        (self.grid_h * self.grid_w) as usize
    }

    /// Whether the span is empty (degenerate; preprocessing never emits one).
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Assign `(t, h, w)` positions to a sequence of `seq_len` tokens containing
/// `spans` (sorted by `start`, non-overlapping).
///
/// Returns the per-token positions and the base the *next* token after the
/// sequence would get — the scalar decode continues from.
///
/// # Panics
/// If spans overlap, are unsorted, or extend past `seq_len`.
pub fn assign_positions(seq_len: usize, spans: &[ImageSpan]) -> (Vec<MropePos>, u32) {
    let mut out = Vec::with_capacity(seq_len);
    let mut base: u32 = 0;
    let mut i = 0usize;

    for span in spans {
        assert!(
            span.start >= i,
            "image spans must be sorted and non-overlapping"
        );
        assert!(
            span.start + span.len() <= seq_len,
            "image span extends past the sequence"
        );

        // Text run up to the span.
        while i < span.start {
            out.push(MropePos::text(base));
            base += 1;
            i += 1;
        }

        // The image grid, all rooted at the entry base.
        let gw = span.grid_w;
        for k in 0..span.len() as u32 {
            out.push(MropePos {
                t: base,
                h: base + k / gw,
                w: base + k % gw,
            });
        }
        i += span.len();
        // max component seen is base + max(grid_h, grid_w) - 1 (t contributes
        // base + 0); advance to one past it.
        base += span.grid_h.max(span.grid_w).max(1);
    }

    // Trailing text.
    while i < seq_len {
        out.push(MropePos::text(base));
        base += 1;
        i += 1;
    }

    (out, base)
}

/// Applies interleaved M-RoPE (llama.cpp `GGML_ROPE_TYPE_IMROPE`) to the
/// leading `rope_dim` dimensions of one head vector, leaving the tail
/// untouched — the three-component sibling of [`crate::rope::apply_rope`].
///
/// Pair `j` (NEOX pairing: dimension `j` with `j + rope_dim/2`) reads its
/// position from channel `j % 3` — `t, h, w` — bounded by `sections`:
/// `h` requires `j < 3*sections[1]`, `w` requires `j < 3*sections[2]`, and
/// everything else falls back to `t`. The frequency uses the *global* pair
/// index (`theta_base^(-2j/rope_dim)`), unlike the vision tower's
/// per-section restart. Ported from `ggml_mrope_cache_init`
/// (`ggml/src/ggml-cpu/ops.cpp`, the `GGML_ROPE_TYPE_IMROPE` branch) with
/// `sections` from GGUF `qwen35moe.rope.dimension_sections` = `[11,11,10]`
/// (the fourth section is 0 and dead).
///
/// With `pos.is_scalar()` this is exactly [`crate::rope::apply_rope`] —
/// the collapse the engine's text path relies on, asserted in the tests.
pub fn apply_imrope(
    head_vec: &[f32],
    pos: MropePos,
    rope_dim: u32,
    sections: [u32; 3],
    theta_base: f32,
) -> Vec<f32> {
    let head_dim = head_vec.len();
    let rope_dim = rope_dim as usize;
    assert_eq!(rope_dim % 2, 0, "apply_imrope: rope_dim must be even");
    assert!(rope_dim <= head_dim);
    let half = rope_dim / 2;
    assert_eq!(
        (sections[0] + sections[1] + sections[2]) as usize,
        half,
        "sections must cover every rotary pair"
    );

    let mut out = head_vec.to_vec();
    for j in 0..half {
        let ju = j as u32;
        let p = if ju % 3 == 1 && ju < 3 * sections[1] {
            pos.h
        } else if ju % 3 == 2 && ju < 3 * sections[2] {
            pos.w
        } else {
            pos.t
        };
        let freq = f64::from(theta_base).powf(-2.0 * j as f64 / rope_dim as f64);
        let angle = f64::from(p) * freq;
        let (sin_a, cos_a) = (angle.sin() as f32, angle.cos() as f32);

        let x0 = head_vec[j];
        let x1 = head_vec[j + half];
        out[j] = x0 * cos_a - x1 * sin_a;
        out[j + half] = x0 * sin_a + x1 * cos_a;
    }
    out
}

/// The `qwen35moe.rope.dimension_sections` value, minus its dead fourth
/// entry.
pub const QWEN3_6_SECTIONS: [u32; 3] = [11, 11, 10];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_positions_are_the_identity() {
        let (pos, next) = assign_positions(5, &[]);
        assert_eq!(next, 5);
        for (p, expect) in pos.iter().zip(0..) {
            assert_eq!(*p, MropePos::text(expect));
            assert!(p.is_scalar());
        }
    }

    #[test]
    fn image_grid_gets_row_and_column_offsets() {
        // 2 text tokens, then a 2x3 merged grid, then 1 text token.
        let span = ImageSpan {
            start: 2,
            grid_h: 2,
            grid_w: 3,
        };
        let (pos, next) = assign_positions(9, &[span]);

        assert_eq!(pos[0], MropePos::text(0));
        assert_eq!(pos[1], MropePos::text(1));
        // Image enters at base 2. t stays 2 for the whole grid.
        assert_eq!(pos[2], MropePos { t: 2, h: 2, w: 2 });
        assert_eq!(pos[3], MropePos { t: 2, h: 2, w: 3 });
        assert_eq!(pos[4], MropePos { t: 2, h: 2, w: 4 });
        assert_eq!(pos[5], MropePos { t: 2, h: 3, w: 2 });
        assert_eq!(pos[6], MropePos { t: 2, h: 3, w: 3 });
        assert_eq!(pos[7], MropePos { t: 2, h: 3, w: 4 });
        // Base advances by max(gh, gw) = 3, to 5.
        assert_eq!(pos[8], MropePos::text(5));
        assert_eq!(next, 6);
    }

    #[test]
    fn an_image_occupies_max_edge_positions_not_area() {
        // A 4x4 merged grid (16 tokens) advances sequence time by only 4.
        let span = ImageSpan {
            start: 0,
            grid_h: 4,
            grid_w: 4,
        };
        let (pos, next) = assign_positions(16, &[span]);
        assert_eq!(pos.len(), 16);
        assert_eq!(next, 4);
    }

    #[test]
    fn adjacent_images_chain_their_bases() {
        let spans = [
            ImageSpan {
                start: 0,
                grid_h: 2,
                grid_w: 2,
            },
            ImageSpan {
                start: 4,
                grid_h: 2,
                grid_w: 2,
            },
        ];
        let (pos, next) = assign_positions(8, &spans);
        // First image at base 0, advances to 2; second at base 2.
        assert_eq!(pos[4], MropePos { t: 2, h: 2, w: 2 });
        assert_eq!(next, 4);
    }

    #[test]
    fn decode_continuation_matches_vllms_delta_rule() {
        // vLLM stores mrope_position_delta = max + 1 - seq_len and decodes at
        // seq_len + delta. Our returned `next` must equal that.
        let span = ImageSpan {
            start: 1,
            grid_h: 6,
            grid_w: 4,
        };
        let seq_len = 1 + 24 + 3;
        let (pos, next) = assign_positions(seq_len, &[span]);
        let max_seen = pos.iter().map(|p| p.t.max(p.h).max(p.w)).max().unwrap();
        assert_eq!(next, max_seen + 1);
    }

    #[test]
    fn scalar_positions_collapse_to_plain_rope_bit_exactly() {
        // The reduction the engine's text path relies on: with t == h == w,
        // interleaved section selection changes nothing, so apply_imrope must
        // equal apply_rope bit for bit — including the untouched 64..256 tail.
        use crate::rng::Xorshift64Star;
        let mut rng = Xorshift64Star::new(7);
        let head = rng.vec_f32(256, -1.0, 1.0);
        for position in [0u32, 1, 511, 131_071] {
            let plain = crate::rope::apply_rope(&head, position, 64, 10_000_000.0);
            let multi = apply_imrope(
                &head,
                MropePos::text(position),
                64,
                QWEN3_6_SECTIONS,
                10_000_000.0,
            );
            assert_eq!(plain, multi, "divergence at position {position}");
        }
    }

    #[test]
    fn diverging_components_select_by_channel() {
        // Pair 0 is a t channel, pair 1 an h channel, pair 2 a w channel.
        // Zero out two components at a time and check the right pairs move.
        let head: Vec<f32> = (0..256).map(|i| (i as f32 * 0.13).sin()).collect();
        let base = apply_imrope(
            &head,
            MropePos { t: 0, h: 0, w: 0 },
            64,
            QWEN3_6_SECTIONS,
            10_000_000.0,
        );
        let h_only = apply_imrope(
            &head,
            MropePos { t: 0, h: 9, w: 0 },
            64,
            QWEN3_6_SECTIONS,
            10_000_000.0,
        );
        // h channels are pairs 1,4,7,...,31; all other pairs unchanged.
        for j in 0..32usize {
            let moved = base[j] != h_only[j] || base[j + 32] != h_only[j + 32];
            let is_h = j % 3 == 1;
            assert_eq!(moved, is_h, "pair {j}: moved={moved} but channel-h={is_h}");
        }
    }

    #[test]
    fn eleven_t_eleven_h_ten_w_channels() {
        // The section arithmetic in the doc comment, checked by counting.
        let (mut t, mut h, mut w) = (0, 0, 0);
        for j in 0u32..32 {
            if j % 3 == 1 && j < 3 * QWEN3_6_SECTIONS[1] {
                h += 1;
            } else if j % 3 == 2 && j < 3 * QWEN3_6_SECTIONS[2] {
                w += 1;
            } else {
                t += 1;
            }
        }
        assert_eq!((t, h, w), (11, 11, 10));
    }

    #[test]
    #[should_panic(expected = "sorted")]
    fn overlapping_spans_panic() {
        let spans = [
            ImageSpan {
                start: 0,
                grid_h: 2,
                grid_w: 2,
            },
            ImageSpan {
                start: 3,
                grid_h: 2,
                grid_w: 2,
            },
        ];
        let _ = assign_positions(16, &spans);
    }
}
