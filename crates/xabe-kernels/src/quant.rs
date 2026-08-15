//! Q8_0 and Q6_K dequantization, ported byte-for-byte from llama.cpp's
//! scalar reference implementation, plus a simple (non-optimal) quantizer
//! for each format used only to build round-trip differential tests.
//!
//! **Dequantization** is the operation that matters for correctness here:
//! this engine reads GGUF weights already quantized by someone else's
//! (optimal, search-based) quantizer, so the CPU reference only needs to
//! unpack their bit layout correctly. Those unpacking formulas are
//! transcribed directly from:
//! - Q8_0: `ggml/src/ggml-quants.c::dequantize_row_q8_0` (line 553-567) and
//!   `::quantize_row_q8_0_ref` (line 276-299) in `llama.cpp`.
//! - Q6_K: `ggml/src/ggml-quants.c::dequantize_row_q6_K` (line 1939-1968)
//!   in `llama.cpp`; block layout from `ggml/src/ggml-common.h` (search
//!   `block_q6_K`, `block_q8_0`).
//!
//! **Quantization** here is a simple, correct-by-construction encoder built
//! only so a synthetic vector can be round-tripped through
//! quantize-then-dequantize and checked against an error bound. It is
//! *not* a port of llama.cpp's actual quantizer, which searches for a
//! least-squares-optimal scale per block (`quantize_row_q6_K_impl`); ours
//! picks the scale directly from the block's max magnitude. That is fine
//! for testing the dequantization bit-unpacking (the thing this engine
//! actually depends on) but would give worse compression than the real
//! quantizer if used to produce a GGUF file.

use half::f16;

/// Superblock size shared by every "k-quant" format (`QK_K` in
/// `ggml-common.h`).
pub const QK_K: usize = 256;
/// Block size for Q8_0 (`QK8_0` in `ggml-common.h`).
pub const QK8_0: usize = 32;

/// A Q8_0 block: 32 int8 quants sharing one fp16 delta.
///
/// Layout matches `block_q8_0` in `ggml-common.h`: `{ ggml_half d; int8_t
/// qs[QK8_0]; }`.
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; QK8_0],
}

/// Dequantizes one Q8_0 block: `y[j] = qs[j] * d`.
///
/// Ported from `dequantize_row_q8_0` (`ggml-quants.c:553`).
pub fn dequantize_q8_0(block: &BlockQ8_0) -> [f32; QK8_0] {
    let d = block.d.to_f32();
    let mut y = [0.0f32; QK8_0];
    for (y_j, &q_j) in y.iter_mut().zip(block.qs.iter()) {
        *y_j = f32::from(q_j) * d;
    }
    y
}

/// Quantizes 32 values into a Q8_0 block.
///
/// Ported from `quantize_row_q8_0_ref` (`ggml-quants.c:276`): per-block
/// delta is `amax / 127`, each element is `round(x / delta)`.
pub fn quantize_q8_0(x: &[f32; QK8_0]) -> BlockQ8_0 {
    let amax = x.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let d = amax / 127.0;
    let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0i8; QK8_0];
    for (j, &v) in x.iter().enumerate() {
        qs[j] = (v * inv_d).round() as i8;
    }
    BlockQ8_0 {
        d: f16::from_f32(d),
        qs,
    }
}

/// A Q6_K superblock: 256 6-bit quants, 16 int8 per-group scales, one fp16
/// super-scale.
///
/// Layout matches `block_q6_K` in `ggml-common.h`: `{ uint8_t ql[QK_K/2];
/// uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; }` — low 4 bits
/// of each 6-bit code live in `ql`, the high 2 bits live in `qh`, packed 4
/// codes per `qh` byte.
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: f16,
}

/// Dequantizes one Q6_K superblock.
///
/// Ported from `dequantize_row_q6_K` (`ggml-quants.c:1939`). The superblock
/// is processed as two 128-element halves; within each half, 32 positions
/// `l` each address four interleaved 6-bit codes (`q1..q4`, at flat offsets
/// `l`, `l+32`, `l+64`, `l+96`), and `is = l / 16` selects which of two
/// per-16-element int8 scales applies. Each code is `(low4 | high2<<4) -
/// 32`, i.e. an unsigned 6-bit value re-centered to `[-32, 31]`.
pub fn dequantize_q6_k(block: &BlockQ6K) -> [f32; QK_K] {
    let d = block.d.to_f32();
    let mut y = [0.0f32; QK_K];

    for half in 0..2 {
        let y_off = half * 128;
        let ql = &block.ql[half * 64..half * 64 + 64];
        let qh = &block.qh[half * 32..half * 32 + 32];
        let sc = &block.scales[half * 8..half * 8 + 8];

        for l in 0..32 {
            let is = l / 16;

            let raw1 = (ql[l] & 0xF) | ((qh[l] & 3) << 4);
            let raw2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            let raw3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            let raw4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);

            let q1 = i32::from(raw1) - 32;
            let q2 = i32::from(raw2) - 32;
            let q3 = i32::from(raw3) - 32;
            let q4 = i32::from(raw4) - 32;

            y[y_off + l] = d * f32::from(sc[is]) * q1 as f32;
            y[y_off + l + 32] = d * f32::from(sc[is + 2]) * q2 as f32;
            y[y_off + l + 64] = d * f32::from(sc[is + 4]) * q3 as f32;
            y[y_off + l + 96] = d * f32::from(sc[is + 6]) * q4 as f32;
        }
    }

    y
}

/// Which of the 16 per-group scales (and which flat superblock positions)
/// a `(half, lane, is)` triple corresponds to, mirroring the indexing in
/// [`dequantize_q6_k`]. `lane` is 0..4 for the four interleaved codes
/// (`q1..q4`), `is` is 0..2. Returns `(scale_index, [16 flat positions])`.
fn q6_k_group_positions(half: usize, lane: usize, is: usize) -> (usize, [usize; 16]) {
    let scale_index = half * 8 + lane * 2 + is;
    let y_off = half * 128 + lane * 32;
    let l_start = is * 16;
    let mut positions = [0usize; 16];
    for (k, pos) in positions.iter_mut().enumerate() {
        *pos = y_off + l_start + k;
    }
    (scale_index, positions)
}

/// Quantizes 256 values into a Q6_K superblock.
///
/// Not a port of llama.cpp's search-based quantizer (see module docs): this
/// picks each group's int8 scale directly from that group's max magnitude,
/// relative to a per-superblock fp16 delta derived from the overall max
/// magnitude, then rounds each element to the nearest representable code
/// in `[-32, 31]`.
pub fn quantize_q6_k(x: &[f32; QK_K]) -> BlockQ6K {
    let global_amax = x.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    // Choose d so that a group scale of 127 (the largest representable
    // int8) together with a code of 32 (the largest representable
    // magnitude) can reach the block's largest element.
    let d = if global_amax > 0.0 {
        global_amax / (32.0 * 127.0)
    } else {
        0.0
    };

    let mut scales = [0i8; QK_K / 16];
    let mut ql = [0u8; QK_K / 2];
    let mut qh = [0u8; QK_K / 4];

    for half in 0..2 {
        for lane in 0..4 {
            for is in 0..2 {
                let (scale_index, positions) = q6_k_group_positions(half, lane, is);

                let group_amax = positions.iter().fold(0.0f32, |acc, &p| acc.max(x[p].abs()));
                let sc = if d > 0.0 {
                    (group_amax / (d * 32.0)).round().clamp(0.0, 127.0) as i8
                } else {
                    0
                };
                scales[scale_index] = sc;

                let denom = d * f32::from(sc);
                let l_start = is * 16;
                for (k, &pos) in positions.iter().enumerate() {
                    let l = l_start + k;
                    let code = if denom != 0.0 {
                        (x[pos] / denom).round().clamp(-32.0, 31.0) as i32
                    } else {
                        0
                    };
                    let raw = (code + 32) as u8; // 0..=63

                    let qh_byte_index = half * 32 + l;
                    let shift = (lane as u32) * 2;
                    qh[qh_byte_index] |= (raw >> 4) << shift;

                    // lane 0 and 1 pack into the low/high nibble of
                    // ql[l], lanes 2 and 3 into ql[l + 32], matching the
                    // dequant read pattern (ql[l]&0xF / ql[l]>>4 for
                    // lanes 0/2, ql[l+32]&0xF / ql[l+32]>>4 for lanes 1/3).
                    let (ql_index, nibble_high) = match lane {
                        0 => (l, false),
                        1 => (l + 32, false),
                        2 => (l, true),
                        3 => (l + 32, true),
                        _ => unreachable!(),
                    };
                    let low4 = raw & 0xF;
                    let ql_byte_index = half * 64 + ql_index;
                    if nibble_high {
                        ql[ql_byte_index] |= low4 << 4;
                    } else {
                        ql[ql_byte_index] |= low4;
                    }
                }
            }
        }
    }

    BlockQ6K {
        ql,
        qh,
        scales,
        d: f16::from_f32(d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::compare;
    use crate::rng::Xorshift64Star;

    #[test]
    fn q8_0_round_trip_stays_within_the_formats_quantization_step() {
        let mut rng = Xorshift64Star::new(1);
        let mut x = [0.0f32; QK8_0];
        for v in &mut x {
            *v = rng.next_f32_range(-4.0, 4.0);
        }

        let block = quantize_q8_0(&x);
        let y = dequantize_q8_0(&block);

        // Max error per element is bounded by half a quantization step:
        // delta = amax/127, so worst case error <= delta/2.
        let amax = x.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let step = amax / 127.0;
        let result = compare(&y, &x);
        assert!(
            result.max_abs_error <= step / 2.0 + 1e-5,
            "q8_0 round-trip error {} exceeds half a quantization step {}",
            result.max_abs_error,
            step / 2.0
        );
    }

    #[test]
    fn q8_0_of_zero_vector_round_trips_to_zero() {
        let x = [0.0f32; QK8_0];
        let block = quantize_q8_0(&x);
        let y = dequantize_q8_0(&block);
        assert_eq!(y, [0.0f32; QK8_0]);
    }

    #[test]
    fn q8_0_saturating_value_maps_to_the_extreme_code() {
        let mut x = [0.0f32; QK8_0];
        x[0] = 10.0;
        x[1] = -10.0;
        let block = quantize_q8_0(&x);
        assert_eq!(block.qs[0], 127);
        assert_eq!(block.qs[1], -127);
    }

    #[test]
    fn q6_k_round_trip_stays_within_the_formats_quantization_step() {
        let mut rng = Xorshift64Star::new(2);
        let mut x = [0.0f32; QK_K];
        for v in &mut x {
            *v = rng.next_f32_range(-4.0, 4.0);
        }

        let block = quantize_q6_k(&x);
        let y = dequantize_q6_k(&block);

        let result = compare(&y, &x);
        // Coarser format (6-bit codes with an int8 group scale on top of
        // an fp16 super-scale, and a non-optimal encoder): a looser but
        // still tight bound.
        assert!(
            result.max_abs_error < 0.35,
            "q6_k round-trip error too large: {result}"
        );
        assert!(
            result.cosine_similarity > 0.999,
            "q6_k round-trip: {result}"
        );
    }

    #[test]
    fn q6_k_of_zero_vector_round_trips_to_zero() {
        let x = [0.0f32; QK_K];
        let block = quantize_q6_k(&x);
        let y = dequantize_q6_k(&block);
        assert_eq!(y, [0.0f32; QK_K]);
    }

    #[test]
    fn q6_k_group_positions_partition_the_superblock_without_overlap() {
        // Every one of the 256 flat positions must be covered by exactly
        // one (half, lane, is) group of 16 — otherwise the quantizer and
        // dequantizer are not addressing the same elements.
        let mut seen = [0u32; QK_K];
        for half in 0..2 {
            for lane in 0..4 {
                for is in 0..2 {
                    let (_, positions) = q6_k_group_positions(half, lane, is);
                    for pos in positions {
                        seen[pos] += 1;
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|&count| count == 1),
            "every position must be covered exactly once"
        );
    }

    #[test]
    fn q6_k_large_magnitude_input_round_trips_without_overflow() {
        let mut rng = Xorshift64Star::new(3);
        let mut x = [0.0f32; QK_K];
        for v in &mut x {
            *v = rng.next_f32_range(-100.0, 100.0);
        }
        let block = quantize_q6_k(&x);
        let y = dequantize_q6_k(&block);
        assert!(y.iter().all(|v| v.is_finite()));
        let result = compare(&y, &x);
        assert!(
            result.cosine_similarity > 0.99,
            "q6_k large-magnitude round-trip: {result}"
        );
    }

    #[test]
    fn dequantize_q6_k_matches_a_hand_built_superblock() {
        // Construct a block where every code is 0 (raw=32, i.e. ql nibble
        // = 0, qh bits = 10) at scale 1, super-scale d=2.0. Every output
        // element should equal 0 (code 0 * anything = 0), so instead pick
        // code = 1 (raw = 33 = 0b100001, low4=1, high2=2) with scale=1 and
        // d=2.0 to get a nonzero, hand-checkable value: 2.0*1*1 = 2.0.
        let mut block = BlockQ6K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [1i8; QK_K / 16],
            d: f16::from_f32(2.0),
        };
        // raw = 33 = 0b10_0001 -> low4 = 0b0001 = 1, high2 = 0b10 = 2.
        // Place this at half=0, lane=0 (q1, offset 0), l=0: low4 goes into
        // ql[0] bits [0:4), high2 goes into qh[0] bits [0:2) (shift = 0).
        block.ql[0] = 1;
        block.qh[0] = 2;

        let y = dequantize_q6_k(&block);
        // code = raw - 32 = 33 - 32 = 1; value = d * scale * code = 2*1*1.
        assert!((y[0] - 2.0).abs() < 1e-4, "y[0] = {}", y[0]);
    }
}
