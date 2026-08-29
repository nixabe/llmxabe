//! Block-quantized dequantization, ported byte-for-byte from llama.cpp's
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
//! - Q4_0: `ggml/src/ggml-quants.c::dequantize_row_q4_0`.
//! - Q4_K: `ggml/src/ggml-quants.c::dequantize_row_q4_K`, and the packed
//!   sub-scale reader `get_scale_min_k4` next to it.
//! - Q5_K: `ggml/src/ggml-quants.c::dequantize_row_q5_K`, same sub-scale
//!   reader plus a separate high-bit plane.
//!
//! ## Why these formats are here when no shipped file uses them
//!
//! `docs/MODEL.md` records that the two target files use exactly four types
//! between them — F32, Q8_0, Q6_K, BF16. Q4_K and Q5_K are what the ordinary
//! community quant of either architecture is built from (`Q4_K_M` is a
//! *mixture* of Q4_K, Q5_K and Q6_K, not uniform Q4_K), and Q4_0 is the
//! legacy 32-element form. A device kernel for any of them is only as
//! trustworthy as the scalar reference it is checked against, so the
//! reference lands first and separately.
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

/// Block size for Q4_0 (`QK4_0` in `ggml-common.h`).
pub const QK4_0: usize = 32;

/// A Q4_0 block: 32 4-bit quants sharing one fp16 delta, two per byte.
///
/// Layout matches `block_q4_0` in `ggml-common.h`: `{ ggml_half d; uint8_t
/// qs[QK4_0/2]; }`.
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: f16,
    pub qs: [u8; QK4_0 / 2],
}

/// Dequantizes one Q4_0 block: `y[j] = (q - 8) * d`.
///
/// Ported from `dequantize_row_q4_0`. **The two nibbles of a byte are not
/// adjacent outputs**: byte `j` carries element `j` in its low nibble and
/// element `j + 16` in its high nibble. Reading them as consecutive would
/// interleave every row and still produce plausible-looking magnitudes,
/// which is precisely the class of bug a differential test exists to catch.
pub fn dequantize_q4_0(block: &BlockQ4_0) -> [f32; QK4_0] {
    let d = block.d.to_f32();
    let mut y = [0.0f32; QK4_0];
    for (j, &b) in block.qs.iter().enumerate() {
        // Operand order `q * d`, as in Q8_0; `d * q` rounds differently.
        y[j] = ((b & 0x0F) as i32 - 8) as f32 * d;
        y[j + QK4_0 / 2] = ((b >> 4) as i32 - 8) as f32 * d;
    }
    y
}

/// Quantizes 32 values into a Q4_0 block.
///
/// Not a port of llama.cpp's `quantize_row_q4_0_ref`, which derives the
/// delta from the most negative element; this picks it from the max
/// magnitude against the 4-bit range. See the module docs on why the
/// quantizer only has to be correct, not optimal.
pub fn quantize_q4_0(x: &[f32; QK4_0]) -> BlockQ4_0 {
    let amax = x.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    // Codes are `raw - 8` for raw in 0..=15, i.e. -8..=7. Scaling by the
    // larger end (8) would let a maximal positive element round to 8 and
    // wrap; 7 is the honest bound for the positive side.
    let d = if amax > 0.0 { amax / 7.0 } else { 0.0 };

    let mut qs = [0u8; QK4_0 / 2];
    for j in 0..QK4_0 / 2 {
        let code = |v: f32| -> u8 {
            let q = if d != 0.0 { (v / d).round() } else { 0.0 };
            (q.clamp(-8.0, 7.0) as i32 + 8) as u8
        };
        qs[j] = code(x[j]) | (code(x[j + QK4_0 / 2]) << 4);
    }
    BlockQ4_0 {
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

/// Bytes of packed 6-bit sub-scales and sub-mins in a Q4_K / Q5_K superblock
/// (`K_SCALE_SIZE` in `ggml-common.h`).
pub const K_SCALE_SIZE: usize = 12;

/// Unpack the `j`th 6-bit scale/min pair from a Q4_K / Q5_K `scales` array.
///
/// A byte-for-byte port of `get_scale_min_k4` in `ggml-quants.c`. Eight
/// pairs are packed into twelve bytes, and the packing is *not* uniform: the
/// first four pairs are the low 6 bits of `q[j]` and `q[j+4]`, while the last
/// four are assembled from a low nibble in `q[j+4]` and a high bit-pair
/// borrowed from `q[j-4]` or `q[j]`. Getting the second branch wrong yields
/// scales that are plausible but too small by a factor of up to 4, which
/// shows up as a quietly duller model rather than as garbage — the exact
/// failure mode `AGENTS.md` calls the highest-likelihood risk here.
fn get_scale_min_k4(j: usize, q: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Pack eight 6-bit scale/min pairs into the twelve bytes
/// [`get_scale_min_k4`] reads. The exact inverse, for the quantizers below.
fn set_scale_min_k4(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; K_SCALE_SIZE] {
    let mut q = [0u8; K_SCALE_SIZE];
    for j in 0..4 {
        q[j] = scales[j] & 63;
        q[j + 4] = mins[j] & 63;
    }
    for j in 4..8 {
        q[j + 4] = (scales[j] & 0x0F) | ((mins[j] & 0x0F) << 4);
        q[j - 4] |= (scales[j] >> 4) << 6;
        q[j] |= (mins[j] >> 4) << 6;
    }
    q
}

/// A Q4_K superblock: 256 4-bit quants in eight 32-element groups, each with
/// its own 6-bit scale and 6-bit min, over two fp16 super-scales.
///
/// Layout matches `block_q4_K` in `ggml-common.h`: `{ ggml_half d; ggml_half
/// dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; }`. Unlike Q6_K
/// this format is *affine* — it carries a min as well as a scale, so a value
/// is `d*sc*q - dmin*m` and not a pure product.
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; QK_K / 2],
}

/// Dequantizes one Q4_K superblock.
///
/// Ported from `dequantize_row_q4_K`. The superblock is four 64-element
/// passes; each pass takes 32 bytes of `qs` and emits the low nibbles of all
/// 32 (under sub-scale `2g`) followed by the high nibbles of all 32 (under
/// sub-scale `2g+1`). So byte `l` of a pass carries elements `64g + l` and
/// `64g + 32 + l` — again not adjacent outputs.
pub fn dequantize_q4_k(block: &BlockQ4K) -> [f32; QK_K] {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();
    let mut y = [0.0f32; QK_K];

    for g in 0..4 {
        let q = &block.qs[g * 32..g * 32 + 32];
        for sub in 0..2 {
            let (sc, m) = get_scale_min_k4(2 * g + sub, &block.scales);
            // `d1` and `m1` are formed once per group in the reference, and
            // the value is `d1 * q - m1`. Kept in that shape, operand for
            // operand, so a device kernel can be bit-identical to it.
            let d1 = d * f32::from(sc);
            let m1 = dmin * f32::from(m);
            for l in 0..32 {
                let code = if sub == 0 { q[l] & 0x0F } else { q[l] >> 4 };
                y[g * 64 + sub * 32 + l] = d1 * f32::from(code) - m1;
            }
        }
    }
    y
}

/// A Q5_K superblock: as Q4_K plus one extra high bit per element, in a
/// separate 32-byte plane.
///
/// Layout matches `block_q5_K` in `ggml-common.h`: `{ ggml_half d; ggml_half
/// dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t
/// qs[QK_K/2]; }`. Note `qh` precedes `qs`, which is the opposite of the
/// order the dequant loop reads them in.
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}

/// Dequantizes one Q5_K superblock.
///
/// Ported from `dequantize_row_q5_K`. Identical to Q4_K except that each
/// code gains a fifth bit from `qh`, giving a 0..=31 range instead of
/// 0..=15. **`qh` is not advanced between passes**: all four passes index the
/// same 32 bytes by `l`, and it is the *bit position* that moves, `2g + sub`.
/// The reference spells this as two masks `u1`/`u2` shifted left by two each
/// pass, which is the same thing said less directly.
pub fn dequantize_q5_k(block: &BlockQ5K) -> [f32; QK_K] {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();
    let mut y = [0.0f32; QK_K];

    for g in 0..4 {
        let q = &block.qs[g * 32..g * 32 + 32];
        for sub in 0..2 {
            let (sc, m) = get_scale_min_k4(2 * g + sub, &block.scales);
            let d1 = d * f32::from(sc);
            let m1 = dmin * f32::from(m);
            let bit = 2 * g + sub;
            for l in 0..32 {
                let low = if sub == 0 { q[l] & 0x0F } else { q[l] >> 4 };
                let high = (block.qh[l] >> bit) & 1;
                let code = low + (high << 4);
                y[g * 64 + sub * 32 + l] = d1 * f32::from(code) - m1;
            }
        }
    }
    y
}

/// Quantizes 256 values into a Q4_K superblock.
///
/// Not a port of llama.cpp's search-based `quantize_row_q4_K_impl`: this
/// fits each 32-element group with the affine map that spans its own
/// `[min, max]`, then quantizes the two per-group parameters to the 6 bits
/// the format gives them. Correct rather than optimal — see the module docs.
pub fn quantize_q4_k(x: &[f32; QK_K]) -> BlockQ4K {
    quantize_k_affine::<15>(x).into_q4_k()
}

/// Quantizes 256 values into a Q5_K superblock. As [`quantize_q4_k`], with
/// codes in `0..=31`.
pub fn quantize_q5_k(x: &[f32; QK_K]) -> BlockQ5K {
    quantize_k_affine::<31>(x).into_q5_k()
}

/// The affine k-quant fit shared by Q4_K and Q5_K.
///
/// Both formats differ only in the code range and in where the extra bit is
/// stored, so the search runs once and the caller decides how to pack it.
struct KAffine {
    d: f32,
    dmin: f32,
    scales: [u8; 8],
    mins: [u8; 8],
    codes: [u8; QK_K],
}

fn quantize_k_affine<const MAX_CODE: u8>(x: &[f32; QK_K]) -> KAffine {
    // Per-group scale and min, in float, before either is quantized.
    let mut gs = [0.0f32; 8];
    let mut gm = [0.0f32; 8];
    for g in 0..8 {
        let group = &x[g * 32..g * 32 + 32];
        let lo = group.iter().fold(f32::INFINITY, |a, &v| a.min(v));
        let hi = group.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        // The format stores `-min` as an unsigned magnitude, so a group whose
        // values are all positive gets min 0 rather than a negative one.
        let lo = lo.min(0.0);
        gs[g] = (hi - lo) / f32::from(MAX_CODE);
        gm[g] = -lo;
    }

    // The two super-scales quantize the per-group parameters to 6 bits.
    let smax = gs.iter().fold(0.0f32, |a, &v| a.max(v));
    let mmax = gm.iter().fold(0.0f32, |a, &v| a.max(v));
    let d = smax / 63.0;
    let dmin = mmax / 63.0;

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for g in 0..8 {
        scales[g] = if d > 0.0 {
            (gs[g] / d).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
        mins[g] = if dmin > 0.0 {
            (gm[g] / dmin).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
    }

    // Encode against the *quantized* parameters, not the float ones, so the
    // round trip sees the error the format actually has.
    let mut codes = [0u8; QK_K];
    for g in 0..8 {
        let d1 = d * f32::from(scales[g]);
        let m1 = dmin * f32::from(mins[g]);
        for l in 0..32 {
            let i = g * 32 + l;
            let c = if d1 > 0.0 { (x[i] + m1) / d1 } else { 0.0 };
            codes[i] = c.round().clamp(0.0, f32::from(MAX_CODE)) as u8;
        }
    }

    KAffine {
        d,
        dmin,
        scales,
        mins,
        codes,
    }
}

impl KAffine {
    /// Scatter the flat codes into the format's group-of-64 nibble layout.
    ///
    /// Element `64g + sub*32 + l` lives in the `sub` nibble of `qs[32g + l]`,
    /// which is the read pattern [`dequantize_q4_k`] documents, written
    /// backwards.
    fn pack_nibbles(&self) -> [u8; QK_K / 2] {
        let mut qs = [0u8; QK_K / 2];
        for g in 0..4 {
            for l in 0..32 {
                let lo = self.codes[g * 64 + l] & 0x0F;
                let hi = self.codes[g * 64 + 32 + l] & 0x0F;
                qs[g * 32 + l] = lo | (hi << 4);
            }
        }
        qs
    }

    fn into_q4_k(self) -> BlockQ4K {
        BlockQ4K {
            d: f16::from_f32(self.d),
            dmin: f16::from_f32(self.dmin),
            scales: set_scale_min_k4(&self.scales, &self.mins),
            qs: self.pack_nibbles(),
        }
    }

    fn into_q5_k(self) -> BlockQ5K {
        // The fifth bit of element `64g + sub*32 + l` is bit `2g + sub` of
        // `qh[l]` — one plane reused by all four passes, as the dequant docs
        // spell out.
        let mut qh = [0u8; QK_K / 8];
        for (l, slot) in qh.iter_mut().enumerate() {
            for g in 0..4 {
                for sub in 0..2 {
                    let code = self.codes[g * 64 + sub * 32 + l];
                    *slot |= ((code >> 4) & 1) << (2 * g + sub);
                }
            }
        }
        BlockQ5K {
            d: f16::from_f32(self.d),
            dmin: f16::from_f32(self.dmin),
            scales: set_scale_min_k4(&self.scales, &self.mins),
            qh,
            qs: self.pack_nibbles(),
        }
    }
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

// ---------------------------------------------------------------------
// Decoding whole rows from GGUF bytes
// ---------------------------------------------------------------------
//
// The block structs above mirror `ggml-common.h` field for field, but Rust
// gives no layout guarantee for them, so the bytes are parsed explicitly
// rather than transmuted. That is the correct call regardless of speed:
// these are reference paths, and a `#[repr(C)]` transmute would silently
// depend on padding rules that differ from C's for `[i8; 32]` followed by
// nothing.

/// Serialized size of one Q8_0 block: `ggml_half` + 32 × `int8_t`.
pub const BLOCK_Q8_0_BYTES: usize = 2 + QK8_0;
/// Serialized size of one Q6_K superblock: `ql` + `qh` + `scales` +
/// `ggml_half`.
pub const BLOCK_Q6_K_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
/// Serialized size of one Q4_0 block: `ggml_half` + 16 packed nibble pairs.
pub const BLOCK_Q4_0_BYTES: usize = 2 + QK4_0 / 2;
/// Serialized size of one Q4_K superblock: two `ggml_half` + `scales` + `qs`.
pub const BLOCK_Q4_K_BYTES: usize = 4 + K_SCALE_SIZE + QK_K / 2;
/// Serialized size of one Q5_K superblock: as Q4_K plus the `qh` bit plane.
pub const BLOCK_Q5_K_BYTES: usize = 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;

/// A block's bytes were the wrong length for its format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSizeError {
    /// Format name, for the message.
    pub format: &'static str,
    /// Bytes the format requires.
    pub expected: usize,
    /// Bytes supplied.
    pub found: usize,
}

impl core::fmt::Display for BlockSizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} needs {} bytes per block, got {}",
            self.format, self.expected, self.found
        )
    }
}

impl std::error::Error for BlockSizeError {}

impl BlockQ8_0 {
    /// Parse one block from its on-disk representation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BlockSizeError> {
        if bytes.len() != BLOCK_Q8_0_BYTES {
            return Err(BlockSizeError {
                format: "q8_0",
                expected: BLOCK_Q8_0_BYTES,
                found: bytes.len(),
            });
        }
        let d = f16::from_le_bytes([bytes[0], bytes[1]]);
        let mut qs = [0i8; QK8_0];
        for (q, &b) in qs.iter_mut().zip(&bytes[2..]) {
            *q = b as i8;
        }
        Ok(Self { d, qs })
    }

    /// Write this block in its on-disk representation.
    ///
    /// The exact inverse of [`Self::from_bytes`], and the reason it exists is
    /// the differential harness: a device kernel for a format no shipped file
    /// stores has nothing to read unless a test can *build* a tensor in that
    /// format, and building one by hand in each test is how two tests end up
    /// disagreeing about a bit layout.
    pub fn to_bytes(&self) -> [u8; BLOCK_Q8_0_BYTES] {
        let mut out = [0u8; BLOCK_Q8_0_BYTES];
        out[..2].copy_from_slice(&self.d.to_le_bytes());
        for (b, &q) in out[2..].iter_mut().zip(&self.qs) {
            *b = q as u8;
        }
        out
    }
}

impl BlockQ6K {
    /// Parse one superblock from its on-disk representation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BlockSizeError> {
        if bytes.len() != BLOCK_Q6_K_BYTES {
            return Err(BlockSizeError {
                format: "q6_K",
                expected: BLOCK_Q6_K_BYTES,
                found: bytes.len(),
            });
        }
        const QL: usize = QK_K / 2;
        const QH: usize = QK_K / 4;
        const SC: usize = QK_K / 16;

        let mut ql = [0u8; QL];
        ql.copy_from_slice(&bytes[..QL]);
        let mut qh = [0u8; QH];
        qh.copy_from_slice(&bytes[QL..QL + QH]);
        let mut scales = [0i8; SC];
        for (s, &b) in scales.iter_mut().zip(&bytes[QL + QH..QL + QH + SC]) {
            *s = b as i8;
        }
        let d_off = QL + QH + SC;
        let d = f16::from_le_bytes([bytes[d_off], bytes[d_off + 1]]);
        Ok(Self { ql, qh, scales, d })
    }

    /// Write this superblock in its on-disk representation.
    ///
    /// The exact inverse of [`Self::from_bytes`]; see
    /// [`BlockQ8_0::to_bytes`] for why the harness needs it.
    pub fn to_bytes(&self) -> [u8; BLOCK_Q6_K_BYTES] {
        const QL: usize = QK_K / 2;
        const QH: usize = QK_K / 4;
        const SC: usize = QK_K / 16;

        let mut out = [0u8; BLOCK_Q6_K_BYTES];
        out[..QL].copy_from_slice(&self.ql);
        out[QL..QL + QH].copy_from_slice(&self.qh);
        for (b, &sc) in out[QL + QH..QL + QH + SC].iter_mut().zip(&self.scales) {
            *b = sc as u8;
        }
        out[QL + QH + SC..].copy_from_slice(&self.d.to_le_bytes());
        out
    }
}

impl BlockQ4_0 {
    /// Parse one block from its on-disk representation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BlockSizeError> {
        if bytes.len() != BLOCK_Q4_0_BYTES {
            return Err(BlockSizeError {
                format: "q4_0",
                expected: BLOCK_Q4_0_BYTES,
                found: bytes.len(),
            });
        }
        let d = f16::from_le_bytes([bytes[0], bytes[1]]);
        let mut qs = [0u8; QK4_0 / 2];
        qs.copy_from_slice(&bytes[2..]);
        Ok(Self { d, qs })
    }

    /// Write this block in its on-disk representation.
    pub fn to_bytes(&self) -> [u8; BLOCK_Q4_0_BYTES] {
        let mut out = [0u8; BLOCK_Q4_0_BYTES];
        out[..2].copy_from_slice(&self.d.to_le_bytes());
        out[2..].copy_from_slice(&self.qs);
        out
    }
}

impl BlockQ4K {
    /// Parse one superblock from its on-disk representation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BlockSizeError> {
        if bytes.len() != BLOCK_Q4_K_BYTES {
            return Err(BlockSizeError {
                format: "q4_K",
                expected: BLOCK_Q4_K_BYTES,
                found: bytes.len(),
            });
        }
        let d = f16::from_le_bytes([bytes[0], bytes[1]]);
        let dmin = f16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales = [0u8; K_SCALE_SIZE];
        scales.copy_from_slice(&bytes[4..4 + K_SCALE_SIZE]);
        let mut qs = [0u8; QK_K / 2];
        qs.copy_from_slice(&bytes[4 + K_SCALE_SIZE..]);
        Ok(Self {
            d,
            dmin,
            scales,
            qs,
        })
    }

    /// Write this superblock in its on-disk representation.
    pub fn to_bytes(&self) -> [u8; BLOCK_Q4_K_BYTES] {
        let mut out = [0u8; BLOCK_Q4_K_BYTES];
        out[..2].copy_from_slice(&self.d.to_le_bytes());
        out[2..4].copy_from_slice(&self.dmin.to_le_bytes());
        out[4..4 + K_SCALE_SIZE].copy_from_slice(&self.scales);
        out[4 + K_SCALE_SIZE..].copy_from_slice(&self.qs);
        out
    }
}

impl BlockQ5K {
    /// Parse one superblock from its on-disk representation.
    ///
    /// `qh` comes *before* `qs` on disk, which is the opposite of the order
    /// the dequant loop uses them in; see [`BlockQ5K`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BlockSizeError> {
        if bytes.len() != BLOCK_Q5_K_BYTES {
            return Err(BlockSizeError {
                format: "q5_K",
                expected: BLOCK_Q5_K_BYTES,
                found: bytes.len(),
            });
        }
        const QH_OFF: usize = 4 + K_SCALE_SIZE;
        const QS_OFF: usize = QH_OFF + QK_K / 8;

        let d = f16::from_le_bytes([bytes[0], bytes[1]]);
        let dmin = f16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales = [0u8; K_SCALE_SIZE];
        scales.copy_from_slice(&bytes[4..QH_OFF]);
        let mut qh = [0u8; QK_K / 8];
        qh.copy_from_slice(&bytes[QH_OFF..QS_OFF]);
        let mut qs = [0u8; QK_K / 2];
        qs.copy_from_slice(&bytes[QS_OFF..]);
        Ok(Self {
            d,
            dmin,
            scales,
            qh,
            qs,
        })
    }

    /// Write this superblock in its on-disk representation.
    pub fn to_bytes(&self) -> [u8; BLOCK_Q5_K_BYTES] {
        const QH_OFF: usize = 4 + K_SCALE_SIZE;
        const QS_OFF: usize = QH_OFF + QK_K / 8;

        let mut out = [0u8; BLOCK_Q5_K_BYTES];
        out[..2].copy_from_slice(&self.d.to_le_bytes());
        out[2..4].copy_from_slice(&self.dmin.to_le_bytes());
        out[4..QH_OFF].copy_from_slice(&self.scales);
        out[QH_OFF..QS_OFF].copy_from_slice(&self.qh);
        out[QS_OFF..].copy_from_slice(&self.qs);
        out
    }
}

/// Dequantize a whole Q8_0 row.
///
/// `bytes` must be a whole number of blocks; the returned length is
/// `bytes.len() / BLOCK_Q8_0_BYTES * QK8_0`.
pub fn dequantize_row_q8_0(bytes: &[u8]) -> Result<Vec<f32>, BlockSizeError> {
    if !bytes.len().is_multiple_of(BLOCK_Q8_0_BYTES) {
        return Err(BlockSizeError {
            format: "q8_0 row",
            expected: BLOCK_Q8_0_BYTES,
            found: bytes.len() % BLOCK_Q8_0_BYTES,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_Q8_0_BYTES * QK8_0);
    for chunk in bytes.as_chunks::<BLOCK_Q8_0_BYTES>().0 {
        out.extend_from_slice(&dequantize_q8_0(&BlockQ8_0::from_bytes(chunk)?));
    }
    Ok(out)
}

/// Dequantize a whole Q6_K row.
pub fn dequantize_row_q6_k(bytes: &[u8]) -> Result<Vec<f32>, BlockSizeError> {
    if !bytes.len().is_multiple_of(BLOCK_Q6_K_BYTES) {
        return Err(BlockSizeError {
            format: "q6_K row",
            expected: BLOCK_Q6_K_BYTES,
            found: bytes.len() % BLOCK_Q6_K_BYTES,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_Q6_K_BYTES * QK_K);
    for chunk in bytes.as_chunks::<BLOCK_Q6_K_BYTES>().0 {
        out.extend_from_slice(&dequantize_q6_k(&BlockQ6K::from_bytes(chunk)?));
    }
    Ok(out)
}

/// Dequantize a whole Q4_0 row.
pub fn dequantize_row_q4_0(bytes: &[u8]) -> Result<Vec<f32>, BlockSizeError> {
    if !bytes.len().is_multiple_of(BLOCK_Q4_0_BYTES) {
        return Err(BlockSizeError {
            format: "q4_0 row",
            expected: BLOCK_Q4_0_BYTES,
            found: bytes.len() % BLOCK_Q4_0_BYTES,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_Q4_0_BYTES * QK4_0);
    for chunk in bytes.as_chunks::<BLOCK_Q4_0_BYTES>().0 {
        out.extend_from_slice(&dequantize_q4_0(&BlockQ4_0::from_bytes(chunk)?));
    }
    Ok(out)
}

/// Dequantize a whole Q4_K row.
pub fn dequantize_row_q4_k(bytes: &[u8]) -> Result<Vec<f32>, BlockSizeError> {
    if !bytes.len().is_multiple_of(BLOCK_Q4_K_BYTES) {
        return Err(BlockSizeError {
            format: "q4_K row",
            expected: BLOCK_Q4_K_BYTES,
            found: bytes.len() % BLOCK_Q4_K_BYTES,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_Q4_K_BYTES * QK_K);
    for chunk in bytes.as_chunks::<BLOCK_Q4_K_BYTES>().0 {
        out.extend_from_slice(&dequantize_q4_k(&BlockQ4K::from_bytes(chunk)?));
    }
    Ok(out)
}

/// Dequantize a whole Q5_K row.
pub fn dequantize_row_q5_k(bytes: &[u8]) -> Result<Vec<f32>, BlockSizeError> {
    if !bytes.len().is_multiple_of(BLOCK_Q5_K_BYTES) {
        return Err(BlockSizeError {
            format: "q5_K row",
            expected: BLOCK_Q5_K_BYTES,
            found: bytes.len() % BLOCK_Q5_K_BYTES,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_Q5_K_BYTES * QK_K);
    for chunk in bytes.as_chunks::<BLOCK_Q5_K_BYTES>().0 {
        out.extend_from_slice(&dequantize_q5_k(&BlockQ5K::from_bytes(chunk)?));
    }
    Ok(out)
}

/// Widen an fp16 row to fp32.
pub fn dequantize_row_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_le_bytes(*c).to_f32())
        .collect()
}

/// Widen a bf16 row to fp32.
///
/// bf16 is fp32 with the low 16 mantissa bits removed, so widening is a
/// shift — not the fp16 conversion, which has a different exponent width.
/// Getting these two confused produces values wrong by large powers of two,
/// which is exactly the kind of error that still looks like a plausible
/// weight distribution.
pub fn dequantize_row_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
        .collect()
}

#[cfg(test)]
mod row_tests {
    use super::*;

    #[test]
    fn block_sizes_match_the_ggml_layout() {
        // These are the numbers GGUF's own tensor directory is computed
        // from, so a mismatch here misaligns every block after the first.
        assert_eq!(BLOCK_Q8_0_BYTES, 34);
        assert_eq!(BLOCK_Q6_K_BYTES, 210);
        assert_eq!(BLOCK_Q4_0_BYTES, 18);
        assert_eq!(BLOCK_Q4_K_BYTES, 144);
        assert_eq!(BLOCK_Q5_K_BYTES, 176);
    }

    /// The packed 6-bit sub-scale codec, over every value both branches of
    /// `get_scale_min_k4` can produce.
    ///
    /// This is the highest-risk transcription in the file. The `j >= 4`
    /// branch borrows two high bits from a *different* byte than the one it
    /// takes the low nibble from, and getting it wrong scales a quarter of
    /// every Q4_K/Q5_K tensor by a wrong factor while leaving the result
    /// finite and plausible. Exhaustive over the 6-bit range rather than
    /// sampled, because the failure is in specific bit positions.
    #[test]
    fn packed_sub_scales_round_trip_over_the_whole_six_bit_range() {
        for base in 0..64u8 {
            let mut scales = [0u8; 8];
            let mut mins = [0u8; 8];
            for j in 0..8 {
                // Distinct per slot, so a codec that mixed two slots up
                // cannot pass by symmetry.
                scales[j] = (base + j as u8) & 63;
                mins[j] = (base + 8 + j as u8) & 63;
            }
            let packed = set_scale_min_k4(&scales, &mins);
            for j in 0..8 {
                let (sc, m) = get_scale_min_k4(j, &packed);
                assert_eq!(sc, scales[j], "scale {j} at base {base}");
                assert_eq!(m, mins[j], "min {j} at base {base}");
            }
        }
    }

    /// The nibble and bit-plane scatter both k-quants use, checked by
    /// building a superblock with known codes and reading back *where* they
    /// land.
    ///
    /// A code placed in the wrong nibble, or a `qh` bit written to the wrong
    /// pass, permutes the row while keeping every magnitude in range — the
    /// class of bug that survives an eyeball check of generated text. This
    /// pins position directly rather than through a round trip: with `d = 1`,
    /// every sub-scale 1 and every sub-min 0, a code *is* its own output, so
    /// any disagreement is a layout error and not quantization error.
    #[test]
    fn k_quant_codes_land_at_the_positions_the_dequant_reads() {
        let scales = set_scale_min_k4(&[1; 8], &[0; 8]);

        // Byte `32g + l` carries element `64g + l` in its low nibble and
        // element `64g + 32 + l` in its high nibble.
        let mut qs = [0u8; QK_K / 2];
        for g in 0..4 {
            for l in 0..32 {
                qs[g * 32 + l] = ((g + 1) as u8) | ((((l % 15) + 1) as u8) << 4);
            }
        }

        let y = dequantize_q4_k(&BlockQ4K {
            d: f16::from_f32(1.0),
            dmin: f16::from_f32(0.0),
            scales,
            qs,
        });
        for g in 0..4 {
            for l in 0..32 {
                assert_eq!(
                    y[g * 64 + l],
                    (g + 1) as f32,
                    "q4_K low nibble of byte {} belongs at {}",
                    g * 32 + l,
                    g * 64 + l,
                );
                assert_eq!(
                    y[g * 64 + 32 + l],
                    ((l % 15) + 1) as f32,
                    "q4_K high nibble of byte {} belongs at {}",
                    g * 32 + l,
                    g * 64 + 32 + l,
                );
            }
        }

        // Q5_K reuses one 32-byte `qh` plane across all four passes; the bit
        // that moves is `2g + sub`, not the byte. Set exactly one bit and
        // check exactly one element gains 16.
        for g in 0..4 {
            for sub in 0..2 {
                let mut qh = [0u8; QK_K / 8];
                qh[7] = 1 << (2 * g + sub);
                let y = dequantize_q5_k(&BlockQ5K {
                    d: f16::from_f32(1.0),
                    dmin: f16::from_f32(0.0),
                    scales,
                    qh,
                    qs: [0u8; QK_K / 2],
                });
                let want = g * 64 + sub * 32 + 7;
                for (i, &v) in y.iter().enumerate() {
                    let expect = if i == want { 16.0 } else { 0.0 };
                    assert_eq!(
                        v,
                        expect,
                        "q5_K high bit {} of qh[7] belongs at {want}, not {i}",
                        2 * g + sub,
                    );
                }
            }
        }
    }

    /// Q4_0's two nibbles are elements `j` and `j + 16`, not `2j` and
    /// `2j + 1`. Same reasoning as the k-quant position test.
    #[test]
    fn q4_0_nibbles_are_sixteen_apart_not_adjacent() {
        let mut qs = [0u8; QK4_0 / 2];
        // Low nibble 8 (code 0), high nibble 9 (code +1) in byte 0 only.
        qs[0] = 8 | (9 << 4);
        for q in qs.iter_mut().skip(1) {
            *q = 8 | (8 << 4);
        }
        let y = dequantize_q4_0(&BlockQ4_0 {
            d: f16::from_f32(1.0),
            qs,
        });
        assert_eq!(y[0], 0.0);
        assert_eq!(y[16], 1.0, "the high nibble of byte 0 is element 16");
        assert_eq!(y[1], 0.0, "element 1 is the low nibble of byte 1");
    }

    #[test]
    fn q4_0_round_trips_through_bytes() {
        let mut x = [0.0f32; QK4_0];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 - 16.0) * 0.25;
        }
        let block = quantize_q4_0(&x);
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q4_0_BYTES);

        let parsed = BlockQ4_0::from_bytes(&bytes).expect("parses");
        assert_eq!(parsed.d.to_bits(), block.d.to_bits());
        assert_eq!(parsed.qs, block.qs);
        assert_eq!(dequantize_q4_0(&parsed), dequantize_q4_0(&block));

        // A whole row through the same path, to pin the block stride.
        let row = dequantize_row_q4_0(&bytes).expect("row parses");
        assert_eq!(row.len(), QK4_0);
    }

    #[test]
    fn q4_k_round_trips_through_bytes() {
        let mut x = [0.0f32; QK_K];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 37) as f32 - 18.0) * 0.05;
        }
        let block = quantize_q4_k(&x);
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q4_K_BYTES);

        let parsed = BlockQ4K::from_bytes(&bytes).expect("parses");
        assert_eq!(parsed.d.to_bits(), block.d.to_bits());
        assert_eq!(parsed.dmin.to_bits(), block.dmin.to_bits());
        assert_eq!(parsed.scales, block.scales);
        assert_eq!(parsed.qs, block.qs);
        assert_eq!(dequantize_q4_k(&parsed), dequantize_q4_k(&block));
        assert_eq!(dequantize_row_q4_k(&bytes).expect("row").len(), QK_K);
    }

    #[test]
    fn q5_k_round_trips_through_bytes() {
        let mut x = [0.0f32; QK_K];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 53) as f32 - 26.0) * 0.03;
        }
        let block = quantize_q5_k(&x);
        let bytes = block.to_bytes();
        assert_eq!(bytes.len(), BLOCK_Q5_K_BYTES);

        let parsed = BlockQ5K::from_bytes(&bytes).expect("parses");
        assert_eq!(parsed.d.to_bits(), block.d.to_bits());
        assert_eq!(parsed.dmin.to_bits(), block.dmin.to_bits());
        assert_eq!(parsed.scales, block.scales);
        assert_eq!(parsed.qh, block.qh, "the high-bit plane must survive");
        assert_eq!(parsed.qs, block.qs);
        assert_eq!(dequantize_q5_k(&parsed), dequantize_q5_k(&block));
        assert_eq!(dequantize_row_q5_k(&bytes).expect("row").len(), QK_K);
    }

    /// Each format's round trip must be accurate to roughly its own step
    /// size, and — the part that actually distinguishes them — Q5_K must beat
    /// Q4_K on the same input.
    ///
    /// An encoder that silently ignored Q5_K's fifth bit would still pass a
    /// loose absolute bound; it cannot pass the comparison.
    ///
    /// Q6_K is deliberately *not* in the comparison. It is a symmetric format
    /// with no min, so which of it and Q5_K wins depends on how centred the
    /// data is — on an all-positive block Q6_K throws away half its range and
    /// loses badly. That is a property of the formats, not a bug, and
    /// asserting an order between them would encode a claim that is only true
    /// for some inputs.
    #[test]
    fn the_fifth_bit_of_q5_k_buys_accuracy_over_q4_k() {
        let mut x = [0.0f32; QK_K];
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for v in x.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // Weight-like: centred on zero, order 1.
            *v = (seed >> 40) as f32 / 8_388_608.0 - 1.0;
        }
        let err = |y: [f32; QK_K]| {
            x.iter()
                .zip(y.iter())
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
        };
        let e4 = err(dequantize_q4_k(&quantize_q4_k(&x)));
        let e5 = err(dequantize_q5_k(&quantize_q5_k(&x)));

        let span = x.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v))
            - x.iter().fold(f32::INFINITY, |m, v| m.min(*v));
        // 15 codes over a 32-element group, so a group's step is at most
        // span/15; allow a factor of two for the 6-bit scale quantization.
        assert!(e4 < span / 7.0, "q4_K max error {e4} against span {span}");
        assert!(e5 < e4, "q5_K ({e5}) must beat q4_K ({e4})");
    }

    #[test]
    fn q8_0_round_trips_through_bytes() {
        let mut x = [0.0f32; QK8_0];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 - 16.0) * 0.25;
        }
        let block = quantize_q8_0(&x);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&block.d.to_le_bytes());
        bytes.extend(block.qs.iter().map(|&q| q as u8));

        // `to_bytes` must agree with the hand-built serialization above,
        // which is the layout `from_bytes` and every kernel already read.
        assert_eq!(block.to_bytes().as_slice(), bytes.as_slice());

        let parsed = BlockQ8_0::from_bytes(&bytes).expect("parses");
        assert_eq!(parsed.d.to_bits(), block.d.to_bits());
        assert_eq!(parsed.qs, block.qs);
        assert_eq!(dequantize_q8_0(&parsed), dequantize_q8_0(&block));
    }

    #[test]
    fn q6_k_round_trips_through_bytes() {
        let mut x = [0.0f32; QK_K];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 61) as f32 - 30.0) * 0.1;
        }
        let block = quantize_q6_k(&x);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&block.ql);
        bytes.extend_from_slice(&block.qh);
        bytes.extend(block.scales.iter().map(|&s| s as u8));
        bytes.extend_from_slice(&block.d.to_le_bytes());
        assert_eq!(bytes.len(), BLOCK_Q6_K_BYTES);
        assert_eq!(block.to_bytes().as_slice(), bytes.as_slice());

        let parsed = BlockQ6K::from_bytes(&bytes).expect("parses");
        assert_eq!(dequantize_q6_k(&parsed), dequantize_q6_k(&block));
    }

    #[test]
    fn negative_scales_survive_the_byte_round_trip() {
        // `scales` and `qs` are int8 on disk but arrive as u8. Reading them
        // unsigned would flip the sign of roughly half of every tensor while
        // leaving the magnitudes plausible.
        let bytes = [0xFFu8; BLOCK_Q8_0_BYTES];
        let block = BlockQ8_0::from_bytes(&bytes).expect("parses");
        assert!(block.qs.iter().all(|&q| q == -1), "int8 read as unsigned");
    }

    #[test]
    fn bf16_widening_is_a_shift_not_an_fp16_conversion() {
        // 1.0f32 is 0x3F800000; its bf16 form is the top half, 0x3F80.
        let one = dequantize_row_bf16(&0x3F80u16.to_le_bytes());
        assert_eq!(one, vec![1.0f32]);
        // The same bit pattern read as fp16 is 1.875, which is what a
        // confused implementation would return.
        let as_f16 = dequantize_row_f16(&0x3F80u16.to_le_bytes());
        assert!((as_f16[0] - 1.875).abs() < 1e-6, "{:?}", as_f16);
    }

    #[test]
    fn a_row_that_is_not_a_whole_number_of_blocks_is_rejected() {
        assert!(dequantize_row_q6_k(&[0u8; BLOCK_Q6_K_BYTES + 1]).is_err());
        assert!(dequantize_row_q8_0(&[0u8; 33]).is_err());
    }
}
