//! IEEE 754 binary16 conversion, on the host.
//!
//! The device has `cvt.rn.f16.f32` and `cvt.f32.f16`; the host needs the same
//! two so that a differential test can feed the CPU reference **exactly the
//! values the kernel will see**. Without that, a test comparing an fp16 kernel
//! against an fp32 reference measures the operand rounding rather than the
//! kernel, and the only way to pass it is to widen the tolerance until it stops
//! measuring anything.
//!
//! Round-to-nearest-even, matching `cvt.rn`. Subnormals, overflow to infinity,
//! and NaN payloads are all handled rather than assumed away: the KV cache
//! holds post-RoPE keys whose components really do reach both ends of the
//! range, and a silent flush-to-zero here would show up as a tolerance failure
//! somewhere far from its cause.

/// Round an `f32` to binary16, returning the bit pattern.
pub fn to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let biased = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x007f_ffff;

    if biased == 0xff {
        // Infinity, or a NaN whose payload must stay non-zero after the shift.
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }

    let exp = biased - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00; // overflows binary16's range
    }
    if exp <= 0 {
        // Subnormal in binary16, or under even that. The implicit leading one
        // comes back for the shift, which is `14 - exp` because the result's
        // exponent is pinned at the subnormal minimum.
        if exp < -10 {
            return sign;
        }
        let full = mant | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let half = 1u32 << (shift - 1);
        let sticky = full & (half - 1);
        let mut r = full >> shift;
        if (full & half) != 0 && (sticky != 0 || (r & 1) != 0) {
            r += 1;
        }
        return sign | r as u16;
    }

    let half = 1u32 << 12;
    let sticky = mant & (half - 1);
    let mut m = mant >> 13;
    let mut e = exp;
    if (mant & half) != 0 && (sticky != 0 || (m & 1) != 0) {
        m += 1;
        if m == 0x400 {
            // The round carried out of the significand and into the exponent.
            m = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | m as u16
}

/// Widen a binary16 bit pattern back to `f32`. Exact — every binary16 is
/// representable.
pub fn from_f16_bits(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign); // signed zero
        }
        // Subnormal: normalize by shifting the leading one into place and
        // paying for it in the exponent.
        let mut m = mant;
        let mut e: i32 = -1;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        let e32 = (e + 1 - 14 + 127) as u32;
        return f32::from_bits(sign | (e32 << 23) | ((m & 0x3ff) << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// Round through binary16 and back, which is what the KV cache does to every
/// key and value it stores.
pub fn round_through_f16(x: f32) -> f32 {
    from_f16_bits(to_f16_bits(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_representable_values_survive_the_round_trip() {
        for &x in &[
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            2.0,
            1024.0,
            -1024.0,
            65504.0,
            -65504.0,
            0.25,
            6.103_515_6e-5,
        ] {
            assert_eq!(round_through_f16(x), x, "{x} is representable in binary16");
        }
    }

    #[test]
    fn rounding_is_to_nearest_even_at_the_tie() {
        // Halfway between two binary16 neighbours near 1.0, whose spacing is
        // 2^-10. 1.0 + 2^-11 ties between 1.0 (even significand) and
        // 1 + 2^-10 (odd), so it must go to 1.0.
        let tie_down = 1.0f32 + 2f32.powi(-11);
        assert_eq!(round_through_f16(tie_down), 1.0);
        // 1 + 2^-10 + 2^-11 ties between 1 + 2^-10 (odd) and 1 + 2^-9 (even),
        // so it must go up.
        let tie_up = 1.0f32 + 2f32.powi(-10) + 2f32.powi(-11);
        assert_eq!(round_through_f16(tie_up), 1.0 + 2f32.powi(-9));
    }

    #[test]
    fn the_relative_error_never_exceeds_half_an_ulp() {
        // 2^-11 is half the 2^-10 significand step, so this is the bound the
        // differential tests' tolerances are derived from.
        let mut rng = crate::rng::Xorshift64Star::new(0x00F1_6000);
        let mut worst = 0.0f32;
        for _ in 0..200_000 {
            let x = rng.next_f32_range(-4.0, 4.0);
            if x == 0.0 {
                continue;
            }
            worst = worst.max(((round_through_f16(x) - x) / x).abs());
        }
        assert!(
            worst <= 2f32.powi(-11),
            "worst relative error {worst:.3e} exceeds half an ulp {:.3e}",
            2f32.powi(-11),
        );
    }

    #[test]
    fn magnitudes_past_the_range_saturate_rather_than_wrap() {
        assert_eq!(round_through_f16(1e30), f32::INFINITY);
        assert_eq!(round_through_f16(-1e30), f32::NEG_INFINITY);
        // Half of the smallest subnormal rounds to zero, keeping its sign.
        assert_eq!(round_through_f16(1e-10).to_bits(), 0);
        assert_eq!(round_through_f16(-1e-10).to_bits(), 0x8000_0000);
    }

    #[test]
    fn subnormals_round_trip_through_the_normalizing_branch() {
        // 2^-24 is the smallest positive binary16 subnormal.
        let smallest = 2f32.powi(-24);
        assert_eq!(round_through_f16(smallest), smallest);
        assert_eq!(round_through_f16(2f32.powi(-20)), 2f32.powi(-20));
    }
}
