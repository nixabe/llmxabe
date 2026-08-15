//! A small, dependency-free, deterministic pseudo-random generator.
//!
//! Differential tests need reproducible inputs: the same seed must produce
//! the same tensor on every run, on every machine, forever, so that a
//! regression can be replayed exactly. Pulling in the `rand` crate for this
//! would add a dependency whose own version bumps could silently change the
//! stream. This is [xorshift64*](https://en.wikipedia.org/wiki/Xorshift#xorshift*),
//! a well-known 64-bit generator with a documented, fixed transition
//! function — good enough statistically for generating test tensors, and
//! small enough to vendor and read in thirty seconds.
#[derive(Debug, Clone)]
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    /// Seeds the generator. `seed` must be non-zero (xorshift has a fixed
    /// point at all-zero state); zero is remapped to a fixed non-zero
    /// constant so callers can't accidentally hang the stream.
    pub const fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits give a value exactly representable in f32 mantissa.
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    /// A uniform `f32` in `[low, high)`.
    pub fn next_f32_range(&mut self, low: f32, high: f32) -> f32 {
        low + self.next_f32() * (high - low)
    }

    /// A uniform `u32` in `[0, bound)`. `bound` must be non-zero.
    pub fn next_u32_below(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "next_u32_below requires a non-zero bound");
        (self.next_u64() % u64::from(bound)) as u32
    }

    /// A vector of `n` values drawn uniformly from `[low, high)`.
    pub fn vec_f32(&mut self, n: usize, low: f32, high: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32_range(low, high)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_reproduces_the_same_stream() {
        let mut a = Xorshift64Star::new(42);
        let mut b = Xorshift64Star::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Xorshift64Star::new(1);
        let mut b = Xorshift64Star::new(2);
        let sa: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let sb: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        assert_ne!(sa, sb);
    }

    #[test]
    fn zero_seed_does_not_hang_or_stay_zero() {
        let mut rng = Xorshift64Star::new(0);
        assert_ne!(rng.next_u64(), 0);
    }

    #[test]
    fn next_f32_stays_in_unit_interval() {
        let mut rng = Xorshift64Star::new(7);
        for _ in 0..10_000 {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x), "value {x} escaped [0, 1)");
        }
    }

    #[test]
    fn next_u32_below_respects_the_bound() {
        let mut rng = Xorshift64Star::new(99);
        for _ in 0..10_000 {
            let x = rng.next_u32_below(7);
            assert!(x < 7);
        }
    }
}
