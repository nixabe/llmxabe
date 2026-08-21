//! Host-side token sampling: temperature, top-k, and top-p over a logit
//! vector the device copied back.
//!
//! # Why the host, when argmax runs on the device
//!
//! Greedy decoding reduces 993 KiB of logits to four bytes on the device,
//! inside the captured decode graph, and that path is untouched: a request
//! that does not ask for sampling costs exactly what it cost before. A
//! request that does ask pays for one logits row across PCIe and an `O(vocab)`
//! host pass per token. That price buys three things a device sampler would
//! have to re-earn: the filtering pipeline is plain code with unit tests
//! rather than a kernel needing a differential harness, per-request RNG state
//! lives in ordinary structs instead of graph-stable device buffers, and the
//! captured decode graph does not change shape at all. If sampling ever
//! dominates a profile, the WHY NOT table is where the measured case for a
//! device sampler belongs.
//!
//! # Distribution semantics
//!
//! The filters chain the way llama.cpp's sampler chain does
//! (`src/llama-sampling.cpp`, `llama_sampler_top_k_impl` /
//! `llama_sampler_top_p_impl`): top-k keeps the k largest logits, then top-p
//! keeps the smallest prefix of the survivors — sorted by descending
//! probability, normalized over the survivors — whose cumulative mass reaches
//! `p`, and the final draw renormalizes over what is left. Temperature scales
//! log-probabilities before either filter looks at them.
//!
//! Speculative decoding needs no changes to stay exact under sampling: the
//! runtime samples the *target* model's token and accepts a draft only when
//! it equals that token, so every emitted token is drawn from the true
//! conditional distribution given the tokens before it. Lower temperatures
//! just accept fewer drafts.

/// What a request asked the sampler to do.
///
/// `temperature == 0` is greedy argmax — the caller-visible contract every
/// serving API documents — and the runtime treats it as "no sampler at all",
/// which is what keeps the greedy hot path free of this module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature. Zero selects greedy argmax; values above zero
    /// scale the logits by `1/temperature` before sampling.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens. Zero disables the filter.
    pub top_k: u32,
    /// Keep the smallest set of tokens whose cumulative probability reaches
    /// `top_p`. One disables the filter.
    pub top_p: f32,
    /// RNG seed. Equal seeds with equal parameters draw equal token
    /// sequences; the serving layer fills this with entropy when the caller
    /// did not pin it.
    pub seed: u64,
}

impl SamplingParams {
    /// Greedy argmax: what every request did before this module existed.
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: 0,
    };

    /// Whether these parameters are greedy argmax in disguise, so the
    /// on-device path can serve them.
    ///
    /// `top_k == 1` is greedy no matter the temperature: one candidate
    /// survives the filter, and sampling among one candidate is a lookup.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }
}

/// xoshiro256++, seeded through SplitMix64.
///
/// Written out rather than pulled from the `rand` crate because the workspace
/// does not depend on `rand`, and a serving engine's sampler needs exactly
/// two operations: seed deterministically, draw `u64`s. Algorithms from
/// Blackman & Vigna, <https://prng.di.unimi.it/> (public domain reference
/// implementations `xoshiro256plusplus.c` and `splitmix64.c`).
#[derive(Debug, Clone)]
struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    fn new(seed: u64) -> Self {
        // SplitMix64 expands one word into the four-word state; the reference
        // seeding recommendation, and it cannot produce the all-zero state.
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`: the high 53 bits, the standard double conversion.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Per-sequence sampler state: the parameters and the RNG they seed.
#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplingParams,
    rng: Xoshiro256pp,
}

/// Candidate sizes the top-p escalation tries before giving up and taking
/// the whole vocabulary. Nucleus cuts land within the first few hundred
/// tokens for any distribution a language model actually emits; the tail
/// sizes exist so a near-uniform distribution still gets the exact answer
/// rather than a truncated one.
const TOP_P_LADDER: [usize; 4] = [128, 1024, 16_384, usize::MAX];

impl Sampler {
    /// A sampler for `params`. Callers should route [`SamplingParams::is_greedy`]
    /// parameters to the device argmax path instead; a greedy `Sampler` would
    /// answer correctly but pay the host round-trip for nothing.
    pub fn new(params: SamplingParams) -> Self {
        Self {
            rng: Xoshiro256pp::new(params.seed),
            params,
        }
    }

    /// Draw one token id from `logits`.
    ///
    /// `scratch` is caller-owned candidate storage, cleared and refilled here,
    /// so a serving runtime can pre-size it once (AGENTS.md rule 6: no
    /// allocation on the hot path). Non-finite logits are treated as
    /// negative infinity — absent, exactly as a probability of zero would be.
    pub fn sample(&mut self, logits: &[f32], scratch: &mut Vec<(f32, u32)>) -> i32 {
        assert!(
            !logits.is_empty(),
            "cannot sample from an empty logit vector"
        );
        scratch.clear();
        scratch.extend(logits.iter().enumerate().map(|(index, &logit)| {
            let logit = if logit.is_finite() {
                logit
            } else {
                f32::NEG_INFINITY
            };
            (logit, index as u32)
        }));

        let top_k = self.params.top_k as usize;
        if top_k > 0 && top_k < scratch.len() {
            scratch.select_nth_unstable_by(top_k - 1, |a, b| b.0.total_cmp(&a.0));
            scratch.truncate(top_k);
        }

        // Every survivor at -inf means every logit was non-finite; sampling
        // weights would be NaN. Answer index 0, the same token the device
        // argmax answers for an all-NaN vector.
        let max = scratch
            .iter()
            .map(|&(logit, _)| logit)
            .fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return 0;
        }

        let inv_t = f64::from(self.params.temperature).recip();
        let weight = |logit: f32| (f64::from(logit - max) * inv_t).exp();

        if self.params.top_p < 1.0 {
            let target = f64::from(self.params.top_p.max(0.0));
            // The mass the cut must reach is measured against *all* current
            // candidates, so it is summed before any of them are dropped.
            let total: f64 = scratch.iter().map(|&(logit, _)| weight(logit)).sum();
            for cap in TOP_P_LADDER {
                let sorted = cap.min(scratch.len());
                if sorted < scratch.len() {
                    // Partition the `sorted` largest to the front, then order
                    // just that prefix: `select_nth` alone leaves the prefix
                    // unordered, and the nucleus is defined by a *descending*
                    // cumulative scan.
                    scratch.select_nth_unstable_by(sorted - 1, |a, b| b.0.total_cmp(&a.0));
                }
                scratch[..sorted].sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                let mut mass = 0.0f64;
                let mut keep = None;
                for (index, &(logit, _)) in scratch.iter().take(sorted).enumerate() {
                    mass += weight(logit);
                    if mass >= target * total {
                        keep = Some(index + 1);
                        break;
                    }
                }
                match keep {
                    Some(keep) => {
                        scratch.truncate(keep);
                        break;
                    }
                    // The scan covered every candidate without reaching the
                    // target — floating-point shortfall at `top_p` ~ 1. Keep
                    // everything; that is what the target asked for.
                    None if sorted == scratch.len() => break,
                    None => {}
                }
            }
        }

        let total: f64 = scratch.iter().map(|&(logit, _)| weight(logit)).sum();
        let mut draw = self.rng.next_f64() * total;
        for &(logit, index) in scratch.iter() {
            draw -= weight(logit);
            if draw <= 0.0 {
                return index as i32;
            }
        }
        // Rounding pushed the draw past the last cumulative step; the last
        // candidate is the one the draw was converging on.
        scratch.last().map_or(0, |&(_, index)| index as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(temperature: f32, top_k: u32, top_p: f32, seed: u64) -> Sampler {
        Sampler::new(SamplingParams {
            temperature,
            top_k,
            top_p,
            seed,
        })
    }

    fn draw_many(sampler: &mut Sampler, logits: &[f32], n: usize) -> Vec<i32> {
        let mut scratch = Vec::new();
        (0..n)
            .map(|_| sampler.sample(logits, &mut scratch))
            .collect()
    }

    #[test]
    fn zero_temperature_and_top_k_one_are_greedy() {
        assert!(SamplingParams::GREEDY.is_greedy());
        assert!(
            SamplingParams {
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
                seed: 7,
            }
            .is_greedy()
        );
        assert!(
            !SamplingParams {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                seed: 7,
            }
            .is_greedy()
        );
    }

    #[test]
    fn equal_seeds_draw_equal_sequences_and_different_seeds_diverge() {
        let logits: Vec<f32> = (0..512).map(|i| ((i * 37) % 101) as f32 * 0.05).collect();
        let a = draw_many(&mut sampler(1.0, 0, 1.0, 42), &logits, 64);
        let b = draw_many(&mut sampler(1.0, 0, 1.0, 42), &logits, 64);
        let c = draw_many(&mut sampler(1.0, 0, 1.0, 43), &logits, 64);
        assert_eq!(a, b, "the same seed must replay the same tokens");
        assert_ne!(
            a, c,
            "different seeds drawing identically is vanishingly unlikely"
        );
    }

    #[test]
    fn near_zero_temperature_concentrates_on_the_argmax() {
        let logits = [1.0f32, 3.0, 2.0, -1.0];
        let drawn = draw_many(&mut sampler(1e-4, 0, 1.0, 5), &logits, 100);
        assert!(drawn.iter().all(|&token| token == 1));
    }

    #[test]
    fn top_k_bounds_the_support() {
        let logits: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        // The three largest logits are indices 97, 98, 99.
        let drawn = draw_many(&mut sampler(2.0, 3, 1.0, 11), &logits, 500);
        assert!(drawn.iter().all(|&token| token >= 97));
        assert!(
            drawn.iter().any(|&token| token != 99),
            "high temperature over three candidates should not collapse to one"
        );
    }

    #[test]
    fn top_p_drops_the_tail() {
        // Probabilities 0.5, 0.3, 0.2 at temperature 1: a 0.7 nucleus keeps
        // exactly the first two (0.5 alone misses, 0.5 + 0.3 crosses).
        let logits = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        let drawn = draw_many(&mut sampler(1.0, 0, 0.7, 3), &logits, 500);
        assert!(drawn.iter().all(|&token| token == 0 || token == 1));
        assert!(drawn.contains(&0) && drawn.contains(&1));
    }

    #[test]
    fn a_tiny_top_p_is_argmax() {
        let logits = [0.1f32, 0.9, 0.3];
        let drawn = draw_many(&mut sampler(1.5, 0, 1e-6, 9), &logits, 100);
        assert!(drawn.iter().all(|&token| token == 1));
    }

    #[test]
    fn top_p_escalates_past_the_first_ladder_rung_when_the_head_is_flat() {
        // 4,096 equal logits: a 0.5 nucleus needs 2,048 candidates, which is
        // past the 128 and 1,024 rungs. Every index must stay reachable
        // within the kept half, and none outside it is checkable — but the
        // draw must not panic and must stay in range, which is what breaks if
        // the escalation truncates at a rung that has not reached the mass.
        let logits = vec![0.0f32; 4096];
        let drawn = draw_many(&mut sampler(1.0, 0, 0.5, 17), &logits, 200);
        assert!(drawn.iter().all(|&token| (0..4096).contains(&token)));
        let distinct: std::collections::HashSet<_> = drawn.iter().collect();
        assert!(
            distinct.len() > 50,
            "a flat distribution should not collapse"
        );
    }

    #[test]
    fn observed_frequencies_track_the_distribution() {
        // Two tokens at 3:1 odds. 10,000 draws put the observed frequency
        // within ±0.03 of 0.75 with overwhelming probability (sigma ~0.004).
        let logits = [0.75f32.ln(), 0.25f32.ln()];
        let drawn = draw_many(&mut sampler(1.0, 0, 1.0, 23), &logits, 10_000);
        let zeros = drawn.iter().filter(|&&token| token == 0).count() as f64;
        let frequency = zeros / drawn.len() as f64;
        assert!(
            (frequency - 0.75).abs() < 0.03,
            "observed {frequency}, expected 0.75 ± 0.03"
        );
    }

    #[test]
    fn temperature_flattens_the_distribution() {
        let logits = [2.0f32, 0.0];
        let cold = draw_many(&mut sampler(0.5, 0, 1.0, 31), &logits, 4_000);
        let hot = draw_many(&mut sampler(2.0, 0, 1.0, 31), &logits, 4_000);
        let ones = |drawn: &[i32]| drawn.iter().filter(|&&token| token == 1).count();
        assert!(
            ones(&hot) > ones(&cold) + 200,
            "temperature 2 must reach the minority token far more often than 0.5"
        );
    }

    #[test]
    fn non_finite_logits_are_excluded_not_propagated() {
        let logits = [f32::NAN, 5.0, f32::NEG_INFINITY, f32::INFINITY];
        let drawn = draw_many(&mut sampler(1.0, 0, 1.0, 13), &logits, 100);
        assert!(drawn.iter().all(|&token| token == 1));
        // All non-finite answers index 0, matching the device argmax's
        // all-NaN behavior.
        let all_bad = [f32::NAN, f32::NAN];
        assert_eq!(
            draw_many(&mut sampler(1.0, 0, 1.0, 13), &all_bad, 1),
            vec![0]
        );
    }
}
