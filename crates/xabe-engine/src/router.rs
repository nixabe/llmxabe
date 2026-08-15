//! Cache-aware request routing across workers.
//!
//! Three workers each hold a complete copy of the model, so any request can go
//! to any of them and the answer is identical. Routing is purely a cost
//! question: which worker will compute this request most cheaply, accounting
//! for what it already has cached and how loaded it already is.
//!
//! The scoring function is
//!
//! ```text
//! score(w) = a * prefix_match(w) - b * queue_pressure(w) - c * kv_utilization(w)
//! ```
//!
//! Cache affinity pulls a request toward the worker holding its prefix; the
//! load terms push back when that worker is saturated. Without the load terms
//! a popular prefix would pin all traffic to one card while two sit idle.

use crate::worker::WorkerId;

/// Weights for the routing score.
///
/// All three inputs are normalized to roughly `[0, 1]` before weighting (see
/// [`WorkerLoad`]), which is what makes these coefficients comparable to each
/// other. Scoring raw token counts against a utilization fraction would make
/// the weights carry units and the balance impossible to reason about.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouterConfig {
    /// `a` — reward for holding a prefix of the incoming request.
    pub prefix_weight: f64,
    /// `b` — penalty for already-queued work.
    pub queue_weight: f64,
    /// `c` — penalty for a nearly-full KV pool.
    pub kv_weight: f64,
}

impl RouterConfig {
    /// Starting weights.
    ///
    /// Prefix affinity dominates because avoiding prefill is worth far more
    /// than balancing a few queued tokens: a 32K prefix hit saves 32K tokens
    /// of compute, while a slightly longer queue costs milliseconds. KV
    /// utilization is weighted between the two because running a pool to
    /// exhaustion causes preemption, whose cost is not local to this request.
    ///
    /// These are tuning parameters, not constants. They trade
    /// time-to-first-token against inter-token latency, and the right balance
    /// depends on traffic shape — in particular on how much prefix sharing
    /// the workload actually exhibits.
    pub const fn balanced() -> Self {
        Self {
            prefix_weight: 1.0,
            queue_weight: 0.3,
            kv_weight: 0.5,
        }
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

/// A worker's current state, as the router sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkerLoad {
    /// Which worker this describes.
    pub id: WorkerId,
    /// Tokens of the incoming request already cached on this worker, after
    /// the GDN retention-boundary truncation described in `docs/CACHE.md`.
    ///
    /// Truncated rather than raw, because an untruncated match overstates the
    /// saving: the recurrent state is only resumable at a retained snapshot.
    pub matched_tokens: u32,
    /// Tokens queued and not yet computed on this worker.
    pub queued_tokens: u32,
    /// Fraction of the attention block pool currently in use, in `[0, 1]`.
    pub kv_utilization: f64,
    /// Whether this worker can accept the request at all.
    ///
    /// A worker that cannot admit is excluded outright, regardless of how
    /// good its cache affinity is. Admission is a hard constraint; scoring is
    /// only a preference.
    pub can_admit: bool,
}

impl WorkerLoad {
    /// Fraction of the incoming request already cached here, in `[0, 1]`.
    ///
    /// Normalizing by prompt length rather than using raw matched tokens is
    /// what makes the score comparable across requests of different sizes: a
    /// 1,000-token match means something very different for a 1,200-token
    /// prompt than for a 100,000-token one.
    fn prefix_fraction(&self, prompt_tokens: u32) -> f64 {
        if prompt_tokens == 0 {
            return 0.0;
        }
        f64::from(self.matched_tokens.min(prompt_tokens)) / f64::from(prompt_tokens)
    }

    /// Queued work scaled by the per-step token budget.
    ///
    /// A queue of one budget's worth is one step of delay, so this reads as
    /// "steps of work ahead of you". Unbounded above, deliberately — a badly
    /// backed-up worker should keep getting worse scores rather than
    /// saturating at some ceiling.
    fn queue_pressure(&self, token_budget: u32) -> f64 {
        if token_budget == 0 {
            return 0.0;
        }
        f64::from(self.queued_tokens) / f64::from(token_budget)
    }

    /// The routing score for this worker against a given request.
    pub fn score(&self, cfg: &RouterConfig, prompt_tokens: u32, token_budget: u32) -> f64 {
        cfg.prefix_weight * self.prefix_fraction(prompt_tokens)
            - cfg.queue_weight * self.queue_pressure(token_budget)
            - cfg.kv_weight * self.kv_utilization
    }
}

/// Why routing produced no worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingError {
    /// There are no workers at all.
    NoWorkers,
    /// Every worker refused admission.
    ///
    /// Distinct from [`Self::NoWorkers`]: the fleet exists and is simply
    /// full, so the caller should queue the request rather than fail it.
    AllWorkersSaturated,
}

impl core::fmt::Display for RoutingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoWorkers => write!(f, "no workers registered"),
            Self::AllWorkersSaturated => {
                write!(f, "every worker refused admission; queue rather than fail")
            }
        }
    }
}

impl core::error::Error for RoutingError {}

/// A routing decision, with the score that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Routed {
    /// The chosen worker.
    pub worker: WorkerId,
    /// Its score. Retained so routing decisions can be logged and explained
    /// rather than merely observed.
    pub score: f64,
    /// Tokens this worker already holds — the prefill that routing avoided.
    pub matched_tokens: u32,
}

/// Pick the worker that should serve a request.
///
/// Workers that cannot admit are excluded before scoring. Ties break toward
/// the lower worker id, which keeps routing deterministic and therefore
/// testable.
pub fn route(
    cfg: &RouterConfig,
    loads: &[WorkerLoad],
    prompt_tokens: u32,
    token_budget: u32,
) -> Result<Routed, RoutingError> {
    if loads.is_empty() {
        return Err(RoutingError::NoWorkers);
    }

    let mut best: Option<Routed> = None;
    for load in loads.iter().filter(|l| l.can_admit) {
        let score = load.score(cfg, prompt_tokens, token_budget);
        let better = match &best {
            None => true,
            // Strictly greater, so an earlier (lower-id) worker wins ties.
            Some(b) => score > b.score,
        };
        if better {
            best = Some(Routed {
                worker: load.id,
                score,
                matched_tokens: load.matched_tokens,
            });
        }
    }

    best.ok_or(RoutingError::AllWorkersSaturated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(id: u32, matched: u32, queued: u32, kv: f64) -> WorkerLoad {
        WorkerLoad {
            id: WorkerId(id),
            matched_tokens: matched,
            queued_tokens: queued,
            kv_utilization: kv,
            can_admit: true,
        }
    }

    const BUDGET: u32 = 4096;

    #[test]
    fn an_empty_fleet_is_distinguishable_from_a_saturated_one() {
        // The caller must be able to tell "misconfigured" from "busy": one is
        // a failure, the other means queue and retry.
        assert_eq!(
            route(&RouterConfig::balanced(), &[], 100, BUDGET),
            Err(RoutingError::NoWorkers)
        );

        let full = [WorkerLoad {
            can_admit: false,
            ..load(0, 0, 0, 0.9)
        }];
        assert_eq!(
            route(&RouterConfig::balanced(), &full, 100, BUDGET),
            Err(RoutingError::AllWorkersSaturated)
        );
    }

    #[test]
    fn cache_affinity_wins_when_load_is_equal() {
        let loads = [
            load(0, 0, 0, 0.1),
            load(1, 8000, 0, 0.1),
            load(2, 0, 0, 0.1),
        ];
        let r = route(&RouterConfig::balanced(), &loads, 10_000, BUDGET).unwrap();
        assert_eq!(r.worker, WorkerId(1));
        assert_eq!(r.matched_tokens, 8000);
    }

    #[test]
    fn a_saturated_worker_loses_despite_holding_the_prefix() {
        // This is the behaviour the load terms exist for. Without them a
        // popular prefix pins all traffic to one card while two sit idle.
        let loads = [
            // Holds the whole prefix, but is deeply backed up and nearly full.
            load(0, 10_000, 40 * BUDGET, 0.98),
            // Holds nothing, but is idle.
            load(1, 0, 0, 0.05),
        ];
        let r = route(&RouterConfig::balanced(), &loads, 10_000, BUDGET).unwrap();
        assert_eq!(
            r.worker,
            WorkerId(1),
            "cache affinity must not override extreme load"
        );
    }

    #[test]
    fn a_worker_that_cannot_admit_is_never_chosen() {
        let loads = [
            WorkerLoad {
                can_admit: false,
                ..load(0, 10_000, 0, 0.0)
            },
            load(1, 0, 0, 0.9),
        ];
        let r = route(&RouterConfig::balanced(), &loads, 10_000, BUDGET).unwrap();
        assert_eq!(
            r.worker,
            WorkerId(1),
            "admission is a hard constraint, not a preference"
        );
    }

    #[test]
    fn prefix_credit_is_relative_to_prompt_length() {
        // 1,000 matched tokens is nearly everything for a short prompt and
        // almost nothing for a long one. The score must reflect that, or
        // routing skews by request size rather than by cache value.
        let w = load(0, 1000, 0, 0.0);
        let cfg = RouterConfig::balanced();
        let short = w.score(&cfg, 1_200, BUDGET);
        let long = w.score(&cfg, 100_000, BUDGET);
        assert!(
            short > long,
            "same match, shorter prompt: {short} should exceed {long}"
        );
    }

    #[test]
    fn a_match_longer_than_the_prompt_does_not_score_above_a_full_match() {
        // Guards against a stale or over-long match inflating the score past
        // the natural ceiling of 1.0 * prefix_weight.
        let cfg = RouterConfig::balanced();
        let exact = load(0, 1000, 0, 0.0).score(&cfg, 1000, BUDGET);
        let over = load(1, 5000, 0, 0.0).score(&cfg, 1000, BUDGET);
        assert!((exact - over).abs() < 1e-12);
    }

    #[test]
    fn ties_break_deterministically_toward_the_lower_worker_id() {
        let loads = [
            load(0, 500, 0, 0.2),
            load(1, 500, 0, 0.2),
            load(2, 500, 0, 0.2),
        ];
        for _ in 0..8 {
            let r = route(&RouterConfig::balanced(), &loads, 1000, BUDGET).unwrap();
            assert_eq!(r.worker, WorkerId(0));
        }
    }

    #[test]
    fn queue_pressure_grows_without_saturating() {
        // A badly backed-up worker should keep getting worse, not plateau.
        let cfg = RouterConfig::balanced();
        let a = load(0, 0, 10 * BUDGET, 0.0).score(&cfg, 1000, BUDGET);
        let b = load(0, 0, 100 * BUDGET, 0.0).score(&cfg, 1000, BUDGET);
        assert!(b < a);
    }

    #[test]
    fn zero_length_prompts_and_budgets_do_not_produce_nan() {
        let cfg = RouterConfig::balanced();
        let s = load(0, 0, 0, 0.5).score(&cfg, 0, 0);
        assert!(s.is_finite(), "score must stay finite for degenerate input");
    }
}
