//! Sampling strategies for text generation.
//!
//! Supports temperature scaling, top-k filtering, top-p (nucleus) filtering,
//! and repetition penalty. The [`Sampler`] converts a logit vector into a
//! single token ID using these strategies in order:
//!
//! 1. **Temperature scaling** — divide logits by temperature (0 = greedy argmax)
//! 2. **Top-k** — keep only the k highest-probability candidates
//! 3. **Softmax** — convert scaled logits to probabilities
//! 4. **Top-p** — keep the smallest set of tokens whose cumulative probability exceeds p
//! 5. **Weighted random selection** — sample from the filtered distribution

use std::cmp::Ordering;

use crate::error::RuntimeResult;

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for softmax scaling. 0.0 = greedy.
    pub temperature: f32,
    /// Top-k filtering (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) threshold (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Maximum number of new tokens to generate per request.
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            max_tokens: 128,
        }
    }
}

/// Token sampler.
///
/// Owns a reusable `probs_buf` that is grown on first use and then reused across
/// all subsequent `sample()` calls, eliminating the ~1.8 MB per-call heap
/// allocation that a fresh `Vec` would require for a 151 936-token vocabulary.
#[derive(Debug)]
pub struct Sampler {
    params: SamplingParams,
    rng_state: u64,
    /// Reusable working buffer for `(token_index, scaled_logit)` pairs.
    ///
    /// After `select_nth_unstable_by` + `drain` the buffer holds only the top-k
    /// candidates (capacity stays at `vocab_size`).  `clear()` on the next call
    /// resets length to zero without freeing the backing store, so subsequent
    /// `extend()` calls never reallocate.
    probs_buf: Vec<(usize, f32)>,
    /// Running history of generated token IDs for repetition penalty.
    /// Bug #2 fix: Track tokens to apply repetition penalty during sampling.
    generated_tokens: Vec<u32>,
}

impl Sampler {
    /// Create a new sampler with the given parameters and seed.
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        Self {
            params,
            rng_state: seed,
            probs_buf: Vec::new(),
            generated_tokens: Vec::new(),
        }
    }

    /// Reset the generated token history (e.g., for new conversation turn).
    pub fn reset(&mut self) {
        self.generated_tokens.clear();
    }

    /// Simple xorshift64 PRNG — no external dependency needed.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Sample a token index from logits.
    #[tracing::instrument(skip(self, logits), fields(vocab_size = logits.len()), level = "debug")]
    pub fn sample(&mut self, logits: &mut [f32]) -> RuntimeResult<u32> {
        if logits.is_empty() {
            return Ok(0);
        }

        // Apply repetition penalty BEFORE temperature scaling (Bug #2 fix)
        // Modify logits in place based on generated token history
        if self.params.repetition_penalty > 1.0 && !self.generated_tokens.is_empty() {
            let penalty = self.params.repetition_penalty;
            for &token_id in &self.generated_tokens {
                let idx = token_id as usize;
                if idx < logits.len() {
                    // For tokens already generated:
                    // - Positive logits are divided by penalty (discouraged)
                    // - Negative logits are multiplied by penalty (further discouraged)
                    let logit = logits[idx];
                    logits[idx] = if logit >= 0.0 {
                        logit / penalty
                    } else {
                        logit * penalty
                    };
                }
            }
        }

        // Greedy if temperature is ~0
        if self.params.temperature < 1e-6 {
            let token = argmax(logits) as u32;
            self.generated_tokens.push(token);
            return Ok(token);
        }

        // Populate the reusable buffer with temperature-scaled logits.
        // On the first call this allocates `vocab_size × 12` bytes; every
        // subsequent call reuses the existing backing store (len is reset to 0
        // by `clear()`, capacity is preserved from the previous call).
        self.probs_buf.clear();
        self.probs_buf.extend(
            logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v / self.params.temperature)),
        );

        // Top-k filtering — O(n) average via partial selection rather than O(n log n) full sort.
        // `select_nth_unstable_by` rearranges `probs_buf` so that element at index `cutoff` is in
        // its fully-sorted position, all elements before it are ≤ it (lower scaled logits), and all
        // elements after it are ≥ it (higher scaled logits).  Draining the prefix leaves exactly
        // the top-k elements in arbitrary order, which is sufficient for softmax + sampling.
        if self.params.top_k > 0 && self.params.top_k < self.probs_buf.len() {
            let k = self.params.top_k;
            let cutoff = self.probs_buf.len() - k;
            self.probs_buf.select_nth_unstable_by(cutoff, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
            });
            self.probs_buf.drain(..cutoff);
        }

        // Softmax
        let max_val = self
            .probs_buf
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (_, v) in self.probs_buf.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        for (_, v) in self.probs_buf.iter_mut() {
            *v /= sum;
        }

        // Top-p filtering
        if self.params.top_p < 1.0 {
            self.probs_buf
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            let mut cum = 0.0f32;
            let cutoff = self
                .probs_buf
                .iter()
                .position(|&(_, p)| {
                    cum += p;
                    cum > self.params.top_p
                })
                .unwrap_or(self.probs_buf.len().saturating_sub(1));
            self.probs_buf.truncate(cutoff + 1);

            // Re-normalize
            let sum: f32 = self.probs_buf.iter().map(|(_, p)| p).sum();
            for (_, p) in self.probs_buf.iter_mut() {
                *p /= sum;
            }
        }

        // Pre-compute random value before the immutable borrow of `probs_buf`
        // to satisfy the borrow checker: `next_u64` takes `&mut self` which
        // would conflict with an active `&self.probs_buf` borrow.
        let rand_val = (self.next_u64() as f64 / u64::MAX as f64) as f32;

        // Weighted random selection
        let mut cum = 0.0f32;
        let mut chosen_token = self.probs_buf[0].0 as u32;
        for &(idx, p) in &self.probs_buf {
            cum += p;
            if rand_val <= cum {
                chosen_token = idx as u32;
                break;
            }
        }

        // Track the chosen token for future repetition penalty applications
        self.generated_tokens.push(chosen_token);
        Ok(chosen_token)
    }

    /// Get current parameters.
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }
}

/// Return the index of the maximum element.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling() {
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(params.clone(), 42);
        let mut logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sampler.sample(&mut logits).expect("sampling should succeed");
        assert_eq!(token, 3); // index of 0.9
    }

    #[test]
    fn sampling_returns_valid_index() {
        let params = SamplingParams::default();
        let mut sampler = Sampler::new(params, 12345);
        let mut logits = vec![0.0f32; 100];
        for _ in 0..50 {
            let token = sampler.sample(&mut logits).expect("sampling should succeed");
            assert!(token < 100);
        }
    }

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn buffer_reuse_across_calls() {
        // Verify the probs_buf is correctly reused without incorrect state leaking.
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 5,
            top_p: 1.0, // disable top-p so we control exactly
            repetition_penalty: 1.0,
            max_tokens: 128,
        };
        let mut sampler = Sampler::new(params, 99);
        let mut logits: Vec<f32> = (0..200).map(|i| i as f32 * 0.01).collect();
        for _ in 0..20 {
            let token = sampler.sample(&mut logits).expect("sampling should succeed");
            // Top-k=5 on ascending logits: only the last 5 indices (195-199) are valid
            assert!(token >= 195, "expected token ≥ 195, got {token}");
        }
    }

    #[test]
    fn repetition_penalty_affects_sampling() {
        // Bug #2 fix verification: repetition_penalty should affect sampling
        // Test that a token in history with high penalty is not chosen
        
        // Setup: Create two identical samplers
        let params_base = SamplingParams {
            temperature: 0.0, // greedy for predictability
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 128,
        };

        // Sampler 1: No penalty
        let mut sampler_no_penalty = Sampler::new(params_base.clone(), 42);

        // Sampler 2: Will have penalty applied to token 5
        let mut sampler_with_penalty = Sampler::new(params_base.clone(), 42);
        
        // First call for sampler_with_penalty: Generate token 5 to add it to history
        let mut logits_first = vec![0.1f32; 100];
        logits_first[5] = 100.0; // Strongly prefer token 5
        let _ = sampler_with_penalty.sample(&mut logits_first).expect("first sample");

        // Now update penalty
        sampler_with_penalty.params.repetition_penalty = 100.0;

        // Second call: Both samplers see identical logits
        let mut logits_second = vec![0.1f32; 100];
        logits_second[5] = 100.0; // Token 5 is highest
        logits_second[50] = 50.0; // Token 50 is second highest

        let token_no_penalty = sampler_no_penalty.sample(&mut logits_second).expect("no penalty");
        let token_with_penalty = sampler_with_penalty.sample(&mut logits_second).expect("with penalty");

        // Without penalty, token 5 should be chosen (highest logit)
        assert_eq!(token_no_penalty, 5, "without penalty, token 5 should win");
        
        // With penalty, token 5's logit becomes 100.0 / 100.0 = 1.0
        // Token 50's logit stays 50.0, so token 50 should win
        assert_eq!(token_with_penalty, 50, "with penalty, token 5 should be suppressed");
    }

    #[test]
    fn sampler_reset_clears_history() {
        let params = SamplingParams::default();
        let mut sampler = Sampler::new(params.clone(), 42);

        // Generate some tokens
        let mut logits = vec![1.0f32; 100];
        logits[10] = 10.0;
        logits[20] = 20.0;
        let _ = sampler.sample(&mut logits);
        let _ = sampler.sample(&mut logits);

        // Verify history has tokens
        assert!(!sampler.generated_tokens.is_empty());

        // Reset
        sampler.reset();

        // Verify history is cleared
        assert!(sampler.generated_tokens.is_empty());
    }
}
