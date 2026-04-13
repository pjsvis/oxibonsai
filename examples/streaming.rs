//! Streaming token generation demonstration.
//!
//! Shows how to use `generate_streaming` to receive tokens one at a
//! time through a Tokio MPSC channel, allowing real-time output as
//! the model generates.

use std::sync::Arc;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::presets::SamplingPreset;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== OxiBonsai Streaming Demo ===\n");

    // ─── Setup ───────────────────────────────────────────────────
    let config = Qwen3Config::tiny_test();
    let params = SamplingPreset::Conversational.params();
    let mut engine = InferenceEngine::new(config, params, 12345);

    let metrics = Arc::new(InferenceMetrics::new());
    engine.set_metrics(Arc::clone(&metrics));

    // Dummy prompt tokens
    let prompt_tokens: Vec<u32> = vec![151644, 1820, 2660, 315, 4298];
    let max_tokens: usize = 15;

    println!(
        "Prompt: {} tokens, generating up to {} tokens",
        prompt_tokens.len(),
        max_tokens,
    );

    // ─── Streaming generation ────────────────────────────────────
    // Create an unbounded channel for streaming tokens.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<u32>();

    // Spawn the generation on a blocking thread because the engine
    // uses synchronous computation internally.
    let prompt = prompt_tokens.clone();
    let handle =
        tokio::task::spawn_blocking(move || engine.generate_streaming(&prompt, max_tokens, &tx));

    // ─── Process tokens as they arrive ───────────────────────────
    println!("\nStreaming tokens:");
    let mut received_count: usize = 0;
    while let Some(token_id) = rx.recv().await {
        received_count += 1;
        print!("[{}] ", token_id);
    }
    println!();

    // Wait for generation to finish
    let total_generated = handle.await??;

    // ─── Summary ─────────────────────────────────────────────────
    println!("\n--- Streaming Summary ---");
    println!("Tokens received via channel: {}", received_count);
    println!("Total generated (returned):  {}", total_generated);
    println!(
        "Metrics counter:             {}",
        metrics.tokens_generated_total.get()
    );

    println!("\nDone.");
    Ok(())
}
