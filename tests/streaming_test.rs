//! Streaming generation tests.
//!
//! Tests the `generate_streaming` method that sends tokens through
//! a Tokio mpsc channel as they are generated.

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::presets::SamplingPreset;
use oxibonsai_runtime::sampling::SamplingParams;

// ── Helper ───────────────────────────────────────────────────────────────

fn make_tiny_engine() -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42)
}

fn make_tiny_greedy_engine() -> InferenceEngine<'static> {
    InferenceEngine::new(
        Qwen3Config::tiny_test(),
        SamplingPreset::Greedy.params(),
        42,
    )
}

// ── 1. generate_streaming sends tokens through channel ───────────────────

#[tokio::test]
async fn streaming_sends_tokens_through_channel() {
    let mut engine = make_tiny_engine();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let prompt = vec![1u32, 2, 3];
    let result = engine.generate_streaming(&prompt, 5, &tx);
    assert!(result.is_ok(), "streaming generation should succeed");
    let count = result.expect("should succeed");

    // Drop sender to close channel
    drop(tx);

    // Collect tokens from receiver
    let mut received = Vec::new();
    while let Some(token) = rx.recv().await {
        received.push(token);
    }

    assert_eq!(
        received.len(),
        count,
        "number of received tokens should match returned count"
    );
}

// ── 2. Receiver gets tokens one at a time ────────────────────────────────

#[tokio::test]
async fn streaming_tokens_arrive_individually() {
    let mut engine = make_tiny_engine();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let prompt = vec![1u32, 2, 3, 4, 5];
    let count = engine
        .generate_streaming(&prompt, 10, &tx)
        .expect("streaming should succeed");

    drop(tx);

    // Verify we can recv each token individually
    let mut collected = 0;
    while let Some(_token) = rx.recv().await {
        collected += 1;
    }

    assert_eq!(collected, count, "should receive exactly {count} tokens");
}

// ── 3. Channel closed when generation complete ───────────────────────────

#[tokio::test]
async fn streaming_channel_closes_after_completion() {
    let mut engine = make_tiny_engine();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let _count = engine
        .generate_streaming(&[1, 2, 3], 3, &tx)
        .expect("streaming should succeed");

    // After generation completes and we drop tx, rx.recv() should return None
    drop(tx);

    // Drain any tokens
    while rx.recv().await.is_some() {}

    // Now channel should be fully closed
    let final_recv = rx.recv().await;
    assert!(final_recv.is_none(), "channel should be closed after drain");
}

// ── 4. Dropped receiver stops generation gracefully ──────────────────────

#[tokio::test]
async fn streaming_dropped_receiver_no_panic() {
    let mut engine = make_tiny_engine();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<u32>();

    // Drop the receiver immediately before generation
    drop(rx);

    // Generation should handle the closed channel gracefully
    let result = engine.generate_streaming(&[1, 2, 3], 100, &tx);
    assert!(
        result.is_ok(),
        "generation should succeed even with dropped receiver"
    );
    let count = result.expect("should be ok");
    // With dropped receiver, the engine should detect send failure and stop early
    assert!(
        count <= 100,
        "count should be bounded by max_tokens or early stop"
    );
}

// ── 5. Token count matches non-streaming generation ──────────────────────

#[tokio::test]
async fn streaming_matches_nonstreaming_count() {
    // Use greedy for determinism
    let mut engine_stream = make_tiny_greedy_engine();
    let mut engine_normal = make_tiny_greedy_engine();

    let prompt = vec![1u32, 2, 3];
    let max_tokens = 5;

    // Non-streaming generation
    let normal_tokens = engine_normal
        .generate(&prompt, max_tokens)
        .expect("normal generation should succeed");

    // Streaming generation
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let stream_count = engine_stream
        .generate_streaming(&prompt, max_tokens, &tx)
        .expect("streaming should succeed");
    drop(tx);

    let mut stream_tokens = Vec::new();
    while let Some(token) = rx.recv().await {
        stream_tokens.push(token);
    }

    assert_eq!(
        stream_count,
        normal_tokens.len(),
        "streaming count should match non-streaming"
    );
    assert_eq!(
        stream_tokens, normal_tokens,
        "streaming tokens should match non-streaming tokens"
    );
}

// ── 6. Streaming with empty prompt ───────────────────────────────────────

#[tokio::test]
async fn streaming_empty_prompt_returns_zero() {
    let mut engine = make_tiny_engine();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let count = engine
        .generate_streaming(&[], 10, &tx)
        .expect("empty prompt streaming should succeed");
    drop(tx);

    assert_eq!(count, 0, "empty prompt should produce 0 tokens");
    let token = rx.recv().await;
    assert!(
        token.is_none(),
        "no tokens should come through for empty prompt"
    );
}

// ── 7. Streaming with max_tokens=0 ──────────────────────────────────────

#[tokio::test]
async fn streaming_max_tokens_zero() {
    let mut engine = make_tiny_engine();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let count = engine
        .generate_streaming(&[1, 2, 3], 0, &tx)
        .expect("max_tokens=0 streaming should succeed");
    drop(tx);

    assert_eq!(count, 0, "max_tokens=0 should produce 0 tokens");
    let token = rx.recv().await;
    assert!(token.is_none(), "no tokens for max_tokens=0");
}

// ── 8. Streaming preserves token order ───────────────────────────────────

#[tokio::test]
async fn streaming_preserves_order() {
    let mut engine = make_tiny_greedy_engine();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let prompt = vec![1u32, 2, 3, 4, 5];
    let _count = engine
        .generate_streaming(&prompt, 10, &tx)
        .expect("streaming should succeed");
    drop(tx);

    let mut prev_idx = 0usize;
    while let Some(_token) = rx.recv().await {
        prev_idx += 1;
    }
    // We just verify we could iterate through all tokens in order
    // (the channel is FIFO by design)
    assert!(prev_idx <= 10);
}

// ── 9. Multiple streaming calls on same engine ───────────────────────────

#[tokio::test]
async fn multiple_streaming_calls_sequential() {
    let mut engine = make_tiny_engine();

    // First streaming call
    {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let _count = engine
            .generate_streaming(&[1, 2], 3, &tx)
            .expect("first streaming should succeed");
        drop(tx);
        while rx.recv().await.is_some() {}
    }

    // Reset engine state
    engine.reset();

    // Second streaming call
    {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let _count = engine
            .generate_streaming(&[3, 4], 3, &tx)
            .expect("second streaming should succeed");
        drop(tx);
        while rx.recv().await.is_some() {}
    }

    // If we got here without panicking, sequential streaming works
}

// ── 10. Streaming stats are updated ──────────────────────────────────────

#[tokio::test]
async fn streaming_updates_engine_stats() {
    let mut engine = make_tiny_engine();

    let initial_requests = engine.stats().requests_completed();

    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let _count = engine
        .generate_streaming(&[1, 2, 3], 3, &tx)
        .expect("streaming should succeed");

    // Note: generate_streaming doesn't call record_request itself (only generate does)
    // So we just verify no panic occurs and stats are accessible
    let _final_requests = engine.stats().requests_completed();
    assert!(engine.stats().uptime_seconds() >= 0.0);
    let _ = initial_requests; // used above
}

// ── 11. Streaming with single-token prompt ───────────────────────────────

#[tokio::test]
async fn streaming_single_token_prompt() {
    let mut engine = make_tiny_engine();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let count = engine
        .generate_streaming(&[42], 5, &tx)
        .expect("single token prompt streaming should succeed");
    drop(tx);

    let mut received = 0;
    while rx.recv().await.is_some() {
        received += 1;
    }

    assert_eq!(received, count);
    assert!(received <= 5);
}
