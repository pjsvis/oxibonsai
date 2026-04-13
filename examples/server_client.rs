//! Server setup demonstration.
//!
//! Shows how to programmatically create the OxiBonsai HTTP server
//! router, start it on a random port, and make requests against
//! the health and model listing endpoints.
//!
//! NOTE: This example starts a real HTTP server and connects to it.
//! It demonstrates the full server lifecycle without requiring a
//! real model file.

use std::sync::Arc;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_kernels::OneBitKernel;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::presets::SamplingPreset;
use oxibonsai_runtime::server::create_router_with_metrics;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== OxiBonsai Server Demo ===\n");

    // ─── 1. Create the engine ────────────────────────────────────
    let config = Qwen3Config::tiny_test();
    let params = SamplingPreset::Balanced.params();

    // SAFETY: InferenceEngine<'static> requires that the model data
    // outlives the engine. For the test config (no GGUF mmap), this
    // is trivially satisfied because all data is owned.
    let engine = InferenceEngine::new(config, params, 42);
    let metrics = Arc::new(InferenceMetrics::new());

    println!("Engine created with kernel: {}", engine.kernel().name());

    // ─── 2. Build the router ─────────────────────────────────────
    // The router provides these endpoints:
    //   POST /v1/chat/completions  — OpenAI-compatible chat
    //   GET  /v1/models            — List available models
    //   GET  /health               — Health check
    //   GET  /metrics              — Prometheus metrics
    let router = create_router_with_metrics(engine, None, Arc::clone(&metrics));

    println!("Router created with endpoints:");
    println!("  POST /v1/chat/completions");
    println!("  GET  /v1/models");
    println!("  GET  /health");
    println!("  GET  /metrics");

    // ─── 3. Start the server on a random port ────────────────────
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    println!("\nServer listening on http://{}", addr);

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, router)
            .await
            .expect("server should run");
    });

    // Give the server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // ─── 4. Make requests ────────────────────────────────────────
    println!("\n--- Making Requests ---\n");

    // Health check
    let health_url = format!("http://{}/health", addr);
    let resp = reqwest::get(&health_url).await?;
    println!("GET /health => {} {}", resp.status(), resp.text().await?);

    // List models
    let models_url = format!("http://{}/v1/models", addr);
    let resp = reqwest::get(&models_url).await?;
    let body: serde_json::Value = resp.json().await?;
    println!("GET /v1/models => {}", serde_json::to_string_pretty(&body)?);

    // Metrics
    let metrics_url = format!("http://{}/metrics", addr);
    let resp = reqwest::get(&metrics_url).await?;
    let metrics_text = resp.text().await?;
    println!("GET /metrics => ({} bytes)", metrics_text.len());
    // Show first few lines
    for line in metrics_text.lines().take(10) {
        println!("  {}", line);
    }
    if metrics_text.lines().count() > 10 {
        println!("  ... (truncated)");
    }

    // ─── 5. Shutdown ─────────────────────────────────────────────
    println!("\nShutting down server...");
    server_handle.abort();
    println!("Done.");

    Ok(())
}
