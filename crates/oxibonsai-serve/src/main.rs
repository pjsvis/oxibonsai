//! OxiBonsai server binary.
//!
//! Usage:
//!   oxibonsai-serve [OPTIONS]
//!
//! Options:
//!   --host <HOST>         Bind host (default: 0.0.0.0)
//!   --port <PORT>         Bind port (default: 8080)
//!   --model <PATH>        Path to GGUF model file
//!   --tokenizer <PATH>    Path to tokenizer (optional)
//!   --max-tokens <N>      Default max tokens (default: 256)
//!   --temperature <F>     Default temperature (default: 0.7)
//!   --seed <N>            RNG seed (default: 42)
//!   --log-level <LEVEL>   Logging level: error/warn/info/debug/trace (default: info)
//!   --help                Show this help
//!   --version             Show version

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{create_router, serve_with_shutdown, shutdown_signal};
use oxibonsai_serve::{args::parse_args_from, banner};
use std::net::SocketAddr;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // ── 1. Parse command-line arguments ──────────────────────────────────
    let argv: Vec<String> = std::env::args().collect();
    let server_args = match parse_args_from(&argv)? {
        Some(a) => a,
        // --help or --version was printed; exit cleanly.
        None => return Ok(()),
    };

    // ── 2. Set up tracing ─────────────────────────────────────────────────
    let filter = tracing_subscriber::EnvFilter::try_new(&server_args.log_level)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();

    // ── 3. Print banner ───────────────────────────────────────────────────
    banner::print_banner();
    info!(
        "{}",
        banner::startup_message(&server_args.host, server_args.port)
    );

    // ── 4. Build inference engine ─────────────────────────────────────────
    //
    // When a GGUF model path is provided we would normally memory-map and load
    // it.  That loader is not yet wired through the public API, so we fall back
    // to the tiny_test configuration for now and log a warning when a path was
    // supplied but cannot be used.
    if let Some(ref path) = server_args.model_path {
        tracing::warn!(
            model_path = %path,
            "GGUF loading is not yet implemented; using tiny_test engine"
        );
    }

    let config = Qwen3Config::tiny_test();
    let sampling = SamplingParams {
        temperature: server_args.temperature,
        ..SamplingParams::default()
    };
    // max_tokens is forwarded to the server via the per-request field in the
    // chat completion handler; SamplingParams itself does not carry it.
    let _ = server_args.max_tokens; // acknowledged, used at request time

    let engine = InferenceEngine::new(config, sampling, server_args.seed);

    // ── 5. Build router ───────────────────────────────────────────────────
    //
    // The tokenizer bridge is optional; if no tokenizer path was given the
    // server falls back to raw token-ID arrays in responses.
    let tokenizer = None; // tokenizer loading is feature-gated in the runtime crate

    let router = create_router(engine, tokenizer);

    // ── 6. Resolve bind address ───────────────────────────────────────────
    let addr_str = format!("{}:{}", server_args.host, server_args.port);
    let addr: SocketAddr = addr_str
        .parse()
        .map_err(|e| format!("invalid bind address '{}': {}", addr_str, e))?;

    info!(%addr, "starting listener");

    // ── 7. Serve with graceful shutdown ───────────────────────────────────
    serve_with_shutdown(router, addr, shutdown_signal()).await?;

    info!("oxibonsai-serve exited cleanly");
    Ok(())
}
