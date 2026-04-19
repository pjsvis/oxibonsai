//! OxiBonsai server binary.
//!
//! Layered configuration loader (`defaults < TOML < env < CLI`) followed by
//! model / tokenizer wiring and an Axum-based HTTP server with OpenAI-style
//! chat-completion endpoints.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{create_router, serve_with_shutdown, shutdown_signal};
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;
use oxibonsai_serve::{
    args::parse_args_from,
    banner,
    config::{PartialServerConfig, ServerConfig},
    env::parse_process_env,
};
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            error!(%err, "oxibonsai-serve startup failed");
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // ── 1. Parse command-line arguments ──────────────────────────────────
    let argv: Vec<String> = std::env::args().collect();
    let cli_args = match parse_args_from(&argv)? {
        Some(a) => a,
        // --help or --version was printed; exit cleanly.
        None => return Ok(()),
    };

    // ── 2. Install a *temporary* tracing subscriber so config / env / CLI
    //       parsing errors show up cleanly.  It will be replaced once the
    //       final `log_level` is known.
    let bootstrap_filter = tracing_subscriber::EnvFilter::try_new(&cli_args.log_level)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(bootstrap_filter)
        .with_target(false)
        .compact()
        .try_init();

    // ── 3. Load layered configuration ────────────────────────────────────
    let toml_path: Option<PathBuf> = cli_args.config_path.as_ref().map(PathBuf::from);
    let env_partial = parse_process_env()?;
    let cli_partial: PartialServerConfig = cli_args.to_partial();

    let config = ServerConfig::load(toml_path.as_deref(), Some(env_partial), Some(cli_partial))?;

    // ── 4. Print banner ───────────────────────────────────────────────────
    banner::print_banner();
    info!(
        "{}",
        banner::startup_message(&config.bind.host, config.bind.port)
    );

    // ── 5. Build inference engine ─────────────────────────────────────────
    //
    // If a GGUF model path is configured we load it eagerly via
    // `InferenceEngine::from_gguf_path`.  Any failure is *fatal* — the
    // operator asked for a specific model, so silently falling back to a
    // tiny test config would be misleading.
    let sampling = SamplingParams {
        temperature: config.sampling.default_temperature,
        top_p: config.sampling.default_top_p,
        ..SamplingParams::default()
    };

    let engine: InferenceEngine<'static> = match config.model.path.as_ref() {
        Some(path) => {
            info!(path = %path.display(), "loading GGUF model");
            match InferenceEngine::from_gguf_path(
                path,
                sampling.clone(),
                config.seed,
                config.limits.max_input_tokens,
            ) {
                Ok(e) => {
                    info!("GGUF model loaded");
                    e
                }
                Err(err) => {
                    error!(
                        path = %path.display(),
                        %err,
                        "failed to load GGUF model"
                    );
                    return Err(format!("failed to load GGUF model: {err}").into());
                }
            }
        }
        None => {
            warn!("no --model path supplied; falling back to tiny_test engine");
            let tiny = Qwen3Config::tiny_test();
            InferenceEngine::new(tiny, sampling, config.seed)
        }
    };

    // ── 6. Load tokenizer (optional) ──────────────────────────────────────
    let tokenizer = match config.tokenizer.path.as_ref() {
        Some(path) => match TokenizerBridge::from_file(&path.display().to_string()) {
            Ok(t) => {
                info!(path = %path.display(), "tokenizer loaded");
                Some(t)
            }
            Err(err) => {
                error!(path = %path.display(), %err, "failed to load tokenizer");
                return Err(format!("failed to load tokenizer: {err}").into());
            }
        },
        None => {
            info!("no tokenizer path supplied; server will fall back to raw token-IDs");
            None
        }
    };

    // ── 7. Build router (with optional bearer auth) ───────────────────────
    let base_router = create_router(engine, tokenizer);
    let router = if let Some(ref token) = config.auth.bearer_token {
        let state = middleware::BearerAuthState {
            token: token.clone(),
        };
        info!("bearer-token authentication enabled");
        base_router.layer(axum::middleware::from_fn_with_state(
            state,
            middleware::bearer_auth,
        ))
    } else {
        base_router
    };

    // ── 8. Resolve bind address ───────────────────────────────────────────
    let addr_str = format!("{}:{}", config.bind.host, config.bind.port);
    let addr: SocketAddr = addr_str
        .parse()
        .map_err(|e| format!("invalid bind address '{addr_str}': {e}"))?;

    info!(%addr, "starting listener");

    // ── 9. Serve with graceful shutdown ───────────────────────────────────
    serve_with_shutdown(router, addr, shutdown_signal()).await?;

    info!("oxibonsai-serve exited cleanly");
    Ok(())
}

/// Bearer-auth middleware.
///
/// Kept inline here (rather than in `oxibonsai-runtime`) because auth is a
/// deployment concern of the server binary, not the inference core.
mod middleware {
    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{header, Request, StatusCode};
    use axum::middleware::Next;
    use axum::response::{IntoResponse, Response};
    use axum::Json;

    /// State shared by the bearer-auth middleware.
    #[derive(Debug, Clone)]
    pub struct BearerAuthState {
        /// The expected token.  Any request that does not present exactly this
        /// token in `Authorization: Bearer <token>` is rejected with 401.
        pub token: String,
    }

    /// `axum::middleware::from_fn_with_state` handler.
    pub async fn bearer_auth(
        State(state): State<BearerAuthState>,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        // Allow `/health` and `/metrics` through unauthenticated — they are
        // needed for load balancers and Prometheus scrapers.
        let path = req.uri().path();
        if path == "/health" || path == "/metrics" {
            return next.run(req).await;
        }

        let header_value = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());

        let presented = match header_value.and_then(|h| h.strip_prefix("Bearer ")) {
            Some(tok) => tok.trim(),
            None => {
                return unauthorized("missing or malformed Authorization header").into_response();
            }
        };

        if presented != state.token {
            return unauthorized("invalid bearer token").into_response();
        }

        next.run(req).await
    }

    fn unauthorized(msg: &str) -> (StatusCode, Json<serde_json::Value>) {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": {
                    "message": msg,
                    "type": "auth_error",
                    "param": null,
                    "code": null,
                }
            })),
        )
    }
}
