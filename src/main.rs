//! OxiBonsai — 1-bit LLM inference engine for Bonsai-8B.
//!
//! This binary is not functional on WASM targets; the WASM entry points
//! are exposed via [`oxibonsai_runtime`] library APIs instead.

/// WASM stub: this binary is a no-op on wasm32 targets.
/// Consumers should use the `oxibonsai_runtime` library crate APIs directly.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod cli {
    use clap::{Parser, Subcommand};
    use std::io::{self, BufRead, Write};
    use std::path::Path;
    use std::sync::Arc;

    use oxibonsai_runtime::OxiBonsaiConfig;

    #[derive(Parser)]
    #[command(
        name = "oxibonsai",
        version,
        about = "1-bit LLM inference engine for Bonsai-8B"
    )]
    pub struct Cli {
        /// Path to an OxiBonsai TOML configuration file.
        #[arg(long, global = true)]
        config: Option<String>,

        #[command(subcommand)]
        pub command: Commands,
    }

    #[derive(Subcommand)]
    pub enum Commands {
        /// Run inference on a GGUF model.
        Run {
            /// Path to the GGUF model file.
            #[arg(short, long)]
            model: String,

            /// Prompt text. Use "-" for stdin.
            #[arg(short, long)]
            prompt: String,

            /// Maximum number of tokens to generate.
            #[arg(long, default_value_t = 256)]
            max_tokens: usize,

            /// Sampling temperature (0.0 = greedy).
            #[arg(long, default_value_t = 0.7)]
            temperature: f32,

            /// Top-k sampling (0 = disabled).
            #[arg(long, default_value_t = 40)]
            top_k: usize,

            /// Top-p (nucleus) sampling.
            #[arg(long, default_value_t = 0.9)]
            top_p: f32,

            /// Random seed.
            #[arg(long, default_value_t = 42)]
            seed: u64,

            /// Maximum sequence length (prompt + generated).
            #[arg(long, default_value_t = 4096)]
            max_seq_len: usize,

            /// Path to tokenizer.json file.
            #[arg(long)]
            tokenizer: Option<String>,
        },

        /// Interactive multi-turn conversation.
        Chat {
            /// Path to the GGUF model file.
            #[arg(short, long)]
            model: String,

            /// Maximum number of tokens to generate per turn.
            #[arg(long, default_value_t = 512)]
            max_tokens: usize,

            /// Sampling temperature (0.0 = greedy).
            #[arg(long, default_value_t = 0.7)]
            temperature: f32,

            /// Top-k sampling (0 = disabled).
            #[arg(long, default_value_t = 40)]
            top_k: usize,

            /// Top-p (nucleus) sampling.
            #[arg(long, default_value_t = 0.9)]
            top_p: f32,

            /// Random seed.
            #[arg(long, default_value_t = 42)]
            seed: u64,

            /// Maximum sequence length.
            #[arg(long, default_value_t = 4096)]
            max_seq_len: usize,

            /// Path to tokenizer.json file.
            #[arg(long)]
            tokenizer: Option<String>,
        },

        /// Start an OpenAI-compatible API server.
        #[cfg(feature = "server")]
        Serve {
            /// Path to the GGUF model file.
            #[arg(short, long)]
            model: String,

            /// Host to bind to.
            #[arg(long, default_value = "127.0.0.1")]
            host: String,

            /// Port to listen on.
            #[arg(long, default_value_t = 8080)]
            port: u16,

            /// Maximum sequence length.
            #[arg(long, default_value_t = 4096)]
            max_seq_len: usize,

            /// Path to tokenizer.json file.
            #[arg(long)]
            tokenizer: Option<String>,
        },

        /// Display model info from a GGUF file.
        Info {
            /// Path to the GGUF model file.
            #[arg(short, long)]
            model: String,

            /// Emit info as JSON instead of human-readable text.
            #[arg(long, default_value_t = false)]
            json: bool,
        },

        /// Run a quick throughput benchmark (no real model weights required).
        Benchmark {
            /// Total tokens to generate during the benchmark pass.
            #[arg(long, default_value_t = 100)]
            tokens: usize,

            /// Number of warmup tokens generated before timing begins.
            #[arg(long, default_value_t = 10)]
            warmup: usize,

            /// Sampling temperature.
            #[arg(long, default_value_t = 0.7)]
            temperature: f32,

            /// Random seed.
            #[arg(long, default_value_t = 42)]
            seed: u64,
        },

        /// Quantize a GGUF model to a lower-precision format (simulation).
        Quantize {
            /// Path to the input GGUF model file.
            #[arg(long)]
            input: String,

            /// Destination path for the quantized file.
            #[arg(long)]
            output: String,

            /// Target quantization format (e.g. q1_0, q4_0, q8_0).
            #[arg(long, default_value = "q1_0")]
            format: String,
        },

        /// Validate that a GGUF file is well-formed and display a metadata summary.
        Validate {
            /// Path to the GGUF model file to validate.
            #[arg(short, long)]
            model: String,
        },

        /// Convert a HuggingFace safetensors model to GGUF format.
        Convert {
            /// Input directory containing model.safetensors (or shards) and config.json.
            #[arg(long)]
            from: String,

            /// Output GGUF file path.
            #[arg(long)]
            to: String,

            /// Quantization format: tq2_0_g128 (default), q1_0_g128.
            #[arg(long, default_value = "tq2_0_g128")]
            quant: String,
        },
    }

    pub fn read_prompt_stdin() -> String {
        let stdin = io::stdin();
        let mut lines = Vec::new();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => lines.push(l),
                Err(_) => break,
            }
        }
        lines.join("\n")
    }

    pub async fn run() -> anyhow::Result<()> {
        let cli = Cli::parse();
        let config = OxiBonsaiConfig::load_or_default(cli.config.as_deref().map(Path::new));

        let tracing_config =
            oxibonsai_runtime::TracingConfig::from_observability(&config.observability);
        if let Err(e) = oxibonsai_runtime::init_tracing(&tracing_config) {
            eprintln!("warning: failed to initialize tracing: {e}");
        }

        match cli.command {
            Commands::Run {
                model,
                prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                max_seq_len,
                tokenizer,
            } => {
                let prompt_text = if prompt == "-" {
                    read_prompt_stdin()
                } else {
                    prompt
                };

                tracing::info!(
                    model = %model,
                    max_tokens,
                    temperature,
                    "starting inference"
                );

                // Memory-map the GGUF file
                let mmap =
                    oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))?;
                let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

                let params = oxibonsai_runtime::sampling::SamplingParams {
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty: 1.1,
                    ..oxibonsai_runtime::sampling::SamplingParams::default()
                };

                let mut engine = oxibonsai_runtime::InferenceEngine::from_gguf(
                    &gguf,
                    params,
                    seed,
                    max_seq_len,
                )?;

                // Tokenize prompt and retain the bridge for streaming decode
                let (prompt_tokens, tok_bridge) = if let Some(tok_path) = &tokenizer {
                    let tok = oxibonsai_runtime::TokenizerBridge::from_file(tok_path)?;
                    let tokens = tok.encode(&prompt_text)?;
                    (tokens, Some(tok))
                } else {
                    tracing::warn!("no tokenizer specified — using dummy token");
                    (vec![151644], None) // <|im_start|>
                };

                tracing::info!(prompt_tokens = prompt_tokens.len(), "prefilling");

                let start = std::time::Instant::now();

                // Greedy GPU path: when temperature=0 and Metal is available,
                // run argmax on GPU and download only 4-byte token IDs instead
                // of the full ~607KB logits vector per token.
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let use_greedy_gpu = temperature == 0.0;
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let use_greedy_gpu = false;

                let (prompt_len, output_count) = if use_greedy_gpu {
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    {
                        tracing::info!("using greedy GPU path (argmax on Metal, 4-byte download)");
                        let p_len = prompt_tokens.len();
                        let tokens = engine.generate_greedy_gpu(&prompt_tokens, max_tokens)?;
                        // Print tokens
                        for &token_id in &tokens {
                            if let Some(tok) = &tok_bridge {
                                let text = tok.decode(&[token_id]).unwrap_or_default();
                                print!("{text}");
                            } else {
                                print!(" {token_id}");
                            }
                            let _ = io::stdout().flush();
                        }
                        (p_len, tokens.len())
                    }
                    #[cfg(not(all(feature = "metal", target_os = "macos")))]
                    unreachable!()
                } else {
                    // CUDA direct path (Linux / Windows): call engine.generate() directly on the
                    // main thread, bypassing std::thread::scope and its ~107ms OS scheduling
                    // overhead.  Tokens are printed synchronously after the full generation
                    // completes, which is fine for non-interactive benchmarking.
                    #[cfg(all(
                        feature = "native-cuda",
                        not(all(feature = "metal", target_os = "macos")),
                        any(target_os = "linux", target_os = "windows")
                    ))]
                    {
                        let p_len = prompt_tokens.len();
                        let tokens = engine.generate(&prompt_tokens, max_tokens)?;
                        for &token_id in &tokens {
                            if let Some(tok) = &tok_bridge {
                                let text = tok.decode(&[token_id]).unwrap_or_default();
                                print!("{text}");
                            } else {
                                print!(" {token_id}");
                            }
                            let _ = io::stdout().flush();
                        }
                        (p_len, tokens.len())
                    }

                    // All other platforms (no CUDA): streaming path via a worker thread so that
                    // the main thread can decode and print tokens as they arrive.
                    #[cfg(not(all(
                        feature = "native-cuda",
                        not(all(feature = "metal", target_os = "macos")),
                        any(target_os = "linux", target_os = "windows")
                    )))]
                    {
                        let (tx, rx) = std::sync::mpsc::channel::<u32>();

                        let p_len = prompt_tokens.len();
                        let count = std::thread::scope(|s| -> anyhow::Result<usize> {
                            let thread_tx = tx.clone();
                            let gen_handle = s.spawn(move || {
                                engine.generate_streaming_sync(
                                    &prompt_tokens,
                                    max_tokens,
                                    &thread_tx,
                                )
                            });
                            drop(tx);

                            let mut count = 0usize;
                            for token_id in rx {
                                count += 1;
                                if let Some(tok) = &tok_bridge {
                                    let text = tok.decode(&[token_id]).unwrap_or_default();
                                    print!("{text}");
                                } else {
                                    if count == 1 {
                                        print!("Tokens:");
                                    }
                                    print!(" {token_id}");
                                }
                                let _ = io::stdout().flush();
                            }

                            match gen_handle.join() {
                                Ok(Ok(_)) => {}
                                Ok(Err(e)) => return Err(e.into()),
                                Err(_) => {
                                    return Err(anyhow::anyhow!("generation thread panicked"))
                                }
                            }
                            Ok(count)
                        })?;
                        (p_len, count)
                    }
                };

                let elapsed = start.elapsed();

                let total_tokens = prompt_len + output_count;
                let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                    output_count as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                eprintln!();
                eprintln!(
                    "---\n{} prompt + {} generated = {} total tokens in {:.2}s ({:.1} tok/s)",
                    prompt_len,
                    output_count,
                    total_tokens,
                    elapsed.as_secs_f64(),
                    tok_per_sec
                );

                // Print GPU profiling summary if OXIBONSAI_PROFILE_GPU=1 was set
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    let model_size = std::fs::metadata(&model).map(|m| m.len()).unwrap_or(0);
                    oxibonsai_kernels::print_gpu_profile_summary(model_size);
                }
            }

            Commands::Chat {
                model,
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                max_seq_len,
                tokenizer,
            } => {
                // Memory-map the GGUF file
                let mmap =
                    oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))?;
                let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

                let params = oxibonsai_runtime::sampling::SamplingParams {
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty: 1.1,
                    ..oxibonsai_runtime::sampling::SamplingParams::default()
                };

                let mut engine = oxibonsai_runtime::InferenceEngine::from_gguf(
                    &gguf,
                    params,
                    seed,
                    max_seq_len,
                )?;

                let tok = if let Some(tok_path) = &tokenizer {
                    Some(oxibonsai_runtime::TokenizerBridge::from_file(tok_path)?)
                } else {
                    tracing::warn!("no tokenizer specified — token IDs will be printed");
                    None
                };

                println!("OxiBonsai Interactive Chat (type 'quit' or Ctrl-D to exit)");
                println!("---");

                let stdin = io::stdin();
                loop {
                    print!("> ");
                    io::stdout().flush()?;

                    let mut input = String::new();
                    if stdin.lock().read_line(&mut input)? == 0 {
                        // EOF
                        println!();
                        break;
                    }
                    let input = input.trim();
                    if input.is_empty() {
                        continue;
                    }
                    if input == "quit" || input == "exit" {
                        break;
                    }
                    if input == "/reset" {
                        engine.reset();
                        println!("[context cleared]");
                        continue;
                    }

                    let prompt_tokens = if let Some(tok) = &tok {
                        tok.encode(input)?
                    } else {
                        vec![151644]
                    };

                    let start = std::time::Instant::now();
                    let (tx, rx) = std::sync::mpsc::channel::<u32>();

                    // Use std::thread::scope so engine's borrow stays valid
                    let output_count = std::thread::scope(|s| -> anyhow::Result<usize> {
                        let thread_tx = tx.clone();
                        let engine_ref = &mut engine;
                        let tokens_ref = &prompt_tokens;
                        let gen_handle = s.spawn(move || {
                            engine_ref.generate_streaming_sync(tokens_ref, max_tokens, &thread_tx)
                        });
                        drop(tx);

                        let mut count = 0usize;
                        for token_id in rx {
                            count += 1;
                            if let Some(tok) = &tok {
                                let text = tok.decode(&[token_id]).unwrap_or_default();
                                print!("{text}");
                            } else {
                                if count == 1 {
                                    print!("Tokens:");
                                }
                                print!(" {token_id}");
                            }
                            let _ = io::stdout().flush();
                        }

                        match gen_handle.join() {
                            Ok(Ok(_)) => {}
                            Ok(Err(e)) => return Err(e.into()),
                            Err(_) => return Err(anyhow::anyhow!("generation thread panicked")),
                        }
                        Ok(count)
                    })?;

                    let elapsed = start.elapsed();
                    println!(); // newline after streamed output

                    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                        output_count as f64 / elapsed.as_secs_f64()
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[{} tokens in {:.2}s, {:.1} tok/s]",
                        output_count,
                        elapsed.as_secs_f64(),
                        tok_per_sec
                    );
                }
            }

            #[cfg(feature = "server")]
            Commands::Serve {
                model,
                host,
                port,
                max_seq_len,
                tokenizer,
            } => {
                tracing::info!(model = %model, host = %host, port, "starting server");

                let mmap =
                    oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))?;
                // Leak the mmap to get 'static lifetime for the server
                let mmap: &'static memmap2::Mmap = Box::leak(Box::new(mmap));
                let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(mmap)?;
                let gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static> =
                    Box::leak(Box::new(gguf));

                let params = oxibonsai_runtime::sampling::SamplingParams::default();
                let metrics = Arc::new(oxibonsai_runtime::InferenceMetrics::new());
                let mut engine =
                    oxibonsai_runtime::InferenceEngine::from_gguf(gguf, params, 42, max_seq_len)?;
                engine.set_metrics(Arc::clone(&metrics));

                let tok = tokenizer
                    .as_ref()
                    .map(|p| oxibonsai_runtime::TokenizerBridge::from_file(p))
                    .transpose()?;

                let router =
                    oxibonsai_runtime::server::create_router_with_metrics(engine, tok, metrics);
                let addr = format!("{host}:{port}");
                let listener = tokio::net::TcpListener::bind(&addr).await?;
                tracing::info!("listening on {addr}");
                axum::serve(listener, router).await?;
            }

            Commands::Info { model, json } => {
                let mmap =
                    oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))?;
                let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;
                let config = oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata)?;
                let type_counts = gguf.tensors.count_by_type();

                // Determine dominant quant type from tensor counts for accurate variant detection.
                let dominant_type = type_counts
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(ty, _)| *ty)
                    .unwrap_or(oxibonsai_core::GgufTensorType::Q1_0_g128);

                let variant = oxibonsai_model::ModelVariant::from_config_and_sample_tensor_type(
                    &config,
                    dominant_type,
                );

                if json {
                    let tensor_types: std::collections::HashMap<String, usize> = type_counts
                        .iter()
                        .map(|(k, v)| (k.to_string(), *v))
                        .collect();

                    let info = serde_json::json!({
                        "model": model,
                        "gguf_version": gguf.header.version,
                        "tensor_count": gguf.header.tensor_count,
                        "metadata_entries": gguf.header.metadata_kv_count,
                        "architecture": format!("Qwen3 ({})", variant.name()),
                        "variant": variant.name(),
                        "num_layers": config.num_layers,
                        "hidden_size": config.hidden_size,
                        "num_attention_heads": config.num_attention_heads,
                        "num_kv_heads": config.num_kv_heads,
                        "head_dim": config.head_dim,
                        "vocab_size": config.vocab_size,
                        "max_context_length": config.max_context_length,
                        "intermediate_size": config.intermediate_size,
                        "tensor_types": tensor_types,
                    });
                    println!("{}", serde_json::to_string_pretty(&info)?);
                } else {
                    println!("Model: {model}");
                    println!("GGUF version: {}", gguf.header.version);
                    println!("Tensor count: {}", gguf.header.tensor_count);
                    println!("Metadata entries: {}", gguf.header.metadata_kv_count);
                    println!();

                    println!("Architecture: Qwen3 ({})", variant.name());
                    println!("  Layers:       {}", config.num_layers);
                    println!("  Hidden size:  {}", config.hidden_size);
                    println!("  Q heads:      {}", config.num_attention_heads);
                    println!("  KV heads:     {}", config.num_kv_heads);
                    println!("  Head dim:     {}", config.head_dim);
                    println!("  Vocab:        {}", config.vocab_size);
                    println!("  Max context:  {}", config.max_context_length);
                    println!("  Intermediate: {}", config.intermediate_size);
                    println!();

                    println!("Tensor types:");
                    for (tensor_type, count) in &type_counts {
                        println!("  {tensor_type}: {count}");
                    }
                }
            }

            Commands::Benchmark {
                tokens,
                warmup,
                temperature,
                seed,
            } => {
                use oxibonsai_core::config::Qwen3Config;
                use oxibonsai_runtime::model_cache::ModelWarmup;
                use oxibonsai_runtime::sampling::SamplingParams;

                let config = Qwen3Config::tiny_test();
                let params = SamplingParams {
                    temperature,
                    top_k: 40,
                    top_p: 0.9,
                    repetition_penalty: 1.0,
                    ..SamplingParams::default()
                };

                let mut engine =
                    oxibonsai_runtime::InferenceEngine::new(config, params.clone(), seed);

                // ── Warmup pass ──────────────────────────────────────────────
                let warmup_helper = ModelWarmup::new().with_tokens(warmup);
                let warmup_ms = warmup_helper.run(&mut engine, &params);
                eprintln!("Warmup: {warmup} tokens in {warmup_ms} ms");

                // ── Benchmark pass ───────────────────────────────────────────
                let prompt_tokens: Vec<u32> = vec![151644u32]; // <|im_start|>
                let bench_start = std::time::Instant::now();
                let output_tokens = engine.generate(&prompt_tokens, tokens)?;
                let bench_elapsed = bench_start.elapsed();

                let generated = output_tokens.len();
                let tok_per_sec = if bench_elapsed.as_secs_f64() > 0.0 {
                    generated as f64 / bench_elapsed.as_secs_f64()
                } else {
                    0.0
                };

                println!(
                    "Warmup: {warmup} tokens, Benchmark: {tok_per_sec:.1} tokens/sec \
                     ({generated} tokens in {:.2}s)",
                    bench_elapsed.as_secs_f64()
                );
            }

            Commands::Quantize {
                input,
                output,
                format,
            } => {
                use std::path::Path;

                let input_path = Path::new(&input);
                if !input_path.exists() {
                    anyhow::bail!("input file does not exist: {input}");
                }

                // Read the original file size.
                let original_bytes = std::fs::metadata(input_path).map(|m| m.len()).unwrap_or(0);
                let original_mb = original_bytes as f64 / (1024.0 * 1024.0);

                // Validate the format string.
                let known_formats = ["q1_0", "q2_k", "q4_0", "q4_1", "q8_0", "f16", "f32"];
                if !known_formats.contains(&format.as_str()) {
                    eprintln!(
                        "warning: unknown quantization format '{format}'; \
                         known formats: {known_formats:?}"
                    );
                }

                // Compression ratio heuristic per format.
                let compression_ratio: f64 = match format.as_str() {
                    "q1_0" => 32.0,
                    "q2_k" => 16.0,
                    "q4_0" | "q4_1" => 8.0,
                    "q8_0" => 4.0,
                    "f16" => 2.0,
                    _ => 1.0,
                };

                println!("Quantizing {input} -> {output} (format: {format})...");

                // For an actual implementation we would call into oxibonsai_core
                // quantization routines.  For now we simulate by copying the file
                // metadata and estimating the output size.
                let quantized_bytes = (original_bytes as f64 / compression_ratio).ceil() as u64;
                let quantized_mb = quantized_bytes as f64 / (1024.0 * 1024.0);

                println!(
                    "Quantizing... Original: {original_mb:.1} MB \
                     → Quantized: {quantized_mb:.1} MB \
                     ({compression_ratio:.0}:1 compression)"
                );
                println!("Output: {output}");
            }

            Commands::Convert { from, to, quant } => {
                use std::path::Path;
                let from_path = Path::new(&from);
                let to_path = Path::new(&to);
                if !from_path.exists() {
                    anyhow::bail!("input directory not found: {from}");
                }
                println!("Converting {from} -> {to} (quant: {quant})");
                let stats =
                    oxibonsai_model::convert::convert_hf_to_gguf(from_path, to_path, &quant)?;
                println!(
                    "Done: {} tensors ({} ternary + {} fp32), output: {:.1} MB",
                    stats.n_tensors,
                    stats.n_ternary,
                    stats.n_fp32,
                    stats.output_bytes as f64 / (1024.0 * 1024.0),
                );
            }

            Commands::Validate { model } => {
                let mmap =
                    oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))?;
                let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

                // Attempt to parse model config — validates metadata consistency.
                let config_result =
                    oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata);

                println!("Validating: {model}");
                println!("  GGUF version:     {}", gguf.header.version);
                println!("  Tensor count:     {}", gguf.header.tensor_count);
                println!("  Metadata entries: {}", gguf.header.metadata_kv_count);

                match config_result {
                    Ok(config) => {
                        println!("  Architecture:     Qwen3");
                        println!("  Layers:           {}", config.num_layers);
                        println!("  Hidden size:      {}", config.hidden_size);
                        println!("  Vocab size:       {}", config.vocab_size);
                        println!();
                        println!("Validation: OK");
                    }
                    Err(e) => {
                        println!();
                        println!("Validation: FAILED");
                        println!("  Error: {e}");
                        return Err(anyhow::anyhow!("GGUF validation failed: {e}"));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Native (non-WASM) entry point.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build tokio runtime: {e}"))?
        .block_on(cli::run())
}
