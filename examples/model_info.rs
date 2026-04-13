//! Model information display demonstration.
//!
//! Shows how to inspect model variants, parameter counts,
//! expected file sizes, and memory requirements.

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_model::model_registry::ModelVariant;
use oxibonsai_runtime::convenience::{
    estimate_memory_requirements, format_bytes, format_token_count,
};

fn main() -> anyhow::Result<()> {
    println!("=== OxiBonsai Model Information ===\n");

    // ─── 1. Known model variants ─────────────────────────────────
    println!("--- Known Model Variants ---\n");
    println!(
        "{:<15} {:>12} {:>12} {:>8} {:>6}",
        "Variant", "Parameters", "File Size", "Layers", "Hidden"
    );
    println!("{}", "-".repeat(60));

    for variant in ModelVariant::known_variants() {
        let config = variant.default_config();
        println!(
            "{:<15} {:>12} {:>12} {:>8} {:>6}",
            variant.name(),
            format_param_count(variant.param_count()),
            format_bytes(variant.expected_model_size_bytes()),
            config.num_layers,
            config.hidden_size,
        );
    }

    // ─── 2. Variant detection from config ────────────────────────
    println!("\n--- Variant Detection ---\n");

    let configs = [
        ("Bonsai-8B config", Qwen3Config::bonsai_8b()),
        ("Bonsai-4B config", Qwen3Config::bonsai_4b()),
        ("Bonsai-1.7B config", Qwen3Config::bonsai_1_7b()),
        ("Tiny test config", Qwen3Config::tiny_test()),
    ];

    for (label, config) in &configs {
        let detected = ModelVariant::from_config(config);
        println!(
            "{:<20} => {} (known={})",
            label,
            detected.name(),
            detected.is_known(),
        );
    }

    // ─── 3. Custom (unrecognized) configuration ──────────────────
    println!("\n--- Custom Configuration ---\n");

    let mut custom_config = Qwen3Config::bonsai_8b();
    custom_config.num_layers = 48;
    custom_config.hidden_size = 8192;
    custom_config.model_name = "Custom-70B".to_string();

    let variant = ModelVariant::from_config(&custom_config);
    println!(
        "Config: layers={}, hidden={}",
        custom_config.num_layers, custom_config.hidden_size
    );
    println!(
        "Detected variant: {} (known={})",
        variant.name(),
        variant.is_known()
    );

    // ─── 4. Detailed model info via BonsaiModel ──────────────────
    println!("\n--- Detailed Model Info (BonsaiModel) ---\n");

    let model = BonsaiModel::new(Qwen3Config::bonsai_8b());
    println!("Variant:           {}", model.variant());
    println!("Layers:            {}", model.num_layers());
    println!("Hidden size:       {}", model.hidden_size());
    println!(
        "Context length:    {}",
        format_token_count(model.context_length())
    );
    println!(
        "Parameters:        {}",
        format_param_count(model.num_parameters())
    );
    println!(
        "Model size (disk): {}",
        format_bytes(model.model_size_bytes())
    );
    println!(
        "KV cache memory:   {}",
        format_bytes(model.kv_cache_memory_bytes() as u64)
    );

    // ─── 5. Memory estimation for each variant ───────────────────
    println!("\n--- Memory Estimates (max_seq_len=4096) ---\n");
    println!(
        "{:<15} {:>12} {:>12} {:>12} {:>12}",
        "Variant", "Weights", "KV Cache", "Total", "Fits?"
    );
    println!("{}", "-".repeat(68));

    for variant in ModelVariant::known_variants() {
        let config = variant.default_config();
        let est = estimate_memory_requirements(
            variant.expected_model_size_bytes(),
            4096,
            config.num_kv_heads,
            config.head_dim,
            config.num_layers,
        );
        println!(
            "{:<15} {:>12} {:>12} {:>12} {:>12}",
            variant.name(),
            format_bytes(est.model_weights_bytes),
            format_bytes(est.kv_cache_bytes),
            format_bytes(est.total_bytes),
            if est.fits_in_memory { "yes" } else { "no" },
        );
    }

    println!("\nDone.");
    Ok(())
}

/// Format a parameter count for human-readable display.
fn format_param_count(count: u64) -> String {
    if count == 0 {
        return "unknown".to_string();
    }
    if count < 1_000_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else {
        format!("{:.2}B", count as f64 / 1_000_000_000.0)
    }
}
