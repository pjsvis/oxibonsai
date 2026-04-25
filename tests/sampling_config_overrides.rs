//! Sampling config precedence chain tests.
//!
//! Tests the precedence chain: defaults < TOML < CLI
//! for all [sampling] parameters in OxiBonsaiConfig.

use std::fs;
use std::io::Write;
use tempfile::TempDir;

use oxibonsai_runtime::config::{OxiBonsaiConfig, SamplingConfig};

/// Test that SamplingConfig defaults are applied when no TOML is present.
#[test]
fn sampling_config_defaults() {
    let config = SamplingConfig::default();
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, 40);
    assert_eq!(config.top_p, 0.9);
    assert_eq!(config.repetition_penalty, 1.1);
    assert_eq!(config.max_tokens, 512);
}

/// Test that TOML parsing correctly loads [sampling] section.
#[test]
fn sampling_config_from_toml() {
    let toml_content = r#"
[sampling]
temperature = 0.5
top_k = 20
top_p = 0.8
repetition_penalty = 1.3
max_tokens = 256
"#;

    let temp_dir = TempDir::new().expect("temp dir");
    let config_path = temp_dir.path().join("config.toml");
    fs::File::create(&config_path)
        .expect("create config file")
        .write_all(toml_content.as_bytes())
        .expect("write config");

    let config = OxiBonsaiConfig::load_or_default(Some(&config_path));

    assert_eq!(config.sampling.temperature, 0.5);
    assert_eq!(config.sampling.top_k, 20);
    assert_eq!(config.sampling.top_p, 0.8);
    assert_eq!(config.sampling.repetition_penalty, 1.3);
    assert_eq!(config.sampling.max_tokens, 256);
}

/// Test that missing [sampling] section falls back to defaults.
#[test]
fn sampling_config_missing_section_uses_defaults() {
    let toml_content = r#"
[model]
max_seq_len = 4096
"#;

    let temp_dir = TempDir::new().expect("temp dir");
    let config_path = temp_dir.path().join("config.toml");
    fs::File::create(&config_path)
        .expect("create config file")
        .write_all(toml_content.as_bytes())
        .expect("write config");

    let config = OxiBonsaiConfig::load_or_default(Some(&config_path));

    // Should fall back to defaults
    assert_eq!(config.sampling.temperature, 0.7);
    assert_eq!(config.sampling.top_k, 40);
    assert_eq!(config.sampling.top_p, 0.9);
    assert_eq!(config.sampling.repetition_penalty, 1.1);
    assert_eq!(config.sampling.max_tokens, 512);
}

/// Test that partial [sampling] section only overrides specified fields.
#[test]
fn sampling_config_partial_toml() {
    let toml_content = r#"
[sampling]
repetition_penalty = 1.5
"#;

    let temp_dir = TempDir::new().expect("temp dir");
    let config_path = temp_dir.path().join("config.toml");
    fs::File::create(&config_path)
        .expect("create config file")
        .write_all(toml_content.as_bytes())
        .expect("write config");

    let config = OxiBonsaiConfig::load_or_default(Some(&config_path));

    // repetition_penalty from TOML, rest from defaults
    assert_eq!(config.sampling.repetition_penalty, 1.5);
    assert_eq!(config.sampling.temperature, 0.7);
    assert_eq!(config.sampling.top_k, 40);
    assert_eq!(config.sampling.top_p, 0.9);
    assert_eq!(config.sampling.max_tokens, 512);
}

/// Test CLI override precedence: CLI > TOML > defaults.
/// This tests the logic that would be applied in main.rs.
#[test]
fn cli_overrides_toml() {
    let toml_content = r#"
[sampling]
repetition_penalty = 1.3
temperature = 0.5
top_k = 20
top_p = 0.8
max_tokens = 256
"#;

    let temp_dir = TempDir::new().expect("temp dir");
    let config_path = temp_dir.path().join("config.toml");
    fs::File::create(&config_path)
        .expect("create config file")
        .write_all(toml_content.as_bytes())
        .expect("write config");

    let config = OxiBonsaiConfig::load_or_default(Some(&config_path));

    // Simulate CLI override values (e.g., from clap --repetition-penalty 1.2)
    let cli_repetition_penalty = Some(1.2f32);
    let cli_temperature = Some(0.3f32);

    let final_repetition_penalty =
        cli_repetition_penalty.unwrap_or(config.sampling.repetition_penalty);
    let final_temperature = cli_temperature.unwrap_or(config.sampling.temperature);

    assert_eq!(final_repetition_penalty, 1.2); // CLI wins over TOML
    assert_eq!(final_temperature, 0.3); // CLI wins over TOML
}

/// Test that absence of CLI flag uses TOML value.
#[test]
fn no_cli_flag_uses_toml() {
    let toml_content = r#"
[sampling]
repetition_penalty = 1.4
temperature = 0.6
"#;

    let temp_dir = TempDir::new().expect("temp dir");
    let config_path = temp_dir.path().join("config.toml");
    fs::File::create(&config_path)
        .expect("create config file")
        .write_all(toml_content.as_bytes())
        .expect("write config");

    let config = OxiBonsaiConfig::load_or_default(Some(&config_path));

    // No CLI override (None)
    let cli_repetition_penalty: Option<f32> = None;

    let final_repetition_penalty =
        cli_repetition_penalty.unwrap_or(config.sampling.repetition_penalty);

    assert_eq!(final_repetition_penalty, 1.4); // TOML value used
}

/// Test that absence of both CLI and TOML uses defaults.
#[test]
fn no_cli_no_toml_uses_defaults() {
    let toml_content = r#"
[model]
max_seq_len = 4096
"#;

    let temp_dir = TempDir::new().expect("temp dir");
    let config_path = temp_dir.path().join("config.toml");
    fs::File::create(&config_path)
        .expect("create config file")
        .write_all(toml_content.as_bytes())
        .expect("write config");

    let config = OxiBonsaiConfig::load_or_default(Some(&config_path));

    // No CLI override
    let cli_repetition_penalty: Option<f32> = None;

    let final_repetition_penalty =
        cli_repetition_penalty.unwrap_or(config.sampling.repetition_penalty);

    assert_eq!(final_repetition_penalty, 1.1); // Default value
}

/// Test full precedence chain: CLI > TOML > defaults
#[test]
fn full_precedence_chain_defaults_toml_cli() {
    // Case 1: No TOML, no CLI → defaults
    let defaults = SamplingConfig::default();
    let none: Option<f32> = None;
    let result = none.unwrap_or(defaults.repetition_penalty);
    assert_eq!(result, 1.1);

    // Case 2: TOML but no CLI → TOML
    let toml_value = 1.3f32;
    let none: Option<f32> = None;
    let result = none.unwrap_or(toml_value);
    assert_eq!(result, 1.3);

    // Case 3: CLI override → CLI wins
    let cli_value = Some(1.2f32);
    let toml_value = 1.3f32;
    let result = cli_value.unwrap_or(toml_value);
    assert_eq!(result, 1.2);
}

/// Test that repetition_penalty value of 1.0 is valid (disabled).
#[test]
fn repetition_penalty_minimum_valid() {
    let toml_content = r#"
[sampling]
repetition_penalty = 1.0
"#;

    let temp_dir = TempDir::new().expect("temp dir");
    let config_path = temp_dir.path().join("config.toml");
    fs::File::create(&config_path)
        .expect("create config file")
        .write_all(toml_content.as_bytes())
        .expect("write config");

    let config = OxiBonsaiConfig::load_or_default(Some(&config_path));

    assert_eq!(config.sampling.repetition_penalty, 1.0);
}

/// Test that repetition_penalty < 1.0 is valid in TOML (validation is at CLI parse time).
#[test]
fn repetition_penalty_below_one_valid_in_toml() {
    let toml_content = r#"
[sampling]
repetition_penalty = 0.9
"#;

    let temp_dir = TempDir::new().expect("temp dir");
    let config_path = temp_dir.path().join("config.toml");
    fs::File::create(&config_path)
        .expect("create config file")
        .write_all(toml_content.as_bytes())
        .expect("write config");

    let config = OxiBonsaiConfig::load_or_default(Some(&config_path));

    // TOML allows < 1.0; CLI parser enforces >= 1.0
    assert_eq!(config.sampling.repetition_penalty, 0.9);
}