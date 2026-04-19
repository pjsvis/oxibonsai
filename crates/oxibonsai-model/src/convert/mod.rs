//! HuggingFace safetensors → OxiBonsai GGUF conversion.
//!
//! Converts a HuggingFace model directory (containing `model.safetensors` or
//! sharded safetensors files and `config.json`) into an OxiBonsai GGUF file
//! with TQ2_0_g128 quantisation for weight tensors and FP32 for norm tensors.
//!
//! A sibling [`onnx`] module provides the same output format from HuggingFace
//! MatMulNBits-quantized ONNX models (e.g. `onnx-community/Ternary-Bonsai-1.7B-ONNX`).
//! Shared helpers live in [`common`].
//!
//! # Usage
//!
//! ```no_run
//! use std::path::Path;
//! use oxibonsai_model::convert::convert_hf_to_gguf;
//!
//! let stats = convert_hf_to_gguf(
//!     Path::new("/path/to/Ternary-Bonsai-1.7B-unpacked"),
//!     Path::new("/path/to/output.gguf"),
//!     "tq2_0_g128",
//! ).expect("conversion failed");
//!
//! println!("Converted {} tensors", stats.n_tensors);
//! ```

pub mod common;
pub mod name_map;
pub mod onnx;

use std::collections::{BTreeMap, HashMap};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::Context;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};
use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;

use crate::convert::common::{
    blocks_to_bytes, pad_to_multiple_of_128, read_config_json, write_metadata,
};
use crate::convert::name_map::hf_to_gguf_name;

pub use crate::convert::common::ConvertStats;

// ─── Public API ───────────────────────────────────────────────────────────────

/// Convert a HuggingFace safetensors model directory to an OxiBonsai GGUF file.
///
/// # Arguments
///
/// * `from_dir` — Directory containing `model.safetensors` (or sharded files
///   plus `model.safetensors.index.json`) and `config.json`.
/// * `to_path` — Destination path for the GGUF file.
/// * `quant` — Quantisation format; only `"tq2_0_g128"` is currently supported.
///
/// # Errors
///
/// Returns an error if the directory does not contain the expected files, if
/// any tensor cannot be converted, or if the output file cannot be written.
pub fn convert_hf_to_gguf(
    from_dir: &Path,
    to_path: &Path,
    quant: &str,
) -> anyhow::Result<ConvertStats> {
    if quant != "tq2_0_g128" {
        anyhow::bail!(
            "unsupported quantisation format '{}'; only 'tq2_0_g128' is supported",
            quant
        );
    }

    // ── 1. Read config.json ──────────────────────────────────────────────────
    let config = read_config_json(&from_dir.join("config.json"))?;

    // ── 2. Collect shard paths ───────────────────────────────────────────────
    let shard_paths = discover_shard_paths(from_dir)?;

    // ── 3. Load all raw tensor bytes from shards ─────────────────────────────
    // We store each shard's raw bytes so that SafeTensors can borrow from them.
    let shard_bytes_list: Vec<Vec<u8>> = shard_paths
        .iter()
        .map(|p| std::fs::read(p).with_context(|| format!("reading shard {:?}", p)))
        .collect::<anyhow::Result<_>>()?;

    // Parse SafeTensors views from each shard (borrows from shard_bytes_list).
    let parsed_shards: Vec<SafeTensors<'_>> = shard_bytes_list
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            SafeTensors::deserialize(bytes)
                .with_context(|| format!("parsing shard {:?}", shard_paths[i]))
        })
        .collect::<anyhow::Result<_>>()?;

    // Collect (hf_name → shard_index) for all tensors across shards.
    let mut name_to_shard: HashMap<&str, usize> = HashMap::new();
    for (shard_idx, shard) in parsed_shards.iter().enumerate() {
        for name in shard.names() {
            name_to_shard.insert(name, shard_idx);
        }
    }

    // ── 4. Build GGUF writer ─────────────────────────────────────────────────
    let mut writer = GgufWriter::new();

    // Model name derived from directory basename.
    let model_name = from_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    write_metadata(&mut writer, &config, model_name)?;

    // ── 5. Determine tied-embedding flag ────────────────────────────────────
    let tie_word_embeddings = config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // ── 6. Collect and sort tensor names so output is deterministic ──────────
    // Sort by GGUF name to get a canonical ordering (blk.0 before blk.1 etc.).
    let mut gguf_entries: BTreeMap<String, TensorEntryPending> = BTreeMap::new();
    let mut embed_tokens_f32: Option<Vec<f32>> = None;

    for (hf_name, &shard_idx) in &name_to_shard {
        let mapped = match hf_to_gguf_name(hf_name) {
            Some(m) => m,
            None => {
                tracing::debug!(hf_name, "skipping unmapped tensor");
                continue;
            }
        };

        let shard = &parsed_shards[shard_idx];
        let view = shard
            .tensor(hf_name)
            .with_context(|| format!("tensor '{}' not found in shard", hf_name))?;

        let f32_data = to_f32_vec(view.dtype(), view.data());
        if f32_data.is_empty() && !view.data().is_empty() {
            tracing::warn!(hf_name, dtype = ?view.dtype(), "unsupported dtype — skipping tensor");
            continue;
        }

        // Keep embed_tokens for tied embedding duplication if needed.
        if mapped.gguf_name == "token_embd.weight" && tie_word_embeddings {
            embed_tokens_f32 = Some(f32_data.clone());
        }

        let shape_hf = view.shape();
        // GGUF shape = reversed HF shape (outermost dimension last).
        let gguf_shape: Vec<u64> = shape_hf.iter().rev().map(|&d| d as u64).collect();

        gguf_entries.insert(
            mapped.gguf_name.clone(),
            TensorEntryPending {
                gguf_name: mapped.gguf_name,
                is_norm: mapped.is_norm,
                gguf_shape,
                f32_data,
            },
        );
    }

    // ── 7. Handle tied embeddings ────────────────────────────────────────────
    // If tie_word_embeddings is true and output.weight is absent, duplicate
    // token_embd.weight as output.weight (the loader hard-requires it).
    if tie_word_embeddings && !gguf_entries.contains_key("output.weight") {
        if let Some(embed_f32) = embed_tokens_f32 {
            let embed_entry = gguf_entries
                .get("token_embd.weight")
                .with_context(|| "tie_word_embeddings=true but token_embd.weight not found")?;
            let shape = embed_entry.gguf_shape.clone();
            tracing::info!("tie_word_embeddings=true: duplicating token_embd as output.weight");
            gguf_entries.insert(
                "output.weight".to_string(),
                TensorEntryPending {
                    gguf_name: "output.weight".to_string(),
                    is_norm: false,
                    gguf_shape: shape,
                    f32_data: embed_f32,
                },
            );
        }
    }

    // ── 8. Quantize and add tensors to writer ────────────────────────────────
    let mut stats = ConvertStats::default();

    for pending in gguf_entries.values() {
        let (raw_bytes, tensor_type) = if pending.is_norm {
            // FP32 norm tensor
            let raw: Vec<u8> = pending
                .f32_data
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            (raw, TensorType::F32)
        } else {
            // TQ2_0_g128 quantised tensor
            let f32_data = pad_to_multiple_of_128(&pending.f32_data);
            let blocks = BlockTQ2_0_g128::quantize(&f32_data)
                .with_context(|| format!("quantizing tensor '{}'", pending.gguf_name))?;
            let raw = blocks_to_bytes(&blocks);
            (raw, TensorType::TQ2_0_g128)
        };

        println!(
            "  converting {} {:?} -> {}",
            pending.gguf_name,
            pending.gguf_shape,
            if pending.is_norm { "F32" } else { "TQ2_0_g128" }
        );

        writer.add_tensor(TensorEntry {
            name: pending.gguf_name.clone(),
            shape: pending.gguf_shape.clone(),
            tensor_type,
            data: raw_bytes,
        });

        if pending.is_norm {
            stats.n_fp32 += 1;
        } else {
            stats.n_ternary += 1;
        }
        stats.n_tensors += 1;
    }

    // ── 9. Write GGUF file ───────────────────────────────────────────────────
    let out_file = std::fs::File::create(to_path)
        .with_context(|| format!("creating output file {:?}", to_path))?;
    let mut buf_writer = BufWriter::new(out_file);
    let bytes_written = writer
        .write(&mut buf_writer)
        .map_err(|e| anyhow::anyhow!("GGUF write error: {}", e))?;

    stats.output_bytes = bytes_written;
    Ok(stats)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// A pending tensor entry before quantization.
struct TensorEntryPending {
    gguf_name: String,
    is_norm: bool,
    gguf_shape: Vec<u64>,
    f32_data: Vec<f32>,
}

/// Discover shard file paths from the model directory.
///
/// Prefers a single `model.safetensors`; falls back to the shards listed in
/// `model.safetensors.index.json`.
fn discover_shard_paths(from_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let single = from_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let index_path = from_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        anyhow::bail!(
            "neither model.safetensors nor model.safetensors.index.json found in {:?}",
            from_dir
        );
    }

    let raw = std::fs::read_to_string(&index_path)
        .with_context(|| format!("reading {:?}", index_path))?;
    let index: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {:?}", index_path))?;

    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .with_context(|| format!("missing 'weight_map' in {:?}", index_path))?;

    // Collect unique shard filenames, preserving first-seen order.
    let mut shard_names: Vec<String> = Vec::new();
    for file_name in weight_map.values() {
        if let Some(s) = file_name.as_str() {
            if !shard_names.contains(&s.to_string()) {
                shard_names.push(s.to_string());
            }
        }
    }
    shard_names.sort(); // canonical ordering

    let paths: Vec<PathBuf> = shard_names.iter().map(|name| from_dir.join(name)).collect();

    Ok(paths)
}

/// Convert raw safetensors bytes to a `Vec<f32>` according to the dtype.
///
/// Returns an empty vec for unsupported dtypes (caller should warn and skip).
fn to_f32_vec(dtype: Dtype, data: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        _ => vec![],
    }
}
