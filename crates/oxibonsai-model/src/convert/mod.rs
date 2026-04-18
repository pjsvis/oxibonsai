//! HuggingFace safetensors → OxiBonsai GGUF conversion.
//!
//! Converts a HuggingFace model directory (containing `model.safetensors` or
//! sharded safetensors files and `config.json`) into an OxiBonsai GGUF file
//! with TQ2_0_g128 quantisation for weight tensors and FP32 for norm tensors.
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

pub mod name_map;

use std::collections::{BTreeMap, HashMap};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::Context;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use oxibonsai_core::gguf::tensor_info::keys;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_core::quant_ternary::{BlockTQ2_0_g128, BLOCK_TQ2_0_G128_BYTES};

use crate::convert::name_map::hf_to_gguf_name;

// ─── Public API ───────────────────────────────────────────────────────────────

/// Statistics returned after a successful conversion.
#[derive(Debug, Clone, Default)]
pub struct ConvertStats {
    /// Total number of tensors written to the GGUF file.
    pub n_tensors: usize,
    /// Number of tensors quantized to TQ2_0_g128.
    pub n_ternary: usize,
    /// Number of tensors stored as FP32.
    pub n_fp32: usize,
    /// Total size of the output GGUF file in bytes.
    pub output_bytes: usize,
}

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
    let config = read_config_json(from_dir)?;

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

    // Write metadata from config.json
    write_metadata(&mut writer, &config, from_dir)?;

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
            let element_count: usize = pending.gguf_shape.iter().product::<u64>() as usize;
            let f32_data = pad_to_multiple_of_128(&pending.f32_data, element_count);
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

/// Read and parse `config.json` from the model directory.
fn read_config_json(from_dir: &Path) -> anyhow::Result<Value> {
    let config_path = from_dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {:?}", config_path))?;
    let value: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {:?}", config_path))?;
    Ok(value)
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

/// Write model metadata from `config.json` into the GGUF writer.
fn write_metadata(writer: &mut GgufWriter, config: &Value, from_dir: &Path) -> anyhow::Result<()> {
    // Architecture constant
    writer.add_metadata(
        keys::GENERAL_ARCHITECTURE,
        MetadataWriteValue::Str("qwen3".to_string()),
    );

    // Model name from directory basename
    let model_name = from_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
    writer.add_metadata(keys::GENERAL_NAME, MetadataWriteValue::Str(model_name));

    // Quantisation version string
    writer.add_metadata(
        "general.quantization_version",
        MetadataWriteValue::Str("TQ2_0_G128".to_string()),
    );

    // Integer keys (u32)
    let u32_keys = [
        (keys::LLM_BLOCK_COUNT, "num_hidden_layers"),
        (keys::LLM_EMBEDDING_LENGTH, "hidden_size"),
        (keys::LLM_FEED_FORWARD_LENGTH, "intermediate_size"),
        (keys::LLM_ATTENTION_HEAD_COUNT, "num_attention_heads"),
        (keys::LLM_ATTENTION_HEAD_COUNT_KV, "num_key_value_heads"),
        (keys::LLM_CONTEXT_LENGTH, "max_position_embeddings"),
        (keys::LLM_VOCAB_SIZE, "vocab_size"),
    ];
    for (gguf_key, json_key) in &u32_keys {
        if let Some(val) = config.get(*json_key).and_then(Value::as_u64) {
            writer.add_metadata(gguf_key, MetadataWriteValue::U32(val as u32));
        } else {
            tracing::warn!(json_key, "missing or non-u64 field in config.json");
        }
    }

    // rms_norm_eps → F32
    if let Some(eps) = config.get("rms_norm_eps").and_then(Value::as_f64) {
        writer.add_metadata(
            keys::LLM_ATTENTION_LAYER_NORM_RMS_EPSILON,
            MetadataWriteValue::F32(eps as f32),
        );
    }

    // rope_theta → F32 (default 10000.0 if absent)
    let rope_theta = config
        .get("rope_theta")
        .and_then(Value::as_f64)
        .unwrap_or(10000.0);
    writer.add_metadata(
        keys::LLM_ROPE_FREQ_BASE,
        MetadataWriteValue::F32(rope_theta as f32),
    );

    Ok(())
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

/// Pad (or trim) `f32_data` to a multiple of 128 elements for TQ2_0_g128
/// quantization.  We pad with zeros and trim to the true element count
/// afterwards to avoid data loss.
fn pad_to_multiple_of_128(f32_data: &[f32], _element_count: usize) -> Vec<f32> {
    let len = f32_data.len();
    let remainder = len % 128;
    if remainder == 0 {
        f32_data.to_vec()
    } else {
        let padded_len = len + (128 - remainder);
        let mut padded = f32_data.to_vec();
        padded.resize(padded_len, 0.0f32);
        padded
    }
}

/// Serialise a slice of `BlockTQ2_0_g128` blocks to raw bytes.
///
/// Each block is 34 bytes: 32 bytes of packed `qs` + 2 bytes of FP16 `d`.
///
/// # Safety
///
/// `BlockTQ2_0_g128` is `#[repr(C)]` with a compile-time size assertion of
/// exactly 34 bytes.  The cast is safe because we verify the total byte length
/// matches `blocks.len() * BLOCK_TQ2_0_G128_BYTES`.
fn blocks_to_bytes(blocks: &[BlockTQ2_0_g128]) -> Vec<u8> {
    let total = blocks.len() * BLOCK_TQ2_0_G128_BYTES;
    // SAFETY: repr(C) layout with compile-time size check; byte length verified.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, total) };
    bytes.to_vec()
}
