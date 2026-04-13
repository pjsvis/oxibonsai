//! Multi-model support: auto-detect Bonsai model variant from GGUF metadata.
//!
//! The model registry provides automatic detection of model architecture
//! variants (8B, 4B, 1.7B) based on configuration parameters like
//! layer count and hidden dimension size.

use oxibonsai_core::config::Qwen3Config;

/// Known Bonsai model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelVariant {
    /// Bonsai-8B (Qwen3-8B architecture): 36 layers, hidden=4096
    Bonsai8B,
    /// Bonsai-4B: 24 layers, hidden=2560
    Bonsai4B,
    /// Bonsai-1.7B: 16 layers, hidden=1536
    Bonsai1_7B,
    /// Custom or unrecognized architecture
    Custom,
}

impl ModelVariant {
    /// Auto-detect variant from model configuration.
    ///
    /// Matches on the combination of `num_layers` and `hidden_size`
    /// to identify known architectures.
    pub fn from_config(config: &Qwen3Config) -> Self {
        match (config.num_layers, config.hidden_size) {
            (36, 4096) => ModelVariant::Bonsai8B,
            (24, 2560) => ModelVariant::Bonsai4B,
            (16, 1536) => ModelVariant::Bonsai1_7B,
            _ => ModelVariant::Custom,
        }
    }

    /// Get the default configuration for this variant.
    ///
    /// Returns the standard configuration for known variants.
    /// For `Custom`, returns the 8B configuration as a fallback.
    pub fn default_config(&self) -> Qwen3Config {
        match self {
            ModelVariant::Bonsai8B => Qwen3Config::bonsai_8b(),
            ModelVariant::Bonsai4B => Qwen3Config::bonsai_4b(),
            ModelVariant::Bonsai1_7B => Qwen3Config::bonsai_1_7b(),
            ModelVariant::Custom => Qwen3Config::bonsai_8b(),
        }
    }

    /// Human-readable display name for this variant.
    pub fn name(&self) -> &'static str {
        match self {
            ModelVariant::Bonsai8B => "Bonsai-8B",
            ModelVariant::Bonsai4B => "Bonsai-4B",
            ModelVariant::Bonsai1_7B => "Bonsai-1.7B",
            ModelVariant::Custom => "Custom",
        }
    }

    /// Approximate parameter count for this variant.
    ///
    /// Computed as: embedding + attention + ffn + norms + output head.
    /// For 1-bit models, each "parameter" is 1 bit + per-group scale.
    pub fn param_count(&self) -> u64 {
        match self {
            ModelVariant::Bonsai8B => {
                // Qwen3-8B: ~8.03B parameters
                // Embedding: 151936 * 4096 = 622M
                // Per layer: Q(4096*4096) + K(4096*1024) + V(4096*1024) + O(4096*4096)
                //          + gate(4096*14336) + up(4096*14336) + down(14336*4096)
                //          + 2 norms(4096 each)
                // = 16M + 4M + 4M + 16M + 58.7M + 58.7M + 58.7M + 8K = ~216M per layer
                // 36 layers = ~7.78B
                // + embedding(622M) + output(622M) + final norm(4K)
                8_030_000_000
            }
            ModelVariant::Bonsai4B => {
                // 24 layers, hidden=2560, intermediate=6912
                // Per layer: Q(2560*2560) + K(2560*512) + V(2560*512) + O(2560*2560)
                //          + gate(2560*6912) + up(2560*6912) + down(6912*2560) + norms
                // Embedding: 151936 * 2560
                4_020_000_000
            }
            ModelVariant::Bonsai1_7B => {
                // 16 layers, hidden=1536, intermediate=4096
                1_720_000_000
            }
            ModelVariant::Custom => 0,
        }
    }

    /// Expected model file size in bytes for the 1-bit quantized GGUF file.
    ///
    /// Approximate: 1-bit weights use ~1 bit per param + scale factors.
    /// Embeddings and norms are typically stored in FP16 or FP32.
    pub fn expected_model_size_bytes(&self) -> u64 {
        match self {
            ModelVariant::Bonsai8B => {
                // ~8B params at 1 bit = ~1 GB for weights
                // + embeddings in FP16: 151936 * 4096 * 2 = ~1.2 GB
                // + norms in FP32: ~0.01 GB
                // + metadata overhead
                // Total: ~2.2 GB
                2_200_000_000
            }
            ModelVariant::Bonsai4B => {
                // ~4B params at 1 bit = ~0.5 GB
                // + embeddings in FP16: 151936 * 2560 * 2 = ~0.78 GB
                // Total: ~1.3 GB
                1_300_000_000
            }
            ModelVariant::Bonsai1_7B => {
                // ~1.7B params at 1 bit = ~0.21 GB
                // + embeddings in FP16: 151936 * 1536 * 2 = ~0.47 GB
                // Total: ~0.7 GB
                700_000_000
            }
            ModelVariant::Custom => 0,
        }
    }

    /// Return all known (non-Custom) variants.
    pub fn known_variants() -> &'static [ModelVariant] {
        &[
            ModelVariant::Bonsai8B,
            ModelVariant::Bonsai4B,
            ModelVariant::Bonsai1_7B,
        ]
    }

    /// Whether this variant is a known (non-custom) architecture.
    pub fn is_known(&self) -> bool {
        !matches!(self, ModelVariant::Custom)
    }
}

impl std::fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_bonsai_8b() {
        let config = Qwen3Config::bonsai_8b();
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Bonsai8B);
        assert_eq!(ModelVariant::Bonsai8B.name(), "Bonsai-8B");
        assert!(ModelVariant::Bonsai8B.is_known());
    }

    #[test]
    fn detect_bonsai_4b() {
        let config = Qwen3Config::bonsai_4b();
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Bonsai4B);
        assert_eq!(ModelVariant::Bonsai4B.name(), "Bonsai-4B");
        assert!(ModelVariant::Bonsai4B.is_known());
    }

    #[test]
    fn detect_bonsai_1_7b() {
        let config = Qwen3Config::bonsai_1_7b();
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Bonsai1_7B);
        assert_eq!(ModelVariant::Bonsai1_7B.name(), "Bonsai-1.7B");
        assert!(ModelVariant::Bonsai1_7B.is_known());
    }

    #[test]
    fn detect_custom() {
        let mut config = Qwen3Config::bonsai_8b();
        config.num_layers = 48;
        config.hidden_size = 8192;
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Custom);
        assert_eq!(ModelVariant::Custom.name(), "Custom");
        assert!(!ModelVariant::Custom.is_known());
    }

    #[test]
    fn default_configs_roundtrip() {
        for variant in ModelVariant::known_variants() {
            let config = variant.default_config();
            let detected = ModelVariant::from_config(&config);
            assert_eq!(
                *variant, detected,
                "variant {:?} config should round-trip",
                variant
            );
        }
    }

    #[test]
    fn param_counts_are_reasonable() {
        assert!(ModelVariant::Bonsai8B.param_count() > 7_000_000_000);
        assert!(ModelVariant::Bonsai8B.param_count() < 10_000_000_000);

        assert!(ModelVariant::Bonsai4B.param_count() > 3_000_000_000);
        assert!(ModelVariant::Bonsai4B.param_count() < 5_000_000_000);

        assert!(ModelVariant::Bonsai1_7B.param_count() > 1_000_000_000);
        assert!(ModelVariant::Bonsai1_7B.param_count() < 2_500_000_000);

        assert_eq!(ModelVariant::Custom.param_count(), 0);
    }

    #[test]
    fn model_sizes_decrease_with_variant() {
        let size_8b = ModelVariant::Bonsai8B.expected_model_size_bytes();
        let size_4b = ModelVariant::Bonsai4B.expected_model_size_bytes();
        let size_1_7b = ModelVariant::Bonsai1_7B.expected_model_size_bytes();

        assert!(size_8b > size_4b, "8B should be larger than 4B");
        assert!(size_4b > size_1_7b, "4B should be larger than 1.7B");
        assert!(size_1_7b > 0, "1.7B should have nonzero size");
    }

    #[test]
    fn display_trait() {
        assert_eq!(format!("{}", ModelVariant::Bonsai8B), "Bonsai-8B");
        assert_eq!(format!("{}", ModelVariant::Custom), "Custom");
    }

    #[test]
    fn known_variants_list() {
        let variants = ModelVariant::known_variants();
        assert_eq!(variants.len(), 3);
        assert!(variants.contains(&ModelVariant::Bonsai8B));
        assert!(variants.contains(&ModelVariant::Bonsai4B));
        assert!(variants.contains(&ModelVariant::Bonsai1_7B));
    }
}
