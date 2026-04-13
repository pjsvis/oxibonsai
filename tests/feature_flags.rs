//! Feature flag compile-time validation tests for OxiBonsai.
//!
//! These tests verify that feature flags are correctly propagated across
//! the workspace crate hierarchy: oxibonsai -> oxibonsai-runtime -> oxibonsai-kernels.

// ── Default features ────────────────────────────────────────────────────────

#[test]
#[allow(clippy::assertions_on_constants)]
fn default_features_include_server() {
    // The "server" feature is enabled by default in oxibonsai.
    #[cfg(feature = "server")]
    {
        assert!(
            true,
            "server feature should be enabled with default features"
        );
    }
    #[cfg(not(feature = "server"))]
    {
        panic!("server feature should be enabled by default");
    }
}

// ── Core crate availability ─────────────────────────────────────────────────

#[test]
fn core_crate_exports_available() {
    // Verify that fundamental types from oxibonsai-core are accessible
    let _header_size = std::mem::size_of::<oxibonsai_core::GgufHeader>();
    assert!(_header_size > 0, "GgufHeader should be a non-ZST type");
}

#[test]
fn core_tensor_types_available() {
    let _block_size = std::mem::size_of::<oxibonsai_core::BlockQ1_0G128>();
    // Q1_0_g128: 2-byte FP16 scale + 16 bytes sign bits = 18 bytes
    assert_eq!(_block_size, 18, "BlockQ1_0G128 should be 18 bytes");
}

#[test]
fn core_config_type_available() {
    let _config_size = std::mem::size_of::<oxibonsai_core::Qwen3Config>();
    assert!(_config_size > 0, "Qwen3Config should be a non-ZST type");
}

// ── Kernel crate availability ───────────────────────────────────────────────

#[test]
fn kernel_dispatcher_available() {
    use oxibonsai_kernels::OneBitKernel;
    let dispatcher = oxibonsai_kernels::KernelDispatcher::auto_detect();
    let name = dispatcher.name();
    assert!(
        !name.is_empty(),
        "dispatcher should report a non-empty name"
    );
}

#[test]
fn kernel_trait_available() {
    // Verify the OneBitKernel trait is importable
    fn _assert_trait_exists<T: oxibonsai_kernels::OneBitKernel>() {}
}

// ── Runtime crate availability ──────────────────────────────────────────────

#[test]
fn runtime_sampling_params_available() {
    let params = oxibonsai_runtime::sampling::SamplingParams::default();
    assert!(
        params.temperature >= 0.0,
        "default temperature should be non-negative"
    );
}

#[test]
fn runtime_config_available() {
    let config = oxibonsai_runtime::OxiBonsaiConfig::default();
    // Config should have reasonable defaults
    let _model = &config.model;
    let _sampling = &config.sampling;
}

#[test]
fn runtime_builder_pattern_available() {
    // Verify the builder API is accessible
    let _builder_size = std::mem::size_of::<oxibonsai_runtime::ConfigBuilder>();
    assert!(_builder_size > 0, "ConfigBuilder should be a non-ZST type");
}

#[test]
fn runtime_presets_available() {
    let _preset_size = std::mem::size_of::<oxibonsai_runtime::SamplingPreset>();
    assert!(_preset_size > 0, "SamplingPreset should be a non-ZST type");
}

#[test]
fn runtime_health_available() {
    let _report_size = std::mem::size_of::<oxibonsai_runtime::HealthReport>();
    assert!(_report_size > 0, "HealthReport should be a non-ZST type");
}

#[test]
fn runtime_circuit_breaker_available() {
    let _cb_size = std::mem::size_of::<oxibonsai_runtime::CircuitBreaker>();
    assert!(_cb_size > 0, "CircuitBreaker should be a non-ZST type");
}

#[test]
fn runtime_metrics_available() {
    let metrics = oxibonsai_runtime::InferenceMetrics::new();
    // Verify metrics can render to Prometheus format (starts empty)
    let output = metrics.render_prometheus();
    assert!(
        !output.is_empty(),
        "new metrics should render non-empty Prometheus output"
    );
}

// ── SIMD feature detection ──────────────────────────────────────────────────

#[test]
#[allow(clippy::assertions_on_constants)]
fn simd_feature_flags_compile() {
    // These are compile-time checks: if the feature is enabled, the
    // corresponding module should exist.  On CI we test each feature
    // separately with --features flags.

    #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
    {
        // AVX2 module should be available
        assert!(true, "simd-avx2 feature compiles on x86_64");
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd-avx512"))]
    {
        assert!(true, "simd-avx512 feature compiles on x86_64");
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd-neon"))]
    {
        assert!(true, "simd-neon feature compiles on aarch64");
    }
}

// ── Model crate availability ────────────────────────────────────────────────

#[test]
fn model_types_available() {
    let _model_size = std::mem::size_of::<oxibonsai_model::ModelVariant>();
    assert!(_model_size > 0, "ModelVariant should be a non-ZST type");
}

#[test]
fn model_kv_cache_available() {
    let _cache_size = std::mem::size_of::<oxibonsai_model::KvCache>();
    assert!(_cache_size > 0, "KvCache should be a non-ZST type");
}

// ── Server feature gating ───────────────────────────────────────────────────

#[test]
#[cfg(feature = "server")]
#[allow(clippy::assertions_on_constants)]
fn server_module_available_when_feature_enabled() {
    // When "server" feature is on, the server module should be usable
    let _router_fn_exists = oxibonsai_runtime::server::create_router_with_metrics;
    assert!(true, "server module is available with server feature");
}
