//! Metrics system demonstration.
//!
//! Shows how to use the built-in Prometheus-compatible metrics:
//! counters, gauges, histograms, and the full `InferenceMetrics`
//! collection with Prometheus text rendering.

use std::time::Duration;

use oxibonsai_runtime::health::{
    check_kernel_tier, check_kv_cache, check_memory_pressure, check_model_loaded,
    run_health_checks, HealthStatus,
};
use oxibonsai_runtime::metrics::{Counter, Gauge, Histogram, InferenceMetrics};

fn main() -> anyhow::Result<()> {
    println!("=== OxiBonsai Metrics Demo ===\n");

    // ─── 1. Individual metric types ──────────────────────────────
    println!("--- Counter ---");
    let counter = Counter::new("demo_requests_total", "Total demo requests");
    counter.inc();
    counter.inc();
    counter.inc_by(10);
    println!(
        "  {} = {} (help: {})",
        counter.name(),
        counter.get(),
        counter.help()
    );

    println!("\n--- Gauge ---");
    let gauge = Gauge::new("demo_active_connections", "Active connections");
    gauge.set(5.0);
    gauge.inc();
    gauge.inc();
    gauge.dec();
    println!(
        "  {} = {:.0} (help: {})",
        gauge.name(),
        gauge.get(),
        gauge.help()
    );

    println!("\n--- Histogram ---");
    let hist = Histogram::new(
        "demo_request_duration_seconds",
        "Request duration in seconds",
        vec![0.01, 0.05, 0.1, 0.5, 1.0],
    );
    hist.observe(0.003);
    hist.observe(0.042);
    hist.observe(0.087);
    hist.observe(0.250);
    hist.observe(1.500);
    println!("  {} observations:", hist.name());
    println!("  count={}, sum={:.3}", hist.count(), hist.sum());
    for (i, boundary) in hist.bucket_boundaries().iter().enumerate() {
        println!("  le={:.2}: {}", boundary, hist.bucket_count(i));
    }
    println!(
        "  le=+Inf: {}",
        hist.bucket_count(hist.bucket_boundaries().len())
    );

    // Histogram timer
    println!("\n--- Histogram Timer ---");
    let timed_hist = Histogram::new(
        "demo_compute_duration",
        "Computation duration",
        vec![0.001, 0.01, 0.1],
    );
    let result = timed_hist.time(|| {
        // Simulate a short computation
        let mut sum = 0u64;
        for i in 0..100_000 {
            sum = sum.wrapping_add(i);
        }
        sum
    });
    println!("  Timed computation result: {}", result);
    println!(
        "  Recorded duration: {:.6}s ({} observations)",
        timed_hist.sum(),
        timed_hist.count()
    );

    // ─── 2. InferenceMetrics (full collection) ───────────────────
    println!("\n--- InferenceMetrics ---");
    let metrics = InferenceMetrics::new();

    // Simulate some inference activity
    metrics.requests_total.inc_by(42);
    metrics.tokens_generated_total.inc_by(1024);
    metrics.errors_total.inc_by(2);
    metrics.prompt_tokens_total.inc_by(500);
    metrics.active_requests.set(3.0);
    metrics.kv_cache_utilization.set(0.45);
    metrics.model_memory_bytes.set(2_200_000_000.0);

    metrics.prefill_duration_seconds.observe(0.015);
    metrics.prefill_duration_seconds.observe(0.022);
    metrics.decode_token_duration_seconds.observe(0.001);
    metrics.decode_token_duration_seconds.observe(0.002);
    metrics.request_duration_seconds.observe(0.5);
    metrics.request_duration_seconds.observe(1.2);
    metrics.tokens_per_second.observe(45.0);
    metrics.tokens_per_second.observe(52.0);

    println!("  Requests:      {}", metrics.requests_total.get());
    println!("  Tokens:        {}", metrics.tokens_generated_total.get());
    println!("  Errors:        {}", metrics.errors_total.get());
    println!("  Active:        {:.0}", metrics.active_requests.get());
    println!(
        "  KV util:       {:.1}%",
        metrics.kv_cache_utilization.get() * 100.0
    );

    // ─── 3. Prometheus text output ───────────────────────────────
    println!("\n--- Prometheus Output (first 30 lines) ---\n");
    let prom_output = metrics.render_prometheus();
    for (i, line) in prom_output.lines().enumerate() {
        if i >= 30 {
            println!("  ... ({} more lines)", prom_output.lines().count() - 30);
            break;
        }
        println!("  {}", line);
    }

    // ─── 4. Health checks ────────────────────────────────────────
    println!("\n--- Health Checks ---\n");

    let model_check = check_model_loaded(true);
    println!(
        "  Model:    {} ({})",
        model_check.status,
        model_check.details.as_deref().unwrap_or("n/a")
    );

    let mem_check = check_memory_pressure(6_000_000_000, 16_000_000_000);
    println!(
        "  Memory:   {} ({})",
        mem_check.status,
        mem_check.details.as_deref().unwrap_or("n/a")
    );

    let kv_check = check_kv_cache(0.45);
    println!(
        "  KV Cache: {} ({})",
        kv_check.status,
        kv_check.details.as_deref().unwrap_or("n/a")
    );

    let kernel_check = check_kernel_tier("neon");
    println!(
        "  Kernel:   {} ({})",
        kernel_check.status,
        kernel_check.details.as_deref().unwrap_or("n/a")
    );

    // ─── 5. Full health report ───────────────────────────────────
    println!("\n--- Full Health Report ---\n");
    let report = run_health_checks(
        true,
        6_000_000_000,
        16_000_000_000,
        0.45,
        "neon",
        Duration::from_secs(3600),
    );

    println!("  Overall: {}", report.overall);
    println!("  Version: {}", report.version);
    println!("  Uptime:  {:.0}s", report.uptime.as_secs_f64());

    let json = report.to_json();
    println!("\n  JSON output:");
    let pretty = serde_json::to_string_pretty(&json)?;
    for line in pretty.lines() {
        println!("    {}", line);
    }

    // ─── 6. Health status variants ───────────────────────────────
    println!("\n--- Health Status Variants ---");
    let statuses: Vec<HealthStatus> = vec![
        HealthStatus::Healthy,
        HealthStatus::Degraded("memory pressure at 85%".to_string()),
        HealthStatus::Unhealthy("model not loaded".to_string()),
    ];
    for s in &statuses {
        println!(
            "  {} => healthy={} degraded={} unhealthy={}",
            s,
            s.is_healthy(),
            s.is_degraded(),
            s.is_unhealthy(),
        );
    }

    println!("\nDone.");
    Ok(())
}
