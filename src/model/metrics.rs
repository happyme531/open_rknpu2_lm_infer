use crate::npu::Backend;
use serde_json::{Value, json};

pub(crate) fn backend_stats<'a>(
    backends: impl IntoIterator<Item = &'a Backend>,
    head: Option<&Backend>,
) -> Value {
    let mut domains: Vec<Value> = backends.into_iter().map(Backend::stats).collect();
    if let Some(head) = &head {
        domains.push(head.stats());
    }
    let mut result = json!({"domains": domains});
    for key in ["compress_w8a8", "weight_payload_bytes_estimated"] {
        result[key] = json!(domains.iter().any(|d| d[key].as_bool() == Some(true)));
    }
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for (prefix, key) in [("VmRSS:", "rss_kib"), ("VmHWM:", "peak_rss_kib")] {
            if let Some(kib) = status
                .lines()
                .find_map(|line| line.strip_prefix(prefix))
                .and_then(|value| value.split_whitespace().next())
                .and_then(|value| value.parse::<u64>().ok())
            {
                result[key] = json!(kib);
            }
        }
    }
    for key in [
        "loaded_w4",
        "loaded_w8",
        "weight_payload_bytes",
        "workspace_payload_bytes",
        "tuning_material_bytes",
        "workspace_count",
        "tune_cache_hits",
        "tune_cache_misses",
        "tune_calls",
        "strategy_fallbacks",
        "strategy_fallback_calls",
        "weight_reload_calls",
        "weight_profile_creations",
        "attention_cpu_staging_bytes",
    ] {
        result[key] = json!(
            domains
                .iter()
                .map(|d| d[key].as_u64().unwrap_or(0))
                .sum::<u64>()
        );
    }
    result
}
