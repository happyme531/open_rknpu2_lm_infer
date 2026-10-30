#[cfg(not(any(
    feature = "model-qwen3",
    feature = "model-llama",
    feature = "model-granite",
    feature = "model-qwen35",
    feature = "model-minicpmv47",
    feature = "model-ling3tiny"
)))]
compile_error!(
    "select at least one model family, e.g. --features model-qwen3, or use default features"
);

use std::{env, os::unix::process::CommandExt, process::Command};

fn main() {
    if let Err(error) = ensure_openmp_defaults() {
        eprintln!("Error: OpenMP startup: {error}");
        std::process::exit(1);
    }
    run();
}

fn ensure_openmp_defaults() -> std::io::Result<()> {
    let missing: Vec<_> = [("OMP_WAIT_POLICY", "PASSIVE"), ("GOMP_SPINCOUNT", "0")]
        .into_iter()
        .filter(|(name, _)| env::var_os(name).is_none())
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    // libgomp reads these in its constructor, before main or any Tokio threads.
    // exec preserves arguments, stdio and the inherited board-lock descriptor.
    Err(Command::new(env::current_exe()?)
        .args(env::args_os().skip(1))
        .envs(missing)
        .exec())
}

#[tokio::main]
async fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    if let Err(error) =
        open_rknpu2_lm_infer::cli::execute(open_rknpu2_lm_infer::cli::Cli::parse_args()).await
    {
        eprintln!("Error: {error:#}");
        std::process::exit(1);
    }
}
