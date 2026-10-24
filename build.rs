use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=RKNPU_PROJECT_NAME");
    let name =
        env::var("RKNPU_PROJECT_NAME").unwrap_or_else(|_| env::var("CARGO_PKG_NAME").unwrap());
    assert!(
        name.as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
            && name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.')),
        "RKNPU_PROJECT_NAME must start with an ASCII letter or digit and contain only letters, digits, '.', '_' or '-'"
    );
    println!("cargo:rustc-env=RKNPU_PROJECT_NAME={name}");
    println!("cargo:rerun-if-env-changed=RK_NPU_MATMUL_ROOT");
    println!("cargo:rerun-if-env-changed=RK_NPU_ENABLE_TRACE");
    println!("cargo:rerun-if-env-changed=RK_NPU_PERFETTO_ROOT");
    println!("cargo:rustc-check-cfg=cfg(rknpu_trace)");
    let trace = env::var("RK_NPU_ENABLE_TRACE").is_ok_and(|v| v == "1" || v == "ON");
    let root = env::var_os("RK_NPU_MATMUL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("native/rk_npu_matmul")
        })
        .canonicalize()
        .expect("initialize rk_npu_matmul with git submodule update --init --depth 1, or set RK_NPU_MATMUL_ROOT");
    for path in ["CMakeLists.txt", "include", "src"] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    let perfetto = env::var_os("RK_NPU_PERFETTO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("third_party/perfetto-v58.2"));
    let mut native = cmake::Config::new(&root);
    native
        .profile("Release")
        .define("BUILD_TESTING", "OFF")
        .define("RK_NPU_ENABLE_TRACE", if trace { "ON" } else { "OFF" });
    if trace {
        native.define("RK_NPU_PERFETTO_ROOT", &perfetto);
    }
    let output = native.build_target("rk_npu_matmul_static").build();
    let gdn = env::var_os("CARGO_FEATURE_ARCH_QWEN35").is_some();
    let kda = env::var_os("CARGO_FEATURE_MODEL_LING3TINY").is_some();
    // SwiGLU is shared by every model family; recurrence kernels stay optional.
    let cpu_root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("native/cpu_kernels");
    println!("cargo:rerun-if-changed={}", cpu_root.display());
    println!("cargo:rerun-if-changed=native/trace_bridge.cpp");
    let mut cpu = cmake::Config::new(cpu_root);
    cpu.out_dir(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("cpu-kernels"))
        .profile("Release")
        .define("RK_NPU_MATMUL_ROOT", &root)
        .define("RK_NPU_ENABLE_TRACE", if trace { "ON" } else { "OFF" });
    if trace {
        cpu.define("RK_NPU_TRACE_ROOT", &root)
            .define("RK_NPU_PERFETTO_ROOT", &perfetto);
    }
    cpu.define("RK_CPU_GDN", if gdn { "ON" } else { "OFF" })
        .define("RK_CPU_KDA", if kda { "ON" } else { "OFF" });
    let cpu_output = cpu.build_target("rknpu2_cpu_kernels").build();
    println!(
        "cargo:rustc-link-search=native={}",
        cpu_output.join("build").display()
    );
    println!("cargo:rustc-link-lib=static=rknpu2_cpu_kernels");
    println!(
        "cargo:rustc-link-search=native={}",
        output.join("build").display()
    );
    println!("cargo:rustc-link-lib=static=rk_npu_matmul");
    if trace {
        println!("cargo:rustc-cfg=rknpu_trace");
        println!("cargo:rustc-link-lib=dylib=rk_npu_trace");
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            output.join("build").display()
        );
    }
    for library in ["stdc++", "gomp", "pthread", "m", "dl"] {
        println!("cargo:rustc-link-lib={library}");
    }
    let include = root.join("include");
    let bindings = bindgen::Builder::default()
        .rust_target(
            bindgen::RustTarget::stable(85, 0)
                .unwrap_or_else(|_| panic!("unsupported Rust binding target")),
        )
        .rust_edition(bindgen::RustEdition::Edition2024)
        .header(include.join("rk_npu_quant_matmul.h").to_string_lossy())
        .header(include.join("rk_npu_w4a8_tune.h").to_string_lossy())
        .header(include.join("rk_npu_attention_f16.h").to_string_lossy())
        .header(include.join("rk_npu_mla_f16.h").to_string_lossy())
        .header(include.join("rk_npu_mm_tune.h").to_string_lossy())
        .header(include.join("rk_npu_mm_gelu.h").to_string_lossy())
        .header(include.join("rk_npu_gdn.h").to_string_lossy())
        .header(include.join("rk_npu_moe_w8.h").to_string_lossy())
        .clang_arg(format!("-I{}", include.display()))
        .allowlist_function("rk_npu_.*")
        .allowlist_type("rk_npu_.*")
        .allowlist_var("RK_NPU_.*")
        .prepend_enum_name(false)
        .derive_default(true)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("generate bindings for the rk_npu_matmul public C ABI");
    bindings
        .write_to_file(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("npu_bindings.rs"))
        .expect("write NPU bindings");
}
