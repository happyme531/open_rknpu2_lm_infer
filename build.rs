use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=RK_NPU_MATMUL_ROOT");
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
    let output = cmake::Config::new(&root)
        .profile("Release")
        .define("BUILD_TESTING", "OFF")
        .define("RK_NPU_ENABLE_TRACE", "OFF")
        .build_target("rk_npu_matmul_static")
        .build();
    let cpu_root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("native/cpu_kernels");
    println!("cargo:rerun-if-changed={}", cpu_root.display());
    let cpu_output = cmake::Config::new(cpu_root)
        .out_dir(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("cpu-kernels"))
        .profile("Release")
        .define("RK_NPU_MATMUL_ROOT", &root)
        .build_target("rknpu2_cpu_kernels")
        .build();
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
        .header(include.join("rk_npu_mm_tune.h").to_string_lossy())
        .header(include.join("rk_npu_mm_gelu.h").to_string_lossy())
        .header(include.join("rk_npu_gdn.h").to_string_lossy())
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
