use std::path::PathBuf;
use std::process::Command;
use std::io::{self, Write, ErrorKind};

fn main() {
    // 1. Define the input CUDA source file and the desired output PTX file name.
    let kernel_src = "kernels/gradient_descent.cu";
    let kernel_ptx_name = "gradient_descent.ptx";

    // 2. Tell Cargo which files to watch:
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", kernel_src);

    // 3. Get the output directory path set by Cargo (MANDATORY location for Rust linking).
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join(kernel_ptx_name);
    
    // Attempt to compile CUDA kernel
    let compilation_result = Command::new("nvcc")
        .arg(kernel_src)
        .arg("-o")
        .arg(&ptx_path)
        .args(&["-ptx", "-lcublas"])
        .status();

    match compilation_result {
        Ok(status) => {
            if status.success() {
                // Compilation succeeded, PTX file is in $OUT_DIR
                println!("cargo:warning=Successfully compiled CUDA kernel to: {}", ptx_path.display());

                // --- NEW STEP: Copy the PTX file to the project's kernels/ directory for easy inspection ---
                let kernel_copy_dir = PathBuf::from("kernels");
                let kernel_copy_path = kernel_copy_dir.join(kernel_ptx_name);
                
                // Ensure the 'kernels' directory exists before copying
                if let Err(e) = std::fs::create_dir_all(&kernel_copy_dir) {
                     println!("cargo:warning=Could not create 'kernels/' directory for copy: {}", e);
                }

                match std::fs::copy(&ptx_path, &kernel_copy_path) {
                    Ok(_) => {
                        println!("cargo:warning=Copied PTX file to project root for inspection: {}", kernel_copy_path.display());
                    }
                    Err(e) => {
                        // Log a warning, but don't panic as the build technically succeeded
                        println!("cargo:warning=Failed to copy PTX file to kernels/ directory: {}", e);
                    }
                }
                // ------------------------------------------------------------------------------------------

            } else {
                // nvcc was found, but compilation failed (e.g., syntax error in .cu file)
                panic!(
                    "nvcc was found but failed to compile {}. Exit code: {}. Check your CUDA code.",
                    kernel_src,
                    status.code().unwrap_or(-1)
                );
            }
        }
        Err(e) => {
            if e.kind() == ErrorKind::NotFound {
                // === GRACEFUL FALLBACK: NVCC NOT FOUND ===
                println!("cargo:warning=NVCC not found ({}). Skipping CUDA compilation and creating an EMPTY placeholder PTX file.", e);
                println!("cargo:warning=Please ensure your main Rust code checks if the included PTX string is empty and falls back to CPU execution.");

                // Create the empty file the main Rust code expects to exist via `include_str!`.
                let mut file = std::fs::File::create(&ptx_path)
                    .unwrap_or_else(|e| panic!("Failed to create placeholder PTX file at {}: {}", ptx_path.display(), e));
                
                writeln!(file, "").unwrap_or_else(|e| panic!("Failed to write to placeholder PTX file at {}: {}", ptx_path.display(), e));
                
            } else {
                // Some other I/O error (permission denied, etc.)
                panic!("Failed to execute nvcc command: {:?}", e);
            }
        }
    }
}