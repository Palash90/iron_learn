# Iron Learn

A pure Rust machine learning library with GPU-accelerated optimization. Built for efficient tensor operations, gradient-based algorithms, and numerical computing with an emphasis on type safety and correctness.

## Features

- **GPU-Accelerated Training**: CUDA kernels for logistic regression on large datasets
- **Comprehensive Tensor Support**: Multi-dimensional arrays with generic numeric types
- **Optimization Algorithms**: Linear and logistic regression with automatic preprocessing
- **Complex Number Arithmetic**: Native support for complex-valued computations
- **Type-Safe API**: Result-based error handling without panics
- **Zero-Copy Operations**: Borrowing methods for efficient computation reuse

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
iron_learn = "0.5"
```

### Basic Usage

```rust
use iron_learn::Tensor;

// Create 2x2 matrices
let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
let b = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0])?;

// Add tensors without move
let sum = a.add(&b)?;

// Multiply tensors
let product = a.mul(&b)?;
```

## Core Modules

# Iron Learn

A multi-language (Rust + Python + CUDA) learning repository combining a Rust ML
library (`src/`) with several Python utilities and CUDA kernels. This
repository is an educational / experimental ML stack providing tensors,
gradient-based optimization, basic neural network building blocks, and GPU
acceleration for key linear algebra kernels.

This README expands on the project layout, purpose of each major component,
how to build/run, and where to look for specific functionality.

## High-level Overview

- Rust: Core library implementing `Tensor`s, numeric abstractions, optimization
    (gradient descent), neural network primitives, and optional CUDA-backed
    tensor implementations.
- CUDA: `kernels/` contains CUDA kernels used by the Rust `cuda_tensor` and
    `gpu_context` modules for accelerated matrix ops.
- Python: `python_scripts/` contains helper scripts, experiments and small
    neural-network examples used for prototyping and data preprocessing.
- Data/Images: Example JSON metadata and image assets under `data/` and
    `image/` used by demos and scripts.

## Repository Structure (map to code)

- `src/` — Rust library and binaries
    - `tensor/` — Tensor trait and backend implementations (CPU/GPU implementations live in `cpu_tensor/` and `cuda_tensor/`).
    - `neural_network/` — High-level NN builder, layers, activations and loss functions.
    - `gradient_descent.rs` — CPU implementations of linear and logistic regression and helper routines (normalization, bias handling).
    - `gpu_context.rs`, `cuda_tensor/` — GPU initialization and device-backed tensor types (CUDA interop, cublas wrappers, memory pools).
    - `read_file.rs` — Helpers for loading JSON model/data artifacts.
    - `runners.rs` — Small CLI-like routines for running `run_linear`, `run_logistic`, and `run_neural_net` examples.

- `kernels/` — CUDA device code
    - `gpu_kernels.cu` — Implementations for tiled matrix multiplication, elementwise ops, clipping, transpose and reductions.
    - `gpu_kernels.ptx` — Precompiled PTX shipped alongside the CUDA source.

- `python_scripts/` — Python utilities and experiments
    - Top-level scripts: `forward_prop.py`, `k-means.py`, `check_cuda.py`, `plot_graph.py`, etc.
    - `neural_net/` — Small Python builder, activation, layers and helpers used for rapid prototyping and educational examples.

- `data/`, `image/`, `weights/` — Example datasets, model JSONs and saved weights used by demos.

## What each major component does (details)

**Rust: `src/`**

- `tensor` (trait & implementations): Core abstraction exposing operations such as `add`, `mul`, `transpose`, elementwise math (`sin`, `cos`, `sigmoid`, etc.), and shape/data accessors. Backends implement `TensorMath` for mathematical functions.
- `cpu_tensor/`: Pure CPU tensor implementation used for most algorithms and tests.
- `cuda_tensor/`: CUDA-backed tensor implementation with device buffers and memory pooling. Integrates with `gpu_context` and uses kernels from `kernels/`.
- `neural_network/`: Provides a `NeuralNetBuilder` and a runtime `NeuralNet` type. Layers implement a `Layer<T>` trait; `LinearLayer` handles weight matrices and updates while `ActivationLayer` applies elementwise activations (sigmoid/tanh/linear/sin). `loss_functions.rs` contains `MeanSquaredErrorLoss` and `BinaryCrossEntropy` used for backprop.
- `gradient_descent.rs`: Implements `linear_regression`, `logistic_regression`, plus helper functions `predict_linear`, `predict_logistic`, and `gradient_descent` steps. Also contains `normalize_features` exposed from `commons`.
- `gpu_context.rs` and `cuda_tensor/*`: Manage GPU initialization, cuBLAS/cuDNN handles (where applicable), custom device buffers, and a simple memory pool to reduce allocations when transferring data.
- `runners.rs`: Provides convenience runners used by the CLI entrypoints (`main.rs`) to invoke example training runs.

**CUDA: `kernels/gpu_kernels.cu`**

- Implements atomic-safe memory comparison, element-wise math device kernels (exp, sin, cos, sigmoid branch), vector arithmetic, clipping, tiled matrix multiplication (shared-memory, TILE_SIZE 16), transpose and column reductions. These kernels are used by `cuda_tensor` for operations like `mul`, `transpose`, `col_reduce` and elementwise transforms (sigmoid, ln, exp).

**Python: `python_scripts/`**

- `forward_prop.py`: Small educational implementation of a two-layer forward pass and prediction helper. Useful for checking activation shapes and expected behavior.
- `nn.py`: Minimal vectorized helper functions demonstrating element-wise operations, weighted sum and simple neural network forward logic used for educational examples.
- `anomaly_detection.py`: Implements Gaussian estimation and threshold selection (F1-based) for anomaly detection — a direct NumPy port of the typical ML course algorithms (estimate Gaussian and select threshold by F1 score).
- `neural_net/` folder: Small Python builder, activation, layers and helpers used for experimentation and generating JSON `ModelData` artifacts that can be consumed by the Rust `read_file`/builder logic.

## How to build and run

Prerequisites:
- Rust toolchain (stable or nightly depending on local setup): https://rustup.rs
- For GPU builds: CUDA Toolkit and an NVIDIA GPU (driver + nvcc). If you don't plan to use GPU tensors, CPU-only build is fine.
- Python 3.8+ for the scripts (optional). NumPy recommended for running `python_scripts`.

Build the Rust library and examples:

```bash
cargo build --release
```

Run unit tests:

```bash
cargo test
```

Run the Rust demonstration runners (examples):

```bash
# Run the binary entry (see main.rs / main_2.rs for CLI flags)
cargo run --release --example run_linear
```

If you want to enable CUDA-backed tensors, ensure your environment has CUDA installed and visible to the linker. Typical workflow is the same `cargo build` but the code will initialize GPU devices at runtime when `init_gpu()` or `init_context()` is invoked.

Python scripts can be run directly (no extra files created here):

```bash
python python_scripts/forward_prop.py
python python_scripts/k-means.py
python anomaly_detection.py
```

`python_scripts/check_cuda.py` is a small utility to detect/validate CUDA availability from Python; useful for quick GPU checks.

## Where to look for functionality you might extend or inspect

- Implement new layers / activations: `src/neural_network/` (add `Layer` impls and wire into `NeuralNetBuilder`).
- Change tensor math: `src/tensor/` traits and `src/cpu_tensor` / `src/cuda_tensor` implementations.
- Add GPU kernels: `kernels/gpu_kernels.cu` and regenerate PTX if required; then extend `cuda_tensor` to call new kernels.
- Add CLI runners: `src/runners.rs` and `src/main.rs`.

## Notes from code inspection (current state)

- The Rust `NeuralNetBuilder` supports building from scratch or restoring `ModelData` (weights loaded from JSON). Layers expose `get_parameters()` for serialization.
- `ActivationLayer` caches outputs; activations (sigmoid, tanh, sin, linear) are implemented in `activations.rs` using the `TensorMath` trait.
- `LinearLayer` performs forward via `input.mul(&self.weights)` and backward by computing `weights_grad = input.T * error` then updating `self.weights = self.weights - lr * weights_grad`.
- `loss_functions.rs` includes stable BinaryCrossEntropy (clipping predictions to avoid log/divide-by-zero) and an MSE implementation.
- CUDA kernels implement a tiled matrix multiply and common elementwise ops; the kernels use shared memory and boundary checks for correctness.

## Quick pointers for contributors

- Run tests: `cargo test`
- Format: `cargo fmt`
- Lint: `cargo clippy` (may require installing `clippy` via rustup)
- Keep Python examples and the Rust model importer (`read_file.rs`) in sync if you change JSON model formats.


## Detailed Module Reference

**Tensor API**
- **Purpose:** Core trait `Tensor<T>` defines creation (`new`, `zeroes`, `ones`), shape/data accessors (`get_shape`, `get_data`), linear algebra (`add`, `sub`, `mul`, `t`, `multiply`, `div`, `scale`, `clip`) and reducers (`sum`). See `src/tensor/mod.rs` for the trait and basic docs.

**CpuTensor (CPU backend)**
- **Purpose:** `CpuTensor<T>` implements `Tensor<T>` for CPU operations with plain Rust `Vec<T>` storage and row-major layout. See `src/cpu_tensor/mod.rs`.
- **Highlights:** explicit shape validation, element-wise math helpers, an inner `_cpu_mul` implementation optimized for clarity and basic SIMD-friendly loops, and numerically-stable sigmoid in `element_op`.
- **Limitations:** Currently restricted to 1D/2D tensors.

**GpuTensor (CUDA backend)**
- **Purpose:** `GpuTensor<T>` provides a CUDA-backed `Tensor` using device buffers, kernels from `kernels/gpu_kernels.cu`, and optional cublas-accelerated multiplication (`_gpu_mul_cublas`). See `src/cuda_tensor/mod.rs` and `src/gpu_context.rs` for initialization.
- **Highlights:** device memory pooling, kernel launches via `cust::launch!`, functions for elementwise ops (`element_op`), clipping, transpose (naive), tiled matrix multiply, and column-wise reduce. Uses `GPU_CONTEXT` to find module/function handles.
- **Notes:** GPU ops require `init_gpu()` to be called and a valid `GpuContext`. Multiplication defaults to `cublasSgemm` path when available.

**GPU Context**
- **Purpose:** `src/gpu_context.rs` exposes `init_gpu(...)` and `GPU_CONTEXT` (global OnceLock). It stores CUDA `Module`, `Stream`, `CudaMemoryPool`, and a `CublasHandle` used by `GpuTensor`.

**Gradient Descent / Regression**
- **Purpose:** `src/gradient_descent.rs` implements `gradient_descent`, `linear_regression`, `logistic_regression`, and helpers `predict_linear`, `predict_logistic`. The functions expect `Tensor<f64>` with `TensorMath<f64>` support and follow standard batch gradient updates with optional logistic sigmoid.

**Neural Network**
- **Purpose:** High-level NN abstraction in `src/neural_network/`. Use `NeuralNetBuilder` to add `Linear` layers (`LinearLayer`) and activations (`ActivationLayer`) and then `build()` a `NeuralNet` instance.
- **Model Persistence:** `ModelData` (see `src/neural_network/mod.rs`) serializes model metadata:
    - `name`: model name
    - `parameter_count`: total parameters
    - `layers`: list of `LayerData` objects (`layer_type`, `name`, `index`, `weights`, `shape`)
    - `epoch`: last saved epoch
    - `saved_lr`: learning rate saved with the model

**CUDA Kernels (quick API map)**
- Located at `kernels/gpu_kernels.cu`. Main exported kernels (extern "C"):
    - `fill_value(float *out, int n, float value)` — fill buffer
    - `vector_arithmatic(const float *a, const float *b, float *out, int n, unsigned int op)` — add/sub/mul/div
    - `clip(const float *s, float *r, int n, float min, float max)` — clip values
    - `element_op(const float *s, float *r, int n, int op, float scale)` — exp/sin/cos/tanh/sigmoid/log
    - `compare_memory(const float *a, const float *b, size_t size, int *result)` — compare arrays
    - `transpose_naive(const float *A, float *B, int M, int N)` — naive transpose
    - `matrix_mul(const float *A, const float *B, float *C, int M, int N, int K)` — tiled matmul using shared memory
    - `column_reduce(const float *inputMatrix, float *outputSums, int numRows, int numCols)` — column sums

**Python scripts (where to look)**
- `python_scripts/forward_prop.py` — small two-layer forward pass and prediction helper.
- `python_scripts/nn.py` — tiny, educational neural-network helper functions (weighted_sum, vector ops).
- `python_scripts/neural_net/*` — prototype Python-side neural net builder and helpers used for experiments and model JSON creation.
- `anomaly_detection.py` — feature-wise Gaussian estimation and F1-based threshold selection (useful as a standalone script).

**Data & Model JSONs**
- Example dataset: see `data/image.json` (fields: `m`, `n`, `m_test`, `x`, `y`, `x_test`, `y_test`) — JSON contains flattened row-major arrays for `x`/`y`.
- Example model file: see `image/model.json` — follows `ModelData` schema described above. Use `read_file::deserialize_model()` to load models into `ModelData`, and `NeuralNetBuilder::build_from_model()` to restore a runtime `NeuralNet` from `ModelData`.

## Next steps I can take (choose one)
- Add per-module examples for `CpuTensor` and `GpuTensor` (small code snippets).
- Document the `ModelData` JSON format with a minimal example and a small Rust snippet to load it.
- Audit `python_scripts/neural_net` and produce instructions for generating compatible `ModelData` JSON files.

## Example: Load a saved `ModelData` (Rust)

The Rust side can deserialize `ModelData` JSON and restore a runtime `NeuralNet`. Example snippet (uses `CpuTensor<f32>` as the runtime tensor type):

```rust
use iron_learn::{read_file, NeuralNetBuilder, MeanSquaredErrorLoss, CpuTensor};

fn load_model(path: &str) -> Option<iron_learn::neural_network::NeuralNet<CpuTensor<f32>>> {
    // Deserialize the JSON file to ModelData
    let model = read_file::deserialize_model(path)?;

    // Build a runtime NeuralNet from the stored ModelData. We use MSE as a placeholder loss
    let net = NeuralNetBuilder::<CpuTensor<f32>>::build_from_model(model, Box::new(MeanSquaredErrorLoss));

    Some(net)
}

// Usage:
// let net = load_model("image/model.json").expect("failed to load model");
```

Notes:
- Replace `CpuTensor<f32>` with `GpuTensor<f32>` if you want a GPU-backed runtime and have initialized the GPU context.
- `MeanSquaredErrorLoss` is used here as an example; choose the appropriate loss when restoring classification models (e.g., `BinaryCrossEntropy`).

## Exporting `ModelData` from Python (compatible JSON)

The Python prototype in `python_scripts/neural_net` already saves weights (`final_model_weights.npz`) and a simple `myfile.json`. To create a JSON file compatible with Rust `ModelData`, follow this pattern:

Python snippet to export `ModelData`:

```python
import json
import numpy as np

def export_modeldata_py(nn, name="py_model", epoch=0, saved_lr=0.001):
    layers = []
    parameter_count = 0
    layer_index = 0

    for info in nn.layer_info:
        ltype = info['type']
        lname = info['name']

        if ltype == 'LinearLayer':
            W = np.asarray(info['layer'].weights)
            # Ensure row-major flattened list of floats
            flat = W.flatten(order='C').astype(float).tolist()
            shape = [int(W.shape[0]), int(W.shape[1])]
            parameter_count += W.size

            layers.append({
                'layer_type': 'Linear',
                'name': lname,
                'index': layer_index,
                'weights': flat,
                'shape': shape
            })

        else:
            # Activation layers: map the Python activation to Rust LayerType names
            mapping = {
                'ActivationLayer': 'Tanh',  # change as appropriate per activation
                'SinusoidalLayer': 'Sin'
            }
            layers.append({
                'layer_type': mapping.get(ltype, 'Linear'),
                'name': lname,
                'index': layer_index,
                'weights': [],
                'shape': []
            })

        layer_index += 1

    modeldata = {
        'name': name,
        'parameter_count': parameter_count,
        'layers': layers,
        'epoch': epoch,
        'saved_lr': float(saved_lr)
    }

    with open(f"{name}_model.json", 'w') as f:
        json.dump(modeldata, f, indent=2)

    print(f"Exported {name}_model.json")

# Example usage: export_modeldata_py(net, name='image', epoch=100, saved_lr=0.001)
```

Guidance:
- For activation layers, set the `layer_type` to one of Rust's `LayerType` names: `Sigmoid`, `Tanh`, `Sin`, or `Linear`.
- Ensure weights are flattened in row-major (`order='C'`) so Rust's `T::new(shape, weights)` interprets them correctly.
- `parameter_count` is optional for loading (Rust `build_from_model` accepts it) but helpful for bookkeeping.

