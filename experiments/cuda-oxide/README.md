# Rust residual kernel

This standalone experiment compiles `src/main.rs` with NVlabs cuda-oxide
`26754ae52c26c097dc1c465a1e42c4c5d05a3d40` (0.2.1, nightly-2026-08-28).
The engine's opt-in `xabe-cuda/rust-kernels` feature loads its generated PTX
through cudarc. Normal builds do not need cuda-oxide or its nightly. The
embedded PTX is generated code, not an independently maintained kernel.

Install the matching compiler, with the CUDA 13 toolkit and libclang available:

```sh
cargo +nightly-2026-08-28 install --git https://github.com/NVlabs/cuda-oxide \
  --rev 26754ae52c26c097dc1c465a1e42c4c5d05a3d40 cargo-oxide
```

For a local compiler checkout, put its `target/debug` directory on `PATH` and
set `CUDA_OXIDE_BACKEND` to the matching `librustc_codegen_cuda.so`. Both must
come from the pinned revision. Regenerate from the repository root:

```sh
python3 experiments/cuda-oxide/regenerate.py
git diff -- crates/xabe-cuda/src/kernels/rust/tensor_add.ptx
```

On the target host, the build used these environment overrides (the isolated
libclang wheel was installed with `python3 -m pip install --target
/tmp/llmxabe-oxide-python libclang==18.1.1`):

```sh
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0
export LIBCLANG_PATH=/tmp/llmxabe-oxide-python/clang/native
export BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib/gcc/x86_64-linux-gnu/11/include -idirafter /usr/local/cuda-12.4/include'
```

The fallback include path supplies cuRAND headers missing from the minimal
CUDA 13.0 installation. This kernel uses no cuRAND API. It still reads the
CUDA 13.0 driver header first. The engine benchmark uses its ordinary CUDA
12.4 NVRTC dependency for the baseline.

Run outside the sandbox, with an idle GPU and no other builds or GPU work
during timing:

```sh
CUDA_VISIBLE_DEVICES=1 cargo run --release -p xabe-engine --bin bench_rust_add -- \
  crates/xabe-cuda/src/kernels/rust/tensor_add.ptx tensor_add
CUDA_VISIBLE_DEVICES=1 cargo test --release -p xabe-engine \
  --features xabe-cuda/rust-kernels --test layer_ops_differential -- --nocapture
```

The narrow gate checks empty and ragged lengths, decode and prefill shapes,
multiple grid strides, exact CPU agreement, both in-place aliases, guard
elements, and graph replay. It reports six alternating pairs with reversed
order on odd pairs. Each CUDA event interval contains 100 graph-captured
launches. Small shapes therefore reflect repeated cache-hot execution;
whole-model measurements are required before drawing an inference-speed claim.

An optional third argument selects a baseline PTX containing `tensor_add`,
instead of compiling the NVRTC baseline. This compares a compiler or kernel
change directly with the previous Rust artifact. The gate includes all float
alignments modulo 16, independently offset input/output pointers, and all
four-float tail lengths; wider accesses must preserve the sliced-buffer ABI.

To test inside the model, pass `--features xabe-cuda/rust-kernels` to
`bench_forward`, `bench_decode_batch`, and the existing forward tests. Keep
separate target directories for candidate and baseline, build both before
timing, and alternate processes on the same card. The production default
remains NVRTC while this feature is evaluated.

The measured correctness and performance results are recorded in
[BENCHMARKS.md](../../docs/BENCHMARKS.md#rust-kernel-authoring-can-keep-the-existing-launcher).
