#!/usr/bin/env python3
"""Rebuild the pinned Rust kernel and refresh the engine's embedded PTX."""
import hashlib
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "src/main.rs"
OUTPUT = HERE.parents[1] / "crates/xabe-cuda/src/kernels/rust/tensor_add.ptx"
REVISION = "26754ae52c26c097dc1c465a1e42c4c5d05a3d40"

source_before = SOURCE.read_bytes()
subprocess.run(
    ["cargo", "oxide", "build", "--arch", "sm_75", "--", "--release", "--locked"],
    cwd=HERE,
    check=True,
)
if SOURCE.read_bytes() != source_before:
    raise SystemExit("Rust source changed during compilation; refusing to publish PTX")
ptx = (HERE / "xabe_rust_device.ptx").read_text()
if ".target sm_75\n" not in ptx or any(
    f".visible .entry {entry}(" not in ptx
    for entry in ("tensor_add", "swiglu_mul", "sigmoid_gate_mul")
):
    raise SystemExit("compiler output has the wrong target or entry point")
header = (
    "// Generated from experiments/cuda-oxide/src/main.rs; do not edit.\n"
    f"// cuda-oxide: {REVISION}; nightly-2026-08-28; sm_75\n"
    f"// Rust source SHA-256: {hashlib.sha256(source_before).hexdigest()}\n"
)
OUTPUT.write_text(header + ptx)
