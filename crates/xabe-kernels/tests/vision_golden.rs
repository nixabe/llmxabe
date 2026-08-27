//! Cross-implementation check of the vision-tower reference against
//! llama.cpp's own execution of the same mmproj file.
//!
//! llama.cpp's `llama-mtmd-debug` tool feeds a synthetic f32 image
//! (bypassing preprocessing) through the clip graph and, in a debug-callback
//! build, prints every intermediate tensor with a per-tensor sum. This test
//! reconstructs the same synthetic inputs, runs [`xabe_kernels::vision`]'s
//! reference `encode`, and compares the final projector output against the
//! last tensor block of those logs — value by printed value, plus the sum.
//!
//! Captured goldens are never committed (AGENTS.md), so the logs are read
//! from `LLMXABE_MTMD_GOLDEN_DIR` at run time and the test SKIPS without
//! it. To generate, for `IMG` in `gray`/`red`/`cb` and `N` in `64`/`96`:
//!
//! ```sh
//! llama-mtmd-debug -m <model.gguf> --mmproj <mmproj-F16.gguf> \
//!   -p encode --image $IMG -n $N --no-mmproj-offload -ngl 0 --no-warmup \
//!   > $LLMXABE_MTMD_GOLDEN_DIR/golden-$IMG-$N.log 2>&1
//! ```
//!
//! Tolerances budget for llama.cpp's f16 GELU lookup table
//! (`GGML_GELU_FP16`) and SIMD summation order; the weights themselves are
//! identical f16 values on both sides.

use std::path::PathBuf;

use half::f16;
use xabe_gguf::GgufFile;
use xabe_kernels::vision::{self, cell_order_index};
use xabe_model::VisionConfig;
use xabe_model::vision::{VisionRole, VisionWeightSchema};

const DEFAULT_MMPROJ_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf";

fn mmproj_path() -> PathBuf {
    std::env::var_os("LLMXABE_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MMPROJ_PATH))
}

/// Convert one GGUF tensor to f32, whatever of the two mmproj types it is.
fn tensor_f32(file: &GgufFile, name: &str) -> Vec<f32> {
    let info = file
        .tensor(name)
        .unwrap_or_else(|| panic!("missing {name}"));
    let bytes = file.tensor_bytes(name).unwrap();
    match info.ggml_type {
        xabe_gguf::GgmlType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        xabe_gguf::GgmlType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => panic!("{name}: unexpected type {other:?}"),
    }
}

fn load_weights(file: &GgufFile, cfg: &VisionConfig) -> vision::VisionWeights {
    use VisionRole::*;
    let schema = VisionWeightSchema::new(cfg);
    schema.resolve(file).expect("mmproj must match the schema");
    let name =
        |role: VisionRole, layer: Option<u32>| schema.find(role, layer).unwrap().name.clone();
    let global = |role: VisionRole| tensor_f32(file, &name(role, None));

    // Still images run the same frame through both temporal conv slices and
    // sum the results, so the two weights fold into one matrix.
    let mut patch_embed = global(PatchEmbed0);
    for (a, b) in patch_embed.iter_mut().zip(global(PatchEmbed1)) {
        *a += b;
    }

    vision::VisionWeights {
        patch_embed,
        patch_bias: global(PatchBias),
        pos_embed: global(PositionEmbed),
        blocks: (0..cfg.num_layers)
            .map(|l| {
                let t = |role: VisionRole| tensor_f32(file, &name(role, Some(l)));
                vision::VisionBlockWeights {
                    ln1_w: t(Ln1Weight),
                    ln1_b: t(Ln1Bias),
                    qkv_w: t(AttnQkvWeight),
                    qkv_b: t(AttnQkvBias),
                    out_w: t(AttnOutWeight),
                    out_b: t(AttnOutBias),
                    ln2_w: t(Ln2Weight),
                    ln2_b: t(Ln2Bias),
                    up_w: t(FfnUpWeight),
                    up_b: t(FfnUpBias),
                    down_w: t(FfnDownWeight),
                    down_b: t(FfnDownBias),
                }
            })
            .collect(),
        post_ln_w: global(PostLnWeight),
        post_ln_b: global(PostLnBias),
        fc1_w: global(MergerFc1Weight),
        fc1_b: global(MergerFc1Bias),
        fc2_w: global(MergerFc2Weight),
        fc2_b: global(MergerFc2Bias),
    }
}

/// Reproduce mtmd-debug's synthetic image: per-pixel f32 RGB in `[0,1]`,
/// fed to the graph as-is (no normalization).
fn synthetic_pixel(kind: &str, x: u32, y: u32) -> [f32; 3] {
    match kind {
        "gray" => [0.5, 0.5, 0.5],
        "red" => [1.0, 0.0, 0.0],
        "cb" => {
            let v = if (x + y) % 2 == 1 { 0.0 } else { 1.0 };
            [v, v, v]
        }
        other => panic!("unknown synthetic image {other}"),
    }
}

/// Patchify a synthetic image into cell order with the conv layout.
fn synthetic_patches(cfg: &VisionConfig, kind: &str, edge: u32) -> Vec<f32> {
    let patch = cfg.patch_size;
    let grid = edge / patch;
    let plane = (patch * patch) as usize;
    let patch_len = 3 * plane;
    let mut out = vec![0.0f32; (grid * grid) as usize * patch_len];
    for gy in 0..grid {
        for gx in 0..grid {
            let base = cell_order_index(gx, gy, grid) * patch_len;
            for py in 0..patch {
                for px in 0..patch {
                    let rgb = synthetic_pixel(kind, gx * patch + px, gy * patch + py);
                    for (c, &v) in rgb.iter().enumerate() {
                        out[base + (px + patch * py) as usize + plane * c] = v;
                    }
                }
            }
        }
    }
    out
}

/// The printed values of the last tensor block in a mtmd-debug log:
/// per-token first three and last three values (tokens beyond the
/// printer's window are elided with `...`), and the block sum.
struct GoldenBlock {
    /// `(token_index, first_three, last_three)`.
    rows: Vec<(usize, [f32; 3], [f32; 3])>,
    sum: f64,
    n_embd: usize,
    n_tokens: usize,
}

fn parse_last_block(log: &str) -> GoldenBlock {
    // Blocks look like:
    //   common_debug_cb_eval: NAME = (f32) OP(...) = {2048, 4, 1, 1}
    //       [ [ [ a, b, c, ..., x, y, z ], ... ] ]
    //       sum = S
    let mut blocks: Vec<&str> = log.split("common_debug_cb_eval:").collect();
    let last = blocks.pop().expect("log holds at least one tensor block");
    let header = last.lines().next().unwrap();
    let shape: Vec<usize> = header
        .rsplit_once('{')
        .expect("shape in header")
        .1
        .trim_end_matches(['}', '\n'])
        .split(',')
        .filter_map(|t| t.trim().trim_end_matches('}').parse().ok())
        .collect();
    let (n_embd, n_tokens) = (shape[0], shape[1]);

    // Value rows print as `[ a, b, c, ..., x, y, z ],`; a bare `...,` row
    // marks elided middle tokens, after which the remaining rows are the
    // LAST tokens of the tensor.
    let mut leading: Vec<([f32; 3], [f32; 3])> = Vec::new();
    let mut trailing: Vec<([f32; 3], [f32; 3])> = Vec::new();
    let mut elided = false;
    for line in last.lines() {
        let t = line.trim();
        if t == "...," {
            elided = true;
            continue;
        }
        if t.starts_with('[') && t.contains("...") {
            let nums: Vec<f32> = t
                .trim_start_matches('[')
                .trim_end_matches([']', ',', ' '])
                .split(',')
                .filter_map(|tok| tok.trim().parse().ok())
                .collect();
            if nums.len() == 6 {
                let row = ([nums[0], nums[1], nums[2]], [nums[3], nums[4], nums[5]]);
                if elided {
                    trailing.push(row);
                } else {
                    leading.push(row);
                }
            }
        }
    }
    let mut rows: Vec<(usize, [f32; 3], [f32; 3])> = leading
        .into_iter()
        .enumerate()
        .map(|(i, (a, b))| (i, a, b))
        .collect();
    let n_trailing = trailing.len();
    rows.extend(
        trailing
            .into_iter()
            .enumerate()
            .map(|(i, (a, b))| (n_tokens - n_trailing + i, a, b)),
    );
    let sum: f64 = last
        .lines()
        .find_map(|l| l.trim().strip_prefix("sum = "))
        .expect("block sum")
        .trim()
        .parse()
        .unwrap();
    GoldenBlock {
        rows,
        sum,
        n_embd,
        n_tokens,
    }
}

#[test]
fn reference_tower_matches_llama_cpp_on_synthetic_images() {
    let Some(dir) = std::env::var_os("LLMXABE_MTMD_GOLDEN_DIR").map(PathBuf::from) else {
        println!("SKIPPED: LLMXABE_MTMD_GOLDEN_DIR not set; see module docs to generate goldens");
        return;
    };
    let path = mmproj_path();
    if !path.exists() {
        println!("SKIPPED: mmproj not found at {}", path.display());
        return;
    }

    let cfg = VisionConfig::qwen3_6_35b_a3b();
    let file = GgufFile::open(&path).expect("mmproj parses");
    let weights = load_weights(&file, &cfg);

    let mut compared = 0usize;
    for kind in ["gray", "red", "cb"] {
        for edge in [64u32, 96] {
            let log_path = dir.join(format!("golden-{kind}-{edge}.log"));
            let Ok(log) = std::fs::read_to_string(&log_path) else {
                println!("skipping {}: not present", log_path.display());
                continue;
            };
            let golden = parse_last_block(&log);
            let grid = edge / cfg.patch_size;
            assert_eq!(golden.n_embd, cfg.projection_dim as usize);
            assert_eq!(golden.n_tokens, cfg.output_tokens(grid, grid) as usize);

            let patches = synthetic_patches(&cfg, kind, edge);
            let out = vision::encode(&cfg, &weights, &patches, grid, grid);
            let n_embd = golden.n_embd;

            let sum: f64 = out.iter().map(|&v| f64::from(v)).sum();
            let sum_scale = out.iter().map(|&v| f64::from(v).abs()).sum::<f64>();
            let sum_err = (sum - golden.sum).abs() / sum_scale.max(1.0);
            println!(
                "{kind}-{edge}: sum {sum:.4} vs golden {:.4} (rel-to-mass {sum_err:.2e})",
                golden.sum
            );
            assert!(
                sum_err < 2e-3,
                "{kind}-{edge}: output mass diverges from llama.cpp"
            );

            assert!(
                !golden.rows.is_empty(),
                "no value rows parsed from {}",
                log_path.display()
            );
            for &(tok, first, last) in &golden.rows {
                let row = &out[tok * n_embd..(tok + 1) * n_embd];
                for (i, &g) in first.iter().enumerate() {
                    let d = (row[i] - g).abs();
                    assert!(
                        d < 5e-3,
                        "{kind}-{edge} token {tok} value {i}: {} vs {g}",
                        row[i]
                    );
                }
                for (i, &g) in last.iter().enumerate() {
                    let idx = n_embd - 3 + i;
                    let d = (row[idx] - g).abs();
                    assert!(
                        d < 5e-3,
                        "{kind}-{edge} token {tok} value {idx}: {} vs {g}",
                        row[idx]
                    );
                }
            }
            compared += 1;
        }
    }
    assert!(compared > 0, "golden dir set but no logs found");
    println!("compared {compared} golden logs");
}
