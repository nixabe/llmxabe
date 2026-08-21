//! Reference forward pass through the vision tower and projector.
//!
//! Weight layout follows GGUF/ggml convention: a tensor with dims
//! `[in, out]` is row-major with `out` rows of length `in`, so
//! `w[o * in + i]` multiplies input element `i` into output element `o` —
//! matching `ggml_mul_mat(w, x)`.

use crate::vision::cell_order_index;
use xabe_model::VisionConfig;

/// One transformer block's weights, f32.
#[derive(Debug, Clone)]
pub struct VisionBlockWeights {
    /// `v.blk.N.ln1.weight` / `.bias` — LayerNorm before attention.
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    /// `v.blk.N.attn_qkv.weight` `[hidden, 3*hidden]` — rows `[Q | K | V]`.
    pub qkv_w: Vec<f32>,
    pub qkv_b: Vec<f32>,
    /// `v.blk.N.attn_out.weight` `[hidden, hidden]`.
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
    /// `v.blk.N.ln2.weight` / `.bias` — LayerNorm before the MLP.
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    /// `v.blk.N.ffn_up.weight` `[hidden, ffn]`.
    pub up_w: Vec<f32>,
    pub up_b: Vec<f32>,
    /// `v.blk.N.ffn_down.weight` `[ffn, hidden]`.
    pub down_w: Vec<f32>,
    pub down_b: Vec<f32>,
}

/// The whole tower's weights, f32.
///
/// `patch_embed` is the *sum* of the two temporal conv slices
/// (`v.patch_embd.weight` + `v.patch_embd.weight.1`): a still image feeds
/// the same frame through both and the results are added
/// (`build_inp_with_temporal_merge`, `tools/mtmd/models/qwen2vl.cpp`), so
/// for image-only serving the sum can be folded at load time.
#[derive(Debug, Clone)]
pub struct VisionWeights {
    /// Summed patch conv, `[patch_elems/2, hidden]` = `[768, 1152]` rows.
    /// Row layout per patch: `x + 16*y + 256*c`.
    pub patch_embed: Vec<f32>,
    /// `v.patch_embd.bias`.
    pub patch_bias: Vec<f32>,
    /// `v.position_embd.weight` `[hidden, 48*48]` — raw grid order,
    /// row-major over the 48×48 grid.
    pub pos_embed: Vec<f32>,
    /// The 27 blocks.
    pub blocks: Vec<VisionBlockWeights>,
    /// `v.post_ln.weight` / `.bias`.
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Vec<f32>,
    /// `mm.0.weight` `[4608, 4608]` / `.bias`.
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    /// `mm.2.weight` `[4608, 2048]` / `.bias`.
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
}

/// ggml's tanh-approximation GELU (`ggml_gelu_f32`,
/// `ggml/src/ggml-cpu/vec.h`), computed exactly in f32.
///
/// llama.cpp's CPU backend additionally rounds input and output through a
/// f16 lookup table (`GGML_GELU_FP16`); its CUDA backend computes this
/// formula directly. The reference matches the formula — golden
/// comparisons against the CPU backend must budget for the table's f16
/// rounding.
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEF_A: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + COEF_A * x * x)).tanh())
}

/// Full LayerNorm with weight and bias over the last dimension.
fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(w.iter().zip(b))
        .map(|(&v, (&w, &b))| (v - mean) * inv * w + b)
        .collect()
}

/// `out[o] = Σ_i w[o*in + i] * x[i] + b[o]`.
fn matvec(w: &[f32], b: &[f32], x: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    assert_eq!(w.len(), in_dim * out_dim);
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x).map(|(&w, &x)| w * x).sum::<f32>() + b[o]
        })
        .collect()
}

/// Vision M-RoPE over one 72-dim head vector, in place.
///
/// All `head_dim` dimensions rotate, NEOX pairs `(j, j+36)`. Pairs
/// `j < 18` rotate by the patch row with frequency `theta^(-j/18)`; pairs
/// `j >= 18` by the patch column with frequency restarting at
/// `theta^(-(j-18)/18)` — the "independent sections" behaviour of
/// `GGML_ROPE_TYPE_VISION` (`ggml_mrope_cache_init`,
/// `ggml/src/ggml-cpu/ops.cpp`) with sections `[18,18,18,18]`, of which
/// only the first two are reachable.
fn vision_rope(head: &mut [f32], row: u32, col: u32, theta_base: f32) {
    let half = head.len() / 2; // 36
    let quarter = half / 2; // 18
    for j in 0..half {
        let (pos, k) = if j < quarter {
            (row, j)
        } else {
            (col, j - quarter)
        };
        let freq = f64::from(theta_base).powf(-(k as f64) / quarter as f64);
        let angle = f64::from(pos) * freq;
        let (sin_a, cos_a) = (angle.sin() as f32, angle.cos() as f32);
        let x0 = head[j];
        let x1 = head[j + half];
        head[j] = x0 * cos_a - x1 * sin_a;
        head[j + half] = x0 * sin_a + x1 * cos_a;
    }
}

/// Bilinear align-corners resize of the learned 48×48 position grid to
/// `(grid_h, grid_w)` patches, returned in **cell order** ready to add.
///
/// Matches `ggml_interpolate(..., GGML_SCALE_MODE_BILINEAR |
/// GGML_SCALE_FLAG_ALIGN_CORNERS)` as called from
/// `clip_graph::resize_position_embeddings` (`tools/mtmd/clip.cpp`):
/// `src = i * (n-1) / (out-1)`, corners exact. When the target grid *is*
/// 48×48 the graph skips interpolation entirely; the formula degenerates
/// to the identity there, so no special case is needed.
pub fn resized_pos_embed(
    pos_embed: &[f32],
    hidden: usize,
    grid_edge: usize,
    grid_h: u32,
    grid_w: u32,
) -> Vec<f32> {
    let n_patches = (grid_h * grid_w) as usize;
    let mut out = vec![0.0f32; n_patches * hidden];
    assert!(grid_h >= 2 && grid_w >= 2, "min image edge is 2 patches");
    let sy = (grid_edge - 1) as f64 / (grid_h - 1) as f64;
    let sx = (grid_edge - 1) as f64 / (grid_w - 1) as f64;
    for y in 0..grid_h {
        let fy = y as f64 * sy;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(grid_edge - 1);
        let dy = (fy - y0 as f64) as f32;
        for x in 0..grid_w {
            let fx = x as f64 * sx;
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(grid_edge - 1);
            let dx = (fx - x0 as f64) as f32;
            let dst = cell_order_index(x, y, grid_w) * hidden;
            let (r00, r10) = (
                &pos_embed[(y0 * grid_edge + x0) * hidden..][..hidden],
                &pos_embed[(y0 * grid_edge + x1) * hidden..][..hidden],
            );
            let (r01, r11) = (
                &pos_embed[(y1 * grid_edge + x0) * hidden..][..hidden],
                &pos_embed[(y1 * grid_edge + x1) * hidden..][..hidden],
            );
            for c in 0..hidden {
                let top = r00[c] * (1.0 - dx) + r10[c] * dx;
                let bot = r01[c] * (1.0 - dx) + r11[c] * dx;
                out[dst + c] = top * (1.0 - dy) + bot * dy;
            }
        }
    }
    out
}

/// Encode one image through the tower and projector.
///
/// `patches` holds `grid_h * grid_w` flattened normalized patches in
/// **cell order** (see [`cell_order_index`]), each of
/// `3 * patch_size * patch_size` elements laid out `x + 16*y + 256*c` to
/// match the conv weight. Returns `output_tokens * projection_dim` f32
/// embeddings, row-major over merged 2×2 cells.
pub fn encode(
    cfg: &VisionConfig,
    w: &VisionWeights,
    patches: &[f32],
    grid_h: u32,
    grid_w: u32,
) -> Vec<f32> {
    let hidden = cfg.hidden_size as usize;
    let n_patches = (grid_h * grid_w) as usize;
    let patch_len = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
    assert_eq!(patches.len(), n_patches * patch_len, "patch buffer size");
    assert!(
        grid_h.is_multiple_of(cfg.spatial_merge) && grid_w.is_multiple_of(cfg.spatial_merge),
        "grid must be a multiple of the merge size"
    );

    // Patch embedding (summed temporal conv) + bias + position embedding.
    let pos = resized_pos_embed(
        &w.pos_embed,
        hidden,
        cfg.pos_grid_edge() as usize,
        grid_h,
        grid_w,
    );
    let mut x: Vec<Vec<f32>> = (0..n_patches)
        .map(|p| {
            let mut row = matvec(
                &w.patch_embed,
                &w.patch_bias,
                &patches[p * patch_len..(p + 1) * patch_len],
                hidden,
            );
            for (c, r) in row.iter_mut().enumerate() {
                *r += pos[p * hidden + c];
            }
            row
        })
        .collect();

    // Patch (row, col) per sequence position — the same walk that fills the
    // `positions` input in clip.cpp, inverted through cell_order_index.
    let mut coords = vec![(0u32, 0u32); n_patches];
    for y in 0..grid_h {
        for x_ in 0..grid_w {
            coords[cell_order_index(x_, y, grid_w)] = (y, x_);
        }
    }

    let heads = cfg.num_heads as usize;
    let d_head = cfg.head_dim() as usize;
    let scale = 1.0 / (d_head as f32).sqrt();

    for blk in &w.blocks {
        // Attention.
        let mut q = vec![vec![0.0f32; hidden]; n_patches];
        let mut k = vec![vec![0.0f32; hidden]; n_patches];
        let mut v = vec![vec![0.0f32; hidden]; n_patches];
        for p in 0..n_patches {
            let normed = layer_norm(&x[p], &blk.ln1_w, &blk.ln1_b, cfg.ln_eps);
            let qkv = matvec(&blk.qkv_w, &blk.qkv_b, &normed, 3 * hidden);
            q[p].copy_from_slice(&qkv[..hidden]);
            k[p].copy_from_slice(&qkv[hidden..2 * hidden]);
            v[p].copy_from_slice(&qkv[2 * hidden..]);
            let (row, col) = coords[p];
            for h in 0..heads {
                vision_rope(
                    &mut q[p][h * d_head..(h + 1) * d_head],
                    row,
                    col,
                    cfg.rope_theta,
                );
                vision_rope(
                    &mut k[p][h * d_head..(h + 1) * d_head],
                    row,
                    col,
                    cfg.rope_theta,
                );
            }
        }

        for p in 0..n_patches {
            let mut attn_out = vec![0.0f32; hidden];
            for h in 0..heads {
                let qh = &q[p][h * d_head..(h + 1) * d_head];
                // Bidirectional, unmasked scores over every patch.
                let mut scores: Vec<f32> = (0..n_patches)
                    .map(|j| {
                        let kh = &k[j][h * d_head..(h + 1) * d_head];
                        qh.iter().zip(kh).map(|(&a, &b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    total += *s;
                }
                for (j, &s) in scores.iter().enumerate() {
                    let vh = &v[j][h * d_head..(h + 1) * d_head];
                    let weight = s / total;
                    for (c, &vv) in vh.iter().enumerate() {
                        attn_out[h * d_head + c] += weight * vv;
                    }
                }
            }
            let projected = matvec(&blk.out_w, &blk.out_b, &attn_out, hidden);
            for (c, pr) in projected.iter().enumerate() {
                x[p][c] += pr;
            }
        }

        // MLP.
        for row in x.iter_mut() {
            let normed = layer_norm(row, &blk.ln2_w, &blk.ln2_b, cfg.ln_eps);
            let mut up = matvec(&blk.up_w, &blk.up_b, &normed, cfg.ffn_size as usize);
            for u in up.iter_mut() {
                *u = gelu(*u);
            }
            let down = matvec(&blk.down_w, &blk.down_b, &up, hidden);
            for (c, d) in down.iter().enumerate() {
                row[c] += d;
            }
        }
    }

    // Post-LN, 2×2 merge of consecutive positions, projector.
    let merge = cfg.merge_factor() as usize;
    let merged_dim = cfg.merger_input_dim() as usize;
    let proj_dim = cfg.projection_dim as usize;
    let n_tokens = n_patches / merge;
    let mut out = Vec::with_capacity(n_tokens * proj_dim);
    for m in 0..n_tokens {
        let mut merged = Vec::with_capacity(merged_dim);
        for row in x.iter().skip(m * merge).take(merge) {
            merged.extend(layer_norm(row, &w.post_ln_w, &w.post_ln_b, cfg.ln_eps));
        }
        let mut h = matvec(&w.fc1_w, &w.fc1_b, &merged, merged_dim);
        for u in h.iter_mut() {
            *u = gelu(*u);
        }
        out.extend(matvec(&w.fc2_w, &w.fc2_b, &h, proj_dim));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_matches_known_values() {
        // gelu(0)=0, gelu is odd-ish: gelu(x)+gelu(-x)=x·1... spot values
        // against the published tanh approximation.
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841192).abs() < 1e-5);
        assert!((gelu(-1.0) + 0.158808).abs() < 1e-5);
        assert!((gelu(3.0) - 2.996363).abs() < 1e-5);
    }

    #[test]
    fn layer_norm_zero_means_and_unit_scales() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let b = vec![0.0; 4];
        let out = layer_norm(&x, &w, &b, 0.0);
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-6);
        assert!((var - 1.0).abs() < 1e-5);
    }

    #[test]
    fn vision_rope_at_origin_is_identity() {
        let mut head: Vec<f32> = (0..72).map(|i| (i as f32 * 0.31).sin()).collect();
        let orig = head.clone();
        vision_rope(&mut head, 0, 0, 10_000.0);
        assert_eq!(head, orig);
    }

    #[test]
    fn vision_rope_row_only_moves_first_half_pairs() {
        let base: Vec<f32> = (0..72).map(|i| (i as f32 * 0.17).cos()).collect();
        let mut moved = base.clone();
        vision_rope(&mut moved, 5, 0, 10_000.0);
        for j in 0..36 {
            let changed = moved[j] != base[j] || moved[j + 36] != base[j + 36];
            assert_eq!(changed, j < 18, "pair {j}");
        }
    }

    #[test]
    fn pos_embed_resize_to_native_grid_is_identity() {
        // grid_edge=4 stand-in: resizing 4×4 -> 4×4 must reproduce the
        // grid exactly (modulo the cell reorder).
        let hidden = 2usize;
        let edge = 4usize;
        let grid: Vec<f32> = (0..edge * edge * hidden).map(|i| i as f32).collect();
        let out = resized_pos_embed(&grid, hidden, edge, 4, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let dst = cell_order_index(x, y, 4) * hidden;
                let src = (y as usize * edge + x as usize) * hidden;
                assert_eq!(&out[dst..dst + hidden], &grid[src..src + hidden]);
            }
        }
    }

    #[test]
    fn pos_embed_resize_corners_are_exact() {
        // Align-corners: output corners equal input corners regardless of
        // scale.
        let hidden = 1usize;
        let edge = 48usize;
        let grid: Vec<f32> = (0..edge * edge).map(|i| (i as f32).sqrt()).collect();
        let (gh, gw) = (6u32, 10u32);
        let out = resized_pos_embed(&grid, hidden, edge, gh, gw);
        let corner = |x: u32, y: u32| out[cell_order_index(x, y, gw)];
        assert_eq!(corner(0, 0), grid[0]);
        assert_eq!(corner(gw - 1, 0), grid[edge - 1]);
        assert_eq!(corner(0, gh - 1), grid[(edge - 1) * edge]);
        assert_eq!(corner(gw - 1, gh - 1), grid[edge * edge - 1]);
    }

    fn tiny_cfg() -> VisionConfig {
        VisionConfig {
            num_layers: 2,
            hidden_size: 8,
            num_heads: 2,
            ffn_size: 16,
            image_size: 64,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge: 2,
            projection_dim: 6,
            ln_eps: 1e-6,
            rope_theta: 10_000.0,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        }
    }

    fn tiny_weights(cfg: &VisionConfig, seed: u64) -> VisionWeights {
        use crate::rng::Xorshift64Star;
        let mut rng = Xorshift64Star::new(seed);
        let h = cfg.hidden_size as usize;
        let ffn = cfg.ffn_size as usize;
        let patch = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
        let edge = cfg.pos_grid_edge() as usize;
        let merged = cfg.merger_input_dim() as usize;
        let proj = cfg.projection_dim as usize;
        let mut v = |n: usize| rng.vec_f32(n, -0.1, 0.1);
        VisionWeights {
            patch_embed: v(patch * h),
            patch_bias: v(h),
            pos_embed: v(edge * edge * h),
            blocks: (0..cfg.num_layers)
                .map(|_| VisionBlockWeights {
                    ln1_w: v(h).iter().map(|x| 1.0 + x).collect(),
                    ln1_b: v(h),
                    qkv_w: v(h * 3 * h),
                    qkv_b: v(3 * h),
                    out_w: v(h * h),
                    out_b: v(h),
                    ln2_w: v(h).iter().map(|x| 1.0 + x).collect(),
                    ln2_b: v(h),
                    up_w: v(h * ffn),
                    up_b: v(ffn),
                    down_w: v(ffn * h),
                    down_b: v(h),
                })
                .collect(),
            post_ln_w: v(h).iter().map(|x| 1.0 + x).collect(),
            post_ln_b: v(h),
            fc1_w: v(merged * merged),
            fc1_b: v(merged),
            fc2_w: v(merged * proj),
            fc2_b: v(proj),
        }
    }

    #[test]
    fn encode_produces_the_expected_shape() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, 3);
        let patch = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
        let (gh, gw) = (2u32, 4u32);
        let patches = vec![0.25f32; (gh * gw) as usize * patch];
        let out = encode(&cfg, &w, &patches, gh, gw);
        assert_eq!(
            out.len(),
            cfg.output_tokens(gh, gw) as usize * cfg.projection_dim as usize
        );
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn encode_is_deterministic_and_input_sensitive() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, 5);
        let patch = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
        let patches = vec![0.5f32; 8 * patch];
        let a = encode(&cfg, &w, &patches, 2, 4);
        let b = encode(&cfg, &w, &patches, 2, 4);
        assert_eq!(a, b);
        let mut patches2 = patches;
        patches2[0] += 0.1;
        let c = encode(&cfg, &w, &patches2, 2, 4);
        assert_ne!(a, c);
    }

    #[test]
    fn attention_mixes_across_every_patch() {
        // Bidirectional attention: perturbing the LAST patch must change
        // the FIRST output token. A causal implementation would not.
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, 9);
        let patch = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
        let patches = vec![0.1f32; 8 * patch];
        let a = encode(&cfg, &w, &patches, 2, 4);
        let mut patches2 = patches;
        let last = patches2.len() - 1;
        patches2[last] = 0.9;
        let b = encode(&cfg, &w, &patches2, 2, 4);
        let first_token = cfg.projection_dim as usize;
        assert_ne!(&a[..first_token], &b[..first_token]);
    }
}
