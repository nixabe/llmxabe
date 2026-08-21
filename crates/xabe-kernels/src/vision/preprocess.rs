//! Image preprocessing: smart-resize, bilinear rescale, normalize,
//! patchify.
//!
//! Ported from llama.cpp `tools/mtmd/mtmd-image.cpp`:
//! `img_tool::calc_size_preserved_ratio` (the HF `smart_resize`),
//! `img_tool::resize` with `PAD_CEIL` (centered black padding), and
//! `resize_bilinear` (align-corners on u8 with truncation). Normalization
//! order is `tools/mtmd/clip-impl.h`: scale to `[0,1]` first, then
//! `(x - mean) / std` per channel.

use crate::vision::cell_order_index;
use xabe_model::VisionConfig;

/// Pixel-count bounds, in *output tokens* worth of pixels.
///
/// llama.cpp sets `set_limit_image_tokens(8, 4096)` for `qwen3vl_merger`
/// (`tools/mtmd/clip.cpp`): one output token covers
/// `patch^2 * merge^2 = 1024` pixels, so images are scaled into
/// `[8*1024, 4096*1024]` pixels before patch alignment.
pub const MIN_IMAGE_TOKENS: u32 = 8;
pub const MAX_IMAGE_TOKENS: u32 = 4096;

/// The multiple every image edge is aligned to: `patch * merge` = 32 px.
pub fn align_edge(cfg: &VisionConfig) -> u32 {
    cfg.patch_size * cfg.spatial_merge
}

fn round_by(v: f64, f: f64) -> u32 {
    ((v / f).round() * f) as u32
}
fn ceil_by(v: f64, f: f64) -> u32 {
    ((v / f).ceil() * f) as u32
}
fn floor_by(v: f64, f: f64) -> u32 {
    ((v / f).floor() * f) as u32
}

/// Target size for a `w × h` input: aspect-preserving, aligned to 32,
/// clamped into the pixel budget. Both beta branches divide the
/// **original** dimensions, matching upstream exactly.
pub fn smart_resize(cfg: &VisionConfig, w: u32, h: u32) -> (u32, u32) {
    let f = f64::from(align_edge(cfg));
    let token_px = f64::from(align_edge(cfg) * align_edge(cfg));
    let min_px = f64::from(MIN_IMAGE_TOKENS) * token_px;
    let max_px = f64::from(MAX_IMAGE_TOKENS) * token_px;
    let (wf, hf) = (f64::from(w), f64::from(h));

    let mut w_bar = round_by(wf, f).max(align_edge(cfg));
    let mut h_bar = round_by(hf, f).max(align_edge(cfg));
    let area = f64::from(w_bar) * f64::from(h_bar);
    if area > max_px {
        let beta = (wf * hf / max_px).sqrt();
        h_bar = floor_by(hf / beta, f).max(align_edge(cfg));
        w_bar = floor_by(wf / beta, f).max(align_edge(cfg));
    } else if area < min_px {
        let beta = (min_px / (wf * hf)).sqrt();
        h_bar = ceil_by(hf * beta, f);
        w_bar = ceil_by(wf * beta, f);
    }
    (w_bar, h_bar)
}

/// Bilinear resize of interleaved RGB u8, align-corners, truncating to u8 —
/// `img_tool::resize_bilinear` in `tools/mtmd/mtmd-image.cpp`.
fn resize_bilinear(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; (dw * dh * 3) as usize];
    let x_ratio = if dw > 1 {
        (sw - 1) as f32 / (dw - 1) as f32
    } else {
        0.0
    };
    let y_ratio = if dh > 1 {
        (sh - 1) as f32 / (dh - 1) as f32
    } else {
        0.0
    };
    for y in 0..dh {
        let py = y as f32 * y_ratio;
        let y0 = py.floor() as u32;
        let y1 = (y0 + 1).min(sh - 1);
        let yf = py - y0 as f32;
        for x in 0..dw {
            let px = x as f32 * x_ratio;
            let x0 = px.floor() as u32;
            let x1 = (x0 + 1).min(sw - 1);
            let xf = px - x0 as f32;
            for c in 0..3u32 {
                let at = |xx: u32, yy: u32| src[((yy * sw + xx) * 3 + c) as usize] as f32;
                let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * xf;
                let bot = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * xf;
                out[((y * dw + x) * 3 + c) as usize] = (top + (bot - top) * yf) as u8;
            }
        }
    }
    out
}

/// A preprocessed image, ready for the tower.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedImage {
    /// Flattened normalized patches in cell order, each
    /// `3 * patch^2` elements laid out `x + patch*y + patch^2*c`.
    pub patches: Vec<f32>,
    /// Patch-grid height (rows of patches).
    pub grid_h: u32,
    /// Patch-grid width.
    pub grid_w: u32,
}

impl PreprocessedImage {
    /// Language-model tokens this image expands to after the 2×2 merge.
    pub fn output_tokens(&self, cfg: &VisionConfig) -> u32 {
        cfg.output_tokens(self.grid_h, self.grid_w)
    }
}

/// Preprocess an interleaved RGB8 image of size `w × h`.
///
/// Pipeline (each step cited in the module docs): smart-resize target,
/// aspect-preserving bilinear rescale, centered black padding to the
/// target, `[0,1]` scaling, mean/std normalization, patchify into cell
/// order.
pub fn preprocess(cfg: &VisionConfig, rgb: &[u8], w: u32, h: u32) -> PreprocessedImage {
    assert_eq!(rgb.len(), (w * h * 3) as usize, "interleaved RGB8 expected");
    assert!(w > 0 && h > 0);
    let (tw, th) = smart_resize(cfg, w, h);

    // Aspect-preserving fit, then centered pad with black — the PAD_CEIL
    // branch of img_tool::resize.
    let scale = (f64::from(tw) / f64::from(w)).min(f64::from(th) / f64::from(h));
    let new_w = ((f64::from(w) * scale).ceil() as u32).min(tw);
    let new_h = ((f64::from(h) * scale).ceil() as u32).min(th);
    let scaled = if (new_w, new_h) == (w, h) {
        rgb.to_vec()
    } else {
        resize_bilinear(rgb, w, h, new_w, new_h)
    };
    let (off_x, off_y) = ((tw - new_w) / 2, (th - new_h) / 2);
    let mut canvas = vec![0u8; (tw * th * 3) as usize];
    for y in 0..new_h {
        let src = ((y * new_w) * 3) as usize;
        let dst = (((y + off_y) * tw + off_x) * 3) as usize;
        canvas[dst..dst + (new_w * 3) as usize]
            .copy_from_slice(&scaled[src..src + (new_w * 3) as usize]);
    }

    // Normalize and patchify.
    let patch = cfg.patch_size;
    let grid_w = tw / patch;
    let grid_h = th / patch;
    let patch_len = (3 * patch * patch) as usize;
    let mut patches = vec![0.0f32; (grid_w * grid_h) as usize * patch_len];
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let base = cell_order_index(gx, gy, grid_w) * patch_len;
            for c in 0..3u32 {
                let mean = cfg.image_mean[c as usize];
                let std = cfg.image_std[c as usize];
                for py in 0..patch {
                    for px in 0..patch {
                        let sx = gx * patch + px;
                        let sy = gy * patch + py;
                        let v = canvas[((sy * tw + sx) * 3 + c) as usize] as f32 / 255.0;
                        patches[base + (px + patch * py + patch * patch * c) as usize] =
                            (v - mean) / std;
                    }
                }
            }
        }
    }

    PreprocessedImage {
        patches,
        grid_h,
        grid_w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VisionConfig {
        VisionConfig::qwen3_6_35b_a3b()
    }

    #[test]
    fn already_aligned_sizes_pass_through() {
        assert_eq!(smart_resize(&cfg(), 768, 768), (768, 768));
        assert_eq!(smart_resize(&cfg(), 96, 128), (96, 128));
    }

    #[test]
    fn sizes_round_to_the_nearest_multiple_of_32() {
        assert_eq!(smart_resize(&cfg(), 100, 100), (96, 96));
        assert_eq!(smart_resize(&cfg(), 113, 97), (128, 96));
    }

    #[test]
    fn images_below_the_pixel_floor_scale_up() {
        // 64x64 = 4096 px < 8192 px minimum: beta = sqrt(2), 64*sqrt(2) ≈
        // 90.5, ceil to 96 on both edges.
        assert_eq!(smart_resize(&cfg(), 64, 64), (96, 96));
    }

    #[test]
    fn tiny_images_scale_up_to_the_minimum_budget() {
        // 10x10 = 100 px < 8192 px minimum; beta = sqrt(8192/100) ≈ 9.05,
        // 10*9.05 = 90.5 -> ceil to 96. 96*96 = 9216 >= 8192.
        let (w, h) = smart_resize(&cfg(), 10, 10);
        assert_eq!((w, h), (96, 96));
        assert!(w * h >= MIN_IMAGE_TOKENS * 1024);
    }

    #[test]
    fn huge_images_scale_down_to_the_maximum_budget() {
        let (w, h) = smart_resize(&cfg(), 10_000, 10_000);
        assert!(w * h <= MAX_IMAGE_TOKENS * 1024);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        // Aspect preserved: square in, square out.
        assert_eq!(w, h);
    }

    #[test]
    fn extreme_aspect_ratios_keep_the_minimum_edge() {
        let (w, h) = smart_resize(&cfg(), 5000, 40);
        assert!(w >= 32 && h >= 32);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
    }

    #[test]
    fn identity_resize_needs_no_interpolation() {
        let cfg = cfg();
        // 96x96 solid color (9216 px, above the floor): every normalized
        // value is exact.
        let rgb = vec![128u8; 96 * 96 * 3];
        let img = preprocess(&cfg, &rgb, 96, 96);
        assert_eq!((img.grid_w, img.grid_h), (6, 6));
        assert_eq!(img.output_tokens(&cfg), 9);
        let expect = (128.0 / 255.0 - 0.5) / 0.5;
        assert!(img.patches.iter().all(|&v| (v - expect).abs() < 1e-6));
    }

    #[test]
    fn channels_are_planar_within_a_patch() {
        let cfg = cfg();
        // Pure red 96x96: R channel = (1.0-0.5)/0.5 = 1, G/B = -1.
        let mut rgb = vec![0u8; 96 * 96 * 3];
        for px in rgb.as_chunks_mut::<3>().0 {
            px[0] = 255;
        }
        let img = preprocess(&cfg, &rgb, 96, 96);
        let plane = (cfg.patch_size * cfg.patch_size) as usize;
        for p in 0..36 {
            let patch = &img.patches[p * 3 * plane..(p + 1) * 3 * plane];
            assert!(patch[..plane].iter().all(|&v| (v - 1.0).abs() < 1e-6));
            assert!(patch[plane..].iter().all(|&v| (v + 1.0).abs() < 1e-6));
        }
    }

    #[test]
    fn patchify_places_pixels_in_cell_order() {
        let cfg = cfg();
        // 96x96, unique value per 16x16 patch: patch (gx,gy) is filled with
        // 24*gy+4*gx. After preprocessing, sequence slot
        // cell_order_index(gx,gy) must hold that value.
        let mut rgb = vec![0u8; 96 * 96 * 3];
        for y in 0..96u32 {
            for x in 0..96u32 {
                let v = ((y / 16) * 24 + (x / 16) * 4) as u8;
                let i = ((y * 96 + x) * 3) as usize;
                rgb[i] = v;
                rgb[i + 1] = v;
                rgb[i + 2] = v;
            }
        }
        let img = preprocess(&cfg, &rgb, 96, 96);
        let patch_len = (3 * cfg.patch_size * cfg.patch_size) as usize;
        for gy in 0..6u32 {
            for gx in 0..6u32 {
                let slot = cell_order_index(gx, gy, 6);
                let got = img.patches[slot * patch_len];
                let raw = (gy * 24 + gx * 4) as f32;
                let expect = (raw / 255.0 - 0.5) / 0.5;
                assert!(
                    (got - expect).abs() < 1e-6,
                    "patch ({gx},{gy}) at slot {slot}: got {got}, expected {expect}"
                );
            }
        }
    }

    #[test]
    fn output_token_count_stays_within_the_declared_budget() {
        let cfg = cfg();
        for (w, h) in [(48, 48), (640, 480), (1920, 1080), (4096, 4096), (33, 5000)] {
            let (tw, th) = smart_resize(&cfg, w, h);
            let tokens = (tw / 32) * (th / 32);
            assert!(
                (MIN_IMAGE_TOKENS..=MAX_IMAGE_TOKENS).contains(&tokens),
                "{w}x{h} -> {tw}x{th} = {tokens} tokens"
            );
        }
    }
}
