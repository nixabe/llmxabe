//! CPU reference for the Qwen3.6 vision tower (`qwen3vl_merger` mmproj).
//!
//! Scalar f32 throughout, per this crate's charter: obviously correct and
//! readable against the cited upstream math, never fast. The CUDA vision
//! path is validated against [`tower::encode`] by differential test, and
//! [`tower::encode`] itself is validated stage-by-stage against llama.cpp's
//! own execution of the same mmproj file (`llama-mtmd-debug`, see
//! `tests/vision_golden.rs`).
//!
//! Ported from `tools/mtmd/models/qwen3vl.cpp` (graph),
//! `tools/mtmd/models/qwen2vl.cpp` (`build_inp_with_temporal_merge`),
//! `tools/mtmd/clip.cpp` (position filling, pos-embd resize) and
//! `tools/mtmd/mtmd-image.cpp` (preprocessing) in llama.cpp.

pub mod preprocess;
pub mod tower;

pub use preprocess::{
    PreprocessedImage, preprocess, preprocess_bounded, smart_resize, smart_resize_bounded,
};
pub use tower::{VisionBlockWeights, VisionWeights, encode};

/// Index of patch `(x, y)` in the transformer's token order.
///
/// The sequence is ordered 2×2-cell-major: cells row-major over the merged
/// grid, and top-left, top-right, bottom-left, bottom-right inside each
/// cell. Derived element-wise from the permute/reshape chain at the top of
/// `clip_graph_qwen3vl::build` (`tools/mtmd/models/qwen3vl.cpp`), and
/// identical to the `positions` fill order in `clip.cpp`'s
/// `set_input` for `PROJECTOR_TYPE_QWEN3VL`.
pub fn cell_order_index(x: u32, y: u32, patches_w: u32) -> usize {
    let (wx, dx) = (x / 2, x % 2);
    let (hy, dy) = (y / 2, y % 2);
    (dx + 2 * dy + 4 * wx + 2 * patches_w * hy) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_order_walks_2x2_cells_row_major() {
        // 4x2 patch grid: cells (0,0) and (1,0); order inside a cell is
        // TL, TR, BL, BR.
        let order: Vec<usize> = [
            (0u32, 0u32),
            (1, 0),
            (0, 1),
            (1, 1), // first cell
            (2, 0),
            (3, 0),
            (2, 1),
            (3, 1), // second cell
        ]
        .iter()
        .map(|&(x, y)| cell_order_index(x, y, 4))
        .collect();
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn cell_order_is_a_permutation() {
        let (pw, ph) = (6u32, 4u32);
        let mut seen = vec![false; (pw * ph) as usize];
        for y in 0..ph {
            for x in 0..pw {
                let i = cell_order_index(x, y, pw);
                assert!(!seen[i], "index {i} hit twice");
                seen[i] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }
}
