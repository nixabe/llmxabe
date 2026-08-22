//! Image placements: where a request's images sit in its token stream.
//!
//! The prompt keeps the real `<|image_pad|>` token id in every image slot —
//! the vocabulary check, the tokenizer round-trip and the embed kernel all
//! keep working untouched — and this side channel carries what those slots
//! *mean*: which image, what merged grid, and a content hash that makes the
//! prefix cache image-aware (see the `prefix` module).

/// One image's placement inside a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImagePlacement {
    /// Index of the image's first embedding token in the prompt.
    pub start: usize,
    /// Merged grid height — rows of language-model tokens.
    pub grid_h: u32,
    /// Merged grid width.
    pub grid_w: u32,
    /// Hash of the *preprocessed* image content (patches and grid), not the
    /// container bytes: two files that decode and resize identically are
    /// the same image to the cache, and the same file at a different
    /// `--image-max-tokens` is not.
    pub content_hash: u64,
}

impl ImagePlacement {
    /// Language-model tokens this image occupies.
    pub fn tokens(&self) -> usize {
        (self.grid_h * self.grid_w) as usize
    }

    /// One past the last token index.
    pub fn end(&self) -> usize {
        self.start + self.tokens()
    }

    /// The hash lane substituted for the `<|image_pad|>` at `start + offset`
    /// when naming cache blocks.
    ///
    /// Each slot gets an independent 32-bit digest of `(content_hash,
    /// offset)`, so any block overlapping an image span by even one token
    /// carries at least 32 bits of that image's identity, and a block
    /// overlapping by `k` tokens carries `32k` — the SGLang scheme
    /// (`_compute_pad_value`, `sglang/srt/managers/schedule_batch.py`)
    /// strengthened from one shared lane per image to one per slot.
    pub fn lane(&self, offset: usize) -> u32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        self.content_hash.hash(&mut hasher);
        offset.hash(&mut hasher);
        let mixed = hasher.finish();
        (mixed >> 32) as u32 ^ mixed as u32
    }
}

/// One image ready for admission: its placement plus its preprocessed
/// pixels. The patches travel to the runtime, which encodes them through
/// the vision tower at admit time; only the embeddings survive after that.
#[derive(Debug, Clone)]
pub struct SequenceImage {
    /// Where the image's tokens sit in the prompt.
    pub placement: ImagePlacement,
    /// Normalized patches in cell order (see `xabe_kernels::vision`).
    pub image: xabe_kernels::vision::PreprocessedImage,
}

/// The rotary-base offset in force at prompt position `pos`.
///
/// Every image span before `pos` advances rope "time" by `max(grid_h,
/// grid_w)` while occupying `grid_h * grid_w` tokens, so the base falls
/// behind the token index by their difference, accumulated. Inside a span
/// the delta of the span's *start* applies (the per-token components are
/// taken from [`fill_mrope_triples`] there, not from this scalar).
pub fn rope_delta_at(images: &[ImagePlacement], pos: usize) -> i32 {
    let mut delta = 0i64;
    for img in images {
        if img.end() <= pos {
            let advance = i64::from(img.grid_h.max(img.grid_w));
            delta += advance - img.tokens() as i64;
        }
    }
    delta as i32
}

/// Whether the chunk `[start, start + width)` contains any image token.
pub fn chunk_overlaps_images(images: &[ImagePlacement], start: usize, width: usize) -> bool {
    images
        .iter()
        .any(|img| img.start < start + width && start < img.end())
}

/// Fill `(t, h, w)` triples for the chunk `[start, start + width)` into
/// `out` (cleared first) — the per-token rotary positions the imrope
/// kernel consumes on image-overlapping chunks.
///
/// Matches `xabe_kernels::mrope::assign_positions` over the whole
/// sequence, restricted to the chunk, computed in O(width + images);
/// asserted equal in the tests below.
pub fn fill_mrope_triples(
    images: &[ImagePlacement],
    start: usize,
    width: usize,
    out: &mut Vec<i32>,
) {
    out.clear();
    out.reserve(3 * width);
    for pos in start..start + width {
        if let Some(img) = images
            .iter()
            .find(|img| pos >= img.start && pos < img.end())
        {
            let base = img.start as i32 + rope_delta_at(images, img.start);
            let k = (pos - img.start) as u32;
            out.push(base);
            out.push(base + (k / img.grid_w) as i32);
            out.push(base + (k % img.grid_w) as i32);
        } else {
            let q = pos as i32 + rope_delta_at(images, pos);
            out.push(q);
            out.push(q);
            out.push(q);
        }
    }
}

/// Validate placements against a prompt: sorted, non-overlapping, in range.
///
/// Returns the first violation as a human-readable reason.
pub fn validate_placements(placements: &[ImagePlacement], prompt_len: usize) -> Result<(), String> {
    let mut previous_end = 0usize;
    for (i, p) in placements.iter().enumerate() {
        if p.grid_h == 0 || p.grid_w == 0 {
            return Err(format!("image {i} has an empty grid"));
        }
        if p.start < previous_end {
            return Err(format!("image {i} overlaps its predecessor"));
        }
        if p.end() > prompt_len {
            return Err(format!(
                "image {i} spans tokens {}..{} of a {prompt_len}-token prompt",
                p.start,
                p.end()
            ));
        }
        previous_end = p.end();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement(start: usize, hash: u64) -> ImagePlacement {
        ImagePlacement {
            start,
            grid_h: 2,
            grid_w: 3,
            content_hash: hash,
        }
    }

    #[test]
    fn lanes_differ_per_slot_and_per_image() {
        let a = placement(0, 0xDEAD_BEEF);
        let b = placement(0, 0xDEAD_BEE0);
        assert_ne!(a.lane(0), a.lane(1), "slots of one image must differ");
        assert_ne!(a.lane(0), b.lane(0), "images must differ at every slot");
    }

    #[test]
    fn lanes_are_deterministic() {
        let a = placement(7, 42);
        assert_eq!(
            a.lane(3),
            placement(99, 42).lane(3),
            "start does not enter the lane"
        );
    }

    #[test]
    fn triples_match_the_reference_assignment() {
        use xabe_kernels::mrope::{ImageSpan, assign_positions};
        let images = [
            ImagePlacement {
                start: 3,
                grid_h: 4,
                grid_w: 2,
                content_hash: 1,
            },
            ImagePlacement {
                start: 15,
                grid_h: 2,
                grid_w: 5,
                content_hash: 2,
            },
        ];
        let seq_len = 30usize;
        let spans: Vec<ImageSpan> = images
            .iter()
            .map(|p| ImageSpan {
                start: p.start,
                grid_h: p.grid_h,
                grid_w: p.grid_w,
            })
            .collect();
        let (reference, next) = assign_positions(seq_len, &spans);

        // Whole-sequence fill in one chunk, and again in ragged chunks.
        for chunks in [vec![(0usize, 30usize)], vec![(0, 7), (7, 10), (17, 13)]] {
            let mut got = Vec::new();
            let mut buf = Vec::new();
            for (start, width) in chunks {
                fill_mrope_triples(&images, start, width, &mut buf);
                got.extend_from_slice(&buf);
            }
            for (i, r) in reference.iter().enumerate() {
                assert_eq!(
                    &got[3 * i..3 * i + 3],
                    &[r.t as i32, r.h as i32, r.w as i32],
                    "token {i}"
                );
            }
        }

        // The scalar decode continuation equals the reference's next base.
        assert_eq!(
            seq_len as i32 + rope_delta_at(&images, seq_len),
            next as i32
        );
    }

    #[test]
    fn overlap_detection_covers_edges() {
        let images = [ImagePlacement {
            start: 4,
            grid_h: 2,
            grid_w: 2,
            content_hash: 0,
        }];
        assert!(!chunk_overlaps_images(&images, 0, 4));
        assert!(chunk_overlaps_images(&images, 0, 5));
        assert!(chunk_overlaps_images(&images, 7, 1));
        assert!(!chunk_overlaps_images(&images, 8, 10));
    }

    #[test]
    fn validation_rejects_overlap_and_overhang() {
        let ok = [placement(0, 1), placement(6, 2)];
        assert!(validate_placements(&ok, 12).is_ok());
        let overlapping = [placement(0, 1), placement(5, 2)];
        assert!(validate_placements(&overlapping, 64).is_err());
        let overhanging = [placement(0, 1)];
        assert!(validate_placements(&overhanging, 5).is_err());
        assert!(validate_placements(&[], 0).is_ok());
    }
}
