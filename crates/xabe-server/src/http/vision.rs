//! The serving side of image input: decoding what the wire carries,
//! preprocessing it, and expanding image markers into the token spans the
//! engine consumes.
//!
//! Images arrive as base64 — an OpenAI `image_url` data URI, an Anthropic
//! `image` source block, or a Responses `input_image` — and every dialect
//! folds them into the same [`DecodedImage`]. Rendering inserts one
//! [`IMAGE_MARKER`] per image into the prompt text; after tokenization,
//! [`expand_images`] preprocesses each image, replaces its single
//! `<|image_pad|>` token with one pad per merged patch, and derives the
//! [`ImagePlacement`]s the engine's M-RoPE and prefix cache need.
//!
//! Everything here is gated on `--mmproj`: without it, a request that
//! carries images is refused with a 400 that names the flag, and a request
//! without images never touches this module past the empty-images early
//! return.

use base64::Engine as _;
use std::hash::{Hash, Hasher};

use xabe_engine::image::{ImagePlacement, SequenceImage};
use xabe_kernels::vision::{preprocess_bounded, smart_resize_bounded};
use xabe_model::VisionConfig;

use super::error::{ApiError, Dialect};

/// The markup one image renders as, exactly as the model's chat template
/// spells it. The middle token is the one [`expand_images`] expands.
pub(crate) const IMAGE_MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// The vocabulary spellings resolved against the tokenizer at startup.
pub(crate) const IMAGE_PAD: &str = "<|image_pad|>";
pub(crate) const VISION_START: &str = "<|vision_start|>";
pub(crate) const VISION_END: &str = "<|vision_end|>";

/// Decoded pixels cap. A compressed image expands ~3 bytes per pixel here;
/// this bounds a small upload from decompressing into gigabytes of RGB.
const MAX_DECODED_PIXELS: u64 = 8192 * 8192;

/// What `--mmproj` and `--image-max-tokens` configure, before the tokenizer
/// exists.
#[derive(Debug, Clone)]
pub struct VisionServingConfig {
    /// The vision tower's geometry — patch size, merge factor, pixel
    /// normalization.
    pub config: VisionConfig,
    /// The most language-model tokens one image may occupy; larger images
    /// are resized down to fit.
    pub max_tokens: u32,
}

/// The resolved serving state: the configuration plus the pad token id the
/// expansion matches on.
#[derive(Debug)]
pub(crate) struct VisionServing {
    pub(crate) config: VisionConfig,
    pub(crate) max_tokens: u32,
    pub(crate) image_pad: u32,
}

/// An image decoded to interleaved RGB8, dialect differences already gone.
#[derive(Debug, Clone)]
pub(crate) struct DecodedImage {
    pub(crate) rgb: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

fn decode_bytes(bytes: &[u8]) -> Result<DecodedImage, String> {
    let decoded = image::load_from_memory(bytes)
        .map_err(|failure| format!("the image could not be decoded: {failure}"))?;
    let (width, height) = (decoded.width(), decoded.height());
    if u64::from(width) * u64::from(height) > MAX_DECODED_PIXELS {
        return Err(format!(
            "the image decodes to {width}x{height} pixels, over the {MAX_DECODED_PIXELS} \
             pixel limit"
        ));
    }
    let rgb = decoded.into_rgb8();
    Ok(DecodedImage {
        rgb: rgb.into_raw(),
        width,
        height,
    })
}

/// Decode a base64 payload, as Anthropic's `source.data` carries it.
pub(crate) fn decode_base64_image(data: &str) -> Result<DecodedImage, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .map_err(|failure| format!("the image payload is not valid base64: {failure}"))?;
    decode_bytes(&bytes)
}

/// Decode an image URL. Only `data:` URIs are accepted: this server does
/// not fetch remote content on a caller's behalf.
pub(crate) fn decode_image_url(url: &str) -> Result<DecodedImage, String> {
    let Some(rest) = url.strip_prefix("data:") else {
        let scheme = url.split(':').next().unwrap_or("").to_owned();
        return Err(format!(
            "only `data:` image URLs are supported — this server does not fetch remote \
             images (got a `{scheme}:` URL); send the image inline as base64"
        ));
    };
    let Some((metadata, payload)) = rest.split_once(',') else {
        return Err(
            "a `data:` URL needs a comma between its media type and its payload".to_owned(),
        );
    };
    if !metadata.ends_with(";base64") {
        return Err("only base64 `data:` URLs are supported".to_owned());
    }
    decode_base64_image(payload)
}

/// The 400 a request with images gets on a server that has no vision tower.
fn vision_disabled(dialect: Dialect) -> ApiError {
    ApiError::bad_request(
        dialect,
        "this request carries images, but the server was started without a vision \
         projector; restart it with --mmproj <path> to enable image input",
    )
}

/// Hash the *preprocessed* content — patches and grid — so two files that
/// decode and resize identically name the same cache entries, and the same
/// file under a different token ceiling does not (see
/// [`ImagePlacement::content_hash`]).
fn content_hash(image: &xabe_kernels::vision::PreprocessedImage) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    image.grid_h.hash(&mut hasher);
    image.grid_w.hash(&mut hasher);
    for &value in &image.patches {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Expand each `<|image_pad|>` token into one pad per merged patch and pair
/// it with its preprocessed image.
///
/// The rendered prompt carries exactly one pad per image (see
/// [`IMAGE_MARKER`]), in message order, so pads and `images` zip
/// positionally. A pad without an image means the caller typed the literal
/// marker into a message that also carries real images — refused, because
/// guessing which pad is decorative would misplace every span after it.
pub(crate) fn expand_images(
    vision: Option<&VisionServing>,
    dialect: Dialect,
    tokens: Vec<u32>,
    images: &[DecodedImage],
) -> Result<(Vec<u32>, Vec<SequenceImage>), ApiError> {
    if images.is_empty() {
        return Ok((tokens, Vec::new()));
    }
    let Some(vision) = vision else {
        return Err(vision_disabled(dialect));
    };
    let mut expanded = Vec::with_capacity(tokens.len());
    let mut sequence_images = Vec::with_capacity(images.len());
    let mut next = 0usize;
    for &token in &tokens {
        if token != vision.image_pad {
            expanded.push(token);
            continue;
        }
        let Some(image) = images.get(next) else {
            return Err(ApiError::bad_request(
                dialect,
                format!(
                    "the prompt contains more `{IMAGE_PAD}` markers than images ({} sent); \
                     literal `{IMAGE_PAD}` text is not supported alongside image parts",
                    images.len()
                ),
            ));
        };
        next += 1;
        let preprocessed = preprocess_bounded(
            &vision.config,
            &image.rgb,
            image.width,
            image.height,
            vision.max_tokens,
        );
        let merge = vision.config.spatial_merge;
        let placement = ImagePlacement {
            start: expanded.len(),
            grid_h: preprocessed.grid_h / merge,
            grid_w: preprocessed.grid_w / merge,
            content_hash: content_hash(&preprocessed),
        };
        expanded.extend(std::iter::repeat_n(token, placement.tokens()));
        sequence_images.push(SequenceImage {
            placement,
            image: preprocessed,
        });
    }
    debug_assert_eq!(
        next,
        images.len(),
        "every folded image inserted its own marker"
    );
    Ok((expanded, sequence_images))
}

/// The token count [`expand_images`] would produce, without preprocessing
/// any pixels — what `count_tokens` prices a prompt with.
pub(crate) fn expanded_token_count(
    vision: Option<&VisionServing>,
    dialect: Dialect,
    token_count: usize,
    images: &[DecodedImage],
) -> Result<usize, ApiError> {
    if images.is_empty() {
        return Ok(token_count);
    }
    let Some(vision) = vision else {
        return Err(vision_disabled(dialect));
    };
    let mut count = token_count;
    for image in images {
        let (width, height) =
            smart_resize_bounded(&vision.config, image.width, image.height, vision.max_tokens);
        let patch = vision.config.patch_size;
        // The single rendered pad becomes one pad per merged patch.
        count += vision
            .config
            .output_tokens(height / patch, width / patch)
            .saturating_sub(1) as usize;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serving() -> VisionServing {
        VisionServing {
            config: VisionConfig::qwen3_6_35b_a3b(),
            max_tokens: 1024,
            image_pad: 7,
        }
    }

    fn gray(width: u32, height: u32) -> DecodedImage {
        DecodedImage {
            rgb: vec![128u8; (width * height * 3) as usize],
            width,
            height,
        }
    }

    #[test]
    fn expansion_replaces_one_pad_with_the_merged_token_span() {
        let serving = serving();
        // 96x96 -> 6x6 patches -> 3x3 merged tokens.
        let (tokens, images) = expand_images(
            Some(&serving),
            Dialect::OpenAi,
            vec![1, 2, 7, 3],
            &[gray(96, 96)],
        )
        .expect("one marker, one image");
        assert_eq!(tokens, vec![1, 2, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3]);
        assert_eq!(images.len(), 1);
        let placement = images[0].placement;
        assert_eq!(placement.start, 2);
        assert_eq!((placement.grid_h, placement.grid_w), (3, 3));
        assert_eq!(placement.tokens(), 9);
    }

    #[test]
    fn two_identical_images_hash_identically_and_two_different_do_not() {
        let serving = serving();
        let (_, images) = expand_images(
            Some(&serving),
            Dialect::OpenAi,
            vec![7, 0, 7],
            &[gray(96, 96), gray(96, 96)],
        )
        .expect("two markers, two images");
        assert_eq!(
            images[0].placement.content_hash,
            images[1].placement.content_hash
        );
        let mut red = gray(96, 96);
        for pixel in red.rgb.as_chunks_mut::<3>().0 {
            pixel[0] = 255;
        }
        let (_, unequal) = expand_images(
            Some(&serving),
            Dialect::OpenAi,
            vec![7, 0, 7],
            &[gray(96, 96), red],
        )
        .expect("two markers, two images");
        assert_ne!(
            unequal[0].placement.content_hash,
            unequal[1].placement.content_hash
        );
    }

    #[test]
    fn images_without_a_vision_tower_are_refused() {
        let refused = expand_images(None, Dialect::OpenAi, vec![7], &[gray(96, 96)])
            .expect_err("no --mmproj means no images");
        assert!(refused.message().contains("--mmproj"), "{refused:?}");
        assert!(
            expanded_token_count(None, Dialect::OpenAi, 4, &[gray(96, 96)]).is_err(),
            "counting must refuse what serving refuses"
        );
    }

    #[test]
    fn a_literal_pad_beyond_the_sent_images_is_refused() {
        let serving = serving();
        let refused = expand_images(Some(&serving), Dialect::OpenAi, vec![7, 7], &[gray(96, 96)])
            .expect_err("two markers, one image");
        assert!(refused.message().contains("more"), "{refused:?}");
    }

    #[test]
    fn no_images_passes_tokens_through_even_with_pads_present() {
        // Literal marker text in a text-only request is just text; nothing
        // expands and nothing is refused, with or without a tower.
        let (tokens, images) =
            expand_images(None, Dialect::OpenAi, vec![7, 7], &[]).expect("no images, no work");
        assert_eq!(tokens, vec![7, 7]);
        assert!(images.is_empty());
    }

    #[test]
    fn the_count_matches_a_real_expansion() {
        let serving = serving();
        let images = [gray(96, 96), gray(640, 480)];
        let (expanded, _) = expand_images(
            Some(&serving),
            Dialect::OpenAi,
            vec![1, 7, 2, 7, 3],
            &images,
        )
        .expect("expands");
        assert_eq!(
            expanded_token_count(Some(&serving), Dialect::OpenAi, 5, &images).expect("counts"),
            expanded.len()
        );
    }

    #[test]
    fn the_token_ceiling_bounds_every_expansion() {
        let serving = VisionServing {
            max_tokens: 64,
            ..serving()
        };
        let (_, images) = expand_images(
            Some(&serving),
            Dialect::OpenAi,
            vec![7],
            &[gray(1920, 1080)],
        )
        .expect("expands");
        assert!(
            images[0].placement.tokens() <= 64,
            "{} tokens",
            images[0].placement.tokens()
        );
    }

    #[test]
    fn data_uris_decode_and_remote_urls_are_refused() {
        // A 1x1 red PNG.
        const PNG_1X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4\
                               2mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        let image = decode_image_url(&format!("data:image/png;base64,{PNG_1X1}"))
            .expect("a data URI decodes");
        assert_eq!((image.width, image.height), (1, 1));
        assert_eq!(image.rgb.len(), 3);

        let refused =
            decode_image_url("https://example.com/cat.png").expect_err("remote fetch is refused");
        assert!(refused.contains("does not fetch"), "{refused}");
        assert!(decode_image_url("data:image/png,plain").is_err());
        assert!(decode_base64_image("!!!").is_err());
        assert!(
            decode_base64_image(PNG_1X1).is_ok(),
            "bare base64 decodes without a data: wrapper"
        );
    }
}
