//! Resolve the vision weight schema against the real mmproj file.
//!
//! [`VisionWeightSchema`] derives every expected tensor name and dimension
//! from [`VisionConfig`] alone; this closes the loop against the 899 MB
//! `mmproj-F16.gguf` that ships beside the language model — every tensor
//! present, every shape as predicted, nothing left over.
//!
//! The mmproj is a *separate* file whose path the server takes explicitly
//! (`--mmproj`), so this test takes its own `LLMXABE_MMPROJ` override
//! rather than deriving the path from `LLMXABE_MODEL`.
//!
//! If the file is absent the test SKIPS with a printed message. Per
//! `AGENTS.md`, a skipped test is reported as skipped and is not evidence
//! the schema is right.

use std::path::PathBuf;

use xabe_gguf::GgufFile;
use xabe_model::vision::{VisionConfig, VisionRole, VisionWeightSchema};

const DEFAULT_MMPROJ_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf";

fn mmproj_path() -> PathBuf {
    std::env::var_os("LLMXABE_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MMPROJ_PATH))
}

fn open() -> Option<GgufFile> {
    let path = mmproj_path();
    if !path.exists() {
        println!(
            "SKIPPED: mmproj file not found at {}; set LLMXABE_MMPROJ to override",
            path.display()
        );
        return None;
    }
    Some(GgufFile::open(&path).expect("mmproj file must parse as valid GGUF v3"))
}

#[test]
fn vision_schema_resolves_against_the_real_mmproj_with_no_mismatches() {
    let Some(file) = open() else { return };
    let config = VisionConfig::qwen3_6_35b_a3b();
    let schema = VisionWeightSchema::new(&config);

    if let Err(errors) = schema.resolve(&file) {
        for e in &errors {
            eprintln!("  {e}");
        }
        panic!("{} tensor(s) did not match the vision schema", errors.len());
    }

    // The schema is exhaustive: every tensor in the file is claimed. This is
    // the assertion that would catch a tensor the loader silently ignores —
    // a deepstack merger appearing in a future re-export, for example.
    assert_eq!(
        schema.specs().len(),
        file.n_tensors(),
        "schema names {} tensors but the file holds {}",
        schema.specs().len(),
        file.n_tensors(),
    );
    println!("resolved {} vision tensors", schema.specs().len());
}

#[test]
fn mmproj_metadata_matches_the_transcribed_config() {
    // `VisionConfig` is transcribed by hand; the file's own metadata is the
    // ground truth. A silent re-export with, say, a different merge size
    // must fail here rather than produce garbage embeddings.
    let Some(file) = open() else { return };
    let c = VisionConfig::qwen3_6_35b_a3b();

    let get = |key: &str| {
        file.get_u32(key)
            .unwrap_or_else(|| panic!("mmproj metadata `{key}` missing"))
    };
    assert_eq!(get("clip.vision.block_count"), c.num_layers);
    assert_eq!(get("clip.vision.embedding_length"), c.hidden_size);
    assert_eq!(get("clip.vision.attention.head_count"), c.num_heads);
    assert_eq!(get("clip.vision.feed_forward_length"), c.ffn_size);
    assert_eq!(get("clip.vision.image_size"), c.image_size);
    assert_eq!(get("clip.vision.patch_size"), c.patch_size);
    assert_eq!(get("clip.vision.spatial_merge_size"), c.spatial_merge);
    assert_eq!(get("clip.vision.projection_dim"), c.projection_dim);
}

#[test]
fn this_export_ships_no_deepstack_layers() {
    // The engine splices projector output at exactly one point. Deepstack
    // would feed extra features into early LLM layers — a structurally
    // different integration this engine does not implement. If a future
    // mmproj export turns deepstack on, refuse it loudly at load time
    // rather than silently dropping the extra features.
    let Some(file) = open() else { return };
    let flags = file
        .get_bool_array("clip.vision.is_deepstack_layers")
        .expect("deepstack flags present in qwen3vl mmproj");
    assert_eq!(flags.len(), 27);
    assert!(
        flags.iter().all(|&f| !f),
        "mmproj declares active deepstack layers; the loader must reject this file"
    );
    // And no merger tensors beyond mm.0 / mm.2 exist.
    let schema = VisionWeightSchema::new(&VisionConfig::qwen3_6_35b_a3b());
    assert!(schema.find(VisionRole::MergerFc1Weight, None).is_some());
}
