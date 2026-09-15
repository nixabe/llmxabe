use xabe_gguf::{GgufArray as A, GgufFile, GgufValue as V};
use xabe_model::{DFlashConfig, ModelConfig, VisionConfig};

fn file(values: &[(&str, V)]) -> GgufFile {
    fn text(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend((values.len() as u64).to_le_bytes());
    for (key, v) in values {
        text(&mut out, key);
        match v {
            V::U32(n) => {
                out.extend(4u32.to_le_bytes());
                out.extend(n.to_le_bytes());
            }
            V::F32(n) => {
                out.extend(6u32.to_le_bytes());
                out.extend(n.to_le_bytes());
            }
            V::Bool(v) => {
                out.extend(7u32.to_le_bytes());
                out.push(u8::from(*v));
            }
            V::String(s) => {
                out.extend(8u32.to_le_bytes());
                text(&mut out, s);
            }
            V::Array(a) => {
                out.extend(9u32.to_le_bytes());
                let (tag, len) = match a {
                    A::F32(v) => (6u32, v.len()),
                    A::I32(v) => (5, v.len()),
                    A::Bool(v) => (7, v.len()),
                    _ => panic!("fixture type"),
                };
                out.extend(tag.to_le_bytes());
                out.extend((len as u64).to_le_bytes());
                match a {
                    A::F32(v) => {
                        for n in v {
                            out.extend(n.to_le_bytes());
                        }
                    }
                    A::I32(v) => {
                        for n in v {
                            out.extend(n.to_le_bytes());
                        }
                    }
                    A::Bool(v) => {
                        for n in v {
                            out.push(u8::from(*n));
                        }
                    }
                    _ => unreachable!(),
                }
            }
            _ => panic!("fixture type"),
        }
    }
    out.resize(out.len().next_multiple_of(32), 0);
    GgufFile::from_bytes(out).unwrap()
}
fn vision(c: VisionConfig) -> Vec<(&'static str, V)> {
    let mut v = vec![
        ("general.architecture", V::String("clip".into())),
        ("clip.projector_type", V::String("qwen3vl_merger".into())),
        ("clip.use_gelu", V::Bool(true)),
        ("clip.vision.attention.layer_norm_epsilon", V::F32(c.ln_eps)),
        (
            "clip.vision.image_mean",
            V::Array(A::F32(c.image_mean.to_vec())),
        ),
        (
            "clip.vision.image_std",
            V::Array(A::F32(c.image_std.to_vec())),
        ),
        (
            "clip.vision.is_deepstack_layers",
            V::Array(A::Bool(vec![false; c.num_layers as usize])),
        ),
    ];
    for (k, n) in [
        ("clip.vision.block_count", c.num_layers),
        ("clip.vision.embedding_length", c.hidden_size),
        ("clip.vision.attention.head_count", c.num_heads),
        ("clip.vision.feed_forward_length", c.ffn_size),
        ("clip.vision.image_size", c.image_size),
        ("clip.vision.patch_size", c.patch_size),
        ("clip.vision.spatial_merge_size", c.spatial_merge),
        ("clip.vision.projection_dim", c.projection_dim),
    ] {
        v.push((k, V::U32(n)));
    }
    v
}
fn draft(c: &DFlashConfig) -> Vec<(&'static str, V)> {
    let mut v = vec![
        ("general.architecture", V::String("dflash".into())),
        ("dflash.attention.layer_norm_rms_epsilon", V::F32(c.rms_eps)),
        ("dflash.rope.freq_base", V::F32(c.rope_theta)),
        (
            "dflash.attention.sliding_window_pattern",
            V::Array(A::Bool(c.swa_pattern.clone())),
        ),
        (
            "dflash.target_layers",
            V::Array(A::I32(c.target_layers.iter().map(|n| *n as i32).collect())),
        ),
    ];
    for (k, n) in [
        ("dflash.block_count", c.num_layers),
        ("dflash.embedding_length", c.hidden_size),
        ("dflash.attention.head_count", c.num_q_heads),
        ("dflash.attention.head_count_kv", c.num_kv_heads),
        ("dflash.attention.key_length", c.head_dim),
        ("dflash.attention.value_length", c.head_dim),
        ("dflash.feed_forward_length", c.ffn_size),
        ("dflash.block_size", c.block_size),
        ("dflash.attention.sliding_window", c.sliding_window),
        ("tokenizer.ggml.mask_token_id", c.mask_token_id),
    ] {
        v.push((k, V::U32(n)));
    }
    v
}
fn replace(v: &mut [(&str, V)], key: &str, value: V) {
    v.iter_mut().find(|(k, _)| *k == key).unwrap().1 = value;
}

#[test]
fn auxiliary_geometry_tracks_metadata_instead_of_presets() {
    let target = ModelConfig::qwen3_8_27b();
    let mut c = VisionConfig::for_model(&target);
    c.num_layers = 3;
    c.hidden_size = 256;
    c.num_heads = 8;
    c.ffn_size = 512;
    c.image_size = 224;
    c.patch_size = 8;
    c.ln_eps = 1e-5;
    c.image_mean = [0.4, 0.3, 0.2];
    c.image_std = [0.2, 0.3, 0.4];
    assert_eq!(
        VisionConfig::from_gguf(&file(&vision(c)), &target).unwrap(),
        c
    );
    let mut d = DFlashConfig::qwen3_6_35b_a3b();
    d.hidden_size = target.hidden_size;
    d.num_layers = 2;
    d.num_q_heads = 16;
    d.num_kv_heads = 4;
    d.head_dim = 64;
    d.ffn_size = 1024;
    d.block_size = 8;
    d.mask_token_id = 17;
    d.target_layers = vec![1, 10, 60];
    d.swa_pattern = vec![true, false];
    d.sliding_window = 512;
    d.rms_eps = 1e-5;
    d.rope_theta = 10000.0;
    assert_eq!(
        DFlashConfig::from_gguf(&file(&draft(&d)), &target).unwrap(),
        d
    );
}
#[test]
fn vision_rejects_incompatible_or_malformed_metadata() {
    let target = ModelConfig::qwen3_6_35b_a3b();
    for (key, value) in [
        ("clip.vision.projection_dim", V::U32(5120)),
        ("clip.vision.attention.head_count", V::U32(0)),
        ("clip.vision.patch_size", V::U32(17)),
        ("clip.vision.spatial_merge_size", V::U32(3)),
        ("clip.use_gelu", V::Bool(false)),
        ("clip.vision.image_std", V::Array(A::F32(vec![0.0; 3]))),
        (
            "clip.vision.image_mean",
            V::Array(A::F32(vec![f32::NAN; 3])),
        ),
        (
            "clip.vision.is_deepstack_layers",
            V::Array(A::Bool(vec![true; 27])),
        ),
    ] {
        let mut v = vision(VisionConfig::for_model(&target));
        replace(&mut v, key, value);
        assert!(
            VisionConfig::from_gguf(&file(&v), &target).is_err(),
            "{key}"
        );
    }
}
#[test]
fn dflash_rejects_invalid_target_bindings_and_patterns() {
    let target = ModelConfig::qwen3_6_35b_a3b();
    for (key, value) in [
        ("dflash.embedding_length", V::U32(5120)),
        ("dflash.target_layers", V::Array(A::I32(vec![0, 1]))),
        ("dflash.target_layers", V::Array(A::I32(vec![1, 1]))),
        ("dflash.target_layers", V::Array(A::I32(vec![40]))),
        ("tokenizer.ggml.mask_token_id", V::U32(target.vocab_size)),
        ("dflash.block_size", V::U32(1)),
        ("dflash.attention.head_count_kv", V::U32(3)),
        ("dflash.attention.value_length", V::U32(64)),
        (
            "dflash.attention.sliding_window_pattern",
            V::Array(A::Bool(vec![false])),
        ),
    ] {
        let mut v = draft(&DFlashConfig::qwen3_6_35b_a3b());
        replace(&mut v, key, value);
        assert!(
            DFlashConfig::from_gguf(&file(&v), &target).is_err(),
            "{key}"
        );
    }
}
#[test]
fn installed_auxiliary_files_match_metadata_derived_schemas() {
    for (target, dir) in [
        (ModelConfig::qwen3_6_35b_a3b(), "Qwen3.6-35B-A3B-GGUF"),
        (ModelConfig::qwen3_8_27b(), "Qwen3.8-27B-GGUF"),
    ] {
        let path = format!(
            "{}/../../models/{dir}/mmproj-F16.gguf",
            env!("CARGO_MANIFEST_DIR")
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!("SKIPPED: {path}");
            continue;
        }
        let f = GgufFile::open(path).unwrap();
        let c = VisionConfig::from_gguf(&f, &target).unwrap();
        assert_eq!(c, VisionConfig::for_model(&target));
        xabe_model::vision::VisionWeightSchema::new(&c)
            .resolve(&f)
            .unwrap();
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/Qwen3.6-35B-A3B-GGUF/qwen36-35b-a3b-dflash-Q8_0.gguf"
    );
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIPPED: {path}");
        return;
    }
    let f = GgufFile::open(path).unwrap();
    let c = DFlashConfig::from_gguf(&f, &ModelConfig::qwen3_6_35b_a3b()).unwrap();
    assert_eq!(c, DFlashConfig::qwen3_6_35b_a3b());
    xabe_model::dflash::DFlashWeightSchema::new(&c)
        .resolve(&c, &f)
        .unwrap();
}
