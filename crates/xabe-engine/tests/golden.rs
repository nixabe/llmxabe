//! The llama.cpp oracle: loader and shape/layout gate for the captured golden
//! forward pass.
//!
//! Story G006's objective is a full forward pass whose logits match llama.cpp.
//! Until something recorded llama.cpp's answer for a fixed prompt, that gate
//! was not executable — there was nothing to compare against. This file is the
//! reader for what [`docs/ORACLE.md`](../../../docs/ORACLE.md) captures, plus
//! the tests that prove the capture is intact and that its memory layout is
//! read the same way this repository's own GGUF loader reads the file.
//!
//! ## Using it from another test
//!
//! ```ignore
//! #[path = "golden.rs"]
//! mod golden;
//!
//! #[test]
//! fn my_layer_matches_llama_cpp() {
//!     let Some(g) = golden::setup() else { return };
//!     let entering = g.f32("l_out-2");   // hidden state entering block 3
//!     let leaving  = g.f32("l_out-3");   // hidden state leaving block 3
//!     // ...
//! }
//! ```
//!
//! The tests below run again in whichever binary includes this file. They are
//! cheap relative to a device test and idempotent, so that is deliberate: a
//! consumer that includes the module also inherits the integrity check.
//!
//! ## Layout — the thing that silently produces a transposed comparison
//!
//! ggml stores `ne[0]` as the **fastest-varying** dimension, which is the
//! opposite of reading a `[rows][cols]` array row-major. A tensor logged as
//! `ne=[2048, 19]` is 19 columns of 2048 contiguous floats, not 2048 rows of
//! 19. Every payload in the container is written in ggml logical order —
//! `i0 + ne0*(i1 + ne1*(i2 + ne2*i3))` — with strided views de-strided at
//! capture time, so element `(i0, i1)` is at flat index `i1 * ne0 + i0`.
//!
//! [`input_embedding_matches_this_repos_own_gguf_reader`] is the proof that
//! this convention is the right one and that it agrees with `xabe-gguf`: it
//! dequantizes `token_embd.weight` straight out of the model file and compares
//! it to llama.cpp's captured embedding output, element for element.
//!
//! SKIPS — reporting that it skipped — when the golden file is absent. It is
//! ~55 MiB of captured activations and is deliberately not committed; see
//! `docs/ORACLE.md` for the one command that regenerates it.

#![allow(dead_code)]

use std::path::PathBuf;

use xabe_gguf::GgufFile;
use xabe_model::config::{LayerKind, ModelConfig};

/// Where the capture lands by default, relative to the workspace root.
const DEFAULT_GOLDEN_PATH: &str = ".golden/qwen36-golden.bin";

/// The model the golden was captured from, for the layout proof.
const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// The prompt the golden was captured for. Recorded here so a stale capture
/// taken against a different prompt fails loudly rather than comparing against
/// the wrong activations.
pub const GOLDEN_PROMPT: &str = "The capital of France is Paris. The capital of Germany is Berlin. \
     The capital of Japan is";

/// Token ids the prompt tokenizes to, `add_special = true`. This model has
/// `add_bos = false`, so there is no leading BOS.
pub const GOLDEN_TOKENS: [i32; 19] = [
    760, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 9564, 369, 19241, 13, 561, 6511, 314,
    6124, 369,
];

/// The token llama.cpp's logits select for the position after the prompt.
/// `' Tokyo'` — the prompt is a two-shot capital-city pattern, so an oracle
/// that decoded something else would be a capture bug rather than a model
/// property.
pub const GOLDEN_ARGMAX: usize = 25358;

const MAGIC: &[u8; 8] = b"XABEGOLD";
const VERSION: u32 = 1;

/// Element type of one captured record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// Activations, and anything ggml held as a float type. Captured as f32.
    F32,
    /// Token ids.
    I32,
}

/// One captured tensor.
#[derive(Debug, Clone)]
pub struct Record {
    /// The ggml graph node name, exactly as `llm_graph_context::cb` set it:
    /// `"{tag}-{layer}"` for per-layer nodes, bare for globals. Two records
    /// may share a name — see [`Golden::all`].
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Dimensions in ggml order. `ne[0]` is fastest-varying.
    pub ne: [i64; 4],
    /// Payload, in ggml logical order.
    pub f32_data: Vec<f32>,
    /// Payload for [`Dtype::I32`] records.
    pub i32_data: Vec<i32>,
}

impl Record {
    /// Total element count.
    pub fn len(&self) -> usize {
        match self.dtype {
            Dtype::F32 => self.f32_data.len(),
            Dtype::I32 => self.i32_data.len(),
        }
    }

    /// Whether the record holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dimensions with trailing 1s dropped, which is how the capture log and
    /// llama.cpp's own debug output read.
    pub fn shape(&self) -> Vec<i64> {
        let mut d = self.ne.to_vec();
        while d.len() > 1 && *d.last().unwrap() == 1 {
            d.pop();
        }
        d
    }

    /// Column `i1` of a 2-D record: `ne[0]` contiguous elements.
    ///
    /// This is the accessor that encodes the ggml convention, so callers do
    /// not each get a chance to transpose it independently.
    pub fn column(&self, i1: usize) -> &[f32] {
        let n0 = self.ne[0] as usize;
        assert!(
            (i1 as i64) < self.ne[1],
            "{}: column {i1} out of range for ne[1] = {}",
            self.name,
            self.ne[1],
        );
        &self.f32_data[i1 * n0..(i1 + 1) * n0]
    }
}

/// Every tensor captured in one llama.cpp forward pass.
#[derive(Debug, Clone)]
pub struct Golden {
    records: Vec<Record>,
}

/// Why a golden file failed to load.
#[derive(Debug)]
pub enum GoldenError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The file is not a golden container, or is a version this reader does
    /// not understand.
    Format(String),
}

impl std::fmt::Display for GoldenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Format(m) => write!(f, "{m}"),
        }
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], GoldenError> {
        if self.p + n > self.b.len() {
            return Err(GoldenError::Format(format!(
                "truncated at byte {}: wanted {n} more, {} remain",
                self.p,
                self.b.len() - self.p,
            )));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, GoldenError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, GoldenError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, GoldenError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

impl Golden {
    /// Parse a golden container.
    ///
    /// The format is described in `docs/ORACLE.md`; it is deliberately a flat
    /// sequence of length-prefixed records so this reader needs no
    /// dependencies beyond the standard library.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, GoldenError> {
        let path = path.into();
        let bytes = std::fs::read(&path).map_err(GoldenError::Io)?;
        let mut c = Cursor { b: &bytes, p: 0 };

        if c.take(8)? != MAGIC {
            return Err(GoldenError::Format(format!(
                "{} is not a golden container (bad magic)",
                path.display(),
            )));
        }
        let version = c.u32()?;
        if version != VERSION {
            return Err(GoldenError::Format(format!(
                "golden container version {version}, this reader understands {VERSION}",
            )));
        }
        let n_records = c.u32()? as usize;

        let mut records = Vec::with_capacity(n_records);
        for i in 0..n_records {
            let name_len = c.u32()? as usize;
            let name = String::from_utf8(c.take(name_len)?.to_vec()).map_err(|e| {
                GoldenError::Format(format!("record {i} has a non-UTF-8 name: {e}"))
            })?;
            let dtype = match c.u32()? {
                0 => Dtype::F32,
                1 => Dtype::I32,
                other => {
                    return Err(GoldenError::Format(format!(
                        "record `{name}` has unknown dtype {other}",
                    )));
                }
            };
            let ne = [c.i64()?, c.i64()?, c.i64()?, c.i64()?];
            let n_elements = c.u64()? as usize;

            let declared: i64 = ne.iter().product();
            if declared != n_elements as i64 {
                return Err(GoldenError::Format(format!(
                    "record `{name}`: ne {ne:?} implies {declared} elements, header says {n_elements}",
                )));
            }

            let raw = c.take(n_elements * 4)?;
            let (mut f32_data, mut i32_data) = (Vec::new(), Vec::new());
            match dtype {
                Dtype::F32 => {
                    f32_data.reserve_exact(n_elements);
                    f32_data.extend(
                        raw.as_chunks::<4>()
                            .0
                            .iter()
                            .copied()
                            .map(f32::from_le_bytes),
                    );
                }
                Dtype::I32 => {
                    i32_data.reserve_exact(n_elements);
                    i32_data.extend(
                        raw.as_chunks::<4>()
                            .0
                            .iter()
                            .copied()
                            .map(i32::from_le_bytes),
                    );
                }
            }

            records.push(Record {
                name,
                dtype,
                ne,
                f32_data,
                i32_data,
            });
        }

        if c.p != bytes.len() {
            return Err(GoldenError::Format(format!(
                "{} bytes of trailing data after {n_records} records",
                bytes.len() - c.p,
            )));
        }

        Ok(Self { records })
    }

    /// Every record, in capture (graph execution) order.
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// All records carrying `name`.
    ///
    /// **Names are not unique.** llama.cpp calls `cb(t, "Kcur", il)` twice in
    /// a Gated Attention layer — once on the raw projection, once on the
    /// post-RoPE tensor — so `Kcur-3` names two different tensors with two
    /// different shapes. Anything comparing against `Kcur-*` has to say which
    /// one it means.
    pub fn all(&self, name: &str) -> Vec<&Record> {
        self.records.iter().filter(|r| r.name == name).collect()
    }

    /// The **last** record carrying `name`, i.e. the value that node held at
    /// the end of the forward pass.
    pub fn get(&self, name: &str) -> Option<&Record> {
        self.records.iter().rev().find(|r| r.name == name)
    }

    /// [`Self::get`], panicking with the available names on a miss.
    pub fn expect(&self, name: &str) -> &Record {
        self.get(name).unwrap_or_else(|| {
            panic!(
                "golden has no tensor `{name}`; it holds {} records — \
                 regenerate with a filter that covers it (see docs/ORACLE.md)",
                self.records.len(),
            )
        })
    }

    /// Float payload of the last record carrying `name`.
    pub fn f32(&self, name: &str) -> &[f32] {
        let r = self.expect(name);
        assert_eq!(r.dtype, Dtype::F32, "`{name}` is not a float tensor");
        &r.f32_data
    }

    /// The prompt's token ids.
    pub fn tokens(&self) -> &[i32] {
        let r = self.expect("api.tokens");
        assert_eq!(r.dtype, Dtype::I32);
        &r.i32_data
    }

    /// Number of prompt tokens the pass ran over.
    pub fn n_tokens(&self) -> usize {
        self.tokens().len()
    }

    /// Final logits over the full vocabulary, for the last prompt position.
    ///
    /// Taken from `llama_get_logits_ith`, and asserted below to be
    /// bit-identical to the `result_output` graph node.
    pub fn logits(&self) -> &[f32] {
        self.f32("api.logits")
    }

    /// Hidden state **entering** transformer block `layer`, `[hidden, tokens]`.
    ///
    /// Block 0's input is the embedding output; every later block's input is
    /// the previous block's `l_out`. Naming it once here keeps the off-by-one
    /// out of every call site.
    pub fn block_input(&self, layer: u32) -> &Record {
        match layer {
            0 => self.expect("model.input_embed"),
            n => self.expect(&format!("l_out-{}", n - 1)),
        }
    }

    /// Hidden state **leaving** transformer block `layer`, `[hidden, tokens]`.
    pub fn block_output(&self, layer: u32) -> &Record {
        self.expect(&format!("l_out-{layer}"))
    }
}

/// Load the golden, or report that the test is skipping.
///
/// Mirrors the `setup()` helpers in the differential tests: a missing input is
/// a skip that says so, never a silent pass.
pub fn setup() -> Option<Golden> {
    let path = golden_path();
    if !path.exists() {
        println!(
            "SKIPPED: golden capture not found at {}; \
             regenerate it with the procedure in docs/ORACLE.md, \
             or set LLMXABE_GOLDEN to point at one",
            path.display(),
        );
        return None;
    }
    match Golden::load(&path) {
        Ok(g) => Some(g),
        Err(e) => {
            // A corrupt file is not a skip. A test that quietly passed on a
            // truncated capture would be worse than no oracle at all.
            panic!("golden at {} failed to parse: {e}", path.display());
        }
    }
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <root>/crates/xabe-engine.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("manifest dir has a workspace root above it")
        .to_path_buf()
}

fn golden_path() -> PathBuf {
    std::env::var_os("LLMXABE_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join(DEFAULT_GOLDEN_PATH))
}

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn golden_loads_with_the_prompt_and_geometry_it_was_captured_for() {
    let Some(g) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();

    println!(
        "golden: {} records, {} tokens",
        g.records().len(),
        g.n_tokens(),
    );

    // Pin the prompt. A capture taken against a different prompt would load
    // fine and compare against the wrong activations everywhere.
    assert_eq!(
        g.tokens(),
        &GOLDEN_TOKENS,
        "golden was captured for a different prompt than this file describes",
    );

    let n_tokens = g.n_tokens() as i64;
    let hidden = i64::from(config.hidden_size);
    let vocab = i64::from(config.vocab_size);

    let embed = g.expect("model.input_embed");
    assert_eq!(embed.shape(), vec![hidden, n_tokens]);

    // `result_norm` is post-`get_rows`: llama.cpp keeps only the positions it
    // was asked to produce logits for, which for a plain prefill is the last
    // one. `h_nextn` is the same norm before that selection, so it still has
    // every token — that is the one to compare a full-sequence final norm
    // against.
    assert_eq!(g.expect("h_nextn").shape(), vec![hidden, n_tokens]);
    assert_eq!(g.expect("result_norm").shape(), vec![hidden]);
    assert_eq!(g.expect("result_output").shape(), vec![vocab]);
    assert_eq!(g.logits().len(), vocab as usize);

    println!(
        "  model.input_embed {:?}, h_nextn {:?}, result_output {:?}",
        embed.shape(),
        g.expect("h_nextn").shape(),
        g.expect("result_output").shape(),
    );
}

#[test]
fn every_transformer_block_boundary_is_captured() {
    let Some(g) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let n_tokens = g.n_tokens() as i64;
    let hidden = i64::from(config.hidden_size);

    // The residual stream at every block boundary, plus the three residual
    // waypoints inside each block. This is what makes a forward pass
    // bisectable: a mismatch at `l_out-17` with `l_out-16` clean localizes the
    // fault to one block without re-running anything.
    for layer in 0..config.num_layers {
        for tag in [
            "attn_norm",
            "attn_residual",
            "attn_post_norm",
            "ffn_out",
            "l_out",
        ] {
            let r = g.expect(&format!("{tag}-{layer}"));
            assert_eq!(
                r.shape(),
                vec![hidden, n_tokens],
                "{tag}-{layer} has the wrong shape",
            );
            assert!(
                r.f32_data.iter().all(|v| v.is_finite()),
                "{tag}-{layer} contains a non-finite value",
            );
        }
    }

    // Block 0's input is the embedding output, not an `l_out`.
    assert_eq!(
        g.block_input(0).name,
        "model.input_embed",
        "block 0 must read the embedding table output",
    );
    assert_eq!(g.block_input(1).name, "l_out-0");
    assert_eq!(g.block_output(39).name, "l_out-39");

    println!(
        "{} blocks x 5 waypoints captured, all finite, all [{hidden}, {n_tokens}]",
        config.num_layers,
    );
}

#[test]
fn the_packed_query_gate_projection_is_interleaved_per_head_not_split_in_half() {
    // The second layout trap, proved from captured data rather than from
    // reading upstream source.
    //
    // `blk.N.attn_q.weight` is `[hidden, head_dim * q_heads * 2]`, and
    // `xabe_model::weights` documents the extra width as query and output gate
    // *interleaved per head*, stride `head_dim * 2`. The golden carries all
    // three tensors involved — the full projection, the query view, and the
    // gate view — so the claim is checkable:
    //
    //   Qcur_reshaped[d, h, t]  ==  Qcur_full[h * 2 * head_dim + d,           t]
    //   gate_reshaped[h*hd + d, t] == Qcur_full[h * 2 * head_dim + head_dim + d, t]
    //
    // A loader that split `attn_q` down the middle instead would read
    // `Qcur_full[h * head_dim + d]`, which agrees only for head 0.
    let Some(g) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let a = config.attention;
    let hd = a.head_dim as usize;
    let nh = a.q_heads as usize;
    let n_tokens = g.n_tokens();

    let full = g.expect("Qcur_full-3");
    let query = g.expect("Qcur_reshaped-3");
    let gate = g.expect("gate_reshaped-3");

    let (mut bad_q, mut bad_gate, mut bad_contiguous) = (0usize, 0usize, 0usize);
    for t in 0..n_tokens {
        let f = full.column(t);
        for h in 0..nh {
            for d in 0..hd {
                let interleaved_q = f[h * 2 * hd + d];
                let interleaved_gate = f[h * 2 * hd + hd + d];
                let contiguous_q = f[h * hd + d];
                let ref_q = query.f32_data[(t * nh + h) * hd + d];
                let ref_gate = gate.column(t)[h * hd + d];

                if interleaved_q.to_bits() != ref_q.to_bits() {
                    bad_q += 1;
                }
                if interleaved_gate.to_bits() != ref_gate.to_bits() {
                    bad_gate += 1;
                }
                if contiguous_q.to_bits() != ref_q.to_bits() {
                    bad_contiguous += 1;
                }
            }
        }
    }

    let total = n_tokens * nh * hd;
    println!(
        "attn_q interleave over {total} elements: stride-2*head_dim mismatches \
         q={bad_q} gate={bad_gate}; split-in-half mismatches {bad_contiguous}",
    );
    assert_eq!(bad_q, 0, "the query half is not at stride head_dim * 2");
    assert_eq!(bad_gate, 0, "the gate half is not at stride head_dim * 2");
    // Head 0 coincides under both readings; everything else must not, or the
    // test proves nothing.
    assert_eq!(
        bad_contiguous,
        total - n_tokens * hd,
        "splitting attn_q in half should disagree on every head but the first",
    );
}

#[test]
fn mixer_intermediates_match_the_layer_kind_and_the_real_head_geometry() {
    let Some(g) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let n_tokens = g.n_tokens() as i64;

    // The capture covers three GDN blocks and two Gated Attention blocks.
    // Assert the pattern derivation agrees with which internals actually
    // exist, so a wrong `attention_offset` cannot pass unnoticed.
    for layer in [0u32, 4, 20] {
        assert_eq!(config.layer_kind(layer), LayerKind::GatedDeltaNet);
        assert!(g.get(&format!("conv_output_silu-{layer}")).is_some());
        assert!(g.get(&format!("Qcur_full-{layer}")).is_none());
    }
    for layer in [3u32, 39] {
        assert_eq!(config.layer_kind(layer), LayerKind::GatedAttention);
        assert!(g.get(&format!("Qcur_full-{layer}")).is_some());
        assert!(g.get(&format!("conv_output_silu-{layer}")).is_none());
    }

    let gdn = config.gdn;
    let head_dim = i64::from(gdn.head_dim);
    let key_dim = i64::from(gdn.qk_heads) * head_dim;
    let value_dim = i64::from(gdn.value_heads) * head_dim;

    // The fused q/k/v stream the depthwise convolution runs over.
    let conv_dim = key_dim * 2 + value_dim;
    assert_eq!(
        g.expect("linear_attn_qkv_mixed-0").shape(),
        vec![conv_dim, n_tokens],
    );
    assert_eq!(
        g.expect("conv_output_raw-0").shape(),
        vec![conv_dim, n_tokens]
    );
    assert_eq!(
        g.expect("conv_output_silu-0").shape(),
        vec![conv_dim, n_tokens],
    );

    // The SiLU between the convolution and the q/k/v slice — the op
    // `docs/KERNELS.md` records as easy to drop. Two separately captured
    // tensors either side of it means a GDN block that forgot it fails here,
    // not three layers later.
    let raw = g.expect("conv_output_raw-0");
    let silu = g.expect("conv_output_silu-0");
    let mut worst = 0.0f32;
    for (&x, &y) in raw.f32_data.iter().zip(&silu.f32_data) {
        worst = worst.max((y - x / (1.0 + (-x).exp())).abs());
    }
    assert!(
        worst < 1e-5,
        "conv_output_silu-0 is not silu(conv_output_raw-0): worst deviation {worst:.3e}",
    );
    println!("conv_output_silu-0 == silu(conv_output_raw-0) to {worst:.3e}");

    // Post-convolution q/k carry 16 qk heads and v carries 32 value heads.
    // Note these are *not* pre-broadcast to 32: this build takes the fused GDN
    // path, which skips the explicit `ggml_repeat_4d`.
    assert_eq!(
        g.expect("q_conv_predelta-0").shape(),
        vec![head_dim, i64::from(gdn.qk_heads), n_tokens],
    );
    assert_eq!(
        g.expect("v_conv_predelta-0").shape(),
        vec![head_dim, i64::from(gdn.value_heads), n_tokens],
    );
    assert_eq!(
        g.expect("state_predelta-0").shape(),
        vec![head_dim, head_dim, i64::from(gdn.value_heads)],
    );
    // A fresh sequence starts from a zeroed recurrent state.
    assert!(g.f32("state_predelta-0").iter().all(|&v| v == 0.0));

    let attn = config.attention;
    let a_head = i64::from(attn.head_dim);
    let q_dim = i64::from(attn.q_heads) * a_head;

    // The packed query+gate projection: double width, because each head's
    // query is followed by that head's output gate. This is the shape that
    // looks like a bug and is not — see `xabe_model::weights`.
    assert_eq!(g.expect("Qcur_full-3").shape(), vec![q_dim * 2, n_tokens]);
    assert_eq!(
        g.expect("Qcur_reshaped-3").shape(),
        vec![a_head, i64::from(attn.q_heads), n_tokens],
        "the query half must be a strided view of stride head_dim*2, not the first half",
    );
    assert_eq!(g.expect("gate_reshaped-3").shape(), vec![q_dim, n_tokens]);
    assert_eq!(
        g.expect("attn_pregate-3").shape(),
        vec![q_dim, n_tokens],
        "attention output before the sigmoid gate",
    );

    // `Kcur-3` is captured twice under one name. Assert both, so a consumer
    // that grabs the wrong one has a documented reason available.
    let kcur = g.all("Kcur-3");
    assert_eq!(
        kcur.len(),
        2,
        "Kcur-3 should be captured pre- and post-RoPE"
    );
    assert_eq!(
        kcur[0].shape(),
        vec![i64::from(attn.kv_heads) * a_head, n_tokens],
        "first Kcur-3 is the raw projection",
    );
    assert_eq!(
        kcur[1].shape(),
        vec![a_head, i64::from(attn.kv_heads), n_tokens],
        "second Kcur-3 is post-norm, post-RoPE",
    );

    println!(
        "GDN blocks 0/4/20 and attention blocks 3/39 carry the geometry \
         ModelConfig derives ({} qk / {} v heads, {} q / {} kv heads)",
        gdn.qk_heads, gdn.value_heads, attn.q_heads, attn.kv_heads,
    );
}

#[test]
fn the_captured_logits_are_the_same_tensor_the_public_api_returns() {
    let Some(g) = setup() else { return };

    // `result_output` comes off the graph callback mid-compute;
    // `api.logits` comes from `llama_get_logits_ith` after the decode
    // returned. If these ever disagreed, the intermediate captures would be
    // from a different execution than the logits, and every per-layer
    // comparison built on this file would be meaningless.
    let node = g.f32("result_output");
    let api = g.logits();
    assert_eq!(node.len(), api.len());
    assert!(
        node.iter()
            .zip(api)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the `result_output` graph node and llama_get_logits_ith disagree",
    );

    let (best, &top) = api
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("logits are non-empty");
    assert_eq!(
        best, GOLDEN_ARGMAX,
        "llama.cpp's argmax moved; the capture is stale or the model changed",
    );
    assert!(api.iter().all(|v| v.is_finite()));

    println!(
        "argmax = {best} (' Tokyo'), logit = {top:.6}, {} entries",
        api.len()
    );
}

#[test]
fn input_embedding_matches_this_repos_own_gguf_reader() {
    // THE LAYOUT PROOF.
    //
    // llama.cpp's `model.input_embed` is `ggml_get_rows(token_embd.weight,
    // tokens)`. If `xabe-gguf` reads `token_embd.weight` with the same
    // convention — `ne[0] = 2048` fastest-varying, so one token's embedding is
    // 2048 *contiguous* elements — then dequantizing row `tokens[i]` straight
    // out of the file must reproduce column `i` of the capture.
    //
    // If the convention were the other way round, a row would be a stride-
    // 248,320 gather and this would not match on a single element. So this is
    // the check that stops a transposed reading from being mistaken for a
    // kernel bug later.
    let Some(g) = setup() else { return };
    let path = model_path();
    if !path.exists() {
        println!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display(),
        );
        return;
    }
    let file = GgufFile::open(&path).expect("model file must parse as valid GGUF v3");
    let config = ModelConfig::qwen3_6_35b_a3b();

    let info = file
        .tensor("token_embd.weight")
        .expect("the file must carry an embedding table");
    let hidden = u64::from(config.hidden_size);
    assert_eq!(
        info.dims,
        vec![hidden, u64::from(config.vocab_size)],
        "token_embd.weight is [hidden, vocab] in ggml order, hidden fastest",
    );
    assert_eq!(
        info.ggml_type,
        xabe_gguf::GgmlType::Q8_0,
        "this proof dequantizes Q8_0; the file's embedding type changed",
    );

    let bytes = file
        .tensor_bytes("token_embd.weight")
        .expect("tensor bytes are mapped");
    let row_bytes =
        (hidden as usize / xabe_kernels::quant::QK8_0) * xabe_kernels::quant::BLOCK_Q8_0_BYTES;

    let embed = g.expect("model.input_embed");
    let mut exact = 0usize;
    let mut total = 0usize;
    let mut worst = 0.0f32;

    for (i, &tid) in g.tokens().iter().enumerate() {
        let off = tid as usize * row_bytes;
        let row = xabe_kernels::quant::dequantize_row_q8_0(&bytes[off..off + row_bytes])
            .expect("embedding row is a whole number of Q8_0 blocks");
        let captured = embed.column(i);
        assert_eq!(row.len(), captured.len());
        for (&a, &b) in row.iter().zip(captured) {
            total += 1;
            if a.to_bits() == b.to_bits() {
                exact += 1;
            }
            worst = worst.max((a - b).abs());
        }
    }

    println!(
        "token_embd.weight dequantized by xabe-gguf + xabe-kernels vs llama.cpp's \
         model.input_embed: {exact}/{total} elements bit-identical, max_abs {worst:.3e}",
    );
    assert_eq!(
        exact, total,
        "Q8_0 dequantization is multiply-only, so this must be exact, not close",
    );

    // And show the transposed reading is genuinely different, so the match
    // above is not a coincidence of a nearly-symmetric tensor.
    let n0 = embed.ne[0] as usize;
    let transposed: Vec<f32> = (0..5).map(|j| embed.f32_data[j * n0]).collect();
    let correct = &embed.column(0)[..5];
    assert_ne!(
        transposed.as_slice(),
        correct,
        "a transposed reading produced the same values; this proof is not discriminating",
    );
    println!("  correct column 0 head: {correct:?}");
    println!("  transposed misreading: {transposed:?}");
}
