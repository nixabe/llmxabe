//! The vocabulary a grammar is masked against, and the walk that turns a
//! cursor set into a token mask.
//!
//! llama.cpp asks the same question token by token
//! (`llama_grammar_accept_str` over each candidate's piece). Doing that over
//! a quarter of a million tokens at every constrained step is most of the
//! cost, so the pieces are held in a byte trie instead: the walk descends
//! only where the grammar can still go, and a whole subtree dies with its
//! first impossible byte. In the markup — a tool name, a parameter name, a
//! closing tag — that prunes to a handful of nodes.
//!
//! The one place the trie does not help is a long free-text parameter value,
//! where every token is admissible and the walk would visit the whole thing.
//! That case has its own shortcut; see [`Vocab::mask`].

use crate::machine::{CLOSE, Cursor, Machine, StepScratch};

/// The token pieces a constraint masks over, indexed by token id.
#[derive(Debug)]
pub struct Vocab {
    /// Node ranges into the edge arrays; one more entry than there are nodes.
    edge_start: Vec<u32>,
    edge_byte: Vec<u8>,
    edge_target: Vec<u32>,
    /// The token that ends at each node, or `-1`.
    node_token: Vec<i32>,
    /// Every token whose bytes contain the first byte of [`CLOSE`], paired
    /// with those bytes. The free-text shortcut checks only these.
    breaking: Vec<(u32, Vec<u8>)>,
    /// Bitmask of the tokens the free-text shortcut admits outright: every
    /// token that is neither end-of-generation nor in `breaking`.
    free_text: Vec<u64>,
    /// End-of-generation tokens, admissible only where the grammar is
    /// satisfied.
    eog: Vec<u32>,
    /// Every piece end to end, indexed by `piece_start`, so a constraint can
    /// look up the bytes of the token that was actually emitted without the
    /// engine having to carry a tokenizer.
    piece_bytes: Vec<u8>,
    piece_start: Vec<u32>,
    len: usize,
}

impl Vocab {
    /// Build the trie over `pieces`, the byte spelling of every token in
    /// vocabulary order. A token with no byte spelling — an unused slot in a
    /// padded vocabulary — is left out and can never be emitted under a
    /// constraint.
    ///
    /// End-of-generation tokens are held aside rather than put in the trie:
    /// they end the response, they do not extend it.
    pub fn new(pieces: &[Vec<u8>], eog: &[u32]) -> Self {
        let len = pieces.len();
        let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut node_token: Vec<i32> = vec![-1];
        let mut breaking = Vec::new();
        let mut piece_bytes = Vec::new();
        let mut piece_start = Vec::with_capacity(len + 1);
        let mut free_text = vec![u64::MAX; len.div_ceil(64)];
        // The tail of the last word holds bits past the vocabulary.
        if !len.is_multiple_of(64) {
            let last = free_text.len() - 1;
            free_text[last] = (1u64 << (len % 64)) - 1;
        }
        let clear = |mask: &mut Vec<u64>, token: usize| {
            mask[token / 64] &= !(1u64 << (token % 64));
        };
        for &token in eog {
            if (token as usize) < len {
                clear(&mut free_text, token as usize);
            }
        }
        for (token, piece) in pieces.iter().enumerate() {
            piece_start.push(piece_bytes.len() as u32);
            piece_bytes.extend_from_slice(piece);
            if piece.is_empty() || eog.contains(&(token as u32)) {
                clear(&mut free_text, token);
                continue;
            }
            if piece.contains(&CLOSE[0]) {
                breaking.push((token as u32, piece.clone()));
                clear(&mut free_text, token);
            }
            let mut node = 0usize;
            for &byte in piece {
                node = match children[node].iter().find(|(edge, _)| *edge == byte) {
                    Some((_, next)) => *next as usize,
                    None => {
                        children.push(Vec::new());
                        node_token.push(-1);
                        let next = children.len() - 1;
                        children[node].push((byte, next as u32));
                        next
                    }
                };
            }
            node_token[node] = token as i32;
        }
        piece_start.push(piece_bytes.len() as u32);
        let mut edge_start = Vec::with_capacity(children.len() + 1);
        let mut edge_byte = Vec::new();
        let mut edge_target = Vec::new();
        for edges in &mut children {
            edge_start.push(edge_byte.len() as u32);
            edges.sort_unstable_by_key(|(byte, _)| *byte);
            for &(byte, target) in edges.iter() {
                edge_byte.push(byte);
                edge_target.push(target);
            }
        }
        edge_start.push(edge_byte.len() as u32);
        Self {
            edge_start,
            edge_byte,
            edge_target,
            node_token,
            breaking,
            free_text,
            eog: eog.to_vec(),
            piece_bytes,
            piece_start,
            len,
        }
    }

    /// The bytes of one token, or nothing for a token outside the
    /// vocabulary.
    pub(crate) fn piece(&self, token: usize) -> &[u8] {
        if token >= self.len {
            return &[];
        }
        let (from, to) = (
            self.piece_start[token] as usize,
            self.piece_start[token + 1] as usize,
        );
        &self.piece_bytes[from..to]
    }

    /// How many tokens the mask covers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drive `logits` to `-inf` everywhere the grammar cannot go next.
    ///
    /// Every token the machine admits keeps its logit; the rest are excluded
    /// before the sampler or the argmax ever sees them.
    pub(crate) fn mask(
        &self,
        machine: &Machine,
        cursors: &[Cursor],
        scratch: &mut MaskScratch,
        logits: &mut [f32],
    ) {
        scratch.allow.clear();
        scratch.allow.resize(self.len.div_ceil(64), 0);
        if machine.is_free_text(cursors) {
            // Nothing of the delimiter is matched, so a token that cannot
            // start matching it leaves the cursor set exactly as it is. Only
            // the ones that can need walking.
            scratch.allow.copy_from_slice(&self.free_text);
            for (token, piece) in &self.breaking {
                if self.survives(machine, cursors, piece, scratch) {
                    scratch.allow[*token as usize / 64] |= 1 << (*token % 64);
                }
            }
        } else {
            self.walk(machine, cursors, scratch);
        }
        if machine.accepts_end(cursors) {
            for &token in &self.eog {
                if (token as usize) < self.len {
                    scratch.allow[token as usize / 64] |= 1 << (token % 64);
                }
            }
        }
        // A word at a time, because both interesting cases are uniform: in
        // the markup almost every word is empty, and in a free-text value
        // almost every word is full. Touching a quarter of a million logits
        // one at a time cost more than the trie walk did.
        for (word, &allow) in scratch.allow.iter().enumerate() {
            if allow == u64::MAX {
                continue;
            }
            let from = word * 64;
            let to = (from + 64).min(self.len);
            if allow == 0 {
                logits[from..to].fill(f32::NEG_INFINITY);
                continue;
            }
            for (bit, logit) in logits[from..to].iter_mut().enumerate() {
                if allow & (1 << bit) == 0 {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        // A logits row wider than the vocabulary carries ids that spell
        // nothing; none of them may be chosen.
        if logits.len() > self.len {
            logits[self.len..].fill(f32::NEG_INFINITY);
        }
    }

    /// Whether feeding `piece` leaves the machine alive.
    fn survives(
        &self,
        machine: &Machine,
        cursors: &[Cursor],
        piece: &[u8],
        scratch: &mut MaskScratch,
    ) -> bool {
        scratch.probe.clear();
        scratch.probe.extend_from_slice(cursors);
        piece
            .iter()
            .all(|&byte| machine.step(&mut scratch.probe, byte, &mut scratch.step))
    }

    /// Depth-first over the trie, carrying the cursor set down each edge and
    /// abandoning a subtree the moment the machine dies on it.
    fn walk(&self, machine: &Machine, cursors: &[Cursor], scratch: &mut MaskScratch) {
        let MaskScratch {
            allow,
            levels,
            frames,
            step,
            ..
        } = scratch;
        if levels.is_empty() {
            levels.push(Vec::new());
        }
        levels[0].clear();
        levels[0].extend_from_slice(cursors);
        frames.clear();
        frames.push(Frame {
            node: 0,
            edge: self.edge_start[0],
            depth: 0,
        });
        while let Some(frame) = frames.last_mut() {
            let node = frame.node as usize;
            if frame.edge >= self.edge_start[node + 1] {
                frames.pop();
                continue;
            }
            let edge = frame.edge as usize;
            let depth = frame.depth;
            frame.edge += 1;
            let byte = self.edge_byte[edge];
            let target = self.edge_target[edge];
            if levels.len() <= depth + 1 {
                levels.push(Vec::new());
            }
            let (upper, lower) = levels.split_at_mut(depth + 1);
            lower[0].clear();
            lower[0].extend_from_slice(&upper[depth]);
            if !machine.step(&mut lower[0], byte, step) {
                continue;
            }
            let token = self.node_token[target as usize];
            if token >= 0 {
                allow[token as usize / 64] |= 1 << (token as usize % 64);
            }
            frames.push(Frame {
                node: target,
                edge: self.edge_start[target as usize],
                depth: depth + 1,
            });
        }
    }
}

#[derive(Debug)]
struct Frame {
    node: u32,
    edge: u32,
    depth: usize,
}

/// Buffers the mask walk reuses, so a constrained step allocates nothing
/// after the first one.
#[derive(Debug, Default)]
pub struct MaskScratch {
    allow: Vec<u64>,
    levels: Vec<Vec<Cursor>>,
    frames: Vec<Frame>,
    probe: Vec<Cursor>,
    step: StepScratch,
}
