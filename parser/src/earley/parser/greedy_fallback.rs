//! Opt-in recovery for globally greedy lexemes.
//!
//! The ordinary parser remains untouched. This module wraps a `ParserState`
//! with sidecar metadata that remembers the most recent accepting boundary.
//! Short failed extensions are replayed; long-lived extensions maintain one
//! incrementally advanced fallback shadow.
//!
//! Core invariants:
//! - `tags` has exactly one entry per `inner.lexer_stack` entry.
//! - `shadow`, when present, represents the same input bytes interpreted from
//!   the saved accepting boundary.
//! - speculative trie traversal is fully reversible through `undos` and
//!   `trie_snapshot`.
//! - definitive shadow promotion retains enough primary history for rollback.

use super::*;

#[derive(Clone)]
struct SavedGreedyParserState {
    stack_len: usize,
    undos_len: usize,
    promoted_flush: bool,
    shadow: Option<Box<SavedGreedyParserState>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GreedyCheckpoint {
    None,
    Replay,
    Shadow,
    Promoted,
}

#[derive(Clone)]
struct GreedyLexerStackUndo {
    trigger_len: usize,
    start: usize,
    old_tail: Vec<LexerState>,
    old_tags: Vec<GreedyCheckpoint>,
}

#[derive(Clone)]
struct GreedyPromotionUndo {
    trigger_len: usize,
    primary: GreedyParserState,
}

#[derive(Clone, Copy)]
struct GreedyTrieSnapshot {
    tags_len: usize,
    undos_len: usize,
    promoted_flush: bool,
}

#[derive(Clone)]
pub(super) struct GreedyParserState {
    pub(super) inner: ParserState,
    tags: Vec<GreedyCheckpoint>,
    undos: Vec<GreedyLexerStackUndo>,
    shadow: Option<Box<GreedyParserState>>,
    promotion_undo: Option<Box<GreedyPromotionUndo>>,
    promoted_flush: bool,
    trie_snapshot: Option<GreedyTrieSnapshot>,
}

impl GreedyCheckpoint {
    #[inline(always)]
    fn advances_shadow(self) -> bool {
        matches!(self, Self::Shadow | Self::Promoted)
    }
}

const GREEDY_SHADOW_THRESHOLD: usize = 64;

#[derive(Clone, Copy, Debug)]
enum GreedyDefinitiveByteResult {
    Rejected,
    Primary,
    Promoted(usize),
}

impl GreedyParserState {
    pub(super) fn new(inner: ParserState) -> Self {
        let tags = vec![GreedyCheckpoint::None; inner.lexer_stack.len()];
        Self {
            inner,
            tags,
            undos: Vec::new(),
            shadow: None,
            promotion_undo: None,
            promoted_flush: false,
            trie_snapshot: None,
        }
    }

    #[inline(always)]
    fn assert_sidecar(&self) {
        debug_assert_eq!(self.tags.len(), self.inner.lexer_stack.len());
    }

    #[inline(always)]
    fn tag(&self) -> GreedyCheckpoint {
        self.assert_sidecar();
        *self.tags.last().unwrap()
    }

    #[inline(always)]
    fn push_lexer_state(&mut self, state: LexerState, tag: GreedyCheckpoint) {
        self.inner.lexer_stack.push(state);
        self.tags.push(tag);
    }

    fn sync_after_inner_push(&mut self, old_len: usize, tag: GreedyCheckpoint) {
        debug_assert!(self.inner.lexer_stack.len() >= old_len);
        self.tags
            .truncate(self.inner.lexer_stack.len().min(self.tags.len()));
        while self.tags.len() < self.inner.lexer_stack.len() {
            self.tags.push(tag);
        }
        if self.inner.lexer_stack.len() > old_len {
            *self.tags.last_mut().unwrap() = tag;
        }
        self.assert_sidecar();
    }

    fn with_shadow<T>(&mut self, f: impl FnOnce(&mut GreedyParserState) -> T) -> Option<T> {
        let mut shadow = self.shadow.take()?;
        debug_assert!(shadow.inner.shared_box.lexer_opt.is_none());
        shadow.inner.shared_box = std::mem::take(&mut self.inner.shared_box);
        let result = f(&mut shadow);
        self.inner.shared_box = std::mem::take(&mut shadow.inner.shared_box);
        self.shadow = Some(shadow);
        Some(result)
    }

    fn clone_for_shadow(&mut self) -> GreedyParserState {
        debug_assert!(self.inner.scratch.definitive);
        debug_assert!(self.undos.is_empty());
        let shared = std::mem::take(&mut self.inner.shared_box);
        let old_shadow = self.shadow.take();
        let mut clone = self.clone();
        self.shadow = old_shadow;
        self.inner.shared_box = shared;

        clone.inner.shared_box = Box::default();
        clone.shadow = None;
        clone.promotion_undo = None;
        clone.promoted_flush = false;
        clone.trie_snapshot = None;
        clone.undos.clear();
        clone.inner.bias_cache = None;
        clone.inner.trie_lexer_stack = usize::MAX;
        clone.inner.trie_grammar_stack = 0;
        clone
    }

    fn active_uses_shadow(&self) -> bool {
        self.promoted_flush || self.tag() == GreedyCheckpoint::Promoted
    }

    fn active_row_is_accepting(&self) -> bool {
        if self.active_uses_shadow() {
            self.shadow
                .as_ref()
                .is_some_and(|shadow| shadow.active_row_is_accepting())
        } else {
            self.inner.row_is_accepting()
        }
    }

    fn maybe_materialize_shadow(&mut self) {
        if !self.inner.scratch.definitive
            || self.shadow.is_some()
            || self.tag() != GreedyCheckpoint::Replay
            || self.checkpoint_distance().unwrap_or(0) < GREEDY_SHADOW_THRESHOLD
        {
            return;
        }

        let mut shadow = self.clone_for_shadow();
        shadow.inner.shared_box = std::mem::take(&mut self.inner.shared_box);
        let ok = shadow.recover_previous_greedy_match(None, false);
        self.inner.shared_box = std::mem::take(&mut shadow.inner.shared_box);
        if ok {
            shadow.undos.clear();
            shadow.inner.bias_cache = None;
            self.shadow = Some(Box::new(shadow));
            *self.tags.last_mut().unwrap() = GreedyCheckpoint::Shadow;
            self.inner.bias_cache = None;
        }
    }

    fn promote_shadow_definitive(&mut self, consumed_byte: bool) {
        let trigger_len = self.inner.bytes.len() + usize::from(consumed_byte);
        let mut shadow = self
            .shadow
            .take()
            .expect("missing greedy shadow during promotion");
        debug_assert!(shadow.inner.scratch.definitive);
        debug_assert!(shadow.inner.shared_box.lexer_opt.is_none());

        // Token and byte histories are identical before the promotion
        // boundary. Move the token mapping to the active fallback instead of
        // copying it; rollback transfers the active history back before
        // truncating to the requested boundary.
        shadow.inner.byte_to_token_idx = std::mem::take(&mut self.inner.byte_to_token_idx);
        shadow.inner.token_idx = self.inner.token_idx;
        shadow.inner.shared_box = std::mem::take(&mut self.inner.shared_box);
        shadow.promotion_undo = None;

        let mut primary = std::mem::replace(self, *shadow);
        // The active fallback already owns the same input bytes. Keeping a
        // second copy in every promotion undo would make repeated promotions
        // unnecessarily expensive. The bytes are transferred back on rollback.
        primary.inner.bytes = Vec::new();
        self.promotion_undo = Some(Box::new(GreedyPromotionUndo {
            trigger_len,
            primary,
        }));
        self.inner.bias_cache = None;
    }

    fn restore_promotions_for_rollback(&mut self, new_len: usize) {
        while self
            .promotion_undo
            .as_ref()
            .is_some_and(|undo| new_len < undo.trigger_len)
        {
            let undo = self.promotion_undo.take().unwrap();
            let shared = std::mem::take(&mut self.inner.shared_box);
            let bytes = std::mem::take(&mut self.inner.bytes);
            let byte_to_token_idx = std::mem::take(&mut self.inner.byte_to_token_idx);
            let token_idx = self.inner.token_idx;
            let mut primary = undo.primary;
            debug_assert!(bytes.len() >= new_len);
            debug_assert!(byte_to_token_idx.len() >= new_len);
            primary.inner.shared_box = shared;
            primary.inner.bytes = bytes;
            primary.inner.byte_to_token_idx = byte_to_token_idx;
            primary.inner.token_idx = token_idx;
            *self = primary;
        }
    }

    fn advance_parser(&mut self, pre: PreLexeme, tag: GreedyCheckpoint) -> bool {
        let old_len = self.inner.lexer_stack.len();
        let ok = self.inner.advance_parser(pre);
        self.sync_after_inner_push(old_len, tag);
        ok
    }

    fn special_pre_lexeme(&mut self, state: StateID, tag: GreedyCheckpoint) -> bool {
        let old_len = self.inner.lexer_stack.len();
        let ok = self.inner.special_pre_lexeme(state);
        self.sync_after_inner_push(old_len, tag);
        ok
    }

    fn checkpoint_start_index(&self) -> Option<usize> {
        let tag = self.tag();
        if !matches!(tag, GreedyCheckpoint::Replay | GreedyCheckpoint::Shadow) {
            return None;
        }
        let curr = self.inner.lexer_state();
        let mut idx = self.inner.lexer_stack.len() - 1;
        while idx > 0 {
            if self.inner.lexer_stack[idx].row_idx != curr.row_idx
                || self.tags[idx] == GreedyCheckpoint::None
            {
                break;
            }
            idx -= 1;
        }
        let start = idx + 1;
        if start >= self.inner.lexer_stack.len()
            || self.inner.lexer_stack[start - 1].row_idx != curr.row_idx
        {
            None
        } else {
            Some(start)
        }
    }

    fn checkpoint_distance(&self) -> Option<usize> {
        Some(self.inner.lexer_stack.len() - self.checkpoint_start_index()?)
    }

    fn checkpoint_lexer_state(&self) -> Option<StateID> {
        let start = self.checkpoint_start_index()?;
        Some(self.inner.lexer_stack[start - 1].lexer_state)
    }

    fn restore_undos_to(&mut self, target_len: usize) {
        while self.undos.len() > target_len {
            let undo = self.undos.pop().unwrap();
            self.inner.lexer_stack.truncate(undo.start);
            self.inner.lexer_stack.extend_from_slice(&undo.old_tail);
            self.tags.truncate(undo.start);
            self.tags.extend_from_slice(&undo.old_tags);
        }
        self.assert_sidecar();
    }

    fn truncate_stacks(&mut self, target_len: usize) {
        while self
            .undos
            .last()
            .is_some_and(|undo| undo.trigger_len > target_len)
        {
            let undo = self.undos.pop().unwrap();
            self.inner.lexer_stack.truncate(undo.start);
            self.inner.lexer_stack.extend_from_slice(&undo.old_tail);
            self.tags.truncate(undo.start);
            self.tags.extend_from_slice(&undo.old_tags);
        }
        self.inner.lexer_stack.truncate(target_len);
        self.tags.truncate(target_len);
        self.assert_sidecar();
    }

    fn pop_lexer_states(&mut self, n: usize) {
        let target = self.inner.lexer_stack.len().saturating_sub(n);
        if !self.inner.scratch.definitive && self.shadow.is_some() {
            let removed = &self.tags[target..];
            let shadow_bytes = removed.iter().filter(|tag| tag.advances_shadow()).count();
            if shadow_bytes > 0 {
                self.with_shadow(|shadow| shadow.pop_lexer_states(shadow_bytes));
            }
        }
        self.truncate_stacks(target);
    }

    fn previous_greedy_checkpoint(&mut self) -> Option<(usize, PreLexeme)> {
        let transition_idx = self.checkpoint_start_index()?;
        let checkpoint = transition_idx.checked_sub(1)?;
        let state = self.inner.lexer_stack[checkpoint];
        let curr = self.inner.lexer_state();
        debug_assert_eq!(state.row_idx, curr.row_idx);
        match self.inner.lexer_mut().try_lexeme_end(state.lexer_state) {
            LexerResult::Lexeme(pre) => Some((checkpoint, pre)),
            _ => panic!("greedy checkpoint was not accepting"),
        }
    }

    fn recover_previous_greedy_match(&mut self, byte: Option<u8>, flush_at_end: bool) -> bool {
        let Some((checkpoint_idx, mut pre_lexeme)) = self.previous_greedy_checkpoint() else {
            return false;
        };

        let old_len = self.inner.lexer_stack.len();
        let old_tail = self.inner.lexer_stack[checkpoint_idx + 1..].to_vec();
        let old_tags = self.tags[checkpoint_idx + 1..].to_vec();
        let replay: Vec<u8> = old_tail
            .iter()
            .filter_map(|state| state.byte)
            .chain(byte)
            .collect();
        if replay.len() != old_tail.len() + usize::from(byte.is_some()) || replay.is_empty() {
            return false;
        }

        let undo_len = self.undos.len();
        let captures = self.inner.captures.clone();
        let row_infos = self.inner.row_infos.clone();
        let backtrack_byte_count = self.inner.backtrack_byte_count;
        let lexer_stack_flush_position = self.inner.lexer_stack_flush_position;

        self.inner.lexer_stack.truncate(checkpoint_idx + 1);
        self.tags.truncate(checkpoint_idx + 1);

        pre_lexeme.byte = replay.first().copied();
        pre_lexeme.byte_next_row = true;
        let mut ok = self.advance_parser(pre_lexeme, GreedyCheckpoint::None);
        for &b in replay.iter().skip(1) {
            if !ok {
                break;
            }
            ok = self.advance_byte_replay(b, self.inner.scratch.log_enabled());
        }
        if ok && flush_at_end {
            ok = self.flush_lexer();
        }

        let expected_len = old_len + usize::from(byte.is_some());
        if ok && self.inner.lexer_stack.len() >= expected_len {
            self.undos.push(GreedyLexerStackUndo {
                trigger_len: self.inner.lexer_stack.len(),
                start: checkpoint_idx + 1,
                old_tail,
                old_tags,
            });
            true
        } else {
            self.restore_undos_to(undo_len);
            self.inner.lexer_stack.truncate(checkpoint_idx + 1);
            self.inner.lexer_stack.extend_from_slice(&old_tail);
            self.tags.truncate(checkpoint_idx + 1);
            self.tags.extend_from_slice(&old_tags);
            self.inner.captures = captures;
            self.inner.row_infos = row_infos;
            self.inner.backtrack_byte_count = backtrack_byte_count;
            self.inner.lexer_stack_flush_position = lexer_stack_flush_position;
            self.assert_sidecar();
            false
        }
    }

    fn advance_byte_replay(&mut self, byte: u8, enable_logging: bool) -> bool {
        let curr = self.inner.lexer_state();
        let result = self
            .inner
            .lexer_mut()
            .advance_greedy(curr.lexer_state, byte, enable_logging);
        if result.is_error() {
            self.recover_previous_greedy_match(Some(byte), false)
        } else {
            self.advance_lexer_or_parser(result, curr, None)
        }
    }

    fn advance_lexer_or_parser(
        &mut self,
        result: GreedyLexerResult,
        curr: LexerState,
        tag_override: Option<GreedyCheckpoint>,
    ) -> bool {
        let inherited = self.tag();
        match result {
            GreedyLexerResult::State(next_state, byte) => {
                self.push_lexer_state(
                    LexerState {
                        row_idx: curr.row_idx,
                        lexer_state: next_state,
                        byte: Some(byte),
                    },
                    tag_override.unwrap_or(inherited),
                );
                true
            }
            GreedyLexerResult::CheckpointStart(next_state, byte) => {
                self.push_lexer_state(
                    LexerState {
                        row_idx: curr.row_idx,
                        lexer_state: next_state,
                        byte: Some(byte),
                    },
                    tag_override.unwrap_or(GreedyCheckpoint::Replay),
                );
                true
            }
            GreedyLexerResult::CheckpointResolved(next_state, byte) => {
                if self.inner.scratch.definitive {
                    self.shadow = None;
                }
                self.push_lexer_state(
                    LexerState {
                        row_idx: curr.row_idx,
                        lexer_state: next_state,
                        byte: Some(byte),
                    },
                    GreedyCheckpoint::None,
                );
                true
            }
            GreedyLexerResult::Error => false,
            GreedyLexerResult::Lexeme(pre) => {
                if self.inner.scratch.definitive {
                    self.shadow = None;
                }
                self.advance_parser(pre, GreedyCheckpoint::None)
            }
            GreedyLexerResult::SpecialToken(state) => {
                if self.inner.scratch.definitive {
                    self.shadow = None;
                }
                self.special_pre_lexeme(state, GreedyCheckpoint::None)
            }
        }
    }

    fn advance_shadow_speculative(&mut self, byte: u8, enable_logging: bool) -> bool {
        self.with_shadow(|shadow| shadow.advance_byte_speculative(byte, enable_logging))
            .unwrap_or(false)
    }

    fn advance_shadow_definitive(&mut self, byte: u8) -> (bool, usize) {
        self.with_shadow(|shadow| shadow.try_push_byte_definitive(Some(byte)))
            .unwrap_or((false, 0))
    }

    fn push_promoted_placeholder(&mut self, curr: LexerState, byte: u8) {
        self.push_lexer_state(
            LexerState {
                row_idx: curr.row_idx,
                lexer_state: curr.lexer_state,
                byte: Some(byte),
            },
            GreedyCheckpoint::Promoted,
        );
    }

    fn advance_byte_speculative(&mut self, byte: u8, enable_logging: bool) -> bool {
        let curr = self.inner.lexer_state();
        if self.tag() == GreedyCheckpoint::Promoted {
            if self.advance_shadow_speculative(byte, enable_logging) {
                self.push_promoted_placeholder(curr, byte);
                return true;
            }
            return false;
        }

        let result = self
            .inner
            .lexer_mut()
            .advance_greedy(curr.lexer_state, byte, enable_logging);
        match result {
            GreedyLexerResult::Error => match self.tag() {
                GreedyCheckpoint::Replay => self.recover_previous_greedy_match(Some(byte), false),
                GreedyCheckpoint::Shadow => {
                    let shadow_ok = self.advance_shadow_speculative(byte, enable_logging);
                    if shadow_ok {
                        self.push_promoted_placeholder(curr, byte);
                        true
                    } else {
                        false
                    }
                }
                GreedyCheckpoint::None => false,
                GreedyCheckpoint::Promoted => unreachable!(),
            },
            GreedyLexerResult::State(_, _) if self.tag() == GreedyCheckpoint::Shadow => {
                let shadow_ok = self.advance_shadow_speculative(byte, enable_logging);
                let tag = if shadow_ok {
                    GreedyCheckpoint::Shadow
                } else {
                    GreedyCheckpoint::None
                };
                self.advance_lexer_or_parser(result, curr, Some(tag))
            }
            GreedyLexerResult::CheckpointResolved(_, _)
            | GreedyLexerResult::Lexeme(_)
            | GreedyLexerResult::SpecialToken(_) => {
                self.advance_lexer_or_parser(result, curr, None)
            }
            GreedyLexerResult::CheckpointStart(_, _) => {
                self.advance_lexer_or_parser(result, curr, Some(GreedyCheckpoint::Replay))
            }
            _ => self.advance_lexer_or_parser(result, curr, None),
        }
    }

    fn advance_byte_definitive(
        &mut self,
        byte: u8,
        enable_logging: bool,
    ) -> GreedyDefinitiveByteResult {
        let curr = self.inner.lexer_state();
        debug_assert!(self.tag() != GreedyCheckpoint::Promoted);
        let result = self
            .inner
            .lexer_mut()
            .advance_greedy(curr.lexer_state, byte, enable_logging);
        match result {
            GreedyLexerResult::Error => match self.tag() {
                GreedyCheckpoint::Replay => {
                    if self.recover_previous_greedy_match(Some(byte), false) {
                        GreedyDefinitiveByteResult::Primary
                    } else {
                        GreedyDefinitiveByteResult::Rejected
                    }
                }
                GreedyCheckpoint::Shadow => {
                    let (ok, backtrack) = self.advance_shadow_definitive(byte);
                    if ok {
                        self.promote_shadow_definitive(true);
                        GreedyDefinitiveByteResult::Promoted(backtrack)
                    } else {
                        GreedyDefinitiveByteResult::Rejected
                    }
                }
                GreedyCheckpoint::None => GreedyDefinitiveByteResult::Rejected,
                GreedyCheckpoint::Promoted => unreachable!(),
            },
            GreedyLexerResult::State(_, _) if self.tag() == GreedyCheckpoint::Shadow => {
                let (shadow_ok, shadow_backtrack) = self.advance_shadow_definitive(byte);
                debug_assert_eq!(shadow_backtrack, 0);
                if !shadow_ok {
                    self.shadow = None;
                }
                let tag = if shadow_ok {
                    GreedyCheckpoint::Shadow
                } else {
                    GreedyCheckpoint::None
                };
                if self.advance_lexer_or_parser(result, curr, Some(tag)) {
                    GreedyDefinitiveByteResult::Primary
                } else {
                    GreedyDefinitiveByteResult::Rejected
                }
            }
            GreedyLexerResult::CheckpointResolved(_, _)
            | GreedyLexerResult::Lexeme(_)
            | GreedyLexerResult::SpecialToken(_) => {
                self.shadow = None;
                if self.advance_lexer_or_parser(result, curr, None) {
                    GreedyDefinitiveByteResult::Primary
                } else {
                    GreedyDefinitiveByteResult::Rejected
                }
            }
            GreedyLexerResult::CheckpointStart(_, _) => {
                self.shadow = None;
                if self.advance_lexer_or_parser(result, curr, Some(GreedyCheckpoint::Replay)) {
                    GreedyDefinitiveByteResult::Primary
                } else {
                    GreedyDefinitiveByteResult::Rejected
                }
            }
            GreedyLexerResult::State(_, _) => {
                if self.advance_lexer_or_parser(result, curr, None) {
                    GreedyDefinitiveByteResult::Primary
                } else {
                    GreedyDefinitiveByteResult::Rejected
                }
            }
        }
    }

    fn trie_started_inner(&mut self, lbl: &str) {
        debug_assert!(self.trie_snapshot.is_none());
        if self.shadow.is_some() {
            self.with_shadow(|shadow| shadow.trie_started_inner(lbl));
        }
        self.trie_snapshot = Some(GreedyTrieSnapshot {
            tags_len: self.tags.len(),
            undos_len: self.undos.len(),
            promoted_flush: self.promoted_flush,
        });
        self.inner.trie_started_inner(lbl);
    }

    fn trie_finished_inner(&mut self) {
        let snapshot = self
            .trie_snapshot
            .take()
            .expect("greedy trie traversal finished without a matching start");
        self.restore_undos_to(snapshot.undos_len);
        self.tags.truncate(snapshot.tags_len);
        self.inner.trie_finished_inner();
        self.tags.truncate(self.inner.lexer_stack.len());
        if self.shadow.is_some() {
            self.with_shadow(|shadow| shadow.trie_finished_inner());
        }
        self.promoted_flush = snapshot.promoted_flush;
        self.assert_sidecar();
    }

    fn flush_lexer(&mut self) -> bool {
        if !self.inner.has_pending_lexeme_bytes() {
            return true;
        }
        if self.active_uses_shadow() {
            return self
                .with_shadow(|shadow| shadow.flush_lexer())
                .unwrap_or(false);
        }
        let curr = self.inner.lexer_state();
        let result: GreedyLexerResult = self
            .inner
            .lexer_mut()
            .try_lexeme_end(curr.lexer_state)
            .into();
        let old_len = self.inner.lexer_stack.len();
        let ok = if result.is_error() {
            match self.tag() {
                GreedyCheckpoint::Replay => self.recover_previous_greedy_match(None, true),
                GreedyCheckpoint::Shadow => {
                    let definitive = self.inner.scratch.definitive;
                    let ok = self
                        .with_shadow(|shadow| shadow.flush_lexer())
                        .unwrap_or(false);
                    if ok {
                        if definitive {
                            self.promote_shadow_definitive(false);
                        } else {
                            self.promoted_flush = true;
                        }
                    }
                    ok
                }
                GreedyCheckpoint::None => false,
                GreedyCheckpoint::Promoted => unreachable!(),
            }
        } else {
            if self.inner.scratch.definitive {
                self.shadow = None;
            }
            self.advance_lexer_or_parser(result, curr, None)
        };
        if self.inner.lexer_stack.len() != old_len
            && (self.inner.scratch.definitive || self.tag() != GreedyCheckpoint::Promoted)
        {
            self.inner.lexer_stack_flush_position = old_len;
        }
        ok
    }

    fn run_speculative<T>(&mut self, lbl: &str, f: impl FnOnce(&mut Self) -> T) -> T {
        self.trie_started_inner(lbl);
        let result = f(self);
        self.trie_finished_inner();
        result
    }

    fn lexer_allows_eos(&mut self) -> bool {
        if self.active_uses_shadow() || self.tag() == GreedyCheckpoint::Shadow {
            return self
                .with_shadow(|shadow| shadow.lexer_allows_eos())
                .unwrap_or(false);
        }
        let state = match self.tag() {
            GreedyCheckpoint::Replay => self.checkpoint_lexer_state(),
            GreedyCheckpoint::None => Some(self.inner.lexer_state().lexer_state),
            GreedyCheckpoint::Shadow | GreedyCheckpoint::Promoted => unreachable!(),
        };
        state.is_some_and(|state| self.inner.lexer_mut().allows_eos(state))
    }

    pub(super) fn compute_bias(&mut self, computer: &dyn BiasComputer, start: &[u8]) -> SimpleVob {
        let t0 = Instant::now();
        let cacheable =
            start.is_empty() && self.tag() == GreedyCheckpoint::None && self.shadow.is_none();
        if cacheable {
            let curr = self.inner.lexer_state();
            let pending = self.inner.has_pending_lexeme_bytes();
            if let Some(cache) = &self.inner.bias_cache {
                if cache.lexer_state == curr.lexer_state
                    && cache.row_idx == curr.row_idx
                    && cache.has_pending_lexeme_bytes == pending
                {
                    let d = t0.elapsed();
                    self.inner.stats.compute_time_us += d.as_micros() as u64;
                    self.inner.perf_counters.compute_bias.record(d);
                    return cache.mask.clone();
                }
            }
        }

        let limits = self.inner.limits.clone();
        self.inner.lexer_mut().dfa.set_fuel(limits.step_lexer_fuel);
        self.inner
            .lexer_mut()
            .dfa
            .set_max_states(limits.max_lexer_states);
        self.inner.max_all_items = self
            .inner
            .stats
            .all_items
            .saturating_add(limits.step_max_items);
        let mut set = {
            let mut rec = GreedyParserRecognizer { state: self };
            computer.compute_bias_greedy(&mut rec, start)
        };
        if self.inner.stats.all_items > self.inner.max_all_items
            && self.inner.parser_error.is_none()
        {
            self.inner.parser_error = Some(format!(
                "Too many items (limit {}; mask); try avoiding single-byte/short lexemes",
                limits.step_max_items
            ));
        }
        self.inner.max_all_items = usize::MAX;
        self.inner.stats.lexer_cost = self.inner.lexer().dfa.total_fuel_spent();

        if self.inner.special_token_marker_token != INVALID_TOKEN {
            set.disallow_token(self.inner.special_token_marker_token);
        }

        if start.is_empty() {
            self.run_speculative("token_ranges", |state| {
                if state.flush_lexer() {
                    state.allow_active_token_ranges(&mut set);
                }
            });
        }

        let eos = computer.trie().eos_token();
        if eos != INVALID_TOKEN && start.is_empty() && self.lexer_allows_eos() {
            set.allow_token(eos);
        }

        if cacheable {
            let curr = self.inner.lexer_state();
            self.inner.bias_cache = Some(BiasCache {
                lexer_state: curr.lexer_state,
                row_idx: curr.row_idx,
                has_pending_lexeme_bytes: self.inner.has_pending_lexeme_bytes(),
                mask: set.clone(),
            });
        } else {
            self.inner.bias_cache = None;
        }

        let d = t0.elapsed();
        self.inner.stats.compute_time_us += d.as_micros() as u64;
        self.inner.perf_counters.compute_bias.record(d);
        set
    }

    fn is_accepting_inner(&mut self) -> bool {
        self.flush_lexer() && self.active_row_is_accepting()
    }

    pub(super) fn is_accepting(&mut self) -> bool {
        self.run_speculative("is_accepting", |state| state.is_accepting_inner())
    }

    fn try_push_byte_definitive(&mut self, byte: Option<u8>) -> (bool, usize) {
        assert!(self.inner.scratch.definitive);
        assert_eq!(self.inner.backtrack_byte_count, 0);
        if let Some(byte) = byte {
            self.inner.stats.definitive_bytes += 1;
            match self.advance_byte_definitive(byte, true) {
                GreedyDefinitiveByteResult::Rejected => (false, 0),
                GreedyDefinitiveByteResult::Promoted(backtrack) => {
                    self.maybe_materialize_shadow();
                    (true, backtrack)
                }
                GreedyDefinitiveByteResult::Primary => {
                    self.inner.bytes.push(byte);
                    let backtrack = std::mem::take(&mut self.inner.backtrack_byte_count);
                    if backtrack > 0 {
                        assert!(self.inner.lexer_spec().has_stop);
                        self.inner.last_force_bytes_len = usize::MAX;
                        self.inner
                            .bytes
                            .truncate(self.inner.bytes.len().saturating_sub(backtrack));
                    }
                    self.maybe_materialize_shadow();
                    (true, backtrack)
                }
            }
        } else {
            self.shadow = None;
            let curr = self.inner.lexer_state();
            let result: GreedyLexerResult = self
                .inner
                .lexer_mut()
                .force_lexeme_end(curr.lexer_state)
                .into();
            if self.advance_lexer_or_parser(result, curr, None) {
                (true, 0)
            } else {
                (false, 0)
            }
        }
    }

    pub(super) fn rollback(&mut self, n_bytes: usize) -> Result<()> {
        debug!("greedy rollback: {} bytes", n_bytes);
        ensure!(self.inner.parser_error.is_none(), "rollback: parser error");
        self.inner.assert_definitive();
        ensure!(
            n_bytes <= self.inner.byte_to_token_idx.len(),
            "rollback: too many bytes {} > {}",
            n_bytes,
            self.inner.byte_to_token_idx.len()
        );

        let new_len = self.inner.byte_to_token_idx.len() - n_bytes;
        self.restore_promotions_for_rollback(new_len);
        self.shadow = None;
        self.inner.byte_to_token_idx.truncate(new_len);
        self.inner.bytes.truncate(new_len);
        self.truncate_stacks(new_len + 1);
        self.undos.clear();

        self.inner.row_infos.truncate(self.inner.num_rows());
        self.inner.token_idx = *self.inner.byte_to_token_idx.last().unwrap_or(&0) as usize;
        self.inner.last_force_bytes_len = usize::MAX;
        self.inner.bias_cache = None;
        self.inner.lexer_stack_top_eos = false;
        self.inner.rows_valid_end = self.inner.num_rows();
        if let Some(tag) = self.tags.last_mut() {
            if matches!(*tag, GreedyCheckpoint::Shadow | GreedyCheckpoint::Promoted) {
                *tag = GreedyCheckpoint::Replay;
            }
        }
        self.maybe_materialize_shadow();
        self.inner.assert_definitive();
        self.assert_sidecar();
        Ok(())
    }

    fn save_state(&self) -> SavedGreedyParserState {
        self.assert_sidecar();
        SavedGreedyParserState {
            stack_len: self.inner.lexer_stack.len(),
            undos_len: self.undos.len(),
            promoted_flush: self.promoted_flush,
            shadow: self
                .shadow
                .as_ref()
                .map(|shadow| Box::new(shadow.save_state())),
        }
    }

    fn restore_state(&mut self, state: SavedGreedyParserState) {
        let pop = self.inner.lexer_stack.len().saturating_sub(state.stack_len);
        self.pop_lexer_states(pop);
        self.restore_undos_to(state.undos_len);
        self.tags.truncate(state.stack_len);
        self.promoted_flush = state.promoted_flush;
        if let (Some(shadow), Some(saved_shadow)) = (self.shadow.as_mut(), state.shadow) {
            shadow.restore_state(*saved_shadow);
        }
        self.assert_sidecar();
    }

    fn token_range_lexemes(&self) -> Vec<&LexemeSpec> {
        let state = self.inner.lexer_state().lexer_state;
        let possible = self.inner.lexer().possible_lexemes(state);
        self.inner.lexer_spec().token_range_lexemes(possible)
    }

    fn active_token_range_match(&mut self, tok_id: TokenId) -> Option<LexemeIdx> {
        if self.active_uses_shadow() {
            self.with_shadow(|shadow| shadow.active_token_range_match(tok_id))
                .flatten()
        } else {
            self.inner
                .token_range_lexemes()
                .into_iter()
                .find(|spec| spec.contains_token(tok_id))
                .map(|spec| spec.idx)
        }
    }

    fn allow_active_token_ranges(&mut self, set: &mut SimpleVob) {
        if self.active_uses_shadow() {
            self.with_shadow(|shadow| shadow.allow_active_token_ranges(set));
        } else {
            for spec in self.token_range_lexemes() {
                for range in &spec.token_ranges {
                    set.allow_range(range.clone());
                }
            }
        }
    }

    fn flush_and_check_numeric(&mut self, tok_id: TokenId) -> Option<LexemeIdx> {
        if self.flush_lexer() {
            self.active_token_range_match(tok_id)
        } else {
            None
        }
    }

    fn add_numeric_token(&mut self, idx: LexemeIdx, tok_bytes: &[u8]) -> Result<()> {
        let lexer_state = self.inner.lexer_state();
        for &byte in &tok_bytes[..tok_bytes.len() - 1] {
            self.push_lexer_state(
                LexerState {
                    byte: Some(byte),
                    ..lexer_state
                },
                GreedyCheckpoint::None,
            );
        }
        if self.inner.scratch.definitive {
            self.inner.bytes.extend_from_slice(tok_bytes);
            for _ in 0..tok_bytes.len() {
                self.inner
                    .byte_to_token_idx
                    .push(self.inner.token_idx.try_into().unwrap());
            }
        }
        let ok = self.advance_parser(
            PreLexeme::just_idx(MatchingLexemesIdx::Single(idx)),
            GreedyCheckpoint::None,
        );
        ensure!(ok, "failed to advance parser after adding numeric token");
        if self.inner.scratch.definitive {
            let row_idx = self.inner.num_rows() - 1;
            self.inner.row_infos[row_idx].apply_token_idx(self.inner.token_idx);
        }
        Ok(())
    }

    pub(super) fn validate_tokens(&mut self, tokens: &[TokenId]) -> usize {
        self.inner.assert_definitive();
        self.run_speculative("validate_tokens", |state| {
            state.inner.scratch.log_override = true;
            let mut applied_idx = state.inner.byte_to_token_idx.len();
            let tok_env = state.inner.tok_env.clone();
            let trie = tok_env.tok_trie();
            let eos = trie.eos_token();
            let mut recog = GreedyParserRecognizer { state };

            for (tidx, &tok) in tokens.iter().enumerate() {
                let state = &mut recog.state;
                if tok == eos {
                    return if applied_idx == state.inner.bytes.len() && state.is_accepting_inner() {
                        tidx + 1
                    } else {
                        tidx
                    };
                }

                if applied_idx >= state.inner.bytes.len() {
                    let saved = state.save_state();
                    if let Some(idx) = state.flush_and_check_numeric(tok) {
                        let numeric_bytes = trie.decode_as_special(tok);
                        let ok = state.add_numeric_token(idx, &numeric_bytes);
                        assert!(ok.is_ok());
                        continue;
                    }
                    state.restore_state(saved);
                }

                let token_bytes = trie.decode_raw(&[tok]);
                let token_bytes = if applied_idx < state.inner.bytes.len()
                    && state.inner.bytes[applied_idx] == TokTrie::SPECIAL_TOKEN_MARKER
                {
                    trie.decode_as_special(tok)
                } else {
                    token_bytes
                };

                for &byte in &token_bytes {
                    if applied_idx < recog.state.inner.bytes.len() {
                        if recog.state.inner.bytes[applied_idx] == byte {
                            applied_idx += 1;
                        } else {
                            return tidx;
                        }
                    } else if byte != TokTrie::SPECIAL_TOKEN_MARKER && recog.try_push_byte(byte) {
                        continue;
                    } else {
                        return tidx;
                    }
                }
            }
            tokens.len()
        })
    }

    fn apply_token_row_metadata(&mut self, applied_idx0: usize) {
        let mut row_to_apply = self.inner.num_rows() - 1;
        while row_to_apply > 0 {
            if self.inner.row_infos[row_to_apply].start_byte_idx <= applied_idx0 {
                break;
            }
            row_to_apply -= 1;
        }
        for idx in row_to_apply..self.inner.num_rows() {
            if self.inner.row_infos[idx].start_byte_idx >= applied_idx0 {
                self.inner.row_infos[idx].set_token_idx(self.inner.token_idx);
            } else {
                self.inner.row_infos[idx].apply_token_idx(self.inner.token_idx);
            }
        }
    }

    fn finish_shadow_token(&mut self, applied_idx0: usize) {
        let suffix = self.inner.byte_to_token_idx[applied_idx0..].to_vec();
        let token_idx = self.inner.token_idx;
        if let Some(shadow) = self.shadow.as_mut() {
            debug_assert!(shadow.inner.bytes.len() >= applied_idx0 + suffix.len());
            shadow.inner.byte_to_token_idx.truncate(applied_idx0);
            shadow.inner.byte_to_token_idx.extend_from_slice(&suffix);
            shadow.inner.token_idx = token_idx;
            shadow.apply_token_row_metadata(applied_idx0);
            shadow.finish_shadow_token(applied_idx0);
        }
    }

    pub(super) fn increment_token_idx_recursive(&mut self) {
        self.inner.token_idx += 1;
        if let Some(shadow) = self.shadow.as_mut() {
            shadow.increment_token_idx_recursive();
        }
    }

    pub(super) fn resize_token_mapping_recursive(&mut self, new_len: usize, value: u32) {
        self.inner.byte_to_token_idx.resize(new_len, value);
        if let Some(shadow) = self.shadow.as_mut() {
            shadow.resize_token_mapping_recursive(new_len, value);
        }
    }

    pub(super) fn truncate_token_mapping_recursive(&mut self, new_len: usize) {
        self.inner.byte_to_token_idx.truncate(new_len);
        if let Some(shadow) = self.shadow.as_mut() {
            shadow.truncate_token_mapping_recursive(new_len);
        }
    }

    pub(super) fn temperature(&self) -> Option<f32> {
        let mut temp = self.inner.temperature().unwrap_or(0.0);
        if let Some(shadow) = &self.shadow {
            temp = temp.max(shadow.temperature().unwrap_or(0.0));
        }
        if temp < 0.00000001 {
            None
        } else {
            Some(temp)
        }
    }

    pub(super) fn apply_token(&mut self, tok_bytes: &[u8], tok_id: TokenId) -> Result<usize> {
        self.inner.assert_definitive();
        let mut check_lexer_max_tokens = false;
        let applied_idx0 = self.inner.byte_to_token_idx.len();

        if self.inner.tok_env.tok_trie().token(tok_id) == tok_bytes
            && self.inner.byte_to_token_idx.len() == self.inner.bytes.len()
        {
            let applies = self
                .run_speculative("numeric_apply_token", |state| {
                    state.flush_and_check_numeric(tok_id)
                })
                .is_some();
            if applies {
                let row_idx = self.inner.num_rows() - 1;
                self.inner.row_infos[row_idx].apply_token_idx(self.inner.token_idx);
                self.inner.lexer_stack_flush_position = 0;
                let idx = self.flush_and_check_numeric(tok_id).unwrap();
                self.add_numeric_token(idx, tok_bytes)?;
                if self.inner.lexer_stack_flush_position > 0 {
                    let position = self.inner.lexer_stack_flush_position;
                    assert!(position + 1 < self.inner.lexer_stack.len());
                    self.inner.lexer_stack.remove(position);
                    self.tags.remove(position);
                }
                self.apply_token_row_metadata(applied_idx0);
                self.finish_shadow_token(applied_idx0);
                self.inner.assert_definitive();
                self.assert_sidecar();
                return Ok(0);
            }
        }

        for (bidx, &byte) in tok_bytes.iter().enumerate() {
            check_lexer_max_tokens = false;
            let applied_idx = self.inner.byte_to_token_idx.len();
            if applied_idx >= self.inner.bytes.len() {
                assert_eq!(applied_idx, self.inner.bytes.len());
                let row_idx = self.inner.num_rows() - 1;
                self.inner.row_infos[row_idx].apply_token_idx(self.inner.token_idx);
                let (ok, backtrack) = self.try_push_byte_definitive(Some(byte));
                if !ok {
                    bail!(
                        "token {:?} doesn't satisfy the grammar; byte {:?} fails parse",
                        String::from_utf8_lossy(tok_bytes),
                        byte as char,
                    );
                }
                if backtrack > 0 {
                    self.truncate_token_mapping_recursive(self.inner.bytes.len());
                    return Ok(backtrack + tok_bytes.len() - bidx - 1);
                }
                if row_idx == self.inner.num_rows() - 1 {
                    check_lexer_max_tokens = true;
                }
            } else {
                if bidx == 0 && self.inner.bytes[applied_idx] == TokTrie::SPECIAL_TOKEN_MARKER {
                    if let Some(token_id) =
                        self.inner.tok_env.tok_trie().token_id_at_bytes(tok_bytes)
                    {
                        if let Some((len, token_id2)) =
                            parse_numeric_token(&self.inner.bytes[applied_idx + 1..])
                        {
                            if token_id == token_id2 {
                                let token_idx = self.inner.token_idx.try_into().unwrap();
                                for _ in 0..len + 1 {
                                    self.inner.byte_to_token_idx.push(token_idx);
                                }
                                break;
                            }
                        }
                    }
                }
                if self.inner.bytes[applied_idx] != byte {
                    bail!(
                        "token {:?} doesn't satisfy the grammar; forced bytes: got {:?}; applying {:?}",
                        String::from_utf8_lossy(tok_bytes),
                        self.inner.bytes[applied_idx] as char,
                        byte as char,
                    );
                }
            }
            self.inner
                .byte_to_token_idx
                .push(self.inner.token_idx.try_into().unwrap());
        }

        self.apply_token_row_metadata(applied_idx0);
        self.finish_shadow_token(applied_idx0);

        if check_lexer_max_tokens {
            let row_idx = self.inner.num_rows() - 1;
            let mut pop_classes = HashSet::default();
            let mut stack_ptr = self.inner.rows[row_idx].grammar_stack_ptr;
            while stack_ptr.as_usize() > 0 {
                let top = &self.inner.scratch.grammar_stack[stack_ptr.as_usize()];
                if top.token_horizon <= self.inner.token_idx as u32 + 1 {
                    pop_classes.insert(top.grammar_id);
                    stack_ptr = top.back_ptr;
                } else {
                    break;
                }
            }

            let info = &self.inner.row_infos[row_idx];
            let info_tokens = std::cmp::max(
                0,
                self.inner.token_idx as isize + 1 - info.token_idx_start as isize,
            ) as usize;
            let lexer_state = self.inner.lexer_state().lexer_state;
            let mut limit = self.inner.lexer_spec().alloc_lexeme_set();
            let mut num_limit = 0;
            {
                let possible = self.inner.lexer().possible_lexemes(lexer_state);
                for lexeme in possible.iter() {
                    let spec = self.inner.lexer_spec().lexeme_spec(lexeme);
                    if info_tokens < spec.max_tokens() && !pop_classes.contains(&spec.class()) {
                        limit.add(lexeme);
                    } else {
                        num_limit += 1;
                    }
                }
            }
            if num_limit > 0 {
                let new_state = self.inner.lexer_mut().limit_state_to(lexer_state, &limit);
                if new_state.is_dead() {
                    let (ok, backtrack) = self.try_push_byte_definitive(None);
                    assert_eq!(backtrack, 0);
                    if !ok {
                        return Ok(0);
                    }
                } else {
                    self.inner.lexer_stack.last_mut().unwrap().lexer_state = new_state;
                }
            }
        }

        let item_count = self.inner.curr_row().item_indices().count();
        if item_count > self.inner.limits.max_items_in_row {
            bail!(
                "Current row has {} items; max is {}; consider making your grammar left-recursive if it's right-recursive",
                item_count,
                self.inner.limits.max_items_in_row,
            );
        }
        self.inner.assert_definitive();
        self.assert_sidecar();
        Ok(0)
    }

    fn forced_byte(&mut self) -> Option<u8> {
        if self.is_accepting() {
            return None;
        }
        let lexer_state = self.inner.lexer_state();
        let quick_res = self.inner.lexer_mut().next_byte(lexer_state.lexer_state);
        if self.tag() == GreedyCheckpoint::None && self.shadow.is_none() {
            if let NextByte::ForcedByte(byte) = quick_res {
                return Some(byte);
            }
        }

        self.run_speculative("forced_byte", |state| {
            let mut recognizer = GreedyParserRecognizer { state };
            if let NextByte::SomeBytes2([a, b]) = quick_res {
                if recognizer.try_push_byte(a) {
                    recognizer.pop_bytes(1);
                    if recognizer.try_push_byte(b) {
                        recognizer.pop_bytes(1);
                        return None;
                    }
                }
            }

            let first = quick_res.some_bytes().first().copied().unwrap_or(b' ');
            let mut byte = first;
            let mut forced = None;
            loop {
                if recognizer.try_push_byte(byte) {
                    recognizer.pop_bytes(1);
                    if forced.is_some() {
                        return None;
                    }
                    forced = Some(byte);
                }
                byte = byte.wrapping_add(1);
                if byte == first {
                    break;
                }
            }
            forced
        })
    }

    fn unique_token_range_id(&mut self) -> Option<TokenId> {
        if self.active_uses_shadow() {
            return self
                .with_shadow(|shadow| shadow.unique_token_range_id())
                .flatten();
        }
        let mut unique = None;
        'spec: for spec in self.token_range_lexemes() {
            for range in &spec.token_ranges {
                if range.start() == range.end() {
                    let token = *range.start();
                    if unique.is_none() || unique == Some(token) {
                        unique = Some(token);
                    } else {
                        unique = None;
                        break 'spec;
                    }
                } else {
                    unique = None;
                    break 'spec;
                }
            }
        }
        unique
    }

    pub(super) fn force_bytes(&mut self) {
        self.inner.assert_definitive();
        if !self.inner.needs_force_bytes() {
            return;
        }
        let limit = self.inner.limits.step_max_items;
        self.inner.max_all_items = self.inner.stats.all_items.saturating_add(limit);
        while let Some(byte) = self.forced_byte() {
            if byte == TokTrie::SPECIAL_TOKEN_MARKER {
                assert!(!self.inner.has_pending_lexeme_bytes());
                let Some(token_id) = self.unique_token_range_id() else {
                    break;
                };
                let mut bytes = format!("X[{token_id}]").into_bytes();
                bytes[0] = TokTrie::SPECIAL_TOKEN_MARKER;
                let mut all_ok = true;
                for byte in bytes {
                    let (ok, backtrack) = self.try_push_byte_definitive(Some(byte));
                    assert_eq!(backtrack, 0);
                    if !ok {
                        all_ok = false;
                        break;
                    }
                }
                if !all_ok {
                    break;
                }
                continue;
            }

            let (ok, backtrack) = self.try_push_byte_definitive(Some(byte));
            assert_eq!(backtrack, 0);
            if !ok {
                break;
            }
        }
        if self.inner.stats.all_items > self.inner.max_all_items
            && self.inner.parser_error.is_none()
        {
            self.inner.parser_error = Some(format!(
                "Too many items (limit {limit}; ff_tokens); try avoiding single-byte/short lexemes"
            ));
        }
        self.inner.max_all_items = usize::MAX;
        self.inner.last_force_bytes_len = self.inner.bytes.len();
        self.inner.assert_definitive();
    }

    pub(super) fn scan_eos(&mut self) -> bool {
        self.inner.assert_definitive();
        let lexer_eos = self.lexer_allows_eos();
        let prev_len = self.inner.lexer_stack.len();
        if !self.flush_lexer() {
            return false;
        }
        if lexer_eos {
            return true;
        }
        if self.inner.lexer_stack.len() != prev_len {
            self.inner.lexer_stack_top_eos = true;
        }
        false
    }
}

#[doc(hidden)]
pub struct GreedyParserRecognizer<'a> {
    pub(super) state: &'a mut GreedyParserState,
}

impl GreedyParserRecognizer<'_> {
    pub fn has_greedy_checkpoint(&self) -> bool {
        self.state.tag() != GreedyCheckpoint::None || self.state.shadow.is_some()
    }

    pub fn lexer_mut(&mut self) -> &mut Lexer {
        self.state.inner.lexer_mut()
    }

    pub fn lexer(&self) -> &Lexer {
        self.state.inner.lexer()
    }

    pub fn lexer_state(&self) -> StateID {
        self.state.inner.lexer_state().lexer_state
    }

    pub fn stats_mut(&mut self) -> &mut ParserStats {
        &mut self.state.inner.stats
    }

    pub fn metrics_mut(&mut self) -> &mut ParserMetrics {
        &mut self.state.inner.metrics
    }
}

impl Recognizer for GreedyParserRecognizer<'_> {
    #[inline(always)]
    fn pop_bytes(&mut self, num: usize) {
        if ITEM_TRACE {
            self.state
                .inner
                .trace_byte_stack
                .truncate(self.state.inner.trace_byte_stack.len() - num);
        }
        self.state.pop_lexer_states(num);
    }

    fn collapse(&mut self) {}

    fn trie_started(&mut self, lbl: &str) {
        self.state.trie_started_inner(lbl);
    }

    fn trie_finished(&mut self) {
        self.state.trie_finished_inner();
    }

    #[inline(always)]
    fn try_push_byte(&mut self, byte: u8) -> bool {
        if ITEM_TRACE {
            self.state.inner.trace_byte_stack.push(byte);
        }
        let ok = self.state.advance_byte_speculative(byte, false);
        if ITEM_TRACE && !ok {
            self.state.inner.trace_byte_stack.pop();
        }
        ok
    }

    fn save_stats(&mut self, nodes_walked: usize) {
        self.state.inner.stats.trie_nodes_walked += nodes_walked;
    }
}
