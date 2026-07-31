use super::*;

const SHADOW_THRESHOLD: usize = 64;

#[derive(Clone)]
struct Snapshot {
    state: ParserState,
    replay: GreedyReplay,
}

#[derive(Clone)]
struct SpecUndo {
    trigger_lexer_len: usize,
    previous: Box<Snapshot>,
    shadow_promotion: bool,
}

#[derive(Clone)]
struct PromotionUndo {
    trigger_byte_len: usize,
    previous: Snapshot,
}

#[derive(Clone, Default)]
pub(super) struct GreedyReplay {
    accepting: Vec<usize>,
    shadow: Option<Box<Snapshot>>,
    shadow_source: Option<usize>,
    spec_undo: Option<SpecUndo>,
    promotion_undo: Option<Box<PromotionUndo>>,
}

struct Context<'a> {
    state: &'a mut ParserState,
    replay: &'a mut GreedyReplay,
}

fn with_replay<T>(state: &mut ParserState, f: impl FnOnce(&mut Context<'_>) -> T) -> T {
    let Some(mut replay) = state.greedy_replay.take() else {
        panic!("greedy replay requested for an ordinary parser");
    };
    let result = {
        let mut context = Context {
            state,
            replay: &mut replay,
        };
        f(&mut context)
    };
    state.greedy_replay = Some(replay);
    result
}

fn with_snapshot<T>(
    owner: &mut ParserState,
    snapshot: &mut Snapshot,
    f: impl FnOnce(&mut ParserState) -> T,
) -> T {
    snapshot.state.shared_box = std::mem::take(&mut owner.shared_box);
    snapshot.state.greedy_replay = Some(Box::new(std::mem::take(&mut snapshot.replay)));
    let result = f(&mut snapshot.state);
    snapshot.replay = *snapshot.state.greedy_replay.take().unwrap();
    owner.shared_box = std::mem::take(&mut snapshot.state.shared_box);
    result
}

impl Context<'_> {
    fn snapshot(&mut self) -> Snapshot {
        debug_assert!(self.state.greedy_replay.is_none());
        let shared = std::mem::take(&mut self.state.shared_box);
        let state = self.state.clone();
        self.state.shared_box = shared;
        debug_assert!(state.shared_box.lexer_opt.is_none());
        Snapshot {
            state,
            replay: self.replay.clone(),
        }
    }

    fn restore(&mut self, mut snapshot: Snapshot) {
        snapshot.state.shared_box = std::mem::take(&mut self.state.shared_box);
        *self.state = snapshot.state;
        *self.replay = snapshot.replay;
    }

    fn latest_accepting(&mut self) -> Option<(usize, PreLexeme, Vec<u8>)> {
        let top = self.state.lexer_state();

        // Speculative candidates are short tokenizer paths. Scan only the
        // bytes added by the current trie traversal; committed history is
        // tracked incrementally below.
        if !self.state.scratch.definitive {
            let floor = self
                .state
                .trie_lexer_stack
                .min(self.state.lexer_stack.len());
            for idx in (floor..self.state.lexer_stack.len()).rev() {
                let item = self.state.lexer_stack[idx];
                if item.row_idx != top.row_idx {
                    break;
                }
                if let LexerResult::Lexeme(pre) =
                    self.state.lexer_mut().try_lexeme_end(item.lexer_state)
                {
                    let replay = self.state.lexer_stack[idx + 1..]
                        .iter()
                        .map(|state| state.byte)
                        .collect::<Option<Vec<_>>>()?;
                    return Some((idx, pre, replay));
                }
            }
        }

        let idx = *self.replay.accepting.last()?;
        let item = *self.state.lexer_stack.get(idx)?;
        if item.row_idx != top.row_idx {
            return None;
        }
        let LexerResult::Lexeme(pre) = self.state.lexer_mut().try_lexeme_end(item.lexer_state)
        else {
            return None;
        };
        let replay = self.state.lexer_stack[idx + 1..]
            .iter()
            .map(|state| state.byte)
            .collect::<Option<Vec<_>>>()?;
        Some((idx, pre, replay))
    }

    fn truncate_lexer(&mut self, target: usize) {
        self.state.lexer_stack.truncate(target);
        while self
            .replay
            .accepting
            .last()
            .is_some_and(|idx| *idx >= target)
        {
            self.replay.accepting.pop();
        }
        if self.replay.shadow_source.is_some_and(|idx| idx >= target) {
            self.replay.shadow = None;
            self.replay.shadow_source = None;
        }
    }

    fn record_top_if_accepting(&mut self) {
        let idx = self.state.lexer_stack.len() - 1;
        let lexer_state = self.state.lexer_stack[idx].lexer_state;
        if matches!(
            self.state.lexer_mut().try_lexeme_end(lexer_state),
            LexerResult::Lexeme(_)
        ) && self.replay.accepting.last() != Some(&idx)
        {
            self.replay.accepting.push(idx);
            self.replay.shadow = None;
            self.replay.shadow_source = None;
        }
    }

    fn finish_active_shadow_speculation(&mut self) {
        self.restore_all_spec();
        self.state
            .scratch
            .grammar_stack
            .truncate(self.state.trie_grammar_stack);
        self.truncate_lexer(self.state.trie_lexer_stack);
        self.state.scratch.definitive = true;
        self.state.rows_valid_end = self.state.num_rows();
        self.state.scratch.log_override = false;
        self.state.lexer_stack_flush_position = 0;
    }

    fn swap_snapshot(&mut self, mut snapshot: Box<Snapshot>) -> Box<Snapshot> {
        let shared = std::mem::take(&mut self.state.shared_box);
        std::mem::swap(self.state, &mut snapshot.state);
        self.state.shared_box = shared;
        std::mem::swap(self.replay, &mut snapshot.replay);
        snapshot
    }

    fn restore_shadow_promotion(&mut self, previous: Box<Snapshot>) {
        self.finish_active_shadow_speculation();
        let shadow = self.swap_snapshot(previous);
        self.replay.shadow = Some(shadow);
    }

    fn restore_spec_undo(&mut self, undo: SpecUndo) {
        if undo.shadow_promotion {
            self.restore_shadow_promotion(undo.previous);
        } else {
            self.restore(*undo.previous);
        }
    }

    fn restore_spec_to(&mut self, target: usize) {
        while self
            .replay
            .spec_undo
            .as_ref()
            .is_some_and(|undo| undo.trigger_lexer_len > target)
        {
            let undo = self.replay.spec_undo.take().unwrap();
            self.restore_spec_undo(undo);
        }
        self.truncate_lexer(target);
    }

    fn restore_all_spec(&mut self) {
        while let Some(undo) = self.replay.spec_undo.take() {
            self.restore_spec_undo(undo);
        }
    }

    fn recover_from_shadow_speculative(
        &mut self,
        byte: Option<u8>,
        flush_end: bool,
    ) -> Option<bool> {
        let floor = self
            .state
            .trie_lexer_stack
            .min(self.state.lexer_stack.len());
        let end = self.state.lexer_stack.len();
        let shadow = self.replay.shadow.take()?;
        let previous = self.swap_snapshot(shadow);
        self.state.trie_started_inner("greedy_shadow");

        let mut ok = true;
        for idx in floor..end {
            let Some(byte) = previous.state.lexer_stack[idx].byte else {
                ok = false;
                break;
            };
            if !self.try_push_speculative(byte) {
                ok = false;
                break;
            }
        }
        if ok {
            if let Some(byte) = byte {
                ok = self.try_push_speculative(byte);
            }
        }
        if ok && flush_end {
            ok = self.flush_speculative();
        }
        if !ok {
            self.restore_shadow_promotion(previous);
            return Some(false);
        }

        let undo = SpecUndo {
            trigger_lexer_len: self.state.lexer_stack.len(),
            previous,
            shadow_promotion: true,
        };
        let mut slot = &mut self.replay.spec_undo;
        while let Some(existing) = slot {
            slot = &mut existing.previous.replay.spec_undo;
        }
        *slot = Some(undo);
        self.state.bias_cache = None;
        Some(true)
    }

    fn recover_from_shadow_definitive(
        &mut self,
        byte: Option<u8>,
        flush_end: bool,
    ) -> Option<(bool, usize)> {
        let source = self.replay.shadow_source;
        let mut shadow = self.replay.shadow.take()?;
        self.replay.shadow_source = None;
        let prefix_len = shadow.state.bytes.len();
        if prefix_len > self.state.bytes.len() {
            self.replay.shadow = Some(shadow);
            self.replay.shadow_source = source;
            return Some((false, 0));
        }
        let replay_bytes = self.state.bytes[prefix_len..].to_vec();
        let parent_mapping = self.state.byte_to_token_idx.clone();
        let previous = self.snapshot();
        let result = with_snapshot(self.state, &mut shadow, |state| {
            let mapped_existing = state.bytes.len().min(parent_mapping.len());
            if state.byte_to_token_idx.len() < mapped_existing {
                state.byte_to_token_idx.extend_from_slice(
                    &parent_mapping[state.byte_to_token_idx.len()..mapped_existing],
                );
            }

            let mut backtrack = 0;
            for (idx, replay_byte) in replay_bytes.iter().copied().enumerate() {
                let (ok, bt) = state.try_push_byte_definitive(Some(replay_byte));
                if !ok || bt > 0 {
                    return (false, bt);
                }
                let absolute_idx = prefix_len + idx;
                if let Some(&token_idx) = parent_mapping.get(absolute_idx) {
                    state.byte_to_token_idx.push(token_idx);
                }
            }
            if let Some(byte) = byte {
                let (ok, bt) = state.try_push_byte_definitive(Some(byte));
                if !ok || bt > 0 {
                    return (false, bt);
                }
                backtrack = bt;
            }
            if flush_end && !state.flush_lexer() {
                return (false, 0);
            }
            (true, backtrack)
        });
        if !result.0 {
            self.replay.shadow = Some(shadow);
            self.replay.shadow_source = source;
            return Some(result);
        }
        shadow.replay.promotion_undo = Some(Box::new(PromotionUndo {
            trigger_byte_len: previous.state.bytes.len() + usize::from(byte.is_some()),
            previous,
        }));
        shadow.replay.shadow_source = None;
        self.restore(*shadow);
        self.state.bias_cache = None;
        Some(result)
    }

    fn recover_speculative(&mut self, byte: Option<u8>, flush_end: bool) -> bool {
        if let Some(result) = self.recover_from_shadow_speculative(byte, flush_end) {
            return result;
        }
        let Some((checkpoint, mut pre, mut bytes)) = self.latest_accepting() else {
            return false;
        };
        bytes.extend(byte);
        let Some((&first, rest)) = bytes.split_first() else {
            return false;
        };

        let previous = Box::new(self.snapshot());
        self.truncate_lexer(checkpoint + 1);
        self.state.lexer_stack_top_eos = false;
        self.state.lexer_stack_flush_position = 0;

        pre.byte = Some(first);
        pre.byte_next_row = true;
        let mut ok = self.state.advance_parser(pre);
        if ok {
            self.record_top_if_accepting();
        }
        for &byte in rest {
            if !ok {
                break;
            }
            ok = self.try_push_speculative(byte);
        }
        if ok && flush_end {
            ok = self.flush_speculative();
        }

        if ok {
            self.replay.spec_undo = Some(SpecUndo {
                trigger_lexer_len: self.state.lexer_stack.len(),
                previous,
                shadow_promotion: false,
            });
            self.state.bias_cache = None;
            true
        } else {
            self.restore(*previous);
            false
        }
    }

    fn try_push_speculative(&mut self, byte: u8) -> bool {
        let current = self.state.lexer_state();
        let result = self
            .state
            .lexer_mut()
            .advance(current.lexer_state, byte, false);
        if result.is_error() {
            self.recover_speculative(Some(byte), false)
        } else {
            let is_state = matches!(result, LexerResult::State(_, _));
            let ok = self.state.advance_lexer_or_parser(result, current);
            if ok && is_state {
                self.record_top_if_accepting();
            }
            ok
        }
    }

    fn flush_speculative(&mut self) -> bool {
        if self.state.flush_lexer_raw() {
            true
        } else {
            self.recover_speculative(None, true)
        }
    }

    fn recover_definitive(&mut self, byte: Option<u8>, flush_end: bool) -> (bool, usize) {
        if let Some(result) = self.recover_from_shadow_definitive(byte, flush_end) {
            return result;
        }
        let Some((checkpoint, mut pre, replay)) = self.latest_accepting() else {
            return (false, 0);
        };
        let existing_len = replay.len();
        let mut bytes = replay;
        bytes.extend(byte);
        let Some((&first, rest)) = bytes.split_first() else {
            return (false, 0);
        };

        let previous = self.snapshot();
        if existing_len > self.state.bytes.len() {
            self.restore(previous);
            return (false, 0);
        }
        let prefix_len = self.state.bytes.len() - existing_len;
        let token_idx = self.state.token_idx;

        self.truncate_lexer(checkpoint + 1);
        self.state.bytes.truncate(prefix_len);
        self.state
            .byte_to_token_idx
            .truncate(prefix_len.min(self.state.byte_to_token_idx.len()));
        self.state.row_infos.truncate(self.state.num_rows());
        self.state.token_idx = token_idx;
        self.state.last_force_bytes_len = usize::MAX;
        self.state.lexer_stack_top_eos = false;
        self.state.lexer_stack_flush_position = 0;
        self.state.rows_valid_end = self.state.num_rows();
        self.state.bias_cache = None;

        pre.byte = Some(first);
        pre.byte_next_row = true;
        let mut ok = self.state.advance_parser(pre);
        if ok {
            self.record_top_if_accepting();
        }
        let mut backtrack = 0;
        if ok {
            self.state.bytes.push(first);
            if existing_len > 0 && prefix_len < previous.state.byte_to_token_idx.len() {
                self.state
                    .byte_to_token_idx
                    .push(previous.state.byte_to_token_idx[prefix_len]);
            }
        }
        for (offset, &byte) in rest.iter().enumerate() {
            if !ok {
                break;
            }
            (ok, backtrack) = self.try_push_definitive(Some(byte));
            let replay_idx = offset + 1;
            if ok && backtrack == 0 && replay_idx < existing_len {
                let idx = prefix_len + replay_idx;
                if idx < previous.state.byte_to_token_idx.len() {
                    self.state
                        .byte_to_token_idx
                        .push(previous.state.byte_to_token_idx[idx]);
                }
            }
        }
        if ok && backtrack == 0 && flush_end {
            ok = self.flush_definitive();
        }

        if ok {
            self.replay.promotion_undo = Some(Box::new(PromotionUndo {
                trigger_byte_len: previous.state.bytes.len() + usize::from(byte.is_some()),
                previous,
            }));
            self.replay.spec_undo = None;
            self.state.bias_cache = None;
            (true, backtrack)
        } else {
            self.restore(previous);
            (false, 0)
        }
    }

    fn try_push_definitive(&mut self, byte: Option<u8>) -> (bool, usize) {
        let current = self.state.lexer_state();
        let result = if let Some(byte) = byte {
            self.state.stats.definitive_bytes += 1;
            self.state
                .lexer_mut()
                .advance(current.lexer_state, byte, true)
        } else {
            self.state.lexer_mut().force_lexeme_end(current.lexer_state)
        };
        let lexer_error = result.is_error();

        let is_state = matches!(result, LexerResult::State(_, _));
        assert_eq!(self.state.backtrack_byte_count, 0);
        if self.state.advance_lexer_or_parser(result, current) {
            if is_state {
                self.record_top_if_accepting();
            }
            if let Some(byte) = byte {
                self.state.bytes.push(byte);
            }
            let backtrack = std::mem::take(&mut self.state.backtrack_byte_count);
            if backtrack > 0 {
                assert!(self.state.lexer_spec().has_stop);
                self.state.last_force_bytes_len = usize::MAX;
                self.state
                    .bytes
                    .truncate(self.state.bytes.len().saturating_sub(backtrack));
            }
            (true, backtrack)
        } else if lexer_error {
            self.recover_definitive(byte, byte.is_none())
        } else {
            (false, 0)
        }
    }

    fn flush_definitive(&mut self) -> bool {
        if self.state.flush_lexer_raw() {
            true
        } else {
            self.recover_definitive(None, true).0
        }
    }

    fn advance_shadow_token(&mut self, tok_bytes: &[u8], tok_id: TokenId) {
        let source = self.replay.shadow_source;
        let Some(mut shadow) = self.replay.shadow.take() else {
            return;
        };
        let result = with_snapshot(self.state, &mut shadow, |state| {
            state.apply_token(tok_bytes, tok_id)
        });
        if matches!(result, Ok(0)) {
            shadow.state.token_idx += 1;
            self.replay.shadow = Some(shadow);
            self.replay.shadow_source = source;
        } else {
            self.replay.shadow_source = source;
        }
    }

    fn advance_shadow_byte(&mut self, byte: u8) {
        let source = self.replay.shadow_source;
        let Some(mut shadow) = self.replay.shadow.take() else {
            return;
        };
        let result = with_snapshot(self.state, &mut shadow, |state| {
            state.try_push_byte_definitive(Some(byte))
        });
        if result == (true, 0) {
            self.replay.shadow = Some(shadow);
            self.replay.shadow_source = source;
        } else {
            self.replay.shadow_source = source;
        }
    }

    fn maybe_materialize(&mut self, token_idx: usize) {
        if self.replay.shadow.is_some() || self.replay.shadow_source.is_some() {
            return;
        }
        let Some(&source) = self.replay.accepting.last() else {
            return;
        };
        if self.state.lexer_stack[source].row_idx != self.state.lexer_state().row_idx
            || self.state.lexer_stack.len().saturating_sub(source + 1) < SHADOW_THRESHOLD
        {
            return;
        }

        let mut shadow = self.snapshot();
        shadow.replay.shadow = None;
        shadow.replay.shadow_source = None;
        shadow.replay.spec_undo = None;
        shadow.replay.promotion_undo = None;
        let ok = with_snapshot(self.state, &mut shadow, |state| {
            recover_definitive(state, None, false).0
        });
        if ok {
            shadow.replay.spec_undo = None;
            shadow.replay.promotion_undo = None;
            shadow.state.token_idx = token_idx;
            self.replay.shadow = Some(Box::new(shadow));
        }
        self.replay.shadow_source = Some(source);
    }

    fn token_committed(&mut self, tok_bytes: &[u8], tok_id: TokenId) {
        self.advance_shadow_token(tok_bytes, tok_id);
        self.maybe_materialize(self.state.token_idx + 1);
    }

    fn forced_byte_committed(&mut self, byte: u8) {
        self.advance_shadow_byte(byte);
        self.maybe_materialize(self.state.token_idx);
    }

    fn prepare_rollback(&mut self, target: usize) {
        while self
            .replay
            .promotion_undo
            .as_ref()
            .is_some_and(|undo| target < undo.trigger_byte_len)
        {
            let undo = self.replay.promotion_undo.take().unwrap();
            self.restore(undo.previous);
        }
        self.replay.spec_undo = None;
    }
}

pub(super) fn has_checkpoint(state: &ParserState) -> bool {
    let Some(replay) = state.greedy_replay.as_deref() else {
        return false;
    };
    let Some(&idx) = replay.accepting.last() else {
        return false;
    };
    idx + 1 < state.lexer_stack.len()
        && state.lexer_stack[idx].row_idx == state.lexer_state().row_idx
}

pub(super) fn record_top_if_accepting(state: &mut ParserState) {
    if state.greedy_replay.is_none() || !state.scratch.definitive {
        return;
    }
    let idx = state.lexer_stack.len() - 1;
    let lexer_state = state.lexer_stack[idx].lexer_state;
    let accepting = matches!(
        state.lexer_mut().try_lexeme_end(lexer_state),
        LexerResult::Lexeme(_)
    );
    if accepting {
        let replay = state.greedy_replay.as_deref_mut().unwrap();
        if replay.accepting.last() != Some(&idx) {
            replay.accepting.push(idx);
            replay.shadow = None;
            replay.shadow_source = None;
        }
    }
}

pub(super) fn truncate_history(state: &mut ParserState, target: usize) {
    state.lexer_stack.truncate(target);
    if let Some(replay) = state.greedy_replay.as_deref_mut() {
        while replay.accepting.last().is_some_and(|idx| *idx >= target) {
            replay.accepting.pop();
        }
        if replay.shadow_source.is_some_and(|idx| idx >= target) {
            replay.shadow = None;
            replay.shadow_source = None;
        }
    }
}

pub(super) fn recover_speculative(
    state: &mut ParserState,
    byte: Option<u8>,
    flush_end: bool,
) -> bool {
    with_replay(state, |context| {
        context.recover_speculative(byte, flush_end)
    })
}

pub(super) fn recover_definitive(
    state: &mut ParserState,
    byte: Option<u8>,
    flush_end: bool,
) -> (bool, usize) {
    with_replay(state, |context| context.recover_definitive(byte, flush_end))
}

pub(super) fn restore_spec_to(state: &mut ParserState, target: usize) {
    with_replay(state, |context| context.restore_spec_to(target));
}

pub(super) fn restore_all_spec(state: &mut ParserState) {
    if state.greedy_replay.is_some() {
        with_replay(state, |context| context.restore_all_spec());
    }
}

pub(super) fn prepare_rollback(state: &mut ParserState, target: usize) {
    with_replay(state, |context| context.prepare_rollback(target));
}

pub(super) fn token_committed(state: &mut ParserState, tok_bytes: &[u8], tok_id: TokenId) {
    if state.greedy_replay.is_some() {
        with_replay(state, |context| context.token_committed(tok_bytes, tok_id));
    }
}

pub(super) fn forced_byte_committed(state: &mut ParserState, byte: u8) {
    if state.greedy_replay.is_some() {
        with_replay(state, |context| context.forced_byte_committed(byte));
    }
}

pub(super) fn discard_shadow(state: &mut ParserState) {
    if let Some(replay) = state.greedy_replay.as_deref_mut() {
        replay.shadow = None;
    }
}

pub(super) fn accepting_allows_eos(state: &mut ParserState) -> bool {
    let Some(replay) = state.greedy_replay.as_deref() else {
        return false;
    };
    let Some(&idx) = replay.accepting.last() else {
        return false;
    };
    let item = state.lexer_stack[idx];
    item.row_idx == state.lexer_state().row_idx && state.lexer_mut().allows_eos(item.lexer_state)
}
