use super::*;

const SHADOW_THRESHOLD: usize = 64;

#[derive(Clone)]
struct Snapshot {
    state: ParserState,
    replay: GreedyReplay,
}

#[derive(Clone)]
struct Undo {
    trigger: usize,
    previous: Box<Snapshot>,
    shadow_promotion: bool,
}

#[derive(Clone, Default)]
pub(super) struct GreedyReplay {
    accepting: Vec<usize>,
    shadow: Option<Box<Snapshot>>,
    shadow_source: Option<usize>,
    spec_undo: Option<Undo>,
    promotion_undo: Option<Undo>,
}

pub(super) struct Context<'a> {
    state: &'a mut ParserState,
    replay: &'a mut GreedyReplay,
}

pub(super) fn with<T>(state: &mut ParserState, f: impl FnOnce(&mut Context<'_>) -> T) -> T {
    let mut replay = *state
        .shared_box
        .greedy_replay
        .take()
        .expect("greedy replay on ordinary parser");
    let result = f(&mut Context {
        state,
        replay: &mut replay,
    });
    state.shared_box.greedy_replay = Some(Box::new(replay));
    result
}

fn with_snapshot<T>(
    owner: &mut ParserState,
    snapshot: &mut Snapshot,
    f: impl FnOnce(&mut ParserState) -> T,
) -> T {
    snapshot.state.shared_box = std::mem::take(&mut owner.shared_box);
    snapshot.state.shared_box.greedy_replay = Some(Box::new(std::mem::take(&mut snapshot.replay)));
    let result = f(&mut snapshot.state);
    snapshot.replay = *snapshot.state.shared_box.greedy_replay.take().unwrap();
    owner.shared_box = std::mem::take(&mut snapshot.state.shared_box);
    result
}

impl Context<'_> {
    fn snapshot(&mut self) -> Snapshot {
        debug_assert!(self.state.shared_box.greedy_replay.is_none());
        let shared = std::mem::take(&mut self.state.shared_box);
        let state = self.state.clone();
        self.state.shared_box = shared;
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

    fn swap_snapshot(&mut self, mut snapshot: Box<Snapshot>) -> Box<Snapshot> {
        let shared = std::mem::take(&mut self.state.shared_box);
        std::mem::swap(self.state, &mut snapshot.state);
        self.state.shared_box = shared;
        std::mem::swap(self.replay, &mut snapshot.replay);
        snapshot
    }

    fn latest_accepting(&mut self) -> Option<(usize, PreLexeme, Vec<u8>)> {
        let top = self.state.lexer_state();
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
                    let bytes = self.state.lexer_stack[idx + 1..]
                        .iter()
                        .map(|state| state.byte)
                        .collect::<Option<_>>()?;
                    return Some((idx, pre, bytes));
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
        let bytes = self.state.lexer_stack[idx + 1..]
            .iter()
            .map(|state| state.byte)
            .collect::<Option<_>>()?;
        Some((idx, pre, bytes))
    }

    pub(super) fn truncate_lexer(&mut self, target: usize) {
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

    pub(super) fn record_top_if_accepting(&mut self) {
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

    fn finish_shadow_speculation(&mut self) {
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

    fn restore_shadow(&mut self, previous: Box<Snapshot>) {
        self.finish_shadow_speculation();
        let shadow = self.swap_snapshot(previous);
        self.replay.shadow = Some(shadow);
    }

    fn restore_undo(&mut self, undo: Undo) {
        if undo.shadow_promotion {
            self.restore_shadow(undo.previous);
        } else {
            self.restore(*undo.previous);
        }
    }

    pub(super) fn restore_spec_to(&mut self, target: usize) {
        while self
            .replay
            .spec_undo
            .as_ref()
            .is_some_and(|undo| undo.trigger > target)
        {
            let undo = self.replay.spec_undo.take().unwrap();
            self.restore_undo(undo);
        }
        self.truncate_lexer(target);
    }

    pub(super) fn restore_all_spec(&mut self) {
        while let Some(undo) = self.replay.spec_undo.take() {
            self.restore_undo(undo);
        }
    }

    fn try_push<const DEFINITIVE: bool>(&mut self, byte: u8) -> (bool, usize) {
        let current = self.state.lexer_state();
        let result = self
            .state
            .lexer_mut()
            .advance(current.lexer_state, byte, DEFINITIVE);
        if result.is_error() {
            return self.recover::<DEFINITIVE>(Some(byte), false);
        }

        let is_state = matches!(result, LexerResult::State(_, _));
        if !self.state.advance_lexer_or_parser(result, current) {
            return (false, 0);
        }
        if is_state {
            self.record_top_if_accepting();
        }
        if !DEFINITIVE {
            return (true, 0);
        }

        self.state.bytes.push(byte);
        let backtrack = std::mem::take(&mut self.state.backtrack_byte_count);
        if backtrack > 0 {
            assert!(self.state.lexer_spec().has_stop);
            self.state.last_force_bytes_len = usize::MAX;
            self.state
                .bytes
                .truncate(self.state.bytes.len().saturating_sub(backtrack));
        }
        (true, backtrack)
    }

    fn flush<const DEFINITIVE: bool>(&mut self) -> bool {
        self.state.flush_lexer_raw() || self.recover::<DEFINITIVE>(None, true).0
    }

    fn replay_bytes<const DEFINITIVE: bool>(
        &mut self,
        bytes: &[u8],
        existing: usize,
        prefix: usize,
        mapping: &[u32],
    ) -> (bool, usize) {
        let mut backtrack = 0;
        for (offset, &byte) in bytes.iter().enumerate() {
            let (ok, bt) = self.try_push::<DEFINITIVE>(byte);
            if !ok || bt > 0 {
                return (false, bt);
            }
            backtrack = bt;
            if DEFINITIVE && offset < existing {
                if let Some(&token_idx) = mapping.get(prefix + offset) {
                    self.state.byte_to_token_idx.push(token_idx);
                }
            }
        }
        (true, backtrack)
    }

    fn install_undo<const DEFINITIVE: bool>(
        &mut self,
        previous: Box<Snapshot>,
        shadow_promotion: bool,
        added_byte: bool,
    ) {
        let trigger = if DEFINITIVE {
            previous.state.bytes.len() + usize::from(added_byte)
        } else {
            self.state.lexer_stack.len()
        };
        let undo = Undo {
            trigger,
            previous,
            shadow_promotion,
        };
        if DEFINITIVE {
            self.replay.promotion_undo = Some(undo);
            self.replay.spec_undo = None;
        } else {
            let mut slot = &mut self.replay.spec_undo;
            while let Some(existing) = slot {
                slot = &mut existing.previous.replay.spec_undo;
            }
            *slot = Some(undo);
        }
        self.state.bias_cache = None;
    }

    fn recover_shadow<const DEFINITIVE: bool>(
        &mut self,
        byte: Option<u8>,
        flush_end: bool,
    ) -> Option<(bool, usize)> {
        let floor = self
            .state
            .trie_lexer_stack
            .min(self.state.lexer_stack.len());
        let end = self.state.lexer_stack.len();
        let source = self.replay.shadow_source;
        let shadow = self.replay.shadow.take()?;
        if DEFINITIVE {
            self.replay.shadow_source = None;
        }
        let previous = self.swap_snapshot(shadow);
        if !DEFINITIVE {
            self.state.trie_started_inner("greedy_shadow");
        }

        let (prefix, existing, bytes, mapping) = if DEFINITIVE {
            let prefix = self.state.bytes.len();
            if prefix > previous.state.bytes.len() {
                let shadow = self.swap_snapshot(previous);
                self.replay.shadow = Some(shadow);
                self.replay.shadow_source = source;
                return Some((false, 0));
            }
            let bytes = previous.state.bytes[prefix..].to_vec();
            let existing = bytes.len();
            let mapping = previous.state.byte_to_token_idx.clone();
            let mapped = prefix.min(mapping.len());
            if self.state.byte_to_token_idx.len() < mapped {
                self.state
                    .byte_to_token_idx
                    .extend_from_slice(&mapping[self.state.byte_to_token_idx.len()..mapped]);
            }
            (prefix, existing, bytes, mapping)
        } else {
            let Some(bytes) = previous.state.lexer_stack[floor..end]
                .iter()
                .map(|state| state.byte)
                .collect::<Option<Vec<_>>>()
            else {
                self.restore_shadow(previous);
                return Some((false, 0));
            };
            (0, 0, bytes, Vec::new())
        };

        let (mut ok, mut backtrack) =
            self.replay_bytes::<DEFINITIVE>(&bytes, existing, prefix, &mapping);
        if ok {
            if let Some(byte) = byte {
                (ok, backtrack) = self.try_push::<DEFINITIVE>(byte);
            }
        }
        if ok && backtrack == 0 && flush_end {
            ok = self.flush::<DEFINITIVE>();
        }
        if !ok || backtrack > 0 {
            if !DEFINITIVE {
                self.finish_shadow_speculation();
            }
            let shadow = self.swap_snapshot(previous);
            self.replay.shadow = Some(shadow);
            self.replay.shadow_source = source;
            return Some((false, backtrack));
        }

        self.install_undo::<DEFINITIVE>(previous, !DEFINITIVE, byte.is_some());
        Some((true, backtrack))
    }

    fn recover_current<const DEFINITIVE: bool>(
        &mut self,
        byte: Option<u8>,
        flush_end: bool,
    ) -> (bool, usize) {
        let Some((checkpoint, mut pre, mut bytes)) = self.latest_accepting() else {
            return (false, 0);
        };
        let existing = bytes.len();
        bytes.extend(byte);
        let Some((&first, rest)) = bytes.split_first() else {
            return (false, 0);
        };

        let previous = Box::new(self.snapshot());
        let prefix = if DEFINITIVE {
            let Some(prefix) = self.state.bytes.len().checked_sub(existing) else {
                self.restore(*previous);
                return (false, 0);
            };
            self.state.bytes.truncate(prefix);
            self.state
                .byte_to_token_idx
                .truncate(prefix.min(self.state.byte_to_token_idx.len()));
            self.state.row_infos.truncate(self.state.num_rows());
            self.state.token_idx = previous.state.token_idx;
            self.state.last_force_bytes_len = usize::MAX;
            self.state.rows_valid_end = self.state.num_rows();
            prefix
        } else {
            0
        };

        self.truncate_lexer(checkpoint + 1);
        self.state.lexer_stack_top_eos = false;
        self.state.lexer_stack_flush_position = 0;
        self.state.bias_cache = None;

        pre.byte = Some(first);
        pre.byte_next_row = true;
        let mut ok = self.state.advance_parser(pre);
        if ok {
            self.record_top_if_accepting();
            if DEFINITIVE {
                self.state.bytes.push(first);
                if existing > 0 {
                    if let Some(&token_idx) = previous.state.byte_to_token_idx.get(prefix) {
                        self.state.byte_to_token_idx.push(token_idx);
                    }
                }
            }
        }

        let (replay_ok, mut backtrack) = if ok {
            self.replay_bytes::<DEFINITIVE>(
                rest,
                existing.saturating_sub(1),
                prefix + 1,
                &previous.state.byte_to_token_idx,
            )
        } else {
            (false, 0)
        };
        ok &= replay_ok;
        if ok && backtrack == 0 && flush_end {
            ok = self.flush::<DEFINITIVE>();
        }

        if ok && backtrack == 0 {
            self.install_undo::<DEFINITIVE>(previous, false, byte.is_some());
            (true, backtrack)
        } else {
            self.restore(*previous);
            backtrack = 0;
            (false, backtrack)
        }
    }

    pub(super) fn recover<const DEFINITIVE: bool>(
        &mut self,
        byte: Option<u8>,
        flush_end: bool,
    ) -> (bool, usize) {
        self.recover_shadow::<DEFINITIVE>(byte, flush_end)
            .unwrap_or_else(|| self.recover_current::<DEFINITIVE>(byte, flush_end))
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
        }
        self.replay.shadow_source = source;
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
        }
        self.replay.shadow_source = source;
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
            with(state, |context| context.recover::<true>(None, false).0)
        });
        if ok {
            shadow.replay.spec_undo = None;
            shadow.replay.promotion_undo = None;
            shadow.state.token_idx = token_idx;
            self.replay.shadow = Some(Box::new(shadow));
        }
        self.replay.shadow_source = Some(source);
    }

    pub(super) fn token_committed(&mut self, tok_bytes: &[u8], tok_id: TokenId) {
        self.advance_shadow_token(tok_bytes, tok_id);
        self.maybe_materialize(self.state.token_idx + 1);
    }

    pub(super) fn forced_byte_committed(&mut self, byte: u8) {
        self.advance_shadow_byte(byte);
        self.maybe_materialize(self.state.token_idx);
    }

    pub(super) fn discard_shadow(&mut self) {
        self.replay.shadow = None;
    }

    pub(super) fn accepting_allows_eos(&mut self) -> bool {
        let Some(&idx) = self.replay.accepting.last() else {
            return false;
        };
        let item = self.state.lexer_stack[idx];
        item.row_idx == self.state.lexer_state().row_idx
            && self.state.lexer_mut().allows_eos(item.lexer_state)
    }

    pub(super) fn prepare_rollback(&mut self, target: usize) {
        while self
            .replay
            .promotion_undo
            .as_ref()
            .is_some_and(|undo| target < undo.trigger)
        {
            let undo = self.replay.promotion_undo.take().unwrap();
            self.restore(*undo.previous);
        }
        self.replay.spec_undo = None;
    }
}

pub(super) fn has_checkpoint(state: &ParserState) -> bool {
    let Some(replay) = state.shared_box.greedy_replay.as_deref() else {
        return false;
    };
    let Some(&idx) = replay.accepting.last() else {
        return false;
    };
    idx + 1 < state.lexer_stack.len()
        && state.lexer_stack[idx].row_idx == state.lexer_state().row_idx
}
