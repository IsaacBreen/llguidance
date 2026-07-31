use super::*;

const SHADOW_THRESHOLD: usize = 64;

#[derive(Clone)]
struct Snapshot {
    state: ParserState,
    replay: GreedyReplay,
}

#[derive(Clone)]
struct Shadow {
    source: usize,
    snapshot: Box<Snapshot>,
}

struct Attempt {
    source: Option<usize>,
    previous: Box<Snapshot>,
    prefix: usize,
    existing: usize,
    bytes: Vec<u8>,
}

#[derive(Clone)]
struct Undo {
    trigger: usize,
    previous: Box<Snapshot>,
    shadow_source: Option<usize>,
}

#[derive(Clone, Default)]
pub(super) struct GreedyReplay {
    accepting: Vec<usize>,
    shadow: Option<Shadow>,
    spec_undo: Option<Undo>,
    promotion_undo: Option<Undo>,
}

pub(super) struct Context<'a> {
    state: &'a mut ParserState,
    replay: &'a mut GreedyReplay,
}

pub(super) fn with<T>(state: &mut ParserState, f: impl FnOnce(&mut Context<'_>) -> T) -> T {
    let mut replay = *state.shared_box.greedy_replay.take().unwrap();
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

    fn swap(&mut self, mut snapshot: Box<Snapshot>) -> Box<Snapshot> {
        let shared = std::mem::take(&mut self.state.shared_box);
        std::mem::swap(self.state, &mut snapshot.state);
        self.state.shared_box = shared;
        std::mem::swap(self.replay, &mut snapshot.replay);
        snapshot
    }

    fn checkpoint(&mut self, idx: usize, row: u32) -> Option<(usize, PreLexeme, Vec<u8>)> {
        let item = *self.state.lexer_stack.get(idx)?;
        (item.row_idx == row).then_some(())?;
        let LexerResult::Lexeme(pre) = self.state.lexer_mut().try_lexeme_end(item.lexer_state)
        else {
            return None;
        };
        Some((
            idx,
            pre,
            self.state.lexer_stack[idx + 1..]
                .iter()
                .map(|state| state.byte)
                .collect::<Option<_>>()?,
        ))
    }

    fn latest_accepting(&mut self) -> Option<(usize, PreLexeme, Vec<u8>)> {
        let row = self.state.lexer_state().row_idx;
        if !self.state.scratch.definitive {
            let floor = self
                .state
                .trie_lexer_stack
                .min(self.state.lexer_stack.len());
            for idx in (floor..self.state.lexer_stack.len()).rev() {
                if self.state.lexer_stack[idx].row_idx != row {
                    break;
                }
                if let Some(checkpoint) = self.checkpoint(idx, row) {
                    return Some(checkpoint);
                }
            }
        }
        self.checkpoint(*self.replay.accepting.last()?, row)
    }

    pub(super) fn truncate_lexer(&mut self, target: usize) {
        self.state.lexer_stack.truncate(target);
        let keep = self.replay.accepting.partition_point(|&idx| idx < target);
        self.replay.accepting.truncate(keep);
        let stale_shadow = self
            .replay
            .shadow
            .as_ref()
            .is_some_and(|s| s.source >= target);
        if stale_shadow {
            self.replay.shadow = None;
        }
    }

    pub(super) fn record_top_if_accepting(&mut self) {
        let idx = self.state.lexer_stack.len() - 1;
        let state = self.state.lexer_stack[idx].lexer_state;
        let accepting = matches!(
            self.state.lexer_mut().try_lexeme_end(state),
            LexerResult::Lexeme(_)
        );
        if accepting && self.replay.accepting.last() != Some(&idx) {
            self.replay.accepting.push(idx);
            self.replay.shadow = None;
        }
    }

    fn finish_shadow_speculation(&mut self) {
        self.restore_all_spec();
        self.truncate_lexer(self.state.trie_lexer_stack);
        self.state.trie_finished_inner();
    }

    fn restore_saved(&mut self, previous: Box<Snapshot>, source: Option<usize>, finish: bool) {
        if finish {
            self.finish_shadow_speculation();
        }
        if let Some(source) = source {
            let snapshot = self.swap(previous);
            self.replay.shadow = Some(Shadow { source, snapshot });
        } else {
            self.restore(*previous);
        }
    }

    fn restore_undo(&mut self, undo: Undo) {
        let source = undo.shadow_source;
        self.restore_saved(undo.previous, source, source.is_some());
    }

    pub(super) fn restore_spec_to(&mut self, target: usize) {
        while self
            .replay
            .spec_undo
            .as_ref()
            .is_some_and(|u| u.trigger > target)
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

    fn replay_bytes<const DEFINITIVE: bool>(&mut self, attempt: &Attempt) -> (bool, usize) {
        for (offset, &byte) in attempt.bytes.iter().enumerate() {
            let (ok, backtrack) = self.try_push::<DEFINITIVE>(byte);
            if !ok || backtrack > 0 {
                return (false, backtrack);
            }
            if DEFINITIVE && offset < attempt.existing {
                if let Some(&token) = attempt
                    .previous
                    .state
                    .byte_to_token_idx
                    .get(attempt.prefix + offset)
                {
                    self.state.byte_to_token_idx.push(token);
                }
            }
        }
        (true, 0)
    }

    fn install_undo<const DEFINITIVE: bool>(
        &mut self,
        previous: Box<Snapshot>,
        shadow_source: Option<usize>,
        added_byte: bool,
    ) {
        let undo = Undo {
            trigger: if DEFINITIVE {
                previous.state.bytes.len() + usize::from(added_byte)
            } else {
                self.state.lexer_stack.len()
            },
            previous,
            shadow_source,
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

    fn prepare_shadow<const DEFINITIVE: bool>(&mut self) -> Option<Attempt> {
        let Shadow { source, snapshot } = self.replay.shadow.take()?;
        let previous = self.swap(snapshot);
        if !DEFINITIVE {
            self.state.trie_started_inner("greedy_shadow");
        }
        let (prefix, existing, bytes) = if DEFINITIVE {
            let prefix = self.state.bytes.len();
            if prefix > previous.state.bytes.len() {
                self.restore_saved(previous, Some(source), false);
                return None;
            }
            let bytes = previous.state.bytes[prefix..].to_vec();
            let existing = bytes.len();
            let mapping = &previous.state.byte_to_token_idx;
            let mapped = prefix.min(mapping.len());
            if let Some(missing) = mapping.get(self.state.byte_to_token_idx.len()..mapped) {
                self.state.byte_to_token_idx.extend_from_slice(missing);
            }
            (prefix, existing, bytes)
        } else {
            let floor = previous
                .state
                .trie_lexer_stack
                .min(previous.state.lexer_stack.len());
            let Some(bytes) = previous.state.lexer_stack[floor..]
                .iter()
                .map(|state| state.byte)
                .collect::<Option<_>>()
            else {
                self.restore_saved(previous, Some(source), true);
                return None;
            };
            (0, 0, bytes)
        };
        Some(Attempt {
            source: Some(source),
            previous,
            prefix,
            existing,
            bytes,
        })
    }

    fn prepare_current<const DEFINITIVE: bool>(&mut self, byte: Option<u8>) -> Option<Attempt> {
        let (checkpoint, mut pre, mut bytes) = self.latest_accepting()?;
        let existing = bytes.len();
        bytes.extend(byte);
        let (&first, rest) = bytes.split_first()?;
        let previous = Box::new(self.snapshot());
        let prefix = if DEFINITIVE {
            let prefix = self.state.bytes.len().checked_sub(existing)?;
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
        if !self.state.advance_parser(pre) {
            self.restore(*previous);
            return None;
        }
        self.record_top_if_accepting();
        if DEFINITIVE {
            self.state.bytes.push(first);
            if existing > 0 {
                if let Some(&token) = previous.state.byte_to_token_idx.get(prefix) {
                    self.state.byte_to_token_idx.push(token);
                }
            }
        }
        Some(Attempt {
            source: None,
            previous,
            prefix: prefix + 1,
            existing: existing.saturating_sub(1),
            bytes: rest.to_vec(),
        })
    }

    pub(super) fn recover<const DEFINITIVE: bool>(
        &mut self,
        byte: Option<u8>,
        flush_end: bool,
    ) -> (bool, usize) {
        let Some(attempt) = (if self.replay.shadow.is_some() {
            self.prepare_shadow::<DEFINITIVE>()
        } else {
            self.prepare_current::<DEFINITIVE>(byte)
        }) else {
            return (false, 0);
        };
        let from_shadow = attempt.source.is_some();
        let (mut ok, mut backtrack) = self.replay_bytes::<DEFINITIVE>(&attempt);
        if ok && from_shadow {
            if let Some(byte) = byte {
                (ok, backtrack) = self.try_push::<DEFINITIVE>(byte);
            }
        }
        if ok && backtrack == 0 && flush_end {
            ok = self.flush::<DEFINITIVE>();
        }
        if !ok || backtrack > 0 {
            self.restore_saved(attempt.previous, attempt.source, !DEFINITIVE && from_shadow);
            return (false, backtrack);
        }
        let shadow_source = if DEFINITIVE { None } else { attempt.source };
        self.install_undo::<DEFINITIVE>(attempt.previous, shadow_source, byte.is_some());
        (true, backtrack)
    }

    fn update_shadow(&mut self, f: impl FnOnce(&mut ParserState) -> bool) {
        let Some(mut shadow) = self.replay.shadow.take() else {
            return;
        };
        if with_snapshot(self.state, &mut shadow.snapshot, f) {
            self.replay.shadow = Some(shadow);
        }
    }

    fn maybe_materialize(&mut self, token_idx: usize) {
        if self.replay.shadow.is_some() {
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
        let mut snapshot = self.snapshot();
        snapshot.replay.shadow = None;
        snapshot.replay.spec_undo = None;
        snapshot.replay.promotion_undo = None;
        let ok = with_snapshot(self.state, &mut snapshot, |state| {
            with(state, |context| context.recover::<true>(None, false).0)
        });
        if ok {
            snapshot.replay.spec_undo = None;
            snapshot.replay.promotion_undo = None;
            snapshot.state.token_idx = token_idx;
            self.replay.shadow = Some(Shadow {
                source,
                snapshot: Box::new(snapshot),
            });
        }
    }

    pub(super) fn token_committed(&mut self, bytes: &[u8], token: TokenId) {
        self.update_shadow(|state| {
            let ok = matches!(state.apply_token(bytes, token), Ok(0));
            state.token_idx += usize::from(ok);
            ok
        });
        self.maybe_materialize(self.state.token_idx + 1);
    }

    pub(super) fn forced_byte_committed(&mut self, byte: u8) {
        self.update_shadow(|state| state.try_push_byte_definitive(Some(byte)) == (true, 0));
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
            .is_some_and(|u| target < u.trigger)
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
