use super::*;

#[derive(Clone)]
struct Undo {
    trigger: usize,
    state: ParserState,
    replay: GreedyReplay,
}

#[derive(Clone, Default)]
pub(super) struct GreedyReplay {
    promotion: Option<Box<Undo>>,
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

impl Context<'_> {
    fn snapshot(&mut self) -> Box<Undo> {
        let shared = std::mem::take(&mut self.state.shared_box);
        let state = self.state.clone();
        self.state.shared_box = shared;
        Box::new(Undo {
            trigger: 0,
            state,
            replay: self.replay.clone(),
        })
    }

    fn restore(&mut self, mut saved: Box<Undo>) {
        saved.state.shared_box = std::mem::take(&mut self.state.shared_box);
        *self.state = saved.state;
        *self.replay = saved.replay;
    }

    fn latest_accepting(&mut self) -> Option<(usize, PreLexeme)> {
        let row = self.state.lexer_state().row_idx;
        let mut idx = self.state.lexer_stack.len();
        while idx > 0 {
            idx -= 1;
            let item = self.state.lexer_stack[idx];
            if item.row_idx != row {
                return None;
            }
            if let LexerResult::Lexeme(pre) =
                self.state.lexer_mut().try_lexeme_end(item.lexer_state)
            {
                return Some((idx, pre));
            }
        }
        None
    }

    fn push(&mut self, byte: u8) -> (bool, usize) {
        let result = self.state.try_push_byte_definitive(Some(byte));
        if result.0 {
            result
        } else {
            self.recover(Some(byte), false)
        }
    }

    pub(super) fn recover(&mut self, byte: Option<u8>, flush_end: bool) -> (bool, usize) {
        let Some((checkpoint, mut pre)) = self.latest_accepting() else {
            return (false, 0);
        };
        let Some(mut bytes) = self.state.lexer_stack[checkpoint + 1..]
            .iter()
            .map(|state| state.byte)
            .collect::<Option<Vec<_>>>()
        else {
            return (false, 0);
        };
        let existing = bytes.len();
        bytes.extend(byte);
        let Some((&first, rest)) = bytes.split_first() else {
            return (false, 0);
        };
        let previous = self.snapshot();
        let Some(prefix) = self.state.bytes.len().checked_sub(existing) else {
            return (false, 0);
        };
        self.state.bytes.truncate(prefix);
        self.state
            .byte_to_token_idx
            .truncate(prefix.min(self.state.byte_to_token_idx.len()));
        self.state.row_infos.truncate(self.state.num_rows());
        self.state.last_force_bytes_len = usize::MAX;
        self.state.rows_valid_end = self.state.num_rows();
        self.state.lexer_stack.truncate(checkpoint + 1);
        self.state.lexer_stack_top_eos = false;
        self.state.lexer_stack_flush_position = 0;
        pre.byte = Some(first);
        pre.byte_next_row = true;
        let mut ok = self.state.advance_parser(pre);
        let mut backtrack = 0;
        if ok {
            self.state.bytes.push(first);
        }
        for &byte in rest {
            if !ok {
                break;
            }
            (ok, backtrack) = self.push(byte);
        }
        if ok && backtrack == 0 {
            let end = (prefix + existing).min(previous.state.byte_to_token_idx.len());
            if prefix < end {
                self.state
                    .byte_to_token_idx
                    .extend_from_slice(&previous.state.byte_to_token_idx[prefix..end]);
            }
        }
        if ok && backtrack == 0 && flush_end {
            ok = self.state.flush_lexer() || self.recover(None, true).0;
        }
        if !ok || backtrack > 0 {
            self.restore(previous);
            return (false, backtrack);
        }
        let mut previous = previous;
        previous.trigger = previous.state.bytes.len() + usize::from(byte.is_some());
        self.replay.promotion = Some(previous);
        (true, backtrack)
    }

    fn fork_prefix(&mut self) -> bool {
        let Some((checkpoint, pre)) = self.latest_accepting() else {
            return false;
        };
        if checkpoint + 1 == self.state.lexer_stack.len() {
            self.state.lexer_stack.truncate(checkpoint);
            self.state.advance_parser(pre)
        } else {
            self.recover(None, false).0
        }
    }

    pub(super) fn accepting_allows_eos(&mut self) -> bool {
        self.latest_accepting().is_some_and(|(idx, _)| {
            let state = self.state.lexer_stack[idx].lexer_state;
            self.state.lexer_mut().allows_eos(state)
        })
    }

    pub(super) fn prepare_rollback(&mut self, target: usize) {
        while let Some(undo) = self.replay.promotion.take_if(|undo| target < undo.trigger) {
            self.restore(undo);
        }
    }
}

pub(super) fn fork(state: &ParserState) -> Option<ParserState> {
    let mut fork = state.clone();
    fork.shared_box.greedy_replay = Some(Box::default());
    let ok = with(&mut fork, |replay| replay.fork_prefix());
    if ok {
        fork.shared_box.greedy_replay.as_mut()?.promotion = None;
        Some(fork)
    } else {
        None
    }
}
