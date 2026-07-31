use super::*;

#[derive(Clone)]
struct Undo {
    trigger: usize,
    state: ParserState,
    replay: GreedyReplay,
}

#[derive(Clone, Default)]
pub(super) struct GreedyReplay {
    speculative: Option<Box<Undo>>,
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

    pub(super) fn restore_spec(&mut self, target: Option<usize>) {
        while let Some(undo) = self
            .replay
            .speculative
            .take_if(|undo| target.is_none_or(|target| undo.trigger > target))
        {
            self.restore(undo);
        }
        if let Some(target) = target {
            self.state.lexer_stack.truncate(target);
        }
    }

    fn push<const DEFINITIVE: bool>(&mut self, byte: u8) -> (bool, usize) {
        let result = if DEFINITIVE {
            self.state.try_push_byte_definitive(Some(byte))
        } else {
            let ok = ParserRecognizer::<false> { state: self.state }.try_push_byte(byte);
            (ok, 0)
        };
        if result.0 {
            result
        } else {
            self.recover::<DEFINITIVE>(Some(byte))
        }
    }

    pub(super) fn recover<const DEFINITIVE: bool>(&mut self, byte: Option<u8>) -> (bool, usize) {
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
        let prefix = if DEFINITIVE {
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
            prefix
        } else {
            0
        };
        self.state.lexer_stack.truncate(checkpoint + 1);
        self.state.lexer_stack_top_eos = false;
        self.state.lexer_stack_flush_position = 0;
        pre.byte = Some(first);
        pre.byte_next_row = true;
        let mut ok = self.state.advance_parser(pre);
        let mut backtrack = 0;
        if ok && DEFINITIVE {
            self.state.bytes.push(first);
        }
        for &byte in rest {
            if !ok {
                break;
            }
            (ok, backtrack) = self.push::<DEFINITIVE>(byte);
        }
        if DEFINITIVE && ok && backtrack == 0 {
            let end = (prefix + existing).min(previous.state.byte_to_token_idx.len());
            if prefix < end {
                self.state
                    .byte_to_token_idx
                    .extend_from_slice(&previous.state.byte_to_token_idx[prefix..end]);
            }
        }
        if ok && backtrack == 0 && byte.is_none() {
            ok = self.state.flush_lexer() || self.recover::<DEFINITIVE>(None).0;
        }
        if !ok || backtrack > 0 {
            self.restore(previous);
            return (false, backtrack);
        }
        let mut previous = previous;
        previous.trigger = if DEFINITIVE {
            previous.state.bytes.len() + usize::from(byte.is_some())
        } else {
            self.state.lexer_stack.len()
        };
        if DEFINITIVE {
            self.replay.speculative = None;
        }
        let slot = if DEFINITIVE {
            &mut self.replay.promotion
        } else {
            &mut self.replay.speculative
        };
        *slot = Some(previous);
        (true, backtrack)
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
        self.replay.speculative = None;
    }
}
