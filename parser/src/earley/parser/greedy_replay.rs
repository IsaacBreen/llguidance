use super::*;

#[derive(Clone)]
struct Snapshot {
    trigger: usize,
    state: ParserState,
    replay: GreedyReplay,
}

#[derive(Clone, Copy)]
enum Commit<'a> {
    Token(&'a [u8], TokenId),
    Byte(u8),
}

impl Commit<'_> {
    fn apply(self, state: &mut ParserState) -> bool {
        match self {
            Self::Token(bytes, token) => {
                let ok = matches!(state.apply_token(bytes, token), Ok(0));
                state.token_idx += usize::from(ok);
                ok
            }
            Self::Byte(byte) => state.try_push_byte_definitive(Some(byte)) == (true, 0),
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct GreedyReplay {
    promotion: Option<Box<Snapshot>>,
    source: Option<usize>,
    checked: usize,
    frontier_tried: bool,
    frontier: Option<Box<Snapshot>>,
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
    fn snapshot_with(&mut self, replay: GreedyReplay) -> Snapshot {
        let shared = std::mem::take(&mut self.state.shared_box);
        let state = self.state.clone();
        self.state.shared_box = shared;
        Snapshot {
            trigger: 0,
            state,
            replay,
        }
    }

    fn snapshot(&mut self) -> Box<Snapshot> {
        Box::new(self.snapshot_with(self.replay.clone()))
    }

    fn restore(&mut self, mut saved: Box<Snapshot>) {
        saved.state.shared_box = std::mem::take(&mut self.state.shared_box);
        *self.state = saved.state;
        *self.replay = saved.replay;
    }

    fn latest_accepting(&mut self) -> Option<(usize, PreLexeme)> {
        self.refresh_source();
        let idx = self.replay.source?;
        let item = self.state.lexer_stack[idx];
        let LexerResult::Lexeme(pre) = self.state.lexer_mut().try_lexeme_end(item.lexer_state)
        else {
            return None;
        };
        Some((idx, pre))
    }

    fn try_recover(&mut self, byte: Option<u8>, flush_end: bool) -> Option<(bool, usize)> {
        let (checkpoint, mut pre) = self.latest_accepting()?;
        let mut bytes = self.state.lexer_stack[checkpoint + 1..]
            .iter()
            .map(|state| state.byte)
            .collect::<Option<Vec<_>>>()?;
        let existing = bytes.len();
        bytes.extend(byte);
        let (&first, rest) = bytes.split_first()?;
        let previous = self.snapshot();
        let prefix = self.state.bytes.len().checked_sub(existing)?;
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
        self.discard_frontier();
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
            let result = self.state.try_push_byte_definitive(Some(byte));
            (ok, backtrack) = if result.0 {
                result
            } else {
                self.recover(Some(byte), false)
            };
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
            return Some((false, backtrack));
        }
        let mut previous = previous;
        previous.trigger = previous.state.bytes.len() + usize::from(byte.is_some());
        self.replay.promotion = Some(previous);
        Some((true, backtrack))
    }

    pub(super) fn recover(&mut self, byte: Option<u8>, flush_end: bool) -> (bool, usize) {
        self.try_recover(byte, flush_end).unwrap_or((false, 0))
    }

    fn fork_prefix(&mut self) -> bool {
        let Some((checkpoint, pre)) = self.latest_accepting() else {
            return false;
        };
        if checkpoint + 1 == self.state.lexer_stack.len() {
            if !self.state.has_pending_lexeme_bytes() || !self.state.advance_parser(pre) {
                return false;
            }
            let next = self.state.lexer_stack.pop().unwrap();
            self.state.lexer_stack[checkpoint] = next;
            true
        } else {
            self.recover(None, false).0
        }
    }

    fn fork_snapshot(&mut self) -> Option<Box<Snapshot>> {
        let mut snapshot = Box::new(self.snapshot_with(GreedyReplay::default()));
        let ok = with_snapshot(self.state, &mut snapshot, |state| {
            with(state, |replay| replay.fork_prefix())
        });
        ok.then(|| {
            snapshot.replay.promotion = None;
            snapshot
        })
    }

    fn refresh_source(&mut self) {
        if self.replay.checked > self.state.lexer_stack.len() {
            self.discard_frontier();
        }
        let previous = self.replay.source;
        let mut source = self.replay.source;
        for idx in self.replay.checked..self.state.lexer_stack.len() {
            let item = self.state.lexer_stack[idx];
            if source.is_some_and(|idx| self.state.lexer_stack[idx].row_idx != item.row_idx) {
                source = None;
            }
            if matches!(
                self.state.lexer_mut().try_lexeme_end(item.lexer_state),
                LexerResult::Lexeme(_)
            ) {
                source = Some(idx);
            }
        }
        self.replay.checked = self.state.lexer_stack.len();
        if source.is_some_and(|idx| {
            self.state.lexer_stack[idx].row_idx != self.state.lexer_state().row_idx
        }) {
            source = None;
        }
        if source != previous {
            self.replay.source = source;
            self.replay.frontier_tried = false;
            self.replay.frontier = None;
        }
    }

    fn refresh_frontier(&mut self) {
        self.refresh_source();
        let pending = self
            .replay
            .source
            .is_some_and(|idx| idx + 1 < self.state.lexer_stack.len());
        if pending && !self.replay.frontier_tried {
            self.replay.frontier_tried = true;
            self.replay.frontier = self.fork_snapshot();
        }
    }

    fn committed(&mut self, commit: Commit<'_>) {
        if let Some(mut frontier) = self.replay.frontier.take() {
            let ok = with_snapshot(self.state, &mut frontier, |state| {
                let ok = commit.apply(state);
                if ok && state.shared_box.greedy_replay.is_some() {
                    with(state, |replay| replay.committed(commit));
                }
                ok
            });
            if ok {
                self.replay.frontier = Some(frontier);
            }
        }
        self.refresh_frontier();
    }

    pub(super) fn token_committed(&mut self, bytes: &[u8], token: TokenId) {
        self.committed(Commit::Token(bytes, token));
    }

    pub(super) fn forced_byte_committed(&mut self, byte: u8) {
        self.committed(Commit::Byte(byte));
    }

    pub(super) fn discard_frontier(&mut self) {
        self.replay.source = None;
        self.replay.checked = 0;
        self.replay.frontier_tried = false;
        self.replay.frontier = None;
    }

    fn visit_forks(&mut self, f: &mut impl FnMut(&mut ParserState)) {
        if let Some(mut frontier) = self.replay.frontier.take() {
            with_snapshot(self.state, &mut frontier, |state| {
                f(state);
                if state.shared_box.greedy_replay.is_some() {
                    with(state, |replay| replay.visit_forks(f));
                }
            });
            self.replay.frontier = Some(frontier);
        } else if !self.replay.frontier_tried {
            if let Some(mut frontier) = self.fork_snapshot() {
                with_snapshot(self.state, &mut frontier, |state| {
                    f(state);
                    if state.shared_box.greedy_replay.is_some() {
                        with(state, |replay| replay.visit_forks(f));
                    }
                });
            }
        }
    }

    fn collect_forks(&mut self, forks: &mut Vec<ParserState>) {
        self.visit_forks(&mut |state| {
            let mut fork = state.clone();
            fork.shared_box.greedy_replay = None;
            forks.push(fork);
        });
    }

    pub(super) fn accepting_allows_eos(&mut self) -> bool {
        self.refresh_source();
        self.replay.source.is_some_and(|idx| {
            let item = self.state.lexer_stack[idx];
            self.state.lexer_mut().allows_eos(item.lexer_state)
        })
    }

    pub(super) fn prepare_rollback(&mut self, target: usize) {
        while let Some(undo) = self.replay.promotion.take_if(|undo| target < undo.trigger) {
            self.restore(undo);
        }
        self.discard_frontier();
    }
}

pub(super) fn forks(state: &mut ParserState) -> Vec<ParserState> {
    let mut forks = Vec::new();
    if state.shared_box.greedy_replay.is_some() {
        with(state, |replay| replay.collect_forks(&mut forks));
    }
    forks
}

pub(super) fn visit_forks(state: &mut ParserState, mut f: impl FnMut(&mut ParserState)) {
    if state.shared_box.greedy_replay.is_some() {
        with(state, |replay| replay.visit_forks(&mut f));
    }
}

pub(super) fn forced_byte(state: &mut ParserState) -> Option<u8> {
    if state.is_accepting() {
        return None;
    }
    let mut recognizer = ForkRecognizer::new(state);
    recognizer.trie_started("forced_byte");
    let mut allowed = (u8::MIN..=u8::MAX).filter(|&byte| recognizer.byte_allowed(byte));
    let forced = allowed.next().filter(|_| allowed.next().is_none());
    recognizer.trie_finished();
    forced
}

pub(super) fn chop_tokens(
    state: &mut ParserState,
    trie: &TokTrie,
    tokens: &[TokenId],
) -> (usize, usize) {
    if state.shared_box.greedy_replay.is_none() {
        trie.chop_tokens(&mut ParserRecognizer { state }, tokens)
    } else {
        trie.chop_tokens(&mut ForkRecognizer::new(state), tokens)
    }
}

pub(super) fn validate_tokens(state: &mut ParserState, tokens: &[TokenId]) -> usize {
    let mut valid = state.validate_tokens(tokens);
    visit_forks(state, |branch| {
        valid = valid.max(branch.validate_tokens(tokens));
    });
    valid
}

struct ForkRecognizer {
    branches: Vec<ParserState>,
    history: Vec<Vec<ParserState>>,
}

impl ForkRecognizer {
    fn new(state: &mut ParserState) -> Self {
        let mut branches = vec![state.clone()];
        branches.extend(forks(state));
        branches
            .iter_mut()
            .for_each(|state| state.shared_box.greedy_replay = None);
        Self {
            branches,
            history: vec![],
        }
    }
}

impl Recognizer for ForkRecognizer {
    fn pop_bytes(&mut self, num: usize) {
        if num != 0 {
            let target = self.history.len() - num;
            self.branches = self.history.split_off(target).remove(0);
        }
    }

    fn collapse(&mut self) {}

    fn trie_started(&mut self, label: &str) {
        self.branches
            .iter_mut()
            .for_each(|state| state.trie_started_inner(label));
    }

    fn trie_finished(&mut self) {
        self.branches
            .iter_mut()
            .for_each(ParserState::trie_finished_inner);
        self.history.clear();
    }

    fn try_push_byte(&mut self, byte: u8) -> bool {
        let previous = self.branches.clone();
        self.branches
            .retain_mut(|state| ParserRecognizer { state }.try_push_byte(byte));
        if self.branches.is_empty() {
            self.branches = previous;
            false
        } else {
            self.history.push(previous);
            true
        }
    }
}
