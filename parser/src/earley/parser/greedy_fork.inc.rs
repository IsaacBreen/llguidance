struct ForkRecognizer {
    branches: Vec<ParserState>,
    history: Vec<Vec<ParserState>>,
}

impl ForkRecognizer {
    fn new(state: &mut ParserState) -> Self {
        let mut branches = vec![state.clone()];
        branches.extend(greedy_replay::forks(state));
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
            .for_each(|s| s.trie_started_inner(label));
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
