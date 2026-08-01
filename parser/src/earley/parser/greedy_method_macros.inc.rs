macro_rules! greedy_accepting_method {
    () => {
        pub fn is_accepting(&mut self) -> bool {
            if self.shared_box.greedy_replay.is_some() {
                let mut accepting = self.run_speculative("is_accepting", |s| s.is_accepting_inner());
                if !accepting {
                    greedy_replay::visit_forks(self, |branch| {
                        if !accepting {
                            accepting = branch
                                .run_speculative("greedy_is_accepting", |s| s.is_accepting_inner());
                        }
                    });
                }
                accepting
            } else {
                self.run_speculative("is_accepting", |s| s.is_accepting_inner())
            }
        }
    };
}

macro_rules! greedy_forced_byte_method {
    () => {
        fn greedy_forced_byte(&mut self) -> Option<u8> {
            if self.is_accepting() {
                return None;
            }
            let mut recognizer = ForkRecognizer::new(self);
            recognizer.trie_started("forced_byte");
            let forced = {
                let mut allowed = (u8::MIN..=u8::MAX).filter(|&byte| recognizer.byte_allowed(byte));
                allowed.next().filter(|_| allowed.next().is_none())
            };
            recognizer.trie_finished();
            forced
        }
    };
}

macro_rules! greedy_recognizer_access_methods {
    () => {
        pub fn with_recognizer<T>(&mut self, f: impl FnOnce(&mut ParserRecognizer) -> T) -> T {
            assert!(
                self.state.shared_box.greedy_replay.is_none(),
                "with_recognizer is unavailable with greedy_lexeme_fallback"
            );
            self.with_shared(|state| f(&mut ParserRecognizer { state }))
        }

        pub(crate) fn chop_tokens(&mut self, trie: &TokTrie, tokens: &[TokenId]) -> (usize, usize) {
            self.with_shared(|state| {
                if state.shared_box.greedy_replay.is_none() {
                    trie.chop_tokens(&mut ParserRecognizer { state }, tokens)
                } else {
                    trie.chop_tokens(&mut ForkRecognizer::new(state), tokens)
                }
            })
        }
    };
}

macro_rules! greedy_state_transfer_methods {
    () => {
        pub fn apply_token(&mut self, tok_bytes: &[u8], tok_id: TokenId) -> Result<usize> {
            self.with_shared(|state| {
                let result = state.apply_token(tok_bytes, tok_id);
                state.token_idx += 1;
                if state.shared_box.greedy_replay.is_some() {
                    greedy_replay::with(state, |replay| {
                        if matches!(result, Ok(0)) {
                            replay.token_committed(tok_bytes, tok_id);
                        } else {
                            replay.discard_frontier();
                        }
                    });
                }
                result
            })
        }

        fn with_shared<T>(&mut self, f: impl FnOnce(&mut ParserState) -> T) -> T {
            let mut shared = self.shared.lock().unwrap();
            let greedy = self.state.shared_box.greedy_replay.is_some();
            std::mem::swap(&mut self.state.shared_box, &mut shared);
            if greedy {
                std::mem::swap(
                    &mut self.state.shared_box.greedy_replay,
                    &mut shared.greedy_replay,
                );
            }
            let r = f(&mut self.state);
            if greedy {
                std::mem::swap(
                    &mut self.state.shared_box.greedy_replay,
                    &mut shared.greedy_replay,
                );
            }
            std::mem::swap(&mut self.state.shared_box, &mut shared);
            assert!(shared.lexer_opt.is_some());
            r
        }
    };
}
