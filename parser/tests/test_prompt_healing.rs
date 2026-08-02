use llg_test_utils::{get_parser_factory, get_tok_env};
use llguidance::{api::TopLevelGrammar, TokenParser};

fn parser(grammar: &str, prompt: &str) -> (TokenParser, Vec<u32>) {
    let grammar = TopLevelGrammar::from_lark(grammar.to_string());
    let mut parser = get_parser_factory().create_parser(grammar).unwrap();
    let prompt = get_tok_env().tokenize(prompt);
    let processed = parser.process_prompt(prompt);
    (parser, processed)
}

fn single_token(text: &str) -> u32 {
    let tokens = get_tok_env().tokenize(text);
    assert_eq!(
        tokens.len(),
        1,
        "test tokenizer must contain {text:?} as one token"
    );
    assert_eq!(get_tok_env().tok_trie().token(tokens[0]), text.as_bytes());
    tokens[0]
}

fn assert_allowed(parser: &mut TokenParser, token: u32) {
    let mask = parser.compute_mask().unwrap();
    assert!(
        mask.is_allowed(token),
        "token {} should be allowed",
        get_tok_env().tok_trie().token_dbg(token)
    );
}

const NO_FORCING_ABY: &str = r#"
    %llguidance {"no_forcing": true}
    start: "aby"
"#;

#[test]
fn prompt_healed_token_spanning_prefix_is_allowed_and_commits() {
    let aby = single_token("aby");
    let prompt = get_tok_env().tokenize("a");
    let (mut parser, processed) = parser(NO_FORCING_ABY, "a");
    assert!(processed.len() < prompt.len());
    assert!(parser.dump_state().contains("grm_prefix: \"a\""));

    let parser_bytes = parser.parser.get_bytes().to_vec();
    let forced_bytes = parser.parser.currently_forced_bytes().to_vec();
    assert_allowed(&mut parser, aby);
    assert_eq!(parser.parser.get_bytes(), parser_bytes);
    assert_eq!(parser.parser.currently_forced_bytes(), forced_bytes);
    assert_eq!(parser.validate_tokens_raw(&[aby]).unwrap(), 1);
    assert_eq!(parser.consume_token(aby).unwrap(), 0);
    assert_eq!(parser.parser.get_bytes(), b"aby");
    assert_eq!(parser.final_bytes(), b"by");
    assert!(parser.is_accepting());

    let eos = get_tok_env().tok_trie().eos_token();
    assert_allowed(&mut parser, eos);
    assert_eq!(parser.validate_tokens_raw(&[eos]).unwrap(), 1);
}

#[test]
fn token_is_allowed_without_healed_prefix() {
    let aby = single_token("aby");
    let grammar = TopLevelGrammar::from_lark(NO_FORCING_ABY.to_string());
    let mut parser = get_parser_factory().create_parser(grammar).unwrap();
    parser.start_without_prompt();
    assert_allowed(&mut parser, aby);
}

#[test]
fn shorter_tokens_can_satisfy_healed_prefix_incrementally() {
    let a = single_token("a");
    let ab = single_token("ab");
    let grammar = r#"
        %llguidance {"no_forcing": true}
        start: "abcd"
    "#;

    let (mut parser, processed) = parser(grammar, "abc");
    assert!(processed.is_empty());
    assert_allowed(&mut parser, a);
    assert_allowed(&mut parser, ab);
    assert_eq!(parser.validate_tokens_raw(&[a]).unwrap(), 1);

    parser.consume_token(a).unwrap();
    assert_eq!(parser.parser.get_bytes(), b"a");
    assert!(parser.final_bytes().is_empty());
    let b = single_token("b");
    assert_allowed(&mut parser, b);

    parser.rollback(1).unwrap();
    assert_eq!(parser.parser.get_bytes(), b"");
    assert!(parser.final_bytes().is_empty());
    assert_allowed(&mut parser, ab);
}

#[test]
fn token_beyond_healed_prefix_is_checked_by_parser() {
    let aby = single_token("aby");
    let grammar = r#"
        %llguidance {"no_forcing": true}
        start: "abz"
    "#;
    let (mut parser, _) = parser(grammar, "a");
    let mask = parser.compute_mask().unwrap();
    assert!(!mask.is_allowed(aby));
    assert_eq!(parser.validate_tokens_raw(&[aby]).unwrap(), 0);
}

#[test]
fn incorrect_healed_prefix_is_masked() {
    let x = single_token("x");
    let aby = single_token("aby");
    let (mut parser, _) = parser(NO_FORCING_ABY, "a");
    let mask = parser.compute_mask().unwrap();
    assert!(!mask.is_allowed(x));
    assert!(mask.is_allowed(aby));
    assert_eq!(parser.validate_tokens_raw(&[x]).unwrap(), 0);
}

#[test]
fn token_equal_to_healed_prefix_advances_only_that_prefix() {
    let abc = single_token("abc");
    let d = single_token("d");
    let grammar = r#"
        %llguidance {"no_forcing": true}
        start: "abcd"
    "#;
    let (mut parser, _) = parser(grammar, "abc");
    assert_allowed(&mut parser, abc);
    parser.consume_token(abc).unwrap();
    assert_eq!(parser.parser.get_bytes(), b"abc");
    assert!(parser.final_bytes().is_empty());
    assert_allowed(&mut parser, d);
}

#[test]
fn default_forcing_keeps_healed_and_forced_prefixes_distinct() {
    let aa = single_token("aa");
    let by = single_token("by");
    let grammar = r#"start: "a" ("by" | "cz")"#;
    let (mut parser, processed) = parser(grammar, "a");
    assert!(processed.is_empty());
    assert_eq!(parser.parser.currently_forced_bytes(), b"a");

    assert_allowed(&mut parser, aa);
    assert_eq!(parser.validate_tokens_raw(&[aa]).unwrap(), 1);
    parser.consume_token(aa).unwrap();
    assert_eq!(parser.parser.get_bytes(), b"a");
    assert_allowed(&mut parser, by);
    parser.consume_token(by).unwrap();
    assert!(parser.is_accepting());
}

#[test]
fn validation_handles_multiple_tokens_across_prefix_boundary() {
    let a = single_token("a");
    let by = single_token("by");
    let x = single_token("x");
    let (mut parser, _) = parser(NO_FORCING_ABY, "a");
    assert_eq!(parser.validate_tokens_raw(&[a, by]).unwrap(), 2);
    assert_eq!(parser.validate_tokens_raw(&[a, x]).unwrap(), 1);
}

#[test]
fn validation_agrees_with_backtracking_across_healed_prefix() {
    let space_quote = single_token(" \"");
    let grammar = r#"
        start: gen "foo"
        gen[stop=/"/]: /.*/
    "#;
    let (mut parser, processed) = parser(grammar, "Hello, text: ");
    assert_eq!(
        get_tok_env().tok_trie().decode_raw(&processed),
        b"Hello, text:"
    );

    assert_allowed(&mut parser, space_quote);
    assert_eq!(parser.validate_tokens_raw(&[space_quote]).unwrap(), 1);
    assert_eq!(parser.consume_token(space_quote).unwrap(), 1);
}

#[test]
fn commit_and_rollback_restore_healed_prefix_state() {
    let aby = single_token("aby");
    let (mut parser, _) = parser(NO_FORCING_ABY, "a");
    let before = parser.parser.get_bytes().to_vec();

    parser.consume_token(aby).unwrap();
    assert!(parser.is_accepting());
    parser.rollback(1).unwrap();

    assert_eq!(parser.num_tokens(), 0);
    assert_eq!(parser.parser.get_bytes(), before);
    assert!(parser.final_bytes().is_empty());
    assert!(!parser.is_accepting());
    assert_allowed(&mut parser, aby);
    assert_eq!(parser.validate_tokens_raw(&[aby]).unwrap(), 1);
}

#[test]
fn prompt_processing_still_returns_healed_prompt() {
    let prompt = get_tok_env().tokenize("a");
    let (parser, processed) = parser(NO_FORCING_ABY, "a");
    assert_eq!(prompt.len(), 1);
    assert!(processed.is_empty());
    assert_eq!(parser.num_tokens(), 0);
    assert!(parser.final_bytes().is_empty());
}
