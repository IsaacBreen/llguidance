use llguidance::api::{LLGuidanceOptions, TopLevelGrammar};
use llguidance::toktrie::Recognizer;
use sample_parser::{get_parser_factory, get_tok_env};

mod common_lark_utils;
use common_lark_utils::*;

#[test]
fn test_greedy_checkpoint_recovers_after_unfinished_longer_attempt() {
    lark_str_test_many(
        r#"
            %llguidance {"no_forcing": true, "greedy_lexeme_fallback": true}
            start: STEM SUFFIX
            STEM: "list" | "listen"
            SUFFIX: "ed"
        "#,
        &["listed", "listened"],
        &["listd", "listeed", "FINAL_REJECT:list"],
    );
}

#[test]
fn test_greedy_checkpoint_is_still_global_maximal_munch() {
    lark_str_test_many(
        r#"
            %llguidance {"no_forcing": true, "greedy_lexeme_fallback": true}
            start: A B | AB C
            A: "a"
            B: "b"
            AB: "ab"
            C: "c"
        "#,
        &["abc"],
        &["FINAL_REJECT:ab"],
    );
}

#[test]
fn test_greedy_checkpoint_contextual_ipv6() {
    lark_str_test_many(
        r#"
            %llguidance {"no_forcing": true, "greedy_lexeme_fallback": true}
            start: "ping " ipv6
                 | "connect " (ipv6 | HOST_PORT)
            ipv6: HEX "::" HEX
            HEX: /[0-9a-f]+/
            HOST_PORT: /[a-z0-9]+:[0-9]+/
        "#,
        &["ping fe80::1", "connect fe80::1", "connect server:80"],
        &["connect fe80:xyz"],
    );
}

#[test]
fn test_greedy_checkpoint_fstring_chunk() {
    lark_str_test_many(
        r#"
            %llguidance {"no_forcing": true, "greedy_lexeme_fallback": true}
            start: "f\"" FCHUNK? ("{" NAME "}" FCHUNK?)* "\""
            FCHUNK: /(?:[^{}"\\]|\\.|\{\{|\}\})+/
            NAME: /[A-Za-z_][A-Za-z0-9_]*/
        "#,
        &[r#"f"User {name} done""#],
        &[r#"f"User {name done""#],
    );
}

#[test]
fn test_greedy_checkpoint_at_eos() {
    lark_str_test_many(
        r#"
            %llguidance {"no_forcing": true, "greedy_lexeme_fallback": true}
            start: T B
            T: "a" | "abc"
            B: "b"
        "#,
        &["ab"],
        &["FINAL_REJECT:a"],
    );
}

#[test]
fn test_greedy_shadow_long_checkpoint() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T S
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}d/
    "#
    );
    let input = format!("a{}d", "b".repeat(n));
    lark_str_test(&grammar, true, &input, true);
}

#[test]
fn test_greedy_shadow_nested_checkpoint_chain() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T U V
        T: /a|abd{{{n}}}x/
        U: /b|bd{{{n}}}e/
        V: /d{{{n}}}f/
    "#
    );
    let prefix = format!("ab{}", "d".repeat(n));
    let mut parser = make_parser(&grammar, true).unwrap();
    feed_greedy_text(&mut parser, &prefix);

    let token = get_tok_env().tokenize("f")[0];
    for _ in 0..2 {
        let mask = parser.compute_mask().unwrap();
        assert!(mask.is_allowed(token));
        assert_eq!(parser.final_bytes(), prefix.as_bytes());
    }
    consume(&mut parser, token);
    assert!(parser.is_accepting());
}

#[test]
fn test_greedy_shadow_promotes_at_eos() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T S
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}/
    "#
    );
    let input = format!("a{}", "b".repeat(n));
    lark_str_test(&grammar, true, &input, true);
}

fn feed_greedy_text(parser: &mut llguidance::TokenParser, text: &str) {
    let env = get_tok_env();
    for tok in env.tokenize(text) {
        let mask = parser.compute_mask().unwrap();
        assert!(
            mask.is_allowed(tok),
            "rejected {}",
            env.tok_trie().token_dbg(tok)
        );
        consume(parser, tok);
    }
}

#[test]
fn test_greedy_shadow_capture_after_promotion() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: stem S
        stem[capture]: T
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}d/
    "#
    );
    let input = format!("a{}d", "b".repeat(n));
    let mut parser = make_parser(&grammar, true).unwrap();
    feed_greedy_text(&mut parser, &input);
    assert!(parser.is_accepting());
    assert_eq!(parser.get_capture("stem"), Some(&b"a"[..]));
}

#[test]
fn test_greedy_shadow_rollback_and_rebuild() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T S
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}d/
    "#
    );
    let full = format!("a{}d", "b".repeat(n));
    let prefix = format!("a{}", "b".repeat(72));
    let mut parser = make_parser(&grammar, true).unwrap();
    feed_greedy_text(&mut parser, &prefix);
    let rollback_tokens = 4.min(parser.num_tokens());
    parser.rollback(rollback_tokens).unwrap();
    let consumed = parser.final_bytes().len();
    feed_greedy_text(&mut parser, &full[consumed..]);
    assert!(parser.is_accepting());
}

#[test]
fn test_greedy_shadow_deep_clone_diverges_cleanly() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T (S | X)
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}d/
        X: "x"
    "#
    );
    let prefix = format!("a{}", "b".repeat(72));
    let mut fallback = make_parser(&grammar, true).unwrap();
    feed_greedy_text(&mut fallback, &prefix);
    let mut longest = fallback.deep_clone();
    feed_greedy_text(&mut fallback, &format!("{}d", "b".repeat(8)));
    feed_greedy_text(&mut longest, &format!("{}cx", "b".repeat(8)));
    assert!(fallback.is_accepting());
    assert!(longest.is_accepting());
}

#[test]
fn test_greedy_shadow_shared_lexer_clone_diverges_cleanly() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T (S | X)
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}d/
        X: "x"
    "#
    );
    let prefix = format!("a{}", "b".repeat(72));
    let mut fallback = make_parser(&grammar, true).unwrap();
    feed_greedy_text(&mut fallback, &prefix);
    let mut longest = fallback.clone();
    feed_greedy_text(&mut fallback, &format!("{}d", "b".repeat(8)));
    feed_greedy_text(&mut longest, &format!("{}cx", "b".repeat(8)));
    assert!(fallback.is_accepting());
    assert!(longest.is_accepting());
}

#[test]
fn test_greedy_shadow_validate_tokens_restores_state() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T S
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}d/
    "#
    );
    let prefix = format!("a{}", "b".repeat(72));
    let suffix = format!("{}d", "b".repeat(8));
    let mut parser = make_parser(&grammar, true).unwrap();
    feed_greedy_text(&mut parser, &prefix);
    let tokens = get_tok_env().tokenize(&suffix);
    assert_eq!(parser.validate_tokens_raw(&tokens).unwrap(), tokens.len());
    assert_eq!(parser.final_bytes(), prefix.as_bytes());
    feed_greedy_text(&mut parser, &suffix);
    assert!(parser.is_accepting());
}

#[test]
fn test_greedy_shadow_with_forcing_enabled() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"greedy_lexeme_fallback": true}}
        start: T S
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}d/
    "#
    );
    let input = format!("a{}d", "b".repeat(n));
    lark_str_test(&grammar, true, &input, true);
}

#[test]
fn test_greedy_shadow_with_stop_suffix_and_max_tokens() {
    let n = 80usize;
    let prefix = format!("a{}d", "b".repeat(n));

    let stop_grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T body "!"
        T: /a|ab{{{n}}}c/
        body[stop="!"]: /.*/
    "#
    );
    lark_str_test(&stop_grammar, true, &format!("{prefix}!"), true);

    let suffix_grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T body
        T: /a|ab{{{n}}}c/
        body[suffix="!"]: /.*/
    "#
    );
    lark_str_test(&suffix_grammar, true, &format!("{prefix}!"), true);

    let max_tokens_grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T body
        T: /a|ab{{{n}}}c/
        body[max_tokens=100]: /b{{{n}}}d/
    "#
    );
    lark_str_test(&max_tokens_grammar, true, &prefix, true);
}

#[test]
fn test_greedy_fallback_serialized_grammar_option() {
    let n = 80usize;
    let lark = format!(
        r#"
        %llguidance {{"no_forcing": true}}
        start: T S
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}d/
    "#
    );
    let mut grammar = TopLevelGrammar::from_lark(lark);
    grammar.grammars[0].greedy_lexeme_fallback = true;

    let mut parser = get_parser_factory().create_parser(grammar).unwrap();
    parser.start_without_prompt();
    feed_greedy_text(&mut parser, &format!("a{}d", "b".repeat(n)));
    assert!(parser.is_accepting());
}

#[test]
fn test_greedy_fallback_false_is_not_serialized() {
    let value = serde_json::to_value(LLGuidanceOptions::default()).unwrap();
    assert!(value.get("greedy_lexeme_fallback").is_none());
}

#[test]
fn test_greedy_shadow_rollback_across_promotion_then_take_long_match() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: stem (S | X)
        stem[capture]: T
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}d/
        X: "x"
    "#
    );
    let fallback = format!("a{}d", "b".repeat(n));
    let longest = format!("a{}cx", "b".repeat(n));

    let mut parser = make_parser(&grammar, true).unwrap();
    feed_greedy_text(&mut parser, &fallback);
    assert!(parser.is_accepting());
    assert_eq!(parser.get_capture("stem"), Some(&b"a"[..]));

    parser.rollback(1).unwrap();
    let consumed = parser.final_bytes().len();
    assert!(
        longest.as_bytes().starts_with(parser.final_bytes()),
        "rollback did not return to a common prefix: {:?}",
        String::from_utf8_lossy(parser.final_bytes())
    );
    feed_greedy_text(&mut parser, &longest[consumed..]);

    assert!(parser.is_accepting());
    assert_eq!(
        parser.get_capture("stem"),
        Some(&longest.as_bytes()[..longest.len() - 1])
    );
}

#[test]
fn test_greedy_shadow_rollback_across_eos_promotion() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: T (S | X)
        T: /a|ab{{{n}}}c/
        S: /b{{{n}}}/
        X: "x"
    "#
    );
    let fallback = format!("a{}", "b".repeat(n));
    let longest = format!("a{}cx", "b".repeat(n));

    let mut parser = make_parser(&grammar, true).unwrap();
    feed_greedy_text(&mut parser, &fallback);
    assert!(parser.is_accepting());

    parser.rollback(1).unwrap();
    let consumed = parser.final_bytes().len();
    assert!(longest.as_bytes().starts_with(parser.final_bytes()));
    feed_greedy_text(&mut parser, &longest[consumed..]);
    assert!(parser.is_accepting());
}

#[test]
fn test_greedy_shadow_rollback_across_multiple_promotions() {
    let n = 80usize;
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "greedy_lexeme_fallback": true}}
        start: first (S1 | X) second (S2 | Y)
        first[capture]: T1
        second[capture]: T2
        T1: /a|ab{{{n}}}c/
        S1: /b{{{n}}}d/
        X: "x"
        T2: /e|ef{{{n}}}g/
        S2: /f{{{n}}}h/
        Y: "y"
    "#
    );
    let all_fallback = format!("a{}de{}h", "b".repeat(n), "f".repeat(n));
    let second_long = format!("a{}de{}gy", "b".repeat(n), "f".repeat(n));
    let both_long = format!("a{}cxe{}gy", "b".repeat(n), "f".repeat(n));

    let mut parser = make_parser(&grammar, true).unwrap();
    feed_greedy_text(&mut parser, &all_fallback);
    assert!(parser.is_accepting());

    parser.rollback(1).unwrap();
    let consumed = parser.final_bytes().len();
    assert!(second_long.as_bytes().starts_with(parser.final_bytes()));
    feed_greedy_text(&mut parser, &second_long[consumed..]);
    assert!(parser.is_accepting());
    assert_eq!(
        parser.get_capture("second"),
        Some(&second_long.as_bytes()[n + 2..second_long.len() - 1])
    );

    while !both_long.as_bytes().starts_with(parser.final_bytes()) {
        assert!(parser.num_tokens() > 0);
        parser.rollback(1).unwrap();
    }
    let consumed = parser.final_bytes().len();
    feed_greedy_text(&mut parser, &both_long[consumed..]);
    assert!(parser.is_accepting());
    assert_eq!(
        parser.get_capture("first"),
        Some(&both_long.as_bytes()[..n + 2])
    );
    assert_eq!(
        parser.get_capture("second"),
        Some(&both_long.as_bytes()[n + 3..both_long.len() - 1])
    );
}

#[test]
fn test_low_level_recognizer_hook_covers_greedy_fallback() {
    let mut parser = make_parser(
        r#"
            %llguidance {"no_forcing": true, "greedy_lexeme_fallback": true}
            start: T B
            T: "a" | "abc"
            B: "b"
        "#,
        true,
    )
    .unwrap();
    parser.parser.with_any_recognizer(|recognizer| {
        recognizer.trie_started("greedy_hook");
        assert!(recognizer.try_push_byte(b'a'));
        recognizer.pop_bytes(1);
        recognizer.trie_finished();
    });
}
