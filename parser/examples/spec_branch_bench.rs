use std::hint::black_box;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use llguidance::{
    api::TopLevelGrammar,
    toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv},
    Matcher, ParserFactory,
};
use rayon::prelude::*;

const BLOG_SCHEMA_JSON: &str = include_str!("../../sample_parser/data/blog.schema.json");

// Branches diverge into different greedy lexemes, then stay inside those lexemes.
const DIVERGENT_GRAMMAR: &str = r#"
start: "a" /a+/ "0"
     | "b" /b+/ "1"
     | "c" /c+/ "2"
     | "d" /d+/ "3"
     | "e" /e+/ "4"
     | "f" /f+/ "5"
     | "g" /g+/ "6"
     | "h" /h+/ "7"
"#;

// Each speculative token completes one lexeme and advances to a new Earley row.
// This prevents the state-local whole-mask cache from turning steps 2..K into
// same-(lexer_state,row_idx) cache hits.
const ROW_CHANGING_GRAMMAR: &str = r#"
start: "A" T0 T0 T0 T0 T0 T0 T0 T0 "!"
     | "B" T1 T1 T1 T1 T1 T1 T1 T1 "!"
     | "C" T2 T2 T2 T2 T2 T2 T2 T2 "!"
     | "D" T3 T3 T3 T3 T3 T3 T3 T3 "!"
     | "E" T4 T4 T4 T4 T4 T4 T4 T4 "!"
     | "F" T5 T5 T5 T5 T5 T5 T5 T5 "!"
     | "G" T6 T6 T6 T6 T6 T6 T6 T6 "!"
     | "H" T7 T7 T7 T7 T7 T7 T7 T7 "!"
T0: /[a-c]/
T1: /[d-f]/
T2: /[g-i]/
T3: /[j-l]/
T4: /[m-o]/
T5: /[p-r]/
T6: /[s-u]/
T7: /[v-x]/
"#;

const LONG_LEXEME_GRAMMAR: &str = r#"start: /[a-z]+/"#;
const MANY_ROWS_GRAMMAR: &str = r#"start: "a"+"#;

struct SyntheticTokEnv {
    trie: TokTrie,
}

impl TokenizerEnv for SyntheticTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie.greedy_tokenize(s)
    }

    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

fn synthetic_tok_env(vocab_size: usize) -> TokEnv {
    let eos_token = (vocab_size - 1) as TokenId;
    let mut tokens = Vec::with_capacity(vocab_size);
    for byte in 0u8..=255 {
        tokens.push(vec![byte]);
    }

    // Spread synthetic long tokens over structurals and the alphabet so branch-specific
    // character classes all have substantial vocabulary subtrees.
    let prefixes: &[u8] = b" \"{[\\abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for i in 0..(vocab_size - tokens.len() - 1) {
        let mut tok = Vec::with_capacity(5);
        tok.push(prefixes[i % prefixes.len()]);
        tok.extend_from_slice(&(i as u32).to_le_bytes());
        tokens.push(tok);
    }
    tokens.push(b"\xFF<|eos|>".to_vec());
    Arc::new(SyntheticTokEnv {
        trie: TokTrie::from(&TokRxInfo::new(vocab_size as u32, eos_token), &tokens),
    })
}

fn blog_grammar() -> TopLevelGrammar {
    let schema: serde_json::Value = serde_json::from_str(BLOG_SCHEMA_JSON).unwrap();
    TopLevelGrammar::from_json_schema(schema)
}

fn make_matcher(tok_env: &TokEnv, grammar: TopLevelGrammar) -> Matcher {
    let mut factory = ParserFactory::new_simple(tok_env).unwrap();
    factory.quiet();
    Matcher::new(factory.create_parser(grammar))
}

fn consume_bytes(m: &mut Matcher, bytes: &[u8]) {
    for &b in bytes {
        m.consume_token(b as TokenId).unwrap();
    }
}

fn percentile_ns(mut xs: Vec<u128>, q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_unstable();
    let idx = ((xs.len() - 1) as f64 * q).round() as usize;
    xs[idx] as f64
}

fn fmt_us(ns: f64) -> f64 {
    ns / 1000.0
}

#[derive(Clone, Copy, Debug)]
enum CloneKind {
    Shared,
    Deep,
}

fn clone_matcher(base: &Matcher, kind: CloneKind) -> Matcher {
    match kind {
        CloneKind::Shared => base.clone(),
        CloneKind::Deep => base.deep_clone(),
    }
}

#[derive(Default)]
struct RunStats {
    wall: Vec<u128>,
    masks: Vec<u128>,
    commits: Vec<u128>,
}

fn branch_work(mut m: Matcher, token: TokenId, steps: usize) -> (Vec<u128>, Vec<u128>) {
    let mut masks = Vec::with_capacity(steps);
    let mut commits = Vec::with_capacity(steps);
    for _ in 0..steps {
        let t0 = Instant::now();
        let mask = m.compute_mask().unwrap();
        masks.push(t0.elapsed().as_nanos());
        assert!(mask.is_allowed(token));

        let t0 = Instant::now();
        m.consume_token(token).unwrap();
        commits.push(t0.elapsed().as_nanos());
    }
    black_box(m);
    (masks, commits)
}

fn prepare_branches(
    base: &Matcher,
    kind: CloneKind,
    branch_tokens: &[(TokenId, TokenId)],
    branches: usize,
) -> Vec<(Matcher, TokenId)> {
    (0..branches)
        .map(|i| {
            let (root_tok, body_tok) = branch_tokens[i % branch_tokens.len()];
            let mut m = clone_matcher(base, kind);
            // Model the common-root mask as already computed once. The first sampled token
            // causes branch divergence; timings begin with the next per-branch mask.
            m.consume_token(root_tok).unwrap();
            (m, body_tok)
        })
        .collect()
}

fn bench_sequential(
    base: &Matcher,
    kind: CloneKind,
    branch_tokens: &[(TokenId, TokenId)],
    branches: usize,
    steps: usize,
    reps: usize,
) -> RunStats {
    let mut out = RunStats::default();
    for _ in 0..reps {
        let work = prepare_branches(base, kind, branch_tokens, branches);
        let t0 = Instant::now();
        for (m, tok) in work {
            let (masks, commits) = branch_work(m, tok, steps);
            out.masks.extend(masks);
            out.commits.extend(commits);
        }
        out.wall.push(t0.elapsed().as_nanos());
    }
    out
}

// Uses Rayon's persistent global pool, matching LLExecutor's basic execution model
// and avoiding fresh OS-thread creation in the timed region.
fn bench_parallel_rayon(
    base: &Matcher,
    kind: CloneKind,
    branch_tokens: &[(TokenId, TokenId)],
    branches: usize,
    steps: usize,
    reps: usize,
) -> RunStats {
    let mut out = RunStats::default();
    // Force pool initialization outside timing.
    (0..branches).into_par_iter().for_each(|_| black_box(()));

    for _ in 0..reps {
        let work = prepare_branches(base, kind, branch_tokens, branches);
        let t0 = Instant::now();
        let results: Vec<_> = work
            .into_par_iter()
            .map(|(m, tok)| branch_work(m, tok, steps))
            .collect();
        out.wall.push(t0.elapsed().as_nanos());
        for (masks, commits) in results {
            out.masks.extend(masks);
            out.commits.extend(commits);
        }
    }
    out
}

fn print_run(label: &str, s: RunStats, branches: usize, steps: usize) {
    let events = branches * steps;
    let wall_med = percentile_ns(s.wall.clone(), 0.50);
    println!(
        "RUN {label:30} B={branches:2} K={steps:2} wall_med_us={:9.3} wall_per_maskcommit_us={:8.3} mask_p50_us={:8.3} mask_p90_us={:8.3} mask_p99_us={:8.3} commit_p50_us={:8.3} commit_p99_us={:8.3}",
        fmt_us(wall_med),
        fmt_us(wall_med) / events as f64,
        fmt_us(percentile_ns(s.masks.clone(), 0.50)),
        fmt_us(percentile_ns(s.masks.clone(), 0.90)),
        fmt_us(percentile_ns(s.masks, 0.99)),
        fmt_us(percentile_ns(s.commits.clone(), 0.50)),
        fmt_us(percentile_ns(s.commits, 0.99)),
    );
}

fn bench_branch_case(
    name: &str,
    base: &Matcher,
    branch_tokens: &[(TokenId, TokenId)],
    steps: usize,
) {
    println!("\n=== BRANCH CASE {name} ===");
    for branches in [1usize, 2, 4, 8] {
        let reps = if branches <= 2 { 100 } else { 70 };
        print_run(
            "seq/shared-lexer",
            bench_sequential(
                base,
                CloneKind::Shared,
                branch_tokens,
                branches,
                steps,
                reps,
            ),
            branches,
            steps,
        );
        print_run(
            "rayon/shared-lexer",
            bench_parallel_rayon(
                base,
                CloneKind::Shared,
                branch_tokens,
                branches,
                steps,
                reps,
            ),
            branches,
            steps,
        );
        print_run(
            "rayon/deep-independent-lexer",
            bench_parallel_rayon(
                base,
                CloneKind::Deep,
                branch_tokens,
                branches,
                steps,
                reps,
            ),
            branches,
            steps,
        );
    }
}

fn bench_clone_cost_case(tok_env: &TokEnv, name: &str, grammar: &str) {
    println!("\n=== CLONE COST {name} ===");
    for prefix_len in [0usize, 64, 512, 4096, 16384, 65536] {
        let mut base = make_matcher(
            tok_env,
            TopLevelGrammar::from_lark(grammar.to_string()),
        );
        consume_bytes(&mut base, &vec![b'a'; prefix_len]);

        for kind in [CloneKind::Shared, CloneKind::Deep] {
            let reps = match prefix_len {
                0..=64 => 2000,
                65..=512 => 1000,
                513..=4096 => 300,
                4097..=16384 => 100,
                _ => 30,
            };
            let mut times = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t0 = Instant::now();
                let c = clone_matcher(&base, kind);
                black_box(c);
                times.push(t0.elapsed().as_nanos());
            }
            println!(
                "CLONE case={name:13} kind={kind:?} prefix_tokens={prefix_len:5} p50_us={:9.3} p99_us={:9.3}",
                fmt_us(percentile_ns(times.clone(), 0.50)),
                fmt_us(percentile_ns(times, 0.99)),
            );
        }
    }
}

fn bench_rollback_case(tok_env: &TokEnv, name: &str, grammar: &str) {
    println!("\n=== ROLLBACK COST {name} ===");
    let mut m = make_matcher(
        tok_env,
        TopLevelGrammar::from_lark(grammar.to_string()),
    );
    consume_bytes(&mut m, &vec![b'a'; 4096]);

    for n in [1usize, 4, 8, 16, 64] {
        let reps = 5000;
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            m.rollback(n).unwrap();
            times.push(t0.elapsed().as_nanos());
            for _ in 0..n {
                m.consume_token(b'a' as TokenId).unwrap();
            }
        }
        println!(
            "ROLLBACK case={name:13} n={n:2} p50_us={:8.3} p99_us={:8.3}",
            fmt_us(percentile_ns(times.clone(), 0.50)),
            fmt_us(percentile_ns(times, 0.99)),
        );
    }
}

fn main() {
    println!("available_parallelism={:?}", std::thread::available_parallelism());
    println!("rayon_threads={}", rayon::current_num_threads());

    // Timer floor for context.
    let mut timer = Vec::with_capacity(20_000);
    for _ in 0..20_000 {
        let t0 = Instant::now();
        timer.push(t0.elapsed().as_nanos());
    }
    println!(
        "timer_p50_ns={:.0} timer_p99_ns={:.0}",
        percentile_ns(timer.clone(), 0.50),
        percentile_ns(timer, 0.99)
    );

    let vocab_size = 128_000;
    println!("building synthetic vocab size={vocab_size}");
    let tok_env = synthetic_tok_env(vocab_size);

    bench_clone_cost_case(&tok_env, "long-lexeme", LONG_LEXEME_GRAMMAR);
    bench_clone_cost_case(&tok_env, "many-rows", MANY_ROWS_GRAMMAR);
    bench_rollback_case(&tok_env, "long-lexeme", LONG_LEXEME_GRAMMAR);
    bench_rollback_case(&tok_env, "many-rows", MANY_ROWS_GRAMMAR);

    let letters = [
        (b'a' as TokenId, b'a' as TokenId),
        (b'b' as TokenId, b'b' as TokenId),
        (b'c' as TokenId, b'c' as TokenId),
        (b'd' as TokenId, b'd' as TokenId),
        (b'e' as TokenId, b'e' as TokenId),
        (b'f' as TokenId, b'f' as TokenId),
        (b'g' as TokenId, b'g' as TokenId),
        (b'h' as TokenId, b'h' as TokenId),
    ];

    // Realistic JSON string interior: branches contain different bytes, but grammar state
    // often collapses to the same string-interior lexer state. After the first cold mask,
    // the current bias cache can make subsequent masks exceptionally cheap.
    let mut json_base = make_matcher(&tok_env, blog_grammar());
    consume_bytes(&mut json_base, b"{\"title\":\"");
    bench_branch_case("json-string-equivalent", &json_base, &letters, 8);

    // Distinct greedy lexer paths, but then each branch remains inside one lexeme.
    let divergent_base = make_matcher(
        &tok_env,
        TopLevelGrammar::from_lark(DIVERGENT_GRAMMAR.to_string()),
    );
    bench_branch_case("divergent-long-lexeme", &divergent_base, &letters, 8);

    // Every body token completes a terminal and changes the Earley row. This is closer to
    // the case where every speculative depth genuinely needs a new uncached mask.
    let row_base = make_matcher(
        &tok_env,
        TopLevelGrammar::from_lark(ROW_CHANGING_GRAMMAR.to_string()),
    );
    let row_tokens = [
        (b'A' as TokenId, b'a' as TokenId),
        (b'B' as TokenId, b'd' as TokenId),
        (b'C' as TokenId, b'g' as TokenId),
        (b'D' as TokenId, b'j' as TokenId),
        (b'E' as TokenId, b'm' as TokenId),
        (b'F' as TokenId, b'p' as TokenId),
        (b'G' as TokenId, b's' as TokenId),
        (b'H' as TokenId, b'v' as TokenId),
    ];
    bench_branch_case("row-changing-mask-miss", &row_base, &row_tokens, 8);

    thread::sleep(Duration::from_millis(20));
}
