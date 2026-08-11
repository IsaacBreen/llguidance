use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use llguidance::{
    api::TopLevelGrammar,
    toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv},
    Matcher, ParserFactory,
};

const BLOG_SCHEMA_JSON: &str = include_str!("../../sample_parser/data/blog.schema.json");
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
const LONG_GRAMMAR: &str = r#"start: /[a-z]+/"#;

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
    let prefixes: &[u8] = b" \"{[\\etaoin";
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
    branch_tokens: &[TokenId],
    branches: usize,
) -> Vec<(Matcher, TokenId)> {
    (0..branches)
        .map(|i| {
            let tok = branch_tokens[i % branch_tokens.len()];
            let mut m = clone_matcher(base, kind);
            // Model the first speculative token as having already diverged from the common root.
            // The common-root mask is shared by all speculators and should not be charged B times.
            m.consume_token(tok).unwrap();
            (m, tok)
        })
        .collect()
}

fn bench_sequential(
    base: &Matcher,
    kind: CloneKind,
    branch_tokens: &[TokenId],
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

fn bench_parallel(
    base: &Matcher,
    kind: CloneKind,
    branch_tokens: &[TokenId],
    branches: usize,
    steps: usize,
    reps: usize,
) -> RunStats {
    let mut out = RunStats::default();
    for _ in 0..reps {
        let work = prepare_branches(base, kind, branch_tokens, branches);
        let barrier = Arc::new(Barrier::new(branches + 1));
        let (wall, results) = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(branches);
            for (m, tok) in work {
                let barrier = barrier.clone();
                handles.push(scope.spawn(move || {
                    barrier.wait();
                    branch_work(m, tok, steps)
                }));
            }
            let t0 = Instant::now();
            barrier.wait();
            let results: Vec<_> = handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect();
            (t0.elapsed().as_nanos(), results)
        });
        out.wall.push(wall);
        for (masks, commits) in results {
            out.masks.extend(masks);
            out.commits.extend(commits);
        }
    }
    out
}

fn print_run(label: &str, s: RunStats, branches: usize, steps: usize) {
    let events = branches * steps;
    println!(
        "RUN {label:30} B={branches:2} K={steps:2} wall_med_us={:9.3} wall_per_maskcommit_us={:8.3} mask_p50_us={:8.3} mask_p99_us={:8.3} commit_p50_us={:8.3} commit_p99_us={:8.3}",
        fmt_us(percentile_ns(s.wall.clone(), 0.50)),
        fmt_us(percentile_ns(s.wall, 0.50)) / events as f64,
        fmt_us(percentile_ns(s.masks.clone(), 0.50)),
        fmt_us(percentile_ns(s.masks, 0.99)),
        fmt_us(percentile_ns(s.commits.clone(), 0.50)),
        fmt_us(percentile_ns(s.commits, 0.99)),
    );
}

fn bench_branch_case(
    name: &str,
    base: &Matcher,
    branch_tokens: &[TokenId],
    max_branches: usize,
    steps: usize,
) {
    println!("\n=== BRANCH CASE {name} ===");
    for branches in [1usize, 2, 4, 8] {
        if branches > max_branches {
            continue;
        }
        let reps = if branches <= 2 { 80 } else { 50 };
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
            "par/shared-lexer",
            bench_parallel(
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
            "par/deep-independent-lexer",
            bench_parallel(
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

fn bench_clone_cost(tok_env: &TokEnv) {
    println!("\n=== CLONE COST VS PREFIX LENGTH ===");
    for prefix_len in [0usize, 64, 512, 4096, 16384] {
        let mut base = make_matcher(
            tok_env,
            TopLevelGrammar::from_lark(LONG_GRAMMAR.to_string()),
        );
        consume_bytes(&mut base, &vec![b'a'; prefix_len]);

        for kind in [CloneKind::Shared, CloneKind::Deep] {
            let reps = match prefix_len {
                0..=64 => 2000,
                65..=512 => 1000,
                513..=4096 => 300,
                _ => 80,
            };
            let mut times = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t0 = Instant::now();
                let c = clone_matcher(&base, kind);
                black_box(c);
                times.push(t0.elapsed().as_nanos());
            }
            println!(
                "CLONE kind={kind:?} prefix_tokens={prefix_len:5} p50_us={:9.3} p99_us={:9.3}",
                fmt_us(percentile_ns(times.clone(), 0.50)),
                fmt_us(percentile_ns(times, 0.99)),
            );
        }
    }
}

fn bench_rollback(tok_env: &TokEnv) {
    println!("\n=== ROLLBACK COST ===");
    let mut m = make_matcher(
        tok_env,
        TopLevelGrammar::from_lark(LONG_GRAMMAR.to_string()),
    );
    consume_bytes(&mut m, &vec![b'a'; 4096]);

    // Timer floor for context.
    let mut timer = Vec::with_capacity(20_000);
    for _ in 0..20_000 {
        let t0 = Instant::now();
        timer.push(t0.elapsed().as_nanos());
    }
    println!(
        "TIMER p50_ns={:.0} p99_ns={:.0}",
        percentile_ns(timer.clone(), 0.50),
        percentile_ns(timer, 0.99)
    );

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
            "ROLLBACK n={n:2} p50_us={:8.3} p99_us={:8.3}",
            fmt_us(percentile_ns(times.clone(), 0.50)),
            fmt_us(percentile_ns(times, 0.99)),
        );
    }
}

fn main() {
    println!("available_parallelism={:?}", std::thread::available_parallelism());
    println!("pid={}", std::process::id());

    let vocab_size = 128_000;
    println!("building synthetic vocab size={vocab_size}");
    let tok_env = synthetic_tok_env(vocab_size);

    bench_clone_cost(&tok_env);
    bench_rollback(&tok_env);

    // Case 1: realistic JSON string interior. Different speculators emit different letters,
    // but their grammar configuration is usually equivalent. This is favorable to potential
    // cross-branch state/mask deduplication.
    let mut json_base = make_matcher(&tok_env, blog_grammar());
    consume_bytes(&mut json_base, b"{\"title\":\"");
    bench_branch_case(
        "json-string-equivalent",
        &json_base,
        &[b'a' as TokenId, b'b' as TokenId, b'c' as TokenId, b'd' as TokenId,
          b'e' as TokenId, b'f' as TokenId, b'g' as TokenId, b'h' as TokenId],
        8,
        8,
    );

    // Case 2: force the sibling branches into distinct parser/lexer paths.
    let divergent_base = make_matcher(
        &tok_env,
        TopLevelGrammar::from_lark(DIVERGENT_GRAMMAR.to_string()),
    );
    bench_branch_case(
        "parser-divergent",
        &divergent_base,
        &[b'a' as TokenId, b'b' as TokenId, b'c' as TokenId, b'd' as TokenId,
          b'e' as TokenId, b'f' as TokenId, b'g' as TokenId, b'h' as TokenId],
        8,
        8,
    );

    // Keep output visibly separated from cargo messages.
    thread::sleep(Duration::from_millis(20));
}
