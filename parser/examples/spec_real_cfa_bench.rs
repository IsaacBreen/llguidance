use std::cmp::Reverse;
use std::env;
use std::hint::black_box;
use std::fs;
use std::sync::Arc;
use std::time::Instant;

use llguidance::{
    api::TopLevelGrammar,
    toktrie::{SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv},
    Matcher, ParserFactory,
};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use toktrie_hf_tokenizers::ByteTokenizer;

const CFA_VOCAB_SIZE: usize = 128_256;
const CFA_BASE_VOCAB_SIZE: usize = 128_000;
const CFA_EOT_TOKEN_ID: TokenId = 128_009;
const SCAN_REPS: usize = 5;
const SPEC_STEPS: usize = 8;
const SPEC_REPS: usize = 20;

#[derive(Deserialize, Clone)]
struct RealCase {
    name: String,
    schema: Value,
    examples: Vec<String>,
}

struct CfaTokenizerEnv {
    tokenizer: ByteTokenizer,
    trie: TokTrie,
    token_bytes: Vec<Vec<u8>>,
}

impl TokenizerEnv for CfaTokenizerEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie.tokenize_with_greedy_fallback(s, |s| {
            self.tokenizer
                .hf_tokenizer
                .encode(s, false)
                .expect("tokenizer error")
                .get_ids()
                .to_vec()
        })
    }

    fn tokenize_is_canonical(&self) -> bool {
        true
    }
}

fn cfa_special_bytes(token_id: usize) -> Vec<u8> {
    match token_id {
        128_000 => b"<|begin_of_text|>".to_vec(),
        128_001 => b"<|end_of_text|>".to_vec(),
        128_002 => b"<|reserved_special_token_0|>".to_vec(),
        128_003 => b"<|reserved_special_token_1|>".to_vec(),
        128_004 => b"<|finetune_right_pad_id|>".to_vec(),
        128_005 => b"<|reserved_special_token_2|>".to_vec(),
        128_006 => b"<|start_header_id|>".to_vec(),
        128_007 => b"<|end_header_id|>".to_vec(),
        128_008 => b"<|eom_id|>".to_vec(),
        128_009 => b"<|eot_id|>".to_vec(),
        128_010 => b"<|python_tag|>".to_vec(),
        128_011..=128_255 => {
            format!("<|reserved_special_token_{}|>", token_id - 128_008).into_bytes()
        }
        _ => panic!("not a CFA special token id: {token_id}"),
    }
}

fn load_cfa_tok_env(path: &str) -> (TokEnv, Arc<CfaTokenizerEnv>) {
    let tokenizer = ByteTokenizer::from_file(path).expect("load Llama-3.1 tokenizer.json");
    let mut token_bytes = tokenizer.token_bytes();
    assert_eq!(
        token_bytes.len(),
        CFA_VOCAB_SIZE,
        "expected full Llama-3.1-8B-Instruct vocabulary"
    );
    for token_id in CFA_BASE_VOCAB_SIZE..CFA_VOCAB_SIZE {
        token_bytes[token_id] = cfa_special_bytes(token_id);
    }

    let info = TokRxInfo::new(CFA_VOCAB_SIZE as u32, CFA_EOT_TOKEN_ID);
    let trie = TokTrie::from(&info, &token_bytes);
    let concrete = Arc::new(CfaTokenizerEnv {
        tokenizer,
        trie,
        token_bytes,
    });
    let env: TokEnv = concrete.clone();
    (env, concrete)
}

fn make_matcher(tok_env: &TokEnv, schema: &Value) -> Matcher {
    let mut factory = ParserFactory::new_simple(tok_env).unwrap();
    factory.quiet();
    Matcher::new(factory.create_parser(TopLevelGrammar::from_json_schema(schema.clone())))
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

fn token_debug(env: &CfaTokenizerEnv, token: TokenId) -> String {
    let bytes = &env.token_bytes[token as usize];
    let text = String::from_utf8_lossy(bytes);
    format!("id={token} bytes={bytes:?} text={text:?}")
}

#[derive(Clone, Debug)]
struct StepTiming {
    example_idx: usize,
    step_idx: usize,
    token_id: TokenId,
    mask_p50_ns: f64,
    mask_p99_ns: f64,
    commit_p50_ns: f64,
    token_count: usize,
}

fn tokenize_examples(case: &RealCase, tok_env: &TokEnv) -> Vec<Vec<TokenId>> {
    case.examples
        .iter()
        .map(|s| tok_env.tokenize_bytes(s.as_bytes()))
        .collect()
}

fn scan_case(
    case: &RealCase,
    tok_env: &TokEnv,
    concrete_env: &CfaTokenizerEnv,
) -> (Vec<Vec<TokenId>>, Vec<StepTiming>) {
    let tokenized = tokenize_examples(case, tok_env);
    let mut all_steps = Vec::new();

    println!("\n=== REAL CFA REPLAY {} ===", case.name);
    println!(
        "examples={} schema_json_bytes={} scan_reps={}",
        case.examples.len(),
        serde_json::to_vec(&case.schema).unwrap().len(),
        SCAN_REPS
    );

    for (example_idx, tokens) in tokenized.iter().enumerate() {
        println!(
            "EXAMPLE case={} idx={} utf8_bytes={} tokens={}",
            case.name,
            example_idx,
            case.examples[example_idx].as_bytes().len(),
            tokens.len()
        );
        let mut masks: Vec<Vec<u128>> = (0..tokens.len())
            .map(|_| Vec::with_capacity(SCAN_REPS))
            .collect();
        let mut commits: Vec<Vec<u128>> = (0..tokens.len())
            .map(|_| Vec::with_capacity(SCAN_REPS))
            .collect();

        for rep in 0..SCAN_REPS {
            let mut matcher = make_matcher(tok_env, &case.schema);
            for (step_idx, &token) in tokens.iter().enumerate() {
                let t0 = Instant::now();
                let mask = matcher.compute_mask().unwrap_or_else(|e| {
                    panic!(
                        "compute_mask failed case={} example={} step={} rep={}: {e:?}",
                        case.name, example_idx, step_idx, rep
                    )
                });
                masks[step_idx].push(t0.elapsed().as_nanos());
                assert!(
                    mask.is_allowed(token),
                    "real positive token rejected case={} example={} step={} {}",
                    case.name,
                    example_idx,
                    step_idx,
                    token_debug(concrete_env, token)
                );

                let t0 = Instant::now();
                matcher.consume_token(token).unwrap();
                commits[step_idx].push(t0.elapsed().as_nanos());
            }
        }

        let mut example_steps = Vec::with_capacity(tokens.len());
        for (step_idx, &token_id) in tokens.iter().enumerate() {
            let rec = StepTiming {
                example_idx,
                step_idx,
                token_id,
                mask_p50_ns: percentile_ns(masks[step_idx].clone(), 0.50),
                mask_p99_ns: percentile_ns(masks[step_idx].clone(), 0.99),
                commit_p50_ns: percentile_ns(commits[step_idx].clone(), 0.50),
                token_count: tokens.len(),
            };
            example_steps.push(rec.clone());
            all_steps.push(rec);
        }
        example_steps.sort_by(|a, b| {
            b.mask_p50_ns
                .partial_cmp(&a.mask_p50_ns)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for rec in example_steps.iter().take(8) {
            println!(
                "SLOWSTEP case={} example={} step={}/{} mask_p50_us={:.3} mask_p99_us={:.3} commit_p50_us={:.3} {}",
                case.name,
                rec.example_idx,
                rec.step_idx,
                rec.token_count,
                fmt_us(rec.mask_p50_ns),
                fmt_us(rec.mask_p99_ns),
                fmt_us(rec.commit_p50_ns),
                token_debug(concrete_env, rec.token_id),
            );
        }
    }

    all_steps.sort_by(|a, b| {
        b.mask_p50_ns
            .partial_cmp(&a.mask_p50_ns)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("TOP CASE STEPS {}", case.name);
    for rec in all_steps.iter().take(12) {
        println!(
            "TOPSTEP case={} example={} step={}/{} mask_p50_us={:.3} mask_p99_us={:.3} commit_p50_us={:.3} {}",
            case.name,
            rec.example_idx,
            rec.step_idx,
            rec.token_count,
            fmt_us(rec.mask_p50_ns),
            fmt_us(rec.mask_p99_ns),
            fmt_us(rec.commit_p50_ns),
            token_debug(concrete_env, rec.token_id),
        );
    }

    (tokenized, all_steps)
}

fn build_real_state_before_step(
    case: &RealCase,
    tok_env: &TokEnv,
    tokens: &[TokenId],
    step_idx: usize,
) -> Matcher {
    let mut matcher = make_matcher(tok_env, &case.schema);
    for &token in &tokens[..step_idx] {
        let mask = matcher.compute_mask().unwrap();
        assert!(mask.is_allowed(token));
        matcher.consume_token(token).unwrap();
    }
    matcher
}

fn normal_candidate_tokens(mask: &SimpleVob, env: &CfaTokenizerEnv) -> Vec<TokenId> {
    let mut out = Vec::new();
    for token_id in 0..CFA_BASE_VOCAB_SIZE as TokenId {
        if !mask.is_allowed(token_id) {
            continue;
        }
        let bytes = &env.token_bytes[token_id as usize];
        if bytes.is_empty() || bytes.len() > 48 {
            continue;
        }
        let Ok(text) = std::str::from_utf8(bytes) else {
            continue;
        };
        if text
            .bytes()
            .all(|b| b >= 0x20 || matches!(b, b'\n' | b'\r' | b'\t'))
        {
            out.push(token_id);
        }
    }
    out.sort_by_key(|&id| (Reverse(env.token_bytes[id as usize].len()), id));
    out
}

fn rotated_candidates(mut candidates: Vec<TokenId>, salt: usize) -> Vec<TokenId> {
    if !candidates.is_empty() {
        let n = salt % candidates.len();
        candidates.rotate_left(n);
    }
    candidates
}

fn plan_suffix(
    mut state: Matcher,
    env: &CfaTokenizerEnv,
    depth: usize,
    salt: usize,
) -> Option<Vec<TokenId>> {
    if depth == 0 {
        return Some(Vec::new());
    }
    let mask = state.compute_mask().ok()?;
    let candidates = rotated_candidates(normal_candidate_tokens(&mask, env), salt);
    for (idx, token) in candidates.into_iter().take(32).enumerate() {
        let mut child = state.deep_clone();
        if child.consume_token(token).is_err() {
            continue;
        }
        if let Some(rest) = plan_suffix(child, env, depth - 1, salt.wrapping_mul(17) + idx + 1) {
            let mut path = Vec::with_capacity(depth);
            path.push(token);
            path.extend(rest);
            return Some(path);
        }
    }
    None
}

#[derive(Clone)]
struct BranchPath {
    root: TokenId,
    suffix: Vec<TokenId>,
}

fn plan_branches(
    base: &Matcher,
    root_mask: &SimpleVob,
    env: &CfaTokenizerEnv,
    actual_tokens: &[TokenId],
    step_idx: usize,
) -> Vec<BranchPath> {
    let actual_root = actual_tokens[step_idx];
    let mut roots = normal_candidate_tokens(root_mask, env);
    roots.retain(|&t| t != actual_root);
    roots.insert(0, actual_root);

    let mut paths = Vec::new();
    for (root_rank, root) in roots.into_iter().take(96).enumerate() {
        if paths.len() >= 8 {
            break;
        }

        if root_rank == 0 && step_idx + 1 + SPEC_STEPS <= actual_tokens.len() {
            paths.push(BranchPath {
                root,
                suffix: actual_tokens[step_idx + 1..step_idx + 1 + SPEC_STEPS].to_vec(),
            });
            continue;
        }

        let mut planner = base.deep_clone();
        if planner.consume_token(root).is_err() {
            continue;
        }
        if let Some(suffix) = plan_suffix(planner, env, SPEC_STEPS, root_rank + 1) {
            paths.push(BranchPath { root, suffix });
        }
    }
    paths
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

fn bench_clone_cost(base: &Matcher, name: &str) {
    for kind in [CloneKind::Shared, CloneKind::Deep] {
        let reps = 500;
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            black_box(clone_matcher(base, kind));
            times.push(t0.elapsed().as_nanos());
        }
        println!(
            "REALCLONE case={} kind={kind:?} p50_us={:.3} p99_us={:.3}",
            name,
            fmt_us(percentile_ns(times.clone(), 0.50)),
            fmt_us(percentile_ns(times, 0.99)),
        );
    }
}

#[derive(Default)]
struct RunStats {
    prep: Vec<u128>,
    wall: Vec<u128>,
    total: Vec<u128>,
    masks: Vec<u128>,
    commits: Vec<u128>,
}

fn prepare_work(
    base: &Matcher,
    kind: CloneKind,
    paths: &[BranchPath],
    branches: usize,
) -> Vec<(Matcher, Vec<TokenId>)> {
    paths[..branches]
        .iter()
        .map(|path| {
            let mut matcher = clone_matcher(base, kind);
            matcher.consume_token(path.root).unwrap();
            (matcher, path.suffix.clone())
        })
        .collect()
}

fn branch_work(mut matcher: Matcher, suffix: Vec<TokenId>) -> (Vec<u128>, Vec<u128>) {
    let mut masks = Vec::with_capacity(suffix.len());
    let mut commits = Vec::with_capacity(suffix.len());
    for token in suffix {
        let t0 = Instant::now();
        let mask = matcher.compute_mask().unwrap();
        masks.push(t0.elapsed().as_nanos());
        assert!(mask.is_allowed(token));

        let t0 = Instant::now();
        matcher.consume_token(token).unwrap();
        commits.push(t0.elapsed().as_nanos());
    }
    black_box(matcher);
    (masks, commits)
}

fn bench_serial(
    base: &Matcher,
    kind: CloneKind,
    paths: &[BranchPath],
    branches: usize,
    reps: usize,
) -> RunStats {
    let mut out = RunStats::default();
    for _ in 0..reps {
        let t_total = Instant::now();
        let t_prep = Instant::now();
        let work = prepare_work(base, kind, paths, branches);
        out.prep.push(t_prep.elapsed().as_nanos());
        let t_wall = Instant::now();
        for (matcher, suffix) in work {
            let (masks, commits) = branch_work(matcher, suffix);
            out.masks.extend(masks);
            out.commits.extend(commits);
        }
        out.wall.push(t_wall.elapsed().as_nanos());
        out.total.push(t_total.elapsed().as_nanos());
    }
    out
}

fn bench_parallel(
    base: &Matcher,
    kind: CloneKind,
    paths: &[BranchPath],
    branches: usize,
    reps: usize,
) -> RunStats {
    let mut out = RunStats::default();
    (0..branches).into_par_iter().for_each(|_| black_box(()));
    for _ in 0..reps {
        let t_total = Instant::now();
        let t_prep = Instant::now();
        let work = prepare_work(base, kind, paths, branches);
        out.prep.push(t_prep.elapsed().as_nanos());
        let t_wall = Instant::now();
        let results: Vec<_> = work
            .into_par_iter()
            .map(|(matcher, suffix)| branch_work(matcher, suffix))
            .collect();
        out.wall.push(t_wall.elapsed().as_nanos());
        out.total.push(t_total.elapsed().as_nanos());
        for (masks, commits) in results {
            out.masks.extend(masks);
            out.commits.extend(commits);
        }
    }
    out
}

fn print_run(label: &str, stats: RunStats, branches: usize) {
    println!(
        "REALRUN {label:31} B={branches:2} K={SPEC_STEPS:2} prep_p50_us={:9.3} wall_p50_us={:10.3} total_p50_us={:10.3} mask_p50_us={:9.3} mask_p90_us={:9.3} mask_p99_us={:10.3} commit_p50_us={:8.3} commit_p99_us={:9.3}",
        fmt_us(percentile_ns(stats.prep, 0.50)),
        fmt_us(percentile_ns(stats.wall, 0.50)),
        fmt_us(percentile_ns(stats.total, 0.50)),
        fmt_us(percentile_ns(stats.masks.clone(), 0.50)),
        fmt_us(percentile_ns(stats.masks.clone(), 0.90)),
        fmt_us(percentile_ns(stats.masks, 0.99)),
        fmt_us(percentile_ns(stats.commits.clone(), 0.50)),
        fmt_us(percentile_ns(stats.commits, 0.99)),
    );
}

fn benchmark_selected_state(
    case: &RealCase,
    tok_env: &TokEnv,
    concrete_env: &CfaTokenizerEnv,
    tokenized: &[Vec<TokenId>],
    selected: &StepTiming,
) {
    println!(
        "\n=== REAL SPEC STATE {} example={} step={}/{} replay_mask_p50_us={:.3} ===",
        case.name,
        selected.example_idx,
        selected.step_idx,
        selected.token_count,
        fmt_us(selected.mask_p50_ns),
    );

    let tokens = &tokenized[selected.example_idx];
    let mut base = build_real_state_before_step(case, tok_env, tokens, selected.step_idx);
    let t0 = Instant::now();
    let root_mask = base.compute_mask().unwrap();
    let root_once_ns = t0.elapsed().as_nanos();
    assert!(root_mask.is_allowed(tokens[selected.step_idx]));
    println!(
        "REALROOT case={} root_once_us={:.3} allowed_actual=true actual={}",
        case.name,
        fmt_us(root_once_ns as f64),
        token_debug(concrete_env, tokens[selected.step_idx]),
    );

    bench_clone_cost(&base, &case.name);
    let paths = plan_branches(&base, &root_mask, concrete_env, tokens, selected.step_idx);
    println!("REALPATHS case={} planned={}", case.name, paths.len());
    for (idx, path) in paths.iter().enumerate() {
        println!(
            "REALPATH case={} branch={} root={} suffix_ids={:?}",
            case.name,
            idx,
            token_debug(concrete_env, path.root),
            path.suffix,
        );
    }
    assert!(!paths.is_empty(), "failed to plan any speculative branch");

    for branches in [1usize, 2, 4, 8] {
        if branches > paths.len() {
            continue;
        }
        print_run(
            "serial/shared-lexer",
            bench_serial(&base, CloneKind::Shared, &paths, branches, SPEC_REPS),
            branches,
        );
        print_run(
            "rayon/shared-lexer",
            bench_parallel(&base, CloneKind::Shared, &paths, branches, SPEC_REPS),
            branches,
        );
        print_run(
            "rayon/deep-independent-lexer",
            bench_parallel(&base, CloneKind::Deep, &paths, branches, SPEC_REPS),
            branches,
        );
    }
}

fn main() {
    println!("available_parallelism={:?}", std::thread::available_parallelism());
    println!("rayon_threads={}", rayon::current_num_threads());

    let cases_path = env::var("REAL_CFA_CASES").expect("REAL_CFA_CASES");
    let tokenizer_path = env::var("REAL_CFA_TOKENIZER").expect("REAL_CFA_TOKENIZER");
    let cases: Vec<RealCase> = serde_json::from_slice(&fs::read(cases_path).unwrap()).unwrap();
    let (tok_env, concrete_env) = load_cfa_tok_env(&tokenizer_path);
    println!(
        "CFA_TOKENIZER vocab={} eos={} tokenizer={}",
        concrete_env.token_bytes.len(),
        CFA_EOT_TOKEN_ID,
        tokenizer_path,
    );

    for case in &cases {
        let (tokenized, steps) = scan_case(case, &tok_env, &concrete_env);
        let selected = steps.first().expect("nonempty positive example").clone();
        benchmark_selected_state(case, &tok_env, &concrete_env, &tokenized, &selected);
    }
}
