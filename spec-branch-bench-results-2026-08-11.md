# llguidance masked speculative-branch benchmark — 2026-08-11

## Provenance

- Upstream llguidance base: `guidance-ai/llguidance@dbaf504d498b6aeede06ae57adc6f7c2c4848c59` (v1.8.0, `Bump version to 1.8.0`).
- Temporary benchmark commit before this result note: `6c6e14e373f6ecbe1dcf27ae313cbea373fc4559`.
- GitHub Actions run: `31459449637`, successful.
- Runner: Ubuntu 24.04, AMD EPYC 9V74, 4 logical CPUs / 2 physical cores, Rayon threads = 4.
- Rust 1.95.0.
- Synthetic tokenizer vocabulary: 128,000 tokens.

The benchmark models multiple masked autoregressive speculators. The common root mask is computed conceptually once and is not charged once per branch. Each branch first consumes a different root token to diverge, then performs K=8 repetitions of `compute_mask(); consume_token()`.

Modes:

- `seq/shared-lexer`: ordinary llguidance clones, run serially.
- `rayon/shared-lexer`: ordinary clones run in parallel through a persistent Rayon pool. Ordinary clones share llguidance's lexer `Arc<Mutex<...>>`.
- `rayon/deep-independent-lexer`: `deep_clone()` branches, so each branch has an independent lexer, run through the same Rayon pool.

The benchmark also measures ordinary/deep clone time versus prefix-history shape and rollback time.

## Key result: row-changing / cache-miss masked speculation

This grammar deliberately ends a one-byte terminal at each speculative token, changing the Earley row and defeating the state-local same-state whole-mask cache each depth.

| B | mode | median wall for B×8 mask+commit steps | mask p50 | mask p99 | commit p50 | commit p99 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | serial shared | 74.811 us | 8.473 us | 15.713 us | 0.701 us | 1.362 us |
| 2 | serial shared | 149.392 us | 8.452 us | 14.392 us | 0.711 us | 1.442 us |
| 2 | parallel shared | 226.046 us | 11.828 us | 88.070 us | 1.192 us | 61.041 us |
| 2 | parallel independent | 132.187 us | 10.526 us | 18.107 us | 1.191 us | 3.425 us |
| 4 | serial shared | 299.985 us | 8.443 us | 17.456 us | 0.651 us | 1.112 us |
| 4 | parallel shared | 494.454 us | 13.871 us | 264.212 us | 1.442 us | 195.840 us |
| 4 | parallel independent | 187.188 us | 12.218 us | 25.959 us | 1.392 us | 21.281 us |
| 8 | serial shared | 616.235 us | 8.443 us | 18.287 us | 0.681 us | 0.972 us |
| 8 | parallel shared | 972.363 us | 13.680 us | 367.977 us | 1.372 us | 256.421 us |
| 8 | parallel independent | 360.185 us | 12.108 us | 44.616 us | 1.342 us | 35.623 us |

Interpretation: current ordinary sibling clones contend strongly on the shared lexer mutex. At B=8, attempting to parallelize ordinary clones is ~1.58x slower than simply running all eight branches serially, while independent lexers are ~1.71x faster than serial despite the runner having only two physical cores. The enormous shared-clone p99 mask/commit times are largely lock/wait effects, not intrinsic single-state mask computation.

## JSON-string interior

A realistic JSON string interior is highly cache-friendly: each branch has one cold mask around 0.54 ms on this synthetic 128k vocabulary, then most subsequent masks are sub-microsecond whole-mask cache hits.

| B | mode | median wall for B×8 steps | mask p50 | mask p99 |
|---:|---|---:|---:|---:|
| 1 | serial shared | 557.618 us | 0.661 us | 588.484 us |
| 4 | serial shared | 2237.411 us | 0.711 us | 561.283 us |
| 4 | parallel shared | 2345.501 us | 0.901 us | 2293.244 us |
| 4 | parallel independent | 1108.916 us | 1.241 us | 1011.271 us |
| 8 | serial shared | 4477.275 us | 0.691 us | 558.019 us |
| 8 | parallel shared | 4708.789 us | 0.921 us | 2335.367 us |
| 8 | parallel independent | 2177.112 us | 1.232 us | 1017.620 us |

Again the shared lexer prevents useful sibling parallelism; independent lexers roughly halve B=4/B=8 wall time on this two-physical-core runner.

## Cheap long-lexeme case

When each branch stays in one simple greedy lexeme, one mask is around 9–10 us and later cache hits are sub-microsecond. Work is so cheap that scheduling overhead matters.

At B=8, K=8:

- serial shared: 226.696 us wall;
- parallel shared: 453.032 us wall;
- parallel independent: 220.347 us wall.

Thus independent parallelism merely breaks even here on this small runner, whereas shared-lexer parallelism is about 2x slower.

## Clone cost depends strongly on parser-history shape

### Long lexeme (`start: /[a-z]+/`)

| prefix tokens | ordinary/shared clone p50 | deep clone p50 |
|---:|---:|---:|
| 0 | 0.381 us | 4.066 us |
| 64 | 0.440 us | 4.257 us |
| 512 | 0.822 us | 5.047 us |
| 4,096 | 3.014 us | 14.021 us |
| 16,384 | 9.534 us | 43.895 us |
| 65,536 | 42.884 us | 81.762 us |

### Many Earley rows (`start: "a"+`)

| prefix tokens | ordinary/shared clone p50 | deep clone p50 |
|---:|---:|---:|
| 0 | 0.461 us | 4.136 us |
| 64 | 3.315 us | 10.366 us |
| 512 | 23.054 us | 49.343 us |
| 4,096 | 176.853 us | 358.752 us |
| 16,384 | 734.159 us | 1,459.296 us |
| 65,536 | 2,852.584 us | 5,860.779 us |

This demonstrates that current clone cost is not just a small fixed cost. It scales with accumulated parser history and can reach milliseconds on long histories with many Earley rows. Sharing the lexer roughly halves the expensive-history clone cost, but then sibling mask/commit operations contend on that lexer if run concurrently.

## Rollback is extremely cheap

At a 4,096-token prefix:

| rollback tokens | long-lexeme p50 | many-rows p50 | many-rows p99 |
|---:|---:|---:|---:|
| 1 | 0.140 us | 0.130 us | 0.140 us |
| 4 | 0.160 us | 0.170 us | 0.301 us |
| 8 | 0.180 us | 0.250 us | 0.291 us |
| 16 | 0.220 us | 0.471 us | 0.561 us |
| 64 | 0.440 us | 1.692 us | 1.953 us |

So linear speculative decoding using advance-then-rollback is already very favorable in current llguidance. The issue is concurrent sibling states, not rollback.

## Experimental conclusions

1. Current ordinary llguidance sibling clones are a poor representation for concurrent masked speculators because the shared lexer mutex serializes/contends in both mask and commit paths.
2. This is not evidence that llguidance's underlying mask algorithm is inherently slow. In the deliberately cache-missing synthetic case, serial masks were ~8.4 us p50; with independent lexers, sibling branches parallelized substantially.
3. The mutex problem is therefore an implementation/design-of-state-sharing issue that can plausibly be removed by changing how lazy lexer/DFA state is shared.
4. Current cloning itself is a separate real cost: parser-state cloning scales strongly with accumulated Earley history. Persistent/COW parser history would be needed for near-O(1) forks.
5. Rollback is already negligible for normal speculative draft lengths.
6. These synthetic tests do not establish full-corpus latency tails or GPU-overlapped end-to-end speculative throughput. Existing real-corpus MaskBench/CFA results remain relevant for mask tails, and an end-to-end model benchmark would be the next experimental layer.
