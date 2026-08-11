# Real CFA speculative branch benchmark — 2026-08-11

## Provenance / setup

- llguidance upstream base: `guidance-ai/llguidance@dbaf504d498b6aeede06ae57adc6f7c2c4848c59` (v1.8.0).
- Benchmark branch: `bench/spec-real-cfa-20260811`.
- Refined GitHub Actions run: `31461523980`, successful.
- Runner: Ubuntu 24.04, AMD EPYC 9V74, 4 logical CPUs / 2 physical cores, Rayon threads = 4.
- Exact JSONSchemaBench data commit: `ba103c73756198dd9b149ddc7db7867da7a077f6`.
- Problems: `Github_hard---o14404` (slow-ish) and `Github_hard---o62060` (slow/pathological).
- Llama-3.1-8B-Instruct tokenizer semantics matching CFA: 128,256 token IDs, EOT id 128009. Downloaded tokenizer SHA256 `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`.
- Each positive JSONSchemaBench example was replayed five times. The state with the largest median `compute_mask()` was selected for speculative branching.
- The common root mask is measured once. From that state, eight valid sibling paths are constructed. Branch 0 follows the real recorded example. Alternate roots are printable base-vocab tokens <=12 bytes, ordered by token ID; this avoids the prior stress run's 48-character repeated-punctuation roots, but is still not a model-logit-based sampling policy.
- Speculative workload per branch: K=8 repetitions of `compute_mask(); consume_token()` after branch divergence.
- `prep` is clone + root-token consumption; `wall` is the K=8 post-divergence work; `total` includes prep + wall.

## `Github_hard---o14404` — slow-ish real problem

Positive examples have 752 and 722 Llama tokens. The slowest observed real state was example 0, step 36/752, where the actual next token is `Button` (id 1597).

- replay mask p50: **298.631 us**
- replay max-of-five / reported p99: **309.477 us**
- one separately rebuilt root mask: **298.169 us**
- ordinary/shared clone at that state: p50 **2.354 us**, p99 **5.087 us**
- deep/independent-lexer clone: p50 **40.902 us**, p99 **56.866 us**

Eight planned root tokens were the real `Button` plus short valid alternatives `A`, `B`, `D`, `G`, `K`, `P`, `R`.

### K=8 speculative workload

| B | mode | prep p50 | wall p50 | total p50 | mask p50 | mask p99 | commit p99 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | serial/shared | 4.547 us | 100.612 us | 105.410 us | 10.346 us | 44.467 us | 1.322 us |
| 2 | serial/shared | 8.062 us | 170.477 us | 179.270 us | 8.933 us | 32.790 us | 1.202 us |
| 2 | Rayon/shared | 15.864 us | 276.617 us | 290.378 us | 16.976 us | 165.970 us | 81.823 us |
| 2 | Rayon/deep independent | 116.746 us | 207.313 us | 331.070 us | 13.170 us | 44.637 us | 1.913 us |
| 4 | serial/shared | 15.874 us | 316.357 us | 332.592 us | 9.023 us | 42.714 us | 1.573 us |
| 4 | Rayon/shared | 30.787 us | 535.858 us | 565.453 us | 22.655 us | 224.849 us | 127.007 us |
| 4 | Rayon/deep independent | 226.311 us | 279.236 us | 514.736 us | 15.424 us | 61.112 us | 2.704 us |
| 8 | serial/shared | 32.248 us | **571.893 us** | **604.822 us** | 8.853 us | 34.272 us | 1.572 us |
| 8 | Rayon/shared | 49.455 us | **929.643 us** | **978.196 us** | 20.531 us | 216.857 us | 153.802 us |
| 8 | Rayon/deep independent | 456.589 us | **554.767 us** | **1,015.562 us** | 15.133 us | 71.448 us | 2.904 us |

Interpretation: at this moderately hard real state, the post-divergence work is too cheap for parallelism to pay on a two-physical-core runner. Ordinary shared-lexer parallelism is actively harmful because of lock contention. Independent lexers remove the contention tail, but their wall time only roughly matches serial at B=8, while fresh deep-fork preparation makes total latency substantially worse. Persistent independent speculators would avoid most of that prep cost.

## `Github_hard---o62060` — real pathological llguidance problem

Positive examples have 1312 and 1306 Llama tokens. The bad region is inside the real `serviceSummary` string containing `This is an example service.`. The slowest selected state is example 0, step 872/1312, with actual next token `.\",` (id 10684).

The slow region is broad rather than a one-off:

| step | actual token | replay mask p50 |
|---:|---|---:|
| 872 | `.\",` | **9.118 ms** |
| 868 | ` is` | **7.695 ms** |
| 871 | ` service` | **7.322 ms** |
| 870 | ` example` | **7.099 ms** |
| 869 | ` an` | **6.904 ms** |
| 866 | ` \"` | **4.716 ms** |
| 763 | `it` | **4.364 ms** |
| 762 | `Benef` | **4.271 ms** |

At the selected state:

- replay mask p50: **9.118 ms**
- replay max-of-five / reported p99: **10.104 ms**
- one separately rebuilt root mask: **8.898 ms**
- ordinary/shared clone p50: **36.936 us**, p99 **53.251 us**
- deep/independent-lexer clone p50: **1.718 ms**, p99 **1.976 ms**

The refined alternate branch roots were single-byte printable tokens `!`, `\"`, `#`, `$`, `%`, `&`, `'`; branch 0 followed the actual recorded continuation. They are legal short-token alternatives but are not claimed to be likely model drafts.

### K=8 speculative workload

| B | mode | prep p50 | wall p50 | total p50 | mask p50 | mask p90 | mask p99 | commit p99 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | serial/shared | 37.516 us | 123.336 us | 163.126 us | 11.057 us | 19.859 us | 130.748 us | 1.822 us |
| 2 | serial/shared | 133.692 us | 12.038 ms | 12.166 ms | 1.433 ms | 1.484 ms | 2.857 ms | 2.624 us |
| 2 | Rayon/shared | 128.935 us | 12.045 ms | 12.153 ms | 1.444 ms | 1.498 ms | **11.818 ms** | **11.812 ms** |
| 2 | Rayon/deep independent | 4.743 ms | 12.756 ms | 17.616 ms | 1.433 ms | 1.486 ms | 1.637 ms | 2.964 us |
| 4 | serial/shared | 220.833 us | 23.806 ms | 24.026 ms | 1.428 ms | 1.474 ms | 1.524 ms | 2.675 us |
| 4 | Rayon/shared | 224.639 us | 24.056 ms | 24.317 ms | 1.456 ms | **11.837 ms** | **23.638 ms** | **11.882 ms** |
| 4 | Rayon/deep independent | 8.677 ms | 17.390 ms | 25.976 ms | 1.442 ms | 2.088 ms | 3.759 ms | 3.375 us |
| 8 | serial/shared | 405.963 us | **75.143 ms** | **75.565 ms** | 1.455 ms | 1.518 ms | 5.296 ms | 3.004 us |
| 8 | Rayon/shared | 427.865 us | **71.457 ms** | **71.890 ms** | 1.467 ms | **13.249 ms** | **36.170 ms** | **11.902 ms** |
| 8 | Rayon/deep independent | **15.848 ms** | **35.300 ms** | **51.244 ms** | 2.502 ms | 2.591 ms | 2.714 ms | 3.415 us |

Interpretation:

1. The known o62060 llguidance tail is reproduced cleanly on the exact real schema, Llama-3.1 vocabulary and real positive-example prefix: one mask is ~9.1 ms and several consecutive masks are 6–8 ms.
2. Current ordinary clone cost is already nontrivial at this 1312-token state (~37 us), while deep clone is very expensive (~1.7 ms each), demonstrating real history-scaled fork cost.
3. After divergence, the hard alternate branches frequently cost ~1.4 ms per mask. With eight branches, serial post-divergence work is ~75.1 ms.
4. Ordinary shared-lexer parallelism barely improves B=8 wall time (~75.1 -> 71.5 ms) and creates extreme lock tails (mask p99 ~36.2 ms; commit p99 ~11.9 ms).
5. Independent lexers roughly halve persistent-branch wall time at B=8 (~75.1 -> 35.3 ms) even on only two physical cores.
6. Fresh deep forking costs ~15.8 ms for eight branches, but the total is still better than serial here (~51.2 ms vs ~75.6 ms) because the downstream constraint work is so expensive. For persistent speculators, where the deep-fork setup is amortized, the benefit is substantially larger.
7. This does not make the alternate-path mix a model-realistic speculative-decoding distribution; no draft-model logits were used. It does establish the cost behavior of current llguidance on a genuinely large, genuinely slow CFA state under legal short-token sibling divergence.

## Bottom line

The synthetic benchmark was not misleading about the shared lexer mutex, but it understated how consequential real llguidance tail states can be. On a moderate real problem, parallel constraint evaluation is not worth it. On o62060, independent branch-local lexer state materially helps, while current shared-lexer clones produce severe contention tails. Current deep cloning is also expensive enough on long real histories that a persistent/COW fork representation would matter substantially.