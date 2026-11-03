# Results

`threshold_sweep.json`, when generated locally, contains synthetic-benchmark
findings from the threshold-based patch-reuse experiment. The matching
`docs/figures/threshold_tradeoff.png` charts refresh fraction and feature error.
The JSON records the input manifest, patch size, thresholds, feature metrics,
latency definitions, and patch-operation proxy. These results describe only
the synthetic sequence manifest used for the run; they are not evidence of
semantic correctness or generalization. Generated outputs are ignored by Git
by default.

`change_detection.json`, when generated locally, compares the approximate
gate's actual patch refresh decisions with pixel-change masks. Detection
confusion counts exclude each episode's first frame; feature approximation and
computation summaries include it. The matching
`docs/figures/change_detection_tradeoff.png` charts transition refresh fraction
and missed-change rate. The result is specific to its synthetic manifest and
does not measure semantic change detection.

`research_benchmark.json` and `research_benchmark.csv`, when generated
locally, contain aggregate and per-transformation results for the configured
training, validation, and held-out test partitions. Detection metrics exclude
episode-initial frames; all-frame computation counts include them. The JSON
also records deterministic episode-bootstrap confidence intervals and the
predeclared threshold policy. Test results are reported separately and are not
used to select a threshold. These are synthetic-benchmark findings only.

`global_change_validation.json`, when generated locally, reports a validation-
only comparison of exact reuse, local thresholds, and local 0.05 with the
optional global fallback. The matching
`docs/figures/global_change_comparison.png` charts recall, transition refresh,
and full-frame refresh frequency. Detector runtime is measured separately.
This evaluation uses only a validation-prefixed manifest and does not load the
held-out test partition.

`pretrained_cache_evaluation.json` records a CPU evaluation of the pinned frozen
DINOv2 independent-frame baseline and exact whole-frame cache. It includes the
four deterministic synthetic validation conditions, exact model and
preprocessing metadata, runtime versions, the seeded policy execution order,
untimed warmup call counts, per-condition cache and feature-equivalence
metrics, latency summaries, cache-overhead and tensor-payload estimates, and
all raw timed trials. `pretrained_cache_evaluation.csv` contains one row per
timed policy/condition/repetition. The matching
`docs/figures/pretrained_cache_latency.png` plots mean latency with sample
standard-deviation whiskers, and `docs/figures/pretrained_cache_hit_rate.png`
plots whole-frame cache hit rates. Reproduce all artifacts with:

```bash
.venv/bin/python scripts/evaluate_pretrained_caching.py \
  --config configs/pretrained_cache_evaluation.json
```

The runner is CPU-only and loads the pinned checkpoint with downloads disabled.
It reads images into memory before timing; measured latency includes
preprocessing, hashing and exact comparison, cache storage/retrieval, and
complete DINOv2 forwards on misses, but excludes disk I/O, model loading,
feature-equivalence audits, serialization, and plotting. Cache-hit rate and
forward-call reductions are reported separately from wall-clock latency. In
the frequent widespread condition, every adjacent transition changes every
pixel, the cache records zero hits, and the measured cache path is slightly
slower. Feature comparison is numerical equivalence evidence, not semantic
accuracy. These synthetic results do not establish natural-video
generalization or partial-token computation savings. Cache memory estimates
cover resident tensor payloads only and exclude Python and container overhead.
Generated inputs and result artifacts are excluded from Git by default.

`pretrained_cache_ablation.json` and `pretrained_cache_ablation.csv` record the
Task 16 validation-only capacity sweep. Reproduce the capacity report and its
figure with the locally cached pinned model:

```bash
.venv/bin/python scripts/ablate_pretrained_cache.py \
  --config configs/pretrained_cache_ablation.json
.venv/bin/python scripts/compare_research_baselines.py
```

The runner uses four deterministic 128×128 synthetic sequences (64–65 frames)
with seeds 516016, 516116, 516216, and 516316; policy order uses seed 516017.
Each sequence is preloaded before timing. Independent inference and exact
whole-frame capacities 1, 2, 4, 8, 16, and 32 receive three warmups and five
timed repetitions in deterministic shuffled order. Timing includes
preprocessing, full input hashing and exact equality verification, complete
model forwards on misses, and cache storage or retrieval. Model loading, disk
I/O, independent feature audits, plotting, and serialization are excluded. The
CSV contains the 140 timed policy passes; JSON preserves the full order
schedule and all summaries.

The local CPU run used frozen `facebook/dinov2-small`, revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`, with the existing pinned resize,
crop, and normalization settings. It ran on macOS 26.6.2 arm64, Python 3.13.7,
PyTorch 2.15.0, Transformers 5.22.1, and Pillow 13.0.0. Each output was a
16×16×384 spatial-token tensor. Every condition-capacity comparison matched
independent representations exactly: maximum and mean absolute difference 0,
cosine similarity 1.0. This is numerical equivalence, not semantic accuracy.

| Validation trace | Frames / unique preprocessed inputs | Cache forwards for capacities 1, 2, 4, 8, 16, 32 | Cache-hit rates in the same order | Median independent latency | Cached latency range | Speedup range | Payload range |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Repeated identical | 64 / 1 | 1, 1, 1, 1, 1, 1 | 98.4% throughout | 1.173 s | 0.083–0.084 s | 13.980–14.055× | 0.949 MiB |
| Recurring nonconsecutive | 65 / 32 | 65, 64, 62, 58, 50, 32 | 0%, 1.5%, 4.6%, 10.8%, 23.1%, 50.8% | 1.162 s | 0.617–1.181 s | 0.985–1.883× | 0.949–30.375 MiB |
| Sparse changes | 64 / 4 | 4 throughout | 93.8% throughout | 1.146 s | 0.136–0.137 s | 8.383–8.426× | 0.949–3.797 MiB |
| Frequent widespread changes | 64 / 64 | 64 throughout | 0% throughout | 1.143 s | 1.161–1.165 s | 0.981–0.984× | 0.949–30.375 MiB |

The recurring sequence shows capacity-dependent reuse: capacity 32 retained all
32 distinct inputs, yielding 33 hits and 32 forwards, while capacity 1 had no
hits. For the sparse blocks, capacity 4 retained every distinct input and
larger capacities added memory without reducing forwards. When every
preprocessed frame was different, no capacity avoided a forward and the cache
path was slower despite consuming tensor memory. The cache reuses complete
frame representations only; it does not compute tokens selectively.

Across 30 timed cache trials, the cache made 3,725 forwards. Their
capacity-matched independent reference represents 7,710 forwards, or 3,985
avoided calls over those paired trials. The experiment measured 1,285 actual
independent control forwards because one independent pass per condition and
repetition is shared across the six capacity comparisons. These are separate
scopes; the aggregate avoided-call count is not a wall-clock speedup. Latencies
in the table compare each cache capacity with its independent median within the
same trace. The payload estimate excludes Python and container structures,
allocator overhead, and in-flight tensors.

`final_baseline_comparison.json` consolidates the validation-only handcrafted
RGB-statistics and pretrained DINOv2 experiments while keeping them in
separate tables. It identifies source reports and configurations, includes
patch-size and global-change studies, labels the Task 14 demo as non-comparable
provenance, and records unavailable hardware metadata where source reports did
not preserve it. The aggregate `research_benchmark` report and CSV are excluded
without opening because repository documentation says they contain
held-out-test results. See the Task 16 section in the root README for complete
comparison tables, global-fallback failure cases, and threats to validity.

## Task 17 final evaluation

`final_evaluation.json` and `final_evaluation.csv` contain the actual frozen
Task 17 run. The handcrafted RGB-statistics policies were evaluated on the
configured synthetic test manifest: 30 episodes, 240 frames, 210 transitions,
128×128 images, 16-pixel patches, and seeds 37900–37929. Earlier experiments
had already inspected this synthetic held-out partition; the new run is not
blind. Policies and settings were frozen from validation before the run.
Independent encoding, exact patch reuse, approximate reuse at 0.01, and the
existing global fallback (local 0.05, changed-pixel fraction 0.50) are all
included.

`pretrained_cache_evaluation` inside the same JSON records local-only CPU
DINOv2 inference on the fixed synthetic validation workload, with four
conditions and four 8-frame episodes per condition. Each policy/episode
received three untimed warmups and five timed passes in deterministic shuffled
order. The model is `facebook/dinov2-small` at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`; spatial-token outputs have shape
16×16×384. The exact whole-frame cache had capacity 8 and reset per episode.
No test-partition images were used by the pretrained experiment.

The full run used 1,000 percentile episode-cluster bootstrap repetitions,
95% intervals, seed 10675315 for handcrafted metrics and 10675316 (condition
offsets recorded in JSON) for paired DINOv2 summaries. Frame dependence is
retained within each resampled episode. Paired comparisons use identical
episode indices. These intervals describe dataset-sampling uncertainty and do
not by themselves establish statistical significance. Five repeated timed
passes are retained separately to describe measurement variability.

| Handcrafted method | Episodes / transitions | F1 (95% bootstrap CI) | Missed-change rate | Feature MAE | All-frame patch-operation fraction | Median latency / 240 frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent encoding | 30 / 210 | 0.356 (0.290–0.426) | 0.000 | 0 | 1.000 | 1.295 s |
| Exact patch reuse | 30 / 210 | 1.000 (1.000–1.000) | 0.000 | 0 | 0.315 | 0.439 s |
| Approximate reuse, 0.01 | 30 / 210 | 0.997 (0.995–0.999) | 0.005 | 0.0000194 | 0.314 | 0.722 s |
| Global fallback | 30 / 210 | 0.982 (0.971–0.989) | 0.035 | 0.000586 | 0.308 | 0.963 s |

F1, precision, recall, missed-change rate, and false-refresh rate classify
transition patches only. The first frame of each episode initializes the
cache and is excluded from classification. Feature MAE and patch-operation
fractions include all frames. The operation fraction counts handcrafted
six-value RGB patch feature computations, not neural work. Independent
encoding predicts every transition patch refreshed; its recall is complete,
but false-refresh rate is 1.0 because unchanged patches are also refreshed.
For unchanged-only transitions, F1 is 0 by the existing undefined-positive
convention.

| Validation condition | Frames per pass | Independent / cached full forwards | Hits / rate | Median latency, independent / cache | Measured speedup | Peak tensor payload | Maximum feature difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identical consecutive | 32 | 32 / 4 | 28 / 87.5% | 0.611 / 0.107 s | 5.689× | 0.95 MiB | 0 |
| Sparse changes | 32 | 32 / 8 | 24 / 75.0% | 0.589 / 0.176 s | 3.337× | 1.90 MiB | 0 |
| Moderate changes | 32 | 32 / 16 | 16 / 50.0% | 0.601 / 0.322 s | 1.869× | 3.80 MiB | 0 |
| Frequent widespread changes | 32 | 32 / 32 | 0 / 0.0% | 0.622 / 0.651 s | 0.956× | 7.59 MiB | 0 |

The paired episode-bootstrap 95% intervals for the ratio of summed per-episode
median latencies were 5.612–5.759× (identical), 3.301–3.344× (sparse),
1.846–1.850× (moderate), and 0.963–0.993× (frequent widespread). The point
speedup column above is the ratio of medians from the five repeated complete
condition passes. The bootstrap ratio uses per-episode medians, so the two
estimands can differ slightly. Repeated timing standard deviations are recorded
separately and are not these episode-sampling intervals.

For all four DINOv2 conditions, mean absolute difference was 0 and cosine
similarity was 1.0; every feature comparison was within the configured
1e-5 maximum-difference and 0.999999 minimum-cosine tolerances. These are
numerical equivalence results, not semantic accuracy. When all preprocessed
inputs change, caching avoids no full model forwards; preprocessing, lookup,
comparison, and memory costs remain. Cache hits are not treated as speedup.

The runtime was macOS 26.6.2, arm64, 12 logical CPUs, Python 3.13.7, PyTorch
2.15.0, Transformers 5.22.1, and Pillow 13.0.0. The processor string reported
`arm`; a more specific chip model was not available to the runtime. JSON
preserves the exact software and preprocessing metadata, model revision,
seeds, episode records, raw timed trials, execution schedules, and source
configuration hashes. The first three DINO conditions use seeds 515015–515018;
frequent widespread uses 515115–515118. Policy-order seeds are 517017 for
handcrafted and 515215 for DINOv2. The CSV is an episode-level table; JSON
contains all raw timing repetitions.

Reproduce the final run and its statistical artifacts using the cached model
checkpoint:

```bash
.venv/bin/python scripts/run_final_evaluation.py --config configs/final_evaluation.json
.venv/bin/python scripts/analyze_final_results.py --input results/final_evaluation.json --config configs/final_evaluation.json
```

`final_statistics.json` contains the overall and transformation-stratified
metrics, 95% episode-bootstrap intervals, paired contrasts against independent
RGB encoding, and separate repeated-timing variability. The figures are
`docs/figures/final_accuracy_comparison.png` and
`docs/figures/final_efficiency_comparison.png`. The latter deliberately
separates handcrafted patch-operation proxies from complete DINOv2 forward
calls and measured CPU latency.

Observed handcrafted-feature failure patterns were concentrated in partial
occlusion and recoloring under the local-threshold global-fallback policy
(29.5% and 33.9% missed-change rates). The global detector measures changed
pixel area, not semantic importance or change magnitude; small-area local
changes may not trigger its full refresh. A broad low-magnitude image change
can trigger one unnecessarily. The synthetic masks label RGB pixel differences
only. These results do not establish semantic accuracy or natural-video
generalization, and the DINOv2 cache reuses complete frame representations
without selective token computation.
