# DeltaVision: Stateful Visual Encoding Under Change

**Technical research report · version 0.1.0 · synthetic benchmark study**

## Abstract

This report studies when cached visual state can avoid repeated feature
computation while preserving representation correctness. It compares a
handcrafted six-statistic RGB patch encoder with exact patch reuse,
threshold-based approximate reuse, and a local/global refresh policy. A
separate experiment compares frozen DINOv2-small inference with exact
whole-frame representation caching. On a previously inspected synthetic
held-out partition, exact RGB patch reuse had transition F1 1.000 (95% episode
bootstrap interval 1.000–1.000), zero feature error, and 31.5% of the
independent patch-operation count. Approximate reuse at 0.01 had F1 0.997,
0.5% missed changes, and nonzero feature error; the global fallback had F1
0.982 and 3.5% missed changes. On four-episode-per-condition DINOv2 validation
workloads, exact whole-frame caching reduced complete model forwards by
50–87.5% and measured 1.869–5.689× ratios when inputs recurred. It avoided no
forwards and measured 0.956× when every input changed. Cached representations
matched independent outputs exactly in these runs. These controlled synthetic
results do not measure semantic accuracy, downstream task quality, or
natural-video generalization.

## 1. Introduction

Sequential observations can contain repeated visual content. Recomputing a
feature representation for every frame is straightforward, but may repeat
work when an input is unchanged. DeltaVision asks whether a stateful system can
reuse such work while preserving the output of an independent reference.

This repository evaluates two different settings. The handcrafted setting
computes RGB means and population standard deviations for fixed image patches
and can refresh selected patch vectors. The pretrained setting runs a frozen
vision transformer and can reuse a complete frame representation when its
preprocessed input recurs. The two settings use different representations and
computation measures; patch-operation counts are not DINOv2 forward calls.

DeltaVision is an independent research software project. This report is a
technical record of the experiments, not a claim of publication, peer review,
or a novel model architecture.

## 2. Research Questions

1. When does exact visual-state reuse reduce computation while preserving the
   output of independent feature encoding?
2. How do fixed approximate RGB-difference thresholds trade feature staleness
   and change detection against handcrafted patch work?
3. For a frozen DINOv2 encoder, how do exact whole-frame cache hits affect
   complete model-forward counts, measured CPU time, and resident tensor
   storage?

Here, **representation correctness** means equality or measured numerical
agreement with the independent encoder output for the same input and model
configuration. It does not mean that the representation is semantically
correct or useful for a downstream task.

## 3. Related Work

The Vision Transformer represents an image as a sequence of embedded patches
[1]. DINOv2 provides self-supervised pretrained visual features using a Vision
Transformer backbone [2]. The model card for the specific `facebook/dinov2-small`
checkpoint describes the model and its feature-extraction interface [3]. The
repository pins the model snapshot and processor configuration [4, 5].

The experiments here use deterministic memoization for complete DINOv2
representations and simple per-patch RGB reuse policies. They do not implement
or compare learned temporal gating, token pruning, or selective recomputation
methods. No novelty claim about caching or temporal vision methods is made.

## 4. Methodology

### 4.1 Synthetic benchmark and ground truth

The benchmark generator renders small scenes with geometric shapes and
produces deterministic RGB frame sequences. Its configured transformations
are movement, recoloring, appearance, disappearance, partial occlusion, global
brightness, and unchanged frames. A generated mask pixel is positive when any
rendered RGB channel differs from the corresponding pixel in the preceding
frame. A patch is labeled changed when at least one mask pixel lies in it.
These labels describe rendered pixel changes; they are not semantic object
annotations or measures of scene meaning.

The final handcrafted workload uses 128×128 frames, eight frames per episode,
and 16×16 pixel patches, giving an 8×8 patch grid. Each patch has six
handcrafted values: normalized R, G, and B means and population standard
deviations. Patches are visited in row-major order.

### 4.2 Exact and approximate RGB reuse

The independent RGB baseline computes all patch vectors from each frame.
Exact patch reuse compares the raw RGB bytes at each patch position with the
preceding frame in the same episode and reuses a feature vector only when
those bytes match exactly. Changed patches are recomputed with the independent
feature function. The cache resets at episode boundaries.

Approximate reuse computes normalized mean absolute RGB difference per patch
and reuses a feature when its score is at most 0.01. The current policy updates
its pixel reference each frame, so repeated small changes can leave the
retained feature increasingly different from the current image. The global
fallback combines the configured local threshold 0.05 with a full-refresh
trigger when the changed-pixel fraction exceeds 0.50. Its global statistic is
an area test, not a semantic change detector.

### 4.3 Frozen DINOv2 and exact whole-frame cache

The learned-feature experiment uses `facebook/dinov2-small`, pinned to revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`, with deterministic CPU inference,
evaluation mode, and frozen parameters. Preprocessing converts to RGB, resizes
the shortest edge to 256 using bicubic interpolation, center-crops to 224×224,
rescales by 1/255, and normalizes with ImageNet mean `(0.485, 0.456, 0.406)`
and standard deviation `(0.229, 0.224, 0.225)`. The evaluated representation
is the 16×16×384 spatial-token tensor.

For each episode, the exact cache hashes the complete preprocessed model
input, then checks exact tensor equality for candidate hits. A hit returns the
complete cached representation. A miss runs a full model forward and stores
the complete selected representation. Capacity is eight entries per episode
for the final DINOv2 evaluation. This does not recompute spatial tokens
selectively: every miss still runs the entire DINOv2 model.
No selective DINOv2 token computation is implemented.

## 5. Experimental Setup

The synthetic benchmark configuration defines 100 development episodes
(seeds 17000–17099), 30 validation episodes (27100–27129), and 30 final
handcrafted episodes (37900–37929). Each final handcrafted episode has eight
frames, for 240 frames and 210 evaluated transitions. The configuration fixes
the tested policies before final evaluation.

The held-out synthetic partition was inspected during earlier experiments.
The Task 17 final report therefore records a **previously inspected held-out
evaluation**, not a blind test. The final handcrafted policies were fixed
using validation reports before the Task 17 run; that prespecification does
not erase the earlier test exposure.

The DINOv2 final comparison uses validation data only: four eight-frame
episodes per condition across identical consecutive frames, sparse changes,
moderate changes, and frequent widespread changes. Capacity ablation uses
four deterministic validation traces of 64–65 frames each. Neither pretrained
experiment accesses the held-out test partition.

The Task 17 run used three untimed warmups and five timed repetitions. Frames
were loaded before timing. Handcrafted timings include encoding, patch
comparisons, and cache updates; DINOv2 timings include preprocessing, hashing,
exact equality lookup, cache retrieval or storage, and complete forwards on
misses. Both timing protocols exclude disk reads and result serialization;
DINOv2 timing also excludes model loading and the separate feature audit.

The recorded host was macOS 26.6.2 on arm64, with 12 logical CPUs. The runtime
reported Python 3.13.7, PyTorch 2.15.0, Transformers 5.22.1, and Pillow 13.0.0;
CUDA was unavailable. The runtime identified the processor as `arm` and did
not report a more specific chip model.

## 6. Baselines

| Family | Method | Output or reuse rule | Cost quantity |
| --- | --- | --- | --- |
| Handcrafted | Independent RGB statistics | Recompute six values for every patch | Patch-feature operations |
| Handcrafted | Exact patch reuse | Reuse only byte-identical raw RGB patches | Recomputed patch-feature operations |
| Handcrafted | Approximate reuse, threshold 0.01 | Reuse when normalized mean RGB difference is at most 0.01 | Recomputed patch-feature operations plus comparison work |
| Handcrafted | Global fallback, local 0.05 / global 0.50 | Local RGB difference with a whole-frame changed-area trigger | Recomputed patch-feature operations plus detector work |
| Pretrained | Independent DINOv2 | One full frozen-model forward per frame | Full neural-network forwards and measured latency |
| Pretrained | Exact whole-frame DINOv2 cache | Reuse the whole output only for an exactly equal preprocessed input | Full forwards avoided, cache overhead, and measured latency |

The handcrafted operation fraction is a work proxy, not an estimate of
neural-network FLOPs. The DINOv2 call count is a count of complete model
forwards. These values cannot be compared as if they measured the same work.

## 7. Results

### 7.1 Handcrafted RGB statistics

The following values come from `results/final_evaluation.json` and
`results/final_statistics.json`. Transition metrics classify patch refresh
decisions against rendered pixel-change labels. F1 intervals are 95% episode-
cluster bootstrap intervals.

| Method | Precision | Recall | F1 (95% CI) | Missed changes | False refresh | Feature MAE | All-frame operation fraction | Median CPU seconds / 240 frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent RGB | 0.2168 | 1.0000 | 0.3564 (0.2899–0.4262) | 0.0000 | 1.0000 | 0 | 1.0000 | 1.295 |
| Exact patch reuse | 1.0000 | 1.0000 | 1.0000 (1.0000–1.0000) | 0.0000 | 0.0000 | 0 | 0.3147 | 0.439 |
| Approximate reuse, 0.01 | 1.0000 | 0.9949 | 0.9974 (0.9951–0.9992) | 0.0051 | 0.0000 | 0.0000194 | 0.3137 | 0.722 |
| Global fallback | 1.0000 | 0.9650 | 0.9822 (0.9709–0.9889) | 0.0350 | 0.0000 | 0.000586 | 0.3081 | 0.963 |

Precision, recall, F1, missed-change rate, and false-refresh rate exclude each
episode's initialization frame. Feature MAE and the operation fraction
include all frames. Independent encoding refreshes every transition patch,
which gives full recall but also refreshes every unchanged patch. F1 is 0 for
the unchanged-only transformation under the evaluator's convention for a
class with no positive labels.

Exact patch reuse reproduced the independent handcrafted feature vectors for
unchanged byte-level patches and used 31.5% of the independent all-frame
patch-operation count. Approximate and global policies executed slightly
fewer patch feature computations, but their comparison work and feature error
are separate costs. In this run, their median end-to-end CPU times were
greater than exact reuse.

### 7.2 Frozen DINOv2

Each validation condition contains 32 frames (four episodes × eight frames).
“Ratio” is independent median latency divided by cached median latency, so a
value below 1 indicates that the cache path was slower. Maximum and mean
feature differences were 0 and cosine similarity was 1.0 for every condition.

| Validation condition | Independent / cached forwards | Cache hits / hit rate | Median seconds, independent / cache | Measured ratio | Paired episode-bootstrap 95% CI* | Peak tensor payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identical consecutive | 32 / 4 | 28 / 87.5% | 0.611 / 0.107 | 5.689× | 5.612–5.759 | 0.95 MiB |
| Sparse changes | 32 / 8 | 24 / 75.0% | 0.589 / 0.176 | 3.337× | 3.301–3.344 | 1.90 MiB |
| Moderate changes | 32 / 16 | 16 / 50.0% | 0.601 / 0.322 | 1.869× | 1.846–1.850 | 3.80 MiB |
| Frequent widespread changes | 32 / 32 | 0 / 0.0% | 0.622 / 0.651 | 0.956× | 0.963–0.993 | 7.59 MiB |

All features had shape 16×16×384. The cache's median speedup ratio is the
ratio of the two whole-condition latency medians. *The interval is instead a
paired bootstrap of per-episode median latency ratios, so it has a different
estimand and its center need not equal the whole-condition point estimate.
Intervals resample four episodes and should be read cautiously. Five timed
repetitions quantify measurement variation separately from episode-sampling
uncertainty.

## 8. Ablation Studies

The Task 16 validation-only capacity ablation compared capacities 1, 2, 4, 8,
16, and 32 using three warmups and five timed repetitions. The table below
shows the observed capacity 1-to-32 range from
`results/pretrained_cache_ablation.json`; timing ratios are independent/cache
median ratios. Resident payload estimates exclude Python object, allocator,
and in-flight tensor overhead.

| Synthetic trace | Frames; distinct inputs | Capacity 1 → 32: full forwards | Cache hit rate | Median timing ratio | Payload estimate |
| --- | --- | ---: | ---: | ---: | ---: |
| Repeated identical | 64; 1 | 1 → 1 | 98.4% → 98.4% | 14.011× → 13.989× | 0.949 MiB → 0.949 MiB |
| Recurring nonconsecutive | 65; 32 | 65 → 32 | 0% → 50.8% | 0.985× → 1.883× | 0.949 MiB → 30.375 MiB |
| Sparse changes | 64; 4 | 4 → 4 | 93.8% → 93.8% | 8.408× → 8.392× | 0.949 MiB → 3.797 MiB |
| Frequent widespread | 64; 64 | 64 → 64 | 0% → 0% | 0.984× → 0.981× | 0.949 MiB → 30.375 MiB |

For the recurring trace, capacity 8 produced 58 forwards, 7 hits (10.8%), a
1.095× ratio, and 7.594 MiB of tensor payload. Increasing capacity helped
only when it retained inputs that recurred later. It did not reduce forwards
for all-unique inputs, where storage and lookup remained overhead.

## 9. Failure Analysis

In the previously inspected held-out handcrafted evaluation, approximate
reuse at 0.01 missed 3.85% of changed patches under partial occlusion and
4.96% under recoloring. The global fallback missed 29.49% and 33.88%,
respectively. These category rates are patch-level comparisons to the
synthetic RGB masks and do not indicate semantic object recognition. The
global-area trigger can leave small-area changes below 0.50; a broad brightness
change can instead force a full refresh and remove reuse for that transition.

The failure gallery contains deterministic validation examples of a small
approximate-reuse miss, a global brightness shift, partial occlusion, and
recoloring. Its DINOv2 examples show a frequent-change condition with zero
hits and a measured slowdown. Gallery masks visualize RGB differences, not
semantic annotations. See the [gallery index](failure_gallery/index.md).

The DINOv2 failure condition changes every preprocessed input. In that case,
all 32 frames still execute a full model forward; preprocessing, hashing,
equality checking, storage, and retrieval do not substitute for neural
computation.

## 10. Discussion

The evidence supports a narrow conclusion. Exact reuse can avoid work when an
input or patch is unchanged, and it preserves the deterministic reference
output for that unchanged input. Approximate reuse extends reuse to inputs
that differ, but can retain stale handcrafted features. A global-area rule
can prompt broad recomputation, yet its area statistic is not a proxy for
semantic importance.

For DINOv2, exact whole-frame caching reduced full forwards and measured CPU
latency on these repeated synthetic inputs. Capacity mattered when inputs
recurred nonconsecutively. When all inputs differed, the cache had no hits and
was slower. Hit rate and avoided forwards are therefore not substitutes for
end-to-end timing. The experiments provide no evidence for selective neural
token computation or acceleration on GPUs.

## 11. Limitations and Threats to Validity

- The benchmark uses small, rendered geometric scenes and RGB-difference
  masks. It does not provide semantic labels, real video, or a downstream
  prediction task.
- The synthetic held-out partition had prior exposure in earlier work. It is
  not a blind test, and results must not be generalized as though it were.
- The DINOv2 validation conditions include four episodes per condition. The
  episode-bootstrap intervals have limited resolution and do not establish
  statistical significance or population-wide performance.
- CPU timings describe one macOS ARM host and one software stack. They do not
  predict GPU performance or other systems. Five repeated timings capture
  measurement variation on the fixed workloads, not new data samples.
- Cache-memory values count resident tensor payload only. Runtime structures,
  allocator overhead, and in-flight tensors add memory use.
- Feature equality establishes numerical agreement with the configured
  independent encoder on the tested inputs. It does not establish semantic
  accuracy, feature quality, or downstream utility.
- The study does not compare learned refresh policies, selective neural-token
  methods, or natural-video workloads, and makes no novelty claim.

## 12. Conclusion

Visual-state reuse helps when the complete state being reused is unchanged
and occurs often enough to offset preprocessing, comparison, and storage
costs. Exact patch reuse reproduced handcrafted features on byte-identical
patches; approximate policies introduced measurable stale-feature error and
missed changes. The frozen DINOv2 cache returned exactly matching complete
representations for exact preprocessed-input hits, reduced model forwards on
repeated validation inputs, and was slower when every input changed. This is
evidence about exact caching on the specified synthetic workloads, not about
semantic correctness or real-world video.

## References

1. Alexey Dosovitskiy et al. “An Image is Worth 16×16 Words: Transformers for
   Image Recognition at Scale.” arXiv:2010.11929, 2020.
   [arXiv record](https://arxiv.org/abs/2010.11929).
2. Maxime Oquab et al. “DINOv2: Learning Robust Visual Features without
   Supervision.” arXiv:2304.07193, 2023.
   [arXiv record](https://arxiv.org/abs/2304.07193).
3. Hugging Face. [`facebook/dinov2-small` model card](https://huggingface.co/facebook/dinov2-small).
   The card states that it was written by the Hugging Face team, not the team
   releasing DINOv2.
4. Hugging Face. [`facebook/dinov2-small` pinned revision](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)
   and [revision-pinned processor configuration](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/preprocessor_config.json).
5. Meta AI. [DINOv2 repository model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).

## Reproducibility Appendix

The repository contains configuration files and runners for the synthetic
benchmark, final evaluation, cache ablation, figures, and demo. See
[`docs/reproducibility.md`](reproducibility.md) for setup, exact commands,
expected output paths, local-checkpoint requirements, and lightweight tests.

The final numeric tables above were transcribed from
`results/final_evaluation.json`, `results/final_statistics.json`, and
`results/pretrained_cache_ablation.json`; their experiment protocols are
documented in [`results/README.md`](../results/README.md). Result JSON/CSV
files and generated data are excluded from Git by default, so regenerate them
locally to inspect the complete machine-readable records.
