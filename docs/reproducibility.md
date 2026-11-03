# Reproducibility Guide

This guide describes how to regenerate the synthetic inputs and saved outputs
used by the project. Generated datasets, result JSON/CSV files, and model
weights are local artifacts and are excluded from Git by default.

## Python and installation

The package requires Python 3.11 or later. The Task 17 run recorded Python
3.13.7, PyTorch 2.15.0, Transformers 5.22.1, and Pillow 13.0.0 on macOS 26.6.2
arm64 with 12 logical CPUs. The processor was reported as `arm`; the runtime
did not identify a more specific chip. CUDA was unavailable. No dependency
lockfile is included, so versions installed from the minimum requirements can
differ from the recorded run.

Install the project, development tools, and Pillow for visual tests and
figures without installing the pretrained-model stack:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

For experiments that load DINOv2, install the optional pretrained
dependencies:

```bash
.venv/bin/python -m pip install -e ".[dev,pretrained]"
```

The pretrained extra includes PyTorch and Transformers. Installation does not
download the DINOv2 checkpoint. The project itself does not include or
redistribute weights.

## Model checkpoint and preprocessing

The configured model is `facebook/dinov2-small` at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`, with `spatial_tokens` output and
CPU inference. The pinned preprocessing converts to RGB, resizes the shortest
edge to 256 using bicubic interpolation, center-crops to 224×224, rescales by
1/255, and applies mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`. The evaluated spatial-token shape is 16×16×384.

The baseline config uses `data/pretrained_cache` and checkpoint downloads are
disabled for the recorded evaluations. If the exact revision is not already
in the local cache, a pretrained run exits with an error; it does not silently
download weights. To obtain weights manually, follow the model provider's
terms and place the pinned files in the configured local cache. Do not add
model weights to this repository.

## Lightweight smoke tests without model weights

The pytest suite uses mocked pretrained model and processor objects for model
adapter tests; it does not download or load the DINOv2 checkpoint. Pillow is
needed for visual tests, but PyTorch and Transformers are not required for the
lightweight synthetic RGB runs.

```bash
.venv/bin/python -m pytest -q
```

Generate a small deterministic sequence and run the independent and exact
handcrafted baselines:

```bash
.venv/bin/python scripts/build_sequences.py \
  --output data/generated/smoke --episodes 4 --seed 17
.venv/bin/python scripts/run_baseline.py \
  data/generated/smoke/manifest.jsonl \
  --output data/generated/smoke_baseline.jsonl --patch-size 16
.venv/bin/python scripts/run_stateful.py \
  data/generated/smoke/manifest.jsonl \
  --output data/generated/smoke_stateful.jsonl --patch-size 16
```

These commands use the repository's deterministic geometric generator. A
mask pixel is positive when any rendered RGB channel differs from the
preceding frame. This is pixel-difference ground truth, not semantic labeling.

## Benchmark data generation

To regenerate the configured training, validation, and held-out partitions:

```bash
.venv/bin/python scripts/build_research_benchmark.py \
  --config configs/research_benchmark.json \
  --output data/generated/research_benchmark
```

This writes deterministic manifests and PPM/PGM files under
`data/generated/research_benchmark/`. The partitions contain 100 development
episodes (seeds 17000–17099), 30 validation episodes (27100–27129), and 30
synthetic held-out episodes (37900–37929). The held-out partition was
inspected in earlier experiments. Regenerating or rerunning its evaluation
does not make it blind; do not use it to select a policy.

The small standalone generator is also available for custom smoke data:

```bash
.venv/bin/python scripts/build_sequences.py \
  --output data/generated/demo --episodes 4 --seed 17 \
  --width 128 --height 128 --sequence-length 8
```

## Handcrafted baseline commands

Run an approximate threshold sweep on an existing manifest:

```bash
.venv/bin/python scripts/run_threshold_sweep.py \
  data/generated/demo/manifest.jsonl \
  --output results/threshold_sweep.json \
  --figure docs/figures/threshold_tradeoff.png --patch-size 16
```

Run the validation-only global-change analysis using the configured validation
manifest:

```bash
.venv/bin/python scripts/evaluate_global_change.py \
  --config configs/global_change_validation.json
```

The first command uses the manifest supplied on its command line. The second
reads only the validation partition named by its configuration.

## Pretrained DINOv2 commands

Every command in this section requires the locally cached pinned DINOv2
checkpoint and the optional pretrained dependencies. The supplied configs
disable checkpoint downloads. Obtaining the checkpoint and installing Python
packages may require network access; once dependencies and weights are cached,
the experiment runners do not need network access. These are the resource-
intensive steps: they run repeated full DINOv2-small CPU forwards, and the
capacity ablation repeats the workload across six cache capacities. They can
take substantially longer than the lightweight RGB smoke checks.

Run the independent-frame adapter on a manifest:

```bash
.venv/bin/python scripts/run_pretrained_baseline.py \
  data/generated/demo/manifest.jsonl \
  --config configs/pretrained_baseline.json \
  --output-dir results/pretrained_baseline
```

Run the validation workload comparing independent inference with exact
whole-frame caching:

```bash
.venv/bin/python scripts/evaluate_pretrained_caching.py \
  --config configs/pretrained_cache_evaluation.json
```

Run the validation-only cache-capacity ablation (capacities 1, 2, 4, 8, 16,
and 32):

```bash
.venv/bin/python scripts/ablate_pretrained_cache.py \
  --config configs/pretrained_cache_ablation.json
```

The independent and stateful policies use identical frames and preprocessing.
The stateful DINOv2 method reuses a complete representation only when the
complete preprocessed input matches exactly. It does not selectively compute
tokens.

## Final evaluation and statistical analysis

The following reproduces the saved Task 17 reports. It includes handcrafted
metrics on the **previously inspected** synthetic held-out partition and
DINOv2 results on validation data; it is not a blind-test procedure. It
requires generated benchmark data and the locally cached DINOv2 checkpoint.

```bash
.venv/bin/python scripts/run_final_evaluation.py \
  --config configs/final_evaluation.json
.venv/bin/python scripts/analyze_final_results.py \
  --input results/final_evaluation.json \
  --config configs/final_evaluation.json \
  --output results/final_statistics.json \
  --figure-dir docs/figures
```

The evaluation writes `results/final_evaluation.json` and `.csv`; analysis
writes `results/final_statistics.json` and the final accuracy/efficiency
figures. The JSON contains configuration hashes, split provenance, runtime
versions, raw timing repetitions, episode metrics, and bootstrap intervals.
Bootstrap sampling uses 1,000 paired episode-cluster replicates. It quantifies
episode-sampling variation for these synthetic inputs; repeated timing
variation is recorded separately.

## Figures and demonstration

Once the saved evaluation, statistics, capacity-ablation, and global-change
outputs and their validation manifests are available, generate the report
figures, failure gallery, and demo:

```bash
.venv/bin/python scripts/create_research_demo.py
```

Expected outputs include:

- `docs/figures/architecture.png`
- `docs/figures/final_research_summary.png`
- `docs/figures/accuracy_efficiency_tradeoff.png`
- `docs/figures/pretrained_speedup.png`
- `docs/figures/cache_capacity_summary.png`
- `docs/failure_gallery/` cards, index, and provenance manifest
- `docs/demo/deltavision_demo.gif` and metadata JSON

The demo command reads saved DINOv2 metrics but performs no pretrained
inference. It creates handcrafted RGB visuals on CPU. An MP4 is written only
when a local `ffmpeg` executable is available; it is optional.

## Expected output and artifact policy

Synthetic frames, masks, feature arrays, result JSON/CSV files, and checkpoint
weights are local generated artifacts and are ignored by Git. The README,
technical report, result documentation, selected figures, failure gallery,
and demo are the lightweight presentation artifacts. Regenerate ignored
results locally before inspecting or citing their complete machine-readable
records.

| Reproduction step | Principal outputs |
| --- | --- |
| Build research benchmark | `data/generated/research_benchmark/{train,validation,test}/manifest.jsonl` and frame/mask files |
| Global-change validation | `results/global_change_validation.json` and its configured validation figure |
| DINOv2 condition evaluation | `results/pretrained_cache_evaluation.json`, `.csv`, timing figures, and validation manifests |
| DINOv2 capacity ablation | `results/pretrained_cache_ablation.json`, `.csv`, and capacity figure |
| Final evaluation | `results/final_evaluation.json` and `.csv` |
| Final statistical analysis | `results/final_statistics.json`, accuracy and efficiency figures |
| Research visuals and demo | Five Task 18 figures, gallery cards and manifest, GIF and metadata JSON |

Result and data outputs are intentionally not committed by the default ignore
rules. Figure exceptions in `.gitignore` keep the selected README figures
trackable; the raw result and dataset trees remain local.

## Runtime limitations

The recorded measurements are CPU-only measurements on one macOS ARM host.
They are workload- and system-specific and do not establish GPU speedup.
Handcrafted patch-operation proxies and full DINOv2 forward counts measure
different work. Resident cache-payload estimates omit Python object,
allocator, and in-flight tensor overhead. Synthetic masks do not establish
semantic accuracy, and these experiments do not evaluate natural video or a
downstream prediction task.
