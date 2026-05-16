# GPU SES Benchmark

This repository includes a Docker-based benchmark path for collecting GPU
runtime, memory, and failure statistics across all SES sampling methods on the
real PDB dataset in `Data/01-benchmark_pdbs`.

## Requirements

- Ubuntu 22.04 host with recent NVIDIA drivers.
- Docker with the NVIDIA Container Toolkit configured.
- The benchmark dataset available at `Data/01-benchmark_pdbs`.

The Docker image uses CUDA 12.1 runtime libraries and installs PyTorch from the
CUDA 12.1 wheel index. Override `SES_BENCH_TORCH_INDEX_URL` if the server needs
a different PyTorch CUDA wheel family.

## One-Command Full Run

From the repository root:

```bash
scripts/run_gpu_benchmarks.sh
```

The script builds `protein-surfaces-gpu-bench:latest`, starts a container with
`--gpus all`, runs all implemented methods on every PDB file, and streams JSONL
results to:

```text
tmp/gpu_benchmarks/ses_gpu_benchmark_<program version>.jsonl
```

For non-quick modes the wrapper appends the mode name unless
`SES_BENCH_OUTPUT` is set, for example
`ses_gpu_benchmark_<program version>_detail.jsonl`.

A compact summary is written next to it as:

```text
tmp/gpu_benchmarks/ses_gpu_benchmark_<program version>.summary.json
```

Send the JSONL and summary files back for regression analysis. The default run
uses `--mode quick`: all methods, all PDB files, largest molecules first,
points-only output, no profiler overhead, no Chrome trace export, and fsync
after every JSONL record. That keeps the file useful after interruption without
turning routine regression control into a trace dump.

The benchmark has three modes:

- `quick`: fast regression control over the selected PDB directory. Defaults to
  all methods, `--interfaces points`, `--sweep-preset none`, `--repeats 1`,
  largest molecules first, no reference metrics, no internal profiling, and no
  PyTorch profiler.
- `detail`: compact critical-path profiling. Defaults to internal function
  summaries plus a limited PyTorch top-op profile, but stores profile details in
  binary `.pt` artifacts under `tmp/gpu_benchmarks/profiles/` instead of
  inlining them into the streaming JSONL.
- `sweep`: parameter-grid tuning. Defaults to the focused sweep preset, three
  repeats, no profiling, and the same compact JSONL records as quick mode.

Chrome trace JSON export is disabled in every mode unless
`--torch-profile-export-traces` is explicitly passed. The compact top-op summary
and internal function summary are usually enough for hot-path optimization.

Each run records `program_version`, defaulting to `0.0.3`. For release
benchmarks, set `SES_BENCH_PROGRAM_VERSION` to the same semantic version as the
GitHub tag, for example `0.1.0`.

The wrapper automatically passes `--resume` by default, so restarting the same
program version skips already recorded molecule/method/parameter/repeat results
and only fills missing records. Use `--overwrite` when you intentionally want
to rebuild the benchmark file for the same version.

## Release Optimization Loop

The intended release workflow is:

1. After code changes, run a quick benchmark for the candidate version.
2. Compare the quick result with the latest release benchmark.
3. If the comparison is clean, cut the next release benchmark/tag.
4. If it regresses, run detail mode for the affected methods/molecules and
   optimize the hot path.
5. If new parameters or heuristics were introduced, run sweep mode and adjust
   defaults or `auto` selection from the sweep result.
6. Repeat from quick mode until the candidate is clean.

Quick release-gate run:

```bash
SES_BENCH_PROGRAM_VERSION=0.0.4 \
scripts/run_gpu_benchmarks.sh --mode quick
```

Compare it with the latest release JSONL:

```bash
python scripts/analyze_gpu_benchmarks.py compare \
  --baseline tmp/gpu_benchmarks/releases/ses_gpu_benchmark_0.0.3.jsonl \
  --current tmp/gpu_benchmarks/ses_gpu_benchmark_0.0.4.jsonl \
  --fail-on-regression \
  --output tmp/gpu_benchmarks/ses_gpu_benchmark_0.0.4.compare.json
```

The wrapper can run that comparison automatically after collection:

```bash
SES_BENCH_PROGRAM_VERSION=0.0.4 \
SES_BENCH_BASELINE_OUTPUT=tmp/gpu_benchmarks/releases/ses_gpu_benchmark_0.0.3.jsonl \
SES_BENCH_COMPARE_OUTPUT=tmp/gpu_benchmarks/ses_gpu_benchmark_0.0.4.compare.json \
SES_BENCH_COMPARE_FAIL=1 \
scripts/run_gpu_benchmarks.sh --mode quick
```

The comparison groups records by molecule, method, method variant, and
interface mode. It reports aggregate variant medians, the worst per-molecule
regressions, improvements, missing cases, and point-count changes.

When quick comparison finds a regression, collect compact detail profiles on
the problematic slice:

```bash
SES_BENCH_PROGRAM_VERSION=0.0.4 \
scripts/run_gpu_benchmarks.sh --mode detail \
  --methods tiled_analytic \
  --limit 20 \
  --output tmp/gpu_benchmarks/ses_gpu_benchmark_0.0.4_detail.jsonl

python scripts/analyze_gpu_benchmarks.py profiles \
  --benchmark tmp/gpu_benchmarks/ses_gpu_benchmark_0.0.4_detail.jsonl \
  --output tmp/gpu_benchmarks/ses_gpu_benchmark_0.0.4_detail.profiles.json
```

When tuning defaults or `auto` heuristics, run a parameter grid and rank
variants:

```bash
SES_BENCH_PROGRAM_VERSION=0.0.4 \
scripts/run_gpu_benchmarks.sh --mode sweep \
  --methods analytic,tiled_analytic \
  --interfaces points,normals,adjacency

python scripts/analyze_gpu_benchmarks.py sweep \
  --benchmark tmp/gpu_benchmarks/ses_gpu_benchmark_0.0.4_sweep.jsonl \
  --output tmp/gpu_benchmarks/ses_gpu_benchmark_0.0.4_sweep.analysis.json
```

## Smoke Run

Use a small limit before launching the full dataset:

```bash
scripts/run_gpu_benchmarks.sh --limit 10 --log-every 1
```

Largest molecules already run first in quick mode. To use a different order:

```bash
scripts/run_gpu_benchmarks.sh --molecule-order name --limit 10 --log-every 1
```

CPU-only smoke runs are also possible for validating the benchmark driver inside
the dev container:

```bash
python scripts/benchmark_ses_gpu.py --device cpu --no-require-cuda \
  --data-dir tests/data/pdb --limit 1 --overwrite
```

## Existing Container

If the benchmark image or another suitable CUDA container is already running,
set `SES_BENCH_CONTAINER` to its container name or id. The wrapper will skip
`docker build` and `docker run`, then execute the benchmark inside that
container with `docker exec`:

```bash
SES_BENCH_CONTAINER=ses-gpu-bench \
scripts/run_gpu_benchmarks.sh --limit 10 --log-every 1
```

The existing container must already have GPU access, Python dependencies, and
the repository mounted or copied at `/workspace`. `SES_BENCH_OUTPUT`,
`SES_BENCH_DATA_DIR`, and `SES_BENCH_SURFACE_DIR` are interpreted inside that
container workdir when they are relative paths. If the repository is at a
different path inside the container, set `SES_BENCH_CONTAINER_WORKDIR`:

```bash
SES_BENCH_CONTAINER=ses-gpu-bench \
SES_BENCH_CONTAINER_WORKDIR=/work/protein-surfaces \
scripts/run_gpu_benchmarks.sh --limit 10
```

If you are already attached to the container shell, run the wrapper directly in
the current environment:

```bash
SES_BENCH_RUN_LOCAL=1 scripts/run_gpu_benchmarks.sh --limit 10
```

In existing-container or local-current-environment modes, Python dependencies
must already be installed. To let the wrapper install `requirements.txt` in
that environment first, add `SES_BENCH_INSTALL_DEPS=1`:

```bash
SES_BENCH_RUN_LOCAL=1 \
SES_BENCH_INSTALL_DEPS=1 \
scripts/run_gpu_benchmarks.sh --limit 10
```

## Resuming

Results are written after every molecule/method pair. The wrapper uses a stable
versioned output path and enables resume by default:

```bash
scripts/run_gpu_benchmarks.sh
```

For a custom output path, keep using the same file:

```bash
SES_BENCH_OUTPUT=tmp/gpu_benchmarks/full.jsonl \
scripts/run_gpu_benchmarks.sh
```

Resume skips records whose method parameter hash already exists in that JSONL
file for the same `program_version`. If you change benchmark parameters, the
method parameter hash changes and the missing records are added. Set
`SES_BENCH_AUTO_RESUME=0` to restore the older fail-if-output-exists behavior.

## Parameter Sweeps

The benchmark can run every method across parameter variants. This is the main
way to find the fastest useful settings before changing code.

Focused sweep, one parameter varied at a time:

```bash
scripts/run_gpu_benchmarks.sh --mode sweep
```

Broader sweep:

```bash
scripts/run_gpu_benchmarks.sh --mode sweep --sweep-preset broad
```

Explicit method-specific sweeps:

```bash
scripts/run_gpu_benchmarks.sh \
  --mode sweep \
  --methods projected,sdf,tiled_analytic \
  --projected-m-values 96,160,192,230,320 \
  --sdf-m-values 16,26,34,64 \
  --sdf-smoothness-values 0.15,0.2,0.3 \
  --tile-size-values auto,128,256,512 \
  --tiled-atom-density-scale-values 0.75,1,1.55,3 \
  --tiled-pair-density-scale-values 0,0.75,1,1.55 \
  --tiled-probe-density-scale-values 0,0.75,1,1.55
```

By default, sweep presets create one-axis variants so the result remains
interpretable. Add `--sweep-cartesian` when you intentionally want a Cartesian
product. The summary JSON includes `variant_summaries` and
`fastest_variants_by_method`.

Sweep mode defaults to repeated clean runs with no profiling. The summary
exposes both `wall_seconds` and `clean_wall_seconds`; in sweep mode they should
match unless profiling is explicitly enabled:

```bash
scripts/run_gpu_benchmarks.sh --mode sweep --repeats 5
```

## Reference Metrics

Reference metrics are disabled by default in all modes because they add a
separate nearest-neighbor pass. When matching `.ply` files are present in
`Data/01-benchmark_surfaces`, enable them to record approximate quality metrics
against reference vertices:

- generated point count to reference vertex count ratio;
- point-to-reference and reference-to-point nearest-distance summaries;
- symmetric mean nearest-distance score.

Distances are computed on deterministic subsamples so they stay practical on
the full dataset:

```bash
scripts/run_gpu_benchmarks.sh --reference-metrics --reference-sample-size 8192
```

## Profiling Detail

Use `detail` mode when the quick run finds a regression and you need the hot
path:

```bash
scripts/run_gpu_benchmarks.sh --mode detail --methods tiled_analytic --limit 10
```

Each profiled result writes a compact `profile_artifact` path in the JSONL. The
default artifact format is binary `.pt` and contains:

- `internal_profile.top_functions`: inclusive SES helper timings plus maximum
  input/output tensor counts and bytes;
- `internal_profile.top_calls`: slow individual helper calls;
- `torch_profile.top_ops`: top CUDA, CPU, and memory operators.

The profile payload is not inlined into JSONL by default. Inline it only for
small diagnostic runs:

```bash
scripts/run_gpu_benchmarks.sh --mode detail --limit 3 --inline-profile-details
```

For deeper CUDA/operator traces on a limited subset, explicitly export Chrome
trace JSON files. Keep the limit small; these are for viewer inspection, not
routine optimization loops:

```bash
scripts/run_gpu_benchmarks.sh --limit 5 --torch-profile-limit 5 \
  --torch-profile-export-traces
```

Operator input-shape recording is disabled by default because shape-heavy traces
can become enormous. Re-enable it only for a tiny diagnostic subset:

```bash
scripts/run_gpu_benchmarks.sh --mode detail --limit 3 \
  --torch-profile-record-shapes
```

For maximum structure detail on a small diagnostic run, include nested sample
summaries:

```bash
scripts/run_gpu_benchmarks.sh --mode detail --limit 3 --profile-sample-structures
```

## Sharding

For multi-GPU or multi-server runs, start one process per shard and write each
shard to a separate output file:

```bash
SES_BENCH_OUTPUT=tmp/gpu_benchmarks/full_shard0.jsonl \
NVIDIA_VISIBLE_DEVICES=0 \
scripts/run_gpu_benchmarks.sh --shard-count 4 --shard-index 0

SES_BENCH_OUTPUT=tmp/gpu_benchmarks/full_shard1.jsonl \
NVIDIA_VISIBLE_DEVICES=1 \
scripts/run_gpu_benchmarks.sh --shard-count 4 --shard-index 1
```

## Important Parameters

- `--methods`: comma-separated subset, or `all`.
- `--mode`: `quick`, `detail`, or `sweep`. Default: `quick`.
- `--interfaces`: independent output variants to benchmark. Default: `points`,
  which requests only point coordinates. Use comma-separated `features`,
  `normals`, and `adjacency` to add isolated feature, normal, and graph
  measurements, or `all` for all four variants.
- `--sweep-preset`: `none`, `focused`, or `broad`; defaults to `focused`
  only in `sweep` mode.
- `--repeats`: repeated runs per molecule/method/variant. Defaults to `1` in
  quick/detail mode and `3` in sweep mode.
- `--largest-first`: shortcut for largest PDBs first, which is already the
  default order.
- `--molecule-order`: `name`, `atom_count_desc`, `atom_count_asc`, `file_size_desc`, or `file_size_asc`.
- `--point-area`: analytic target area per point. Default: `0.5`.
- `--projected-m`: projection seeds per atom. Default: `192`.
- `--sdf-m`: SDF seeds per atom. Default: `26`.
- `--analytic-oversample-factor`: analytic candidate oversampling. Default: `1.0`.
- `--tiled-point-area`: tiled analytic target area per point. Defaults to
  `--point-area`, which is `0.5` unless overridden.
- `--tiled-atom-density-scale`: tiled contact density multiplier. Default: `1.0`.
- `--tiled-pair-density-scale`: tiled pair-torus density multiplier. Default: `1.0`.
- `--tiled-probe-density-scale`: tiled fixed-probe density multiplier. Default: `1.0`.
- `--adjacency-weight`: graph edge weights, `euclidean` or `geodesic`. Default:
  `euclidean`.
- `--adjacency-neighbors`: maximum outgoing graph neighbors per point before
  symmetrization. Default: `6`.
- `--tile-size` / `--tile-overlap`: default to `auto`; numeric tile-size
  sweeps use overlap `4.0` unless overridden. For `tiled_analytic`, `auto`
  prefers the largest candidate tile that fits the built-in 3 GiB tile-work
  estimate. If one tile covers the molecule, `tiled_analytic` uses the same
  analytic block pipeline as `sample_analytic_points`.
- `--reference-sample-size`: reference quality subsample size. Default: `4096`.
- `--torch-profile-limit`: number of method/variant runs to profile with the
  PyTorch profiler. Defaults to `20` in detail mode and `0` otherwise.
- `--torch-profile-export-traces`: write Chrome trace JSON files. Disabled by
  default.
- `--profile-artifact-format`: `none`, `pt`, or `json`. Detail mode defaults to
  `pt`.
- `--profile-internals-every`: collect internal function profiles every N method/variant runs.
- `--max-atoms`: optional debugging guard for very large structures.
- `--dtype`: `float32` by default; use `float64` for precision comparisons.

The defaults follow the 0.0.3 GPU default, tile-size, and analytic/tiled deep
runs. The tiled analytic default uses `point_area=0.5` and
`atom/pair/probe_density_scale=1.0`; the projected, SDF, and analytic defaults
target a similar median point density.

## Useful Environment Variables

- `SES_BENCH_IMAGE`: Docker image tag.
- `SES_BENCH_OUTPUT`: JSONL output path.
- `SES_BENCH_MODE`: `quick`, `detail`, or `sweep`. Default: `quick`.
- `SES_BENCH_DATA_DIR`: PDB dataset directory.
- `SES_BENCH_SURFACE_DIR`: PLY reference surface directory.
- `SES_BENCH_PROGRAM_VERSION`: semantic program version recorded in benchmark output. Default: `0.0.3`.
- `SES_BENCH_INTERFACES`: default interface modes passed by the wrapper. Default: `points`.
- `SES_BENCH_MOLECULE_ORDER`: default PDB order, for example `atom_count_desc`.
- `SES_BENCH_REPEATS`: override mode-specific repeat defaults.
- `SES_BENCH_SWEEP_PRESET`: override mode-specific sweep defaults.
- `SES_BENCH_TORCH_PROFILE_LIMIT`: override mode-specific PyTorch profiler limits.
- `SES_BENCH_PROFILE_ARTIFACT_FORMAT`: `none`, `pt`, or `json`.
- `SES_BENCH_BASELINE_OUTPUT`: optional release JSONL for automatic post-run
  quick comparison.
- `SES_BENCH_COMPARE_OUTPUT`: optional JSON output path for automatic
  comparison reports.
- `SES_BENCH_COMPARE_FAIL=1`: make the wrapper fail when automatic comparison
  finds regressions, point-count changes, or missing baseline cases.
- `SES_BENCH_AUTO_RESUME=0`: disable wrapper auto-resume. Auto-resume is enabled by default.
- `SES_BENCH_CONTAINER`: run inside an already-running Docker container with `docker exec`.
- `SES_BENCH_CONTAINER_WORKDIR`: repository path inside that container. Default: `/workspace`.
- `SES_BENCH_EXEC_USER`: optional user passed to `docker exec --user`.
- `SES_BENCH_RUN_LOCAL=1`: run the benchmark directly in the current environment.
- `SES_BENCH_INSTALL_DEPS=1`: install Python dependencies before local or existing-container runs.
- `SES_BENCH_SKIP_BUILD=1`: reuse an already built image.
- `SES_BENCH_TORCH_INDEX_URL`: PyTorch wheel index, for example CUDA 12.4.
- `NVIDIA_VISIBLE_DEVICES`: restrict the container to selected GPU ids.
