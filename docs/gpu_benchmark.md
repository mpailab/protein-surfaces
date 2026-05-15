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

A compact summary is written next to it as:

```text
tmp/gpu_benchmarks/ses_gpu_benchmark_<program version>.summary.json
```

Send both files back for optimization work. The JSONL file is the most useful
artifact because it includes per-molecule timings, peak CUDA memory, point
counts, molecule geometry, internal function timings, intermediate tensor sizes,
reference-surface distance metrics, errors, tracebacks, environment metadata,
and git status.

The wrapper defaults are deliberately diagnostic: `--sweep-preset focused`,
`--repeats 3`, and `--torch-profile-limit 100`. That gives profiled repeats on
a limited prefix plus cleaner timing repeats for each molecule/method/variant.
PyTorch traces can be large and slow to write, so use a lower
`--torch-profile-limit` for throughput-focused full-dataset runs. Override the
defaults with normal CLI flags, or set `SES_BENCH_SWEEP_PRESET`,
`SES_BENCH_REPEATS`, and `SES_BENCH_TORCH_PROFILE_LIMIT` before running the
wrapper.

Each run records `program_version`, defaulting to `0.0.3`. For release
benchmarks, set `SES_BENCH_PROGRAM_VERSION` to the same semantic version as the
GitHub tag, for example `0.1.0`.

The wrapper automatically passes `--resume` by default, so restarting the same
program version skips already recorded molecule/method/parameter/repeat results
and only fills missing records. Use `--overwrite` when you intentionally want
to rebuild the benchmark file for the same version.

## Smoke Run

Use a small limit before launching the full dataset:

```bash
scripts/run_gpu_benchmarks.sh --limit 10 --log-every 1
```

To stress the largest molecules first:

```bash
scripts/run_gpu_benchmarks.sh --largest-first --limit 10 --log-every 1
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
scripts/run_gpu_benchmarks.sh --sweep-preset focused
```

Broader sweep:

```bash
scripts/run_gpu_benchmarks.sh --sweep-preset broad
```

Explicit method-specific sweeps:

```bash
scripts/run_gpu_benchmarks.sh \
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

For more stable timings without losing diagnostic detail, use repeated runs.
With the default `--profile-only-first-repeat`, repeat `0` keeps internal
profiling while later repeats are cleaner timing samples. The summary exposes
both `wall_seconds` and `clean_wall_seconds`, so profiler overhead does not hide
the fastest useful variant:

```bash
scripts/run_gpu_benchmarks.sh --sweep-preset focused --repeats 3
```

## Reference Metrics

When matching `.ply` files are present in `Data/01-benchmark_surfaces`, each
method result also records approximate quality metrics against reference
vertices:

- generated point count to reference vertex count ratio;
- point-to-reference and reference-to-point nearest-distance summaries;
- symmetric mean nearest-distance score.

Distances are computed on deterministic subsamples so they stay practical on
the full dataset. Increase or reduce the sample size with:

```bash
scripts/run_gpu_benchmarks.sh --reference-sample-size 8192
```

Disable reference metrics only for pure runtime debugging:

```bash
scripts/run_gpu_benchmarks.sh --no-reference-metrics
```

## Profiling Detail

Internal section profiling is enabled by default. Each benchmark result includes
`internal_profile.top_functions` and `internal_profile.top_calls` with inclusive
wall time plus maximum input/output tensor sizes. This is designed to identify
hot helper functions and oversized intermediate structures.

For cleaner throughput-only measurements, disable internal profiling:

```bash
scripts/run_gpu_benchmarks.sh --no-profile-internals
```

For deeper CUDA/operator traces on a limited subset, enable PyTorch profiler:

```bash
scripts/run_gpu_benchmarks.sh --limit 20 --torch-profile-limit 20
```

Trace files are written under `tmp/gpu_benchmarks/traces/` and can be opened in
Chrome trace viewer or Perfetto. Records with `torch_profile` include top CUDA,
CPU, and memory ops, and are marked because `wall_seconds` includes profiler
overhead. Prefer the `clean_*` summary metrics for throughput comparisons.
Operator input-shape recording is disabled by default because the 0.0.1 GPU
run showed that shape-heavy traces can grow into multi-gigabyte files. Re-enable
it only for a small diagnostic subset:

```bash
scripts/run_gpu_benchmarks.sh --limit 5 --torch-profile-limit 5 \
  --torch-profile-record-shapes
```

To keep top-op summaries in JSONL without exporting trace files:

```bash
scripts/run_gpu_benchmarks.sh --limit 20 --torch-profile-limit 20 \
  --no-torch-profile-export-traces
```

For maximum structure detail on a small diagnostic run, include nested sample
summaries:

```bash
scripts/run_gpu_benchmarks.sh --limit 5 --profile-sample-structures
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
- `--interfaces`: independent output variants to benchmark. Default: `points`,
  which requests only point coordinates. Use comma-separated `features`,
  `normals`, and `adjacency` to add isolated feature, normal, and graph
  measurements, or `all` for all four variants.
- `--sweep-preset`: `none`, `focused`, or `broad`.
- `--repeats`: repeated runs per molecule/method/variant.
- `--largest-first`: run the largest PDBs first, using an atom-count estimate.
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
- `--torch-profile-limit`: number of method/variant runs to trace with PyTorch profiler.
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
- `SES_BENCH_DATA_DIR`: PDB dataset directory.
- `SES_BENCH_SURFACE_DIR`: PLY reference surface directory.
- `SES_BENCH_PROGRAM_VERSION`: semantic program version recorded in benchmark output. Default: `0.0.3`.
- `SES_BENCH_INTERFACES`: default interface modes passed by the wrapper. Default: `points`.
- `SES_BENCH_MOLECULE_ORDER`: default PDB order, for example `atom_count_desc`.
- `SES_BENCH_AUTO_RESUME=0`: disable wrapper auto-resume. Auto-resume is enabled by default.
- `SES_BENCH_CONTAINER`: run inside an already-running Docker container with `docker exec`.
- `SES_BENCH_CONTAINER_WORKDIR`: repository path inside that container. Default: `/workspace`.
- `SES_BENCH_EXEC_USER`: optional user passed to `docker exec --user`.
- `SES_BENCH_RUN_LOCAL=1`: run the benchmark directly in the current environment.
- `SES_BENCH_INSTALL_DEPS=1`: install Python dependencies before local or existing-container runs.
- `SES_BENCH_SKIP_BUILD=1`: reuse an already built image.
- `SES_BENCH_TORCH_INDEX_URL`: PyTorch wheel index, for example CUDA 12.4.
- `NVIDIA_VISIBLE_DEVICES`: restrict the container to selected GPU ids.
