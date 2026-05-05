#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

IMAGE="${SES_BENCH_IMAGE:-protein-surfaces-gpu-bench:latest}"
DOCKERFILE="${SES_BENCH_DOCKERFILE:-Dockerfile.gpu}"
PYTHON_VERSION="${SES_BENCH_PYTHON_VERSION:-3.9}"
TORCH_INDEX_URL="${SES_BENCH_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
OUTPUT="${SES_BENCH_OUTPUT:-tmp/gpu_benchmarks/ses_gpu_benchmark_$(date -u +%Y%m%dT%H%M%SZ).jsonl}"
DATA_DIR="${SES_BENCH_DATA_DIR:-Data/01-benchmark_pdbs}"
SURFACE_DIR="${SES_BENCH_SURFACE_DIR:-Data/01-benchmark_surfaces}"
SKIP_BUILD="${SES_BENCH_SKIP_BUILD:-0}"
DEFAULT_SWEEP_PRESET="${SES_BENCH_SWEEP_PRESET:-focused}"
DEFAULT_REPEATS="${SES_BENCH_REPEATS:-3}"
DEFAULT_TORCH_PROFILE_LIMIT="${SES_BENCH_TORCH_PROFILE_LIMIT:-100}"
DEFAULT_ARGS=(
  --sweep-preset "${DEFAULT_SWEEP_PRESET}"
  --repeats "${DEFAULT_REPEATS}"
  --torch-profile-limit "${DEFAULT_TORCH_PROFILE_LIMIT}"
)

mkdir -p "$(dirname "${OUTPUT}")"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ses-gpu-bench] Data directory does not exist: ${DATA_DIR}" >&2
  exit 1
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "[ses-gpu-bench] Building ${IMAGE} from ${DOCKERFILE}"
  docker build \
    -f "${DOCKERFILE}" \
    -t "${IMAGE}" \
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
    --build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}" \
    .
fi

echo "[ses-gpu-bench] Running benchmark in ${IMAGE}"
echo "[ses-gpu-bench] Output: ${OUTPUT}"
echo "[ses-gpu-bench] Defaults: sweep=${DEFAULT_SWEEP_PRESET}, repeats=${DEFAULT_REPEATS}, torch_profile_limit=${DEFAULT_TORCH_PROFILE_LIMIT}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e "PYTHONUNBUFFERED=1" \
  -e "SES_BENCH_OUTPUT=${OUTPUT}" \
  -e "SES_BENCH_DATA_DIR=${DATA_DIR}" \
  -e "SES_BENCH_SURFACE_DIR=${SURFACE_DIR}" \
  -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}" \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  python scripts/benchmark_ses_gpu.py "${DEFAULT_ARGS[@]}" "$@"
