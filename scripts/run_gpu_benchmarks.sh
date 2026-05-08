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
INSTALL_DEPS="${SES_BENCH_INSTALL_DEPS:-0}"
CONTAINER="${SES_BENCH_CONTAINER:-}"
CONTAINER_WORKDIR="${SES_BENCH_CONTAINER_WORKDIR:-/workspace}"
EXEC_USER="${SES_BENCH_EXEC_USER:-}"
RUN_LOCAL="${SES_BENCH_RUN_LOCAL:-0}"
PROGRAM_VERSION="${SES_BENCH_PROGRAM_VERSION:-0.0.1}"
DEFAULT_SWEEP_PRESET="${SES_BENCH_SWEEP_PRESET:-focused}"
DEFAULT_REPEATS="${SES_BENCH_REPEATS:-3}"
DEFAULT_TORCH_PROFILE_LIMIT="${SES_BENCH_TORCH_PROFILE_LIMIT:-100}"
DEFAULT_ARGS=(
  --program-version "${PROGRAM_VERSION}"
  --sweep-preset "${DEFAULT_SWEEP_PRESET}"
  --repeats "${DEFAULT_REPEATS}"
  --torch-profile-limit "${DEFAULT_TORCH_PROFILE_LIMIT}"
)

print_defaults() {
  echo "[ses-gpu-bench] Output: ${OUTPUT}"
  echo "[ses-gpu-bench] Wrapper defaults before CLI overrides: program_version=${PROGRAM_VERSION}, sweep=${DEFAULT_SWEEP_PRESET}, repeats=${DEFAULT_REPEATS}, torch_profile_limit=${DEFAULT_TORCH_PROFILE_LIMIT}"
}

missing_dependency_message() {
  local where="$1"
  echo "[ses-gpu-bench] Missing Python dependencies in ${where}." >&2
  echo "[ses-gpu-bench] Required modules include Bio, numpy, scipy, and torch." >&2
  echo "[ses-gpu-bench] Re-run with SES_BENCH_INSTALL_DEPS=1 to install requirements.txt there." >&2
}

check_python_deps() {
  python -c 'import importlib.util, sys; missing=[name for name in ("Bio", "numpy", "scipy", "torch") if importlib.util.find_spec(name) is None]; print(",".join(missing)); sys.exit(1 if missing else 0)'
}

install_local_deps() {
  echo "[ses-gpu-bench] Installing Python dependencies in the current environment"
  python -m pip install torch --index-url "${TORCH_INDEX_URL}"
  python -m pip install -r requirements.txt --extra-index-url "${TORCH_INDEX_URL}"
}

prepare_local_python() {
  if check_python_deps >/dev/null; then
    return
  fi
  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    install_local_deps
    check_python_deps >/dev/null
    return
  fi
  missing_dependency_message "the current environment"
  exit 1
}

check_container_python_deps() {
  docker exec -w "${CONTAINER_WORKDIR}" "${CONTAINER}" \
    python -c 'import importlib.util, sys; missing=[name for name in ("Bio", "numpy", "scipy", "torch") if importlib.util.find_spec(name) is None]; print(",".join(missing)); sys.exit(1 if missing else 0)'
}

install_container_deps() {
  echo "[ses-gpu-bench] Installing Python dependencies in ${CONTAINER}"
  docker exec -w "${CONTAINER_WORKDIR}" "${CONTAINER}" \
    python -m pip install torch --index-url "${TORCH_INDEX_URL}"
  docker exec -w "${CONTAINER_WORKDIR}" "${CONTAINER}" \
    python -m pip install -r requirements.txt --extra-index-url "${TORCH_INDEX_URL}"
}

prepare_container_python() {
  if check_container_python_deps >/dev/null; then
    return
  fi
  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    install_container_deps
    check_container_python_deps >/dev/null
    return
  fi
  missing_dependency_message "container ${CONTAINER}"
  exit 1
}

ensure_local_paths() {
  mkdir -p "$(dirname "${OUTPUT}")"
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[ses-gpu-bench] Data directory does not exist: ${DATA_DIR}" >&2
    exit 1
  fi
}

ensure_container_paths() {
  local output_dir
  output_dir="$(dirname "${OUTPUT}")"
  if ! docker exec -w "${CONTAINER_WORKDIR}" "${CONTAINER}" test -d "${DATA_DIR}" >/dev/null 2>&1; then
    echo "[ses-gpu-bench] Data directory does not exist in ${CONTAINER}: ${DATA_DIR}" >&2
    exit 1
  fi
  docker exec -w "${CONTAINER_WORKDIR}" "${CONTAINER}" mkdir -p "${output_dir}"
}

if [[ "${RUN_LOCAL}" == "1" ]]; then
  ensure_local_paths
  echo "[ses-gpu-bench] Running benchmark in the current environment"
  print_defaults
  prepare_local_python
  PYTHONUNBUFFERED=1 \
  SES_BENCH_OUTPUT="${OUTPUT}" \
  SES_BENCH_DATA_DIR="${DATA_DIR}" \
  SES_BENCH_SURFACE_DIR="${SURFACE_DIR}" \
  SES_BENCH_PROGRAM_VERSION="${PROGRAM_VERSION}" \
    python scripts/benchmark_ses_gpu.py "${DEFAULT_ARGS[@]}" "$@"
  exit 0
fi

if [[ -n "${CONTAINER}" ]]; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != "true" ]]; then
    echo "[ses-gpu-bench] Docker container is not running: ${CONTAINER}" >&2
    exit 1
  fi
  ensure_container_paths
  echo "[ses-gpu-bench] Running benchmark in existing container ${CONTAINER}"
  print_defaults
  prepare_container_python
  DOCKER_EXEC_ARGS=(
    exec
    -w "${CONTAINER_WORKDIR}"
    -e "PYTHONUNBUFFERED=1"
    -e "SES_BENCH_OUTPUT=${OUTPUT}"
    -e "SES_BENCH_DATA_DIR=${DATA_DIR}"
    -e "SES_BENCH_SURFACE_DIR=${SURFACE_DIR}"
    -e "SES_BENCH_PROGRAM_VERSION=${PROGRAM_VERSION}"
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}"
  )
  if [[ -n "${EXEC_USER}" ]]; then
    DOCKER_EXEC_ARGS+=(--user "${EXEC_USER}")
  fi
  docker "${DOCKER_EXEC_ARGS[@]}" \
    "${CONTAINER}" \
    python scripts/benchmark_ses_gpu.py "${DEFAULT_ARGS[@]}" "$@"
  exit 0
fi

ensure_local_paths

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
print_defaults

docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e "PYTHONUNBUFFERED=1" \
  -e "SES_BENCH_OUTPUT=${OUTPUT}" \
  -e "SES_BENCH_DATA_DIR=${DATA_DIR}" \
  -e "SES_BENCH_SURFACE_DIR=${SURFACE_DIR}" \
  -e "SES_BENCH_PROGRAM_VERSION=${PROGRAM_VERSION}" \
  -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}" \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  python scripts/benchmark_ses_gpu.py "${DEFAULT_ARGS[@]}" "$@"
