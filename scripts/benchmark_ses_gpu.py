#!/usr/bin/env python
"""Run SES method benchmarks on a PDB dataset.

The script is intentionally streaming-oriented: every method/molecule result is
written to JSONL immediately, so a long GPU run still leaves useful partial
data after an error, CUDA OOM, or manual interruption.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gc
import hashlib
import inspect
import itertools
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.profiler

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import read_pdb_tensors  # noqa: E402
import ses.analytic as ses_analytic  # noqa: E402
import ses.projection as ses_projection  # noqa: E402
import ses.sdf as ses_sdf  # noqa: E402
import ses.tiled_analytic as ses_tiled_analytic  # noqa: E402


SCHEMA_VERSION = 1
PROGRAM_VERSION = "0.0.1"
BENCHMARK_DRIVER_VERSION = "0.0.1"
METHOD_ORDER = ("analytic", "projected", "sdf", "tiled_analytic")
MB = 1024 * 1024
_ACTIVE_SECTION_PROFILER: Optional["SectionProfiler"] = None
_INSTALLED_PROFILE_WRAPPERS: List[Tuple[Any, str, Any]] = []
_PROFILE_MODULES = (ses_projection, ses_analytic, ses_sdf, ses_tiled_analytic)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _tensor_meta(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "bytes": _tensor_nbytes(tensor),
        "requires_grad": bool(tensor.requires_grad),
    }


def _summarize_object(
    value: Any,
    *,
    depth: int = 0,
    max_items: int = 6,
    max_depth: int = 2,
) -> Dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return _tensor_meta(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": type(value).__name__, "value": value}
    if dataclasses.is_dataclass(value):
        if depth >= max_depth:
            return {"kind": type(value).__name__}
        fields = {}
        for field in dataclasses.fields(value):
            try:
                field_value = getattr(value, field.name)
            except AttributeError:
                continue
            fields[field.name] = _summarize_object(
                field_value,
                depth=depth + 1,
                max_items=max_items,
                max_depth=max_depth,
            )
        return {"kind": type(value).__name__, "fields": fields}
    if isinstance(value, dict):
        if depth >= max_depth:
            return {"kind": "dict", "len": len(value)}
        items = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                break
            items[str(key)] = _summarize_object(
                item,
                depth=depth + 1,
                max_items=max_items,
                max_depth=max_depth,
            )
        return {"kind": "dict", "len": len(value), "items": items}
    if isinstance(value, (list, tuple)):
        if depth >= max_depth:
            return {"kind": type(value).__name__, "len": len(value)}
        return {
            "kind": type(value).__name__,
            "len": len(value),
            "items": [
                _summarize_object(
                    item,
                    depth=depth + 1,
                    max_items=max_items,
                    max_depth=max_depth,
                )
                for item in value[:max_items]
            ],
        }
    return {"kind": type(value).__name__}


def _tensor_tree_stats(value: Any, *, max_depth: int = 4) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "tensor_count": 0,
        "tensor_bytes": 0,
        "max_tensor_bytes": 0,
        "max_tensor": None,
    }

    def walk(item: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(item, torch.Tensor):
            meta = _tensor_meta(item)
            stats["tensor_count"] += 1
            stats["tensor_bytes"] += meta["bytes"]
            if meta["bytes"] > stats["max_tensor_bytes"]:
                stats["max_tensor_bytes"] = meta["bytes"]
                stats["max_tensor"] = meta
            return
        if dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                if hasattr(item, field.name):
                    walk(getattr(item, field.name), depth + 1)
            return
        if isinstance(item, dict):
            for nested in item.values():
                walk(nested, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                walk(nested, depth + 1)

    walk(value, 0)
    return stats


class SectionProfiler:
    """Inclusive function-level profiler for SES internals."""

    def __init__(
        self,
        *,
        device: torch.device,
        capture_shapes: bool,
        sample_structures: bool,
        record_cuda_events: bool,
        synchronize_cuda: bool,
        top_call_count: int,
        max_summary_items: int,
    ) -> None:
        self.device = device
        self.capture_shapes = capture_shapes
        self.sample_structures = sample_structures
        self.record_cuda_events = record_cuda_events
        self.synchronize_cuda = synchronize_cuda
        self.top_call_count = int(top_call_count)
        self.max_summary_items = int(max_summary_items)
        self.depth = 0
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.top_calls: List[Dict[str, Any]] = []

    def __enter__(self) -> "SectionProfiler":
        global _ACTIVE_SECTION_PROFILER
        self._previous = _ACTIVE_SECTION_PROFILER
        _ACTIVE_SECTION_PROFILER = self
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        global _ACTIVE_SECTION_PROFILER
        _ACTIVE_SECTION_PROFILER = self._previous

    def call(self, name: str, fn: Any, args: tuple, kwargs: dict) -> Any:
        input_stats = (
            _tensor_tree_stats((args, kwargs)) if self.capture_shapes else {}
        )
        input_summary = None
        if (
            self.capture_shapes
            and self.sample_structures
            and input_stats.get("tensor_bytes", 0)
        ):
            input_summary = _summarize_object(
                (args, kwargs),
                max_items=self.max_summary_items,
            )

        cuda_start = None
        cuda_end = None
        if (
            self.record_cuda_events
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            if self.synchronize_cuda:
                torch.cuda.synchronize(self.device)
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()

        start = time.perf_counter()
        self.depth += 1
        status = "ok"
        try:
            with torch.profiler.record_function(name):
                result = fn(*args, **kwargs)
            return result
        except Exception:
            status = "error"
            raise
        finally:
            self.depth -= 1
            if cuda_end is not None:
                cuda_end.record()
            if self.device.type == "cuda" and torch.cuda.is_available():
                if self.synchronize_cuda:
                    torch.cuda.synchronize(self.device)
            wall_seconds = time.perf_counter() - start
            cuda_event_ms = None
            if cuda_start is not None and cuda_end is not None:
                try:
                    torch.cuda.synchronize(self.device)
                    cuda_event_ms = float(cuda_start.elapsed_time(cuda_end))
                except RuntimeError:
                    cuda_event_ms = None

            output_stats = (
                _tensor_tree_stats(locals().get("result")) if self.capture_shapes else {}
            )
            output_summary = None
            if (
                self.capture_shapes
                and self.sample_structures
                and output_stats.get("tensor_bytes", 0)
            ):
                output_summary = _summarize_object(
                    locals().get("result"),
                    max_items=self.max_summary_items,
                )
            self._record(
                name=name,
                status=status,
                depth=self.depth,
                wall_seconds=wall_seconds,
                cuda_event_ms=cuda_event_ms,
                input_stats=input_stats,
                output_stats=output_stats,
                input_summary=input_summary,
                output_summary=output_summary,
            )

    def _record(
        self,
        *,
        name: str,
        status: str,
        depth: int,
        wall_seconds: float,
        cuda_event_ms: Optional[float],
        input_stats: Dict[str, Any],
        output_stats: Dict[str, Any],
        input_summary: Optional[Dict[str, Any]],
        output_summary: Optional[Dict[str, Any]],
    ) -> None:
        stats = self.stats.setdefault(
            name,
            {
                "calls": 0,
                "errors": 0,
                "wall_seconds_total": 0.0,
                "wall_seconds_max": 0.0,
                "cuda_event_ms_total": 0.0,
                "cuda_event_ms_max": 0.0,
                "cuda_event_calls": 0,
                "max_input_tensor_bytes": 0,
                "max_output_tensor_bytes": 0,
                "max_input_tensor": None,
                "max_output_tensor": None,
                "sample_input": None,
                "sample_output": None,
            },
        )
        stats["calls"] += 1
        if status != "ok":
            stats["errors"] += 1
        stats["wall_seconds_total"] += wall_seconds
        stats["wall_seconds_max"] = max(stats["wall_seconds_max"], wall_seconds)
        if cuda_event_ms is not None:
            stats["cuda_event_calls"] += 1
            stats["cuda_event_ms_total"] += cuda_event_ms
            stats["cuda_event_ms_max"] = max(stats["cuda_event_ms_max"], cuda_event_ms)

        input_bytes = int(input_stats.get("tensor_bytes", 0))
        output_bytes = int(output_stats.get("tensor_bytes", 0))
        if input_bytes > stats["max_input_tensor_bytes"]:
            stats["max_input_tensor_bytes"] = input_bytes
            stats["max_input_tensor"] = input_stats.get("max_tensor")
            stats["sample_input"] = input_summary
        if output_bytes > stats["max_output_tensor_bytes"]:
            stats["max_output_tensor_bytes"] = output_bytes
            stats["max_output_tensor"] = output_stats.get("max_tensor")
            stats["sample_output"] = output_summary

        call_record = {
            "name": name,
            "status": status,
            "depth": depth,
            "wall_seconds": wall_seconds,
            "cuda_event_ms": cuda_event_ms,
            "input_tensor_bytes": input_bytes,
            "output_tensor_bytes": output_bytes,
            "max_input_tensor": input_stats.get("max_tensor"),
            "max_output_tensor": output_stats.get("max_tensor"),
        }
        self.top_calls.append(call_record)
        self.top_calls.sort(key=lambda item: item["wall_seconds"], reverse=True)
        del self.top_calls[self.top_call_count :]

    def summary(self, *, limit: int) -> Dict[str, Any]:
        functions = []
        for name, stats in self.stats.items():
            calls = max(1, int(stats["calls"]))
            item = dict(stats)
            item["name"] = name
            item["wall_seconds_mean"] = stats["wall_seconds_total"] / calls
            if stats["cuda_event_calls"]:
                item["cuda_event_ms_mean"] = (
                    stats["cuda_event_ms_total"] / stats["cuda_event_calls"]
                )
            else:
                item["cuda_event_ms_mean"] = None
            functions.append(item)
        functions.sort(key=lambda item: item["wall_seconds_total"], reverse=True)
        return {
            "function_count": len(functions),
            "top_functions": functions[:limit],
            "top_calls": self.top_calls[: self.top_call_count],
        }


def _make_profile_wrapper(module_name: str, public_name: str, fn: Any) -> Any:
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        profiler = _ACTIVE_SECTION_PROFILER
        if profiler is None:
            return fn(*args, **kwargs)
        return profiler.call(f"{module_name}.{public_name}", fn, args, kwargs)

    wrapped.__ses_benchmark_wrapped__ = True
    return wrapped


def _install_internal_profile_wrappers() -> None:
    if _INSTALLED_PROFILE_WRAPPERS:
        return
    for module in _PROFILE_MODULES:
        for name, value in list(vars(module).items()):
            if name.startswith("__") or not inspect.isfunction(value):
                continue
            if getattr(value, "__ses_benchmark_wrapped__", False):
                continue
            source_module = getattr(value, "__module__", "")
            if not source_module.startswith("ses."):
                continue
            wrapped = _make_profile_wrapper(module.__name__, name, value)
            setattr(module, name, wrapped)
            _INSTALLED_PROFILE_WRAPPERS.append((module, name, value))


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _default_output_path() -> str:
    program_version = os.environ.get("SES_BENCH_PROGRAM_VERSION", PROGRAM_VERSION)
    return f"tmp/gpu_benchmarks/ses_gpu_benchmark_{program_version}.jsonl"


def _parse_methods(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(METHOD_ORDER)
    methods: List[str] = []
    for item in value.split(","):
        method = item.strip()
        if not method:
            continue
        if method not in METHOD_ORDER:
            allowed = ", ".join(METHOD_ORDER)
            raise argparse.ArgumentTypeError(
                f"unknown method {method!r}; expected one of: {allowed}"
            )
        if method not in methods:
            methods.append(method)
    if not methods:
        raise argparse.ArgumentTypeError("at least one method is required")
    return methods


def _parse_csv_values(value: Optional[str], caster: Any) -> Optional[List[Any]]:
    if value is None or value == "":
        return None
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "none":
            values.append(None)
        elif item == "auto":
            values.append("auto")
        else:
            values.append(caster(item))
    return values or None


def _parse_csv_floats(value: Optional[str]) -> Optional[List[Optional[float]]]:
    return _parse_csv_values(value, float)


def _parse_csv_ints(value: Optional[str]) -> Optional[List[Optional[int]]]:
    return _parse_csv_values(value, int)


def _parse_csv_float_or_auto(value: Optional[str]) -> Optional[List[Any]]:
    return _parse_csv_values(value, float)


def _parse_float_or_auto(value: str) -> Any:
    if value == "auto":
        return "auto"
    return float(value)


def _parse_optional_int(value: str) -> Optional[int]:
    if value.lower() == "none":
        return None
    return int(value)


def _parse_program_version(value: str) -> str:
    parts = value.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(
            "program version must use semantic form X.Y.Z, for example 0.0.1"
        )
    return value


def _dtype_from_name(name: str) -> torch.dtype:
    dtype_by_name = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return dtype_by_name[name]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "dtype must be one of: float32, float64"
        ) from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    return str(value)


def _stable_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _current_rss_mb() -> Optional[float]:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    try:
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / MB
    except (OSError, IndexError, ValueError):
        return None


def _peak_rss_mb() -> Optional[float]:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / MB
    return peak / 1024


def _run_command(args: Sequence[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            list(args),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_metadata() -> Dict[str, Any]:
    return {
        "commit": _run_command(["git", "rev-parse", "HEAD"]),
        "short_status": _run_command(["git", "status", "--short"]),
    }


def _nvidia_smi() -> Optional[str]:
    return _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )


def _environment(device: torch.device) -> Dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": [props.major, props.minor],
                    "total_memory_mb": props.total_memory / MB,
                    "multi_processor_count": props.multi_processor_count,
                }
            )

    return {
        "created_at_utc": _utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "selected_device": str(device),
        "cuda_devices": cuda_devices,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "nvidia_smi": _nvidia_smi(),
        "git": _git_metadata(),
    }


def _default_surface_dir(data_dir: str) -> str:
    data_path = Path(data_dir)
    if data_path.name == "01-benchmark_pdbs":
        return str(data_path.with_name("01-benchmark_surfaces"))
    return str(data_path.parent / "01-benchmark_surfaces")


def _method_params(
    args: argparse.Namespace,
    method: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if method == "analytic":
        params = {
            "point_area": args.point_area,
            "oversample_factor": args.analytic_oversample_factor,
            "atom_filter_samples": args.analytic_atom_filter_samples,
            "pair_filter_samples": args.analytic_pair_filter_samples,
            "probe_density_scale": args.analytic_probe_density_scale,
            "max_probe_support_atoms": args.analytic_max_probe_support_atoms,
            "support_tolerance": args.analytic_support_tolerance,
            "dedup_tolerance": args.analytic_dedup_tolerance,
            "max_probe_triples": args.analytic_max_probe_triples,
            "grid_spacing": args.grid_spacing,
            "max_grid_points": args.analytic_max_grid_points,
        }
    elif method == "projected":
        params = {
            "m": args.projected_m,
        }
    elif method == "sdf":
        params = {
            "m": args.sdf_m,
            "smoothness": args.sdf_smoothness,
            "iterations": args.sdf_iterations,
            "level_tolerance": args.sdf_level_tolerance,
            "subsample_spacing": args.sdf_subsample_spacing,
            "feature_threshold": args.sdf_feature_threshold,
            "grid_spacing": args.grid_spacing,
            "max_grid_points": args.sdf_max_grid_points,
            "pairwise_element_budget": args.pairwise_element_budget,
        }
    elif method == "tiled_analytic":
        params = {
            "point_area": args.tiled_point_area or args.point_area,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "atom_density_scale": args.tiled_atom_density_scale,
            "pair_density_scale": args.tiled_pair_density_scale,
            "probe_density_scale": args.tiled_probe_density_scale,
            "dedup_tolerance": args.tiled_dedup_tolerance,
            "exact_accessibility": args.tiled_exact_accessibility,
            "grid_spacing": args.grid_spacing,
            "max_grid_points": args.tiled_max_grid_points,
            "max_probe_triples": args.tiled_max_probe_triples,
            "pairwise_element_budget": args.pairwise_element_budget,
        }
    else:
        raise ValueError(f"unknown method: {method}")
    if overrides:
        params.update(overrides)
    return params


def _common_hash_payload(
    args: argparse.Namespace,
    method: str,
    method_params: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "device": args.device,
        "dtype": args.dtype,
        "probe_radius": args.probe_radius,
        "loader": {
            "center": args.center,
            "include_hetatm": args.include_hetatm,
            "include_hydrogens": args.include_hydrogens,
            "unknown_elements": args.unknown_elements,
        },
        "method_params": method_params,
    }


def _variant_name(overrides: Dict[str, Any]) -> str:
    if not overrides:
        return "default"
    parts = []
    for key, value in sorted(overrides.items()):
        text = str(value).replace(".", "p").replace("/", "_")
        parts.append(f"{key}={text}")
    return "__".join(parts)


def _dedupe_variants(variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for variant in variants:
        key = (variant["method"], json.dumps(variant["method_params"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def _single_axis_overrides(
    base: Dict[str, Any],
    values_by_key: Dict[str, Sequence[Any]],
) -> List[Dict[str, Any]]:
    overrides: List[Dict[str, Any]] = []
    for key, values in values_by_key.items():
        for value in values:
            if base.get(key) == value:
                continue
            overrides.append({key: value})
    return overrides


def _cartesian_overrides(values_by_key: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(values_by_key)
    values = [list(values_by_key[key]) for key in keys]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def _explicit_values(args: argparse.Namespace, method: str) -> Dict[str, Sequence[Any]]:
    if method == "analytic":
        values = {
            "point_area": args.analytic_point_area_values,
            "oversample_factor": args.analytic_oversample_factor_values,
            "atom_filter_samples": args.analytic_atom_filter_samples_values,
            "pair_filter_samples": args.analytic_pair_filter_samples_values,
            "probe_density_scale": args.analytic_probe_density_scale_values,
        }
    elif method == "projected":
        values = {"m": args.projected_m_values}
    elif method == "sdf":
        values = {
            "m": args.sdf_m_values,
            "smoothness": args.sdf_smoothness_values,
            "iterations": args.sdf_iterations_values,
            "subsample_spacing": args.sdf_subsample_spacing_values,
        }
    elif method == "tiled_analytic":
        values = {
            "point_area": args.tiled_point_area_values,
            "tile_size": args.tile_size_values,
            "tile_overlap": args.tile_overlap_values,
            "atom_density_scale": args.tiled_atom_density_scale_values,
            "pair_density_scale": args.tiled_pair_density_scale_values,
            "probe_density_scale": args.tiled_probe_density_scale_values,
            "dedup_tolerance": args.tiled_dedup_tolerance_values,
        }
    else:
        raise ValueError(f"unknown method: {method}")
    return {key: value for key, value in values.items() if value is not None}


def _preset_values(method: str, base: Dict[str, Any], preset: str) -> Dict[str, Sequence[Any]]:
    if preset == "none":
        return {}
    if method == "analytic":
        focused = {
            "point_area": [0.25, base["point_area"], 1.0],
            "oversample_factor": [1.0, base["oversample_factor"], 1.5],
            "atom_filter_samples": [32, base["atom_filter_samples"], 128],
            "pair_filter_samples": [8, base["pair_filter_samples"], 24],
        }
        broad = {
            "point_area": [0.2, 0.35, 0.5, 0.75, 1.0],
            "oversample_factor": [1.0, 1.25, 1.5],
            "atom_filter_samples": [32, 64, 128],
            "pair_filter_samples": [8, 12, 24],
        }
    elif method == "projected":
        focused = {"m": [96, 160, base["m"], 320]}
        broad = {"m": [64, 96, 128, 160, 230, 320, 448]}
    elif method == "sdf":
        focused = {
            "m": [16, base["m"], 64],
            "smoothness": [0.15, base["smoothness"], 0.3],
            "iterations": [4, base["iterations"], 8],
        }
        broad = {
            "m": [12, 16, 24, 34, 48, 64],
            "smoothness": [0.1, 0.15, 0.2, 0.3, 0.45],
            "iterations": [3, 4, 6, 8],
        }
    elif method == "tiled_analytic":
        focused = {
            "tile_size": ["auto", 24.0, 48.0, 64.0],
            "tile_overlap": ["auto", 3.0, 4.0, 6.0],
            "atom_density_scale": [4.0, base["atom_density_scale"], 10.0],
            "pair_density_scale": [0.0, 0.75, 1.55],
            "probe_density_scale": [0.0, 0.75, 1.55],
        }
        broad = {
            "tile_size": ["auto", 24.0, 32.0, 48.0, 64.0],
            "tile_overlap": ["auto", 3.0, 4.0, 6.0],
            "atom_density_scale": [3.5, 5.0, 7.0, 9.0, 12.0],
            "pair_density_scale": [0.0, 0.75, 1.55],
            "probe_density_scale": [0.0, 0.75, 1.55],
        }
    else:
        raise ValueError(f"unknown method: {method}")
    return broad if preset == "broad" else focused


def _sweep_overrides(
    args: argparse.Namespace,
    method: str,
    base: Dict[str, Any],
) -> List[Dict[str, Any]]:
    explicit = _explicit_values(args, method)
    if explicit:
        values = {**_preset_values(method, base, args.sweep_preset), **explicit}
    else:
        values = _preset_values(method, base, args.sweep_preset)
    if not values:
        return []
    if args.sweep_cartesian:
        overrides = _cartesian_overrides(values)
    else:
        overrides = _single_axis_overrides(base, values)
    max_variants = args.sweep_max_variants_per_method
    if max_variants is not None:
        overrides = overrides[: max(0, int(max_variants) - 1)]
    return overrides


def _build_variants(
    args: argparse.Namespace,
    methods: Sequence[str],
) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for method in methods:
        default_params = _method_params(args, method)
        method_variants = [
            {
                "method": method,
                "variant_name": "default",
                "overrides": {},
                "method_params": default_params,
            }
        ]
        for overrides in _sweep_overrides(args, method, default_params):
            params = _method_params(args, method, overrides)
            method_variants.append(
                {
                    "method": method,
                    "variant_name": _variant_name(overrides),
                    "overrides": overrides,
                    "method_params": params,
                }
            )
        for variant in _dedupe_variants(method_variants):
            variant["hash"] = _stable_hash(
                _common_hash_payload(args, method, variant["method_params"])
            )
            variants.append(variant)
    return variants


def _all_parameters(
    args: argparse.Namespace,
    variants: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "program_version": args.program_version,
        "benchmark_driver_version": BENCHMARK_DRIVER_VERSION,
        "probe_radius": args.probe_radius,
        "dtype": args.dtype,
        "device": args.device,
        "molecule_order": args.molecule_order,
        "loader": {
            "center": args.center,
            "include_hetatm": args.include_hetatm,
            "include_hydrogens": args.include_hydrogens,
            "unknown_elements": args.unknown_elements,
        },
        "sweep": {
            "preset": args.sweep_preset,
            "cartesian": args.sweep_cartesian,
            "max_variants_per_method": args.sweep_max_variants_per_method,
            "variant_count": len(variants),
            "repeats": args.repeats,
            "profile_only_first_repeat": args.profile_only_first_repeat,
        },
        "reference_metrics": {
            "enabled": args.reference_metrics,
            "surface_dir": args.surface_dir,
            "sample_size": args.reference_sample_size,
            "distance_budget": args.reference_distance_budget,
        },
        "variants": [
            {
                "method": variant["method"],
                "variant_name": variant["variant_name"],
                "hash": variant["hash"],
                "params": variant["method_params"],
            }
            for variant in variants
        ],
        "profiling": {
            "profile_internals": args.profile_internals,
            "profile_internals_every": args.profile_internals_every,
            "profile_internals_limit_runs": args.profile_internals_limit_runs,
            "profile_shapes": args.profile_shapes,
            "profile_sample_structures": args.profile_sample_structures,
            "profile_record_cuda_events": args.profile_record_cuda_events,
            "profile_synchronize_cuda": args.profile_synchronize_cuda,
            "torch_profile_limit": args.torch_profile_limit,
            "torch_profile_every": args.torch_profile_every,
            "torch_profile_dir": args.torch_profile_dir,
        },
        "calibration_note": (
            "Defaults follow the existing CPU calibration for point_area=0.5: "
            "projected m=230, SDF m=34, and tiled analytic atom density scale=7 "
            "with pair/probe tiled patches disabled for speed. Sweep presets and "
            "*-values flags can vary these settings for throughput/quality tuning."
        ),
    }


def _pdb_atom_is_hydrogen(line: str) -> bool:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element == "H"
    atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
    atom_name = atom_name.lstrip("0123456789")
    return atom_name.startswith("H")


def _estimate_pdb_atom_count(path: Path, args: argparse.Namespace) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("ATOM"):
                    pass
                elif line.startswith("HETATM"):
                    if not args.include_hetatm:
                        continue
                else:
                    continue
                if not args.include_hydrogens and _pdb_atom_is_hydrogen(line):
                    continue
                count += 1
    except OSError:
        return -1
    return count


def _select_pdbs(args: argparse.Namespace) -> List[Path]:
    data_dir = Path(args.data_dir)
    pdbs = sorted(data_dir.glob("*.pdb"))
    if args.molecule_order == "atom_count_desc":
        pdbs.sort(key=lambda path: (-_estimate_pdb_atom_count(path, args), path.name))
    elif args.molecule_order == "atom_count_asc":
        pdbs.sort(key=lambda path: (_estimate_pdb_atom_count(path, args), path.name))
    elif args.molecule_order == "file_size_desc":
        pdbs.sort(key=lambda path: (-path.stat().st_size, path.name))
    elif args.molecule_order == "file_size_asc":
        pdbs.sort(key=lambda path: (path.stat().st_size, path.name))
    elif args.molecule_order != "name":
        raise ValueError(f"unknown molecule order: {args.molecule_order}")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    if args.shard_count > 1:
        pdbs = [
            path
            for index, path in enumerate(pdbs)
            if index % args.shard_count == args.shard_index
        ]
    if args.offset:
        pdbs = pdbs[args.offset :]
    if args.limit is not None:
        pdbs = pdbs[: args.limit]
    return pdbs


def _load_existing_results(
    output_path: Path,
    variants: Sequence[Dict[str, Any]],
    program_version: str,
) -> Tuple[set, List[Dict[str, Any]]]:
    keys = set()
    results: List[Dict[str, Any]] = []
    known_hashes = {
        (variant["method"], variant["hash"])
        for variant in variants
    }
    if not output_path.exists():
        return keys, results

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "benchmark_result":
                continue
            method = record.get("method")
            method_hash = record.get("method_parameter_hash")
            if (method, method_hash) not in known_hashes:
                continue
            if record.get("program_version") != program_version:
                continue
            key = (
                record.get("pdb_id"),
                method,
                method_hash,
                int(record.get("repeat_index", 0)),
            )
            keys.add(key)
            results.append(record)
    return keys, results


class JsonlWriter:
    def __init__(self, path: Path, append: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a" if append else "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        json.dump(record, self._handle, default=_json_default, sort_keys=True)
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _base_result(
    *,
    args: argparse.Namespace,
    run_id: str,
    pdb_path: Path,
    variant: Dict[str, Any],
    repeat_index: int,
    atom_count: Optional[int],
    load_seconds: Optional[float],
) -> Dict[str, Any]:
    method = variant["method"]
    return {
        "event": "benchmark_result",
        "schema_version": SCHEMA_VERSION,
        "program_version": args.program_version,
        "benchmark_driver_version": BENCHMARK_DRIVER_VERSION,
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "pdb_id": pdb_path.stem,
        "path": str(pdb_path),
        "method": method,
        "variant_name": variant["variant_name"],
        "method_parameter_hash": variant["hash"],
        "method_params": variant["method_params"],
        "method_overrides": variant["overrides"],
        "repeat_index": int(repeat_index),
        "atom_count": atom_count,
        "load_seconds": load_seconds,
    }


def _call_method(
    method: str,
    coords: torch.Tensor,
    radii: torch.Tensor,
    args: argparse.Namespace,
    method_params: Dict[str, Any],
) -> torch.Tensor:
    if method == "analytic":
        return ses_analytic.sample_analytic_points(
            coords,
            radii,
            args.probe_radius,
            point_area=method_params["point_area"],
            oversample_factor=method_params["oversample_factor"],
            probe_density_scale=method_params["probe_density_scale"],
            include_atom_features=False,
            atom_filter_samples=method_params["atom_filter_samples"],
            pair_filter_samples=method_params["pair_filter_samples"],
            max_probe_support_atoms=method_params["max_probe_support_atoms"],
            support_tolerance=method_params["support_tolerance"],
            dedup_tolerance=method_params["dedup_tolerance"],
            max_probe_triples=method_params["max_probe_triples"],
            grid_spacing=method_params["grid_spacing"],
            max_grid_points=method_params["max_grid_points"],
        )
    if method == "projected":
        return ses_projection.sample_projected_points(
            coords,
            radii,
            method_params["m"],
            args.probe_radius,
            include_atom_features=False,
        )
    if method == "sdf":
        return ses_sdf.sample_sdf_points(
            coords,
            radii,
            method_params["m"],
            args.probe_radius,
            smoothness=method_params["smoothness"],
            iterations=method_params["iterations"],
            level_tolerance=method_params["level_tolerance"],
            subsample_spacing=method_params["subsample_spacing"],
            feature_threshold=method_params["feature_threshold"],
            include_atom_features=False,
            grid_spacing=method_params["grid_spacing"],
            max_grid_points=method_params["max_grid_points"],
            pairwise_element_budget=method_params["pairwise_element_budget"],
        )
    if method == "tiled_analytic":
        return ses_tiled_analytic.sample_tiled_analytic_points(
            coords,
            radii,
            args.probe_radius,
            point_area=method_params["point_area"],
            tile_size=method_params["tile_size"],
            tile_overlap=method_params["tile_overlap"],
            atom_density_scale=method_params["atom_density_scale"],
            pair_density_scale=method_params["pair_density_scale"],
            probe_density_scale=method_params["probe_density_scale"],
            dedup_tolerance=method_params["dedup_tolerance"],
            exact_accessibility=method_params["exact_accessibility"],
            include_atom_features=False,
            grid_spacing=method_params["grid_spacing"],
            max_grid_points=method_params["max_grid_points"],
            max_probe_triples=method_params["max_probe_triples"],
            pairwise_element_budget=method_params["pairwise_element_budget"],
        )
    raise ValueError(f"unknown method: {method}")


def _cuda_memory(device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_allocated_mb": None,
            "gpu_reserved_mb": None,
            "gpu_peak_allocated_mb": None,
            "gpu_peak_reserved_mb": None,
            "gpu_free_mb": None,
            "gpu_total_mb": None,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated(device) / MB,
        "gpu_reserved_mb": torch.cuda.memory_reserved(device) / MB,
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / MB,
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / MB,
        "gpu_free_mb": free_bytes / MB,
        "gpu_total_mb": total_bytes / MB,
    }


def _safe_cuda_synchronize(device: torch.device) -> Optional[str]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        return str(exc)
    return None


def _read_ply_vertices(path: Path) -> torch.Tensor:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")
        vertex_count = None
        fmt = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} ended before PLY header finished")
            stripped = line.strip()
            if stripped.startswith("format "):
                fmt = stripped.split()[1]
            elif stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[2])
            elif stripped == "end_header":
                break
        if fmt != "ascii":
            raise ValueError(f"Only ascii PLY files are supported, got {fmt!r} in {path}")
        if vertex_count is None:
            raise ValueError(f"{path} does not declare vertex count")
        vertices = []
        for _ in range(vertex_count):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} ended before all vertices were read")
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed PLY vertex row in {path}")
            vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return torch.tensor(vertices, dtype=torch.float32)


def _load_reference_vertices(
    pdb_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    if not args.reference_metrics:
        return None, {"enabled": False}
    surface_path = Path(args.surface_dir) / f"{pdb_path.stem}.ply"
    if not surface_path.exists():
        return None, {
            "enabled": True,
            "status": "missing_surface",
            "path": str(surface_path),
        }
    try:
        start = time.perf_counter()
        vertices_cpu = _read_ply_vertices(surface_path)
        vertices = vertices_cpu.to(device=device, dtype=dtype)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - benchmark should continue.
        return None, {
            "enabled": True,
            "status": "load_error",
            "path": str(surface_path),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    with torch.no_grad():
        bbox_min = vertices.min(dim=0).values.detach().cpu().tolist()
        bbox_max = vertices.max(dim=0).values.detach().cpu().tolist()
        bbox_span = (
            vertices.max(dim=0).values - vertices.min(dim=0).values
        ).detach().cpu().tolist()
    return vertices, {
        "enabled": True,
        "status": "ok",
        "path": str(surface_path),
        "load_seconds": load_seconds,
        "vertex_count": int(vertices.shape[0]),
        "vertices": _tensor_meta(vertices),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_span": bbox_span,
    }


def _deterministic_subsample(points: torch.Tensor, max_count: int) -> torch.Tensor:
    if max_count <= 0 or points.shape[0] <= max_count:
        return points
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=max_count,
        device=points.device,
    ).round().to(torch.long)
    return points[indices]


def _nearest_distances(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    distance_budget: int,
) -> torch.Tensor:
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return torch.empty((0,), dtype=query.dtype, device=query.device)
    rows = max(1, int(distance_budget) // max(1, int(reference.shape[0])))
    distances = []
    for start in range(0, query.shape[0], rows):
        stop = min(start + rows, query.shape[0])
        block = torch.cdist(query[start:stop], reference)
        distances.append(block.min(dim=1).values)
    return torch.cat(distances, dim=0)


def _distance_summary(distances: torch.Tensor) -> Dict[str, Optional[float]]:
    if distances.numel() == 0:
        return {"mean": None, "median": None, "p90": None, "p95": None, "max": None}
    values = distances.detach().float().cpu()
    quantiles = torch.quantile(
        values,
        torch.tensor([0.5, 0.9, 0.95], dtype=values.dtype),
    )
    return {
        "mean": float(values.mean().item()),
        "median": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "max": float(values.max().item()),
    }


def _reference_metrics(
    points: torch.Tensor,
    reference_vertices: Optional[torch.Tensor],
    reference_info: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    if not args.reference_metrics:
        return {"enabled": False}
    if reference_vertices is None:
        return dict(reference_info)
    start = time.perf_counter()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        sampled_points = _deterministic_subsample(points.detach(), args.reference_sample_size)
        sampled_reference = _deterministic_subsample(
            reference_vertices.detach(),
            args.reference_sample_size,
        )
        point_to_reference = _nearest_distances(
            sampled_points,
            sampled_reference,
            distance_budget=args.reference_distance_budget,
        )
        reference_to_point = _nearest_distances(
            sampled_reference,
            sampled_points,
            distance_budget=args.reference_distance_budget,
        )
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    quality_memory = _cuda_memory(device)
    p2r = _distance_summary(point_to_reference)
    r2p = _distance_summary(reference_to_point)
    symmetric_mean = None
    if p2r["mean"] is not None and r2p["mean"] is not None:
        symmetric_mean = 0.5 * (p2r["mean"] + r2p["mean"])
    return {
        "enabled": True,
        "status": "ok",
        "path": reference_info.get("path"),
        "seconds": seconds,
        "reference_vertex_count": int(reference_vertices.shape[0]),
        "sampled_point_count": int(sampled_points.shape[0]),
        "sampled_reference_count": int(sampled_reference.shape[0]),
        "point_count_to_reference_ratio": (
            float(points.shape[0]) / float(reference_vertices.shape[0])
            if reference_vertices.shape[0]
            else None
        ),
        "point_to_reference_distance": p2r,
        "reference_to_point_distance": r2p,
        "symmetric_mean_distance": symmetric_mean,
        "gpu_peak_allocated_mb": quality_memory.get("gpu_peak_allocated_mb"),
        "gpu_peak_reserved_mb": quality_memory.get("gpu_peak_reserved_mb"),
    }


def _molecule_stats(coords: torch.Tensor, radii: torch.Tensor) -> Dict[str, Any]:
    with torch.no_grad():
        bbox_min = coords.min(dim=0).values.detach().cpu().tolist()
        bbox_max = coords.max(dim=0).values.detach().cpu().tolist()
        bbox_span = (coords.max(dim=0).values - coords.min(dim=0).values).detach().cpu().tolist()
        return {
            "atom_count": int(coords.shape[0]),
            "coords": _tensor_meta(coords),
            "radii": _tensor_meta(radii),
            "input_tensor_bytes": _tensor_nbytes(coords) + _tensor_nbytes(radii),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_span": bbox_span,
            "bbox_volume": float(bbox_span[0] * bbox_span[1] * bbox_span[2]),
            "radii_min": float(radii.min().item()),
            "radii_mean": float(radii.mean().item()),
            "radii_max": float(radii.max().item()),
        }


def _event_value(event: Any, names: Sequence[str]) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _torch_profiler_top_ops(
    profiler: torch.profiler.profile,
    *,
    limit: int,
) -> Dict[str, List[Dict[str, Any]]]:
    try:
        averages = list(profiler.key_averages(group_by_input_shape=True))
    except Exception:
        return {"cuda_time": [], "cpu_time": [], "memory": []}

    def serialize(event: Any) -> Dict[str, Any]:
        return {
            "key": getattr(event, "key", ""),
            "count": int(getattr(event, "count", 0)),
            "cpu_time_total_us": _event_value(event, ("cpu_time_total",)),
            "self_cpu_time_total_us": _event_value(event, ("self_cpu_time_total",)),
            "cuda_time_total_us": _event_value(event, ("cuda_time_total", "device_time_total")),
            "self_cuda_time_total_us": _event_value(
                event,
                ("self_cuda_time_total", "self_device_time_total"),
            ),
            "cpu_memory_usage_bytes": int(getattr(event, "cpu_memory_usage", 0) or 0),
            "cuda_memory_usage_bytes": int(
                getattr(event, "cuda_memory_usage", None)
                or getattr(event, "device_memory_usage", 0)
                or 0
            ),
            "input_shapes": getattr(event, "input_shapes", None),
        }

    return {
        "cuda_time": [
            serialize(event)
            for event in sorted(
                averages,
                key=lambda item: _event_value(
                    item,
                    ("cuda_time_total", "device_time_total"),
                ),
                reverse=True,
            )[:limit]
        ],
        "cpu_time": [
            serialize(event)
            for event in sorted(
                averages,
                key=lambda item: _event_value(item, ("cpu_time_total",)),
                reverse=True,
            )[:limit]
        ],
        "memory": [
            serialize(event)
            for event in sorted(
                averages,
                key=lambda item: abs(
                    int(
                        getattr(item, "cuda_memory_usage", None)
                        or getattr(item, "device_memory_usage", 0)
                        or 0
                    )
                ),
                reverse=True,
            )[:limit]
        ],
    }


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value)


def _torch_profile_path(
    args: argparse.Namespace,
    *,
    run_id: str,
    pdb_id: str,
    variant: Dict[str, Any],
    run_index: int,
) -> Path:
    profile_dir = Path(args.torch_profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    name = "__".join(
        [
            run_id,
            f"{run_index:06d}",
            pdb_id,
            variant["method"],
            _safe_filename(variant["variant_name"]),
            variant["hash"],
        ]
    )
    return profile_dir / f"{name}.trace.json"


def _should_profile_internals(args: argparse.Namespace, run_index: int) -> bool:
    if not args.profile_internals:
        return False
    if args.profile_internals_limit_runs is not None and run_index > args.profile_internals_limit_runs:
        return False
    return args.profile_internals_every > 0 and run_index % args.profile_internals_every == 0


def _should_profile_run(args: argparse.Namespace, run_index: int, repeat_index: int) -> bool:
    if args.profile_only_first_repeat and repeat_index != 0:
        return False
    return _should_profile_internals(args, run_index)


def _should_torch_profile(
    args: argparse.Namespace,
    run_index: int,
    profiles_written: int,
    repeat_index: int,
) -> bool:
    if args.profile_only_first_repeat and repeat_index != 0:
        return False
    if args.torch_profile_limit and profiles_written < args.torch_profile_limit:
        return True
    return args.torch_profile_every > 0 and run_index % args.torch_profile_every == 0


def _run_one_method(
    *,
    args: argparse.Namespace,
    run_id: str,
    pdb_path: Path,
    variant: Dict[str, Any],
    repeat_index: int,
    coords: torch.Tensor,
    radii: torch.Tensor,
    atom_count: int,
    load_seconds: float,
    device: torch.device,
    molecule_stats: Dict[str, Any],
    reference_vertices: Optional[torch.Tensor],
    reference_info: Dict[str, Any],
    run_index: int,
    profile_internals: bool,
    torch_profile: bool,
) -> Dict[str, Any]:
    method = variant["method"]
    method_params = variant["method_params"]
    result = _base_result(
        args=args,
        run_id=run_id,
        pdb_path=pdb_path,
        variant=variant,
        repeat_index=repeat_index,
        atom_count=atom_count,
        load_seconds=load_seconds,
    )
    cpu_rss_before = _current_rss_mb()
    cpu_peak_before = _peak_rss_mb()
    result.update(
        {
            "device": str(device),
            "dtype": args.dtype,
            "run_index": run_index,
            "repeat_index": int(repeat_index),
            "molecule": molecule_stats,
            "profiling": {
                "internal_profile_enabled": profile_internals,
                "torch_profile_enabled": torch_profile,
                "torch_profile_note": (
                    "wall_seconds includes torch profiler overhead"
                    if torch_profile
                    else None
                ),
            },
            "cpu_rss_before_mb": cpu_rss_before,
            "cpu_peak_rss_before_mb": cpu_peak_before,
        }
    )

    if device.type == "cuda" and torch.cuda.is_available():
        _safe_cuda_synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = None
        end_event = None

    wall_start = time.perf_counter()
    points: Optional[torch.Tensor] = None
    cuda_sync_error = None
    section_summary = None
    torch_profile_summary = None
    quality_metrics = None
    try:
        section_profiler = (
            SectionProfiler(
                device=device,
                capture_shapes=args.profile_shapes,
                sample_structures=args.profile_sample_structures,
                record_cuda_events=args.profile_record_cuda_events,
                synchronize_cuda=args.profile_synchronize_cuda,
                top_call_count=args.profile_top_calls,
                max_summary_items=args.profile_max_summary_items,
            )
            if profile_internals
            else None
        )
        torch_profile_path = (
            _torch_profile_path(
                args,
                run_id=run_id,
                pdb_id=pdb_path.stem,
                variant=variant,
                run_index=run_index,
            )
            if torch_profile
            else None
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch_profile and device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.no_grad():
            if torch_profile:
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=args.torch_profile_with_stack,
                ) as profiler:
                    if section_profiler is None:
                        points = _call_method(method, coords, radii, args, method_params)
                    else:
                        with section_profiler:
                            points = _call_method(method, coords, radii, args, method_params)
                if torch_profile_path is not None:
                    profiler.export_chrome_trace(str(torch_profile_path))
                torch_profile_summary = {
                    "trace_path": str(torch_profile_path) if torch_profile_path else None,
                    "top_ops": _torch_profiler_top_ops(
                        profiler,
                        limit=args.torch_profile_top_ops,
                    ),
                }
            elif section_profiler is None:
                points = _call_method(method, coords, radii, args, method_params)
            else:
                with section_profiler:
                    points = _call_method(method, coords, radii, args, method_params)
        if section_profiler is not None:
            section_summary = section_profiler.summary(limit=args.profile_top_functions)
        if end_event is not None:
            end_event.record()
        cuda_sync_error = _safe_cuda_synchronize(device)
        wall_seconds = time.perf_counter() - wall_start

        cuda_event_ms = None
        if start_event is not None and end_event is not None and cuda_sync_error is None:
            cuda_event_ms = float(start_event.elapsed_time(end_event))

        memory = _cuda_memory(device)
        finite_points = bool(torch.isfinite(points).all().item()) if points.numel() else True
        quality_metrics = _reference_metrics(
            points,
            reference_vertices,
            reference_info,
            args,
            device,
        )
        result.update(
            {
                "status": "ok",
                "wall_seconds": wall_seconds,
                "cuda_event_ms": cuda_event_ms,
                "point_count": int(points.shape[0]),
                "points_tensor_bytes": _tensor_nbytes(points),
                "points_per_atom": (
                    float(points.shape[0]) / atom_count if atom_count else None
                ),
                "points_per_wall_second": (
                    float(points.shape[0]) / wall_seconds if wall_seconds > 0 else None
                ),
                "finite_points": finite_points,
                "points_device": str(points.device),
                "points_dtype": str(points.dtype).replace("torch.", ""),
                "cuda_sync_error": cuda_sync_error,
                "reference_metrics": quality_metrics,
            }
        )
        cpu_rss_after = _current_rss_mb()
        cpu_peak_after = _peak_rss_mb()
        result.update(
            {
                "cpu_rss_after_mb": cpu_rss_after,
                "cpu_peak_rss_after_mb": cpu_peak_after,
                "cpu_rss_delta_mb": (
                    cpu_rss_after - cpu_rss_before
                    if cpu_rss_after is not None and cpu_rss_before is not None
                    else None
                ),
                "cpu_peak_delta_mb": (
                    cpu_peak_after - cpu_peak_before
                    if cpu_peak_after is not None and cpu_peak_before is not None
                    else None
                ),
            }
        )
        result.update(memory)
        if section_summary is not None:
            result["internal_profile"] = section_summary
        if torch_profile_summary is not None:
            result["torch_profile"] = torch_profile_summary
    except Exception as exc:  # noqa: BLE001 - benchmark must keep going.
        cuda_sync_error = _safe_cuda_synchronize(device)
        cpu_rss_after = _current_rss_mb()
        cpu_peak_after = _peak_rss_mb()
        result.update(
            {
                "status": "error",
                "wall_seconds": time.perf_counter() - wall_start,
                "cuda_event_ms": None,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=12)[-6000:],
                "cuda_sync_error": cuda_sync_error,
                "cpu_rss_after_mb": cpu_rss_after,
                "cpu_peak_rss_after_mb": cpu_peak_after,
                "cpu_rss_delta_mb": (
                    cpu_rss_after - cpu_rss_before
                    if cpu_rss_after is not None and cpu_rss_before is not None
                    else None
                ),
                "cpu_peak_delta_mb": (
                    cpu_peak_after - cpu_peak_before
                    if cpu_peak_after is not None and cpu_peak_before is not None
                    else None
                ),
            }
        )
        result.update(_cuda_memory(device))
        if section_summary is not None:
            result["internal_profile"] = section_summary
        if torch_profile_summary is not None:
            result["torch_profile"] = torch_profile_summary
    finally:
        del points
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _load_pdb(
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    load_start = time.perf_counter()
    protein = read_pdb_tensors(
        path,
        center=args.center,
        include_hetatm=args.include_hetatm,
        include_hydrogens=args.include_hydrogens,
        unknown_elements=args.unknown_elements,
        dtype=dtype,
    )
    coords = protein.atom_coords.to(device=device, dtype=dtype)
    radii = protein.atom_radii.to(device=device, dtype=dtype)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start
    return coords, radii, int(coords.shape[0]), load_seconds


def _write_load_error_results(
    *,
    writer: JsonlWriter,
    result_records: List[Dict[str, Any]],
    args: argparse.Namespace,
    run_id: str,
    pdb_path: Path,
    runs: Sequence[Tuple[Dict[str, Any], int]],
    exc: Exception,
) -> None:
    for variant, repeat_index in runs:
        record = _base_result(
            args=args,
            run_id=run_id,
            pdb_path=pdb_path,
            variant=variant,
            repeat_index=repeat_index,
            atom_count=None,
            load_seconds=None,
        )
        record.update(
            {
                "status": "load_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=12)[-6000:],
            }
        )
        writer.write(record)
        result_records.append(record)


def _write_skip_results(
    *,
    writer: JsonlWriter,
    result_records: List[Dict[str, Any]],
    args: argparse.Namespace,
    run_id: str,
    pdb_path: Path,
    runs: Sequence[Tuple[Dict[str, Any], int]],
    atom_count: int,
    load_seconds: float,
    reason: str,
) -> None:
    for variant, repeat_index in runs:
        record = _base_result(
            args=args,
            run_id=run_id,
            pdb_path=pdb_path,
            variant=variant,
            repeat_index=repeat_index,
            atom_count=atom_count,
            load_seconds=load_seconds,
        )
        record.update({"status": reason})
        writer.write(record)
        result_records.append(record)


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _numeric_summary(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return {"mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "mean": sum(cleaned) / len(cleaned),
        "median": _percentile(cleaned, 0.5),
        "p90": _percentile(cleaned, 0.9),
        "p95": _percentile(cleaned, 0.95),
        "max": max(cleaned),
    }


def _clean_timing_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        record
        for record in records
        if not record.get("profiling", {}).get("internal_profile_enabled")
        and not record.get("profiling", {}).get("torch_profile_enabled")
    ]


def _profile_status_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for record in records:
        profiling = record.get("profiling", {})
        if profiling.get("torch_profile_enabled"):
            counts["torch_profiled"] += 1
        if profiling.get("internal_profile_enabled"):
            counts["internal_profiled"] += 1
        if (
            not profiling.get("torch_profile_enabled")
            and not profiling.get("internal_profile_enabled")
        ):
            counts["clean_timing"] += 1
    return dict(counts)


def _reference_status_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(
        Counter(
            record.get("reference_metrics", {}).get("status", "not_recorded")
            for record in records
            if record.get("status") == "ok"
        )
    )


def _summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_variant: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in results:
        method = record.get("method")
        if method:
            by_method[str(method)].append(record)
            by_variant[
                (
                    str(method),
                    str(record.get("variant_name", "default")),
                    str(record.get("method_parameter_hash", "")),
                )
            ].append(record)

    method_summaries: Dict[str, Any] = {}
    for method in METHOD_ORDER:
        records = by_method.get(method, [])
        ok_records = [record for record in records if record.get("status") == "ok"]
        clean_records = _clean_timing_records(ok_records)
        reference_records = [
            record.get("reference_metrics", {})
            for record in ok_records
            if record.get("reference_metrics", {}).get("status") == "ok"
        ]
        method_summaries[method] = {
            "records": len(records),
            "status_counts": dict(Counter(record.get("status") for record in records)),
            "profiling_counts": _profile_status_counts(ok_records),
            "reference_status_counts": _reference_status_counts(records),
            "wall_seconds": _numeric_summary(
                record.get("wall_seconds") for record in ok_records
            ),
            "clean_wall_seconds": _numeric_summary(
                record.get("wall_seconds") for record in clean_records
            ),
            "cuda_event_ms": _numeric_summary(
                record.get("cuda_event_ms") for record in ok_records
            ),
            "clean_cuda_event_ms": _numeric_summary(
                record.get("cuda_event_ms") for record in clean_records
            ),
            "gpu_peak_allocated_mb": _numeric_summary(
                record.get("gpu_peak_allocated_mb") for record in ok_records
            ),
            "gpu_peak_reserved_mb": _numeric_summary(
                record.get("gpu_peak_reserved_mb") for record in ok_records
            ),
            "gpu_free_mb": _numeric_summary(
                record.get("gpu_free_mb") for record in ok_records
            ),
            "cpu_rss_delta_mb": _numeric_summary(
                record.get("cpu_rss_delta_mb") for record in ok_records
            ),
            "cpu_peak_delta_mb": _numeric_summary(
                record.get("cpu_peak_delta_mb") for record in ok_records
            ),
            "point_count": _numeric_summary(record.get("point_count") for record in ok_records),
            "atom_count": _numeric_summary(record.get("atom_count") for record in ok_records),
            "points_per_wall_second": _numeric_summary(
                record.get("points_per_wall_second") for record in ok_records
            ),
            "clean_points_per_wall_second": _numeric_summary(
                record.get("points_per_wall_second") for record in clean_records
            ),
            "reference_symmetric_mean_distance": _numeric_summary(
                record.get("symmetric_mean_distance") for record in reference_records
            ),
            "point_count_to_reference_ratio": _numeric_summary(
                record.get("point_count_to_reference_ratio") for record in reference_records
            ),
            "reference_metric_seconds": _numeric_summary(
                record.get("seconds") for record in reference_records
            ),
        }

    variant_summaries: List[Dict[str, Any]] = []
    for (method, variant_name, method_hash), records in by_variant.items():
        ok_records = [record for record in records if record.get("status") == "ok"]
        clean_records = _clean_timing_records(ok_records)
        reference_records = [
            record.get("reference_metrics", {})
            for record in ok_records
            if record.get("reference_metrics", {}).get("status") == "ok"
        ]
        params = ok_records[0].get("method_params") if ok_records else records[0].get("method_params")
        variant_summaries.append(
            {
                "method": method,
                "variant_name": variant_name,
                "method_parameter_hash": method_hash,
                "method_params": params,
                "records": len(records),
                "status_counts": dict(Counter(record.get("status") for record in records)),
                "profiling_counts": _profile_status_counts(ok_records),
                "reference_status_counts": _reference_status_counts(records),
                "wall_seconds": _numeric_summary(
                    record.get("wall_seconds") for record in ok_records
                ),
                "clean_wall_seconds": _numeric_summary(
                    record.get("wall_seconds") for record in clean_records
                ),
                "cuda_event_ms": _numeric_summary(
                    record.get("cuda_event_ms") for record in ok_records
                ),
                "clean_cuda_event_ms": _numeric_summary(
                    record.get("cuda_event_ms") for record in clean_records
                ),
                "gpu_peak_allocated_mb": _numeric_summary(
                    record.get("gpu_peak_allocated_mb") for record in ok_records
                ),
                "gpu_peak_reserved_mb": _numeric_summary(
                    record.get("gpu_peak_reserved_mb") for record in ok_records
                ),
                "gpu_free_mb": _numeric_summary(
                    record.get("gpu_free_mb") for record in ok_records
                ),
                "cpu_rss_delta_mb": _numeric_summary(
                    record.get("cpu_rss_delta_mb") for record in ok_records
                ),
                "cpu_peak_delta_mb": _numeric_summary(
                    record.get("cpu_peak_delta_mb") for record in ok_records
                ),
                "point_count": _numeric_summary(
                    record.get("point_count") for record in ok_records
                ),
                "points_per_wall_second": _numeric_summary(
                    record.get("points_per_wall_second") for record in ok_records
                ),
                "clean_points_per_wall_second": _numeric_summary(
                    record.get("points_per_wall_second") for record in clean_records
                ),
                "reference_symmetric_mean_distance": _numeric_summary(
                    record.get("symmetric_mean_distance") for record in reference_records
                ),
                "point_count_to_reference_ratio": _numeric_summary(
                    record.get("point_count_to_reference_ratio") for record in reference_records
                ),
                "reference_metric_seconds": _numeric_summary(
                    record.get("seconds") for record in reference_records
                ),
            }
        )
    variant_summaries.sort(
        key=lambda item: (
            item["method"],
            (
                item["clean_wall_seconds"]["median"]
                if item["clean_wall_seconds"]["median"] is not None
                else item["wall_seconds"]["median"]
            )
            if item["wall_seconds"]["median"] is not None
            else float("inf"),
        )
    )

    fastest_variants: Dict[str, List[Dict[str, Any]]] = {}
    for method in METHOD_ORDER:
        fastest_variants[method] = [
            item
            for item in variant_summaries
            if item["method"] == method and item["wall_seconds"]["median"] is not None
        ][:10]

    ok_results = [record for record in results if record.get("status") == "ok"]
    error_results = [
        record
        for record in results
        if record.get("status") not in {None, "ok", "skipped_max_atoms"}
    ]
    slowest = sorted(
        ok_results,
        key=lambda record: record.get("wall_seconds") or 0,
        reverse=True,
    )[:25]
    highest_memory = sorted(
        ok_results,
        key=lambda record: record.get("gpu_peak_allocated_mb") or 0,
        reverse=True,
    )[:25]

    def slim(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pdb_id": record.get("pdb_id"),
            "method": record.get("method"),
            "variant_name": record.get("variant_name"),
            "status": record.get("status"),
            "atom_count": record.get("atom_count"),
            "point_count": record.get("point_count"),
            "wall_seconds": record.get("wall_seconds"),
            "cuda_event_ms": record.get("cuda_event_ms"),
            "gpu_peak_allocated_mb": record.get("gpu_peak_allocated_mb"),
            "points_per_wall_second": record.get("points_per_wall_second"),
            "reference_symmetric_mean_distance": record.get(
                "reference_metrics",
                {},
            ).get("symmetric_mean_distance"),
            "point_count_to_reference_ratio": record.get(
                "reference_metrics",
                {},
            ).get("point_count_to_reference_ratio"),
            "error_type": record.get("error_type"),
            "error_message": record.get("error_message"),
        }

    return {
        "total_records": len(results),
        "ok_records": len(ok_results),
        "non_ok_records": len(results) - len(ok_results),
        "method_summaries": method_summaries,
        "variant_summaries": variant_summaries,
        "fastest_variants_by_method": fastest_variants,
        "slowest_ok_results": [slim(record) for record in slowest],
        "highest_gpu_memory_ok_results": [slim(record) for record in highest_memory],
        "errors": [slim(record) for record in error_results[:50]],
    }


def _write_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _warmup_device(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    sample = torch.randn((1024, 3), device=device)
    _ = sample @ sample.T
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark SES samplers on a PDB dataset and write JSONL results."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SES_BENCH_DATA_DIR", "Data/01-benchmark_pdbs"),
        help="Directory containing benchmark .pdb files.",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("SES_BENCH_OUTPUT", _default_output_path()),
        help="Streaming JSONL output path.",
    )
    parser.add_argument(
        "--summary-output",
        default=os.environ.get("SES_BENCH_SUMMARY_OUTPUT"),
        help="Summary JSON path. Defaults to OUTPUT with .summary.json suffix.",
    )
    parser.add_argument(
        "--program-version",
        type=_parse_program_version,
        default=os.environ.get("SES_BENCH_PROGRAM_VERSION", PROGRAM_VERSION),
        help=(
            "Program version for benchmark comparisons. Use X.Y.Z, matching "
            "the GitHub release/tag version when available."
        ),
    )
    parser.add_argument("--methods", type=_parse_methods, default=list(METHOD_ORDER))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--molecule-order",
        choices=(
            "name",
            "atom_count_desc",
            "atom_count_asc",
            "file_size_desc",
            "file_size_asc",
        ),
        default=os.environ.get("SES_BENCH_MOLECULE_ORDER", "name"),
        help=(
            "Order PDB files before shard/offset/limit selection. "
            "Use atom_count_desc to run the largest molecules first."
        ),
    )
    parser.add_argument(
        "--largest-first",
        action="store_true",
        help="Shortcut for --molecule-order atom_count_desc.",
    )
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs per molecule/method/variant.",
    )
    parser.add_argument(
        "--profile-only-first-repeat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --repeats > 1, keep later repeats cleaner for timing.",
    )
    parser.add_argument(
        "--sweep-preset",
        choices=("none", "focused", "broad"),
        default=os.environ.get("SES_BENCH_SWEEP_PRESET", "none"),
        help="Parameter sweep preset. focused varies one parameter at a time; broad is larger.",
    )
    parser.add_argument(
        "--sweep-cartesian",
        action="store_true",
        help="Use Cartesian products for sweep values instead of one-axis variants.",
    )
    parser.add_argument(
        "--sweep-max-variants-per-method",
        type=int,
        default=None,
        help="Optional cap including the default variant.",
    )

    parser.add_argument("--profile-internals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-internals-every", type=int, default=1)
    parser.add_argument("--profile-internals-limit-runs", type=int, default=None)
    parser.add_argument("--profile-shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-sample-structures", action="store_true")
    parser.add_argument("--profile-record-cuda-events", action="store_true")
    parser.add_argument("--profile-synchronize-cuda", action="store_true")
    parser.add_argument("--profile-top-functions", type=int, default=40)
    parser.add_argument("--profile-top-calls", type=int, default=12)
    parser.add_argument("--profile-max-summary-items", type=int, default=6)

    parser.add_argument("--torch-profile-limit", type=int, default=0)
    parser.add_argument("--torch-profile-every", type=int, default=0)
    parser.add_argument("--torch-profile-dir", default="tmp/gpu_benchmarks/traces")
    parser.add_argument("--torch-profile-top-ops", type=int, default=30)
    parser.add_argument("--torch-profile-with-stack", action="store_true")

    parser.add_argument("--device", default=os.environ.get("SES_BENCH_DEVICE", "cuda"))
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail immediately if the selected device is not CUDA.",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--include-hetatm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-hydrogens", action="store_true")
    parser.add_argument(
        "--unknown-elements",
        choices=("error", "skip"),
        default="skip",
        help="How to handle PDB elements outside C,H,O,N,S,SE.",
    )

    parser.add_argument("--probe-radius", type=float, default=1.4)
    parser.add_argument(
        "--surface-dir",
        default=os.environ.get("SES_BENCH_SURFACE_DIR"),
    )
    parser.add_argument("--reference-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-sample-size", type=int, default=4096)
    parser.add_argument("--reference-distance-budget", type=int, default=16_000_000)
    parser.add_argument("--grid-spacing", type=float, default=None)
    parser.add_argument("--point-area", type=float, default=0.5)
    parser.add_argument("--pairwise-element-budget", type=int, default=64_000_000)

    parser.add_argument("--projected-m", type=int, default=230)
    parser.add_argument("--projected-m-values", type=_parse_csv_ints, default=None)

    parser.add_argument("--sdf-m", type=int, default=34)
    parser.add_argument("--sdf-m-values", type=_parse_csv_ints, default=None)
    parser.add_argument("--sdf-smoothness", type=float, default=0.2)
    parser.add_argument("--sdf-smoothness-values", type=_parse_csv_floats, default=None)
    parser.add_argument("--sdf-iterations", type=int, default=6)
    parser.add_argument("--sdf-iterations-values", type=_parse_csv_ints, default=None)
    parser.add_argument("--sdf-level-tolerance", type=float, default=0.05)
    parser.add_argument("--sdf-subsample-spacing", type=float, default=None)
    parser.add_argument("--sdf-subsample-spacing-values", type=_parse_csv_floats, default=None)
    parser.add_argument("--sdf-feature-threshold", type=float, default=0.1)
    parser.add_argument("--sdf-max-grid-points", type=int, default=500_000)

    parser.add_argument("--analytic-atom-filter-samples", type=int, default=64)
    parser.add_argument("--analytic-atom-filter-samples-values", type=_parse_csv_ints, default=None)
    parser.add_argument("--analytic-pair-filter-samples", type=int, default=12)
    parser.add_argument("--analytic-pair-filter-samples-values", type=_parse_csv_ints, default=None)
    parser.add_argument("--analytic-oversample-factor", type=float, default=1.25)
    parser.add_argument("--analytic-oversample-factor-values", type=_parse_csv_floats, default=None)
    parser.add_argument("--analytic-point-area-values", type=_parse_csv_floats, default=None)
    parser.add_argument("--analytic-probe-density-scale", type=float, default=1.0)
    parser.add_argument("--analytic-probe-density-scale-values", type=_parse_csv_floats, default=None)
    parser.add_argument("--analytic-max-probe-support-atoms", type=int, default=16)
    parser.add_argument("--analytic-support-tolerance", type=float, default=1e-3)
    parser.add_argument("--analytic-dedup-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--analytic-max-probe-triples",
        type=_parse_optional_int,
        default=5_000_000,
    )
    parser.add_argument("--analytic-max-grid-points", type=int, default=500_000)

    parser.add_argument("--tiled-point-area", type=float, default=None)
    parser.add_argument("--tiled-point-area-values", type=_parse_csv_floats, default=None)
    parser.add_argument("--tile-size", type=_parse_float_or_auto, default="auto")
    parser.add_argument("--tile-size-values", type=_parse_csv_float_or_auto, default=None)
    parser.add_argument("--tile-overlap", type=_parse_float_or_auto, default="auto")
    parser.add_argument("--tile-overlap-values", type=_parse_csv_float_or_auto, default=None)
    parser.add_argument("--tiled-atom-density-scale", type=float, default=7.0)
    parser.add_argument("--tiled-atom-density-scale-values", type=_parse_csv_floats, default=None)
    parser.add_argument("--tiled-pair-density-scale", type=float, default=0.0)
    parser.add_argument("--tiled-pair-density-scale-values", type=_parse_csv_floats, default=None)
    parser.add_argument("--tiled-probe-density-scale", type=float, default=0.0)
    parser.add_argument("--tiled-probe-density-scale-values", type=_parse_csv_floats, default=None)
    parser.add_argument("--tiled-dedup-tolerance", type=float, default=0.05)
    parser.add_argument("--tiled-dedup-tolerance-values", type=_parse_csv_floats, default=None)
    parser.add_argument("--tiled-exact-accessibility", action="store_true")
    parser.add_argument("--tiled-max-grid-points", type=int, default=500_000)
    parser.add_argument(
        "--tiled-max-probe-triples",
        type=_parse_optional_int,
        default=5_000_000,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.largest_first:
        args.molecule_order = "atom_count_desc"
    if args.surface_dir is None:
        args.surface_dir = _default_surface_dir(args.data_dir)
    methods = args.methods if isinstance(args.methods, list) else _parse_methods(args.methods)
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.reference_sample_size < 1:
        raise SystemExit("--reference-sample-size must be positive")
    if args.reference_distance_budget < 1:
        raise SystemExit("--reference-distance-budget must be positive")
    if args.profile_internals_every < 1:
        raise SystemExit("--profile-internals-every must be positive")
    if args.profile_internals and not _INSTALLED_PROFILE_WRAPPERS:
        _install_internal_profile_wrappers()
    output_path = Path(args.output)
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else output_path.with_suffix(".summary.json")
    )

    if output_path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(
            f"{output_path} already exists; use --resume, --overwrite, or a new --output"
        )

    device = torch.device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("--require-cuda is enabled, but --device is not a CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA device requested, but torch.cuda.is_available() is false")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        args.device = str(device)
    elif args.require_cuda:
        raise SystemExit("CUDA is required; pass --no-require-cuda for CPU smoke runs")

    dtype = _dtype_from_name(args.dtype)
    variants = _build_variants(args, methods)
    variant_runs = [
        (variant, repeat_index)
        for variant in variants
        for repeat_index in range(args.repeats)
    ]

    pdbs = _select_pdbs(args)
    run_id = uuid.uuid4().hex[:12]
    append = bool(args.resume and output_path.exists())
    existing_keys, result_records = (
        _load_existing_results(output_path, variants, args.program_version)
        if append
        else (set(), [])
    )
    writer = JsonlWriter(output_path, append=append)

    start_record = {
        "event": "run_start",
        "schema_version": SCHEMA_VERSION,
        "program_version": args.program_version,
        "benchmark_driver_version": BENCHMARK_DRIVER_VERSION,
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "argv": list(sys.argv if argv is None else argv),
        "environment": _environment(device),
        "selection": {
            "data_dir": str(Path(args.data_dir)),
            "available_pdb_count": len(sorted(Path(args.data_dir).glob("*.pdb"))),
            "selected_pdb_count": len(pdbs),
            "surface_dir": args.surface_dir,
            "limit": args.limit,
            "offset": args.offset,
            "molecule_order": args.molecule_order,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "max_atoms": args.max_atoms,
        },
        "parameters": _all_parameters(args, variants),
        "resume": {
            "enabled": args.resume,
            "matching_existing_results": len(existing_keys),
            "program_version": args.program_version,
        },
        "output": {
            "jsonl": str(output_path),
            "summary_json": str(summary_path),
        },
    }
    writer.write(start_record)

    print(
        f"[ses-gpu-bench] run_id={run_id} methods={','.join(methods)} "
        f"variants={len(variants)} repeats={args.repeats} "
        f"molecules={len(pdbs)} output={output_path}",
        flush=True,
    )
    _warmup_device(device)

    attempted = 0
    skipped_resume = 0
    torch_profiles_written = 0
    try:
        for pdb_index, pdb_path in enumerate(pdbs, start=1):
            pending_runs = [
                (variant, repeat_index)
                for variant, repeat_index in variant_runs
                if (
                    pdb_path.stem,
                    variant["method"],
                    variant["hash"],
                    repeat_index,
                )
                not in existing_keys
            ]
            if not pending_runs:
                skipped_resume += len(variant_runs)
                continue

            try:
                coords, radii, atom_count, load_seconds = _load_pdb(
                    pdb_path,
                    args,
                    device,
                    dtype,
                )
            except Exception as exc:  # noqa: BLE001 - report and continue.
                _write_load_error_results(
                    writer=writer,
                    result_records=result_records,
                    args=args,
                    run_id=run_id,
                    pdb_path=pdb_path,
                    runs=pending_runs,
                    exc=exc,
                )
                if args.fail_fast:
                    raise
                continue

            if args.max_atoms is not None and atom_count > args.max_atoms:
                _write_skip_results(
                    writer=writer,
                    result_records=result_records,
                    args=args,
                    run_id=run_id,
                    pdb_path=pdb_path,
                    runs=pending_runs,
                    atom_count=atom_count,
                    load_seconds=load_seconds,
                    reason="skipped_max_atoms",
                )
                del coords, radii
                gc.collect()
                continue

            molecule_stats = _molecule_stats(coords, radii)
            reference_vertices, reference_info = _load_reference_vertices(
                pdb_path,
                args,
                device,
                dtype,
            )
            molecule_stats["reference_surface"] = reference_info
            for variant, repeat_index in pending_runs:
                run_index = attempted + 1
                should_torch_profile = _should_torch_profile(
                    args,
                    run_index,
                    torch_profiles_written,
                    repeat_index,
                )
                record = _run_one_method(
                    args=args,
                    run_id=run_id,
                    pdb_path=pdb_path,
                    variant=variant,
                    coords=coords,
                    radii=radii,
                    atom_count=atom_count,
                    load_seconds=load_seconds,
                    device=device,
                    molecule_stats=molecule_stats,
                    reference_vertices=reference_vertices,
                    reference_info=reference_info,
                    repeat_index=repeat_index,
                    run_index=run_index,
                    profile_internals=_should_profile_run(args, run_index, repeat_index),
                    torch_profile=should_torch_profile,
                )
                if should_torch_profile:
                    torch_profiles_written += 1
                writer.write(record)
                result_records.append(record)
                attempted += 1
                if record.get("status") != "ok" and args.fail_fast:
                    raise RuntimeError(
                        f"{pdb_path.stem}/{variant['method']}/{variant['variant_name']} failed: "
                        f"{record.get('error_type')} {record.get('error_message')}"
                    )

            del coords, radii, reference_vertices
            gc.collect()
            if args.log_every and pdb_index % args.log_every == 0:
                status_counts = Counter(
                    record.get("status") for record in result_records
                )
                print(
                    f"[ses-gpu-bench] {pdb_index}/{len(pdbs)} molecules, "
                    f"attempted={attempted}, ok={status_counts.get('ok', 0)}, "
                    f"non_ok={len(result_records) - status_counts.get('ok', 0)}",
                    flush=True,
                )
    finally:
        summary = {
            "event": "run_end",
            "schema_version": SCHEMA_VERSION,
            "program_version": args.program_version,
            "benchmark_driver_version": BENCHMARK_DRIVER_VERSION,
            "run_id": run_id,
            "created_at_utc": _utc_now(),
            "output_jsonl": str(output_path),
            "summary_json": str(summary_path),
            "attempted_in_this_run": attempted,
            "skipped_by_resume": skipped_resume,
            "torch_profiles_written": torch_profiles_written,
            "summary": _summarize(result_records),
        }
        writer.write(summary)
        writer.close()
        _write_summary(summary_path, summary)
        print(
            f"[ses-gpu-bench] wrote summary={summary_path} "
            f"records={summary['summary']['total_records']}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
