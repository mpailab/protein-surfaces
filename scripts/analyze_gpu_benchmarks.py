#!/usr/bin/env python
"""Analyze SES GPU benchmark artifacts.

The benchmark writer is optimized for crash-safe collection. This companion
script is optimized for the release loop:

1. compare a quick run with the latest release baseline;
2. summarize detail-mode profile artifacts when a regression appears;
3. rank sweep-mode variants when defaults or auto heuristics need tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CaseKey = Tuple[str, str, str, str, str]
VariantKey = Tuple[str, str, str, str]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{line_number}: invalid JSONL record: {exc}"
                ) from exc
    return records


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _result_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            record
            for record in _read_jsonl(path)
            if record.get("event") == "benchmark_result"
        ]
    payload = _read_json(path)
    records = payload.get("records")
    if isinstance(records, list):
        return [
            record
            for record in records
            if record.get("event") == "benchmark_result"
        ]
    raise SystemExit(
        f"{path} does not contain benchmark_result records. "
        "Use the streaming .jsonl file for compare/profile analysis."
    )


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = sorted(float(value) for value in values if value is not None)
    if not cleaned:
        return None
    middle = len(cleaned) // 2
    if len(cleaned) % 2:
        return cleaned[middle]
    return 0.5 * (cleaned[middle - 1] + cleaned[middle])


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _clean_timing_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean = []
    for record in records:
        profiling = record.get("profiling", {})
        if profiling.get("internal_profile_enabled"):
            continue
        if profiling.get("torch_profile_enabled"):
            continue
        clean.append(record)
    return clean


def _case_key(record: Dict[str, Any]) -> CaseKey:
    interface_params = record.get("interface_params") or {}
    return (
        str(record.get("pdb_id", "")),
        str(record.get("method", "")),
        str(record.get("method_variant_name") or record.get("variant_name", "default")),
        str(record.get("interface_mode", "points")),
        str(interface_params.get("scenario_mode", "independent")),
    )


def _variant_key_from_case(key: CaseKey) -> VariantKey:
    _, method, method_variant_name, interface_mode, scenario_mode = key
    return (method, method_variant_name, interface_mode, scenario_mode)


def _case_stats(records: Sequence[Dict[str, Any]]) -> Dict[CaseKey, Dict[str, Any]]:
    grouped: Dict[CaseKey, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") != "ok":
            continue
        if record.get("wall_seconds") is None:
            continue
        grouped[_case_key(record)].append(record)

    stats: Dict[CaseKey, Dict[str, Any]] = {}
    for key, case_records in grouped.items():
        timing_records = _clean_timing_records(case_records) or case_records
        stats[key] = {
            "records": len(case_records),
            "clean_records": len(_clean_timing_records(case_records)),
            "wall_seconds": _median(
                record.get("wall_seconds") for record in timing_records
            ),
            "cuda_event_ms": _median(
                record.get("cuda_event_ms") for record in timing_records
            ),
            "point_count": _median(record.get("point_count") for record in case_records),
            "points_per_wall_second": _median(
                record.get("points_per_wall_second") for record in timing_records
            ),
            "gpu_peak_allocated_mb": _median(
                record.get("gpu_peak_allocated_mb") for record in case_records
            ),
            "atom_count": _median(record.get("atom_count") for record in case_records),
            "method_params": case_records[0].get("method_params", {}),
        }
    return stats


def compare_benchmarks(
    baseline_records: Sequence[Dict[str, Any]],
    current_records: Sequence[Dict[str, Any]],
    *,
    regression_threshold: float,
    improvement_threshold: float,
    point_count_tolerance: float,
) -> Dict[str, Any]:
    baseline = _case_stats(baseline_records)
    current = _case_stats(current_records)
    common_keys = sorted(set(baseline) & set(current))
    comparisons = []
    for key in common_keys:
        base = baseline[key]
        cur = current[key]
        ratio = _safe_ratio(cur["wall_seconds"], base["wall_seconds"])
        point_ratio = _safe_ratio(cur["point_count"], base["point_count"])
        comparisons.append(
            {
                "pdb_id": key[0],
                "method": key[1],
                "method_variant_name": key[2],
                "interface_mode": key[3],
                "interface_scenarios": key[4],
                "baseline_wall_seconds": base["wall_seconds"],
                "current_wall_seconds": cur["wall_seconds"],
                "wall_ratio": ratio,
                "speedup": _safe_ratio(base["wall_seconds"], cur["wall_seconds"]),
                "baseline_point_count": base["point_count"],
                "current_point_count": cur["point_count"],
                "point_count_ratio": point_ratio,
                "gpu_peak_allocated_mb": cur["gpu_peak_allocated_mb"],
                "regression": (
                    ratio is not None and ratio > 1.0 + regression_threshold
                ),
                "improvement": (
                    ratio is not None and ratio < 1.0 - improvement_threshold
                ),
                "point_count_changed": (
                    point_ratio is not None
                    and abs(point_ratio - 1.0) > point_count_tolerance
                ),
            }
        )

    by_variant: Dict[VariantKey, List[Dict[str, Any]]] = defaultdict(list)
    for item in comparisons:
        by_variant[
            (
                item["method"],
                item["method_variant_name"],
                item["interface_mode"],
                item["interface_scenarios"],
            )
        ].append(item)

    variant_summaries = []
    for key, items in by_variant.items():
        ratios = [item["wall_ratio"] for item in items if item["wall_ratio"] is not None]
        worst = max(
            items,
            key=lambda item: item["wall_ratio"] or -math.inf,
        )
        best = min(
            items,
            key=lambda item: item["wall_ratio"] or math.inf,
        )
        variant_summaries.append(
            {
                "method": key[0],
                "method_variant_name": key[1],
                "interface_mode": key[2],
                "interface_scenarios": key[3],
                "cases": len(items),
                "median_wall_ratio": _median(ratios),
                "median_speedup": (
                    _safe_ratio(1.0, _median(ratios)) if _median(ratios) else None
                ),
                "regression_cases": sum(1 for item in items if item["regression"]),
                "improvement_cases": sum(1 for item in items if item["improvement"]),
                "point_count_changed_cases": sum(
                    1 for item in items if item["point_count_changed"]
                ),
                "worst_case": {
                    "pdb_id": worst["pdb_id"],
                    "wall_ratio": worst["wall_ratio"],
                    "baseline_wall_seconds": worst["baseline_wall_seconds"],
                    "current_wall_seconds": worst["current_wall_seconds"],
                },
                "best_case": {
                    "pdb_id": best["pdb_id"],
                    "wall_ratio": best["wall_ratio"],
                    "baseline_wall_seconds": best["baseline_wall_seconds"],
                    "current_wall_seconds": best["current_wall_seconds"],
                },
            }
        )

    variant_summaries.sort(
        key=lambda item: (
            item["median_wall_ratio"]
            if item["median_wall_ratio"] is not None
            else -math.inf
        ),
        reverse=True,
    )
    regressions = [item for item in comparisons if item["regression"]]
    improvements = [item for item in comparisons if item["improvement"]]
    point_changes = [item for item in comparisons if item["point_count_changed"]]
    regressions.sort(key=lambda item: item["wall_ratio"] or 0.0, reverse=True)
    improvements.sort(key=lambda item: item["wall_ratio"] or math.inf)

    return {
        "common_cases": len(common_keys),
        "baseline_only_cases": len(set(baseline) - set(current)),
        "current_only_cases": len(set(current) - set(baseline)),
        "regression_threshold": regression_threshold,
        "improvement_threshold": improvement_threshold,
        "point_count_tolerance": point_count_tolerance,
        "regression_cases": len(regressions),
        "improvement_cases": len(improvements),
        "point_count_changed_cases": len(point_changes),
        "variant_summaries": variant_summaries,
        "top_regressions": regressions[:25],
        "top_improvements": improvements[:25],
        "top_point_count_changes": sorted(
            point_changes,
            key=lambda item: abs((item["point_count_ratio"] or 1.0) - 1.0),
            reverse=True,
        )[:25],
    }


def _format_seconds(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}s"


def _format_ratio(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def _print_compare_report(report: Dict[str, Any]) -> None:
    print("GPU Benchmark Compare")
    print(f"Common cases: {report['common_cases']}")
    print(
        "Regressions: "
        f"{report['regression_cases']} "
        f"(threshold +{report['regression_threshold']:.0%})"
    )
    print(
        "Improvements: "
        f"{report['improvement_cases']} "
        f"(threshold -{report['improvement_threshold']:.0%})"
    )
    print(f"Point count changes: {report['point_count_changed_cases']}")
    print(f"Baseline-only cases: {report['baseline_only_cases']}")
    print(f"Current-only cases: {report['current_only_cases']}")

    print("\nWorst variant medians:")
    for item in report["variant_summaries"][:10]:
        worst = item["worst_case"]
        print(
            "  "
            f"{item['method']} {item['method_variant_name']} {item['interface_mode']}: "
            f"{item['interface_scenarios']}: "
            f"median={_format_ratio(item['median_wall_ratio'])}, "
            f"cases={item['cases']}, regressions={item['regression_cases']}, "
            f"worst={worst['pdb_id']} {_format_ratio(worst['wall_ratio'])}"
        )

    if report["top_regressions"]:
        print("\nTop per-molecule regressions:")
        for item in report["top_regressions"][:10]:
            print(
                "  "
                f"{item['pdb_id']} {item['method']} "
                f"{item['method_variant_name']} {item['interface_mode']} "
                f"{item['interface_scenarios']}: "
                f"{_format_ratio(item['wall_ratio'])} "
                f"({_format_seconds(item['baseline_wall_seconds'])} -> "
                f"{_format_seconds(item['current_wall_seconds'])})"
            )

    if report["top_improvements"]:
        print("\nTop per-molecule improvements:")
        for item in report["top_improvements"][:10]:
            print(
                "  "
                f"{item['pdb_id']} {item['method']} "
                f"{item['method_variant_name']} {item['interface_mode']} "
                f"{item['interface_scenarios']}: "
                f"{_format_ratio(item['wall_ratio'])} "
                f"({_format_seconds(item['baseline_wall_seconds'])} -> "
                f"{_format_seconds(item['current_wall_seconds'])})"
            )


def _resolve_artifact_path(raw_path: str, benchmark_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return benchmark_path.parent / path


def _load_profile_payload(path: Path) -> Dict[str, Any]:
    if path.suffix == ".pt":
        import torch

        return torch.load(path, map_location="cpu")
    if path.suffix == ".json":
        return _read_json(path)
    raise ValueError(f"unsupported profile artifact extension: {path}")


def _profile_payloads(
    benchmark_path: Path,
    *,
    profile_dir: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    payloads: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen_paths = set()

    for record in _result_records(benchmark_path):
        artifact = record.get("profile_artifact") or {}
        artifact_path = artifact.get("path")
        if artifact_path:
            path = _resolve_artifact_path(str(artifact_path), benchmark_path)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                payloads.append(_load_profile_payload(path))
            except Exception as exc:  # noqa: BLE001 - report all broken artifacts.
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if record.get("internal_profile") or record.get("torch_profile"):
            payloads.append(
                {
                    "pdb_id": record.get("pdb_id"),
                    "method": record.get("method"),
                    "variant_name": record.get("variant_name"),
                    "interface_mode": record.get("interface_mode"),
                    "interface_params": record.get("interface_params"),
                    "internal_profile": record.get("internal_profile"),
                    "torch_profile": record.get("torch_profile"),
                }
            )

    if profile_dir is not None:
        for path in sorted(profile_dir.glob("*.profile.*")):
            if path in seen_paths:
                continue
            try:
                payloads.append(_load_profile_payload(path))
                seen_paths.add(path)
            except Exception as exc:  # noqa: BLE001 - report all broken artifacts.
                errors.append(f"{path}: {type(exc).__name__}: {exc}")

    return payloads, errors


def summarize_profiles(
    benchmark_path: Path,
    *,
    profile_dir: Optional[Path],
) -> Dict[str, Any]:
    payloads, errors = _profile_payloads(benchmark_path, profile_dir=profile_dir)
    function_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"profiles": 0, "calls": 0, "wall_seconds_total": 0.0}
    )
    op_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "profiles": 0,
            "count": 0,
            "cuda_time_total_us": 0.0,
            "cpu_time_total_us": 0.0,
            "cuda_memory_usage_bytes": 0,
        }
    )
    top_calls = []
    for payload in payloads:
        interface_params = payload.get("interface_params") or {}
        scenario_mode = interface_params.get("scenario_mode", "independent")
        internal = payload.get("internal_profile") or {}
        for item in internal.get("top_functions", []) or []:
            name = str(item.get("name", ""))
            stats = function_stats[name]
            stats["profiles"] += 1
            stats["calls"] += int(item.get("calls", 0) or 0)
            stats["wall_seconds_total"] += float(item.get("wall_seconds_total", 0.0) or 0.0)
        for item in internal.get("top_calls", []) or []:
            top_calls.append(
                {
                    "pdb_id": payload.get("pdb_id"),
                    "method": payload.get("method"),
                    "interface_mode": payload.get("interface_mode"),
                    "interface_scenarios": scenario_mode,
                    "name": item.get("name"),
                    "wall_seconds": item.get("wall_seconds"),
                    "input_tensor_bytes": item.get("input_tensor_bytes"),
                    "output_tensor_bytes": item.get("output_tensor_bytes"),
                }
            )

        torch_profile = payload.get("torch_profile") or {}
        top_ops = torch_profile.get("top_ops") or {}
        for bucket in ("cuda_time", "cpu_time", "memory"):
            for item in top_ops.get(bucket, []) or []:
                key = str(item.get("key", ""))
                stats = op_stats[key]
                stats["profiles"] += 1
                stats["count"] += int(item.get("count", 0) or 0)
                stats["cuda_time_total_us"] += float(
                    item.get("cuda_time_total_us", 0.0) or 0.0
                )
                stats["cpu_time_total_us"] += float(
                    item.get("cpu_time_total_us", 0.0) or 0.0
                )
                stats["cuda_memory_usage_bytes"] += int(
                    item.get("cuda_memory_usage_bytes", 0) or 0
                )

    functions = [
        {"name": name, **stats}
        for name, stats in function_stats.items()
        if name
    ]
    functions.sort(key=lambda item: item["wall_seconds_total"], reverse=True)
    ops = [
        {"key": key, **stats}
        for key, stats in op_stats.items()
        if key
    ]
    ops.sort(
        key=lambda item: (
            item["cuda_time_total_us"],
            item["cpu_time_total_us"],
            abs(item["cuda_memory_usage_bytes"]),
        ),
        reverse=True,
    )
    top_calls.sort(key=lambda item: item.get("wall_seconds") or 0.0, reverse=True)
    return {
        "profile_count": len(payloads),
        "artifact_errors": errors,
        "top_functions": functions[:50],
        "top_ops": ops[:50],
        "top_calls": top_calls[:50],
    }


def _print_profile_report(report: Dict[str, Any], *, limit: int) -> None:
    print("GPU Benchmark Profile Summary")
    print(f"Profiles loaded: {report['profile_count']}")
    if report["artifact_errors"]:
        print(f"Artifact errors: {len(report['artifact_errors'])}")
        for error in report["artifact_errors"][:limit]:
            print(f"  {error}")

    print("\nTop internal functions:")
    for item in report["top_functions"][:limit]:
        print(
            "  "
            f"{item['name']}: {item['wall_seconds_total']:.4f}s, "
            f"calls={item['calls']}, profiles={item['profiles']}"
        )

    print("\nTop PyTorch ops:")
    for item in report["top_ops"][:limit]:
        print(
            "  "
            f"{item['key']}: cuda={item['cuda_time_total_us']:.0f}us, "
            f"cpu={item['cpu_time_total_us']:.0f}us, "
            f"cuda_mem={item['cuda_memory_usage_bytes']}"
        )

    print("\nSlowest individual internal calls:")
    for item in report["top_calls"][:limit]:
        print(
            "  "
            f"{item.get('pdb_id')} {item.get('method')} "
            f"{item.get('interface_mode')} {item.get('interface_scenarios')} "
            f"{item.get('name')}: "
            f"{_format_seconds(item.get('wall_seconds'))}"
        )


def summarize_sweep(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") != "ok":
            continue
        interface_params = record.get("interface_params") or {}
        key = (
            str(record.get("method", "")),
            str(record.get("interface_mode", "points")),
            str(interface_params.get("scenario_mode", "independent")),
            str(record.get("method_variant_name") or record.get("variant_name", "default")),
            str(record.get("method_parameter_hash", "")),
        )
        grouped[key].append(record)

    rows = []
    defaults: Dict[Tuple[str, str, str], Optional[float]] = {}
    for key, row_records in grouped.items():
        method, interface_mode, scenario_mode, variant_name, method_hash = key
        timing_records = _clean_timing_records(row_records) or row_records
        wall_seconds = _median(record.get("wall_seconds") for record in timing_records)
        row = {
            "method": method,
            "interface_mode": interface_mode,
            "interface_scenarios": scenario_mode,
            "method_variant_name": variant_name,
            "method_parameter_hash": method_hash,
            "records": len(row_records),
            "wall_seconds": wall_seconds,
            "point_count": _median(record.get("point_count") for record in row_records),
            "gpu_peak_allocated_mb": _median(
                record.get("gpu_peak_allocated_mb") for record in row_records
            ),
            "method_params": row_records[0].get("method_params", {}),
        }
        rows.append(row)
        if variant_name == "default":
            defaults[(method, interface_mode, scenario_mode)] = wall_seconds

    for row in rows:
        default_seconds = defaults.get(
            (row["method"], row["interface_mode"], row["interface_scenarios"])
        )
        row["speedup_vs_default"] = _safe_ratio(default_seconds, row["wall_seconds"])
    rows.sort(
        key=lambda item: (
            item["method"],
            item["interface_mode"],
            item["interface_scenarios"],
            item["wall_seconds"] if item["wall_seconds"] is not None else math.inf,
        )
    )

    status_counts = dict(Counter(record.get("status") for record in records))
    return {
        "status_counts": status_counts,
        "variant_count": len(rows),
        "variants": rows,
    }


def _print_sweep_report(report: Dict[str, Any], *, limit: int) -> None:
    print("GPU Benchmark Sweep Summary")
    print(f"Status counts: {report['status_counts']}")
    print(f"Variants: {report['variant_count']}")
    by_method_interface: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in report["variants"]:
        by_method_interface[
            (row["method"], row["interface_mode"], row["interface_scenarios"])
        ].append(row)
    for key in sorted(by_method_interface):
        print(f"\n{key[0]} {key[1]} {key[2]}:")
        for row in by_method_interface[key][:limit]:
            print(
                "  "
                f"{row['method_variant_name']}: "
                f"{_format_seconds(row['wall_seconds'])}, "
                f"speedup_vs_default={_format_ratio(row['speedup_vs_default'])}, "
                f"records={row['records']}, params={row['method_params']}"
            )


def _write_optional_json(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze SES GPU benchmark JSONL and profile artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="Compare quick run with a release baseline.")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--current", required=True, type=Path)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--regression-threshold", type=float, default=0.10)
    compare.add_argument("--improvement-threshold", type=float, default=0.05)
    compare.add_argument("--point-count-tolerance", type=float, default=0.05)
    compare.add_argument("--fail-on-regression", action="store_true")

    profiles = subparsers.add_parser("profiles", help="Summarize detail profile artifacts.")
    profiles.add_argument("--benchmark", required=True, type=Path)
    profiles.add_argument("--profile-dir", type=Path)
    profiles.add_argument("--output", type=Path)
    profiles.add_argument("--limit", type=int, default=20)

    sweep = subparsers.add_parser("sweep", help="Rank parameter sweep variants.")
    sweep.add_argument("--benchmark", required=True, type=Path)
    sweep.add_argument("--output", type=Path)
    sweep.add_argument("--limit", type=int, default=8)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "compare":
        report = compare_benchmarks(
            _result_records(args.baseline),
            _result_records(args.current),
            regression_threshold=args.regression_threshold,
            improvement_threshold=args.improvement_threshold,
            point_count_tolerance=args.point_count_tolerance,
        )
        _print_compare_report(report)
        _write_optional_json(args.output, report)
        if args.fail_on_regression and (
            report["regression_cases"]
            or report["point_count_changed_cases"]
            or report["baseline_only_cases"]
        ):
            return 1
        return 0

    if args.command == "profiles":
        report = summarize_profiles(args.benchmark, profile_dir=args.profile_dir)
        _print_profile_report(report, limit=args.limit)
        _write_optional_json(args.output, report)
        return 1 if report["artifact_errors"] else 0

    if args.command == "sweep":
        report = summarize_sweep(_result_records(args.benchmark))
        _print_sweep_report(report, limit=args.limit)
        _write_optional_json(args.output, report)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
