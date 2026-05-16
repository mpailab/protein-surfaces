import json
from pathlib import Path

import torch

from scripts import analyze_gpu_benchmarks as analysis


def _write_jsonl(path: Path, records):
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _result(
    *,
    pdb_id="1ABC_A",
    method="projected",
    variant="default",
    interface="points",
    seconds=1.0,
    points=100,
    status="ok",
    profile=False,
):
    return {
        "event": "benchmark_result",
        "status": status,
        "pdb_id": pdb_id,
        "method": method,
        "variant_name": variant,
        "method_variant_name": variant,
        "interface_mode": interface,
        "method_parameter_hash": f"{method}-{variant}",
        "method_params": {"m": 192} if method == "projected" else {},
        "wall_seconds": seconds,
        "point_count": points,
        "points_per_wall_second": points / seconds,
        "profiling": {
            "internal_profile_enabled": profile,
            "torch_profile_enabled": profile,
        },
    }


def test_compare_benchmarks_reports_regressions_and_point_count_changes():
    baseline = [
        _result(seconds=1.0, points=100),
        _result(pdb_id="2DEF_A", seconds=2.0, points=200),
    ]
    current = [
        _result(seconds=1.3, points=100),
        _result(pdb_id="2DEF_A", seconds=1.8, points=240),
    ]

    report = analysis.compare_benchmarks(
        baseline,
        current,
        regression_threshold=0.10,
        improvement_threshold=0.05,
        point_count_tolerance=0.05,
    )

    assert report["common_cases"] == 2
    assert report["regression_cases"] == 1
    assert report["improvement_cases"] == 1
    assert report["point_count_changed_cases"] == 1
    assert report["top_regressions"][0]["pdb_id"] == "1ABC_A"


def test_profile_summary_loads_referenced_pt_artifact(tmp_path):
    artifact_path = tmp_path / "profile.profile.pt"
    torch.save(
        {
            "pdb_id": "1ABC_A",
            "method": "projected",
            "interface_mode": "points",
            "internal_profile": {
                "top_functions": [
                    {
                        "name": "ses.projection.sample_projected_points",
                        "calls": 2,
                        "wall_seconds_total": 1.5,
                    }
                ],
                "top_calls": [
                    {
                        "name": "ses.projection.sample_projected_points",
                        "wall_seconds": 1.0,
                    }
                ],
            },
            "torch_profile": {
                "top_ops": {
                    "cuda_time": [
                        {
                            "key": "aten::cdist",
                            "count": 3,
                            "cuda_time_total_us": 400.0,
                        }
                    ]
                }
            },
        },
        artifact_path,
    )
    benchmark_path = tmp_path / "detail.jsonl"
    record = _result(profile=True)
    record["profile_artifact"] = {"path": str(artifact_path), "format": "pt"}
    _write_jsonl(benchmark_path, [record])

    report = analysis.summarize_profiles(benchmark_path, profile_dir=None)

    assert report["profile_count"] == 1
    assert report["artifact_errors"] == []
    assert report["top_functions"][0]["name"] == "ses.projection.sample_projected_points"
    assert report["top_functions"][0]["wall_seconds_total"] == 1.5
    assert report["top_ops"][0]["key"] == "aten::cdist"


def test_sweep_summary_ranks_variants_against_default():
    records = [
        _result(variant="default", seconds=2.0),
        _result(variant="m=160", seconds=1.0),
        _result(variant="m=320", seconds=3.0),
    ]

    report = analysis.summarize_sweep(records)

    assert report["variant_count"] == 3
    fastest = report["variants"][0]
    assert fastest["method_variant_name"] == "m=160"
    assert fastest["speedup_vs_default"] == 2.0
