import torch

from scripts import benchmark_ses_gpu as bench


def _parser_args(*extra: str):
    args = bench._build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--no-require-cuda",
            "--data-dir",
            "tests/data/pdb",
            "--no-reference-metrics",
            "--no-profile-internals",
            *extra,
        ]
    )
    bench._apply_mode_defaults(args)
    return args


def test_gpu_benchmark_quick_mode_defaults_are_compact() -> None:
    args = _parser_args()

    assert args.mode == "quick"
    assert args.molecule_order == "atom_count_desc"
    assert args.repeats == 1
    assert args.sweep_preset == "none"
    assert args.profile_internals is False
    assert args.torch_profile_limit == 0
    assert args.torch_profile_export_traces is False
    assert args.profile_artifact_format == "none"
    assert args.fsync is True


def test_gpu_benchmark_detail_mode_defaults_to_compact_artifacts() -> None:
    args = _parser_args(
        "--mode",
        "detail",
        "--profile-internals",
        "--reference-metrics",
    )

    assert args.repeats == 1
    assert args.sweep_preset == "none"
    assert args.reference_metrics is True
    assert args.profile_internals is True
    assert args.profile_shapes is True
    assert args.torch_profile_limit == 20
    assert args.torch_profile_memory is True
    assert args.torch_profile_export_traces is False
    assert args.profile_artifact_format == "pt"
    assert args.inline_profile_details is False


def test_gpu_benchmark_sweep_mode_defaults_to_focused_grid() -> None:
    args = _parser_args("--mode", "sweep")

    assert args.repeats == 3
    assert args.sweep_preset == "focused"
    assert args.profile_internals is False
    assert args.torch_profile_limit == 0


def test_gpu_benchmark_profile_details_can_be_written_as_artifact(tmp_path) -> None:
    args = _parser_args(
        "--mode",
        "detail",
        "--profile-internals",
        "--profile-artifact-dir",
        str(tmp_path),
    )
    variant = {
        "method": "projected",
        "variant_name": "default",
        "interface_mode": "points",
        "hash": "abc123",
    }
    result = {"status": "ok", "repeat_index": 0}
    section_summary = {"top_functions": [{"name": "ses.projected.foo"}]}

    bench._attach_profile_details(
        result,
        args=args,
        run_id="run1",
        pdb_path=tmp_path / "1ABC_A.pdb",
        variant=variant,
        run_index=1,
        section_summary=section_summary,
        torch_profile_summary=None,
    )

    artifact = result["profile_artifact"]
    assert artifact["format"] == "pt"
    assert artifact["file_size_bytes"] > 0
    assert "internal_profile" not in result
    payload = torch.load(artifact["path"])
    assert payload["pdb_id"] == "1ABC_A"
    assert payload["internal_profile"] == section_summary


def test_gpu_benchmark_density_defaults_are_calibrated() -> None:
    args = _parser_args()

    assert args.program_version == "0.0.4"
    assert args.point_area == 0.5
    assert args.analytic_oversample_factor == 1.0
    assert args.projected_m == 192
    assert args.sdf_m == 26
    assert args.tiled_point_area is None
    assert bench._method_params(args, "tiled_analytic")["point_area"] == 0.5
    assert args.tiled_atom_density_scale == 1.0
    assert args.tiled_pair_density_scale == 1.0
    assert args.tiled_probe_density_scale == 1.0


def test_gpu_benchmark_interface_variants_are_hashed_separately() -> None:
    args = _parser_args(
        "--methods",
        "projected",
        "--interfaces",
        "all",
    )
    variants = bench._build_variants(args, ["projected"])

    assert [variant["interface_mode"] for variant in variants] == [
        "points",
        "features",
        "normals",
        "adjacency",
    ]
    assert [variant["variant_name"] for variant in variants] == [
        "default",
        "default__interface=features",
        "default__interface=normals",
        "default__interface=adjacency",
    ]
    assert len({variant["hash"] for variant in variants}) == 4
    assert [
        (
            variant["interface_params"]["include_atom_features"],
            variant["interface_params"]["include_normals"],
            variant["interface_params"]["include_adjacency"],
        )
        for variant in variants
    ] == [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ]


def test_gpu_benchmark_interface_parser_keeps_requested_variants_independent() -> None:
    assert bench._parse_interface_modes("features,normals,adjacency") == [
        "features",
        "normals",
        "adjacency",
    ]
    assert bench._parse_interface_modes("normals_adjacency") == [
        "features",
        "normals",
        "adjacency",
    ]


def test_gpu_benchmark_split_sampler_output_reads_independent_adjacency() -> None:
    args = _parser_args("--interfaces", "adjacency")
    interface_params = bench._interface_params(args, "adjacency")
    points = torch.zeros((3, 3), dtype=torch.float32)
    indices = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    values = torch.ones(4, dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()

    split = bench._split_sampler_output(
        (points, adjacency),
        interface_params,
    )

    assert split[0] is points
    assert split[1] is None
    assert split[2] is None
    assert split[3] is adjacency


def test_gpu_benchmark_points_interface_is_geometry_only() -> None:
    args = _parser_args("--interfaces", "points")
    interface_params = bench._interface_params(args, "points")
    points = torch.zeros((3, 3), dtype=torch.float32)

    split = bench._split_sampler_output(points, interface_params)
    stats = bench._output_stats(*split)

    assert split[0] is points
    assert split[1:] == (None, None, None)
    assert stats["output_tensor_count"] == 1
    assert stats["atom_features_present"] is False
    assert stats["normals_present"] is False
    assert stats["adjacency_present"] is False


def test_gpu_benchmark_output_stats_include_features_normals_and_sparse_adjacency() -> None:
    points = torch.zeros((3, 3), dtype=torch.float32)
    atom_features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    normals = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    values = torch.ones(4, dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()

    stats = bench._output_stats(points, atom_features, normals, adjacency)

    assert stats["output_tensor_count"] == 4
    assert stats["atom_features_present"] is True
    assert stats["atom_features_shape"] == [3, 2]
    assert stats["finite_atom_features"] is True
    assert abs(stats["atom_feature_active_mean"] - 4 / 3) < 1e-6
    assert stats["atom_feature_active_max"] == 2
    assert stats["normals_present"] is True
    assert stats["finite_normals"] is True
    assert stats["normal_unit_max_abs_error"] == 0.0
    assert stats["adjacency_present"] is True
    assert stats["adjacency_nnz"] == 4
    assert stats["adjacency_mean_degree"] == 4 / 3
    assert stats["adjacency_tensor_bytes"] > 0
