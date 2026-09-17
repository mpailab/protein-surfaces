"""Small correctness fixtures for the surface-graph preprocessing script."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "ses-gnn" / "build_graphs.py"
SPEC = importlib.util.spec_from_file_location("ses_gnn_build_graphs", SCRIPT)
build_graphs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_graphs)


def _reference_fps(points, k):
    """Deliberately simple independent reference, including deterministic ties."""
    if k >= len(points):
        return list(range(len(points)))
    selected = []
    remaining = list(range(len(points)))
    for _ in range(k):
        if not selected:
            next_index = 0
        else:
            next_index = max(
                remaining,
                key=lambda index: min(
                    sum((a - b) ** 2 for a, b in zip(points[index], points[other]))
                    for other in selected
                ),
            )
        selected.append(next_index)
        remaining.remove(next_index)
    return selected


def _reference_graph(pos, norms, r_local, r_mid, r_global, k_mid, k_global, sigma):
    points = pos.detach().cpu().tolist()
    normals = norms.detach().cpu().tolist()
    edges = {}
    for source, point in enumerate(points):
        candidates = [[], [], []]
        for dest, other in enumerate(points):
            if source == dest:
                continue
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(point, other)))
            if distance < r_local:
                candidates[0].append(dest)
            elif distance < r_mid:
                candidates[1].append(dest)
            elif distance < r_global:
                candidates[2].append(dest)
        selected = candidates[0]
        for neighbors, limit in zip(candidates[1:], (k_mid, k_global)):
            selected += [neighbors[i] for i in _reference_fps([points[j] for j in neighbors], limit)]
        for dest in selected:
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(point, points[dest])))
            dot = sum(a * b for a, b in zip(normals[source], normals[dest]))
            weight = math.exp(-(distance ** 2) / (2 * sigma ** 2)) * (1 + dot) / 2
            edges[source, dest] = (distance, dot, weight)
    return edges


def _assert_matches_reference(pos, norms, params, **options):
    edge_index, edge_attr = build_graphs.build_multiscale_graph(pos, norms, **params, **options)
    expected = _reference_graph(pos, norms, **params)
    output_device = torch.device(options.get("output_device") or pos.device)
    assert edge_index.dtype == torch.long
    assert edge_index.device == output_device
    assert edge_attr.device == output_device
    assert edge_attr.dtype == torch.promote_types(pos.dtype, norms.dtype)
    assert edge_index.shape == (2, len(expected))
    assert edge_attr.shape == (len(expected), 3)
    edges = list(map(tuple, edge_index.cpu().T.tolist()))
    assert len(set(edges)) == len(edges), "FPS must not select the same neighbor twice"
    assert set(edges) == set(expected)
    expected_attr = torch.tensor(
        [expected[edge] for edge in edges], dtype=edge_attr.dtype, device=output_device
    ).reshape(-1, 3)
    torch.testing.assert_close(edge_attr, expected_attr)
    return edge_index, edge_attr


@pytest.mark.parametrize("k", [0, 1, 3, 5, 8])
def test_fps_handles_duplicate_coordinates_and_zero_or_large_k(k):
    points = torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.],
                           [1., 0., 0.], [0., 0., 0.]], dtype=torch.float64)
    indices = build_graphs.farthest_point_sampling(points, k)
    assert indices.dtype == torch.long
    assert indices.device == points.device
    assert indices.tolist() == _reference_fps(points.tolist(), k)
    assert len(indices.unique()) == min(k, len(points))


def test_fps_empty_and_large_translated_coordinates(monkeypatch):
    def reject_cdist(*args, **kwargs):
        raise AssertionError("FPS must use direct differences for stable distances")

    monkeypatch.setattr(torch, "cdist", reject_cdist)
    generator = torch.Generator().manual_seed(817)
    points = torch.randint(-24, 24, (40, 3), generator=generator).float() / 4
    translated = points + 2 ** 18
    expected = _reference_fps(points.tolist(), 9)
    assert build_graphs.farthest_point_sampling(points, 9).tolist() == expected
    assert build_graphs.farthest_point_sampling(translated, 9).tolist() == expected
    assert build_graphs.farthest_point_sampling(torch.empty((0, 3)), 4).shape == (0,)


@pytest.mark.parametrize("k", [0, 2])
@pytest.mark.parametrize("points", [
    torch.zeros(3),
    torch.zeros((2, 2)),
    torch.zeros((2, 3), dtype=torch.int64),
    torch.zeros((2, 3), dtype=torch.float16),
    torch.full((2, 3), float("nan")),
    torch.full((2, 3), float("inf")),
], ids=["rank", "coordinates", "integer", "half", "nan", "infinity"])
def test_fps_rejects_invalid_points_even_when_no_sampling_is_needed(points, k):
    with pytest.raises(ValueError):
        build_graphs.farthest_point_sampling(points, k)


@pytest.mark.parametrize("chunk_size,budget", [(2, 100), (4, 7), (4, 3), (1, 1)])
def test_candidate_batches_preserve_all_rows_and_neighbors_within_budgets(chunk_size, budget):
    # Three separated clusters exercise large, small and singleton neighborhoods.
    coordinates = torch.zeros((9, 3), dtype=torch.float64).numpy()
    coordinates[:, 0] = [0., 0.1, 0.2, 0.3, 0.4, 10., 10.1, 10.2, 20.]
    radius = 0.75
    tree = build_graphs.cKDTree(coordinates)
    covered_sources = []
    exceeded_budget = False
    for start, num_rows, rows, neighbors in build_graphs._candidate_batches(
        tree, coordinates, radius, chunk_size, budget, workers=1
    ):
        assert 1 <= num_rows <= chunk_size
        assert len(rows) == len(neighbors)
        if len(neighbors) > budget:
            assert num_rows == 1, "Only one indivisible neighborhood may exceed the budget"
            exceeded_budget = True
        expected_rows, expected_neighbors = [], []
        for row in range(num_rows):
            source = start + row
            covered_sources.append(source)
            # Independent exhaustive search also checks sorted IDs and self candidates.
            for neighbor, point in enumerate(coordinates):
                if sum((a - b) ** 2 for a, b in zip(coordinates[source], point)) <= radius ** 2:
                    expected_rows.append(row)
                    expected_neighbors.append(neighbor)
        assert rows.tolist() == expected_rows
        assert neighbors.tolist() == expected_neighbors
    assert covered_sources == list(range(len(coordinates)))
    assert exceeded_budget == (budget < 5)


def test_local_neighborhood_is_not_limited_to_32_edges():
    pos = torch.zeros((41, 3), dtype=torch.float64)
    pos[:, 0] = torch.arange(41, dtype=pos.dtype) / 100
    norms = torch.tensor([0., 0., 1.], dtype=pos.dtype).expand_as(pos)
    params = dict(r_local=1., r_mid=2., r_global=4., k_mid=0, k_global=0, sigma=3.)
    edges, _ = _assert_matches_reference(pos, norms, params, chunk_size=7, max_candidate_pairs=13)
    assert torch.equal(torch.bincount(edges[0]), torch.full((41,), 40))


def test_fps_sees_all_candidates_in_both_annuli():
    generator = torch.Generator().manual_seed(519)
    directions = torch.randn((40, 3), generator=generator, dtype=torch.float64)
    directions /= directions.norm(dim=1, keepdim=True)
    mid_radius = 2.8 + torch.rand((40, 1), generator=generator, dtype=torch.float64) * 0.4
    global_radius = 5.7 + torch.rand((40, 1), generator=generator, dtype=torch.float64) * 0.6
    pos = torch.cat((torch.zeros((1, 3), dtype=torch.float64),
                     directions * mid_radius, directions * global_radius))
    norms = torch.tensor([0., 0., 1.], dtype=pos.dtype).expand_as(pos)
    params = dict(r_local=2., r_mid=4., r_global=8., k_mid=5, k_global=4, sigma=3.)
    edges, _ = _assert_matches_reference(pos, norms, params, chunk_size=11, max_candidate_pairs=32)
    center_neighbors = edges[1, edges[0] == 0]
    assert (center_neighbors <= 40).sum() == 5
    assert (center_neighbors > 40).sum() == 4


def test_grouped_fps_selects_distinct_neighbors_with_coincident_coordinates():
    pos = torch.zeros((13, 3), dtype=torch.float64)
    pos[1:7, 0] = 3
    pos[7:, 0] = 6
    norms = torch.tensor([0., 0., 1.], dtype=pos.dtype).expand_as(pos)
    params = dict(r_local=2., r_mid=4., r_global=8., k_mid=3, k_global=3, sigma=3.)
    edges, _ = _assert_matches_reference(pos, norms, params, max_candidate_pairs=7)
    assert set(edges[1, edges[0] == 0].tolist()) == {1, 2, 3, 7, 8, 9}


@pytest.mark.parametrize("chunk_size,budget", [(1, 1), (7, 19), (1024, 262144)])
@pytest.mark.parametrize("pos_dtype", [torch.float32, torch.float64])
def test_chunk_and_candidate_budget_preserve_graph(chunk_size, budget, pos_dtype):
    generator = torch.Generator().manual_seed(409)
    pos = torch.randn((27, 3), generator=generator, dtype=pos_dtype) * 2
    norms = torch.randn((27, 3), generator=generator, dtype=torch.float32)
    norms /= norms.norm(dim=1, keepdim=True)
    params = dict(r_local=1.1, r_mid=3.2, r_global=5.5, k_mid=4, k_global=3, sigma=1.7)
    _assert_matches_reference(pos, norms, params, chunk_size=chunk_size, max_candidate_pairs=budget)


def test_exact_radius_boundaries_and_zero_sampling():
    pos = torch.tensor([[0., 0., 0.], [0.5, 0., 0.], [1., 0., 0.],
                        [2., 0., 0.], [4., 0., 0.]], dtype=torch.float64)
    norms = torch.tensor([[0., 0., 2.], [0., 0., -1.], [1., 0., 0.],
                          [0., 1., 0.], [0., 0., 1.]], dtype=pos.dtype)
    params = dict(r_local=1., r_mid=2., r_global=4., k_mid=0, k_global=8, sigma=2.)
    edges, _ = _assert_matches_reference(pos, norms, params)
    assert set(edges[1, edges[0] == 0].tolist()) == {1, 3}
    params.update(k_mid=8, k_global=0)
    edges, _ = _assert_matches_reference(pos, norms, params)
    assert set(edges[1, edges[0] == 0].tolist()) == {1, 2}


@pytest.mark.parametrize("count,spacing", [(0, 1.), (1, 1.), (4, 100.), (4, 0.)])
def test_empty_isolated_and_coincident_points(count, spacing):
    pos = torch.zeros((count, 3), dtype=torch.float32)
    pos[:, 0] = torch.arange(count) * spacing
    norms = torch.ones_like(pos, dtype=torch.float64)
    params = dict(r_local=1., r_mid=2., r_global=4., k_mid=0, k_global=0, sigma=3.)
    _assert_matches_reference(pos, norms, params, max_candidate_pairs=1)


@pytest.mark.parametrize("name,value,error", [
    ("r_local", 0., "radii"),
    ("r_mid", 8., "radii"),
    ("r_global", float("inf"), "radii"),
    ("sigma", 0., "sigma"),
    ("k_mid", -1, "k_mid"),
    ("k_global", 1.5, "k_global"),
    ("chunk_size", 0, "chunk_size"),
    ("max_candidate_pairs", True, "max_candidate_pairs"),
    ("workers", 0, "workers"),
    ("workers", -1.0, "workers"),
])
def test_graph_rejects_invalid_parameters_before_building_neighbors(name, value, error):
    pos = torch.zeros((2, 3))
    params = dict(r_local=2., r_mid=4., r_global=8., k_mid=2, k_global=2, sigma=3.)
    params[name] = value
    with pytest.raises(ValueError, match=error):
        build_graphs.build_multiscale_graph(pos, torch.ones_like(pos), **params)


@pytest.mark.parametrize("pos,norms", [
    (torch.zeros(3), torch.zeros(3)),
    (torch.zeros((2, 3)), torch.zeros((1, 3))),
    (torch.zeros((2, 3), dtype=torch.int64), torch.zeros((2, 3))),
    (torch.zeros((2, 3)), torch.zeros((2, 3), dtype=torch.float16)),
    (torch.full((2, 3), float("nan")), torch.zeros((2, 3))),
    (torch.zeros((2, 3)), torch.full((2, 3), float("inf"))),
], ids=["rank", "mismatched_normals", "integer_positions", "half_normals",
        "nan_positions", "infinite_normals"])
def test_graph_rejects_invalid_positions_and_normals(pos, norms):
    params = dict(r_local=2., r_mid=4., r_global=8., k_mid=2, k_global=2, sigma=3.)
    with pytest.raises(ValueError):
        build_graphs.build_multiscale_graph(pos, norms, **params)


def test_process_file_preserves_training_payload_and_skips_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.], [6., 0., 0.]])
    raw = dict(
        pos=pos,
        norms=torch.tensor([0., 0., 1.]).expand_as(pos).clone(),
        x=torch.arange(20, dtype=torch.float32).reshape(4, 5),
        y=torch.tensor([0., 1., 1., 0.]),
        pdb_id="fixture_A",
        mode="ppi",
        receptor_chains=["A", "B"],
        ligand_chains=["C"],
    )
    raw_path = tmp_path / "fixture.pt"
    out_dir = tmp_path / "graphs"
    out_dir.mkdir()
    torch.save(raw, raw_path)
    params = dict(r_local=2., r_mid=4., r_global=8., k_mid=2, k_global=2, sigma=3.)
    assert build_graphs.process_file(str(raw_path), str(out_dir), params) is True
    saved_path = out_dir / raw_path.name
    saved = torch.load(saved_path, map_location="cpu", weights_only=False)
    for key, value in raw.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(saved[key], value)
            assert saved[key].device.type == "cpu"
        else:
            assert saved[key] == value
    expected = _reference_graph(pos, raw["norms"], **params)
    edges = list(map(tuple, saved["edge_index"].T.tolist()))
    assert set(edges) == set(expected)
    torch.testing.assert_close(saved["edge_attr"], torch.tensor([expected[edge] for edge in edges]))
    original_contents = saved_path.read_bytes()

    def reject_reload(*args, **kwargs):
        raise AssertionError("Existing output should be skipped before loading the input")

    monkeypatch.setattr(torch, "load", reject_reload)
    assert build_graphs.process_file(str(raw_path), str(out_dir), {}) is True
    assert saved_path.read_bytes() == original_contents


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("output_device", [None, "cpu"])
def test_cuda_graph_matches_cpu_reference(output_device):
    generator = torch.Generator().manual_seed(144)
    pos = torch.randn((21, 3), generator=generator, dtype=torch.float64).cuda()
    norms = torch.randn((21, 3), generator=generator, dtype=torch.float64).cuda()
    norms /= norms.norm(dim=1, keepdim=True)
    params = dict(r_local=0.7, r_mid=1.5, r_global=3., k_mid=4, k_global=3, sigma=1.2)
    _assert_matches_reference(pos, norms, params, chunk_size=6, max_candidate_pairs=17,
                              output_device=output_device)
