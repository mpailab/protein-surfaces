#!/usr/bin/env python3
"""Build exact multiscale SES graphs with bounded candidate batches and FPS.

Neighbor lookup uses one CPU KD-tree. Geometry and grouped FPS run on the
input tensor device; only the final graph, not all global-radius candidates,
is retained. This is an offline, non-differentiable preprocessing operation.
"""

import argparse
import glob
import math
import operator
import os
from itertools import chain
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
import torch


# These limits bound temporary candidate storage, not the final graph size.
# A single neighborhood is never truncated, even if it exceeds the pair budget.
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_MAX_CANDIDATE_PAIRS = 262144


def _integer(name: str, value: int, minimum: int = 0) -> int:
    """Validate an integer option without silently truncating floats.

    Args:
        name: Option name used in validation errors.
        value: Python integer or integer-like scalar implementing ``__index__``.
            Booleans are rejected, although Python treats them as integers.
        minimum: Inclusive lower bound for the value.

    Returns:
        The validated value as a Python integer.

    Raises:
        ValueError: If the value is not an integer or is below ``minimum``.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@torch.no_grad()
def farthest_point_sampling(points: torch.Tensor, k: int) -> torch.Tensor:
    """Return up to k distinct indices, starting at 0; break ties by index.

    Squared coordinate differences avoid both square roots and cdist's
    matrix-multiplication path, which loses precision for translated clouds.

    Args:
        points: Finite float32/float64 coordinates of shape [M, 3], on CPU or
            CUDA. Coordinates should use the same length units on every axis.
        k: Nonnegative maximum number of points to select.

    Returns:
        Long tensor [min(M, k)] on ``points.device`` containing distinct input
        indices in FPS selection order. If M <= k, returns all indices in
        input order; if M == 0 or k == 0, returns an empty tensor. Coincident
        coordinates are allowed and remain distinct candidates by index.

    Raises:
        ValueError: If ``k`` is not a nonnegative integer, or the coordinates
            have an invalid shape, dtype or nonfinite value.

    Notes:
        This preprocessing helper does not track gradients. It uses O(M)
        working memory and O(M * min(M, k)) work when sampling is needed.
    """
    k = _integer("k", k)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [M, 3]")
    if points.dtype not in (torch.float32, torch.float64):
        raise ValueError("points must use float32 or float64")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite")
    count = points.shape[0]
    if count <= k:
        return torch.arange(count, device=points.device)
    indices = torch.empty(k, dtype=torch.long, device=points.device)
    if k == 0:
        return indices
    indices[0] = 0
    distances = points.new_full((count,), float("inf"))
    for step in range(1, k):
        delta = points - points[indices[step - 1]]
        torch.minimum(distances, delta.square().sum(dim=-1), out=distances)
        # Previously selected points stay excluded even for coincident points.
        distances[indices[step - 1]] = -1
        indices[step] = distances.argmax()
    return indices


def _sample_neighbor_indices(pos: torch.Tensor, rows: torch.Tensor,
                             neighbors: torch.Tensor, k: int,
                             num_rows: int) -> torch.Tensor:
    """Select offsets into source-sorted ragged neighbors, without padding.

    Small neighborhoods pass through. All larger neighborhoods share k FPS
    iterations, using segmented reductions instead of a Python loop per node.

    Args:
        pos: Validated surface coordinates [N, 3].
        rows: Long tensor [C] of source IDs relative to the current block,
            in nondecreasing order and in the range [0, num_rows).
        neighbors: Long tensor [C] of global destination IDs into ``pos``.
            IDs within each row must be distinct and sorted in ascending order.
        k: Nonnegative outgoing neighbor limit for this distance interval.
        num_rows: Number of source vertices in the block, including empty rows.
            Both index tensors must be on ``pos.device``.

    Returns:
        Long tensor of selected offsets into ``rows`` and ``neighbors``, on
        ``pos.device``. Each row contributes min(degree, k) offsets. Small rows
        precede sampled rows, so the output need not be sorted by source ID.

    Notes:
        Called after validation and with gradients disabled by the builder.
        Working memory is O(C + num_rows); no [num_rows, max_degree] padding
        is allocated.
        Boolean compaction may synchronize CUDA once per operation, but there
        is no Python iteration or host transfer for each individual source.
    """
    size = neighbors.numel()
    offsets = torch.arange(size, device=pos.device)
    if k == 0 or size == 0:
        return offsets[:0]
    counts = torch.bincount(rows, minlength=num_rows)
    dense = counts[rows] > k
    dense_offsets = offsets[dense]
    if dense_offsets.numel() == 0:
        return offsets
    sparse_offsets = offsets[~dense]
    # Compact nonempty sampled rows to segment IDs 0..num_groups-1. This keeps
    # empty rows out of FPS and makes the initial index a valid candidate.
    _, groups, counts = torch.unique_consecutive(
        rows[dense], return_inverse=True, return_counts=True
    )
    points = pos[neighbors[dense]]
    size = points.shape[0]
    num_groups = counts.numel()
    current = counts.cumsum(0) - counts
    selected = torch.empty((num_groups, k), dtype=torch.long, device=pos.device)
    distances = points.new_full((size,), float("inf"))
    offsets = torch.arange(size, device=pos.device)
    maximum = points.new_empty(num_groups)
    for step in range(k):
        selected[:, step] = current
        if step + 1 == k:
            break
        delta = points - points[current][groups]
        torch.minimum(distances, delta.square().sum(dim=-1), out=distances)
        distances[current] = -1
        # First find each segment's largest remaining distance, then select
        # the smallest candidate offset among ties for a reproducible choice.
        maximum.fill_(-1)
        maximum.scatter_reduce_(0, groups, distances, reduce="amax", include_self=True)
        candidates = torch.where(distances == maximum[groups], offsets, size)
        current = torch.full((num_groups,), size, dtype=torch.long, device=pos.device)
        current.scatter_reduce_(0, groups, candidates, reduce="amin", include_self=True)
    return torch.cat((sparse_offsets, dense_offsets[selected.reshape(-1)]))


def _candidate_batches(tree: cKDTree, coordinates: np.ndarray, radius: float,
                       chunk_size: int, max_candidate_pairs: int,
                       workers: int) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield sorted neighbor indices; retain only O(N) counts between queries.

    A single dense neighborhood may exceed the pair budget: it is kept intact
    for exact FPS. All other batches obey both the row and candidate budgets.

    Args:
        tree: CPU spatial index built from ``coordinates`` in their input order.
        coordinates: Array [N, 3] of all indexed points; preferably ``tree.data``
            to reuse its contiguous float64 storage for queries.
        radius: Positive search radius in coordinate units, including any
            numerical margin. Final distance filtering happens in the caller.
        chunk_size: Positive maximum number of source vertices per batch.
        max_candidate_pairs: Positive candidate budget, counting self matches.
        workers: Positive number of SciPy query threads, or -1 for all CPUs.

    Yields:
        ``(start, num_rows, rows, neighbors)`` for consecutive source blocks.
        ``start`` is the first global source ID, ``num_rows`` is the block
        length, and the two int64 arrays [C] hold relative source IDs and
        global destination IDs. Entries are sorted by source then destination;
        self matches and candidates on the search boundary are still included.

    Notes:
        A count-only query sizes batches before any neighbor lists are built.
        Thus temporary neighbor storage scales with one batch, plus O(N) count
        and prefix-sum arrays. The tree and final output have separate costs.
    """
    counts = tree.query_ball_point(coordinates, radius, workers=workers, return_length=True)
    indptr = np.empty(len(coordinates) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    start = 0
    while start < len(coordinates):
        # Prefix sums allow choosing a budget-fitting block without inspecting
        # every row in Python. Force one row if that row alone exceeds budget.
        budget_end = int(np.searchsorted(
            indptr, indptr[start] + max_candidate_pairs, side="right"
        )) - 1
        stop = min(len(coordinates), start + chunk_size, max(start + 1, budget_end))
        neighborhoods = tree.query_ball_point(
            coordinates[start:stop], radius, workers=workers, return_sorted=True
        )
        neighbors = np.fromiter(
            chain.from_iterable(neighborhoods), dtype=np.int64,
            count=int(indptr[stop] - indptr[start]),
        )
        rows = np.repeat(np.arange(stop - start, dtype=np.int64), counts[start:stop])
        # Release Python lists before geometry/FPS for this batch is computed.
        del neighborhoods
        yield start, stop - start, rows, neighbors
        start = stop


def _validate_parameters(r_local: float, r_mid: float, r_global: float,
                         k_mid: int, k_global: int, sigma: float,
                         chunk_size: int, max_candidate_pairs: int,
                         workers: int) -> None:
    """Validate shared builder and CLI configuration before allocating work.

    Args:
        r_local: Positive finite outer radius of the local interval.
        r_mid: Finite middle radius, strictly greater than ``r_local``.
        r_global: Finite outer radius, strictly greater than ``r_mid``.
        k_mid: Nonnegative integer limit for middle-interval neighbors.
        k_global: Nonnegative integer limit for outer-interval neighbors.
        sigma: Positive finite Gaussian length scale, in coordinate units.
        chunk_size: Positive integer maximum number of sources per block.
        max_candidate_pairs: Positive integer candidate budget per block.
        workers: Positive integer number of KD-tree threads, or -1 for all CPUs.

    Returns:
        None. The supplied values are checked without changing them.

    Raises:
        ValueError: If a radius, length scale, count or worker setting is invalid.
    """
    if not all(math.isfinite(r) for r in (r_local, r_mid, r_global)) or not (
        0 < r_local < r_mid < r_global
    ):
        raise ValueError("radii must satisfy 0 < r_local < r_mid < r_global")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    _integer("k_mid", k_mid)
    _integer("k_global", k_global)
    _integer("chunk_size", chunk_size, 1)
    _integer("max_candidate_pairs", max_candidate_pairs, 1)
    workers = _integer("workers", workers, -1)
    if workers == 0:
        raise ValueError("workers must be -1 or a positive integer")


@torch.no_grad()
def build_multiscale_graph(pos: torch.Tensor, norms: torch.Tensor,
                           r_local: float, r_mid: float, r_global: float,
                           k_mid: int, k_global: int, sigma: float,
                           *, chunk_size: int = DEFAULT_CHUNK_SIZE,
                           max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
                           workers: int = 1,
                           output_device: Optional[Union[str, torch.device]] = None
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return directed edges and [distance, normal dot product, weight].

    All non-self neighbors in [0, r_local) are retained; FPS selects at most
    k_mid/k_global outgoing neighbors in [r_local, r_mid)/[r_mid, r_global).
    Neighbor IDs are sorted before FPS, making its seed and tie breaks stable
    across batch sizes. No implicit cap is applied to the candidate search.

    The KD-tree runs on CPU even for CUDA inputs. Candidate geometry and FPS
    run on pos.device. Set output_device='cpu' to accumulate large final graphs
    outside GPU memory. Inputs must be finite float32/float64 tensors [N, 3].

    Args:
        pos: Surface coordinates [N, 3], conventionally in angstroms.
        norms: Surface normals [N, 3], on the same device as ``pos``. Normals
            are used as supplied; their dot product is a cosine only for unit
            vectors. Their dtype may differ from the coordinate dtype.
        r_local: Positive outer radius of the local interval; all neighbors
            strictly below this distance are retained.
        r_mid: Outer radius of the middle interval, greater than ``r_local``.
        r_global: Outer radius of the global interval, greater than ``r_mid``.
        k_mid: Maximum middle-interval neighbors per source; zero disables them.
        k_global: Maximum global-interval neighbors per source; zero disables them.
        sigma: Positive Gaussian length scale, in the same units as ``pos``.
        chunk_size: Maximum number of source vertices in a candidate block.
        max_candidate_pairs: Candidate budget before filtering. A single row
            larger than this budget is kept intact rather than truncated.
        workers: KD-tree CPU query threads; -1 uses all available CPUs.
        output_device: Device for returned tensors and accumulated output
            blocks. None retains ``pos.device``; it does not control computation.

    Returns:
        ``(edge_index, edge_attr)`` with shapes [2, E] and [E, 3]. The long
        ``edge_index`` stores global source IDs in row 0 and destination IDs
        in row 1. Edges are directed, without self loops or duplicate pairs.
        ``edge_attr`` columns are distance d, normal dot product c, and
        ``exp(-d**2 / (2 * sigma**2)) * (1 + c) / 2``. Its dtype is the promoted
        dtype of ``pos`` and ``norms``. Empty graphs retain these shapes and
        dtypes. Edge order may change with block size, while each attribute row
        always corresponds to the same column of ``edge_index``.

    Raises:
        ValueError: If tensor shapes, dtypes, devices, finite values, or scalar
            parameters fail validation.

    Notes:
        This is offline preprocessing; gradients are disabled. GPU inputs
        require a CPU coordinate copy for the tree and a device transfer of
        candidate indices for each block. Final concatenation temporarily
        duplicates output storage. The candidate budget does not cap the
        O(E) result; dense local neighborhoods can produce O(N**2) edges.
    """
    _validate_parameters(r_local, r_mid, r_global, k_mid, k_global, sigma,
                         chunk_size, max_candidate_pairs, workers)
    if pos.ndim != 2 or pos.shape[1] != 3 or norms.shape != pos.shape:
        raise ValueError("pos and norms must have matching shapes [N, 3]")
    if pos.device != norms.device:
        raise ValueError("pos and norms must be on the same device")
    if pos.dtype not in (torch.float32, torch.float64) or norms.dtype not in (
        torch.float32, torch.float64
    ):
        raise ValueError("pos and norms must use float32 or float64")
    if not bool(torch.isfinite(pos).all()) or not bool(torch.isfinite(norms).all()):
        raise ValueError("pos and norms must be finite")
    output_device = pos.device if output_device is None else torch.device(output_device)
    attr_dtype = torch.promote_types(pos.dtype, norms.dtype)
    if len(pos) < 2:
        return (torch.empty((2, 0), dtype=torch.long, device=output_device),
                torch.empty((0, 3), dtype=attr_dtype, device=output_device))

    coordinates = pos.detach().cpu().numpy()
    tree = cKDTree(coordinates)
    coordinates = tree.data  # Reuse the tree's contiguous float64 query storage.
    search_radius = r_global if k_global else (r_mid if k_mid else r_local)
    # KD-tree arithmetic is double precision; slightly over-query so tensor
    # rounding near a cutoff cannot hide a valid candidate. Filter below.
    search_radius *= 1 + 8 * torch.finfo(pos.dtype).eps
    edge_chunks, attr_chunks = [], []
    for start, num_rows, rows_np, neighbors_np in _candidate_batches(
        tree, coordinates, search_radius, chunk_size, max_candidate_pairs, workers
    ):
        candidates = torch.from_numpy(np.stack((rows_np, neighbors_np))).to(pos.device)
        rows, neighbors = candidates
        sources = rows + start
        delta = pos[sources] - pos[neighbors]
        distance_sq = delta.square().sum(dim=-1)
        del delta
        not_self = sources != neighbors
        local = torch.nonzero(not_self & (distance_sq < r_local ** 2)).flatten()
        selected = [local]
        # The same global-radius query supplies three disjoint intervals. FPS
        # preserves outgoing degree limits; no reverse edges are added here.
        for lower, upper, k in ((r_local, r_mid, k_mid), (r_mid, r_global, k_global)):
            if k == 0:
                continue
            annulus = torch.nonzero(
                not_self & (distance_sq >= lower ** 2) & (distance_sq < upper ** 2)
            ).flatten()
            chosen = _sample_neighbor_indices(
                pos, rows[annulus], neighbors[annulus], k, num_rows
            )
            selected.append(annulus[chosen])
        selected = torch.cat(selected)
        if selected.numel() == 0:
            continue
        src, dst = sources[selected], neighbors[selected]
        edge_index = torch.stack((src, dst))
        squared = distance_sq[selected]
        # Reuse filtered squared distances. Square roots are only needed for
        # the retained edges' distance feature, not during candidate selection.
        distance = squared.sqrt()
        normal_cos = (norms[src] * norms[dst]).sum(dim=-1)
        weight = torch.exp(-squared / (2 * sigma ** 2)) * (1 + normal_cos) / 2
        edge_attr = torch.stack((distance, normal_cos, weight), dim=-1)
        edge_chunks.append(edge_index.to(output_device))
        attr_chunks.append(edge_attr.to(output_device))

    if not edge_chunks:
        return (torch.empty((2, 0), dtype=torch.long, device=output_device),
                torch.empty((0, 3), dtype=attr_dtype, device=output_device))
    return torch.cat(edge_chunks, dim=1), torch.cat(attr_chunks, dim=0)


@torch.no_grad()
def process_file(raw_path: Union[str, os.PathLike],
                 out_dir: Union[str, os.PathLike], params: Dict[str, Any]) -> bool:
    """Load one surface, build its graph and save the training dictionary.

    Args:
        raw_path: Path to a trusted torch-saved dictionary containing ``pos``
            and ``norms`` [N, 3], node features ``x`` [N, F], and labels ``y``.
        out_dir: Existing output directory. The input basename is preserved.
            Existing output files are skipped without loading the input or
            comparing graph parameters; use a new directory when rebuilding.
        params: Required builder settings ``r_local``, ``r_mid``, ``r_global``,
            ``k_mid``, ``k_global`` and ``sigma``. Optional ``device`` defaults
            to 'auto' (CUDA if available, otherwise CPU). ``chunk_size``,
            ``max_candidate_pairs`` and ``workers`` use the builder defaults.

    Returns:
        True when the graph was saved or the destination already existed.
        Saved tensors reside on CPU. The node payload is preserved alongside
        ``edge_index`` and ``edge_attr``; optional ``pdb_id``, ``mode``,
        ``receptor_chains`` and ``ligand_chains`` retain their original values
        or receive the legacy defaults '', 'ligand', [], and [], respectively.

    Raises:
        Exceptions from loading, validation, graph construction or saving are
        propagated to the caller; a failed build does not return False.
    """
    out_path = os.path.join(out_dir, os.path.basename(raw_path))
    if os.path.exists(out_path):
        return True
    raw_data = torch.load(raw_path, map_location="cpu", weights_only=False)
    requested_device = params.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    # Keep features, labels and the original surface on CPU; only coordinates
    # and normals are needed by device-side geometry. Emit completed blocks
    # directly to CPU so the entire graph need not fit in GPU memory.
    edge_index, edge_attr = build_multiscale_graph(
        raw_data["pos"].to(device), raw_data["norms"].to(device),
        r_local=params["r_local"], r_mid=params["r_mid"], r_global=params["r_global"],
        k_mid=params["k_mid"], k_global=params["k_global"], sigma=params["sigma"],
        chunk_size=params.get("chunk_size", DEFAULT_CHUNK_SIZE),
        max_candidate_pairs=params.get("max_candidate_pairs", DEFAULT_MAX_CANDIDATE_PAIRS),
        workers=params.get("workers", 1), output_device="cpu",
    )
    graph_data = {
        "x": raw_data["x"],
        "pos": raw_data["pos"],
        "norms": raw_data["norms"],
        "y": raw_data["y"],
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pdb_id": raw_data.get("pdb_id", ""),
        "mode": raw_data.get("mode", "ligand"),
        "receptor_chains": raw_data.get("receptor_chains", []),
        "ligand_chains": raw_data.get("ligand_chains", []),
    }
    torch.save(graph_data, out_path)
    # Let the CUDA caching allocator reuse buffers for the next molecule.
    return True


def main() -> None:
    """Parse CLI arguments, build graphs serially, and report per-file errors.

    Arguments are read from ``sys.argv``; ``--help`` documents paths and graph
    settings. This entry point creates the output directory, processes sorted
    input .pt paths, and continues with the next file after a processing error.

    Returns:
        None. Graphs are written to disk and progress is printed to stdout.

    Raises:
        SystemExit: Argparse exits for ``--help`` or invalid CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description='Построение мультимасштабных графов с FPS'
    )
    parser.add_argument('--raw_dir', type=str, default='./processed/raw',
                        help='Папка с сырыми .pt файлами')
    parser.add_argument('--out_dir', type=str, default='./processed/graphs_multiscale',
                        help='Папка для сохранения графов')
    parser.add_argument('--r_local', type=float, default=2.0,
                        help='Локальный радиус (все рёбра)')
    parser.add_argument('--r_mid', type=float, default=4.0,
                        help='Средний радиус (для отбора центроидов)')
    parser.add_argument('--r_global', type=float, default=8.0,
                        help='Глобальный радиус (для отбора центроидов)')
    parser.add_argument('--k_mid', type=int, default=10,
                        help='Число центроидов на среднем масштабе')
    parser.add_argument('--k_global', type=int, default=10,
                        help='Число центроидов на глобальном масштабе')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Параметр затухания для edge_w')
    parser.add_argument('--n_files', type=int, default=None,
                        help='Число файлов для обработки (для теста)')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                        help='Устройство для геометрии и FPS (поиск соседей на CPU)')
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help='Максимум вершин в блоке поиска соседей')
    parser.add_argument('--max_candidate_pairs', type=int, default=DEFAULT_MAX_CANDIDATE_PAIRS,
                        help='Бюджет кандидатов в блоке; одно соседство не обрезается')
    parser.add_argument('--workers', type=int, default=1,
                        help='Потоки KD-tree; -1 использует все CPU')
    args = parser.parse_args()
    try:
        _validate_parameters(args.r_local, args.r_mid, args.r_global,
                             args.k_mid, args.k_global, args.sigma,
                             args.chunk_size, args.max_candidate_pairs, args.workers)
    except ValueError as exc:
        parser.error(str(exc))
    
    params = {
        'r_local': args.r_local,
        'r_mid': args.r_mid,
        'r_global': args.r_global,
        'k_mid': args.k_mid,
        'k_global': args.k_global,
        'sigma': args.sigma,
        'device': args.device,
        'chunk_size': args.chunk_size,
        'max_candidate_pairs': args.max_candidate_pairs,
        'workers': args.workers,
    }
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    raw_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.pt")))
    if args.n_files:
        raw_paths = raw_paths[:args.n_files]
    
    print(f"🔧 Параметры графа:")
    print(f"   Локальный радиус: {params['r_local']}Å (все рёбра)")
    print(f"   Средний радиус: {params['r_mid']}Å, центроидов: {params['k_mid']}")
    print(f"   Глобальный радиус: {params['r_global']}Å, центроидов: {params['k_global']}")
    print(f"   SIGMA: {params['sigma']}")
    print(f"📂 Найдено {len(raw_paths)} файлов")
    
    try:
        from tqdm import tqdm
    except ImportError:
        pass  # Progress bars are optional in the lightweight CPU environment.
    else:
        raw_paths = tqdm(raw_paths, desc="Построение графов")
    saved = 0
    for raw_path in raw_paths:
        try:
            if process_file(raw_path, args.out_dir, params):
                saved += 1
        except Exception as e:
            print(f"❌ Ошибка в {raw_path}: {e}")
    
    print(f"✅ Сохранено {saved} графов в {args.out_dir}")


if __name__ == '__main__':
    main()
