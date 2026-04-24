"""Atom tensor IO for PDB and dMaSIF-style NPY data.

External interfaces:
    ProteinAtomTensors: container for atom coordinates, types, radii and metadata.
    read_pdb_tensors: read a PDB file into torch tensors.
    load_structure_np: read a PDB file into dMaSIF-compatible NumPy arrays.
    save_protein_npy / load_protein_npy: save and load tensor sets as .npy files.
    convert_pdb_to_npy / convert_pdbs: convert one PDB file or a directory of PDBs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from Bio.PDB import PDBParser


# dMaSIF uses six atom channels in this exact order.
ELEMENTS: Tuple[str, ...] = ("C", "H", "O", "N", "S", "SE")
_ELEMENT_TO_INDEX: Dict[str, int] = {
    element: idx for idx, element in enumerate(ELEMENTS)
}
_INDEX_TO_ELEMENT: Dict[int, str] = {
    idx: element for element, idx in _ELEMENT_TO_INDEX.items()
}

# Van der Waals radii used by dMaSIF's atom-type smoothness code, converted
# from picometers to Angstroms.
ELEMENT_RADII: Dict[str, float] = {
    "C": 1.70,
    "H": 1.10,
    "O": 1.52,
    "N": 1.55,
    "S": 1.80,
    "SE": 1.90,
}

_FLOAT_FIELDS = {"atom_coords", "atom_types", "atom_radii", "occupancy", "b_factors"}
_LONG_FIELDS = {
    "atom_type_indices",
    "atom_serials",
    "residue_numbers",
    "chain_indices",
    "model_indices",
}
_STRING_FIELDS = {
    "atom_names",
    "residue_names",
    "chain_ids",
    "insertion_codes",
    "elements",
    "records",
}

_NPY_SUFFIXES: Dict[str, str] = {
    "atom_coords": "atomxyz",
    "atom_types": "atomtypes",
    "atom_radii": "atomradii",
    "atom_type_indices": "atomtype_indices",
    "atom_serials": "atomserials",
    "residue_numbers": "residue_numbers",
    "chain_indices": "chain_indices",
    "model_indices": "model_indices",
    "occupancy": "occupancy",
    "b_factors": "bfactors",
    "atom_names": "atomnames",
    "residue_names": "residue_names",
    "chain_ids": "chain_ids",
    "insertion_codes": "insertion_codes",
    "elements": "elements",
    "records": "records",
}


PathLike = Union[str, Path]

__all__ = [
    "ELEMENTS",
    "ELEMENT_RADII",
    "ProteinAtomTensors",
    "read_pdb_tensors",
    "load_structure_np",
    "save_protein_npy",
    "load_protein_npy",
    "convert_pdb_to_npy",
    "convert_pdbs",
]


@dataclass(frozen=True)
class ProteinAtomTensors(Mapping[str, Any]):
    """Atom-level tensors and metadata for a PDB structure.

    The core tensors mirror dMaSIF's atom input convention:
    ``atom_coords`` has shape ``(n_atoms, 3)``, ``atom_types`` is the
    ``(n_atoms, 6)`` one-hot encoding for ``C,H,O,N,S,SE``, and
    ``atom_radii`` has shape ``(n_atoms,)``.

    Attributes:
        atom_coords: Atom center coordinates, shape ``(n_atoms, 3)``.
        atom_types: One-hot atom element encoding in ``ELEMENTS`` order, shape
            ``(n_atoms, 6)``.
        atom_radii: Van der Waals radii in Angstroms, shape ``(n_atoms,)``.
        atom_type_indices: Integer atom type ids in ``ELEMENTS`` order, shape
            ``(n_atoms,)``.
        atom_serials: PDB atom serial numbers, shape ``(n_atoms,)``.
        residue_numbers: PDB residue sequence numbers, shape ``(n_atoms,)``.
        chain_indices: Contiguous integer chain ids, shape ``(n_atoms,)``.
        model_indices: Integer model ids from the PDB file, shape
            ``(n_atoms,)``.
        occupancy: Atom occupancy values, shape ``(n_atoms,)``.
        b_factors: Atom temperature-factor values, shape ``(n_atoms,)``.
        atom_names: PDB atom names, one string per atom.
        residue_names: PDB residue names, one string per atom.
        chain_ids: PDB chain ids, one string per atom.
        insertion_codes: PDB residue insertion codes, one string per atom.
        elements: Normalized chemical element labels, one string per atom.
        records: PDB record labels, either ``ATOM`` or ``HETATM``.
    """

    atom_coords: torch.Tensor
    atom_types: torch.Tensor
    atom_radii: torch.Tensor
    atom_type_indices: torch.Tensor
    atom_serials: torch.Tensor
    residue_numbers: torch.Tensor
    chain_indices: torch.Tensor
    model_indices: torch.Tensor
    occupancy: torch.Tensor
    b_factors: torch.Tensor
    atom_names: Tuple[str, ...] = ()
    residue_names: Tuple[str, ...] = ()
    chain_ids: Tuple[str, ...] = ()
    insertion_codes: Tuple[str, ...] = ()
    elements: Tuple[str, ...] = ()
    records: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate tensor and metadata shapes after dataclass construction."""

        num_atoms = int(self.atom_coords.shape[0])
        if self.atom_coords.ndim != 2 or self.atom_coords.shape[1] != 3:
            raise ValueError("atom_coords must have shape (n_atoms, 3)")
        if self.atom_types.shape != (num_atoms, len(ELEMENTS)):
            raise ValueError("atom_types must have shape (n_atoms, 6)")
        for name in (
            "atom_radii",
            "atom_type_indices",
            "atom_serials",
            "residue_numbers",
            "chain_indices",
            "model_indices",
            "occupancy",
            "b_factors",
        ):
            value = getattr(self, name)
            if value.ndim != 1 or int(value.shape[0]) != num_atoms:
                raise ValueError(f"{name} must have shape (n_atoms,)")
        for name in _STRING_FIELDS:
            value = getattr(self, name)
            if value and len(value) != num_atoms:
                raise ValueError(f"{name} must have one value per atom")

    @property
    def xyz(self) -> torch.Tensor:
        """dMaSIF-compatible alias for atom coordinates."""

        return self.atom_coords

    @property
    def types(self) -> torch.Tensor:
        """dMaSIF-compatible alias for one-hot atom types."""

        return self.atom_types

    @property
    def radii(self) -> torch.Tensor:
        """dMaSIF-compatible alias for atom radii."""

        return self.atom_radii

    def __getitem__(self, key: str) -> Any:
        """Return a field by mapping-style key lookup.

        Args:
            key: Field name. The aliases ``xyz``, ``types`` and ``radii`` map
                to ``atom_coords``, ``atom_types`` and ``atom_radii``.

        Returns:
            The requested tensor or metadata tuple.
        """

        aliases = {
            "xyz": "atom_coords",
            "types": "atom_types",
            "radii": "atom_radii",
        }
        return getattr(self, aliases.get(key, key))

    def __iter__(self) -> Iterator[str]:
        """Iterate over mapping-style field names.

        Returns:
            Iterator over keys returned by :meth:`as_dict`.
        """

        return iter(self.as_dict())

    def __len__(self) -> int:
        """Return the number of mapping-style fields.

        Returns:
            Number of keys returned by :meth:`as_dict`.
        """

        return len(self.as_dict())

    def as_dict(self) -> Dict[str, Any]:
        """Return all tensor and metadata fields as a dictionary.

        Returns:
            Dictionary with the canonical ``ProteinAtomTensors`` field names.
        """

        return {
            "atom_coords": self.atom_coords,
            "atom_types": self.atom_types,
            "atom_radii": self.atom_radii,
            "atom_type_indices": self.atom_type_indices,
            "atom_serials": self.atom_serials,
            "residue_numbers": self.residue_numbers,
            "chain_indices": self.chain_indices,
            "model_indices": self.model_indices,
            "occupancy": self.occupancy,
            "b_factors": self.b_factors,
            "atom_names": self.atom_names,
            "residue_names": self.residue_names,
            "chain_ids": self.chain_ids,
            "insertion_codes": self.insertion_codes,
            "elements": self.elements,
            "records": self.records,
        }

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "ProteinAtomTensors":
        """Move tensor fields to a device and optionally cast floating tensors.

        Args:
            device: Target device for all tensor fields. If ``None``, the
                current device is preserved.
            dtype: Target dtype for floating-point tensor fields. Integer
                tensor fields keep their integer dtype.

        Returns:
            New ``ProteinAtomTensors`` object with moved tensor fields and
            unchanged string metadata.
        """

        values: Dict[str, Any] = {}
        for name, value in self.as_dict().items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    values[name] = value.to(device=device, dtype=dtype or value.dtype)
                else:
                    values[name] = value.to(device=device)
            else:
                values[name] = value
        return ProteinAtomTensors(**values)


def read_pdb_tensors(
    pdb_path: PathLike,
    *,
    center: bool = False,
    chain_ids: Optional[Union[str, Sequence[str]]] = None,
    include_hetatm: bool = True,
    include_hydrogens: bool = True,
    accepted_altlocs: Sequence[str] = ("", " ", "A", "1"),
    unknown_elements: str = "error",
    dtype: torch.dtype = torch.float32,
) -> ProteinAtomTensors:
    """Read a PDB file and return atom coordinates, types, radii and metadata.

    The PDB structure is parsed with Biopython. Atom elements are normalized to
    the six dMaSIF channels listed in ``ELEMENTS``.

    Args:
        pdb_path: Path to a ``.pdb`` file.
        center: If true, subtract the mean atom coordinate.
        chain_ids: Optional chain id or sequence of chain ids to keep.
        include_hetatm: Whether to keep ``HETATM`` records.
        include_hydrogens: Whether to keep hydrogen atoms.
        accepted_altlocs: Alternate-location identifiers to keep.
        unknown_elements: ``"error"`` to fail on elements outside dMaSIF's
            six channels, or ``"skip"`` to ignore those atoms.
        dtype: Floating dtype for returned tensors.

    Returns:
        ``ProteinAtomTensors`` with atom coordinates, one-hot types, radii and
        per-atom metadata.
    """

    if unknown_elements not in {"error", "skip"}:
        raise ValueError("unknown_elements must be 'error' or 'skip'")

    chain_filter = _normalize_chain_filter(chain_ids)
    accepted_altlocs_set = {altloc.strip() for altloc in accepted_altlocs}

    coords = []
    type_indices = []
    radii = []
    serials = []
    residue_numbers = []
    chain_indices = []
    model_indices = []
    occupancy = []
    b_factors = []
    atom_names = []
    residue_names = []
    chain_id_values = []
    insertion_codes = []
    elements = []
    records = []

    chain_to_index: Dict[str, int] = {}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(pdb_path))

    for model_index, model in enumerate(structure):
        for chain in model:
            chain_id = chain.id.strip()
            if chain_filter is not None and chain_id not in chain_filter:
                continue

            if chain_id not in chain_to_index:
                chain_to_index[chain_id] = len(chain_to_index)

            for residue in chain.get_unpacked_list():
                hetero_flag, residue_number, insertion_code = residue.id
                record = "ATOM" if hetero_flag == " " else "HETATM"
                if record == "HETATM" and not include_hetatm:
                    continue

                for atom in residue.get_unpacked_list():
                    altloc = atom.get_altloc().strip()
                    if altloc not in accepted_altlocs_set:
                        continue

                    atom_name = atom.get_name().strip()
                    element = _normalize_element(atom.element, atom_name)
                    if not include_hydrogens and element == "H":
                        continue
                    if element not in _ELEMENT_TO_INDEX:
                        if unknown_elements == "skip":
                            continue
                        atom_context = (
                            f"{residue.get_resname().strip()} {chain_id}"
                            f"{residue_number} atom {atom_name}"
                        )
                        raise ValueError(
                            f"Unsupported element {element!r} for {atom_context}; "
                            f"supported elements are {', '.join(ELEMENTS)}"
                        )

                    type_idx = _ELEMENT_TO_INDEX[element]
                    coords.append(tuple(float(value) for value in atom.get_coord()))
                    type_indices.append(type_idx)
                    radii.append(ELEMENT_RADII[element])
                    serials.append(atom.get_serial_number() or len(serials) + 1)
                    residue_numbers.append(int(residue_number))
                    chain_indices.append(chain_to_index[chain_id])
                    model_indices.append(model_index)
                    atom_occupancy = atom.get_occupancy()
                    occupancy.append(
                        1.0 if atom_occupancy is None else float(atom_occupancy)
                    )
                    b_factors.append(float(atom.get_bfactor()))
                    atom_names.append(atom_name)
                    residue_names.append(residue.get_resname().strip())
                    chain_id_values.append(chain_id)
                    insertion_codes.append(insertion_code.strip())
                    elements.append(element)
                    records.append(record)

    if not coords:
        raise ValueError(f"No supported atoms found in {pdb_path}")

    coords_array = np.asarray(coords, dtype=np.float32)
    if center:
        coords_array = coords_array - coords_array.mean(axis=0, keepdims=True)

    type_index_array = np.asarray(type_indices, dtype=np.int64)
    types_array = np.zeros((type_index_array.shape[0], len(ELEMENTS)), dtype=np.float32)
    types_array[np.arange(type_index_array.shape[0]), type_index_array] = 1.0

    return ProteinAtomTensors(
        atom_coords=torch.as_tensor(coords_array, dtype=dtype),
        atom_types=torch.as_tensor(types_array, dtype=dtype),
        atom_radii=torch.as_tensor(radii, dtype=dtype),
        atom_type_indices=torch.as_tensor(type_index_array, dtype=torch.long),
        atom_serials=torch.as_tensor(serials, dtype=torch.long),
        residue_numbers=torch.as_tensor(residue_numbers, dtype=torch.long),
        chain_indices=torch.as_tensor(chain_indices, dtype=torch.long),
        model_indices=torch.as_tensor(model_indices, dtype=torch.long),
        occupancy=torch.as_tensor(occupancy, dtype=dtype),
        b_factors=torch.as_tensor(b_factors, dtype=dtype),
        atom_names=tuple(atom_names),
        residue_names=tuple(residue_names),
        chain_ids=tuple(chain_id_values),
        insertion_codes=tuple(insertion_codes),
        elements=tuple(elements),
        records=tuple(records),
    )


def _load_pdb_tensors(pdb_path: PathLike, **kwargs: Any) -> ProteinAtomTensors:
    """Alias for :func:`read_pdb_tensors`.

    Args:
        pdb_path: Path to a ``.pdb`` file.
        **kwargs: Keyword arguments forwarded to :func:`read_pdb_tensors`.

    Returns:
        ``ProteinAtomTensors`` loaded from ``pdb_path``.
    """

    return read_pdb_tensors(pdb_path, **kwargs)


def load_structure_np(
    fname: PathLike,
    center: bool = False,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """dMaSIF-compatible PDB loader that returns NumPy arrays.

    Returns at least ``{"xyz": (n, 3), "types": (n, 6)}``, matching
    ``data_preprocessing/convert_pdb2npy.py`` in dMaSIF, with additional
    atom-level arrays for radii and metadata.

    Args:
        fname: Path to a ``.pdb`` file.
        center: If true, subtract the mean atom coordinate.
        **kwargs: Keyword arguments forwarded to :func:`read_pdb_tensors`.

    Returns:
        Dictionary with dMaSIF keys ``xyz`` and ``types`` plus ``radii`` and
        additional atom metadata arrays.
    """

    tensors = read_pdb_tensors(fname, center=center, **kwargs)
    arrays = _tensors_to_numpy(tensors)
    return {
        "xyz": arrays["atom_coords"],
        "types": arrays["atom_types"],
        "radii": arrays["atom_radii"],
        "atom_type_indices": arrays["atom_type_indices"],
        "atom_serials": arrays["atom_serials"],
        "residue_numbers": arrays["residue_numbers"],
        "chain_indices": arrays["chain_indices"],
        "model_indices": arrays["model_indices"],
        "occupancy": arrays["occupancy"],
        "b_factors": arrays["b_factors"],
        "atom_names": arrays["atom_names"],
        "residue_names": arrays["residue_names"],
        "chain_ids": arrays["chain_ids"],
        "insertion_codes": arrays["insertion_codes"],
        "elements": arrays["elements"],
        "records": arrays["records"],
    }


def save_protein_npy(
    tensors: Union[ProteinAtomTensors, Mapping[str, Any]],
    data_dir: PathLike,
    pdb_id: str,
    *,
    include_metadata: bool = True,
) -> None:
    """Save atom tensors as dMaSIF-style ``.npy`` files.

    The core dMaSIF files are ``<pdb_id>_atomxyz.npy`` and
    ``<pdb_id>_atomtypes.npy``. This function also writes radii and numeric
    metadata, plus string metadata when ``include_metadata`` is true.

    Args:
        tensors: ``ProteinAtomTensors`` or mapping with equivalent keys. The
            dMaSIF aliases ``xyz``, ``types`` and ``radii`` are accepted.
        data_dir: Directory where ``.npy`` files will be written.
        pdb_id: Filename prefix used for all saved arrays.
        include_metadata: If true, save string metadata arrays in addition to
            numeric tensors.

    Returns:
        ``None``. Files are written to ``data_dir``.
    """

    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protein = _ensure_protein_tensors(tensors)
    arrays = _tensors_to_numpy(protein)

    for field, suffix in _NPY_SUFFIXES.items():
        if field in _STRING_FIELDS and not include_metadata:
            continue
        value = arrays[field]
        if field in _STRING_FIELDS and value.size == 0:
            continue
        np.save(output_dir / f"{pdb_id}_{suffix}.npy", value)


def load_protein_npy(
    pdb_id: str,
    data_dir: PathLike,
    *,
    center: bool = False,
    dtype: torch.dtype = torch.float32,
) -> ProteinAtomTensors:
    """Load atom tensors saved by :func:`save_protein_npy` or dMaSIF.

    Args:
        pdb_id: Filename prefix used when the arrays were saved.
        data_dir: Directory containing ``<pdb_id>_atomxyz.npy`` and
            ``<pdb_id>_atomtypes.npy``.
        center: If true, subtract the mean atom coordinate after loading.
        dtype: Floating dtype for returned tensor fields.

    Returns:
        ``ProteinAtomTensors`` reconstructed from the saved arrays. Metadata
        missing from older dMaSIF-style exports is filled with defaults.
    """

    input_dir = Path(data_dir)
    required = ("atom_coords", "atom_types")
    arrays: Dict[str, np.ndarray] = {}
    for field in required:
        arrays[field] = _load_required_npy(input_dir, pdb_id, _NPY_SUFFIXES[field])

    num_atoms = int(arrays["atom_coords"].shape[0])
    optional_defaults = _default_optional_arrays(arrays["atom_types"], num_atoms)
    for field, default in optional_defaults.items():
        suffix = _NPY_SUFFIXES[field]
        path = input_dir / f"{pdb_id}_{suffix}.npy"
        arrays[field] = np.load(path, allow_pickle=False) if path.exists() else default

    if center:
        atom_coords = arrays["atom_coords"].astype(np.float32, copy=True)
        atom_coords -= atom_coords.mean(axis=0, keepdims=True)
        arrays["atom_coords"] = atom_coords

    return _numpy_to_tensors(arrays, dtype=dtype)


def _save_tensors_npy(
    tensors: Union[ProteinAtomTensors, Mapping[str, Any]],
    data_dir: PathLike,
    prefix: str,
    *,
    include_metadata: bool = True,
) -> None:
    """Alias for :func:`save_protein_npy` with a neutral name.

    Args:
        tensors: ``ProteinAtomTensors`` or mapping with equivalent keys.
        data_dir: Directory where ``.npy`` files will be written.
        prefix: Filename prefix used for all saved arrays.
        include_metadata: If true, save string metadata arrays.

    Returns:
        ``None``. Files are written to ``data_dir``.
    """

    save_protein_npy(tensors, data_dir, prefix, include_metadata=include_metadata)


def _load_tensors_npy(
    data_dir: PathLike,
    prefix: str,
    *,
    center: bool = False,
    dtype: torch.dtype = torch.float32,
) -> ProteinAtomTensors:
    """Alias for :func:`load_protein_npy` with a neutral name.

    Args:
        data_dir: Directory containing saved ``.npy`` files.
        prefix: Filename prefix used when the arrays were saved.
        center: If true, subtract the mean atom coordinate after loading.
        dtype: Floating dtype for returned tensor fields.

    Returns:
        ``ProteinAtomTensors`` reconstructed from saved ``.npy`` files.
    """

    return load_protein_npy(prefix, data_dir, center=center, dtype=dtype)


def convert_pdb_to_npy(
    pdb_path: PathLike,
    npy_dir: PathLike,
    *,
    pdb_id: Optional[str] = None,
    center: bool = False,
    include_metadata: bool = True,
    **kwargs: Any,
) -> ProteinAtomTensors:
    """Read one PDB file, save tensors to ``.npy`` files, and return them.

    Args:
        pdb_path: Path to the input ``.pdb`` file.
        npy_dir: Directory where ``.npy`` files will be written.
        pdb_id: Optional filename prefix. If omitted, ``pdb_path.stem`` is used.
        center: If true, subtract the mean atom coordinate before saving.
        include_metadata: If true, save string metadata arrays.
        **kwargs: Keyword arguments forwarded to :func:`read_pdb_tensors`.

    Returns:
        ``ProteinAtomTensors`` parsed from ``pdb_path``.
    """

    path = Path(pdb_path)
    protein_id = pdb_id or path.stem
    tensors = read_pdb_tensors(path, center=center, **kwargs)
    save_protein_npy(tensors, npy_dir, protein_id, include_metadata=include_metadata)
    return tensors


def convert_pdbs(
    pdb_dir: PathLike,
    npy_dir: PathLike,
    *,
    include_metadata: bool = True,
    **kwargs: Any,
) -> None:
    """Convert all ``*.pdb`` files in a directory to dMaSIF-style ``.npy`` files.

    Args:
        pdb_dir: Directory containing input ``.pdb`` files.
        npy_dir: Directory where converted arrays will be written.
        include_metadata: If true, save string metadata arrays.
        **kwargs: Keyword arguments forwarded to :func:`read_pdb_tensors`.

    Returns:
        ``None``. One set of ``.npy`` files is written per PDB file.
    """

    output_dir = Path(npy_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pdb_path in sorted(Path(pdb_dir).glob("*.pdb")):
        convert_pdb_to_npy(
            pdb_path,
            output_dir,
            pdb_id=pdb_path.stem,
            include_metadata=include_metadata,
            **kwargs,
        )


def _tensors_to_numpy(tensors: ProteinAtomTensors) -> Dict[str, np.ndarray]:
    """Convert a :class:`ProteinAtomTensors` object to NumPy arrays.

    Args:
        tensors: Atom tensor container to convert.

    Returns:
        Dictionary with the same canonical field names as
        ``ProteinAtomTensors.as_dict`` and NumPy array values.
    """

    arrays: Dict[str, np.ndarray] = {}
    for name, value in tensors.as_dict().items():
        if isinstance(value, torch.Tensor):
            arrays[name] = value.detach().cpu().numpy()
        else:
            arrays[name] = np.asarray(value, dtype=str)
    return arrays


def _numpy_to_tensors(
    arrays: Mapping[str, np.ndarray],
    *,
    dtype: torch.dtype = torch.float32,
) -> ProteinAtomTensors:
    """Build :class:`ProteinAtomTensors` from NumPy arrays.

    Args:
        arrays: Mapping containing all canonical ``ProteinAtomTensors`` fields
            as NumPy arrays.
        dtype: Floating dtype for returned tensor fields.

    Returns:
        ``ProteinAtomTensors`` object with tensor fields converted from NumPy.
    """

    values: Dict[str, Any] = {}
    for field in _FLOAT_FIELDS:
        values[field] = torch.as_tensor(arrays[field], dtype=dtype)
    for field in _LONG_FIELDS:
        values[field] = torch.as_tensor(arrays[field], dtype=torch.long)
    for field in _STRING_FIELDS:
        values[field] = tuple(str(value) for value in arrays[field].tolist())
    return ProteinAtomTensors(**values)


def _ensure_protein_tensors(
    tensors: Union[ProteinAtomTensors, Mapping[str, Any]],
    *,
    dtype: torch.dtype = torch.float32,
) -> ProteinAtomTensors:
    """Normalize a mapping or ``ProteinAtomTensors`` to ``ProteinAtomTensors``.

    Args:
        tensors: Existing ``ProteinAtomTensors`` object or mapping. The aliases
            ``xyz``, ``types`` and ``radii`` are accepted.
        dtype: Floating dtype for returned tensor fields.

    Returns:
        Validated ``ProteinAtomTensors`` object. Missing optional metadata is
        filled with deterministic defaults.
    """

    if isinstance(tensors, ProteinAtomTensors):
        return tensors

    source = dict(tensors)
    if "xyz" in source and "atom_coords" not in source:
        source["atom_coords"] = source["xyz"]
    if "types" in source and "atom_types" not in source:
        source["atom_types"] = source["types"]
    if "radii" in source and "atom_radii" not in source:
        source["atom_radii"] = source["radii"]

    atom_coords = _as_float_tensor(source["atom_coords"], dtype)
    atom_types = _as_float_tensor(source["atom_types"], dtype)
    num_atoms = int(atom_coords.shape[0])
    if "atom_type_indices" in source:
        atom_type_indices = _as_long_tensor(source["atom_type_indices"])
    else:
        atom_type_indices = atom_types.argmax(dim=1).to(torch.long)
    if "atom_radii" in source:
        atom_radii = _as_float_tensor(source["atom_radii"], dtype).reshape(-1)
    else:
        atom_radii = _radii_from_types(atom_types, dtype=dtype)

    values: Dict[str, Any] = {
        "atom_coords": atom_coords,
        "atom_types": atom_types,
        "atom_radii": atom_radii,
        "atom_type_indices": atom_type_indices,
        "atom_serials": _as_long_tensor(
            source.get("atom_serials", np.arange(1, num_atoms + 1))
        ),
        "residue_numbers": _as_long_tensor(
            source.get("residue_numbers", np.zeros(num_atoms, dtype=np.int64))
        ),
        "chain_indices": _as_long_tensor(
            source.get("chain_indices", np.zeros(num_atoms, dtype=np.int64))
        ),
        "model_indices": _as_long_tensor(
            source.get("model_indices", np.zeros(num_atoms, dtype=np.int64))
        ),
        "occupancy": _as_float_tensor(
            source.get("occupancy", np.ones(num_atoms, dtype=np.float32)),
            dtype,
        ),
        "b_factors": _as_float_tensor(
            source.get("b_factors", np.zeros(num_atoms, dtype=np.float32)),
            dtype,
        ),
    }

    for field in _STRING_FIELDS:
        raw_value = source.get(field, ())
        values[field] = tuple(str(value) for value in raw_value)

    return ProteinAtomTensors(**values)


def _radii_from_types(
    atom_types: Union[np.ndarray, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute atom radii from dMaSIF one-hot atom-type channels.

    Args:
        atom_types: One-hot atom element encoding, shape ``(n_atoms, 6)``.
        dtype: Floating dtype used when ``atom_types`` is not already a tensor.

    Returns:
        Atom radii in Angstroms, shape ``(n_atoms,)``.
    """

    atom_types_tensor = _as_float_tensor(atom_types, dtype)
    if atom_types_tensor.ndim != 2 or atom_types_tensor.shape[1] != len(ELEMENTS):
        raise ValueError("atom_types must have shape (n_atoms, 6)")
    radii = torch.tensor(
        [ELEMENT_RADII[element] for element in ELEMENTS],
        dtype=atom_types_tensor.dtype,
        device=atom_types_tensor.device,
    )
    return atom_types_tensor @ radii


def _default_optional_arrays(
    atom_types: np.ndarray,
    num_atoms: int,
) -> Dict[str, np.ndarray]:
    """Build fallback arrays for optional metadata missing from ``.npy`` files.

    Args:
        atom_types: One-hot atom element encoding, shape ``(n_atoms, 6)``.
        num_atoms: Number of atoms represented by the loaded arrays.

    Returns:
        Dictionary of default arrays for all optional ``ProteinAtomTensors``
        fields.
    """

    atom_type_indices = atom_types.argmax(axis=1).astype(np.int64)
    radii = np.asarray(
        [ELEMENT_RADII[_INDEX_TO_ELEMENT[int(idx)]] for idx in atom_type_indices],
        dtype=np.float32,
    )
    elements = np.asarray(
        [_INDEX_TO_ELEMENT[int(idx)] for idx in atom_type_indices],
        dtype=str,
    )

    return {
        "atom_radii": radii,
        "atom_type_indices": atom_type_indices,
        "atom_serials": np.arange(1, num_atoms + 1, dtype=np.int64),
        "residue_numbers": np.zeros(num_atoms, dtype=np.int64),
        "chain_indices": np.zeros(num_atoms, dtype=np.int64),
        "model_indices": np.zeros(num_atoms, dtype=np.int64),
        "occupancy": np.ones(num_atoms, dtype=np.float32),
        "b_factors": np.zeros(num_atoms, dtype=np.float32),
        "atom_names": np.asarray((), dtype=str),
        "residue_names": np.asarray((), dtype=str),
        "chain_ids": np.asarray((), dtype=str),
        "insertion_codes": np.asarray((), dtype=str),
        "elements": elements,
        "records": np.asarray((), dtype=str),
    }


def _load_required_npy(data_dir: Path, pdb_id: str, suffix: str) -> np.ndarray:
    """Load a required ``.npy`` array and raise if it is missing.

    Args:
        data_dir: Directory containing saved arrays.
        pdb_id: Filename prefix used when arrays were saved.
        suffix: Field-specific filename suffix without ``.npy``.

    Returns:
        Loaded NumPy array.
    """

    path = data_dir / f"{pdb_id}_{suffix}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def _as_float_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Convert a value to a floating-point tensor.

    Args:
        value: Array-like value to convert.
        dtype: Target floating-point dtype.

    Returns:
        Tensor created with ``torch.as_tensor``.
    """

    return torch.as_tensor(value, dtype=dtype)


def _as_long_tensor(value: Any) -> torch.Tensor:
    """Convert a value to a flattened ``torch.long`` tensor.

    Args:
        value: Array-like value to convert.

    Returns:
        One-dimensional integer tensor.
    """

    return torch.as_tensor(value, dtype=torch.long).reshape(-1)


def _normalize_chain_filter(
    chain_ids: Optional[Union[str, Sequence[str]]],
) -> Optional[set]:
    """Normalize user-provided chain filters to a set of chain ids.

    Args:
        chain_ids: ``None``, a single chain-id string, or a sequence of
            chain-id values.

    Returns:
        ``None`` when no filtering is requested, otherwise a set of stripped
        chain-id strings.
    """

    if chain_ids is None:
        return None
    if isinstance(chain_ids, str):
        if len(chain_ids) <= 1:
            return {chain_ids.strip()}
        return {chain_id.strip() for chain_id in chain_ids}
    return {str(chain_id).strip() for chain_id in chain_ids}


def _normalize_element(element: str, atom_name: str) -> str:
    """Normalize an element label using the atom name as a fallback.

    Args:
        element: Element label reported by Biopython.
        atom_name: PDB atom name used when the element label is empty.

    Returns:
        Uppercase element label compatible with dMaSIF channels when possible.
    """

    element = element.strip().upper()
    if element:
        return element

    cleaned = "".join(ch for ch in atom_name.strip().upper() if ch.isalpha())
    if not cleaned:
        return ""
    if len(cleaned) >= 2 and cleaned[:2] in _ELEMENT_TO_INDEX:
        return cleaned[:2]
    return cleaned[:1]
