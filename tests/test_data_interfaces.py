from pathlib import Path

import numpy as np
import pytest
import torch

from data import (
    ELEMENTS,
    ELEMENT_RADII,
    ProteinAtomTensors,
    _as_float_tensor,
    _as_long_tensor,
    _default_optional_arrays,
    _ensure_protein_tensors,
    _load_required_npy,
    _load_pdb_tensors,
    _load_tensors_npy,
    _numpy_to_tensors,
    _normalize_chain_filter,
    _normalize_element,
    _radii_from_types,
    _save_tensors_npy,
    _tensors_to_numpy,
    convert_pdb_to_npy,
    convert_pdbs,
    load_protein_npy,
    load_structure_np,
    read_pdb_tensors,
    save_protein_npy,
)


TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
PDB_DIR = TEST_DATA_DIR / "pdb"
NPY_DIR = TEST_DATA_DIR / "npy"
PDB_IDS = ("2PQ2_B", "4FT4_Q", "4Q6I_J")
EXPECTED_ATOM_COUNTS = {
    "2PQ2_B": 55,
    "4FT4_Q": 58,
    "4Q6I_J": 24,
}
EXPECTED_CHAINS = {
    "2PQ2_B": "B",
    "4FT4_Q": "Q",
    "4Q6I_J": "J",
}


def _one_hot(type_indices):
    atom_types = torch.zeros((len(type_indices), len(ELEMENTS)), dtype=torch.float32)
    atom_types[torch.arange(len(type_indices)), torch.tensor(type_indices)] = 1.0
    return atom_types


def _toy_protein() -> ProteinAtomTensors:
    atom_types = _one_hot([0, 3])
    return ProteinAtomTensors(
        atom_coords=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        atom_types=atom_types,
        atom_radii=torch.tensor([ELEMENT_RADII["C"], ELEMENT_RADII["N"]]),
        atom_type_indices=torch.tensor([0, 3], dtype=torch.long),
        atom_serials=torch.tensor([10, 11], dtype=torch.long),
        residue_numbers=torch.tensor([1, 1], dtype=torch.long),
        chain_indices=torch.tensor([0, 0], dtype=torch.long),
        model_indices=torch.tensor([0, 0], dtype=torch.long),
        occupancy=torch.tensor([1.0, 0.5]),
        b_factors=torch.tensor([2.0, 3.0]),
        atom_names=("C", "N"),
        residue_names=("GLY", "GLY"),
        chain_ids=("A", "A"),
        insertion_codes=("", ""),
        elements=("C", "N"),
        records=("ATOM", "ATOM"),
    )


def _pdb_atom_line(
    serial: int,
    name: str,
    resname: str,
    chain_id: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    element: str,
    *,
    record: str = "ATOM",
    altloc: str = " ",
    occupancy: float = 1.0,
    b_factor: float = 0.0,
    insertion_code: str = " ",
) -> str:
    return (
        f"{record:<6}{serial:5d} {name:>4}{altloc:1}{resname:>3} "
        f"{chain_id:1}{residue_number:4d}{insertion_code:1}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}"
        f"          {element:>2}\n"
    )


def _write_pdb(path, lines):
    path.write_text("".join(lines) + "TER\nEND\n")
    return path


def test_protein_atom_tensors_mapping_aliases_and_to() -> None:
    protein = read_pdb_tensors(PDB_DIR / "4Q6I_J.pdb")

    assert protein["xyz"] is protein.atom_coords
    assert protein["types"] is protein.atom_types
    assert protein["radii"] is protein.atom_radii
    assert set(iter(protein)) == set(protein.as_dict())
    assert len(protein) == len(protein.as_dict())

    moved = protein.to(dtype=torch.float64)

    assert moved.atom_coords.dtype == torch.float64
    assert moved.atom_types.dtype == torch.float64
    assert moved.atom_serials.dtype == torch.long
    assert moved.atom_names == protein.atom_names


def test_protein_atom_tensors_validates_shapes() -> None:
    values = _toy_protein().as_dict()
    values["atom_coords"] = torch.zeros((2, 2))

    with pytest.raises(ValueError, match="atom_coords"):
        ProteinAtomTensors(**values)

    values = _toy_protein().as_dict()
    values["atom_names"] = ("C",)

    with pytest.raises(ValueError, match="atom_names"):
        ProteinAtomTensors(**values)


@pytest.mark.parametrize("pdb_id", PDB_IDS)
def test_read_pdb_tensors_returns_dmasif_atom_tensors_for_fixture_data(
    pdb_id: str,
) -> None:
    protein = read_pdb_tensors(PDB_DIR / f"{pdb_id}.pdb")

    assert protein.atom_coords.shape == (EXPECTED_ATOM_COUNTS[pdb_id], 3)
    assert protein.atom_types.shape == (EXPECTED_ATOM_COUNTS[pdb_id], len(ELEMENTS))
    assert protein.atom_radii.shape == (EXPECTED_ATOM_COUNTS[pdb_id],)
    assert protein.atom_names[:3] == ("N", "CA", "C")
    assert set(protein.chain_ids) == {EXPECTED_CHAINS[pdb_id]}
    assert protein.elements[0] == "N"
    assert torch.equal(protein.atom_types[0], torch.tensor([0, 0, 0, 1, 0, 0]))
    assert torch.isclose(protein.atom_radii[0], torch.tensor(ELEMENT_RADII["N"]))
    assert torch.allclose(protein.atom_types.sum(dim=1), torch.ones(len(protein.elements)))


def test_read_pdb_tensors_can_filter_chain_and_hydrogens() -> None:
    protein = read_pdb_tensors(
        PDB_DIR / "2PQ2_B.pdb",
        chain_ids="B",
        include_hydrogens=False,
    )

    assert set(protein.chain_ids) == {"B"}
    assert "H" not in set(protein.elements)


def test_read_pdb_tensors_centers_atoms_and_excludes_hetatm(tmp_path) -> None:
    pdb_path = _write_pdb(
        tmp_path / "small.pdb",
        [
            _pdb_atom_line(1, "C", "GLY", "A", 1, 0.0, 0.0, 0.0, "C"),
            _pdb_atom_line(2, "N", "GLY", "A", 1, 2.0, 0.0, 0.0, "N"),
            _pdb_atom_line(
                3,
                "O",
                "HOH",
                "A",
                2,
                10.0,
                0.0,
                0.0,
                "O",
                record="HETATM",
            ),
        ],
    )

    protein = read_pdb_tensors(
        pdb_path,
        center=True,
        include_hetatm=False,
        dtype=torch.float64,
    )

    assert protein.atom_coords.dtype == torch.float64
    assert torch.allclose(
        protein.atom_coords,
        torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert protein.records == ("ATOM", "ATOM")


def test_read_pdb_tensors_filters_alternate_locations(tmp_path) -> None:
    pdb_path = _write_pdb(
        tmp_path / "altloc.pdb",
        [
            _pdb_atom_line(1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C", altloc="A"),
            _pdb_atom_line(2, "CA", "GLY", "A", 1, 1.0, 0.0, 0.0, "C", altloc="B"),
        ],
    )

    protein = read_pdb_tensors(pdb_path, accepted_altlocs=("B",))

    assert protein.atom_names == ("CA",)
    assert torch.allclose(protein.atom_coords[0], torch.tensor([1.0, 0.0, 0.0]))


def test_read_pdb_tensors_handles_unknown_elements(tmp_path) -> None:
    pdb_path = _write_pdb(
        tmp_path / "unknown.pdb",
        [
            _pdb_atom_line(1, "C", "GLY", "A", 1, 0.0, 0.0, 0.0, "C"),
            _pdb_atom_line(
                2,
                "ZN",
                "ZN",
                "A",
                2,
                1.0,
                0.0,
                0.0,
                "ZN",
                record="HETATM",
            ),
        ],
    )

    with pytest.raises(ValueError, match="Unsupported element"):
        read_pdb_tensors(pdb_path)

    skipped = read_pdb_tensors(pdb_path, unknown_elements="skip")
    assert skipped.elements == ("C",)

    with pytest.raises(ValueError, match="unknown_elements"):
        read_pdb_tensors(pdb_path, unknown_elements="drop")


def test_load_pdb_tensors_alias_matches_read_pdb_tensors() -> None:
    protein = _load_pdb_tensors(PDB_DIR / "4Q6I_J.pdb", include_hydrogens=False)
    direct = read_pdb_tensors(PDB_DIR / "4Q6I_J.pdb", include_hydrogens=False)

    assert torch.allclose(protein.atom_coords, direct.atom_coords)
    assert torch.equal(protein.atom_types, direct.atom_types)
    assert protein.atom_names[:5] == direct.atom_names[:5]


@pytest.mark.parametrize("pdb_id", PDB_IDS)
def test_load_structure_np_matches_dmasif_keys_for_fixture_data(pdb_id: str) -> None:
    protein = load_structure_np(PDB_DIR / f"{pdb_id}.pdb")
    parsed = read_pdb_tensors(PDB_DIR / f"{pdb_id}.pdb")

    assert protein["xyz"].shape == (EXPECTED_ATOM_COUNTS[pdb_id], 3)
    assert protein["types"].shape == (EXPECTED_ATOM_COUNTS[pdb_id], len(ELEMENTS))
    assert protein["radii"].shape == (EXPECTED_ATOM_COUNTS[pdb_id],)
    assert np.allclose(protein["types"].sum(axis=1), 1.0)
    assert np.allclose(protein["xyz"], parsed.atom_coords.numpy())
    assert np.allclose(protein["types"], parsed.atom_types.numpy())
    assert np.allclose(protein["radii"], parsed.atom_radii.numpy())


def test_load_structure_np_can_center_coordinates() -> None:
    protein = load_structure_np(PDB_DIR / "4FT4_Q.pdb", center=True)

    assert np.allclose(protein["xyz"].mean(axis=0), 0.0, atol=1e-4)


def test_reference_npy_files_exist_for_fixture_data() -> None:
    expected_suffixes = {
        "atomxyz",
        "atomtypes",
        "atomradii",
        "atomtype_indices",
        "atomserials",
        "residue_numbers",
        "chain_indices",
        "model_indices",
        "occupancy",
        "bfactors",
        "atomnames",
        "residue_names",
        "chain_ids",
        "insertion_codes",
        "elements",
        "records",
    }

    assert len(list(PDB_DIR.glob("*.pdb"))) == len(PDB_IDS)
    assert len(PDB_IDS) <= 10
    for pdb_id in PDB_IDS:
        assert (PDB_DIR / f"{pdb_id}.pdb").exists()
        for suffix in expected_suffixes:
            assert (NPY_DIR / f"{pdb_id}_{suffix}.npy").exists()


def test_save_and_load_protein_npy_roundtrip_for_fixture_data(tmp_path) -> None:
    protein = read_pdb_tensors(PDB_DIR / "4FT4_Q.pdb")

    save_protein_npy(protein, tmp_path, "sample")
    restored = load_protein_npy("sample", tmp_path)

    assert torch.allclose(restored.atom_coords, protein.atom_coords)
    assert torch.allclose(restored.atom_types, protein.atom_types)
    assert torch.allclose(restored.atom_radii, protein.atom_radii)
    assert torch.equal(restored.atom_serials, protein.atom_serials)
    assert restored.atom_names[:3] == protein.atom_names[:3]


@pytest.mark.parametrize("pdb_id", PDB_IDS)
def test_load_protein_npy_reads_reference_fixture_arrays(pdb_id: str) -> None:
    parsed = read_pdb_tensors(PDB_DIR / f"{pdb_id}.pdb")
    restored = load_protein_npy(pdb_id, NPY_DIR)

    assert torch.allclose(restored.atom_coords, parsed.atom_coords)
    assert torch.allclose(restored.atom_types, parsed.atom_types)
    assert torch.allclose(restored.atom_radii, parsed.atom_radii)
    assert torch.equal(restored.atom_serials, parsed.atom_serials)
    assert restored.atom_names == parsed.atom_names
    assert restored.residue_names == parsed.residue_names
    assert restored.chain_ids == parsed.chain_ids


def test_load_protein_npy_fills_defaults_for_legacy_dmasif_files(tmp_path) -> None:
    atom_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    atom_types = _one_hot([0, 2]).numpy()
    np.save(tmp_path / "legacy_atomxyz.npy", atom_coords)
    np.save(tmp_path / "legacy_atomtypes.npy", atom_types)

    protein = load_protein_npy("legacy", tmp_path, center=True, dtype=torch.float64)

    assert protein.atom_coords.dtype == torch.float64
    assert torch.allclose(protein.atom_coords.mean(dim=0), torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(
        protein.atom_radii,
        torch.tensor([ELEMENT_RADII["C"], ELEMENT_RADII["O"]], dtype=torch.float64),
    )
    assert protein.elements == ("C", "O")
    assert protein.atom_names == ()


def test_save_and_load_tensors_npy_aliases_accept_mapping_inputs(tmp_path) -> None:
    protein = _toy_protein()
    _save_tensors_npy(
        {"xyz": protein.atom_coords, "types": protein.atom_types},
        tmp_path,
        "alias",
        include_metadata=False,
    )

    restored = _load_tensors_npy(tmp_path, "alias")

    assert (tmp_path / "alias_atomxyz.npy").exists()
    assert (tmp_path / "alias_atomtypes.npy").exists()
    assert not (tmp_path / "alias_atomnames.npy").exists()
    assert torch.allclose(restored.atom_coords, protein.atom_coords)
    assert torch.allclose(restored.atom_radii, protein.atom_radii)


def test_convert_pdb_to_npy_saves_and_returns_tensors(tmp_path) -> None:
    pdb_path = PDB_DIR / "4Q6I_J.pdb"
    output_dir = tmp_path / "npys"

    protein = convert_pdb_to_npy(
        pdb_path,
        output_dir,
        pdb_id="custom",
        center=True,
        include_metadata=False,
    )

    assert torch.allclose(protein.atom_coords.mean(dim=0), torch.zeros(3), atol=1e-4)
    assert (output_dir / "custom_atomxyz.npy").exists()
    assert (output_dir / "custom_atomtypes.npy").exists()
    assert not (output_dir / "custom_atomnames.npy").exists()
    restored = load_protein_npy("custom", output_dir)
    assert torch.allclose(restored.atom_coords, protein.atom_coords)


def test_convert_pdbs_converts_every_pdb_in_directory(tmp_path) -> None:
    npy_dir = tmp_path / "npys"

    convert_pdbs(PDB_DIR, npy_dir, include_metadata=False)

    for pdb_id in PDB_IDS:
        assert (npy_dir / f"{pdb_id}_atomxyz.npy").exists()
        assert (npy_dir / f"{pdb_id}_atomtypes.npy").exists()
        assert not (npy_dir / f"{pdb_id}_atomnames.npy").exists()
        assert (
            load_protein_npy(pdb_id, npy_dir).atom_coords.shape[0]
            == EXPECTED_ATOM_COUNTS[pdb_id]
        )


def test_tensors_to_numpy_and_numpy_to_tensors_roundtrip() -> None:
    protein = _toy_protein()
    arrays = _tensors_to_numpy(protein)

    restored = _numpy_to_tensors(arrays, dtype=torch.float64)

    assert arrays["atom_coords"].shape == (2, 3)
    assert restored.atom_coords.dtype == torch.float64
    assert torch.allclose(restored.atom_coords, protein.atom_coords.to(torch.float64))
    assert restored.atom_names == protein.atom_names


def test_ensure_protein_tensors_preserves_instances_and_fills_defaults() -> None:
    protein = _toy_protein()

    assert _ensure_protein_tensors(protein) is protein

    normalized = _ensure_protein_tensors(
        {"xyz": protein.atom_coords.numpy(), "types": protein.atom_types.numpy()},
        dtype=torch.float64,
    )

    assert normalized.atom_coords.dtype == torch.float64
    assert torch.equal(normalized.atom_serials, torch.tensor([1, 2]))
    assert torch.allclose(
        normalized.atom_radii,
        torch.tensor([ELEMENT_RADII["C"], ELEMENT_RADII["N"]], dtype=torch.float64),
    )
    assert normalized.atom_names == ()


def test_radii_from_types_returns_expected_values_and_validates_shape() -> None:
    radii = _radii_from_types(_one_hot([0, 1, 5]).numpy(), dtype=torch.float64)

    assert torch.allclose(
        radii,
        torch.tensor(
            [ELEMENT_RADII["C"], ELEMENT_RADII["H"], ELEMENT_RADII["SE"]],
            dtype=torch.float64,
        ),
    )

    with pytest.raises(ValueError, match="atom_types"):
        _radii_from_types(np.zeros((2, 5), dtype=np.float32))


def test_private_array_and_normalization_helpers(tmp_path) -> None:
    defaults = _default_optional_arrays(_one_hot([0, 2]).numpy(), 2)
    assert np.allclose(defaults["atom_radii"], [ELEMENT_RADII["C"], ELEMENT_RADII["O"]])
    assert defaults["elements"].tolist() == ["C", "O"]

    np.save(tmp_path / "sample_atomxyz.npy", np.array([[1.0, 2.0, 3.0]]))
    assert _load_required_npy(tmp_path, "sample", "atomxyz").shape == (1, 3)
    with pytest.raises(FileNotFoundError):
        _load_required_npy(tmp_path, "missing", "atomxyz")

    assert _as_float_tensor([1, 2], torch.float64).dtype == torch.float64
    assert torch.equal(_as_long_tensor([[1, 2]]), torch.tensor([1, 2]))
    assert _normalize_chain_filter(None) is None
    assert _normalize_chain_filter("AB") == {"A", "B"}
    assert _normalize_chain_filter([" A ", "B"]) == {"A", "B"}
    assert _normalize_element("", " SE ") == "SE"
    assert _normalize_element("", " CA ") == "C"
    assert _normalize_element("se", "") == "SE"
