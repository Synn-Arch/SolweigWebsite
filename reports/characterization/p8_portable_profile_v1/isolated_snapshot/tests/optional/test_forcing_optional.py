"""Real local forcing conversion against isolated original CPU artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from solweig_light.forcing import local

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "tests/reference/forcing_original_cpu"
MANIFEST = json.loads((PACKET / "manifest.json").read_text())
WRF_PACKET = ROOT / "tests/reference/wrf_patched_cpu"
WRF_MANIFEST = json.loads((WRF_PACKET / "manifest.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_original_packet_provenance():
    assert MANIFEST["evidence_class"] == "original_upstream_cpu_reference"
    assert MANIFEST["upstream_commit"] == "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"
    assert MANIFEST["upstream_patch_hash"] is None
    for path, expected in MANIFEST["files"].items():
        assert sha(PACKET / path) == expected
    wrf = next(c for c in MANIFEST["cases"] if c["label"] == "wrf-datetime-tuple-failure")
    assert wrf["status"] == "original_failure"
    assert wrf["exception_type"] == "TypeError"
    assert "datetime.datetime" in wrf["exception_message"] and "tuple" in wrf["exception_message"]


def test_wrf_timestamp_repair_provenance():
    assert WRF_MANIFEST["evidence_class"] == "patched_upstream_cpu_reference"
    assert WRF_MANIFEST["repair_policy"] == "wrf_timestamp_v1"
    assert WRF_MANIFEST["source_sha256"] == MANIFEST["source_sha256"]
    assert sha(ROOT / WRF_MANIFEST["upstream_patch"]) == WRF_MANIFEST["upstream_patch_sha256"]
    for path, expected in WRF_MANIFEST["files"].items():
        assert sha(WRF_PACKET / path) == expected
    assert sum(case["status"] == "success" for case in WRF_MANIFEST["cases"]) == 3
    failure = next(case for case in WRF_MANIFEST["cases"] if case["status"] == "patched_failure")
    assert failure["label"] == "duplicate-domain-shape-failure"
    assert failure["exception_type"] == "ValueError"


# The original WRF failure remains checked above; the authorized timestamp
# repair is compared only with the separately labeled patched oracle.
CASES = [(case, PACKET) for case in MANIFEST["cases"] if case["function"] != "process_wrfout_data"]
CASES += [(case, WRF_PACKET) for case in WRF_MANIFEST["cases"]]


@pytest.mark.parametrize("case,packet", CASES, ids=[case["label"] for case, _ in CASES])
def test_local_original_forcing(case, packet, tmp_path):
    import netCDF4
    kwargs = {}
    for key, value in case["arguments"].items():
        if isinstance(value, dict):
            path = value["fixture_path"]
            kwargs[key] = str(tmp_path / path) if key in ("output_file", "preprocess_dir") else str(packet / path)
        else:
            kwargs[key] = value
    if "output_file" in kwargs:
        Path(kwargs["output_file"]).parent.mkdir(parents=True, exist_ok=True)
    function = getattr(local, case["function"])
    if case["status"] != "success":
        with pytest.raises(getattr(__import__("builtins"), case["exception_type"])) as error:
            function(**kwargs)
        assert str(error.value) == case["exception_message"]
        return
    function(**kwargs)
    if case["function"] == "process_metfiles":
        expected_folder = packet / case["arguments"]["preprocess_dir"]["fixture_path"] / "metfiles"
        actual_folder = Path(kwargs["preprocess_dir"]) / "metfiles"
        assert sorted(p.name for p in actual_folder.glob("*.txt")) == sorted(p.name for p in expected_folder.glob("*.txt"))
        for path in expected_folder.glob("*.txt"):
            assert (actual_folder / path.name).read_bytes() == path.read_bytes()
    else:
        expected = packet / case["arguments"]["output_file"]["fixture_path"]
        with netCDF4.Dataset(expected) as golden, netCDF4.Dataset(kwargs["output_file"]) as actual:
            assert actual.ncattrs() == golden.ncattrs()
            assert list(actual.dimensions) == list(golden.dimensions)
            assert list(actual.variables) == list(golden.variables)
            for name, variable in golden.variables.items():
                candidate = actual.variables[name]
                assert candidate.dtype == variable.dtype
                assert candidate.dimensions == variable.dimensions
                assert {k: candidate.getncattr(k) for k in candidate.ncattrs()} == {k: variable.getncattr(k) for k in variable.ncattrs()}
                assert candidate.filters() == variable.filters()
                assert np.array_equal(np.ma.getmaskarray(candidate[:]), np.ma.getmaskarray(variable[:]))
                np.testing.assert_array_equal(candidate[:].filled(np.nan), variable[:].filled(np.nan))


def test_patched_wrf_wind_rotation_units_domain_order():
    import netCDF4
    with netCDF4.Dataset(WRF_PACKET / "three-hour-rotation.nc") as ds:
        assert ds.variables["T2"].units == "K"
        assert ds.variables["PSFC"].units == "Pa"
        assert ds.variables["WIND"].shape == (3, 3, 4)
        assert ds.variables["WIND"][0, 0, 0] == 0
        assert ds.variables["WDIR"][0, 0, 0] == 0
        # Grid wind (-1, 1) rotates to earth wind (-1.4, -.2).
        u = np.float32(-1) * np.float32(.6) - np.float32(1) * np.float32(.8)
        v = np.float32(1) * np.float32(.6) + np.float32(-1) * np.float32(.8)
        expected = (np.float32(270) - np.degrees(np.arctan2(v, u))) % np.float32(360)
        np.testing.assert_array_equal(ds.variables["WDIR"][0, 2, 0], expected)
        assert ds.variables["RH2"][:].min() >= 0 and ds.variables["RH2"][:].max() <= 100
    with netCDF4.Dataset(WRF_PACKET / "domain-sort-retained.nc") as ds:
        assert ds.variables["T2"][0, 0, 0] == 281  # domain 2 precedes domain 3.
        assert ds.variables["T2"][1, 0, 0] == 280
        assert ds.variables["time"][1] - ds.variables["time"][0] == 1


def test_dst_and_metfile_contract():
    path = PACKET / "met-uhi/metfiles/metfile_nearest_2024-11-03.txt"
    rows = np.loadtxt(path, skiprows=1)
    assert rows.shape == (25, 25)
    assert np.count_nonzero(rows[:, 2] == 1) == 2
    assert path.read_text().splitlines()[0].split()[13:15] == ["rain", "Kdn"]
    assert (rows[:, 11] < 100).all()  # Kelvin converted to Celsius.
    assert (rows[:, 12] < 200).all()  # Pa converted to kPa.
    assert (rows[:, 24] >= 0).all()
    disabled = np.loadtxt(PACKET / "met-standard/metfiles/metfile_nearest_2024-11-03.txt", skiprows=1)
    assert np.count_nonzero(disabled[:, 24]) == 0
    masked = np.loadtxt(PACKET / "met-masked/metfiles/metfile_nearest_2024-11-03.txt", skiprows=1)
    assert masked[0, 9] == -999
    assert masked[1, 24] == 0
    assert masked[2, 23] == 5
    missing = np.loadtxt(PACKET / "met-missing/metfiles/metfile_nearest_2024-11-03.txt", skiprows=1)
    assert (missing[:, 10] == -999).all()
    assert (missing[:, 23] == -999).all()
    assert (missing[:, 24] == 0).all()


def test_original_uhi_threshold_wind_floor_and_nan_component():
    case = MANIFEST["component_cases"][0]
    with np.load(PACKET / case["input"]) as inputs:
        actual = local.compute_uhi_cycle_from_arrays(**{key: inputs[key] for key in inputs.files})
    with np.load(PACKET / case["output"]) as golden:
        np.testing.assert_array_equal(actual, golden["uhi_cycle"])
    assert actual.dtype == np.float32
    assert actual[1, 0, 0] > 0
    assert actual[3, 0, 0] == 0  # SW just above 5 W/m2 is outside night.


def test_core_forcing_import_does_not_load_optional_dependencies():
    code = "import sys; import solweig_light.forcing; assert not any(n in sys.modules for n in ('torch','xarray','pandas','netCDF4','timezonefinder','matplotlib'))"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("filename", ["wrfout_d03_2024-06-21_12_00_00", "wrfout_d03_2024-06-21_12:00:00", "wrfout_d03_2024-06-21_12"])
def test_wrf_parser_tuple_contract(filename):
    import datetime
    assert local.extract_datetime_strict(filename) == (datetime.datetime(2024, 6, 21, 12), 3)
