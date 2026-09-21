"""Public raw-TIFF preprocessing against actual original forcing integration."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from solweig_light import preprocess

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "tests/reference/public_forcing_original_cpu"
MANIFEST = json.loads((PACKET / "manifest.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_public_original_integration_provenance():
    assert MANIFEST["upstream_commit"] == "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"
    forcing = json.loads((ROOT / "tests/reference/forcing_original_cpu/manifest.json").read_text())
    assert MANIFEST["source_sha256"]["preprocessor.py"] == forcing["source_sha256"]
    for path, expected in MANIFEST["files"].items():
        assert sha(PACKET / path) == expected
    repair = MANIFEST["wrf_repair"]
    assert repair["policy"] == "wrf_timestamp_v1"
    assert sha(ROOT / repair["patch"]) == repair["patch_sha256"]
    original_failure = next(c for c in MANIFEST["cases"] if c["label"] == "wrf-original-failure")
    assert original_failure["evidence_class"] == "original_upstream_cpu_reference"
    assert original_failure["status"] == "failure"
    assert original_failure["exception_type"] == "TypeError"
    assert "datetime.datetime" in original_failure["exception_message"]
    assert "tuple" in original_failure["exception_message"]
    assert sum(c["status"] == "success" for c in MANIFEST["cases"]) == 5


def assert_netcdf_equal(actual_path, expected_path):
    import netCDF4
    with netCDF4.Dataset(actual_path) as actual, netCDF4.Dataset(expected_path) as expected:
        assert {k: actual.getncattr(k) for k in actual.ncattrs()} == {k: expected.getncattr(k) for k in expected.ncattrs()}
        assert {k: (len(v), v.isunlimited()) for k, v in actual.dimensions.items()} == {k: (len(v), v.isunlimited()) for k, v in expected.dimensions.items()}
        assert list(actual.variables) == list(expected.variables)
        for name, golden in expected.variables.items():
            candidate = actual.variables[name]
            assert candidate.dimensions == golden.dimensions
            assert candidate.dtype == golden.dtype
            assert candidate.filters() == golden.filters()
            assert {k: candidate.getncattr(k) for k in candidate.ncattrs()} == {k: golden.getncattr(k) for k in golden.ncattrs()}
            np.testing.assert_array_equal(np.ma.getmaskarray(candidate[:]), np.ma.getmaskarray(golden[:]))
            np.testing.assert_array_equal(candidate[:].filled(np.nan), golden[:].filled(np.nan))


def assert_tiff_equal(actual_path, expected_path):
    from osgeo import gdal
    actual, golden = gdal.Open(str(actual_path)), gdal.Open(str(expected_path))
    assert actual.GetGeoTransform() == golden.GetGeoTransform()
    assert actual.GetProjection() == golden.GetProjection()
    assert actual.GetMetadata() == golden.GetMetadata()
    assert actual.RasterCount == golden.RasterCount
    for index in range(1, golden.RasterCount + 1):
        candidate, reference = actual.GetRasterBand(index), golden.GetRasterBand(index)
        assert candidate.DataType == reference.DataType
        assert candidate.GetNoDataValue() == reference.GetNoDataValue()
        assert candidate.GetMetadata() == reference.GetMetadata()
        np.testing.assert_array_equal(candidate.ReadAsArray(), reference.ReadAsArray())


# WRF compares with the authorized patched oracle; its original public failure
# remains independently retained and checked in the provenance test.
CASES = [case for case in MANIFEST["cases"] if case["label"] != "wrf-original-failure"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["label"])
def test_public_preprocess_real_forcing(case, tmp_path):
    kwargs = {}
    for key, value in case["arguments"].items():
        if isinstance(value, dict):
            if "repository_path" in value:
                kwargs[key] = str(ROOT / value["repository_path"])
            elif key == "preprocess_dir":
                kwargs[key] = str(tmp_path / value["packet_path"])
            else:
                kwargs[key] = str(PACKET / value["packet_path"])
        else:
            kwargs[key] = value
    if case["status"] == "failure":
        assert case["evidence_class"] == "patched_upstream_cpu_reference"
        with pytest.raises(getattr(__import__("builtins"), case["exception_type"])) as error:
            preprocess(**kwargs)
        assert str(error.value) == case["exception_message"]
    else:
        assert preprocess(**kwargs) == kwargs["preprocess_dir"]
    actual = Path(kwargs["preprocess_dir"])
    expected = PACKET / case["arguments"]["preprocess_dir"]["packet_path"]
    paths = sorted(path.relative_to(expected) for path in expected.rglob("*") if path.is_file())
    assert paths == sorted(path.relative_to(actual) for path in actual.rglob("*") if path.is_file())
    assert len([path for path in paths if path.suffix == ".tif"]) == 16
    for path in paths:
        if path.suffix == ".txt":
            assert (actual / path).read_bytes() == (expected / path).read_bytes()
            rows = np.loadtxt(actual / path, skiprows=1)
            assert rows.shape == (3 if kwargs["data_source_type"] == "wrfout" else 25, 25)
            if not kwargs["use_uhi"]:
                assert (rows[:, -1] == 0).all()
        elif path.suffix == ".nc":
            assert_netcdf_equal(actual / path, expected / path)
        elif path.suffix == ".tif":
            assert_tiff_equal(actual / path, expected / path)
    assert "torch" not in sys.modules
