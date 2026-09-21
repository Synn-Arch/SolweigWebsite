#!/usr/bin/env python3
"""Generate deterministic input fixtures; never simulation goldens.

Run with .venv-oracle/bin/python tools/generate_fixtures.py. The manifest's
thermal_comfort kwargs use paths relative to the fixture directory; an oracle
adapter must resolve base_path and own_met_file before invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from osgeo import gdal, osr

UPSTREAM_COMMIT = "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"
HEADER = "iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output: Path, upstream: Path) -> dict:
    gdal.UseExceptions()
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}; found {commit}")
    output.mkdir(parents=True, exist_ok=True)
    shape = (32, 35)
    dem = np.full(shape, 3.0, dtype=np.float32)
    buildings = dem.copy()
    buildings[12:19, 14:22] += 10.0
    trees = np.zeros(shape, dtype=np.float32)
    trees[7:10, 6:9] = 8.0
    trees[22:25, 25:28] = 6.0
    # UTM 18N near New York: square one-meter pixels, north-up.
    transform = (583000.0, 1.0, 0.0, 4507000.0, 0.0, -1.0)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32618)
    for filename, array in [("Building_DSM.tif", buildings), ("DEM.tif", dem), ("Trees.tif", trees)]:
        path = output / filename
        ds = gdal.GetDriverByName("GTiff").Create(str(path), shape[1], shape[0], 1, gdal.GDT_Float32)
        ds.SetGeoTransform(transform)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(array)
        ds.FlushCache()
        ds = None
        ds = gdal.Open(str(path))
        assert (ds.RasterYSize, ds.RasterXSize) == shape
        assert ds.GetGeoTransform() == transform
        assert np.array_equal(ds.ReadAsArray(), array)
        ds = None
    # Complete hourly local day, July 18 in leap year 2020 (day 200).
    # Columns follow preprocessor's own-met header and utci_process indices.
    met = np.full((24, 24), -999.0, dtype=np.float64)
    for hour in range(24):
        daylight = max(0.0, np.sin(np.pi * (hour - 6) / 14)) if 6 <= hour <= 20 else 0.0
        met[hour, :4] = (2020, 200, hour, 0)
        met[hour, 9:15] = (2.0 + 0.5 * daylight, 70.0 - 20.0 * daylight,
                          23.0 + 7.0 * daylight, 101.3, 0.0, 700.0 * daylight)
        met[hour, 21:24] = (150.0 * daylight, 550.0 * daylight, (hour * 30.0) % 360)
    np.savetxt(output / "met.txt", met, fmt="%.8f", header=HEADER, comments="")
    loaded = np.loadtxt(output / "met.txt", skiprows=1, delimiter=" ")
    assert loaded.shape == (24, 24) and np.all(np.isfinite(loaded))
    assert np.all(loaded[:, 1] == 200) and np.array_equal(loaded[:, 2], np.arange(24))
    files = {name: sha256(output / name) for name in ("Building_DSM.tif", "DEM.tif", "Trees.tif", "met.txt")}
    kwargs = dict(base_path=".", selected_date_str="2020-07-18", building_dsm_filename="Building_DSM.tif",
                  dem_filename="DEM.tif", trees_filename="Trees.tif", landcover_filename=None,
                  ERA_5_z0_find=False, tile_size=64, overlap=0, use_own_met=True,
                  start_time=None, end_time=None, data_source_type=None, data_folder=None,
                  own_met_file="met.txt", use_uhi=False)
    kwargs.update({f"save_{name}": True for name in ("tmrt", "svf", "kup", "kdown", "lup", "ldown", "shadow", "wbgt", "ta", "wind")})
    manifest = dict(schema_version=1, fixture_id="isolated_block_sparse_trees_32x35",
                    evidence_class="synthetic_input_fixture_not_reference_output", seed=None,
                    determinism="explicit geometry and analytic hourly forcing; no random sampling",
                    upstream_repository="https://github.com/nvnsudharsan/SOLWEIG-GPU", upstream_commit=commit,
                    source_sha256={name: sha256(upstream / "solweig_gpu" / name) for name in
                                   ("solweig_gpu.py", "preprocessor.py", "utci_process.py")},
                    generator_sha256=sha256(Path(__file__)), files_sha256=files,
                    fixture_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                    raster=dict(rows=32, columns=35, epsg=32618, geotransform=transform,
                                dtype="float32", nodata=None, terrain_m=3.0, building_height_m=10.0,
                                tree_semantics="height above terrain; zero away from canopy", canopy_heights_m=[6.0, 8.0]),
                    meteorology=dict(rows=24, columns=24, header=HEADER.split(),
                                     time_basis="local clock inferred by upstream from raster centre and selected date",
                                     units=dict(Wind="m/s", RH="percent", Td="degree Celsius", press="kPa", Kdn="W/m2")),
                    invocation=dict(function="solweig_gpu.thermal_comfort", kwargs=kwargs),
                    validation="GDAL read-back geometry/values and 24x24 met read-back passed",
                    limitations=["Initial P0 scene only; does not cover the full plan 10.4 fixture matrix.",
                                 "No numerical or scientific simulation validation implied.",
                                 "Upstream wrapper builds SVF before radiation; save_svf's warm-cache behavior remains upstream-defined."])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/generated/own_met_small"))
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/SOLWEIG-GPU"))
    args = parser.parse_args()
    result = generate(args.output, args.upstream)
    print(json.dumps({"fixture_sha256": result["fixture_sha256"], "output": str(args.output), "validation": result["validation"]}, indent=2))
