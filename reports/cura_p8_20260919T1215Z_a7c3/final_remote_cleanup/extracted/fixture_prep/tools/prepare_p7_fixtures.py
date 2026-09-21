"""Prepare additional deterministic synthetic inputs, never reference outputs.

Existing benchmark fixtures/protocols are deliberately not modified. Real
scenes require separate source/licensing and classification evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np
from osgeo import gdal, osr


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire_zenodo(output):
    """Download only the pinned modest raster bundle, with size/hash limits."""
    output.mkdir(parents=True, exist_ok=False)
    record_url = "https://zenodo.org/api/records/21081622"
    with urllib.request.urlopen(record_url, timeout=30) as response:
        record = json.load(response)
    if record["metadata"]["license"]["id"] != "gpl-3.0-or-later":
        raise RuntimeError("Pinned record licensing changed; inspect before acquisition")
    source = next(item for item in record["files"] if item["key"] == "Input_rasters.zip")
    expected_size = 94896401
    expected_md5 = "3da64b3eeb71a6838057674838c6c4a7"
    if source["size"] != expected_size or source["checksum"] != "md5:" + expected_md5:
        raise RuntimeError("Pinned source identity changed")
    (output / "record.json").write_text(json.dumps(record, indent=2)+"\n")
    temporary = output / "Input_rasters.zip.partial"
    digest = hashlib.md5()
    count = 0
    with urllib.request.urlopen(source["links"]["self"], timeout=60) as response, temporary.open("xb") as handle:
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > expected_size:
                raise RuntimeError("Source exceeds pinned download resource limit")
            digest.update(chunk)
            handle.write(chunk)
    if count != expected_size or digest.hexdigest() != expected_md5:
        raise RuntimeError("Source size/checksum verification failed; partial retained")
    destination = output / "Input_rasters.zip"
    temporary.rename(destination)
    (output / "acquisition_manifest.json").write_text(json.dumps(dict(
        evidence_class="external_source_archive_not_admitted_real_fixture", source_url=record_url,
        archive_url=source["links"]["self"], record_license="gpl-3.0-or-later",
        bytes=count, md5=digest.hexdigest(), sha256=sha(destination),
        generator_sha256=sha(Path(__file__)),
        unresolved=["acquisition lineage", "height/datum semantics", "real scene coverage"]), indent=2)+"\n")
    print(destination, sha(destination), flush=True)


def inventory_zenodo(archive, output):
    """Density inventory only: metadata gaps prohibit real-fixture admission."""
    if sha(archive) != "9429fac970a29b3bd21217cbd0876f61d3df35e923efd077b7754e9134960a34":
        raise RuntimeError("Inventory requires the pinned source archive")
    gdal.UseExceptions()
    datasets = {name: gdal.Open("/vsizip/" + str(archive.resolve()) + "/Input_rasters/" + name + ".tif")
                for name in ("DEM", "Building_DSM", "Trees", "Landcover")}
    windows = []
    for row in range(0, datasets["DEM"].RasterYSize - 255, 256):
        for col in range(0, datasets["DEM"].RasterXSize - 255, 256):
            arrays = {name: ds.ReadAsArray(col, row, 256, 256) for name, ds in datasets.items()}
            if not all(np.isfinite(a).all() for a in arrays.values()) or (arrays["Landcover"] == -np.finfo(np.float32).max).any():
                continue
            windows.append(dict(row=row, col=col,
                                building_fraction=float(np.mean(arrays["Building_DSM"] - arrays["DEM"] >= 2)),
                                canopy_fraction=float(np.mean(arrays["Trees"] > 0)),
                                dem_range=[float(arrays["DEM"].min()), float(arrays["DEM"].max())],
                                trees_range=[float(arrays["Trees"].min()), float(arrays["Trees"].max())],
                                landcover_values=np.unique(arrays["Landcover"]).tolist()))
    if not windows:
        raise RuntimeError("No valid density windows")
    selected = dict(dense_urban_candidate=max(windows, key=lambda w: w["building_fraction"]),
                    vegetation_rich_candidate=max(windows, key=lambda w: w["canopy_fraction"]))
    sparse = [w for w in windows if w["building_fraction"] > 0 and w["canopy_fraction"] > 0]
    if sparse:
        selected["sparse_candidate"] = min(sparse, key=lambda w: w["building_fraction"] + w["canopy_fraction"])
    report = dict(evidence_class="source_data_density_inventory_not_admitted_real_fixture", window_size=256,
                  resolution_m=2, archive_sha256=sha(archive), generator_sha256=sha(Path(__file__)),
                  windows_considered=len(windows),
                  trees_zero_policy="Included as canopy absence for inventory only; source NoData=0 conflict remains explicit",
                  selection=selected, windows=windows,
                  unresolved=["source acquisition lineage", "height derivation and vertical datum", "Trees zero NoData semantics"])
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(output, len(windows), flush=True)


def prepare_real_windows(archive, output):
    """Retain source-published model-input semantics and original 2 m grid."""
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=False)
    inventory_zenodo(archive, output / "selection_inventory.json")
    selection = json.loads((output / "selection_inventory.json").read_text())["selection"]
    met_source = root / "tests/reference/small_original_cpu/scene/met.txt"
    kwargs_source = root / "reports/initial_kwargs.json"
    offsets = [(w["row"], w["col"]) for w in selection.values()]
    if len(offsets) != 3 or len(set(offsets)) != 3:
        raise RuntimeError("Three distinct real sample density windows required")
    for label, window in selection.items():
        path = output / label.removesuffix("_candidate")
        path.mkdir()
        row, col = window["row"], window["col"]
        for name in ("DEM.tif", "Building_DSM.tif", "Trees.tif", "Landcover.tif"):
            source = "/vsizip/" + str(archive.resolve()) + "/Input_rasters/" + name
            ds = gdal.Translate(str(path / name), source, srcWin=[col, row, 256, 256], format="GTiff")
            ds = None
            original = gdal.Open(source)
            check = gdal.Open(str(path / name))
            if not np.array_equal(original.ReadAsArray(col, row, 256, 256), check.ReadAsArray()):
                raise RuntimeError("Lossless source-window read-back failed")
            original = check = None
        shutil.copyfile(met_source, path / "met.txt")
        kwargs = json.loads(kwargs_source.read_text())
        kwargs.update(base_path=".", own_met_file="met.txt", tile_size=3600, overlap=20,
                      landcover_filename=None)
        (path / "kwargs.json").write_text(json.dumps(kwargs, indent=2)+"\n")
        hashes = {p.name: sha(p) for p in sorted(path.iterdir())}
        check = gdal.Open(str(path / "DEM.tif"))
        arrays = {name: gdal.Open(str(path / (name + ".tif"))).ReadAsArray()
                  for name in ("DEM", "Building_DSM", "Trees")}
        delta = arrays["Building_DSM"] - arrays["DEM"]
        manifest = dict(schema_version=1, evidence_class="source_published_model_input_window_with_synthetic_forcing",
                        fixture_id=path.name, files_sha256=hashes,
                        fixture_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                        source_record="https://zenodo.org/records/21081622", record_license="gpl-3.0-or-later",
                        archive_sha256=sha(archive), generator_sha256=sha(Path(__file__)),
                        source_window=window, rows=256, cols=256, transform=check.GetGeoTransform(), epsg=32614,
                        resolution_m=2, timesteps=24, patch_option=2, tile_size=3600, overlap=20,
                        forcing="synthetic analytic P0 meteorology; not observed Austin forcing",
                        source_met_sha256=sha(met_source), source_kwargs_sha256=sha(kwargs_source),
                        interpretation="source documentation: DEM excludes buildings; Building_DSM includes terrain; Trees canopy height; zero interpreted as canopy absence as upstream model inputs",
                        landcover="source values retained but disabled (None); no undocumented mapping admitted",
                        validation="source-window arrays read back exactly; no resampling",
                        value_audit=dict(building_minus_dem_min_m=float(delta.min()),
                                         building_minus_dem_max_m=float(delta.max()),
                                         building_below_terrain_pixels=int(np.count_nonzero(delta < 0)),
                                         negative_tree_height_pixels=int(np.count_nonzero(arrays["Trees"] < 0)),
                                         nonfinite_pixels=int(sum(np.count_nonzero(~np.isfinite(a)) for a in arrays.values()))),
                        limitations=["Unknown original acquisition lineage and vertical datum", "Trees source NoData is zero; retained unchanged",
                                     "Relative density labels within one published sample, not independent observed regions",
                                     "Not independent scientific ground truth; no numerical comparison or performance result"])
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(path, manifest["fixture_sha256"], flush=True)


def geometry(family, size):
    dem = np.full((size, size), 3, dtype=np.float32)
    building = dem.copy()
    trees = np.zeros_like(dem)
    for row in range(0, size, 32):
        for col in range(0, size, 32):
            if family == "dense_urban":
                height = 16 + 8 * ((row // 32 + col // 32) % 3)
                building[row+4:row+28, col+4:col+28] += height
                # Courtyard with unchanged terrain, surrounded by high walls.
                building[row+12:row+20, col+12:col+20] = dem[row+12:row+20, col+12:col+20]
                trees[row+1:row+3, col+14:col+18] = 6
            else:
                building[row+13:row+19, col+13:col+19] += 10
                # Broad canopy with cross-shaped gaps and a central building.
                trees[row+2:row+30, col+2:col+30] = 12 + 2 * ((row // 32 + col // 32) % 3)
                trees[row+14:row+18, col:col+32] = 0
                trees[row:row+32, col+14:col+18] = 0
                trees[row+13:row+19, col+13:col+19] = 0
    return dem, building, trees


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256])
    parser.add_argument("--families", nargs="+", choices=["dense_urban", "vegetation_rich"],
                        default=["dense_urban", "vegetation_rich"])
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/generated/p7"))
    parser.add_argument("--fetch-zenodo", action="store_true",
                        help="Acquire only the pinned raster archive into a new --output directory; no extraction/model execution")
    parser.add_argument("--inventory-zenodo", type=Path, help="Pinned archive to inventory; --output names a new JSON file")
    parser.add_argument("--prepare-real-zenodo", type=Path, help="Prepare fixed source-published sample windows with synthetic P0 forcing")
    args = parser.parse_args()
    if args.prepare_real_zenodo:
        if args.fetch_zenodo or args.inventory_zenodo or args.output.exists():
            parser.error("Real preparation requires an absent --output and no other source mode")
        prepare_real_windows(args.prepare_real_zenodo, args.output)
        return
    if args.inventory_zenodo:
        if args.fetch_zenodo or args.output.exists():
            parser.error("Inventory --output must be absent and cannot combine with acquisition")
        inventory_zenodo(args.inventory_zenodo, args.output)
        return
    if args.fetch_zenodo:
        if args.output.exists():
            parser.error("Acquisition --output must be absent; existing sources cannot be overwritten")
        acquire_zenodo(args.output)
        return
    if any(size < 32 or size > 3600 or size % 32 for size in args.sizes):
        parser.error("Sizes must be multiples of 32 from 32 through 3600; no implicit workload shrinking")
    destinations = [args.output / f"{family}_{size}" for family in args.families for size in args.sizes]
    if len(set(destinations)) != len(destinations) or any(p.exists() for p in destinations):
        parser.error("Every destination must be unique and absent; existing fixtures cannot be overwritten")
    root = Path(__file__).resolve().parents[1]
    met_source = root / "tests/reference/small_original_cpu/scene/met.txt"
    kwargs_source = root / "reports/initial_kwargs.json"
    gdal.UseExceptions()
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32618)
    for family in args.families:
        for size in args.sizes:
            path = args.output / f"{family}_{size}"
            path.mkdir(parents=True)
            dem, building, trees = geometry(family, size)
            transform = (583017.5-size/2, 1., 0., 4506984.+size/2, 0., -1.)
            for name, values in (("DEM.tif", dem), ("Building_DSM.tif", building), ("Trees.tif", trees)):
                ds = gdal.GetDriverByName("GTiff").Create(str(path / name), size, size, 1, gdal.GDT_Float32)
                ds.SetGeoTransform(transform)
                ds.SetProjection(srs.ExportToWkt())
                ds.GetRasterBand(1).WriteArray(values)
                ds = None
                check = gdal.Open(str(path / name))
                if not np.array_equal(check.ReadAsArray(), values) or check.GetGeoTransform() != transform:
                    raise RuntimeError(f"Raster read-back failed: {path / name}")
                if not osr.SpatialReference(wkt=check.GetProjection()).IsSame(srs):
                    raise RuntimeError(f"CRS read-back failed: {path / name}")
                check = None
            shutil.copyfile(met_source, path / "met.txt")
            kwargs = json.loads(kwargs_source.read_text())
            kwargs.update(base_path=".", own_met_file="met.txt", tile_size=3600, overlap=20)
            (path / "kwargs.json").write_text(json.dumps(kwargs, indent=2)+"\n")
            hashes = {p.name: sha(p) for p in sorted(path.iterdir())}
            manifest = dict(schema_version=1, evidence_class="synthetic_input_fixture_not_reference_output",
                            fixture_id=path.name, family=family, files_sha256=hashes,
                            fixture_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                            generator_sha256=sha(Path(__file__)), source_met_sha256=sha(met_source),
                            source_kwargs_sha256=sha(kwargs_source), rows=size, cols=size,
                            transform=transform, epsg=32618, dtype="float32", nodata=None,
                            timesteps=24, patch_option=2, tile_size=3600, overlap=20,
                            building_fraction=float(np.mean(building-dem >= 2)),
                            canopy_fraction=float(np.mean(trees > 0)),
                            visible_output_fraction=float(np.mean(building-dem < 2)),
                            determinism="explicit 32-pixel motif; no RNG; synthetic analytic P0 forcing copied byte-for-byte",
                            tree_semantics="height above terrain; zero away from canopy",
                            validation="GDAL values, transform, CRS read-back passed",
                            limitations=["Not real observed geometry or forcing", "No model outputs or numerical equivalence generated",
                                         "Input preparation alone does not admit a benchmark; differential prerequisites remain required"])
            (path / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            print(path, manifest["fixture_sha256"], flush=True)


if __name__ == "__main__":
    main()
