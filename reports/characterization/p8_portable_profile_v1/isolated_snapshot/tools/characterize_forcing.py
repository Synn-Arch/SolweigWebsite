"""Collect unmodified, pinned original local ERA5/UHI/WRF forcing evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_wrf(output):
    """Execute an isolated source copy with only the authorized tuple-index fix."""
    import types
    import numpy as np
    import netCDF4
    import xarray as xr
    from solweig_gpu import preprocessor as original
    checkout = ROOT / ".upstream/SOLWEIG-GPU"
    if subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise RuntimeError("upstream commit mismatch")
    if subprocess.check_output(["git", "-C", str(checkout), "diff", "--name-only"], text=True).strip():
        raise RuntimeError("upstream source modified")
    if sha(original.__file__) != sha(checkout / "solweig_gpu/preprocessor.py") or "solweig_light" in sys.modules:
        raise RuntimeError("original source isolation failed")
    source = Path(original.__file__).read_text()
    before = "start_time <= extract_datetime_strict(f) <= end_time"
    after = "start_time <= extract_datetime_strict(f)[0] <= end_time"
    if source.count(before) != 1:
        raise RuntimeError("repair does not apply uniquely")
    output.mkdir(parents=True, exist_ok=False)
    patched_path = output / "preprocessor_wrf_timestamp_v1.py"
    patched_path.write_text(source.replace(before, after))
    patched = types.ModuleType("original_wrf_timestamp_v1")
    patched.__file__ = str(patched_path)
    exec(compile(patched_path.read_text(), str(patched_path), "exec"), patched.__dict__)
    patch = ROOT / "docs/upstream_patches/wrf_timestamp_v1.patch"
    report = {"evidence_class": "patched_upstream_cpu_reference", "upstream_commit": COMMIT,
              "source_sha256": sha(original.__file__), "patched_source_sha256": sha(patched_path),
              "upstream_patch": str(patch.relative_to(ROOT)), "upstream_patch_sha256": sha(patch),
              "repair_policy": "wrf_timestamp_v1", "repair_rationale": "Extract datetime from parser tuple only for inclusive window filtering; retain datetime/domain tuple sort.",
              "collector_sha256": sha(__file__), "invocation": sys.argv,
              "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                              "numpy": np.__version__, "xarray": xr.__version__, "netCDF4": netCDF4.__version__}, "cases": []}

    def fixture(folder, filename, index, rotation):
        folder.mkdir(parents=True, exist_ok=True)
        shape = (1, 3, 4)
        dims = ("Time", "south_north", "west_east")
        spatial = np.arange(12, dtype=np.float32).reshape(shape)
        u = (spatial % 4 - 1).copy(); v = (spatial % 3 - 1).copy()
        u[0, 0, 0] = 0; v[0, 0, 0] = 0
        u[0, 0, 1] = np.float32(.01); v[0, 0, 1] = 0
        variables = {"T2": (dims, 280 + spatial + np.float32(index)), "TSK": (dims, 290 + spatial),
                     "Q2": (dims, np.full(shape, .005, np.float32)), "PSFC": (dims, 100000 + spatial * 100),
                     "U10": (dims, u), "V10": (dims, v), "SWDOWN": (dims, spatial * 40),
                     "XLAT": (dims, np.broadcast_to(np.array([40.9, 40.7, 40.5], np.float32)[None, :, None], shape)),
                     "XLONG": (dims, np.broadcast_to(np.array([-74.3, -74.1, -73.9, -73.7], np.float32)[None, None, :], shape))}
        if rotation:
            rdims, rshape = (dims, shape) if rotation == "3d" else (dims[1:], shape[1:])
            variables["COSALPHA"] = (rdims, np.full(rshape, .6, np.float32))
            variables["SINALPHA"] = (rdims, np.full(rshape, .8, np.float32))
        xr.Dataset(variables).to_netcdf(folder / filename)

    definitions = [
        ("three-hour-rotation", "2024-06-21 12:00:00", "2024-06-21 14:00:00",
         [("wrfout_d03_2024-06-21_12_00_00", "2d"), ("wrfout_d02_2024-06-21_13:00:00", "3d"), ("wrfout_d03_2024-06-21_14", None)]),
        ("domain-sort-retained", "2024-06-21 12:00:00", "2024-06-21 13:00:00",
         [("wrfout_d03_2024-06-21_12", None), ("wrfout_d02_2024-06-21_12", None)]),
        ("duplicate-domain-shape-failure", "2024-06-21 12:00:00", "2024-06-21 12:00:00",
         [("wrfout_d03_2024-06-21_12", None), ("wrfout_d02_2024-06-21_12", None)]),
        ("single-calm-unrotated", "2024-06-21 12:00:00", "2024-06-21 12:00:00",
         [("wrfout_d01_2024-06-21_12", None)]),
    ]
    for label, start, end, files in definitions:
        folder = output / label
        for i, (filename, rotation) in enumerate(files):
            fixture(folder, filename, i, rotation)
        kwargs = dict(start_time=start, end_time=end, folder_path=str(folder), output_file=str(output / f"{label}.nc"))
        entry = {"label": label, "function": "process_wrfout_data",
                 "arguments": {**kwargs, "folder_path": {"fixture_path": label}, "output_file": {"fixture_path": f"{label}.nc"}},
                 "ordered_source_filenames": sorted([f for f, _ in files], key=patched.extract_datetime_strict)}
        try:
            patched.process_wrfout_data(**kwargs)
            entry["status"] = "success"
        except Exception as error:
            entry.update(status="patched_failure", exception_type=type(error).__name__, exception_message=str(error))
        report["cases"].append(entry)
    report["files"] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Captured {len(report['cases'])} patched WRF calls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wrf-patched-output", type=Path)
    args = parser.parse_args()
    if args.wrf_patched_output:
        collect_wrf(args.wrf_patched_output)
        return
    if args.output is None:
        parser.error("--output or --wrf-patched-output is required")
    import numpy as np
    import xarray as xr
    import netCDF4
    from osgeo import gdal, osr
    from solweig_gpu import preprocessor as original
    checkout = ROOT / ".upstream/SOLWEIG-GPU"
    if subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise RuntimeError("upstream commit mismatch")
    if subprocess.check_output(["git", "-C", str(checkout), "diff", "--name-only"], text=True).strip():
        raise RuntimeError("upstream source modified")
    if sha(original.__file__) != sha(checkout / "solweig_gpu/preprocessor.py") or "solweig_light" in sys.modules:
        raise RuntimeError("original source isolation failed")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"evidence_class": "original_upstream_cpu_reference", "upstream_commit": COMMIT,
              "upstream_patch_hash": None, "source_sha256": sha(original.__file__),
              "collector_sha256": sha(__file__), "invocation": sys.argv,
              "environment": {"python": sys.version, "executable": sys.executable,
                              "platform": platform.platform(), "original_module": original.__file__,
                              "numpy": np.__version__, "xarray": xr.__version__, "netCDF4": netCDF4.__version__},
              "cases": []}
    def run(label, name, kwargs):
        entry = {"label": label, "function": name, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()}}
        try:
            getattr(original, name)(**kwargs)
            entry["status"] = "success"
        except Exception as error:
            entry.update(status="original_failure", exception_type=type(error).__name__, exception_message=str(error))
        report["cases"].append(entry)
        return entry

    # The audited WRF defect occurs before any meteorological variable is read.
    wrf = args.output / "wrf"
    wrf.mkdir()
    for suffix in ("12_00_00", "13:00:00", "14"):
        with netCDF4.Dataset(wrf / f"wrfout_d03_2024-06-21_{suffix}", "w") as ds:
            ds.createDimension("Time", 1)
    run("wrf-datetime-tuple-failure", "process_wrfout_data",
        dict(start_time="2024-06-21 12:00:00", end_time="2024-06-21 14:00:00", folder_path=wrf, output_file=args.output / "wrf-output.nc"))

    # Real NetCDF files span the DST fall-back day; streams omit different hours.
    times = np.arange(np.datetime64("2024-11-01T00"), np.datetime64("2024-11-06T00"), np.timedelta64(1, "h"))
    lat, lon = np.array([40.9, 40.7, 40.5], np.float32), np.array([-74.2, -74.0, -73.8], np.float32)
    hour = np.arange(len(times), dtype=np.float32)[:, None, None]
    spatial = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    shape = (len(times), 3, 3)
    t2 = np.broadcast_to(285 + 7 * np.sin(hour * np.float32(np.pi / 12)) + spatial, shape).copy()
    u = np.broadcast_to((hour % 4 - 2) + spatial * .1, shape).copy()
    v = np.broadcast_to((hour % 3 - 1) - spatial * .2, shape).copy()
    u[0] = 0; v[0] = 0
    radiation = np.broadcast_to(np.maximum(0, 700 * np.sin((hour % 24 - 6) * np.float32(np.pi / 12))), shape).copy()
    radiation[2, 0, 0] = 5.0; radiation[3, 0, 0] = np.nan
    t2[10, 2, 2] = np.nan
    for coord in ("time", "valid_time"):
        folder = args.output / coord
        folder.mkdir()
        coords = {coord: times, "latitude": lat, "longitude": lon}
        dims = (coord, "latitude", "longitude")
        instant = xr.Dataset({"t2m": (dims, t2), "d2m": (dims, t2 - 3),
                              "sp": (dims, np.broadcast_to(100000 + spatial * 100, shape)),
                              "u10": (dims, u), "v10": (dims, v)}, coords=coords)
        accum = xr.Dataset({"ssrd": (dims, radiation * np.float32(3600))}, coords=coords)
        instant.isel({coord: [i for i in range(len(times)) if i != 35]}).to_netcdf(folder / "data_stream-oper_stepType-instant.nc")
        accum.isel({coord: [i for i in range(len(times)) if i != 36]}).to_netcdf(folder / "data_stream-oper_stepType-accum.nc")
        for uhi in (False, True):
            output = args.output / f"{coord}-{'uhi' if uhi else 'standard'}.nc"
            run(f"{coord}-{'uhi' if uhi else 'standard'}", "process_era5_data_uhi" if uhi else "process_era5_data",
                dict(start_time="2024-11-03 00:00:00", end_time="2024-11-04 12:00:00", folder_path=folder, output_file=output))
        run(f"{coord}-empty-window", "process_era5_data",
            dict(start_time="2020-01-01 00:00:00", end_time="2020-01-01 01:00:00", folder_path=folder, output_file=args.output / "empty.nc"))

    # Streams with intersecting and disjoint timestamps exercise inclusive alignment.
    for label, accum_indices in (("intersection", [48, 50, 51]), ("disjoint", [60, 61])):
        folder = args.output / label
        folder.mkdir()
        with xr.open_dataset(args.output / "time/data_stream-oper_stepType-instant.nc") as source:
            source.sel(time=times[[48, 49, 50]]).to_netcdf(folder / "data_stream-oper_stepType-instant.nc")
        with xr.open_dataset(args.output / "time/data_stream-oper_stepType-accum.nc") as source:
            source.sel(time=times[accum_indices]).to_netcdf(folder / "data_stream-oper_stepType-accum.nc")
        run(label, "process_era5_data", dict(start_time="2024-11-03 00:00:00", end_time="2024-11-03 23:00:00",
            folder_path=folder, output_file=args.output / f"{label}.nc"))
    # Characterize the original forecast-axis normalization, including failures.
    for label, two_dimensional in (("scalar-time-step", False), ("two-axis-time-step", True)):
        folder = args.output / label
        folder.mkdir()
        for stream in ("instant", "accum"):
            with xr.open_dataset(args.output / f"time/data_stream-oper_stepType-{stream}.nc") as source:
                source = source.sel(time=times[[48, 49, 50]]).load()
            source = source.rename({"time": "step"}).assign_coords(step=np.arange(3).astype("timedelta64[h]"))
            if two_dimensional:
                source = source.expand_dims(time=[times[48]])
            else:
                source = source.assign_coords(time=times[48])
            source.to_netcdf(folder / f"data_stream-oper_stepType-{stream}.nc")
        run(label, "process_era5_data", dict(start_time="2024-11-03 00:00:00", end_time="2024-11-03 02:00:00",
            folder_path=folder, output_file=args.output / f"{label}.nc"))

    tiles = args.output / "tiles"
    tiles.mkdir()
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
    for label, bounds in (("nearest", (-74.02, 40.72, .02)), ("polygon", (-74.3, 41.0, .7)), ("outside", (-75., 41.5, .1))):
        ds = gdal.GetDriverByName("GTiff").Create(str(tiles / f"DEM_{label}.tif"), 2, 2, 1, gdal.GDT_Float32)
        ds.SetProjection(srs.ExportToWkt()); ds.SetGeoTransform((bounds[0], bounds[2] / 2, 0, bounds[1], 0, -bounds[2] / 2))
        ds.GetRasterBand(1).WriteArray(np.ones((2, 2), np.float32)); ds = None
    for uhi in (False, True):
        preprocess = args.output / f"met-{'uhi' if uhi else 'standard'}"
        run(f"met-DST-{'uhi' if uhi else 'standard'}", "process_metfiles",
            dict(netcdf_file=args.output / "time-uhi.nc", raster_folder=tiles, base_path=args.output,
                 selected_date_str="2024-11-03", preprocess_dir=preprocess, use_uhi=uhi))
    # Explicitly masked forcing, direction wrap, missing variables and fallback.
    masked = args.output / "masked.nc"
    import shutil
    shutil.copyfile(args.output / "time-uhi.nc", masked)
    with netCDF4.Dataset(masked, "a") as ds:
        # UTC 04:00 is local midnight; these rows are inside the sampled day.
        ds.variables["WIND"][4, :, :] = np.ma.masked
        ds.variables["UHI_CYCLE"][5, :, :] = np.ma.masked
        ds.variables["WDIR"][6, :, :] = 725
    run("met-masked-fallback", "process_metfiles",
        dict(netcdf_file=masked, raster_folder=tiles, base_path=args.output, selected_date_str="2024-11-03",
             preprocess_dir=args.output / "met-masked", use_uhi=True))
    missing = args.output / "missing.nc"
    shutil.copyfile(masked, missing)
    with netCDF4.Dataset(missing, "a") as ds:
        ds.renameVariable("RH2", "UNUSED_RH2")
        ds.renameVariable("WDIR", "UNUSED_WDIR")
        ds.renameVariable("UHI_CYCLE", "UNUSED_UHI")
    run("met-missing-variables", "process_metfiles",
        dict(netcdf_file=missing, raster_folder=tiles, base_path=args.output, selected_date_str="2024-11-03",
             preprocess_dir=args.output / "met-missing", use_uhi=True))
    # Independent component capture includes exact threshold, wind floor and NaNs.
    edge_times = np.arange(np.datetime64("2024-11-03T00"), np.datetime64("2024-11-05T00"), np.timedelta64(1, "h"))
    edge_sw = np.tile(np.array([0, -1, 5, 5.0001, np.nan, 100, 300, 0], np.float32), 6).reshape(48, 1, 1)
    edge_t = (280 + np.arange(48, dtype=np.float32) % 24).reshape(48, 1, 1)
    edge_u = np.full_like(edge_t, .5)
    edge_v = np.full_like(edge_t, .5)
    np.savez_compressed(args.output / "uhi-edge-input.npz", times=edge_times, t2_k=edge_t, swdown_wm2=edge_sw, u_wind_ms=edge_u, v_wind_ms=edge_v)
    edge_output = original.compute_uhi_cycle_from_arrays(edge_times, edge_t, edge_sw, edge_u, edge_v)
    np.savez_compressed(args.output / "uhi-edge-output.npz", uhi_cycle=edge_output)
    report["component_cases"] = [{"function": "compute_uhi_cycle_from_arrays", "input": "uhi-edge-input.npz", "output": "uhi-edge-output.npz", "status": "success"}]
    report["files"] = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    # Paths relative to the fixture root make candidate replay relocatable.
    prefix = str(args.output) + "/"
    for case in report["cases"]:
        for key, value in case["arguments"].items():
            if isinstance(value, str) and value.startswith(prefix):
                case["arguments"][key] = {"fixture_path": value[len(prefix):]}
            elif value == str(args.output):
                case["arguments"][key] = {"fixture_path": "."}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Captured {len(report['cases'])} original forcing calls")


if __name__ == "__main__":
    main()
