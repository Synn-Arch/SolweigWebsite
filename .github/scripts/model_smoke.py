"""Run solweig_light the way the course backend does, on any OS, and check the results.

    python .github/scripts/model_smoke.py web/scene <empty work folder>

Mirrors the student backend (week 6/7 model.py): copy the inputs into a fresh
folder, optionally raise the canopy, call thermal_comfort with the same
arguments, read TMRT/UTCI, then delete the folder. Also checks that a second
holder of an output lock is refused while the first holds it.
"""
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from osgeo import gdal

from solweig_light import RuntimeOptions, runtime_options, thermal_comfort
from solweig_light import _portable

gdal.UseExceptions()

SIZE, OFFSET = 96, 208                       # a small window from the middle of the scene
MET = "ownmet_Forcing_data.txt"
THREADS = min(4, os.cpu_count() or 1)


def make_scene(source, folder):
    folder.mkdir(parents=True)
    for name in ("Building_DSM.tif", "DEM.tif", "Trees.tif"):
        gdal.Translate(str(folder / name), str(source / name), srcWin=[OFFSET, OFFSET, SIZE, SIZE])
    shutil.copy(source / MET, folder / MET)


def raise_canopy(folder):
    ds = gdal.Open(str(folder / "Trees.tif"), gdal.GA_Update)
    band = ds.GetRasterBand(1)
    canopy = band.ReadAsArray().astype("float32")
    canopy[40:48, 40:48] = np.maximum(canopy[40:48, 40:48], 12.0)
    band.WriteArray(canopy)
    ds = None


def run(folder):
    options = RuntimeOptions(memory_budget_bytes=6 * 1024**3, cpu_budget=THREADS,
                             threads_per_worker=THREADS, block_pixels=1024)
    started = time.perf_counter()
    with runtime_options(options):
        thermal_comfort(base_path=str(folder), selected_date_str="2020-08-13",
                        building_dsm_filename="Building_DSM.tif", dem_filename="DEM.tif",
                        trees_filename="Trees.tif", landcover_filename=None, tile_size=600, overlap=20,
                        use_own_met=True, own_met_file=str(folder / MET), save_tmrt=True)
    seconds = time.perf_counter() - started
    result = {}
    for var in ("TMRT", "UTCI"):
        ds = gdal.Open(str(folder / "output_folder" / "0_0" / f"{var}_0_0.tif"))
        data = ds.ReadAsArray()
        ds = None
        assert data.shape == (24, SIZE, SIZE), (var, data.shape)
        assert np.isfinite(data[14]).any(), f"{var} 14:00 is empty"
        result[var] = data
    print(f"  {folder.name}: {seconds:.1f} s, mean TMRT 14:00 = {np.nanmean(result['TMRT'][14]):.6f}, "
          f"mean UTCI 14:00 = {np.nanmean(result['UTCI'][14]):.6f}")
    return result


def check_locks(folder):
    path = folder / "probe.lock"
    first, second = open(path, "a+b"), open(path, "a+b")
    try:
        _portable.lock_exclusive(first.fileno(), blocking=False)
        try:
            _portable.lock_exclusive(second.fileno(), blocking=False)
        except BlockingIOError:
            pass
        else:
            raise AssertionError("a second lock holder was not refused")
        _portable.unlock(first.fileno())
        _portable.lock_exclusive(second.fileno(), blocking=False)
        _portable.unlock(second.fileno())
    finally:
        first.close()
        second.close()
    print("  locks: a second holder is refused, and gets the lock once it is free")


def main():
    source, work = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"platform {sys.platform}, python {sys.version.split()[0]}, gdal {gdal.__version__}, threads {THREADS}")
    work.mkdir(parents=True, exist_ok=True)
    check_locks(work)

    baseline_dir, again_dir, trees_dir = work / "baseline", work / "baseline_again", work / "with_tree"
    make_scene(source, baseline_dir)
    baseline = run(baseline_dir)
    run(baseline_dir)                        # the same folder a second time: existing outputs and locks

    make_scene(source, again_dir)
    again = run(again_dir)
    for var in baseline:
        assert np.array_equal(baseline[var], again[var], equal_nan=True), f"{var} differs between identical runs"
    print("  identical inputs give identical outputs")

    make_scene(source, trees_dir)
    raise_canopy(trees_dir)
    trees = run(trees_dir)
    cooling = np.nanmin(trees["TMRT"][14] - baseline["TMRT"][14])
    assert cooling < -1.0, f"a new tree should cool TMRT at 14:00; largest change {cooling:.2f}"
    print(f"  a new tree cools TMRT at 14:00 by up to {-cooling:.1f} C")

    for folder in (baseline_dir, again_dir, trees_dir):
        shutil.rmtree(folder)                # the course backend deletes each run folder like this
    print("  run folders deleted")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
