"""Run Solweig-Lite over a few tile folders the way the course's tile script does, on any OS.

    python .github/scripts/solweig_lite_tiles.py web/scene <empty work folder>

Cuts two small tiles (with Landcover.tif and a weather file) out of the scene,
then runs each one like the students' run_tiles.py: several threads in one
worker, the weather file found in the tile folder, the date read from its
first row, land cover on, no ERA5 roughness. Checks both tiles produce 24
hourly TMRT and UTCI bands.
"""
import datetime
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from osgeo import gdal

from solweig_light import RuntimeOptions, runtime_options, thermal_comfort

gdal.UseExceptions()

SIZE = 96
TILES = {"r1c1_C": (208, 208), "r1c2_E": (304, 208)}       # tile name -> (column, row) offset
THREADS = max(1, (os.cpu_count() or 2) - 1)
MEMORY_BUDGET = 6 * 1024**3        # the model checks each run fits in this (its own guess follows free RAM)


def date_of(met):
    """A UMEP weather file starts each row with the year and the day of the year."""
    with open(met) as rows:
        rows.readline()                                   # skip the header line
        year, day = rows.readline().split()[:2]
    first = datetime.date(int(float(year)), 1, 1) + datetime.timedelta(days=int(float(day)) - 1)
    return first.isoformat()


def run(folder):
    met = next(folder.glob("*.txt"))                       # the one weather file in the tile folder
    options = RuntimeOptions(memory_budget_bytes=MEMORY_BUDGET, cpu_budget=THREADS, workers=1,
                             threads_per_worker=THREADS, block_pixels=1024)
    started = time.perf_counter()
    with runtime_options(options):
        thermal_comfort(
            base_path=str(folder),
            selected_date_str=date_of(met),
            own_met_file=str(met),
            landcover_filename="Landcover.tif",
            ERA_5_z0_find=False,                           # we have our own weather file
        )
    return time.perf_counter() - started


def main():
    source, sample = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"platform {sys.platform}, python {sys.version.split()[0]}, gdal {gdal.__version__}, threads {THREADS}")
    for name, (col, row) in TILES.items():
        folder = sample / name
        folder.mkdir(parents=True)
        for raster in ("Building_DSM.tif", "DEM.tif", "Trees.tif", "Landcover.tif"):
            gdal.Translate(str(folder / raster), str(source / raster), srcWin=[col, row, SIZE, SIZE])
        shutil.copy(source / "ownmet_Forcing_data.txt", folder / "Austin_typical_2009_UTC-5.txt")

    for name in TILES:
        seconds = run(sample / name)
        out = sample / name / "output_folder" / "0_0"
        means = {}
        for var in ("TMRT", "UTCI"):
            ds = gdal.Open(str(out / f"{var}_0_0.tif"))
            data = ds.ReadAsArray()
            ds = None
            assert data.shape == (24, SIZE, SIZE), (name, var, data.shape)
            assert np.isfinite(data[14]).any(), f"{name} {var} 14:00 is empty"
            means[var] = float(np.nanmean(data[14]))
        print(f"  {name}: {seconds:.1f} s, date {date_of(next((sample / name).glob('*.txt')))}, "
              f"mean TMRT 14:00 = {means['TMRT']:.6f}, mean UTCI 14:00 = {means['UTCI']:.6f}")
    shutil.rmtree(sample)
    print("TILE RUNS PASSED")


if __name__ == "__main__":                                 # needed on Windows and macOS
    main()
