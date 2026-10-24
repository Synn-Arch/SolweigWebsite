"""Does a deep run folder break Solweig-Lite on Windows? Run it like the course backend at several depths.

    python .github/scripts/longpath_check.py web/scene <base folder>

The model writes files up to ~180 characters below its run folder (the
geometry cache). Without Windows long-path support a full path stops at 260
characters, so a run folder deeper than ~80 characters should fail. Each case
prints the folder length and either OK or the error the backend would show.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from osgeo import gdal

from solweig_light import RuntimeOptions, runtime_options, thermal_comfort

gdal.UseExceptions()

SIZE, OFFSET = 96, 208
MET = "ownmet_Forcing_data.txt"


def scene_at(base, length):
    """A run folder whose full path is `length` characters long, with the rasters in it."""
    pad = length - len(str(base)) - len("\\scene") - 1
    work = base / ("x" * pad) / "scene"
    work.mkdir(parents=True)
    for name in ("Building_DSM.tif", "DEM.tif", "Trees.tif"):
        gdal.Translate(str(work / name), str(SOURCE / name), srcWin=[OFFSET, OFFSET, SIZE, SIZE])
    shutil.copy(SOURCE / MET, work / MET)
    return work


def run(work, **extra):
    threads = min(4, os.cpu_count() or 1)
    options = RuntimeOptions(memory_budget_bytes=6 * 1024**3, cpu_budget=threads,
                             threads_per_worker=threads, block_pixels=1024, **extra)
    try:
        with runtime_options(options):
            thermal_comfort(base_path=str(work), selected_date_str="2020-08-13",
                            building_dsm_filename="Building_DSM.tif", dem_filename="DEM.tif",
                            trees_filename="Trees.tif", landcover_filename=None, tile_size=600,
                            overlap=20, use_own_met=True, own_met_file=str(work / MET), save_tmrt=True)
        ok = (work / "output_folder" / "0_0" / "TMRT_0_0.tif").exists()
        return "OK" if ok else "no output"
    except Exception as error:                      # what the course backend would show
        return f"Failed: {type(error).__name__}: {error}"


def longest_below(work):
    return max((len(str(p)) - len(str(work)) for p in work.rglob("*")), default=0)


def main():
    global SOURCE
    SOURCE, base = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    base.mkdir(parents=True, exist_ok=True)
    print(f"platform {sys.platform}, base {base} ({len(str(base))} chars)")
    for length in (60, 75, 90, 110):
        work = scene_at(base / f"case{length}", length)
        print(f"  run folder {len(str(work))} chars: {run(work)} (longest path below it: {longest_below(work)})")
    work = scene_at(base / "case110cache", 110)
    cache = Path(tempfile.gettempdir()) / "slc"
    print(f"  run folder {len(str(work))} chars, cache_dir={cache}: {run(work, cache_dir=str(cache))}"
          f" (longest path below it: {longest_below(work)})")
    shutil.rmtree(base, ignore_errors=True)
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
