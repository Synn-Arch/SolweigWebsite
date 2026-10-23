"""Run Solweig-Lite from a Jupyter kernel, the way the course notebook does, on any OS.

    python .github/scripts/notebook_smoke.py web/scene <empty work folder>

Cuts one small tile (with Landcover.tif and a weather file) out of the scene,
writes a notebook whose cells call the model like the students' notebook
(several threads, a 6 GiB memory budget, land cover, no ERA5 roughness), and
executes it in a fresh kernel. Fails if any cell raises or the outputs are
missing.
"""
import shutil
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_notebook
from osgeo import gdal

gdal.UseExceptions()

SIZE, OFFSET = 96, 208

CELLS = [
    """import os
from pathlib import Path

from solweig_light import RuntimeOptions, runtime_options, thermal_comfort""",
    """def run(tile, met, date):
    threads = max(1, os.cpu_count() - 1)
    options = RuntimeOptions(memory_budget_bytes=6 * 1024**3, cpu_budget=threads,
                             workers=1, threads_per_worker=threads, block_pixels=1024)
    with runtime_options(options):
        thermal_comfort(
            base_path=str(tile),
            selected_date_str=date,
            own_met_file=str(met),
            landcover_filename="Landcover.tif",
            ERA_5_z0_find=False,
        )""",
    """tile = Path(TILE)
run(tile, next(tile.glob("*.txt")), "2009-08-11")""",
    """import numpy as np
from osgeo import gdal

gdal.UseExceptions()
data = gdal.Open(str(tile / "output_folder" / "0_0" / "TMRT_0_0.tif")).ReadAsArray()
assert data.shape == (24, SIZE, SIZE), data.shape
print("mean TMRT 14:00", round(float(np.nanmean(data[14])), 6))""",
]


def main():
    source, work = Path(sys.argv[1]), Path(sys.argv[2])
    tile = work / "tile"
    tile.mkdir(parents=True)
    for name in ("Building_DSM.tif", "DEM.tif", "Trees.tif", "Landcover.tif"):
        gdal.Translate(str(tile / name), str(source / name), srcWin=[OFFSET, OFFSET, SIZE, SIZE])
    shutil.copy(source / "ownmet_Forcing_data.txt", tile / "Austin_typical_2009_UTC-5.txt")

    header = f"TILE = {str(tile)!r}\nSIZE = {SIZE}"
    nb = new_notebook(cells=[new_code_cell(header)] + [new_code_cell(c) for c in CELLS])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, kernel_name="python3").execute()
    printed = [o.get("text", "") for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "stream"]
    print(sys.platform, "|", [line for text in printed for line in text.splitlines() if "mean TMRT" in line])
    shutil.rmtree(work)
    print("NOTEBOOK RUN PASSED")


if __name__ == "__main__":
    main()
