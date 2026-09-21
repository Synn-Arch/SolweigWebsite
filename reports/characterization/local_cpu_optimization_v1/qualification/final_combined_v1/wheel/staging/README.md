# SOLWEIG-light

A CPU-native SOLWEIG implementation under development, using NumPy, SciPy and
Numba. The compatibility baseline is SOLWEIG-GPU commit
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec`.

The local own-met TIFF workflow runs without Torch or CUDA. It preserves the
chronological model and streams output bands. Original CPU references verify
24-step cases with cold/warm geometry, land cover, UHI and directional wind.
This is a correctness-first implementation, not a completed release or a
measured performance improvement.

## Development installation

The currently exercised environment is Python 3.11 on macOS ARM64 with native
GDAL 3.13.3. Exact Python dependencies are recorded in
`requirements/macos-arm64-py311.txt`. Install native GDAL before its matching
Python bindings, and use a separate environment from upstream SOLWEIG-GPU.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/macos-arm64-py311.txt
.venv/bin/python -m pip install .
```

```python
from solweig_light import thermal_comfort

thermal_comfort(
    base_path="/absolute/path/to/scene",
    selected_date_str="2020-07-18",
    own_met_file="/absolute/path/to/scene/met.txt",
    ERA_5_z0_find=False,
)
```

The scene contains `Building_DSM.tif`, `DEM.tif` and `Trees.tif`. Explicit relative
meteorological filenames are resolved against the working directory, as upstream.
Default logical tiling and physical options follow the pinned public interface.

The main distribution owns `solweig_light` and the `solweig-light` executable.
Legacy `solweig_gpu` imports and `thermal_comfort` are supplied separately by the
opt-in companion distribution; see [compatibility](docs/compatibility.md).

## Verification and remaining work

```sh
.venv/bin/python -m pytest tests/differential/test_pipeline_reference.py -q
```

This executes real raw/prepared TIFF cases, checks every output band against
original upstream CPU artifacts, and requires Torch to be absent. Broader
source-inspection tests additionally require the exact upstream checkout under
`.upstream/SOLWEIG-GPU`; no upstream package is imported into the candidate runtime.

P0 baseline characterization and the P1 correctness-first gate are complete.
P2 compiled wall/aspect and UTCI kernels and P3 shadows/SVF/lossless visibility
have passed their local verification gates. P4 ground-view/radiation and
chronological integration also passed (2,879 installed-wheel tests).
Geometry measurements and their regressions are recorded in
[the P3 report](reports/p3_geometry_measurements.md); these are not end-to-end
speedup claims. [P4 component measurements](reports/p4_radiation_measurements.md)
record substantial patch-radiation runtime regressions that still require tuning.
Local ERA5/UHI and WRF processing, input construction and roughness-driven wind
generation are implemented. Optional installed-wheel checks cover real local
files; input acquisition uses explicitly labeled offline mocks. WRF references
use the documented one-line timestamp repair, with original failures retained.
Install integrations with `[forcing]`, `[wind]`, or `[inputs]`; see
[optional environment evidence](docs/optional_environment.md).
P5 passed its installed-wheel gates (2,881 core, 58 optional, and 34
forcing-only tests); see [the P5 report](docs/p5_optional.md).
PR CI is configured but has not executed remotely.
P6 cache validation, restart and bounded tile/wind execution passed independent
review: 3,029 core, 78 optional and 34 forcing-only installed tests, with no skips.
[Core resource measurements](reports/p6_runtime_measurements.md) and
[wind resource measurements](reports/p6_wind_measurements.md) record their
source identities and small-workload limitations.
Further kernel optimization, scientific coverage and the full performance
matrix remain required. CUDA is unavailable on the development machine.

See [progress](docs/progress.md), [numerical contract](docs/numerical_contract.md),
[model deviations](docs/model_deviations.md), `TASKS.yaml`, and
`SOLWEIG_LIGHT_IMPLEMENTATION_PLAN.md` for evidence and completion gates.
