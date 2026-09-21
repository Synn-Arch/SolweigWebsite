# SOLWEIG-light

A CPU-native SOLWEIG implementation under development, using NumPy, SciPy and
Numba. The compatibility baseline is SOLWEIG-GPU commit
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec`.

The local own-met TIFF workflow runs without Torch or CUDA. It preserves the
chronological model and streams output bands. Original CPU references verify
24-step cases with cold/warm geometry, land cover, UHI and directional wind.
This is a correctness-first implementation, not a completed release. A local
candidate-to-candidate optimization matrix passed independent promotion
review. The reviewed source is integrated and passed installed-wheel
verification.

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

An isolated exact optimization candidate based on checkpoint `8ca23d4`
completed its matrix with all 50 same-host pairs exact against that checkpoint. The
selected four-thread, 1,024-pixel-block setting had a 1.627 median paired
baseline/candidate ratio in the primary geometry-warm cell; cold, first-use,
single-thread and unchanged-default guards were measured separately. This is
local Apple M1 Pro candidate-to-candidate evidence. It is neither an
original-upstream speedup claim nor a P7/P8 completion claim. Independent
review approved local integration; all 49 package files matched the qualified
source and the final installed-wheel checks passed 97 tests without skips. See the
[local CPU optimization report](reports/local_cpu_optimization.md) for the
complete matrix, exactness evidence, review status and limitations.

P7/P8 still require the full original-upstream and tuned-CPU matrix, larger
2048/default-3600 or explicit memory-limit coverage, multi-day execution,
final-source Linux evidence and hosted CI. CUDA is unavailable on the
development machine. Four approved inherited scientific exceptions remain
failed and documented; they do not close the remaining release gates.

See [progress](docs/progress.md), [numerical contract](docs/numerical_contract.md),
[model deviations](docs/model_deviations.md), `TASKS.yaml`, and
`SOLWEIG_LIGHT_IMPLEMENTATION_PLAN.md` for evidence and completion gates.
