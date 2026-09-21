# P5 optional workflows: evidence and boundaries

P5 adds local forcing conversion, directional wind generation and input
construction to the CPU implementation. Final-source installed-wheel runs
passed 58 optional tests in 16.69 seconds and 34 forcing-extra tests in 4.75
seconds, each with zero failures, errors or skips. The final core run passed
2,881 tests in 220.53 seconds with zero skips, including nine genuine
chronological TIFF cases. All three records below identify the same final wheel
and verify its installed files against the worktree source.
This document does not change milestone status or establish repository release
completion.

## Installed-wheel gates

Run from the repository root, with the candidate installed as a wheel rather
than imported through `PYTHONPATH`:

```sh
.venv-wheel/bin/python -m pytest tests/differential tests/unit -q --junitxml=reports/p5_installed_tests.xml
.venv-optional/bin/python -m pytest tests/optional -q --basetemp=reports/runs/p5-optional-pytest --junitxml=reports/p5_optional_installed_tests.xml
.venv-forcing/bin/python -m pytest tests/optional/test_forcing_optional.py tests/optional/test_public_forcing.py -q --basetemp=reports/runs/p5-forcing-pytest --junitxml=reports/p5_forcing_extra_tests.xml
```

The corresponding durable records are
[`p5_installed_verification.json`](../reports/p5_installed_verification.json),
[`p5_optional_installed_verification.json`](../reports/p5_optional_installed_verification.json)
and [`p5_forcing_extra_verification.json`](../reports/p5_forcing_extra_verification.json).
The recorders check executed JUnit results, installed/source/wheel agreement,
dependency absence and reference provenance. The record commands are:

```sh
.venv-wheel/bin/python tools/record_p1_verification.py --milestone p5
.venv-optional/bin/python tools/record_p5_optional_verification.py
.venv-forcing/bin/python tools/record_p5_optional_verification.py --forcing-only
```

The core environment excludes Torch, xarray and acquisition clients. The clean
forcing-only installation excludes Torch, rasterio, GeoPandas and acquisition
clients while executing real NetCDF-to-metfile and public preprocessing tests.
Optional tests use the frozen environment described in
[`optional_environment.md`](optional_environment.md); their known native import
and rasterio warnings remain visible.

## Reference classes and exercised behavior

All upstream packets identify SOLWEIG-GPU commit
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec`. Their manifests distinguish source,
collector, fixture, environment and artifact provenance. References are
collected in the isolated original oracle, never generated from the candidate.

| Packet | Evidence and scope |
| --- | --- |
| [`forcing_original_cpu`](../tests/reference/forcing_original_cpu/manifest.json) | Original CPU forcing behavior, including the original WRF failure. Local ERA5/UHI conversion tests cover variables, units, masks, spatial sampling and timestamps. |
| [`public_forcing_original_cpu`](../tests/reference/public_forcing_original_cpu/manifest.json) | Public preprocessing integration against real local files, including time-coordinate variants, forcing selection and daylight-saving behavior. Patched WRF cases retain separate labels. |
| [`wrf_patched_cpu`](../tests/reference/wrf_patched_cpu/manifest.json) | Explicit `wrf_timestamp_v1` repaired reference; successful WRF comparisons apply to this versioned policy. |
| [`wind_original_cpu`](../tests/reference/wind_original_cpu/manifest.json) | Original directional wind generation, including twelve directions, geometry processing, precedence and raster metadata. |
| [`public_roughness_original_cpu`](../tests/reference/public_roughness_original_cpu/manifest.json) | Public roughness workflow generates and consumes all directional outputs through local preprocessing. |
| [`inputs_original_cpu`](../tests/reference/inputs_original_cpu/manifest.json) | Unchanged AST-extracted original local helpers, with import/bootstrap statements excluded. This is helper evidence, not an unmodified full-module execution. |

The original WRF filter compares the `(datetime, domain)` tuple returned by its
filename parser with a datetime and raises `TypeError`. The one-line
[`wrf_timestamp_v1.patch`](upstream_patches/wrf_timestamp_v1.patch) selects tuple
element zero only at that comparison. Candidate behavior follows this explicit
repair. Tuple ordering, inclusive endpoints and the subsequent meteorological
calculations retain upstream semantics. The observed duplicate-domain/hour
shape failure remains recorded and tested; it is not repaired or counted as a
successful WRF conversion. See [`model_deviations.md`](model_deviations.md).

Input tests execute real clipping, reprojection, grid alignment, polygon
rasterization, WorldCover remapping, DSM addition, tree cleanup and NetCDF
normalization. Nominatim, OSMnx and Earth Engine test responses are offline
mocks. They establish callable boundaries and downstream local behavior,
not authenticated acquisition. WFS/Earth Engine export adapters remain callable
but are not live-service verification. The input builder follows the executed
upstream entrypoint: it produces static rasters; the original docstring's
additional met/wind promise does not correspond to executed calls. Separate
forcing, ERA5 acquisition and wind helpers remain available.

## Public and compatibility surface

The seven interfaces are `thermal_comfort`, `preprocess`, `build_inputs`,
`build_wind_ext_coeff`, `run_walls_aspect`, `calculate_svf` and `run_utci_tiles`.
Installed API tests check their argument contracts; numerical and local
integration tests exercise the applicable workflows. This interface coverage
does not convert mocked acquisition into live-service evidence.

The normal namespace is `solweig_light`. The separately installed compatibility
distribution supplies `solweig_gpu` imports and the legacy `thermal_comfort`
executable. Installed tests verify forwarding, the matching base dependency,
optional-module imports without service dependencies, executable help and
rejection of a synthetic upstream-package conflict before forwarding. These
conflict tests are labeled synthetic installation tests.

## Unavailable or unexecuted evidence

Authenticated acquisition was not executed. CUDA is not available. Remote CI
is configured but was not executed here. No remote publication or mutation was
performed. No P5 performance claim follows from these correctness gates.
Broader robustness, benchmark and release requirements remain governed by the
implementation plan and task gates.
