# Optional local verification environment

Core own-met verification uses the minimal installed wheel environment, with
Torch, xarray and acquisition dependencies absent. Local optional verification
uses `requirements/optional-macos-arm64-py311.txt` and `tests/optional`. Neither
gate replaces the other, and neither uses unconditional skips.

Install user-facing integrations through `solweig-light[forcing]`,
`solweig-light[wind]` or `solweig-light[inputs]`. The frozen optional test
environment exercises local files and offline acquisition responses; it does
not install or authenticate every service client in the `inputs` extra.

Both the original oracle and candidate optional environment emit the same
NumPy ndarray-size warning when netCDF4 1.7.4 is imported after resetting warning
filters. It originates in `netCDF4._netCDF4.abi3.so`. The exact reproducer is:

```sh
.venv-optional/bin/python -c 'import numpy; import warnings; warnings.simplefilter("error"); import netCDF4'
```

This matches [netCDF4 issue 1354](https://github.com/Unidata/netcdf4-python/issues/1354).
The project's [merged fix 1471](https://github.com/Unidata/netcdf4-python/pull/1471)
changes the opaque ndarray declaration's size check. This evidence supports
identifying the known declaration warning; it does not establish universal ABI
compatibility. The real NetCDF value, mask, metadata and write/read tests pass
in this recorded environment. We retain the published dependency and visible
warning rather than introducing a locally patched native library during parity
verification. Rasterio also emits an upstream affine-operator deprecation
warning in local builder tests.

Authenticated/live acquisition remains unverified. Offline service responses
are labeled as mocks; actual downstream raster/vector/NetCDF transformations
execute against real files.

## Final P5 installed-wheel verification

The evidence summary and exact gate commands are in
[`p5_optional.md`](p5_optional.md). Final-source installed-wheel runs passed 58
optional tests in 16.69 seconds and 34 clean forcing-extra tests in 4.75 seconds
with zero skips. The final core run passed 2,881 tests in 220.53 seconds, with
nine genuine chronological TIFF cases and zero skips. All three installed-wheel
records verify the same final wheel against the worktree source.

The final records are
[`p5_installed_verification.json`](../reports/p5_installed_verification.json),
[`p5_optional_installed_verification.json`](../reports/p5_optional_installed_verification.json)
and [`p5_forcing_extra_verification.json`](../reports/p5_forcing_extra_verification.json).
These bind executed JUnit results to the installed wheel, source files,
environment and reference manifests. The forcing-only gate verifies the
`forcing` extra without Torch, rasterio, GeoPandas or acquisition clients;
Shapely is an explicit dependency of its local geographic normalization.

Original CPU, explicitly patched WRF, AST-extracted input-helper and offline
mock evidence remain distinct. `wrf_timestamp_v1` does not repair the recorded
duplicate-domain/hour failure. Authenticated acquisition, unavailable CUDA and
unexecuted remote CI remain outside the executed local verification claims.
