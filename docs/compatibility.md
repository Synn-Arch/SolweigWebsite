# Opt-in legacy compatibility

The normal distribution is `solweig-light`, imported as `solweig_light`.
It does not install the `solweig_gpu` namespace or `thermal_comfort` executable.
The separate `solweig-light-compat` distribution supplies those legacy names and
requires exactly the matching main package version.

Install both wheels into a clean environment with the main runtime dependencies:

```sh
python -m pip install solweig_light-0.1.0.dev0-py3-none-any.whl solweig_light_compat-0.1.0.dev0-py3-none-any.whl
```

For a local development install, use `python -m pip install . ./compat` from the
repository root. Upstream reference environments must remain separate.

The following imports forward the same CPU function objects, preserving their
Python signatures:

```python
from solweig_gpu import thermal_comfort, preprocess, run_walls_aspect, calculate_svf, run_utci_tiles
from solweig_gpu.solweig_gpu import thermal_comfort
```

The `thermal_comfort` executable delegates to `solweig_light.cli`. Implemented
numerical modules are also forwarded under `solweig_gpu.solweig`, `shadow`,
`walls_aspect`, `calculate_utci`, `calculate_wbgt`, `preprocessor`,
`sun_position`, and `Tgmaps_v1`.

The companion exports all seven public functions, including `build_inputs` and
`build_wind_ext_coeff`. Their optional dependencies load only when called.
Legacy `create_inputs` and `wind_ext_coeff` module imports are forwarded too.
Acquisition verification uses explicitly offline mocks; authenticated live
services have not been verified. WRF conversion uses the documented
`wrf_timestamp_v1` minimal repair and separately labeled patched references.


Do not install `solweig-gpu` and `solweig-light-compat` in the same environment.
They own overlapping package files and executable names. The companion checks
installed `solweig-gpu` distribution metadata before forwarding and raises
`ImportError` when it detects that conflict. This cannot prevent an installation
from overwriting files, or detect every unmanaged source checkout. If upstream
files win the import resolution, the companion cannot execute its diagnostic.
Use a clean environment to avoid that ambiguity.

If both distributions were installed, remove the conflicting distribution and
reinstall the intended owner: uninstalling an overlapping package can delete
files belonging to the other distribution. A fresh environment is preferable.
The compatibility tests create synthetic upstream distribution metadata to
verify the rejection; they never install upstream into the candidate or oracle
environment.


## Retained scientific limitations

The compatibility profile retains upstream SVF initialization and float32 ground-view accumulation. The user approved a narrow release exception for four named independent scientific-check failures; they remain recorded as failures, not passes. See [model deviations](model_deviations.md#approved-compatibility-release-exception-inherited-svf-and-umep-failures) and the [evidence-bound policy](../reports/scientific_release_exceptions.json). The separately approved portable-math reference amendment is documented in [portable math profile v1](portable_math_profile.md); release qualification remains incomplete.


The development runtime uses `solweig-portable-sleef-5a1d179d-v1` in both normal and opt-in legacy namespaces, without a Torch or MKL runtime dependency. Geometry and checkpoint identities include its implementation/runtime fingerprint; incompatible old checkpoint state is rejected, and derived geometry is regenerated. OriginalCura/MKL comparisons remain separately labeled rather than promised as universal backend parity.
