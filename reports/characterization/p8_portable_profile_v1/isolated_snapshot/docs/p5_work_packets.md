# P5 preparation — start after P4 acceptance

P5 is complete; final evidence is in `docs/p5_optional.md`. Root owns API/preprocessor integration, exports,
compatibility distribution, dependency extras, CLI and installed-wheel gates.
Workers must not edit those shared files concurrently.

1. Meteorological adapters: own new forcing modules, tests and original-only
   collectors. Execute real local ERA5/UHI/WRF NetCDF normalization, spatial
   sampling and metfile construction. Verify columns, units, masks, timestamps,
   time-coordinate variants, inclusive intersections, SSRD conversion and DST.
   First reproduce the audited WRF datetime/tuple failure; do not silently repair
   upstream or present an expected failure as an operational WRF workflow.
2. Directional wind: own a new wind module, tests and collector. Preserve all
   twelve directions, rotations, connectivity, wake combination, clipping,
   first-time roughness selection, building-file precedence and raster metadata.
   DSM fallback must retain original behavior rather than subtracting DEM.
3. Input construction: own local transformation/acquisition modules, tests and
   collector. Run real local clipping, reprojection, alignment, rasterization,
   remapping, DSM/tree processing and NetCDF normalization. Callable network
   boundaries include Nominatim, OSMnx, WFS/Earth Engine and exports. Record/replay
   tests must be labelled offline mocks, never live-service verification.

Use optional lazy imports and declared extras; do not copy upstream's import-time
package installation loop. Core own-met execution must remain free of Torch,
xarray and acquisition clients. Reuse existing optional-local, artifact, API,
CLI and compatibility fixtures, but do not treat wind consumption as evidence
for wind generation, or own-met input as evidence for ERA5/WRF conversion.

A required upstream repair without an existing policy is an escalation boundary.
Keep original failure, any approved patched oracle and candidate output distinct.
No remote publication or authenticated external mutation is implied by these
local development packets.
