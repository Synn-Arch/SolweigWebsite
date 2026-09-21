# SOLWEIG-light: CPU implementation and verification specification

## 0. Decision and evidence status

Build a CPU-native implementation of the existing model, not a reduced-physics model and not a Torch-to-NumPy text substitution. Preserve the public Python functions, legacy command-line arguments, TIFF inputs, logical tiling, and output semantics through a thin compatibility layer. Put the numerical implementation in independently testable, memory-bounded NumPy/Numba kernels.

The recommended optimization order is: establish an executable reference; remove unnecessary output retention; compile wall/aspect and scalar comfort calculations; port exact shadow traversal; reduce visibility storage; fuse ground-view and anisotropic radiation loops; reuse static geometry; then tune parallel scheduling and advanced algorithms. Measurement may change the order after the baseline, but not the correctness requirements.

**Evidence status:** This specification is based on static inspection of the upstream source. No upstream simulation, profiler, numerical comparison, or performance benchmark was executed during this review. All speedup targets below are proposed engineering gates, not measured results. A faster CPU implementation than upstream's CPU fallback is a plausible objective. Being faster than the CUDA implementation on every workload is not an established or defensible promise.

Reference repository: `https://github.com/nvnsudharsan/SOLWEIG-GPU`

Pinned upstream commit: `0d7fe742abeeddd890dd58fc76ed7f78bd47faec`

Commit date: September 2, 2026. Package version in `solweig_gpu/__init__.py`: `2.0.0`. Review date: September 18, 2026. References [S1]–[S16] and [D1]–[D4] are listed at the end. Codex must inspect the complete files and dependency graph at the pinned commit before implementation. This review inspected the principal paths and selected supporting modules, not every upstream line or external scientific reference.

The upstream source includes GPL version 3-or-later notices. Preserve source attribution, notices, and provenance in the derivative repository; do not relabel copied or translated implementation as unrelated permissively licensed code. Record third-party notices separately. [S1, S2]

## 1. Non-negotiable product contract

### 1.1 What “same computation” means

The default compatibility path must retain the upstream spatial resolution, actual sky-patch coordinates and weights, timestep sequence, vegetation treatment, meteorological interpretation, model constants, radiation components, thermal-delay state, and recipient masks. An optimization cannot silently use fewer patches, shorter rays, less vegetation, an isotropic sky, coarser TIFFs, fewer timesteps, a surrogate model, or different wind/temperature corrections.

Separate three concerns:

- **Execution configuration:** threads, internal block size, memory budget, native cache location. These must not change model results beyond the numerical contract.
- **Compatibility semantics:** the pinned upstream's defined behavior, including documented source quirks that affect valid runs.
- **Scientific corrections or approximations:** separately named, versioned, explicitly selected behavior with its own validation and cache identity. These are not the mechanism used to claim default speedup.

Compatibility does not require matching ZIP byte streams, TIFF compression byte layout, temporary files, progress timings, or branding. It does require matching the observable data and metadata contract. New validation for unsafe or previously undefined inputs must be documented as such rather than presented as identical exception behavior.

### 1.2 Public Python surface

Preserve all seven exported functions, not just `thermal_comfort`: `thermal_comfort`, `preprocess`, `build_inputs`, `build_wind_ext_coeff`, `run_walls_aspect`, `calculate_svf`, and `run_utci_tiles`. The source, not the README alone, defines the initial signature inventory. [S1, S2]

The signature seed for the main entry point is:

```python
thermal_comfort(
    base_path, selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif', trees_filename='Trees.tif',
    landcover_filename=None, ERA_5_z0_find=True,
    tile_size=3600, overlap=20, use_own_met=True,
    start_time=None, end_time=None, data_source_type=None,
    data_folder=None, own_met_file=None, use_uhi=True,
    save_tmrt=True, save_svf=False, save_kup=False,
    save_kdown=False, save_lup=False, save_ldown=False,
    save_shadow=False, save_wbgt=False,
    save_ta=False, save_wind=False,
)
```

Preserve positional argument ordering, keyword spelling/case, defaults, return behavior, and path resolution. Generate a full machine-readable signature snapshot from the actual pinned code during task P0. The signature above is a reviewed seed, not a substitute for that snapshot.

Other important contracts include:

| Function | Behavior to retain |
|---|---|
| `preprocess` | Custom raster paths, optional land cover and wind coefficients, own met/ERA5/WRF, optional preprocessing directory, return its path. |
| `run_walls_aspect` | Consume the preprocessing directory and write the existing `walls/` and `aspect/` tile files. |
| `calculate_svf` | Its `base_path` is the preprocessing directory; retain `patch_option=2`, `overwrite=False`, all three existing SVF artifacts and their schemas. |
| `run_utci_tiles` | Retain tile selection, output flags, directory conventions, and `None` return value. |
| `build_wind_ext_coeff` | Retain all keyword-only physical parameters, directional outputs, and directory return value. |
| `build_inputs` | Retain the optional input-building workflow and return value. Isolate external services and optional dependencies rather than replacing the function with a stub. |

Generate and test signatures and representative calls for every public function. Inventory additional documented low-level imports. Full compatibility with arbitrary private Torch tensor internals is not implied by the seven-function contract; classify any additional required imports explicitly before release.

### 1.3 Import and executable compatibility

Use `solweig-light` as the main distribution and `solweig_light` as its normal import namespace. Provide a new `solweig-light` executable. Both expose the existing function/argument semantics without requiring Torch.

For genuinely unchanged imports and commands, ship an **opt-in companion distribution in the same repository**, `solweig-light-compat`, which depends on `solweig-light` and supplies the needed `solweig_gpu` forwarding modules and `thermal_comfort` entry point. Do not call changing an import statement “zero-change compatibility.”

The compatibility distribution replaces upstream in a clean environment. Do not install it alongside `solweig-gpu`: they would own overlapping import paths and potentially executable names. Include a diagnostic and explicit installation documentation for this conflict. Run upstream and light in separate subprocess environments for differential tests. Use static source/signature extraction where importing upstream would load heavy dependencies.

Do not return Torch tensors or import Torch in the default TIFF-to-TIFF path. An optional adapter for extra low-level interfaces is permissible only if expressly scoped and separately tested; it must not be a hidden numerical fallback.

### 1.4 CLI contract

Retain `thermal_comfort --base_path ... --date ...` and these legacy flags: [S3]

```text
--building_dsm --dem --trees --landcover
--tile_size --overlap
--use_own_met --own_metfile --data_source_type --data_folder
--start --end --era5_z0_find --use_uhi
--save_tmrt --save_svf --save_kup --save_kdown
--save_lup --save_ldown --save_shadow --save_wbgt
--save_ta --save_wind --help --version
```

Preserve accepted boolean strings: yes/true/t/1 and no/false/f/0, case-insensitively. Note the real naming difference between CLI `--own_metfile` and Python `own_met_file`.

The source Python default is `ERA_5_z0_find=True`, while the CLI resolves its unspecified value from whether `data_folder` is provided. Preserve this distinction. Test command errors, return codes, required arguments and path resolution. Permit intentional branding/version differences in snapshots. Add CPU tuning through new optional flags, environment variables, or a separate configuration API without altering legacy signatures.

### 1.5 Raster, forcing and output contract

Building DSM is building-plus-terrain elevation in the radiation path. DEM is terrain elevation. Trees are treated as heights in the reviewed preparation code, including a quarter-height trunk zone. Preserve the actual conversions, not an interpretation inferred from filenames. Wind preprocessing has a different height requirement, discussed in Section 8. [S4, S10]

Preserve logical tile names `i_j`, directory layout, raster dimensions, geotransform, CRS, band order, values, finite/NaN/sentinel locations, and per-band `Time` metadata. Do not introduce different roof masking into every radiation field simply because UTCI masks buildings.

Runtime outputs are under `{base_path}/output_folder/{tile_key}/`, including `UTCI_{key}.tif`, `TMRT_{key}.tif`, optional `Kup`, `Kdown`, `Lup`, `Ldown`, `Shadow`, `WBGT`, `Ta`, and `Wind` files. These use float32 GeoTIFF bands. `SVF_{key}.tif` has a cache-dependent write condition in the pinned driver: it is written when `save_svf` is true and the cache did not already exist. Characterize both cold and warm behavior instead of assuming the flag always creates this duplicate output. [S4]

Standalone SVF artifacts are:

```text
SVF/SkyViewFactor_{key}.tif
SVF/svfs_{key}.zip
SVF/shadowmats_{key}.npz
```

The NPZ members are `shadowmat`, `vegshadowmat`, and `vbshmat`, exported as float32 arrays in `(rows, cols, patches)` order. The ZIP holds the existing 15 named SVF TIFFs. The runtime also creates SVF cache artifacts on a cache miss independently of the ordinary `save_svf` flag. Maintain these external contracts in compatibility mode. A packed native cache is additional internal storage, not a replacement file disguised under the legacy schema. [S4, S5]

## 2. Source audit and optimization hypotheses

| Source finding | Consequence and proposed action |
|---|---|
| `walls_aspect.findwalls` traverses pixels in Python; the aspect routine repeats nested loops for 180 rotated filters. | Compile exact stencils and wall-only sparse filters. Preserve rotation interpolation, border bounds, strict tie ordering and fallback derivatives. [S6] |
| `shadow.shadow` runs a Python-controlled ray-step loop over full-tile Torch operations and temporary buffers. | Port the full discrete recurrence to compiled loops. Compare step-major fused loops with pixel/block-major gathers rather than assuming one traversal always wins. [S5] |
| SVF allocates three float32 `N × P` visibility arrays; the driver adds another dense `diffsh` array. | Store visibility losslessly in compact form, derive diffuse visibility during use, and consume bounded blocks. [S4, S5] |
| Anisotropic shortwave and longwave integration repeatedly sweep patches and allocate full-raster intermediate fields. | Hoist static angular factors and fuse each pixel's patch reductions while retaining the original patch order. [S7] |
| `gvf_2018a` searches 18 directions and invokes `sunonsurface_2018a` for each daytime step. | Separate fixed traversal geometry from changing radiation/temperature values; reuse only the former. [S7] |
| The timestep driver appends UTCI, TMRT, four radiation components and shadow for every step regardless of most save flags, then stacks the lists. | Stream requested bands and retain only live physics/state buffers. [S4] |
| The UTCI polynomial is a long array expression with repeated powers and temporary tensors. | Compile a scalar polynomial and fuse its evaluation with the pointwise comfort calculation. [S8] |
| SVF cache eligibility uses file existence, without validating input content or patch configuration. | Add dependency-addressed native caches and explicit validation/import rules for legacy caches. [S4] |
| Legacy SVF ZIP generation uses shared temporary names such as `svf.tif` in one output directory. | Give each tile/attempt a unique temporary directory before enabling concurrent SVF writers. [S5] |
| The selected end-to-end test unconditionally calls `pytest.skip`. | Add genuinely executed reference comparisons; inherited test success is not end-to-end verification. [S12] |

These are structural findings, not a measured ranking of runtime cost. In P0 measure both the complete user workflow and its stages. In particular, walls/aspect may dominate an initial run, whereas radiation may dominate runs with geometry already cached.

### 2.1 Memory arithmetic

The implemented default patch option is **2**, whose band counts sum to **153**. Other implemented options sum to 145, 305 and 609. Some source docstrings instead mention 144/2304, which is not the actual option table. Snapshot the generated patch values and order. [S5]

For `N = rows × cols` and `P = 153`, the three float32 visibility cubes require `3 × N × P × 4` bytes. At 1,000 × 1,000 pixels this is 1.836 GB in decimal units. Including `diffsh` makes 2.448 GB, about 2.28 GiB, before state, scratch arrays and outputs. At 3,600 × 3,600 pixels those four arrays alone are about 31.73 GB, or 29.55 GiB. These are calculated storage sizes, not observed peak RSS.

If and only if all three channels are demonstrably binary, packing each channel across patches takes `3 × N × ceil(P/8)` bytes, or 60 MB for one million pixels. Do not assume the binary invariant from variable names: inspect values on short-ray, zenith, first-step vegetation, bush and boundary fixtures. If a channel has additional exact categories, use a lossless codebook/multibit encoding. If unexpected fractional or nonfinite values occur, use a lossless fallback and report them. A cast to `bool` is not acceptable evidence of equivalence.

Start with validated compact integer storage where possible, then benchmark packing. Decode only the active block, or decode a value directly in a reduction. Never reconstruct all four full-domain float cubes for normal execution. Legacy NPZ export can use disk-backed `.npy` members and bounded streaming into a ZIP container rather than a second in-memory float cube.

## 3. Repository and data architecture

Recommended repository structure:

```text
solweig-light/
  pyproject.toml
  src/solweig_light/
    __init__.py              # Seven compatibility-equivalent public exports
    api.py                  # Thin argument/path adapters
    cli.py                  # Legacy semantics and optional CPU controls
    config.py               # ExecutionConfig and explicit model policy
    pipeline.py             # Stage orchestration and chronological driver
    models.py               # Scene, forcing, state, outputs and workspace
    io/
      rasters.py            # One geospatial I/O boundary
      meteorology.py        # Text format and normalized forcing
      outputs.py            # Streaming multi-band writers
      legacy_svf.py         # Exact external ZIP/NPZ schemas
    geometry/
      walls.py
      ray_schedule.py
      shadows.py
      svf.py
      ground_view.py
      visibility.py
    radiation/
      solar.py
      shortwave.py
      longwave.py
      surface_temperature.py
      tmrt.py
    comfort/
      utci.py
      utci_coefficients.py
      wbgt.py
    forcing/
      era5.py
      wrf.py
      uhi.py
    inputs/
      build.py              # Optional network/input acquisition
      wind.py
    runtime/
      cache.py
      scheduler.py
      memory.py
      diagnostics.py
  compat/                   # Separate opt-in compatibility distribution
  tests/
    contract/ unit/ differential/ integration/ scientific/
  tools/
    snapshot_upstream.py
    generate_fixtures.py
    generate_reference.py
    compare_outputs.py
    benchmark.py
  benchmarks/
    cases/ protocols/ results/
  docs/
    compatibility.md
    model_deviations.md
    performance.md
    provenance.md
```

Use plain Python dataclasses at the orchestration boundary and typed arrays/scalars inside compiled functions. Avoid a monolithic Numba class and avoid passing GDAL datasets, dictionaries, pandas objects or Python callbacks into hot kernels.

Define these ownership boundaries:

| Object | Contents and lifetime |
|---|---|
| `StaticScene` | Input arrays, masks, geometry metadata and logical domain. Read-only after normalization. |
| `ForcingTimeline` | Ordered timestamps, base meteorology, UHI, wind direction, solar geometry, location and units. Scalars stay scalar when spatially uniform. |
| `GeometryCache` | Walls/aspects, patch tables, compact visibility, derived angular factors, and verified reusable traversal descriptors. |
| `SimulationState` | All carried temperature-delay maps and scalar flags/counters, including `CI`, `firstdaytime`, `timeadd`, `timestepdec`, directional `Tgmap1*`, and `TgOut1`. |
| `Workspace` | Reusable per-worker buffers with explicit live ranges and initialized regions. |
| `OutputPlan` | Requested artifacts and the dependency closure of fields needed to compute them. |

Separate *logical tiles* from *execution blocks*. A block controls locality and memory, not the model's domain, forcing cell aggregation, solar location, padding, or tile boundaries. Current preprocessing and solar preparation use tile-specific information, so changing public `tile_size` is not automatically a result-preserving optimization. [S2, S4, S9]

The chronological loop is conceptually:

```text
prepare and validate logical scene
load/build static geometry and forcing
initialize complete state
open requested output datasets
for timestep in original order:
    calculate timestep scalars and select wind direction
    compute direct shadows and surface terms
    gather neighboring radiative contributions using read-only source fields
    update thermal-delay state with the same ordering as the reference
    reduce shortwave/longwave components and compute TMRT
    evaluate UTCI and optional WBGT/diagnostics
    write requested bands/blocks and release dead scratch buffers
finalize datasets and publish completion manifest
```

Do not parallelize dependent timesteps. Where a stage reads neighbors, do not overwrite its input field while another block still needs the old value. Use a stage barrier and separate input/output buffers or prove that the update is pointwise. Nighttime skips daytime-only work but still computes longwave and performs the original state transitions. [S4, S7]

## 4. Numerical implementation policy

Use NumPy for array ownership, SciPy for already compiled operations whose exact semantics are required, and Numba for hot loops. Keep a straightforward scalar/NumPy reference implementation for small tests, separate from optimized code. The production path must not silently fall back to Torch or object-mode hot loops.

Start hot kernels with `@njit(cache=True, nogil=True, fastmath=False)`. Enable `parallel=True` only after serial differential tests pass and ownership/dependency analysis establishes safe parallel iterations. Numba supports efficient explicit loops; fast-math can change reassociation and nonfinite results, so it is not a default speed switch. [D1, D2]

`fastmath=False` is necessary but not sufficient for parity. Explicitly characterize and preserve dtype promotion, rounding of ray coordinates, comparison thresholds, powers/transcendentals, signed zeros where relevant, min/max NaN behavior, integer conversions, division-by-zero behavior, and evaluation order. A direct scalar port can promote float32 intermediates to float64 and change a shadow threshold. Use explicit casts at semantically relevant boundaries when necessary. A float64 scientific oracle is useful but does not define float32 legacy compatibility.

Pass configurable constants and model tables as arguments or immutable versioned arrays. Do not rely on mutating module globals that a cached Numba function may have captured. Version/clear compiled caches when helper implementations or semantics change; Numba's cache has dependency-invalidation limitations. Test fresh and warm processes. [D4]

Produce Numba type and parallel diagnostics for optimized kernels. Inspect generated code or hardware counters only when a measured bottleneck requires it. A `parallel=True` decorator by itself is not proof of useful parallelism or SIMD. Ensure masks and negative ray offsets use deliberately chosen signed index types. [D2]

Keep serial and parallel implementations sharing scalar logic where doing so does not compromise the independence of the reference oracle. Do not spend engineering effort replacing fast compiled SciPy routines solely to remove a dependency. Consider Cython or C++ only for a measured residual bottleneck that Numba cannot adequately express, with the same verification gates.

## 5. Kernel-specific optimization work

### 5.1 Wall heights and aspect

Compile `findwalls` as a four-neighbor maximum stencil with the exact wall threshold, excluded border and output dtype. Do not substitute a different gradient detector. [S6]

For aspect, generate the 180 rotated filters once per scale using the original SciPy rotation orders, reshape/boundary options, rounding and special-angle edits. Convert each filter into sparse offsets and coefficients. Iterate only over wall pixels in the same supported interior. Within each pixel, visit angles in the original order and update only when the new score is strictly greater, preserving tie selection and the height-side comparison. Preserve the gradient fallback when the chosen angle remains zero.

Parallelize ownership of wall pixels. Benchmark sparse filtering against equivalent compiled convolution only after parity. Cache the exact filter tables by scale and implementation version. Test non-square arrays, coarse/fine resolution, corners, nearly equal competing orientations, and thresholds immediately below/at/above three meters.

### 5.2 Direct and sky-patch shadows

There are different shadow implementations: SVF's `shadow` and the wall-height shadow routine used in `Solweig_2022a_calc`. Both must be ported, including wall sun/shade quantities, vegetation top/trunk logic, combined masks and bush handling. They cannot be replaced by a single simplified building-shadow function. [S5, S7]

First implement a faithful compiled recurrence with the original ray schedule and global conditions. Precompute exact `(dx, dy, dz)` schedules and source/destination bounds per direction/altitude, with the original numerical rounding and termination. Reuse schedules for identical geometry/angles only. Remove Python/Torch scalar orchestration, repeated scalar trig, and whole-array expression temporaries.

Then compare two implementations on representative scenes:

1. A compiled step-major traversal, retaining global step conditions and reusing workspace arrays.
2. A compiled pixel/block-major traversal with scalar per-ray state and contiguous output ownership.

The second can avoid repeated full-image writes, but reads scattered source cells. Its benefit is an empirical question. Audit global bush predicates, first-step corrections, padded zeros and state retained outside updated slices before claiming pixel independence. Keep a correct step-major fallback for cases whose independence is not established.

Do not stop a ray merely because a building shadow is found when vegetation, wall height or combined outputs can still change. Do not replace an absolute-elevation termination test with a height-range test without equivalence evidence. Test zero azimuth, quadrant boundaries, tie-rounded offsets, zenith, shallow sun, one-step rays, negative elevations and domain edges.

### 5.3 SVF and visibility

Preserve the exact patch table, order, directional intervals, annulus weights, clipping and special vegetation corrections, including the small directional correction in the SVF tail. Preserve the hardcoded transmission used to form total SVF in compatibility mode. [S5]

Compute each patch's visibility, accumulate all required directional SVFs, and encode visibility losslessly. Avoid retaining full temporary float masks for every patch. Use one writer per packed word or pixel to avoid bit-setting races. Validate bit order, last-word padding, byte order, codebooks and dimensions with round-trip tests.

Support memory-mapped native cache storage with bounded access. Prototype simple `.npy`/memmap blocks plus an atomic manifest before adding a more elaborate storage system. Use Zarr or another chunked format only if a measured compression/concurrency need justifies the dependency.

Preserve all information consumed later by anisotropic radiation. A total SVF alone is insufficient. Calculate `diffsh` from the original channel values and transmission while reducing radiation instead of storing another `N × P` float array. Export legacy sidecars exactly when the compatibility contract requires them, using unique per-tile temporary directories.

### 5.4 Ground-view radiation

In the reviewed implementation, 18 search directions and much of the addressing are fixed, but `shadow`, wall sun exposure, temperatures and emitted/reflected radiation change each timestep. Therefore cache geometry descriptors, not the complete values returned by `gvf_2018a`. [S7]

Precompute the 18 offset schedules and valid bounds. Derive more compact stopping/visibility descriptors where the original binary building-mask recurrence permits them. Evaluate multiple emitted/reflected channels in one gather traversal and accumulate cardinal components together. Do not allocate an `N × directions × distance` index table by default: it can replace one memory problem with another. Compare formula-based addressing, compact stopping indices, and selective sparse descriptors.

Audit in-place changes and outside-slice workspace history in `sunonsurface_2018a`. Strict compatibility must reproduce their observable effects or explicitly classify the affected case as an upstream deviation. A clean zero-fill in a rewritten kernel is not automatically equivalent to the original.

Use stage-local input snapshots for dynamic neighbor fields. Test all directional components and delayed state, not only final TMRT, because errors can cancel in a summed quantity.

### 5.5 Anisotropic shortwave and longwave

Cache immutable patch altitude/azimuth, solid angles, trigonometric factors, cardinal-direction membership, altitude-group membership, and any visibility-weighted geometry moments proven independent of forcing. Avoid recreating patch arrays or transferring them to NumPy to call `unique` every timestep. [S7]

Compute Perez/sky-emissivity coefficients once per timestep and location. In the compiled reduction, parallelize pixels/blocks and keep each pixel's patch loop serial in the original order. Accumulate outputs into scalar local variables rather than allocating a full raster for every patch contribution.

Retain sunlit/shaded wall classification and vegetation/building distinctions. Do not replace dynamic visibility-weighted sky radiation with an isotropic SVF multiplication.

The reviewed longwave path has a second patch sweep for reflected longwave after sky terms are known. A promising subsequent optimization is to precompute the masked angular sums used in that reflection and apply the dynamic field once. This changes floating-point grouping, so it must pass numerical gates before becoming default. Apply the same reasoning to altitude-group factorizations: prove the algebra and measure the result, rather than assuming a low-rank or grouped approximation is exact.

Fuse pointwise flux combination, the fourth-root TMRT calculation, and comfort evaluation where it reduces memory traffic without obscuring diagnostics. Preserve `273.2` in the pinned TMRT conversion and all existing constants in compatibility mode. Retain a diagnostic route that exposes intermediate components for testing. [S7]

### 5.6 UTCI and WBGT

Translate the complete UTCI polynomial into a compiled scalar routine. Extract coefficient/exponent data mechanically from the pinned expression, retain provenance and a coefficient checksum, and verify reconstructed terms against the source. Preserve the initial expression/evaluation semantics before experimenting with common subexpressions, precomputed powers or multivariate Horner evaluation. A different evaluator must pass the same maximum-error gates. [S8]

Compute saturation vapor pressure once per timestep when air temperature and humidity are uniform. Fuse valid-recipient selection and UTCI output assignment. Keep scalar forcing scalar rather than creating constant full-raster matrices. Do not remove building or vegetation cells from geometry/radiation merely because they are not UTCI recipients.

Preserve the upstream missing-value policy and the driver's `0.15 m/s` wind floor. Replacing this with another library's validity clipping is a model change. Test physical-validity diagnostics separately from compatibility of the polynomial on its accepted inputs.

WBGT is part of feature parity. The current pipeline computes wet-bulb temperature on the meteorological timeline and uses spatial globe temperature in the output calculation. Retain that reuse; optimize the spatial calculation first. Preserve units, iteration limit, convergence criterion and treatment of nonconvergence. A per-element early-convergence rewrite can change results relative to the existing array-wide stopping test and requires explicit verification. Independently verify the shade/sun selection and Kelvin/Celsius conversions rather than silently correcting them while porting. [S4, S11]

## 6. Streaming I/O, memory and caching

Open only requested output datasets and write one timestep/band or bounded window at a time. Retain all intermediates required by TMRT and UTCI even if their save flags are false, but do not append them to time-history lists. Maintain original band metadata and layout semantics. A writer must finish consuming a buffer before the buffer is reused.

Use a single owner per writable GDAL dataset. Give workers independent dataset handles and output files, and use unique temporary directories. Avoid sharing ordinary dataset/band objects across worker threads or forking after native I/O activity. These constraints follow GDAL's threading guidance. [D3]

Keep native cache storage separate from compatibility exports. A cache key must include relevant input content fingerprints, shape, full transform and CRS, NoData semantics, logical domain/extent, boundary policy, dtype policy, actual patch arrays/order, geometry/model implementation version, wall/filter parameters, canopy/trunk definitions and transmission wherever used by the cached quantity. Forcing caches additionally include timestamps, selected date, location/timezone policy, source units, UHI/wind options and input content. Cache dependencies should be per quantity rather than one unnecessarily broad key.

Hashing is work: record its time, cache verified file fingerprints in manifests, and define the trust model. File size and modification time alone are not authoritative content identity for externally mutable rasters. Never reuse an incompatible native cache silently.

Import legacy ZIP/NPZ caches only through an explicit validation/trust policy. They lack sufficient provenance for unconditional scientific equivalence. Check member schemas, shapes, patch count, dtypes and coordinate metadata; recompute when required provenance cannot be established. A trusted-legacy mode must say what was trusted. In particular, standalone `patch_option` and the driver's hardcoded default must not silently disagree. [S2, S4]

Publish cache entries atomically with a manifest/checksum written last. Use locks or equivalent single-producer coordination, detect partial/corrupt files, and test concurrent identical requests. Preserve original user inputs; do not repair them in place. Report failure as failure, not as a successfully completed partial tile.

Create a live-array inventory and memory estimator. The normal hot path should use compact `O(NP)` geometry and `O(N)` state, without float `O(NP)` working cubes or `O(TN)` retained output histories. For fixed scene size, adding timesteps must not cause linear raster-memory growth; small forcing metadata may grow with time. Include mapped/native buffers, writer queues and all workers in measurements.

If a logical tile cannot fit its live state budget, use stage-wise execution blocks and, where necessary, disk-backed state. Do not silently shrink the model's logical tile/domain to meet memory limits. Estimate memory before scheduling and provide a clear actionable error when the selected budget cannot support the required execution plan.

## 7. Tiling and parallel execution

### 7.1 Preserve logical tile semantics

Upstream advances tile origins by `tile_size` and adds overlap toward increasing column and row coordinates only. It does not create a symmetric halo. Preserve these exact windows, edge truncation and output extents in compatibility mode. Merely reducing the public tile size to improve CPU cache behavior can change the answer. [S9]

Internal blocks must sample the original logical tile, retain its boundary/padding semantics, and use its fixed meteorological and solar context. The same logical input run with different internal block sizes must pass differential tests. Surface-radiation gathers can require fields beyond a block, so blocks cannot simply be simulated independently from start to finish without analyzing those dependencies.

A separate domain-correct boundary mode can use symmetric context and crop to a non-overlapping output core. For a flat-grid shadow with a conservative relative obstruction-height bound `Δz`, pixel size `r` in meters, and positive solar altitude `α`, a necessary context estimate is:

```text
halo_pixels >= ceil(Δz / (r * tan(α)))
```

This is a geometric bound, not a complete halo algorithm. Include terrain elevation relative to every possible receiver, vegetation, wall/surface-neighborhood support, discrete rounding margin, and the lowest relevant SVF patch. At shallow positive altitude the support can approach the full available domain. Do not hide an arbitrary maximum radius in the exact mode. Use source-domain context, an exact proven acceleration structure, or clearly report insufficient domain support.

Verify symmetric-domain mode against a whole-domain reference with identical forcing and solar geometry. Its outputs can legitimately differ from legacy one-sided tiles. Keep separate cache keys and validation reports; do not mix a boundary correction into a claimed semantics-preserving speedup.

### 7.2 Thread and worker ownership

Default to one active logical tile with Numba parallelism over independent pixels/blocks. For multiple small tiles, benchmark a spawned process pool with an explicitly smaller native thread count per worker. Do not assume maximum workers times maximum threads is optimal.

Select budgets from CPU affinity/container limits and available memory. Enforce an aggregate CPU budget and a memory-based worker cap. Account for Numba, BLAS/OpenMP, compression, existing wind-direction threads, and I/O activity. Configure native libraries before their worker pools initialize. Do not change global thread settings concurrently from arbitrary user threads.

Keep one owner for each output array region and packed word. Patches should not concurrently add into the same SVF pixel without a tested deterministic reduction. Prefer pixel ownership so each pixel sees the original patch sequence. The same rule applies to overlapping wake updates in wind preprocessing.

Warm relevant JIT specializations deliberately and record whether that time is included. Avoid a compilation stampede across workers. Test worker startup on Linux, macOS and Windows with the chosen threading layer; spawn processes instead of inheriting already initialized native runtime state.

No timestep-level parallelism is allowed across thermal-delay dependencies. Independent scenes or separately initialized simulations can run concurrently. A restart must serialize every state component and its exact timestamp/configuration identity; restarting from output TMRT alone is not equivalent to resuming the model.

## 8. Full v2 feature coverage and dependencies

### 8.1 Meteorological preprocessing

Keep own-met processing usable with only the core runtime dependencies. Lazy-load xarray/netCDF-related packages for ERA5/WRF, and acquisition packages for `build_inputs`. Avoid downloading or authenticating during a normal user-supplied TIFF/custom-met run. [S2, S9]

Preserve met column order and units: air temperature, humidity, wind speed, pressure, global/diffuse/direct radiation, optional wind direction, and optional UHI. The reviewed driver reads global radiation at index 14, direction at index 23 and UHI at index 24. Do not rely on a rewritten header alone to establish compatibility. [S4]

For ERA5/WRF preprocessing, precompute tile-to-grid selection/aggregation mappings. The reviewed metfile loop can read a full source time slice repeatedly across tiles and variables; batch or share those reads without changing nearest-cell versus polygon-mean selection. Group identical forcing selections when safe, but retain tile-specific geographic/solar context. [S9]

Validate ERA5 time-coordinate normalization and accumulation/unit conversion using small frozen NetCDF fixtures. Validate WRF filename parsing, Earth-relative wind rotation from `COSALPHA`/`SINALPHA`, selected times and dimensionality. Preserve local-day handling and explicitly test daylight-saving transitions. Do not silently regularize time gaps or replace duplicated local timestamps with invented samples.

Precompute UHI forcing once for a given source timeline/grid configuration. Preserve its daily aggregation, nighttime threshold and sine-profile details in compatibility mode. A different UHI model is outside the optimization scope. [S9]

### 8.2 Directional wind coefficients

Port and verify wind preprocessing as its own CPU stage. It is already largely NumPy/SciPy based, with rotated rasters, connected components and Python loops over building wake segments. Keep compiled SciPy image operations initially, cache shared direction-independent maps, and compile the measured component/segment loops. Bound concurrency because every rotation can allocate another large raster. Preserve interpolation, padding, connected-component connectivity and overlapping wake-combination semantics. [S10]

The wind module prefers `Buildings.tif` as height above ground, and explicitly warns that its `Building_DSM.tif` fallback does not subtract the DEM. This differs from the radiation DSM convention. Do not silently subtract terrain only in the optimized version and call the result a performance improvement. Preserve legacy behavior with a diagnostic, and offer any normalized-height correction as a documented policy with independent tests. [S10]

The reviewed wind helper selects the first ERA5 time value for surface roughness, while a public wrapper docstring describes averaging available times. Snapshot executable behavior and register the documentation discrepancy. Preserve all 12 direction bins and their nearest-30-degree selection, including ties and wraparound. Test missing bins, missing direction, single-raster legacy inputs and the distinction between low-level support and the staged wrapper's actual file-discovery behavior. [S2, S4, S10]

### 8.3 Optional input construction

Retain `build_inputs` and its actual argument forwarding in an optional extra. Separate network acquisition from deterministic local transformations so the latter can be tested offline. Use small recorded service-response fixtures for normal CI; keep authenticated live smoke tests explicitly opt-in. Do not present a mocked acquisition test as proof of live service compatibility.

Benchmark downloaded/local inputs separately. A network fetch must not be counted as CPU model acceleration. Preserve provenance for acquired rasters and meteorological data.

## 9. Upstream behavior register

Create `docs/model_deviations.md` before optimizing. Every item needs a reproducer, affected versions/modes, reference output, proposed treatment and validation status. The following are source observations or review hypotheses, not a claim that every one is a scientifically confirmed bug:

| Item | Required treatment |
|---|---|
| One-sided tile overlap and tile-local radiation domains. | Preserve legacy windows; validate a separate context-correct mode. [S9] |
| Cache existence checks without model/input identity. | Add explicit validation and a documented trusted-legacy migration path. [S4] |
| Fixed SVF transmission versus seasonal direct-shadow transmission. | Preserve both dependencies and include them in relevant cache keys. [S4, S5] |
| `273.2` in TMRT conversion, versus other Kelvin offsets elsewhere. | Preserve the pinned constant; review scientifically in a separate change. [S7] |
| UHI is added to comfort-stage air temperature after radiation received the base air temperature. | Preserve the dataflow unless a separately versioned correction is selected. [S4] |
| WBGT shade/sun condition and temperature-unit conversions. | Verify independently; do not rename/reverse branches or change units without a dedicated correction test. [S4, S11] |
| Median terrain altitude is replaced by `3.` when positive in location preparation. | Record and reproduce; do not quietly reinterpret it as actual median altitude. [S4] |
| Land-cover normalization acts on a NumPy view/copy derived from a Torch tensor. | Test CPU/GPU aliasing and downstream land-cover values; this is a potential device-dependent behavior, not verified here. [S4] |
| Shadow/GVF temporary-array and in-place mutation ordering. | Test edge/first-step effects before changing loop order or buffer initialization. [S5, S7] |
| Single-row met input through `np.loadtxt` without explicit 2D normalization. | Characterize the failure and document any new supported single-timestep behavior. [S2] |
| Timestamps use selected base date plus band hour/minute. | Do not silently claim correct multi-day or DST-disambiguated metadata if the reference does not provide it. Version improvements separately. [S4] |
| Wind DSM height convention and first-time versus mean roughness documentation. | Preserve/report actual behavior; test corrected policies separately. [S10] |
| Broad exceptions can turn wind/preprocessing failures into warnings. | Characterize defined legacy behavior; modern strict mode must distinguish failed, skipped and completed stages. [S2, S6] |

Use the upstream CPU path as the executable CPU differential baseline, but also compare upstream CUDA outputs when available. If they differ materially, do not silently assume the CPU fallback represents the GPU outputs the user currently relies on. Diagnose the difference and make any compatibility-profile choice explicit before publishing parity claims. An unresolved device-dependent discrepancy blocks a universal parity claim for the affected cases.

## 10. Verification plan

### 10.1 Build an independent reference harness

Create a locked upstream environment at the pinned commit and a separate light environment. Ensure upstream CPU runs cannot select CUDA before importing any upstream modules. Record Python, Torch, NumPy, SciPy, GDAL and all relevant package versions, OS, architecture, thread settings and source hash.

Run upstream through a subprocess adapter that captures outputs and selected intermediate arrays. Avoid import-namespace collisions with the compatibility shim. Store small reference arrays in safe, non-pickled formats with fixture hashes and provenance. Generate reference outputs from upstream, never from the candidate implementation.

If upstream cannot execute a case, report the exact failure. A necessary minimal repair must be a separate audited patch with a patch hash, rationale and before/after evidence. Such outputs are a **patched reference**, not an unmodified-upstream result. Keep failing cases visible; neither skip them silently nor invent goldens.

Reference categories should distinguish: original upstream CPU, original upstream CUDA, minimally patched upstream, independently calculated analytical/scalar expectations, and candidate output. Never merge these labels into a single unexplained “expected” file.

### 10.2 Contract tests

Test installed wheels, not only an editable source tree. In an environment without Torch or CUDA, check all seven exported signatures, return values, legacy import paths in the compatibility distribution, exact CLI argument parsing and functional calls.

Test relative and absolute TIFF paths, custom preprocessing directory, selected tile keys, all save flags, directional and legacy wind paths, existing and missing cache outputs, and meaningful failure modes. Cover every save flag independently plus all-off/default/all-on combinations and selected interactions. Do not require all 1,024 combinations when a dependency-coverage analysis demonstrates a smaller complete suite.

Compare TIFF dimensions, full affine transform, equivalent CRS, band dtype, exact band count/order, per-band `Time`, masks and sentinel locations. Match NoData *metadata presence/value* separately from array NaNs. The writer must not convert one convention into the other without a documented contract change. Compare ZIP member names and their TIFF schemas, and NPZ names/shapes/dtypes/order. Compression bytes and container timestamps need not be identical.

Test hot-cache versus cold-cache artifact behavior, including the duplicate-SVF condition. Add corruption, interrupted-write, permission-error and simultaneous-writer tests. A completed manifest must imply every requested artifact is complete.

### 10.3 Numerical differential tests

Compare at intermediate stage boundaries as well as final outputs:

```text
normalized inputs and masks
wall height and aspect
patch coordinates and weights
building / vegetation / combined visibility and wall-shadow outputs
all SVF channels
solar position and per-timestep scalar coefficients
direct/diffuse/reflected shortwave components
sky/vegetation/wall/reflected longwave components
all ground-view and cardinal components
thermal-delay state after every timestep
TMRT, UTCI, WBGT, Ta and wind diagnostics
```

A comparison first checks shapes, finite/NaN/positive-infinity/negative-infinity and sentinel masks. It then compares values on the same valid cells. For each field report bias, MAE, RMSE, p95/p99 absolute error, maximum absolute error, relevant relative error, coordinates/timestep of worst errors, and spatial error maps. Separate seams, shadows, roofs, canopy and open-sky regions. A small whole-domain RMSE does not excuse a wrong shadow boundary or a different missing-value mask.

Candidate initial strict-mode limits are below. These are proposed numerical acceptance budgets for reference-valid fixtures, not physical uncertainty estimates or measured attainable errors. In P0/P1 characterize the oracle's precision and cross-platform variability, then freeze justified budgets before optimizing. Do not widen them merely to pass a faster implementation.

| Field | Proposed initial gate |
|---|---|
| Categorical masks, tile indices, patch identity/order, missing-value locations | Exact equality. Preserve extra visibility categories, not only presumed binary classes. |
| Wall heights for the exact stencil | Exact float32 equality on finite inputs. |
| Aspect filter winner and discrete classification | Exact choice; continuous gradient fallback uses a separately characterized angular tolerance. Compare angles modulo 360 only where meaningful. |
| SVF fields | Maximum absolute error at most `1e-6`. |
| Radiative flux components | `atol=0.05 W/m²`, `rtol=1e-5`, plus inspect maximum absolute error and bias. |
| TMRT | Maximum absolute error at most `0.01 °C`. |
| UTCI and WBGT | Maximum absolute error at most `0.02 °C` on the validated reference domain. |
| Thermal state and scalar forcing intermediates | Field-specific frozen limits with units, derived from the above propagation budget; never omit them. |

If the same algorithm has unavoidable cross-platform differences, record the explanation and platform-qualified contract. Maintain stronger exact tests for geometry classifications because a tiny floating-point difference can change a discrete occlusion result. Test out-of-range accepted legacy inputs separately; do not quietly apply a new clipping rule to make those cases easier.

### 10.4 Fixture matrix

Use small synthetic scenes in a projected, meter-based CRS for fast tests and licensed real scenes for realistic validation. Deterministically generate both geometry and forcing; save seeds and hashes. The fixtures must cover:

| Dimension | Required cases |
|---|---|
| Geometry | Empty/open terrain, isolated block, street canyon, courtyard, dense blocks, sloped terrain, negative elevation, asymmetric/non-square scenes. |
| Vegetation | No trees, canopy/trunk gaps, isolated trees, dense canopy, bush cases, canopy near/over buildings, leaf-season boundaries. |
| Solar angles | Night, sunrise/sunset, shallow positive altitude, zenith, quadrant/cardinal boundaries, azimuth wrap, ray-offset rounding ties and one-step rays. |
| Materials | All supported land-cover values, remapping of invalid and vegetation classes, water handling and equal/contrasting emissivity/albedo. |
| Forcing | Clear/cloudy days, changing wind, all direction bins and ties, missing direction/UHI columns, single timestep, continuous day/night sequences and multi-day kernel-state tests. |
| Calendar | Leap day/year, local-day boundaries, DST transitions, missing/duplicate timestamps and subhourly inputs where the contract supports them. Unsupported cases must have explicit outcomes. |
| Geospatial I/O | Offsets, mismatched origins, differing CRS encodings, non-square pixels, rotated grids, edge tiles, finite NoData sentinels, NaNs, missing layers and malformed files. |
| Caches/runtime | Empty/warm/corrupt caches, different patch options, changed trees/DEM, interrupted export, differing block/thread/worker counts, restarts and repeated calls in one process. |
| Forcing adapters | Frozen tiny ERA5 and WRF NetCDF files, WRF grid rotation, source time/unit variants, missing variables and UHI on/off. |

Reject or explicitly normalize unsupported geographic/non-square/rotated grids according to the selected input policy. Do not treat degrees as meters merely because a test TIFF opens successfully. Avoid a “flat” fixture where the tree raster unintentionally contains the terrain's absolute elevation as canopy height.

Use sizes such as 16×19 and 64×64 for kernel tests where each kernel supports them, and at least one genuine non-mocked small end-to-end scene large enough for every stencil/ray path. Select real scenes spanning sparse suburban, dense urban and vegetation-rich geometry. Test all implemented patch options independently of the default simulation path.

### 10.5 Independent scientific checks

Reference agreement checks implementation equivalence, not physical validity. Add independently derived checks with clearly stated assumptions: four-neighbor wall heights; shadow direction and geometric length for an isolated block on flat terrain; no-obstacle visibility; bounded SVF for valid cases; absence of direct solar flux at night; known Stefan-Boltzmann subexpressions; directional wind selection; and independently evaluated UTCI coefficient tests.

Use a separately pinned and inspected UMEP/SOLWEIG implementation for matched model components where versions and configuration actually agree. Do not assume an older SOLWEIG release has identical whole-model behavior to this v2 fork. Derive analytical/scalar tests independently of the optimized helper functions to avoid shared mistakes.

Metamorphic tests need preconditions. Internal blocking, thread count and repeated identical inputs should not materially change results. Building occlusion should be monotone when increasing an isolated obstacle under fixed receiver/domain conditions. Do not assert exact rotational symmetry for an asymmetric discrete sky-patch table, or vertical-translation invariance in a legacy implementation with zero-filled boundaries and absolute-elevation bounds, without proving those conditions.

Where UTCI/WBGT stress categories are used downstream, report category changes near thresholds in addition to continuous errors. The point of these tests is not to silently add a new category definition to the API.

Measured environmental validation would be a separate scientific study using observed forcing and temperatures. Passing software tests does not establish new physical accuracy beyond the underlying model.

### 10.6 Concurrency and CI gates

Run serial/parallel and multiple internal block sizes on the same fixtures; compare every state/output field. Stress repeated cache creation, non-overlapping writes, randomized tile order, worker failure, cancellation and restart. Instrument actual parallel execution. Avoid shared mutable Python containers inside `prange`. [D2]

PR CI must include a real CPU-only end-to-end case without an unconditional skip. Test a declared, locked Python/NumPy/Numba/SciPy compatibility matrix rather than promising every current version. Test Linux, macOS and Windows on available runners; include ARM64 support where claimed. Check import/help and the own-met TIFF path with optional acquisition/forcing packages absent.

Nightly CI runs larger real-data differential cases, multi-day state sequences, all patch modes and optional-feature fixtures. Use a stable dedicated host for performance gates; shared CI runners are for functional tests and gross resource failures, not tight speed regressions. Missing GPU hardware may legitimately mark CUDA comparison unavailable, but cannot mark it passed.

## 11. Performance protocol and acceptance

### 11.1 Instrument the complete workload

Report wall-clock time for import/startup, JIT compile/load, input validation/fingerprinting, raster reads, forcing preprocessing, walls/aspect, SVF computation, cache read/export, direct shadows, GVF, shortwave, longwave, state update, TMRT/UTCI, WBGT, and output writes/compression. Also report user-visible end-to-end time, peak process-tree RSS, CPU utilization, thread/worker counts, bytes read/written and cache size.

Use a lightweight stage timer in production diagnostics. Use sampling/native profiling where needed to distinguish Python overhead, native arithmetic, memory traffic and I/O. Python allocation tracing alone is not sufficient for native/mapped buffers or child processes.

Upstream's printed tile timer starts after substantial setup and is not the complete user workflow. Use an external harness around the chosen API/CLI boundary instead of comparing printed messages. [S4]

### 11.2 Baselines and equal-work rules

Measure:

- Upstream pinned CPU path with its defaults, then with reasonable documented thread tuning on the same machine.
- Light with one native thread and with the selected full CPU budget.
- Upstream pinned CUDA path on available hardware, reported as a separate system configuration.

Use the stronger appropriate upstream CPU baseline for the primary CPU speedup claim. Keep logical tiling, raster resolution, patch option, timestep count, forcing, physical options, masks and requested artifacts identical. CUDA timings need completion synchronization at timed kernel boundaries; externally timed completed CLI output includes all work. Never compare an asynchronous launch time with a completed CPU result.

Evaluate matched scenarios:

| Scenario | Included work |
|---|---|
| First use | Fresh process, empty JIT cache, no geometry cache, complete requested outputs. |
| Compiled but geometry-cold | Fresh process with compatible compiled cache, no geometry cache. |
| Geometry-warm | Same validated geometry already prepared for both implementations; include cache loading and requested output work. |
| Kernel-only | Preloaded data and explicitly excluded I/O/compilation; explanatory result, not advertised end-to-end speedup. |

Record operating-system file-cache conditions rather than claiming a cold disk when only application caches were deleted. Report native-cache and legacy-cache scenarios distinctly. Charge required legacy NPZ/ZIP exports to the workflows that request or promise them. A cache-only speedup should not conceal unchanged cold computation.

Use randomized paired A/B ordering, independent processes and at least five repetitions where feasible. For expensive large cases, justify fewer repetitions and report the uncertainty. Report median and tail times, variation/confidence intervals, not only the fastest trial. Record CPU model, allowed cores, memory, OS, package/compiler versions, storage, power mode and GPU model where applicable. Report out-of-memory and failed cases rather than dropping them from aggregates.

Benchmark sizes should include a startup-sensitive small scene, approximately 256², 1024² and 2048² scenes, and the public 3600-pixel logical-tile default or a justified memory-limited case. Include one-step, full-day and longer repeated-day workloads; distinguish the legacy single-date output API from extended state-kernel tests. Use multiple geometry densities. Record the fraction of sunlit, canopy, building and output-valid cells.

### 11.3 Proposed release goals

Freeze hardware and benchmark cases in P0 before implementation tuning. Proposed initial goals for medium/large reference-valid scenes are at least **2× complete geometry-cold speedup** and **3× geometry-warm simulation speedup** relative to the best documented upstream CPU baseline, with **at least 4× lower peak memory** on visibility-dominated cases. Wall/aspect and UTCI kernel speedups of 5–10× are optimization hypotheses, not release claims until measured.

These are goals, not estimates. Baseline characterization may justify different targets for small, I/O-dominated or unusual geometry cases. Any revision must be recorded before evaluating the relevant optimization, not selected after seeing a failing result. Do not weaken numerical tolerances to meet a performance target. No material unexplained regression should remain in a declared priority workload; report smaller-workload startup tradeoffs explicitly.

Do not set a blanket gate requiring CPU to beat every GPU. Report the crossover by scene size, geometry reuse, timestep count and hardware. The product claim should name the regimes actually supported by results.

Irrespective of the relative speed target, release requires: no full time-history output stacks in the streaming path; a tested memory budget; no uncontrolled nested worker pools; working cold-start execution; no repeated static geometry recomputation on a validated warm run; and correct behavior when a requested scene exceeds available memory.

## 12. Dependency-ordered Codex work packages

Use small reviewable changes. Each optimized kernel needs a baseline, a semantic proof sketch, differential results and a benchmark. The task list is a build plan, not permission to stop after an impressive wall-height microbenchmark.

| ID | Deliverable | Dependencies | Acceptance evidence |
|---|---|---|---|
| P0 | Pin source; inspect complete call graph; snapshot API/CLI/artifacts; create isolated oracle and initial profiling harness; open behavior register. | None | Real small upstream run or clearly labeled failure/patch, recorded environment, signature manifest, raw timings, frozen benchmark/tolerance proposal. |
| P1 | New package, compatibility distribution, typed data model, raster/met readers, chronological reference CPU path and streaming writer. | P0 | Seven public surfaces inventoried, actual own-met TIFF-to-TIFF execution, no Torch in light runtime, exact artifact schema tests. Track optional-feature implementation gaps rather than claiming completion. |
| P2 | Exact compiled wall-height/aspect and scalar UTCI kernels. | P1 | Stencil/filter tie cases, coefficient checksum, serial/parallel parity and individual timing/allocation reports. |
| P3 | Both shadow families, actual patch tables, all SVFs, compact lossless visibility and bounded legacy export. | P1 | All patch options, short-ray/zenith/bush cases, visibility round trips, cache/export races, numerical comparison and peak-memory report. |
| P4 | GVF, anisotropic radiation, surface temperature/state, TMRT and full chronological integration. | P2, P3 | Intermediate component and state comparisons through day/night and restart sequences; no isotropic shortcuts or timestep races. |
| P5 | Full optional-path parity: WBGT, directional/legacy wind, UHI, ERA5, WRF and input construction; documented compatibility shim installation. | P1, P4 | Real local fixture integration for every path, optional dependency isolation, service mocks labeled separately from live tests, all public functions operational. |
| P6 | Dependency-addressed caching, memory estimator, execution blocks, thread/process scheduler and failure/restart robustness. | P3, P4, P5 | Block/thread/worker invariance, corrupt/stale/concurrent caches, process-tree memory limits, cold/warm timings and complete output manifests. |
| P7 | Advanced measured optimization experiments and selected promotion to default. | P6 | Experiment log, unchanged accuracy contract, speed/memory benefit on representative scenes and no priority-workload regressions. |
| P8 | Independent validation, full benchmark matrix, packaging/CI and release documentation. | P5, P6, P7 | Passed non-skipped end-to-end tests, raw comparisons/timings, supported-platform matrix, clearly qualified CPU/GPU performance claims. |

P1 is a correctness-first milestone, not a declaration of a fully optimized or feature-complete release. Some low-risk writer/cache primitives can be implemented earlier than their final integration package. P8's scientific checks should be developed throughout, not postponed until the end.

### 12.1 Advanced optimization experiments for P7

Only pursue these after profiling the full working implementation:

**Exact building-only horizon or scan recurrence.** Test directional prefix/max-plus sweeps and reuse of an exact discrete horizon across genuinely identical azimuth sample paths. Cardinal cases may be simpler. Oblique ray rounding introduces phase-dependent sample sequences, so a standard scanline shortcut is not automatically equivalent to upstream's shifted-array traversal. Vegetation requires canopy/trunk intervals and ordering, not a single opaque horizon.

**Conservative hierarchical skipping.** A spatial hierarchy of source maxima could skip ray intervals only where a bound proves that none of the required building, vegetation or wall outputs can change. Preserve discrete sample locations and termination. Record node visits and actual speedup; construction and branch overhead may outweigh saved samples for small scenes.

**Algebraic angular moment reuse.** Precompute static masked angular sums for reflected longwave and any factorizable sky terms. Preserve dynamic sunlit/shaded classification. Treat floating-point reassociation as a numerical change requiring the frozen tests, even when the real-number algebra is identical.

**Layout and vectorization variants.** Compare pixel-major compact masks, patch/block-major storage, byte coding versus bit packing, and direct decoded reductions. Avoid per-timestep transposes of the full domain. Tune blocks against measured cache/memory behavior, not a universal hardcoded tile size.

**Exact coefficient specialization.** Specialize UTCI for uniform timestep temperature/humidity and retain variation only in wind/TMRT where applicable. Benchmark Horner versus explicit powers and check cancellation-sensitive inputs. Do not substitute a fitted polynomial or lookup interpolation.

**Alternative compiled extension.** Use Cython/C++ only for a demonstrated residual bottleneck with a stable array interface, portable build story and matching oracle tests. Do not introduce multiple numerical backends without a measurable maintenance/performance justification.

Every experiment must record its preconditions, complexity, added storage, construction cost, numerical effect and workload crossover. Reject an experiment that speeds one microbenchmark while making the complete priority workload slower.

## 13. Required reports and definition of done

Codex must produce:

```text
docs/compatibility.md
docs/model_deviations.md
docs/provenance.md
reports/source_manifest.json
reports/contract_snapshot.json
reports/reference_environment.json
reports/numerical_comparison.json
reports/performance_results.json
reports/peak_memory.json
reports/optimization_experiments.md
```

Reports must separate passed, failed, skipped, not available and not yet measured. Each numerical/performance result identifies source commit, candidate commit, fixture hash, configuration, environment, cache state, invocation and artifact paths. Failed comparisons include their worst coordinates and reproducible commands.

The release is complete only when the full default TIFF/custom-met pipeline and all seven public workflows work on CPU; no required path uses a placeholder or GPU fallback; the selected compatibility and independent checks actually execute; legacy interfaces and outputs are verified; supported optional workflows are covered; numerical budgets are met; memory is bounded; and measured performance supports the published claims.

A report containing only microbenchmarks, only mocked end-to-end tests, or only speed without output comparison is not a successful execution of this specification. An unresolved upstream failure is a visible issue, not a fabricated verification result.

## 14. Source references

All repository references below use commit `0d7fe742abeeddd890dd58fc76ed7f78bd47faec`. The reviewed regions are evidence anchors, not a statement that other regions can be skipped during implementation.

- [S1] Public exports and version: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/__init__.py`
- [S2] Public wrappers, staged execution and main function: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/solweig_gpu.py`
- [S3] CLI parser and forwarding: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/cli.py`
- [S4] Raster preparation, cache loading, chronological driver and output writers: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/utci_process.py`
- [S5] SVF shadows, patch definitions, visibility storage and legacy exports: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/shadow.py`
- [S6] Wall and aspect algorithms: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/walls_aspect.py`
- [S7] Main radiation calculation, ground-view traversal, temperature delay, longwave integration and TMRT: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/solweig.py`
- [S8] UTCI polynomial and input handling: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/calculate_utci.py`
- [S9] Tiling, metfile creation, ERA5/WRF and UHI preparation: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/preprocessor.py`
- [S10] Wind input interpretation, rotations and wake loops: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/wind_ext_coeff.py`
- [S11] Wet-bulb implementation, including array-wide convergence: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/calculate_wbgt.py`
- [S12] Unconditionally skipped integration test: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/tests/test_integration_end_to_end.py`
- [S13] Commit metadata: `https://github.com/nvnsudharsan/SOLWEIG-GPU/commit/0d7fe742abeeddd890dd58fc76ed7f78bd47faec`
- [S14] Optional input construction, identified through the public wrapper; complete implementation audit remains a P0 task: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/create_inputs.py`
- [S15] Solar helper, identified through driver imports; complete implementation audit remains a P0 task: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/sun_position.py`
- [S16] Land-cover mapping helper, identified through driver imports; complete implementation audit remains a P0 task: `https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/Tgmaps_v1.py`
- [D1] Numba performance tips, nopython loops and fast-math semantics: `https://numba.readthedocs.io/en/stable/user/performance-tips.html`
- [D2] Numba parallel ownership, reductions and diagnostics: `https://numba.readthedocs.io/en/stable/user/parallel.html`
- [D3] GDAL thread/process safety: `https://gdal.org/en/stable/user/multithreading.html`
- [D4] Numba compiled-cache behavior and limitations: `https://numba.readthedocs.io/en/stable/developer/caching.html`
