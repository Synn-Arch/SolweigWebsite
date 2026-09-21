# Chronological driver audit

Source inspected in full: `solweig_gpu/utci_process.py` (932 lines), commit
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec`. Findings below are static unless
explicitly connected to the recorded CPU run.

## Dataflow and ownership

`run_utci_tiles` discovers matched files and invokes `compute_utci` sequentially.
The driver reads DSM, trees, DEM, walls and aspects into float32 Torch tensors;
all met columns are tensors from the float64 `np.loadtxt` matrix. Its module-wide
device chooses CUDA if visible, otherwise CPU. Torch scalar/array promotion must
be characterized, not inferred from source dtype annotations.

`compute_utci` builds tile-centre geographic position and local UTC offset from
the selected date. Positive median elevation is replaced with 3 m. The solar
helper receives the entire met matrix. Trees are clipped at zero; absolute
canopy/trunk arrays for radiation add DSM while the maximum-elevation bound
uses DEM. Building recipients use `(DSM - DEM) < 2` after an in-place remapping.
Land-cover normalization operates on a CPU NumPy view of the original tensor;
the shared-storage effect differs from a CUDA-to-CPU copy.

Directional wind requires all 12 bins if supplied as a dictionary; the nearest
bin uses floor after adding 15 degrees and wraps at 360. Invalid/missing direction
uses an all-ones coefficient. The low-level function accepts a single raster,
but the public wrapper's directional-only filename mapper does not discover it.
Comfort wind is clamped to 0.15 m/s. UHI affects comfort temperature and wet bulb,
not the base temperature sent to radiation.

State carried in chronological order is `CI`, `firstdaytime`, `timeadd`,
`timestepdec`, `Tgmap1`, `Tgmap1E/S/W/N`, and `TgOut1`. `Twater` updates when
land-cover handling requires it. The radiation function returns both diagnostic
components and replacement state. Restart must preserve all of these and the
timeline position. The midnight CI look-ahead branch checks the length of the
tuple returned by `np.where`; its effective reachability needs a targeted test.

## Geometry and exports

`_svf_cache_exists` checks only three file paths. Cached directional fields are
read from a 15-member ZIP, visibility from three float32 NPZ members, and total
SVF from a TIFF. The driver hardcodes patch option 2 and transmission 0.03;
direct-shadow transmission changes with the strict leaf-season boundaries.
It allocates dense diffuse visibility from building and vegetation channels.

The loop retains seven raster histories unconditionally (UTCI, TMRT, four fluxes,
shadow), plus requested WBGT/Ta/wind. It converts them into another stacked array
before writing. This is the principal `O(time × pixels)` output lifetime to remove
in P1 without removing any physical intermediate.

Every runtime TIFF uses float32, input dimensions/transform/projection, and
per-band `Time` derived from selected date plus the met hour/minute. No explicit
NoData tag is set. UTCI alone is assigned NaN outside the building recipient mask.
Other output masks must not be inferred from UTCI. `save_svf=True` writes the
runtime duplicate only on a cache miss. The staged public workflow precomputes
SVF, so the successful initial CPU run explicitly logged that the duplicate
was skipped. ZIP compression bytes are not a semantic contract.

## Verification boundary

`reports/runs/dependencies_cpu` executed the full 24-step synthetic case without
source patches. That establishes executability for one configuration only.
Component/state interception, land-cover aliasing tests, missing-value inputs,
directional/legacy wind tests, cache miss variants, and restart comparisons remain
pending. No candidate parity or scientific validity is established by this run.
