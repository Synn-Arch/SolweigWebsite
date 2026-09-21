# P6 wind memory and ownership inventory

This source-inspection inventory supports resource admission. It is not a hard
process-tree RSS bound or a measured peak. Raw RSS characterization must report
its workload, hardware, native-thread environment, dependency versions and source
hashes separately. No wind memory measurement or speedup is claimed here.

Public wind entry snapshots scoped `RuntimeOptions`. Before full raster arrays
or local roughness extraction, it guards input contents, opens building/tree
metadata, verifies dimensions and admits the requested directional workload.
After normalized heights are available, it checks content again and refines
admission for untrimmed height-dependent wake ramps before coefficient,
footprint, smoothing or direction work begins.

The estimate uses the largest expanded forward grid and largest expanded inverse
grid across requested directions. Inverse rotation applies to the expanded wake
raster with `reshape=True` before center cropping; estimating only the forward
grid misses this allocation. Shape bounds use absolute sine/cosine extents with
rounding margin, preserving cardinal shape swaps.

| Shared accounting | Reserve |
| --- | ---: |
| Normalized input/coefficient planes and preprocessing scratch | 80 float32 planes on original grid |
| Tree footprint distance/predicate/native scratch and coordinate vectors | 16 bytes per footprint cell plus vector reserve |
| Gaussian kernel support | 32 bytes per conservative largest support element |
| Native/library overhead | 256 MiB |

| Per-direction accounting | Reserve |
| --- | ---: |
| Labels, component slice objects and object bookkeeping | 512 bytes per forward-grid cell |
| Forward rotation/wake numeric scratch | 64 float32 planes on forward grid |
| Expanded inverse rotation/scratch | 32 float64 planes on inverse grid |
| Smoothing, result and writer clipping copies | 32 float32 planes on original grid |
| Morphology and Gaussian support | Separate footprint/kernel reserves |
| Wake profiles and intermediate copies | 32 bytes per conservative profile element |
| Native/write overhead | 64 MiB |

The footprint reserves the full radius `max(1, round(10 / max(pixel_size, 1e-6)))`,
even when the actual scene has no trees. Gaussian support follows the largest
40 m smoothing sigma and the original default support. Refined backward ramp
length uses maximum normalized obstacle height, a minimum one-cell depth and
maximum rotated width in the original wake formula, with numerical margin.
Forward support uses the width bound. Neither ramp nor footprint is truncated.
These allowances intentionally overestimate typical scenes; unusual resolution,
aspect ratio or height can still require explicit refusal.

Admission charges shared plus direction memory to each candidate worker. Shared
buffers are therefore conservatively charged repeatedly. Runtime worker, CPU,
native-thread and memory controls cap direction count; the legacy `max_workers`
argument adds an upper bound. Directions and logical domains are unchanged.

Directions execute the original NumPy vector/reduction and SciPy ndimage work,
with one executor thread per active direction. CPU admission reserves
`threads_per_worker` slots for each direction; configured values above one are
conservative reservations, not requests to parallelize these kernels. TIFF
reader and writer handles receive per-dataset `NUM_THREADS=1` execution options,
including compression fallback and stage validation. Parent environment and
GDAL configuration remain unchanged. The inspected macOS SciPy ndimage binary
imports no OpenMP, pthread or BLAS entrypoints; this is implementation evidence
for the tested environment, not a promise about arbitrary replacement native
libraries. Native process/thread measurements remain required to qualify CPU
and RSS behavior on release configurations.

Only the admitted number of futures are submitted. A completed future is removed
before writing, its result is released before replacement submission, and no
long-lived completed-future dictionary retains the twelve raster outputs. During
writing at most one fewer direction can compute. Completion metadata records
admitted workers, peak pending futures and the refined inventory; these are
accounting/scheduler records rather than measured RSS.

One writer stages TIFFs outside the legacy artifact directory, preserving the
original ZSTD/DEFLATE fallback and metadata. Resolved-path destination locks use
the same `.solweig-light-locks` convention as transactional thermal publication.
All staged TIFF blocks are readable before any promised file is replaced. The
completion manifest under `.solweig-light/wind-completions` outside the legacy
input/output directory publishes last with complete output content hashes and
source/policy identity. Ordinary replacement or source-mutation failures restore
the prior artifact set and manifest. Process death releases ownership locks;
any interrupted publication remains distinguishable through a missing or stale
completion manifest.

Verification uses original upstream wind packets at admitted concurrency 1/2/4,
including all twelve directions and the original invalid-block failure. Public
roughness tests consume all 24 wind bands with exact directional fields and the
unchanged UTCI gate. Focused tests inspect admission before raster loading,
refined support refusal, live future/result lifetimes, source mutation, shared
transaction ownership and ordinary publication rollback.
