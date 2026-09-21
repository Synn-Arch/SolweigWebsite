# P6 implementation contracts

P5 was accepted before implementation started. Architecture reviewed against
implementation-plan §6–7 and the current pipeline. Root owns `api.py`,
`pipeline.py`, `models.py`, `__init__.py`, radiation dispatch and integration tests.
Workers must not edit those shared files. No scientific-policy change is needed.

## Configuration and scheduling

Keep all seven legacy signatures unchanged. Provide an immutable `RuntimeOptions`
and scoped `runtime_options(...)` context, backed by ContextVar rather than a
mutable process-wide configuration. Snapshot once at public entry and pass the
snapshot to internal work. Defaults preserve logical domains and chronological
ordering. Additional controls cover cache root/trust, checkpoint/restart,
execution-block pixels, total CPU budget, worker count and memory budget.

Workers receive explicit serializable job/options records. If using subprocess
workers, set native thread environment in each child's `Popen(env=...)` before
package import; never temporarily mutate the parent's environment or thread
settings while concurrent callers may run. Use independent GDAL handles and
single-owner destinations. Admit only the number of workers whose combined
estimated live memory and native threads fit the configured budgets.

Radiation already accepts `block_pixels`; expose this through internal dispatch
without changing logical tile size, patch count, forcing or ray support. Full
neighbor-readable snapshots remain immutable across blocks. If a tile cannot
fit the supported plan, fail before computation with an actionable estimate;
never shrink logical tiles as a memory workaround. Disk-backed state/stage-wise
execution requires an explicit live-array inventory and verification.

## Worker A: dependency-addressed geometry cache

Own new `cache/` modules and focused tests. Reuse
`geometry/visibility_native.py` storage/ownership; do not invent another codec.
Proposed root-facing contract: content identity helpers plus a cache store that
accepts a JSON-safe dependency manifest and a producer returning the existing
SVF field mapping. It returns an owned mapping/context that closes all native
mappings explicitly. Publication is atomic, with payload hashes and a final
manifest; identical concurrent producers coordinate via locks. Crashes release
locks and leave only uncommitted staging data. Never delete a payload that a
reader may still have mapped.

Dependency identity includes actual input content, full raster metadata/NoData,
logical domain, geometry-construction policy, dtype, actual patch arrays/order,
canopy/trunk/transmission policy and implementation/model version. File size and
mtime alone are insufficient. Include upstream-style array normalization rules
and distinguish standalone versus driver construction if their inputs differ.

Default legacy handling validates schemas and recomputes when identity is
missing. An explicit trusted-legacy mode records the absent provenance trusted,
while still rejecting shape/patch-count/metadata errors. Never modify original
legacy inputs in place. Native cache lives separately from promised legacy
exports. A native hit must not be substituted for `available_on_disk`: that
existing flag controls conditional `save_svf` artifacts and must retain its
separate meaning.

Gate: content change with preserved size/mtime; every declared identity field;
corrupt/missing payload or manifest; interrupted producer; concurrent identical
requests; correct mapped lifetimes; original-reference field comparisons.

## Worker B: checkpoint and output transaction

Own new persistence modules and `io/rasters.py`, plus focused tests. Keep the
existing writer interface usable while adding explicit transaction hooks.
Persist every SimulationState dataclass field including Twater, with exact
Python/NumPy scalar/array dtype and shape. Do not use pickle. Identity includes
scene, forcing content, date/timezone, relevant runtime/model policy and output
schema; record the next chronological timestep.

Each run/tile owns unique staged GeoTIFFs under a destination lock. Flush all
requested bands first, then atomically publish checkpoint state and committed
cursor. On recovery validate staged artifacts, replay uncommitted work, and
resume only a fully committed boundary. TIFF existence is not completion.
Publish complete output files and a completion manifest last, accounting for
conditional SVF output. Failures must remain failures; interrupted partial
results cannot be mistaken for complete outputs. Preserve final legacy names,
band schemas, masks and timestamps.

Gate: scalar/array/list state round trips, stale identity, corrupt data,
interruption at each publication boundary, competing output owners, resumed
versus uninterrupted real chronological TIFF comparisons.

## Worker C: resource planner and bounded scheduler

Own new runtime modules and tests; agree RuntimeOptions schema with root before
editing. Inventory packed/raw fallback visibility, live raster/state arrays,
wind coefficients, decoded blocks, JIT/native overhead and writer buffers.
Resource admission must bound aggregate workers and native threads. Include
process-tree measurements, not just Python/NumPy counters.

Gate: deterministic admission/errors, independent job ownership, failure
propagation and cleanup, worker/thread/block invariance. Root adds genuine
multi-tile integration and fixed-scene short/long timeline RSS measurements.

## Integration and acceptance

Merge one coherent behavior at a time, then run its targeted original-reference
checks. Final gate includes cold/warm caches, stale/corrupt/interrupted/concurrent
cases, restart, block/thread/worker settings, bounded output history and measured
process-tree memory. Preserve the frozen P4 benchmark matrix and disclose its
patch-radiation regressions until measured improvements are established.

## Follow-up coverage identified during integration

Core resource characterization freezes source while it runs. Optional wind
preprocessing still needs a resource pass: the current direction executor uses
host CPU count and retains every completed future until the stage ends. Bound
admission, live direction results and worker count without changing rotations,
wake formulas, selected roughness time or output directions. Reuse the original
wind/public roughness references to verify this execution-only change.

InputGuard is content revalidation, not a filesystem snapshot. Callers must keep
sources stable during an invocation. The core captures/checks content around
reads, geometry production and publication; explicit trusted legacy inputs also
participate in those checks and checkpoint identity.

Wind follow-up packet (read-only design accepted for implementation after the
core measurement freeze): own `wind.py`, new `wind_resources.py`, and focused
resource/optional wind tests. Snapshot runtime options; guard source content;
admit metadata-only before raster reads, then refine after heights are known.
`max_workers` remains an upper bound. Charge shared plus per-direction buffers,
cap active tasks by CPU/memory/worker limits and bound submitted futures/results.
One writer stages direction files under the same resolved-destination locks.
Preserve all numeric functions, compression fallback and sorted return values.

The inventory must cover BOTH the expanded forward rotation and expanded
inverse rotation before cropping, labeling/object scratch, result clipping,
footprints/Gaussian support, and height-dependent wake-ramp lengths. Never
truncate those supports to fit the budget. Validate original twelve-direction
cases at actual admitted concurrency 1/2/4 with explicit runtime budgets, plus
public roughness consumption and resource/recovery/ownership cases.
