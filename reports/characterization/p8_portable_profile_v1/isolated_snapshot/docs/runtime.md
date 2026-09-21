# Runtime controls and recovery

The seven compatibility signatures are unchanged. CPU execution controls are
scoped separately:

```python
from solweig_light import RuntimeOptions, runtime_options, run_utci_tiles

with runtime_options(RuntimeOptions(
    cpu_budget=4,
    workers=2,
    threads_per_worker=2,
    memory_budget_bytes=4 * 1024**3,
    block_pixels=128,
)):
    run_utci_tiles(base_path, preprocess_dir, "2020-07-18")
```

The options object is immutable and the context is local to the calling context.
Public tile execution starts independent processes with native thread settings
in place before numerical imports, including the one-worker case. Timesteps
within each tile remain chronological. Block size changes radiation working
buffers, never logical tile geometry, overlap, forcing or patch count.

Memory admission uses a conservative live-array inventory and the available
CPU/memory limits. It is an estimate, not a hard operating-system RSS limit;
measured evidence belongs in the P6 resource report. A job exceeding admission
fails before its numerical tile/geometry computation. No automatic reduction
of logical tile dimensions is used. See `p6_memory_inventory.md` for the
inventory and measurement limitations. Wider preprocessing/acquisition-stage
resource coverage is documented separately in `wind_memory_inventory.md`.

Native geometry is content-addressed and checksummed. Default legacy-cache
handling recomputes independently when provenance is absent. Standalone SVF
export conflicts require explicit `overwrite=True`; thermal execution can use
validated native geometry while preserving unverified legacy files. See
`cache_policy.md`. `legacy_cache_policy="trust"` explicitly trusts missing
legacy provenance while still checking schemas, dimensions and spatial
metadata; the trusted input hashes are included in checkpoint identity.

Source files must remain stable for a workflow invocation. Content checks
around reading, computation and publication reject detected changes; these
checks do not provide atomic filesystem snapshots. Paths, sizes and modification
times alone are never accepted as scientific content identity.

Each tile stages output bands and serializes its complete carried state after
`checkpoint_interval` steps (default 1). Only a flushed, committed boundary can
be resumed. State includes all delay/temperature maps, scalar values and water
temperature. Uncommitted bands are replayed, and old state generations are
removed only after a new checkpoint is durable. Requested TIFF history stays
on disk; the engine does not retain its completed raster outputs.

After an interrupted call, repeat it with identical inputs, date and output
options under `RuntimeOptions(resume=True)`. Operational block, CPU and memory
settings may change. A mismatched or corrupt checkpoint is rejected. A completed
run with `resume=True` validates its published files; a normal fresh run can
replace a previously validated complete run. Incomplete work is never silently
discarded by a fresh invocation.

Transaction data and completion manifests live under
`<base_path>/.solweig-light/transactions`. Only the completion manifest certifies
all requested files: individual renames are recoverable but cannot be atomic
as a set. Per-destination locks exclude competing writers, including standalone
geometry export and pipeline publication. The final TIFF names, masks,
timestamps and metadata retain the compatibility contract.

P6 passed its installed-wheel and robustness gate. The initial
process-tree measurements are recorded in
[`reports/p6_runtime_measurements.md`](../reports/p6_runtime_measurements.md);
they cover the frozen small-scene workloads and do not establish a universal
memory bound. Release qualification remains pending.
