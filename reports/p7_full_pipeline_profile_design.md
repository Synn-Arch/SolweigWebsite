# P7 full chronological candidate profile design

Status: tool authored and syntax-checked only; no profile executed during the active
timing window.

`tools/profile_p7_pipeline.py` profiles the actual one-tile numerical worker with
Python's deterministic profiler:

`python -m cProfile -o worker.pstats -m solweig_light.runtime_worker --job ... --options ...`

This places cProfile inside the process that imports `pipeline.py`, computes cold
geometry, executes every chronological timestep, evaluates comfort indices, and
writes outputs.  It does not profile only the scheduler parent and does not
reimplement or mock physics.

## Frozen workload and isolation

The input is the existing `tests/reference/small_original_cpu/scene` fixture:
32 by 35 pixels, 24 original hourly forcing rows, default option 2 with 153 sky
patches.  The tool copies it to a newly created run directory, removes copied
outputs, transactions, native cache, and legacy SVF artifacts, and then runs one
cold full-physics tile.  All ten save flags are true.  Runtime options use one
worker, the requested native thread count, `block_pixels=128`, checkpoint interval
one, a 4 GiB admission budget, cache enabled/recompute, and resume disabled.

The copied scene and outputs live under `setup/scene`; the geometry cache is
`setup/geometry_cache`; the fresh Numba cache is `jit`; and raw/derived profiler
artifacts are under `profile`.  These locations are distinct, and the original
fixture is never mutated.  A fresh run path is mandatory, and the tool rejects a
run destination nested inside either the fixture or candidate source tree.

Before execution the tool records SHA-256 inventories for the original fixture and
complete candidate source tree, plus its own hash.  At execution it records the
exact worker job/options/command, copied-input inventory, Python/platform/package
environment, native thread variables, JIT/cache policy, and artifact paths.  After
execution it verifies all ten GeoTIFFs are 32 by 35 with 24 timestamped bands and
that the three cold SVF artifacts were published.  It re-hashes the original
fixture and complete candidate source, records both final inventories and equality
results, and fails if either changed during profiling.  The final copied-scene
inventory is retained.

The worker is always launched with the absolute, non-symlink-resolved value of
`sys.executable` from the profiler tool.  Preserving that virtual-environment
launcher path keeps its site-packages active; resolving the symlink could select
the base interpreter.  The underlying real path is recorded separately for
provenance.  There is no `--python` override, so the recorded Python version and
package inventory describe the actual worker environment rather than a potentially
different parent environment.

## Artifacts and interpretation

The primary artifact is `profile/worker.pstats`.  The tool also writes a complete
cumulative sorted text table and JSON function table with primitive/call counts,
internal time, and cumulative time.  Standard output and error are captured.

cProfile adds material interpreter and call-hook overhead, so its wall time and
function times are attribution diagnostics rather than benchmark measurements.
Numba and native-library work mostly appears as time charged to Python call sites;
the profiler cannot attribute instructions within compiled kernels.  The profile
is one candidate run with uncontrolled OS page cache.  It cannot support a speedup,
regression, scalability, or peak-memory claim.  Existing frozen paired benchmark
evidence remains authoritative for those questions.

## Deferred commands and validation

Dry plan, which performs only standard-library validation and hashing:

`.venv-light/bin/python tools/profile_p7_pipeline.py --run reports/characterization/p7_full_pipeline_profile`

Execute after the timing window closes:

`.venv-light/bin/python tools/profile_p7_pipeline.py --run reports/characterization/p7_full_pipeline_profile --threads 1 --execute`

Before accepting the result, inspect `outcome.json` for
`profile_completed_not_a_benchmark`, confirm output validation passed, verify the
worker stderr is empty or explained, confirm the provenance source inventory still
matches the reviewed candidate, and open the cumulative table alongside raw
`worker.pstats`.  A failed worker, missing raw profile, incomplete 24-band output,
or missing cold SVF publication invalidates the capture; it must remain recorded as
a failed attempt rather than being summarized as profiling evidence.
