# P6 core runtime characterization

Eight cases completed under the frozen `benchmarks/protocols/p6_runtime_v1.json`
workload. Raw evidence, source/environment/protocol/harness hashes, logs and
output verification are in
[`characterization/p6_runtime_v1_retry2_psutil`](characterization/p6_runtime_v1_retry2_psutil).
This is development characterization, not a release benchmark or speedup claim.

| Workload | Process elapsed (s) | Peak summed RSS (MiB) |
| --- | ---: | ---: |
| 24 h, cold, one worker | 8.879 | 385.0 |
| 24 h, warm, one worker | 8.169 | 384.8 |
| 96 h, cold, one worker | 16.408 | 375.8 |
| 96 h, warm, one worker | 16.972 | 382.9 |
| 288 h, cold, one worker | 36.130 | 386.1 |
| 288 h, warm, one worker | 36.907 | 374.4 |
| 24 h, two tiles, one worker | 14.972 | 382.4 |
| 24 h, two tiles, two workers | 8.873 | 599.1 |

Every case exited successfully below the 4 GiB configured budget. RSS sampling
summed the parent and recursive descendants every 20 ms (254–1,147 samples per
case). Summing RSS can double-count shared pages and miss peaks between samples.
Each row is one trial; elapsed time includes imports, actual JIT/cache state,
geometry, simulation and output. No uncertainty or paired speedup estimate is
supported by this design.

All outputs passed the harness's per-tile field/dimension/band checks, had
per-band Time metadata, and matched artifact hashes in the durable completion
manifest. Both independent tiles were checked in worker-count cases. Warm runs
had native cache manifests whose identities were verified by the cache reader.
Candidate source hashes remained unchanged throughout the successful run.

The scene is only 32×35 pixels with 153 patches. The 96/288-hour forcings repeat
the original day's meteorology with advanced calendar fields; they are not
long-horizon original-oracle comparisons. RSS has no visible growing trend in
these samples, but the small fixture and single trials do not establish a
universal memory scaling bound. A separate 96-hour weak-reference integration
test verifies that completed raster outputs are released rather than retained
as time history; state-generation tests verify bounded checkpoint storage.

Two setup failures remain preserved in the preceding run directories. The
initial attempt stopped before measurement because the isolated development
environment lacked psutil; psutil 7.2.2 was installed there. A retry exposed a
warm-case cleanup bug in the harness. The correction has its own harness hash;
these failed attempts are not counted as successful cases. The numerical
candidate was unchanged by those harness/environment repairs.

These measurements describe the source hashes recorded with this run, before
the final wind controls and cgroup headroom/hierarchy fixes. They do not measure
the final wheel. The core workload used an explicit 4 GiB budget on macOS, so
the later default Linux cgroup-discovery changes do not alter its admission
configuration; wind generation was outside this core workload.

P6 subsequently passed its final installed-wheel and independent review gates.
Final wind measurements are in `p6_wind_measurements.md`. P4's recorded
patch-radiation regressions remain unresolved; this table does not replace
that matrix or satisfy P8 release benchmarks.
