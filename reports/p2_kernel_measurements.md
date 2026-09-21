# P2 kernel measurements

These are kernel-only measurements on the local Apple M1 Pro, not end-to-end speedups. All 28 configurations passed equal-input hashes, exact wall/aspect output checks and the unchanged 0.02 C UTCI gate. Five warm trials per configuration and every first call are retained in `reports/characterization/p2_kernels_v1/`.

The table uses the predeclared four-thread upstream configuration and one-thread serial/four-thread parallel candidate paths. All other predeclared thread configurations remain in `reports/p2_kernel_comparison.json`; this table does not establish an optimal CPU budget.

| Kernel | Upstream 4-thread warm median (ms) | Candidate serial warm median (ms) | Candidate 4-thread warm median (ms) | Candidate serial first call (ms) |
|---|---:|---:|---:|---:|
| walls | 606.964 | 0.534 | 0.311 | 598.731 |
| aspect | 224.284 | 4.911 | 4.543 | 552.109 |
| utci | 83.897 | 46.773 | 13.279 | 2262.468 |
| utci_uniform | 84.008 | 35.420 | 9.744 | 2276.948 |

First calls include relevant compilation and filter construction, but exclude process startup, imports and fixture construction. Candidate UTCI first calls are slower than upstream; warm timings cannot be substituted for first-use application costs. Wall/aspect outputs were exact. Benchmark maximum UTCI errors were 0.0040283203125 C (variable forcing) and 0.00025513768196105957 C (uniform forcing). The broader original characterization reached 0.017578125 C, relatively close to the frozen 0.02 C limit.

Raw process-lifetime peak RSS includes imports, fixtures, JIT and output serialization. It is not a kernel allocation measurement or a process-tree/application memory claim. No P8 memory or end-to-end performance gate is satisfied by these results. Trial min/max spread is in the summary; five within-process warm samples do not characterize cross-process or cross-machine uncertainty.

Source snapshots, the frozen protocol, executable harnesses, unfiltered logs, output arrays, hashes and the zero-failure execution manifest are preserved beside the raw trials. The final installed-wheel source differs from the benchmark snapshot only in subsequently corrected descriptive UTCI comments; exact tested-source hashes distinguish the two states.

A separate warm allocation audit (`reports/p2_allocation_audit.json`) records
output bytes, Python/NumPy-visible traced peaks and native NRT allocation counts
for all four kernels in serial and parallel one/four/ten-thread configurations.
All sixteen calls balance native allocation/free and meminfo counts after output
release. Tracemalloc does not cover every native allocation, and NRT counts do
not provide byte peaks; this is allocation accounting, not a complete peak-memory
or cross-runtime reduction claim.
