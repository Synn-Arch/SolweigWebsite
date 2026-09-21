# Performance evidence status

No candidate performance claim exists. P0 has measured one original-upstream
workload on an Apple M1 Pro (10 logical CPUs, 16 GiB RAM), macOS ARM64.
The fixture is 32×35 pixels, 153 sky patches, buildings and sparse vegetation,
24 chronological timesteps, all output flags requested. Inputs, model options
and artifacts are identical across the native-thread configurations.

`reports/upstream_cpu_tuning.json` records the predeclared protocol, randomized
trial order, exact invocations, all trial measurements and summaries. Raw logs,
installed environments and copies of the harness are retained under
`reports/characterization/tuning_small_frozen/`. Each of the fifteen trials
passed exact comparison of nineteen standalone TIFFs against the initial run.
ZIP/NPZ equality and intermediate-state equality were not checked in these trials.

| Native configuration | Trials | Median elapsed seconds | Observed min–max seconds | Sample standard deviation |
|---|---:|---:|---:|---:|
| Upstream default | 5 | 9.006 | 8.110–9.413 | 0.574 |
| One thread | 5 | 9.252 | 8.424–10.408 | 0.877 |
| Four threads | 5 | 8.739 | 8.138–9.104 | 0.356 |

Four threads has the lowest observed median on this tiny case. The distributions
overlap; these measurements do not establish a reliable tuning benefit or an
optimal budget for larger workloads. They are baseline characterization on a
development host, not a dedicated release performance host.

Elapsed time includes child imports, source checks and the full public workflow,
including preprocessing, walls, SVF, simulation and output I/O. Fixture copying
is excluded. Every trial starts a fresh process with no geometry cache; the OS
file cache is uncontrolled. Upstream helper compilation during imports is
included. No candidate first-use/JIT behavior has been measured. Native budgets
set Torch/OMP/MKL/OpenBLAS/NumExpr; the upstream process-worker policy is unchanged.

Memory is sampled at 20 ms as summed RSS across the child process tree. This can
count shared resident pages more than once and can miss peaks between samples.
Raw samples' maxima are reported per trial, not described as exact peak physical
memory. Candidate memory reduction is unmeasured.

The prior batch is explicitly invalidated in
`reports/upstream_cpu_tuning_invalidated.json`: a transient harness edit added
source-copy I/O to two trials. It was retained rather than filtered into the
valid batch; the full experiment was rerun with a runner-hash check before and
after each trial. No claim uses those invalidated timings.

Remaining work includes geometry-warm and kernel-only baselines, larger sizes
and densities, longer timelines, multiple output plans, process-tree memory
planning, stage profiles, matched candidate trials, and uncertainty estimates
appropriate for release claims. CUDA is unavailable, not passed. The proposed
2× cold, 3× warm and 4× memory targets remain provisional.
