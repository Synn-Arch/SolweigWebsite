# Draft Cura CPU/CUDA benchmark execution plan

This is a proposal for review, not a frozen amendment or authorization to make
performance claims. It does not change `benchmark_v1.json`, `comparison_v1.json`,
or an existing harness.

## Scope and equal-work matrix

The companion JSON fixes ten 24-step TIFF workloads: the existing 32×35 small
case; repeated-block, dense-urban and vegetation-rich 256 cases; the same three
families at 1024; repeated-block and vegetation-rich at 2048; and one 3600×3600
public logical tile. Every synthetic run uses 1 m resolution, option 2 with 153
patches, overlap 20 where applicable, identical forcing and masks, and all ten
save flags. The 32×35 fixture retains its existing tile-size exception of 64.
This statement applies to the synthetic matrix only.

A separate mandatory matrix contains licensed real sparse, dense-urban and
vegetation-rich windows. These augment the synthetic families and retain their
native 2 m cell size; resampling them to 1 m would change the physical workload.
Each real window runs first-use, compiled/geometry-cold and geometry-warm with
all outputs. Admission requires source/license provenance, exact native fixture
hashes, density inventories and numerical agreement before timing.

Two further mandatory matrices cover output plans and timelines. The output
matrix runs exact public defaults, all ten flags off, and all ten flags on for
the small, repeated-block 256 and all three real windows in both cold and warm
regimes. The timeline matrix records public one-step outcomes on the small and
256 repeated-block fixtures, retains 24-step coverage for every end-to-end
fixture, and executes a separate 72-step/three-day state-kernel sequence without
resetting carried state. Every carried thermal field is checked after every
timestep and restart boundary. Kernel-only timing remains diagnostic.

Each workload runs in first-use only where declared, compiled/geometry-cold, and
geometry-warm regimes. A warm run must load a hash-validated cache produced by
the same backend from the same immutable fixture. Cache construction is recorded
separately and is never hidden inside the other backend's setup. OS page-cache
state is observed rather than described as cold.

The fixed system set is upstream CPU default; upstream CPU at 1, 4, 18 and 36
Torch threads; candidate at one thread and a 36-core budget; and upstream CUDA on
one explicitly selected RTX A6000. Upstream default is always reported. The
strongest CPU baseline is selected per workload/regime from the five
predeclared upstream CPU configurations by lowest median among numerically valid
runs. Candidate process workers remain one so the 36-core budget cannot become
an uncontrolled process/thread product.

Normal cells use five repetitions. The two 2048 cases and 3600 case use three,
declared before results because their full artifacts are expensive. Trial order
is randomized with seed 20260918, every trial uses a fresh process, and failures
remain in the schedule.

## Isolation and provenance

Create three new subdirectories inside the existing owned Cura root:
`envs/upstream-cpu`, `envs/upstream-cuda`, and `envs/candidate`. Install the
upstream commit `0d7fe742...` from a clean, separately hashed checkout into both
upstream environments. Use the recorded oracle dependency set, including
Torch 2.14.0 and GDAL 3.13.3; the CUDA environment must additionally freeze the
exact CUDA runtime/build tag. Install the candidate from an immutable wheel and
record the wheel hash. Never place `solweig_gpu` compatibility modules in either
upstream environment.

Before timing, capture source and environment hashes, `conda list --explicit` or
equivalent lock data, Python/platform/compiler details, CPU topology and affinity,
governor/frequency state, NUMA binding, filesystem/mount identity, `nvidia-smi -q`,
driver/runtime versions, GPU UUID and selected device. A trial records its exact
command, environment, effective kwargs, fixture hashes, cache hashes, timestamps,
20 ms process-tree RSS samples, output inventory, stdout/stderr and exit status.
CUDA records GPU utilization and peak device memory from a separate sampler.

Numerical verification is outside timed regions. It uses the unchanged
`comparison_v1.json` tolerances and metadata/artifact rules. Upstream CPU,
upstream CUDA and candidate outputs retain distinct evidence classes. A cell
with a failed comparison cannot enter a speedup aggregate.

## Minimal proposed runner work

Do not alter `tools/run_reference.py`. Copy it to a new draft CUDA runner and
make only these reviewable changes:

1. Add `--device {cpu,cuda}` and `--cuda-device` arguments. CPU mode retains the
   current rejection of visible CUDA and clears `CUDA_VISIBLE_DEVICES`.
2. CUDA mode requires `torch.cuda.is_available()`, exactly one visible selected
   device and the expected GPU UUID; it must fail rather than fall back to CPU.
3. Preserve the same public entrypoint, kwargs, fixture copying, output requests,
   source pin/cleanliness checks, installed-source hash checks and 20 ms RSS
   sampler. Add GPU memory/utilization sampling without changing model calls.
4. Record Torch/CUDA/cuDNN versions and device properties. Call
   `torch.cuda.synchronize()` after import/device initialization immediately
   before the timed workflow body and again after the public call and completed
   output writes. External process time remains the end-to-end measure; the
   synchronized body is an additional diagnostic, not its replacement.
5. Generalize a new top-level orchestrator to schedule the fixed systems and
   regimes. It may reuse comparison helpers, but must bind itself and all copied
   runners by hash before execution. Existing P7 tools currently force
   `CUDA_VISIBLE_DEVICES=''` and compare candidate variants, so they are not a
   valid CUDA release harness unchanged.

## Disk and cleanup sequence

Cura had 46,279,958,528 free bytes after fixture preparation. The owned task
root occupied 6,142,186,178 bytes and the prepared fixture tree occupied
442,726,991 bytes at that preflight.
Before fixture generation and before every trial, compute conservative sizes for
one immutable fixture, one scene copy, all requested TIFF/ZIP/NPZ outputs, native
and legacy caches, JIT files and evidence. Require that estimate plus a 10 GiB
reserve and never begin below 20 GiB free. Keep at most one large trial scene and
one comparison peer live. Stream hashes and compact measurement records to the
result directory; download and verify them after every completed pair. Delete
only task-created trial scene/output/cache directories listed in the cleanup
ledger after their evidence is verified locally. Retain immutable source,
environment locks and result manifests until the whole matrix is accepted and
downloaded. The existing Cura root remains retained for this review stage.

## Feasibility and unresolved gates

The remote host and driver can support upstream CUDA in principle, and the four
A6000 devices were idle during assessment. The current isolated P8 environment
has the candidate numerical stack but no Torch, pandas, xarray, netCDF4,
matplotlib, shapely or upstream checkout. A separate upstream environment is
therefore required. The recorded oracle used Torch 2.14.0; availability of a
matching Linux CUDA build compatible with driver 550.144.03 must be resolved and
locked before calling CUDA feasible in practice.

Deterministic input preparation is now recorded in
`fixture_prep/fixture_preparation_manifest.json` (SHA-256
`4d0888bcf8cce28f3bd3d46f3c2c199cb0961a7816cc7ab39228531bf454afad`).
The unchanged generators produced repeated-block 1024, 2048 and 3600 inputs;
dense-urban and vegetation-rich 1024 and 2048 inputs; and three 256×256 native
2 m windows from the pinned 94,896,401-byte Zenodo archive. The archive SHA-256
is `9429fac970a29b3bd21217cbd0876f61d3df35e923efd077b7754e9134960a34`.
All declared input-file hashes were recomputed successfully. This prepares
inputs only; none is numerically admitted and no benchmark was executed.

The unchanged P7 generator rejects 3600 for dense-urban and vegetation-rich
because 3600 is not divisible by its required 32-pixel motif. A 3584 substitute
would change the declared workload, so this remains a freeze blocker if those
families are added at public-tile size. The current draft only declares the
repeated-block 3600 case, which was generated successfully.

The conservative single-trial estimates include three uncompressed 153-patch
visibility channels, ten 24-band float32 outputs, sixteen float32 SVF fields,
three float32 input rasters, 20% headroom and 1 GiB for evidence. They are
4,687,554,150 bytes at 1024, 15,528,991,129 bytes at 2048 and 45,739,085,824
bytes at 3600. With the mandatory 10 GiB reserve, only 35,542,540,288 bytes were
usable. Therefore 1024 and 2048 pass this disk-only preflight, while 3600
all-on is not admitted on the current filesystem. The separate 12 GiB
process-tree RSS gate and numerical gates still apply to every case.

The next safe command after review is validation-only: a new orchestrator should
load the draft, resolve every fixture/environment/harness hash, calculate the
complete trial schedule and disk estimates, and emit `validated_not_executed`.
No model import or numerical workload is needed for that validation pass.
