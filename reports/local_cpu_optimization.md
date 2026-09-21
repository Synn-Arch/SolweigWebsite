# Local CPU optimization evidence

**Status: integrated and verified after independent promotion review.**
Independent review approved all frozen promotion gates, and the exact
`combined_v1` source and permanent regressions are integrated. Final
installed-wheel verification passed 97 tests without failures or skips; all
49 source, wheel and installed files matched before and after the checks.
These results
compare the reviewed candidate with the local candidate baseline at checkpoint
`8ca23d444a3b05bdeb76655329c0a09c3dc4d6e8`. They do not measure original
SOLWEIG-GPU and do not close P7 or P8.

## Scope and method

The candidate source inventory is
`dc6e151cfcd6c8bd476921e850666d22f6dfcd2237eb8b0afa649736002af0c6`.
Measurements ran on one 10-core Apple M1 Pro host with 16 GiB RAM, ARM64 and
macOS 26.6.2. Every cell used one worker, 153 sky patches, all 24
chronological timesteps, every required output and checkpoint interval one.
The baseline used 128 block pixels. The selected candidate setting used four
native threads and 1,024 block pixels; explicit one-thread and unchanged
128-block controls were also measured.

Each of the ten cells contains five sequential baseline/candidate pairs with
alternating order. `geometry warm` means setup was completed outside the timed
distribution and both compatible JIT and geometry caches were retained for a
timed `run_utci_tiles` call. `geometry cold` removed prepared geometry while
retaining the compatible JIT cache and timed full `thermal_comfort`. `first
use` timed full `thermal_comfort` with empty private geometry and JIT caches.
The OS page cache was uncontrolled, so none of these is a disk-cold result.

In the table, **B median** and **C median** are separate medians of the five
baseline and candidate elapsed-time samples. **Paired median B/C** is the
median of the five within-pair ratios, so it need not equal B median divided by
C median. Values above one favor the candidate. The confidence interval is
for the paired geometric mean B/C: all 3,125 ordered five-pair log-ratio
resamples were enumerated, then the 2.5% and 97.5% linear quantiles were taken.
Only the primary cell's lower endpoint is a predeclared promotion gate; the
other intervals are descriptive. Five pairs do not model thermal drift or
cross-host variability.

| Cell and role | Fixture | Regime | Threads | Blocks B→C | B median (s) | C median (s) | Paired median B/C | GM 95% interval | Median sampled peak RSS B/C (bytes) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 primary | repeated_block_256 | geometry warm | 4 | 128→1024 | 20.964 | 12.887 | 1.627 | 1.593–1.649 | 488587264/482213888 |
| 1 single-thread | repeated_block_256 | geometry warm | 1 | 128→1024 | 20.783 | 18.620 | 1.117 | 1.105–1.143 | 506740736/526696448 |
| 2 small | small_original_cpu | geometry warm | 4 | 128→1024 | 5.102 | 4.087 | 1.247 | 1.202–1.281 | 458964992/436731904 |
| 3 dense | dense_urban_256 | geometry warm | 4 | 128→1024 | 20.955 | 13.070 | 1.603 | 1.576–1.640 | 516341760/523649024 |
| 4 vegetation | vegetation_rich_256 | geometry warm | 4 | 128→1024 | 21.427 | 13.291 | 1.615 | 1.529–1.655 | 515784704/480460800 |
| 5 geometry-cold | repeated_block_256 | geometry cold | 4 | 128→1024 | 28.714 | 20.737 | 1.391 | 1.372–1.412 | 503021568/500088832 |
| 6 one-thread startup | small_original_cpu | first use | 1 | 128→1024 | 12.336 | 12.430 | 0.998 | 0.963–1.016 | 541704192/478248960 |
| 7 selected startup | repeated_block_256 | first use | 4 | 128→1024 | 36.766 | 30.341 | 1.214 | 1.199–1.220 | 563838976/561840128 |
| 8 default warm | repeated_block_256 | geometry warm | 1 | 128→128 | 20.950 | 19.415 | 1.067 | 1.059–1.106 | 488931328/465453056 |
| 9 default startup | small_original_cpu | first use | 1 | 128→128 | 12.723 | 12.405 | 1.032 | 0.979–1.037 | 523468800/541245440 |

All 50 pairs completed and were exact between the baseline and candidate for
the ten 24-band TIFFs, lossless geometry exports and final carried state.
Independent review approved every frozen benefit, regression, startup and RSS
gate. No startup tradeoff rule was triggered. The one-thread 1,024-block
first-use cell was essentially neutral: its paired median B/C was 0.998 and
the separately calculated median candidate/baseline ratio was 1.002, within
the frozen first-use limit.

RSS is a sampled diagnostic, not an allocation bound. The monitor summed the
process tree every configured 20 ms, which can double-count shared pages and
miss shorter peaks. Monitor work also made observed gaps longer than the
configured sleep in some trials. Every trial remained below the frozen 12 GiB
cap and every cell passed its relative median-RSS guard. The maximum observed
summed process-tree RSS across all 100 trials was 604553216 bytes.

## Correctness and feasibility evidence

The exact final candidate also passed correctness-only admissions on the
1,024-square dense and vegetation fixtures. Candidate diagnostic elapsed time
and sampled summed process-tree peak RSS were 400.827 s and 1629618176 bytes
(1.518 GiB) for dense1024, and 341.365 s and 1594834944 bytes (1.485 GiB) for
vegetation1024. Each was one run,
so these values demonstrate local feasibility only; they are not paired
speedups or performance distributions.

For both large cases, the candidate matched the E0 baseline exactly for final
state and artifacts and passed the original-reference checks for all 240
output bands and 18 geometry rasters. Separate small-case traces at one thread
with block 128 and at four and ten threads with block 1,024 each matched E0 for
all 72 captured events. No every-timestep trace was captured for the large
1,024 cases.

The four named scientific exceptions under policy
`inherited-svf-umep-compatibility-v1` remain failed with their approved
inherited-limitation disposition: two open-scene SVF unity checks and two UMEP
ground-view land-cover variants. Their tests, tolerances and model behavior
remain unchanged. This disposition does not approve other failures or imply
release completion.

## Claim boundaries and remaining work

The paired evidence is a same-host comparison of checkpoint `8ca23d4` with the
exact `combined_v1` candidate. It supports no original-upstream, tuned-upstream,
GPU, Linux or cross-machine speedup claim. Compatible-JIT-warm trials can still
compile deliberately uncached kernels. Only the last complete output histories
were retained for each side; every trial retains exact comparison records,
state/geometry/output hashes and raw measurements.

P7/P8 still require the full original-upstream and strongest tuned-upstream CPU
matrix, large 2048 and default-3600 workloads or explicit memory-limit outcomes,
multi-day sequences, final-source Linux qualification and hosted CI. The local
source integration and final installed verification do not close those gates.

The frozen definitions and raw summary are
[`promotion_method.md`](characterization/local_cpu_optimization_v1/promotion_method.md)
and
[`promotion_statistics_v1.json`](characterization/local_cpu_optimization_v1/promotion_statistics_v1.json).
The independent approval is recorded in
[`final_promotion_review.json`](characterization/local_cpu_optimization_v1/reviews/final_promotion_review.json).
The final installed-wheel check, exact invocation, source identities and JUnit
are linked in
[`postcopy_report.json`](characterization/local_cpu_optimization_v1/integration/postcopy_v1/postcopy_report.json).
