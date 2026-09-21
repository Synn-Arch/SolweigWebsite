# Optimization evidence index — release incomplete

The detailed P7 experiment register is [docs/p7_experiments.md](../docs/p7_experiments.md).
It records profiling attribution, native visibility decoding, classification
preparation, startup/memory tradeoffs, and the expanded-fixture failures that
required separate compatibility repairs. The two-file persistent-JIT optimization is now integrated after independent review and installed-wheel validation; this does not complete P7.

The accepted P6 snapshot, angular-only repair snapshot and final three-file
repair-only snapshot remain distinct under `characterization`. Both P7
experiments are excluded from the repair-only baseline. Failed comparisons
remain recorded; neither tolerances nor fixtures were removed.

The repaired candidate passed the unchanged 29-configuration component matrix
at `characterization/p7_repaired_v2_component_matrix`. This is component-only
evidence. Full-pipeline paired evidence completed under
`characterization/p7_pairs_v4`, with five pairs for each one/four-thread budget
and cold/geometry-warm regime on seven frozen scenes: all 140 pairs passed.
`p7_completed_matrix_review.json` records measured ratios, memory ranges and
the small four-thread geometry-warm median regression. Partial v3 trials were
interrupted for an evidence-validation defect and are not claim-eligible.

See [performance_results.json](performance_results.json) and
[peak_memory.json](peak_memory.json) for source-bound measurement references,
historical limitations and outstanding release gates. Their existence does
not imply P7 or P8 completion. Rejected/deferred advanced strategies and the
eventual promotion decision must be recorded after the full evidence review;
unimplemented strategies must not be described as measured experiments.

## Coverage against plan §12.1

| Strategy | Actual evidence/state | Remaining decision or evidence |
|---|---|---|
| Exact building-only horizon/scan recurrence | Reference-bound single-translation experiment rejected: rounded oblique paths change support and masks | More complex phase-indexed schemes remain untested; sampled cardinal mask agreement does not prove arbitrary-threshold horizon equality |
| Conservative hierarchical skipping | Building-only four-sample rectangle-max bound passes 15 reference-bound cases; Python quadtree rejected for overhead | Packed compiled structure is untested; vegetation/wall outputs need independent simultaneous bounds |
| Angular moment reuse | Isolated six-moment tile bundle passes correctness; 40 valid full-pipeline pairs reject promotion on elapsed time | All eight median candidate/baseline ratios exceed1; retain existing ordered implementation |
| Layout/decoding variants | P7 native bounded decoding, including raw/categorical channels; current compact/dense component comparisons passed | All 140 pairs passed; ranges and the small four-thread warm regression are retained in p7_pair_measurements.md; no memory-reduction claim |
| Uniform-forcing UTCI specialization | Implemented and measured in P2; see `p2_kernel_measurements.md` and source-bound raw trials | Historical specialization remains; full-coefficient Horner experiment now rejected at unchanged0.02°C gate; no current full-workload benefit claim |
| Alternative Cython/C++ extension | Not implemented or benchmarked | Deferred by p7_angular_moment_design.json: retained profiles include compilation and predate cache promotion, with no demonstrated Numba limitation justifying another backend |

The P7 initial attribution profile covers warmed patch-radiation components.
Full chronological cold profiles subsequently executed on Cura (small scene)
and locally (256-square scene), with source/fixture guards and output schema
validation. Evidence is in `cura_p8_20260919T1215Z_a7c3/profile_small24_t1`
and `characterization/p7_full_profile_repeated256_v2`. The first local attempt
is retained with its profiler-only dimension-validation failure. Warmed diagnostics and the distinct-cache experiment subsequently completed; their results and limitations are recorded below. Final lead review remains necessary; no unexecuted strategy is being
labeled a failed measurement or a completed experiment.


## Persistent-JIT integration

`characterization/p7_persistent_jit_promotion/report.json` records the accepted two-file cache change, 831 passing targeted/subsystem tests, permanent distinct-dispatcher cache regressions, and installed-wheel empty/populated-cache exact output comparisons. The 40-pair experiment in `characterization/p7_persistent_jit_serial_variant/pairs_v1` compares current optimized baseline to the distinct-cache snapshot; all pairs pass, but first-use ratios remain mixed. Compatible-cache warm medians improve 1.52–1.71× (small) and 1.12–1.16× (256-square). These are not original-upstream speedups. Frozen snapshots retain their original identities after main-source promotion.

Cura dense1024 original-CPU admission remains failed under the unchanged TMRT gate. The ASVF diagnosis identifies original platform-dependent unary math (M1 SLEEF versus Cura MKL VML), not a cache regression. Standalone Torch-free MKL reproduces all 189,220 tested ASVF patterns on Cura; downstream strict-classifier boundary parity and a qualified packaging design remain open. No ASVF or classifier repair has entered production.
