# P7 paired full-pipeline measurements — incomplete matrix

These comparisons use **P6 plus angular and SVF-weight compatibility repairs**
as the baseline. They do not compare against original upstream or unchanged
accepted P6. The optimized candidate preserves the same workload. P7 promotion
remains pending the entire seven-scene matrix and review of regressions.

## Completed small scene

All 20 paired comparisons passed numerical, artifact and cache gates. Final
source, protocol, evidence and hardware guards passed. Each row has five pairs
on the frozen Apple M1 Pro host. Ratios are baseline elapsed time divided by
candidate elapsed time; values above one favor the candidate.

| Regime | Native threads | Median paired ratio | Observed ratio range |
|---|---:|---:|---:|
| Geometry cold | 1 | 1.075 | 1.023–1.205 |
| Geometry cold | 4 | 1.065 | 1.014–1.076 |
| Geometry warm | 1 | 1.098 | 1.038–1.153 |
| Geometry warm | 4 | 0.987 | 0.913–1.079 |

The four-thread geometry-warm result is mixed and does not establish an
improvement. Its possible regression remains part of the promotion review.
No trial has been excluded based on its timing. All regimes use fresh processes
and JIT caches and include imports, compilation and I/O; geometry-warm does not
mean JIT-warm. Summed process-tree RSS ranges across measured calls were
341,327,872–457,113,600 bytes for the baseline and
356,220,928–472,252,416 bytes for the candidate. Sampling can miss short peaks
and double-count shared pages.

Raw paired times, differences, dispersion, sampled memory traces, environment,
invocations and frozen hashes are under
[`characterization/p7_pairs_v4/small`](characterization/p7_pairs_v4/small).
These small-scene results do not include the other scenes.

## Completed repeated-block 256 scene

All 20 pairs passed, with frozen inputs unchanged and final global evidence
eligibility true. Each group contains five valid pairs and zero failed pairs.
The baseline, host, full physical workload, fresh-process/JIT policy and ratio
definition are as above.

| Regime | Native threads | Median paired ratio | Observed ratio range |
|---|---:|---:|---:|
| cold | 1 | 3.547 | 3.439–3.556 |
| cold | 4 | 3.497 | 3.402–3.503 |
| geometry_warm | 1 | 4.399 | 4.258–4.476 |
| geometry_warm | 4 | 4.337 | 4.272–4.361 |

Raw trials, memory measurements and final guards are in
[`characterization/p7_pairs_v4/repeated_block_256`](characterization/p7_pairs_v4/repeated_block_256).
These results establish a measured benefit for this scene against repaired P6;
they are not upstream speedup claims or a completed P7 promotion decision.
The remaining scene results are recorded below as each completes.

## Completed dense-urban 256 scene

All 20 numerical, artifact and cache comparisons passed. Final frozen-input
and global evidence guards passed; each group has five valid pairs and zero
failures. The baseline is repaired P6, with the same host, workload and
fresh-process/JIT policy described above.

| Regime | Native threads | Median paired ratio | Observed ratio range |
|---|---:|---:|---:|
| cold | 1 | 3.232 | 3.152–3.252 |
| cold | 4 | 3.190 | 3.153–3.275 |
| geometry_warm | 1 | 4.448 | 4.401–4.460 |
| geometry_warm | 4 | 4.402 | 4.306–4.446 |

Raw times, dispersion, sampled process-tree memory and final guards are in
[`characterization/p7_pairs_v4/dense_urban_256`](characterization/p7_pairs_v4/dense_urban_256).
The observed benefit applies to this scene against repaired P6; it is not an
upstream speedup claim. The subsequent scene results follow below; P7 promotion remains pending.

## Completed vegetation-rich 256 scene

All 20 numerical, artifact and cache comparisons passed. Final frozen-input
and global evidence guards passed; each group has five valid pairs and zero
failures. The baseline, host, workload and fresh-process/JIT policy are as above.

| Regime | Native threads | Median paired ratio | Observed ratio range |
|---|---:|---:|---:|
| cold | 1 | 3.288 | 3.199–3.374 |
| cold | 4 | 3.296 | 3.238–3.349 |
| geometry_warm | 1 | 4.429 | 4.293–4.441 |
| geometry_warm | 4 | 4.209 | 4.127–4.305 |

Raw times, dispersion, sampled process-tree memory and final guards are in
[`characterization/p7_pairs_v4/vegetation_rich_256`](characterization/p7_pairs_v4/vegetation_rich_256).
The observed benefit applies to this scene against repaired P6; it is not an
upstream speedup claim. All three real-data windows remain; the orchestrator
has started real_dense_urban. P7 promotion remains pending.

## Real dense-urban scene (v4)

All 20 pairs passed artifact/numerical and cache comparisons; final frozen-input and global evidence guards passed. Each group has five valid pairs and zero failures. Ratios compare repaired P6 to candidate on the frozen M1 workload, not original upstream. Geometry-warm runs still include fresh-process JIT.

| Regime | Threads | Median ratio | Observed range | Paired seconds standard error |
|---|---:|---:|---:|---:|
| Cold | 1 | 2.904354 | 2.790303–2.934001 | 0.675584 |
| Cold | 4 | 2.885401 | 2.813714–2.897235 | 0.569997 |
| Geometry warm | 1 | 4.410939 | 4.062422–4.551977 | 1.052889 |
| Geometry warm | 4 | 4.338172 | 4.048912–4.425635 | 0.886509 |

Raw evidence: `reports/characterization/p7_pairs_v4/real_dense_urban/{summary,pairs,trials}.json`. Two scenes and full P7 promotion review remain outstanding.

## real_vegetation_rich (v4)

All 20 pairs pass numerical/artifact/cache comparisons and final frozen-input/global evidence guards. Each subgroup has five valid pairs, zero failed pairs. Comparison is repaired P6 versus candidate, not original upstream; geometry-warm includes fresh-process JIT.

| Regime | Threads | Median ratio | Observed range | Paired seconds standard error |
|---|---:|---:|---:|---:|
| cold | 1 | 2.782724 | 2.530215–2.862372 | 1.242353 |
| cold | 4 | 2.780870 | 2.589243–2.891544 | 1.618935 |
| geometry_warm | 1 | 4.298243 | 4.107246–4.366476 | 0.731436 |
| geometry_warm | 4 | 4.272485 | 4.166161–4.429909 | 0.900890 |

Raw evidence: `reports/characterization/p7_pairs_v4/real_vegetation_rich/{summary,pairs,trials}.json`.

## real_sparse (v4)

All 20 pairs pass numerical/artifact/cache comparisons and final frozen-input/global evidence guards. Each subgroup has five valid pairs, zero failed pairs. Comparison is repaired P6 versus candidate, not original upstream; geometry-warm includes fresh-process JIT.

| Regime | Threads | Median ratio | Observed range | Paired seconds standard error |
|---|---:|---:|---:|---:|
| cold | 1 | 2.938818 | 2.917044–2.959172 | 0.148748 |
| cold | 4 | 2.918282 | 2.882471–2.927545 | 0.291874 |
| geometry_warm | 1 | 4.515183 | 4.491135–4.555300 | 0.310571 |
| geometry_warm | 4 | 4.417327 | 4.353227–4.440575 | 0.141222 |

Raw evidence: `reports/characterization/p7_pairs_v4/real_sparse/{summary,pairs,trials}.json`.

All seven scenes are now executed and verified (140 pairs). P7 promotion/regression review remains required; no upstream speedup or release-completion claim is established.
