# P3 geometry measurements

Scope: geometry kernels, full SVF computation and legacy visibility loading only. These are not end-to-end simulation speedups. Frozen protocol: `benchmarks/protocols/p3_geometry_v1.json`. All 43 configurations passed equal-input and numerical comparisons. Raw trials, outputs, logs, source snapshots and harness: `reports/characterization/p3_geometry_v1`.

Hardware: Apple M1 Pro, 10 logical CPUs, 16 GiB RAM, macOS ARM64. Each configuration ran in a fresh process with an empty candidate JIT cache. First-call timing includes kernel compilation but excludes imports and fixture construction. SVF export occurs after the compute timer. Peak RSS is the process lifetime high-water mark, including imports, compilation and final exports; it is not process-tree or stage-only memory. Warm ranges below describe repeated calls in one process, not independent-process uncertainty.

| Task | Backend/method | Threads | Bush | First s | Warm median s [min, max] | Peak RSS MB | Retained payload MB |
|---|---|---:|---|---:|---|---:|---:|
| load | candidate/step/serial | 1 | False | 1.0052 | 0.9279 [0.9025, 0.9627] | 144.2 | 3.76 |
| load | upstream/step/serial | 1 | False | 0.0485 | 0.0305 [0.0284, 0.0388] | 387.9 | 120.32 |
| sky | candidate/pixel/parallel | 10 | True | 1.9715 | 0.0112 [0.0103, 0.0142] | 172.0 | 0.25 |
| sky | candidate/pixel/parallel | 10 | False | 1.0404 | 0.0014 [0.0013, 0.0014] | 150.8 | 0.25 |
| sky | candidate/pixel/parallel | 1 | True | 1.9006 | 0.0137 [0.0131, 0.0138] | 171.9 | 0.25 |
| sky | candidate/pixel/parallel | 1 | False | 1.0091 | 0.0051 [0.0050, 0.0053] | 155.3 | 0.25 |
| sky | candidate/pixel/parallel | 4 | True | 1.9551 | 0.0084 [0.0080, 0.0087] | 172.0 | 0.25 |
| sky | candidate/pixel/parallel | 4 | False | 0.9981 | 0.0016 [0.0016, 0.0021] | 154.2 | 0.25 |
| sky | candidate/pixel/serial | 1 | True | 0.9962 | 0.0251 [0.0249, 0.0255] | 157.1 | 0.25 |
| sky | candidate/pixel/serial | 1 | False | 0.4938 | 0.0050 [0.0049, 0.0050] | 148.4 | 0.25 |
| sky | candidate/step/parallel | 10 | True | 1.9325 | 0.0106 [0.0102, 0.0113] | 172.8 | 0.25 |
| sky | candidate/step/parallel | 10 | False | 1.9558 | 0.0086 [0.0082, 0.0117] | 162.4 | 0.25 |
| sky | candidate/step/parallel | 1 | True | 1.8868 | 0.0138 [0.0136, 0.0152] | 170.5 | 0.25 |
| sky | candidate/step/parallel | 1 | False | 1.8760 | 0.0119 [0.0118, 0.0124] | 170.0 | 0.25 |
| sky | candidate/step/parallel | 4 | True | 1.8858 | 0.0078 [0.0077, 0.0081] | 167.3 | 0.25 |
| sky | candidate/step/parallel | 4 | False | 1.9202 | 0.0060 [0.0057, 0.0063] | 167.3 | 0.25 |
| sky | candidate/step/serial | 1 | True | 0.9979 | 0.0252 [0.0252, 0.0255] | 155.9 | 0.25 |
| sky | candidate/step/serial | 1 | False | 1.1564 | 0.0235 [0.0231, 0.0238] | 156.8 | 0.25 |
| sky | upstream/step/serial | 10 | True | 0.0106 | 0.0105 [0.0101, 0.0107] | 268.6 | 0.25 |
| sky | upstream/step/serial | 10 | False | 0.0103 | 0.0095 [0.0094, 0.0108] | 269.6 | 0.25 |
| sky | upstream/step/serial | 1 | True | 0.0303 | 0.0105 [0.0102, 0.0116] | 260.8 | 0.25 |
| sky | upstream/step/serial | 1 | False | 0.0352 | 0.0097 [0.0092, 0.0101] | 246.0 | 0.25 |
| sky | upstream/step/serial | 4 | True | 0.0108 | 0.0103 [0.0100, 0.0108] | 268.6 | 0.25 |
| sky | upstream/step/serial | 4 | False | 0.0105 | 0.0096 [0.0095, 0.0099] | 268.6 | 0.25 |
| svf | candidate/pixel/serial | 1 | False | 3.3677 | 2.7770 [2.7519, 2.7961] | 155.5 | 3.76 |
| svf | candidate/step/serial | 1 | False | 11.9462 | 10.7114 [10.6515, 10.7276] | 166.6 | 3.76 |
| svf | upstream/step/serial | 10 | False | 12.8514 | 12.6075 [12.4388, 12.9824] | 401.1 | 120.32 |
| svf | upstream/step/serial | 1 | False | 3.5915 | 3.3513 [3.3371, 3.6602] | 373.6 | 120.32 |
| svf | upstream/step/serial | 4 | False | 7.5981 | 7.5269 [7.5093, 7.6797] | 405.5 | 120.32 |
| wall13 | candidate/step/parallel | 10 | False | 1.1425 | 0.0009 [0.0009, 0.0013] | 194.3 | 0.41 |
| wall13 | candidate/step/parallel | 1 | False | 1.1189 | 0.0019 [0.0018, 0.0019] | 194.6 | 0.41 |
| wall13 | candidate/step/parallel | 4 | False | 1.1111 | 0.0009 [0.0009, 0.0009] | 194.6 | 0.41 |
| wall13 | candidate/step/serial | 1 | False | 0.9838 | 0.0015 [0.0014, 0.0015] | 187.6 | 0.41 |
| wall13 | upstream/step/serial | 10 | False | 0.0042 | 0.0029 [0.0027, 0.0033] | 268.6 | 0.41 |
| wall13 | upstream/step/serial | 1 | False | 0.0347 | 0.0029 [0.0027, 0.0031] | 250.7 | 0.41 |
| wall13 | upstream/step/serial | 4 | False | 0.0034 | 0.0029 [0.0027, 0.0030] | 268.7 | 0.41 |
| wall23 | candidate/step/parallel | 10 | False | 1.6542 | 0.0019 [0.0018, 0.0022] | 206.3 | 0.66 |
| wall23 | candidate/step/parallel | 1 | False | 1.7826 | 0.0059 [0.0058, 0.0061] | 206.3 | 0.66 |
| wall23 | candidate/step/parallel | 4 | False | 1.7427 | 0.0021 [0.0021, 0.0021] | 199.2 | 0.66 |
| wall23 | candidate/step/serial | 1 | False | 1.3361 | 0.0062 [0.0060, 0.0065] | 193.0 | 0.66 |
| wall23 | upstream/step/serial | 10 | False | 0.0153 | 0.0147 [0.0139, 0.0186] | 259.3 | 0.66 |
| wall23 | upstream/step/serial | 1 | False | 0.0174 | 0.0142 [0.0135, 0.0145] | 241.8 | 0.66 |
| wall23 | upstream/step/serial | 4 | False | 0.0151 | 0.0144 [0.0140, 0.0148] | 257.9 | 0.66 |

The selected serial pixel SVF configuration is faster and uses less peak memory than the strongest measured upstream configuration (one thread) on this scene, but does not establish a general 2x runtime or 4x process-memory improvement. Visibility payload is 32x smaller here because all channels are binary; ternary and raw fallback data have different compression. Object overhead is excluded from payload bytes.

Legacy compact archive loading is substantially slower than dense upstream loading, despite lower process peak RSS. Positive-bush serial shadow execution is also slower than upstream. First-call JIT costs can dominate small kernels. These regressions are retained in the table, not removed from the workload.

After measurement, the public sky wrapper was changed from step-serial to the measured pixel-serial implementation, whose positive-bush branch delegates to step-serial. The archived benchmark source records the pre-selection wrapper and explicitly selected both variants. Native mmap storage was added separately and is not measured by this matrix. Automatic dependency-valid geometry reuse and complete process-tree memory control remain P6. CUDA is unavailable.
