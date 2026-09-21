# P4 component measurements

All 29 frozen configurations passed output, dtype, mask and observable-mutation comparisons. These are synthetic 128×140 component calls with 153 sky patches, not chronological end-to-end timings. Original inputs, all raw trials, output arrays, logs, environments, protocol, and guarded source snapshots are retained in `reports/characterization/p4_radiation_v1`.

Hardware: Apple M1 Pro, 10 logical CPUs, 16 GiB RAM, macOS ARM64. Each configuration used a fresh process and empty JIT cache. First-call times include kernel compilation and descriptor construction, excluding imports, fixture loading and fresh input copying. Each call receives fresh inputs. Warm ranges are three repeats within one process, not independent-process confidence intervals. Peak RSS covers the entire process, including imports, fixture representations, checking, compilation and output export; it is not stage-only or process-tree memory.

| Component | Backend / visibility / execution | Threads | First s | Warm median s [min, max] | Lifetime RSS MB |
|---|---|---:|---:|---|---:|
| Kside_veg_v2022a | candidate / compact / parallel | 10 | 2.5756 | 0.8860 [0.8798, 0.8953] | 247.2 |
| Kside_veg_v2022a | candidate / dense / parallel | 10 | 2.2113 | 0.6047 [0.5935, 0.6123] | 368.3 |
| Kside_veg_v2022a | candidate / compact / parallel | 1 | 2.6106 | 0.8946 [0.8711, 0.9310] | 215.5 |
| Kside_veg_v2022a | candidate / dense / parallel | 1 | 2.1922 | 0.5589 [0.5567, 0.5621] | 350.7 |
| Kside_veg_v2022a | candidate / compact / parallel | 4 | 2.5381 | 0.8646 [0.8621, 0.8717] | 230.8 |
| Kside_veg_v2022a | candidate / dense / parallel | 4 | 2.1970 | 0.5757 [0.5744, 0.5763] | 353.4 |
| Kside_veg_v2022a | candidate / compact / serial | 1 | 2.0091 | 0.8637 [0.8628, 0.8663] | 257.0 |
| Kside_veg_v2022a | candidate / dense / serial | 1 | 1.7447 | 0.5597 [0.5579, 0.5614] | 356.0 |
| Kside_veg_v2022a | upstream / dense / serial | 10 | 0.0905 | 0.0882 [0.0870, 0.0885] | 431.5 |
| Kside_veg_v2022a | upstream / dense / serial | 1 | 0.1065 | 0.0778 [0.0723, 0.0814] | 422.3 |
| Kside_veg_v2022a | upstream / dense / serial | 4 | 0.0931 | 0.0860 [0.0831, 0.0883] | 430.4 |
| Lcyl_v2022a | candidate / compact / parallel | 10 | 2.7289 | 0.5141 [0.5119, 0.5215] | 226.6 |
| Lcyl_v2022a | candidate / dense / parallel | 10 | 2.6091 | 0.3955 [0.3893, 0.3968] | 327.4 |
| Lcyl_v2022a | candidate / compact / parallel | 1 | 2.5460 | 0.5251 [0.4980, 0.5393] | 259.9 |
| Lcyl_v2022a | candidate / dense / parallel | 1 | 2.5456 | 0.3874 [0.3813, 0.3884] | 346.2 |
| Lcyl_v2022a | candidate / compact / parallel | 4 | 3.1733 | 0.4968 [0.4933, 0.4969] | 235.4 |
| Lcyl_v2022a | candidate / dense / parallel | 4 | 2.4480 | 0.3798 [0.3768, 0.3873] | 344.8 |
| Lcyl_v2022a | candidate / compact / serial | 1 | 2.6678 | 0.5016 [0.4991, 0.5046] | 266.0 |
| Lcyl_v2022a | candidate / dense / serial | 1 | 2.6363 | 0.3748 [0.3719, 0.3770] | 353.2 |
| Lcyl_v2022a | upstream / dense / serial | 10 | 0.1789 | 0.1689 [0.1660, 0.1699] | 391.3 |
| Lcyl_v2022a | upstream / dense / serial | 1 | 0.1913 | 0.1467 [0.1464, 0.1585] | 379.6 |
| Lcyl_v2022a | upstream / dense / serial | 4 | 0.1682 | 0.1688 [0.1639, 0.1718] | 391.4 |
| gvf_2018a | candidate / dense / parallel | 10 | 1.5836 | 0.0289 [0.0285, 0.0307] | 214.5 |
| gvf_2018a | candidate / dense / parallel | 1 | 1.4710 | 0.0554 [0.0551, 0.0577] | 206.6 |
| gvf_2018a | candidate / dense / parallel | 4 | 1.5165 | 0.0301 [0.0295, 0.0313] | 213.0 |
| gvf_2018a | candidate / dense / serial | 1 | 1.4978 | 0.0661 [0.0659, 0.0671] | 209.7 |
| gvf_2018a | upstream / dense / serial | 10 | 0.0976 | 0.0885 [0.0882, 0.0904] | 281.5 |
| gvf_2018a | upstream / dense / serial | 1 | 0.1228 | 0.0893 [0.0886, 0.0903] | 267.4 |
| gvf_2018a | upstream / dense / serial | 4 | 0.0926 | 0.0904 [0.0889, 0.0924] | 282.1 |

Ground-view warm execution improves on the measured upstream configurations; its first call is slower because of compilation. Patch radiation shows substantial runtime regressions at the frozen 128-pixel decode block: serial compact shortwave is about 11.1× the fastest measured upstream warm time, and serial compact longwave about 3.4×. Dense candidate paths also regress. Parallel threads do not remove the dominant cost at this block size. Compact candidate process peaks are lower, but import/fixture/compiler effects prevent attributing the entire RSS difference to the reduction alone.

These measurements do not establish the exact bottleneck; repeated Python block preparation, visibility decoding and classification are candidates for profiling. No default end-to-end speedup is claimed. Retain this matrix when tuning execution blocks or native decoding in P6/P7; do not replace the measured regressions with a different workload or reinterpret warm kernels as complete simulations. Release performance gates remain outstanding.

The fixture was captured before the explicit candidate_block_pixels=128 field was added to the protocol; both hashes are recorded. Spatial shape, forcing, patch count and original data were unchanged. All runs use the final frozen protocol and source hash guards. CUDA is unavailable.
