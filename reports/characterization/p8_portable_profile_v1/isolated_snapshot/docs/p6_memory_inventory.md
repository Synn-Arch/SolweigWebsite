# P6 runtime memory inventory

This is the conservative admission inventory for `solweig_light.runtime`. It
is a planning estimate, not a hard process-tree RSS bound. P6 measurements
must report summed resident sets separately and may not present this estimate
as observed peak memory.

One full-plane equivalent is `rows * cols` float32 values. The inventory uses
192 such planes for live scene and numerical work:

| Family | Plane equivalents | Contents represented |
| --- | ---: | --- |
| Scene and static geometry | 32 | DSM, DEM, trees, building/vegetation masks, walls/aspects, SVF directional fields, and derived geometry |
| Chronological state and forcing | 24 | carried temperature/delay state, meteorology, solar coordinates, masks, and material grids |
| Engine outputs and diagnostics | 48 | the engine return family, comfort/radiation fields, shadow and requested output staging |
| Radiation and ground-view scratch | 88 | patch reductions, anisotropic terms, longwave/shortwave intermediates, and temporary reductions |
| **Subtotal** | **192** | **float32-equivalent live full planes** |

The estimate reserves another 32 full planes for dtype64 promotion and mixed
dtype temporaries. It separately adds 12 directional wind coefficient planes
by default. Raw visibility fallback is always retained as
`3 * rows * cols * patches * 4` bytes for the three float32 channels, with
patch option 2 using 153 patches unless the caller supplies another count.

Decoded patch blocks are accounted as
`block_pixels * patches * (windchannels + 4) * 4` bytes. A fixed native/JIT
allowance covers compiled code and native workspaces. The allowance is not a
claim that those allocations are bounded by the configured value; process-tree
RSS measurements remain the evidence for actual memory behavior.

The estimate must reject a tile whose single-job inventory exceeds the memory
budget before computation. Aggregate worker admission sums the conservative
per-job estimates and caps native threads independently.

When `memory_budget_bytes` is omitted, the default is a conservative fraction
of the smallest available view among host physical RAM, host available RAM,
and the process's finite cgroup memory headroom. Cgroup headroom is computed
per current cgroup and ancestor as `memory.max - memory.current` (or the v1
`memory.limit_in_bytes - memory.usage_in_bytes` pair), then the minimum is
used. This is a point-in-time admission snapshot: other processes can consume
the headroom after resolution. Zero or exhausted headroom resolves to zero and
causes admission to reject work; it is never converted into a one-byte or
one-pixel fallback.
