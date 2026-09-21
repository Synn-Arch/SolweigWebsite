# P4 radiation execution contract

The chronological engine retains its source-derived NumPy parameterization and
state updates. Compiled dispatch replaces receiver traversals and patch
reductions; it does not reduce rays, patches, vegetation channels or timesteps.

Ground-view schedules contain offsets and slice bounds only. Their keys include
shape and exact scalar representations. Each direction reconstructs its dynamic
radiance/albedo inputs and snapshots them before receiver execution. A receiver
owns its running minimum, scratch history and channel sums. Values outside a
new ray slice retain their previous value, as in the original recurrence. Water
`Tg` mutation occurs after the first directional radiance calculation, and
`sunwall` normalization remains observable. Direction reductions retain their
original order. Unsupported float64 raster profiles use the array reference.

Patch geometry retains immutable position, solid-angle, trigonometric,
altitude-group and cardinal metadata in an eight-entry cache. Geometry keys
exclude luminance. Emissivity, Perez coefficients, solar classification and
surface radiance remain dynamic. Shortwave rotation-dependent factors are
recomputed. Longwave retains both ordered sweeps, including reflection after
sky totals are known.

Visibility is decoded into bounded receiver blocks. Each receiver accumulates
patch contributions serially in sky order; parallel workers do not share
accumulators or packed writes. Dense compatibility inputs and compact runtime
inputs use the same reduction. A block changes memory locality, not logical
scene extent, sky geometry, solar coordinates or forcing. The normal engine
selects serial kernels; parallel variants are separately verified.

Scalar argument distinctions remain observable. A native scalar and a
zero-dimensional tensor-origin array are not interchangeable for every original
function. Reference-order dispatch preserves the original box-selection and
longwave-tangent failures without rejecting inactive branches. See
`docs/model_deviations.md` and the typed patch packet.

Verification includes original component boundaries, targeted ground-view
mutation snapshots, typed patch fixtures, irregular thermal-delay thresholds,
nine TIFF workflows and a two-day chronological sequence. Test-only serialized
state handoff verifies numerical resumption; it does not provide production
checkpoint persistence, output recovery or cache dependency validation. Those
remain P6 work.

The fixed component timing protocol is
`benchmarks/protocols/p4_radiation_v1.json`. Its synthetic repeated spatial
fixtures preserve each component's full workload and input types. Component
first-call and warm timings must not be described as end-to-end performance.
