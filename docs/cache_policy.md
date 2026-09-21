# Geometry cache and standalone exports

Native geometry is stored separately under `.solweig-light/cache` (or the
configured cache directory). Identities hash raster contents and record raster
metadata, patch arrays and order, normalization, dependencies, and implementation
sources. Standalone construction has its own identity because its normalization
need not match thermal-driver construction. Native payloads are immutable;
checksummed manifests publish completed generations. Replaced generations remain
on disk so readers can retain mapped payloads safely. Per-key Unix advisory locks
coordinate identical requests and release automatically if a process crashes.

Source files are content-checked before and after identity capture, around
geometry production, before legacy replacement, and before completion manifest
publication. Detected mutation refuses the operation; a producer cannot commit
geometry under an identity captured from different source bytes. Mutation while
staging preserves previous exports, and mutation during replacement invokes the
ordinary rollback before committing a new completion manifest. These checks
detect changing sources at the transaction boundaries rather than locking
external programs that may edit the inputs.

Standalone and thermal preprocessing geometry preparation run memory admission
before loading any raster arrays or producing cold geometry. Admission uses DSM
dimensions read from metadata and the requested patch table's actual count. The
current planner uses its conservative full-tile pipeline inventory for this
geometry-only stage, including thermal/radiation reserves that may overestimate
the stage's memory. A rejected tile requires a larger supported budget; logical
domains, resolution, rays, and sky patches are never reduced to fit it.

The legacy `SVF` directory retains exactly its promised TIFF, ZIP, and NPZ names.
Export completion manifests and locks live in
`.solweig-light/export-manifests`. A native cache hit does not establish legacy
artifact presence or provenance.

Standalone exports also hold the same per-destination locks as transactional
thermal output publication, under `.solweig-light-locks`. Both workflows hash
resolved destination paths and acquire locks in sorted order without waiting
while holding a subset. A competing owner causes an explicit busy error before
standalone geometry production or transactional export publication. The
standalone set lock serializes callers across its three artifact destinations;
all destination locks remain held through final manifest publication or rollback.

For standalone `calculate_svf(..., overwrite=False)`, an existing complete set of
three legacy artifacts is reused only when a matching export manifest, current
file content hashes, and schemas prove validity. Without that proof, the current
native geometry is obtained or computed and every legacy value is compared
exactly, including decoded visibility, dtype, shape, patch count, spatial
metadata, and NoData. Equal exports are left unchanged and receive a validation
manifest. Unequal or malformed exports are preserved and cause an actionable
error requesting `overwrite=True`. No numerical tolerance is loosened to accept
legacy exports. Upstream scientific comparisons retain their separately frozen
tolerances.

`overwrite=True` stages and validates all three new artifacts before replacing
any destination. A destination lock covers the complete set; the completion
manifest publishes last. Ordinary publication errors restore previous exports
and their manifest. Abrupt process termination can leave an incomplete set or
mixed generations; file hashes and schema validation prevent treating that as
completed publication. A partial pre-existing set follows the upstream rule:
regenerate the entire set. Export production failures preserve existing files.

The thermal driver retains its separate legacy-artifact-presence policy. It does
not use native hits as the legacy `available_on_disk` flag. Explicit trusted
legacy loading logs missing dependency identity and still enforces schemas;
ordinary own-met execution recomputes geometry when provenance is absent.
