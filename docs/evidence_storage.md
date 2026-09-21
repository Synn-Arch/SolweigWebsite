# Evidence storage

The repository keeps concise reports, protocols, provenance records, and source
snapshots under `reports/` when tests or scripts use them as evidence. The
authoritative source snapshots include `p8_utci_official_source`,
`p8_umep_source`, and `p7_p6_baseline/src`; these remain visible to Git. The
root reports and the small review documents named in
`reports/local_evidence_inventory.json` are retained for the same reason.

Generated rasters, array dumps, compiled Numba artifacts, JIT and geometry
caches, local environments, output folders, processed inputs, and the complete
`reports/runs/` tree remain local-only. The scoped rules live in `.gitignore`.
They preserve report text and JSON by default; only explicitly identified
large sampling or whole-tree inventory captures are additionally ignored.

This policy does not delete or move local evidence. The inventory records
observed paths and sizes from the checkpoint and points to existing manifests
or hash records without recomputing them. Local-only evidence is therefore not
part of a clone and cannot be reproduced from this document alone: rerunning a
case may require the exact fixture, dependency environment, upstream commit,
hardware, cache state, and remote input recorded by its own protocol.

The local CPU optimization experiment additionally records its two large
baseline sampling captures in
[`storage_manifest.json`](../reports/characterization/local_cpu_optimization_v1/storage_manifest.json).
Those captures remain local with verified hashes. The paired trial
measurements, RSS JSONL samples, exact comparisons, protocols and source
snapshots remain tracked; generated raster histories and wheel binaries
follow the same local-only policy above.

To review the prospective tracked set without adding files, run:

```sh
git ls-files --others --exclude-standard
```
