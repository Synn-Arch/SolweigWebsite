# P7 fixture preparation audit

This record covers input preparation only. No numerical comparison or benchmark
is implied. Existing `small`, `medium256`, `repeated_blocks`, their generator,
and `benchmarks/protocols/benchmark_v1.json` are retained unchanged.

## Synthetic additional families

`tools/prepare_p7_fixtures.py` defines two explicit 32-pixel motifs, with no
random sampling. `dense_urban` has 16/24/32 m courtyard blocks; building footprint
is 50% and canopy footprint 0.78125%. `vegetation_rich` has isolated 10 m blocks,
12/14/16 m broad canopy with cross-shaped gaps; building footprint is 3.515625%
and canopy footprint 55.859375%. Fractions are measured from generated inputs;
neither family is an observed real scene. Terrain is 3 m, tree values are canopy
height above terrain, and all rasters are float32, EPSG:32618, 1 m square pixels,
with no NoData. The geographic centre and 24-row analytic P0 forcing are retained.
Full initial save flags and 153-patch workload remain in force. Logical tile
settings are 3600/20, including for 256-pixel input domains.

```
.venv-light/bin/python tools/prepare_p7_fixtures.py
```

The default prepares two 256x256 scenes. Larger multiples of 32 through 3600
can be requested explicitly; there is no automatic shrinking or benchmark
execution. Three resident float32 rasters require roughly 12 bytes/pixel,
excluding GDAL/read-back temporaries. Simulation memory is a separate admission
decision under the frozen 12 GiB process-tree limit. Existing destinations are
rejected. GDAL read-back verified values, transform, and CRS. Durable manifests
and Python/GDAL/NumPy versions are in `reports/p7_fixture_preparation.json`;
generated inputs and per-scene manifests live under `tests/fixtures/generated/p7`.
Independent regeneration in `/tmp/solweig-p7-final-reproduction-20260918` produced
identical fixture hashes for both scenes; invalid size 31 was rejected with exit
2. No numerical pipeline or benchmark was executed.
The candidate Git repository has unborn HEAD, so provenance uses working-tree
generator/input hashes rather than inventing a commit.

## Real source audit

No real raster is shipped in the pinned upstream checkout. Its examples link
[Zenodo 18283037](https://zenodo.org/records/18283037), while its README links
[Zenodo 21081622](https://zenodo.org/records/21081622). The record APIs identify
both licenses as `gpl-3.0-or-later`; the rendered HTML license field omits the
value. Both list the same `Input_rasters.zip`, 94,896,401 bytes,
MD5 `3da64b3eeb71a6838057674838c6c4a7`. Record-level licensing is not proof of the
underlying acquisition lineage, vertical datum, or canopy-height derivation.
The bundle is a source candidate until those are inspected. Its raw API metadata
and archive are kept under `tests/fixtures/generated/p7_sources/zenodo_21081622`.
The 11.1 GB forcing archive is not needed and is not acquired.

Acquisition completed: archive SHA256 is
`9429fac970a29b3bd21217cbd0876f61d3df35e923efd077b7754e9134960a34`;
byte count and advertised MD5 matched. The archive contains only four rasters
and a land-cover style file, with no acquisition report. All four rasters are
3367x3913 float32, aligned EPSG:32614, **2 m** pixels (not the synthetic benchmark's
1 m resolution). DEM and building rasters have no NoData; trees declare zero
NoData, land cover declares float32 minimum. Metadata contains only
`AREA_OR_POINT=Area`, without vertical datum or acquisition lineage.
`reports/p7_real_source_audit.json` records the exact grid and band inventory.
The older record README lists required layer names and Austin forcing examples,
but does not resolve raster acquisition or height derivation. The source record and
pinned upstream README explicitly document these as model input rasters: DEM
excludes buildings, Building_DSM includes terrain, Trees contains canopy height.
This permits source-published model-input windows for compatibility/performance
with those intended semantics; it does not establish independent scientific
ground truth or observational validation. No resampling to 1 m is
silently performed. Trees' zero-NoData declaration conflicts with the normal
zero-for-no-canopy interpretation and remains explicit.

The source acquisition can be reproduced into an absent directory with:

```
.venv-light/bin/python tools/prepare_p7_fixtures.py --fetch-zenodo --output tests/fixtures/generated/p7_sources/reproduction
```

This mode pins record license, byte count and archive MD5, streams with a hard
94,896,401-byte download bound, writes a partial archive until validation passes,
and records SHA256 and unresolved provenance in an acquisition manifest. It
does not extract untrusted archive paths or run the model.

As an alternative acquisition route, [USGS 3DEP products](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services)
and [LidarExplorer](https://www.usgs.gov/tools/lidarexplorer) provide terrain and
point-cloud sources. Deriving separate buildings and canopy from point-cloud
classes would require a frozen, documented derivation and dataset-specific
metadata. A terrain DEM alone cannot supply these real fixture families.
The upstream `build_inputs` acquisition route additionally requires Google
Earth Engine authentication; no authenticated live acquisition is established.

## Prospective real-window rule, frozen before performance inspection

If the archive supports aligned terrain/building/canopy semantics, inspect
nonoverlapping 256x256 windows anchored at raster origin (row-major order); omit
partial edge windows and windows with invalid pixels. Define building fraction
as `mean(Building_DSM - DEM >= 2)` and canopy fraction as `mean(Trees > 0)`.
Choose dense urban by largest building fraction, vegetation rich by largest
canopy fraction, and sparse by smallest building-plus-canopy fraction among
windows with nonzero building and canopy fractions. Ties choose first row-major
window. Require three distinct windows; labels describe relative sample density,
not a claim of independent city/biome coverage. Publish measured fractions and
source offsets before using windows in performance work. Retain each entire
declared logical input domain with its transform; no later crop or physics
reduction is allowed. Published source documentation supplies intended model-input height semantics;
unknown original acquisition lineage and vertical datum remain limitations. Land cover, forcing availability, source hashes, license metadata,
selection rule and derivations must be recorded. This is relative sparse/urban/vegetation coverage within one source-published
sample; independently acquired regional/observational coverage remains absent.

The density inventory was executed for 195 complete windows without model
execution; results are in `reports/p7_real_window_inventory.json`. It treats
zero trees as absent canopy **for inventory only**, retaining the metadata
conflict above. The three distinct candidate offsets are urban `(0,1280)`
(29.3930% building, 9.8770% canopy), vegetation `(0,256)` (19.2352%, 52.0218%),
and sparse `(2560,1536)` (3.7949%, 8.4396%). All contain land-cover values
1, 2, 4, 5, 6. These are relative density labels. They were subsequently admitted as
source-published model-input fixtures and written losslessly to
`tests/fixtures/generated/p7_real`, after the lead verified the source record
input semantics. No performance results had been inspected. Reproduce the inventory into an absent
JSON destination:

```
.venv-light/bin/python tools/prepare_p7_fixtures.py --inventory-zenodo tests/fixtures/generated/p7_sources/zenodo_21081622/Input_rasters.zip --output /tmp/solweig-p7-source-inventory.json
```

Preparation command:

```
.venv-light/bin/python tools/prepare_p7_fixtures.py --prepare-real-zenodo tests/fixtures/generated/p7_sources/zenodo_21081622/Input_rasters.zip --output tests/fixtures/generated/p7_real
```

Full 256x256 domains, original 2 m resolution, source NoData metadata (including
Trees=0), and source raster values are preserved. For this frozen selection,
invalid means nonfinite numeric pixels or the land-cover finite NoData sentinel;
Trees=0 remains in the inventory as absent canopy under published input semantics.
This clarifies the prospective invalid-pixel rule before window creation and
without observing performance. Land cover is retained but disabled (`None`)
because no mapping is asserted. The 24-row analytic P0 meteorology is explicitly
synthetic forcing, not observed Austin weather. Tile size/overlap remain 3600/20,
all initial output flags retained. Three-window read-back is exact, with no
negative tree heights, building elevations below terrain, or nonfinite numeric
pixels in the three geometry layers. Archive plus windows occupy under 0.3 GB;
no full-scene model or heavy benchmark was executed.

Next: establish expanded component and chronological numerical comparisons
before admitting performance trials; retain unknown lineage/datum and the
Trees zero-NoData quirk as explicit limitations.
