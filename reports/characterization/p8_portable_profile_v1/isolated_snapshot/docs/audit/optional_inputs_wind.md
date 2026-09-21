# Optional input construction and directional wind audit

This audit is a static inspection of the pinned upstream files at commit
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec`:

- `.upstream/SOLWEIG-GPU/solweig_gpu/create_inputs.py` (1,851 lines)
- `.upstream/SOLWEIG-GPU/solweig_gpu/wind_ext_coeff.py` (1,049 lines)

No upstream function was executed while producing this document. Statements in
the “Observed” column are source claims; “Validation needed” identifies work
that still requires fixtures or a live service. This scope corresponds to the
optional-input and directional-wind requirements in implementation-plan §8.2–8.3.

## Function graph and inspected coverage

The complete top-level callable/class inventory was inspected with line
anchors:

| File | Inventory |
|---|---|
| `wind_ext_coeff.py` | `_tlog` 23, `_ensure_dir` 27, `_find_first_file` 31, `_find_building_raster` 54, `_find_tree_raster` 90, `_find_era5_fsr_file` 108, `_building_raster_midpoint_lonlat` 121, `_coord_name` 148, `_era5_target_lon` 161, `_read_z0_from_fsr_at_raster_midpoint` 178, `_center_crop_or_pad` 249, `_rotate_full_extent` 270, `_coeff_at_z_trees` 317, `_building_wake_lr_from_rot` 384, `_trees_wake_lr_from_rot` 504, `_gaussian_smooth` 643, `_save_like_meta` 670, `_read_building_height` 706, `_compute_wind_full_domain` 726, `calculate_wind_ext_coeff` 932. |
| `create_inputs.py` | `_ensure` 36; dataclasses `BBoxes` 86, `GridSpec` 95, `Paths` 102; `build_paths` 125; logging/helpers `tlog` 339, `timed` 342, `ensure_dir` 351, `fail` 354, `clear_osmnx_cache` 358; geometry/service functions `utm_epsg_from_latlon` 392, `compute_bounding_boxes` 397, `reverse_geocode` 431, `safe_initialize_ee` 446, `_bbox_to_poly` 492, `_get_osm_polygons` 496, `_format_gba_lon` 512, `_format_gba_lat` 517, `_gba_tile_name` 522, `_fetch_gba_buildings_tiled` 545, `_clip_polygons` 607, `_grid_cells` 620, `_ee_split_config` 635, `_fetch_gba_buildings_ee_subset` 656, `build_vectors_from_osm_gba` 762; downloads ` _generate_tiles` 814, `download_worldcover` 833, `download_lcz` 883, `download_tree_dsm` 896, `download_dem` 982; local transforms `reclassify_esa_worldcover_inplace` 997, `compute_reference_grid` 1028, `_resample_to_grid` 1036, `_write_all_zeros_raster` 1052, `_save_array_like` 1060, `_safe_tag` 1078, `_safe_log_ratio_at_z` 1082, `_win_stats` 1112, `_lambdaf_edges` 1161, `_coeff_at_z_buildings` 1252, `_fill_nearest` 1303, `_read_vector_or_warn` 1320, `rasterize_vector_checked` 1337, `rasterize_polygons_to_array` 1367, `create_building_dsm_and_clean_trees` 1380, `check_raster_alignment` 1408, `final_report_table` 1440; forcing/orchestration `download_and_embed_era5` 1464, `run_create_inputs` 1668, `main` 1825. |

The wind call graph is:

```text
calculate_wind_ext_coeff
 ├─ _find_building_raster → _find_first_file
 ├─ _find_tree_raster → _find_first_file
 ├─ _find_era5_fsr_file
 ├─ _read_z0_from_fsr_at_raster_midpoint
 │   ├─ _building_raster_midpoint_lonlat → rasterio.warp.transform
 │   ├─ _coord_name, _era5_target_lon
 │   └─ xarray.open_dataset (optional at call time)
 └─ _compute_wind_full_domain
     ├─ _read_building_height
     ├─ _coeff_at_z_trees
     ├─ scipy grey_erosion / gaussian_filter
     └─ per direction: _rotate_full_extent → wake helpers → _save_like_meta
```

`create_inputs.run_create_inputs` calls, in order: `build_paths`,
`compute_bounding_boxes`, `safe_initialize_ee`, `build_vectors_from_osm_gba`,
`download_worldcover`, `download_tree_dsm`, `download_dem`,
`compute_reference_grid`, three `_resample_to_grid` calls,
`rasterize_vector_checked`, `reclassify_esa_worldcover_inplace`,
`create_building_dsm_and_clean_trees`, and `check_raster_alignment`. It returns
the output directory string at line 1823. `download_lcz` is implemented but is
not called by `run_create_inputs` (lines 882–893 and 1752–1755). The CLI
`main` at lines 1825–1847 calls `run_create_inputs` with `args.year_start` and
`args.year_end`, but those arguments are not defined by its parser and are not
parameters of `run_create_inputs`; this is an observed upstream defect requiring
an explicit compatibility decision.

## Directional wind coefficients

### Public contract and inputs

`calculate_wind_ext_coeff(input_dir, era5_dir, *, directions=range(0,360,30),
z0_ref=0.03, hmin_b=1.0, hmin_t=1.0, z_eval=10.0, zref=10.0, LAI_t=2.0,
a0_t=0.5, a1_t=0.4, alpha_min_t=0.2, alpha_max_t=2.5, coeff_min=0.1,
coeff_max=1.0, lp_min_open=0.02, max_workers=None)` returns a sorted list of
`Path` objects (932–951, 1001–1049). It requires both directories and searches
only their top level. Building selection prefers `Buildings.tif`, then case
variants, `Building_Height.tif`, and finally `Building_DSM.tif` patterns;
`WindCoeff_` files are excluded (54–88). Tree selection is `Trees.tif` and
tree-DSM variants (90–105). ERA5 roughness requires exactly
`data_stream-oper_stepType-instant.nc` (108–118).

`Building_DSM.tif` is accepted as a fallback with a warning, but the function
does not subtract DEM: `_read_building_height` treats the selected raster as
obstacle height above ground and thresholds values below `hmin_b` to zero
(706–723). This is a compatibility-critical distinction from the radiation
DSM convention and the input builder's `Building_DSM = DEM + height` operation
(1380–1406).

The building and tree arrays must have the same shape (760–765); the tree mask
uses finite values at least `hmin_t` (767–769). Raster output copies the building
profile, forces one float32 band, NaN nodata, tiled ZSTD level 3, and falls back
to tiled Deflate/predictor 3 if the first write fails (670–704). Each requested
direction is named `WindCoeff_dir{angle:03d}.tif` (921–925). There is no explicit
direction normalization; arbitrary supplied integer angles are formatted with
three digits (or more if needed).

### Roughness and coefficient behavior

The wrapper default fallback roughness is `z0_ref=0.03` m (937), while the
internal `_compute_wind_full_domain` default is `0.7` m (737). The wrapper
passes the extracted/fallback value through, so the public default is 0.03 m
unless valid ERA5 `fsr` replaces it. `_read_z0_from_fsr_at_raster_midpoint`
uses the building bounds midpoint transformed to EPSG:4326, selects `fsr` at
the nearest latitude/longitude grid point, and selects the first time-like
dimension value; invalid/missing values retain the fallback (121–247). This
observes the executable helper behavior; any docstring claim of averaging times
would be a discrepancy to register and test.

Tree local coefficients use `z_eval=10`, `LAI`, `a0`, `a1`, alpha clamps and
coefficient clamps. The public values differ from helper defaults: LAI 2.0,
`a0=0.5`, `a1=0.4`, alpha 0.2–2.5, `lp_min_open=0.02`, output 0.1–1.0
(317–381, 776–790, 932–950). Building and tree wakes are computed in rotated
coordinates. Rotations preserve full extent, use cardinal-angle fast paths,
order 0 for masks and order 1 for heights/coefficients, and center crop/pad to
the original shape (249–315, 856–906). Connected components are labeled with
`scipy.ndimage.label`; overlapping wake segments combine by `minimum`, with
front/back ramps derived from component width, depth and mean height
(384–501, 504–640). Tree coefficients additionally undergo a circular
approximately 10 m grey-erosion footprint (795–807, 894–898).

Every direction runs in a `ThreadPoolExecutor`; default workers are bounded by
the number of directions and `os.cpu_count()`, but an explicit `max_workers`
is accepted (908–917). Results are written as futures complete, then sorted
only in the returned list. The output directory, raster dimensions, metadata,
interpolation, connected-component connectivity, and minimum-combination order
must be covered by CPU fixtures. No runtime validation was performed here.

## Optional input construction

### Dependencies and service boundary

At module import, `_ensure` attempts to install missing required packages via
`pip` (36–61), then imports Earth Engine (`ee`), geemap, GeoPandas/Fiona,
NumPy/Pandas, rasterio, requests, OSMnx, geopy, pyproj, shapely, xarray, cftime,
netCDF4 and SciPy (62–79). This makes importing the optional module network and
package-manager sensitive. The CPU-native core must therefore lazy-import this
workflow or isolate it behind an optional extra, as required by plan §8.3.

External boundaries are: Nominatim reverse geocoding (431–442), Google Earth
Engine initialization/authentication and exports (446–483, 833–992, 1464–1664),
OSM via OSMnx (496–510), the Global Building Atlas WFS
`https://tubvsig-so2sat-vm1.srv.mwn.de/geoserver/wfs` layer
`global3D:lod1_global` (487–604), and the GBA Earth Engine asset fallback
`projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS` (656–758). Local,
deterministic boundaries are raster alignment, reprojection, classification,
rasterization, DSM construction and NetCDF normalization (997–1462,
1593–1664). Recorded service responses can test local transformations offline;
that cannot establish live-service compatibility.

### Defaults, geometry and schemas

`run_create_inputs(lat, lon, city=None, km_buffer=3, km_reduced_lat=1,
km_reduced_lon=1, base_folder=None, resolution=2)` returns
`{base_folder}/{city}_for_solweig/` (1668–1707). If no city is given,
Nominatim supplies one; the module default base is `/Users/` when that path
exists, otherwise the module directory (160–169). Bounding boxes use a
latitude/longitude km approximation, add a 1 km margin for OSM/ERA5, choose UTM
from longitude/hemisphere, and shrink the UTM polygon by the reduced distances
(392–426). The target grid is north-up, square, `ceil(extent/resolution)` and
uses the UTM CRS (1028–1034).

`Paths` names the intermediate/output schema: source `ESA_WorldCover_2020.tif`,
`tree_dsm_1m_merged.tif`, `downloaded_dem.tif`, vectors
`vegetation.geojson`, `water.geojson`, `building.geojson`, `impervious.geojson`,
and output `Vegetation.tif`, `Water.tif`, `Buildings.tif`, `Landuse.tif`,
`Trees.tif`, `DEM.tif`, `Building_DSM.tif`, `LCZ.tif`, and
`WindCoeff_dir000.tif` (102–156). In the current orchestration, only building
vector/raster is required; vegetation, water, impervious, LCZ and wind output
paths are declared but not built by `run_create_inputs`.

Buildings prefer cached GeoJSON, then tiled GBA WFS, GBA Earth Engine export,
and OSM building polygons. Missing numeric `height` becomes 10.0 m and is
written as `HEIGHT_ROOF` (762–810). WorldCover v200 is downloaded at 10 m and
reclassified in place: 50→1 paved, 20/95→3, 10→4, 30/40/90→5, 60/70/100→6,
80→7, 0→0; pixels with building raster >1 overwrite to class 2
(833–880, 997–1026). Trees and DEM are bilinear-resampled; WorldCover is
nearest-resampled (1760–1765). Building DSM is float32 DEM plus building
height, and tree pixels over building height >1 are set to zero in place
(1380–1406). Alignment checks compare CRS, resolution, rounded bounds and
shape (1408–1438).

### ERA5 output schema and quirks

`download_and_embed_era5(out_nc, bbox_osm, year_start, year_end, data_dir)`
extracts a bbox-center point from Earth Engine ERA5 hourly data, month by month,
through the last completed month; each month retries `getRegion` three times
with a ten-second delay (1464–1565). It requests temperature, dew point,
pressure, 10 m winds, shortwave, longwave and geopotential, plus an available
roughness alias. It writes a single-point xarray NetCDF with dimensions
`time,lat,lon`, standardized variables in `WANTED_VARS` order:
`T2M,RH2M,SWDOWN,HGT,LWDOWN,PSFC,U10M,V10M,Z0` when present (1600–1658).
Relative humidity uses the Magnus constants 17.625 and 243.04 and is clipped
0–100; `ssrd`/`strd` are divided by 3600 to W m⁻²; geopotential is divided by
`G0=9.80665` to `surface_altitude`; dew point and geopotential intermediates
are dropped. The file is written to `.tmp` and atomically renamed with
`os.replace` (1612–1664). The roughness metadata says metres and
`surface_roughness_length` (1612–1618).

`run_create_inputs` never calls `download_and_embed_era5`, despite its docstring
claiming meteorological NetCDF output (1668–1687, 1752–1823). The CLI also
passes undefined `year_start`/`year_end` names (1825–1847). These are source
observations, not repaired behavior. Compatibility tests should preserve the
defined public contract deliberately and classify unexecuted acquisition as
not validated until a policy and fixtures exist.

## Required follow-up validation

Before implementation claims parity, add offline fixtures for: building DSM
versus above-ground building height; missing/invalid `fsr`; longitude 0–360
versus −180–180; all 12 default directions and wraparound/tie selection;
rotation/cardinal interpolation; connected-component wake overlap and borders;
NaN/nodata and output metadata; WorldCover mapping and building overwrite;
tree removal and DSM arithmetic; CRS/grid alignment; NetCDF units, RH and
surface-altitude derivation; stale/partial files and atomic output. Add separate
opt-in live smoke tests for Earth Engine, OSM/Nominatim and GBA. Label mocked
service responses as mocks, and record unavailable credentials/network as
unavailable rather than passing them.
