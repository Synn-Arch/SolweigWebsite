"""Serial chronological CPU driver with streaming outputs.

Derived from SOLWEIG-GPU utci_process.py, commit
0d7fe742abeeddd890dd58fc76ed7f78bd47faec.
Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan.
Distributed under GNU GPL version 3 or (at your option) any later version.
"""
import datetime
from contextlib import ExitStack
from pathlib import Path
import re

import numpy as np
from osgeo import gdal, osr
import pytz
from timezonefinder import TimezoneFinder

from .models import (StaticScene, ForcingTimeline, GeometryCache, SimulationState,
                     Workspace, OutputPlan, STATE_NAMES)
from .comfort.utci import utci_calculator_uniform
from .comfort.wbgt import isobaric_wet_bulb_temperature_from_rh, black_globe_temperature
from .geometry.svf import svf_calculator_compact as svf_calculator
from .geometry.visibility import import_visibility_npz, LazyDiffVisibility
from .io.rasters import read_raster, write_single, StreamingOutputs
from .radiation.engine import Solweig_2022a_calc
from .radiation.materials import Tgmaps_v1
from .radiation.solar import Solweig_2015a_metdata_noload

RETURN_NAMES = ("Tmrt Kdown Kup Ldown Lup Tg ea esky I0 CI shadow firstdaytime timestepdec "
                "timeadd Tgmap1 Tgmap1E Tgmap1S Tgmap1W Tgmap1N Keast Ksouth Kwest Knorth "
                "Least Lsouth Lwest Lnorth KsideI TgOut1 TgOut radI radD Lside L_patches CI_Tg CI_TgG KsideD dRad Kside").split()
SVF_NAMES = ("svf svfaveg svfE svfEaveg svfEveg svfN svfNaveg svfNveg svfS svfSaveg svfSveg svfveg "
             "svfW svfWaveg svfWveg vegshmat vbshvegshmat shmat svftotal").split()


def files_by_key(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".tif", ".txt"}:
            continue
        match = re.search(r"_(\d+)_(\d+)", path.name)
        if match:
            result["_".join(match.groups())] = path
    return result


def directional_wind_files(directory, tile):
    result = {}
    if not Path(directory).is_dir():
        return result
    for path in sorted(Path(directory).glob("*.tif")):
        match = re.fullmatch(r"WindCoeff_dir(\d{3})_(-?\d+)_(-?\d+)\.tif", path.name)
        if match and "_".join(match.groups()[1:]) == tile:
            result[int(match.group(1)) % 360] = path
    return result


def load_svf(directory, tile):
    directory = Path(directory)
    values = {}
    for name in SVF_NAMES[:15]:
        path = f"/vsizip/{(directory / ('svfs_' + tile + '.zip')).resolve()}/{name}.tif"
        values[name], _ = read_raster(path)
    matrices = import_visibility_npz(directory / f"shadowmats_{tile}.npz")
    for name, saved in (("shmat", "shadowmat"), ("vegshmat", "vegshadowmat"), ("vbshvegshmat", "vbshmat")):
        values[name] = matrices[saved]
    values["svftotal"], _ = read_raster(directory / f"SkyViewFactor_{tile}.tif")
    return values


def _dem_median(dem):
    """Pinned Torch median: propagate any NaN and select the lower middle."""
    if np.isnan(dem).any():
        return float('nan')
    return float(np.partition(dem.ravel(), (dem.size - 1) // 2)[(dem.size - 1) // 2])


def run_tile(base_path, preprocess_dir, selected_date_str, tile, paths, flags, *, runtime=None):
    from .runtime import get_runtime_options, runtime_options, plan_admission

    options = runtime or get_runtime_options()
    job = dict(base_path=str(base_path), preprocess_dir=str(preprocess_dir),
               selected_date_str=selected_date_str, tile=tile, paths=paths, flags=flags)
    plan_admission([job], options)
    with runtime_options(options), ExitStack() as resources:
        return _run_tile(base_path, preprocess_dir, selected_date_str, tile, paths, flags,
                         options, resources)


def _run_tile(base_path, preprocess_dir, selected_date_str, tile, paths, flags, runtime, resources):
    """Execute one original logical tile, retaining no raster time history."""
    from .identities import InputGuard

    wind_files = directional_wind_files(Path(preprocess_dir) / 'WindCoeff', tile)
    guarded_paths = dict(paths, **{f'wind_{direction}': path for direction, path in wind_files.items()})
    input_guard = InputGuard(guarded_paths)
    output_plan = OutputPlan.from_flags(flags)
    a, metadata = read_raster(paths["Building_DSM"])
    trees, _ = read_raster(paths["Trees"])
    dem, _ = read_raster(paths["DEM"])
    walls, _ = read_raster(paths["walls"])
    aspects, _ = read_raster(paths["aspect"])
    met = np.loadtxt(paths["metfiles"], skiprows=1, delimiter=" ")
    if met.ndim != 2:
        raise ValueError("Own-met input must contain multiple rows, as required by the pinned public driver")
    rows, cols = a.shape
    scale = 1 / metadata.transform[1]
    old = osr.SpatialReference()
    old.ImportFromWkt(metadata.projection)
    old.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    new = osr.SpatialReference()
    new.ImportFromEPSG(4326)
    new.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(old, new)
    lon, lat = transform.TransformPoint(metadata.transform[0] + metadata.transform[1] * cols / 2,
                                        metadata.transform[3] + metadata.transform[5] * rows / 2)[:2]
    # Torch.median returns the lower middle element, not NumPy's even-length mean.
    altitude_m = _dem_median(dem)
    if altitude_m > 0:
        altitude_m = 3.0
    location = {"longitude": lon, "latitude": lat, "altitude": altitude_m}
    date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d")
    timezone = pytz.timezone(TimezoneFinder().timezone_at(lat=lat, lng=lon) or "UTC")
    utc = timezone.localize(date).utcoffset().total_seconds() / 3600
    _, altitude, azimuth, zen, jday, _, dectime, altmax = Solweig_2015a_metdata_noload(met, location, utc)

    trees[trees < 0] = 0
    vegdem, vegdem2 = trees + dem, trees * np.float32(.25) + dem
    bush = np.logical_not(vegdem2 * vegdem) * vegdem
    vegdsm, vegdsm2 = trees + a, trees * np.float32(.25) + a
    vegdsm[vegdsm == a] = 0
    vegdsm2[vegdsm2 == a] = 0
    amaxvalue = np.maximum(a.max(), vegdem.max())
    buildings = a - dem
    buildings[buildings < 2] = 1
    buildings[buildings >= 2] = 0
    valid_mask = buildings == 1
    zero = np.zeros((rows, cols), dtype=np.float32)
    landcover = int("Landcover" in paths)
    lcgrid = False
    if landcover:
        lcgrid, _ = read_raster(paths["Landcover"])
        lcgrid[(lcgrid < 1) | (lcgrid > 7)] = 6
        lcgrid[(lcgrid == 3) | (lcgrid == 4)] = 5
        table = np.loadtxt(Path(__file__).parent / "data/landcoverclasses_2016a.txt", skiprows=1, usecols=range(1, 7))
        materials = Tgmaps_v1(lcgrid, table)
        TgK, Tstart, alb_grid, emis_grid = [np.asarray(v, dtype=np.float32) for v in materials[:4]]
        TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall = [np.asarray(v, dtype=np.float32) for v in materials[4:]]
    else:
        TgK, Tstart = zero + .37, zero - 3.41
        alb_grid, emis_grid = zero + .15, zero + .95
        TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall = .37, -3.41, 15., 15.

    scene = StaticScene(a, trees, dem, buildings, vegdsm, vegdsm2, bush, amaxvalue,
                        valid_mask, metadata, scale, landcover, lcgrid)

    svf_dir = Path(preprocess_dir) / "SVF"
    cache_available = all((svf_dir / name).is_file() for name in
                          (f"SkyViewFactor_{tile}.tif", f"svfs_{tile}.zip", f"shadowmats_{tile}.npz"))
    from .cache import GeometryStore, load_legacy_geometry, content_fingerprint
    from .identities import simulation_identity
    from .geometry.svf import save_svf_zip_npz_outputs
    from .geometry.shadows import create_patches
    from .geometry.recipe import numerical_geometry_recipe, guarded_producer

    # One shared numerical recipe with the standalone route: same identity,
    # same native key, one production per logical tile when caching is on.
    recipe = numerical_geometry_recipe(paths, 2)
    geometry_key = recipe.identity
    input_guard.check()
    geometry = None
    legacy_guard = None
    if cache_available and runtime.legacy_cache_policy == 'trust':
        legacy_paths = [svf_dir / name for name in
                        (f'svfs_{tile}.zip', f'shadowmats_{tile}.npz', f'SkyViewFactor_{tile}.tif')]
        legacy_guard = InputGuard({path.name: path for path in legacy_paths})
        geometry = load_legacy_geometry(*legacy_paths, trust=True, shape=a.shape,
            patch_count=int(np.sum(create_patches(2)[4])), geotransform=metadata.transform,
            projection=metadata.projection)
        legacy_guard.check()
        geometry_key = dict(geometry_key, trusted_legacy=legacy_guard.fingerprints)

    def produce_geometry():
        return guarded_producer(recipe, input_guard.check)()

    if geometry is None:
        if runtime.cache_enabled:
            root = runtime.cache_dir or str(Path(preprocess_dir) / '.solweig-light/cache')
            handle = resources.enter_context(GeometryStore(root).get_or_create(geometry_key, produce_geometry))
            geometry = handle.fields
        else:
            geometry = produce_geometry()
    svf, svfveg = geometry["svf"], geometry["svfveg"]
    svfbuveg = svf - (1 - svfveg) * np.float32(1 - .03)
    from .radiation._math_profile import asvf as prepare_asvf
    asvf = prepare_asvf(svf)
    diffsh = LazyDiffVisibility(geometry["shmat"], geometry["vegshmat"])
    tmp = np.maximum(svf + svfveg - 1, np.float32(0))
    with np.errstate(divide="ignore"):
        svfalfa = np.arcsin(np.exp(np.log(1 - tmp) / 2))

    geometry_cache = GeometryCache(walls, aspects, geometry, svfbuveg, asvf, diffsh, svfalfa, cache_available)

    forcing = {name: met[:, index] for name, index in (("Ta", 11), ("RH", 10), ("radG", 14), ("radD", 21), ("radI", 22), ("P", 12))}
    wind = met[:, 9]
    wind_direction = met[:, 23] if met.shape[1] > 23 else np.full(len(met), -999.)
    uhi = met[:, 24] if met.shape[1] > 24 else np.zeros(len(met))
    psi = np.where((met[:, 1] > 97) & (met[:, 1] < 300), np.float32(.03), np.float32(.5))
    wetbulb = None
    if output_plan.needs_wetbulb:
        wetbulb = isobaric_wet_bulb_temperature_from_rh(p=forcing["P"] * 1000,
            T=forcing["Ta"] + uhi + 273.15, rh=forcing["RH"], phase="liquid", method="Romps", limit=True)
    timeline = ForcingTimeline(met, forcing, wind, wind_direction, uhi, psi, altitude, azimuth,
                               zen, jday, dectime, altmax, location, wetbulb)
    coefficients = {}
    if wind_files:
        missing = sorted(set(range(0, 360, 30)) - set(wind_files))
        if missing:
            raise FileNotFoundError(f"Tile {tile} missing wind directions: {missing}")
        for direction, path in wind_files.items():
            coefficients[direction], _ = read_raster(path)
            if coefficients[direction].shape != a.shape:
                raise ValueError("Wind coefficient shape differs from DSM")
    ones = np.ones_like(a)
    workspace = Workspace(zero, ones)
    state = SimulationState.initial(workspace.zero, timeline.dectime[1] - timeline.dectime[0])
    args = dict(dsm=scene.dsm, scale=scene.scale, rows=rows, cols=cols, vegdem=scene.vegdsm, vegdem2=scene.vegdsm2,
        albedo_b=.2, absK=.7, absL=.95, ewall=.9, Fside=.22, Fup=.06, Fcyl=.28,
        usevegdem=1, onlyglobal=1, buildings=scene.buildings, location=timeline.location, landcover=scene.landcover,
        lc_grid=scene.lcgrid, dirwalls=geometry_cache.aspects, walls=geometry_cache.walls, cyl=True, elvis=0, amaxvalue=scene.amaxvalue,
        bush=scene.bush, TgK=TgK, Tstart=Tstart, alb_grid=alb_grid, emis_grid=emis_grid,
        TgK_wall=TgK_wall, Tstart_wall=Tstart_wall, TmaxLST=TmaxLST, TmaxLST_wall=TmaxLST_wall,
        first=np.round(np.float32(1.1)), second=np.round(np.float32(1.1) * np.float32(20)),
        svfalfa=geometry_cache.svfalfa, svfbuveg=geometry_cache.svfbuveg,
        diffsh=geometry_cache.diffsh, anisotropic_sky=1, asvf=geometry_cache.asvf, patch_option=2)
    args.update({name: value for name, value in geometry_cache.svfs.items() if name != "svftotal"})
    output_dir = Path(base_path) / "output_folder" / tile
    from .persistence import TransactionalOutputs
    identity = simulation_identity(paths, wind_files, selected_date_str, tile, flags, location, utc)
    identity['geometry'] = geometry_key
    input_guard.check()
    if legacy_guard is not None:
        legacy_guard.check()
    transaction_dir = Path(base_path) / '.solweig-light/transactions'
    with TransactionalOutputs(output_dir, tile, scene.metadata, timeline.met, selected_date_str,
            output_plan.requested, transaction_dir=transaction_dir, identity=identity,
            resume=runtime.resume) as writer:
        restored = writer.restore()
        start = 0
        if restored is not None:
            start, state = restored
        # Private demand profiles: the pipeline consumes only the primary
        # cylinder-longwave outputs (cardinal diagnostics are NOT_REQUESTED,
        # never zeros) and the admitted cylinder-anisotropic shortwave
        # scratch profile. Both restore on exit; public API is unchanged
        # (D09: never surfaced through RuntimeOptions or public signatures).
        from .radiation import cylinder_longwave as _cyl_lw
        from .radiation import cylinder_shortwave as _cyl_sw
        assert args['cyl'] and args['anisotropic_sky'] == 1, \
            'reduced private demand profiles require the standard cylinder-anisotropic workflow'
        _previous_lw_demand = _cyl_lw.set_demand(_cyl_lw.CylinderLongwaveDemand.PIPELINE_CYLINDERS_ANISOTROPIC)
        # cylinder_shortwave.set_demand_profile deliberately returns None
        # (set-only API); capture the ambient profile before switching.
        _previous_sw_profile = _cyl_sw.demand_profile()
        _cyl_sw.set_demand_profile(_cyl_sw.PIPELINE_CYLINDER_ANISOTROPIC)
        try:
            for i in range(start, len(timeline.met)):
                if scene.landcover and (i == 0 or timeline.dectime[i] % 1 == 0):
                    state.Twater = np.mean(timeline.meteorology["Ta"][timeline.jday[0] == np.floor(timeline.dectime[i])])
                if timeline.dectime[i] % 1 == 0:
                    state.CI = 1.0  # Original np.where tuple-length branch always selects this for 1D dectime.
                dynamic = timeline.at(i)
                # C6-20: the anisotropic Lside fast path is opted into per
                # thread by the private demand context (recipe: driver-side
                # placement); non-admitted inputs still fall back internally.
                from .radiation.pipeline_demand import RadiationDemand, radiation_demand
                with radiation_demand(RadiationDemand.PIPELINE_CYLINDER_ANISOTROPIC):
                    result = Solweig_2022a_calc(i=i, **args, **state.engine_arguments(), **dynamic, altitude=timeline.altitude[0, i],
                        azimuth=timeline.azimuth[0, i], zen=timeline.zen[0, i], jday=timeline.jday[0, i],
                        psi=timeline.psi[i], dectime=timeline.dectime[i], altmax=timeline.altmax[0, i], Twater=state.Twater)
                fields = dict(zip(RETURN_NAMES, result))
                state.accept(fields)
                direction = timeline.wind_direction[i]
                direction = int(np.floor(((direction % 360) + 15) / 30) * 30) % 360 if np.isfinite(direction) and direction >= 0 else None
                speed = np.maximum(coefficients.get(direction, workspace.ones) * np.float32(timeline.wind[i]), np.float32(.15))
                temperature = (workspace.zero + np.float32(timeline.meteorology["Ta"][i])) + np.float32(timeline.uhi[i])
                tmrt = workspace.zero + fields["Tmrt"]
                utci = utci_calculator_uniform(temperature[0, 0], np.float32(timeline.meteorology["RH"][i]), tmrt, speed)
                utci[~scene.valid_mask] = np.nan
                output = {"UTCI": utci, "TMRT": fields["Tmrt"], "Kup": fields["Kup"], "Kdown": fields["Kdown"],
                          "Lup": fields["Lup"], "Ldown": fields["Ldown"], "Shadow": fields["shadow"], "Ta": temperature, "Wind": speed}
                if timeline.wetbulb is not None:
                    hcg = (6.3 / .46821) * np.power(speed, .6)
                    globe = black_globe_temperature(hcg, tmrt, temperature, emissivity=.95)
                    sun = np.float32(.7 * timeline.wetbulb[i]) + .3 * globe
                    shade = np.float32(.7 * timeline.wetbulb[i]) + .2 * globe + .1 * temperature
                    output["WBGT"] = np.where(fields["shadow"] < .1, sun, shade)
                writer.write(i, output)
                if (i + 1) % runtime.checkpoint_interval == 0 or i + 1 == len(timeline.met):
                    writer.checkpoint(i + 1, state)
                # State owns its carried maps. Release completed diagnostic and
                # comfort rasters before the next kernel allocates their successors.
                del result, fields, output, tmrt, utci, speed, temperature
        finally:
            _cyl_sw.set_demand_profile(_previous_sw_profile)
            _cyl_lw.set_demand(_previous_lw_demand)
            # N9 default route: the bounded region owner executes the
            # stream whenever a tile took it (threads>1). Release its
            # threads between tiles; a no-op for tiles that never
            # dispatched (the machinery is vendored under the package and
            # lazily imported).
            try:
                from solweig_light._native_dispatch.region import \
                    shutdown_all_pools as _shutdown_pools
            except ImportError:
                pass
            else:
                _shutdown_pools()
        extra = {}
        input_guard.check()
        publishing = writer.publication_path.exists() or writer.completion_path.exists()
        if not publishing and not cache_available:
            template = gdal.Open(str(paths['Building_DSM']))
            save_svf_zip_npz_outputs(str(writer.staging_directory), template, **geometry, number=tile)
            template = None
            for name in (f'svfs_{tile}.zip', f'shadowmats_{tile}.npz', f'SkyViewFactor_{tile}.tif'):
                extra[svf_dir / name] = writer.staging_directory / name
        if not publishing and output_plan.save_svf and not geometry_cache.available_on_disk:
            staged = writer.staging_directory / f'SVF_{tile}.tif'
            write_single(staged, geometry_cache.svfs['svftotal'], scene.metadata)
            extra[output_dir / f'SVF_{tile}.tif'] = staged
        input_guard.check()
        if legacy_guard is not None:
            legacy_guard.check()
        writer.complete(extra_artifacts=extra)
