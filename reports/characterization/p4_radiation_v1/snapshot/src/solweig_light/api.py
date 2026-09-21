"""Public CPU workflows matching the pinned local TIFF/own-met contract.

ERA5/WRF acquisition and wind generation remain deferred to P5.
Derived from SOLWEIG-GPU solweig_gpu.py, commit
0d7fe742abeeddd890dd58fc76ed7f78bd47faec; GPL-3.0-or-later.
"""
import os
from pathlib import Path
import re
from typing import Optional, List

from .preprocessor import preprocess


def run_walls_aspect(preprocess_dir: str) -> None:
    """Write wall height and aspect for every Building_DSM TIFF, serially."""
    import numpy as np
    from .geometry.walls import findwalls, filter1Goodwin_as_aspect_v3
    from .io.rasters import read_raster, write_single

    directory = Path(preprocess_dir)
    for subdir in ('walls', 'aspect'):
        (directory / subdir).mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(directory / 'Building_DSM'):
        if not filename.endswith('.tif') or filename.startswith('.'):
            continue
        try:
            dsm, metadata = read_raster(directory / 'Building_DSM' / filename)
            if np.all(np.isnan(dsm)):
                print(f'Skipping {filename}, invalid DEM.')
                continue
            walls = findwalls(dsm, 3.0)
            aspects = filter1Goodwin_as_aspect_v3(walls, 1 / metadata.transform[1], dsm)
            tile = filename[13:-4]
            write_single(directory / 'walls' / f'walls_{tile}.tif', walls, metadata)
            write_single(directory / 'aspect' / f'aspect_{tile}.tif', aspects, metadata)
        except Exception as error:
            # The original wall worker reports bad tiles and continues.
            print(f'Error processing {filename}: {error}')


def calculate_svf(base_path: str, patch_option: int = 2, overwrite: bool = False) -> None:
    """Write original standalone SVF TIFF, ZIP and NPZ artifacts per tile."""
    from .geometry.svf import svf_calculator_compact as svf_calculator

    directory = Path(base_path)
    output = directory / 'SVF'
    output.mkdir(parents=True, exist_ok=True)
    folders = [directory / name for name in ['Building_DSM', 'DEM', 'Trees']]
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f'Required directory not found: {folder}')

    def map_tiles(folder, prefix):
        result = {}
        for filename in os.listdir(folder):
            if filename.startswith('.') or not filename.lower().endswith('.tif'):
                continue
            match = re.match(rf'^{re.escape(prefix)}_(.+)\.tif$', filename)
            if match:
                result[match.group(1)] = str(folder / filename)
        return result

    maps = [map_tiles(folder, name) for folder, name in zip(folders, ['Building_DSM', 'DEM', 'Trees'])]
    building, dem, trees = maps
    common = sorted(set(building) & set(dem) & set(trees))
    if not common:
        raise RuntimeError('No matching SVF tiles found across Building_DSM/, DEM/, and Trees/.')
    for mapping, name in [(dem, 'DEM'), (trees, 'Trees')]:
        missing = set(building) - set(mapping)
        if missing:
            print(f'[WARNING] {len(missing)} Building_DSM tiles are missing matching {name} tiles.')
    print(f'[INFO] Found {len(common)} matching SVF tiles.')
    print(f'[INFO] Writing outputs to: {output}')
    for tile in common:
        expected = [output / name for name in [f'SkyViewFactor_{tile}.tif', f'svfs_{tile}.zip', f'shadowmats_{tile}.npz']]
        if not overwrite and all(path.exists() for path in expected):
            continue
        svf_calculator(patch_option=patch_option, save_rasters=True, building_dsm_path=building[tile],
                       tree_path=trees[tile], dem_path=dem[tile], output_dir=str(output), number=tile)
    print('[INFO] Standalone SVF calculation complete.')


def run_utci_tiles(
    base_path: str,
    preprocess_dir: str,
    selected_date_str: str,
    tile_keys: Optional[List[str]] = None,
    save_tmrt: bool = True,
    save_svf: bool = False,
    save_kup: bool = False,
    save_kdown: bool = False,
    save_lup: bool = False,
    save_ldown: bool = False,
    save_shadow: bool = False,
    save_wbgt: bool = False,
    save_ta: bool = False,
    save_wind: bool = False,
) -> None:
    """Run matched logical tiles chronologically with bounded output storage."""
    from .pipeline import files_by_key, run_tile

    directory = Path(preprocess_dir)
    def matching_files(folder, extension):
        # Directory maps are extension-specific upstream. Recover a matching
        # file if an unrelated extension replaced its key in the shared mapper.
        mapping = {key: path for key, path in files_by_key(folder).items() if path.suffix == extension}
        for path in sorted(folder.iterdir()):
            if path.suffix != extension:
                continue
            match = re.search(r'_(\d+)_(\d+)', path.name)
            if match:
                mapping['_'.join(match.groups())] = path
        return mapping

    required = ['Building_DSM', 'Trees', 'DEM', 'metfiles', 'walls', 'aspect']
    maps = {name: matching_files(directory / name, '.txt' if name == 'metfiles' else '.tif')
            for name in required}
    landcover = matching_files(directory / 'Landcover', '.tif') if (directory / 'Landcover').is_dir() else {}
    common = set.intersection(*(set(mapping) for mapping in maps.values()))
    if landcover:
        common &= set(landcover)
    if tile_keys is not None:
        common &= set(tile_keys)
        if not common:
            raise ValueError(f'No tiles to run; tile_keys={tile_keys} not found in preprocess_dir')
    flags = dict(save_tmrt=save_tmrt, save_svf=save_svf, save_kup=save_kup, save_kdown=save_kdown,
                 save_lup=save_lup, save_ldown=save_ldown, save_shadow=save_shadow, save_wbgt=save_wbgt,
                 save_ta=save_ta, save_wind=save_wind)
    print('Running Solweig ...')
    for tile in sorted(common, key=lambda key: tuple(int(value) for value in key.split('_'))):
        paths = {name: mapping[tile] for name, mapping in maps.items()}
        if landcover:
            paths['Landcover'] = landcover[tile]
        run_tile(base_path, preprocess_dir, selected_date_str, tile, paths, flags)


def thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename: Optional[str] = None,
    ERA_5_z0_find=True,
    tile_size=3600,
    overlap=20,
    use_own_met=True,
    start_time=None,
    end_time=None,
    data_source_type=None,
    data_folder=None,
    own_met_file=None,
    use_uhi=True,
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    save_wbgt=False,
    save_ta=False,
    save_wind=False,
):
    """Execute the full local own-met TIFF workflow on CPU.

    Remote acquisition and roughness-driven wind generation are deferred to P5.
    Own-met UHI column 24 is preserved independently of use_uhi, as upstream.
    """
    windcoeff_folder = None
    if ERA_5_z0_find:
        try:
            roughness = Path(data_folder) / 'data_stream-oper_stepType-instant.nc' if data_folder is not None else None
            roughness_available = roughness is not None and roughness.is_file()
        except (TypeError, ValueError, OSError):
            roughness_available = False
        if roughness_available:
            raise NotImplementedError('ERA5 roughness-driven wind generation is deferred to P5; use preprocess with existing wind coefficient rasters')
        print('Could not find ERA-5 file with roughness length')
    preprocess_dir = preprocess(base_path=base_path, selected_date_str=selected_date_str,
        building_dsm_filename=building_dsm_filename, dem_filename=dem_filename, trees_filename=trees_filename,
        landcover_filename=landcover_filename, windcoeff_folder=windcoeff_folder, tile_size=tile_size, overlap=overlap,
        use_own_met=use_own_met, start_time=start_time, end_time=end_time, data_source_type=data_source_type,
        data_folder=data_folder, own_met_file=own_met_file, use_uhi=use_uhi)
    run_walls_aspect(preprocess_dir)
    calculate_svf(preprocess_dir, patch_option=2, overwrite=False)
    run_utci_tiles(base_path=base_path, preprocess_dir=preprocess_dir, selected_date_str=selected_date_str,
        tile_keys=None, save_tmrt=save_tmrt, save_svf=save_svf, save_kup=save_kup, save_kdown=save_kdown,
        save_lup=save_lup, save_ldown=save_ldown, save_shadow=save_shadow, save_wbgt=save_wbgt,
        save_ta=save_ta, save_wind=save_wind)
