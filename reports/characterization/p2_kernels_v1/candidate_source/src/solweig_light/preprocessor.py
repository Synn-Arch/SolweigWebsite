#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.


"""Native own-met TIFF tiling; optional acquisition is deferred explicitly."""
import os
import re
import glob
import shutil
from typing import Optional

def check_rasters(files):
    """
    Check that all provided raster files have matching dimensions, pixel size, and CRS.

    Parameters:
        files (list): List of raster file paths.

    Returns:
        bool: True if all checks pass, raises ValueError/FileNotFoundError otherwise.
    """
    from osgeo import gdal
    gdal.UseExceptions()
    if not files:
        raise ValueError("No raster files provided.")

    ref_file = files[0]
    ds = gdal.Open(ref_file)
    if ds is None:
        raise FileNotFoundError(f"Could not open {ref_file}")
    ref_width = ds.RasterXSize
    ref_height = ds.RasterYSize
    ref_gt = ds.GetGeoTransform()  # (originX, pixelWidth, rot, originY, rot, pixelHeight)
    ref_pixel_width = ref_gt[1]
    ref_pixel_height = ref_gt[5]  # typically negative
    ref_crs = ds.GetProjection()
    ds = None

    for f in files[1:]:
        ds = gdal.Open(f)
        if ds is None:
            raise FileNotFoundError(f"Could not open {f}")
        if ds.RasterXSize != ref_width or ds.RasterYSize != ref_height:
            raise ValueError("Error: Raster dimensions do not match.")
        gt = ds.GetGeoTransform()
        pixel_width = gt[1]
        pixel_height = gt[5]
        if pixel_width != ref_pixel_width or pixel_height != ref_pixel_height:
            raise ValueError("Error: Pixel sizes do not match.")
        if ds.GetProjection() != ref_crs:
            raise ValueError("Error: CRS does not match.")
        ds = None

    return True

def _resolve_path(base_path, path_or_name):
    if path_or_name is None:
        return None
    if os.path.isabs(path_or_name):
        return path_or_name
    return os.path.join(base_path, path_or_name)

def find_windcoeff_files(base_path, windcoeff_filename):
    """
    windcoeff_filename can be:
      - None: no wind coefficient
      - folder path: read WindCoeff_dir*.tif inside it
      - single file path: old behavior / fallback
      - glob pattern: e.g. 'WindCoeff_dir*.tif'
    """
    if windcoeff_filename is None:
        return []

    wind_path = _resolve_path(base_path, windcoeff_filename)

    if os.path.isdir(wind_path):
        files = sorted(glob.glob(os.path.join(wind_path, "WindCoeff_dir*.tif")))
    elif any(ch in wind_path for ch in ["*", "?", "["]):
        files = sorted(glob.glob(wind_path))
    elif os.path.isfile(wind_path):
        files = [wind_path]
    else:
        files = []

    if not files:
        raise FileNotFoundError(f"No wind coefficient files found from: {wind_path}")

    dir_files = [
        f for f in files
        if re.search(r"WindCoeff_dir\d{3}\.tif$", os.path.basename(f))
    ]

    if dir_files:
        files = sorted(dir_files)

        found_dirs = []
        for f in files:
            m = re.search(r"WindCoeff_dir(\d{3})\.tif$", os.path.basename(f))
            if m:
                found_dirs.append(int(m.group(1)))

        expected_dirs = list(range(0, 360, 30))
        missing = sorted(set(expected_dirs) - set(found_dirs))

        if missing:
            raise FileNotFoundError(
                "Missing directional wind coefficient rasters: "
                + ", ".join(f"WindCoeff_dir{d:03d}.tif" for d in missing)
            )

    return files

def create_tiles_to_folder(infile, tilesize, overlap, out_folder, tile_prefix, clear_folder=False):
    """
    Tile a raster into a specified folder using a specified filename prefix.

    Example output:
      out_folder/WindCoeff_dir030_0_0.tif
    """
    from osgeo import gdal
    gdal.UseExceptions()
    if overlap < 0 or overlap >= tilesize:
        raise ValueError("overlap must be 0 ≤ overlap < tilesize")

    ds = gdal.Open(infile)
    if ds is None:
        raise FileNotFoundError(f"Could not open {infile}")

    width = ds.RasterXSize
    height = ds.RasterYSize

    if clear_folder and os.path.exists(out_folder):
        shutil.rmtree(out_folder)

    os.makedirs(out_folder, exist_ok=True)

    if tilesize >= width and tilesize >= height:
        outfile = os.path.join(out_folder, f"{tile_prefix}_0_0.tif")
        options = gdal.TranslateOptions(format="GTiff", srcWin=[0, 0, width, height])
        gdal.Translate(outfile, ds, options=options)
        print(f"Created single tile: {outfile}")
        ds = None
        return

    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            tile_width = min(tilesize + overlap, width - i)
            tile_height = min(tilesize + overlap, height - j)

            outfile = os.path.join(out_folder, f"{tile_prefix}_{i}_{j}.tif")
            options = gdal.TranslateOptions(
                format="GTiff",
                srcWin=[i, j, tile_width, tile_height]
            )
            gdal.Translate(outfile, ds, options=options)
            print(f"Created tile: {outfile}")

    ds = None

def create_windcoeff_tiles(windcoeff_files, tilesize, overlap, preprocess_dir):
    """
    Tile all directional wind coefficient rasters into one folder:

      preprocess_dir/WindCoeff/WindCoeff_dir000_i_j.tif
      preprocess_dir/WindCoeff/WindCoeff_dir030_i_j.tif
      ...
    """
    if not windcoeff_files:
        print("No wind coefficient rasters provided. Skipping wind coefficient tiling.")
        return

    out_folder = os.path.join(preprocess_dir, "WindCoeff")

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)

    for wind_fp in windcoeff_files:
        tile_prefix = os.path.splitext(os.path.basename(wind_fp))[0]

        print(f"Creating wind coefficient tiles for {tile_prefix}...")
        create_tiles_to_folder(
            infile=wind_fp,
            tilesize=tilesize,
            overlap=overlap,
            out_folder=out_folder,
            tile_prefix=tile_prefix,
            clear_folder=False
        )

def create_tiles(infile, tilesize, overlap, tile_type, preprocess_dir):
    """
    Tile a raster file into smaller chunks.

    Normal rasters are written as:
      preprocess_dir/DEM/DEM_i_j.tif
      preprocess_dir/Trees/Trees_i_j.tif
      etc.
    """
    out_folder = os.path.join(preprocess_dir, tile_type)

    create_tiles_to_folder(
        infile=infile,
        tilesize=tilesize,
        overlap=overlap,
        out_folder=out_folder,
        tile_prefix=tile_type,
        clear_folder=True
    )

def create_met_files(base_path, source_met_file, preprocess_dir):
    """
    Copy a given met file to multiple outputs based on the raster tile filenames.

    Parameters:
        base_path (str): Base directory containing input rasters.
        source_met_file (str): Path to user-provided met file.
        preprocess_dir (str): Directory for preprocessing outputs (pre_processing_outputs).
    """
    raster_folder = os.path.join(preprocess_dir, 'Building_DSM')
    target_folder = os.path.join(preprocess_dir, 'metfiles')

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    else:
        shutil.rmtree(target_folder)
        os.makedirs(target_folder)
    
    for file in os.listdir(raster_folder):
        if file.lower().endswith('.tif'):
            name_without_ext = os.path.splitext(file)[0]
            prefix = 'Building_DSM_'
            if name_without_ext.startswith(prefix):
                digits = name_without_ext[len(prefix):]
                new_filename = f'metfile_{digits}.txt'
                target_met_file = os.path.join(target_folder, new_filename)
                shutil.copy(source_met_file, target_met_file)
                print(f"Copied to {target_met_file}")

def ppr(base_path, building_dsm_filename, dem_filename, trees_filename,
         landcover_filename, windcoeff_filename,
         tile_size, overlap, selected_date_str, use_own_met,
         start_time=None, end_time=None, data_source_type=None, data_folder=None,
         own_met_file=None, preprocess_dir=None, use_uhi=True):
    """
    Preprocessing routine to validate raster files, generate tiles, and prepare metfiles for SOLWEIG.

    Parameters:
        base_path (str): Base working directory containing input rasters.
        building_dsm_filename (str): Filename of building DSM raster.
        dem_filename (str): Filename of DEM raster.
        trees_filename (str): Filename of trees raster.
        landcover_filename (str): Filename of landcover raster or None.
        windcoeff_filename (str): Filename of wind coefficient raster or None.
        tile_size (int): Tile size in pixels.
        overlap (int): Overlap between tiles in pixels.
        selected_date_str (str): Selected date (YYYY-MM-DD).
        use_own_met (bool): Whether to use a user-provided met file.
        start_time (str): Start datetime (required if not using own met file).
        end_time (str): End datetime (required if not using own met file).
        data_source_type (str): Either 'ERA5' or 'wrfout'.
        data_folder (str): Folder containing input NetCDF files.
        own_met_file (str): Path to user-provided met file (used if use_own_met is True).
        preprocess_dir (str): Directory for preprocessing outputs.
        use_uhi (bool): Whether to use UHI-aware ERA5 preprocessing.
    """
    if not use_own_met:
        raise NotImplementedError("ERA5/WRF acquisition and meteorological processing are deferred to P5")

    if preprocess_dir is None:
        preprocess_dir = os.path.join(base_path, "processed_inputs")
    os.makedirs(preprocess_dir, exist_ok=True)

    building_dsm_path = os.path.join(base_path, building_dsm_filename)
    dem_path = os.path.join(base_path, dem_filename)
    trees_path = os.path.join(base_path, trees_filename)

    landcover_path = None
    landcover_path = None
    windcoeff_files = []

    if landcover_filename is not None:
        landcover_path = _resolve_path(base_path, landcover_filename)

    windcoeff_files = find_windcoeff_files(base_path, windcoeff_filename)

    # Check that all rasters have matching dimensions, pixel size, and CRS.
    try:
        raster_list = [building_dsm_path, dem_path, trees_path]

        if landcover_path is not None:
            raster_list.append(landcover_path)

        # Check all directional wind coefficient rasters too
        if windcoeff_files:
            raster_list.extend(windcoeff_files)

        check_rasters(raster_list)

    except ValueError as error:
        print(error)
        exit(1)

    rasters = {
        "Building_DSM": building_dsm_path,
        "DEM": dem_path,
        "Trees": trees_path
    }

    if landcover_path is not None:
        rasters["Landcover"] = landcover_path

    for tile_type, raster in rasters.items():
        print(f"Creating tiles for {tile_type}...")
        create_tiles(raster, tile_size, overlap, tile_type, preprocess_dir)

    # Directional wind coefficient rasters are handled separately
    # so the 12 files do not overwrite each other.
    if windcoeff_files:
        create_windcoeff_tiles(windcoeff_files, tile_size, overlap, preprocess_dir)
    else:
        print("Wind coefficient not used; skipping WindCoeff tiles.")

    # For metfiles processing, we use the DEM tiles folder.
    dem_tiles_folder = os.path.join(preprocess_dir, "DEM")

    # Choose between own met file or processed NetCDF file.
    if use_own_met:
        if own_met_file is None:
            print("Error: Please provide the path to your own met file.")
            exit(1)
        create_met_files(base_path, own_met_file, preprocess_dir)
    else:
        raise NotImplementedError("ERA5/WRF acquisition and meteorological processing are deferred to P5")

def preprocess(
    base_path: str,
    selected_date_str: str,
    building_dsm_filename: str = 'Building_DSM.tif',
    dem_filename: str = 'DEM.tif',
    trees_filename: str = 'Trees.tif',
    landcover_filename: Optional[str] = None,
    windcoeff_folder: Optional[str] = None,
    tile_size: int = 3600,
    overlap: int = 20,
    use_own_met: bool = True,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    data_source_type: Optional[str] = None,
    data_folder: Optional[str] = None,
    own_met_file: Optional[str] = None,
    preprocess_dir: Optional[str] = None,
    use_uhi: bool = True,
) -> str:
    """
    Run preprocessing only: validate rasters, create tiles, and prepare metfiles.

    Use this when you want to run preprocessing once and then call
    :func:`run_walls_aspect` and :func:`run_utci_tiles` separately.

    Args:
        base_path:
            Base directory; used to resolve relative raster paths.

        selected_date_str:
            Simulation date 'YYYY-MM-DD'.

        building_dsm_filename, dem_filename, trees_filename, landcover_filename:
            Raster paths or filenames. Relative paths are resolved against base_path.

        windcoeff_folder:
            Wind coefficient input. Can be:
              - None: do not use wind coefficients
              - folder path containing WindCoeff_dir*.tif
              - glob pattern such as "WindCoeff_dir*.tif"
              - single legacy wind coefficient raster

            Relative paths are resolved against base_path.

            For directional wind coefficients, the expected files are:
              WindCoeff_dir000.tif
              WindCoeff_dir030.tif
              ...
              WindCoeff_dir330.tif

        tile_size:
            Tile size in pixels.

        overlap:
            Overlap between tiles in pixels.

        use_own_met:
            If True, use own_met_file; else use ERA5/WRF.

        start_time, end_time:
            Required for ERA5/WRF, in UTC format 'YYYY-MM-DD HH:MM:SS'.

        data_source_type:
            'ERA5' or 'wrfout' when use_own_met is False.

        data_folder:
            Folder with ERA5/WRF NetCDF files when use_own_met is False.

        own_met_file:
            Path to custom met file when use_own_met is True.

        preprocess_dir:
            Directory for preprocessing outputs. Defaults to
            '{base_path}/processed_inputs'.

        use_uhi:
            If True, use UHI-aware ERA5 processing and write UHI_CYCLE/uhii
            into generated metfiles when available. If False, use standard ERA5
            processing and write uhii = 0.0.

    Returns:
        The path to the preprocessing directory.
    """
    import os

    if preprocess_dir is None:
        preprocess_dir = os.path.join(base_path, "processed_inputs")

    os.makedirs(preprocess_dir, exist_ok=True)

    ppr(
        base_path,
        building_dsm_filename,
        dem_filename,
        trees_filename,
        landcover_filename,
        windcoeff_folder,
        tile_size,
        overlap,
        selected_date_str,
        use_own_met,
        start_time,
        end_time,
        data_source_type,
        data_folder,
        own_met_file,
        preprocess_dir=preprocess_dir,
        use_uhi=use_uhi,
    )

    return preprocess_dir
