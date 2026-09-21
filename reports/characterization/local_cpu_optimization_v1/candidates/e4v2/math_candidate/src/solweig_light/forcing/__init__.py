"""Optional local meteorological adapters (dependencies loaded on execution)."""
from .local import (process_era5_data, process_era5_data_uhi, process_wrfout_data,
                    process_metfiles, compute_uhi_cycle_from_arrays,
                    infer_timezone_from_grid, utc_times_to_local_naive,
                    extract_datetime_strict)
