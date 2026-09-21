import time
from solweig_light import RuntimeOptions, runtime_options, thermal_comfort

SCENE = "/Users/sunghosynn/Documents/Code/SolweigLight/solweig_scene_small"

t0 = time.perf_counter()
with runtime_options(RuntimeOptions(
    memory_budget_bytes=6 * 1024**3,
    cpu_budget=4,
    workers=1,
    threads_per_worker=4,
    block_pixels=1024,
)):
    thermal_comfort(
        base_path=SCENE,
        selected_date_str="2020-08-13",
        landcover_filename="Landcover.tif",
        own_met_file=f"{SCENE}/ownmet_Forcing_data.txt",
        ERA_5_z0_find=False,
        use_uhi=False,
        tile_size=1024,
        overlap=0,
        save_tmrt=True,
    )
elapsed = time.perf_counter() - t0
print(f"[DONE] Duration: {elapsed/60:.1f}mins ({elapsed:.0f}seconds)")