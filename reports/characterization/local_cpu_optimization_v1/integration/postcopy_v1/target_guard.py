"""Same-process guard for the post-copy installed wheel tests."""
from pathlib import Path
import json
import os
import platform
import sys


def pytest_configure(config):
    import numba
    import solweig_light

    target = Path(os.environ["SOLWEIG_POSTCOPY_SITE_TARGET"]).resolve()
    package = Path(solweig_light.__file__).resolve()
    if target not in package.parents:
        raise RuntimeError(f"package escaped explicit post-copy target: {package}")
    if "torch" in sys.modules:
        raise RuntimeError("torch imported before post-copy tests")
    modules = {}
    for name in (
        "solweig_light.api", "solweig_light.pipeline", "solweig_light.runtime",
        "solweig_light.geometry.visibility_compiled",
        "solweig_light.radiation.engine", "solweig_light.radiation.ground_view",
        "solweig_light.radiation._jit_cache", "solweig_light.radiation._math_profile",
    ):
        module = __import__(name, fromlist=["*"])
        origin = Path(module.__file__).resolve()
        if target not in origin.parents:
            raise RuntimeError(f"module escaped explicit post-copy target: {name}={origin}")
        modules[name] = str(origin)
    record = {
        "schema": "local-cpu-optimization-postcopy-target-guard.v1",
        "target": str(target),
        "package_origin": str(package),
        "module_origins": modules,
        "torch_imported_at_configure": "torch" in sys.modules,
        "numba_config_threads": int(numba.config.NUMBA_NUM_THREADS),
        "numba_effective_threads_at_configure": int(numba.get_num_threads()),
        "requested_thread_values_exercised_by_test": [1, 4, 10],
        "environment": {
            name: os.environ.get(name)
            for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_CACHE_DIR",
                         "TMPDIR")
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    path = Path(os.environ["SOLWEIG_POSTCOPY_GUARD_RECORD"])
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
