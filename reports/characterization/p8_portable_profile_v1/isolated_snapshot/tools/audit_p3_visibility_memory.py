"""Reproducible Python-allocation audit; not an RSS or speed benchmark."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import tracemalloc

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from solweig_light.geometry.visibility import VisibilityBuilder, export_visibility_npz, import_visibility_npz
from solweig_light.geometry.visibility_native import save_native_visibility, open_native_visibility


def measure(operation):
    gc.collect()
    tracemalloc.start()
    value = operation()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, {'python_traced_current_bytes': current, 'python_traced_peak_bytes': peak}


def audit():
    shape = (256, 256, 153)
    budget = 64 * 1024
    plane = ((np.arange(shape[0])[:, None] + np.arange(shape[1])[None, :]) % 2).astype(np.float32)
    builder = VisibilityBuilder(shape, max_workspace_bytes=budget)
    for _ in range(shape[2]):
        builder.append(plane)
    packed = builder.finish()
    del builder
    with tempfile.TemporaryDirectory(prefix='p3-visibility-memory-') as directory:
        directory = Path(directory)
        manifest = directory / 'channel.json'
        save_native_visibility(manifest, packed, max_workspace_bytes=budget)
        encoded_bytes = packed.nbytes
        del packed
        channels, opened = measure(lambda: [open_native_visibility(manifest, max_workspace_bytes=budget) for _ in range(3)])
        try:
            decoded, decode = measure(lambda: channels[0].decode_patch(76))
            np.testing.assert_array_equal(decoded.view(np.uint32), plane.view(np.uint32))
            decoded_bytes = decoded.nbytes
            del decoded
            legacy = directory / 'visibility.npz'
            stats, exported = measure(lambda: export_visibility_npz(legacy, *channels, max_workspace_bytes=budget))
            restored = import_visibility_npz(legacy, max_workspace_bytes=budget)
            exact = True
            for channel in restored.values():
                for patch in range(shape[2]):
                    exact &= np.array_equal(channel.decode_patch(patch).view(np.uint32), plane.view(np.uint32))
            assert exact
            modes = list(channels[0].modes)
            no_payload_copy = all(isinstance(c._mapping, np.memmap) and not c._mapping.flags.writeable and
                                  all(not isinstance(p.payload, bytes) for p in c._patches) for c in channels)
            assert no_payload_copy
        finally:
            for channel in channels:
                channel.close()
        assert all(c.closed for c in channels)
    return {'audit_version': 1, 'workload': {'shape': list(shape), 'channels': 3,
            'values': ['canonical float32 +0', 'canonical float32 +1'],
            'patches_per_channel': shape[2], 'encoded_bytes_per_channel': encoded_bytes,
            'dense_bytes_per_channel': int(np.prod(shape)) * 4,
            'workspace_bytes': budget},
            'environment': {'python': sys.version, 'numpy': np.__version__},
            'provenance': {'invocation': 'PYTHONPATH=src .venv-light/bin/python tools/audit_p3_visibility_memory.py',
                           'source_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                                             for name in ('tools/audit_p3_visibility_memory.py',
                                                          'src/solweig_light/geometry/visibility.py',
                                                          'src/solweig_light/geometry/visibility_native.py')}},
            'native_open_three_channels': opened, 'decode_one_plane': dict(decode, result_bytes=decoded_bytes),
            'legacy_export_three_channels': dict(exported, explicit_workspace_statistics=stats),
            'verification': {'readonly_memmap_without_bytes_payload_copy': no_payload_copy,
                             'all_patch_modes_binary': all(m == 'binary' for m in modes),
                             'all_459_exported_planes_bit_exact': bool(exact),
                             'all_three_mappings_explicitly_closed': all(c.closed for c in channels)},
            'limitations': ['tracemalloc measures traced Python/NumPy allocations, not process-tree RSS or all native allocations.',
                            'Memory-mapped resident disk pages, OS page cache, zlib internal buffers and untraced native allocations are excluded.',
                            'Import roundtrip is correctness verification; its allocations are not included in export measurements.',
                            'No timing or speedup claim; encoded storage and a returned decoded plane are separate from temporary workspace.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('reports/p3_visibility_memory.json'))
    args = parser.parse_args()
    report = audit()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
