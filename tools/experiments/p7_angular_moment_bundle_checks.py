#!/usr/bin/env python3
"""Ownership, identity, bounded-decode, fallback and cleanup checks."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import threading
import weakref
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from solweig_light.geometry.visibility import PackedVisibility
from solweig_light.radiation import patch_radiation
from tools.experiments.p7_angular_moment import load_packet
from tools.experiments.p7_angular_moment_bundle import TileMomentOwner, build_bundle


class TrackedPacked:
    dtype = np.dtype(np.float32)
    def __init__(self, value, limit):
        self.value = PackedVisibility.from_dense(value)
        self.shape = self.value.shape
        self.limit = limit
        self.decoded = []
    def decode_pixels(self, patch, start, stop):
        assert stop-start <= self.limit
        self.decoded.append((patch, start, stop))
        return self.value.decode_pixels(patch, start, stop)
    def __array__(self, *args, **kwargs): raise AssertionError("full packed decode")


def main():
    root = ROOT / "tests/reference/patch_radiation_original_cpu"
    manifest = json.loads((root / "manifest.json").read_text())
    case = next(c for c in manifest["cases"] if c["status"] == "captured" and c["function"] == "define_patch_characteristics")
    values = load_packet(case, manifest)
    patches = values["patch_altitude"].size
    dense = [values[k].reshape(-1, patches) for k in ("shmat", "vegshmat", "vbshvegshmat")]
    geometry = patch_radiation.patch_geometry(np.column_stack((values["patch_altitude"], values["patch_azimuth"])))
    angular = (values["steradian"], geometry.sine, geometry.cosine,
               geometry.longwave_cardinal_cosine, geometry.reflection_cardinal)
    base = build_bundle(*dense, *angular, block_pixels=7)
    assert not base.values.flags.writeable and base.values.nbytes == base.pixels*48

    tracked = [TrackedPacked(x.reshape(values["rows"], values["cols"], patches), 7) for x in dense]
    packed = build_bundle(*tracked, *angular, block_pixels=7)
    np.testing.assert_array_equal(packed.values, base.values)
    assert packed.identity == base.identity and all(x.decoded for x in tracked)

    identities = {"base": base.identity}
    changed = dense[0].copy(); changed.flat[0] = np.float32(1-changed.flat[0])
    identities["visibility_changed"] = build_bundle(changed, dense[1], dense[2], *angular).identity
    solid = np.asarray(angular[0]).copy(); solid[0] = np.nextafter(solid[0], np.float32(np.inf))
    identities["solid_changed"] = build_bundle(*dense, solid, *angular[1:]).identity
    order = np.arange(patches)[::-1]
    identities["reordered"] = build_bundle(*(x[:, order] for x in dense),
        solid[order], angular[1][order], angular[2][order], angular[3][order], angular[4][order]).identity
    assert len(set(identities.values())) == len(identities)

    owners, errors = [], []
    def construct():
        try:
            with TileMomentOwner() as owner:
                bundle = owner.build(*dense, *angular, block_pixels=7)
                owners.append((bundle.identity, owner.allocation_bytes, owner.count))
            assert owner.count == 0 and owner.allocation_bytes == 0 and owner.closed
        except BaseException as error: errors.append(repr(error))
    threads = [threading.Thread(target=construct) for _ in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert not errors and len(owners) == 4 and len({x[0] for x in owners}) == 1

    owner = TileMomentOwner()
    held = owner.build(*dense, *angular)
    reference = weakref.ref(held.values)
    del held; owner.close(); gc.collect()
    assert reference() is None
    try:
        owner.build(*dense, *angular)
        raise AssertionError("closed owner accepted construction")
    except RuntimeError: pass
    failed = TileMomentOwner()
    try:
        failed.build(*dense, *angular, block_pixels=1, fail_after_blocks=1)
        raise AssertionError("injected failure did not propagate")
    except RuntimeError as error:
        assert "injected" in str(error)
    assert failed.count == 0 and failed.allocation_bytes == 0

    result = {"schema":"p7_angular_moment_bundle_checks_v1", "status":"pass",
              "pixels":base.pixels, "patches":base.patches, "allocation_bytes":base.values.nbytes,
              "identity_invalidation":identities, "packed_max_block":7,
              "packed_decode_calls":[len(x.decoded) for x in tracked],
              "concurrent_tile_owners":len(owners), "failure_cleanup":"pass",
              "lifetime_release":"pass", "closed_owner_fail_closed":"pass"}
    out = ROOT / "reports/characterization/p7_angular_moment_experiment/bundle_checks.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
