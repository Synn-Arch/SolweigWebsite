"""Private phase-level memory admission for v6 phase scheduling (C6-42).

The reservation model is derived from the C6-03 *actual lifetime inventory*
(``optimization_v6_continue/evidence/memory/m1_stage_live_array_inventory.md``
through ``m4_instrumented_run.json``, commit ``e7a2d6ec``), not from
``RuntimeOptions`` guesses.  The public ``solweig_light.runtime`` admission
model is untouched; this module is inert until the integrator wires it (see
``optimization_v6_continue/evidence/mem_adm/m7_integration_recipe.md``).

What this corrects relative to ``runtime.estimate_memory``/``plan_admission``
(each miss is M2 §6 a-g, with the M1 stage evidence behind it):

1. GDAL block cache is charged per process (M2 §6a: the C-level cache is
   invisible to numpy/tracemalloc; default 5% of physical RAM, matching the
   unset-``GDAL_CACHEMAX`` behaviour observed in M2 §5).
2. The 32-plane float64 reserve is priced at 8 bytes/value, not 4 (M2 §6b).
   This prices the legacy promotion reserve at its denominated dtype — D04:
   "correct float64 accounting means eight-byte arrays, not just a label on
   four-byte planes".  Per the C6-60 review (F7) no resident float64 family
   exists inside a simulation tile (walls/aspects are float32 residents);
   the reserve is purely against hypothetical dependency promotions, which
   is what the ``runtime.py`` comment always claimed.
3. Per-write digest copies and the checkpoint transit pulse are charged
   (M2 §6c/d: 10 + 12 full-plane equivalents).
4. Export-overlap stream buffers are charged where exports share the window
   (M2 §6e, M1 stage 1 and stage 6 windows).
5. The parent process footprint is charged once per process tree (M2 §6f,
   corroborated at 0.22-0.26 GiB by the M4 instrumented run).
6. Phase shape is honoured: preprocess, geometry and simulation carry
   different reservations instead of every phase paying the full
   simulation estimate (M2 §6g).

Failure semantics (C6-03 m3): admission rejection raises the existing
``runtime.ResourceAdmissionError`` and must happen in-process only.  The
class is not pickle-safe; across a worker boundary the scheduler rebuilds it
through ``runtime._child_exception``, never through plain pickling.  A phase
that can never fit raises in both policies; ``policy="queue"`` only defers
jobs that individually fit but exceed the safe parallel width.  Missing
shape data is an explicit error, never a zero-byte reservation.  All
arithmetic is Python integers: no float rounding, no ``numpy.seterr``
manipulation, and no negative intermediate values.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Mapping, Sequence

from .runtime import (
    DEFAULT_PATCHES,
    DEFAULT_WIND_CHANNELS,
    DTYPE64_RESERVED_PLANES,
    LIVE_FULL_PLANE_EQUIVALENTS,
    ResourceAdmissionError,
    _physical_memory_bytes,
)

__all__ = [
    "DEFAULT_EXPORT_STREAM_BYTES",
    "DEFAULT_GDAL_CACHE_RATIO",
    "DEFAULT_PARENT_FOOTPRINT_BYTES",
    "DenseCubeFallbackNotChargedError",
    "INVENTORY_LINES",
    "InventoryLine",
    "PHASES",
    "PHASE_GEOMETRY",
    "PHASE_PREPROCESS",
    "PHASE_SIMULATION",
    "PhaseAdmissionPlan",
    "PhaseJob",
    "Reservation",
    "ResourceAdmissionError",  # re-exported for type identity (runtime.py)
    "TileShapeDescriptor",
    "VISIBILITY_BINARY",
    "VISIBILITY_NATIVE_CACHE",
    "VISIBILITY_RAW",
    "VISIBILITY_TERNARY",
    "VISIBILITY_UNKNOWN_COLD",
    "admission_inputs_from_options",
    "default_gdal_cache_bytes",
    "legacy_estimate_total_bytes",
    "plan_phase_admission",
    "preprocess_reservation",
    "shape_from_building_dsm",
    "geometry_reservation",
    "simulation_reservation",
]


MIB = 1024 * 1024

PHASE_PREPROCESS = "preprocess"
PHASE_GEOMETRY = "geometry"
PHASE_SIMULATION = "simulation"
PHASES = (PHASE_PREPROCESS, PHASE_GEOMETRY, PHASE_SIMULATION)

VISIBILITY_BINARY = "binary"
VISIBILITY_TERNARY = "ternary"
VISIBILITY_RAW = "raw"
VISIBILITY_NATIVE_CACHE = "native_cache"
VISIBILITY_UNKNOWN_COLD = "unknown_cold"

# Encoded payload bits per pixel per channel, from the PackedVisibility
# payload validation at geometry/visibility.py:90-92 (binary
# (pixels*1+7)//8, ternary (pixels*2+7)//8, raw pixels*4).  ``unknown_cold``
# reserves the raw fallback: D04 requires cold unknown modes to reserve the
# fallback rather than predict binary unconditionally.  ``native_cache``
# carries the same byte charge as raw, but as mapped (read-only) pages:
# "a read-only mapping may become resident; mmap is not free memory" (D04),
# and the packed-heap raw term is absent in that mode (M2 §7).
_PACKED_BITS_PER_PIXEL = {
    VISIBILITY_BINARY: 1,
    VISIBILITY_TERNARY: 2,
    VISIBILITY_RAW: 32,
}
VISIBILITY_MODES = (
    VISIBILITY_BINARY,
    VISIBILITY_TERNARY,
    VISIBILITY_RAW,
    VISIBILITY_NATIVE_CACHE,
    VISIBILITY_UNKNOWN_COLD,
)

# GDAL block cache: per-process default when GDAL_CACHEMAX is unset is 5% of
# usable physical RAM (M2 §5).  Charged per worker process because the pool
# spawns separate processes (M2 §6a: "W workers can hold W x 5%").
DEFAULT_GDAL_CACHE_RATIO = (5, 100)
# Parent footprint: CLI/scheduler process holding bookkeeping and GDAL schema
# probes.  M2 §6(f) observed 0.2-0.4 GiB; M4 measured 0.22-0.26 GiB baseline
# interpreter + numba/LLVM + GDAL libraries.  Default is the top of the range.
DEFAULT_PARENT_FOOTPRINT_BYTES = 429_496_730  # ceil(0.4 GiB)
# Bounded streaming buffers for in-window publication (GDAL band buffers and
# NPZ stream chunks; M2 §6e "tens of MiB, bounded by streaming").  Charged
# whenever exports share the phase window (M2 §4 windows 1-2), which is the
# default because publication always overlaps resident arrays today.
DEFAULT_EXPORT_STREAM_BYTES = 64 * MIB

# Legacy simulation envelope, kept as accounting units (runtime.py:285-286):
# 192 full-plane f32 equivalents cover the enumerated M1 stage 0/3/4 families
# (see INVENTORY_LINES and the reconciliation test), and the separate
# 32-plane reserve is repriced at float64 (correction 2 above: reserve
# pricing, not a claim about resident arrays).
SIMULATION_LIVE_F32_PLANES = LIVE_FULL_PLANE_EQUIVALENTS
SIMULATION_F64_RESERVE_PLANES = DTYPE64_RESERVED_PLANES
# M1 stage 2 / M2 P0: 6-8 float64 planes in ``filter1Goodwin``-family
# intermediates plus the float64 ``y`` accumulator (walls.py:181); charge 10.
PREPROCESS_F64_PLANES = 10
# M1 stage 1 geometry window: scene/input derived planes (stage 0 range
# 30-40, charged at the top), 15 SVF field planes (svf.py:111) and the 3
# derived planes svfbuveg/asvf/svfalfa (pipeline.py:193-200).
GEOMETRY_SCENE_PLANES = 40
GEOMETRY_SVF_PLANES = 15
GEOMETRY_DERIVED_PLANES = 3
GEOMETRY_F32_PLANES = GEOMETRY_SCENE_PLANES + GEOMETRY_SVF_PLANES + GEOMETRY_DERIVED_PLANES
# Dense cube fallback of the non-compact SVF route (svf.py:126; M1: "only
# non-compact route").  Not charged by default; ``dense_fallback=True`` adds
# it for callers that know they run the legacy dense route.
DENSE_CUBE_FALLBACK_PLANES = 153
# Write pulses inside the simulation loop (M2 §6c/d): per-write digest
# copies of up to 10 output planes (persistence.py:456-468) plus the
# checkpoint transit of 6 state planes x2 (f64-width transit + BytesIO copy,
# persistence.py:84-104,160-173).
DIGEST_PULSE_PLANES = 10
CHECKPOINT_PULSE_PLANES = 12
SIMULATION_WRITE_PULSE_PLANES = DIGEST_PULSE_PLANES + CHECKPOINT_PULSE_PLANES


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class InventoryLine:
    """One row of the derived lifetime inventory (phase x array x dtype x
    shape x lifetime), transcribed from the C6-03 M1 stage tables.

    ``planes`` counts full-plane equivalents of ``(rows, cols)`` values;
    ``bytes_per_value`` is 4 or 8 for array rows and 0 for rows whose charge
    is a byte term rather than a plane count (packed payloads, mapped pages).
    """

    phase: str
    family: str
    dtype: str
    shape: str
    planes: int
    bytes_per_value: int
    lifetime: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return dict(
            phase=self.phase,
            family=self.family,
            dtype=self.dtype,
            shape=self.shape,
            planes=self.planes,
            bytes_per_value=self.bytes_per_value,
            lifetime=self.lifetime,
            source=self.source,
        )


INVENTORY_LINES: tuple[InventoryLine, ...] = (
    # --- preprocess (M1 stage 2) -----------------------------------------
    InventoryLine(
        PHASE_PREPROCESS, "walls/aspect intermediates", "float64", "full_plane",
        PREPROCESS_F64_PLANES, 8, "transient_stage",
        "M1 S2; walls.py:181; walls_compiled.py:39,49",
    ),
    # --- geometry (M1 stage 1) -------------------------------------------
    InventoryLine(
        PHASE_GEOMETRY, "scene/input derived planes", "float32", "full_plane",
        GEOMETRY_SCENE_PLANES, 4, "persistent_tile", "M1 S0 (range 30-40, top charged)",
    ),
    InventoryLine(
        PHASE_GEOMETRY, "SVF field planes", "float32", "full_plane",
        GEOMETRY_SVF_PLANES, 4, "persistent_tile", "M1 S1; geometry/svf.py:111",
    ),
    InventoryLine(
        PHASE_GEOMETRY, "svfbuveg/asvf/svfalfa", "float32", "full_plane",
        GEOMETRY_DERIVED_PLANES, 4, "persistent_tile", "M1 S1; pipeline.py:193-200",
    ),
    InventoryLine(
        PHASE_GEOMETRY, "packed visibility payloads (shmat/vegshmat/vbshvegshmat)",
        "packed", "packed_channels", 0, 0, "persistent_tile",
        "M1 S1; geometry/visibility.py:90-92 (1/2/32 bits per pixel); charged as payload_bytes",
    ),
    InventoryLine(
        PHASE_GEOMETRY, "dense cube fallback mats (non-compact route only)",
        "float32", "full_plane",
        DENSE_CUBE_FALLBACK_PLANES, 4, "transient_stage",
        "M1 S1; geometry/svf.py:126; excluded unless dense_fallback=True",
    ),
    # --- simulation (M1 stages 0/3/4/5) ----------------------------------
    InventoryLine(
        PHASE_SIMULATION, "scene/input derived planes", "float32", "full_plane",
        GEOMETRY_SCENE_PLANES, 4, "persistent_tile", "M1 S0",
    ),
    InventoryLine(
        PHASE_SIMULATION, "SVF fields + svfbuveg/asvf/svfalfa", "float32",
        "full_plane", GEOMETRY_SVF_PLANES + GEOMETRY_DERIVED_PLANES, 4,
        "persistent_tile", "M1 S3",
    ),
    InventoryLine(
        PHASE_SIMULATION, "walls/aspects residents", "float32", "full_plane",
        2, 4, "persistent_tile",
        "M1 S0 (dtype corrected by C6-60 review F7); read_raster astype f32 "
        "(io/rasters.py:24), published GDT_Float32 (io/rasters.py:31-35), "
        "GeometryCache passthrough (pipeline.py:202); float64 exists only as "
        "preprocess intermediates. Inside the 192-plane envelope",
    ),
    InventoryLine(
        PHASE_SIMULATION, "wind coefficient planes", "float32", "full_plane",
        DEFAULT_WIND_CHANNELS, 4, "persistent_tile", "M1 S0; runtime.py:281",
    ),
    InventoryLine(
        PHASE_SIMULATION, "SimulationState planes", "float32", "full_plane",
        6, 4, "persistent_tile", "M1 S3; models.py; mutated per timestep",
    ),
    InventoryLine(
        PHASE_SIMULATION, "GVF serial internals", "float32", "full_plane",
        40, 4, "transient_timestep",
        "M1 S4; ground_view.py:218,216-217,240-263,413-428,443-450",
    ),
    InventoryLine(
        PHASE_SIMULATION, "wall-shadow output", "float32", "full_plane",
        8, 4, "transient_timestep", "M1 S4; wall_shadows.py:77-105,140-200",
    ),
    InventoryLine(
        PHASE_SIMULATION, "compiled Kside output", "float32", "full_plane",
        7, 4, "transient_timestep", "M1 S4; patch_radiation.py:544,560-582",
    ),
    InventoryLine(
        PHASE_SIMULATION, "compiled define_patch output", "float32", "full_plane",
        11, 4, "transient_timestep", "M1 S4; patch_radiation.py:883",
    ),
    InventoryLine(
        PHASE_SIMULATION, "compiled Lcyl output", "float32", "full_plane",
        4, 4, "transient_timestep", "M1 S4; patch_radiation.py:903-932",
    ),
    InventoryLine(
        PHASE_SIMULATION, "dRad anisotropic decode pair", "float32", "full_plane",
        2, 4, "transient_timestep", "M1 S4; engine.py:1563",
    ),
    InventoryLine(
        PHASE_SIMULATION, "cylindric_wedge internals", "float32", "full_plane",
        10, 4, "transient_timestep", "M1 S4 (engine)",
    ),
    InventoryLine(
        PHASE_SIMULATION, "output dict (Tmrt/UTCI/requested bands)", "float32",
        "full_plane", 10, 4, "transient_timestep", "M1 S4; pipeline.py:258-283",
    ),
    InventoryLine(
        PHASE_SIMULATION, "numpy-fallback Kside intermediates", "float32",
        "full_plane", 30, 4, "transient_stage",
        "M1 S4; engine.py:441-502; fallback route only",
    ),
    InventoryLine(
        PHASE_SIMULATION, "packed visibility payloads", "packed",
        "packed_channels", 0, 0, "persistent_tile",
        "M1 S3; charged as payload_bytes (raw worst 1.79 GiB @1024/P153)",
    ),
    InventoryLine(
        PHASE_SIMULATION, "per-write digest copies", "float32", "full_plane",
        DIGEST_PULSE_PLANES, 4, "pulse", "M1 S5; persistence.py:456-468",
    ),
    InventoryLine(
        PHASE_SIMULATION, "checkpoint state transit (6 planes x2)", "float32",
        "full_plane", CHECKPOINT_PULSE_PLANES, 4, "pulse",
        "M1 S5; persistence.py:84-104,160-173",
    ),
)


@dataclasses.dataclass(frozen=True)
class TileShapeDescriptor:
    """JSON-safe raster shape of one logical tile job."""

    rows: int
    cols: int
    patches: int = DEFAULT_PATCHES
    windchannels: int = DEFAULT_WIND_CHANNELS
    block_pixels: int = 128

    def __post_init__(self) -> None:
        for name in ("rows", "cols", "patches", "windchannels", "block_pixels"):
            _positive_int(getattr(self, name), name)

    @property
    def pixels(self) -> int:
        return self.rows * self.cols

    def to_dict(self) -> dict[str, int]:
        return dict(
            rows=self.rows,
            cols=self.cols,
            patches=self.patches,
            windchannels=self.windchannels,
            block_pixels=self.block_pixels,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TileShapeDescriptor":
        return cls(**dict(payload))


@dataclasses.dataclass(frozen=True)
class Reservation:
    """Per-phase byte reservation with auditable components.

    ``payload_bytes``   packed visibility channels (or their mmap equivalent).
    ``live_array_bytes`` full-plane live envelope priced at the real dtypes.
    ``decoded_block_bytes`` compiled per-block decode scratch.
    ``native_bytes``    JIT/LLVM/BLAS allowance (runtime.py:489-490 formula).
    ``gdal_cache_bytes`` per-process GDAL block cache (correction 1).
    ``write_pulse_bytes`` digest + checkpoint pulses (correction 3).
    ``export_stream_bytes`` bounded publication stream buffers (correction 4).
    ``mapped_bytes``    resident read-only mapping pages (native_cache mode).
    """

    phase: str
    shape: TileShapeDescriptor
    visibility_mode: str
    payload_bytes: int
    live_array_bytes: int
    decoded_block_bytes: int
    native_bytes: int
    gdal_cache_bytes: int
    write_pulse_bytes: int
    export_stream_bytes: int
    mapped_bytes: int
    total_bytes: int

    @property
    def pixels(self) -> int:
        return self.shape.pixels

    def inventory(self) -> dict[str, Any]:
        """JSON-safe component breakdown, plane counts included."""
        plane_bytes = self.pixels * 4
        return {
            "phase": self.phase,
            "visibility_mode": self.visibility_mode,
            "pixels": self.pixels,
            "patches": self.shape.patches,
            "windchannels": self.shape.windchannels,
            "block_pixels": self.shape.block_pixels,
            "payload_bytes": self.payload_bytes,
            "live_array_bytes": self.live_array_bytes,
            "decoded_block_bytes": self.decoded_block_bytes,
            "native_bytes": self.native_bytes,
            "gdal_cache_bytes": self.gdal_cache_bytes,
            "write_pulse_bytes": self.write_pulse_bytes,
            "export_stream_bytes": self.export_stream_bytes,
            "mapped_bytes": self.mapped_bytes,
            "total_bytes": self.total_bytes,
            "live_array_plane_equivalents_f32": (
                self.live_array_bytes // plane_bytes if plane_bytes else 0
            ),
            "payload_plane_equivalents_f32": (
                self.payload_bytes // plane_bytes if plane_bytes else 0
            ),
        }

    _FIELDS = (
        "payload_bytes",
        "live_array_bytes",
        "decoded_block_bytes",
        "native_bytes",
        "gdal_cache_bytes",
        "write_pulse_bytes",
        "export_stream_bytes",
        "mapped_bytes",
        "total_bytes",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "shape": self.shape.to_dict(),
            "visibility_mode": self.visibility_mode,
            **{name: getattr(self, name) for name in self._FIELDS},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Reservation":
        data = dict(payload)
        shape = TileShapeDescriptor.from_dict(data.pop("shape"))
        known = {"phase", "visibility_mode", *cls._FIELDS}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown reservation fields: {sorted(unknown)}")
        return cls(shape=shape, **data)


def _native_allowance(windchannels: int) -> int:
    """JIT/LLVM/BLAS allowance, identical formula to runtime.py:489-490.

    Compiled kernels do not duplicate per native thread (M2 §8), so the
    threads dimension adds no charge; BLAS nested pools stay inside the
    allowance.
    """
    return 256 * MIB + 64 * MIB * min(8, windchannels)


def _payload_bytes(pixels: int, patches: int, mode: str) -> tuple[int, int]:
    """Return (heap payload bytes, mapped page bytes) for a visibility mode.

    Three channels per tile (shmat/vegshmat/vbshvegshmat).  ``unknown_cold``
    reserves the raw fallback (D04).  ``native_cache`` payloads are mmap
    backed and charged as resident mapped pages instead of heap bytes; the
    charge is deliberately the same magnitude (M2 §7: over-reserving is the
    safe direction).
    """
    if mode == VISIBILITY_NATIVE_CACHE:
        total = 3 * patches * ((pixels * _PACKED_BITS_PER_PIXEL[VISIBILITY_RAW] + 7) // 8)
        return 0, total
    if mode == VISIBILITY_UNKNOWN_COLD:
        bits = _PACKED_BITS_PER_PIXEL[VISIBILITY_RAW]
    else:
        bits = _PACKED_BITS_PER_PIXEL[mode]
    total = 3 * patches * ((pixels * bits + 7) // 8)
    return total, 0


def _decoded_block_bytes(shape: TileShapeDescriptor) -> int:
    return shape.block_pixels * shape.patches * (shape.windchannels + 4) * 4


def _resolved_gdal_cache(gdal_cache_bytes: int | None) -> int:
    if gdal_cache_bytes is None:
        return default_gdal_cache_bytes()
    return _non_negative_int(gdal_cache_bytes, "gdal_cache_bytes")


def default_gdal_cache_bytes() -> int:
    """Per-process GDAL block cache: 5% of probed physical RAM.

    Matches the unset-``GDAL_CACHEMAX`` default (M2 §5).  If the probe fell
    back to its 1 GiB floor the charge is correspondingly small, which is a
    documented limitation, not an admission relaxation: callers that set
    ``GDAL_CACHEMAX`` explicitly should pass ``gdal_cache_bytes``.
    """
    numerator, denominator = DEFAULT_GDAL_CACHE_RATIO
    return (_physical_memory_bytes() * numerator) // denominator


def legacy_estimate_total_bytes(shape: TileShapeDescriptor) -> int:
    """Reproduce ``runtime.estimate_memory(...).total_bytes`` from first principles.

    This is the pre-C6-42 model (runtime.py:457-502), kept as the anchor for
    the M2 worst case: at 1024x1024/153/12/block1024 it returns exactly
    3,730,374,656 bytes (3.4742 GiB).  Note the float64 reserve is priced at
    4 bytes here — that is the legacy mispricing this module corrects.
    """
    pixels = shape.pixels
    raw_visibility = 3 * pixels * shape.patches * 4
    live_arrays = (
        SIMULATION_LIVE_F32_PLANES + SIMULATION_F64_RESERVE_PLANES + shape.windchannels
    ) * pixels * 4
    decoded = _decoded_block_bytes(shape)
    native = _native_allowance(shape.windchannels)
    return raw_visibility + live_arrays + decoded + native


def preprocess_reservation(
    shape: TileShapeDescriptor,
    *,
    gdal_cache_bytes: int | None = None,
    native_bytes: int | None = None,
    export_stream_bytes: int | None = DEFAULT_EXPORT_STREAM_BYTES,
) -> Reservation:
    """Reservation for the walls/aspect preprocess phase (M1 stage 2, M2 P0).

    No visibility payload and no simulation envelope exists in this window;
    the phase previously paid the full 3.4742 GiB simulation estimate while
    using float64 intermediates plus streams (M2 §6g).  The corrected charge
    is the f64 intermediates + GDAL cache + native allowance + write stream.
    """
    gdal_cache = _resolved_gdal_cache(gdal_cache_bytes)
    native = (
        _native_allowance(shape.windchannels)
        if native_bytes is None
        else _non_negative_int(native_bytes, "native_bytes")
    )
    stream = (
        DEFAULT_EXPORT_STREAM_BYTES
        if export_stream_bytes is None
        else _non_negative_int(export_stream_bytes, "export_stream_bytes")
    )
    live = shape.pixels * PREPROCESS_F64_PLANES * 8
    total = live + gdal_cache + native + stream
    return Reservation(
        PHASE_PREPROCESS, shape, "none",
        0, live, 0, native, gdal_cache, 0, stream, 0, total,
    )


def geometry_reservation(
    shape: TileShapeDescriptor,
    *,
    visibility_mode: str = VISIBILITY_UNKNOWN_COLD,
    dense_fallback: bool = False,
    export_overlap: bool = True,
    gdal_cache_bytes: int | None = None,
    native_bytes: int | None = None,
    export_stream_bytes: int | None = DEFAULT_EXPORT_STREAM_BYTES,
) -> Reservation:
    """Reservation for the geometry production/publication phase (M1 stage 1, M2 P1/P1').

    Charges the packed payload (raw fallback for cold unknown modes), the
    scene/SVF/derived plane set, the native allowance, the per-process GDAL
    cache, and — because publication overlaps resident arrays in every route
    today (M2 §4 windows 1-2) — the bounded export stream allowance unless
    ``export_overlap=False`` is asserted by the caller.  ``dense_fallback``
    adds the 153-plane dense cube of the non-compact SVF route (svf.py:126).
    """
    if visibility_mode not in VISIBILITY_MODES:
        raise ValueError(
            f"visibility_mode must be one of {VISIBILITY_MODES}; got {visibility_mode!r}"
        )
    gdal_cache = _resolved_gdal_cache(gdal_cache_bytes)
    native = (
        _native_allowance(shape.windchannels)
        if native_bytes is None
        else _non_negative_int(native_bytes, "native_bytes")
    )
    stream = (
        DEFAULT_EXPORT_STREAM_BYTES
        if export_stream_bytes is None
        else _non_negative_int(export_stream_bytes, "export_stream_bytes")
    )
    if not export_overlap:
        stream = 0
    payload, mapped = _payload_bytes(shape.pixels, shape.patches, visibility_mode)
    live = shape.pixels * GEOMETRY_F32_PLANES * 4
    if dense_fallback:
        live += shape.pixels * DENSE_CUBE_FALLBACK_PLANES * 4
    total = payload + live + gdal_cache + native + stream + mapped
    return Reservation(
        PHASE_GEOMETRY, shape, visibility_mode,
        payload, live, 0, native, gdal_cache, 0, stream, mapped, total,
    )


def simulation_reservation(
    shape: TileShapeDescriptor,
    *,
    visibility_mode: str = VISIBILITY_UNKNOWN_COLD,
    export_overlap: bool = True,
    gdal_cache_bytes: int | None = None,
    native_bytes: int | None = None,
    export_stream_bytes: int | None = DEFAULT_EXPORT_STREAM_BYTES,
) -> Reservation:
    """Reservation for the simulation phase (M1 stages 3-6, M2 P2/P3).

    Keeps the legacy 192-plane f32 envelope and the wind channels, reprices
    the 32-plane float64 promotion reserve at 8 bytes/value (reserve pricing
    per D04; no resident f64 family exists in-tile — C6-60 review F7), and
    adds the GDAL cache, the write pulses, the export overlap stream and
    (native_cache mode) the resident mapping charge on top of the legacy
    components.
    """
    if visibility_mode not in VISIBILITY_MODES:
        raise ValueError(
            f"visibility_mode must be one of {VISIBILITY_MODES}; got {visibility_mode!r}"
        )
    gdal_cache = _resolved_gdal_cache(gdal_cache_bytes)
    native = (
        _native_allowance(shape.windchannels)
        if native_bytes is None
        else _non_negative_int(native_bytes, "native_bytes")
    )
    stream = (
        DEFAULT_EXPORT_STREAM_BYTES
        if export_stream_bytes is None
        else _non_negative_int(export_stream_bytes, "export_stream_bytes")
    )
    if not export_overlap:
        stream = 0
    payload, mapped = _payload_bytes(shape.pixels, shape.patches, visibility_mode)
    live = (
        (SIMULATION_LIVE_F32_PLANES + shape.windchannels) * shape.pixels * 4
        + SIMULATION_F64_RESERVE_PLANES * shape.pixels * 8
    )
    decoded = _decoded_block_bytes(shape)
    pulse = shape.pixels * SIMULATION_WRITE_PULSE_PLANES * 4
    total = payload + live + decoded + native + gdal_cache + pulse + stream + mapped
    return Reservation(
        PHASE_SIMULATION, shape, visibility_mode,
        payload, live, decoded, native, gdal_cache, pulse, stream, mapped, total,
    )


_RESERVATION_BY_PHASE = {
    PHASE_PREPROCESS: preprocess_reservation,
    PHASE_GEOMETRY: geometry_reservation,
    PHASE_SIMULATION: simulation_reservation,
}


@dataclasses.dataclass(frozen=True)
class PhaseJob:
    """JSON-safe phase job descriptor (D09 ``PhaseJob`` memory dimension).

    ``visibility_mode`` and ``export_overlap`` describe the visibility
    payload and in-window publication of geometry and simulation jobs and
    are forwarded to their reservations.  Preprocess jobs have no visibility
    payload and no publication window of their own, so non-default values
    for the two fields are rejected rather than silently ignored (a
    misdescribed job must fail explicitly, not reserve the wrong phase).
    """

    phase: str
    rows: int
    cols: int
    patches: int = DEFAULT_PATCHES
    windchannels: int = DEFAULT_WIND_CHANNELS
    block_pixels: int = 128
    visibility_mode: str = VISIBILITY_UNKNOWN_COLD
    dense_fallback: bool = False
    export_overlap: bool = True
    tile: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}; got {self.phase!r}")
        if self.visibility_mode not in VISIBILITY_MODES:
            raise ValueError(
                f"visibility_mode must be one of {VISIBILITY_MODES}; got {self.visibility_mode!r}"
            )
        if self.tile is not None and not isinstance(self.tile, str):
            raise TypeError("tile must be a string or None")
        if not isinstance(self.dense_fallback, bool) or not isinstance(self.export_overlap, bool):
            raise TypeError("dense_fallback and export_overlap must be bools")
        # Validate dimension types eagerly so a malformed job cannot survive
        # until admission with undefined shapes.
        TileShapeDescriptor(
            self.rows, self.cols, self.patches, self.windchannels, self.block_pixels
        )

    @property
    def shape(self) -> TileShapeDescriptor:
        return TileShapeDescriptor(
            self.rows, self.cols, self.patches, self.windchannels, self.block_pixels
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(
            phase=self.phase,
            rows=self.rows,
            cols=self.cols,
            patches=self.patches,
            windchannels=self.windchannels,
            block_pixels=self.block_pixels,
            visibility_mode=self.visibility_mode,
            dense_fallback=self.dense_fallback,
            export_overlap=self.export_overlap,
            tile=self.tile,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhaseJob":
        return cls(**dict(payload))


class DenseCubeFallbackNotChargedError(ValueError):
    """A geometry job requested the dense route without declaring it.

    Kept separate from ``ResourceAdmissionError``: this is a descriptor
    completeness error (missing data is an explicit failure, never a fake
    zero), not a budget exhaustion.
    """


def _describe(job: PhaseJob) -> str:
    return f"{job.phase} job {job.tile or ''}".strip() + (
        f" ({job.rows}x{job.cols}, patches={job.patches})"
    )


@dataclasses.dataclass(frozen=True)
class PhaseAdmissionPlan:
    """JSON-safe admission decision for a set of phase jobs.

    ``admissible_workers`` is the largest K such that *any* K jobs from the
    family fit simultaneously (K-largest bound, matching the deterministic
    rule in ``runtime.plan_admission``).  ``policy="reject"`` raises
    ``ResourceAdmissionError`` in-process when the requested concurrency
    exceeds K; ``policy="queue"`` returns ``status="queued"`` and lets the
    scheduler run at most K jobs at once.  Jobs that individually exceed the
    per-tree reservation base raise in both policies: queueing cannot make
    an impossible job fit.
    """

    status: str
    policy: str
    budget_bytes: int
    parent_footprint_bytes: int
    writer_queue_bytes: int
    reservation_base_bytes: int
    requested_workers: int
    admissible_workers: int
    admitted_count: int
    deferred_count: int
    threads_per_worker: int
    reservations: tuple[Reservation, ...]

    @property
    def native_threads(self) -> int:
        return self.admissible_workers * self.threads_per_worker

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "policy": self.policy,
            "budget_bytes": self.budget_bytes,
            "parent_footprint_bytes": self.parent_footprint_bytes,
            "writer_queue_bytes": self.writer_queue_bytes,
            "reservation_base_bytes": self.reservation_base_bytes,
            "requested_workers": self.requested_workers,
            "admissible_workers": self.admissible_workers,
            "admitted_count": self.admitted_count,
            "deferred_count": self.deferred_count,
            "threads_per_worker": self.threads_per_worker,
            "native_threads": self.native_threads,
            "reservations": [item.to_dict() for item in self.reservations],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhaseAdmissionPlan":
        data = dict(payload)
        data.pop("native_threads", None)  # derived property, recomputed
        data["reservations"] = tuple(
            Reservation.from_dict(item) for item in data["reservations"]
        )
        return cls(**data)


def admission_inputs_from_options(options: Any) -> dict[str, int]:
    """JSON-safe admission inputs from an existing ``RuntimeOptions``.

    Uses only the existing ``workers``/``cpu_budget``/``threads_per_worker``
    memory fields (D04: no public RuntimeOptions signature/default changes).
    """
    cpu_workers = max(1, options.cpu_budget // options.threads_per_worker)
    return {
        "budget_bytes": options.resolved_memory_budget_bytes,
        "active_workers": min(options.workers, cpu_workers),
        "threads_per_worker": options.threads_per_worker,
    }


def shape_from_building_dsm(
    paths: Mapping[str, Any],
    *,
    patches: int = DEFAULT_PATCHES,
    windchannels: int = DEFAULT_WIND_CHANNELS,
    block_pixels: int = 128,
) -> TileShapeDescriptor:
    """Read-only raster shape for a tile job, mirroring ``runtime._job_estimate``.

    Opens the Building_DSM for metadata only (no arrays are read) with a lazy
    gdal import, exactly like the existing admission path.  A missing or
    unreadable raster raises ``ResourceAdmissionError`` naming the path —
    missing shape data is an explicit failure, never a zero-byte reservation.
    """
    dsm = paths.get("Building_DSM") if isinstance(paths, Mapping) else None
    if not dsm:
        raise ResourceAdmissionError(
            "paths['Building_DSM'] is required to size the tile before "
            "resource admission"
        )
    try:
        from osgeo import gdal

        dataset = gdal.Open(os.fspath(dsm), gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError("GDAL could not open the Building_DSM")
        rows, cols = dataset.RasterYSize, dataset.RasterXSize
        dataset = None
    except Exception as error:
        raise ResourceAdmissionError(
            f"tile shape could not be read from Building_DSM {dsm!r}: {error}; "
            "provide verified rows/cols before resource admission"
        ) from error
    return TileShapeDescriptor(
        rows, cols, _positive_int(patches, "patches"),
        _positive_int(windchannels, "windchannels"),
        _positive_int(block_pixels, "block_pixels"),
    )


def plan_phase_admission(
    jobs: Sequence[PhaseJob],
    *,
    budget_bytes: int,
    active_workers: int | None = None,
    parent_footprint_bytes: int = DEFAULT_PARENT_FOOTPRINT_BYTES,
    writer_queue_bytes: int = 0,
    gdal_cache_bytes: int | None = None,
    native_bytes: int | None = None,
    threads_per_worker: int = 1,
    policy: str = "reject",
) -> PhaseAdmissionPlan:
    """Admit phase jobs against the process-tree memory budget.

    ``M_total = parent_footprint + sum(active reservations) + writer_queue``
    (D04 phase-aware resource model) must stay within ``budget_bytes``.
    Per-phase reservations come from the M1-derived lifetime inventory, so a
    preprocess job no longer pays the simulation estimate (M2 §6g) while the
    GDAL cache, float64 reserve, write pulses, export streams, mapped pages
    and parent footprint are all charged (M2 §6a-f).

    Raises ``ResourceAdmissionError`` (in-process only; see module docstring)
    when ``policy="reject"`` and the requested concurrency does not fit, and
    in both policies when a single job can never fit or the tree overheads
    already exhaust the budget.  Never performs negative-budget arithmetic.
    """
    _positive_int(budget_bytes, "budget_bytes")
    _non_negative_int(parent_footprint_bytes, "parent_footprint_bytes")
    _non_negative_int(writer_queue_bytes, "writer_queue_bytes")
    _positive_int(threads_per_worker, "threads_per_worker")
    if policy not in ("reject", "queue"):
        raise ValueError(f"policy must be 'reject' or 'queue'; got {policy!r}")
    jobs = tuple(jobs)
    if not jobs:
        raise ValueError("plan_phase_admission requires at least one job")

    base = budget_bytes - parent_footprint_bytes - writer_queue_bytes
    if base <= 0:
        raise ResourceAdmissionError(
            f"process-tree overhead (parent {parent_footprint_bytes:,} B + writer "
            f"queue {writer_queue_bytes:,} B) already exhausts the {budget_bytes:,} B "
            "budget before any phase reservation"
        )

    reservations = []
    for job in jobs:
        builder = _RESERVATION_BY_PHASE[job.phase]
        kwargs: dict[str, Any] = {
            "gdal_cache_bytes": gdal_cache_bytes,
            "native_bytes": native_bytes,
        }
        if job.phase == PHASE_PREPROCESS:
            if job.visibility_mode != VISIBILITY_UNKNOWN_COLD or not job.export_overlap:
                raise ValueError(
                    f"{_describe(job)}: preprocess has no visibility payload and "
                    "no publication window; visibility_mode/export_overlap do "
                    "not apply to preprocess jobs"
                )
        else:
            kwargs["visibility_mode"] = job.visibility_mode
            kwargs["export_overlap"] = job.export_overlap
        if job.phase == PHASE_GEOMETRY:
            kwargs["dense_fallback"] = job.dense_fallback
        elif job.dense_fallback:
            raise DenseCubeFallbackNotChargedError(
                f"{_describe(job)}: dense_fallback is a geometry-phase descriptor"
            )
        reservations.append(builder(job.shape, **kwargs))

    infeasible = [
        (job, reservation)
        for job, reservation in zip(jobs, reservations)
        if reservation.total_bytes > base
    ]
    if infeasible:
        job, reservation = infeasible[0]
        raise ResourceAdmissionError(
            f"{_describe(job)} needs about {reservation.total_bytes:,} B but only "
            f"{base:,} B of the {budget_bytes:,} B budget remains after parent "
            f"({parent_footprint_bytes:,} B) and writer queue ({writer_queue_bytes:,} B); "
            "queueing cannot make this phase fit. Reduce the tile shape, cap "
            "GDAL_CACHEMAX, or use supported disk-backed execution with an explicit "
            "live-array inventory"
        )

    requested = len(jobs) if active_workers is None else _positive_int(
        active_workers, "active_workers"
    )
    ordered = sorted(
        (item.total_bytes for item in reservations), reverse=True
    )
    admissible = 0
    running = 0
    for value in ordered:
        if running + value > base:
            break
        running += value
        admissible += 1
    admissible = min(admissible, requested)

    status = "admitted"
    if requested > admissible:
        if policy == "reject":
            top = ordered[:requested]
            raise ResourceAdmissionError(
                f"requested {requested} concurrent phase jobs but only "
                f"{admissible} fit: the {requested} largest reservations sum to "
                f"{sum(top):,} B against a {base:,} B reservation base "
                f"(budget {budget_bytes:,} B, parent {parent_footprint_bytes:,} B, "
                f"writer queue {writer_queue_bytes:,} B); run at most {admissible} "
                "at once or queue the remainder"
            )
        status = "queued"

    return PhaseAdmissionPlan(
        status=status,
        policy=policy,
        budget_bytes=budget_bytes,
        parent_footprint_bytes=parent_footprint_bytes,
        writer_queue_bytes=writer_queue_bytes,
        reservation_base_bytes=base,
        requested_workers=requested,
        admissible_workers=admissible,
        admitted_count=admissible,
        deferred_count=max(0, requested - admissible),
        threads_per_worker=threads_per_worker,
        reservations=tuple(reservations),
    )
