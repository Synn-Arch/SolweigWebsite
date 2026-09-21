"""Live ownership boundaries for the serial chronological orchestrator.

Containers retain array/scalar identities. They do not cast, copy timestep
results, retain histories, or define persistence/cache validation policy.
"""
from dataclasses import dataclass, field
from typing import Any
import numpy as np

STATE_NAMES = ('CI', 'firstdaytime', 'timestepdec', 'timeadd', 'Tgmap1',
               'Tgmap1E', 'Tgmap1S', 'Tgmap1W', 'Tgmap1N', 'TgOut1')


@dataclass(frozen=True, slots=True)
class StaticScene:
    """One logical tile after normalization; arrays are read-only by ownership."""
    dsm: np.ndarray
    trees: np.ndarray
    dem: np.ndarray
    buildings: np.ndarray
    vegdsm: np.ndarray
    vegdsm2: np.ndarray
    bush: np.ndarray
    amaxvalue: Any
    valid_mask: np.ndarray
    metadata: Any
    scale: Any
    landcover: int
    lcgrid: Any


@dataclass(frozen=True, slots=True)
class ForcingTimeline:
    """Ordered forcing and solar coordinates; uniform forcing stays scalar."""
    met: np.ndarray
    meteorology: dict
    wind: np.ndarray
    wind_direction: np.ndarray
    uhi: np.ndarray
    psi: np.ndarray
    altitude: np.ndarray
    azimuth: np.ndarray
    zen: np.ndarray
    jday: np.ndarray
    dectime: np.ndarray
    altmax: np.ndarray
    location: dict
    wetbulb: Any

    def at(self, index):
        return {name: values[index] for name, values in self.meteorology.items()}


@dataclass(frozen=True, slots=True)
class GeometryCache:
    """Compact geometry with a separate legacy export-presence artifact flag."""
    walls: np.ndarray
    aspects: np.ndarray
    svfs: dict
    svfbuveg: np.ndarray
    asvf: np.ndarray
    diffsh: Any
    svfalfa: np.ndarray
    available_on_disk: bool


@dataclass(slots=True)
class SimulationState:
    """Complete carried engine state plus the daily water-temperature input."""
    CI: Any
    firstdaytime: Any
    timestepdec: Any
    timeadd: Any
    Tgmap1: np.ndarray
    Tgmap1E: np.ndarray
    Tgmap1S: np.ndarray
    Tgmap1W: np.ndarray
    Tgmap1N: np.ndarray
    TgOut1: np.ndarray
    Twater: Any = field(default_factory=list)

    @classmethod
    def initial(cls, zero, timestepdec):
        return cls(1.0, 1.0, timestepdec, 0.0,
                   *(zero.copy() for _ in range(6)))

    def engine_arguments(self):
        return {name: getattr(self, name) for name in STATE_NAMES}

    def accept(self, fields):
        # Take ownership of exactly the arrays/scalars returned by the engine.
        # Replacement releases the previous generation; no history is retained.
        for name in STATE_NAMES:
            setattr(self, name, fields[name])


@dataclass(frozen=True, slots=True)
class Workspace:
    """Tile-lifetime neutral buffers; engine temporaries remain engine-owned.

    Both arrays are initialized over the complete logical domain before use.
    They are reusable broadcast bases and must not be overwritten.
    """
    zero: np.ndarray
    ones: np.ndarray


OUTPUT_NAMES = {'save_tmrt': 'TMRT', 'save_kup': 'Kup', 'save_kdown': 'Kdown',
                'save_lup': 'Lup', 'save_ldown': 'Ldown', 'save_shadow': 'Shadow',
                'save_wbgt': 'WBGT', 'save_ta': 'Ta', 'save_wind': 'Wind'}


@dataclass(frozen=True, slots=True)
class OutputPlan:
    """Requested bands and additional comfort dependencies.

    All physical engine intermediates remain required independently of saved
    bands; output flags never prune the numerical model.
    """
    requested: tuple
    save_svf: bool
    needs_wetbulb: bool

    @classmethod
    def from_flags(cls, flags):
        return cls(('UTCI',) + tuple(name for flag, name in OUTPUT_NAMES.items()
                                     if flags.get(flag, False)),
                   bool(flags.get('save_svf', False)),
                   bool(flags.get('save_wbgt', False)))
