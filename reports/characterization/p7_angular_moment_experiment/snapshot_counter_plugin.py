"""Test-only lifecycle counter for the isolated source snapshot."""
from solweig_light.radiation.angular_moments import TileMomentOwner

COUNTS = {"build": 0, "close": 0}
_build = TileMomentOwner.build
_close = TileMomentOwner.close


def build(self, *args, **kwargs):
    COUNTS["build"] += 1
    return _build(self, *args, **kwargs)


def close(self):
    if not self.closed:
        COUNTS["close"] += 1
    return _close(self)


TileMomentOwner.build = build
TileMomentOwner.close = close


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0:
        # Eight in-process cases; legacy raw runs in its own worker and is covered
        # by output correctness rather than this parent-process counter.
        assert COUNTS == {"build": 8, "close": 8}, COUNTS
