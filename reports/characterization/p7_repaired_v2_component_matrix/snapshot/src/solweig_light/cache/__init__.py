"""Read-only legacy validation and dependency-addressed native geometry cache."""
from .geometry import (
    GeometryStore, GeometryHandle, content_fingerprint, array_fingerprint,
    raster_fingerprint, SVF_FIELDS, ARRAY_FIELDS, VISIBILITY_FIELDS,
)
from .legacy import validate_legacy_geometry, load_legacy_geometry

__all__ = [
    'GeometryStore', 'GeometryHandle', 'content_fingerprint', 'array_fingerprint',
    'raster_fingerprint', 'validate_legacy_geometry', 'load_legacy_geometry',
    'SVF_FIELDS', 'ARRAY_FIELDS', 'VISIBILITY_FIELDS',
]
