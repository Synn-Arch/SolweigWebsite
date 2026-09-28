"""Private common numerical geometry recipe shared by both production routes.

The standalone export service (``geometry/service.py``) and the pipeline tile
driver (``pipeline.py``) previously extended or used ``geometry_identity``
differently, so one cold logical tile was produced twice under two native
keys despite bitwise-identical normalized inputs and outputs (census C6-02).
This module is the one numerical dependency identity and the one
normalizer/producer body that both routes derive, so an admitted cold tile
is produced once under one native key when the dependency-addressed cache is
enabled.

Payload restrictions (D09): the recipe carries content fingerprints, patch
configuration, the implementation closure and the source paths re-read at
production time. It contains no GDAL dataset, no mutable global, no forcing
data and no export destination. Export-operation provenance (construction
label, exporter implementation, overwrite/presence policy) is a separate
identity each route builds around the recipe digest; it never changes the
native production key.

Numerical contract: ``normalize`` and ``GeometryRecipe.produce`` are
line-for-line the previously duplicated normalization bodies
(``geometry/service.py`` ``_producer`` and the ``pipeline.py`` geometry
branch) feeding the same ``svf_calculator_compact``, so producer outputs are
bitwise identical to both prior routes on the same inputs. Typed math uses
plain NumPy operators exactly as before: no fastmath, no reassociation, no
changed dtype promotions, signed zeros, NaN comparisons or warnings.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from ..cache.geometry import canonical_json
from ..identities import geometry_identity, implementation_identity

# New cache identity policy version for the shared producer: the numerical
# key is deliberately a third, distinct key (old generations remain on disk
# unused). The closure is the conservative one: both wrapper modules
# (pipeline.py, geometry/service.py) plus this module stay inside the
# numerical implementation identity, so either wrapper changing rekeys.
RECIPE_POLICY = 'common-numerical-geometry-v1'
RECIPE_FILES = ('geometry/service.py', 'geometry/recipe.py')
INPUT_NAMES = ('Building_DSM', 'Trees', 'DEM')
# The nineteen producer results in the exact tuple order that
# geometry/svf.py returns; geometry/service.py RESULT_NAMES and pipeline.py
# SVF_NAMES are the identical sequence.
RESULT_NAMES = tuple('svf svfaveg svfE svfEaveg svfEveg svfN svfNaveg svfNveg svfS '
                     'svfSaveg svfSveg svfveg svfW svfWaveg svfWveg vegshmat '
                     'vbshvegshmat shmat svftotal'.split())
# Export-operation policy label for the standalone route's manifest identity.
OPERATION_POLICY = 'overwrite-explicit-complete-set-validated-compare-v1'


def normalize(tree, dem, a):
    """Shared numerical normalization; mutates ``tree`` in place as before.

    Line-for-line the body both routes executed (``geometry/service.py``
    ``_producer`` and ``pipeline.py`` lines around the scene construction):
    negative trees clamp to zero, trunk/canopy and building-relative
    vegetation arrays follow the original expressions and dtype promotions,
    and ``amaxvalue`` keeps the exact float32 maximum semantics.
    """
    tree[tree < 0] = 0
    height = tree + dem
    trunkheight = tree * np.float32(.25) + dem
    bush = np.logical_not(trunkheight * height) * height
    vegdem = tree + a
    vegdem[vegdem == a] = 0
    vegdem2 = tree * np.float32(.25) + a
    vegdem2[vegdem2 == a] = 0
    amaxvalue = np.maximum(a.max(), height.max())
    return a, vegdem, vegdem2, bush, amaxvalue


@dataclass(frozen=True)
class GeometryRecipe:
    """Immutable numerical dependency identity plus its producer paths.

    ``identity`` is the frozen canonical dict handed to
    ``GeometryStore.get_or_create``; ``paths`` are the read-only source
    locations re-opened at production time. Neither is mutated by this
    module after construction.
    """
    identity: dict
    paths: dict
    digest: str = field(default='')

    def native_key(self, store):
        """The native production key this recipe derives for ``store``."""
        return store.key_for(self.identity)

    def export_identity(self, *, construction, exporter_implementation,
                        operation_policy=OPERATION_POLICY):
        """Separate export-operation provenance wrapping the recipe digest.

        This identity is for the standalone export manifest only: it proves
        which export wrapper and operation policy produced the published
        artifacts, while ``numerical_recipe.digest`` references the exact
        numerical generation without rekeying native production.
        """
        return {'numerical_recipe': {'policy': RECIPE_POLICY, 'digest': self.digest},
                'construction': str(construction),
                'exporter_implementation': exporter_implementation,
                'operation_policy': str(operation_policy)}

    def produce(self):
        """Read the three source rasters and run the shared numerical producer.

        Returns the nineteen producer results keyed by ``RESULT_NAMES``:
        sixteen float32 SVF arrays, three packed visibility channels and
        ``svftotal`` — exactly what both routes' producers returned.
        """
        from osgeo import gdal
        from .svf import svf_calculator_compact
        template = gdal.Open(str(self.paths['Building_DSM']))
        try:
            a = template.GetRasterBand(1).ReadAsArray().astype(np.float32)
            scale = 1 / template.GetGeoTransform()[1]
        finally:
            template = None
        def read(name):
            dataset = gdal.Open(str(self.paths[name]))
            try:
                return dataset.ReadAsArray().astype(np.float32)
            finally:
                dataset = None
        tree, dem = read('Trees'), read('DEM')
        a, vegdem, vegdem2, bush, amaxvalue = normalize(tree, dem, a)
        values = svf_calculator_compact(self._patch_option, amaxvalue, a, vegdem, vegdem2,
                                        bush, scale, save_rasters=False)
        return dict(zip(RESULT_NAMES, values))

    @property
    def _patch_option(self):
        return self.identity['patch_option']


def numerical_geometry_recipe(paths, patch_option):
    """Build the immutable recipe for one logical tile from source paths.

    ``paths`` may carry extra workflow entries; only the three numerical
    inputs are selected. The identity is the complete ``geometry_identity``
    domain under the new ``RECIPE_POLICY`` with the export wrapper and this
    module added to the implementation closure, frozen through strict JSON
    canonicalization exactly as the store freezes it.
    """
    try:
        sources = {name: Path(paths[name]) for name in INPUT_NAMES}
    except KeyError as error:
        raise ValueError(f'Missing numerical geometry input {error.args[0]}; '
                         f'required: {", ".join(INPUT_NAMES)}') from error
    identity = geometry_identity(sources, patch_option)
    identity['policy'] = RECIPE_POLICY
    identity['implementation'] = dict(identity['implementation'],
                                      **implementation_identity(RECIPE_FILES))
    # Freeze the caller-visible payload so in-place mutation cannot alter an
    # in-flight key, mirroring GeometryStore.get_or_create. Plain dicts are
    # required downstream (strict JSON canonicalization and the store's
    # mapping checks); treat them as read-only by convention.
    identity = json.loads(canonical_json(identity))
    digest = hashlib.sha256(canonical_json(identity)).hexdigest()
    return GeometryRecipe(identity=identity, paths=sources, digest=digest)


def guarded_producer(recipe, check=None):
    """Wrap ``recipe.produce`` with the caller's input-stability checks.

    The check callable is route-owned acquisition policy (``InputGuard``);
    it runs before and after production exactly as the standalone service
    scheduled its producer, leaving detection semantics unchanged.
    """
    def producer():
        if check is not None:
            check()
        fields = recipe.produce()
        if check is not None:
            check()
        return fields
    return producer
