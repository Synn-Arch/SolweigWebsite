"""Pinned signature checks for all seven public workflows."""
import inspect
import json
from pathlib import Path

import pytest
import solweig_light

CONTRACT = json.loads((Path(__file__).resolve().parents[2] /
                       'reports/contract_snapshot.json').read_text())['public_functions']


@pytest.mark.parametrize('name', ['thermal_comfort', 'preprocess', 'run_walls_aspect',
                                  'calculate_svf', 'run_utci_tiles', 'build_inputs', 'build_wind_ext_coeff'])
def test_public_argument_contract(name):
    signature = inspect.signature(getattr(solweig_light, name))
    expected = CONTRACT[name]['parameters']
    assert list(signature.parameters) == [parameter['name'] for parameter in expected]
    for record, actual in zip(expected, signature.parameters.values()):
        assert actual.kind.name.lower() == record['kind']
        default = inspect.Parameter.empty if record['default'] is None else record['default_value']
        if isinstance(default, dict):
            assert default == {'expression': 'tuple(range(0, 360, 30))'}
            default = tuple(range(0, 360, 30))
        assert actual.default == default, actual.name
