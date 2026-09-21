"""48-hour original-driver comparison; numerical handoff, not persistent restart."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from solweig_light.models import SimulationState, STATE_NAMES
from solweig_light.radiation import engine
from solweig_light.geometry.visibility import PackedVisibility, LazyDiffVisibility

ROOT = Path(__file__).resolve().parents[1] / 'reference/state_sequence_original_cpu'
MANIFEST = json.loads((ROOT / 'boundaries/manifest.json').read_text())
PROTOCOL = json.loads((ROOT.parents[2] / 'benchmarks/protocols/comparison_v1.json').read_text())


def load(event):
    path = ROOT / 'boundaries' / event['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == event['sha256']
    values = {}
    with np.load(path) as archive:
        for name, spec in event['fields'].items():
            if '/' in name:
                continue
            if spec['kind'] == 'dict':
                values[name] = {key: archive[name + '/' + key].item() for key in spec['keys']}
            elif spec['kind'] == 'none':
                values[name] = None
            elif spec['kind'] == 'list':
                assert spec['length'] == 0
                values[name] = []
            else:
                value = archive[name].copy()
                typ = spec['original_python_type']
                values[name] = value.item() if typ.startswith('builtins.') else value[()] if typ.startswith('numpy.') and spec['kind'] == 'scalar' else value
    for name in ('shmat', 'vegshmat', 'vbshvegshmat'):
        values[name] = PackedVisibility.from_dense(values[name])
    values['diffsh'] = LazyDiffVisibility(values['shmat'], values['vegshmat'])
    return values


def compare(event, result):
    with np.load(ROOT / 'boundaries' / event['path']) as archive:
        for name, actual in zip(event['fields'], result, strict=True):
            if event['fields'][name]['kind'] == 'none':
                assert actual is None
                continue
            actual = np.asarray(actual)
            expected = archive[name]
            assert actual.shape == expected.shape, name
            for mask in (np.isnan, np.isposinf, np.isneginf):
                np.testing.assert_array_equal(mask(actual), mask(expected), err_msg=name)
            budget = PROTOCOL['field_rules']['Solweig_2022a_calc/' + name]
            if 'columns' in budget:
                np.testing.assert_array_equal(actual[:, :2], expected[:, :2])
                actual, expected = actual[:, 2], expected[:, 2]
                budget = budget['columns'][2]
            if budget.get('rule', '').startswith('exact'):
                np.testing.assert_array_equal(actual, expected, err_msg=name)
            else:
                finite = np.isfinite(expected)
                limit = budget.get('max_abs', budget.get('atol', 0)) + budget.get('rtol', 0) * np.abs(expected[finite])
                error = np.abs(actual[finite] - expected[finite])
                assert np.all(error <= limit), f"step {event['timestep']} {name}: max error {error.max()}"


@pytest.mark.parametrize('split', [6, 18, 23, 24, 25, 30, 42])
def test_original_two_days_with_midnight_state_handoff(split):
    inputs = [e for e in MANIFEST['events'] if e['boundary'] == 'input']
    outputs = {e['timestep']: e for e in MANIFEST['events'] if e['boundary'] == 'output'}
    assert len(inputs) == 48
    state = None
    for event in inputs:
        values = load(event)
        if state is None:
            state = SimulationState(**{name: values[name] for name in STATE_NAMES}, Twater=values['Twater'])
        if event['timestep'] == split:
            # Test-only handoff: production checkpoint validation is a P6 gate.
            packet = {'cursor': split, 'state': state, 'forcing_context': inputs[split:]}
            restored = pickle.loads(pickle.dumps(packet, protocol=5))
            assert restored['cursor'] == event['timestep']
            assert restored['forcing_context'] == inputs[split:]
            for name in (*STATE_NAMES, 'Twater'):
                before, after = getattr(state, name), getattr(restored['state'], name)
                assert type(before) is type(after), name
                if isinstance(before, np.ndarray):
                    assert before.dtype == after.dtype
                    assert not np.shares_memory(before, after)
                np.testing.assert_array_equal(before, after, err_msg=name)
            state = restored['state']
        forcing_water, forcing_ci = values['Twater'], values['CI']
        values.update(state.engine_arguments())
        # Exact original-driver daily resets supplied by captured forcing context.
        state.Twater = forcing_water
        values['Twater'] = state.Twater
        if float(values['dectime']) % 1 == 0:
            values['CI'] = forcing_ci
        with np.errstate(all='ignore'):
            result = engine.Solweig_2022a_calc(**values)
        compare(outputs[event['timestep']], result)
        state.accept(dict(zip(outputs[event['timestep']]['fields'], result, strict=True)))
