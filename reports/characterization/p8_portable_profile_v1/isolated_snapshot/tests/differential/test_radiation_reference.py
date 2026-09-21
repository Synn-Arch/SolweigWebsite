"""Replay original CPU call boundaries without importing the upstream package."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from solweig_light.radiation import engine, patch_radiation

ROOT = Path(__file__).resolve().parents[1] / 'reference/small_original_cpu/boundaries'
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())
PROTOCOL = json.loads((ROOT.parents[3] / 'benchmarks/protocols/comparison_v1.json').read_text())

def load_input(event, compact=False):
    with np.load(ROOT / event['path']) as archive:
        values = {}
        for name, spec in event['fields'].items():
            if '/' in name:
                continue
            kind = spec['kind']
            if kind == 'dict':
                values[name] = {key: archive[name + '/' + key].item() for key in spec['keys']}
            elif kind == 'list':
                assert spec['length'] == 0
                values[name] = []
            elif kind == 'none':
                values[name] = None
            else:
                value = archive[name].copy()
                # The collector records dtype but does not distinguish Python
                # floats from NumPy scalars. Recover driver types from source:
                # solar/time values are indexed from float64 NumPy arrays.
                if name in {'altitude', 'azimuth', 'zen', 'dectime', 'altmax'}:
                    values[name] = value[()]
                elif kind == 'array' or value.dtype == np.float32 or name in {'jday', 'Ta', 'RH', 'radG', 'radD', 'radI', 'P', 'amaxvalue'}:
                    values[name] = value
                else:
                    values[name] = value.item()
        if compact:
            from solweig_light.geometry.visibility import PackedVisibility, LazyDiffVisibility
            for name in ('shmat', 'vegshmat', 'vbshvegshmat'):
                values[name] = PackedVisibility.from_dense(values[name])
            values['diffsh'] = LazyDiffVisibility(values['shmat'], values['vegshmat'])
        return values

def compare(event, output, delay_index=0):
    with np.load(ROOT / event['path']) as archive:
        for name, actual in zip(event['fields'], output):
            spec = event['fields'][name]
            if spec['kind'] == 'none':
                assert actual is None
                continue
            expected = archive[name]
            actual = np.asarray(actual)
            context = f"step {event['timestep']} {event['function']}/{name}"
            assert actual.shape == expected.shape, context
            for mask in (np.isnan, np.isposinf, np.isneginf):
                np.testing.assert_array_equal(mask(actual), mask(expected), err_msg=context)
            budget = PROTOCOL['field_rules'][event['function'] + '/' + name]
            if 'columns' in budget:
                np.testing.assert_array_equal(actual[:, :2], expected[:, :2], err_msg=context)
                actual, expected = actual[:, 2], expected[:, 2]
                budget = budget['columns'][2]
            if 'rule' in budget and budget['rule'].startswith('exact'):
                np.testing.assert_array_equal(actual, expected, err_msg=context)
                continue
            if 'flux_atol' in budget:
                budget = {'max_abs': budget['temperature_max_abs']} if delay_index == 5 else {'atol': budget['flux_atol'], 'rtol': budget['flux_rtol']}
            atol = budget.get('max_abs', budget.get('atol', 0))
            rtol = budget.get('rtol', 0)
            finite = np.isfinite(expected)
            error = np.abs(actual[finite] - expected[finite])
            limits = atol + rtol * np.abs(expected[finite])
            assert np.all(error <= limits), f"{context}: max error {error.max()} at flattened finite index {error.argmax()}"

@pytest.mark.parametrize('event', [event for event in MANIFEST['events'] if event['boundary'] == 'input'], ids=lambda event: f"step-{event['timestep']}")
@pytest.mark.parametrize('compact', [False, True])
def test_original_cpu_radiation_boundaries(event, monkeypatch, compact):
    references = [item for item in MANIFEST['events'] if item['boundary'] == 'output' and item['timestep'] == event['timestep']]
    captured = []
    for name in {item['function'] for item in references} - {'Solweig_2022a_calc'}:
        # The compiled longwave driver calls its module-local ordered sweep.
        # Observe that sweep directly so all original intermediate fields remain
        # checked; do not drop the nested boundary from the oracle.
        owner = patch_radiation if name == 'define_patch_characteristics' else engine
        original = getattr(owner, name)
        def wrapper(*args, _original=original, _name=name, **kwargs):
            result = _original(*args, **kwargs)
            captured.append((_name, deepcopy(result)))
            return result
        monkeypatch.setattr(owner, name, wrapper)
    with np.errstate(all='ignore'):
        result = engine.Solweig_2022a_calc(**load_input(event, compact=compact))
    captured.append(('Solweig_2022a_calc', result))
    assert [name for name, _ in captured] == [item['function'] for item in references]
    delay_index = 0
    for reference, (_, output) in zip(references, captured):
        compare(reference, output, delay_index)
        if reference['function'] == 'TsWaveDelay_2015a':
            delay_index += 1

@pytest.mark.parametrize('compact', [False, True])
def test_carried_state_across_original_day_night_sequence(compact):
    carried = {}
    for event in MANIFEST['events']:
        if event['boundary'] != 'input':
            continue
        inputs = load_input(event, compact=compact)
        inputs.update(carried)
        with np.errstate(all='ignore'):
            output = engine.Solweig_2022a_calc(**inputs)
        reference = next(item for item in MANIFEST['events'] if item['boundary'] == 'output' and item['function'] == 'Solweig_2022a_calc' and item['timestep'] == event['timestep'])
        compare(reference, output)
        names = list(reference['fields'])
        carried = {name: deepcopy(output[names.index(name)]) for name in ('firstdaytime', 'timeadd', 'timestepdec', 'Tgmap1', 'Tgmap1E', 'Tgmap1S', 'Tgmap1W', 'Tgmap1N', 'CI', 'TgOut1')}
