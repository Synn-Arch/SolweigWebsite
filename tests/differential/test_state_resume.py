"""Numerical state handoff only; this is not the P6 persistent restart API."""
from copy import deepcopy
import pickle

import numpy as np
import pytest

from solweig_light.models import SimulationState, STATE_NAMES
from solweig_light.radiation import engine
from test_radiation_reference import MANIFEST, load_input, compare

INPUTS = [e for e in MANIFEST['events'] if e['boundary'] == 'input']
OUTPUTS = {e['timestep']: e for e in MANIFEST['events']
           if e['boundary'] == 'output' and e['function'] == 'Solweig_2022a_calc'}


def advance(event, state):
    inputs = load_input(event, compact=True)
    if state is None:
        state = SimulationState(**{k: inputs[k] for k in STATE_NAMES},
                                Twater=inputs['Twater'])
    # Forcing and daily Twater policy are supplied by the original timeline.
    state.Twater = inputs['Twater']
    inputs.update(state.engine_arguments())
    inputs['Twater'] = state.Twater
    with np.errstate(all='ignore'):
        outputs = engine.Solweig_2022a_calc(**inputs)
    reference = OUTPUTS[event['timestep']]
    compare(reference, outputs)
    state.accept(dict(zip(reference['fields'], outputs, strict=True)))
    return state, outputs


@pytest.mark.parametrize('split', [1, 6, 12, 18, 23])
def test_complete_state_roundtrip_resume(split):
    uninterrupted = []
    state = None
    for event in INPUTS:
        state, output = advance(event, state)
        uninterrupted.append(deepcopy(output))
    state = None
    for event in INPUTS[:split]:
        state, _ = advance(event, state)
    # Private, test-generated serialization preserves scalar types and arrays.
    # Production persistence, corruption handling and output recovery are P6.
    packet = {'cursor': split, 'state': state,
              'forcing_context': deepcopy(INPUTS[split:])}
    restored = pickle.loads(pickle.dumps(packet, protocol=5))
    resumed = restored['state']
    for name in (*STATE_NAMES, 'Twater'):
        before, after = getattr(state, name), getattr(resumed, name)
        assert type(before) is type(after)
        if isinstance(before, np.ndarray):
            assert before.dtype == after.dtype
            assert not np.shares_memory(before, after)
        np.testing.assert_array_equal(before, after)
    assert restored['cursor'] == split
    for index, event in enumerate(restored['forcing_context'], start=split):
        resumed, actual = advance(event, resumed)
        for observed, expected in zip(actual, uninterrupted[index], strict=True):
            if expected is None:
                assert observed is None
            else:
                np.testing.assert_array_equal(observed, expected)
