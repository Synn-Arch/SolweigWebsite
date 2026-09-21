"""Chronological state ownership: independent initialization and no history."""
import weakref
import numpy as np
from solweig_light.models import SimulationState, STATE_NAMES, OutputPlan, ForcingTimeline


def test_state_initial_maps_have_independent_ownership():
    zero=np.zeros((3,4),dtype=np.float32)
    timestep=np.float64(.0416666667)
    state=SimulationState.initial(zero,timestep)
    assert state.timestepdec is timestep
    arrays=[getattr(state,name) for name in STATE_NAMES if name.startswith('Tg')]
    assert len(arrays)==6
    for array in arrays:
        np.testing.assert_array_equal(array,zero)
        assert array.dtype==np.float32
        assert not np.shares_memory(array,zero)
    for index,array in enumerate(arrays):
        for other in arrays[index+1:]:assert not np.shares_memory(array,other)
    arrays[0][0,0]=9
    assert arrays[1][0,0]==0
    assert state.Twater==[]


def test_state_accepts_exact_scalar_array_results_and_releases_old_maps():
    state=SimulationState.initial(np.zeros((2,3),dtype=np.float32),np.float64(.125))
    old=weakref.ref(state.Tgmap1)
    fields={name: np.full((2,3),index,dtype=np.float32) if name.startswith('Tg') else np.float64(index+.25)
            for index,name in enumerate(STATE_NAMES)}
    water=np.float64(18.75)
    state.Twater=water
    state.accept(fields)
    assert old() is None
    for name in STATE_NAMES:assert getattr(state,name) is fields[name]
    assert state.Twater is water
    arguments=state.engine_arguments()
    assert tuple(arguments)==STATE_NAMES
    assert 'Twater' not in arguments
    for name in STATE_NAMES:assert arguments[name] is fields[name]


def test_output_plan_retains_utci_and_wbgt_dependency():
    plan=OutputPlan.from_flags({'save_wbgt':True,'save_tmrt':True,'save_svf':True})
    assert plan.requested==('UTCI','TMRT','WBGT')
    assert plan.needs_wetbulb and plan.save_svf
    default=OutputPlan.from_flags({})
    assert default.requested==('UTCI',)
    assert not default.needs_wetbulb and not default.save_svf


def test_uniform_forcing_is_scalar_without_conversion():
    met=np.zeros((2,25),dtype=np.float64)
    meteorology={'Ta':met[:,11],'RH':met[:,10]}
    solar=np.zeros((1,2),dtype=np.float64)
    timeline=ForcingTimeline(met,meteorology,met[:,9],met[:,23],met[:,24],met[:,1],
                             solar,solar,solar,solar,met[:,1],solar,{},None)
    assert timeline.met is met
    values=timeline.at(0)
    assert set(values)=={'Ta','RH'}
    assert all(isinstance(value,np.float64) and value.ndim==0 for value in values.values())
