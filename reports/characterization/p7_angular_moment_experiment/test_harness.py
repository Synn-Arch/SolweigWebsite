import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[3]
SPEC=importlib.util.spec_from_file_location('angular_harness',ROOT/'tools/run_p7_angular_moment.py')
HARNESS=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(HARNESS)
PROTOCOL=ROOT/'reports/characterization/p7_angular_moment_experiment/paired_protocol_v3.json'


def test_dry_validation_frozen_schedule_and_balance():
    result=HARNESS.validate_protocol(PROTOCOL)
    assert len(result['schedule'])==40
    for key in {(x['case'],x['regime'],x['budget']) for x in result['schedule']}:
        first=[x['order'][0] for x in result['schedule'] if (x['case'],x['regime'],x['budget'])==key]
        assert sorted((first.count('baseline'),first.count('candidate')))==[2,3]


def test_execution_is_blocked_before_lead_review():
    with pytest.raises(ValueError,match='reviewed_frozen'):
        HARNESS.validate_protocol(PROTOCOL,for_execution=True)


def pairs(candidate_seconds):
    result=[]
    for case in ('small','repeated_block_256'):
        for regime in (HARNESS.REGIME_FIRST,HARNESS.REGIME_WARM):
            for budget in (1,4):
                values=candidate_seconds(case,regime,budget)
                for repetition,value in enumerate(values):
                    result.append({'case':case,'regime':regime,'budget':budget,'repetition':repetition,
                                   'passed':True,'baseline_seconds':10.,'candidate_seconds':value})
    return result


def protocol():
    return {'matrix':{'scenes':['small','repeated_block_256'],
        'regimes':[HARNESS.REGIME_FIRST,HARNESS.REGIME_WARM],'native_budgets':[1,4],'repetitions':5}}


def test_ratio_direction_slower_fails_and_faster_passes():
    slow=HARNESS.summarize_pairs(pairs(lambda *_:[11.]*5),protocol())
    assert all(x['median_ratio']==1.1 for x in slow)
    assert not HARNESS.angular_acceptance(slow)['passed']
    fast=HARNESS.summarize_pairs(pairs(lambda *_:[9.5]*5),protocol())
    assert all(x['median_ratio']==.95 for x in fast)
    accepted=HARNESS.angular_acceptance(fast)
    assert accepted['ratio_definition']=='candidate_elapsed / baseline_elapsed' and accepted['passed']


def test_priority_requires_four_of_five_and_failed_group_never_crashes():
    def three_of_five(case,regime,budget):
        return [9.4,9.5,9.6,10.1,10.2] if case=='repeated_block_256' and regime==HARNESS.REGIME_WARM else [9.5]*5
    groups=HARNESS.summarize_pairs(pairs(three_of_five),protocol())
    assert not HARNESS.angular_acceptance(groups)['passed']
    groups[-1].update(subgroup_passed=False,median_ratio=None,raw_paired_ratios=[])
    assert HARNESS.angular_acceptance(groups)['passed'] is False
