"""Original thermal delay threshold behavior; no candidate-generated oracle."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from solweig_light.radiation.engine import TsWaveDelay_2015a

ROOT=Path(__file__).resolve().parents[1]/'reference/delay_original_cpu'
MANIFEST=json.loads((ROOT/'manifest.json').read_text())

@pytest.mark.parametrize('case',MANIFEST['cases'],ids=lambda c:c['name'])
def test_original_delay_thresholds(case):
    for field in ('input','output'):
        assert hashlib.sha256((ROOT/case[field]).read_bytes()).hexdigest()==case[field+'_sha256']
    with np.load(ROOT/case['input']) as source:
        values={k:v.copy() if v.ndim else v.item() for k,v in source.items()}
    result=TsWaveDelay_2015a(**values)
    with np.load(ROOT/case['output']) as expected:
        for index,actual in enumerate(result):
            target=expected[str(index)]
            assert np.asarray(actual).shape==target.shape
            np.testing.assert_array_equal(np.isnan(actual),np.isnan(target))
            if index==1:
                np.testing.assert_array_equal(actual,target)
            else:
                # Stricter temperature gate also covers flux-valued use.
                np.testing.assert_allclose(actual,target,rtol=0,atol=.01,equal_nan=True)
