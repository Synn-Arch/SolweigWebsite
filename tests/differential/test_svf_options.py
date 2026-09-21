"""Full original SVF option contracts, including upstream-failing option 4."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from solweig_light.geometry.svf import svf_calculator, svf_calculator_compact

REFERENCE=Path(__file__).resolve().parents[1]/'reference/svf_original_cpu'
MANIFEST=json.loads((REFERENCE/'manifest.json').read_text())


@pytest.mark.parametrize('case',MANIFEST['cases'],ids=lambda case:case['name'])
@pytest.mark.parametrize('compact', [False, True])
def test_original_full_svf_option(case, compact):
    calculate = svf_calculator_compact if compact else svf_calculator
    path=REFERENCE/case['input_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==case['input_sha256']
    with np.load(path,allow_pickle=False) as archive:arguments={name:archive[name] for name in archive.files}
    arguments['scale']=float(arguments['scale']);arguments['patch_option']=int(arguments['patch_option'])
    if case['status']=='original_failure':
        assert case['option']==4 and case['exception_type']=='TypeError'
        with pytest.raises(TypeError):calculate(**arguments)
        return
    outputs=calculate(**arguments)
    path=REFERENCE/case['output_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==case['output_sha256']
    with np.load(path,allow_pickle=False) as archive:
        assert len(outputs)==len(archive.files)==19
        for name,actual in zip(archive.files,outputs):
            expected=archive[name]
            assert actual.shape==expected.shape and actual.dtype==expected.dtype
            if compact and name.endswith('mat'):
                for patch in range(expected.shape[2]):
                    np.testing.assert_array_equal(actual[:, :, patch], expected[:, :, patch], err_msg=name)
                with pytest.raises(TypeError):np.asarray(actual)
                continue
            for mask in (np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(mask(actual),mask(expected))
            if name.endswith('mat'):np.testing.assert_array_equal(actual,expected,err_msg=name)
            else:np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6,err_msg=name)
