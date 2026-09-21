"""Independent physical bounds for the default full-patch SVF calculation.

Authored during the P7 benchmark quiet window; execution is intentionally deferred.
"""
import numpy as np
import pytest

from solweig_light.geometry.svf import svf_calculator, svf_calculator_compact


PATCH_OPTION = 2
PATCH_COUNT = 153
MAX_FLOAT32_UPDATES = 1830
FLOAT32_UNIT_ROUNDOFF = 2.0 ** -24
FLOAT32_ALLOWANCE = 2.5e-4
SVF_FIELD_COUNT = 15
VEGETATION_OPEN_FIELDS = (1, 3, 4, 6, 7, 9, 10, 11, 13, 14)


def _calculation(kind):
    rows, cols = 9, 13
    a = np.zeros((rows, cols), dtype=np.float32)
    vegdem = np.zeros_like(a)
    vegdem2 = np.zeros_like(a)
    bush = np.zeros_like(a)
    if kind == "obstructed":
        a[3:6, 3:7] = np.float32(8)
        vegdem[2:7, 9:11] = np.float32(12)
        vegdem2[2:7, 9:11] = np.float32(3)
        bush[1:3, 1:4] = np.float32(4)
        vegdem[1:3, 1:4] = np.float32(4)
        amaxvalue = np.float32(12)
    else:
        assert kind == "open"
        amaxvalue = np.float32(0)
    return dict(
        patch_option=PATCH_OPTION,
        amaxvalue=amaxvalue,
        a=a,
        vegdem=vegdem,
        vegdem2=vegdem2,
        bush=bush,
        scale=np.float32(1),
    )


def _assert_bounded_svf(outputs):
    assert len(outputs) == 19
    for visibility in outputs[15:18]:
        assert visibility.shape == (9, 13, PATCH_COUNT)
    svf_fields = outputs[:SVF_FIELD_COUNT]
    svftotal = outputs[18]
    for field in (*svf_fields, svftotal):
        assert field.dtype == np.float32
        assert field.shape == (9, 13)
        assert np.isfinite(field).all()
        assert np.min(field) >= -FLOAT32_ALLOWANCE
        assert np.max(field) <= 1.0 + FLOAT32_ALLOWANCE


@pytest.mark.parametrize("calculator", (svf_calculator, svf_calculator_compact))
@pytest.mark.parametrize("kind", ("open", "obstructed"))
def test_default_full_patch_svf_fields_are_physically_bounded(calculator, kind):
    outputs = calculator(**_calculation(kind))
    _assert_bounded_svf(outputs)
    if kind == "open":
        for index in VEGETATION_OPEN_FIELDS:
            np.testing.assert_allclose(
                outputs[index], np.float32(1), rtol=0, atol=FLOAT32_ALLOWANCE
            )


def test_float32_allowance_is_fixed_from_default_patch_operation_count():
    # Option 2 has 153 patches. Its eight annuli perform
    # 12*(31+30+28+24+19+13+7) + 6*1 = 1830 field updates.
    assert PATCH_COUNT == 31 + 30 + 28 + 24 + 19 + 13 + 7 + 1
    assert MAX_FLOAT32_UPDATES == 12 * (PATCH_COUNT - 1) + 6
    gamma_two_n = (
        2 * MAX_FLOAT32_UPDATES * FLOAT32_UNIT_ROUNDOFF
        / (1 - 2 * MAX_FLOAT32_UPDATES * FLOAT32_UNIT_ROUNDOFF)
    )
    assert gamma_two_n < FLOAT32_ALLOWANCE
    assert FLOAT32_ALLOWANCE < 3e-4
