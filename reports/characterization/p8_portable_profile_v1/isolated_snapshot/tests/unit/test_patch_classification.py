"""Exact classification masks against the retained scalar/NumPy engine."""
import numpy as np
import pytest
from solweig_light.radiation import engine as e, patch_radiation as p


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('solar_type',[float,np.float32,np.float64,lambda x:np.array(x,dtype=np.float32),lambda x:np.array(x,dtype=np.float64)])
@pytest.mark.parametrize('angle',[0.,-0.,35.,-10.,np.nan])
def test_mixed_scalar_masks(dtype,solar_type,angle):
    geometry=p.patch_geometry(np.array([[0,0],[6,90],[35,180],[90,270],[35,360]],np.float32))
    altitude=np.array(angle,dtype=dtype)
    azimuth=solar_type(90.000001)
    edge=np.float32(np.deg2rad(35))
    asvf=np.array([0.,-0.,edge,np.nextafter(edge,np.float32(-np.inf)),np.nextafter(edge,np.float32(np.inf)),np.nan,np.inf,-np.inf],dtype=dtype).reshape(2,4)
    for active in (None,np.array([1,0,1,0,1],bool),np.zeros(5,bool)):
        with np.errstate(all='ignore'):
            sun,shade=p._classes(altitude,azimuth,geometry,asvf,1,7,active)
            expected=[np.zeros_like(sun),np.zeros_like(shade)]
            for patch in range(5):
                if active is None or active[patch]:
                    masks=e.shaded_or_sunlit(altitude,azimuth,geometry.altitude[patch],geometry.azimuth[patch],asvf.reshape(-1)[1:7,None])
                    for dest,mask in zip(expected,masks):dest[:,patch]=mask[:,0]
        np.testing.assert_array_equal(sun,expected[0])
        np.testing.assert_array_equal(shade,expected[1])


def test_scalar_exception_and_inactive_patches():
    geometry=p.patch_geometry(np.array([[6,0],[90,180]],np.float32))
    field=np.zeros((2,3),np.float32)
    with pytest.raises(TypeError,match='tensor-origin'):
        p._classes(20.,0.,geometry,field,0,6)
    for mask in p._classes(20.,0.,geometry,field,0,6,np.zeros(2,bool)):
        assert not mask.any()


def test_strict_equal_boundary():
    geometry=p.patch_geometry(np.array([[0,0],[0,180]],np.float32))
    sun,shade=p._classes(np.array(0.,np.float32),0.,geometry,np.array([[0.,-0.]],np.float32),0,2)
    assert not sun.any()
    assert not shade.any()


@pytest.mark.parametrize('solar_dtype',[np.float32,np.float64])
@pytest.mark.parametrize('azimuth_dtype',[np.float32,np.float64])
def test_exact_nextafter_thresholds(solar_dtype,azimuth_dtype):
    from dataclasses import replace
    geometry=p.patch_geometry(np.array([[35,180]],np.float32))
    altitude=np.array(35.000001,dtype=solar_dtype)
    azimuth=azimuth_dtype(90.00001)
    field=np.array([[0.,-0.,.3,.7,np.nan]],np.float32)
    difference=np.abs(e._operate(np.subtract,azimuth,geometry.azimuth[0]))
    xi=np.cos(e._operate(np.multiply,difference,e._divide(np.pi,180.)))
    yi=e._operate(np.multiply,e._operate(np.multiply,2,xi),np.tan(e._operate(np.multiply,altitude,e._divide(np.pi,180.))))
    threshold=e._operate(np.multiply,np.arctan(e._operate(np.add,np.tan(field),np.where(yi>0,0.,yi))),e._divide(180.,np.pi))[0,3]
    for boundary in (threshold,np.nextafter(threshold,np.float32(-np.inf)),np.nextafter(threshold,np.float32(np.inf))):
        custom=replace(geometry,altitude=np.array([boundary],np.float32))
        actual=p._classes(altitude,azimuth,custom,field,0,field.size)
        expected=e.shaded_or_sunlit(altitude,azimuth,boundary,custom.azimuth[0],field.reshape(-1,1))
        for result,reference in zip(actual,expected):np.testing.assert_array_equal(result,reference)
