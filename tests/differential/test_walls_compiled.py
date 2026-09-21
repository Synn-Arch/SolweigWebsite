"""Exact compiled wall/aspect comparisons against original CPU fixtures."""
import hashlib
import itertools
import json
from pathlib import Path

import numba
import numpy as np
import pytest
from solweig_light.geometry.walls import (
    findwalls_serial, findwalls_parallel,
    filter1Goodwin_as_aspect_v3_serial, filter1Goodwin_as_aspect_v3_parallel,
    filter_tables, _cached_filters,
)

REFERENCE=Path(__file__).parents[1]/'reference'
CASES=[]
FILTERS=[]
for directory in ('walls_original_cpu','walls_expanded_original_cpu'):
    root=REFERENCE/directory
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['evidence_class']=='original_upstream_cpu'
    assert manifest['source_commit']=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
    assert manifest['repair'] is None and not manifest['candidate_outputs']
    CASES.extend((root,case) for case in manifest['fixtures'])
    FILTERS.extend((root,table) for table in manifest.get('filters',[]))


def assert_exact(value,expected):
    assert value.dtype==expected.dtype
    np.testing.assert_array_equal(value,expected)
    # array equality considers signed zeros equal; check them independently.
    finite=np.isfinite(expected)
    np.testing.assert_array_equal(np.signbit(value[finite]),np.signbit(expected[finite]))


@pytest.mark.parametrize('root,case',CASES,ids=[f'{r.name}/{c["file"]}' for r,c in CASES])
@pytest.mark.parametrize('threads',[None,1,4,10],ids=['serial','parallel1','parallel4','parallel10'])
def test_original_stencil_and_aspect(root,case,threads):
    path=root/case['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==case['sha256']
    with np.load(path,allow_pickle=False) as fixture:
        dsm=fixture['dsm'];original=dsm.copy()
        scale=float(fixture['scale']);threshold=float(fixture['walllimit'])
        previous=numba.get_num_threads()
        try:
            with np.errstate(invalid='ignore',over='ignore'):
                if threads is None:
                    walls=findwalls_serial(dsm,threshold)
                    aspect=filter1Goodwin_as_aspect_v3_serial(walls,scale,dsm)
                else:
                    numba.set_num_threads(threads)
                    walls=findwalls_parallel(dsm,threshold)
                    aspect=filter1Goodwin_as_aspect_v3_parallel(walls,scale,dsm)
            assert_exact(walls,fixture['walls'])
            assert_exact(aspect,fixture['aspect'])
            assert_exact(dsm,original)
        finally:
            numba.set_num_threads(previous)


@pytest.mark.parametrize('root,table',FILTERS,ids=[c['file'] for _,c in FILTERS])
def test_all_180_sparse_filters_match_original_capture(root,table):
    path=root/table['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==table['sha256']
    with np.load(path,allow_pickle=False) as fixture:
        tables=filter_tables(table['scale'])
        half=tables.filtersize//2
        scores=np.zeros((180,tables.filtersize,tables.filtersize),dtype=np.float64)
        sides=np.zeros_like(scores)
        for angle in range(180):
            start,end=tables.score_pointers[angle:angle+2]
            scores[angle,tables.score_rows[start:end]+half,tables.score_cols[start:end]+half]=tables.score_coefficients[start:end]
            start,middle,end=tables.side_pointers[angle]
            sides[angle,tables.side_rows[start:middle]+half,tables.side_cols[start:middle]+half]=1
            sides[angle,tables.side_rows[middle:end]+half,tables.side_cols[middle:end]+half]=2
        assert_exact(scores,fixture['score'])
        assert_exact(sides,fixture['sides'])


def test_filter_cache_is_bounded_exact_and_immutable():
    first=filter_tables(1.)
    assert filter_tables(1.) is first
    assert filter_tables(np.nextafter(1.,2.)) is not first
    assert _cached_filters.cache_info().maxsize==8
    for name,array in vars(first).items():
        if isinstance(array,np.ndarray):
            assert not array.flags.writeable
            with pytest.raises(ValueError):array.setflags(write=True)
            with pytest.raises(ValueError):array.flat[0]=0


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('threads',[None,1,4,10])
def test_unthresholded_signed_zero_cross_reduction(dtype,threads):
    # A positive wall threshold hides zero signs. Exercise the accepted
    # nonpositive threshold against the exact native NumPy cross reduction.
    previous=numba.get_num_threads()
    try:
        if threads is not None:numba.set_num_threads(threads)
        kernel=findwalls_serial if threads is None else findwalls_parallel
        for values in itertools.product([0.,-0.],repeat=5):
            dsm=np.zeros((3,3),dtype=dtype)
            dsm[0,1],dsm[1,0],dsm[1,2],dsm[2,1],dsm[1,1]=values
            expected=np.zeros((3,3),dtype=np.float64)
            expected[1,1]=np.float64(np.max(np.asarray(values[:4],dtype=dtype)))-np.float64(dsm[1,1])
            assert_exact(kernel(dsm,-1.),expected)
    finally:
        numba.set_num_threads(previous)
