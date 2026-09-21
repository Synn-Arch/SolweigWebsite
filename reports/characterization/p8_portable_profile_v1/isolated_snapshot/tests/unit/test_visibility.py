"""Lossless visibility bits, bounded decoding and streamed legacy interchange."""
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import weakref
import zipfile

import numpy as np
import pytest
from solweig_light.geometry.visibility import (CODEC_VERSION,CODEBOOK_BITS,LEGACY_MEMBERS,
    PackedVisibility,VisibilityBuilder,LazyDiffVisibility,export_visibility_npz,import_visibility_npz)


def same_bits(actual,expected):
    assert actual.shape==expected.shape
    assert actual.dtype==np.float32
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def mixed_cube(shape=(5,7,5)):
    cube=np.empty(shape,np.float32)
    cube[:,:,0]=np.resize(np.array([0,1],np.float32),shape[:2])
    cube[:,:,1]=np.resize(np.array([0,1,2],np.float32),shape[:2])
    weird=np.array([0x80000000,0x3e4ccccd,0x7fc00001,0x7fc00234,0x7f800000,0xff800000,0x7fa00001,0xffc00001],np.uint32).view(np.float32)
    cube[:,:,2]=np.resize(weird,shape[:2])
    cube[:,:,3]=1
    cube[:,:,4]=np.resize(np.array([2,1],np.float32),shape[:2])
    return cube


@pytest.mark.parametrize('shape',[(1,1,5),(2,3,5),(5,7,5),(9,13,5)])
@pytest.mark.parametrize('budget',[128,256,1024])
def test_binary_ternary_and_raw_partial_boundaries(shape,budget):
    source=mixed_cube(shape);encoded=PackedVisibility.from_dense(source,max_workspace_bytes=budget)
    assert encoded.shape==shape and encoded.dtype==np.float32 and encoded.ndim==3
    assert encoded.codec_version==CODEC_VERSION==1
    assert encoded.modes[0]=='binary' and encoded.modes[2]=='raw'
    # Shape1x1 contains only +0 in patch1, so it needs only one bit.
    assert encoded.modes[1]==('binary' if shape[:2]==(1,1) else 'ternary')
    for patch in range(shape[2]):
        same_bits(encoded[:,:,patch],source[:,:,patch])
        same_bits(encoded[:,:,patch-5],source[:,:,patch])
        for start in range(source[:,:,patch].size+1):
            stop=min(source[:,:,patch].size,start+3)
            same_bits(encoded.decode_pixels(patch,start,stop),source[:,:,patch].ravel()[start:stop])
    same_bits(encoded.to_dense(),source)
    assert encoded.nbytes==encoded.encoded_nbytes==encoded.storage_nbytes


def test_known_codebooks_are_exact_uint32_and_packed_storage():
    assert CODEBOOK_BITS==(0,0x3f800000,0x40000000)
    source=np.zeros((16,16,3),np.float32);source[:,:,0]=1;source[:,:,1]=2
    encoded=PackedVisibility.from_dense(source)
    assert encoded.modes==('binary','ternary','binary')
    assert encoded.nbytes==32+64+32
    assert all(isinstance(patch.payload,bytes) for patch in encoded._patches)
    with pytest.raises((AttributeError,TypeError)):encoded.shape=(1,1,1)
    source[:]=.3
    assert np.all(encoded[:,:,0]==1) and np.all(encoded[:,:,1]==2)
    decoded=encoded[:,:,0];decoded[:]=.7
    assert np.all(encoded[:,:,0]==1)


def test_builder_releases_completed_plane_and_freezes_order():
    builder=VisibilityBuilder((3,5,2),max_workspace_bytes=128)
    plane=np.ones((3,5),np.float32);reference=weakref.ref(plane)
    builder.append(plane);del plane;gc.collect()
    assert reference() is None and builder.patches_added==1
    with pytest.raises(ValueError,match='every patch'):builder.finish()
    builder.append(np.full((3,5),2,np.float32));encoded=builder.finish()
    assert builder.finish() is encoded
    np.testing.assert_array_equal(encoded[:,:,0],np.ones((3,5),np.float32))
    np.testing.assert_array_equal(encoded[:,:,1],np.full((3,5),2,np.float32))
    with pytest.raises(RuntimeError,match='finished'):builder.append(np.zeros((3,5),np.float32))


def test_implicit_dense_and_patch_slice_requests_fail():
    encoded=PackedVisibility.from_dense(np.zeros((2,3,4),np.float32))
    with pytest.raises(TypeError,match='Implicit dense'):np.asarray(encoded)
    with pytest.raises(TypeError):encoded[:,:,:]
    with pytest.raises(IndexError):encoded[0]
    with pytest.raises(IndexError):encoded[:,:,4]
    with pytest.raises(TypeError):encoded[:,:,True]
    with pytest.raises(IndexError):encoded.decode_pixels(0,-1,1)


def test_lazy_diff_preserves_original_float32_operation_order():
    shadow=mixed_cube();vegetation=np.roll(mixed_cube(),1,axis=2)
    sh=PackedVisibility.from_dense(shadow);veg=PackedVisibility.from_dense(vegetation)
    lazy=LazyDiffVisibility(sh,veg)
    assert lazy.shape==shadow.shape and lazy.dtype==np.float32
    with pytest.raises(TypeError,match='Implicit dense'):np.asarray(lazy)
    with np.errstate(all='ignore'):
        expected=shadow-(np.float32(1)-vegetation)*np.float32(1-.03)
        for patch in range(shadow.shape[2]):
            same_bits(lazy[:,:,patch],expected[:,:,patch])
            same_bits(lazy.decode_pixels(patch,3,8),expected[:,:,patch].ravel()[3:8])


def test_big_endian_source_bit_patterns_are_lossless():
    native=mixed_cube()
    big=native.view(np.uint32).astype('>u4').view('>f4')
    encoded=PackedVisibility.from_dense(big,max_workspace_bytes=128)
    same_bits(encoded.to_dense(),native)


class SegmentOnly:
    """Fail if an exporter requests a full plane or implicit dense conversion."""
    def __init__(self,channel):self.channel=channel;self.shape=channel.shape;self.dtype=channel.dtype;self.maximum_pixels=0
    def __array__(self,*args,**kwargs):raise AssertionError('Exporter materialized a full cube')
    def __getitem__(self,index):raise AssertionError('Exporter decoded a complete plane')
    def decode_patch(self,index):raise AssertionError('Exporter decoded a complete plane')
    def decode_pixels(self,patch,start,stop):
        self.maximum_pixels=max(self.maximum_pixels,stop-start)
        return self.channel.decode_pixels(patch,start,stop)


@pytest.mark.parametrize('budget',[128,256,1024,4096])
@pytest.mark.parametrize('patches',[5,41])
def test_streamed_npz_exact_member_bits_schema_and_chunk_budget(tmp_path,budget,patches):
    original=np.resize(mixed_cube(),(11,13,patches)).astype(np.float32)
    encoded=PackedVisibility.from_dense(original,max_workspace_bytes=budget)
    tracked=[SegmentOnly(encoded) for _ in range(3)]
    target=tmp_path/'shadowmats_0_0.npz'
    stats=export_visibility_npz(target,*tracked,max_workspace_bytes=budget)
    assert stats['conservative_peak_numeric_workspace_bytes']<=budget
    assert all(channel.maximum_pixels*64<=budget for channel in tracked)
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist()==[name+'.npy' for name in LEGACY_MEMBERS]
        for name in LEGACY_MEMBERS:
            with archive.open(name+'.npy') as stream:
                version=np.lib.format.read_magic(stream)
                shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                assert version==(1,0) and shape==original.shape and not fortran and dtype==np.float32
    with np.load(target,allow_pickle=False) as dense:
        assert dense.files==list(LEGACY_MEMBERS)
        for name in LEGACY_MEMBERS:same_bits(dense[name],original)
    scratch=tmp_path/'scratch';scratch.mkdir()
    compact=import_visibility_npz(target,max_workspace_bytes=budget,scratch_dir=scratch)
    assert set(compact)==set(LEGACY_MEMBERS)
    for channel in compact.values():same_bits(channel.to_dense(),original)
    assert list(scratch.iterdir())==[]
    assert not list(tmp_path.glob('.shadowmats*'))


def test_concurrent_distinct_tile_exports_and_imports(tmp_path):
    a=mixed_cube();b=np.roll(a,1,axis=0)
    channels=[PackedVisibility.from_dense(array,max_workspace_bytes=128) for array in [a,b]]
    paths=[tmp_path/'shadowmats_0_0.npz',tmp_path/'shadowmats_100_0.npz']
    def run(index):
        export_visibility_npz(paths[index],*[channels[index]]*3,max_workspace_bytes=256)
        return import_visibility_npz(paths[index],max_workspace_bytes=128,scratch_dir=tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:results=list(executor.map(run,[0,1]))
    for result,source in zip(results,[a,b]):
        for channel in result.values():same_bits(channel.to_dense(),source)
    assert sorted(path.name for path in tmp_path.iterdir())==sorted(path.name for path in paths)


def test_interrupted_export_keeps_previous_file_and_cleans_temporary(tmp_path):
    encoded=PackedVisibility.from_dense(mixed_cube())
    target=tmp_path/'shadowmats_0_0.npz';target.write_bytes(b'previous complete artifact')
    class Interrupted(SegmentOnly):
        def decode_pixels(self,*args):raise RuntimeError('Injected interrupted export')
    with pytest.raises(RuntimeError,match='interrupted'):
        export_visibility_npz(target,Interrupted(encoded),encoded,encoded,max_workspace_bytes=128)
    assert target.read_bytes()==b'previous complete artifact'
    assert list(tmp_path.iterdir())==[target]


def test_original_legacy_npz_import_and_bit_roundtrip(tmp_path):
    path=Path(__file__).resolve().parents[2]/'tests/reference/optional_original_cpu/base/processed_inputs/SVF/shadowmats_0_0.npz'
    compact=import_visibility_npz(path,max_workspace_bytes=1024,scratch_dir=tmp_path)
    target=tmp_path/'roundtrip.npz';export_visibility_npz(target,*(compact[name] for name in LEGACY_MEMBERS),max_workspace_bytes=1024)
    with np.load(path,allow_pickle=False) as original,np.load(target,allow_pickle=False) as exported:
        for name in LEGACY_MEMBERS:same_bits(exported[name],original[name])


@pytest.mark.parametrize('shape',[(0,3,2),(2,0,3),(2,3,0)])
def test_empty_shapes_roundtrip(tmp_path,shape):
    original=np.empty(shape,np.float32);encoded=PackedVisibility.from_dense(original)
    target=tmp_path/'empty.npz';export_visibility_npz(target,encoded,encoded,encoded,max_workspace_bytes=128)
    imported=import_visibility_npz(target,max_workspace_bytes=128)
    for channel in imported.values():same_bits(channel.to_dense(),original)


def test_input_validation(tmp_path):
    with pytest.raises(TypeError):PackedVisibility.from_dense(np.zeros((2,3,4),np.float64))
    with pytest.raises(ValueError):VisibilityBuilder((1,2,-1))
    with pytest.raises(ValueError):VisibilityBuilder((1,2,3),max_workspace_bytes=127)
    builder=VisibilityBuilder((2,3,1))
    with pytest.raises(ValueError):builder.append(np.zeros((3,2),np.float32))
    encoded=PackedVisibility.from_dense(np.zeros((2,3,1),np.float32))
    wrong=PackedVisibility.from_dense(np.zeros((3,2,1),np.float32))
    with pytest.raises(ValueError):LazyDiffVisibility(encoded,wrong)
    with pytest.raises(ValueError):export_visibility_npz(tmp_path/'bad.npz',encoded,encoded,wrong)
