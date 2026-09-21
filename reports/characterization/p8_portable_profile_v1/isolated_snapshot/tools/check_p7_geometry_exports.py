"""Compare expanded original/candidate SVF ZIP and visibility NPZ exports."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
from osgeo import gdal


def compare_exports(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    expected = {p.name: p for p in reference.glob('*') if p.suffix in ('.zip', '.npz')}
    actual = {p.name: p for p in candidate.glob('*') if p.suffix in ('.zip', '.npz')}
    assert actual.keys() == expected.keys() and expected
    records = []
    for name in sorted(expected):
        if name.endswith('.npz'):
            with np.load(expected[name], allow_pickle=False) as left, np.load(actual[name], allow_pickle=False) as right:
                assert left.files == right.files, name
                for field in left.files:
                    a, b = left[field], right[field]
                    assert a.dtype == b.dtype and a.shape == b.shape, (name, field)
                    assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), (name, field)
                    records.append({'artifact': name, 'field': field, 'shape': list(a.shape), 'bitwise_equal': True})
                    del a, b
        else:
            with zipfile.ZipFile(expected[name]) as left, zipfile.ZipFile(actual[name]) as right:
                assert left.namelist() == right.namelist(), name
                for field in left.namelist():
                    a = gdal.Open('/vsizip/' + str(expected[name].resolve()) + '/' + field)
                    b = gdal.Open('/vsizip/' + str(actual[name].resolve()) + '/' + field)
                    assert a is not None and b is not None
                    assert (a.RasterXSize, a.RasterYSize, a.RasterCount, a.GetGeoTransform(), a.GetProjection(), a.GetMetadata()) == (b.RasterXSize, b.RasterYSize, b.RasterCount, b.GetGeoTransform(), b.GetProjection(), b.GetMetadata()), (name, field)
                    maximum = 0.
                    for index in range(1, a.RasterCount + 1):
                        u, v = a.GetRasterBand(index), b.GetRasterBand(index)
                        assert u.DataType == v.DataType and u.GetMetadata() == v.GetMetadata()
                        un, vn = u.GetNoDataValue(), v.GetNoDataValue()
                        assert un == vn or (un is not None and vn is not None and np.isnan(un) and np.isnan(vn))
                        x, y = u.ReadAsArray(), v.ReadAsArray()
                        assert np.array_equal(u.GetMaskBand().ReadAsArray(), v.GetMaskBand().ReadAsArray())
                        for fn in (np.isnan, np.isposinf, np.isneginf):
                            assert np.array_equal(fn(x), fn(y)), (name, field)
                        finite = np.isfinite(x)
                        error = np.abs(x[finite].astype(np.float64) - y[finite].astype(np.float64))
                        maximum = max(maximum, float(error.max()) if error.size else 0.)
                    assert maximum <= 1e-6, (name, field, maximum)
                    records.append({'artifact': name, 'field': field, 'max_abs': maximum, 'frozen_max_abs': 1e-6})
                    a = b = None
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    records = compare_exports(args.reference, args.candidate)
    args.report.write_text(json.dumps({'status': 'passed', 'records': records}, indent=2) + '\n')
    print(len(records), 'geometry export fields passed')


if __name__ == '__main__':
    main()
