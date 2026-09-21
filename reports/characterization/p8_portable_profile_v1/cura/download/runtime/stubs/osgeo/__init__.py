'''Import-only classifier harness shim; no GDAL operation is called.''' 
class _Gdal:
    @staticmethod
    def UseExceptions():
        return None
gdal = _Gdal()
class _Osr:
    pass
osr = _Osr()
