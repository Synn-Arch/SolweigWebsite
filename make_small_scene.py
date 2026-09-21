"""Austin 샘플에서 조지아텍 캠퍼스 규모(기본 1024x1024 픽셀)의 작은 장면을 잘라낸다.

사용 예:
    python make_small_scene.py                       # 기본값으로 자르기만
    python make_small_scene.py --run                 # 자른 뒤 바로 SOLWEIG 실행
    python make_small_scene.py --xoff 800 --yoff 900 --size 1024 --run

잘린 파일은 --out 폴더에 원본과 같은 이름(Building_DSM.tif, DEM.tif, Trees.tif,
Landcover.tif)으로 저장되고, 기상 파일(ownmet_Forcing_data.txt)은 복사된다.
"""

import argparse
import shutil
import sys
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()

RASTERS = ["Building_DSM", "DEM", "Trees", "Landcover"]
MET_FILE = "ownmet_Forcing_data.txt"


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(here / "solweig_scene"), help="원본 장면 폴더")
    p.add_argument("--out", default=str(here / "solweig_scene_small"), help="잘린 장면을 저장할 폴더")
    p.add_argument("--xoff", type=int, default=1200, help="자를 창의 왼쪽 픽셀 위치")
    p.add_argument("--yoff", type=int, default=1400, help="자를 창의 위쪽 픽셀 위치")
    p.add_argument("--size", type=int, default=1024, help="자를 창의 한 변 픽셀 수 (정사각형)")
    p.add_argument("--run", action="store_true", help="자른 뒤 바로 thermal_comfort 실행")
    p.add_argument("--date", default="2020-08-13", help="시뮬레이션 날짜 (met 파일과 일치해야 함)")
    p.add_argument("--memory-gb", type=float, default=6.0, help="어드미션 메모리 예산 (GB)")
    p.add_argument("--threads", type=int, default=4, help="네이티브 스레드 수 (성능 코어 수 이하)")
    return p.parse_args()


def crop(src: Path, out: Path, xoff: int, yoff: int, size: int) -> None:
    out.mkdir(parents=True, exist_ok=True)

    ref = gdal.Open(str(src / f"{RASTERS[0]}.tif"))
    width, height = ref.RasterXSize, ref.RasterYSize
    ref = None
    if xoff < 0 or yoff < 0 or xoff + size > width or yoff + size > height:
        sys.exit(
            f"자를 창이 원본 범위를 벗어납니다: 원본 {width}x{height}, "
            f"요청 xoff={xoff}, yoff={yoff}, size={size}"
        )

    for name in RASTERS:
        infile = src / f"{name}.tif"
        outfile = out / f"{name}.tif"
        if not infile.exists():
            if name == "Landcover":
                print(f"[SKIP] {infile} 없음 (선택 입력)")
                continue
            sys.exit(f"필수 입력이 없습니다: {infile}")
        gdal.Translate(str(outfile), str(infile), format="GTiff", srcWin=[xoff, yoff, size, size])
        ds = gdal.Open(str(outfile))
        print(f"[OK] {outfile.name}: {ds.RasterXSize}x{ds.RasterYSize}")
        ds = None

    met_src = src / MET_FILE
    if met_src.exists():
        shutil.copy2(met_src, out / MET_FILE)
        print(f"[OK] {MET_FILE} 복사")
    else:
        print(f"[WARN] {met_src} 없음. 실행 전에 기상 파일을 {out}에 넣어야 합니다.")


def run(out: Path, date: str, size: int, memory_gb: float, threads: int) -> None:
    from solweig_light import RuntimeOptions, runtime_options, thermal_comfort

    landcover = "Landcover.tif" if (out / "Landcover.tif").exists() else None
    options = RuntimeOptions(
        memory_budget_bytes=int(memory_gb * 1024**3),
        cpu_budget=threads,
        workers=1,
        threads_per_worker=threads,
        block_pixels=1024,
    )
    with runtime_options(options):
        thermal_comfort(
            base_path=str(out),
            selected_date_str=date,
            landcover_filename=landcover,
            own_met_file=str(out / MET_FILE),
            ERA_5_z0_find=False,
            use_uhi=False,
            tile_size=size,
            overlap=0,
            save_tmrt=True,
        )
    print(f"[DONE] 결과: {out / 'output_folder'}")


def main():
    args = parse_args()
    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    crop(src, out, args.xoff, args.yoff, args.size)
    if args.run:
        run(out, args.date, args.size, args.memory_gb, args.threads)


if __name__ == "__main__":
    main()
