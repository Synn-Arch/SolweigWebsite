#!/usr/bin/env bash
set -euo pipefail
# Run in the copied, ledger-owned packet directory on native x86 AVX512.
cc -O2 -fno-fast-math -ffp-contract=off -mavx512f -mfma -fPIC -shared native_svml.c svml_z0_tan_s_la.s -lm -o libsvml_tan_probe.so
tar -xzf sleef-5a1d179.tar.gz
cmake -S sleef-5a1d179df9cf652951b59010a2d2075372d67f68 -B build -DCMAKE_BUILD_TYPE=Release -DSLEEF_BUILD_TESTS=OFF -DSLEEF_BUILD_DFT=OFF -DSLEEF_BUILD_QUAD=OFF -DSLEEF_BUILD_GNUABI_LIBS=OFF -DSLEEF_BUILD_BENCH=OFF -DBUILD_SHARED_LIBS=ON
cmake --build build --target sleef --parallel 1
cc -O2 -fno-fast-math -ffp-contract=off -mavx512f -mfma -fPIC -shared native_sleef.c -Ibuild/include -Lbuild/lib -Wl,-rpath,"$PWD/build/lib" -lsleef -lm -o libsleef_tan_probe.so
