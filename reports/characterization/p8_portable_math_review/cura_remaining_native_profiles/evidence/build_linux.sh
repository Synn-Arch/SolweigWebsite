#!/usr/bin/env bash
set -euo pipefail
# One native build process; no installation into production environments.
tar -xzf openlibm.tar.gz
tar -xzf aocl-libm-ose.tar.gz
cmake -S openlibm-5fe399749f9276eaa0b8403e507470da05cbbb3f -B build-openlibm -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-Bsymbolic-functions
cmake --build build-openlibm --parallel 1
(cd aocl-libm-ose-29fd054f383e6c5e2dec2fce781d5220059f1836 && cmake --preset dev-release-gcc --fresh && cmake --build --preset dev-release-gcc --parallel 1)
cc -O2 -fno-fast-math -ffp-contract=off -fPIC -shared function_loop.c -o libfunction_loop.so
