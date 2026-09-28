#!/bin/zsh
# B7-20 build: lw_primary.ispc -> liblw_native_g4.dylib / liblw_native_g8.dylib
# Toolchain pinned: ISPC 1.31.0 (LLVM 22.1.8), Apple clang (arm64).
# No fast-math, math-lib=default, FMA contraction explicitly disabled.
# The emitted assembly FMA audit is part of the evidence (see audit_fma.sh).
set -euo pipefail
cd "$(dirname "$0")"

ISPC=${ISPC:-/opt/homebrew/bin/ispc}
CC=${CC:-clang}

COMMON_FLAGS=(-O2 --opt=disable-fma --math-lib=default --pic)

for gang in g4 g8; do
  case $gang in
    g4) target=neon-i32x4 ;;
    g8) target=neon-i32x8 ;;
  esac
  "$ISPC" lw_primary.ispc -o lw_primary_${gang}.o -h lw_primary_${gang}.h \
      --target=$target "${COMMON_FLAGS[@]}"
  "$ISPC" lw_primary.ispc --emit-asm -o lw_primary_${gang}.s \
      --target=$target "${COMMON_FLAGS[@]}"
  "$CC" -arch arm64 -dynamiclib -o liblw_native_${gang}.dylib lw_primary_${gang}.o
  echo "built liblw_native_${gang}.dylib (target=$target)"
done

# FMA audit: any fused multiply-add mnemonic in the emitted text asm is a hard failure.
echo "--- FMA audit ---"
if grep -nE 'fmadd|fmla|fmsub|fnmadd|fnmsub|fnmla' lw_primary_g4.s lw_primary_g8.s; then
  echo "FMA AUDIT FAILED: contraction present" >&2
  exit 1
else
  echo "FMA AUDIT OK: no fused multiply-add mnemonics in neon-i32x4/neon-i32x8 assembly"
fi
