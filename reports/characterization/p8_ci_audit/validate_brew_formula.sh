#!/usr/bin/env bash
set -euo pipefail
set -x

formula_commit=3c01d2c89227ced6d17c28d6a784b675c991b33e
formula_sha256=71ecda409f008f599b15b10f0fe6f51f8f7d052ee91f1082a801ce2ca9d3c05f
tap_name=solweig-ci-audit/pinned
formula_dir="$(mktemp -d)"

cleanup() {
  if brew tap | grep -Fxq "${tap_name}"; then
    HOMEBREW_NO_AUTO_UPDATE=1 brew untap "${tap_name}"
  fi
}
trap cleanup EXIT

curl --fail --location --silent --show-error \
  "https://raw.githubusercontent.com/Homebrew/homebrew-core/${formula_commit}/Formula/g/gdal.rb" \
  --output "${formula_dir}/gdal.rb"
echo "${formula_sha256}  ${formula_dir}/gdal.rb" | shasum -a 256 --check
HOMEBREW_NO_AUTO_UPDATE=1 brew tap-new "${tap_name}"
cp "${formula_dir}/gdal.rb" "$(brew --repository "${tap_name}")/Formula/gdal.rb"
HOMEBREW_NO_AUTO_UPDATE=1 brew trust --formula "${tap_name}/gdal"
HOMEBREW_NO_AUTO_UPDATE=1 brew readall "${tap_name}"
