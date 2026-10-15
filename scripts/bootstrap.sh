#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
tools_root="${workspace_root}/.tools"
deps_root="${workspace_root}/.deps"
cmake_root="${tools_root}/cmake"
systemc_source="${deps_root}/systemc-src"
systemc_build="${deps_root}/systemc-build"
systemc_install="${deps_root}/systemc"

mkdir -p "${tools_root}" "${deps_root}"

if [[ ! -x "${cmake_root}/bin/cmake" ]]; then
    python3 -m pip install --target "${cmake_root}" cmake==3.31.6
fi

cmake_cmd=(env "PYTHONPATH=${cmake_root}" "${cmake_root}/bin/cmake")

if [[ ! -d "${systemc_source}/.git" ]]; then
    git clone --depth 1 --branch 3.0.2 \
        https://github.com/accellera-official/systemc.git "${systemc_source}"
fi

"${cmake_cmd[@]}" -S "${systemc_source}" -B "${systemc_build}" \
    -DCMAKE_INSTALL_PREFIX="${systemc_install}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DBUILD_SHARED_LIBS=OFF \
    -DENABLE_EXAMPLES=OFF
"${cmake_cmd[@]}" --build "${systemc_build}" -j2
"${cmake_cmd[@]}" --install "${systemc_build}"

"${cmake_cmd[@]}" -S "${repo_root}" -B "${repo_root}/build" \
    -DCMAKE_PREFIX_PATH="${systemc_install}"
"${cmake_cmd[@]}" --build "${repo_root}/build" -j2
"${cmake_cmd[@]}" --build "${repo_root}/build" --target test
