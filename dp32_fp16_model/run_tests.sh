#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
compiler="${CXX:-g++}"
"$compiler" -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude src/dp32.cpp tests/testbench.cpp -o build/testbench
"$compiler" -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude src/dp32.cpp src/demo.cpp -o build/demo
./build/demo
./build/testbench "${1:-tests/golden_vectors.txt}"
