#!/usr/bin/env sh
set -eu

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQF_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/qf_rates_demo
python3 scripts/python_reference.py

