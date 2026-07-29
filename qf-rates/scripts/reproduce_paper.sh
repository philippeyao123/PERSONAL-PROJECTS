#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${QF_PAPER_BUILD_DIR:-${repo_root}/build-paper}"

cmake -S "${repo_root}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQF_BUILD_BENCHMARKS=ON \
  -DQF_WARNINGS_AS_ERRORS=ON
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure
"${build_dir}/qf_rates_validation" "${repo_root}/paper/data"

python3 -c "import pybind11, QuantLib"
python_build_dir="${build_dir}-python"
cmake -S "${repo_root}" -B "${python_build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQF_BUILD_TESTS=OFF \
  -DQF_BUILD_EXAMPLES=OFF \
  -DQF_BUILD_PYTHON=ON \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build "${python_build_dir}" --parallel
PYTHONPATH="${python_build_dir}" python3 "${repo_root}/scripts/python_reference.py" \
  --require-quantlib \
  --require-bindings \
  --output-directory "${repo_root}/paper/data"
python3 "${repo_root}/scripts/generate_paper_artifacts.py"

mkdir -p "${repo_root}/output/pdf"
if command -v latexmk >/dev/null 2>&1; then
  mkdir -p "${repo_root}/paper/build"
  latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
    -outdir="${repo_root}/paper/build" "${repo_root}/paper/main.tex"
  cp "${repo_root}/paper/build/main.pdf" "${repo_root}/output/pdf/qf-rates-paper.pdf"
elif command -v tectonic >/dev/null 2>&1; then
  tectonic "${repo_root}/paper/main.tex" \
    --outdir "${repo_root}/output/pdf" --keep-intermediates
  mv "${repo_root}/output/pdf/main.pdf" "${repo_root}/output/pdf/qf-rates-paper.pdf"
else
  echo "Install latexmk or Tectonic to compile the paper." >&2
  exit 1
fi

echo "Paper: ${repo_root}/output/pdf/qf-rates-paper.pdf"
