#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${repo_root}/output/pdf"
archive="${output_dir}/qf-rates-arxiv-source.tar.gz"

mkdir -p "${output_dir}"
cd "${repo_root}"

files=(
  paper/main.tex
  paper/references.bib
  paper/generated/results_macros.tex
  paper/generated/quantlib_table.tex
  paper/generated/quantlib_grid_summary_table.tex
  paper/generated/risk_validation_table.tex
  paper/generated/lsm_table.tex
  paper/generated/mc_diagnostics_table.tex
  paper/generated/calibration_residuals_table.tex
  paper/generated/multistart_table.tex
  paper/generated/lsm_oos_table.tex
  paper/generated/stress_summary_table.tex
  paper/generated/wrong_way_grid_table.tex
  paper/figures/g2pp_mc_convergence.pdf
  paper/figures/g2pp_mc_standardized_error.pdf
  paper/figures/g2pp_time_step_convergence.pdf
  paper/figures/g2pp_stress_grid.pdf
  paper/figures/g2pp_stress_moneyness.pdf
  paper/figures/g2pp_multistart_calibration.pdf
  paper/figures/g2pp_multistart_parameters.pdf
  paper/figures/g2pp_calibration_residuals.pdf
  paper/figures/lsm_convergence.pdf
  paper/figures/lsm_out_of_sample.pdf
  paper/figures/lsm_exercise_distribution.pdf
  paper/figures/quantlib_comparison.pdf
  paper/figures/quantlib_g2pp_grid.pdf
  paper/figures/g2pp_risk_validation.pdf
  paper/figures/variance_reduction.pdf
  paper/figures/wrong_way_risk_sensitivity.pdf
)

if [[ -f paper/main.bbl ]]; then
  files+=(paper/main.bbl)
elif [[ -f paper/build/main.bbl ]]; then
  files+=(paper/build/main.bbl)
fi

tar -czf "${archive}" "${files[@]}"
echo "arXiv source archive: ${archive}"
