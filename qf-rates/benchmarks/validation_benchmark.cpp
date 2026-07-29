#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "qf/rates/calibration.hpp"
#include "qf/rates/lsm.hpp"
#include "qf/rates/monte_carlo.hpp"
#include "qf/rates/swap.hpp"
#include "qf/rates/xva.hpp"

namespace {

struct VarianceRow {
  std::string method;
  double standard_error{};
  double variance_ratio{};
};

struct LsmRow {
  std::size_t paths{};
  std::uint64_t seed{};
  std::string basis;
  double price{};
  double standard_error{};
};

struct MonteCarloRow {
  std::size_t paths{};
  std::uint64_t seed{};
  double deterministic{};
  double price{};
  double standard_error{};
};

struct TimeConvergenceRow {
  std::size_t time_steps{};
  std::size_t paths{};
  std::uint64_t seed{};
  double deterministic{};
  double price{};
  double standard_error{};
  double difference_from_deterministic{};
  double paired_bias_vs_finest{};
  double paired_bias_standard_error{};
};

struct StressGridRow {
  std::string scenario;
  double expiry{};
  double tenor{};
  double moneyness_basis_points{};
  double strike{};
  double a{};
  double b{};
  double sigma{};
  double eta{};
  double rho{};
  double price{};
};

struct CalibrationRunRow {
  std::size_t run{};
  bool selected{};
  double initial_a{};
  double initial_b{};
  double initial_sigma{};
  double initial_eta{};
  double initial_rho{};
  double calibrated_a{};
  double calibrated_b{};
  double calibrated_sigma{};
  double calibrated_eta{};
  double calibrated_rho{};
  double rmse{};
  std::size_t iterations{};
  bool converged{};
};

struct CalibrationDiagnosticRow {
  double expiry{};
  double tenor{};
  double market_volatility{};
  double model_volatility{};
  double error_basis_points{};
};

struct LsmOutOfSampleRow {
  std::size_t training_paths{};
  std::size_t valuation_paths{};
  std::uint64_t training_seed{};
  std::uint64_t valuation_seed{};
  std::string basis;
  double training_price{};
  double valuation_price{};
  double standard_error{};
  double optimism{};
  double non_exercise_probability{};
  std::array<double, 3> exercise_probabilities{};
};

struct WrongWayRiskRow {
  double beta{};
  double cva{};
  std::vector<double> epe;
};

void write_csv(const std::filesystem::path& directory,
               const std::vector<VarianceRow>& variance_rows, const std::vector<LsmRow>& lsm_rows,
               const std::vector<MonteCarloRow>& monte_carlo_rows,
               const std::vector<TimeConvergenceRow>& time_rows,
               const std::vector<StressGridRow>& stress_rows,
               const std::vector<CalibrationRunRow>& calibration_rows,
               const std::vector<CalibrationDiagnosticRow>& calibration_diagnostics,
               const std::vector<LsmOutOfSampleRow>& out_of_sample_rows,
               const std::vector<WrongWayRiskRow>& wrong_way_rows,
               std::span<const double> exposure_times) {
  std::filesystem::create_directories(directory);
  std::ofstream variance_file(directory / "variance_reduction.csv");
  variance_file << "method,standard_error,variance_ratio\n" << std::setprecision(12);
  for (const auto& row : variance_rows) {
    variance_file << row.method << ',' << row.standard_error << ',' << row.variance_ratio << '\n';
  }

  std::ofstream lsm_file(directory / "lsm_convergence.csv");
  lsm_file << "paths,seed,basis,price,standard_error\n" << std::setprecision(12);
  for (const auto& row : lsm_rows) {
    lsm_file << row.paths << ',' << row.seed << ',' << row.basis << ',' << row.price << ','
             << row.standard_error << '\n';
  }

  std::ofstream monte_carlo_file(directory / "g2pp_monte_carlo_convergence.csv");
  monte_carlo_file << "paths,seed,deterministic,price,standard_error\n" << std::setprecision(12);
  for (const auto& row : monte_carlo_rows) {
    monte_carlo_file << row.paths << ',' << row.seed << ',' << row.deterministic << ',' << row.price
                     << ',' << row.standard_error << '\n';
  }

  std::ofstream time_file(directory / "g2pp_time_step_convergence.csv");
  time_file << "time_steps,paths,seed,deterministic,price,standard_error,"
               "difference_from_deterministic,paired_bias_vs_finest,"
               "paired_bias_standard_error\n"
            << std::setprecision(12);
  for (const auto& row : time_rows) {
    time_file << row.time_steps << ',' << row.paths << ',' << row.seed << ',' << row.deterministic
              << ',' << row.price << ',' << row.standard_error << ','
              << row.difference_from_deterministic << ',' << row.paired_bias_vs_finest << ','
              << row.paired_bias_standard_error << '\n';
  }

  std::ofstream stress_file(directory / "g2pp_stress_grid.csv");
  stress_file << "scenario,expiry,tenor,moneyness_basis_points,strike,a,b,sigma,eta,rho,price\n"
              << std::setprecision(12);
  for (const auto& row : stress_rows) {
    stress_file << row.scenario << ',' << row.expiry << ',' << row.tenor << ','
                << row.moneyness_basis_points << ',' << row.strike << ',' << row.a << ',' << row.b
                << ',' << row.sigma << ',' << row.eta << ',' << row.rho << ',' << row.price << '\n';
  }

  std::ofstream calibration_file(directory / "g2pp_multistart_calibration.csv");
  calibration_file << "run,selected,initial_a,initial_b,initial_sigma,initial_eta,initial_rho,"
                      "calibrated_a,calibrated_b,calibrated_sigma,calibrated_eta,calibrated_rho,"
                      "rmse,iterations,converged\n"
                   << std::setprecision(12);
  for (const auto& row : calibration_rows) {
    calibration_file << row.run << ',' << row.selected << ',' << row.initial_a << ','
                     << row.initial_b << ',' << row.initial_sigma << ',' << row.initial_eta << ','
                     << row.initial_rho << ',' << row.calibrated_a << ',' << row.calibrated_b << ','
                     << row.calibrated_sigma << ',' << row.calibrated_eta << ','
                     << row.calibrated_rho << ',' << row.rmse << ',' << row.iterations << ','
                     << row.converged << '\n';
  }

  std::ofstream calibration_diagnostic_file(directory / "g2pp_calibration_residuals.csv");
  calibration_diagnostic_file
      << "expiry,tenor,market_volatility,model_volatility,error_basis_points\n"
      << std::setprecision(12);
  for (const auto& row : calibration_diagnostics) {
    calibration_diagnostic_file << row.expiry << ',' << row.tenor << ',' << row.market_volatility
                                << ',' << row.model_volatility << ',' << row.error_basis_points
                                << '\n';
  }

  std::ofstream out_of_sample_file(directory / "lsm_out_of_sample.csv");
  out_of_sample_file
      << "training_paths,valuation_paths,training_seed,valuation_seed,basis,"
         "training_price,valuation_price,standard_error,optimism,non_exercise_probability,"
         "exercise_probability_1y,exercise_probability_2y,exercise_probability_3y\n"
      << std::setprecision(12);
  for (const auto& row : out_of_sample_rows) {
    out_of_sample_file << row.training_paths << ',' << row.valuation_paths << ','
                       << row.training_seed << ',' << row.valuation_seed << ',' << row.basis << ','
                       << row.training_price << ',' << row.valuation_price << ','
                       << row.standard_error << ',' << row.optimism << ','
                       << row.non_exercise_probability << ',' << row.exercise_probabilities[0]
                       << ',' << row.exercise_probabilities[1] << ','
                       << row.exercise_probabilities[2] << '\n';
  }

  std::ofstream wrong_way_file(directory / "wrong_way_risk.csv");
  wrong_way_file << "case,cva\n" << std::setprecision(12);
  for (const auto& row : wrong_way_rows) {
    if (row.beta == 0.0) {
      wrong_way_file << "Independent exposure," << row.cva << '\n';
    } else if (row.beta == 20.0) {
      wrong_way_file << "Proxy WWR beta=20," << row.cva << '\n';
    }
  }

  std::ofstream wrong_way_grid_file(directory / "wrong_way_risk_grid.csv");
  wrong_way_grid_file << "beta,cva";
  for (const double time : exposure_times) {
    wrong_way_grid_file << ",epe_" << time << "y";
  }
  wrong_way_grid_file << '\n' << std::setprecision(12);
  for (const auto& row : wrong_way_rows) {
    wrong_way_grid_file << row.beta << ',' << row.cva;
    for (const double exposure : row.epe) {
      wrong_way_grid_file << ',' << exposure;
    }
    wrong_way_grid_file << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 2) {
    std::cerr << "Usage: qf_rates_validation [output-directory]\n";
    return 2;
  }
  constexpr double forward = 100.0;
  constexpr double strike = 100.0;
  constexpr double volatility = 0.20;
  const auto terminal = [](double normal) {
    return forward * std::exp(-0.5 * volatility * volatility + volatility * normal);
  };
  const auto payoff = [&](double normal) { return std::max(terminal(normal) - strike, 0.0); };
  const auto plain = qf::monte_carlo_normal(payoff, {100000, 1, 17, false});
  const auto antithetic = qf::monte_carlo_normal(payoff, {100000, 1, 17, true});
  const auto controlled = qf::monte_carlo_normal(payoff, {100000, 1, 17, false}, terminal, forward);

  std::cout << std::fixed << std::setprecision(8);
  std::cout << "# Variance reduction\n\n"
            << "| Method | Standard error | Variance ratio vs plain |\n"
            << "|---|---:|---:|\n";
  std::vector<VarianceRow> variance_rows;
  const auto row = [&](const std::string& name, const qf::MonteCarloResult& result) {
    const double ratio = std::pow(result.standard_error / plain.standard_error, 2.0);
    std::cout << "| " << name << " | " << result.standard_error << " | " << ratio << " |\n";
    variance_rows.push_back({name, result.standard_error, ratio});
  };
  row("Plain", plain);
  row("Antithetic", antithetic);
  row("Control variate", controlled);

  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const qf::G2ppModel model(curve, {});
  const qf::BermudanSwaption bermudan{{1.0, 2.0, 3.0}, 6.0, 0.026, 1.0, qf::OptionType::Call, 1.0};
  std::vector<LsmRow> lsm_rows;
  std::cout << "\n# LSM convergence\n\n"
            << "| Paths | Seed | Basis | Price | Standard error |\n"
            << "|---:|---:|---|---:|---:|\n";
  for (const std::size_t paths : {2000U, 5000U, 10000U}) {
    for (const std::uint64_t seed : {7U, 42U}) {
      for (const auto basis : {qf::LsmBasis::Linear, qf::LsmBasis::Quadratic}) {
        const auto result = qf::g2pp_bermudan_lsm(model, bermudan, {paths, 12, seed, basis});
        std::cout << "| " << paths << " | " << seed << " | "
                  << (basis == qf::LsmBasis::Linear ? "Linear" : "Quadratic") << " | "
                  << result.price << " | " << result.standard_error << " |\n";
        lsm_rows.push_back({paths, seed, basis == qf::LsmBasis::Linear ? "Linear" : "Quadratic",
                            result.price, result.standard_error});
      }
    }
  }

  const auto swap = qf::make_vanilla_swap(2.0, 7.0, 0.025);
  const qf::EuropeanSwaption european{2.0, qf::Schedule(2.0, 7.0, 1.0),
                                      swap.par_rate(*curve, *curve), 1.0, qf::OptionType::Call};
  const double deterministic = qf::g2pp_european_swaption(model, european);
  std::vector<MonteCarloRow> monte_carlo_rows;
  std::cout << "\n# G2++ Monte-Carlo convergence\n\n"
            << "| Paths | Seed | Deterministic | Monte-Carlo | Standard error |\n"
            << "|---:|---:|---:|---:|---:|\n";
  for (const std::size_t paths : {2000U, 5000U, 10000U, 30000U}) {
    for (const std::uint64_t seed : {7U, 42U}) {
      const auto result = qf::g2pp_european_swaption_mc(model, european, {paths, 96, seed, true});
      std::cout << "| " << paths << " | " << seed << " | " << deterministic << " | " << result.price
                << " | " << result.standard_error << " |\n";
      monte_carlo_rows.push_back({paths, seed, deterministic, result.price, result.standard_error});
    }
  }

  std::vector<TimeConvergenceRow> time_rows;
  std::cout << "\n# G2++ Monte-Carlo time-step convergence\n\n"
            << "| Time steps | Paths | Seed | Price | Difference from deterministic | "
               "Paired bias vs finest | Paired bias SE |\n"
            << "|---:|---:|---:|---:|---:|---:|---:|\n";
  constexpr std::size_t time_convergence_paths = 20000U;
  for (const std::uint64_t seed : {7U, 42U}) {
    qf::MonteCarloTimeConvergenceConfig config;
    config.paths = time_convergence_paths;
    config.seed = seed;
    const auto convergence =
        qf::g2pp_european_swaption_mc_time_convergence(model, european, config);
    for (const auto& result : convergence) {
      const double difference = result.price - deterministic;
      std::cout << "| " << result.time_steps << " | " << time_convergence_paths << " | " << seed
                << " | " << result.price << " | " << difference << " | "
                << result.paired_bias_vs_finest << " | " << result.paired_bias_standard_error
                << " |\n";
      time_rows.push_back({result.time_steps, time_convergence_paths, seed, deterministic,
                           result.price, result.standard_error, difference,
                           result.paired_bias_vs_finest, result.paired_bias_standard_error});
    }
  }

  struct StressScenario {
    std::string name;
    qf::G2ppParameters parameters;
  };
  const std::vector<StressScenario> stress_scenarios{
      {"low_volatility", {0.10, 0.30, 0.005, 0.0075, -0.70}},
      {"base", {0.10, 0.30, 0.010, 0.0150, -0.70}},
      {"high_volatility", {0.10, 0.30, 0.020, 0.0300, -0.70}},
      {"fast_mean_reversion", {0.30, 0.80, 0.010, 0.0150, -0.70}},
      {"weak_correlation", {0.10, 0.30, 0.010, 0.0150, -0.10}},
  };
  std::vector<StressGridRow> stress_rows;
  for (const auto& scenario : stress_scenarios) {
    const qf::G2ppModel stressed_model(curve, scenario.parameters);
    for (const double expiry : {1.0, 2.0, 5.0}) {
      for (const double tenor : {2.0, 5.0, 10.0}) {
        const auto underlying = qf::make_vanilla_swap(expiry, expiry + tenor, 0.025);
        const double atm = underlying.par_rate(*curve, *curve);
        for (const double moneyness_basis_points : {-100.0, 0.0, 100.0}) {
          const double stressed_strike = atm + moneyness_basis_points * 1.0e-4;
          const qf::EuropeanSwaption stressed_swaption{expiry,
                                                       qf::Schedule(expiry, expiry + tenor, 1.0),
                                                       stressed_strike, 1.0, qf::OptionType::Call};
          const double price = qf::g2pp_european_swaption(stressed_model, stressed_swaption);
          stress_rows.push_back({scenario.name, expiry, tenor, moneyness_basis_points,
                                 stressed_strike, scenario.parameters.a, scenario.parameters.b,
                                 scenario.parameters.sigma, scenario.parameters.eta,
                                 scenario.parameters.rho, price});
        }
      }
    }
  }
  std::cout << "\n# G2++ stress grid\n\nGenerated " << stress_rows.size()
            << " deterministic cells across five parameter regimes, three expiries, "
               "three tenors and three moneyness levels.\n";

  const std::vector<qf::SwaptionQuote> calibration_quotes{
      {1.0, 5.0, 0.026, 0.0060, 1.0},  {2.0, 5.0, 0.026, 0.0065, 1.0},
      {3.0, 5.0, 0.026, 0.0068, 1.0},  {1.0, 10.0, 0.026, 0.0064, 1.0},
      {2.0, 10.0, 0.026, 0.0069, 1.0}, {3.0, 10.0, 0.026, 0.0072, 1.0},
  };
  const auto multistart = qf::calibrate_g2pp_multistart(curve, calibration_quotes, {8U, 2026U});
  std::vector<CalibrationRunRow> calibration_rows;
  std::vector<CalibrationDiagnosticRow> calibration_diagnostics;
  std::cout << "\n# G2++ multi-start calibration\n\n"
            << "| Run | Selected | Initial a | Initial b | Initial sigma | Initial eta | "
               "Initial rho | RMSE |\n"
            << "|---:|:---:|---:|---:|---:|---:|---:|---:|\n";
  for (std::size_t index = 0; index < multistart.runs.size(); ++index) {
    const auto& run = multistart.runs[index];
    const auto& calibrated = run.calibration.parameters;
    const bool selected = index == multistart.best_run;
    std::cout << "| " << index << " | " << (selected ? "yes" : "no") << " | " << run.initial.a
              << " | " << run.initial.b << " | " << run.initial.sigma << " | " << run.initial.eta
              << " | " << run.initial.rho << " | " << run.calibration.rmse << " |\n";
    calibration_rows.push_back({index, selected, run.initial.a, run.initial.b, run.initial.sigma,
                                run.initial.eta, run.initial.rho, calibrated.a, calibrated.b,
                                calibrated.sigma, calibrated.eta, calibrated.rho,
                                run.calibration.rmse, run.calibration.iterations,
                                run.calibration.converged});
  }
  for (const auto& diagnostic : multistart.best.diagnostics) {
    calibration_diagnostics.push_back({diagnostic.expiry, diagnostic.tenor,
                                       diagnostic.market_volatility, diagnostic.model_volatility,
                                       diagnostic.error_basis_points});
  }

  std::vector<LsmOutOfSampleRow> out_of_sample_rows;
  std::cout << "\n# LSM out-of-sample policy evaluation\n\n"
            << "| Training paths | Valuation paths | Training seed | Valuation seed | Basis | "
               "Training price | Valuation price | Optimism | Standard error |\n"
            << "|---:|---:|---:|---:|---|---:|---:|---:|---:|\n";
  constexpr std::size_t valuation_paths = 10000U;
  for (const std::size_t training_paths : {2000U, 5000U, 10000U}) {
    for (const std::uint64_t training_seed : {7U, 42U}) {
      const std::uint64_t valuation_seed = 1000U + training_seed;
      for (const auto basis_choice : {qf::LsmBasis::Linear, qf::LsmBasis::Quadratic}) {
        const auto result = qf::g2pp_bermudan_lsm_out_of_sample(
            model, bermudan,
            {training_paths, valuation_paths, 12U, training_seed, valuation_seed, basis_choice});
        const std::string basis_name =
            basis_choice == qf::LsmBasis::Linear ? "Linear" : "Quadratic";
        const double optimism = result.training_price - result.price;
        std::cout << "| " << training_paths << " | " << valuation_paths << " | " << training_seed
                  << " | " << valuation_seed << " | " << basis_name << " | "
                  << result.training_price << " | " << result.price << " | " << optimism << " | "
                  << result.standard_error << " |\n";
        out_of_sample_rows.push_back(
            {training_paths,
             valuation_paths,
             training_seed,
             valuation_seed,
             basis_name,
             result.training_price,
             result.price,
             result.standard_error,
             optimism,
             result.non_exercise_probability,
             {result.exercise_probabilities[0], result.exercise_probabilities[1],
              result.exercise_probabilities[2]}});
      }
    }
  }

  const std::vector<qf::InterestRateSwap> swaps{qf::make_vanilla_swap(0.0, 7.0, 0.02, 1'000'000.0)};
  const std::vector<double> times{0.5, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<WrongWayRiskRow> wrong_way_rows;
  for (const double beta : {0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0}) {
    const auto exposure = qf::simulate_swap_exposure(model, swaps, times, 10000, 19, true, beta);
    const auto xva = qf::compute_xva(*curve, exposure);
    wrong_way_rows.push_back({beta, xva.cva, xva.profile.epe});
  }
  const auto& independent = wrong_way_rows.front();
  const auto& beta_twenty = wrong_way_rows[4];
  std::cout << "\n# Wrong-way risk\n\n"
            << "| Case | CVA |\n"
            << "|---|---:|\n"
            << "| Independent exposure | " << independent.cva << " |\n"
            << "| Proxy WWR beta=20 | " << beta_twenty.cva << " |\n";

  if (argc == 2) {
    write_csv(argv[1], variance_rows, lsm_rows, monte_carlo_rows, time_rows, stress_rows,
              calibration_rows, calibration_diagnostics, out_of_sample_rows, wrong_way_rows, times);
  }
}
