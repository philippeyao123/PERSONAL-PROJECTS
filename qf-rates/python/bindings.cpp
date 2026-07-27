#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "qf/rates/options.hpp"
#include "qf/rates/swap.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qf_rates_python, module) {
  module.doc() = "Python bindings for the qf-rates C++ pricing library";
  py::enum_<qf::OptionType>(module, "OptionType")
      .value("Call", qf::OptionType::Call)
      .value("Put", qf::OptionType::Put);
  py::class_<qf::OptionResult>(module, "OptionResult")
      .def_readonly("price", &qf::OptionResult::price)
      .def_readonly("delta", &qf::OptionResult::delta)
      .def_readonly("gamma", &qf::OptionResult::gamma)
      .def_readonly("vega", &qf::OptionResult::vega);
  module.def("black76", &qf::black76, py::arg("type"), py::arg("forward"), py::arg("strike"),
             py::arg("volatility"), py::arg("expiry"), py::arg("discount") = 1.0,
             py::arg("notional") = 1.0);
  module.def("bachelier", &qf::bachelier, py::arg("type"), py::arg("forward"), py::arg("strike"),
             py::arg("normal_volatility"), py::arg("expiry"), py::arg("discount") = 1.0,
             py::arg("notional") = 1.0);
}
