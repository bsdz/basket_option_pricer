#include <pybind11/eigen/matrix.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define VERSION_INFO __TIMESTAMP__

#include "basket_option_pricer/basket_options.hpp"

using namespace BOP::BasketOptionPricer;

using std::string;

namespace py = pybind11;

PYBIND11_MODULE(basket_option_pricer, m) {
    m.doc() = R"pbdoc(
        Basket Option Pricer
        -----------------------

        .. currentmodule:: basket_option_pricer

        .. autosummary::
           :toctree: _generate

    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    m.def("calculate", &calculate,
          "Calculate price and greeks for basket option.");

    py::class_<BOP::BasketOptionPricer::Result>(m, "Result")
        .def(py::init<>())
        .def_readonly("tau", &Result::tau)
        .def_readonly("kappa", &Result::kappa)
        .def_readonly("cp", &Result::cp)
        .def_readonly("rho", &Result::rho)
        .def_readonly("skew", &Result::skew)
        .def_readonly("dist", &Result::dist)
        .def_readonly("delta", &Result::delta)
        .def_readonly("theta", &Result::theta)
        .def_readonly("vega", &Result::vega);
}