#include "fullModel.h"

namespace py = pybind11;

//FullFrictionModel hh;
//int ss = hh.step(0.1);

PYBIND11_MODULE(FrictionModelCPPClass, var) {
    var.doc() = "pybind11 example module for a class";
    
    py::class_<FullFrictionModel>(var, "FullFrictionModel")
        .def(py::init<>())
        .def("step", &FullFrictionModel::step);
}