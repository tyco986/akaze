#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "akazemat2numpy.hpp"
#include "AKAZE.h"

namespace py = pybind11;
using namespace libAKAZECU;

PYBIND11_MODULE(libakaze_pybindings, m) {
    m.doc() = "AKAZE CUDA Python bindings (pybind11, no OpenCV)";

    py::class_<AKAZEOptions>(m, "AKAZEOptions")
        .def(py::init<>())
        .def("setWidth", &AKAZEOptions::setWidth)
        .def("setHeight", &AKAZEOptions::setHeight)
        .def_readwrite("omax", &AKAZEOptions::omax)
        .def_readwrite("nsublevels", &AKAZEOptions::nsublevels)
        .def_readwrite("dthreshold", &AKAZEOptions::dthreshold);

    py::class_<AKAZE>(m, "AKAZE")
        .def(py::init<AKAZEOptions>())
        .def("Create_Nonlinear_Scale_Space", [](AKAZE& self, py::array_t<float> img) {
            AkazeMat mat = numpy_to_mat(img);
            return self.Create_Nonlinear_Scale_Space(mat);
        })
        .def("Feature_Detection", [](AKAZE& self) {
            return mat_to_numpy(self.Feature_Detection_());
        })
        .def("Compute_Descriptors", [](AKAZE& self) {
            auto p = self.Compute_Descriptors_Seq();
            return py::make_tuple(mat_to_numpy(p.first), mat_to_numpy(p.second));
        });

    py::class_<Matcher>(m, "Matcher")
        .def(py::init<>())
        .def("BFMatch", [](Matcher& self, py::array_t<unsigned char> desc_query,
                          py::array_t<unsigned char> desc_train) {
            py::buffer_info q = desc_query.request();
            py::buffer_info t = desc_train.request();
            if (q.ndim != 2 || t.ndim != 2)
                throw std::runtime_error("Descriptors must be 2D arrays");
            AkazeMat mq(static_cast<int>(q.shape[0]), static_cast<int>(q.shape[1]),
                        AKAZE_8UC1);
            std::memcpy(mq.data, q.ptr,
                        static_cast<size_t>(q.shape[0]) * q.shape[1]);
            AkazeMat mt(static_cast<int>(t.shape[0]), static_cast<int>(t.shape[1]),
                        AKAZE_8UC1);
            std::memcpy(mt.data, t.ptr,
                        static_cast<size_t>(t.shape[0]) * t.shape[1]);
            AkazeMat result = self.bfmatch_(mq, mt);
            return mat_to_numpy(result);
        });
}
