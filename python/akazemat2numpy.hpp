/**
 * AkazeMat <-> pybind11 numpy array conversion (no OpenCV)
 */
#pragma once

#include <pybind11/numpy.h>
#include <cstring>
#include <vector>
#include "akaze_types.h"

namespace py = pybind11;

inline AkazeMat numpy_to_mat(py::array_t<float> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Expected 2D float array (height, width)");
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    AkazeMat m(rows, cols, AKAZE_32FC1);
    std::memcpy(m.data, buf.ptr, static_cast<size_t>(rows) * cols * sizeof(float));
    return m;
}

inline py::array_t<float> mat_to_numpy_f32(const AkazeMat& m) {
    if (m.empty())
        return py::array_t<float>(std::vector<long>{0, 0});
    py::array_t<float> arr({m.rows, m.cols});
    py::buffer_info buf = arr.request();
    std::memcpy(buf.ptr, m.data, static_cast<size_t>(m.rows) * m.cols * sizeof(float));
    return arr;
}

inline py::array_t<unsigned char> mat_to_numpy_u8(const AkazeMat& m) {
    if (m.empty())
        return py::array_t<unsigned char>(std::vector<long>{0, 0});
    py::array_t<unsigned char> arr({m.rows, m.cols});
    py::buffer_info buf = arr.request();
    std::memcpy(buf.ptr, m.data, static_cast<size_t>(m.rows) * m.cols * sizeof(unsigned char));
    return arr;
}

inline py::object mat_to_numpy(const AkazeMat& m) {
    if (m.empty())
        return py::none();
    switch (m.type()) {
        case AKAZE_32FC1: return mat_to_numpy_f32(m);
        case AKAZE_8UC1:  return mat_to_numpy_u8(m);
        default:
            throw std::runtime_error("Unsupported AkazeMat type for numpy conversion");
    }
}
