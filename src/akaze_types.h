/**
 * @file akaze_types.h
 * @brief Lightweight replacements for cv::KeyPoint, cv::DMatch, cv::Mat
 *        used throughout the AKAZE CUDA pipeline. These types have identical
 *        memory layout to their OpenCV counterparts so that existing CUDA
 *        kernels work without change.
 */

#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>

/* ---- Type constants (match OpenCV values) ---- */
enum {
    AKAZE_8UC1  = 0,   // CV_8UC1
    AKAZE_32SC1 = 4,   // CV_32SC1
    AKAZE_32FC1 = 5    // CV_32FC1
};

inline size_t akaze_elem_size(int type) {
    switch (type) {
        case AKAZE_8UC1:  return 1;
        case AKAZE_32SC1: return 4;
        case AKAZE_32FC1: return 4;
        default:          return 0;
    }
}

/* ---- Geometry helpers ---- */
struct AkazePoint2f {
    float x, y;
};

struct AkazeSize {
    int width, height;
    AkazeSize() : width(0), height(0) {}
    AkazeSize(int w, int h) : width(w), height(h) {}
};

/* ---- KeyPoint (must match cv::KeyPoint memory layout) ---- */
struct AkazeKeyPoint {
    AkazePoint2f pt;
    float size;
    float angle;
    float response;
    int   octave;
    int   class_id;
};

/* ---- DMatch (must match cv::DMatch memory layout) ---- */
struct AkazeMatch {
    int   queryIdx;
    int   trainIdx;
    int   imgIdx;
    float distance;
};

/* ---- Minimal 2-D matrix (replaces cv::Mat for host-side buffers) ---- */
class AkazeMat {
public:
    unsigned char* data;
    int rows, cols;

    AkazeMat()
        : data(nullptr), rows(0), cols(0), type_(0), step_(0), owns_(false) {}

    AkazeMat(int r, int c, int t)
        : data(nullptr), rows(0), cols(0), type_(0), step_(0), owns_(false) {
        create(r, c, t);
    }

    AkazeMat(int r, int c, int t, void* ptr)
        : data(static_cast<unsigned char*>(ptr)), rows(r), cols(c),
          type_(t), step_(c * akaze_elem_size(t)), owns_(false) {}

    ~AkazeMat() { release(); }

    /* Copy: deep-copy owned data, shallow-copy views */
    AkazeMat(const AkazeMat& o)
        : data(nullptr), rows(o.rows), cols(o.cols),
          type_(o.type_), step_(o.step_), owns_(false) {
        if (o.owns_ && o.data && o.rows > 0) {
            size_t sz = static_cast<size_t>(rows) * step_;
            data = new unsigned char[sz];
            std::memcpy(data, o.data, sz);
            owns_ = true;
        } else {
            data = o.data;
        }
    }

    AkazeMat& operator=(const AkazeMat& o) {
        if (this == &o) return *this;
        release();
        rows  = o.rows;
        cols  = o.cols;
        type_ = o.type_;
        step_ = o.step_;
        if (o.owns_ && o.data && o.rows > 0) {
            size_t sz = static_cast<size_t>(rows) * step_;
            data = new unsigned char[sz];
            std::memcpy(data, o.data, sz);
            owns_ = true;
        } else {
            data  = o.data;
            owns_ = false;
        }
        return *this;
    }

    AkazeMat(AkazeMat&& o) noexcept
        : data(o.data), rows(o.rows), cols(o.cols),
          type_(o.type_), step_(o.step_), owns_(o.owns_) {
        o.data = nullptr; o.rows = o.cols = 0; o.owns_ = false;
    }

    AkazeMat& operator=(AkazeMat&& o) noexcept {
        if (this == &o) return *this;
        release();
        data = o.data; rows = o.rows; cols = o.cols;
        type_ = o.type_; step_ = o.step_; owns_ = o.owns_;
        o.data = nullptr; o.rows = o.cols = 0; o.owns_ = false;
        return *this;
    }

    /* ---- allocation ---- */
    void create(int r, int c, int t) {
        if (rows == r && cols == c && type_ == t && data) return;
        release();
        rows = r; cols = c; type_ = t;
        step_ = static_cast<size_t>(c) * akaze_elem_size(t);
        if (r > 0 && c > 0) {
            data = new unsigned char[static_cast<size_t>(r) * step_];
            owns_ = true;
        }
    }

    void create(AkazeSize sz, int t) { create(sz.height, sz.width, t); }

    static AkazeMat zeros(int r, int c, int t) {
        AkazeMat m(r, c, t);
        if (m.data) std::memset(m.data, 0, static_cast<size_t>(r) * m.step_);
        return m;
    }

    /* ---- queries ---- */
    bool   empty()  const { return !data || rows == 0 || cols == 0; }
    int    type()   const { return type_; }
    size_t step1()  const { return step_; }

    /* ---- element access ---- */
    template<typename T> T& at(int i, int j) {
        return *(reinterpret_cast<T*>(data + i * step_) + j);
    }
    template<typename T> const T& at(int i, int j) const {
        return *(reinterpret_cast<const T*>(data + i * step_) + j);
    }
    template<typename T> T* ptr(int row = 0) {
        return reinterpret_cast<T*>(data + row * step_);
    }
    template<typename T> const T* ptr(int row = 0) const {
        return reinterpret_cast<const T*>(data + row * step_);
    }

    /* ---- sub-matrix / copy ---- */
    AkazeMat clone() const {
        AkazeMat m(rows, cols, type_);
        if (data && m.data)
            std::memcpy(m.data, data, static_cast<size_t>(rows) * step_);
        return m;
    }

    void copyTo(AkazeMat& dst) const {
        if (dst.rows != rows || dst.cols != cols || dst.type_ != type_)
            dst.create(rows, cols, type_);
        if (data && dst.data)
            std::memcpy(dst.data, data, static_cast<size_t>(rows) * step_);
    }

    AkazeMat row(int i) const {
        AkazeMat m;
        m.data  = data + i * step_;
        m.rows  = 1; m.cols = cols;
        m.type_ = type_; m.step_ = step_;
        m.owns_ = false;
        return m;
    }

    AkazeMat rowRange(int start, int end) const {
        AkazeMat m;
        m.data  = data + start * step_;
        m.rows  = end - start; m.cols = cols;
        m.type_ = type_; m.step_ = step_;
        m.owns_ = false;
        return m;
    }

    void release() {
        if (owns_ && data) delete[] data;
        data = nullptr; rows = cols = 0; owns_ = false;
    }

private:
    int    type_;
    size_t step_;
    bool   owns_;
};

/* ---- Typed matrix (replaces cv::Mat_<T>) ---- */
template<typename T>
class AkazeMat_ : public AkazeMat {
    static int _auto_type() {
        if (sizeof(T) == 1) return AKAZE_8UC1;
        if (sizeof(T) == 4) {
            /* distinguish int vs float by checking if T is integral */
            T test = T(-1);
            return (test < T(0)) ? AKAZE_32SC1 : AKAZE_32FC1;
        }
        return AKAZE_32FC1;
    }
public:
    AkazeMat_() : AkazeMat() {}
    AkazeMat_(int r, int c) : AkazeMat(r, c, _auto_type()) {}

    T& operator()(int i, int j)       { return at<T>(i, j); }
    const T& operator()(int i, int j) const { return at<T>(i, j); }

    AkazeMat_& operator=(T value) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                at<T>(i, j) = value;
        return *this;
    }

    AkazeMat_ clone() const {
        AkazeMat_ m(rows, cols);
        if (data && m.data)
            std::memcpy(m.data, data, static_cast<size_t>(rows) * step1());
        return m;
    }
};
