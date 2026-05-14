// joint_angle: angle at b formed by segments b→a and b→c, in degrees ∈ [0, 180].
// Returns NaN if either segment has zero length. Mirrors fitcoach.angles.joint_angle.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

double joint_angle_impl(const std::vector<double>& a,
                        const std::vector<double>& b,
                        const std::vector<double>& c) {
    const std::size_t n = a.size();
    if (b.size() != n || c.size() != n) {
        throw py::value_error("a, b, c must have the same length");
    }
    double dot = 0.0;
    double na = 0.0;
    double nc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ba = a[i] - b[i];
        const double bc = c[i] - b[i];
        dot += ba * bc;
        na += ba * ba;
        nc += bc * bc;
    }
    na = std::sqrt(na);
    nc = std::sqrt(nc);
    if (na == 0.0 || nc == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double cos = dot / (na * nc);
    // Clamp to [-1, 1] before acos for numerical safety.
    if (cos < -1.0) cos = -1.0;
    if (cos >  1.0) cos =  1.0;
    return std::acos(cos) * (180.0 / M_PI);
}

}  // namespace

PYBIND11_MODULE(fitcoach_cpp, m) {
    m.doc() = "Native joint-angle math for FitCoach AI";
    m.def("joint_angle", &joint_angle_impl,
          py::arg("a"), py::arg("b"), py::arg("c"),
          "Angle at b formed by segments b→a and b→c, in degrees [0, 180]. "
          "Returns NaN if either segment has zero length.");
}
