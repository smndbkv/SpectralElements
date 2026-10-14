#include "sem/basis/spectral_basis.hpp"

#include <stdexcept>

namespace sem {

namespace {

int checkedDegree(int degree) {
    if (degree < 1) {
        throw std::invalid_argument("SpectralBasis1D: degree must be at least 1");
    }
    return degree;
}

} // namespace

SpectralBasis1D::SpectralBasis1D(int degree)
    : degree_(checkedDegree(degree)),
      quad_(gaussLobattoLegendre(degree_ + 1)),
      lagrange_(quad_.points),
      D_(lagrange_.derivativeMatrix()),
      Dt_(D_.transposed()) {}

} // namespace sem
