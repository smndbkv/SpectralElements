#include "sem/basis/legendre.hpp"

#include <stdexcept>

namespace sem {

LegendreValue legendre(int n, Real x) {
    if (n < 0) {
        throw std::invalid_argument("legendre: degree must be non-negative");
    }
    if (n == 0) {
        return {1, 0};
    }

    // На входе итерации k: L = L_{k-1}, Lprev = L_{k-2} (и так же для производных).
    Real Lprev = 1, L = x;
    Real dLprev = 0, dL = 1;
    for (int k = 2; k <= n; ++k) {
        const Real Lnext  = ((2 * k - 1) * x * L - (k - 1) * Lprev) / k;
        const Real dLnext = dLprev + (2 * k - 1) * L;
        Lprev  = L;
        L      = Lnext;
        dLprev = dL;
        dL     = dLnext;
    }
    return {L, dL};
}

LobattoQ lobattoQ(int n, Real x) {
    if (n < 1) {
        throw std::invalid_argument("lobattoQ: degree must be at least 1");
    }

    // Доходим до Lk = L_n, Lkm1 = L_{n-1}.
    Real Lkm1 = 1, Lk = x;
    for (int k = 1; k < n; ++k) {
        const Real Lkp1 = ((2 * k + 1) * x * Lk - k * Lkm1) / (k + 1);
        Lkm1 = Lk;
        Lk   = Lkp1;
    }
    const Real Lnp1 = ((2 * n + 1) * x * Lk - n * Lkm1) / (n + 1);

    return {Lnp1 - Lkm1, (2 * n + 1) * Lk, Lk};
}

} // namespace sem
