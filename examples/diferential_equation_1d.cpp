// Краевая задача с переменными коэффициентами на одном спектральном элементе:
//
//   u''(x) + c(x) u'(x) + b(x) u(x) = f(x),   x in (-1, 1),
//   u(-1) = u(1) = 0.
//
// Метод: Галёркин на одном элементе (K = 1, J = 1) с базисом Лагранжа степени N на узлах GLL.
// Слабая форма (по частям интегрируется только u'', граничный член исчезает, т.к. v(±1) = 0):
//   int u' v' dx - int c u' v dx - int b u v dx = -int f v dx   для всех v с v(±1) = 0.
// С квадратурой GLL получается система (N+1) x (N+1):
//   (D^T W D - W_c D - W_b) u = -W f,   W = diag(w_i),   W_b = diag(w_i b(x_i)),
//   W_c = diag(w_i c(x_i)),   D_ij = ell_j'(x_i),
// где D^T W D — матрица жёсткости (точная), W_b — матрица масс с коэффициентом b
// (при квадратуре GLL она остаётся диагональной), W_c D — матрица конвекции
// int c ell_j' ell_i dx: она несимметрична (при постоянном c квадратура GLL для неё точна).
// Граничные условия: строки 0 и N заменяются уравнениями u_0 = 0, u_N = 0.
// Система решается методом Гаусса с выбором главного элемента.
//
// Тестовые задачи строятся методом искусственных решений: выбирается u с u(±1) = 0
// и вычисляется f = u'' + c u' + b u. Коэффициенты задаются в одном месте — функции coefB
// и coefC, подпись уравнения для графиков — kEquation; правые части пользуются coefB и
// coefC, поэтому при смене коэффициентов их пересчитывать не нужно (нужно только
// поправить kEquation).
//
// Вывод: нормы ошибки C, C1, L1, L2 и три графика:
//   * точное и численное решения для гауссова пика (N = 36);
//   * сходимость норм C и L2 по N = 2..24 на гладких решениях (1 - x^2) e^x и sin(pi x):
//     экспоненциальное убывание до машинной точности;
//   * поточечная ошибка |u - u_N| при N = 4, 8, 12, 16.
// Замечание: если b велико (больше наименьшего собственного значения -d^2/dx^2,
// pi^2 / 4 на (-1, 1)), задача может стать резонансной, а матрица — вырожденной.

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <numbers>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "sem/analysis/error_norms.hpp"
#include "sem/basis/quadrature.hpp"
#include "sem/basis/spectral_basis.hpp"
#include "sem/core/linear_solve.hpp"
#include "sem/io/plot.hpp"


// Коэффициенты b(x) и c(x) в уравнении u'' + c(x) u' + b(x) u = f — единственное место,
// где они задаются: ими пользуются и сборка системы, и правые части тестовых задач
// (f = u'' + c u' + b u).
sem::Real coefB(sem::Real x) { return 1; }

sem::Real coefC(sem::Real x) { return 100000; }

// Уравнение для подписей графиков (TeX); при смене coefB или coefC поменять и его.
const char *const kEquation = R"($u'' + c\,u' + b\,u = f$, $b(x) = sin(x^2)$, $c(x) = e^{-x^2}$)";

// Решение u'' + c(x) u' + b(x) u = f, u(±1) = 0 на одном элементе степени N
// (метод Галёркина, квадратура GLL):
//   (D^T W D - W_c D - W_b) u = -W f,  W_b = diag(w_i b(x_i)),  W_c = diag(w_i c(x_i)),
//   строки 0 и N заменены на u_0 = 0, u_N = 0.
// Возвращает узловые значения решения.
template<class F>
std::vector<sem::Real> solveHelmholtz(const sem::SpectralBasis1D &basis, F &&f) {
    const int N = basis.degree();
    const int n = basis.numPoints();
    auto weights = basis.weights();
    auto points = basis.points();
    const auto &D = basis.D();
    const auto &Dt = basis.Dt();

    sem::DenseMatrix WD = D;
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            WD(k, j) *= weights[k];
        }
    }
    sem::DenseMatrix A = Dt * WD;


    for (int k = 0; k < n; ++k)
        A(k, k) -= weights[k] * coefB(points[k]); // W_b = diag(w_k b(x_k))

    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            A(k, j) -= weights[k] * coefC(points[k]) * D(k, j); // W_c D

    std::vector<sem::Real> b(n);
    for (int i = 0; i < n; i++)
        b[i] = -weights[i] * f(points[i]);


    //Boundary condition u(+-1) = 0
    for (const int k: {0, N}) {
        for (int j = 0; j <= N; ++j) {
            A(k, j) = 0;
        }
        A(k, k) = 1;
        b[k] = 0;
    }

    return sem::solveGauss(A, b);
}

// Тестовая задача: правая часть f, точное решение u и его производная du.
struct TestProblem {
    std::string name; // подпись для графика (TeX)
    std::function<sem::Real(sem::Real)> f, u, du;
};

// Запуск с флагом --no-show только сохраняет картинки, без окон.
int main(int argc, char **argv) {
    const bool showWindow = !(argc > 1 && std::string_view(argv[1]) == "--no-show");

    const int N = 20; //degree
    // Тестовые задачи для уравнения u'' + c(x) u' + b(x) u = f (f = u'' + c u' + b u).
    // (Задача с f = sin x подходила только для b = 1: при переменном b точного решения
    // в элементарных функциях, как правило, нет.)

    // const sem::Real kw = 4 * std::numbers::pi;
    // const auto f = [kw](sem::Real x) { return (coefB(x) - kw * kw) * std::sin(kw * x) + coefC(x) * kw * std::cos(kw * x); };
    // const auto u = [kw](sem::Real x) { return std::sin(kw * x); };
    // const auto du = [kw](sem::Real x) { return kw * std::cos(kw * x); };

    // const auto f = [](sem::Real x) {
    //     return ((-1 - 4 * x - x * x) + coefC(x) * (1 - 2 * x - x * x) + coefB(x) * (1 - x * x)) * std::exp(x);
    // };
    // const auto u = [](sem::Real x) { return (1 - x * x) * std::exp(x); };
    // const auto du = [](sem::Real x) { return (1 - 2 * x - x * x) * std::exp(x); };

    const sem::Real beta = 50, x0 = 0.3;
    const auto g = [=](sem::Real x) { return std::exp(-beta * (x - x0) * (x - x0)); };
    const sem::Real gp = g(1), gm = g(-1);
    const auto lin = [=](sem::Real x) { return (gp * (1 + x) + gm * (1 - x)) / 2; };

    const auto u = [=](sem::Real x) { return g(x) - lin(x); };
    const auto du = [=](sem::Real x) { return -2 * beta * (x - x0) * g(x) - (gp - gm) / 2; };
    const auto f = [=](sem::Real x) {
        return (4 * beta * beta * (x - x0) * (x - x0) - 2 * beta) * g(x) + coefC(x) * du(x) + coefB(x) * u(x);
        // g'' + c u' + b u
    };

    // 1. Узкий гауссов пик при N = 36: решение и узлы.
    auto basis = sem::SpectralBasis1D(N);
    const auto coeff = solveHelmholtz(basis, f);

    // auto u_approximation = [&basis, &coeff](sem::Real x) -> sem::Real {
    //     sem::Real res = 0;
    //     std::vector<sem::Real> l(N + 1);
    //     basis.lagrange().evaluateBasis(x, l);
    //     for (int i = 0; i < N + 1; i++) {
    //         res += coeff[i] * l[i];
    //     }
    //     return res;
    // };

    auto u_approximation = [&basis, &coeff](sem::Real x) -> sem::Real {
        return basis.evaluate(coeff, x);
    };

    const std::vector<sem::Real> bp{-1, 1}; // один элемент [-1, 1]
    const auto norms = sem::analysis::errorNorms(basis, bp, coeff, u, du);

    printf("Norms of error: C = %.3e  C1 = %.3e  L1 = %.3e  L2 = %.3e\n", norms.c, norms.c1, norms.l1, norms.l2);


    // Подписи на английском: картинка используется в README (docs/images).
    sem::io::Figure fig(std::string(kEquation) + R"( on $[-1, 1]$, $u(\pm 1) = 0$: one spectral element, $N = )" +
                        std::to_string(N) + "$");
    fig.setXLabel("$x$").setYLabel("$u$");
    fig.plotFunction(u, -1, 1, 400, "exact solution", "-"); // функция на равномерной сетке
    fig.plotFunction(u_approximation, -1, 1, 400, "SEM solution", "--");
    fig.plot(basis.points(), coeff, "GLL nodes", "o"); // точки (x, y)

    // 2. Сходимость на гладких решениях: ошибка падает экспоненциально и примерно при N = 16
    //    упирается в машинную точность — дальше её ограничивает только округление.
    // Правые части — для уравнения u'' + c(x) u' + b(x) u = f: f = u'' + c u' + b u.
    const sem::Real pi = std::numbers::pi;
    const std::vector<TestProblem> smooth{
        {
            R"($u = (1 - x^2)\,e^{x}$)",
            [](sem::Real x) {
                return ((-1 - 4 * x - x * x) + coefC(x) * (1 - 2 * x - x * x) + coefB(x) * (1 - x * x)) *
                       std::exp(x);
            },
            [](sem::Real x) { return (1 - x * x) * std::exp(x); },
            [](sem::Real x) { return (1 - 2 * x - x * x) * std::exp(x); }
        },
        {
            R"($u = \sin \pi x$)",
            [pi](sem::Real x) { return (coefB(x) - pi * pi) * std::sin(pi * x) + coefC(x) * pi * std::cos(pi * x); },
            [pi](sem::Real x) { return std::sin(pi * x); },
            [pi](sem::Real x) { return pi * std::cos(pi * x); }
        },
    };
    const std::vector<sem::Real> bpConv{-1, 1};

    const int maxDegree = 24;
    std::vector<sem::Real> degrees;
    for (int N0 = 2; N0 <= maxDegree; ++N0) {
        degrees.push_back(N0);
    }

    sem::io::Figure convergence("Spectral convergence for " + std::string(kEquation) + R"(, $u(\pm 1) = 0$)");
    convergence.setXLabel("degree $N$").setYLabel(R"(error norm of $u - u_N$)" "\n" "(logarithmic scale)")
            .setYScale(sem::io::AxisScale::Log);

    const char *markers[] = {"o", "s"};
    for (std::size_t p = 0; p < smooth.size(); ++p) {
        const TestProblem &problem = smooth[p];
        std::vector<sem::Real> errorC, errorL2;
        printf("\n%s\n  N           C          L2\n", problem.name.c_str());
        for (const sem::Real N0: degrees) {
            const sem::SpectralBasis1D b0(static_cast<int>(N0));
            const auto c0 = solveHelmholtz(b0, problem.f);
            const auto e = sem::analysis::errorNorms(b0, bpConv, c0, problem.u, problem.du);
            errorC.push_back(e.c);
            errorL2.push_back(e.l2);
            printf("%3d  %10.3e  %10.3e\n", static_cast<int>(N0), e.c, e.l2);
        }
        convergence.plot(degrees, errorL2, problem.name + R"(: $L^2$)", std::string(markers[p]) + "-")
                .plot(degrees, errorC, problem.name + R"(: $C$)", std::string(markers[p]) + "--");
    }
    const std::vector<sem::Real> epsilon(degrees.size(), std::numeric_limits<sem::Real>::epsilon());
    convergence.plot(degrees, epsilon, R"(machine $\varepsilon \approx 2.2 \cdot 10^{-16}$)", "k:");

    // 3. Поточечная ошибка |u - u_N| для u = (1 - x^2) e^x: при N = 16 — уровень округления.
    sem::io::Figure pointwise(R"(Pointwise error $|u - u_N|$ for $u = (1 - x^2)\,e^{x}$)");
    pointwise.setXLabel("$x$").setYLabel(R"($|u - u_N|$ (logarithmic scale))")
            .setYScale(sem::io::AxisScale::Log);
    for (const int N0: {4, 8, 12, 16}) {
        const sem::SpectralBasis1D b0(N0);
        const auto c0 = solveHelmholtz(b0, smooth[0].f);
        pointwise.plotFunction([&](sem::Real x) { return std::abs(smooth[0].u(x) - b0.evaluate(c0, x)); },
                               -1, 1, 801, "$N = " + std::to_string(N0) + "$", "-");
    }

    // Все три графика одним запуском Python: окна открываются одновременно.
    sem::io::renderAll(std::vector<sem::io::PlotItem>{
                           {fig.toJson(), "helmholtz_1d.json", "helmholtz_1d.png"},
                           {convergence.toJson(), "helmholtz_1d_convergence.json", "helmholtz_1d_convergence.png"},
                           {pointwise.toJson(), "helmholtz_1d_pointwise_error.json", "helmholtz_1d_pointwise_error.png"}
                       },
                       showWindow);

    return 0;
}
