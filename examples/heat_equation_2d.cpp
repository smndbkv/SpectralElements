// Уравнение теплопроводности в квадрате:
//
//   u_t = u_xx + u_yy   в (-1, 1)^2,  t in (0, T],
//   u = 0 на границе,   u(x, y, 0) = u0(x, y).
//
// Точное решение — сумма двух собственных функций лапласиана, каждая затухает со своей скоростью:
//   u = e^{-lambda1 t} cos(pi x / 2) cos(pi y / 2) + 1/2 e^{-lambda2 t} sin(pi x) sin(2 pi y),
//   lambda1 = pi^2 / 2,  lambda2 = 5 pi^2.
// Вторая (несимметричная) мода затухает в 10 раз быстрее, и решение «округляется».
//
// Дискретизация по пространству: квадрат делится равномерной сеткой на K x K одинаковых
// элементов (sem::BoxMesh), в каждом — базис степени N на узлах GLL (Галёркин, квадратура GLL).
// Соседние элементы делят узлы на общих сторонах, поэтому решение непрерывно.
// Получается система ОДУ  M du/dt = -K u,  где M — диагональная матрица масс, K — матрица
// жёсткости; обе собираются из элементных (на элементе со стороной h: M_e = (h/2)^2 M_ref,
// K_e = K_ref) и не хранятся: K применяется поэлементно (BoxMesh::applyK).
//
// По времени — неявные схемы (BDF), на каждом шаге — задача Гельмгольца:
//   (gamma0 M + dt K) u^{n+1} = M (alpha0 u^n + alpha1 u^{n-1}),
//   неявный Эйлер (BDF1): gamma0 = 1,   alpha = (1, 0),     порядок 1;
//   BDF2:                 gamma0 = 3/2, alpha = (2, -1/2),  порядок 2
//                         (первый шаг BDF2 делается неявным Эйлером, порядок от этого не падает).
// Матрица gamma0 M + dt K симметрична и положительно определена, поэтому система решается
// методом сопряжённых градиентов с предобусловливателем Якоби. Это та же задача, что
// решается для скорости на каждом шаге схемы velocity correction для Навье–Стокса.
//
// Почему не явный Эйлер: он устойчив только при dt < 2 / lambda_max(M^{-1} K), а у
// спектральных элементов lambda_max ~ N^4 / h^2. Пример оценивает lambda_max степенным
// методом и печатает, сколько шагов понадобилось бы явной схеме.
//
// Вывод: ошибка по пространству (интерполяция u0), таблица ошибок по dt для обеих схем,
// график сходимости по времени, значение в точке (0.5, 0.25) во времени, численное решение
// при t = T с границами элементов и анимация решения по шагам времени (heat_2d_movie.gif,
// в окне проигрывается по кругу).
// Запуск с флагом --no-show только сохраняет картинки и анимацию, без окон.

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "sem/analysis/error_norms.hpp"
#include "sem/io/plot.hpp"
#include "sem/mesh/box_mesh.hpp"
#include "sem/solvers/conjugate_gradient.hpp"

using sem::Point;
using sem::Real;

constexpr int Dim = 2;
constexpr Real pi = std::numbers::pi;
constexpr Real lambda1 = pi * pi / 2;
constexpr Real lambda2 = 5 * pi * pi;

using Mesh = sem::BoxMesh<Dim>;

// Точное решение и его градиент.
Real exactU(const Point<Dim> &p, Real t) {
    return std::exp(-lambda1 * t) * std::cos(pi * p[0] / 2) * std::cos(pi * p[1] / 2) +
           0.5 * std::exp(-lambda2 * t) * std::sin(pi * p[0]) * std::sin(2 * pi * p[1]);
}

Point<Dim> exactGrad(const Point<Dim> &p, Real t) {
    const Real e1 = std::exp(-lambda1 * t), e2 = 0.5 * std::exp(-lambda2 * t);
    return {
        -e1 * pi / 2 * std::sin(pi * p[0] / 2) * std::cos(pi * p[1] / 2) +
        e2 * pi * std::cos(pi * p[0]) * std::sin(2 * pi * p[1]),
        -e1 * pi / 2 * std::cos(pi * p[0] / 2) * std::sin(pi * p[1] / 2) +
        e2 * 2 * pi * std::sin(pi * p[0]) * std::cos(2 * pi * p[1])
    };
}

// L2-норма ошибки u - exactU(., t) по всему квадрату: errorNorms на каждом элементе
// (на эталонном элементе, функции пересчитаны через отображение), затем
//   ||e||^2 = sum_e J^2 ||e_ref||^2,   J = h / 2.
Real l2Error(const Mesh &mesh, std::span<const Real> u, Real t) {
    // Нужна только L2: перебор точек для максимумов (C, C1) — минимальный, он самый дорогой.
    sem::analysis::NormOptions options;
    options.quadraturePoints = 6;
    options.samplesPerInterval = 2;
    const Real J = mesh.jacobian();
    std::vector<Real> local(static_cast<std::size_t>(mesh.basis().numPoints()));
    Real sum = 0;
    for (int e = 0; e < mesh.numElements(); ++e) {
        mesh.gather(e, u, local);
        const auto norms = sem::analysis::errorNorms<Dim>(
            mesh.basis(), local,
            [&](const Point<Dim> &xi) { return exactU(mesh.toPhysical(e, xi), t); },
            [&](const Point<Dim> &xi) {
                // производные по xi: d/dxi = J d/dx
                const Point<Dim> g = exactGrad(mesh.toPhysical(e, xi), t);
                return Point<Dim>{J * g[0], J * g[1]};
            },
            options);
        sum += J * J * norms.l2 * norms.l2;
    }
    return std::sqrt(sum);
}

// Результат интегрирования по времени.
struct HeatRun {
    std::vector<Real> u; // узловые значения при t = T
    std::vector<Real> times, probe; // значение в точке-зонде после каждого шага
    int cgIterations = 0; // всего итераций CG за все шаги
};

// Вызывается после каждого шага: момент времени и узловые значения решения.
using StepObserver = std::function<void(Real t, std::span<const Real> u)>;

// u0 — узловые значения начального условия; order = 1 (неявный Эйлер) или 2 (BDF2);
// probePoint — точка, в которой после каждого шага запоминается решение;
// onStep (необязательно) — например, для кадров анимации.
HeatRun solveHeat(const Mesh &mesh, std::span<const Real> u0, Real T, int steps, int order,
                  const Point<Dim> &probePoint, const StepObserver &onStep = {}) {
    const auto size = static_cast<std::size_t>(mesh.numNodes());
    const Real dt = T / steps;

    // Маска граничных узлов: там u = 0 во все моменты времени.
    const std::vector<bool> boundary = mesh.boundaryMask();
    const std::vector<Real> mass = mesh.massDiagonal(); // диагональ собранной M
    const std::vector<Real> diagK = mesh.diagonalK();

    HeatRun run;
    std::vector<Real> current(u0.begin(), u0.end()), previous; // u^n, u^{n-1}
    for (std::size_t k = 0; k < size; ++k) {
        if (boundary[k]) current[k] = 0;
    }
    std::vector<Real> rhs(size), next(size), masked(size);

    for (int n = 1; n <= steps; ++n) {
        // Коэффициенты схемы: первый шаг всегда неявным Эйлером (u^{n-1} ещё нет).
        const bool bdf2 = order == 2 && n > 1;
        const Real gamma0 = bdf2 ? 1.5 : 1.0;

        // Правая часть M (alpha0 u^n + alpha1 u^{n-1}), в граничных узлах 0.
        for (std::size_t k = 0; k < size; ++k) {
            const Real history = bdf2 ? 2 * current[k] - 0.5 * previous[k] : current[k];
            rhs[k] = boundary[k] ? 0 : mass[k] * history;
        }

        // H x = P (gamma0 M + dt K) P x.
        const auto applyH = [&](std::span<const Real> in, std::span<Real> out) {
            for (std::size_t k = 0; k < size; ++k) masked[k] = boundary[k] ? 0 : in[k];
            mesh.applyK(masked, out);
            for (std::size_t k = 0; k < size; ++k) {
                out[k] = boundary[k] ? 0 : gamma0 * mass[k] * masked[k] + dt * out[k];
            }
        };
        std::vector<Real> diagonal(size);
        for (std::size_t k = 0; k < size; ++k) {
            diagonal[k] = boundary[k] ? 1 : gamma0 * mass[k] + dt * diagK[k];
        }

        // Начальное приближение — решение с прошлого шага: оно близко к новому.
        next = current;
        const sem::CgResult cg = sem::conjugateGradient(applyH, sem::JacobiPreconditioner(diagonal), rhs, next,
                                                        {.relativeTolerance = 1e-13});
        run.cgIterations += cg.iterations;

        previous = std::move(current);
        current = next;
        run.times.push_back(n * dt);
        run.probe.push_back(mesh.evaluate(current, probePoint));
        if (onStep) onStep(n * dt, current);
    }
    run.u = std::move(current);
    return run;
}

// Наибольшее собственное значение M^{-1} K на внутренних узлах (степенной метод):
// явный Эйлер устойчив только при dt < 2 / lambda_max.
Real largestEigenvalue(const Mesh &mesh) {
    const auto size = static_cast<std::size_t>(mesh.numNodes());
    const std::vector<bool> boundary = mesh.boundaryMask();
    const std::vector<Real> mass = mesh.massDiagonal();
    std::vector<Real> x(size), y(size);
    for (std::size_t k = 0; k < size; ++k) {
        x[k] = boundary[k] ? 0 : 1 + 0.1 * std::sin(static_cast<Real>(k)); // произвольный ненулевой вектор
    }
    Real lambda = 0;
    for (int it = 0; it < 500; ++it) {
        mesh.applyK(x, y);
        Real norm = 0;
        for (std::size_t k = 0; k < size; ++k) {
            y[k] = boundary[k] ? 0 : y[k] / mass[k];
            norm += y[k] * y[k];
        }
        norm = std::sqrt(norm);
        Real xNorm = 0;
        for (const Real v: x) xNorm += v * v;
        lambda = norm / std::sqrt(xNorm);
        for (std::size_t k = 0; k < size; ++k) x[k] = y[k] / norm;
    }
    return lambda;
}

std::vector<Real> uniformAxis(int n) {
    std::vector<Real> axis(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) axis[static_cast<std::size_t>(i)] = -1 + 2.0 * i / (n - 1);
    return axis;
}

// Значения решения на сетке axis x axis в раскладке Figure2D (x меняется быстрее всего).
std::vector<Real> onGrid(const Mesh &mesh, std::span<const Real> u, const std::vector<Real> &axis) {
    std::vector<Real> values;
    values.reserve(axis.size() * axis.size());
    for (const Real y: axis) {
        for (const Real x: axis) values.push_back(mesh.evaluate(u, {x, y}));
    }
    return values;
}

int main(int argc, char **argv) {
    const bool showWindow = !(argc > 1 && std::string_view(argv[1]) == "--no-show");

    const int K = 30; // элементов по каждому направлению: всего K x K
    const int N = 5; // степень в каждом элементе
    const Real T = 0.05; // за это время вторая мода затухает до ~9% от начальной
    const Point<Dim> probePoint{0.5, 0.25};

    const Mesh mesh(N, K);
    const std::vector<Real> u0 = mesh.interpolate([](const Point<Dim> &p) { return exactU(p, 0); });
    const std::string meshLabel = "$" + std::to_string(K) + R"( \times )" + std::to_string(K) +
                                  "$ elements, $N = " + std::to_string(N) + "$";

    std::printf("mesh: %d x %d elements of size %g, N = %d, %d nodes\n", K, K, mesh.elementSize(), N,
                mesh.numNodes());
    // Ошибка по пространству: интерполяция u0 на сетке. Ошибки по времени ниже неё не опустятся.
    std::printf("spatial error (interpolation of u0): L2 = %.3e\n", l2Error(mesh, u0, 0));

    const Real lambdaMax = largestEigenvalue(mesh);
    std::printf("lambda_max(M^-1 K) = %.3e, explicit Euler needs dt < %.3e (%.0f steps up to T = %g)\n\n",
                lambdaMax, 2 / lambdaMax, std::ceil(T * lambdaMax / 2), T);

    // 1. Сходимость по времени: сетка фиксирована, dt = T / steps уменьшается вдвое.
    const char *names[] = {"implicit Euler", "BDF2"};
    std::vector<Real> dts;
    std::vector<Real> errors[2];
    HeatRun finest;
    std::printf("%6s %10s %16s %8s %16s %8s %9s\n", "steps", "dt", "L2 error Euler", "rate", "L2 error BDF2", "rate",
                "CG/step");
    for (int steps = 4; steps <= 512; steps *= 2) {
        dts.push_back(T / steps);
        Real cgPerStep = 0;
        for (int order = 1; order <= 2; ++order) {
            const HeatRun run = solveHeat(mesh, u0, T, steps, order, probePoint);
            errors[order - 1].push_back(l2Error(mesh, run.u, T));
            cgPerStep = static_cast<Real>(run.cgIterations) / steps;
            if (order == 2 && steps == 512) finest = run;
        }
        // Наблюдаемый порядок: log2(e(2 dt) / e(dt)).
        const auto rate = [&](int s) {
            const auto &e = errors[s];
            return e.size() < 2 ? 0.0 : std::log2(e[e.size() - 2] / e.back());
        };
        std::printf("%6d %10.3e %16.3e %8.2f %16.3e %8.2f %9.1f\n", steps, T / steps, errors[0].back(), rate(0),
                    errors[1].back(), rate(1), cgPerStep);
    }

    sem::io::Figure convergence(R"(Heat equation $u_t = \Delta u$ in $(-1, 1)^2$: error at $t = T$, )" + meshLabel);
    convergence.setXLabel(R"(time step $\Delta t$ (logarithmic scale))")
            .setYLabel(R"($\Vert u - u_N \Vert_{L^2}$ (logarithmic scale))")
            .setXScale(sem::io::AxisScale::Log)
            .setYScale(sem::io::AxisScale::Log)
            .plot(dts, errors[0], names[0], "o-")
            .plot(dts, errors[1], names[1], "s-");
    // Направляющие C dt и C dt^2 через первые точки: прямые с наклоном 1 и 2 в осях log-log.
    std::vector<Real> guide1, guide2;
    for (const Real dt: dts) {
        guide1.push_back(errors[0].front() * dt / dts.front());
        guide2.push_back(errors[1].front() * (dt / dts.front()) * (dt / dts.front()));
    }
    convergence.plot(dts, guide1, R"($\sim \Delta t$)", "k--").plot(dts, guide2, R"($\sim \Delta t^2$)", "k:");

    // 2. Решение в точке-зонде во времени: крупный шаг, чтобы была видна разница схем.
    const int coarseSteps = 5;
    sem::io::Figure history(R"(Solution at $(x, y) = (0.5, 0.25)$, $\Delta t = T / )" + std::to_string(coarseSteps) +
                            "$");
    history.setXLabel("$t$").setYLabel(R"($u(0.5, 0.25, t)$)")
            .plotFunction([&](Real t) { return exactU(probePoint, t); }, 0, T, 200, "exact", "-");
    const char *markers[] = {"o--", "s--"};
    for (int order = 1; order <= 2; ++order) {
        HeatRun run = solveHeat(mesh, u0, T, coarseSteps, order, probePoint);
        run.times.insert(run.times.begin(), 0);
        run.probe.insert(run.probe.begin(), mesh.evaluate(u0, probePoint));
        history.plot(run.times, run.probe, names[order - 1], markers[order - 1]);
    }

    // 3. Численное решение при t = T (BDF2, самый мелкий шаг) на равномерной сетке;
    //    изолинии — точное решение, белые линии — границы элементов.
    const std::vector<Real> axis = uniformAxis(101);
    const std::vector<Real> field = onGrid(mesh, finest.u, axis);

    // Границы элементов — одна ломаная, отрезки разделены NaN (matplotlib их не соединяет).
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    std::vector<Real> linesX, linesY;
    for (int e = 0; e <= K; ++e) {
        const Real c = mesh.lower() + e * mesh.elementSize();
        linesX.insert(linesX.end(), {c, c, nan, mesh.lower(), mesh.upper(), nan});
        linesY.insert(linesY.end(), {mesh.lower(), mesh.upper(), nan, c, c, nan});
    }

    sem::io::Figure2D solution(R"(BDF2 solution at $t = T = )" + std::to_string(T).substr(0, 4) +
                               R"($, $\Delta t = T / 512$, )" + meshLabel);
    solution.setXLabel("$x$").setYLabel("$y$").setColorLabel("$u_N$")
            .setFieldLabel("SEM solution")
            .setField(axis, axis, field)
            .addOverlayFunction([T](const Point<Dim> &p) { return exactU(p, T); }, -1, 1, -1, 1, 101, 101,
                                "exact solution")
            .addPoints(linesX, linesY, "element boundaries", "w-");

    // 4. Анимация: каждый шаг по времени — кадр (BDF2). Интервал длиннее T, чтобы было видно,
    //    как сначала исчезает быстрая несимметричная мода, а потом затухает основная.
    //    Кадры — на сетке 51 x 51.
    const Real movieT = 0.25;
    const int movieSteps = 125;
    const std::vector<Real> movieAxis = uniformAxis(51);

    sem::io::Animation2D movie(R"(Heat equation $u_t = \Delta u$, BDF2, $\Delta t = )" +
                               std::to_string(movieT / movieSteps).substr(0, 5) + "$, " + meshLabel);
    movie.setColorLabel("$u_N$").setIsolines(10).setFramesPerSecond(20).setGrid(movieAxis, movieAxis);
    const auto addFrame = [&](Real t, std::span<const Real> u) {
        char label[32];
        std::snprintf(label, sizeof label, "$t = %.3f$", t);
        movie.addFrame(onGrid(mesh, u, movieAxis), label);
    };
    addFrame(0, u0);
    solveHeat(mesh, u0, movieT, movieSteps, 2, probePoint, addFrame);

    sem::io::renderAll(std::vector<sem::io::PlotItem>{
                           {convergence.toJson(), "heat_2d_convergence.json", "heat_2d_convergence.png"},
                           {history.toJson(), "heat_2d_probe.json", "heat_2d_probe.png"},
                           {solution.toJson(), "heat_2d_solution.json", "heat_2d_solution.png"},
                           {movie.toJson(), "heat_2d_movie.json", "heat_2d_movie.gif"}
                       },
                       showWindow);
    return 0;
}
