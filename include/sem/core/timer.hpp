#pragma once

#include <chrono>
#include <format>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>

namespace sem {

// Секундомер на монотонных часах (steady_clock): показания не скачут при переводе
// системного времени, поэтому подходят для замеров.
//
//   sem::Timer t;
//   ...                                  // работа
//   double ms = t.milliseconds();        // сколько прошло с создания / reset()
//   double lap = t.lapMs();              // то же, но секундомер начинает отсчёт заново
class Timer {
public:
    using Clock = std::chrono::steady_clock;

    Timer() : start_(Clock::now()) {}

    void reset() { start_ = Clock::now(); }

    double seconds() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

    double milliseconds() const { return 1e3 * seconds(); }

    // Время этапа: миллисекунды с прошлого вызова lapMs() (или с создания), затем сброс.
    double lapMs() {
        const Clock::time_point now = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - start_).count();
        start_ = now;
        return ms;
    }

private:
    Clock::time_point start_;
};

// Замер блока кода: при выходе из области видимости печатает «name: 12.345 ms».
//
//   {
//       sem::ScopedTimer timer("нормы ошибки");
//       ...                              // замеряемый код
//   }                                    // здесь печатается время
class ScopedTimer {
public:
    explicit ScopedTimer(std::string name, std::ostream& out = std::cout)
        : name_(std::move(name)), out_(out) {}

    ~ScopedTimer() {
        try {
            out_ << std::format("{}: {:.3f} ms\n", name_, timer_.milliseconds());
        } catch (...) {
            // Деструктор не должен бросать исключения: печать времени не критична.
        }
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string   name_;
    std::ostream& out_;
    Timer         timer_;
};

} // namespace sem
