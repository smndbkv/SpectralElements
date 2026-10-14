#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <sstream>
#include <string>
#include <thread>

#include "sem/core/timer.hpp"

using namespace std::chrono_literals;

TEST_CASE("Timer: measures elapsed time", "[core][timer]") {
    sem::Timer t;
    const double first = t.seconds();
    CHECK(first >= 0);

    std::this_thread::sleep_for(20ms);
    CHECK(t.milliseconds() >= 15);           // сон не короче запрошенного, с запасом на грубость часов
    CHECK(t.seconds() >= first);             // монотонность

    t.reset();
    CHECK(t.milliseconds() < 15);
}

TEST_CASE("Timer: lapMs restarts the stopwatch", "[core][timer]") {
    sem::Timer t;
    std::this_thread::sleep_for(20ms);
    CHECK(t.lapMs() >= 15);
    CHECK(t.lapMs() < 15);                   // сразу после круга почти ничего не прошло
}

TEST_CASE("ScopedTimer: prints the name and time on scope exit", "[core][timer]") {
    std::ostringstream out;
    {
        sem::ScopedTimer timer("этап", out);
        CHECK(out.str().empty());            // до выхода из блока ничего не напечатано
    }
    const std::string text = out.str();
    CHECK(text.starts_with("этап: "));
    CHECK(text.ends_with(" ms\n"));
}
