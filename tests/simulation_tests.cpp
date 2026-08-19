#include "evobrain/random.hpp"
#include "evobrain/simulation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

int failure_count = 0;

// Records a failed boolean expectation while allowing later tests to run.
void expect_true(const bool condition, const std::string_view description)
{
    if (!condition) {
        std::cerr << "FAILED: " << description << '\n';
        ++failure_count;
    }
}

// Records a failed 64-bit equality expectation with both observed values.
void expect_equal(
    const std::uint64_t actual,
    const std::uint64_t expected,
    const std::string_view description)
{
    if (actual != expected) {
        std::cerr << "FAILED: " << description << " (expected " << expected
                  << ", got " << actual << ")\n";
        ++failure_count;
    }
}

// Verifies fixed-step tick accounting through the public simulation API.
void test_simulation_ticks()
{
    evobrain::Simulation simulation(evobrain::SimulationConfig {.seed = 1234});
    expect_equal(simulation.current_tick(), 0, "new simulation starts at zero");

    simulation.tick();
    expect_equal(simulation.current_tick(), 1, "tick advances exactly once");

    simulation.tick();
    simulation.tick();
    expect_equal(simulation.current_tick(), 3, "multiple ticks accumulate");

    simulation.run_for(7);
    expect_equal(simulation.current_tick(), 10, "run_for advances by n ticks");

    simulation.run_for(0);
    expect_equal(simulation.current_tick(), 10, "run_for zero changes nothing");
}

// Verifies run_for has the same tick result as repeated individual ticks.
void test_run_for_equivalence()
{
    evobrain::Simulation run_simulation(evobrain::SimulationConfig {.seed = 9});
    evobrain::Simulation tick_simulation(evobrain::SimulationConfig {.seed = 9});

    run_simulation.run_for(25);
    for (std::uint64_t index = 0; index < 25; ++index) {
        tick_simulation.tick();
    }

    expect_equal(
        run_simulation.current_tick(),
        tick_simulation.current_tick(),
        "run_for matches repeated tick calls");
}

// Verifies seeds reproducibly select deterministic PCG32 sequences.
void test_seed_behavior()
{
    evobrain::Pcg32 first(500);
    evobrain::Pcg32 second(500);
    evobrain::Pcg32 different(501);
    bool found_difference = false;

    for (std::size_t index = 0; index < 16; ++index) {
        const std::uint32_t first_value = first.next();
        expect_equal(first_value, second.next(), "equal seeds match");
        if (first_value != different.next()) {
            found_difference = true;
        }
    }

    expect_true(found_difference, "different seeds select different sequences");
}

// Locks the documented initialization and PCG-XSH-RR output to known values.
void test_known_pcg32_sequence()
{
    constexpr std::array<std::uint32_t, 6> expected {
        3270867926U,
        1795671209U,
        1924641435U,
        1143034755U,
        4121910957U,
        1757328946U,
    };

    evobrain::Pcg32 random(42);
    for (const std::uint32_t expected_value : expected) {
        expect_equal(random.next(), expected_value, "seed 42 PCG32 sequence");
    }
}

} // namespace

// Runs the lightweight core test suite and reports failures through the exit code.
int main()
{
    test_simulation_ticks();
    test_run_for_equivalence();
    test_seed_behavior();
    test_known_pcg32_sequence();

    if (failure_count != 0) {
        std::cerr << failure_count << " test expectation(s) failed\n";
        return 1;
    }

    std::cout << "All evobrain core tests passed\n";
    return 0;
}
