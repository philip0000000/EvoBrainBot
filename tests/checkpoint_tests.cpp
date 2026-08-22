#include "checkpoint_tests.hpp"

#include "evobrain/checkpoint.hpp"
#include "evobrain/simulation.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

int checkpoint_failure_count = 0;

// Records a failed checkpoint expectation while allowing later cases to run.
void expect_checkpoint(const bool condition, const std::string_view description)
{
    if (!condition) {
        std::cerr << "FAILED: " << description << '\n';
        ++checkpoint_failure_count;
    }
}

// Returns a binary string containing one complete saved simulation checkpoint.
std::string saved_checkpoint(const evobrain::Simulation& simulation)
{
    std::ostringstream output(std::ios::out | std::ios::binary);
    evobrain::save_checkpoint(simulation, output);
    return output.str();
}

// Keeps persistence tests bounded so they do not double as population stress tests.
evobrain::SimulationConfig checkpoint_config(const std::uint64_t seed)
{
    evobrain::SimulationConfig config {.seed = seed};
    config.target_food_count = 0;
    config.food_population_threshold = 0;
    config.food_boost_population_threshold = 0;
    config.boosted_food_count = 0;
    config.reproduction_threshold = 100.0;
    return config;
}

// Loads one simulation from an in-memory binary checkpoint string.
evobrain::Simulation load_saved_checkpoint(const std::string& data)
{
    std::istringstream input(data, std::ios::in | std::ios::binary);
    return evobrain::load_checkpoint(input);
}

// Records whether malformed checkpoint data produces a clean exception.
void expect_checkpoint_rejected(
    const std::string& data,
    const std::string_view description)
{
    try {
        static_cast<void>(load_saved_checkpoint(data));
        expect_checkpoint(false, description);
    } catch (const std::exception&) {
    }
}

// Verifies old checkpoints fail with the public compatibility error promised to users.
void expect_unsupported_version(const std::string& data)
{
    try {
        static_cast<void>(load_saved_checkpoint(data));
        expect_checkpoint(false, "version-1 checkpoint is rejected");
    } catch (const std::runtime_error& error) {
        expect_checkpoint(
            std::string_view(error.what()) == "unsupported EvoBrainBot checkpoint version",
            "version-1 checkpoint reports the unsupported-version error");
    } catch (...) {
        expect_checkpoint(false, "version-1 checkpoint uses runtime compatibility error");
    }
}

// Verifies binary save/load preserves every deterministic state field exactly.
void test_checkpoint_round_trip()
{
    evobrain::Simulation original(checkpoint_config(991));
    original.run_for(37);
    evobrain::Simulation restored = load_saved_checkpoint(saved_checkpoint(original));

    expect_checkpoint(original.snapshot() == restored.snapshot(),
        "checkpoint round trip preserves complete state");
}

// Verifies resumed execution equals the same uninterrupted seeded execution.
void test_checkpoint_continuation()
{
    evobrain::Simulation uninterrupted(checkpoint_config(552));
    uninterrupted.run_for(60);

    evobrain::Simulation first_part(checkpoint_config(552));
    first_part.run_for(23);
    evobrain::Simulation resumed = load_saved_checkpoint(saved_checkpoint(first_part));
    resumed.run_for(37);

    expect_checkpoint(uninterrupted.snapshot() == resumed.snapshot(),
        "checkpoint continuation matches uninterrupted execution");
}

// Verifies zero-tick initial state can be checkpointed and restored.
void test_initial_checkpoint()
{
    evobrain::Simulation initial(checkpoint_config(0));
    evobrain::Simulation restored = load_saved_checkpoint(saved_checkpoint(initial));
    expect_checkpoint(initial.snapshot() == restored.snapshot(),
        "initial state checkpoint round trips");
}

// Verifies invalid identifiers, versions, truncation, and trailing data fail.
void test_invalid_checkpoints()
{
    evobrain::Simulation simulation(checkpoint_config(3));
    const std::string valid = saved_checkpoint(simulation);

    std::string invalid_identifier = valid;
    invalid_identifier[0] = 'X';
    expect_checkpoint_rejected(invalid_identifier,
        "invalid checkpoint identifier is rejected");

    std::string unsupported_version = valid;
    // The four-byte little-endian version immediately follows the 8-byte magic.
    unsupported_version[8] = 1;
    expect_unsupported_version(unsupported_version);

    expect_checkpoint_rejected(valid.substr(0, valid.size() / 2),
        "truncated checkpoint is rejected");
    expect_checkpoint_rejected(valid + "extra",
        "checkpoint trailing data is rejected");
}

} // namespace

int run_checkpoint_tests()
{
    test_checkpoint_round_trip();
    test_checkpoint_continuation();
    test_initial_checkpoint();
    test_invalid_checkpoints();
    return checkpoint_failure_count;
}
