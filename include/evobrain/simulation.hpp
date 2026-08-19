#pragma once

#include "evobrain/random.hpp"

#include <cstdint>

namespace evobrain {

// Supplies the explicit inputs required to create a reproducible simulation.
struct SimulationConfig {
    std::uint64_t seed;
};

// Owns the deterministic state of a fixed-step EvoBrainBot simulation.
class Simulation {
public:
    // Creates a simulation at tick zero using the configured random seed.
    explicit Simulation(const SimulationConfig& config) noexcept;

    // Advances all simulation state by exactly one fixed tick.
    void tick() noexcept;

    // Advances the simulation through the requested number of fixed ticks.
    void run_for(std::uint64_t ticks) noexcept;

    // Returns the number of ticks completed by this simulation.
    [[nodiscard]] std::uint64_t current_tick() const noexcept;

private:
    std::uint64_t current_tick_ = 0;
    Pcg32 random_;
};

} // namespace evobrain
