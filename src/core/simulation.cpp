#include "evobrain/simulation.hpp"

#include <cstdint>

namespace evobrain {

Simulation::Simulation(const SimulationConfig& config) noexcept
    : random_(config.seed)
{
}

void Simulation::tick() noexcept
{
    ++current_tick_;
}

void Simulation::run_for(const std::uint64_t ticks) noexcept
{
    for (std::uint64_t tick_index = 0; tick_index < ticks; ++tick_index) {
        tick();
    }
}

std::uint64_t Simulation::current_tick() const noexcept
{
    return current_tick_;
}

} // namespace evobrain
