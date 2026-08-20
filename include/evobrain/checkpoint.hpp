#pragma once

#include "evobrain/simulation.hpp"

#include <iosfwd>

namespace evobrain {

// Writes a versioned binary checkpoint containing all deterministic state.
void save_checkpoint(const Simulation& simulation, std::ostream& output);

// Reads, validates, and restores a versioned binary simulation checkpoint.
[[nodiscard]] Simulation load_checkpoint(std::istream& input);

} // namespace evobrain
