#pragma once

#include <array>
#include <cstddef>

namespace evobrain {

inline constexpr std::size_t brain_input_count = 4;
inline constexpr std::size_t brain_output_count = 2;
inline constexpr std::size_t parameters_per_brain_output = brain_input_count + 1;
inline constexpr std::size_t brain_parameter_count =
    brain_output_count * parameters_per_brain_output;

using BrainParameters = std::array<double, brain_parameter_count>;

// Contains the normalized sensory values consumed by an agent brain.
struct BrainInputs {
    double food_direction_sine = 0.0;
    double food_direction_cosine = 0.0;
    double food_distance = 1.0;
    double energy = 0.0;

    bool operator==(const BrainInputs&) const = default;
};

// Contains the bounded movement decisions produced by an agent brain.
struct BrainOutputs {
    double turn = 0.0;
    double movement = 0.0;

    bool operator==(const BrainOutputs&) const = default;
};

// Evaluates the fixed direct 4-to-2 brain using clamped-linear activation.
[[nodiscard]] BrainOutputs evaluate_brain(
    const BrainParameters& parameters,
    const BrainInputs& inputs) noexcept;

} // namespace evobrain
