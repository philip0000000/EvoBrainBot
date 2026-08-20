#include "evobrain/brain.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace evobrain {
namespace {

// Evaluates one output's four weights and trailing bias.
double evaluate_raw_output(
    const BrainParameters& parameters,
    const std::size_t output_index,
    const std::array<double, brain_input_count>& inputs) noexcept
{
    const std::size_t offset = output_index * parameters_per_brain_output;
    double value = parameters[offset + brain_input_count];
    for (std::size_t input_index = 0; input_index < brain_input_count;
         ++input_index) {
        value += parameters[offset + input_index] * inputs[input_index];
    }
    return value;
}

} // namespace

BrainOutputs evaluate_brain(
    const BrainParameters& parameters,
    const BrainInputs& inputs) noexcept
{
    const std::array input_values {
        inputs.food_direction_sine,
        inputs.food_direction_cosine,
        inputs.food_distance,
        inputs.energy,
    };

    const double turn = std::clamp(
        evaluate_raw_output(parameters, 0, input_values), -1.0, 1.0);
    const double movement_raw = std::clamp(
        evaluate_raw_output(parameters, 1, input_values), -1.0, 1.0);
    return BrainOutputs {
        .turn = turn,
        .movement = (movement_raw + 1.0) * 0.5,
    };
}

} // namespace evobrain
