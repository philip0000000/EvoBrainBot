#include "evobrain/brain.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace evobrain {

BrainOutputs evaluate_brain(const BrainParameters& parameters, const BrainInputs& inputs) noexcept
{
    std::array<double, brain_input_count> input_values {};
    std::size_t input_index = 0;
    for (const VisionRayInputs& ray : inputs.vision) {
        input_values[input_index++] = ray.red;
        input_values[input_index++] = ray.green;
        input_values[input_index++] = ray.blue;
        input_values[input_index++] = ray.proximity;
    }
    input_values[input_index++] = inputs.energy;
    input_values[input_index] = inputs.damage;

    std::array<double, brain_hidden_count> hidden {};
    for (std::size_t hidden_index = 0; hidden_index < brain_hidden_count; ++hidden_index) {
        double value = parameters[hidden_bias_offset + hidden_index];
        const std::size_t weights = hidden_index * brain_input_count;
        for (std::size_t source = 0; source < brain_input_count; ++source) {
            value += parameters[weights + source] * input_values[source];
        }
        hidden[hidden_index] = std::clamp(value, -1.0, 1.0);
    }

    std::array<double, brain_output_count> output {};
    for (std::size_t output_index = 0; output_index < brain_output_count; ++output_index) {
        double value = parameters[output_bias_offset + output_index];
        const std::size_t weights = hidden_output_weight_offset
            + output_index * brain_hidden_count;
        for (std::size_t source = 0; source < brain_hidden_count; ++source) {
            value += parameters[weights + source] * hidden[source];
        }
        output[output_index] = std::clamp(value, -1.0, 1.0);
    }
    return {.turn = output[0], .move = (output[1] + 1.0) * 0.5,
        .eat = (output[2] + 1.0) * 0.5};
}

} // namespace evobrain
