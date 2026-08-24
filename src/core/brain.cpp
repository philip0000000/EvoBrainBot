#include "evobrain/brain.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace evobrain {
namespace {

// Flattens the public sensor structure into the stable numeric input order.
std::array<double, brain_input_count> flatten_inputs(const BrainInputs& inputs) noexcept
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
    return input_values;
}

// Maps the three clamped raw outputs to the simulation's public action ranges.
BrainOutputs map_outputs(const std::array<double, brain_output_count>& output) noexcept
{
    return {.turn = output[0], .move = (output[1] + 1.0) * 0.5,
        .eat = (output[2] + 1.0) * 0.5};
}

} // namespace

BrainStructure founder_brain_structure() noexcept
{
    BrainStructure structure;
    structure.founder_fast_path = 1;
    for (std::size_t hidden = 0; hidden < brain_founder_hidden_count; ++hidden) {
        structure.hidden_active[hidden] = 1;
        for (std::size_t input = 0; input < brain_input_count; ++input) {
            structure.input_hidden_enabled[hidden * brain_input_count + input] = 1;
        }
        for (std::size_t output = 0; output < brain_output_count; ++output) {
            structure.hidden_output_enabled[output * brain_hidden_count + hidden] = 1;
        }
    }
    return structure;
}

bool is_founder_feed_forward(const BrainStructure& structure) noexcept
{
    return structure.founder_fast_path != 0;
}

BrainOutputs evaluate_brain(const BrainParameters& parameters,
    const BrainStructure& structure, BrainState& state, const BrainInputs& inputs) noexcept
{
    const std::array<double, brain_input_count> input_values = flatten_inputs(inputs);

    if (is_founder_feed_forward(structure)) {
        std::array<double, brain_founder_hidden_count> hidden_values {};
        for (std::size_t hidden = 0; hidden < brain_founder_hidden_count; ++hidden) {
            double value = parameters[hidden_bias_offset + hidden];
            const std::size_t weights = hidden * brain_input_count;
            for (std::size_t input = 0; input < brain_input_count; ++input) {
                value += parameters[weights + input] * input_values[input];
            }
            hidden_values[hidden] = std::clamp(value, -1.0, 1.0);
        }
        std::array<double, brain_output_count> output {};
        for (std::size_t output_index = 0; output_index < brain_output_count;
             ++output_index) {
            double value = parameters[output_bias_offset + output_index];
            const std::size_t weights = hidden_output_weight_offset
                + output_index * brain_hidden_count;
            for (std::size_t hidden = 0; hidden < brain_founder_hidden_count; ++hidden) {
                value += parameters[weights + hidden] * hidden_values[hidden];
            }
            output[output_index] = std::clamp(value, -1.0, 1.0);
        }
        state = {};
        return map_outputs(output);
    }

    for (std::size_t hidden = 0; hidden < brain_hidden_count; ++hidden) {
        if (structure.hidden_active[hidden] == 0) {
            state.next_hidden[hidden] = 0.0;
            continue;
        }
        double value = parameters[hidden_bias_offset + hidden];
        const std::size_t input_weights = hidden * brain_input_count;
        for (std::size_t input = 0; input < brain_input_count; ++input) {
            const std::size_t connection = input_weights + input;
            if (structure.input_hidden_enabled[connection] != 0) {
                value += parameters[connection] * input_values[input];
            }
        }
        const std::size_t recurrent_weights = hidden * brain_hidden_count;
        for (std::size_t source = 0; source < brain_hidden_count; ++source) {
            const std::size_t connection = recurrent_weights + source;
            if (structure.recurrent_enabled[connection] != 0
                && structure.hidden_active[source] != 0) {
                // Only the completed previous tick is visible, so loop order is irrelevant.
                value += structure.recurrent_weights[connection]
                    * state.previous_hidden[source];
            }
        }
        state.next_hidden[hidden] = std::clamp(value, -1.0, 1.0);
    }

    std::array<double, brain_output_count> output {};
    for (std::size_t output_index = 0; output_index < brain_output_count; ++output_index) {
        double value = parameters[output_bias_offset + output_index];
        const std::size_t weights = hidden_output_weight_offset
            + output_index * brain_hidden_count;
        const std::size_t enabled = output_index * brain_hidden_count;
        for (std::size_t source = 0; source < brain_hidden_count; ++source) {
            if (structure.hidden_output_enabled[enabled + source] != 0) {
                value += parameters[weights + source] * state.next_hidden[source];
            }
        }
        output[output_index] = std::clamp(value, -1.0, 1.0);
    }
    state.previous_hidden = state.next_hidden;
    return map_outputs(output);
}

BrainOutputs evaluate_brain(const BrainParameters& parameters, const BrainInputs& inputs) noexcept
{
    BrainState state;
    return evaluate_brain(parameters, founder_brain_structure(), state, inputs);
}

} // namespace evobrain
