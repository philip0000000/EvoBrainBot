#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace evobrain {

inline constexpr std::size_t eye_count = 2;
inline constexpr std::size_t rays_per_eye = 3;
inline constexpr std::size_t vision_ray_count = eye_count * rays_per_eye;
inline constexpr std::size_t vision_channels_per_ray = 4;
inline constexpr std::size_t brain_input_count = vision_ray_count * vision_channels_per_ray + 2;
inline constexpr std::size_t brain_founder_hidden_count = 8;
inline constexpr std::size_t brain_hidden_count = 12;
inline constexpr std::size_t brain_output_count = 3;
inline constexpr std::size_t input_hidden_weight_count = brain_input_count * brain_hidden_count;
inline constexpr std::size_t hidden_bias_offset = input_hidden_weight_count;
inline constexpr std::size_t hidden_output_weight_offset = hidden_bias_offset + brain_hidden_count;
inline constexpr std::size_t output_bias_offset =
    hidden_output_weight_offset + brain_hidden_count * brain_output_count;
inline constexpr std::size_t brain_parameter_count = output_bias_offset + brain_output_count;
inline constexpr std::size_t recurrent_weight_count = brain_hidden_count * brain_hidden_count;

using BrainParameters = std::array<double, brain_parameter_count>;

// Stores heritable connection topology separately from numeric brain parameters.
struct BrainStructure {
    std::uint8_t founder_fast_path = 0;
    std::array<std::uint8_t, brain_hidden_count> hidden_active {};
    std::array<std::uint8_t, input_hidden_weight_count> input_hidden_enabled {};
    std::array<std::uint8_t, brain_hidden_count * brain_output_count>
        hidden_output_enabled {};
    std::array<std::uint8_t, recurrent_weight_count> recurrent_enabled {};
    std::array<double, recurrent_weight_count> recurrent_weights {};
    bool operator==(const BrainStructure&) const = default;
};

// Stores per-agent recurrent memory. It persists across ticks but is not inherited.
struct BrainState {
    std::array<double, brain_hidden_count> previous_hidden {};
    std::array<double, brain_hidden_count> next_hidden {};
    bool operator==(const BrainState&) const = default;
};

// Contains the normalized RGB-and-proximity reading of one literal vision ray.
struct VisionRayInputs {
    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
    double proximity = 0.0;
    bool operator==(const VisionRayInputs&) const = default;
};

// Contains the 24 visual values and two internal values consumed by a brain.
struct BrainInputs {
    std::array<VisionRayInputs, vision_ray_count> vision {};
    double energy = 0.0;
    double damage = 0.0;
    bool operator==(const BrainInputs&) const = default;
};

// Contains the bounded decisions produced by an agent brain.
struct BrainOutputs {
    double turn = 0.0;
    // "Move" is the brain command; "movement" remains the noun for costs and limits.
    double move = 0.0;
    double eat = 0.0;
    bool operator==(const BrainOutputs&) const = default;
};

// Creates the default eight-active-neuron feed-forward founder topology.
[[nodiscard]] BrainStructure founder_brain_structure() noexcept;

// Reports whether a topology can use the specialized founder-compatible CPU path.
[[nodiscard]] bool is_founder_feed_forward(const BrainStructure& structure) noexcept;

// Evaluates one brain tick using previous-tick recurrent state and order-independent buffers.
[[nodiscard]] BrainOutputs evaluate_brain(
    const BrainParameters& parameters,
    const BrainStructure& structure,
    BrainState& state,
    const BrainInputs& inputs) noexcept;

// Evaluates the default feed-forward topology without retaining recurrent state.
[[nodiscard]] BrainOutputs evaluate_brain(
    const BrainParameters& parameters,
    const BrainInputs& inputs) noexcept;

} // namespace evobrain
