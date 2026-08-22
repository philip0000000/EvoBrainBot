#pragma once

#include <array>
#include <cstddef>

namespace evobrain {

inline constexpr std::size_t eye_count = 2;
inline constexpr std::size_t rays_per_eye = 3;
inline constexpr std::size_t vision_ray_count = eye_count * rays_per_eye;
inline constexpr std::size_t vision_channels_per_ray = 4;
inline constexpr std::size_t brain_input_count = vision_ray_count * vision_channels_per_ray + 2;
inline constexpr std::size_t brain_hidden_count = 8;
inline constexpr std::size_t brain_output_count = 3;
inline constexpr std::size_t input_hidden_weight_count = brain_input_count * brain_hidden_count;
inline constexpr std::size_t hidden_bias_offset = input_hidden_weight_count;
inline constexpr std::size_t hidden_output_weight_offset = hidden_bias_offset + brain_hidden_count;
inline constexpr std::size_t output_bias_offset =
    hidden_output_weight_offset + brain_hidden_count * brain_output_count;
inline constexpr std::size_t brain_parameter_count = output_bias_offset + brain_output_count;

using BrainParameters = std::array<double, brain_parameter_count>;

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

// Contains the bounded decisions produced by the fixed agent brain.
struct BrainOutputs {
    double turn = 0.0;
    // "Move" is the brain command; "movement" remains the noun for costs and limits.
    double move = 0.0;
    double eat = 0.0;
    bool operator==(const BrainOutputs&) const = default;
};

// Evaluates the fixed 26-to-8-to-3 brain using clamped-linear activation.
[[nodiscard]] BrainOutputs evaluate_brain(
    const BrainParameters& parameters,
    const BrainInputs& inputs) noexcept;

} // namespace evobrain
