#pragma once

#include "evobrain/viewer/camera.hpp"
#include "evobrain/viewer/simulation_worker.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace evobrain::viewer {

// Contains the screen-space centers of one agent and any visible wrapped copies.
struct AgentScreenCopies {
    std::array<Vec2, 9> centers {};
    std::size_t count = 0;
};

// Returns the same zoom-clamped agent body radius used by world rendering.
[[nodiscard]] double agent_visual_radius(double zoom) noexcept;

// Returns whether the I shortcut should toggle without stealing text-entry input.
[[nodiscard]] bool agent_information_shortcut_pressed(
    bool key_pressed,
    bool wants_text_input) noexcept;

// Projects one agent and its required toroidal copies into the viewport.
[[nodiscard]] AgentScreenCopies agent_screen_copies(
    const AgentVisual& agent,
    const Camera& camera,
    const CameraViewport& viewport) noexcept;

// Selects the agent body beneath a screen point using deterministic tie-breaking.
[[nodiscard]] std::optional<std::uint64_t> select_agent_at_screen(
    const RenderSnapshot& snapshot,
    const Camera& camera,
    const CameraViewport& viewport,
    double screen_x,
    double screen_y) noexcept;

} // namespace evobrain::viewer
