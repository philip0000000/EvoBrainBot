#include "evobrain/viewer/agent_selection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace evobrain::viewer {
namespace {

// Returns the base coordinate plus copies needed to show edge overlap.
std::pair<std::array<double, 3>, std::size_t> wrapped_coordinates(
    const double coordinate,
    const double margin,
    const double extent) noexcept
{
    std::array<double, 3> coordinates {coordinate, 0.0, 0.0};
    std::size_t count = 1;
    if (coordinate < margin) {
        coordinates[count++] = coordinate + extent;
    }
    if (coordinate > extent - margin) {
        coordinates[count++] = coordinate - extent;
    }
    return {coordinates, count};
}

} // namespace

double agent_visual_radius(const double zoom) noexcept
{
    return std::clamp(4.0 * zoom, 2.0, 20.0);
}

bool agent_information_shortcut_pressed(
    const bool key_pressed,
    const bool wants_text_input) noexcept
{
    return key_pressed && !wants_text_input;
}

AgentScreenCopies agent_screen_copies(
    const AgentVisual& agent,
    const Camera& camera,
    const CameraViewport& viewport) noexcept
{
    AgentScreenCopies result;
    const double pixels_per_world = std::max(std::min(
        viewport.width / camera.world_width(), viewport.height / camera.world_height())
            * camera.zoom(), 1.0);
    // The renderer's heading extends beyond the body, so copies begin slightly
    // before the circle itself reaches an edge.
    const double wrap_margin = agent_visual_radius(camera.zoom()) * 2.5
        / pixels_per_world;
    const auto [x_coordinates, x_count] =
        wrapped_coordinates(agent.x, wrap_margin, camera.world_width());
    const auto [y_coordinates, y_count] =
        wrapped_coordinates(agent.y, wrap_margin, camera.world_height());
    for (std::size_t x = 0; x < x_count; ++x) {
        for (std::size_t y = 0; y < y_count; ++y) {
            result.centers[result.count++] = camera.world_to_screen(
                {.x = x_coordinates[x], .y = y_coordinates[y]}, viewport);
        }
    }
    return result;
}

std::optional<std::uint64_t> select_agent_at_screen(
    const RenderSnapshot& snapshot,
    const Camera& camera,
    const CameraViewport& viewport,
    const double screen_x,
    const double screen_y) noexcept
{
    if (!snapshot.contains_world) {
        return std::nullopt;
    }
    const Vec2 clicked_world = camera.screen_to_world(screen_x, screen_y, viewport);
    if (clicked_world.x < 0.0 || clicked_world.x >= snapshot.world_width
        || clicked_world.y < 0.0 || clicked_world.y >= snapshot.world_height) {
        // Entity rendering is clipped to the configured world, so invisible circle
        // fragments in the camera-only exterior must not remain clickable.
        return std::nullopt;
    }
    const double pixels_per_world = std::max(std::min(
        viewport.width / camera.world_width(), viewport.height / camera.world_height())
            * camera.zoom(), 1.0);
    const double radius = std::max(snapshot.agent_radius * pixels_per_world, 2.0);
    const double radius_squared = radius * radius;
    double nearest_distance_squared = std::numeric_limits<double>::infinity();
    std::optional<std::uint64_t> selected;
    for (const AgentVisual& agent : snapshot.agents) {
        const AgentScreenCopies copies = agent_screen_copies(agent, camera, viewport);
        for (std::size_t index = 0; index < copies.count; ++index) {
            const double delta_x = copies.centers[index].x - screen_x;
            const double delta_y = copies.centers[index].y - screen_y;
            const double distance_squared = delta_x * delta_x + delta_y * delta_y;
            if (distance_squared > radius_squared) {
                continue;
            }
            if (distance_squared < nearest_distance_squared
                || (distance_squared == nearest_distance_squared
                    && (!selected || agent.id > *selected))) {
                nearest_distance_squared = distance_squared;
                selected = agent.id;
            }
        }
    }
    return selected;
}

} // namespace evobrain::viewer
