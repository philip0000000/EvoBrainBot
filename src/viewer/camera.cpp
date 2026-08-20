#include "evobrain/viewer/camera.hpp"

#include <algorithm>
#include <cmath>

namespace evobrain::viewer {
namespace {

// Returns a positive dimension so minimized windows cannot create infinities.
double valid_dimension(const double value) noexcept
{
    return std::max(value, 1.0);
}

// Clamps one camera axis, centering views wider than the allowed outer range.
double clamp_axis(const double value, const double half_extent) noexcept
{
    const double outer_half_extent = (Camera::outer_maximum - Camera::outer_minimum) * 0.5;
    if (half_extent >= outer_half_extent) {
        return (Camera::outer_minimum + Camera::outer_maximum) * 0.5;
    }
    return std::clamp(
        value,
        Camera::outer_minimum + half_extent,
        Camera::outer_maximum - half_extent);
}

} // namespace

void Camera::reset(const CameraViewport& viewport) noexcept
{
    center_ = Vec2 {.x = 0.5, .y = 0.5};
    zoom_ = 1.0;
    clamp_center(viewport);
}

void Camera::zoom_at(
    const double screen_x,
    const double screen_y,
    const double factor,
    const CameraViewport& viewport) noexcept
{
    if (!std::isfinite(factor) || factor <= 0.0) {
        return;
    }
    const Vec2 anchor_before = screen_to_world(screen_x, screen_y, viewport);
    zoom_ = std::clamp(zoom_ * factor, minimum_zoom, maximum_zoom);
    const Vec2 anchor_after = screen_to_world(screen_x, screen_y, viewport);
    center_.x += anchor_before.x - anchor_after.x;
    center_.y += anchor_before.y - anchor_after.y;
    clamp_center(viewport);
}

void Camera::pan_pixels(
    const double delta_x,
    const double delta_y,
    const CameraViewport& viewport) noexcept
{
    const double pixels_per_world_unit = scale(viewport);
    center_.x -= delta_x / pixels_per_world_unit;
    center_.y -= delta_y / pixels_per_world_unit;
    clamp_center(viewport);
}

void Camera::viewport_changed(const CameraViewport& viewport) noexcept
{
    clamp_center(viewport);
}

Vec2 Camera::world_to_screen(
    const Vec2 world,
    const CameraViewport& viewport) const noexcept
{
    const double pixels_per_world_unit = scale(viewport);
    return Vec2 {
        .x = viewport.x + viewport.width * 0.5
            + (world.x - center_.x) * pixels_per_world_unit,
        .y = viewport.y + viewport.height * 0.5
            + (world.y - center_.y) * pixels_per_world_unit,
    };
}

Vec2 Camera::screen_to_world(
    const double screen_x,
    const double screen_y,
    const CameraViewport& viewport) const noexcept
{
    const double pixels_per_world_unit = scale(viewport);
    return Vec2 {
        .x = center_.x
            + (screen_x - viewport.x - viewport.width * 0.5)
                / pixels_per_world_unit,
        .y = center_.y
            + (screen_y - viewport.y - viewport.height * 0.5)
                / pixels_per_world_unit,
    };
}

WorldBounds Camera::visible_bounds(const CameraViewport& viewport) const noexcept
{
    const double pixels_per_world_unit = scale(viewport);
    const double half_width = valid_dimension(viewport.width) * 0.5 / pixels_per_world_unit;
    const double half_height = valid_dimension(viewport.height) * 0.5 / pixels_per_world_unit;
    return WorldBounds {
        .minimum_x = center_.x - half_width,
        .minimum_y = center_.y - half_height,
        .maximum_x = center_.x + half_width,
        .maximum_y = center_.y + half_height,
    };
}

Vec2 Camera::center() const noexcept
{
    return center_;
}

double Camera::zoom() const noexcept
{
    return zoom_;
}

double Camera::scale(const CameraViewport& viewport) const noexcept
{
    return std::min(valid_dimension(viewport.width), valid_dimension(viewport.height))
        * zoom_;
}

void Camera::clamp_center(const CameraViewport& viewport) noexcept
{
    const WorldBounds bounds = visible_bounds(viewport);
    const double half_width = (bounds.maximum_x - bounds.minimum_x) * 0.5;
    const double half_height = (bounds.maximum_y - bounds.minimum_y) * 0.5;
    center_.x = clamp_axis(center_.x, half_width);
    center_.y = clamp_axis(center_.y, half_height);
}

} // namespace evobrain::viewer
