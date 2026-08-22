#pragma once

#include "evobrain/simulation.hpp"

namespace evobrain::viewer {

// Describes the drawable screen rectangle controlled by the world camera.
struct CameraViewport {
    double x = 0.0;
    double y = 0.0;
    double width = 1.0;
    double height = 1.0;
};

// Stores an axis-aligned world rectangle.
struct WorldBounds {
    double minimum_x = 0.0;
    double minimum_y = 0.0;
    double maximum_x = 0.0;
    double maximum_y = 0.0;
};

// Controls a non-rotating 2D view bounded around configurable world dimensions.
class Camera {
public:
    static constexpr double minimum_zoom = 0.5;
    static constexpr double maximum_zoom = 100.0;

    // Applies positive world dimensions and resets when they change.
    void set_world_dimensions(
        double width,
        double height,
        const CameraViewport& viewport) noexcept;

    // Restores the centered 1x view and clamps it for the current viewport.
    void reset(const CameraViewport& viewport) noexcept;

    // Changes zoom around a screen point while preserving its world position.
    void zoom_at(
        double screen_x,
        double screen_y,
        double factor,
        const CameraViewport& viewport) noexcept;

    // Moves the camera by a screen-space drag and clamps it to the outer world.
    void pan_pixels(
        double delta_x,
        double delta_y,
        const CameraViewport& viewport) noexcept;

    // Reapplies camera limits after a viewport-size change.
    void viewport_changed(const CameraViewport& viewport) noexcept;

    // Converts a point from world coordinates to viewport pixels.
    [[nodiscard]] Vec2 world_to_screen(
        Vec2 world,
        const CameraViewport& viewport) const noexcept;

    // Converts a viewport pixel position to world coordinates.
    [[nodiscard]] Vec2 screen_to_world(
        double screen_x,
        double screen_y,
        const CameraViewport& viewport) const noexcept;

    // Returns the currently visible world-space rectangle.
    [[nodiscard]] WorldBounds visible_bounds(
        const CameraViewport& viewport) const noexcept;

    // Returns the current world-space center.
    [[nodiscard]] Vec2 center() const noexcept;

    // Returns the current magnification relative to the reset view.
    [[nodiscard]] double zoom() const noexcept;

    // Returns the configured horizontal world extent.
    [[nodiscard]] double world_width() const noexcept;

    // Returns the configured vertical world extent.
    [[nodiscard]] double world_height() const noexcept;

    // Returns the camera-only boundary extending half a world on every side.
    [[nodiscard]] WorldBounds outer_bounds() const noexcept;

private:
    // Returns the shared pixels-per-world-unit scale for the viewport.
    [[nodiscard]] double scale(const CameraViewport& viewport) const noexcept;

    // Keeps panning finite and within the two-world-span outer boundary.
    void clamp_center(const CameraViewport& viewport) noexcept;

    Vec2 center_ {.x = 0.5, .y = 0.5};
    double zoom_ = 1.0;
    double world_width_ = 1.0;
    double world_height_ = 1.0;
};

} // namespace evobrain::viewer
