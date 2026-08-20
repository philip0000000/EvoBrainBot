#include "evobrain/checkpoint.hpp"
#include "evobrain/simulation.hpp"
#include "evobrain/viewer/camera.hpp"
#include "evobrain/viewer/simulation_worker.hpp"

#include <Windows.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <thread>

namespace {

int failure_count = 0;

// Records one failed worker expectation while allowing later tests to run.
void expect(const bool condition, const std::string_view description)
{
    if (!condition) {
        std::cerr << "FAILED: " << description << '\n';
        ++failure_count;
    }
}

// Owns an isolated directory used only by this process's checkpoint tests.
class TemporaryDirectory {
public:
    // Recreates a process-unique directory before the first checkpoint test.
    TemporaryDirectory()
        : path_(std::filesystem::temp_directory_path()
            / (L"evobrain-viewer-tests-" + std::to_wstring(GetCurrentProcessId())))
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
        std::filesystem::create_directories(path_);
    }

    // Removes every test checkpoint when the fixture leaves scope.
    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    // Returns the unique directory reserved for this test process.
    [[nodiscard]] const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

// Waits briefly for automatic playback to publish at least the requested tick.
bool wait_for_tick(
    evobrain::viewer::SimulationWorker& worker,
    const std::uint64_t tick)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        worker.request_render_snapshot();
        const auto snapshot = worker.latest_render_snapshot();
        if (snapshot && snapshot->stats.completed_ticks >= tick) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

// Verifies worker steps preserve the core's deterministic tick ordering.
void test_worker_determinism_and_step()
{
    evobrain::Simulation direct(evobrain::SimulationConfig {.seed = 4401});
    const evobrain::SimulationSnapshot initial = direct.snapshot();
    direct.run_for(25);

    evobrain::viewer::SimulationWorker worker;
    expect(worker.load_snapshot(initial).succeeded, "worker accepts a valid snapshot");
    for (int tick = 0; tick < 25; ++tick) {
        expect(worker.step().succeeded, "paused Step succeeds");
    }

    const auto worker_state = worker.copy_simulation_snapshot();
    expect(worker_state.has_value(), "worker exposes an explicit exact snapshot");
    expect(worker_state && *worker_state == direct.snapshot(),
        "worker stepping equals direct core execution");
    expect(worker.status().has_unsaved_changes,
        "stepping marks the loaded simulation unsaved");
}

// Verifies pause completes between ticks and prevents further advancement.
void test_tick_boundary_pause()
{
    evobrain::Simulation simulation(evobrain::SimulationConfig {.seed = 72});
    evobrain::viewer::SimulationWorker worker;
    expect(worker.load_snapshot(simulation.snapshot()).succeeded,
        "pause test loads its simulation");
    expect(worker.set_target_ticks_per_second(1000).succeeded,
        "pause test accepts maximum target rate");
    expect(worker.run().succeeded, "Run starts automatic playback");
    expect(wait_for_tick(worker, 5), "automatic playback completes ticks");
    expect(worker.pause().succeeded, "Pause succeeds at a tick boundary");

    const auto paused = worker.copy_simulation_snapshot();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto later = worker.copy_simulation_snapshot();
    expect(paused && later && *paused == *later,
        "paused worker does not advance after Pause returns");
    expect(worker.status().playback == evobrain::viewer::PlaybackState::paused,
        "worker publishes paused state");
}

// Verifies each published render copy comes from one complete simulation tick.
void test_complete_render_snapshot()
{
    evobrain::Simulation simulation(evobrain::SimulationConfig {.seed = 608});
    simulation.run_for(3);
    evobrain::viewer::SimulationWorker worker;
    expect(worker.load_snapshot(simulation.snapshot()).succeeded,
        "render snapshot test loads its simulation");

    const auto rendered = worker.latest_render_snapshot();
    expect(rendered != nullptr, "load immediately publishes a render snapshot");
    if (!rendered) {
        return;
    }
    expect(rendered->stats == simulation.stats(),
        "render statistics come from the loaded completed tick");
    expect(rendered->agents.size() == simulation.agents().size(),
        "render snapshot contains every agent");
    expect(rendered->food.size() == simulation.food().size(),
        "render snapshot contains every food item");
    expect(rendered->agents.front().x
            == static_cast<float>(simulation.agents().front().position.x)
        && rendered->agents.front().y
            == static_cast<float>(simulation.agents().front().position.y),
        "agent visual coordinates match the same simulation state");
}

// Verifies a failed replacement load leaves the prior simulation untouched.
void test_failed_load_preserves_state(const TemporaryDirectory& directory)
{
    evobrain::Simulation simulation(evobrain::SimulationConfig {.seed = 99});
    simulation.run_for(4);
    evobrain::viewer::SimulationWorker worker;
    expect(worker.load_snapshot(simulation.snapshot()).succeeded,
        "failed-load test loads its initial state");
    const auto before = worker.copy_simulation_snapshot();

    const auto invalid_path = directory.path() / L"invalid.evo";
    {
        std::ofstream invalid(invalid_path, std::ios::binary);
        invalid << "not an EvoBrainBot checkpoint";
    }
    const auto result = worker.load_checkpoint_file(invalid_path);
    const auto after = worker.copy_simulation_snapshot();
    expect(!result.succeeded, "invalid checkpoint load reports failure");
    expect(!result.summary.empty() && !result.detail.empty(),
        "load failure contains a summary and technical detail");
    expect(before && after && *before == *after,
        "invalid checkpoint load preserves the existing simulation");
}

// Verifies atomic save, overwrite protection, reload, and dirty transitions.
void test_save_reload_and_unsaved_state(const TemporaryDirectory& directory)
{
    evobrain::Simulation simulation(evobrain::SimulationConfig {.seed = 818});
    evobrain::viewer::SimulationWorker worker;
    expect(worker.load_snapshot(simulation.snapshot()).succeeded,
        "save test loads its initial state");
    expect(!worker.status().has_unsaved_changes,
        "freshly loaded state starts clean");
    expect(worker.step().succeeded, "save test advances one tick");
    expect(worker.status().has_unsaved_changes, "advanced state becomes dirty");
    const auto advanced = worker.copy_simulation_snapshot();

    const auto checkpoint_path = directory.path() / L"continued.evo";
    expect(worker.save_checkpoint_file(checkpoint_path, false).succeeded,
        "save creates a new checkpoint atomically");
    expect(!worker.status().has_unsaved_changes,
        "successful save marks current state clean");

    expect(worker.step().succeeded, "state can advance after saving");
    expect(!worker.save_checkpoint_file(checkpoint_path, false).succeeded,
        "save refuses to overwrite without approval");
    expect(worker.status().has_unsaved_changes,
        "failed save does not clear the dirty state");
    expect(worker.load_checkpoint_file(checkpoint_path).succeeded,
        "saved checkpoint reloads through the worker");
    const auto restored = worker.copy_simulation_snapshot();
    expect(advanced && restored && *advanced == *restored,
        "saved and reloaded state is exactly equal");
    expect(!worker.status().has_unsaved_changes,
        "successful reload establishes a clean baseline");
}

// Verifies target-rate validation leaves the prior accepted value unchanged.
void test_speed_validation()
{
    evobrain::viewer::SimulationWorker worker;
    expect(worker.set_target_ticks_per_second(240).succeeded,
        "valid target TPS is accepted");
    expect(!worker.set_target_ticks_per_second(0).succeeded,
        "zero target TPS is rejected");
    expect(worker.status().target_ticks_per_second == 240,
        "rejected low target preserves previous value");
    expect(!worker.set_target_ticks_per_second(1001).succeeded,
        "target TPS above maximum is rejected");
    expect(worker.status().target_ticks_per_second == 240,
        "rejected high target preserves previous value");
}

// Verifies destruction stops and joins a worker that is actively advancing.
void test_clean_worker_shutdown()
{
    const auto started = std::chrono::steady_clock::now();
    {
        evobrain::Simulation simulation(evobrain::SimulationConfig {.seed = 7});
        evobrain::viewer::SimulationWorker worker;
        expect(worker.load_snapshot(simulation.snapshot()).succeeded,
            "shutdown test loads its simulation");
        expect(worker.run().succeeded, "shutdown test starts playback");
    }
    expect(std::chrono::steady_clock::now() - started < std::chrono::seconds(2),
        "active worker stops and joins promptly");
}

// Verifies Fast-forward omits world copies, remains responsive, and republishes
// a complete paused world at the exact tick where Pause returns.
void test_fast_forward_snapshot_policy()
{
    evobrain::Simulation simulation(evobrain::SimulationConfig {.seed = 701});
    evobrain::viewer::SimulationWorker worker;
    expect(worker.load_snapshot(simulation.snapshot()).succeeded,
        "fast-forward test loads its simulation");
    expect(worker.fast_forward().succeeded, "Fast-forward starts");
    expect(wait_for_tick(worker, 5), "Fast-forward publishes progress statistics");
    const auto fast_snapshot = worker.latest_render_snapshot();
    expect(fast_snapshot && !fast_snapshot->contains_world,
        "Fast-forward statistics omit world entity vectors");

    const auto pause_started = std::chrono::steady_clock::now();
    expect(worker.pause().succeeded, "Fast-forward pauses successfully");
    expect(std::chrono::steady_clock::now() - pause_started
            < std::chrono::milliseconds(250),
        "Fast-forward Pause remains responsive");
    const auto paused_snapshot = worker.latest_render_snapshot();
    expect(paused_snapshot && paused_snapshot->contains_world,
        "Pause republishes a complete world snapshot");
    expect(paused_snapshot
            && paused_snapshot->agents.size() == paused_snapshot->stats.population
            && paused_snapshot->food.size() == paused_snapshot->stats.food,
        "paused world snapshot contains every current entity");
}

// Verifies reset transforms the unit world consistently for a wide viewport.
void test_camera_reset_and_transform()
{
    const evobrain::viewer::CameraViewport viewport {
        .x = 10.0, .y = 20.0, .width = 1600.0, .height = 900.0};
    evobrain::viewer::Camera camera;
    camera.reset(viewport);

    const evobrain::Vec2 center = camera.world_to_screen({.x = 0.5, .y = 0.5}, viewport);
    expect(center == evobrain::Vec2 {.x = 810.0, .y = 470.0},
        "camera reset centers the world in its viewport");
    expect(camera.screen_to_world(center.x, center.y, viewport)
            == evobrain::Vec2 {.x = 0.5, .y = 0.5},
        "camera screen and world transforms invert at the center");
    expect(camera.zoom() == 1.0, "camera reset restores 1x zoom");
}

// Verifies cursor zoom anchoring, hard zoom limits, and outer pan limits.
void test_camera_zoom_and_pan_limits()
{
    const evobrain::viewer::CameraViewport viewport {
        .width = 900.0, .height = 900.0};
    evobrain::viewer::Camera camera;
    const evobrain::Vec2 before = camera.screen_to_world(675.0, 225.0, viewport);
    camera.zoom_at(675.0, 225.0, 2.0, viewport);
    const evobrain::Vec2 after = camera.screen_to_world(675.0, 225.0, viewport);
    expect(std::abs(before.x - after.x) < 1e-12
            && std::abs(before.y - after.y) < 1e-12,
        "zoom preserves the world point beneath the cursor");

    camera.zoom_at(450.0, 450.0, 1e9, viewport);
    expect(camera.zoom() == evobrain::viewer::Camera::maximum_zoom,
        "camera clamps maximum zoom");
    camera.pan_pixels(-1e9, -1e9, viewport);
    const auto high_bounds = camera.visible_bounds(viewport);
    expect(high_bounds.maximum_x <= evobrain::viewer::Camera::outer_maximum + 1e-12
            && high_bounds.maximum_y <= evobrain::viewer::Camera::outer_maximum + 1e-12,
        "camera pan stops at the outer maximum boundary");
    camera.pan_pixels(1e9, 1e9, viewport);
    const auto low_bounds = camera.visible_bounds(viewport);
    expect(low_bounds.minimum_x >= evobrain::viewer::Camera::outer_minimum - 1e-12
            && low_bounds.minimum_y >= evobrain::viewer::Camera::outer_minimum - 1e-12,
        "camera pan stops at the outer minimum boundary");

    camera.zoom_at(450.0, 450.0, 1e-9, viewport);
    expect(camera.zoom() == evobrain::viewer::Camera::minimum_zoom,
        "camera clamps minimum zoom");
    expect(camera.center() == evobrain::Vec2 {.x = 0.5, .y = 0.5},
        "fully zoomed-out camera centers the complete outer world");
}

// Verifies resizing preserves zoom and center unless a boundary requires clamp.
void test_camera_resize_preservation()
{
    const evobrain::viewer::CameraViewport square {
        .width = 800.0, .height = 800.0};
    evobrain::viewer::Camera camera;
    camera.zoom_at(400.0, 400.0, 4.0, square);
    camera.pan_pixels(-80.0, 40.0, square);
    const evobrain::Vec2 before_center = camera.center();
    const double before_zoom = camera.zoom();

    camera.viewport_changed({.width = 1000.0, .height = 700.0});
    expect(camera.center() == before_center,
        "ordinary resize preserves camera center");
    expect(camera.zoom() == before_zoom,
        "resize preserves camera zoom");
}

} // namespace

// Runs the non-rendering viewer worker and checkpoint operation tests.
int main()
{
    const TemporaryDirectory directory;
    test_worker_determinism_and_step();
    test_tick_boundary_pause();
    test_complete_render_snapshot();
    test_failed_load_preserves_state(directory);
    test_save_reload_and_unsaved_state(directory);
    test_speed_validation();
    test_clean_worker_shutdown();
    test_fast_forward_snapshot_policy();
    test_camera_reset_and_transform();
    test_camera_zoom_and_pan_limits();
    test_camera_resize_preservation();

    if (failure_count != 0) {
        std::cerr << failure_count << " viewer worker test expectation(s) failed\n";
        return 1;
    }
    std::cout << "All viewer worker tests passed\n";
    return 0;
}
