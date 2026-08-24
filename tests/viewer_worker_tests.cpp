#include "evobrain/checkpoint.hpp"
#include "evobrain/simulation.hpp"
#include "evobrain/viewer/agent_selection.hpp"
#include "evobrain/viewer/camera.hpp"
#include "evobrain/viewer/simulation_worker.hpp"

#include <Windows.h>

#include <algorithm>
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
    expect(rendered->reproduction_threshold
            == simulation.config().reproduction_threshold,
        "render snapshot contains the energy-bar reference threshold");
    expect(rendered->world_width == simulation.config().world_width
            && rendered->world_height == simulation.config().world_height,
        "render snapshot contains configured world dimensions");
    expect(rendered->agent_radius == simulation.config().agent_radius
            && rendered->food_radius == simulation.config().food_radius
            && rendered->eye_range == simulation.config().eye_range,
        "render snapshot contains configured body and eye geometry");
    expect(rendered->agents.front().id == simulation.agents().front().id
            && rendered->agents.front().energy
                == static_cast<float>(simulation.agents().front().energy),
        "agent visual contains stable identity and current energy");
    expect(rendered->agents.front().x
            == static_cast<float>(simulation.agents().front().position.x)
        && rendered->agents.front().y
            == static_cast<float>(simulation.agents().front().position.y),
        "agent visual coordinates match the same simulation state");
    expect(rendered->agents.front().diet == simulation.agents().front().diet
            && rendered->agents.front().red
                == static_cast<float>(simulation.agents().front().color.red)
            && rendered->agents.front().green
                == static_cast<float>(simulation.agents().front().color.green)
            && rendered->agents.front().blue
                == static_cast<float>(simulation.agents().front().color.blue),
        "agent visual contains diet and evolved body color");
}

// Verifies full inspection state is published only for the selected stable ID.
void test_selected_agent_details()
{
    evobrain::Simulation simulation(evobrain::SimulationConfig {.seed = 912});
    const evobrain::Agent selected = simulation.agents().front();
    evobrain::viewer::SimulationWorker worker;
    expect(worker.load_snapshot(simulation.snapshot()).succeeded,
        "selection detail test loads its simulation");
    worker.select_agent(selected.id);

    const auto rendered = worker.latest_render_snapshot();
    expect(rendered && rendered->selected_agent.has_value(),
        "selecting a current stable ID publishes full details");
    if (rendered && rendered->selected_agent) {
        const auto& details = *rendered->selected_agent;
        expect(details.id == selected.id && details.position == selected.position
                && details.direction == selected.direction
                && details.energy == selected.energy && details.age == selected.age
                && details.generation == selected.generation
                && details.diet == selected.diet && details.color == selected.color
                && details.mutation_rate == selected.mutation_rate
                && details.mutation_strength == selected.mutation_strength
                && details.prior_bite_damage == selected.prior_bite_damage,
            "selected details match one complete agent state");
        expect(details.brain == selected.brain,
            "selected details contain the exact evolved brain parameters");
    }
    expect(worker.status().selected_agent_id == selected.id,
        "worker status publishes the selected stable ID");

    expect(worker.step().succeeded, "selected detail test advances one tick");
    const auto advanced_state = worker.copy_simulation_snapshot();
    const auto advanced_render = worker.latest_render_snapshot();
    bool details_follow_agent = false;
    if (advanced_state && advanced_render && advanced_render->selected_agent) {
        const auto advanced_agent = std::ranges::find(
            advanced_state->agents, selected.id, &evobrain::Agent::id);
        details_follow_agent = advanced_agent != advanced_state->agents.end()
            && advanced_render->selected_agent->position == advanced_agent->position
            && advanced_render->selected_agent->energy == advanced_agent->energy;
    }
    expect(details_follow_agent,
        "selected details follow the same stable ID after a completed tick");

    expect(worker.load_snapshot(simulation.snapshot()).succeeded,
        "selection detail test can replace the current simulation");
    expect(!worker.status().selected_agent_id
            && !worker.latest_render_snapshot()->selected_agent,
        "loading a replacement simulation clears viewer selection");

    worker.select_agent(std::uint64_t {999'999});
    const auto missing = worker.latest_render_snapshot();
    expect(missing && !missing->selected_agent,
        "a missing stable ID publishes no selected-agent details");
    expect(!worker.status().selected_agent_id,
        "a missing stable ID clears viewer selection");
}

// Verifies a completed tick clears selection when an agent dies and IDs advance.
void test_selected_agent_death()
{
    evobrain::SimulationConfig config {.seed = 43};
    config.initial_population = 1;
    config.minimum_population = 1;
    config.target_food_count = 0;
    config.food_boost_population_threshold = 0;
    config.boosted_food_count = 0;
    config.initial_energy = 0.01;
    config.living_energy_cost = 1.0;
    config.movement_energy_cost = 0.0;
    evobrain::Simulation simulation(config);
    const std::uint64_t doomed_id = simulation.agents().front().id;
    evobrain::viewer::SimulationWorker worker;
    expect(worker.load_snapshot(simulation.snapshot()).succeeded,
        "selected death test loads its simulation");
    worker.select_agent(doomed_id);
    expect(worker.step().succeeded, "selected death test advances one tick");
    const auto rendered = worker.latest_render_snapshot();
    expect(rendered && !rendered->selected_agent,
        "completed state omits details for the dead selected agent");
    expect(!worker.status().selected_agent_id,
        "dead selected agent clears viewer selection despite replacement founders");
}

// Verifies screen-space selection matches bodies, wrapped copies, and stable IDs.
void test_agent_hit_testing()
{
    const evobrain::viewer::CameraViewport viewport {
        .width = 1000.0, .height = 1000.0};
    evobrain::viewer::Camera camera;
    camera.set_world_dimensions(2.5, 2.5, viewport);
    camera.reset(viewport);
    evobrain::viewer::RenderSnapshot snapshot;
    snapshot.agents = {
        {.id = 9, .x = 1.25F, .y = 1.25F},
        {.id = 3, .x = 1.25F, .y = 1.25F},
    };
    expect(evobrain::viewer::select_agent_at_screen(
               snapshot, camera, viewport, 500.0, 500.0)
            == std::uint64_t {9},
        "overlapping equal-distance agents select the highest stable ID");
    expect(!evobrain::viewer::select_agent_at_screen(
                snapshot, camera, viewport, 100.0, 100.0),
        "clicking outside every rendered body selects no agent");

    snapshot.agents = {{.id = 77, .x = 0.001F, .y = 1.25F}};
    expect(evobrain::viewer::select_agent_at_screen(
               snapshot, camera, viewport, 999.0, 500.0)
            == std::uint64_t {77},
        "a visible wrapped body selects its underlying stable agent ID");
    expect(!evobrain::viewer::select_agent_at_screen(
                snapshot, camera, viewport, 1004.0, 500.0),
        "a wrapped body is not clickable through the clipped world exterior");
    snapshot.contains_world = false;
    expect(!evobrain::viewer::select_agent_at_screen(
                snapshot, camera, viewport, 999.0, 500.0),
        "statistics-only snapshots cannot select invisible agents");
}

// Verifies the overlay shortcut is suppressed only during text entry.
void test_agent_information_shortcut()
{
    expect(evobrain::viewer::agent_information_shortcut_pressed(true, false),
        "I toggles agent information outside text input");
    expect(!evobrain::viewer::agent_information_shortcut_pressed(true, true),
        "I is ignored while a text input owns keyboard entry");
    expect(!evobrain::viewer::agent_information_shortcut_pressed(false, false),
        "agent information remains unchanged without an I press");
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

// Verifies backend selection is transient, explicit, and limited to paused state.
void test_brain_backend_selection()
{
    evobrain::viewer::SimulationWorker worker;
    expect(!worker.set_brain_backend(evobrain::BrainBackendKind::cpu).succeeded,
        "backend selection requires a loaded simulation");
    evobrain::Simulation simulation(evobrain::SimulationConfig {.seed = 81});
    expect(worker.load_snapshot(simulation.snapshot()).succeeded,
        "backend test loads its simulation");
    expect(worker.status().brain_backend == evobrain::BrainBackendKind::cpu,
        "viewer defaults to CPU brain evaluation");
    if (worker.status().gpu_backend_available) {
        expect(worker.set_brain_backend(evobrain::BrainBackendKind::gpu).succeeded,
            "available GPU backend can be selected while paused");
    } else {
        expect(!worker.set_brain_backend(evobrain::BrainBackendKind::gpu).succeeded,
            "unavailable GPU backend reports an explicit failure");
        expect(worker.status().brain_backend == evobrain::BrainBackendKind::cpu,
            "failed GPU selection does not silently change backend");
    }
    expect(worker.run().succeeded, "backend test starts playback");
    expect(!worker.set_brain_backend(evobrain::BrainBackendKind::cpu).succeeded,
        "backend cannot change while simulation is running");
    expect(worker.pause().succeeded, "backend test pauses playback");
    expect(worker.set_brain_backend(evobrain::BrainBackendKind::cpu).succeeded,
        "CPU backend can be restored while paused");
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
    const std::uint64_t selected_id = simulation.agents().front().id;
    worker.select_agent(selected_id);
    expect(worker.fast_forward().succeeded, "Fast-forward starts");
    expect(wait_for_tick(worker, 5), "Fast-forward publishes progress statistics");
    const auto fast_snapshot = worker.latest_render_snapshot();
    expect(fast_snapshot && !fast_snapshot->contains_world,
        "Fast-forward statistics omit world entity vectors");
    expect(fast_snapshot && !fast_snapshot->selected_agent
            && worker.status().selected_agent_id == selected_id,
        "Fast-forward omits details while retaining the selected stable ID");

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
    const bool selected_survived = paused_snapshot
        && std::ranges::any_of(paused_snapshot->agents,
            [selected_id](const evobrain::viewer::AgentVisual& agent) {
                return agent.id == selected_id;
            });
    expect(selected_survived
            ? paused_snapshot->selected_agent
                && paused_snapshot->selected_agent->id == selected_id
            : !paused_snapshot->selected_agent
                && !worker.status().selected_agent_id,
        "Pause restores a surviving selection or clears a dead selection");
}

// Verifies reset transforms configurable world dimensions consistently.
void test_camera_reset_and_transform()
{
    const evobrain::viewer::CameraViewport viewport {
        .x = 10.0, .y = 20.0, .width = 1600.0, .height = 900.0};
    evobrain::viewer::Camera camera;
    camera.set_world_dimensions(2.5, 2.5, viewport);
    camera.reset(viewport);

    const evobrain::Vec2 center = camera.world_to_screen({.x = 1.25, .y = 1.25}, viewport);
    expect(center == evobrain::Vec2 {.x = 810.0, .y = 470.0},
        "camera reset centers the world in its viewport");
    expect(camera.screen_to_world(center.x, center.y, viewport)
            == evobrain::Vec2 {.x = 1.25, .y = 1.25},
        "camera screen and world transforms invert at the center");
    expect(camera.zoom() == 1.0, "camera reset restores 1x zoom");
}

// Verifies cursor zoom anchoring, hard zoom limits, and outer pan limits.
void test_camera_zoom_and_pan_limits()
{
    const evobrain::viewer::CameraViewport viewport {
        .width = 900.0, .height = 900.0};
    evobrain::viewer::Camera camera;
    camera.set_world_dimensions(2.5, 2.5, viewport);
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
    const auto outer = camera.outer_bounds();
    expect(high_bounds.maximum_x <= outer.maximum_x + 1e-12
            && high_bounds.maximum_y <= outer.maximum_y + 1e-12,
        "camera pan stops at the outer maximum boundary");
    camera.pan_pixels(1e9, 1e9, viewport);
    const auto low_bounds = camera.visible_bounds(viewport);
    expect(low_bounds.minimum_x >= outer.minimum_x - 1e-12
            && low_bounds.minimum_y >= outer.minimum_y - 1e-12,
        "camera pan stops at the outer minimum boundary");

    camera.zoom_at(450.0, 450.0, 1e-9, viewport);
    expect(camera.zoom() == evobrain::viewer::Camera::minimum_zoom,
        "camera clamps minimum zoom");
    expect(camera.center() == evobrain::Vec2 {.x = 1.25, .y = 1.25},
        "fully zoomed-out camera centers the complete outer world");
}

// Verifies resizing preserves zoom and center unless a boundary requires clamp.
void test_camera_resize_preservation()
{
    const evobrain::viewer::CameraViewport square {
        .width = 800.0, .height = 800.0};
    evobrain::viewer::Camera camera;
    camera.set_world_dimensions(2.5, 2.5, square);
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
    test_selected_agent_details();
    test_selected_agent_death();
    test_agent_hit_testing();
    test_agent_information_shortcut();
    test_failed_load_preserves_state(directory);
    test_save_reload_and_unsaved_state(directory);
    test_speed_validation();
    test_brain_backend_selection();
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
