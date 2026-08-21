#pragma once

#include "evobrain/simulation.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace evobrain::viewer {

// Identifies the mutually exclusive ways in which the worker advances time.
enum class PlaybackState {
    paused,
    running,
    fast_forward,
};

// Reports a recoverable worker operation without throwing through the UI layer.
struct OperationResult {
    bool succeeded = false;
    std::string summary;
    std::string detail;

    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded;
    }
};

// Contains the lightweight fields required to draw and hit-test one agent.
struct AgentVisual {
    std::uint64_t id = 0;
    float x = 0.0F;
    float y = 0.0F;
    float direction = 0.0F;
    float energy = 0.0F;

    bool operator==(const AgentVisual&) const = default;
};

// Contains only the position required to render one food item.
struct FoodVisual {
    float x = 0.0F;
    float y = 0.0F;

    bool operator==(const FoodVisual&) const = default;
};

// Contains the full viewer-only inspection state for one selected agent.
struct SelectedAgentDetails {
    std::uint64_t id = 0;
    Vec2 position;
    double direction = 0.0;
    double energy = 0.0;
    std::uint64_t age = 0;
    std::uint64_t generation = 0;
    BrainParameters brain {};

    bool operator==(const SelectedAgentDetails&) const = default;
};

// Contains one immutable, internally consistent view of a completed tick.
struct RenderSnapshot {
    SimulationStats stats;
    double reproduction_threshold = 1.0;
    std::vector<AgentVisual> agents;
    std::vector<FoodVisual> food;
    std::optional<SelectedAgentDetails> selected_agent;
    bool contains_world = true;

    bool operator==(const RenderSnapshot&) const = default;
};

// Summarizes worker state without exposing its owned Simulation instance.
struct WorkerStatus {
    bool has_simulation = false;
    bool has_unsaved_changes = false;
    PlaybackState playback = PlaybackState::paused;
    int target_ticks_per_second = 60;
    double actual_ticks_per_second = 0.0;
    std::optional<std::uint64_t> selected_agent_id;

    bool operator==(const WorkerStatus&) const = default;
};

// Describes an asynchronous simulation failure for the in-window error modal.
struct WorkerFailure {
    std::string summary;
    std::string detail;
};

// Owns the simulation on a dedicated thread and serializes all state changes.
class SimulationWorker {
public:
    // Starts an idle worker with no simulation loaded.
    SimulationWorker();

    // Stops at a completed tick and joins the worker thread.
    ~SimulationWorker();

    SimulationWorker(const SimulationWorker&) = delete;
    SimulationWorker& operator=(const SimulationWorker&) = delete;
    SimulationWorker(SimulationWorker&&) = delete;
    SimulationWorker& operator=(SimulationWorker&&) = delete;

    // Loads and validates a checkpoint while preserving existing state on failure.
    [[nodiscard]] OperationResult load_checkpoint_file(
        const std::filesystem::path& path);

    // Installs a detached state for tests and future in-memory integrations.
    [[nodiscard]] OperationResult load_snapshot(SimulationSnapshot snapshot);

    // Atomically saves the paused state, optionally replacing an existing file.
    [[nodiscard]] OperationResult save_checkpoint_file(
        const std::filesystem::path& path,
        bool overwrite);

    // Advances exactly one tick when a simulation is loaded and paused.
    [[nodiscard]] OperationResult step();

    // Starts target-rate playback when a simulation is loaded.
    [[nodiscard]] OperationResult run();

    // Advances without pacing or full world snapshots until paused.
    [[nodiscard]] OperationResult fast_forward();

    // Stops playback after the current tick has completed.
    [[nodiscard]] OperationResult pause();

    // Applies a target rate in the inclusive supported range of 1 to 1000.
    [[nodiscard]] OperationResult set_target_ticks_per_second(int value);

    // Selects one current stable agent ID, or clears viewer selection with null.
    void select_agent(std::optional<std::uint64_t> agent_id);

    // Returns the latest immutable render snapshot, or null before a load.
    [[nodiscard]] std::shared_ptr<const RenderSnapshot> latest_render_snapshot() const;

    // Requests one latest-wins full snapshot for the next rendered UI frame.
    void request_render_snapshot() noexcept;

    // Returns an exact detached state for deterministic tests and explicit tools.
    [[nodiscard]] std::optional<SimulationSnapshot> copy_simulation_snapshot();

    // Returns a synchronized summary of worker ownership and playback state.
    [[nodiscard]] WorkerStatus status() const;

    // Takes the latest asynchronous failure so it is presented only once.
    [[nodiscard]] std::optional<WorkerFailure> take_failure();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace evobrain::viewer
