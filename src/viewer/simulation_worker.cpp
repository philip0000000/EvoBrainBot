#include "evobrain/viewer/simulation_worker.hpp"

#include "evobrain/checkpoint.hpp"

#include <Windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace evobrain::viewer {
namespace {

constexpr int minimum_ticks_per_second = 1;
constexpr int maximum_ticks_per_second = 1000;

// Builds a small render-only copy from one completed simulation tick.
std::shared_ptr<const RenderSnapshot> make_render_snapshot(
    const Simulation& simulation,
    const std::optional<std::uint64_t> selected_agent_id,
    const bool include_world)
{
    auto snapshot = std::make_shared<RenderSnapshot>();
    snapshot->stats = simulation.stats();
    snapshot->diagnostics = simulation.diagnostics();
    snapshot->world_width = simulation.config().world_width;
    snapshot->world_height = simulation.config().world_height;
    snapshot->reproduction_threshold = simulation.config().reproduction_threshold;
    snapshot->agent_radius = simulation.config().agent_radius;
    snapshot->food_radius = simulation.config().food_radius;
    snapshot->eye_range = simulation.config().eye_range;
    snapshot->contains_world = include_world;
    if (!include_world) {
        return snapshot;
    }
    snapshot->agents.reserve(simulation.agents().size());
    for (const Agent& agent : simulation.agents()) {
        snapshot->agents.push_back(AgentVisual {
            .id = agent.id,
            .x = static_cast<float>(agent.position.x),
            .y = static_cast<float>(agent.position.y),
            .direction = static_cast<float>(agent.direction),
            .energy = static_cast<float>(agent.energy),
            .diet = agent.diet,
            .red = static_cast<float>(agent.color.red),
            .green = static_cast<float>(agent.color.green),
            .blue = static_cast<float>(agent.color.blue),
        });
        if (selected_agent_id == agent.id) {
            snapshot->selected_agent = SelectedAgentDetails {
                .id = agent.id,
                .position = agent.position,
                .direction = agent.direction,
                .energy = agent.energy,
                .age = agent.age,
                .generation = agent.generation,
                .diet = agent.diet,
                .color = agent.color,
                .mutation_rate = agent.mutation_rate,
                .mutation_strength = agent.mutation_strength,
                .prior_bite_damage = agent.prior_bite_damage,
                .brain = agent.brain,
                .brain_structure = agent.brain_structure,
                .brain_state = agent.brain_state,
            };
        }
    }
    snapshot->food.reserve(simulation.food().size());
    for (const Food& item : simulation.food()) {
        snapshot->food.push_back(FoodVisual {
            .id = item.id,
            .x = static_cast<float>(item.position.x),
            .y = static_cast<float>(item.position.y),
            .energy = static_cast<float>(item.energy),
        });
    }
    // Ascending stable IDs make the highest ID draw last within each entity layer.
    std::ranges::sort(snapshot->agents, {}, &AgentVisual::id);
    std::ranges::sort(snapshot->food, {}, &FoodVisual::id);
    return snapshot;
}

// Returns a non-existing sibling path suitable for an atomic save temporary.
std::filesystem::path temporary_checkpoint_path(
    const std::filesystem::path& destination)
{
    for (std::uint32_t suffix = 0; suffix < 1000; ++suffix) {
        std::filesystem::path candidate = destination;
        candidate += L".tmp-" + std::to_wstring(GetCurrentProcessId()) + L"-"
            + std::to_wstring(suffix);
        std::error_code error;
        if (!std::filesystem::exists(candidate, error) && !error) {
            return candidate;
        }
    }
    throw std::runtime_error("could not reserve a temporary checkpoint filename");
}

// Writes a complete checkpoint and atomically publishes it at the destination.
void save_checkpoint_atomically(
    const Simulation& simulation,
    const std::filesystem::path& destination,
    const bool overwrite)
{
    if (destination.empty()) {
        throw std::invalid_argument("checkpoint path is empty");
    }

    std::error_code exists_error;
    const bool destination_exists = std::filesystem::exists(destination, exists_error);
    if (exists_error) {
        throw std::runtime_error("could not inspect the checkpoint destination");
    }
    if (destination_exists && !overwrite) {
        throw std::runtime_error("a file already exists at the checkpoint destination");
    }

    const std::filesystem::path temporary = temporary_checkpoint_path(destination);
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("could not create the temporary checkpoint file");
        }
        save_checkpoint(simulation, output);
        output.close();
        if (!output) {
            throw std::runtime_error("could not finish writing the checkpoint");
        }

        DWORD move_flags = MOVEFILE_WRITE_THROUGH;
        if (overwrite) {
            move_flags |= MOVEFILE_REPLACE_EXISTING;
        }
        if (!MoveFileExW(temporary.c_str(), destination.c_str(), move_flags)) {
            throw std::system_error(
                static_cast<int>(GetLastError()),
                std::system_category(),
                "could not publish the checkpoint");
        }
    } catch (...) {
        // A failed save must never leave its implementation detail beside the
        // user's checkpoint or damage the prior destination.
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

// Converts a caught operation exception into UI-friendly error information.
OperationResult failed_operation(const char* summary, const std::exception& error)
{
    return OperationResult {
        .succeeded = false,
        .summary = summary,
        .detail = error.what(),
    };
}

} // namespace

class SimulationWorker::Impl {
public:
    // Starts the thread before any commands can be submitted.
    Impl()
        : thread_(&Impl::thread_main, this)
    {
    }

    // Requests termination through the same ordered command path, then joins.
    ~Impl()
    {
        dispatch([this] {
            stopping_ = true;
            playback_ = PlaybackState::paused;
        });
        thread_.join();
    }

    // Queues one operation and waits until the worker has executed it.
    void dispatch(std::function<void()> operation)
    {
        auto completion = std::make_shared<std::promise<void>>();
        std::future<void> completed = completion->get_future();
        {
            std::lock_guard lock(queue_mutex_);
            commands_.push_back(Command {
                .operation = std::move(operation),
                .completion = completion,
            });
        }
        queue_changed_.notify_one();
        completed.get();
    }

    // Publishes a coherent snapshot and status after a worker-side change.
    void publish(const bool include_world = true)
    {
        std::lock_guard lock(published_mutex_);
        published_snapshot_ = simulation_.has_value()
            ? make_render_snapshot(*simulation_, selected_agent_id_, include_world)
            : std::shared_ptr<const RenderSnapshot> {};
        if (include_world && selected_agent_id_ && published_snapshot_
            && !published_snapshot_->selected_agent) {
            // Stable IDs are never reused, so a missing selected agent can be
            // cleared without risking selection of a later replacement.
            selected_agent_id_.reset();
        }
        published_status_ = WorkerStatus {
            .has_simulation = simulation_.has_value(),
            .has_unsaved_changes = unsaved_changes_,
            .playback = playback_,
            .target_ticks_per_second = target_ticks_per_second_,
            .actual_ticks_per_second = actual_ticks_per_second_,
            .brain_backend = simulation_.has_value()
                ? simulation_->brain_backend()
                : BrainBackendKind::cpu,
            .gpu_backend_available = brain_backend_available(BrainBackendKind::gpu),
            .selected_agent_id = selected_agent_id_,
        };
    }

    // Starts a fresh measurement window when playback mode changes.
    void reset_rate_measurement()
    {
        rate_window_started_ = std::chrono::steady_clock::now();
        rate_window_ticks_ = 0;
        actual_ticks_per_second_ = 0.0;
    }

    // Updates actual TPS at the same four-Hz cadence used by Fast-forward stats.
    bool record_completed_tick()
    {
        ++rate_window_ticks_;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = now - rate_window_started_;
        if (elapsed < std::chrono::milliseconds(250)) {
            return false;
        }
        actual_ticks_per_second_ = static_cast<double>(rate_window_ticks_)
            / std::chrono::duration<double>(elapsed).count();
        rate_window_started_ = now;
        rate_window_ticks_ = 0;
        return true;
    }

    // Runs ordered commands and paced simulation ticks until shutdown.
    void thread_main()
    {
        for (;;) {
            Command command;
            {
                std::unique_lock lock(queue_mutex_);
                if (commands_.empty()
                    && (playback_ == PlaybackState::paused || !simulation_)) {
                    queue_changed_.wait(lock, [this] {
                        return !commands_.empty();
                    });
                }
                if (!commands_.empty()) {
                    command = std::move(commands_.front());
                    commands_.pop_front();
                }
            }

            if (command.operation) {
                try {
                    command.operation();
                    command.completion->set_value();
                } catch (...) {
                    command.completion->set_exception(std::current_exception());
                }
                if (stopping_) {
                    return;
                }
                continue;
            }

            const auto tick_started = std::chrono::steady_clock::now();
            try {
                simulation_->tick();
                unsaved_changes_ = true;
                const bool rate_updated = record_completed_tick();
                if (playback_ == PlaybackState::running) {
                    if (render_snapshot_requested_.exchange(false)) {
                        publish();
                    }
                } else if (rate_updated) {
                    // Fast-forward intentionally omits entity vectors so copying
                    // and UI delivery do not throttle the simulation loop.
                    publish(false);
                }
            } catch (const std::exception& error) {
                playback_ = PlaybackState::paused;
                actual_ticks_per_second_ = 0.0;
                {
                    std::lock_guard lock(published_mutex_);
                    pending_failure_ = WorkerFailure {
                        .summary = "The simulation stopped because a tick failed.",
                        .detail = error.what(),
                    };
                }
                publish();
            } catch (...) {
                playback_ = PlaybackState::paused;
                actual_ticks_per_second_ = 0.0;
                {
                    std::lock_guard lock(published_mutex_);
                    pending_failure_ = WorkerFailure {
                        .summary = "The simulation stopped because a tick failed.",
                        .detail = "Unknown worker error",
                    };
                }
                publish();
            }

            if (playback_ == PlaybackState::fast_forward) {
                continue;
            }
            const auto interval = std::chrono::nanoseconds(
                std::chrono::seconds(1)) / target_ticks_per_second_;
            std::unique_lock lock(queue_mutex_);
            queue_changed_.wait_until(lock, tick_started + interval, [this] {
                return !commands_.empty();
            });
        }
    }

    struct Command {
        std::function<void()> operation;
        std::shared_ptr<std::promise<void>> completion;
    };

    std::mutex queue_mutex_;
    std::condition_variable queue_changed_;
    std::deque<Command> commands_;
    std::thread thread_;

    // These values are accessed only by the worker thread.
    std::optional<Simulation> simulation_;
    std::optional<std::uint64_t> selected_agent_id_;
    PlaybackState playback_ = PlaybackState::paused;
    bool unsaved_changes_ = false;
    bool stopping_ = false;
    int target_ticks_per_second_ = 60;
    double actual_ticks_per_second_ = 0.0;
    std::chrono::steady_clock::time_point rate_window_started_ =
        std::chrono::steady_clock::now();
    std::uint64_t rate_window_ticks_ = 0;
    std::atomic_bool render_snapshot_requested_ = false;

    mutable std::mutex published_mutex_;
    std::shared_ptr<const RenderSnapshot> published_snapshot_;
    WorkerStatus published_status_;
    std::optional<WorkerFailure> pending_failure_;
};

SimulationWorker::SimulationWorker()
    : impl_(std::make_unique<Impl>())
{
}

SimulationWorker::~SimulationWorker() = default;

OperationResult SimulationWorker::load_checkpoint_file(
    const std::filesystem::path& path)
{
    OperationResult result;
    impl_->dispatch([&] {
        try {
            std::ifstream input(path, std::ios::binary);
            if (!input) {
                throw std::runtime_error("could not open the checkpoint file");
            }
            Simulation replacement = load_checkpoint(input);
            // Do not mutate the live state until parsing and validation have
            // both completed successfully.
            impl_->simulation_ = std::move(replacement);
            impl_->selected_agent_id_.reset();
            impl_->playback_ = PlaybackState::paused;
            impl_->actual_ticks_per_second_ = 0.0;
            impl_->unsaved_changes_ = false;
            impl_->publish();
            result = OperationResult {.succeeded = true};
        } catch (const std::exception& error) {
            result = failed_operation("The checkpoint could not be opened.", error);
        }
    });
    return result;
}

OperationResult SimulationWorker::load_snapshot(SimulationSnapshot snapshot)
{
    OperationResult result;
    impl_->dispatch([&] {
        try {
            Simulation replacement = Simulation::from_snapshot(std::move(snapshot));
            impl_->simulation_ = std::move(replacement);
            impl_->selected_agent_id_.reset();
            impl_->playback_ = PlaybackState::paused;
            impl_->actual_ticks_per_second_ = 0.0;
            impl_->unsaved_changes_ = false;
            impl_->publish();
            result = OperationResult {.succeeded = true};
        } catch (const std::exception& error) {
            result = failed_operation("The simulation state could not be loaded.", error);
        }
    });
    return result;
}

OperationResult SimulationWorker::save_checkpoint_file(
    const std::filesystem::path& path,
    const bool overwrite)
{
    OperationResult result;
    impl_->dispatch([&] {
        if (!impl_->simulation_) {
            result = OperationResult {
                .summary = "There is no simulation to save.",
            };
            return;
        }
        if (impl_->playback_ != PlaybackState::paused) {
            result = OperationResult {
                .summary = "Pause the simulation before saving.",
            };
            return;
        }
        try {
            save_checkpoint_atomically(*impl_->simulation_, path, overwrite);
            impl_->unsaved_changes_ = false;
            impl_->publish();
            result = OperationResult {.succeeded = true};
        } catch (const std::exception& error) {
            result = failed_operation("The checkpoint could not be saved.", error);
        }
    });
    return result;
}

OperationResult SimulationWorker::step()
{
    OperationResult result;
    impl_->dispatch([&] {
        if (!impl_->simulation_) {
            result = OperationResult {.summary = "No checkpoint is loaded."};
            return;
        }
        if (impl_->playback_ != PlaybackState::paused) {
            result = OperationResult {
                .summary = "Pause the simulation before stepping.",
            };
            return;
        }
        try {
            impl_->simulation_->tick();
            impl_->unsaved_changes_ = true;
            impl_->publish();
            result = OperationResult {.succeeded = true};
        } catch (const std::exception& error) {
            result = failed_operation("The simulation tick failed.", error);
        }
    });
    return result;
}

OperationResult SimulationWorker::run()
{
    OperationResult result;
    impl_->dispatch([&] {
        if (!impl_->simulation_) {
            result = OperationResult {.summary = "No checkpoint is loaded."};
            return;
        }
        impl_->playback_ = PlaybackState::running;
        impl_->reset_rate_measurement();
        impl_->publish();
        result = OperationResult {.succeeded = true};
    });
    impl_->queue_changed_.notify_one();
    return result;
}

OperationResult SimulationWorker::fast_forward()
{
    OperationResult result;
    impl_->dispatch([&] {
        if (!impl_->simulation_) {
            result = OperationResult {.summary = "No checkpoint is loaded."};
            return;
        }
        impl_->playback_ = PlaybackState::fast_forward;
        impl_->reset_rate_measurement();
        impl_->publish(false);
        result = OperationResult {.succeeded = true};
    });
    impl_->queue_changed_.notify_one();
    return result;
}

OperationResult SimulationWorker::pause()
{
    OperationResult result;
    impl_->dispatch([&] {
        impl_->playback_ = PlaybackState::paused;
        impl_->actual_ticks_per_second_ = 0.0;
        impl_->publish();
        result = OperationResult {.succeeded = true};
    });
    return result;
}

OperationResult SimulationWorker::set_target_ticks_per_second(const int value)
{
    OperationResult result;
    impl_->dispatch([&] {
        if (value < minimum_ticks_per_second || value > maximum_ticks_per_second) {
            result = OperationResult {
                .summary = "Target TPS must be between 1 and 1000.",
            };
            return;
        }
        impl_->target_ticks_per_second_ = value;
        impl_->publish();
        result = OperationResult {.succeeded = true};
    });
    return result;
}

OperationResult SimulationWorker::set_brain_backend(const BrainBackendKind backend)
{
    OperationResult result;
    impl_->dispatch([&] {
        if (!impl_->simulation_) {
            result = OperationResult {.summary = "No checkpoint is loaded."};
            return;
        }
        if (impl_->playback_ != PlaybackState::paused) {
            result = OperationResult {
                .summary = "Pause the simulation before changing brain backend.",
            };
            return;
        }
        try {
            impl_->simulation_->set_brain_backend(backend);
            impl_->publish();
            result = OperationResult {.succeeded = true};
        } catch (const std::exception& error) {
            result = failed_operation("The brain backend could not be changed.", error);
        }
    });
    return result;
}

void SimulationWorker::select_agent(const std::optional<std::uint64_t> agent_id)
{
    impl_->dispatch([&] {
        impl_->selected_agent_id_ = agent_id;
        // Selection is viewer metadata. Publishing validates the requested ID
        // against the current completed state without mutating the simulation.
        impl_->publish(impl_->playback_ != PlaybackState::fast_forward);
    });
}

std::shared_ptr<const RenderSnapshot> SimulationWorker::latest_render_snapshot() const
{
    std::lock_guard lock(impl_->published_mutex_);
    return impl_->published_snapshot_;
}

void SimulationWorker::request_render_snapshot() noexcept
{
    impl_->render_snapshot_requested_.store(true);
}

std::optional<SimulationSnapshot> SimulationWorker::copy_simulation_snapshot()
{
    std::optional<SimulationSnapshot> result;
    impl_->dispatch([&] {
        if (impl_->simulation_) {
            result = impl_->simulation_->snapshot();
        }
    });
    return result;
}

WorkerStatus SimulationWorker::status() const
{
    std::lock_guard lock(impl_->published_mutex_);
    return impl_->published_status_;
}

std::optional<WorkerFailure> SimulationWorker::take_failure()
{
    std::lock_guard lock(impl_->published_mutex_);
    std::optional<WorkerFailure> result = std::move(impl_->pending_failure_);
    impl_->pending_failure_.reset();
    return result;
}

} // namespace evobrain::viewer
