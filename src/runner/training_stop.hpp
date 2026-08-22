#pragma once

#include <memory>

namespace evobrain::runner {

// Provides runner-only portable polling for keyboard and process stop requests.
class TrainingStopController {
public:
    // Installs signal handling and configures an attached terminal for Q input.
    TrainingStopController();

    // Restores terminal state and the process's previous signal handlers.
    ~TrainingStopController();

    TrainingStopController(const TrainingStopController&) = delete;
    TrainingStopController& operator=(const TrainingStopController&) = delete;
    TrainingStopController(TrainingStopController&&) = delete;
    TrainingStopController& operator=(TrainingStopController&&) = delete;

    // Returns true after Q, q, SIGINT, or SIGTERM requests graceful stopping.
    [[nodiscard]] bool stop_requested() noexcept;

    // Returns whether no-Enter keyboard input and in-place status are available.
    [[nodiscard]] bool has_interactive_terminal() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace evobrain::runner
