#include "training_stop.hpp"

#include <csignal>
#include <cstdio>

#ifdef _WIN32
#include <Windows.h>
#include <conio.h>
#include <io.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

namespace evobrain::runner {
namespace {

volatile std::sig_atomic_t stop_signal_received = 0;

// Signal handlers only set this flag; checkpoint I/O remains in normal control flow.
extern "C" void request_stop_from_signal(int) noexcept
{
    stop_signal_received = 1;
}

} // namespace

class TrainingStopController::Impl {
public:
    Impl()
        : previous_interrupt_(std::signal(SIGINT, request_stop_from_signal))
        , previous_terminate_(std::signal(SIGTERM, request_stop_from_signal))
    {
        stop_signal_received = 0;
#ifdef _WIN32
        interactive_ = _isatty(_fileno(stdin)) != 0;
        if (interactive_) {
            const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
            DWORD mode = 0;
            if (output != INVALID_HANDLE_VALUE && GetConsoleMode(output, &mode)) {
                static_cast<void>(SetConsoleMode(
                    output, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING));
            }
        }
#else
        interactive_ = isatty(STDIN_FILENO) != 0;
        if (interactive_ && tcgetattr(STDIN_FILENO, &previous_terminal_) == 0) {
            termios terminal = previous_terminal_;
            terminal.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
            terminal.c_cc[VMIN] = 0;
            terminal.c_cc[VTIME] = 0;
            terminal_changed_ = tcsetattr(STDIN_FILENO, TCSANOW, &terminal) == 0;
        } else {
            interactive_ = false;
        }
#endif
    }

    ~Impl()
    {
#ifndef _WIN32
        if (terminal_changed_) {
            static_cast<void>(tcsetattr(
                STDIN_FILENO, TCSANOW, &previous_terminal_));
        }
#endif
        static_cast<void>(std::signal(SIGINT, previous_interrupt_));
        static_cast<void>(std::signal(SIGTERM, previous_terminate_));
    }

    [[nodiscard]] bool stop_requested() noexcept
    {
        if (stop_signal_received != 0) {
            return true;
        }
        if (!interactive_) {
            return false;
        }
#ifdef _WIN32
        while (_kbhit() != 0) {
            const int key = _getch();
            if (key == 'q' || key == 'Q') {
                return true;
            }
        }
#else
        char key = 0;
        while (read(STDIN_FILENO, &key, 1) == 1) {
            if (key == 'q' || key == 'Q') {
                return true;
            }
        }
#endif
        return false;
    }

    [[nodiscard]] bool interactive() const noexcept { return interactive_; }

private:
    using SignalHandler = void (*)(int);
    SignalHandler previous_interrupt_ = SIG_DFL;
    SignalHandler previous_terminate_ = SIG_DFL;
    bool interactive_ = false;
#ifndef _WIN32
    termios previous_terminal_ {};
    bool terminal_changed_ = false;
#endif
};

TrainingStopController::TrainingStopController()
    : impl_(std::make_unique<Impl>())
{
}

TrainingStopController::~TrainingStopController() = default;

bool TrainingStopController::stop_requested() noexcept
{
    return impl_->stop_requested();
}

bool TrainingStopController::has_interactive_terminal() const noexcept
{
    return impl_->interactive();
}

} // namespace evobrain::runner
