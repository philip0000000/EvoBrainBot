#include "command_line.hpp"
#include "training_stop.hpp"

#include "evobrain/checkpoint.hpp"
#include "evobrain/simulation.hpp"

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <ostream>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace evobrain::runner {
namespace {

constexpr std::string_view top_level_help =
    "Usage:\n"
    "  EvoBrainBot [--help]\n"
    "  EvoBrainBot run [<checkpoint>] [--seed <seed>] [--ticks <ticks>]"
    " [--brain-backend cpu|gpu]\n"
    "  EvoBrainBot resume <checkpoint.evo> [--ticks <ticks>]"
    " [--brain-backend cpu|gpu]\n"
    "\n"
    "Run starts a new simulation. Seed defaults to a random integer from 1 to 999.\n"
    "Without ticks, training continues until Q, q, SIGINT, or SIGTERM requests a stop.\n"
    "The checkpoint defaults to autosave.evo; .evo is appended when its filename\n"
    "contains no dot. The checkpoint is saved when training stops.\n"
    "\n"
    "Resume continues the exact checkpoint path supplied. Without ticks, it also\n"
    "continues until Q, q, SIGINT, or SIGTERM requests a stop, then atomically replaces\n"
    "the input checkpoint. Seed and ticks are decimal unsigned 64-bit integers.\n";

constexpr char default_checkpoint_output[] = "autosave.evo";

struct RunOptions {
    std::optional<std::uint64_t> seed;
    std::optional<std::uint64_t> ticks;
    std::optional<std::filesystem::path> checkpoint;
    std::optional<BrainBackendKind> brain_backend;
};

struct ResumeOptions {
    std::filesystem::path checkpoint;
    std::optional<std::uint64_t> ticks;
    std::optional<BrainBackendKind> brain_backend;
};

// Reports a command-line usage error and returns the documented exit code.
int report_usage_error(std::ostream& error, const std::string_view message)
{
    error << "Error: " << message << '\n';
    return usage_error_exit_code;
}

// Reports an execution or file error separately from invalid command syntax.
int report_runtime_error(std::ostream& error, const std::string_view message)
{
    error << "Error: " << message << '\n';
    return runtime_error_exit_code;
}

// Parses a complete decimal unsigned 64-bit integer without accepting signs.
std::optional<std::uint64_t> parse_unsigned_decimal(const std::string_view text)
{
    if (text.empty()) {
        return std::nullopt;
    }

    std::uint64_t value = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result = std::from_chars(begin, end, value, 10);

    // Requiring consumption of the entire token rejects signs, whitespace,
    // suffixes, and other input that is not a plain decimal integer.
    if (result.ec != std::errc {} || result.ptr != end) {
        return std::nullopt;
    }
    return value;
}

// Parses the stable backend names shared with the GUI and diagnostics.
std::optional<BrainBackendKind> parse_brain_backend(const std::string_view text)
{
    if (text == "cpu") return BrainBackendKind::cpu;
    if (text == "gpu") return BrainBackendKind::gpu;
    return std::nullopt;
}

// Generates the small default seed pool exposed by the headless CLI.
std::uint64_t generate_default_seed()
{
    std::random_device entropy_source;
    std::uniform_int_distribution<std::uint64_t> distribution(1, 999);
    return distribution(entropy_source);
}

// Applies run-only checkpoint defaults without changing explicitly suffixed names.
std::filesystem::path normalize_run_checkpoint(
    std::optional<std::filesystem::path> checkpoint)
{
    std::filesystem::path result = checkpoint.value_or(default_checkpoint_output);
    // Keep the filename storage alive while checking only its final component.
    const auto filename = result.filename().native();
    if (!filename.empty()
        && filename.find(std::filesystem::path::value_type {'.'})
            == std::filesystem::path::string_type::npos) {
        result += ".evo";
    }
    return result;
}

// Returns the following option token or reports the shared missing-value error.
std::optional<std::string_view> option_value(
    const std::span<const std::string_view> arguments,
    const std::size_t index,
    std::ostream& error)
{
    if (index + 1 >= arguments.size()
        || arguments[index + 1].starts_with("--")) {
        report_usage_error(error, "missing option value");
        return std::nullopt;
    }
    return arguments[index + 1];
}

// Draws the cumulative training state shown during and after headless execution.
void print_training_status(
    const Simulation& simulation,
    const std::filesystem::path& checkpoint,
    std::ostream& output,
    const bool clear_terminal,
    const bool show_stop_instruction)
{
    if (clear_terminal) {
        output << "\x1b[2J\x1b[H";
    }
    const SimulationStats stats = simulation.stats();
    output << "File: " << checkpoint.string() << '\n'
           << "Brain backend: " << brain_backend_name(simulation.brain_backend()) << '\n'
           << "Seed: " << stats.seed << '\n'
           << "Tick: " << stats.completed_ticks << '\n'
           << "Population: " << stats.population << '\n'
           << "Herbivores: " << stats.herbivores << '\n'
           << "Carnivores: " << stats.carnivores << '\n'
           << "Food: " << stats.food << '\n'
           << "Births: " << stats.births << '\n'
           << "Introduced agents: " << stats.introduced_agents << '\n'
           << "Deaths: " << stats.deaths << '\n'
           << "Agents eaten: " << stats.agents_eaten << '\n';
    if (show_stop_instruction) {
        output << "\nPress Q to stop and save.\n";
    }
    output.flush();
}

// Runs training until its optional tick limit or a graceful stop request.
bool train_simulation(
    Simulation& simulation,
    const std::filesystem::path& checkpoint,
    const std::optional<std::uint64_t> ticks,
    std::ostream& output)
{
    TrainingStopController stop;
    const bool interactive = stop.has_interactive_terminal();
    if (interactive) {
        print_training_status(simulation, checkpoint, output, true, true);
    }

    std::uint64_t executed_ticks = 0;
    auto next_status = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while ((!ticks || executed_ticks < *ticks) && !stop.stop_requested()) {
        simulation.tick();
        ++executed_ticks;
        const auto now = std::chrono::steady_clock::now();
        if (interactive && now >= next_status) {
            print_training_status(simulation, checkpoint, output, true, true);
            next_status = now + std::chrono::milliseconds(250);
        }
    }
    return interactive;
}

// Saves a completed simulation and verifies buffered data reached the stream.
void save_checkpoint_file(
    const Simulation& simulation,
    const std::filesystem::path& path)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("unable to open checkpoint output");
    }
    save_checkpoint(simulation, output);
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to finish checkpoint output");
    }
}

// Loads one complete checkpoint before its input file is released.
Simulation load_checkpoint_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open checkpoint input");
    }
    return load_checkpoint(input);
}

// Returns a non-existing sibling used to publish an in-place save atomically.
std::filesystem::path temporary_checkpoint_path(
    const std::filesystem::path& destination)
{
    for (std::uint32_t suffix = 0; suffix < 1000; ++suffix) {
        std::filesystem::path candidate = destination;
        candidate += ".tmp-" + std::to_string(suffix);
        std::error_code error;
        if (!std::filesystem::exists(candidate, error) && !error) {
            return candidate;
        }
    }
    throw std::runtime_error("could not reserve a temporary checkpoint filename");
}

// Writes beside the input checkpoint before replacing it in one filesystem operation.
void replace_checkpoint_file(
    const Simulation& simulation,
    const std::filesystem::path& destination)
{
    const std::filesystem::path temporary = temporary_checkpoint_path(destination);
    try {
        save_checkpoint_file(simulation, temporary);
#ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), destination.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            throw std::system_error(static_cast<int>(GetLastError()),
                std::system_category(), "unable to replace checkpoint");
        }
#else
        std::error_code replace_error;
        std::filesystem::rename(temporary, destination, replace_error);
        if (replace_error) {
            throw std::system_error(replace_error, "unable to replace checkpoint");
        }
#endif
    } catch (...) {
        // A failed replacement must leave the user's original checkpoint intact.
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

// Parses new-run options and executes a simulation when syntax is valid.
int run_simulation_command(
    const std::span<const std::string_view> arguments,
    std::ostream& output,
    std::ostream& error)
{
    RunOptions options;
    for (std::size_t index = 0; index < arguments.size();) {
        const std::string_view argument = arguments[index];
        if (!argument.starts_with("--")) {
            if (options.checkpoint.has_value()) {
                return report_usage_error(error, "unexpected positional argument");
            }
            options.checkpoint = std::filesystem::u8path(argument);
            ++index;
            continue;
        }
        if (argument != "--seed" && argument != "--ticks"
            && argument != "--brain-backend") {
            return report_usage_error(error, "unknown option");
        }
        const auto value = option_value(arguments, index, error);
        if (!value.has_value()) {
            return usage_error_exit_code;
        }

        if (argument == "--brain-backend") {
            if (options.brain_backend.has_value()) {
                return report_usage_error(error, "duplicate option");
            }
            options.brain_backend = parse_brain_backend(*value);
            if (!options.brain_backend.has_value()) {
                return report_usage_error(error, "brain backend must be cpu or gpu");
            }
        } else {
            std::optional<std::uint64_t>& destination = argument == "--seed"
                ? options.seed
                : options.ticks;
            if (destination.has_value()) {
                return report_usage_error(error, "duplicate option");
            }
            destination = parse_unsigned_decimal(*value);
            if (!destination.has_value()) {
                return report_usage_error(error, "invalid unsigned integer value");
            }
        }
        index += 2;
    }

    const std::filesystem::path checkpoint =
        normalize_run_checkpoint(std::move(options.checkpoint));
    const std::uint64_t seed = options.seed.has_value()
        ? *options.seed
        : generate_default_seed();
    Simulation simulation(SimulationConfig {.seed = seed},
        SimulationExecutionConfig {
            .brain_backend = options.brain_backend.value_or(BrainBackendKind::cpu)});
    const bool interactive =
        train_simulation(simulation, checkpoint, options.ticks, output);
    save_checkpoint_file(simulation, checkpoint);
    print_training_status(simulation, checkpoint, output, interactive, false);
    return success_exit_code;
}

// Parses resume options and continues a validated checkpoint when syntax is valid.
int resume_simulation_command(
    const std::span<const std::string_view> arguments,
    std::ostream& output,
    std::ostream& error)
{
    if (arguments.size() == 1 && arguments.front() == "--help") {
        return report_usage_error(error, "unknown option");
    }

    if (arguments.empty() || arguments.front().starts_with("--")) {
        return report_usage_error(error, "missing checkpoint path");
    }

    ResumeOptions options {.checkpoint = std::filesystem::u8path(arguments.front())};
    for (std::size_t index = 1; index < arguments.size(); index += 2) {
        const std::string_view option = arguments[index];
        if (!option.starts_with("--")) {
            return report_usage_error(error, "unexpected positional argument");
        }
        if (option != "--ticks" && option != "--brain-backend") {
            return report_usage_error(error, "unknown option");
        }
        const auto value = option_value(arguments, index, error);
        if (!value.has_value()) {
            return usage_error_exit_code;
        }

        if (option == "--brain-backend") {
            if (options.brain_backend.has_value()) {
                return report_usage_error(error, "duplicate option");
            }
            options.brain_backend = parse_brain_backend(*value);
            if (!options.brain_backend.has_value()) {
                return report_usage_error(error, "brain backend must be cpu or gpu");
            }
        } else {
            if (options.ticks.has_value()) {
                return report_usage_error(error, "duplicate option");
            }
            options.ticks = parse_unsigned_decimal(*value);
            if (!options.ticks.has_value()) {
                return report_usage_error(error, "invalid unsigned integer value");
            }
        }
    }

    Simulation simulation = load_checkpoint_file(options.checkpoint);
    simulation.set_brain_backend(
        options.brain_backend.value_or(BrainBackendKind::cpu));
    const bool interactive =
        train_simulation(simulation, options.checkpoint, options.ticks, output);
    replace_checkpoint_file(simulation, options.checkpoint);
    print_training_status(simulation, options.checkpoint, output, interactive, false);
    return success_exit_code;
}

} // namespace

int run_command(
    const std::span<const std::string_view> arguments,
    std::ostream& output,
    std::ostream& error)
{
    if (arguments.empty()) {
        output << top_level_help;
        return success_exit_code;
    }
    if (arguments.size() == 1 && arguments.front() == "--help") {
        output << top_level_help;
        return success_exit_code;
    }

    try {
        if (arguments.front() == "run") {
            return run_simulation_command(arguments.subspan(1), output, error);
        }
        if (arguments.front() == "resume") {
            return resume_simulation_command(arguments.subspan(1), output, error);
        }
        return report_usage_error(error, "unknown command");
    } catch (const std::exception& exception) {
        return report_runtime_error(error, exception.what());
    }
}

} // namespace evobrain::runner
