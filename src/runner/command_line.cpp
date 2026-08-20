#include "command_line.hpp"

#include "evobrain/checkpoint.hpp"
#include "evobrain/simulation.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

namespace evobrain::runner {
namespace {

constexpr std::string_view top_level_help =
    "Usage:\n"
    "  EvoBrainBot --help\n"
    "  EvoBrainBot run --help\n"
    "  EvoBrainBot run --seed <seed> --ticks <ticks> [--checkpoint-out <path>]\n"
    "  EvoBrainBot resume --help\n"
    "  EvoBrainBot resume --checkpoint-in <path> --ticks <ticks> "
    "[--checkpoint-out <path>]\n";

constexpr std::string_view run_help =
    "Usage: EvoBrainBot run --seed <seed> --ticks <ticks> "
    "[--checkpoint-out <path>]\n"
    "\n"
    "Seed and ticks are required decimal unsigned 64-bit integers.\n"
    "If checkpoint-out is omitted, the final state is saved to autosave.evo.\n";

constexpr std::string_view resume_help =
    "Usage: EvoBrainBot resume --checkpoint-in <path> --ticks <ticks> "
    "[--checkpoint-out <path>]\n"
    "\n"
    "The checkpoint and an additional decimal unsigned 64-bit tick count are "
    "required.\n"
    "If checkpoint-out is omitted, the final state is saved to autosave.evo.\n";

constexpr char default_checkpoint_output[] = "autosave.evo";

struct RunOptions {
    std::optional<std::uint64_t> seed;
    std::optional<std::uint64_t> ticks;
    std::optional<std::string> checkpoint_output;
};

struct ResumeOptions {
    std::optional<std::string> checkpoint_input;
    std::optional<std::uint64_t> ticks;
    std::optional<std::string> checkpoint_output;
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

// Prints the stable final summary shared by new and resumed runs.
void print_summary(const Simulation& simulation, std::ostream& output)
{
    const SimulationStats stats = simulation.stats();
    output << "Seed: " << stats.seed << '\n'
           << "Completed ticks: " << stats.completed_ticks << '\n'
           << "Population: " << stats.population << '\n'
           << "Food: " << stats.food << '\n'
           << "Births: " << stats.births << '\n'
           << "Introduced agents: " << stats.introduced_agents << '\n'
           << "Deaths: " << stats.deaths << '\n';
}

// Saves a completed simulation and verifies buffered data reached the stream.
void save_checkpoint_file(
    const Simulation& simulation,
    const std::string& path)
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
Simulation load_checkpoint_file(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open checkpoint input");
    }
    return load_checkpoint(input);
}

// Parses new-run options and executes a simulation when syntax is valid.
int run_simulation_command(
    const std::span<const std::string_view> arguments,
    std::ostream& output,
    std::ostream& error)
{
    if (arguments.size() == 1 && arguments.front() == "--help") {
        output << run_help;
        return success_exit_code;
    }

    RunOptions options;
    for (std::size_t index = 0; index < arguments.size(); index += 2) {
        const std::string_view option = arguments[index];
        if (!option.starts_with("--")) {
            return report_usage_error(error, "unexpected positional argument");
        }
        if (option != "--seed" && option != "--ticks"
            && option != "--checkpoint-out") {
            return report_usage_error(error, "unknown option");
        }
        const auto value = option_value(arguments, index, error);
        if (!value.has_value()) {
            return usage_error_exit_code;
        }

        if (option == "--seed" || option == "--ticks") {
            std::optional<std::uint64_t>& destination = option == "--seed"
                ? options.seed
                : options.ticks;
            if (destination.has_value()) {
                return report_usage_error(error, "duplicate option");
            }
            destination = parse_unsigned_decimal(*value);
            if (!destination.has_value()) {
                return report_usage_error(error, "invalid unsigned integer value");
            }
        } else if (option == "--checkpoint-out") {
            if (options.checkpoint_output.has_value()) {
                return report_usage_error(error, "duplicate option");
            }
            options.checkpoint_output = std::string(*value);
        }
    }

    if (!options.seed.has_value()) {
        return report_usage_error(error, "missing required option '--seed'");
    }
    if (!options.ticks.has_value()) {
        return report_usage_error(error, "missing required option '--ticks'");
    }

    Simulation simulation(SimulationConfig {.seed = *options.seed});
    simulation.run_for(*options.ticks);
    save_checkpoint_file(
        simulation,
        options.checkpoint_output.value_or(default_checkpoint_output));
    print_summary(simulation, output);
    return success_exit_code;
}

// Parses resume options and continues a validated checkpoint when syntax is valid.
int resume_simulation_command(
    const std::span<const std::string_view> arguments,
    std::ostream& output,
    std::ostream& error)
{
    if (arguments.size() == 1 && arguments.front() == "--help") {
        output << resume_help;
        return success_exit_code;
    }

    ResumeOptions options;
    for (std::size_t index = 0; index < arguments.size(); index += 2) {
        const std::string_view option = arguments[index];
        if (!option.starts_with("--")) {
            return report_usage_error(error, "unexpected positional argument");
        }
        if (option != "--ticks" && option != "--checkpoint-in"
            && option != "--checkpoint-out") {
            return report_usage_error(error, "unknown option");
        }
        const auto value = option_value(arguments, index, error);
        if (!value.has_value()) {
            return usage_error_exit_code;
        }

        if (option == "--ticks") {
            if (options.ticks.has_value()) {
                return report_usage_error(error, "duplicate option");
            }
            options.ticks = parse_unsigned_decimal(*value);
            if (!options.ticks.has_value()) {
                return report_usage_error(error, "invalid unsigned integer value");
            }
        } else if (option == "--checkpoint-in") {
            if (options.checkpoint_input.has_value()) {
                return report_usage_error(error, "duplicate option");
            }
            options.checkpoint_input = std::string(*value);
        } else if (option == "--checkpoint-out") {
            if (options.checkpoint_output.has_value()) {
                return report_usage_error(error, "duplicate option");
            }
            options.checkpoint_output = std::string(*value);
        }
    }

    if (!options.checkpoint_input.has_value()) {
        return report_usage_error(error, "missing required option '--checkpoint-in'");
    }
    if (!options.ticks.has_value()) {
        return report_usage_error(error, "missing required option '--ticks'");
    }

    Simulation simulation = load_checkpoint_file(*options.checkpoint_input);
    simulation.run_for(*options.ticks);
    save_checkpoint_file(
        simulation,
        options.checkpoint_output.value_or(default_checkpoint_output));
    print_summary(simulation, output);
    return success_exit_code;
}

} // namespace

int run_command(
    const std::span<const std::string_view> arguments,
    std::ostream& output,
    std::ostream& error)
{
    if (arguments.empty()) {
        return report_usage_error(error, "missing command");
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
