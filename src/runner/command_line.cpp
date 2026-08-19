#include "command_line.hpp"

#include "evobrain/simulation.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <span>
#include <string_view>
#include <system_error>

namespace evobrain::runner {
namespace {

constexpr std::string_view top_level_help =
    "Usage:\n"
    "  EvoBrainBot --help\n"
    "  EvoBrainBot run --help\n"
    "  EvoBrainBot run --seed <seed> --ticks <ticks>\n";

constexpr std::string_view run_help =
    "Usage: EvoBrainBot run --seed <seed> --ticks <ticks>\n"
    "\n"
    "Both values are required decimal unsigned 64-bit integers.\n";

struct RunOptions {
    std::optional<std::uint64_t> seed;
    std::optional<std::uint64_t> ticks;
};

// Reports a command-line usage error and returns the documented exit code.
int report_usage_error(std::ostream& error, const std::string_view message)
{
    error << "Error: " << message << '\n';
    return usage_error_exit_code;
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

// Parses run options and executes the simulation when the command is valid.
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
    for (std::size_t index = 0; index < arguments.size();) {
        const std::string_view option = arguments[index];
        std::optional<std::uint64_t>* destination = nullptr;
        if (option == "--seed") {
            destination = &options.seed;
        } else if (option == "--ticks") {
            destination = &options.ticks;
        } else if (option.starts_with("--")) {
            return report_usage_error(error, "unknown option");
        } else {
            return report_usage_error(error, "unexpected positional argument");
        }

        if (destination->has_value()) {
            return report_usage_error(error, "duplicate option");
        }
        if (index + 1 >= arguments.size()
            || arguments[index + 1].starts_with("--")) {
            return report_usage_error(error, "missing option value");
        }

        const auto parsed_value = parse_unsigned_decimal(arguments[index + 1]);
        if (!parsed_value.has_value()) {
            return report_usage_error(error, "invalid unsigned integer value");
        }
        *destination = *parsed_value;
        index += 2;
    }

    if (!options.seed.has_value()) {
        return report_usage_error(error, "missing required option '--seed'");
    }
    if (!options.ticks.has_value()) {
        return report_usage_error(error, "missing required option '--ticks'");
    }

    Simulation simulation(SimulationConfig {.seed = *options.seed});
    simulation.run_for(*options.ticks);
    output << "Seed: " << *options.seed << '\n'
           << "Completed ticks: " << simulation.current_tick() << '\n';
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
    if (arguments.front() != "run") {
        return report_usage_error(error, "unknown command");
    }
    return run_simulation_command(arguments.subspan(1), output, error);
}

} // namespace evobrain::runner
