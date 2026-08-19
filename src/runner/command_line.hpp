#pragma once

#include <iosfwd>
#include <span>
#include <string_view>

namespace evobrain::runner {

inline constexpr int success_exit_code = 0;
inline constexpr int usage_error_exit_code = 2;

// Parses and executes a command using the supplied output streams.
int run_command(
    std::span<const std::string_view> arguments,
    std::ostream& output,
    std::ostream& error);

} // namespace evobrain::runner
