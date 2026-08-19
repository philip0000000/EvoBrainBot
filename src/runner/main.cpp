#include "command_line.hpp"

#include <cstddef>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

// Converts process arguments to non-owning views and delegates runner behavior.
int main(const int argument_count, const char* const argument_values[])
{
    std::vector<std::string_view> arguments;
    // C++ permits a hosted implementation to supply zero process arguments.
    if (argument_count > 1) {
        arguments.reserve(static_cast<std::size_t>(argument_count - 1));
    }
    for (int index = 1; index < argument_count; ++index) {
        arguments.emplace_back(argument_values[index]);
    }

    return evobrain::runner::run_command(
        std::span<const std::string_view>(arguments), std::cout, std::cerr);
}
