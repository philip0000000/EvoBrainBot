#pragma once

#include <cstdint>

namespace evobrain {

// Generates the project-defined deterministic PCG-XSH-RR 64/32 sequence.
class Pcg32 {
public:
    // Initializes the generator from a 64-bit seed on a fixed PCG stream.
    explicit Pcg32(std::uint64_t seed) noexcept;

    // Advances the generator and returns the next deterministic 32-bit value.
    [[nodiscard]] std::uint32_t next() noexcept;

private:
    std::uint64_t state_ = 0;
};

} // namespace evobrain
