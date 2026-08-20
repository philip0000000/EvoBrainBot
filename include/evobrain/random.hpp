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

    // Restores a generator from an exact previously recorded internal state.
    [[nodiscard]] static Pcg32 from_state(std::uint64_t state) noexcept;

    // Returns the exact internal state needed to resume the random sequence.
    [[nodiscard]] std::uint64_t state() const noexcept;

    // Returns a deterministic double in the half-open interval [0, 1).
    [[nodiscard]] double unit_interval() noexcept;

    // Returns a deterministic double in the half-open interval [minimum, maximum).
    [[nodiscard]] double uniform(double minimum, double maximum) noexcept;

    // Returns an unbiased deterministic integer in [0, exclusive_maximum),
    // requiring exclusive_maximum to be positive.
    [[nodiscard]] std::uint32_t bounded(std::uint32_t exclusive_maximum) noexcept;

private:
    struct RestoredStateTag { };

    // Constructs directly from serialized state without running seed initialization.
    Pcg32(std::uint64_t state, RestoredStateTag) noexcept;

    std::uint64_t state_ = 0;
};

} // namespace evobrain
