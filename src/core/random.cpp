#include "evobrain/random.hpp"

#include <cstdint>

namespace evobrain {
namespace {

constexpr std::uint64_t multiplier = 6364136223846793005ULL;
constexpr std::uint64_t increment = 1442695040888963407ULL;

} // namespace

Pcg32::Pcg32(const std::uint64_t seed) noexcept
{
    // The two advances follow PCG's recommended initialization procedure. A
    // fixed odd increment selects one stable stream for this single-seed API.
    static_cast<void>(next());
    state_ += seed;
    static_cast<void>(next());
}

std::uint32_t Pcg32::next() noexcept
{
    const std::uint64_t previous_state = state_;
    state_ = previous_state * multiplier + increment;

    // PCG-XSH-RR uses the old state for both the xorshift and rotation so the
    // result is identical wherever fixed-width unsigned arithmetic is used.
    const auto xorshifted = static_cast<std::uint32_t>(
        ((previous_state >> 18U) ^ previous_state) >> 27U);
    const auto rotation = static_cast<std::uint32_t>(previous_state >> 59U);
    return (xorshifted >> rotation)
        | (xorshifted << ((0U - rotation) & 31U));
}

} // namespace evobrain
