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

Pcg32::Pcg32(const std::uint64_t state, RestoredStateTag) noexcept
    : state_(state)
{
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

Pcg32 Pcg32::from_state(const std::uint64_t state) noexcept
{
    return Pcg32(state, RestoredStateTag {});
}

std::uint64_t Pcg32::state() const noexcept
{
    return state_;
}

double Pcg32::unit_interval() noexcept
{
    // Combining 27 and 26 random bits produces the 53-bit precision carried by
    // an IEEE-754 double without relying on library-defined distributions.
    constexpr double inverse_two_to_53 = 1.0 / 9007199254740992.0;
    const std::uint64_t high = static_cast<std::uint64_t>(next() >> 5U);
    const std::uint64_t low = static_cast<std::uint64_t>(next() >> 6U);
    return static_cast<double>((high << 26U) | low) * inverse_two_to_53;
}

double Pcg32::uniform(const double minimum, const double maximum) noexcept
{
    return minimum + (maximum - minimum) * unit_interval();
}

std::uint32_t Pcg32::bounded(const std::uint32_t exclusive_maximum) noexcept
{
    // Rejection avoids the modulo bias that would otherwise favor low values.
    const std::uint32_t threshold = (0U - exclusive_maximum) % exclusive_maximum;
    for (;;) {
        const std::uint32_t value = next();
        if (value >= threshold) {
            return value % exclusive_maximum;
        }
    }
}

} // namespace evobrain
