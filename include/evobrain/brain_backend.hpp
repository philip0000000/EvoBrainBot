#pragma once

#include "evobrain/brain.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace evobrain {

enum class BrainBackendKind : std::uint8_t { cpu = 0, gpu = 1 };

// Describes one contiguous brain batch shared by CPU and GPU implementations.
struct BrainBatch {
    // Stable, nonzero identities let CUDA retain genomes across host-array reordering.
    std::span<const std::uint64_t> agent_ids;
    std::span<const BrainParameters> parameters;
    std::span<const BrainStructure> structures;
    std::span<BrainState> states;
    std::span<const BrainInputs> inputs;
    std::span<BrainOutputs> outputs;
    // Distinguishes callers because the CUDA implementation owns one shared context.
    const void* cache_identity = nullptr;
    // Membership/order changes are reconciled. A retained ID is assumed to keep
    // identical weights and topology; runtime genome changes must be uploaded
    // explicitly or force a cache reset before the next GPU evaluation.
    bool population_changed = true;
    // Requests a complete recurrent-state upload after an external backend change.
    bool state_changed = true;
    // Invalidates all retained slots, such as on first use or backend replacement.
    bool reset_cache = true;
};

// Returns the stable user-facing name used by CLI, GUI, and diagnostics.
[[nodiscard]] std::string_view brain_backend_name(BrainBackendKind backend) noexcept;

// Reports whether the selected backend can initialize in this build and process.
[[nodiscard]] bool brain_backend_available(BrainBackendKind backend) noexcept;

// Evaluates a complete batch through the selected backend with shared semantics.
void evaluate_brain_batch(
    BrainBackendKind backend,
    BrainBatch batch,
    std::size_t thread_count);

} // namespace evobrain
