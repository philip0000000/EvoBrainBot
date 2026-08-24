#pragma once

#include "evobrain/brain_backend.hpp"

namespace evobrain {

// Reports whether a CUDA device can execute the compiled brain kernels.
[[nodiscard]] bool cuda_brain_backend_available() noexcept;

// Evaluates one host batch through persistent CUDA buffers and returns state/output.
void evaluate_cuda_brain_batch(BrainBatch batch);

} // namespace evobrain
