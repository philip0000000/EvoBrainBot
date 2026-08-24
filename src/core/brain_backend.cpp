#include "evobrain/brain_backend.hpp"

#include "parallel_executor.hpp"

#ifdef EVOBRAIN_HAS_CUDA
#include "cuda_brain_backend.hpp"
#endif

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace evobrain {
namespace {

// Rejects mismatched buffers before a backend can read or write out of bounds.
std::size_t validated_batch_size(const BrainBatch batch)
{
    const std::size_t size = batch.inputs.size();
    if (batch.agent_ids.size() != size || batch.parameters.size() != size
        || batch.structures.size() != size
        || batch.states.size() != size || batch.outputs.size() != size) {
        throw std::invalid_argument("brain batch buffers must have identical sizes");
    }
    return size;
}

// Evaluates the cache-friendly host arrays using the reusable bounded worker pool.
void evaluate_cpu_batch(const BrainBatch batch, const std::size_t requested_threads)
{
    const std::size_t size = validated_batch_size(batch);
    const std::size_t threads = size < 64
        ? 1
        : std::min(requested_threads, std::max<std::size_t>(2, size / 16));
    detail::parallel_executor().for_each_index(size, threads,
        [&](const std::size_t index) {
            batch.outputs[index] = evaluate_brain(batch.parameters[index],
                batch.structures[index], batch.states[index], batch.inputs[index]);
        });
}

} // namespace

std::string_view brain_backend_name(const BrainBackendKind backend) noexcept
{
    return backend == BrainBackendKind::gpu ? "gpu" : "cpu";
}

bool brain_backend_available(const BrainBackendKind backend) noexcept
{
    if (backend == BrainBackendKind::cpu) return true;
#ifdef EVOBRAIN_HAS_CUDA
    static const bool cuda_available = cuda_brain_backend_available();
    return cuda_available;
#else
    return false;
#endif
}

void evaluate_brain_batch(const BrainBackendKind backend, const BrainBatch batch,
    const std::size_t thread_count)
{
    validated_batch_size(batch);
    if (backend == BrainBackendKind::cpu) {
        evaluate_cpu_batch(batch, thread_count);
        return;
    }
#ifdef EVOBRAIN_HAS_CUDA
    evaluate_cuda_brain_batch(batch);
#else
    static_cast<void>(batch);
    static_cast<void>(thread_count);
    throw std::runtime_error(
        "GPU brain backend requested, but this build does not include CUDA support");
#endif
}

} // namespace evobrain
