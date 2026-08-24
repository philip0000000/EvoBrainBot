#include "cuda_brain_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace evobrain {
namespace {

static_assert(std::is_trivially_copyable_v<BrainOutputs>);

// Carries every immutable genome field and initial state for one changed slot.
struct GenomeUpdate {
    std::size_t slot = 0;
    double parameters[brain_parameter_count] {};
    std::uint8_t founder_fast_path = 0;
    std::uint8_t hidden_active[brain_hidden_count] {};
    std::uint8_t input_hidden_enabled[input_hidden_weight_count] {};
    std::uint8_t hidden_output_enabled[brain_hidden_count * brain_output_count] {};
    std::uint8_t recurrent_enabled[recurrent_weight_count] {};
    double recurrent_weights[recurrent_weight_count] {};
    double previous_hidden[brain_hidden_count] {};
    double next_hidden[brain_hidden_count] {};
};

static_assert(std::is_trivially_copyable_v<GenomeUpdate>);

// Converts a CUDA status into a stable exception at the host/backend boundary.
void check_cuda(const cudaError_t status, const char* operation)
{
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename Value>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        if (data_ != nullptr) cudaFree(data_);
    }

    // Grows device storage without reallocating when the population later shrinks.
    [[nodiscard]] bool reserve(const std::size_t count)
    {
        if (count <= capacity_) return false;
        Value* replacement = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement), count * sizeof(Value)),
            "CUDA brain buffer allocation failed");
        if (data_ != nullptr) check_cuda(cudaFree(data_), "CUDA brain buffer release failed");
        data_ = replacement;
        capacity_ = count;
        return true;
    }

    // Uploads exactly one logical buffer after its capacity has been established.
    void upload(const Value* source, const std::size_t count)
    {
        if (count == 0) return;
        check_cuda(cudaMemcpy(data_, source, count * sizeof(Value), cudaMemcpyHostToDevice),
            "CUDA brain upload failed");
    }

    // Downloads exactly one logical buffer after kernel completion.
    void download(Value* destination, const std::size_t count) const
    {
        if (count == 0) return;
        check_cuda(cudaMemcpy(destination, data_, count * sizeof(Value), cudaMemcpyDeviceToHost),
            "CUDA brain download failed");
    }

    // Enqueues a host-to-device transfer in the caller's ordered CUDA stream.
    void upload_async(const Value* source, const std::size_t count,
        const cudaStream_t stream)
    {
        if (count == 0) return;
        check_cuda(cudaMemcpyAsync(data_, source, count * sizeof(Value),
                       cudaMemcpyHostToDevice, stream),
            "CUDA brain asynchronous upload failed");
    }

    // Enqueues a device-to-host transfer for the final per-tick synchronization.
    void download_async(Value* destination, const std::size_t count,
        const cudaStream_t stream) const
    {
        if (count == 0) return;
        check_cuda(cudaMemcpyAsync(destination, data_, count * sizeof(Value),
                       cudaMemcpyDeviceToHost, stream),
            "CUDA brain asynchronous download failed");
    }

    [[nodiscard]] Value* data() noexcept { return data_; }
    [[nodiscard]] const Value* data() const noexcept { return data_; }

private:
    Value* data_ = nullptr;
    std::size_t capacity_ = 0;
};

__device__ __forceinline__ double clamp_activation(const double value)
{
    // Branches match std::clamp semantics, including propagation of NaN values.
    return value < -1.0 ? -1.0 : value > 1.0 ? 1.0 : value;
}

__device__ __forceinline__ double soa_value(const double* values,
    const std::size_t component, const std::size_t stride,
    const std::size_t agent)
{
    return values[component * stride + agent];
}

__device__ __forceinline__ std::uint8_t soa_flag(const std::uint8_t* values,
    const std::size_t component, const std::size_t stride,
    const std::size_t agent)
{
    return values[component * stride + agent];
}

// Scatters one packed host update per block into fixed-stride device arrays.
__global__ void scatter_genome_updates_kernel(const GenomeUpdate* updates,
    const std::size_t update_count, const std::size_t stride,
    double* parameters, std::uint8_t* founder_fast_path,
    std::uint8_t* hidden_active, std::uint8_t* input_hidden_enabled,
    std::uint8_t* hidden_output_enabled, std::uint8_t* recurrent_enabled,
    double* recurrent_weights, double* previous_hidden, double* next_hidden)
{
    const std::size_t update_index = blockIdx.x;
    if (update_index >= update_count) return;
    const GenomeUpdate& update = updates[update_index];
    const std::size_t slot = update.slot;
    if (threadIdx.x == 0) founder_fast_path[slot] = update.founder_fast_path;
    for (std::size_t component = threadIdx.x; component < brain_parameter_count;
         component += blockDim.x) {
        parameters[component * stride + slot] = update.parameters[component];
    }
    for (std::size_t component = threadIdx.x; component < brain_hidden_count;
         component += blockDim.x) {
        hidden_active[component * stride + slot] = update.hidden_active[component];
        previous_hidden[component * stride + slot] = update.previous_hidden[component];
        next_hidden[component * stride + slot] = update.next_hidden[component];
    }
    for (std::size_t component = threadIdx.x; component < input_hidden_weight_count;
         component += blockDim.x) {
        input_hidden_enabled[component * stride + slot]
            = update.input_hidden_enabled[component];
    }
    constexpr std::size_t output_connections = brain_hidden_count * brain_output_count;
    for (std::size_t component = threadIdx.x; component < output_connections;
         component += blockDim.x) {
        hidden_output_enabled[component * stride + slot]
            = update.hidden_output_enabled[component];
    }
    for (std::size_t component = threadIdx.x; component < recurrent_weight_count;
         component += blockDim.x) {
        recurrent_enabled[component * stride + slot]
            = update.recurrent_enabled[component];
        recurrent_weights[component * stride + slot]
            = update.recurrent_weights[component];
    }
}

// Evaluates one complete brain per CUDA thread with fixed accumulation order.
__global__ void evaluate_brains_kernel(const std::size_t slot_extent,
    const std::size_t stride, const std::uint8_t* slot_active,
    const double* parameters, const std::uint8_t* founder_fast_path,
    const std::uint8_t* hidden_active, const std::uint8_t* input_hidden_enabled,
    const std::uint8_t* hidden_output_enabled, const std::uint8_t* recurrent_enabled,
    const double* recurrent_weights, double* previous_hidden, double* next_hidden,
    const double* inputs, BrainOutputs* outputs)
{
    const std::size_t agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= slot_extent || slot_active[agent] == 0) return;

    if (founder_fast_path[agent] != 0) {
        double hidden_values[brain_founder_hidden_count] {};
        for (std::size_t hidden = 0; hidden < brain_founder_hidden_count; ++hidden) {
            double value = soa_value(parameters, hidden_bias_offset + hidden,
                stride, agent);
            const std::size_t weights = hidden * brain_input_count;
            for (std::size_t input = 0; input < brain_input_count; ++input) {
                value += soa_value(parameters, weights + input, stride, agent)
                    * soa_value(inputs, input, stride, agent);
            }
            hidden_values[hidden] = clamp_activation(value);
        }
        double output_values[brain_output_count] {};
        for (std::size_t output = 0; output < brain_output_count; ++output) {
            double value = soa_value(parameters, output_bias_offset + output,
                stride, agent);
            const std::size_t weights = hidden_output_weight_offset
                + output * brain_hidden_count;
            for (std::size_t hidden = 0; hidden < brain_founder_hidden_count; ++hidden) {
                value += soa_value(parameters, weights + hidden, stride, agent)
                    * hidden_values[hidden];
            }
            output_values[output] = clamp_activation(value);
        }
        for (std::size_t hidden = 0; hidden < brain_hidden_count; ++hidden) {
            previous_hidden[hidden * stride + agent] = 0.0;
            next_hidden[hidden * stride + agent] = 0.0;
        }
        outputs[agent] = {.turn = output_values[0],
            .move = (output_values[1] + 1.0) * 0.5,
            .eat = (output_values[2] + 1.0) * 0.5};
        return;
    }

    for (std::size_t hidden = 0; hidden < brain_hidden_count; ++hidden) {
        const std::size_t hidden_state = hidden * stride + agent;
        if (soa_flag(hidden_active, hidden, stride, agent) == 0) {
            next_hidden[hidden_state] = 0.0;
            continue;
        }
        double value = soa_value(parameters, hidden_bias_offset + hidden,
            stride, agent);
        const std::size_t input_weights = hidden * brain_input_count;
        for (std::size_t input = 0; input < brain_input_count; ++input) {
            const std::size_t connection = input_weights + input;
            if (soa_flag(input_hidden_enabled, connection, stride, agent) != 0) {
                value += soa_value(parameters, connection, stride, agent)
                    * soa_value(inputs, input, stride, agent);
            }
        }
        const std::size_t recurrent_base = hidden * brain_hidden_count;
        for (std::size_t source = 0; source < brain_hidden_count; ++source) {
            const std::size_t connection = recurrent_base + source;
            if (soa_flag(recurrent_enabled, connection, stride, agent) != 0
                && soa_flag(hidden_active, source, stride, agent) != 0) {
                // Reads only the completed previous tick, preserving order independence.
                value += soa_value(recurrent_weights, connection, stride, agent)
                    * previous_hidden[source * stride + agent];
            }
        }
        next_hidden[hidden_state] = clamp_activation(value);
    }

    double output_values[brain_output_count] {};
    for (std::size_t output = 0; output < brain_output_count; ++output) {
        double value = soa_value(parameters, output_bias_offset + output,
            stride, agent);
        const std::size_t weights = hidden_output_weight_offset
            + output * brain_hidden_count;
        const std::size_t enabled = output * brain_hidden_count;
        for (std::size_t source = 0; source < brain_hidden_count; ++source) {
            if (soa_flag(hidden_output_enabled, enabled + source,
                    stride, agent) != 0) {
                value += soa_value(parameters, weights + source, stride, agent)
                    * next_hidden[source * stride + agent];
            }
        }
        output_values[output] = clamp_activation(value);
    }
    for (std::size_t hidden = 0; hidden < brain_hidden_count; ++hidden) {
        previous_hidden[hidden * stride + agent]
            = next_hidden[hidden * stride + agent];
    }
    outputs[agent] = {.turn = output_values[0],
        .move = (output_values[1] + 1.0) * 0.5,
        .eat = (output_values[2] + 1.0) * 0.5};
}

class CudaBrainContext {
public:
    // Evaluates a batch while retaining stable-ID genome slots across population churn.
    void evaluate(const BrainBatch batch)
    {
        const std::size_t population = batch.inputs.size();
        const void* identity = batch.cache_identity != nullptr
            ? batch.cache_identity
            : static_cast<const void*>(batch.parameters.data());
        if (population == 0) {
            clear_slots();
            owner_identity_ = identity;
            return;
        }

        const bool allocation_changed = reserve_slots(population);
        const bool rebuild = allocation_changed || owner_identity_ != identity
            || batch.reset_cache || !initialized_;
        bool activity_changed = false;
        if (rebuild) {
            rebuild_slots(batch);
            activity_changed = true;
        } else if (batch.population_changed) {
            activity_changed = reconcile_slots(batch);
        } else {
            updates_host_.clear();
        }
        if (rebuild || batch.population_changed) {
            refresh_identity_mapping(batch.agent_ids.size());
        }
        if (activity_changed) {
            slot_active_.upload(slot_active_host_.data(), slot_extent_);
        }
        upload_updates();
        if (batch.state_changed) pack_and_upload_state(batch);
        pack_and_upload_inputs(batch);

        constexpr unsigned int threads_per_block = 128;
        const unsigned int blocks = static_cast<unsigned int>(
            (slot_extent_ + threads_per_block - 1) / threads_per_block);
        evaluate_brains_kernel<<<blocks, threads_per_block>>>(slot_extent_,
            slot_capacity_, slot_active_.data(), parameters_.data(),
            founder_fast_path_.data(), hidden_active_.data(),
            input_hidden_enabled_.data(), hidden_output_enabled_.data(),
            recurrent_enabled_.data(), recurrent_weights_.data(),
            previous_hidden_.data(), next_hidden_.data(), inputs_.data(), outputs_.data());
        check_cuda(cudaGetLastError(), "CUDA brain kernel launch failed");

        if (!identity_mapping_) outputs_host_.resize(slot_extent_);
        previous_hidden_host_.resize(brain_hidden_count * slot_capacity_);
        outputs_.download_async(identity_mapping_ ? batch.outputs.data()
                                                  : outputs_host_.data(),
            slot_extent_, nullptr);
        check_cuda(cudaMemcpy2DAsync(previous_hidden_host_.data(),
                       slot_capacity_ * sizeof(double), previous_hidden_.data(),
                       slot_capacity_ * sizeof(double), slot_extent_ * sizeof(double),
                       brain_hidden_count, cudaMemcpyDeviceToHost, nullptr),
            "CUDA recurrent-state download failed");
        // One boundary wait covers input upload, kernel work, and both downloads.
        check_cuda(cudaStreamSynchronize(nullptr), "CUDA brain evaluation failed");
        for (std::size_t agent = 0; agent < population; ++agent) {
            const std::size_t slot = batch_to_slot_[agent];
            if (!identity_mapping_) batch.outputs[agent] = outputs_host_[slot];
            for (std::size_t hidden = 0; hidden < brain_hidden_count; ++hidden) {
                const double value
                    = previous_hidden_host_[hidden * slot_capacity_ + slot];
                batch.states[agent].previous_hidden[hidden] = value;
                batch.states[agent].next_hidden[hidden] = value;
            }
        }
        owner_identity_ = identity;
        initialized_ = true;
    }

private:
    // Establishes a dense stride and adds modest headroom only after later growth.
    [[nodiscard]] bool reserve_slots(const std::size_t required)
    {
        if (required <= slot_capacity_) return false;
        std::size_t capacity = required;
        if (slot_capacity_ != 0) {
            const std::size_t headroom = std::max<std::size_t>(128, slot_capacity_ / 8);
            capacity = slot_capacity_ > std::numeric_limits<std::size_t>::max() - headroom
                ? required
                : std::max(required, slot_capacity_ + headroom);
        }
        slot_capacity_ = capacity;
        static_cast<void>(parameters_.reserve(brain_parameter_count * capacity));
        static_cast<void>(founder_fast_path_.reserve(capacity));
        static_cast<void>(hidden_active_.reserve(brain_hidden_count * capacity));
        static_cast<void>(input_hidden_enabled_.reserve(
            input_hidden_weight_count * capacity));
        static_cast<void>(hidden_output_enabled_.reserve(
            brain_hidden_count * brain_output_count * capacity));
        static_cast<void>(recurrent_enabled_.reserve(recurrent_weight_count * capacity));
        static_cast<void>(recurrent_weights_.reserve(recurrent_weight_count * capacity));
        static_cast<void>(previous_hidden_.reserve(brain_hidden_count * capacity));
        static_cast<void>(next_hidden_.reserve(brain_hidden_count * capacity));
        static_cast<void>(inputs_.reserve(brain_input_count * capacity));
        static_cast<void>(outputs_.reserve(capacity));
        static_cast<void>(slot_active_.reserve(capacity));
        slot_active_host_.resize(capacity);
        id_by_slot_.resize(capacity);
        return true;
    }

    // Clears host mappings so the next non-empty batch performs a complete rebuild.
    void clear_slots()
    {
        slot_by_id_.clear();
        free_slots_.clear();
        batch_to_slot_.clear();
        updates_host_.clear();
        slot_extent_ = 0;
        initialized_ = false;
    }

    // Validates stable IDs whenever the membership contract says it may have changed.
    void validate_unique_ids(const BrainBatch batch) const
    {
        std::unordered_set<std::uint64_t> identities;
        identities.reserve(batch.agent_ids.size());
        for (const std::uint64_t identity : batch.agent_ids) {
            if (identity == 0 || !identities.insert(identity).second) {
                throw std::invalid_argument(
                    "CUDA brain batch agent IDs must be unique and nonzero");
            }
        }
    }

    // Packs one changed genome and its initial recurrent state for a scatter update.
    [[nodiscard]] GenomeUpdate make_update(const BrainBatch batch,
        const std::size_t agent, const std::size_t slot) const
    {
        GenomeUpdate update {.slot = slot};
        const BrainStructure& structure = batch.structures[agent];
        update.founder_fast_path = structure.founder_fast_path;
        for (std::size_t component = 0; component < brain_parameter_count; ++component) {
            update.parameters[component] = batch.parameters[agent][component];
        }
        for (std::size_t component = 0; component < brain_hidden_count; ++component) {
            update.hidden_active[component] = structure.hidden_active[component];
            update.previous_hidden[component]
                = batch.states[agent].previous_hidden[component];
            update.next_hidden[component] = batch.states[agent].next_hidden[component];
        }
        for (std::size_t component = 0; component < input_hidden_weight_count;
             ++component) {
            update.input_hidden_enabled[component]
                = structure.input_hidden_enabled[component];
        }
        constexpr std::size_t output_connections
            = brain_hidden_count * brain_output_count;
        for (std::size_t component = 0; component < output_connections; ++component) {
            update.hidden_output_enabled[component]
                = structure.hidden_output_enabled[component];
        }
        for (std::size_t component = 0; component < recurrent_weight_count; ++component) {
            update.recurrent_enabled[component] = structure.recurrent_enabled[component];
            update.recurrent_weights[component] = structure.recurrent_weights[component];
        }
        return update;
    }

    // Rebuilds dense slots after initialization, owner changes, or capacity growth.
    void rebuild_slots(const BrainBatch batch)
    {
        validate_unique_ids(batch);
        slot_by_id_.clear();
        slot_by_id_.reserve(batch.agent_ids.size());
        free_slots_.clear();
        batch_to_slot_.resize(batch.agent_ids.size());
        updates_host_.clear();
        updates_host_.reserve(batch.agent_ids.size());
        std::fill(slot_active_host_.begin(), slot_active_host_.end(), std::uint8_t {0});
        std::fill(id_by_slot_.begin(), id_by_slot_.end(), std::uint64_t {0});
        slot_extent_ = batch.agent_ids.size();
        for (std::size_t agent = 0; agent < batch.agent_ids.size(); ++agent) {
            const std::size_t slot = agent;
            const std::uint64_t identity = batch.agent_ids[agent];
            slot_by_id_.emplace(identity, slot);
            id_by_slot_[slot] = identity;
            slot_active_host_[slot] = 1;
            batch_to_slot_[agent] = slot;
            updates_host_.push_back(make_update(batch, agent, slot));
        }
    }

    // Preserves retained slots, releases deaths, and batches only newborn genomes.
    [[nodiscard]] bool reconcile_slots(const BrainBatch batch)
    {
        validate_unique_ids(batch);
        std::vector<std::uint8_t> retained(slot_extent_, 0);
        std::vector<std::size_t> missing_agents;
        batch_to_slot_.resize(batch.agent_ids.size());
        for (std::size_t agent = 0; agent < batch.agent_ids.size(); ++agent) {
            const auto found = slot_by_id_.find(batch.agent_ids[agent]);
            if (found == slot_by_id_.end()) {
                missing_agents.push_back(agent);
                continue;
            }
            batch_to_slot_[agent] = found->second;
            retained[found->second] = 1;
        }

        bool changed = false;
        for (std::size_t slot = 0; slot < slot_extent_; ++slot) {
            if (slot_active_host_[slot] == 0 || retained[slot] != 0) continue;
            slot_by_id_.erase(id_by_slot_[slot]);
            id_by_slot_[slot] = 0;
            slot_active_host_[slot] = 0;
            free_slots_.push_back(slot);
            changed = true;
        }

        updates_host_.clear();
        updates_host_.reserve(missing_agents.size());
        for (const std::size_t agent : missing_agents) {
            std::size_t slot = 0;
            if (!free_slots_.empty()) {
                slot = free_slots_.back();
                free_slots_.pop_back();
            } else {
                if (slot_extent_ >= slot_capacity_) {
                    throw std::logic_error("CUDA brain slot capacity invariant failed");
                }
                slot = slot_extent_++;
            }
            const std::uint64_t identity = batch.agent_ids[agent];
            slot_by_id_.emplace(identity, slot);
            id_by_slot_[slot] = identity;
            slot_active_host_[slot] = 1;
            batch_to_slot_[agent] = slot;
            updates_host_.push_back(make_update(batch, agent, slot));
            changed = true;
        }

        // Rare dense rebuilding bounds inactive kernel lanes after a lasting collapse.
        if (!batch.agent_ids.empty() && slot_extent_ > batch.agent_ids.size() + 128
            && slot_extent_ / 2 > batch.agent_ids.size()) {
            rebuild_slots(batch);
            return true;
        }
        return changed;
    }

    // Enables direct output transfer while slots still match the dense host order.
    void refresh_identity_mapping(const std::size_t population)
    {
        identity_mapping_ = slot_extent_ == population
            && batch_to_slot_.size() == population;
        for (std::size_t agent = 0; identity_mapping_ && agent < population; ++agent) {
            identity_mapping_ = batch_to_slot_[agent] == agent;
        }
    }

    // Uploads one packed update buffer and scatters changed genomes in one kernel.
    void upload_updates()
    {
        if (updates_host_.empty()) return;
        static_cast<void>(updates_.reserve(updates_host_.size()));
        updates_.upload(updates_host_.data(), updates_host_.size());
        constexpr unsigned int threads = 128;
        scatter_genome_updates_kernel<<<
            static_cast<unsigned int>(updates_host_.size()), threads>>>(
            updates_.data(), updates_host_.size(), slot_capacity_, parameters_.data(),
            founder_fast_path_.data(), hidden_active_.data(),
            input_hidden_enabled_.data(), hidden_output_enabled_.data(),
            recurrent_enabled_.data(), recurrent_weights_.data(),
            previous_hidden_.data(), next_hidden_.data());
        check_cuda(cudaGetLastError(), "CUDA genome update kernel launch failed");
    }

    // Uploads all current recurrent states only after an explicit cache reset.
    void pack_and_upload_state(const BrainBatch batch)
    {
        previous_hidden_host_.resize(brain_hidden_count * slot_capacity_);
        next_hidden_host_.resize(brain_hidden_count * slot_capacity_);
        for (std::size_t agent = 0; agent < batch.agent_ids.size(); ++agent) {
            const std::size_t slot = batch_to_slot_[agent];
            for (std::size_t hidden = 0; hidden < brain_hidden_count; ++hidden) {
                const std::size_t destination = hidden * slot_capacity_ + slot;
                previous_hidden_host_[destination]
                    = batch.states[agent].previous_hidden[hidden];
                next_hidden_host_[destination] = batch.states[agent].next_hidden[hidden];
            }
        }
        check_cuda(cudaMemcpy2D(previous_hidden_.data(), slot_capacity_ * sizeof(double),
                       previous_hidden_host_.data(), slot_capacity_ * sizeof(double),
                       slot_extent_ * sizeof(double), brain_hidden_count,
                       cudaMemcpyHostToDevice),
            "CUDA recurrent-state upload failed");
        check_cuda(cudaMemcpy2D(next_hidden_.data(), slot_capacity_ * sizeof(double),
                       next_hidden_host_.data(), slot_capacity_ * sizeof(double),
                       slot_extent_ * sizeof(double), brain_hidden_count,
                       cudaMemcpyHostToDevice),
            "CUDA next-state upload failed");
    }

    // Flattens sensors directly into persistent slot order for coalesced kernel reads.
    void pack_and_upload_inputs(const BrainBatch batch)
    {
        inputs_host_.resize(brain_input_count * slot_capacity_);
        for (std::size_t agent = 0; agent < batch.agent_ids.size(); ++agent) {
            const std::size_t slot = batch_to_slot_[agent];
            std::size_t component = 0;
            for (const VisionRayInputs& ray : batch.inputs[agent].vision) {
                inputs_host_[component++ * slot_capacity_ + slot] = ray.red;
                inputs_host_[component++ * slot_capacity_ + slot] = ray.green;
                inputs_host_[component++ * slot_capacity_ + slot] = ray.blue;
                inputs_host_[component++ * slot_capacity_ + slot] = ray.proximity;
            }
            inputs_host_[component++ * slot_capacity_ + slot] = batch.inputs[agent].energy;
            inputs_host_[component * slot_capacity_ + slot] = batch.inputs[agent].damage;
        }
        check_cuda(cudaMemcpy2DAsync(inputs_.data(), slot_capacity_ * sizeof(double),
                       inputs_host_.data(), slot_capacity_ * sizeof(double),
                       slot_extent_ * sizeof(double), brain_input_count,
                       cudaMemcpyHostToDevice, nullptr),
            "CUDA brain input upload failed");
    }

    const void* owner_identity_ = nullptr;
    bool initialized_ = false;
    bool identity_mapping_ = false;
    std::size_t slot_capacity_ = 0;
    std::size_t slot_extent_ = 0;
    std::unordered_map<std::uint64_t, std::size_t> slot_by_id_;
    std::vector<std::uint64_t> id_by_slot_;
    std::vector<std::size_t> free_slots_;
    std::vector<std::size_t> batch_to_slot_;

    DeviceBuffer<double> parameters_;
    DeviceBuffer<std::uint8_t> slot_active_;
    DeviceBuffer<std::uint8_t> founder_fast_path_;
    DeviceBuffer<std::uint8_t> hidden_active_;
    DeviceBuffer<std::uint8_t> input_hidden_enabled_;
    DeviceBuffer<std::uint8_t> hidden_output_enabled_;
    DeviceBuffer<std::uint8_t> recurrent_enabled_;
    DeviceBuffer<double> recurrent_weights_;
    DeviceBuffer<double> previous_hidden_;
    DeviceBuffer<double> next_hidden_;
    DeviceBuffer<double> inputs_;
    DeviceBuffer<BrainOutputs> outputs_;
    DeviceBuffer<GenomeUpdate> updates_;

    std::vector<std::uint8_t> slot_active_host_;
    std::vector<GenomeUpdate> updates_host_;
    std::vector<double> previous_hidden_host_;
    std::vector<double> next_hidden_host_;
    std::vector<double> inputs_host_;
    std::vector<BrainOutputs> outputs_host_;
};

} // namespace

bool cuda_brain_backend_available() noexcept
{
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void evaluate_cuda_brain_batch(const BrainBatch batch)
{
    static const bool device_available = cuda_brain_backend_available();
    if (!device_available) {
        throw std::runtime_error("GPU brain backend requested, but no CUDA device is available");
    }
    // One process-wide context prevents independent simulations from racing shared buffers.
    static std::mutex context_mutex;
    static CudaBrainContext context;
    const std::lock_guard lock(context_mutex);
    context.evaluate(batch);
}

} // namespace evobrain
