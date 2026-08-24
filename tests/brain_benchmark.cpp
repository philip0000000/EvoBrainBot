#include "evobrain/brain_backend.hpp"
#include "evobrain/random.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

namespace {

enum class BrainMix { feed_forward_8, recurrent_8, recurrent_12, mixed };

struct Options {
    evobrain::BrainBackendKind backend = evobrain::BrainBackendKind::cpu;
    std::size_t population = 300;
    std::uint64_t ticks = 100;
    std::uint64_t seed = 5;
    std::size_t replacements_per_tick = 0;
    BrainMix mix = BrainMix::feed_forward_8;
};

// Parses one complete unsigned decimal value without signs or suffixes.
template <typename Integer>
bool parse_unsigned(const std::string_view text, Integer& value)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc {} && result.ptr == text.data() + text.size();
}

// Parses the bounded benchmark command so accidental long automated runs are rejected.
bool parse_options(const std::span<char*> arguments, Options& options)
{
    for (std::size_t index = 1; index < arguments.size(); index += 2) {
        if (index + 1 >= arguments.size()) return false;
        const std::string_view option = arguments[index];
        const std::string_view value = arguments[index + 1];
        if (option == "--backend") {
            if (value == "cpu") options.backend = evobrain::BrainBackendKind::cpu;
            else if (value == "gpu") options.backend = evobrain::BrainBackendKind::gpu;
            else return false;
        } else if (option == "--population") {
            if (!parse_unsigned(value, options.population)
                || (options.population != 250 && options.population != 300
                    && options.population != 2'000 && options.population != 3'000
                    && options.population != 5'000 && options.population != 30'000)) {
                return false;
            }
        } else if (option == "--ticks") {
            if (!parse_unsigned(value, options.ticks)
                || (options.ticks != 100 && options.ticks != 500
                    && options.ticks != 1'000)) {
                return false;
            }
        } else if (option == "--seed") {
            if (!parse_unsigned(value, options.seed)) return false;
        } else if (option == "--replacements-per-tick") {
            if (!parse_unsigned(value, options.replacements_per_tick)) return false;
        } else if (option == "--mix") {
            if (value == "feed-forward-8") options.mix = BrainMix::feed_forward_8;
            else if (value == "recurrent-8") options.mix = BrainMix::recurrent_8;
            else if (value == "recurrent-12") options.mix = BrainMix::recurrent_12;
            else if (value == "mixed") options.mix = BrainMix::mixed;
            else return false;
        } else {
            return false;
        }
    }
    return options.replacements_per_tick <= options.population;
}

// Creates one deterministic topology from the four documented benchmark mixes.
evobrain::BrainStructure benchmark_structure(
    const BrainMix requested_mix, const std::size_t agent_index, evobrain::Pcg32& random)
{
    BrainMix mix = requested_mix;
    if (mix == BrainMix::mixed) {
        constexpr BrainMix choices[] {
            BrainMix::feed_forward_8, BrainMix::recurrent_8, BrainMix::recurrent_12};
        mix = choices[agent_index % std::size(choices)];
    }
    evobrain::BrainStructure structure = evobrain::founder_brain_structure();
    if (mix == BrainMix::feed_forward_8) return structure;

    structure.founder_fast_path = 0;
    const std::size_t active_hidden = mix == BrainMix::recurrent_8
        ? evobrain::brain_founder_hidden_count
        : evobrain::brain_hidden_count;
    for (std::size_t hidden = 0; hidden < active_hidden; ++hidden) {
        structure.hidden_active[hidden] = 1;
        for (std::size_t input = 0; input < evobrain::brain_input_count; ++input) {
            structure.input_hidden_enabled[hidden * evobrain::brain_input_count + input] = 1;
        }
        for (std::size_t output = 0; output < evobrain::brain_output_count; ++output) {
            structure.hidden_output_enabled[output * evobrain::brain_hidden_count + hidden] = 1;
        }
    }
    for (std::size_t target = 0; target < active_hidden; ++target) {
        for (std::size_t source = 0; source < active_hidden; ++source) {
            const std::size_t connection = target * evobrain::brain_hidden_count + source;
            structure.recurrent_enabled[connection] = 1;
            structure.recurrent_weights[connection] = random.uniform(-1.0, 1.0);
        }
    }
    return structure;
}

// Returns the stable command-line spelling for one benchmark workload mix.
std::string_view mix_name(const BrainMix mix)
{
    switch (mix) {
    case BrainMix::feed_forward_8: return "feed-forward-8";
    case BrainMix::recurrent_8: return "recurrent-8";
    case BrainMix::recurrent_12: return "recurrent-12";
    case BrainMix::mixed: return "mixed";
    }
    return "unknown";
}

// Reports enough build context to make benchmark records comparable.
std::string_view compiler_name() noexcept
{
#if defined(_MSC_VER)
    return "msvc";
#elif defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#else
    return "unknown";
#endif
}

// Reports whether compiler optimizations and release assertions are selected.
std::string_view build_mode() noexcept
{
#ifdef NDEBUG
    return "release";
#else
    return "debug";
#endif
}

} // namespace

int main(const int argc, char** argv)
{
    Options options;
    if (!parse_options(std::span(argv, static_cast<std::size_t>(argc)), options)) {
        std::cerr << "Usage: evobrain_brain_benchmark [--backend cpu|gpu]"
                     " [--population 250|300|2000|3000|5000|30000]"
                     " [--ticks 100|500|1000]"
                     " [--seed <seed>]"
                     " [--replacements-per-tick <count>]"
                     " [--mix feed-forward-8|recurrent-8|recurrent-12|mixed]\n";
        return 2;
    }
    if (!evobrain::brain_backend_available(options.backend)) {
        std::cerr << "Requested brain backend is unavailable\n";
        return 1;
    }

    const auto initialization_started = std::chrono::steady_clock::now();
    evobrain::Pcg32 random(options.seed);
    std::vector<evobrain::BrainParameters> parameters(options.population);
    std::vector<std::uint64_t> agent_ids(options.population);
    std::vector<evobrain::BrainStructure> structures;
    structures.reserve(options.population);
    std::vector<evobrain::BrainState> states(options.population);
    std::vector<evobrain::BrainInputs> inputs(options.population);
    std::vector<evobrain::BrainOutputs> outputs(options.population);
    for (std::size_t agent = 0; agent < options.population; ++agent) {
        agent_ids[agent] = static_cast<std::uint64_t>(agent + 1);
        for (double& parameter : parameters[agent]) {
            parameter = random.uniform(-1.0, 1.0);
        }
        structures.push_back(benchmark_structure(options.mix, agent, random));
        for (evobrain::VisionRayInputs& ray : inputs[agent].vision) {
            ray = {.red = random.unit_interval(), .green = random.unit_interval(),
                .blue = random.unit_interval(), .proximity = random.unit_interval()};
        }
        inputs[agent].energy = random.unit_interval();
        inputs[agent].damage = random.unit_interval();
    }
    const auto initialization_finished = std::chrono::steady_clock::now();
    const auto evaluation_started = initialization_finished;
    const std::size_t threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    std::uint64_t next_agent_id = static_cast<std::uint64_t>(options.population + 1);
    for (std::uint64_t tick = 0; tick < options.ticks; ++tick) {
        // Replaces stable identities in place to model equal-count birth/death churn.
        for (std::size_t replacement = 0;
             replacement < options.replacements_per_tick; ++replacement) {
            const std::size_t agent = (static_cast<std::size_t>(tick)
                * options.replacements_per_tick + replacement) % options.population;
            agent_ids[agent] = next_agent_id++;
            states[agent] = {};
            parameters[agent][0] = random.uniform(-1.0, 1.0);
        }
        evobrain::evaluate_brain_batch(options.backend,
            {.agent_ids = agent_ids, .parameters = parameters,
                .structures = structures, .states = states,
                .inputs = inputs, .outputs = outputs,
                .cache_identity = parameters.data(),
                .population_changed = tick == 0 || options.replacements_per_tick != 0,
                .state_changed = tick == 0,
                .reset_cache = tick == 0},
            threads);
    }
    const auto evaluation_finished = std::chrono::steady_clock::now();
    const double initialization_ms = std::chrono::duration<double, std::milli>(
        initialization_finished - initialization_started).count();
    const double brain_ms = std::chrono::duration<double, std::milli>(
        evaluation_finished - evaluation_started).count();
    const double evaluations = static_cast<double>(options.population)
        * static_cast<double>(options.ticks);
    double checksum = 0.0;
    for (const evobrain::BrainOutputs& output : outputs) {
        checksum += output.turn + output.move + output.eat;
    }

    std::cout << std::setprecision(17)
              << "backend=" << evobrain::brain_backend_name(options.backend) << '\n'
              << "compiler=" << compiler_name() << '\n'
              << "build_mode=" << build_mode() << '\n'
              << "worker_threads=" << threads << '\n'
              << "seed=" << options.seed << '\n'
              << "population=" << options.population << '\n'
              << "ticks=" << options.ticks << '\n'
              << "replacements_per_tick=" << options.replacements_per_tick << '\n'
              << "mix=" << mix_name(options.mix) << '\n'
              << "initialization_ms=" << initialization_ms << '\n'
              << "brain_ms=" << brain_ms << '\n'
              << "average_ms_per_tick=" << brain_ms / static_cast<double>(options.ticks) << '\n'
              << "evaluations_per_second=" << evaluations / (brain_ms / 1000.0) << '\n'
              << "checksum=" << checksum << '\n';
    return 0;
}
