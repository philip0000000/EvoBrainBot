#include "checkpoint_tests.hpp"

#include "evobrain/brain.hpp"
#include "evobrain/random.hpp"
#include "evobrain/simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

int failure_count = 0;

void expect_true(const bool condition, const std::string_view description)
{
    if (!condition) {
        std::cerr << "FAILED: " << description << '\n';
        ++failure_count;
    }
}

template <typename Actual, typename Expected>
void expect_equal(const Actual& actual, const Expected& expected,
                  const std::string_view description)
{
    if (!(actual == expected)) {
        std::cerr << "FAILED: " << description << '\n';
        ++failure_count;
    }
}

void expect_near(const double actual, const double expected, const double tolerance,
                 const std::string_view description)
{
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "FAILED: " << description << " (expected " << expected
                  << ", got " << actual << ")\n";
        ++failure_count;
    }
}

template <typename Operation>
void expect_invalid_argument(Operation&& operation, const std::string_view description)
{
    try {
        std::forward<Operation>(operation)();
        expect_true(false, description);
    } catch (const std::invalid_argument&) {
    } catch (...) {
        expect_true(false, description);
    }
}

evobrain::SimulationConfig controlled_config(const std::uint64_t seed = 1)
{
    evobrain::SimulationConfig config {.seed = seed};
    config.initial_population = 0;
    config.minimum_population = 0;
    config.target_food_count = 0;
    config.food_population_threshold = 0;
    config.food_bootstrap_population_threshold = 0;
    config.bootstrap_food_count = 0;
    config.food_boost_population_threshold = 0;
    config.boosted_food_count = 0;
    config.world_width = 1.0;
    config.world_height = 1.0;
    config.scattered_rock_probability = 0.0;
    config.cave_region_probability = 0.0;
    config.night_eye_range = 0.25;
    config.day_eye_range = 0.25;
    config.oxygen_refill_per_tick = 0.0;
    config.oxygen_drain_per_tick = 0.0;
    config.initial_energy = 0.5;
    config.living_energy_cost = 0.0;
    config.movement_energy_cost = 0.0;
    config.reproduction_threshold = 10.0;
    config.maximum_movement_per_tick = 0.0;
    config.maximum_turn_per_tick = 0.0;
    config.eat_attempt_energy_cost = 0.0;
    return config;
}

// Creates a stationary test brain with explicit bounded output biases.
evobrain::BrainParameters action_brain(
    const double turn = 0.0, const double move = -1.0, const double eat = -1.0)
{
    evobrain::BrainParameters brain {};
    brain[evobrain::output_bias_offset] = turn;
    brain[evobrain::output_bias_offset + 1] = move;
    brain[evobrain::output_bias_offset + 2] = eat;
    return brain;
}

evobrain::Agent controlled_agent(
    const std::uint64_t id, const evobrain::Vec2 position, const double direction,
    const double energy, const double carnivore_tendency = 0.0)
{
    return {.id = id, .position = position, .direction = direction, .energy = energy,
        .carnivore_tendency = carnivore_tendency, .water_adaptation = 0.5,
        .oxygen = 1.0, .color = {.red = 0.2, .green = 0.3, .blue = 0.4},
        .mutation_rate = evobrain::minimum_mutation_rate, .mutation_strength = 0.1,
        .brain = action_brain()};
}

evobrain::Simulation controlled_simulation(
    const evobrain::SimulationConfig& config, std::vector<evobrain::Agent> agents,
    std::vector<evobrain::Food> food)
{
    evobrain::Simulation initial(config);
    evobrain::SimulationSnapshot snapshot = initial.snapshot();
    snapshot.agents = std::move(agents);
    snapshot.food = std::move(food);
    snapshot.config.target_food_count = std::max(
        snapshot.config.target_food_count,
        static_cast<std::uint64_t>(snapshot.food.size()));
    snapshot.config.bootstrap_food_count = std::max(
        snapshot.config.bootstrap_food_count,
        static_cast<std::uint64_t>(snapshot.food.size()));
    snapshot.config.boosted_food_count = std::max(
        snapshot.config.boosted_food_count,
        static_cast<std::uint64_t>(snapshot.food.size()));
    snapshot.next_agent_id = 1;
    for (const evobrain::Agent& agent : snapshot.agents) {
        snapshot.next_agent_id = std::max(snapshot.next_agent_id, agent.id + 1);
    }
    snapshot.next_food_id = 1;
    for (const evobrain::Food& item : snapshot.food) {
        snapshot.next_food_id = std::max(snapshot.next_food_id, item.id + 1);
    }
    return evobrain::Simulation::from_snapshot(std::move(snapshot));
}

const evobrain::Agent& agent_with_id(const evobrain::Simulation& simulation,
                                    const std::uint64_t id)
{
    const auto found = std::ranges::find(simulation.agents(), id, &evobrain::Agent::id);
    if (found == simulation.agents().end()) throw std::logic_error("test agent missing");
    return *found;
}

void test_random_and_fixed_tick_determinism()
{
    constexpr std::array<std::uint32_t, 3> known {3270867926U, 1795671209U, 1924641435U};
    evobrain::Pcg32 random(42);
    for (const std::uint32_t value : known) expect_equal(random.next(), value, "known PCG value");

    evobrain::Simulation run(evobrain::SimulationConfig {.seed = 9});
    evobrain::Simulation steps(evobrain::SimulationConfig {.seed = 9});
    run.run_for(10);
    for (int index = 0; index < 10; ++index) steps.tick();
    expect_equal(run.snapshot(), steps.snapshot(), "run_for and repeated ticks match");
}

void test_spatial_index_and_thread_determinism()
{
    const evobrain::SimulationConfig config {.seed = 2026};
    evobrain::Simulation serial(config,
        evobrain::SimulationExecutionConfig {.thread_count = 1});
    evobrain::Simulation automatic(config);
    serial.run_for(1'000);
    automatic.run_for(1'000);
    expect_equal(serial.snapshot(), automatic.snapshot(),
        "single-thread and automatic execution produce identical state");
    expect_true(automatic.diagnostics().spatial_columns > 1
            && automatic.diagnostics().spatial_rows > 1,
        "spatial broad phase divides the world into multiple cells");
    expect_true(automatic.diagnostics().vision_candidate_tests
            < automatic.diagnostics().vision_brute_force_tests,
        "spatial broad phase reduces vision candidate tests");
}

void test_brain_topology_and_ranges()
{
    expect_equal(evobrain::brain_parameter_count, std::size_t {515},
        "28-to-16-to-3 brain has 515 feed-forward parameters");
    const evobrain::BrainStructure founder = evobrain::founder_brain_structure();
    expect_equal(std::ranges::count(founder.hidden_active, std::uint8_t {1}),
        std::ptrdiff_t {8}, "founder has eight active hidden neurons");
    expect_true(std::ranges::all_of(
            founder.hidden_active.begin() + evobrain::brain_founder_hidden_count,
            founder.hidden_active.end(), [](const std::uint8_t active) {
                return active == 0;
            }),
        "founder has eight dormant hidden neurons");
    expect_true(std::ranges::none_of(founder.recurrent_enabled,
            [](const std::uint8_t enabled) { return enabled != 0; }),
        "founder starts without recurrent connections");
    evobrain::BrainParameters brain {};
    brain[evobrain::hidden_bias_offset] = 1.0;
    brain[evobrain::hidden_output_weight_offset] = 2.0;
    brain[evobrain::hidden_output_weight_offset + evobrain::brain_hidden_count] = -2.0;
    brain[evobrain::hidden_output_weight_offset + 2 * evobrain::brain_hidden_count] = 2.0;
    const evobrain::BrainOutputs output = evobrain::evaluate_brain(brain, {});
    expect_equal(output.turn, 1.0, "turn clamps to one");
    expect_equal(output.move, 0.0, "Move maps negative clamp to zero");
    expect_equal(output.eat, 1.0, "eat maps positive clamp to one");
    evobrain::BrainStructure general_founder = founder;
    general_founder.founder_fast_path = 0;
    evobrain::BrainState general_state;
    expect_equal(evobrain::evaluate_brain(brain, general_founder, general_state, {}),
        output, "general CPU path agrees with the founder fast path");

    evobrain::BrainStructure memory_structure;
    memory_structure.hidden_active[0] = 1;
    memory_structure.input_hidden_enabled[24] = 1;
    memory_structure.hidden_output_enabled[0] = 1;
    memory_structure.recurrent_enabled[0] = 1;
    memory_structure.recurrent_weights[0] = 1.0;
    evobrain::BrainParameters memory_brain {};
    memory_brain[24] = 1.0;
    memory_brain[evobrain::hidden_output_weight_offset] = 1.0;
    evobrain::BrainState memory_state;
    expect_equal(evobrain::evaluate_brain(memory_brain, memory_structure,
                     memory_state, {.energy = 1.0}).turn,
        1.0, "recurrent brain receives a current input");
    expect_equal(evobrain::evaluate_brain(memory_brain, memory_structure,
                     memory_state, {}).turn,
        1.0, "self-connection retains the previous tick value");

    evobrain::BrainStructure pulse_structure;
    for (std::size_t hidden = 0; hidden < 3; ++hidden) {
        pulse_structure.hidden_active[hidden] = 1;
    }
    pulse_structure.recurrent_enabled[1 * evobrain::brain_hidden_count] = 1;
    pulse_structure.recurrent_weights[1 * evobrain::brain_hidden_count] = 1.0;
    pulse_structure.recurrent_enabled[2 * evobrain::brain_hidden_count + 1] = 1;
    pulse_structure.recurrent_weights[2 * evobrain::brain_hidden_count + 1] = 1.0;
    pulse_structure.recurrent_enabled[2] = 1;
    pulse_structure.recurrent_weights[2] = 1.0;
    pulse_structure.hidden_output_enabled[0] = 1;
    evobrain::BrainParameters pulse_brain {};
    pulse_brain[evobrain::hidden_output_weight_offset] = 1.0;
    evobrain::BrainState pulse_state;
    pulse_state.previous_hidden[0] = 1.0;
    expect_equal(evobrain::evaluate_brain(
                     pulse_brain, pulse_structure, pulse_state, {}).turn,
        0.0, "pulse advances from the first node");
    expect_equal(evobrain::evaluate_brain(
                     pulse_brain, pulse_structure, pulse_state, {}).turn,
        0.0, "pulse advances through the second node");
    expect_equal(evobrain::evaluate_brain(
                     pulse_brain, pulse_structure, pulse_state, {}).turn,
        1.0, "three-node recurrent ring returns a pulse to its output node");
}

// Verifies CUDA matches CPU semantics closely and repeats exactly on one device.
void test_cuda_brain_backend_agreement()
{
    if (!evobrain::brain_backend_available(evobrain::BrainBackendKind::gpu)) return;

    constexpr std::size_t population = 96;
    constexpr std::size_t ticks = 25;
    evobrain::Pcg32 random(5);
    std::vector<evobrain::BrainParameters> parameters(population);
    std::vector<std::uint64_t> agent_ids(population);
    std::vector<evobrain::BrainStructure> structures(population);
    std::vector<evobrain::BrainState> initial_states(population);
    std::vector<evobrain::BrainInputs> inputs(population);
    for (std::size_t agent = 0; agent < population; ++agent) {
        agent_ids[agent] = static_cast<std::uint64_t>(agent + 1);
        for (double& parameter : parameters[agent]) {
            parameter = random.uniform(-0.75, 0.75);
        }
        evobrain::BrainStructure& structure = structures[agent];
        structure = evobrain::founder_brain_structure();
        if (agent % 3 != 0) {
            structure.founder_fast_path = 0;
            const std::size_t active_count = agent % 3 == 1
                ? evobrain::brain_founder_hidden_count
                : evobrain::brain_hidden_count;
            for (std::size_t hidden = 0; hidden < active_count; ++hidden) {
                structure.hidden_active[hidden] = 1;
                for (std::size_t input = 0; input < evobrain::brain_input_count; ++input) {
                    const std::size_t connection = hidden * evobrain::brain_input_count + input;
                    structure.input_hidden_enabled[connection]
                        = (connection + agent) % 5 != 0 ? 1 : 0;
                }
                for (std::size_t output = 0; output < evobrain::brain_output_count; ++output) {
                    structure.hidden_output_enabled[
                        output * evobrain::brain_hidden_count + hidden]
                        = (output + hidden + agent) % 4 != 0 ? 1 : 0;
                }
                initial_states[agent].previous_hidden[hidden]
                    = random.uniform(-0.5, 0.5);
                initial_states[agent].next_hidden[hidden]
                    = initial_states[agent].previous_hidden[hidden];
            }
            for (std::size_t target = 0; target < active_count; ++target) {
                for (std::size_t source = 0; source < active_count; ++source) {
                    const std::size_t connection
                        = target * evobrain::brain_hidden_count + source;
                    if ((target + source + agent) % 3 == 0) {
                        structure.recurrent_enabled[connection] = 1;
                        structure.recurrent_weights[connection]
                            = random.uniform(-0.5, 0.5);
                    }
                }
            }
        }
        for (evobrain::VisionRayInputs& ray : inputs[agent].vision) {
            ray = {.red = random.unit_interval(), .green = random.unit_interval(),
                .blue = random.unit_interval(), .proximity = random.unit_interval()};
        }
        inputs[agent].energy = random.unit_interval();
        inputs[agent].damage = random.unit_interval();
    }

    std::vector<evobrain::BrainState> cpu_states = initial_states;
    std::vector<evobrain::BrainState> gpu_states = initial_states;
    std::vector<evobrain::BrainOutputs> cpu_outputs(population);
    std::vector<evobrain::BrainOutputs> gpu_outputs(population);
    for (std::size_t tick = 0; tick < ticks; ++tick) {
        evobrain::evaluate_brain_batch(evobrain::BrainBackendKind::cpu,
            {.agent_ids = agent_ids, .parameters = parameters,
                .structures = structures, .states = cpu_states,
                .inputs = inputs, .outputs = cpu_outputs,
                .cache_identity = parameters.data(),
                .population_changed = tick == 0, .state_changed = tick == 0,
                .reset_cache = tick == 0},
            1);
        evobrain::evaluate_brain_batch(evobrain::BrainBackendKind::gpu,
            {.agent_ids = agent_ids, .parameters = parameters,
                .structures = structures, .states = gpu_states,
                .inputs = inputs, .outputs = gpu_outputs,
                .cache_identity = parameters.data(),
                .population_changed = tick == 0, .state_changed = tick == 0,
                .reset_cache = tick == 0},
            1);
        for (std::size_t agent = 0; agent < population; ++agent) {
            expect_near(gpu_outputs[agent].turn, cpu_outputs[agent].turn, 1e-12,
                "CUDA turn output agrees with CPU");
            expect_near(gpu_outputs[agent].move, cpu_outputs[agent].move, 1e-12,
                "CUDA move output agrees with CPU");
            expect_near(gpu_outputs[agent].eat, cpu_outputs[agent].eat, 1e-12,
                "CUDA eat output agrees with CPU");
            for (std::size_t hidden = 0; hidden < evobrain::brain_hidden_count; ++hidden) {
                expect_near(gpu_states[agent].previous_hidden[hidden],
                    cpu_states[agent].previous_hidden[hidden], 1e-12,
                    "CUDA recurrent state agrees with CPU");
            }
        }
    }

    const std::vector<evobrain::BrainState> first_gpu_states = gpu_states;
    const std::vector<evobrain::BrainOutputs> first_gpu_outputs = gpu_outputs;
    gpu_states = initial_states;
    for (std::size_t tick = 0; tick < ticks; ++tick) {
        evobrain::evaluate_brain_batch(evobrain::BrainBackendKind::gpu,
            {.agent_ids = agent_ids, .parameters = parameters,
                .structures = structures, .states = gpu_states,
                .inputs = inputs, .outputs = gpu_outputs,
                .cache_identity = parameters.data(),
                .population_changed = tick == 0, .state_changed = tick == 0,
                .reset_cache = tick == 0},
            1);
    }
    expect_equal(gpu_outputs, first_gpu_outputs,
        "repeated CUDA evaluation produces identical outputs");
    expect_equal(gpu_states, first_gpu_states,
        "repeated CUDA evaluation produces identical recurrent state");

    // One death, host reordering, and one birth must update only identity-mapped slots.
    constexpr std::size_t removed = 7;
    agent_ids.erase(agent_ids.begin() + removed);
    parameters.erase(parameters.begin() + removed);
    structures.erase(structures.begin() + removed);
    cpu_states.erase(cpu_states.begin() + removed);
    gpu_states.erase(gpu_states.begin() + removed);
    inputs.erase(inputs.begin() + removed);
    cpu_outputs.erase(cpu_outputs.begin() + removed);
    gpu_outputs.erase(gpu_outputs.begin() + removed);
    constexpr std::size_t reordered_a = 2;
    constexpr std::size_t reordered_b = 41;
    std::swap(agent_ids[reordered_a], agent_ids[reordered_b]);
    std::swap(parameters[reordered_a], parameters[reordered_b]);
    std::swap(structures[reordered_a], structures[reordered_b]);
    std::swap(cpu_states[reordered_a], cpu_states[reordered_b]);
    std::swap(gpu_states[reordered_a], gpu_states[reordered_b]);
    std::swap(inputs[reordered_a], inputs[reordered_b]);
    std::swap(cpu_outputs[reordered_a], cpu_outputs[reordered_b]);
    std::swap(gpu_outputs[reordered_a], gpu_outputs[reordered_b]);

    evobrain::BrainParameters newborn_parameters {};
    for (double& parameter : newborn_parameters) parameter = random.uniform(-0.75, 0.75);
    parameters.push_back(newborn_parameters);
    structures.push_back(evobrain::founder_brain_structure());
    cpu_states.emplace_back();
    gpu_states.emplace_back();
    inputs.push_back(inputs.front());
    cpu_outputs.emplace_back();
    gpu_outputs.emplace_back();
    agent_ids.push_back(10'000);

    evobrain::evaluate_brain_batch(evobrain::BrainBackendKind::cpu,
        {.agent_ids = agent_ids, .parameters = parameters,
            .structures = structures, .states = cpu_states,
            .inputs = inputs, .outputs = cpu_outputs,
            .cache_identity = parameters.data(), .population_changed = true,
            .state_changed = false, .reset_cache = false},
        1);
    evobrain::evaluate_brain_batch(evobrain::BrainBackendKind::gpu,
        {.agent_ids = agent_ids, .parameters = parameters,
            .structures = structures, .states = gpu_states,
            .inputs = inputs, .outputs = gpu_outputs,
            .cache_identity = parameters.data(), .population_changed = true,
            .state_changed = false, .reset_cache = false},
        1);
    for (std::size_t agent = 0; agent < agent_ids.size(); ++agent) {
        expect_near(gpu_outputs[agent].turn, cpu_outputs[agent].turn, 1e-12,
            "incremental CUDA slot turn output agrees with CPU");
        expect_near(gpu_outputs[agent].move, cpu_outputs[agent].move, 1e-12,
            "incremental CUDA slot move output agrees with CPU");
        expect_near(gpu_outputs[agent].eat, cpu_outputs[agent].eat, 1e-12,
            "incremental CUDA slot eat output agrees with CPU");
        for (std::size_t hidden = 0; hidden < evobrain::brain_hidden_count; ++hidden) {
            expect_near(gpu_states[agent].previous_hidden[hidden],
                cpu_states[agent].previous_hidden[hidden], 1e-12,
                "incremental CUDA slot state agrees with CPU");
        }
    }
}

// Verifies bounded full simulations remain numerically close across backends.
void test_simulation_backend_agreement()
{
    if (!evobrain::brain_backend_available(evobrain::BrainBackendKind::gpu)) return;

    const evobrain::SimulationConfig config {.seed = 5};
    evobrain::Simulation cpu(config,
        evobrain::SimulationExecutionConfig {
            .thread_count = 1, .brain_backend = evobrain::BrainBackendKind::cpu});
    evobrain::Simulation gpu(config,
        evobrain::SimulationExecutionConfig {
            .thread_count = 1, .brain_backend = evobrain::BrainBackendKind::gpu});
    cpu.run_for(100);
    gpu.run_for(100);
    expect_equal(gpu.stats(), cpu.stats(),
        "CPU and CUDA simulations retain identical discrete statistics");

    const evobrain::SimulationSnapshot cpu_snapshot = cpu.snapshot();
    const evobrain::SimulationSnapshot gpu_snapshot = gpu.snapshot();
    expect_equal(gpu_snapshot.random_state, cpu_snapshot.random_state,
        "CPU and CUDA simulations retain identical random state");
    expect_equal(gpu_snapshot.agents.size(), cpu_snapshot.agents.size(),
        "CPU and CUDA simulations retain identical agent count");
    expect_equal(gpu_snapshot.food.size(), cpu_snapshot.food.size(),
        "CPU and CUDA simulations retain identical food count");
    for (std::size_t index = 0; index < cpu_snapshot.agents.size(); ++index) {
        const evobrain::Agent& cpu_agent = cpu_snapshot.agents[index];
        const evobrain::Agent& gpu_agent = gpu_snapshot.agents[index];
        expect_equal(gpu_agent.id, cpu_agent.id,
            "CPU and CUDA simulations retain agent identity");
        expect_equal(gpu_agent.age, cpu_agent.age,
            "CPU and CUDA simulations retain agent age");
        expect_equal(gpu_agent.brain, cpu_agent.brain,
            "CPU and CUDA simulations retain agent genomes");
        expect_equal(gpu_agent.brain_structure, cpu_agent.brain_structure,
            "CPU and CUDA simulations retain agent topology");
        expect_near(gpu_agent.position.x, cpu_agent.position.x, 1e-12,
            "CUDA agent x position remains close to CPU");
        expect_near(gpu_agent.position.y, cpu_agent.position.y, 1e-12,
            "CUDA agent y position remains close to CPU");
        expect_near(gpu_agent.direction, cpu_agent.direction, 1e-12,
            "CUDA agent direction remains close to CPU");
        expect_near(gpu_agent.energy, cpu_agent.energy, 1e-12,
            "CUDA agent energy remains close to CPU");
    }
    for (std::size_t index = 0; index < cpu_snapshot.food.size(); ++index) {
        expect_equal(gpu_snapshot.food[index].id, cpu_snapshot.food[index].id,
            "CPU and CUDA simulations retain food identity");
        expect_near(gpu_snapshot.food[index].energy, cpu_snapshot.food[index].energy,
            1e-12, "CUDA food energy remains close to CPU");
    }
}

void test_configuration_and_founders()
{
    evobrain::SimulationConfig invalid {.seed = 1};
    invalid.night_eye_range = 0.0;
    expect_invalid_argument([&] { evobrain::Simulation simulation(invalid); },
        "zero eye range is rejected");

    evobrain::Simulation first(evobrain::SimulationConfig {.seed = 123});
    evobrain::Simulation second(evobrain::SimulationConfig {.seed = 123});
    expect_equal(first.snapshot(), second.snapshot(), "founders are seed reproducible");
    expect_true(first.config().world_width == 5.0 && first.config().world_height == 5.0,
        "default world uses the configured 5 by 5 dimensions");
    expect_true(first.config().food_bootstrap_population_threshold == 50
            && first.config().bootstrap_food_count == 4'000
            && first.config().food_boost_population_threshold == 200
            && first.config().boosted_food_count == 1'000
            && first.config().food_population_threshold == 500
            && first.config().target_food_count == 500
            && first.config().maximum_new_food_per_tick == 5,
        "default population bands configure bootstrap, boosted, normal, then no spawning");
    expect_true(first.config().minimum_fertility > 0.0
            && first.config().terrain_cell_size == 0.04,
        "default terrain keeps food possible and rock cells two agent diameters wide");
    expect_equal(first.config().founder_mutation_rate_minimum,
        evobrain::minimum_mutation_rate,
        "founders use the one-in-seventy-five absolute mutation-rate floor");
    expect_equal(first.config().founder_mutation_rate_maximum, 1.0 / 20.0,
        "founder mutation rates extend to one in twenty");
    expect_equal(evobrain::minimum_mutation_strength, 1.0 / 75.0,
        "mutation strength uses the configured one-seventy-fifth absolute floor");
    expect_equal(first.food().size(), std::size_t {4'000},
        "new simulation starts with its complete initial food supply");
    expect_true(std::ranges::any_of(first.food(), [](const evobrain::Food& item) {
        return item.position.x > 1.0 || item.position.y > 1.0;
    }), "random food placement uses space beyond the former unit world");
    for (const evobrain::Agent& agent : first.agents()) {
        expect_true(agent.trait_mutation_rate_percent >= 20
                && agent.trait_mutation_rate_percent <= 100
                && agent.trait_mutation_rate_percent % 5 == 0,
            "founders choose rates on the inclusive five-percentage-point grid");
        expect_true(agent.color.red >= 0.0 && agent.color.red <= 1.0
                && agent.color.green >= 0.0 && agent.color.green <= 1.0
                && agent.color.blue >= 0.0 && agent.color.blue <= 1.0,
            "founder RGB is normalized");
        expect_true(agent.mutation_rate >= first.config().founder_mutation_rate_minimum
                && agent.mutation_rate <= first.config().founder_mutation_rate_maximum,
            "founder mutation rate uses founder range");
        expect_true(agent.mutation_strength >= first.config().founder_mutation_strength_minimum
                && agent.mutation_strength <= first.config().founder_mutation_strength_maximum,
            "founder mutation strength uses founder range");
        expect_true(agent.carnivore_tendency == 0.0,
            "ordinary random founders begin as herbivores");
        expect_equal(agent.water_adaptation, 0.5,
            "initial founders always begin perfectly amphibious");
        expect_near(agent.oxygen, first.config().maximum_oxygen, 0.0,
            "ordinary random founders begin with full oxygen");
        expect_true(agent.position.x >= 0.0 && agent.position.x < first.config().world_width
                && agent.position.y >= 0.0
                && agent.position.y < first.config().world_height,
            "founders use the complete configured world bounds");
    }
    expect_true(!first.biomes().empty() && !first.terrain().empty(),
        "new simulations expose deterministic biome terrain");

    evobrain::SimulationSnapshot floored_snapshot = first.snapshot();
    floored_snapshot.agents.front().mutation_rate = 0.0;
    floored_snapshot.agents.front().mutation_strength = 0.0;
    const evobrain::Simulation floored =
        evobrain::Simulation::from_snapshot(std::move(floored_snapshot));
    expect_equal(floored.agents().front().mutation_rate,
        evobrain::minimum_mutation_rate,
        "restored zero mutation rate is clamped to the positive floor");
    expect_equal(floored.agents().front().mutation_strength,
        evobrain::minimum_mutation_strength,
        "restored zero mutation strength is clamped to the positive floor");
}

void test_literal_ray_first_hit_and_wrap()
{
    evobrain::SimulationConfig config = controlled_config();
    config.world_width = 2.5;
    config.world_height = 2.5;
    config.maximum_turn_per_tick = 0.25;
    evobrain::Agent observer = controlled_agent(1, {.x = 2.49, .y = 0.5}, 0.0, 1.0);
    // Route the left eye's forward red channel through hidden zero to Turn.
    observer.brain = action_brain();
    observer.brain[8] = 1.0;
    observer.brain[evobrain::hidden_output_weight_offset] = 1.0;
    evobrain::Agent near = controlled_agent(2, {.x = 0.03, .y = 0.505}, 0.0, 1.0);
    near.color = {.red = 1.0, .green = 0.0, .blue = 0.0};
    evobrain::Simulation visible = controlled_simulation(config, {observer, near}, {});
    visible.tick();
    expect_near(agent_with_id(visible, 1).direction, 0.25, 1e-12,
        "finite ray sees a circle across toroidal boundary");

    near.position.x = 0.30;
    evobrain::Simulation out_of_range = controlled_simulation(config, {observer, near}, {});
    out_of_range.tick();
    expect_near(agent_with_id(out_of_range, 1).direction, 0.0, 1e-12,
        "ray does not autocomplete beyond configured range");
}

void test_configured_world_movement_wrap()
{
    evobrain::SimulationConfig config = controlled_config();
    config.world_width = 2.5;
    config.world_height = 2.5;
    config.maximum_movement_per_tick = 0.01;
    evobrain::Agent mover = controlled_agent(1, {.x = 2.495, .y = 1.25}, 0.0, 1.0);
    mover.brain = action_brain(0.0, 1.0, -1.0);
    evobrain::Simulation simulation = controlled_simulation(config, {mover}, {});
    simulation.tick();
    expect_near(simulation.agents().front().position.x, 0.0, 1e-12,
        "perfect amphibian moves at half speed across the configured world boundary");
}

void test_gradual_diet_specific_eating()
{
    evobrain::SimulationConfig config = controlled_config();
    evobrain::Agent herbivore = controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 0.25);
    herbivore.brain = action_brain(0.0, -1.0, 1.0);
    evobrain::Simulation simulation = controlled_simulation(config, {herbivore},
        {{.id = 1, .position = {.x = 0.51, .y = 0.5}, .energy = 0.12}});
    simulation.tick();
    expect_near(simulation.agents().front().energy, 0.30, 1e-12,
        "herbivore receives one configured bite");
    expect_near(simulation.food().front().energy, 0.07, 1e-12,
        "food loses energy gradually");

    evobrain::Agent carnivore = herbivore;
    carnivore.carnivore_tendency = 1.0;
    evobrain::Simulation wrong_food = controlled_simulation(config, {carnivore},
        {{.id = 1, .position = {.x = 0.51, .y = 0.5}, .energy = 0.12}});
    wrong_food.tick();
    expect_near(wrong_food.agents().front().energy, 0.25, 1e-12,
        "carnivore gains nothing from plant food");
    expect_near(wrong_food.food().front().energy, 0.07, 1e-12,
        "strict carnivore still consumes a bitten plant without gaining energy");

    evobrain::Agent blocking_agent = controlled_agent(
        2, {.x = 0.51, .y = 0.5}, 0.0, 0.5, 1.0);
    evobrain::Simulation blocked = controlled_simulation(config,
        {herbivore, blocking_agent},
        {{.id = 1, .position = {.x = 0.51, .y = 0.5}, .energy = 0.12}});
    blocked.tick();
    expect_near(agent_with_id(blocked, 1).energy, 0.25, 1e-12,
        "topmost wrong-diet agent blocks food below it");
    expect_near(blocked.food().front().energy, 0.12, 1e-12,
        "blocked food is not searched as an alternate target");
}

void test_mutual_and_proportional_agent_bites()
{
    evobrain::SimulationConfig config = controlled_config();
    evobrain::Agent left = controlled_agent(1, {.x = 0.50, .y = 0.5}, 0.0, 0.20,
        1.0);
    evobrain::Agent right = controlled_agent(2, {.x = 0.518, .y = 0.5},
        std::numbers::pi_v<double>, 0.20, 1.0);
    left.brain = right.brain = action_brain(0.0, -1.0, 1.0);
    evobrain::Simulation mutual = controlled_simulation(config, {left, right}, {});
    mutual.tick();
    expect_near(agent_with_id(mutual, 1).energy, 0.20, 1e-12,
        "first mutual biter loses and gains simultaneously");
    expect_near(agent_with_id(mutual, 2).energy, 0.20, 1e-12,
        "second mutual biter loses and gains simultaneously");
    expect_near(agent_with_id(mutual, 1).prior_bite_damage, 0.05, 1e-12,
        "bite damage is stored for next tick");

    evobrain::Agent attacker1 = left;
    evobrain::Agent attacker2 = left;
    attacker2.id = 2;
    evobrain::Agent target = controlled_agent(3, {.x = 0.51, .y = 0.5}, 0.0, 0.05,
        0.0);
    target.brain = action_brain();
    evobrain::Simulation shared = controlled_simulation(config,
        {attacker1, attacker2, target}, {});
    shared.tick();
    expect_true(agent_with_id(shared, 1).energy == 0.225
            && agent_with_id(shared, 2).energy == 0.225,
        "insufficient target energy is allocated proportionally");
    expect_true(std::ranges::find(shared.agents(), std::uint64_t {3}, &evobrain::Agent::id)
            == shared.agents().end(), "depleted target is removed");
    expect_equal(shared.stats().agents_eaten, std::uint64_t {1},
        "agent killed through eating is counted");
}

void test_inherited_mutation_and_reproduction_geometry()
{
    evobrain::SimulationConfig config = controlled_config(77);
    config.reproduction_threshold = 1.0;
    evobrain::Agent parent = controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 1.2,
        1.0);
    parent.color = {.red = 0.5, .green = 0.5, .blue = 0.5};
    parent.mutation_rate = 1.0;
    parent.mutation_strength = 0.1;
    evobrain::Simulation simulation = controlled_simulation(config, {parent}, {});
    simulation.tick();
    const evobrain::Agent& child = agent_with_id(simulation, 2);
    expect_true(child.carnivore_tendency >= 0.0 && child.carnivore_tendency <= 1.0,
        "diet tendency is inherited and mutates within its normalized range");
    expect_near(child.oxygen, config.maximum_oxygen, 0.0,
        "new child starts with full oxygen");
    expect_near(child.position.x, 0.48, 1e-12, "child is placed two radii behind parent");
    expect_near(child.direction, std::numbers::pi_v<double>, 1e-12,
        "child faces opposite parent");
    expect_equal(child.age, std::uint64_t {0}, "new child has age zero");
    expect_near(child.prior_bite_damage, 0.0, 0.0, "new child damage starts clear");
    expect_true(std::ranges::all_of(child.brain_state.previous_hidden,
            [](const double value) { return value == 0.0; }),
        "new child does not inherit recurrent memory");
    expect_equal(std::ranges::count(child.brain_structure.hidden_active,
                     std::uint8_t {1}),
        std::ptrdiff_t {16}, "full-rate mutation activates every dormant neuron");
    for (std::size_t hidden = evobrain::brain_founder_hidden_count;
         hidden < evobrain::brain_hidden_count; ++hidden) {
        const bool has_incoming = std::ranges::any_of(
            child.brain_structure.input_hidden_enabled.begin()
                + static_cast<std::ptrdiff_t>(hidden * evobrain::brain_input_count),
            child.brain_structure.input_hidden_enabled.begin()
                + static_cast<std::ptrdiff_t>((hidden + 1) * evobrain::brain_input_count),
            [](const std::uint8_t enabled) { return enabled != 0; });
        bool has_outgoing = false;
        for (std::size_t output = 0; output < evobrain::brain_output_count; ++output) {
            has_outgoing = has_outgoing
                || child.brain_structure.hidden_output_enabled[
                    output * evobrain::brain_hidden_count + hidden]
                    != 0;
        }
        expect_true(has_incoming && has_outgoing,
            "activated dormant neuron receives an incoming and outgoing connection");
    }
    expect_true(std::abs(child.color.red - parent.color.red) <= 0.15 + 1e-12
            && std::abs(child.mutation_rate - parent.mutation_rate) <= 0.002
            && std::abs(child.mutation_strength - parent.mutation_strength) <= 0.01,
        "trait step and brain category scales bound inherited DNA changes");
}

// Full-rate trials reselect ecological values and independently flip each RGB channel.
void test_independent_trait_mutation()
{
    auto config = controlled_config(771);
    config.reproduction_threshold = 1.0;
    bool diets_seen[5] {};
    bool lungs_seen[5] {};
    bool positive_color = false;
    bool negative_color = false;
    bool rate_stayed = false;
    bool rate_decreased = false;
    for (std::uint64_t seed = 1; seed <= 64; ++seed) {
        config.seed = seed;
        auto parent = controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 1.2, 0.0);
        parent.water_adaptation = 1.0;
        parent.color = {.red = 0.0, .green = 0.95, .blue = 0.5};
        parent.trait_mutation_rate_percent = 100;
        auto simulation = controlled_simulation(config, {parent}, {});
        simulation.tick();
        const auto& child = agent_with_id(simulation, 2);
        const auto diet_index = static_cast<int>(child.carnivore_tendency * 4.0);
        const auto lung_index = static_cast<int>(child.water_adaptation * 4.0);
        expect_true(diet_index >= 0 && diet_index <= 4
                && child.carnivore_tendency == diet_index * 0.25,
            "diet selection remains on the five-value grid");
        expect_true(lung_index >= 0 && lung_index <= 4
                && child.water_adaptation == lung_index * 0.25,
            "lung selection remains on the five-value grid");
        if (diet_index >= 0 && diet_index <= 4) diets_seen[diet_index] = true;
        if (lung_index >= 0 && lung_index <= 4) lungs_seen[lung_index] = true;
        expect_near(child.color.red, 0.15, 1e-12, "both signs reflect inward from zero");
        expect_true(std::abs(child.color.green - 0.8) < 1e-12
                || std::abs(child.color.green - 0.9) < 1e-12,
            "upper RGB boundary reflects the positive fixed step");
        expect_near(std::abs(child.color.blue - 0.5), 0.15, 1e-12,
            "every RGB trial uses the parent's 100 percent even when child rate falls");
        positive_color |= child.color.blue > 0.5;
        negative_color |= child.color.blue < 0.5;
        expect_true(child.trait_mutation_rate_percent == 95 || child.trait_mutation_rate_percent == 100,
            "upper rate boundary clamps a fair five-point step");
        rate_stayed |= child.trait_mutation_rate_percent == 100;
        rate_decreased |= child.trait_mutation_rate_percent == 95;
        expect_equal(agent_with_id(simulation, 1).trait_mutation_rate_percent,
            std::uint8_t {100}, "parent rate remains unchanged");
    }
    expect_true(std::ranges::all_of(diets_seen, [](bool seen) { return seen; })
            && std::ranges::all_of(lungs_seen, [](bool seen) { return seen; }),
        "seeded births cover all five values including unchanged and opposite endpoints");
    expect_true(positive_color && negative_color && rate_stayed && rate_decreased,
        "seeded births exercise both coin-flip outcomes");

    bool unchanged_color = false;
    bool changed_color = false;
    for (std::uint64_t seed = 1; seed <= 32; ++seed) {
        config.seed = seed;
        auto parent = controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 1.2, 0.5);
        parent.trait_mutation_rate_percent = 20;
        auto simulation = controlled_simulation(config, {parent}, {});
        simulation.tick();
        const auto& child = agent_with_id(simulation, 2);
        expect_true(child.trait_mutation_rate_percent == 20 || child.trait_mutation_rate_percent == 25,
            "lower rate boundary clamps the inherited five-point step");
        unchanged_color |= child.color.red == parent.color.red;
        changed_color |= child.color.red != parent.color.red;
    }
    expect_true(unchanged_color && changed_color,
        "partial mutation probability permits both inheritance and color mutation");

    auto invalid = evobrain::Simulation(config).snapshot();
    invalid.agents = {controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 0.5)};
    invalid.next_agent_id = 2;
    invalid.agents.front().trait_mutation_rate_percent = 27;
    expect_invalid_argument([&] { (void)evobrain::Simulation::from_snapshot(invalid); },
        "off-grid trait rates are rejected on restore");
    invalid.agents.front().trait_mutation_rate_percent = 20;
    invalid.agents.front().water_adaptation = 0.1;
    expect_invalid_argument([&] { (void)evobrain::Simulation::from_snapshot(invalid); },
        "off-grid lung values are rejected on restore");
}

void test_population_food_boost_and_natural_excess_reduction()
{
    evobrain::SimulationConfig low = controlled_config();
    low.initial_population = 39;
    low.target_food_count = 3;
    low.food_population_threshold = 100;
    low.food_bootstrap_population_threshold = 40;
    low.bootstrap_food_count = 9;
    low.food_boost_population_threshold = 50;
    low.boosted_food_count = 6;
    evobrain::Simulation low_population(low);
    expect_equal(low_population.food().size(), std::size_t {9},
        "population below bootstrap threshold starts with universal food count");

    evobrain::SimulationConfig boosted = low;
    boosted.initial_population = 40;
    evobrain::Simulation boosted_population(boosted);
    expect_equal(boosted_population.food().size(), std::size_t {6},
        "population at bootstrap threshold resumes the ordinary boosted band");

    evobrain::SimulationConfig normal = low;
    normal.initial_population = 50;
    evobrain::Simulation normal_population(normal);
    expect_equal(normal_population.food().size(), std::size_t {3},
        "population at boost threshold uses normal food count");

    evobrain::SimulationConfig crowded = low;
    crowded.initial_population = 3;
    crowded.food_bootstrap_population_threshold = 0;
    crowded.food_boost_population_threshold = 2;
    crowded.food_population_threshold = 3;
    evobrain::Simulation crowded_population(crowded);
    evobrain::SimulationSnapshot crowded_snapshot = crowded_population.snapshot();
    for (evobrain::Agent& agent : crowded_snapshot.agents) {
        agent.brain = action_brain();
    }
    crowded_snapshot.food.clear();
    evobrain::Simulation no_replenishment =
        evobrain::Simulation::from_snapshot(std::move(crowded_snapshot));
    no_replenishment.tick();
    expect_true(no_replenishment.food().empty(),
        "population at replenishment threshold receives no replacement food");

    evobrain::SimulationSnapshot excess = normal_population.snapshot();
    for (evobrain::Agent& agent : excess.agents) {
        agent.brain = action_brain();
    }
    excess.food = {
        {.id = 1, .position = {.x = 0.1, .y = 0.1}, .energy = 0.25},
        {.id = 2, .position = {.x = 0.2, .y = 0.1}, .energy = 0.25},
        {.id = 3, .position = {.x = 0.3, .y = 0.1}, .energy = 0.25},
        {.id = 4, .position = {.x = 0.4, .y = 0.1}, .energy = 0.25},
        {.id = 5, .position = {.x = 0.5, .y = 0.1}, .energy = 0.25},
        {.id = 6, .position = {.x = 0.6, .y = 0.1}, .energy = 0.25},
    };
    excess.next_food_id = 7;
    evobrain::Simulation recovered =
        evobrain::Simulation::from_snapshot(std::move(excess));
    recovered.tick();
    expect_equal(recovered.food().size(), std::size_t {6},
        "food above normal target is not deleted after population recovery");
}

void test_global_food_regrowth_pulse()
{
    evobrain::SimulationConfig config = controlled_config();
    config.target_food_count = 2;
    config.food_population_threshold = 100;
    config.food_boost_population_threshold = 0;
    config.boosted_food_count = 2;
    config.food_regrowth_interval_ticks = 100;
    config.food_regrowth_amount = 0.025;

    evobrain::Agent idle = controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 1.0);
    evobrain::Simulation before_pulse = controlled_simulation(config, {idle},
        {
            {.id = 1, .position = {.x = 0.8, .y = 0.8}, .energy = 0.20},
            {.id = 2, .position = {.x = 0.7, .y = 0.8}, .energy = 0.25},
        });
    evobrain::SimulationSnapshot pulse_snapshot = before_pulse.snapshot();
    pulse_snapshot.current_tick = 99;
    evobrain::Simulation pulse =
        evobrain::Simulation::from_snapshot(std::move(pulse_snapshot));
    pulse.tick();
    expect_near(pulse.food().front().energy, 0.225, 1e-12,
        "surviving food gains energy on global tick 100 pulse");
    expect_near(pulse.food()[1].energy, 0.25, 1e-12,
        "food at maximum energy remains clamped on a regrowth pulse");

    evobrain::Agent eater = idle;
    eater.brain = action_brain(0.0, -1.0, 1.0);
    evobrain::Simulation bitten_before_pulse = controlled_simulation(config, {eater},
        {
            {.id = 1, .position = {.x = 0.51, .y = 0.5}, .energy = 0.25},
            {.id = 2, .position = {.x = 0.8, .y = 0.8}, .energy = 0.25},
        });
    evobrain::SimulationSnapshot bitten_snapshot = bitten_before_pulse.snapshot();
    bitten_snapshot.current_tick = 99;
    evobrain::Simulation bitten =
        evobrain::Simulation::from_snapshot(std::move(bitten_snapshot));
    bitten.tick();
    expect_near(bitten.food().front().energy, 0.225, 1e-12,
        "food bitten on a pulse tick regrows after the bite");
}

void test_population_floor_and_food_replenishment()
{
    evobrain::SimulationConfig config = controlled_config();
    config.initial_population = 2;
    config.minimum_population = 2;
    config.target_food_count = 3;
    config.food_population_threshold = 3;
    config.food_boost_population_threshold = 0;
    config.boosted_food_count = 3;
    config.maximum_new_food_per_tick = 2;
    config.living_energy_cost = 100.0;
    evobrain::Simulation simulation(config);
    evobrain::SimulationSnapshot empty_food = simulation.snapshot();
    empty_food.food.clear();
    empty_food.next_food_id = 1;
    simulation = evobrain::Simulation::from_snapshot(std::move(empty_food));
    simulation.tick();
    expect_equal(simulation.agents().size(), std::size_t {2}, "population floor restores founders");
    expect_equal(simulation.stats().introduced_agents, std::uint64_t {2},
        "introduced founders are counted");
    expect_true(std::ranges::all_of(simulation.agents(), [](const evobrain::Agent& agent) {
        return agent.carnivore_tendency == 0.0 && agent.water_adaptation == 0.5;
    }), "population-floor founders begin as amphibious herbivores");
    expect_equal(simulation.food().size(), std::size_t {2},
        "whole food spawning obeys its per-tick limit");
    simulation.tick();
    expect_equal(simulation.food().size(), std::size_t {3},
        "gradual spawning stops exactly at the population-band ceiling");
}

// Counts major connected media (at least 5% of the map), excluding wetland pools.
// Lifted grid coordinates detect a world-spanning cycle even across wrapped edges.
std::size_t medium_components(const evobrain::Simulation& simulation,
    const evobrain::TerrainMedium medium, bool* const wraps_world = nullptr)
{
    const auto terrain = simulation.terrain();
    const auto columns = static_cast<std::size_t>(std::round(simulation.config().world_width / terrain[0].width));
    const auto rows = terrain.size() / columns;
    std::vector<bool> visited(terrain.size(), false);
    std::vector<int> lifted_x(terrain.size(), 0);
    std::vector<int> lifted_y(terrain.size(), 0);
    if (wraps_world != nullptr) *wraps_world = false;
    std::size_t components = 0;
    for (std::size_t start = 0; start < terrain.size(); ++start) {
        if (visited[start] || terrain[start].medium != medium) continue;
        std::size_t component_size = 0;
        bool winding = false;
        visited[start] = true;
        std::vector<std::size_t> pending {start};
        while (!pending.empty()) {
            const auto current = pending.back();
            pending.pop_back();
            ++component_size;
            const auto x = current % columns;
            const auto y = current / columns;
            const std::array neighbors {y * columns + (x + 1) % columns,
                y * columns + (x + columns - 1) % columns,
                ((y + 1) % rows) * columns + x,
                ((y + rows - 1) % rows) * columns + x};
            const std::array<int, 4> step_x {1, -1, 0, 0};
            const std::array<int, 4> step_y {0, 0, 1, -1};
            for (std::size_t direction = 0; direction < neighbors.size(); ++direction) {
                const auto neighbor = neighbors[direction];
                if (visited[neighbor] && terrain[neighbor].medium == medium) {
                    winding |= lifted_x[neighbor] != lifted_x[current] + step_x[direction]
                        || lifted_y[neighbor] != lifted_y[current] + step_y[direction];
                }
                if (!visited[neighbor] && terrain[neighbor].medium == medium) {
                    visited[neighbor] = true;
                    lifted_x[neighbor] = lifted_x[current] + step_x[direction];
                    lifted_y[neighbor] = lifted_y[current] + step_y[direction];
                    pending.push_back(neighbor);
                }
            }
        }
        if (component_size >= terrain.size() / 20) {
            ++components;
            if (wraps_world != nullptr) *wraps_world |= winding;
        }
    }
    return components;
}

// Presets preserve food density and geography counts while limiting population scaling;
// rare sparse regions never leak
// into the ordinary fertility range, and large deserts include adjacent regions.
void test_world_sizes_and_geography()
{
    const auto small = evobrain::make_world_config(1);
    for (const auto size : {evobrain::WorldSize::small_world, evobrain::WorldSize::medium, evobrain::WorldSize::large}) {
        auto config = evobrain::make_world_config(1, size);
        const auto length = std::uint64_t {1} << static_cast<unsigned>(size);
        const auto area = length * length;
        expect_true(config.world_width == 5.0 * length && config.world_height == 5.0 * length,
            "preset dimensions double on each axis");
        expect_true(config.initial_population == 30 * length && config.minimum_population == 30 * length
                && config.bootstrap_food_count == 4000 * area && config.boosted_food_count == 1000 * area
                && config.target_food_count == 500 * area && config.maximum_new_food_per_tick == 5 * area
                && config.food_bootstrap_population_threshold == 50 * length
                && config.food_boost_population_threshold == 200 * length
                && config.food_population_threshold == 500 * length,
            "population bands scale by length and food supply by area exactly once");
        expect_true(config.food_energy == small.food_energy && config.agent_radius == small.agent_radius
                && config.maximum_movement_per_tick == small.maximum_movement_per_tick
                && config.day_night_cycle_ticks == small.day_night_cycle_ticks
                && config.day_eye_range == small.day_eye_range && config.night_eye_range == small.night_eye_range
                && config.food_regrowth_amount == small.food_regrowth_amount,
            "individual mechanics and time stay independent of world size");
        config.initial_population = config.minimum_population = 0;
        config.bootstrap_food_count = config.boosted_food_count = config.target_food_count = 0;
        for (std::uint64_t seed = 1; seed <= 8; ++seed) {
            config.seed = seed;
            const evobrain::Simulation simulation(config);
            bool land_wraps = false;
            const auto land = medium_components(simulation, evobrain::TerrainMedium::land, &land_wraps);
            const auto water = medium_components(simulation, evobrain::TerrainMedium::water);
            expect_true(size == evobrain::WorldSize::small_world ? land == 1 && water == 1
                    : size == evobrain::WorldSize::medium ? land == 2 && water == 1
                    : land == 3 && water == 1,
                "preset island counts share one surrounding ocean across wrapped edges");
            expect_true(!land_wraps, "rounded continents do not form world-spanning bands");
            const auto regions = simulation.biomes();
            const auto wetland_count = std::ranges::count_if(regions, [](const auto& region) {
                return region.kind == evobrain::BiomeKind::wetland;
            });
            expect_true(size == evobrain::WorldSize::small_world ? wetland_count == 0
                    : size == evobrain::WorldSize::medium ? wetland_count == 1
                    : wetland_count == 2,
                "major wetlands are absent on small and fewer than islands on larger worlds");
            std::size_t sparse = 0;
            double region_area = 0.0;
            for (const auto& region : regions) {
                region_area += region.width * region.height;
                if (region.fertility <= config.sparse_fertility_maximum) {
                    ++sparse;
                    expect_true(region.fertility >= config.minimum_fertility,
                        "sparse regions retain positive food supply");
                } else {
                    expect_true(region.fertility >= config.ordinary_fertility_minimum && region.fertility <= 1.0,
                        "ordinary regions cannot become accidental deserts");
                }
            }
            expect_near(region_area, config.world_width * config.world_height, 1e-8,
                "rectangular regions tile the complete world");
            if (size == evobrain::WorldSize::large) {
                expect_equal(sparse, static_cast<std::size_t>(std::round(regions.size() * 0.15)),
                    "large worlds allocate approximately fifteen percent sparse regions");
                bool adjacent = false;
                for (std::size_t i = 1; i < regions.size(); ++i) {
                    adjacent |= regions[i].position.y == regions[i - 1].position.y
                        && regions[i].fertility <= config.sparse_fertility_maximum
                        && regions[i - 1].fertility <= config.sparse_fertility_maximum;
                }
                expect_true(adjacent, "large deserts may occupy neighboring regions");
            } else {
                expect_true(sparse <= 1, "small and medium have at most one sparse region");
            }
            if (seed == 1) {
                const evobrain::Simulation repeated(config);
                expect_equal(std::vector(simulation.terrain().begin(), simulation.terrain().end()),
                    std::vector(repeated.terrain().begin(), repeated.terrain().end()),
                    "geography repeats for a seed and size");
            }
        }
    }
    expect_invalid_argument([] { (void)evobrain::make_world_config(1, static_cast<evobrain::WorldSize>(99)); },
        "invalid preset enums are rejected");
}

// Attachment neighborhoods contain land, sea and repeated patch alternation.
// Rounded extensions can lie outside the source fertility rectangle, and their
// edges and intact mainland may cut through a patch block.
void test_wetland_patch_mixture()
{
    auto config = evobrain::make_world_config(21, evobrain::WorldSize::medium);
    config.initial_population = config.minimum_population = 0;
    config.bootstrap_food_count = config.boosted_food_count = config.target_food_count = 0;
    config.wetland_water_coverage_minimum = config.wetland_water_coverage_maximum = 0.5;
    const evobrain::Simulation simulation(config);
    const auto cells = simulation.terrain();
    const auto columns = static_cast<std::size_t>(std::round(config.world_width / cells[0].width));
    const auto rows = cells.size() / columns;
    std::size_t wetlands = 0;
    bool mixed_rows = false;
    bool mixed_columns = false;
    for (const auto& region : simulation.biomes()) {
        if (region.kind != evobrain::BiomeKind::wetland) continue;
        ++wetlands;
        // Expand by one region on each side to include the offshore footprint;
        // offset by a whole world so unsigned indices also wrap safely at zero.
        const auto first_x = columns + static_cast<std::size_t>(std::floor(region.position.x / cells[0].width))
            - static_cast<std::size_t>(std::ceil(region.width / cells[0].width));
        const auto first_y = rows + static_cast<std::size_t>(std::floor(region.position.y / cells[0].height))
            - static_cast<std::size_t>(std::ceil(region.height / cells[0].height));
        const auto end_x = columns + static_cast<std::size_t>(std::ceil((region.position.x + 2 * region.width) / cells[0].width));
        const auto end_y = rows + static_cast<std::size_t>(std::ceil((region.position.y + 2 * region.height) / cells[0].height));
        const auto medium_at = [&](std::size_t x, std::size_t y) {
            return cells[(y % rows) * columns + x % columns].medium;
        };
        std::size_t water = 0;
        std::size_t land = 0;
        for (auto y = first_y; y < end_y; ++y) {
            std::size_t changes = 0;
            for (auto x = first_x; x < end_x; ++x) {
                const auto medium = medium_at(x, y);
                medium == evobrain::TerrainMedium::water ? ++water : ++land;
                if (x > first_x && medium_at(x - 1, y) != medium) ++changes;
            }
            mixed_rows |= changes >= 2;
        }
        for (auto x = first_x; x < end_x; ++x) {
            std::size_t changes = 0;
            for (auto y = first_y + 1; y < end_y; ++y) {
                changes += medium_at(x, y) != medium_at(x, y - 1);
            }
            mixed_columns |= changes >= 2;
        }
        expect_true(water > 0 && land > 0,
            "each wetland attachment neighborhood contains mainland and offshore water");
    }
    expect_true(wetlands > 0 && mixed_rows && mixed_columns,
        "wetlands form a two-dimensional patchwork rather than a single coast divider");
}

void test_daylight_and_deterministic_terrain()
{
    evobrain::SimulationConfig config {.seed = 411};
    config.initial_population = 0;
    config.minimum_population = 0;
    config.target_food_count = 0;
    config.food_population_threshold = 0;
    config.food_bootstrap_population_threshold = 0;
    config.bootstrap_food_count = 0;
    config.food_boost_population_threshold = 0;
    config.boosted_food_count = 0;
    config.day_night_cycle_ticks = 100;
    evobrain::Simulation first(config);
    evobrain::Simulation second(config);
    expect_equal(std::vector(first.biomes().begin(), first.biomes().end()),
        std::vector(second.biomes().begin(), second.biomes().end()),
        "biome regions are deterministic for a seed");
    expect_equal(std::vector(first.terrain().begin(), first.terrain().end()),
        std::vector(second.terrain().begin(), second.terrain().end()),
        "terrain medium, fertility, and rocks are deterministic for a seed");
    double covered_area = 0.0;
    for (const evobrain::BiomeRegion& region : first.biomes()) {
        covered_area += region.width * region.height;
    }
    expect_near(covered_area, config.world_width * config.world_height, 1e-10,
        "rectangular biome regions cover the world without gaps or overlap");
    expect_true(std::ranges::all_of(first.terrain(), [&](const evobrain::TerrainCell& cell) {
        return cell.fertility >= config.minimum_fertility && cell.fertility < 1.0;
    }), "every terrain cell retains a nonzero food-spawn weight");
    expect_near(first.light_level(), 0.0, 1e-12, "tick zero begins at full night");
    expect_near(first.current_eye_range(), config.night_eye_range, 1e-12,
        "night uses the configured short sight range");
    evobrain::SimulationSnapshot noon = first.snapshot();
    noon.current_tick = 50;
    const evobrain::Simulation daylight =
        evobrain::Simulation::from_snapshot(std::move(noon));
    expect_near(daylight.light_level(), 1.0, 1e-12,
        "half a cycle reaches full daylight");
    expect_near(daylight.current_eye_range(), config.day_eye_range, 1e-12,
        "daylight uses the configured long sight range");
}

// Both entrance widths remain clear, repeatably, despite maximal scattered rocks.
void test_cave_entrance_clearance()
{
    auto config = controlled_config(512);
    config.world_width = config.world_height = 5.0;
    config.biome_region_minimum_size = config.biome_region_maximum_size = 5.0;
    config.cave_region_probability = 1.0;
    config.scattered_rock_probability = 1.0;
    evobrain::Simulation simulation(config);
    const auto cells = simulation.terrain();
    const auto columns = static_cast<std::size_t>(std::round(config.world_width / cells[0].width));
    expect_true(!cells[2 * columns].rock && !cells[3 * columns].rock,
        "vertical wall entrance spans two adjacent clear cells");
    expect_true(!cells[3].rock && !cells[4].rock,
        "horizontal wall entrance spans two adjacent clear cells");
    std::size_t narrow = 0;
    std::size_t wide = 0;
    // Inspect complete openings along the first vertical and horizontal walls,
    // away from the central spawn clearing and any region-edge truncation.
    for (std::size_t start = 2; start + 3 < columns; start += 5) {
        expect_true(!cells[start * columns].rock && !cells[(start + 1) * columns].rock,
            "every vertical entrance keeps its first two cells clear");
        expect_true(cells[(start - 1) * columns].rock && cells[(start + 3) * columns].rock,
            "vertical entrance is bounded by walls outside its maximum width");
        cells[(start + 2) * columns].rock ? ++narrow : ++wide;
    }
    for (std::size_t start = 3; start + 3 < columns; start += 6) {
        expect_true(!cells[start].rock && !cells[start + 1].rock,
            "every horizontal entrance keeps its first two cells clear");
        expect_true(cells[start - 1].rock && cells[start + 3].rock,
            "horizontal entrance is bounded by walls outside its maximum width");
        cells[start + 2].rock ? ++narrow : ++wide;
    }
    expect_true(narrow > 0 && wide > 0, "seeded caves contain both entrance widths");
    evobrain::Simulation repeated(config);
    expect_equal(std::vector(cells.begin(), cells.end()),
        std::vector(repeated.terrain().begin(), repeated.terrain().end()),
        "entrance width choices repeat for the same seed and configuration");
}

void test_oxygen_and_rock_contact()
{
    evobrain::SimulationConfig oxygen_config = controlled_config(512);
    oxygen_config.oxygen_drain_per_tick = 0.01;
    oxygen_config.suffocation_energy_cost = 0.02;
    evobrain::Simulation terrain_probe(oxygen_config);
    const evobrain::TerrainCell* containing = nullptr;
    for (const evobrain::TerrainCell& cell : terrain_probe.terrain()) {
        if (0.5 >= cell.position.x && 0.5 < cell.position.x + cell.width
            && 0.5 >= cell.position.y && 0.5 < cell.position.y + cell.height) {
            containing = &cell;
            break;
        }
    }
    expect_true(containing != nullptr, "oxygen test position belongs to terrain");
    evobrain::Agent suffocating = controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 0.5);
    suffocating.water_adaptation = containing != nullptr
            && containing->medium == evobrain::TerrainMedium::water
        ? 0.0 : 1.0;
    suffocating.oxygen = 0.005;
    evobrain::Simulation oxygen = controlled_simulation(
        oxygen_config, {suffocating}, {});
    oxygen.tick();
    expect_near(oxygen.agents().front().oxygen, 0.0, 1e-12,
        "incompatible breathing depletes and clamps oxygen");
    expect_near(oxygen.agents().front().energy, 0.48, 1e-12,
        "zero oxygen applies the configured suffocation energy loss");

    evobrain::SimulationConfig rock_config = controlled_config(513);
    rock_config.cave_region_probability = 1.0;
    rock_config.maximum_movement_per_tick = 0.01;
    evobrain::Simulation rock_probe(rock_config);
    const evobrain::TerrainCell* target_rock = nullptr;
    for (const evobrain::TerrainCell& cell : rock_probe.terrain()) {
        const evobrain::Vec2 approach {.x = cell.position.x - rock_config.agent_radius - 0.001,
            .y = cell.position.y + cell.height * 0.5};
        if (cell.rock && approach.x >= 0.0) {
            const auto approach_cell = std::ranges::find_if(rock_probe.terrain(),
                [&](const evobrain::TerrainCell& candidate) {
                    return approach.x >= candidate.position.x
                        && approach.x < candidate.position.x + candidate.width
                        && approach.y >= candidate.position.y
                        && approach.y < candidate.position.y + candidate.height;
                });
            if (approach_cell != rock_probe.terrain().end() && !approach_cell->rock) {
                target_rock = &cell;
                break;
            }
        }
    }
    expect_true(target_rock != nullptr, "cave terrain provides an approachable rock wall");
    if (target_rock != nullptr) {
        const evobrain::Vec2 start {.x = target_rock->position.x
                - rock_config.agent_radius - 0.001,
            .y = target_rock->position.y + target_rock->height * 0.5};
        evobrain::Agent mover = controlled_agent(1, start, 0.0, 0.5);
        mover.brain = action_brain(0.0, 1.0, -1.0);
        evobrain::Simulation collision = controlled_simulation(rock_config, {mover}, {});
        collision.tick();
        expect_equal(collision.agents().front().position, start,
            "rock cells prevent agent movement through them");
        expect_true(collision.agents().front().rock_contact,
            "blocked movement records contact for the following brain tick");
    }
}

} // namespace

int main()
{
    test_random_and_fixed_tick_determinism();
    test_spatial_index_and_thread_determinism();
    test_brain_topology_and_ranges();
    test_cuda_brain_backend_agreement();
    test_simulation_backend_agreement();
    test_configuration_and_founders();
    test_literal_ray_first_hit_and_wrap();
    test_configured_world_movement_wrap();
    test_gradual_diet_specific_eating();
    test_mutual_and_proportional_agent_bites();
    test_inherited_mutation_and_reproduction_geometry();
    test_independent_trait_mutation();
    test_population_food_boost_and_natural_excess_reduction();
    test_global_food_regrowth_pulse();
    test_population_floor_and_food_replenishment();
    test_daylight_and_deterministic_terrain();
    test_world_sizes_and_geography();
    test_wetland_patch_mixture();
    test_oxygen_and_rock_contact();
    test_cave_entrance_clearance();
    failure_count += run_checkpoint_tests();
    if (failure_count != 0) {
        std::cerr << failure_count << " test expectation(s) failed\n";
        return 1;
    }
    std::cout << "All evobrain core tests passed\n";
    return 0;
}
