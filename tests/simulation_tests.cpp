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

// Records a failed boolean expectation while allowing later tests to run.
void expect_true(const bool condition, const std::string_view description)
{
    if (!condition) {
        std::cerr << "FAILED: " << description << '\n';
        ++failure_count;
    }
}

// Records a failed equality expectation without requiring streamable values.
template <typename Actual, typename Expected>
void expect_equal(
    const Actual& actual,
    const Expected& expected,
    const std::string_view description)
{
    if (!(actual == expected)) {
        std::cerr << "FAILED: " << description << '\n';
        ++failure_count;
    }
}

// Records a failed approximate floating-point equality expectation.
void expect_near(
    const double actual,
    const double expected,
    const double tolerance,
    const std::string_view description)
{
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "FAILED: " << description << " (expected " << expected
                  << ", got " << actual << ")\n";
        ++failure_count;
    }
}

// Records whether an operation rejects invalid input with invalid_argument.
template <typename Operation>
void expect_invalid_argument(
    Operation&& operation,
    const std::string_view description)
{
    try {
        std::forward<Operation>(operation)();
        expect_true(false, description);
    } catch (const std::invalid_argument&) {
    } catch (...) {
        expect_true(false, description);
    }
}

// Returns a mechanics-focused configuration with no random initial entities.
evobrain::SimulationConfig controlled_config(const std::uint64_t seed = 1)
{
    evobrain::SimulationConfig config {.seed = seed};
    config.initial_population = 0;
    config.minimum_population = 0;
    config.target_food_count = 0;
    config.food_population_threshold = 100;
    config.initial_energy = 1.0;
    config.food_energy = 0.5;
    config.living_energy_cost = 0.0;
    config.movement_energy_cost = 0.0;
    config.reproduction_threshold = 10.0;
    config.eating_radius = 0.0;
    config.maximum_movement_per_tick = 0.0;
    config.maximum_turn_per_tick = 0.0;
    config.initial_brain_parameter_minimum = 0.0;
    config.initial_brain_parameter_maximum = 0.0;
    config.mutation_strength = 0.1;
    config.brain_parameter_minimum = -4.0;
    config.brain_parameter_maximum = 4.0;
    return config;
}

// Creates a controlled agent and overrides the zero brain's half-speed movement.
evobrain::Agent controlled_agent(
    const std::uint64_t id,
    const evobrain::Vec2 position,
    const double direction,
    const double energy,
    const std::uint64_t generation = 0)
{
    evobrain::Agent agent {
        .id = id,
        .position = position,
        .direction = direction,
        .energy = energy,
        .generation = generation,
    };
    // A movement bias of -1 maps to zero movement after activation.
    agent.brain[evobrain::brain_parameter_count - 1] = -1.0;
    return agent;
}

// Replaces random initialization with an explicit validated test snapshot.
evobrain::Simulation controlled_simulation(
    const evobrain::SimulationConfig& config,
    std::vector<evobrain::Agent> agents,
    std::vector<evobrain::Food> food)
{
    evobrain::Simulation initial(config);
    evobrain::SimulationSnapshot snapshot = initial.snapshot();
    snapshot.agents = std::move(agents);
    snapshot.food = std::move(food);
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

// Returns the agent with the requested stable ID from a read-only state view.
const evobrain::Agent& agent_with_id(
    const evobrain::Simulation& simulation,
    const std::uint64_t id)
{
    const auto found = std::ranges::find(
        simulation.agents(), id, &evobrain::Agent::id);
    if (found == simulation.agents().end()) {
        throw std::logic_error("expected test agent was not found");
    }
    return *found;
}

// Verifies fixed-step tick accounting through the public simulation API.
void test_simulation_ticks()
{
    evobrain::Simulation simulation(evobrain::SimulationConfig {.seed = 1234});
    expect_equal(simulation.current_tick(), std::uint64_t {0},
        "new simulation starts at zero");

    simulation.tick();
    expect_equal(simulation.current_tick(), std::uint64_t {1},
        "tick advances exactly once");

    simulation.tick();
    simulation.tick();
    expect_equal(simulation.current_tick(), std::uint64_t {3},
        "multiple ticks accumulate");

    simulation.run_for(7);
    expect_equal(simulation.current_tick(), std::uint64_t {10},
        "run_for advances by n ticks");

    simulation.run_for(0);
    expect_equal(simulation.current_tick(), std::uint64_t {10},
        "run_for zero changes nothing");
}

// Verifies run_for has the same complete state as repeated individual ticks.
void test_run_for_equivalence()
{
    evobrain::Simulation run_simulation(evobrain::SimulationConfig {.seed = 9});
    evobrain::Simulation tick_simulation(evobrain::SimulationConfig {.seed = 9});

    run_simulation.run_for(25);
    for (std::uint64_t index = 0; index < 25; ++index) {
        tick_simulation.tick();
    }

    expect_equal(run_simulation.snapshot(), tick_simulation.snapshot(),
        "run_for matches repeated tick state");
}

// Verifies seeds reproducibly select deterministic PCG32 sequences.
void test_seed_behavior()
{
    evobrain::Pcg32 first(500);
    evobrain::Pcg32 second(500);
    evobrain::Pcg32 different(501);
    bool found_difference = false;

    for (std::size_t index = 0; index < 16; ++index) {
        const std::uint32_t first_value = first.next();
        expect_equal(first_value, second.next(), "equal seeds match");
        if (first_value != different.next()) {
            found_difference = true;
        }
    }

    expect_true(found_difference, "different seeds select different sequences");
}

// Locks the documented initialization and PCG-XSH-RR output to known values.
void test_known_pcg32_sequence()
{
    constexpr std::array<std::uint32_t, 6> expected {
        3270867926U,
        1795671209U,
        1924641435U,
        1143034755U,
        4121910957U,
        1757328946U,
    };

    evobrain::Pcg32 random(42);
    for (const std::uint32_t expected_value : expected) {
        expect_equal(random.next(), expected_value, "seed 42 PCG32 sequence");
    }
}

// Verifies deterministic real sampling and exact PRNG-state restoration.
void test_random_sampling_and_restore()
{
    evobrain::Pcg32 first(77);
    const double first_sample = first.unit_interval();
    expect_true(first_sample >= 0.0 && first_sample < 1.0,
        "unit sample is in its half-open range");

    const std::uint64_t saved_state = first.state();
    evobrain::Pcg32 restored = evobrain::Pcg32::from_state(saved_state);
    expect_equal(first.next(), restored.next(), "restored random state resumes exactly");

    for (std::size_t index = 0; index < 32; ++index) {
        expect_true(first.bounded(10) < 10, "bounded random value stays in range");
    }
}

// Verifies the direct brain's parameter layout and clamped output ranges.
void test_brain_evaluation()
{
    evobrain::BrainParameters parameters {};
    parameters[0] = 2.0;
    parameters[4] = 0.25;
    parameters[7] = -3.0;
    parameters[9] = 1.0;
    const evobrain::BrainOutputs outputs = evobrain::evaluate_brain(
        parameters,
        evobrain::BrainInputs {
            .food_direction_sine = 0.5,
            .food_distance = 1.0,
        });

    expect_equal(outputs.turn, 1.0, "turn output clamps to positive one");
    expect_equal(outputs.movement, 0.0, "movement output maps negative clamp to zero");
}

// Verifies configuration validation rejects unsafe mechanical relationships.
void test_configuration_validation()
{
    evobrain::SimulationConfig below_minimum {.seed = 1};
    below_minimum.initial_population = 1;
    below_minimum.minimum_population = 2;
    expect_invalid_argument(
        [&below_minimum] { evobrain::Simulation simulation(below_minimum); },
        "initial population below minimum is rejected");

    evobrain::SimulationConfig invalid_energy {.seed = 1};
    invalid_energy.initial_energy = 0.0;
    expect_invalid_argument(
        [&invalid_energy] { evobrain::Simulation simulation(invalid_energy); },
        "nonpositive initial energy is rejected");

    evobrain::SimulationConfig invalid_mutation {.seed = 1};
    invalid_mutation.mutation_strength = 0.0;
    expect_invalid_argument(
        [&invalid_mutation] {
            evobrain::Simulation simulation(invalid_mutation);
        },
        "zero mutation strength is rejected");

    evobrain::SimulationConfig overflowing_mutation {.seed = 1};
    overflowing_mutation.mutation_strength =
        std::numeric_limits<double>::max();
    expect_invalid_argument(
        [&overflowing_mutation] {
            evobrain::Simulation simulation(overflowing_mutation);
        },
        "overflowing mutation range is rejected");

    evobrain::SimulationConfig overflowing_initial_range {.seed = 1};
    overflowing_initial_range.initial_brain_parameter_minimum =
        -std::numeric_limits<double>::max();
    overflowing_initial_range.initial_brain_parameter_maximum =
        std::numeric_limits<double>::max();
    overflowing_initial_range.brain_parameter_minimum =
        -std::numeric_limits<double>::max();
    overflowing_initial_range.brain_parameter_maximum =
        std::numeric_limits<double>::max();
    expect_invalid_argument(
        [&overflowing_initial_range] {
            evobrain::Simulation simulation(overflowing_initial_range);
        },
        "overflowing initial brain range is rejected");

    evobrain::SimulationConfig invalid_brain_range {.seed = 1};
    invalid_brain_range.initial_brain_parameter_maximum = 5.0;
    expect_invalid_argument(
        [&invalid_brain_range] {
            evobrain::Simulation simulation(invalid_brain_range);
        },
        "initial brain range outside limits is rejected");
}

// Verifies seeded initialization creates valid, reproducible world entities.
void test_initialization_and_seeded_state()
{
    evobrain::Simulation first(evobrain::SimulationConfig {.seed = 123});
    evobrain::Simulation second(evobrain::SimulationConfig {.seed = 123});
    evobrain::Simulation different(evobrain::SimulationConfig {.seed = 124});

    expect_equal(first.snapshot(), second.snapshot(),
        "equal seed and configuration initialize equal state");
    expect_true(!(first.snapshot() == different.snapshot()),
        "different seeds initialize different state");
    expect_equal(first.agents().size(), std::size_t {30},
        "default initial population is created");
    expect_equal(first.food().size(), std::size_t {100},
        "default target food is created");
    expect_equal(first.stats().births, std::uint64_t {0},
        "initial agents are not reproduction births");
    expect_equal(first.stats().introduced_agents, std::uint64_t {0},
        "initial agents are not later introductions");
}

// Verifies movement uses clamped output and wraps across the world boundary.
void test_movement_and_wraparound()
{
    evobrain::SimulationConfig config = controlled_config();
    config.maximum_movement_per_tick = 0.01;
    evobrain::Agent agent = controlled_agent(1, {.x = 0.995, .y = 0.5}, 0.0, 1.0);
    agent.brain[evobrain::brain_parameter_count - 1] = 1.0;
    evobrain::Simulation simulation = controlled_simulation(config, {agent}, {});

    simulation.tick();

    expect_near(simulation.agents()[0].position.x, 0.005, 1e-12,
        "movement wraps at the right boundary");
    expect_near(simulation.agents()[0].position.y, 0.5, 1e-12,
        "horizontal movement preserves y");
}

// Verifies turn and movement scaling together with both per-tick energy costs.
void test_turning_movement_and_energy_costs()
{
    evobrain::SimulationConfig config = controlled_config();
    config.maximum_movement_per_tick = 0.01;
    config.maximum_turn_per_tick = 0.25;
    config.living_energy_cost = 0.2;
    config.movement_energy_cost = 0.5;
    evobrain::Agent agent =
        controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 1.0);
    agent.brain[evobrain::parameters_per_brain_output - 1] = 1.0;
    agent.brain[evobrain::brain_parameter_count - 1] = 1.0;
    evobrain::Simulation simulation = controlled_simulation(config, {agent}, {});

    simulation.tick();

    const evobrain::Agent& moved = simulation.agents().front();
    expect_near(moved.direction, 0.25, 1e-12,
        "maximum positive turn output uses the configured turn limit");
    expect_near(moved.position.x, 0.5 + std::cos(0.25) * 0.01, 1e-12,
        "maximum movement output uses the configured movement limit");
    expect_near(moved.position.y, 0.5 + std::sin(0.25) * 0.01, 1e-12,
        "movement follows the updated direction");
    expect_near(moved.energy, 0.795, 1e-12,
        "living and distance-proportional movement costs are both charged");
}

// Verifies food sensing uses shortest wraparound direction across an edge.
void test_nearest_food_wraparound_sensing()
{
    evobrain::SimulationConfig config = controlled_config();
    config.target_food_count = 2;
    config.food_population_threshold = 0;
    config.maximum_turn_per_tick = 0.25;
    evobrain::Agent agent = controlled_agent(1, {.x = 0.99, .y = 0.5}, 0.0, 1.0);
    agent.brain[1] = 2.0;
    evobrain::Simulation simulation = controlled_simulation(
        config,
        {agent},
        {
            {.id = 1, .position = {.x = 0.99, .y = 0.6}},
            {.id = 2, .position = {.x = 0.01, .y = 0.5}},
        });

    simulation.tick();

    expect_near(simulation.agents()[0].direction, 0.25, 1e-12,
        "nearest food across an edge is sensed instead of a lower-ID item");

    evobrain::Agent tie_agent =
        controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 1.0);
    tie_agent.brain[0] = 2.0;
    evobrain::Simulation tie_simulation = controlled_simulation(
        config,
        {tie_agent},
        {
            {.id = 2, .position = {.x = 0.5, .y = 0.6}},
            {.id = 1, .position = {.x = 0.5, .y = 0.4}},
        });

    tie_simulation.tick();

    expect_near(tie_simulation.agents()[0].direction,
        2.0 * std::numbers::pi_v<double> - 0.25, 1e-12,
        "lowest food ID wins an exact sensing-distance tie");
}

// Verifies the no-food sensor supplies zero direction and maximum distance.
void test_no_food_sensor_values()
{
    evobrain::SimulationConfig config = controlled_config();
    config.maximum_turn_per_tick = 0.25;
    evobrain::Agent distance_agent =
        controlled_agent(1, {.x = 0.3, .y = 0.5}, 0.0, 1.0);
    distance_agent.brain[2] = 1.0;
    evobrain::Agent sine_agent =
        controlled_agent(2, {.x = 0.5, .y = 0.5}, 0.0, 1.0);
    sine_agent.brain[0] = 1.0;
    evobrain::Agent cosine_agent =
        controlled_agent(3, {.x = 0.7, .y = 0.5}, 0.0, 1.0);
    cosine_agent.brain[1] = 1.0;
    evobrain::Simulation simulation = controlled_simulation(
        config, {distance_agent, sine_agent, cosine_agent}, {});

    simulation.tick();

    expect_near(agent_with_id(simulation, 1).direction, 0.25, 1e-12,
        "no food supplies normalized distance one");
    expect_near(agent_with_id(simulation, 2).direction, 0.0, 1e-12,
        "no food supplies relative-direction sine zero");
    expect_near(agent_with_id(simulation, 3).direction, 0.0, 1e-12,
        "no food supplies relative-direction cosine zero");
}

// Verifies exhausted agents die before coincident food can be consumed.
void test_death_before_eating()
{
    evobrain::SimulationConfig config = controlled_config();
    config.target_food_count = 1;
    config.food_population_threshold = 0;
    config.living_energy_cost = 1.0;
    config.eating_radius = 0.1;
    evobrain::Simulation simulation = controlled_simulation(
        config,
        {controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 0.5)},
        {{.id = 1, .position = {.x = 0.5, .y = 0.5}}});

    simulation.tick();

    expect_true(simulation.agents().empty(), "exhausted agent is removed");
    expect_equal(simulation.food().size(), std::size_t {1},
        "dead agent cannot consume food");
    expect_equal(simulation.stats().deaths, std::uint64_t {1},
        "death is counted once");
}

// Verifies eating awards energy to the nearest stable-ID winner only once.
void test_food_consumption_and_contention()
{
    evobrain::SimulationConfig config = controlled_config();
    config.target_food_count = 1;
    config.food_population_threshold = 0;
    config.eating_radius = 0.1;
    config.food_energy = 0.5;
    evobrain::Simulation simulation = controlled_simulation(
        config,
        {
            controlled_agent(2, {.x = 0.5, .y = 0.5}, 0.0, 0.25),
            controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 0.25),
        },
        {{.id = 1, .position = {.x = 0.5, .y = 0.5}}});

    simulation.tick();

    expect_true(simulation.food().empty(), "consumed food is removed");
    expect_equal(agent_with_id(simulation, 1).energy, 0.75,
        "lower stable ID wins exact food tie");
    expect_equal(agent_with_id(simulation, 2).energy, 0.25,
        "losing agent receives no food energy");

    evobrain::Simulation nearest_simulation = controlled_simulation(
        config,
        {
            controlled_agent(1, {.x = 0.6, .y = 0.5}, 0.0, 0.25),
            controlled_agent(2, {.x = 0.51, .y = 0.5}, 0.0, 0.25),
        },
        {{.id = 1, .position = {.x = 0.5, .y = 0.5}}});

    nearest_simulation.tick();

    expect_equal(agent_with_id(nearest_simulation, 2).energy, 0.75,
        "nearest agent wins food even when its stable ID is higher");
    expect_equal(agent_with_id(nearest_simulation, 1).energy, 0.25,
        "farther low-ID agent loses food contention");
}

// Verifies reproduction, energy splitting, placement, lineage, and mutation.
void test_reproduction_and_mutation()
{
    evobrain::SimulationConfig config = controlled_config();
    config.target_food_count = 1;
    config.food_population_threshold = 0;
    config.food_energy = 0.25;
    config.eating_radius = 0.1;
    config.reproduction_threshold = 1.0;
    config.maximum_movement_per_tick = 0.01;
    const evobrain::Agent parent =
        controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 0.75);
    evobrain::Simulation simulation = controlled_simulation(
        config,
        {parent},
        {{.id = 1, .position = {.x = 0.5, .y = 0.5}}});

    simulation.tick();

    expect_equal(simulation.agents().size(), std::size_t {2},
        "one eligible parent creates one child");
    const evobrain::Agent& updated_parent = agent_with_id(simulation, 1);
    const evobrain::Agent& child = agent_with_id(simulation, 2);
    expect_equal(updated_parent.energy, 0.5, "parent retains half the energy");
    expect_equal(child.energy, 0.5, "child receives half the energy");
    expect_equal(updated_parent.brain, parent.brain, "parent brain does not mutate");
    expect_equal(child.generation, std::uint64_t {1},
        "child advances one lineage generation");
    expect_equal(child.age, std::uint64_t {0}, "child does not age on birth tick");
    expect_near(child.position.x, 0.49, 1e-12, "child starts behind parent");
    expect_near(child.direction, std::numbers::pi_v<double>, 1e-12,
        "child faces opposite parent");

    std::size_t changed_parameters = 0;
    for (std::size_t index = 0; index < evobrain::brain_parameter_count; ++index) {
        if (child.brain[index] != parent.brain[index]) {
            ++changed_parameters;
        }
    }
    expect_equal(changed_parameters, std::size_t {1},
        "exactly one child brain parameter changes");
    for (std::size_t index = 0; index < evobrain::brain_parameter_count; ++index) {
        expect_true(
            std::abs(child.brain[index] - parent.brain[index])
                <= config.mutation_strength,
            "child mutation remains inside the configured mutation range");
    }
    expect_equal(simulation.stats().births, std::uint64_t {1},
        "reproduction birth is counted");
}

// Verifies newborns wait a tick and controlled conditions reach generations.
void test_multiple_lineage_generations()
{
    evobrain::SimulationConfig config = controlled_config();
    config.reproduction_threshold = 1.0;
    evobrain::Simulation simulation = controlled_simulation(
        config,
        {controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 4.0)},
        {});

    simulation.tick();
    expect_equal(simulation.agents().size(), std::size_t {2},
        "new child does not reproduce on birth tick");
    simulation.tick();

    const auto highest_generation = std::ranges::max_element(
        simulation.agents(), {}, &evobrain::Agent::generation)->generation;
    expect_equal(highest_generation, std::uint64_t {2},
        "controlled simulation reaches multiple lineage generations");
}

// Verifies deaths trigger random generation-zero population restoration.
void test_minimum_population_restoration()
{
    evobrain::SimulationConfig config = controlled_config();
    config.initial_population = 2;
    config.minimum_population = 2;
    config.living_energy_cost = 1.0;
    evobrain::Simulation simulation = controlled_simulation(
        config,
        {
            controlled_agent(1, {.x = 0.2, .y = 0.2}, 0.0, 0.5),
            controlled_agent(2, {.x = 0.8, .y = 0.8}, 0.0, 0.5),
        },
        {});

    simulation.tick();

    expect_equal(simulation.agents().size(), std::size_t {2},
        "population floor is restored");
    expect_equal(simulation.stats().deaths, std::uint64_t {2},
        "founder deaths are counted");
    expect_equal(simulation.stats().introduced_agents, std::uint64_t {2},
        "random introductions are counted separately");
    expect_true(std::ranges::all_of(simulation.agents(), [](const evobrain::Agent& agent) {
        return agent.age == 0 && agent.generation == 0;
    }), "introduced founders do not act and start at generation zero");
}

// Verifies population pressure pauses and later resumes food replacement.
void test_population_controlled_food_replacement()
{
    evobrain::SimulationConfig config = controlled_config();
    config.target_food_count = 2;
    config.food_population_threshold = 1;
    config.eating_radius = 0.1;
    evobrain::Simulation simulation = controlled_simulation(
        config,
        {controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 1.0)},
        {
            {.id = 1, .position = {.x = 0.5, .y = 0.5}},
            {.id = 2, .position = {.x = 0.5, .y = 0.5}},
        });

    simulation.tick();
    expect_true(simulation.food().empty(),
        "food is not replaced at the population threshold");

    evobrain::SimulationSnapshot starving = simulation.snapshot();
    starving.config.living_energy_cost = 100.0;
    evobrain::Simulation resumed =
        evobrain::Simulation::from_snapshot(std::move(starving));
    resumed.tick();
    expect_true(resumed.agents().empty(), "population falls below threshold");
    expect_equal(resumed.food().size(), std::size_t {2},
        "food replacement resumes below threshold");
}

// Verifies replacement food is retained until the tick after it is created.
void test_replacement_food_waits_until_next_tick()
{
    evobrain::SimulationConfig config = controlled_config();
    config.target_food_count = 1;
    config.food_population_threshold = 2;
    config.eating_radius = 1.0;
    evobrain::Simulation simulation = controlled_simulation(
        config,
        {controlled_agent(1, {.x = 0.5, .y = 0.5}, 0.0, 1.0)},
        {{.id = 1, .position = {.x = 0.5, .y = 0.5}}});

    simulation.tick();
    expect_equal(simulation.food().size(), std::size_t {1},
        "replacement food remains after its creation tick");
    const std::uint64_t first_replacement_id = simulation.food().front().id;

    simulation.tick();
    expect_equal(simulation.food().size(), std::size_t {1},
        "consumed replacement food is replenished on the following tick");
    expect_true(simulation.food().front().id != first_replacement_id,
        "food created on the previous tick can be consumed on the next tick");
}

} // namespace

// Runs the lightweight core test suite and reports failures through the exit code.
int main()
{
    test_simulation_ticks();
    test_run_for_equivalence();
    test_seed_behavior();
    test_known_pcg32_sequence();
    test_random_sampling_and_restore();
    test_brain_evaluation();
    test_configuration_validation();
    test_initialization_and_seeded_state();
    test_movement_and_wraparound();
    test_turning_movement_and_energy_costs();
    test_nearest_food_wraparound_sensing();
    test_no_food_sensor_values();
    test_death_before_eating();
    test_food_consumption_and_contention();
    test_reproduction_and_mutation();
    test_multiple_lineage_generations();
    test_minimum_population_restoration();
    test_population_controlled_food_replacement();
    test_replacement_food_waits_until_next_tick();
    failure_count += run_checkpoint_tests();

    if (failure_count != 0) {
        std::cerr << failure_count << " test expectation(s) failed\n";
        return 1;
    }

    std::cout << "All evobrain core tests passed\n";
    return 0;
}
