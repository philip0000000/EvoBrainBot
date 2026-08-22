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
    config.food_boost_population_threshold = 0;
    config.boosted_food_count = 0;
    config.world_width = 1.0;
    config.world_height = 1.0;
    config.carnivore_introduction_herbivore_threshold = 0;
    config.carnivore_introduction_ceiling = 0;
    config.carnivore_introduction_population_ceiling = 0;
    config.carnivore_introduction_batch = 0;
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
    const double energy, const evobrain::Diet diet = evobrain::Diet::herbivore)
{
    return {.id = id, .position = position, .direction = direction, .energy = energy,
        .diet = diet, .color = {.red = 0.2, .green = 0.3, .blue = 0.4},
        .mutation_rate = 0.01, .mutation_strength = 0.1,
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
    expect_equal(evobrain::brain_parameter_count, std::size_t {243},
        "26-to-8-to-3 brain has 243 parameters");
    evobrain::BrainParameters brain {};
    brain[evobrain::hidden_bias_offset] = 1.0;
    brain[evobrain::hidden_output_weight_offset] = 2.0;
    brain[evobrain::hidden_output_weight_offset + evobrain::brain_hidden_count] = -2.0;
    brain[evobrain::hidden_output_weight_offset + 2 * evobrain::brain_hidden_count] = 2.0;
    const evobrain::BrainOutputs output = evobrain::evaluate_brain(brain, {});
    expect_equal(output.turn, 1.0, "turn clamps to one");
    expect_equal(output.move, 0.0, "Move maps negative clamp to zero");
    expect_equal(output.eat, 1.0, "eat maps positive clamp to one");
}

void test_configuration_and_founders()
{
    evobrain::SimulationConfig invalid {.seed = 1};
    invalid.eye_range = 0.0;
    expect_invalid_argument([&] { evobrain::Simulation simulation(invalid); },
        "zero eye range is rejected");

    evobrain::Simulation first(evobrain::SimulationConfig {.seed = 123});
    evobrain::Simulation second(evobrain::SimulationConfig {.seed = 123});
    expect_equal(first.snapshot(), second.snapshot(), "founders are seed reproducible");
    expect_true(first.config().world_width == 2.5 && first.config().world_height == 2.5,
        "default world uses the configured 2.5 by 2.5 dimensions");
    expect_true(first.config().food_boost_population_threshold == 200
            && first.config().boosted_food_count == 1'000
            && first.config().food_population_threshold == 500
            && first.config().target_food_count == 500
            && first.config().maximum_new_food_per_tick == 5,
        "default population bands configure 1000, 500, then no spawning");
    expect_true(first.config().carnivore_introduction_interval_ticks == 500
            && first.config().carnivore_introduction_herbivore_threshold == 200
            && first.config().carnivore_introduction_ceiling == 30
            && first.config().carnivore_introduction_population_ceiling == 500
            && first.config().carnivore_introduction_batch == 15,
        "default carnivore introduction rule matches the ecological thresholds");
    expect_equal(first.food().size(), std::size_t {1'000},
        "new simulation starts with its complete initial food supply");
    expect_true(std::ranges::any_of(first.food(), [](const evobrain::Food& item) {
        return item.position.x > 1.0 || item.position.y > 1.0;
    }), "random food placement uses space beyond the former unit world");
    for (const evobrain::Agent& agent : first.agents()) {
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
        expect_true(agent.diet == evobrain::Diet::herbivore,
            "ordinary random founders are herbivores");
        expect_true(agent.position.x >= 0.0 && agent.position.x < first.config().world_width
                && agent.position.y >= 0.0
                && agent.position.y < first.config().world_height,
            "founders use the complete configured world bounds");
    }
    expect_equal(first.stats().herbivores + first.stats().carnivores,
        first.stats().population, "diet counts sum to population");
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
    expect_near(simulation.agents().front().position.x, 0.005, 1e-12,
        "movement wraps at the configured horizontal world extent");
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
    carnivore.diet = evobrain::Diet::carnivore;
    evobrain::Simulation wrong_food = controlled_simulation(config, {carnivore},
        {{.id = 1, .position = {.x = 0.51, .y = 0.5}, .energy = 0.12}});
    wrong_food.tick();
    expect_near(wrong_food.agents().front().energy, 0.25, 1e-12,
        "carnivore gains nothing from plant food");
    expect_near(wrong_food.food().front().energy, 0.12, 1e-12,
        "wrong-diet plant remains unchanged");

    evobrain::Agent blocking_agent = controlled_agent(
        2, {.x = 0.51, .y = 0.5}, 0.0, 0.5, evobrain::Diet::carnivore);
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
        evobrain::Diet::carnivore);
    evobrain::Agent right = controlled_agent(2, {.x = 0.518, .y = 0.5},
        std::numbers::pi_v<double>, 0.20, evobrain::Diet::carnivore);
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
        evobrain::Diet::herbivore);
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
        evobrain::Diet::carnivore);
    parent.color = {.red = 0.5, .green = 0.5, .blue = 0.5};
    parent.mutation_rate = 1.0;
    parent.mutation_strength = 0.1;
    evobrain::Simulation simulation = controlled_simulation(config, {parent}, {});
    simulation.tick();
    const evobrain::Agent& child = agent_with_id(simulation, 2);
    expect_equal(child.diet, evobrain::Diet::carnivore, "diet is copied without mutation");
    expect_near(child.position.x, 0.48, 1e-12, "child is placed two radii behind parent");
    expect_near(child.direction, std::numbers::pi_v<double>, 1e-12,
        "child faces opposite parent");
    expect_equal(child.age, std::uint64_t {0}, "new child has age zero");
    expect_near(child.prior_bite_damage, 0.0, 0.0, "new child damage starts clear");
    expect_true(std::abs(child.color.red - parent.color.red) <= 0.025
            && std::abs(child.mutation_rate - parent.mutation_rate) <= 0.002
            && std::abs(child.mutation_strength - parent.mutation_strength) <= 0.01,
        "category mutation scales bound inherited DNA changes");
}

void test_population_food_boost_and_natural_excess_reduction()
{
    evobrain::SimulationConfig low = controlled_config();
    low.initial_population = 39;
    low.target_food_count = 3;
    low.food_population_threshold = 100;
    low.food_boost_population_threshold = 40;
    low.boosted_food_count = 6;
    evobrain::Simulation low_population(low);
    expect_equal(low_population.food().size(), std::size_t {6},
        "population below boost threshold starts with boosted food count");

    evobrain::SimulationConfig normal = low;
    normal.initial_population = 40;
    evobrain::Simulation normal_population(normal);
    expect_equal(normal_population.food().size(), std::size_t {3},
        "population at boost threshold uses normal food count");

    evobrain::SimulationConfig crowded = low;
    crowded.initial_population = 3;
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
        return agent.diet == evobrain::Diet::herbivore;
    }), "population-floor founders are always herbivores");
    expect_equal(simulation.food().size(), std::size_t {2},
        "whole food spawning obeys its per-tick limit");
    simulation.tick();
    expect_equal(simulation.food().size(), std::size_t {3},
        "gradual spawning stops exactly at the population-band ceiling");
}

void test_periodic_carnivore_introduction()
{
    evobrain::SimulationConfig config = controlled_config(91);
    config.carnivore_introduction_interval_ticks = 5;
    config.carnivore_introduction_herbivore_threshold = 2;
    config.carnivore_introduction_ceiling = 3;
    config.carnivore_introduction_population_ceiling = 5;
    config.carnivore_introduction_batch = 2;
    evobrain::Simulation simulation = controlled_simulation(config,
        {
            controlled_agent(1, {.x = 0.2, .y = 0.2}, 0.0, 1.0),
            controlled_agent(2, {.x = 0.8, .y = 0.8}, 0.0, 1.0),
        }, {});
    evobrain::SimulationSnapshot before_check = simulation.snapshot();
    before_check.current_tick = 3;
    simulation = evobrain::Simulation::from_snapshot(std::move(before_check));
    simulation.tick();
    expect_equal(simulation.stats().carnivores, std::uint64_t {0},
        "carnivores are not introduced between periodic checks");
    simulation.tick();
    expect_equal(simulation.stats().carnivores, std::uint64_t {2},
        "eligible periodic check introduces one configured cohort");

    evobrain::SimulationSnapshot remainder = simulation.snapshot();
    remainder.current_tick = 9;
    simulation = evobrain::Simulation::from_snapshot(std::move(remainder));
    simulation.tick();
    expect_equal(simulation.stats().carnivores, std::uint64_t {3},
        "periodic introduction adds only the remainder to its carnivore ceiling");
    expect_equal(simulation.stats().population, std::uint64_t {5},
        "periodic introduction also respects the total-population ceiling");

    evobrain::SimulationConfig full_batch = controlled_config(92);
    full_batch.carnivore_introduction_interval_ticks = 5;
    full_batch.carnivore_introduction_herbivore_threshold = 2;
    full_batch.carnivore_introduction_ceiling = 30;
    full_batch.carnivore_introduction_population_ceiling = 50;
    full_batch.carnivore_introduction_batch = 15;
    std::vector<evobrain::Agent> founders {
        controlled_agent(1, {.x = 0.1, .y = 0.1}, 0.0, 1.0),
        controlled_agent(2, {.x = 0.2, .y = 0.2}, 0.0, 1.0),
    };
    for (std::uint64_t id = 3; id < 8; ++id) {
        founders.push_back(controlled_agent(id,
            {.x = 0.1 * static_cast<double>(id), .y = 0.5}, 0.0, 1.0,
            evobrain::Diet::carnivore));
    }
    evobrain::Simulation five_carnivores =
        controlled_simulation(full_batch, founders, {});
    evobrain::SimulationSnapshot full_batch_snapshot = five_carnivores.snapshot();
    full_batch_snapshot.current_tick = 4;
    five_carnivores =
        evobrain::Simulation::from_snapshot(std::move(full_batch_snapshot));
    five_carnivores.tick();
    expect_equal(five_carnivores.stats().carnivores, std::uint64_t {20},
        "five existing carnivores receive the complete fifteen-founder cohort");

    evobrain::SimulationConfig population_capped = full_batch;
    population_capped.carnivore_introduction_population_ceiling = 8;
    evobrain::Simulation near_population_ceiling =
        controlled_simulation(population_capped, std::move(founders), {});
    evobrain::SimulationSnapshot population_snapshot = near_population_ceiling.snapshot();
    population_snapshot.current_tick = 4;
    near_population_ceiling =
        evobrain::Simulation::from_snapshot(std::move(population_snapshot));
    near_population_ceiling.tick();
    expect_equal(near_population_ceiling.stats().population, std::uint64_t {8},
        "carnivore cohort is independently capped by total population room");
    expect_equal(near_population_ceiling.stats().carnivores, std::uint64_t {6},
        "population cap adds only one carnivore when one agent slot remains");
}

} // namespace

int main()
{
    test_random_and_fixed_tick_determinism();
    test_spatial_index_and_thread_determinism();
    test_brain_topology_and_ranges();
    test_configuration_and_founders();
    test_literal_ray_first_hit_and_wrap();
    test_configured_world_movement_wrap();
    test_gradual_diet_specific_eating();
    test_mutual_and_proportional_agent_bites();
    test_inherited_mutation_and_reproduction_geometry();
    test_population_food_boost_and_natural_excess_reduction();
    test_global_food_regrowth_pulse();
    test_population_floor_and_food_replenishment();
    test_periodic_carnivore_introduction();
    failure_count += run_checkpoint_tests();
    if (failure_count != 0) {
        std::cerr << failure_count << " test expectation(s) failed\n";
        return 1;
    }
    std::cout << "All evobrain core tests passed\n";
    return 0;
}
