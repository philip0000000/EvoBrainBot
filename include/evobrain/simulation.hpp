#pragma once

#include "evobrain/brain.hpp"
#include "evobrain/random.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace evobrain {

// Represents a continuous position or displacement in the unit-square world.
struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    bool operator==(const Vec2&) const = default;
};

// Supplies explicit and provisional inputs for a reproducible simulation.
struct SimulationConfig {
    std::uint64_t seed;
    std::uint64_t initial_population = 30;
    std::uint64_t minimum_population = 30;
    std::uint64_t target_food_count = 100;
    std::uint64_t food_population_threshold = 100;
    double initial_energy = 0.5;
    double food_energy = 0.25;
    double living_energy_cost = 0.001;
    double movement_energy_cost = 0.1;
    double reproduction_threshold = 1.0;
    double eating_radius = 0.025;
    double maximum_movement_per_tick = 0.01;
    double maximum_turn_per_tick = 0.25;
    double initial_brain_parameter_minimum = -1.0;
    double initial_brain_parameter_maximum = 1.0;
    double mutation_strength = 0.1;
    double brain_parameter_minimum = -4.0;
    double brain_parameter_maximum = 4.0;

    bool operator==(const SimulationConfig&) const = default;
};

// Stores the complete observable state of one living agent.
struct Agent {
    std::uint64_t id = 0;
    Vec2 position;
    double direction = 0.0;
    double energy = 0.0;
    std::uint64_t age = 0;
    std::uint64_t generation = 0;
    BrainParameters brain {};

    bool operator==(const Agent&) const = default;
};

// Stores the complete observable state of one food item.
struct Food {
    std::uint64_t id = 0;
    Vec2 position;

    bool operator==(const Food&) const = default;
};

// Summarizes current population state and accumulated lifecycle events.
struct SimulationStats {
    std::uint64_t seed = 0;
    std::uint64_t completed_ticks = 0;
    std::uint64_t population = 0;
    std::uint64_t food = 0;
    std::uint64_t births = 0;
    std::uint64_t introduced_agents = 0;
    std::uint64_t deaths = 0;

    bool operator==(const SimulationStats&) const = default;
};

// Owns a detached, serializable copy of every deterministic state value.
struct SimulationSnapshot {
    SimulationConfig config;
    std::uint64_t current_tick = 0;
    std::uint64_t random_state = 0;
    std::uint64_t next_agent_id = 1;
    std::uint64_t next_food_id = 1;
    std::uint64_t births = 0;
    std::uint64_t introduced_agents = 0;
    std::uint64_t deaths = 0;
    std::vector<Agent> agents;
    std::vector<Food> food;

    bool operator==(const SimulationSnapshot&) const = default;
};

// Owns and advances the deterministic state of an EvoBrainBot simulation.
class Simulation {
public:
    // Creates a populated simulation at tick zero using validated configuration.
    explicit Simulation(const SimulationConfig& config);

    // Restores a simulation from a complete validated detached snapshot.
    [[nodiscard]] static Simulation from_snapshot(SimulationSnapshot snapshot);

    // Advances all simulation state by exactly one ordered fixed tick.
    void tick();

    // Advances the simulation through the requested number of fixed ticks.
    void run_for(std::uint64_t ticks);

    // Returns the number of ticks completed by this simulation.
    [[nodiscard]] std::uint64_t current_tick() const noexcept;

    // Returns the immutable configuration, including the original seed.
    [[nodiscard]] const SimulationConfig& config() const noexcept;

    // Returns a non-owning read-only view of the living agents.
    [[nodiscard]] std::span<const Agent> agents() const noexcept;

    // Returns a non-owning read-only view of the current food items.
    [[nodiscard]] std::span<const Food> food() const noexcept;

    // Returns current counts together with accumulated lifecycle statistics.
    [[nodiscard]] SimulationStats stats() const noexcept;

    // Returns a detached copy containing every value required for exact resume.
    [[nodiscard]] SimulationSnapshot snapshot() const;

private:
    struct RestoredSnapshotTag { };
    struct AgentAction {
        double turn = 0.0;
        double movement = 0.0;
    };

    // Takes ownership of an already validated restored snapshot.
    Simulation(SimulationSnapshot snapshot, RestoredSnapshotTag);

    // Creates one random generation-zero agent using the next stable ID.
    [[nodiscard]] Agent create_random_agent();

    // Creates one uniformly positioned food item using the next stable ID.
    [[nodiscard]] Food create_random_food();

    // Evaluates actions against a shared start-of-tick world snapshot.
    [[nodiscard]] std::vector<AgentAction> evaluate_agent_actions();

    // Applies actions, wrapping movement and charging age and energy costs.
    void move_agents_and_charge_energy(std::span<const AgentAction> actions);

    // Removes exhausted agents and records each death exactly once.
    void remove_dead_agents();

    // Assigns each reachable food item to one deterministic living consumer.
    void resolve_food_consumption();

    // Creates at most one inherited and mutated child per eligible parent.
    void reproduce_eligible_agents();

    // Introduces random founders until the configured population floor is met.
    void restore_minimum_population();

    // Restores target food only while population remains below its threshold.
    void replenish_food_if_allowed();

    SimulationConfig config_;
    std::uint64_t current_tick_ = 0;
    Pcg32 random_;
    std::uint64_t next_agent_id_ = 1;
    std::uint64_t next_food_id_ = 1;
    std::uint64_t births_ = 0;
    std::uint64_t introduced_agents_ = 0;
    std::uint64_t deaths_ = 0;
    std::vector<Agent> agents_;
    std::vector<Food> food_;
};

} // namespace evobrain
