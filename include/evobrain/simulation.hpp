#pragma once

#include "evobrain/brain.hpp"
#include "evobrain/brain_backend.hpp"
#include "evobrain/random.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace evobrain {

inline constexpr double minimum_mutation_rate = 0.0001;
inline constexpr double minimum_mutation_strength = 0.001;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
    bool operator==(const Vec2&) const = default;
};

enum class Diet : std::uint8_t { herbivore = 0, carnivore = 1 };

// Stores an evolvable normalized RGB body color.
struct AgentColor {
    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
    bool operator==(const AgentColor&) const = default;
};

inline constexpr AgentColor plant_food_color {.red = 0.10, .green = 0.55, .blue = 0.22};

// Supplies explicit and provisional inputs for a reproducible simulation.
struct SimulationConfig {
    std::uint64_t seed;
    std::uint64_t initial_population = 30;
    std::uint64_t minimum_population = 30;
    std::uint64_t target_food_count = 500;
    std::uint64_t food_population_threshold = 500;
    std::uint64_t food_boost_population_threshold = 200;
    std::uint64_t boosted_food_count = 1'000;
    std::uint64_t maximum_new_food_per_tick = 5;
    std::uint64_t food_regrowth_interval_ticks = 100;
    std::uint64_t carnivore_introduction_interval_ticks = 500;
    std::uint64_t carnivore_introduction_herbivore_threshold = 200;
    std::uint64_t carnivore_introduction_ceiling = 30;
    std::uint64_t carnivore_introduction_population_ceiling = 500;
    std::uint64_t carnivore_introduction_batch = 15;
    double world_width = 2.5;
    double world_height = 2.5;
    double initial_energy = 0.5;
    double food_energy = 0.25;
    double food_regrowth_amount = 0.025;
    double living_energy_cost = 0.001;
    double movement_energy_cost = 0.1;
    double reproduction_threshold = 1.0;
    double maximum_movement_per_tick = 0.01;
    double maximum_turn_per_tick = 0.25;
    double agent_radius = 0.010;
    double food_radius = 0.005;
    double eye_range = 0.250;
    double eat_threshold = 0.50;
    double eat_attempt_energy_cost = 0.001;
    double bite_amount_per_tick = 0.05;
    double initial_brain_parameter_minimum = -1.0;
    double initial_brain_parameter_maximum = 1.0;
    double brain_parameter_minimum = -4.0;
    double brain_parameter_maximum = 4.0;
    double founder_mutation_rate_minimum = 0.005;
    double founder_mutation_rate_maximum = 0.020;
    double founder_mutation_strength_minimum = 0.05;
    double founder_mutation_strength_maximum = 0.20;
    double brain_mutation_scale = 1.0;
    double color_mutation_scale = 0.25;
    double mutation_rate_mutation_scale = 0.02;
    double mutation_strength_mutation_scale = 0.10;
    bool operator==(const SimulationConfig&) const = default;
};

// Stores the complete observable and inherited state of one living agent.
struct Agent {
    std::uint64_t id = 0;
    Vec2 position;
    double direction = 0.0;
    double energy = 0.0;
    std::uint64_t age = 0;
    std::uint64_t generation = 0;
    Diet diet = Diet::herbivore;
    AgentColor color;
    double mutation_rate = 0.01;
    double mutation_strength = 0.1;
    double prior_bite_damage = 0.0;
    BrainParameters brain {};
    BrainStructure brain_structure = founder_brain_structure();
    BrainState brain_state;
    bool operator==(const Agent&) const = default;
};

// Stores one plant-food item's identity, position, and gradually consumed energy.
struct Food {
    std::uint64_t id = 0;
    Vec2 position;
    double energy = 0.0;
    bool operator==(const Food&) const = default;
};

struct SimulationStats {
    std::uint64_t seed = 0;
    std::uint64_t completed_ticks = 0;
    std::uint64_t population = 0;
    std::uint64_t herbivores = 0;
    std::uint64_t carnivores = 0;
    std::uint64_t food = 0;
    std::uint64_t births = 0;
    std::uint64_t introduced_agents = 0;
    std::uint64_t deaths = 0;
    std::uint64_t agents_eaten = 0;
    bool operator==(const SimulationStats&) const = default;
};

// Selects execution resources without changing deterministic simulation state.
struct SimulationExecutionConfig {
    // Zero automatically uses the available logical processors.
    std::size_t thread_count = 0;
    BrainBackendKind brain_backend = BrainBackendKind::cpu;
};

// Reports non-persisted work and timing measurements from the latest tick.
struct SimulationDiagnostics {
    std::size_t spatial_columns = 0;
    std::size_t spatial_rows = 0;
    std::size_t execution_threads = 1;
    std::uint64_t vision_candidate_tests = 0;
    std::uint64_t vision_brute_force_tests = 0;
    std::uint64_t bite_candidate_tests = 0;
    std::uint64_t bite_brute_force_tests = 0;
    double spatial_index_milliseconds = 0.0;
    double sensing_milliseconds = 0.0;
    double brain_milliseconds = 0.0;
    // Retained as the aggregate compatibility measurement while callers migrate.
    double sensing_brain_milliseconds = 0.0;
    double movement_milliseconds = 0.0;
    double bite_milliseconds = 0.0;
    double lifecycle_milliseconds = 0.0;
    double total_milliseconds = 0.0;
};

struct SimulationSnapshot {
    SimulationConfig config;
    std::uint64_t current_tick = 0;
    std::uint64_t random_state = 0;
    std::uint64_t next_agent_id = 1;
    std::uint64_t next_food_id = 1;
    std::uint64_t births = 0;
    std::uint64_t introduced_agents = 0;
    std::uint64_t deaths = 0;
    std::uint64_t agents_eaten = 0;
    std::vector<Agent> agents;
    std::vector<Food> food;
    bool operator==(const SimulationSnapshot&) const = default;
};

// Owns and advances all deterministic state of one predator-prey simulation.
class Simulation {
public:
    // Creates a populated simulation at tick zero using validated configuration.
    explicit Simulation(const SimulationConfig& config,
        SimulationExecutionConfig execution = {});

    // Restores a simulation from a complete validated detached snapshot.
    [[nodiscard]] static Simulation from_snapshot(SimulationSnapshot snapshot,
        SimulationExecutionConfig execution = {});

    // Advances all simulation state by exactly one ordered fixed tick.
    void tick();

    // Advances through the requested number of complete fixed ticks.
    void run_for(std::uint64_t ticks);

    // Returns the number of ticks completed by this simulation.
    [[nodiscard]] std::uint64_t current_tick() const noexcept;

    // Returns the immutable configuration, including the original seed.
    [[nodiscard]] const SimulationConfig& config() const noexcept;

    // Returns a non-owning read-only view of all living agents.
    [[nodiscard]] std::span<const Agent> agents() const noexcept;

    // Returns a non-owning read-only view of all remaining plant food.
    [[nodiscard]] std::span<const Food> food() const noexcept;

    // Returns current populations and accumulated lifecycle statistics.
    [[nodiscard]] SimulationStats stats() const noexcept;

    // Returns transient performance measurements excluded from checkpoints.
    [[nodiscard]] const SimulationDiagnostics& diagnostics() const noexcept;

    // Returns the transient execution backend, which is never checkpointed.
    [[nodiscard]] BrainBackendKind brain_backend() const noexcept;

    // Switches execution backend while preserving host-owned recurrent state.
    void set_brain_backend(BrainBackendKind backend);

    // Returns every deterministic value required to resume exactly.
    [[nodiscard]] SimulationSnapshot snapshot() const;

private:
    struct SpatialCell {
        std::vector<std::size_t> agents;
        std::vector<std::size_t> food;
    };
    struct RestoredSnapshotTag { };
    struct AgentAction {
        std::uint64_t agent_id = 0;
        double turn = 0.0;
        double move = 0.0;
        bool eat = false;
    };
    // Takes ownership of an already validated restored snapshot.
    Simulation(SimulationSnapshot snapshot, RestoredSnapshotTag,
        SimulationExecutionConfig execution);

    // Configures fixed toroidal cells used only as a broad-phase acceleration index.
    void configure_spatial_index();

    // Rebuilds transient cell membership from the current entity arrays.
    void rebuild_spatial_index();

    // Collects conservative broad-phase candidates around one toroidal point.
    void collect_spatial_candidates(Vec2 center, double radius,
        std::vector<std::size_t>& agent_indices,
        std::vector<std::size_t>& food_indices) const;

    // Creates one random generation-zero founder with an explicitly assigned diet.
    [[nodiscard]] Agent create_random_agent(Diet diet);

    // Creates one full-energy plant item using the next stable ID.
    [[nodiscard]] Food create_random_food();

    // Produces six first-hit ray readings and normalized internal sensors.
    [[nodiscard]] BrainInputs sense_agent(const Agent& observer,
        std::span<const std::size_t> agent_candidates,
        std::span<const std::size_t> food_candidates) const;

    // Evaluates all brains against the same completed previous state.
    [[nodiscard]] std::vector<AgentAction> evaluate_agent_actions();

    // Rebuilds contiguous backend arrays only after population or genome changes.
    [[nodiscard]] bool synchronize_brain_batch();

    // Applies movement and all attempt costs without resolving targets.
    void move_agents_and_charge_energy(std::span<const AgentAction> actions);

    // Removes agents exhausted by living, movement, or attempt costs.
    void remove_dead_agents();

    // Selects mouth targets and applies all valid bite transfers together.
    void resolve_bites(std::span<const AgentAction> actions);

    // Creates at most one inherited and mutated child per eligible parent.
    void reproduce_eligible_agents();

    // Introduces independent herbivore founders to restore the population floor.
    void restore_minimum_population();

    // Introduces a bounded carnivore cohort on eligible periodic ticks.
    void introduce_carnivores_if_supported();

    // Adds a bounded number of plants toward the band ceiling without deleting excess.
    void replenish_food_if_allowed();

    // Applies one global energy pulse to surviving food on configured ticks.
    void regrow_food_if_due();

    SimulationConfig config_;
    std::uint64_t current_tick_ = 0;
    Pcg32 random_;
    std::uint64_t next_agent_id_ = 1;
    std::uint64_t next_food_id_ = 1;
    std::uint64_t births_ = 0;
    std::uint64_t introduced_agents_ = 0;
    std::uint64_t deaths_ = 0;
    std::uint64_t agents_eaten_ = 0;
    std::vector<Agent> agents_;
    std::vector<Food> food_;
    std::size_t execution_thread_count_ = 1;
    BrainBackendKind brain_backend_ = BrainBackendKind::cpu;
    bool brain_batch_dirty_ = true;
    bool brain_backend_cache_reset_ = true;
    std::vector<std::uint64_t> brain_ids_batch_;
    std::vector<BrainParameters> brain_parameters_batch_;
    std::vector<BrainStructure> brain_structures_batch_;
    std::vector<BrainState> brain_states_batch_;
    std::vector<BrainInputs> brain_inputs_batch_;
    std::vector<BrainOutputs> brain_outputs_batch_;
    std::size_t spatial_columns_ = 1;
    std::size_t spatial_rows_ = 1;
    double spatial_cell_width_ = 1.0;
    double spatial_cell_height_ = 1.0;
    std::vector<SpatialCell> spatial_cells_;
    SimulationDiagnostics diagnostics_;
};

} // namespace evobrain
