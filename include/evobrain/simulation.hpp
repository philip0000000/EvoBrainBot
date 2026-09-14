#pragma once

#include "evobrain/brain.hpp"
#include "evobrain/brain_backend.hpp"
#include "evobrain/random.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace evobrain {

// Sets the minimum independent mutation probability per mutable value.
inline constexpr double minimum_mutation_rate = 1.0 / 75.0;
// Bounds the inherited strength scale; individual random changes may be smaller.
inline constexpr double minimum_mutation_strength = 1.0 / 75.0;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
    bool operator==(const Vec2&) const = default;
};

enum class TerrainMedium : std::uint8_t { land = 0, water = 1 };
enum class BiomeKind : std::uint8_t { land = 0, wetland = 1, water = 2 };
enum class WorldSize : std::uint8_t { small_world = 0, medium = 1, large = 2 };

// Stores an evolvable normalized RGB body color.
struct AgentColor {
    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
    bool operator==(const AgentColor&) const = default;
};

inline constexpr AgentColor plant_food_color {.red = 0.10, .green = 0.55, .blue = 0.22};
inline constexpr AgentColor rock_color {.red = 0.58, .green = 0.58, .blue = 0.58};

// Describes one deterministic rectangular biome region in world coordinates.
struct BiomeRegion {
    Vec2 position;
    double width = 0.0;
    double height = 0.0;
    // Non-wetland kinds describe the region center; wetland marks an attachment
    // site. Rounded coasts and offshore wetland footprints can cross these rectangles.
    BiomeKind kind = BiomeKind::land;
    double fertility = 0.0;
    double wetland_water_coverage = 0.0;
    bool operator==(const BiomeRegion&) const = default;
};

// Stores behaviorally relevant terrain for one fixed grid-aligned world cell.
struct TerrainCell {
    Vec2 position;
    double width = 0.0;
    double height = 0.0;
    TerrainMedium medium = TerrainMedium::land;
    double fertility = 0.0;
    bool rock = false;
    bool operator==(const TerrainCell&) const = default;
};

// Supplies explicit and provisional inputs for a reproducible simulation.
struct SimulationConfig {
    std::uint64_t seed;
    WorldSize world_size = WorldSize::small_world;
    std::uint64_t initial_population = 30;
    std::uint64_t minimum_population = 30;
    std::uint64_t target_food_count = 500;
    std::uint64_t food_population_threshold = 500;
    std::uint64_t food_bootstrap_population_threshold = 50;
    std::uint64_t bootstrap_food_count = 4'000;
    std::uint64_t food_boost_population_threshold = 200;
    std::uint64_t boosted_food_count = 1'000;
    std::uint64_t maximum_new_food_per_tick = 5;
    std::uint64_t food_regrowth_interval_ticks = 100;
    double world_width = 5.0;
    double world_height = 5.0;
    double terrain_cell_size = 0.04;
    double biome_region_minimum_size = 0.50;
    double biome_region_maximum_size = 1.25;
    double minimum_fertility = 0.05;
    double sparse_fertility_maximum = 0.15;
    double ordinary_fertility_minimum = 0.50;
    double wetland_water_coverage_minimum = 0.25;
    double wetland_water_coverage_maximum = 0.75;
    double scattered_rock_probability = 0.012;
    double cave_region_probability = 1.0 / 7.0;
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
    std::uint64_t day_night_cycle_ticks = 2'000;
    double night_eye_range = 0.10;
    double day_eye_range = 0.25;
    double maximum_oxygen = 1.0;
    double oxygen_refill_per_tick = 0.012;
    double oxygen_drain_per_tick = 0.008;
    double suffocation_energy_cost = 0.005;
    double off_medium_speed_multiplier = 0.50;
    double eat_threshold = 0.50;
    double eat_attempt_energy_cost = 0.001;
    double bite_amount_per_tick = 0.05;
    double initial_brain_parameter_minimum = -1.0;
    double initial_brain_parameter_maximum = 1.0;
    double brain_parameter_minimum = -4.0;
    double brain_parameter_maximum = 4.0;
    double founder_mutation_rate_minimum = minimum_mutation_rate;
    double founder_mutation_rate_maximum = 0.050;
    double founder_mutation_strength_minimum = 0.05;
    double founder_mutation_strength_maximum = 0.20;
    double brain_mutation_scale = 1.0;
    double mutation_rate_mutation_scale = 0.02;
    double mutation_strength_mutation_scale = 0.10;
    bool operator==(const SimulationConfig&) const = default;
};

// Creates a new-world preset, scaling population by length and food by area once.
// Loaded configurations are restored directly and must not be passed through this factory.
[[nodiscard]] SimulationConfig make_world_config(std::uint64_t seed, WorldSize size = WorldSize::small_world);

// Stores the complete observable and inherited state of one living agent.
struct Agent {
    std::uint64_t id = 0;
    Vec2 position;
    double direction = 0.0;
    double energy = 0.0;
    std::uint64_t age = 0;
    std::uint64_t generation = 0;
    double carnivore_tendency = 0.0;
    double water_adaptation = 0.0;
    double oxygen = 1.0;
    bool rock_contact = false;
    AgentColor color;
    double mutation_rate = minimum_mutation_rate;
    double mutation_strength = 0.1;
    // Integer percent preserves the inherited 20, 25, ..., 100 grid exactly.
    std::uint8_t trait_mutation_rate_percent = 20;
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

// Owns and advances all deterministic state of one evolving ecosystem simulation.
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

    // Returns deterministic biome regions used to construct the terrain grid.
    [[nodiscard]] std::span<const BiomeRegion> biomes() const noexcept;

    // Returns fixed terrain cells, including medium, fertility, and rocks.
    [[nodiscard]] std::span<const TerrainCell> terrain() const noexcept;

    // Returns the continuous global light level derived from completed ticks.
    [[nodiscard]] double light_level() const noexcept;

    // Returns the sight range interpolated for the current light level.
    [[nodiscard]] double current_eye_range() const noexcept;

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

    // Creates a generation-zero amphibious herbivore with a random brain at a
    // traversable position, for initialization or population-floor replacement.
    [[nodiscard]] Agent create_random_agent();

    // Creates one full-energy plant, optionally weighting its cell by fertility.
    [[nodiscard]] Food create_random_food(bool fertility_weighted);

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

    // Splits eligible parents' energy with one child, independently mutating
    // brain using its inherited pair and traits using the parent\'s discrete rate.
    void reproduce_eligible_agents();

    // Introduces independent herbivore founders to restore the population floor.
    void restore_minimum_population();

    // Adds a bounded number of plants toward the band ceiling without deleting excess.
    void replenish_food_if_allowed();

    // Applies one global energy pulse to surviving food on configured ticks.
    void regrow_food_if_due();

    // Builds rounded island continents, patchwork wetlands, size-dependent
    // sparse fertility clusters, and 70% two-cell / 30% three-cell cave entrances.
    void generate_terrain();

    // Returns the terrain cell containing a wrapped world position.
    [[nodiscard]] const TerrainCell& terrain_at(Vec2 position) const noexcept;

    // Reports whether a circular agent at the position overlaps a rock cell.
    [[nodiscard]] bool touches_rock(Vec2 position) const noexcept;

    // Finds the first rock hit along a ray, or infinity when unobstructed.
    [[nodiscard]] double ray_rock_distance(
        Vec2 origin, Vec2 direction, double range) const noexcept;

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
    std::vector<BiomeRegion> biomes_;
    std::vector<TerrainCell> terrain_;
    std::size_t terrain_columns_ = 1;
    std::size_t terrain_rows_ = 1;
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
