#include "evobrain/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace evobrain {
namespace {

constexpr double world_size = 1.0;
constexpr double half_world_size = world_size * 0.5;
constexpr double maximum_toroidal_distance =
    0.707106781186547524400844362104849039;
constexpr double full_turn = 2.0 * std::numbers::pi_v<double>;

// Rejects non-finite values before they can poison deterministic simulation state.
void require_finite(const double value, const char* const name)
{
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

// Converts a serialized count to the native container size after range checking.
std::size_t checked_size(const std::uint64_t count, const char* const name)
{
    if (count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(std::string(name) + " is too large");
    }
    return static_cast<std::size_t>(count);
}

// Validates relationships that every new and restored simulation relies on.
void validate_config(const SimulationConfig& config)
{
    checked_size(config.initial_population, "initial population");
    checked_size(config.minimum_population, "minimum population");
    checked_size(config.target_food_count, "target food count");
    if (config.initial_population < config.minimum_population) {
        throw std::invalid_argument(
            "initial population must not be below minimum population");
    }

    require_finite(config.initial_energy, "initial energy");
    require_finite(config.food_energy, "food energy");
    require_finite(config.living_energy_cost, "living energy cost");
    require_finite(config.movement_energy_cost, "movement energy cost");
    require_finite(config.reproduction_threshold, "reproduction threshold");
    require_finite(config.eating_radius, "eating radius");
    require_finite(config.maximum_movement_per_tick, "maximum movement");
    require_finite(config.maximum_turn_per_tick, "maximum turn");
    require_finite(
        config.initial_brain_parameter_minimum,
        "initial brain parameter minimum");
    require_finite(
        config.initial_brain_parameter_maximum,
        "initial brain parameter maximum");
    require_finite(config.mutation_strength, "mutation strength");
    require_finite(config.brain_parameter_minimum, "brain parameter minimum");
    require_finite(config.brain_parameter_maximum, "brain parameter maximum");
    require_finite(
        config.initial_brain_parameter_maximum
            - config.initial_brain_parameter_minimum,
        "initial brain parameter range");
    require_finite(config.mutation_strength * 2.0, "mutation range");

    if (config.initial_energy <= 0.0 || config.food_energy < 0.0
        || config.living_energy_cost < 0.0
        || config.movement_energy_cost < 0.0
        || config.reproduction_threshold <= 0.0
        || config.eating_radius < 0.0
        || config.maximum_movement_per_tick < 0.0
        || config.maximum_turn_per_tick < 0.0
        || config.mutation_strength <= 0.0) {
        throw std::invalid_argument("simulation values are outside valid ranges");
    }
    if (config.brain_parameter_minimum >= config.brain_parameter_maximum) {
        throw std::invalid_argument("brain parameter limits must form a range");
    }
    if (config.initial_brain_parameter_minimum
            > config.initial_brain_parameter_maximum
        || config.initial_brain_parameter_minimum
            < config.brain_parameter_minimum
        || config.initial_brain_parameter_maximum
            > config.brain_parameter_maximum) {
        throw std::invalid_argument(
            "initial brain parameter range must be within parameter limits");
    }
}

// Wraps one finite coordinate into the unit world's half-open interval.
double wrap_coordinate(const double coordinate) noexcept
{
    return coordinate - std::floor(coordinate);
}

// Wraps a finite heading into the half-open interval [0, 2*pi).
double normalize_direction(const double direction) noexcept
{
    double normalized = std::fmod(direction, full_turn);
    if (normalized < 0.0) {
        normalized += full_turn;
    }
    return normalized;
}

// Returns the shortest signed displacement along one wraparound axis.
double wrapped_axis_displacement(const double from, const double to) noexcept
{
    double displacement = to - from;
    if (displacement > half_world_size) {
        displacement -= world_size;
    } else if (displacement < -half_world_size) {
        displacement += world_size;
    }
    return displacement;
}

// Returns the shortest displacement between two points in the toroidal world.
Vec2 toroidal_displacement(const Vec2 from, const Vec2 to) noexcept
{
    return Vec2 {
        .x = wrapped_axis_displacement(from.x, to.x),
        .y = wrapped_axis_displacement(from.y, to.y),
    };
}

// Returns squared toroidal distance, preserving exact comparisons and avoiding sqrt.
double toroidal_distance_squared(const Vec2 first, const Vec2 second) noexcept
{
    const Vec2 displacement = toroidal_displacement(first, second);
    return displacement.x * displacement.x
        + displacement.y * displacement.y;
}

// Validates that a detached snapshot can safely become live simulation state.
void validate_snapshot(const SimulationSnapshot& snapshot)
{
    validate_config(snapshot.config);
    if (snapshot.agents.size()
        < checked_size(snapshot.config.minimum_population, "minimum population")) {
        throw std::invalid_argument("snapshot population is below its minimum");
    }
    if (snapshot.food.size()
        > checked_size(snapshot.config.target_food_count, "target food count")) {
        throw std::invalid_argument("snapshot food exceeds its target count");
    }

    std::unordered_set<std::uint64_t> agent_ids;
    std::uint64_t maximum_agent_id = 0;
    for (const Agent& agent : snapshot.agents) {
        require_finite(agent.position.x, "agent x position");
        require_finite(agent.position.y, "agent y position");
        require_finite(agent.direction, "agent direction");
        require_finite(agent.energy, "agent energy");
        if (agent.id == 0 || !agent_ids.insert(agent.id).second
            || agent.position.x < 0.0 || agent.position.x >= world_size
            || agent.position.y < 0.0 || agent.position.y >= world_size
            || agent.direction < 0.0 || agent.direction >= full_turn
            || agent.energy <= 0.0) {
            throw std::invalid_argument("snapshot contains an invalid agent");
        }
        for (const double parameter : agent.brain) {
            require_finite(parameter, "brain parameter");
            if (parameter < snapshot.config.brain_parameter_minimum
                || parameter > snapshot.config.brain_parameter_maximum) {
                throw std::invalid_argument(
                    "snapshot brain parameter exceeds configured limits");
            }
        }
        maximum_agent_id = std::max(maximum_agent_id, agent.id);
    }
    if (snapshot.next_agent_id == 0
        || snapshot.next_agent_id <= maximum_agent_id) {
        throw std::invalid_argument("snapshot next agent ID is invalid");
    }

    std::unordered_set<std::uint64_t> food_ids;
    std::uint64_t maximum_food_id = 0;
    for (const Food& item : snapshot.food) {
        require_finite(item.position.x, "food x position");
        require_finite(item.position.y, "food y position");
        if (item.id == 0 || !food_ids.insert(item.id).second
            || item.position.x < 0.0 || item.position.x >= world_size
            || item.position.y < 0.0 || item.position.y >= world_size) {
            throw std::invalid_argument("snapshot contains invalid food");
        }
        maximum_food_id = std::max(maximum_food_id, item.id);
    }
    if (snapshot.next_food_id == 0 || snapshot.next_food_id <= maximum_food_id) {
        throw std::invalid_argument("snapshot next food ID is invalid");
    }
}

} // namespace

Simulation::Simulation(const SimulationConfig& config)
    : config_(config)
    , random_(config.seed)
{
    validate_config(config_);
    agents_.reserve(checked_size(config_.initial_population, "initial population"));
    food_.reserve(checked_size(config_.target_food_count, "target food count"));

    while (agents_.size() < config_.initial_population) {
        agents_.push_back(create_random_agent());
    }
    while (food_.size() < config_.target_food_count) {
        food_.push_back(create_random_food());
    }
}

Simulation::Simulation(
    SimulationSnapshot snapshot,
    RestoredSnapshotTag)
    : config_(snapshot.config)
    , current_tick_(snapshot.current_tick)
    , random_(Pcg32::from_state(snapshot.random_state))
    , next_agent_id_(snapshot.next_agent_id)
    , next_food_id_(snapshot.next_food_id)
    , births_(snapshot.births)
    , introduced_agents_(snapshot.introduced_agents)
    , deaths_(snapshot.deaths)
    , agents_(std::move(snapshot.agents))
    , food_(std::move(snapshot.food))
{
}

Simulation Simulation::from_snapshot(SimulationSnapshot snapshot)
{
    validate_snapshot(snapshot);
    return Simulation(std::move(snapshot), RestoredSnapshotTag {});
}

Agent Simulation::create_random_agent()
{
    if (next_agent_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("agent ID space exhausted");
    }

    Agent agent {
        .id = next_agent_id_++,
        .position = Vec2 {
            .x = random_.unit_interval(),
            .y = random_.unit_interval(),
        },
        .direction = random_.uniform(0.0, full_turn),
        .energy = config_.initial_energy,
    };
    for (double& parameter : agent.brain) {
        parameter = random_.uniform(
            config_.initial_brain_parameter_minimum,
            config_.initial_brain_parameter_maximum);
    }
    return agent;
}

Food Simulation::create_random_food()
{
    if (next_food_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("food ID space exhausted");
    }
    return Food {
        .id = next_food_id_++,
        .position = Vec2 {
            .x = random_.unit_interval(),
            .y = random_.unit_interval(),
        },
    };
}

std::vector<Simulation::AgentAction> Simulation::evaluate_agent_actions()
{
    std::vector<AgentAction> actions;
    actions.reserve(agents_.size());

    for (const Agent& agent : agents_) {
        BrainInputs inputs {
            .energy = std::clamp(
                agent.energy / config_.reproduction_threshold, 0.0, 1.0),
        };

        const Food* nearest_food = nullptr;
        double nearest_distance_squared = 0.0;
        for (const Food& item : food_) {
            const double distance_squared =
                toroidal_distance_squared(agent.position, item.position);
            if (nearest_food == nullptr
                || distance_squared < nearest_distance_squared
                || (distance_squared == nearest_distance_squared
                    && item.id < nearest_food->id)) {
                nearest_food = &item;
                nearest_distance_squared = distance_squared;
            }
        }

        if (nearest_food != nullptr) {
            const Vec2 displacement =
                toroidal_displacement(agent.position, nearest_food->position);
            const double distance = std::sqrt(nearest_distance_squared);
            inputs.food_distance = std::clamp(
                distance / maximum_toroidal_distance, 0.0, 1.0);
            if (distance == 0.0) {
                // A coincident food item is treated as directly ahead because
                // it has distance but no mathematically defined direction.
                inputs.food_direction_cosine = 1.0;
            } else {
                const double target_x = displacement.x / distance;
                const double target_y = displacement.y / distance;
                const double heading_x = std::cos(agent.direction);
                const double heading_y = std::sin(agent.direction);
                inputs.food_direction_cosine =
                    heading_x * target_x + heading_y * target_y;
                inputs.food_direction_sine =
                    heading_x * target_y - heading_y * target_x;
            }
        }

        const BrainOutputs output = evaluate_brain(agent.brain, inputs);
        actions.push_back(AgentAction {
            .turn = output.turn,
            .movement = output.movement,
        });
    }
    return actions;
}

void Simulation::move_agents_and_charge_energy(
    const std::span<const AgentAction> actions)
{
    for (std::size_t index = 0; index < agents_.size(); ++index) {
        Agent& agent = agents_[index];
        const AgentAction& action = actions[index];
        agent.direction = normalize_direction(
            agent.direction + action.turn * config_.maximum_turn_per_tick);
        const double distance =
            action.movement * config_.maximum_movement_per_tick;
        agent.position.x = wrap_coordinate(
            agent.position.x + std::cos(agent.direction) * distance);
        agent.position.y = wrap_coordinate(
            agent.position.y + std::sin(agent.direction) * distance);
        agent.energy -= config_.living_energy_cost
            + config_.movement_energy_cost * distance;
        ++agent.age;
    }
}

void Simulation::remove_dead_agents()
{
    const std::size_t previous_size = agents_.size();
    std::erase_if(agents_, [](const Agent& agent) { return agent.energy <= 0.0; });
    deaths_ += static_cast<std::uint64_t>(previous_size - agents_.size());
}

void Simulation::resolve_food_consumption()
{
    const double eating_radius_squared =
        config_.eating_radius * config_.eating_radius;
    std::vector<Food> remaining_food;
    remaining_food.reserve(food_.size());

    for (const Food& item : food_) {
        Agent* consumer = nullptr;
        double consumer_distance_squared = 0.0;
        for (Agent& agent : agents_) {
            const double distance_squared =
                toroidal_distance_squared(agent.position, item.position);
            if (distance_squared <= eating_radius_squared
                && (consumer == nullptr
                    || distance_squared < consumer_distance_squared
                    || (distance_squared == consumer_distance_squared
                        && agent.id < consumer->id))) {
                consumer = &agent;
                consumer_distance_squared = distance_squared;
            }
        }

        if (consumer == nullptr) {
            remaining_food.push_back(item);
        } else {
            consumer->energy += config_.food_energy;
        }
    }
    food_ = std::move(remaining_food);
}

void Simulation::reproduce_eligible_agents()
{
    const std::size_t parent_count = agents_.size();
    std::vector<Agent> children;
    children.reserve(parent_count);

    for (std::size_t index = 0; index < parent_count; ++index) {
        Agent& parent = agents_[index];
        if (parent.energy < config_.reproduction_threshold) {
            continue;
        }
        if (next_agent_id_ == std::numeric_limits<std::uint64_t>::max()
            || parent.generation == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("agent lineage state exhausted");
        }

        parent.energy *= 0.5;
        const double heading_x = std::cos(parent.direction);
        const double heading_y = std::sin(parent.direction);
        Agent child = parent;
        child.id = next_agent_id_++;
        child.position = Vec2 {
            .x = wrap_coordinate(
                parent.position.x
                - heading_x * config_.maximum_movement_per_tick),
            .y = wrap_coordinate(
                parent.position.y
                - heading_y * config_.maximum_movement_per_tick),
        };
        child.direction = normalize_direction(parent.direction + std::numbers::pi_v<double>);
        child.age = 0;
        child.generation = parent.generation + 1;

        const std::size_t parameter_index = random_.bounded(
            static_cast<std::uint32_t>(brain_parameter_count));
        double& parameter = child.brain[parameter_index];
        const double mutation = random_.uniform(
            -config_.mutation_strength, config_.mutation_strength);
        // Only the selected parameter receives the sampled delta. Clamping at
        // a configured limit may leave its numeric value unchanged.
        parameter = std::clamp(
            parameter + mutation,
            config_.brain_parameter_minimum,
            config_.brain_parameter_maximum);

        children.push_back(child);
        ++births_;
    }

    agents_.insert(agents_.end(), children.begin(), children.end());
}

void Simulation::restore_minimum_population()
{
    const std::size_t minimum =
        checked_size(config_.minimum_population, "minimum population");
    while (agents_.size() < minimum) {
        agents_.push_back(create_random_agent());
        ++introduced_agents_;
    }
}

void Simulation::replenish_food_if_allowed()
{
    if (agents_.size() >= config_.food_population_threshold) {
        return;
    }
    const std::size_t target =
        checked_size(config_.target_food_count, "target food count");
    while (food_.size() < target) {
        food_.push_back(create_random_food());
    }
}

void Simulation::tick()
{
    if (current_tick_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("simulation tick counter exhausted");
    }

    const std::vector<AgentAction> actions = evaluate_agent_actions();
    move_agents_and_charge_energy(actions);
    remove_dead_agents();
    resolve_food_consumption();
    reproduce_eligible_agents();
    restore_minimum_population();
    replenish_food_if_allowed();
    ++current_tick_;
}

void Simulation::run_for(const std::uint64_t ticks)
{
    if (ticks > std::numeric_limits<std::uint64_t>::max() - current_tick_) {
        throw std::overflow_error("requested ticks exceed the simulation counter");
    }
    for (std::uint64_t tick_index = 0; tick_index < ticks; ++tick_index) {
        tick();
    }
}

std::uint64_t Simulation::current_tick() const noexcept
{
    return current_tick_;
}

const SimulationConfig& Simulation::config() const noexcept
{
    return config_;
}

std::span<const Agent> Simulation::agents() const noexcept
{
    return agents_;
}

std::span<const Food> Simulation::food() const noexcept
{
    return food_;
}

SimulationStats Simulation::stats() const noexcept
{
    return SimulationStats {
        .seed = config_.seed,
        .completed_ticks = current_tick_,
        .population = static_cast<std::uint64_t>(agents_.size()),
        .food = static_cast<std::uint64_t>(food_.size()),
        .births = births_,
        .introduced_agents = introduced_agents_,
        .deaths = deaths_,
    };
}

SimulationSnapshot Simulation::snapshot() const
{
    return SimulationSnapshot {
        .config = config_,
        .current_tick = current_tick_,
        .random_state = random_.state(),
        .next_agent_id = next_agent_id_,
        .next_food_id = next_food_id_,
        .births = births_,
        .introduced_agents = introduced_agents_,
        .deaths = deaths_,
        .agents = agents_,
        .food = food_,
    };
}

} // namespace evobrain
