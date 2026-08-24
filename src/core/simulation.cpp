#include "evobrain/simulation.hpp"

#include "parallel_executor.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace evobrain {
namespace {

constexpr double full_turn = 2.0 * std::numbers::pi_v<double>;
constexpr std::array<double, vision_ray_count> ray_angle_offsets {
    -std::numbers::pi_v<double> / 2.0, -std::numbers::pi_v<double> / 4.0, 0.0,
    0.0, std::numbers::pi_v<double> / 4.0, std::numbers::pi_v<double> / 2.0,
};

using Clock = std::chrono::steady_clock;

// Converts one measured phase interval to fractional milliseconds for diagnostics.
double elapsed_milliseconds(const Clock::time_point start,
    const Clock::time_point end) noexcept
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Resolves the requested worker count without exceeding available logical processors.
std::size_t resolve_execution_threads(const SimulationExecutionConfig execution) noexcept
{
    const std::size_t available = detail::available_execution_threads();
    return execution.thread_count == 0
        ? available
        : std::clamp(execution.thread_count, std::size_t {1}, available);
}

void require_finite(const double value, const char* const name)
{
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

std::size_t checked_size(const std::uint64_t count, const char* const name)
{
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(std::string(name) + " is too large");
    }
    return static_cast<std::size_t>(count);
}

// Validates all persisted mechanics because restored checkpoints bypass defaults.
void validate_config(const SimulationConfig& config)
{
    checked_size(config.initial_population, "initial population");
    checked_size(config.minimum_population, "minimum population");
    checked_size(config.target_food_count, "target food count");
    checked_size(config.boosted_food_count, "boosted food count");
    checked_size(config.maximum_new_food_per_tick, "maximum new food per tick");
    checked_size(config.carnivore_introduction_ceiling, "carnivore introduction ceiling");
    checked_size(config.carnivore_introduction_population_ceiling,
        "carnivore introduction population ceiling");
    checked_size(config.carnivore_introduction_batch, "carnivore introduction batch");
    if (config.initial_population < config.minimum_population) {
        throw std::invalid_argument("initial population must not be below minimum population");
    }
    const std::array values {
        config.world_width, config.world_height, config.initial_energy,
        config.food_energy, config.food_regrowth_amount,
        config.living_energy_cost,
        config.movement_energy_cost, config.reproduction_threshold,
        config.maximum_movement_per_tick, config.maximum_turn_per_tick,
        config.agent_radius, config.food_radius, config.eye_range, config.eat_threshold,
        config.eat_attempt_energy_cost, config.bite_amount_per_tick,
        config.initial_brain_parameter_minimum, config.initial_brain_parameter_maximum,
        config.brain_parameter_minimum, config.brain_parameter_maximum,
        config.founder_mutation_rate_minimum, config.founder_mutation_rate_maximum,
        config.founder_mutation_strength_minimum, config.founder_mutation_strength_maximum,
        config.brain_mutation_scale, config.color_mutation_scale,
        config.mutation_rate_mutation_scale, config.mutation_strength_mutation_scale,
    };
    for (const double value : values) {
        require_finite(value, "simulation configuration value");
    }
    if (config.world_width <= 0.0 || config.world_height <= 0.0
        || config.initial_energy <= 0.0 || config.food_energy <= 0.0
        || config.food_regrowth_amount < 0.0
        || config.food_regrowth_interval_ticks == 0
        || config.carnivore_introduction_interval_ticks == 0
        || config.boosted_food_count < config.target_food_count
        || config.food_boost_population_threshold > config.food_population_threshold
        || config.living_energy_cost < 0.0 || config.movement_energy_cost < 0.0
        || config.reproduction_threshold <= 0.0 || config.maximum_movement_per_tick < 0.0
        || config.maximum_turn_per_tick < 0.0 || config.agent_radius <= 0.0
        || config.food_radius <= 0.0 || config.eye_range <= 0.0
        || config.eat_threshold < 0.0 || config.eat_threshold > 1.0
        || config.eat_attempt_energy_cost < 0.0 || config.bite_amount_per_tick <= 0.0
        || config.brain_parameter_minimum >= config.brain_parameter_maximum
        || config.initial_brain_parameter_minimum > config.initial_brain_parameter_maximum
        || config.initial_brain_parameter_minimum < config.brain_parameter_minimum
        || config.initial_brain_parameter_maximum > config.brain_parameter_maximum
        || config.founder_mutation_rate_minimum < minimum_mutation_rate
        || config.founder_mutation_rate_maximum > 1.0
        || config.founder_mutation_rate_minimum > config.founder_mutation_rate_maximum
        || config.founder_mutation_strength_minimum < minimum_mutation_strength
        || config.founder_mutation_strength_maximum > 1.0
        || config.founder_mutation_strength_minimum > config.founder_mutation_strength_maximum
        || config.brain_mutation_scale < 0.0 || config.color_mutation_scale < 0.0
        || config.mutation_rate_mutation_scale < 0.0
        || config.mutation_strength_mutation_scale < 0.0) {
        throw std::invalid_argument("simulation values are outside valid ranges");
    }
}

// Wraps one coordinate into the half-open interval for its configured axis.
double wrap_coordinate(const double coordinate, const double extent) noexcept
{
    return coordinate - std::floor(coordinate / extent) * extent;
}

double normalize_direction(const double direction) noexcept
{
    double normalized = std::fmod(direction, full_turn);
    return normalized < 0.0 ? normalized + full_turn : normalized;
}

// Returns the shortest signed displacement along one toroidal axis.
double wrapped_axis_displacement(
    const double from, const double to, const double extent) noexcept
{
    double value = to - from;
    const double half_extent = extent * 0.5;
    if (value > half_extent) value -= extent;
    else if (value < -half_extent) value += extent;
    return value;
}

// Returns the shortest two-dimensional displacement across configured boundaries.
Vec2 toroidal_displacement(
    const Vec2 from, const Vec2 to,
    const double world_width, const double world_height) noexcept
{
    return {.x = wrapped_axis_displacement(from.x, to.x, world_width),
        .y = wrapped_axis_displacement(from.y, to.y, world_height)};
}

// Returns squared toroidal distance without paying for a square root.
double toroidal_distance_squared(
    const Vec2 first, const Vec2 second,
    const double world_width, const double world_height) noexcept
{
    const Vec2 displacement =
        toroidal_displacement(first, second, world_width, world_height);
    return displacement.x * displacement.x + displacement.y * displacement.y;
}

// Accepts only the two serialized diet values understood by the simulation.
bool valid_diet(const Diet diet) noexcept
{
    return diet == Diet::herbivore || diet == Diet::carnivore;
}

// Returns the first forward intersection with a toroidal circle, if in range.
double ray_circle_distance(
    const Vec2 origin, const Vec2 direction, const Vec2 center,
    const double radius, const double range,
    const double world_width, const double world_height) noexcept
{
    const Vec2 displacement =
        toroidal_displacement(origin, center, world_width, world_height);
    const double projection = displacement.x * direction.x + displacement.y * direction.y;
    const double center_distance_squared =
        displacement.x * displacement.x + displacement.y * displacement.y;
    const double perpendicular_squared = center_distance_squared - projection * projection;
    const double radius_squared = radius * radius;
    if (perpendicular_squared > radius_squared || projection + radius < 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    double distance = projection - std::sqrt(std::max(0.0, radius_squared - perpendicular_squared));
    if (distance < 0.0) distance = 0.0;
    return distance <= range ? distance : std::numeric_limits<double>::infinity();
}

void validate_snapshot(const SimulationSnapshot& snapshot)
{
    validate_config(snapshot.config);
    if (snapshot.agents.size() < checked_size(snapshot.config.minimum_population, "minimum population")) {
        throw std::invalid_argument("snapshot population is below its minimum");
    }
    if (snapshot.food.size() > checked_size(
            snapshot.config.boosted_food_count, "boosted food count")) {
        throw std::invalid_argument("snapshot food exceeds its maximum configured count");
    }
    std::unordered_set<std::uint64_t> agent_ids;
    std::uint64_t maximum_agent_id = 0;
    for (const Agent& agent : snapshot.agents) {
        require_finite(agent.position.x, "agent x position");
        require_finite(agent.position.y, "agent y position");
        require_finite(agent.direction, "agent direction");
        require_finite(agent.energy, "agent energy");
        require_finite(agent.color.red, "agent red");
        require_finite(agent.color.green, "agent green");
        require_finite(agent.color.blue, "agent blue");
        require_finite(agent.mutation_rate, "agent mutation rate");
        require_finite(agent.mutation_strength, "agent mutation strength");
        require_finite(agent.prior_bite_damage, "agent bite damage");
        if (agent.id == 0 || !agent_ids.insert(agent.id).second || !valid_diet(agent.diet)
            || agent.position.x < 0.0 || agent.position.x >= snapshot.config.world_width
            || agent.position.y < 0.0 || agent.position.y >= snapshot.config.world_height
            || agent.direction < 0.0 || agent.direction >= full_turn || agent.energy <= 0.0
            || agent.color.red < 0.0 || agent.color.red > 1.0
            || agent.color.green < 0.0 || agent.color.green > 1.0
            || agent.color.blue < 0.0 || agent.color.blue > 1.0
            || agent.mutation_rate < minimum_mutation_rate || agent.mutation_rate > 1.0
            || agent.mutation_strength < minimum_mutation_strength
            || agent.mutation_strength > 1.0
            || agent.prior_bite_damage < 0.0) {
            throw std::invalid_argument("snapshot contains an invalid agent");
        }
        for (const double parameter : agent.brain) {
            require_finite(parameter, "brain parameter");
            if (parameter < snapshot.config.brain_parameter_minimum
                || parameter > snapshot.config.brain_parameter_maximum) {
                throw std::invalid_argument("snapshot brain parameter exceeds configured limits");
            }
        }
        if (agent.brain_structure.founder_fast_path > 1) {
            throw std::invalid_argument("snapshot brain has invalid fast-path state");
        }
        if (agent.brain_structure.founder_fast_path != 0
            && agent.brain_structure != founder_brain_structure()) {
            throw std::invalid_argument("snapshot brain has inconsistent fast-path topology");
        }
        for (const std::uint8_t active : agent.brain_structure.hidden_active) {
            if (active > 1) throw std::invalid_argument("snapshot brain has invalid neuron state");
        }
        const auto validate_mask = [](const auto& mask) {
            return std::ranges::all_of(mask, [](const std::uint8_t enabled) {
                return enabled <= 1;
            });
        };
        if (!validate_mask(agent.brain_structure.input_hidden_enabled)
            || !validate_mask(agent.brain_structure.hidden_output_enabled)
            || !validate_mask(agent.brain_structure.recurrent_enabled)) {
            throw std::invalid_argument("snapshot brain has invalid connection state");
        }
        for (const double weight : agent.brain_structure.recurrent_weights) {
            require_finite(weight, "recurrent brain parameter");
            if (weight < snapshot.config.brain_parameter_minimum
                || weight > snapshot.config.brain_parameter_maximum) {
                throw std::invalid_argument(
                    "snapshot recurrent parameter exceeds configured limits");
            }
        }
        for (const double value : agent.brain_state.previous_hidden) {
            require_finite(value, "previous recurrent brain state");
            if (value < -1.0 || value > 1.0) {
                throw std::invalid_argument("snapshot recurrent state exceeds activation range");
            }
        }
        for (const double value : agent.brain_state.next_hidden) {
            require_finite(value, "next recurrent brain state");
            if (value < -1.0 || value > 1.0) {
                throw std::invalid_argument("snapshot recurrent state exceeds activation range");
            }
        }
        maximum_agent_id = std::max(maximum_agent_id, agent.id);
    }
    if (snapshot.next_agent_id == 0 || snapshot.next_agent_id <= maximum_agent_id) {
        throw std::invalid_argument("snapshot next agent ID is invalid");
    }
    std::unordered_set<std::uint64_t> food_ids;
    std::uint64_t maximum_food_id = 0;
    for (const Food& item : snapshot.food) {
        require_finite(item.position.x, "food x position");
        require_finite(item.position.y, "food y position");
        require_finite(item.energy, "food energy");
        if (item.id == 0 || !food_ids.insert(item.id).second
            || item.position.x < 0.0 || item.position.x >= snapshot.config.world_width
            || item.position.y < 0.0 || item.position.y >= snapshot.config.world_height
            || item.energy <= 0.0) {
            throw std::invalid_argument("snapshot contains invalid food");
        }
        maximum_food_id = std::max(maximum_food_id, item.id);
    }
    if (snapshot.next_food_id == 0 || snapshot.next_food_id <= maximum_food_id) {
        throw std::invalid_argument("snapshot next food ID is invalid");
    }
}

} // namespace

Simulation::Simulation(const SimulationConfig& config,
    const SimulationExecutionConfig execution)
    : config_(config), random_(config.seed),
      execution_thread_count_(resolve_execution_threads(execution)),
      brain_backend_(execution.brain_backend)
{
    validate_config(config_);
    if (!brain_backend_available(brain_backend_)) {
        throw std::runtime_error("requested brain backend is unavailable");
    }
    configure_spatial_index();
    agents_.reserve(checked_size(config_.initial_population, "initial population"));
    food_.reserve(checked_size(config_.boosted_food_count, "boosted food count"));
    // Ordinary founders are herbivores because carnivore evolution requires an
    // established prey population; carnivores enter later through the separate rule.
    while (agents_.size() < config_.initial_population) {
        agents_.push_back(create_random_agent(Diet::herbivore));
    }
    const std::uint64_t initial_food_target = agents_.size()
            < config_.food_boost_population_threshold
        ? config_.boosted_food_count
        : config_.target_food_count;
    // Initial food is complete; only replacement spawning is rate limited.
    while (food_.size() < initial_food_target) food_.push_back(create_random_food());
}

Simulation::Simulation(SimulationSnapshot snapshot, RestoredSnapshotTag,
    const SimulationExecutionConfig execution)
    : config_(snapshot.config), current_tick_(snapshot.current_tick),
      random_(Pcg32::from_state(snapshot.random_state)), next_agent_id_(snapshot.next_agent_id),
      next_food_id_(snapshot.next_food_id), births_(snapshot.births),
      introduced_agents_(snapshot.introduced_agents), deaths_(snapshot.deaths),
      agents_eaten_(snapshot.agents_eaten),
      agents_(std::move(snapshot.agents)), food_(std::move(snapshot.food)),
      execution_thread_count_(resolve_execution_threads(execution)),
      brain_backend_(execution.brain_backend)
{
    if (!brain_backend_available(brain_backend_)) {
        throw std::runtime_error("requested brain backend is unavailable");
    }
    configure_spatial_index();
}

Simulation Simulation::from_snapshot(SimulationSnapshot snapshot,
    const SimulationExecutionConfig execution)
{
    // Old or externally produced values at zero are repaired so a lineage can mutate again.
    snapshot.config.founder_mutation_rate_minimum = std::max(
        snapshot.config.founder_mutation_rate_minimum, minimum_mutation_rate);
    snapshot.config.founder_mutation_rate_maximum = std::max(
        snapshot.config.founder_mutation_rate_maximum,
        snapshot.config.founder_mutation_rate_minimum);
    snapshot.config.founder_mutation_strength_minimum = std::max(
        snapshot.config.founder_mutation_strength_minimum, minimum_mutation_strength);
    snapshot.config.founder_mutation_strength_maximum = std::max(
        snapshot.config.founder_mutation_strength_maximum,
        snapshot.config.founder_mutation_strength_minimum);
    for (Agent& agent : snapshot.agents) {
        agent.mutation_rate = std::max(agent.mutation_rate, minimum_mutation_rate);
        agent.mutation_strength = std::max(agent.mutation_strength, minimum_mutation_strength);
    }
    validate_snapshot(snapshot);
    return Simulation(std::move(snapshot), RestoredSnapshotTag {}, execution);
}

void Simulation::configure_spatial_index()
{
    // Eye-range quartering keeps neighborhood queries narrow without creating tiny cells.
    const double desired_cell_size = std::max({config_.agent_radius * 2.0,
        config_.food_radius * 2.0, config_.eye_range * 0.25});
    spatial_columns_ = std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(config_.world_width / desired_cell_size)));
    spatial_rows_ = std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(config_.world_height / desired_cell_size)));
    spatial_cell_width_ = config_.world_width / static_cast<double>(spatial_columns_);
    spatial_cell_height_ = config_.world_height / static_cast<double>(spatial_rows_);
    spatial_cells_.resize(spatial_columns_ * spatial_rows_);
    diagnostics_.spatial_columns = spatial_columns_;
    diagnostics_.spatial_rows = spatial_rows_;
    diagnostics_.execution_threads = execution_thread_count_;
}

void Simulation::rebuild_spatial_index()
{
    for (SpatialCell& cell : spatial_cells_) {
        cell.agents.clear();
        cell.food.clear();
    }
    const auto cell_index = [&](const Vec2 position) {
        const std::size_t column = std::min(spatial_columns_ - 1,
            static_cast<std::size_t>(position.x / spatial_cell_width_));
        const std::size_t row = std::min(spatial_rows_ - 1,
            static_cast<std::size_t>(position.y / spatial_cell_height_));
        return row * spatial_columns_ + column;
    };
    for (std::size_t index = 0; index < agents_.size(); ++index) {
        spatial_cells_[cell_index(agents_[index].position)].agents.push_back(index);
    }
    for (std::size_t index = 0; index < food_.size(); ++index) {
        spatial_cells_[cell_index(food_[index].position)].food.push_back(index);
    }
}

void Simulation::collect_spatial_candidates(const Vec2 center, const double radius,
    std::vector<std::size_t>& agent_indices,
    std::vector<std::size_t>& food_indices) const
{
    agent_indices.clear();
    food_indices.clear();
    const long long minimum_column = static_cast<long long>(
        std::floor((center.x - radius) / spatial_cell_width_));
    const long long maximum_column = static_cast<long long>(
        std::floor((center.x + radius) / spatial_cell_width_));
    const long long minimum_row = static_cast<long long>(
        std::floor((center.y - radius) / spatial_cell_height_));
    const long long maximum_row = static_cast<long long>(
        std::floor((center.y + radius) / spatial_cell_height_));

    const auto wrap_index = [](const long long value, const std::size_t count) {
        const long long signed_count = static_cast<long long>(count);
        const long long remainder = value % signed_count;
        return static_cast<std::size_t>(remainder < 0 ? remainder + signed_count : remainder);
    };
    const bool all_columns = maximum_column - minimum_column + 1
        >= static_cast<long long>(spatial_columns_);
    const bool all_rows = maximum_row - minimum_row + 1
        >= static_cast<long long>(spatial_rows_);
    const long long column_count = all_columns
        ? static_cast<long long>(spatial_columns_)
        : maximum_column - minimum_column + 1;
    const long long row_count = all_rows
        ? static_cast<long long>(spatial_rows_)
        : maximum_row - minimum_row + 1;
    for (long long row_offset = 0; row_offset < row_count; ++row_offset) {
        const std::size_t row = all_rows
            ? static_cast<std::size_t>(row_offset)
            : wrap_index(minimum_row + row_offset, spatial_rows_);
        for (long long column_offset = 0; column_offset < column_count; ++column_offset) {
            const std::size_t column = all_columns
                ? static_cast<std::size_t>(column_offset)
                : wrap_index(minimum_column + column_offset, spatial_columns_);
            const SpatialCell& cell = spatial_cells_[row * spatial_columns_ + column];
            agent_indices.insert(agent_indices.end(), cell.agents.begin(), cell.agents.end());
            food_indices.insert(food_indices.end(), cell.food.begin(), cell.food.end());
        }
    }
}

Agent Simulation::create_random_agent(const Diet diet)
{
    if (next_agent_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("agent ID space exhausted");
    }
    Agent agent {
        .id = next_agent_id_++,
        .position = {.x = random_.uniform(0.0, config_.world_width),
            .y = random_.uniform(0.0, config_.world_height)},
        .direction = random_.uniform(0.0, full_turn), .energy = config_.initial_energy,
        .diet = diet,
        .color = {.red = random_.unit_interval(), .green = random_.unit_interval(),
            .blue = random_.unit_interval()},
        .mutation_rate = random_.uniform(config_.founder_mutation_rate_minimum,
            config_.founder_mutation_rate_maximum),
        .mutation_strength = random_.uniform(config_.founder_mutation_strength_minimum,
            config_.founder_mutation_strength_maximum),
    };
    // Preserve the original 26-to-8-to-3 initialization order for founder behavior.
    for (std::size_t hidden = 0; hidden < brain_founder_hidden_count; ++hidden) {
        for (std::size_t input = 0; input < brain_input_count; ++input) {
            agent.brain[hidden * brain_input_count + input] = random_.uniform(
                config_.initial_brain_parameter_minimum,
                config_.initial_brain_parameter_maximum);
        }
    }
    for (std::size_t hidden = 0; hidden < brain_founder_hidden_count; ++hidden) {
        agent.brain[hidden_bias_offset + hidden] = random_.uniform(
            config_.initial_brain_parameter_minimum,
            config_.initial_brain_parameter_maximum);
    }
    for (std::size_t output = 0; output < brain_output_count; ++output) {
        for (std::size_t hidden = 0; hidden < brain_founder_hidden_count; ++hidden) {
            agent.brain[hidden_output_weight_offset + output * brain_hidden_count + hidden]
                = random_.uniform(config_.initial_brain_parameter_minimum,
                    config_.initial_brain_parameter_maximum);
        }
    }
    for (std::size_t output = 0; output < brain_output_count; ++output) {
        agent.brain[output_bias_offset + output] = random_.uniform(
            config_.initial_brain_parameter_minimum,
            config_.initial_brain_parameter_maximum);
    }
    // Dormant genes are initialized after the active founder network and remain disabled.
    for (std::size_t hidden = brain_founder_hidden_count; hidden < brain_hidden_count;
         ++hidden) {
        for (std::size_t input = 0; input < brain_input_count; ++input) {
            agent.brain[hidden * brain_input_count + input] = random_.uniform(
                config_.initial_brain_parameter_minimum,
                config_.initial_brain_parameter_maximum);
        }
        agent.brain[hidden_bias_offset + hidden] = random_.uniform(
            config_.initial_brain_parameter_minimum,
            config_.initial_brain_parameter_maximum);
        for (std::size_t output = 0; output < brain_output_count; ++output) {
            agent.brain[hidden_output_weight_offset + output * brain_hidden_count + hidden]
                = random_.uniform(config_.initial_brain_parameter_minimum,
                    config_.initial_brain_parameter_maximum);
        }
    }
    return agent;
}

Food Simulation::create_random_food()
{
    if (next_food_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("food ID space exhausted");
    }
    return {.id = next_food_id_++,
        .position = {.x = random_.uniform(0.0, config_.world_width),
            .y = random_.uniform(0.0, config_.world_height)},
        .energy = config_.food_energy};
}

BrainInputs Simulation::sense_agent(const Agent& observer,
    const std::span<const std::size_t> agent_candidates,
    const std::span<const std::size_t> food_candidates) const
{
    BrainInputs inputs {
        .energy = std::clamp(observer.energy / config_.reproduction_threshold, 0.0, 1.0),
        .damage = std::clamp(observer.prior_bite_damage / config_.bite_amount_per_tick, 0.0, 1.0),
    };
    const Vec2 forward {.x = std::cos(observer.direction), .y = std::sin(observer.direction)};
    const Vec2 left {.x = -forward.y, .y = forward.x};

    for (std::size_t ray_index = 0; ray_index < vision_ray_count; ++ray_index) {
        const bool left_eye = ray_index < rays_per_eye;
        const double side = left_eye ? 0.5 : -0.5;
        const Vec2 origin {
            .x = wrap_coordinate(observer.position.x
                    + forward.x * config_.agent_radius * 0.60
                    + left.x * config_.agent_radius * side,
                config_.world_width),
            .y = wrap_coordinate(observer.position.y
                    + forward.y * config_.agent_radius * 0.60
                    + left.y * config_.agent_radius * side,
                config_.world_height),
        };
        const double angle = observer.direction + ray_angle_offsets[ray_index];
        const Vec2 ray {.x = std::cos(angle), .y = std::sin(angle)};
        double best_distance = std::numeric_limits<double>::infinity();
        int best_layer = -1;
        std::uint64_t best_id = 0;
        AgentColor best_color {};

        const auto consider = [&](const double distance, const int layer,
                                  const std::uint64_t id, const AgentColor color) {
            // Equal-distance hits use render Z-order: agents over food, then highest ID.
            if (distance < best_distance
                || (distance == best_distance
                    && (layer > best_layer || (layer == best_layer && id > best_id)))) {
                best_distance = distance;
                best_layer = layer;
                best_id = id;
                best_color = color;
            }
        };
        for (const std::size_t target_index : agent_candidates) {
            const Agent& target = agents_[target_index];
            if (target.id == observer.id) continue;
            consider(ray_circle_distance(origin, ray, target.position, config_.agent_radius,
                config_.eye_range, config_.world_width, config_.world_height),
                1, target.id, target.color);
        }
        for (const std::size_t food_index : food_candidates) {
            const Food& item = food_[food_index];
            consider(ray_circle_distance(origin, ray, item.position, config_.food_radius,
                config_.eye_range, config_.world_width, config_.world_height),
                0, item.id, plant_food_color);
        }
        if (std::isfinite(best_distance)) {
            inputs.vision[ray_index] = {.red = best_color.red, .green = best_color.green,
                .blue = best_color.blue,
                .proximity = 1.0 - best_distance / config_.eye_range};
        }
    }
    return inputs;
}

bool Simulation::synchronize_brain_batch()
{
    if (!brain_batch_dirty_ && brain_parameters_batch_.size() == agents_.size()) {
        return false;
    }

    brain_parameters_batch_.resize(agents_.size());
    brain_ids_batch_.resize(agents_.size());
    brain_structures_batch_.resize(agents_.size());
    brain_states_batch_.resize(agents_.size());
    brain_inputs_batch_.resize(agents_.size());
    brain_outputs_batch_.resize(agents_.size());
    for (std::size_t index = 0; index < agents_.size(); ++index) {
        brain_ids_batch_[index] = agents_[index].id;
        brain_parameters_batch_[index] = agents_[index].brain;
        brain_structures_batch_[index] = agents_[index].brain_structure;
        brain_states_batch_[index] = agents_[index].brain_state;
    }
    brain_batch_dirty_ = false;
    return true;
}

std::vector<Simulation::AgentAction> Simulation::evaluate_agent_actions()
{
    const bool brain_batch_rebuilt = synchronize_brain_batch();
    std::vector<AgentAction> actions(agents_.size());
    std::vector<std::uint64_t> candidate_tests(agents_.size(), 0);
    std::vector<std::uint64_t> brute_force_tests(agents_.size(), 0);
    const double query_radius = config_.eye_range
        + std::hypot(0.60, 0.50) * config_.agent_radius
        + std::max(config_.agent_radius, config_.food_radius);
    // Very small populations stay serial because waking workers would dominate their work.
    const std::size_t effective_threads = agents_.size() < 64
        ? 1
        : std::min(execution_thread_count_, std::max<std::size_t>(2, agents_.size() / 16));
    diagnostics_.execution_threads = effective_threads;
    const Clock::time_point sensing_start = Clock::now();
    detail::parallel_executor().for_each_index(agents_.size(), effective_threads,
        [&](const std::size_t index) {
            thread_local std::vector<std::size_t> agent_candidates;
            thread_local std::vector<std::size_t> food_candidates;
            collect_spatial_candidates(agents_[index].position, query_radius,
                agent_candidates, food_candidates);
            const Agent& agent = agents_[index];
            brain_inputs_batch_[index] = sense_agent(
                agent, agent_candidates, food_candidates);
            const std::size_t other_candidates = agent_candidates.empty()
                ? 0
                : agent_candidates.size() - 1;
            candidate_tests[index] = static_cast<std::uint64_t>(vision_ray_count)
                * static_cast<std::uint64_t>(other_candidates + food_candidates.size());
            brute_force_tests[index] = static_cast<std::uint64_t>(vision_ray_count)
                * static_cast<std::uint64_t>(agents_.size() - 1 + food_.size());
        });
    const Clock::time_point brain_start = Clock::now();
    diagnostics_.sensing_milliseconds = elapsed_milliseconds(sensing_start, brain_start);
    evaluate_brain_batch(brain_backend_,
        BrainBatch {
            .agent_ids = brain_ids_batch_,
            .parameters = brain_parameters_batch_,
            .structures = brain_structures_batch_,
            .states = brain_states_batch_,
            .inputs = brain_inputs_batch_,
            .outputs = brain_outputs_batch_,
            .cache_identity = this,
            .population_changed = brain_batch_rebuilt,
            .state_changed = brain_backend_cache_reset_,
            .reset_cache = brain_backend_cache_reset_,
        },
        execution_thread_count_);
    brain_backend_cache_reset_ = false;
    diagnostics_.brain_milliseconds = elapsed_milliseconds(brain_start, Clock::now());
    diagnostics_.sensing_brain_milliseconds = diagnostics_.sensing_milliseconds
        + diagnostics_.brain_milliseconds;
    diagnostics_.vision_candidate_tests = 0;
    diagnostics_.vision_brute_force_tests = 0;
    for (std::size_t index = 0; index < agents_.size(); ++index) {
        agents_[index].brain_state = brain_states_batch_[index];
        const BrainOutputs& output = brain_outputs_batch_[index];
        actions[index] = {.agent_id = agents_[index].id, .turn = output.turn,
            .move = output.move, .eat = output.eat >= config_.eat_threshold};
        diagnostics_.vision_candidate_tests += candidate_tests[index];
        diagnostics_.vision_brute_force_tests += brute_force_tests[index];
    }
    return actions;
}

void Simulation::move_agents_and_charge_energy(const std::span<const AgentAction> actions)
{
    for (std::size_t index = 0; index < agents_.size(); ++index) {
        Agent& agent = agents_[index];
        const AgentAction& action = actions[index];
        agent.direction = normalize_direction(agent.direction
            + action.turn * config_.maximum_turn_per_tick);
        const double distance = action.move * config_.maximum_movement_per_tick;
        agent.position.x = wrap_coordinate(
            agent.position.x + std::cos(agent.direction) * distance, config_.world_width);
        agent.position.y = wrap_coordinate(
            agent.position.y + std::sin(agent.direction) * distance, config_.world_height);
        agent.energy -= config_.living_energy_cost + config_.movement_energy_cost * distance
            + (action.eat ? config_.eat_attempt_energy_cost : 0.0);
        ++agent.age;
    }
}

void Simulation::remove_dead_agents()
{
    const std::size_t before = agents_.size();
    std::erase_if(agents_, [](const Agent& agent) { return agent.energy <= 0.0; });
    if (agents_.size() != before) brain_batch_dirty_ = true;
    deaths_ += static_cast<std::uint64_t>(before - agents_.size());
}

void Simulation::resolve_bites(const std::span<const AgentAction> actions)
{
    struct BiteRequest { std::size_t eater; bool target_is_agent; std::size_t target; };
    std::vector<BiteRequest> requests;
    requests.reserve(actions.size());
    std::unordered_map<std::uint64_t, std::size_t> agent_index_by_id;
    agent_index_by_id.reserve(agents_.size());
    for (std::size_t index = 0; index < agents_.size(); ++index) {
        agent_index_by_id.emplace(agents_[index].id, index);
    }
    std::vector<std::size_t> agent_candidates;
    std::vector<std::size_t> food_candidates;
    diagnostics_.bite_candidate_tests = 0;
    diagnostics_.bite_brute_force_tests = 0;
    for (const AgentAction& action : actions) {
        if (!action.eat) continue;
        const auto eater_found = agent_index_by_id.find(action.agent_id);
        if (eater_found == agent_index_by_id.end()) continue;
        const std::size_t eater_index = eater_found->second;
        const Agent& eater = agents_[eater_index];
        const Vec2 mouth {
            .x = wrap_coordinate(eater.position.x
                    + std::cos(eater.direction) * config_.agent_radius,
                config_.world_width),
            .y = wrap_coordinate(eater.position.y
                    + std::sin(eater.direction) * config_.agent_radius,
                config_.world_height),
        };

        collect_spatial_candidates(mouth,
            std::max(config_.agent_radius, config_.food_radius),
            agent_candidates, food_candidates);
        diagnostics_.bite_brute_force_tests += agents_.size();
        diagnostics_.bite_candidate_tests += agent_candidates.size();
        std::size_t top_agent_index = agents_.size();
        for (const std::size_t target_index : agent_candidates) {
            const Agent& target = agents_[target_index];
            if (target.id != eater.id
                && toroidal_distance_squared(mouth, target.position,
                       config_.world_width, config_.world_height)
                    <= config_.agent_radius * config_.agent_radius
                && (top_agent_index == agents_.size()
                    || target.id > agents_[top_agent_index].id)) {
                top_agent_index = target_index;
            }
        }
        std::size_t top_food_index = food_.size();
        if (top_agent_index == agents_.size()) {
            diagnostics_.bite_brute_force_tests += food_.size();
            diagnostics_.bite_candidate_tests += food_candidates.size();
            for (const std::size_t food_index : food_candidates) {
                const Food& item = food_[food_index];
                if (toroidal_distance_squared(mouth, item.position,
                        config_.world_width, config_.world_height)
                        <= config_.food_radius * config_.food_radius
                    && (top_food_index == food_.size()
                        || item.id > food_[top_food_index].id)) {
                    top_food_index = food_index;
                }
            }
        }
        // Diet is checked only after topmost targeting, so a wrong target blocks objects below it.
        if (top_agent_index != agents_.size() && eater.diet == Diet::carnivore) {
            requests.push_back({eater_index, true, top_agent_index});
        } else if (top_food_index != food_.size() && eater.diet == Diet::herbivore) {
            requests.push_back({eater_index, false, top_food_index});
        }
    }

    std::vector<std::size_t> agent_request_counts(agents_.size(), 0);
    std::vector<std::size_t> food_request_counts(food_.size(), 0);
    for (const BiteRequest& request : requests) {
        if (request.target_is_agent) ++agent_request_counts[request.target];
        else ++food_request_counts[request.target];
    }
    std::vector<double> agent_scale(agents_.size(), 0.0);
    std::vector<double> food_scale(food_.size(), 0.0);
    for (std::size_t index = 0; index < agents_.size(); ++index) {
        if (agent_request_counts[index] != 0) {
            agent_scale[index] = std::min(1.0, agents_[index].energy
                / (config_.bite_amount_per_tick
                    * static_cast<double>(agent_request_counts[index])));
        }
    }
    for (std::size_t index = 0; index < food_.size(); ++index) {
        if (food_request_counts[index] != 0) {
            food_scale[index] = std::min(1.0, food_[index].energy
                / (config_.bite_amount_per_tick
                    * static_cast<double>(food_request_counts[index])));
        }
    }

    std::vector<double> agent_energy_delta(agents_.size(), 0.0);
    std::vector<double> agent_damage(agents_.size(), 0.0);
    std::vector<double> food_energy_delta(food_.size(), 0.0);
    for (const BiteRequest& request : requests) {
        const double amount = config_.bite_amount_per_tick
            * (request.target_is_agent
                ? agent_scale[request.target]
                : food_scale[request.target]);
        agent_energy_delta[request.eater] += amount;
        if (request.target_is_agent) {
            agent_energy_delta[request.target] -= amount;
            agent_damage[request.target] += amount;
        } else {
            food_energy_delta[request.target] -= amount;
        }
    }
    for (std::size_t index = 0; index < agents_.size(); ++index) {
        agents_[index].energy += agent_energy_delta[index];
        agents_[index].prior_bite_damage = agent_damage[index];
    }
    for (std::size_t index = 0; index < food_.size(); ++index) {
        food_[index].energy += food_energy_delta[index];
    }
    const std::size_t agents_before = agents_.size();
    std::erase_if(agents_, [&](const Agent& agent) {
        if (agent.energy > 0.0) return false;
        ++agents_eaten_;
        return true;
    });
    if (agents_.size() != agents_before) brain_batch_dirty_ = true;
    deaths_ += static_cast<std::uint64_t>(agents_before - agents_.size());
    std::erase_if(food_, [](const Food& item) { return item.energy <= 0.0; });
}

void Simulation::reproduce_eligible_agents()
{
    const std::size_t parent_count = agents_.size();
    std::vector<Agent> children;
    for (std::size_t index = 0; index < parent_count; ++index) {
        Agent& parent = agents_[index];
        if (parent.energy < config_.reproduction_threshold) continue;
        if (next_agent_id_ == std::numeric_limits<std::uint64_t>::max()
            || parent.generation == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("agent lineage state exhausted");
        }
        parent.energy *= 0.5;
        Agent child = parent;
        child.id = next_agent_id_++;
        child.position = {
            .x = wrap_coordinate(parent.position.x
                    - std::cos(parent.direction) * config_.agent_radius * 2.0,
                config_.world_width),
            .y = wrap_coordinate(parent.position.y
                    - std::sin(parent.direction) * config_.agent_radius * 2.0,
                config_.world_height),
        };
        child.direction = normalize_direction(parent.direction + std::numbers::pi_v<double>);
        child.age = 0;
        child.generation = parent.generation + 1;
        child.prior_bite_damage = 0.0;
        child.brain_state = {};
        const double parent_rate = parent.mutation_rate;
        const double parent_strength = parent.mutation_strength;
        const auto mutate = [&](double& gene, const double minimum, const double maximum,
                                const double category_scale) {
            if (random_.unit_interval() < parent_rate) {
                gene = std::clamp(gene + random_.uniform(-1.0, 1.0) * parent_strength
                    * category_scale, minimum, maximum);
            }
        };
        // Fixed gene and topology order is part of seeded determinism; diet is absent.
        for (double& parameter : child.brain) {
            mutate(parameter, config_.brain_parameter_minimum,
                config_.brain_parameter_maximum, config_.brain_mutation_scale);
        }
        for (std::size_t hidden = brain_founder_hidden_count;
             hidden < brain_hidden_count; ++hidden) {
            if (child.brain_structure.hidden_active[hidden] != 0
                || random_.unit_interval() >= parent_rate) {
                continue;
            }
            child.brain_structure.hidden_active[hidden] = 1;
            child.brain_structure.founder_fast_path = 0;
            const std::size_t input = random_.bounded(
                static_cast<std::uint32_t>(brain_input_count));
            const std::size_t output = random_.bounded(
                static_cast<std::uint32_t>(brain_output_count));
            const std::size_t incoming = hidden * brain_input_count + input;
            const std::size_t outgoing = output * brain_hidden_count + hidden;
            child.brain_structure.input_hidden_enabled[incoming] = 1;
            child.brain_structure.hidden_output_enabled[outgoing] = 1;
            const double connection_scale = parent_strength * config_.brain_mutation_scale;
            child.brain[incoming] = std::clamp(random_.uniform(-connection_scale,
                connection_scale), config_.brain_parameter_minimum,
                config_.brain_parameter_maximum);
            child.brain[hidden_output_weight_offset + outgoing] = std::clamp(
                random_.uniform(-connection_scale, connection_scale),
                config_.brain_parameter_minimum, config_.brain_parameter_maximum);
        }
        for (std::size_t target = 0; target < brain_hidden_count; ++target) {
            if (child.brain_structure.hidden_active[target] == 0) continue;
            for (std::size_t source = 0; source < brain_hidden_count; ++source) {
                if (child.brain_structure.hidden_active[source] == 0) continue;
                const std::size_t connection = target * brain_hidden_count + source;
                if (child.brain_structure.recurrent_enabled[connection] == 0) {
                    if (random_.unit_interval() >= parent_rate) continue;
                    child.brain_structure.recurrent_enabled[connection] = 1;
                    child.brain_structure.founder_fast_path = 0;
                    const double connection_scale =
                        parent_strength * config_.brain_mutation_scale;
                    child.brain_structure.recurrent_weights[connection] = std::clamp(
                        random_.uniform(-connection_scale, connection_scale),
                        config_.brain_parameter_minimum,
                        config_.brain_parameter_maximum);
                } else {
                    mutate(child.brain_structure.recurrent_weights[connection],
                        config_.brain_parameter_minimum,
                        config_.brain_parameter_maximum,
                        config_.brain_mutation_scale);
                }
            }
        }
        mutate(child.color.red, 0.0, 1.0, config_.color_mutation_scale);
        mutate(child.color.green, 0.0, 1.0, config_.color_mutation_scale);
        mutate(child.color.blue, 0.0, 1.0, config_.color_mutation_scale);
        mutate(child.mutation_rate, minimum_mutation_rate, 1.0,
            config_.mutation_rate_mutation_scale);
        mutate(child.mutation_strength, minimum_mutation_strength, 1.0,
            config_.mutation_strength_mutation_scale);
        child.mutation_rate = std::max(child.mutation_rate, minimum_mutation_rate);
        child.mutation_strength = std::max(child.mutation_strength,
            minimum_mutation_strength);
        children.push_back(child);
        ++births_;
    }
    if (!children.empty()) {
        agents_.insert(agents_.end(), children.begin(), children.end());
        brain_batch_dirty_ = true;
    }
}

void Simulation::restore_minimum_population()
{
    const std::size_t minimum = checked_size(config_.minimum_population, "minimum population");
    const std::size_t before = agents_.size();
    // Restoration follows the founder rule so it cannot bypass the prey-gated
    // carnivore introduction policy.
    while (agents_.size() < minimum) {
        agents_.push_back(create_random_agent(Diet::herbivore));
        ++introduced_agents_;
    }
    if (agents_.size() != before) brain_batch_dirty_ = true;
}

void Simulation::introduce_carnivores_if_supported()
{
    const std::uint64_t completed_tick = current_tick_ + 1;
    // Tick modulo is the entire schedule; no transition-sensitive cooldown state exists.
    if (completed_tick % config_.carnivore_introduction_interval_ticks != 0) return;

    const std::uint64_t population = static_cast<std::uint64_t>(agents_.size());
    const std::uint64_t herbivores = static_cast<std::uint64_t>(std::ranges::count(
        agents_, Diet::herbivore, &Agent::diet));
    const std::uint64_t carnivores = population - herbivores;
    if (herbivores < config_.carnivore_introduction_herbivore_threshold
        || carnivores >= config_.carnivore_introduction_ceiling
        || population >= config_.carnivore_introduction_population_ceiling) {
        return;
    }

    // Independent ceilings prevent the periodic cohort from crossing either limit.
    const std::uint64_t count = std::min({config_.carnivore_introduction_batch,
        config_.carnivore_introduction_ceiling - carnivores,
        config_.carnivore_introduction_population_ceiling - population});
    for (std::uint64_t index = 0; index < count; ++index) {
        agents_.push_back(create_random_agent(Diet::carnivore));
        ++introduced_agents_;
    }
    if (count != 0) brain_batch_dirty_ = true;
}

void Simulation::replenish_food_if_allowed()
{
    if (agents_.size() >= config_.food_population_threshold) return;
    const std::uint64_t configured_target = agents_.size()
            < config_.food_boost_population_threshold
        ? config_.boosted_food_count
        : config_.target_food_count;
    const std::size_t target = checked_size(configured_target, "food target");
    // Recovered populations keep excess food until agents consume it naturally.
    const std::size_t missing = target > food_.size() ? target - food_.size() : 0;
    const std::size_t additions = std::min(missing,
        checked_size(config_.maximum_new_food_per_tick, "maximum new food per tick"));
    for (std::size_t index = 0; index < additions; ++index) {
        food_.push_back(create_random_food());
    }
}

void Simulation::regrow_food_if_due()
{
    const std::uint64_t completed_tick = current_tick_ + 1;
    if (completed_tick % config_.food_regrowth_interval_ticks != 0) return;
    for (Food& item : food_) {
        item.energy = std::min(item.energy + config_.food_regrowth_amount,
            config_.food_energy);
    }
}

void Simulation::tick()
{
    if (current_tick_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("simulation tick counter exhausted");
    }
    const Clock::time_point tick_start = Clock::now();
    const Clock::time_point first_index_start = tick_start;
    rebuild_spatial_index();
    const Clock::time_point sensing_start = Clock::now();
    diagnostics_.spatial_index_milliseconds =
        elapsed_milliseconds(first_index_start, sensing_start);
    const std::vector<AgentAction> actions = evaluate_agent_actions();
    const Clock::time_point movement_start = Clock::now();
    diagnostics_.sensing_brain_milliseconds = diagnostics_.sensing_milliseconds
        + diagnostics_.brain_milliseconds;
    move_agents_and_charge_energy(actions);
    remove_dead_agents();
    const Clock::time_point second_index_start = Clock::now();
    diagnostics_.movement_milliseconds =
        elapsed_milliseconds(movement_start, second_index_start);
    rebuild_spatial_index();
    const Clock::time_point bite_start = Clock::now();
    diagnostics_.spatial_index_milliseconds +=
        elapsed_milliseconds(second_index_start, bite_start);
    resolve_bites(actions);
    const Clock::time_point lifecycle_start = Clock::now();
    diagnostics_.bite_milliseconds = elapsed_milliseconds(bite_start, lifecycle_start);
    // Bites remove depleted food first, so a pulse only reaches surviving items.
    regrow_food_if_due();
    reproduce_eligible_agents();
    restore_minimum_population();
    introduce_carnivores_if_supported();
    replenish_food_if_allowed();
    ++current_tick_;
    const Clock::time_point tick_end = Clock::now();
    diagnostics_.lifecycle_milliseconds =
        elapsed_milliseconds(lifecycle_start, tick_end);
    diagnostics_.total_milliseconds = elapsed_milliseconds(tick_start, tick_end);
}

void Simulation::run_for(const std::uint64_t ticks)
{
    if (ticks > std::numeric_limits<std::uint64_t>::max() - current_tick_) {
        throw std::overflow_error("requested ticks exceed the simulation counter");
    }
    for (std::uint64_t index = 0; index < ticks; ++index) tick();
}

std::uint64_t Simulation::current_tick() const noexcept { return current_tick_; }
const SimulationConfig& Simulation::config() const noexcept { return config_; }
std::span<const Agent> Simulation::agents() const noexcept { return agents_; }
std::span<const Food> Simulation::food() const noexcept { return food_; }
const SimulationDiagnostics& Simulation::diagnostics() const noexcept { return diagnostics_; }
BrainBackendKind Simulation::brain_backend() const noexcept { return brain_backend_; }

void Simulation::set_brain_backend(const BrainBackendKind backend)
{
    if (!brain_backend_available(backend)) {
        throw std::runtime_error("requested brain backend is unavailable");
    }
    if (brain_backend_ == backend) return;
    brain_backend_ = backend;
    // A new backend must receive the complete genome and current recurrent state.
    brain_batch_dirty_ = true;
    brain_backend_cache_reset_ = true;
}

SimulationStats Simulation::stats() const noexcept
{
    const std::uint64_t herbivores = static_cast<std::uint64_t>(std::ranges::count(
        agents_, Diet::herbivore, &Agent::diet));
    return {.seed = config_.seed, .completed_ticks = current_tick_,
        .population = static_cast<std::uint64_t>(agents_.size()),
        .herbivores = herbivores,
        .carnivores = static_cast<std::uint64_t>(agents_.size()) - herbivores,
        .food = static_cast<std::uint64_t>(food_.size()), .births = births_,
        .introduced_agents = introduced_agents_, .deaths = deaths_,
        .agents_eaten = agents_eaten_};
}

SimulationSnapshot Simulation::snapshot() const
{
    return {.config = config_, .current_tick = current_tick_, .random_state = random_.state(),
        .next_agent_id = next_agent_id_, .next_food_id = next_food_id_, .births = births_,
        .introduced_agents = introduced_agents_, .deaths = deaths_,
        .agents_eaten = agents_eaten_,
        .agents = agents_, .food = food_};
}

} // namespace evobrain
