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

// Produces stable local terrain noise without consuming simulation RNG state.
std::uint64_t mix_bits(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

// Maps deterministic terrain noise to the half-open unit interval.
double noise_unit(const std::uint64_t value) noexcept
{
    return static_cast<double>(mix_bits(value) >> 11U)
        * (1.0 / 9007199254740992.0);
}

using Clock = std::chrono::steady_clock;

// A rounded, bounded continent in normalized toroidal coordinates. Its outline
// is rasterized once during generation; agents still use ordinary terrain cells.
struct IslandShape {
    Vec2 center;
    double radius_x;
    double radius_y;
    double angle;
    double phase;
};

// Returns elliptical distance and bearing using the nearest wrapped image.
std::pair<double, double> island_coordinates(const IslandShape& island, const Vec2 point)
{
    double dx = point.x - island.center.x;
    double dy = point.y - island.center.y;
    dx -= std::round(dx);
    dy -= std::round(dy);
    const double x = (dx * std::cos(island.angle) + dy * std::sin(island.angle)) / island.radius_x;
    const double y = (-dx * std::sin(island.angle) + dy * std::cos(island.angle)) / island.radius_y;
    return {std::hypot(x, y), std::atan2(y, x)};
}

// Signed radial coast offset in normalized world units; negative is inland.
// This is a smooth generation-only proxy, not an exact collision distance.
double island_coast_offset(const IslandShape& island, const Vec2 point)
{
    const auto [distance, bearing] = island_coordinates(island, point);
    const double outline = 1.0 + 0.06 * std::sin(3.0 * bearing + island.phase)
        + 0.03 * std::sin(5.0 * bearing - island.phase);
    return (distance - outline) * std::min(island.radius_x, island.radius_y);
}

// Gives rounded coastlines broad bulges without stretching them into world-spanning bands.
bool island_land(const std::vector<IslandShape>& islands, const Vec2 point)
{
    return std::ranges::any_of(islands, [&](const IslandShape& island) {
        return island_coast_offset(island, point) <= 0.0;
    });
}

// Spaced templates keep the requested major islands separate across the torus.
// Translation, orientation, aspect and coast phases vary deterministically by seed.
std::vector<IslandShape> make_islands(const std::size_t count, Pcg32& random)
{
    const std::array<Vec2, 3> triple {{{0.25, 0.25}, {0.75, 0.25}, {0.5, 0.75}}};
    const Vec2 offset {random.unit_interval(), random.unit_interval()};
    const bool transpose = random.bounded(2) != 0;
    std::vector<IslandShape> islands;
    for (std::size_t index = 0; index < count; ++index) {
        Vec2 center = count == 1 ? Vec2 {0.5, 0.5}
            : count == 2 ? Vec2 {0.25 + 0.5 * index, 0.5} : triple[index];
        if (transpose) std::swap(center.x, center.y);
        center.x = std::fmod(center.x + offset.x, 1.0);
        center.y = std::fmod(center.y + offset.y, 1.0);
        const double radius = count == 1 ? 0.34 : (count == 2 ? 0.21 : 0.19);
        islands.push_back({center, radius * random.uniform(0.94, 1.04),
            radius * random.uniform(0.94, 1.04), random.uniform(0.0, 2.0 * std::numbers::pi),
            random.uniform(0.0, 2.0 * std::numbers::pi)});
    }
    return islands;
}

// Partitions an axis into bounded rectangles, reserving enough extent for remaining
// regions. The final edge is exact so neither rounding gaps nor overlaps accumulate.
std::vector<double> biome_axis_edges(const double extent, const double minimum,
    const double maximum, const std::size_t preferred_count, Pcg32& random)
{
    const auto least = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(extent / maximum)));
    const auto most = std::max(least, static_cast<std::size_t>(std::floor(extent / minimum)));
    const auto count = std::clamp(preferred_count, least, most);
    std::vector<double> edges {0.0};
    for (std::size_t index = 0; index + 1 < count; ++index) {
        const double remaining = extent - edges.back();
        const double after = static_cast<double>(count - index - 1);
        const double lower = std::max(minimum, remaining - maximum * after);
        const double upper = std::min(maximum, remaining - minimum * after);
        edges.push_back(edges.back() + random.uniform(lower, std::max(lower, upper)));
    }
    edges.push_back(extent);
    return edges;
}

// Selects fertility independently of medium. Small/medium permit at most one
// sparse rectangle; large worlds grow adjacent clusters to approximately 15%.
std::vector<bool> sparse_biome_regions(const WorldSize size, const std::size_t columns,
    const std::size_t rows, Pcg32& random)
{
    const std::size_t count = columns * rows;
    std::vector<bool> sparse(count, false);
    if (size != WorldSize::large) {
        if (random.unit_interval() < (size == WorldSize::small_world ? 0.10 : 0.20)) {
            sparse[random.bounded(static_cast<std::uint32_t>(count))] = true;
        }
        return sparse;
    }
    const std::size_t target = static_cast<std::size_t>(std::round(count * 0.15));
    std::size_t selected = 0;
    while (selected < target) {
        std::size_t seed = random.bounded(static_cast<std::uint32_t>(count));
        while (sparse[seed]) seed = (seed + 1) % count;
        std::vector<std::size_t> frontier {seed};
        const std::size_t cluster_target = std::min(target, selected + 8);
        while (selected < cluster_target && !frontier.empty()) {
            const auto choice = random.bounded(static_cast<std::uint32_t>(frontier.size()));
            const auto current = frontier[choice];
            frontier[choice] = frontier.back();
            frontier.pop_back();
            if (sparse[current]) continue;
            sparse[current] = true;
            ++selected;
            const auto x = current % columns;
            const auto y = current / columns;
            // Region adjacency wraps just like the simulation world.
            frontier.push_back(y * columns + (x + 1) % columns);
            frontier.push_back(y * columns + (x + columns - 1) % columns);
            frontier.push_back(((y + 1) % rows) * columns + x);
            frontier.push_back(((y + rows - 1) % rows) * columns + x);
        }
    }
    return sparse;
}

// Chooses one stable width per entrance, without consuming the agent simulation RNG.
// The orientation, wall index and gap index distinguish entrances within a region.
std::size_t cave_entrance_width(const std::uint64_t seed, const std::size_t region,
    const bool horizontal, const std::size_t wall, const std::size_t gap) noexcept
{
    const std::uint64_t key = mix_bits(seed ^ 0x656e7472616e6365ULL)
        ^ mix_bits(region) ^ mix_bits(static_cast<std::uint64_t>(wall) + 0x100000000ULL)
        ^ mix_bits(static_cast<std::uint64_t>(gap) + 0x200000000ULL)
        ^ (horizontal ? 0x686f72697aULL : 0x76657274ULL);
    return noise_unit(key) < 0.30 ? 3 : 2;
}

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
    checked_size(config.bootstrap_food_count, "bootstrap food count");
    checked_size(config.boosted_food_count, "boosted food count");
    checked_size(config.maximum_new_food_per_tick, "maximum new food per tick");
    if (config.initial_population < config.minimum_population) {
        throw std::invalid_argument("initial population must not be below minimum population");
    }
    const std::array values {
        config.world_width, config.world_height, config.initial_energy,
        config.food_energy, config.food_regrowth_amount,
        config.living_energy_cost,
        config.movement_energy_cost, config.reproduction_threshold,
        config.maximum_movement_per_tick, config.maximum_turn_per_tick,
        config.agent_radius, config.food_radius, config.terrain_cell_size,
        config.biome_region_minimum_size, config.biome_region_maximum_size,
        config.minimum_fertility, config.wetland_water_coverage_minimum,
        config.sparse_fertility_maximum, config.ordinary_fertility_minimum,
        config.wetland_water_coverage_maximum, config.scattered_rock_probability,
        config.cave_region_probability, config.night_eye_range, config.day_eye_range,
        config.maximum_oxygen, config.oxygen_refill_per_tick,
        config.oxygen_drain_per_tick, config.suffocation_energy_cost,
        config.off_medium_speed_multiplier, config.eat_threshold,
        config.eat_attempt_energy_cost, config.bite_amount_per_tick,
        config.initial_brain_parameter_minimum, config.initial_brain_parameter_maximum,
        config.brain_parameter_minimum, config.brain_parameter_maximum,
        config.founder_mutation_rate_minimum, config.founder_mutation_rate_maximum,
        config.founder_mutation_strength_minimum, config.founder_mutation_strength_maximum,
        config.brain_mutation_scale,
        config.mutation_rate_mutation_scale, config.mutation_strength_mutation_scale,
    };
    for (const double value : values) {
        require_finite(value, "simulation configuration value");
    }
    const long double terrain_cell_count = std::ceil(
        static_cast<long double>(config.world_width / config.terrain_cell_size))
        * std::ceil(static_cast<long double>(config.world_height / config.terrain_cell_size));
    if (static_cast<unsigned>(config.world_size) > static_cast<unsigned>(WorldSize::large)
        || config.world_width <= 0.0 || config.world_height <= 0.0
        || config.initial_energy <= 0.0 || config.food_energy <= 0.0
        || config.food_regrowth_amount < 0.0
        || config.food_regrowth_interval_ticks == 0
        || config.day_night_cycle_ticks == 0
        || config.boosted_food_count < config.target_food_count
        || config.food_bootstrap_population_threshold
            > config.food_boost_population_threshold
        || config.food_boost_population_threshold > config.food_population_threshold
        || config.living_energy_cost < 0.0 || config.movement_energy_cost < 0.0
        || config.reproduction_threshold <= 0.0 || config.maximum_movement_per_tick < 0.0
        || config.maximum_turn_per_tick < 0.0 || config.agent_radius <= 0.0
        || config.food_radius <= 0.0 || config.terrain_cell_size <= 0.0
        || config.terrain_cell_size < config.agent_radius * 2.0
        || terrain_cell_count > std::numeric_limits<std::uint32_t>::max()
        || config.biome_region_minimum_size <= 0.0
        || config.biome_region_maximum_size < config.biome_region_minimum_size
        || config.biome_region_minimum_size > config.world_width
        || config.biome_region_minimum_size > config.world_height
        || config.minimum_fertility <= 0.0 || config.minimum_fertility > 1.0
        || config.minimum_fertility > config.sparse_fertility_maximum
        || config.sparse_fertility_maximum >= config.ordinary_fertility_minimum
        || config.ordinary_fertility_minimum > 1.0
        || config.wetland_water_coverage_minimum < 0.0
        || config.wetland_water_coverage_maximum > 1.0
        || config.wetland_water_coverage_minimum
            > config.wetland_water_coverage_maximum
        || config.scattered_rock_probability < 0.0
        || config.scattered_rock_probability > 1.0
        || config.cave_region_probability < 0.0 || config.cave_region_probability > 1.0
        || config.night_eye_range <= 0.0
        || config.day_eye_range < config.night_eye_range
        || config.maximum_oxygen <= 0.0 || config.oxygen_refill_per_tick < 0.0
        || config.oxygen_drain_per_tick < 0.0 || config.suffocation_energy_cost < 0.0
        || config.off_medium_speed_multiplier < 0.0
        || config.off_medium_speed_multiplier > 1.0
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
        || config.brain_mutation_scale < 0.0
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
            std::max(snapshot.config.bootstrap_food_count,
                snapshot.config.boosted_food_count), "maximum food count")) {
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
        require_finite(agent.carnivore_tendency, "agent carnivore tendency");
        require_finite(agent.water_adaptation, "agent water adaptation");
        require_finite(agent.oxygen, "agent oxygen");
        if (agent.id == 0 || !agent_ids.insert(agent.id).second
            || agent.position.x < 0.0 || agent.position.x >= snapshot.config.world_width
            || agent.position.y < 0.0 || agent.position.y >= snapshot.config.world_height
            || agent.direction < 0.0 || agent.direction >= full_turn || agent.energy <= 0.0
            || agent.color.red < 0.0 || agent.color.red > 1.0
            || agent.color.green < 0.0 || agent.color.green > 1.0
            || agent.color.blue < 0.0 || agent.color.blue > 1.0
            || agent.mutation_rate < minimum_mutation_rate || agent.mutation_rate > 1.0
            || agent.mutation_strength < minimum_mutation_strength
            || agent.mutation_strength > 1.0
            || agent.trait_mutation_rate_percent < 20 || agent.trait_mutation_rate_percent > 100
            || agent.trait_mutation_rate_percent % 5 != 0
            || agent.prior_bite_damage < 0.0
            || agent.carnivore_tendency < 0.0 || agent.carnivore_tendency > 1.0
            || agent.water_adaptation < 0.0 || agent.water_adaptation > 1.0
            || std::floor(agent.carnivore_tendency * 4.0) != agent.carnivore_tendency * 4.0
            || std::floor(agent.water_adaptation * 4.0) != agent.water_adaptation * 4.0
            || agent.oxygen < 0.0 || agent.oxygen > snapshot.config.maximum_oxygen) {
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

// Creates explicit preset settings: population bands scale with length, while
// food supply scales with area to preserve encounter density during bootstrap.
SimulationConfig make_world_config(const std::uint64_t seed, const WorldSize size)
{
    if (static_cast<unsigned>(size) > static_cast<unsigned>(WorldSize::large)) {
        throw std::invalid_argument("invalid world size");
    }
    SimulationConfig config {.seed = seed, .world_size = size};
    const std::uint64_t length_scale = std::uint64_t {1} << static_cast<unsigned>(size);
    const std::uint64_t area_scale = length_scale * length_scale;
    config.world_width *= length_scale;
    config.world_height *= length_scale;
    config.initial_population *= length_scale;
    config.minimum_population *= length_scale;
    config.bootstrap_food_count *= area_scale;
    config.boosted_food_count *= area_scale;
    config.target_food_count *= area_scale;
    config.food_bootstrap_population_threshold *= length_scale;
    config.food_boost_population_threshold *= length_scale;
    config.food_population_threshold *= length_scale;
    config.maximum_new_food_per_tick *= area_scale;
    return config;
}

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
    generate_terrain();
    configure_spatial_index();
    agents_.reserve(checked_size(config_.initial_population, "initial population"));
    food_.reserve(checked_size(std::max(config_.bootstrap_food_count,
        config_.boosted_food_count), "maximum food count"));
    while (agents_.size() < config_.initial_population) {
        agents_.push_back(create_random_agent());
    }
    const bool bootstrap = agents_.size() < config_.food_bootstrap_population_threshold;
    const std::uint64_t initial_food_target = bootstrap
        ? config_.bootstrap_food_count
        : (agents_.size() < config_.food_boost_population_threshold
                ? config_.boosted_food_count : config_.target_food_count);
    // Initial food is complete; only replacement spawning is rate limited.
    while (food_.size() < initial_food_target) {
        food_.push_back(create_random_food(!bootstrap));
    }
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
    generate_terrain();
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
    // Day range is the broad-phase maximum; night never needs a larger query.
    const double desired_cell_size = std::max({config_.agent_radius * 2.0,
        config_.food_radius * 2.0, config_.day_eye_range * 0.25});
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

// Rasterizes rounded islands independently of fertility rectangles, with coastal
// wetland mosaics. All geography uses its own RNG; tick RNG is untouched.
void Simulation::generate_terrain()
{
    biomes_.clear();
    terrain_.clear();
    Pcg32 terrain_random(config_.seed ^ 0xd1b54a32d192ed03ULL);
    const std::size_t continents = config_.world_size == WorldSize::small_world ? 1
        : (config_.world_size == WorldSize::medium ? 2 : 3);
    const double typical_size = (config_.biome_region_minimum_size + config_.biome_region_maximum_size) * 0.5;
    const auto x_edges = biome_axis_edges(config_.world_width, config_.biome_region_minimum_size,
        config_.biome_region_maximum_size,
        std::max(8 * continents, static_cast<std::size_t>(std::round(config_.world_width / typical_size))),
        terrain_random);
    const auto y_edges = biome_axis_edges(config_.world_height, config_.biome_region_minimum_size,
        config_.biome_region_maximum_size,
        std::max(8 * continents, static_cast<std::size_t>(std::round(config_.world_height / typical_size))),
        terrain_random);
    const std::size_t biome_columns = x_edges.size() - 1;
    const std::size_t biome_rows = y_edges.size() - 1;
    const auto sparse = sparse_biome_regions(config_.world_size, biome_columns, biome_rows, terrain_random);
    const auto islands = make_islands(continents, terrain_random);
    for (std::size_t row = 0; row < biome_rows; ++row) {
        for (std::size_t column = 0; column < biome_columns; ++column) {
            const std::size_t index = row * biome_columns + column;
            const Vec2 center {(x_edges[column] + x_edges[column + 1]) * 0.5 / config_.world_width,
                (y_edges[row] + y_edges[row + 1]) * 0.5 / config_.world_height};
            // Kind describes the region center; the actual coastline is resolved per
            // terrain cell, so rectangular fertility borders cannot square off islands.
            const bool land = island_land(islands, center);
            biomes_.push_back({.position = {.x = x_edges[column], .y = y_edges[row]},
                .width = x_edges[column + 1] - x_edges[column],
                .height = y_edges[row + 1] - y_edges[row], .kind = land ? BiomeKind::land : BiomeKind::water,
                .fertility = sparse[index]
                    ? terrain_random.uniform(config_.minimum_fertility, config_.sparse_fertility_maximum)
                    : terrain_random.uniform(config_.ordinary_fertility_minimum, 1.0),
                .wetland_water_coverage = land ? 0.0 : 1.0});
        }
    }
    // Wetlands are major landmasses, not one feature per fertility rectangle.
    // Small has none; other presets always have fewer wetlands than islands.
    std::vector<std::pair<IslandShape, std::size_t>> wetlands;
    const std::size_t wetland_count = continents - 1;
    const std::size_t first_parent = wetland_count == 0 ? 0 : terrain_random.bounded(
        static_cast<std::uint32_t>(islands.size()));
    for (std::size_t number = 0; number < wetland_count; ++number) {
        const std::size_t parent = (first_parent + number) % islands.size();
        const auto& island = islands[parent];
        IslandShape best {};
        double best_clearance = -std::numeric_limits<double>::infinity();
        const double phase = terrain_random.uniform(0.0, 2.0 * std::numbers::pi);
        // A bounded angular search favors open sea away from other islands and
        // wetlands. Wrapped distances treat opposite map edges as neighbors.
        for (std::size_t trial = 0; trial < 64; ++trial) {
            const double angle = phase + trial * 2.0 * std::numbers::pi / 64.0;
            const double parent_radius = std::min(island.radius_x, island.radius_y);
            IslandShape candidate {
                {island.center.x + 1.65 * parent_radius * std::cos(angle),
                 island.center.y + 1.65 * parent_radius * std::sin(angle)},
                island.radius_x * 0.95, island.radius_y * 0.95, island.angle, phase};
            double clearance = std::numeric_limits<double>::infinity();
            const auto measure = [&](const IslandShape& other) {
                double dx = candidate.center.x - other.center.x;
                double dy = candidate.center.y - other.center.y;
                dx -= std::round(dx);
                dy -= std::round(dy);
                return std::hypot(dx, dy) - std::max(other.radius_x, other.radius_y)
                    - std::max(candidate.radius_x, candidate.radius_y);
            };
            for (std::size_t other = 0; other < islands.size(); ++other) {
                if (other != parent) clearance = std::min(clearance, measure(islands[other]));
            }
            for (const auto& existing : wetlands) {
                clearance = std::min(clearance, measure(existing.first));
            }
            if (clearance > best_clearance) {
                best = candidate;
                best_clearance = clearance;
            }
        }
        // Comparable radii and this center spacing overlap the parent footprint.
        // The original mainland remains solid; offshore wetland land is patchy.
        Vec2 attachment {island.center.x + (best.center.x - island.center.x) * 0.5,
            island.center.y + (best.center.y - island.center.y) * 0.5};
        attachment.x -= std::floor(attachment.x);
        attachment.y -= std::floor(attachment.y);
        const auto column = static_cast<std::size_t>(std::upper_bound(x_edges.begin(), x_edges.end(),
            attachment.x * config_.world_width) - x_edges.begin() - 1);
        const auto row = static_cast<std::size_t>(std::upper_bound(y_edges.begin(), y_edges.end(),
            attachment.y * config_.world_height) - y_edges.begin() - 1);
        const auto index = row * biome_columns + column;
        auto& source = biomes_[index];
        source.kind = BiomeKind::wetland;
        source.wetland_water_coverage = terrain_random.uniform(
            config_.wetland_water_coverage_minimum, config_.wetland_water_coverage_maximum);
        wetlands.push_back({best, index});
    }

    terrain_columns_ = std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(config_.world_width / config_.terrain_cell_size)));
    terrain_rows_ = std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(config_.world_height / config_.terrain_cell_size)));
    const double cell_width = config_.world_width / static_cast<double>(terrain_columns_);
    const double cell_height = config_.world_height / static_cast<double>(terrain_rows_);
    terrain_.reserve(terrain_columns_ * terrain_rows_);
    for (std::size_t row = 0; row < terrain_rows_; ++row) {
        for (std::size_t column = 0; column < terrain_columns_; ++column) {
            const Vec2 position {.x = column * cell_width, .y = row * cell_height};
            const Vec2 center {.x = position.x + cell_width * 0.5,
                .y = position.y + cell_height * 0.5};
            const auto biome_column = static_cast<std::size_t>(
                std::upper_bound(x_edges.begin(), x_edges.end(), center.x) - x_edges.begin() - 1);
            const auto biome_row = static_cast<std::size_t>(
                std::upper_bound(y_edges.begin(), y_edges.end(), center.y) - y_edges.begin() - 1);
            const std::size_t region_index = biome_row * biome_columns + biome_column;
            const BiomeRegion& region = biomes_[region_index];
            TerrainMedium medium = island_land(islands,
                {center.x / config_.world_width, center.y / config_.world_height})
                ? TerrainMedium::land : TerrainMedium::water;
            for (const auto& [wetland, source_index] : wetlands) {
                // Keep the mainland intact; only the attached offshore extension
                // receives patchwork. Overlapping extensions use their stable order.
                if (medium == TerrainMedium::land) break;
                if (island_coast_offset(wetland,
                        {center.x / config_.world_width, center.y / config_.world_height}) > 0.0) continue;
                const auto& source = biomes_[source_index];
                // One trial per three-cell patch preserves the mosaic. Every patch
                // uses the same coverage, independent of distance to land or ocean.
                const std::uint64_t patch_x = column / 3;
                const std::uint64_t patch_y = row / 3;
                const std::uint64_t key = config_.seed ^ mix_bits(source_index)
                    ^ mix_bits(patch_x + 0x100000000ULL) ^ mix_bits(patch_y + 0x200000000ULL);
                medium = noise_unit(key) < source.wetland_water_coverage
                    ? TerrainMedium::water : TerrainMedium::land;
                break;
            }

            const std::uint64_t cell_key = config_.seed
                ^ (static_cast<std::uint64_t>(row) << 32U)
                ^ static_cast<std::uint64_t>(column);
            const bool cave_region = noise_unit(config_.seed ^ region_index ^ 0xca6eULL)
                < config_.cave_region_probability;
            const std::size_t local_column = static_cast<std::size_t>(
                (center.x - region.position.x) / cell_width);
            const std::size_t local_row = static_cast<std::size_t>(
                (center.y - region.position.y) / cell_height);
            // All cells in a gap share its width choice: 30% three cells, 70% two.
            // Keep entrances clear even where another wall or scattered rock crosses them.
            const bool cave_entrance = cave_region
                && ((local_column % 7 == 0
                        && local_row % 5 >= 2
                        && local_row % 5 < 2 + cave_entrance_width(config_.seed,
                            region_index, false, local_column / 7, local_row / 5))
                    || (local_row % 9 == 0
                        && local_column % 6 >= 3
                        && local_column % 6 < 3 + cave_entrance_width(config_.seed,
                            region_index, true, local_row / 9, local_column / 6)));
            const bool cave_wall = cave_region
                && (local_column % 7 == 0 || local_row % 9 == 0);
            const bool scattered = noise_unit(cell_key ^ 0x726f636bULL)
                < config_.scattered_rock_probability;
            terrain_.push_back({.position = position, .width = cell_width,
                .height = cell_height, .medium = medium,
                .fertility = region.fertility,
                .rock = !cave_entrance && (cave_wall || scattered)});
        }
    }
    // A small deterministic clearing guarantees that even extreme rock settings can spawn life.
    const std::size_t center_column = terrain_columns_ / 2;
    const std::size_t center_row = terrain_rows_ / 2;
    for (std::size_t row = center_row > 0 ? center_row - 1 : center_row;
         row <= std::min(terrain_rows_ - 1, center_row + 1); ++row) {
        for (std::size_t column = center_column > 0 ? center_column - 1 : center_column;
             column <= std::min(terrain_columns_ - 1, center_column + 1); ++column) {
            terrain_[row * terrain_columns_ + column].rock = false;
        }
    }
}

const TerrainCell& Simulation::terrain_at(const Vec2 position) const noexcept
{
    const double x = wrap_coordinate(position.x, config_.world_width);
    const double y = wrap_coordinate(position.y, config_.world_height);
    const std::size_t column = std::min(terrain_columns_ - 1,
        static_cast<std::size_t>(x / config_.world_width * terrain_columns_));
    const std::size_t row = std::min(terrain_rows_ - 1,
        static_cast<std::size_t>(y / config_.world_height * terrain_rows_));
    return terrain_[row * terrain_columns_ + column];
}

bool Simulation::touches_rock(const Vec2 position) const noexcept
{
    constexpr std::array<Vec2, 5> directions {{{0.0, 0.0}, {1.0, 0.0},
        {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}}};
    return std::ranges::any_of(directions, [&](const Vec2 direction) {
        return terrain_at({.x = position.x + direction.x * config_.agent_radius,
            .y = position.y + direction.y * config_.agent_radius}).rock;
    });
}

double Simulation::ray_rock_distance(
    const Vec2 origin, const Vec2 direction, const double range) const noexcept
{
    const double step = std::min(terrain_[0].width, terrain_[0].height) * 0.20;
    for (double distance = 0.0; distance <= range; distance += step) {
        if (terrain_at({.x = origin.x + direction.x * distance,
                .y = origin.y + direction.y * distance}).rock) {
            return distance;
        }
    }
    return std::numeric_limits<double>::infinity();
}

Agent Simulation::create_random_agent()
{
    if (next_agent_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("agent ID space exhausted");
    }
    Vec2 position;
    for (;;) {
        const TerrainCell& cell = terrain_[random_.bounded(
            static_cast<std::uint32_t>(terrain_.size()))];
        if (cell.rock) continue;
        const double margin_x = std::min(config_.agent_radius * 1.01, cell.width * 0.49);
        const double margin_y = std::min(config_.agent_radius * 1.01, cell.height * 0.49);
        position = {.x = random_.uniform(cell.position.x + margin_x,
                        cell.position.x + cell.width - margin_x),
            .y = random_.uniform(cell.position.y + margin_y,
                        cell.position.y + cell.height - margin_y)};
        if (!touches_rock(position)) break;
    }
    Agent agent {
        .id = next_agent_id_++,
        .position = position,
        .direction = random_.uniform(0.0, full_turn), .energy = config_.initial_energy,
        .carnivore_tendency = 0.0,
        // Food-rich bootstrap conditions favor survival in both media over speed.
        // Start founders amphibious so random specialization does not obstruct
        // establishment; descendants can evolve specialization as competition grows.
        .water_adaptation = 0.5,
        .oxygen = config_.maximum_oxygen,
        .color = {.red = random_.unit_interval(), .green = random_.unit_interval(),
            .blue = random_.unit_interval()},
        .mutation_rate = random_.uniform(config_.founder_mutation_rate_minimum,
            config_.founder_mutation_rate_maximum),
        .mutation_strength = random_.uniform(config_.founder_mutation_strength_minimum,
            config_.founder_mutation_strength_maximum),
        // Uniformly sample all seventeen allowed founder rates, including 100%.
        .trait_mutation_rate_percent = static_cast<std::uint8_t>(20 + 5 * random_.bounded(17)),
    };
    // Founders use eight active nodes even though mutation can activate up to sixteen.
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

Food Simulation::create_random_food(const bool fertility_weighted)
{
    if (next_food_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("food ID space exhausted");
    }
    Vec2 position;
    // Bootstrap placement samples all traversable cells equally; normal placement
    // uses fertility as a per-cell acceptance weight without changing global caps.
    for (;;) {
        const TerrainCell& cell = terrain_[random_.bounded(
            static_cast<std::uint32_t>(terrain_.size()))];
        if (cell.rock
            || (fertility_weighted && random_.unit_interval() > cell.fertility)) {
            continue;
        }
        position = {.x = random_.uniform(cell.position.x, cell.position.x + cell.width),
            .y = random_.uniform(cell.position.y, cell.position.y + cell.height)};
        break;
    }
    return {.id = next_food_id_++, .position = position, .energy = config_.food_energy};
}

BrainInputs Simulation::sense_agent(const Agent& observer,
    const std::span<const std::size_t> agent_candidates,
    const std::span<const std::size_t> food_candidates) const
{
    BrainInputs inputs {
        .energy = std::clamp(observer.energy / config_.reproduction_threshold, 0.0, 1.0),
        .damage = std::clamp(observer.prior_bite_damage / config_.bite_amount_per_tick, 0.0, 1.0),
        .oxygen = std::clamp(observer.oxygen / config_.maximum_oxygen, 0.0, 1.0),
        .rock_contact = observer.rock_contact ? 1.0 : 0.0,
    };
    const double eye_range = current_eye_range();
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
        double best_distance = ray_rock_distance(origin, ray, eye_range);
        int best_layer = std::isfinite(best_distance) ? 2 : -1;
        std::uint64_t best_id = 0;
        AgentColor best_color = std::isfinite(best_distance) ? rock_color : AgentColor {};

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
                eye_range, config_.world_width, config_.world_height),
                1, target.id, target.color);
        }
        for (const std::size_t food_index : food_candidates) {
            const Food& item = food_[food_index];
            consider(ray_circle_distance(origin, ray, item.position, config_.food_radius,
                eye_range, config_.world_width, config_.world_height),
                0, item.id, plant_food_color);
        }
        if (std::isfinite(best_distance)) {
            inputs.vision[ray_index] = {.red = best_color.red, .green = best_color.green,
                .blue = best_color.blue,
                .proximity = 1.0 - best_distance / eye_range};
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
    const double query_radius = current_eye_range()
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
        agent.rock_contact = false;
        agent.direction = normalize_direction(agent.direction
            + action.turn * config_.maximum_turn_per_tick);
        const bool water = terrain_at(agent.position).medium == TerrainMedium::water;
        // One trait deliberately governs both locomotion and breathing specialization.
        const double compatibility = water
            ? agent.water_adaptation : 1.0 - agent.water_adaptation;
        // Compatibility at or below the amphibious midpoint uses the configured
        // speed floor; specialization above it scales to full preferred-medium speed.
        const double specialization = std::clamp(2.0 * compatibility - 1.0, 0.0, 1.0);
        const double speed_multiplier = config_.off_medium_speed_multiplier
            + (1.0 - config_.off_medium_speed_multiplier) * specialization;
        const double requested_distance = action.move * config_.maximum_movement_per_tick
            * speed_multiplier;
        double distance = 0.0;
        const double collision_step = std::min(terrain_[0].width, terrain_[0].height) * 0.25;
        // Substeps prevent a large configured movement from tunneling through a rock cell.
        const std::size_t movement_steps = std::max<std::size_t>(1,
            static_cast<std::size_t>(std::ceil(requested_distance / collision_step)));
        const double step_distance = requested_distance / static_cast<double>(movement_steps);
        for (std::size_t step = 0; step < movement_steps; ++step) {
            const Vec2 candidate {
                .x = wrap_coordinate(agent.position.x
                        + std::cos(agent.direction) * step_distance,
                    config_.world_width),
                .y = wrap_coordinate(agent.position.y
                        + std::sin(agent.direction) * step_distance,
                    config_.world_height),
            };
            if (touches_rock(candidate)) {
                agent.rock_contact = true;
                break;
            }
            agent.position = candidate;
            distance += step_distance;
        }
        agent.oxygen = std::clamp(agent.oxygen
                + config_.oxygen_refill_per_tick * compatibility
                - config_.oxygen_drain_per_tick * (1.0 - compatibility),
            0.0, config_.maximum_oxygen);
        agent.energy -= config_.living_energy_cost + config_.movement_energy_cost * distance
            + (action.eat ? config_.eat_attempt_energy_cost : 0.0);
        if (agent.oxygen == 0.0) agent.energy -= config_.suffocation_energy_cost;
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
        if (top_agent_index != agents_.size()) {
            requests.push_back({eater_index, true, top_agent_index});
        } else if (top_food_index != food_.size()) {
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
        const double efficiency = request.target_is_agent
            ? agents_[request.eater].carnivore_tendency
            : 1.0 - agents_[request.eater].carnivore_tendency;
        agent_energy_delta[request.eater] += amount * efficiency;
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
        if (touches_rock(child.position)) child.position = parent.position;
        child.age = 0;
        child.generation = parent.generation + 1;
        child.prior_bite_damage = 0.0;
        child.oxygen = config_.maximum_oxygen;
        child.rock_contact = false;
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
        // Fixed gene and topology order is part of seeded determinism.
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
        // Use the parent's rate for independent trait trials. Reselecting the same
        // ecological value is intentional; a trigger need not change the phenotype.
        const auto trait_mutation_triggers = [&]() {
            return random_.bounded(100) < parent.trait_mutation_rate_percent;
        };
        const auto mutate_color = [&](double& channel) {
            if (trait_mutation_triggers()) {
                const double candidate = channel + (random_.bounded(2) == 0 ? -0.15 : 0.15);
                // Reflect the fixed step to preserve the normalized RGB range.
                channel = candidate < 0.0 ? -candidate
                    : (candidate > 1.0 ? 2.0 - candidate : candidate);
            }
        };
        mutate_color(child.color.red);
        mutate_color(child.color.green);
        mutate_color(child.color.blue);
        if (trait_mutation_triggers()) {
            child.carnivore_tendency = static_cast<double>(random_.bounded(5)) * 0.25;
        }
        if (trait_mutation_triggers()) {
            child.water_adaptation = static_cast<double>(random_.bounded(5)) * 0.25;
        }
        mutate(child.mutation_rate, minimum_mutation_rate, 1.0,
            config_.mutation_rate_mutation_scale);
        mutate(child.mutation_strength, minimum_mutation_strength, 1.0,
            config_.mutation_strength_mutation_scale);
        child.mutation_rate = std::max(child.mutation_rate, minimum_mutation_rate);
        child.mutation_strength = std::max(child.mutation_strength,
            minimum_mutation_strength);
        // Every birth adjusts the inherited rate by a fair +/-5 percentage-point
        // step. Clamp outward steps at the limits; this never changes this birth's trials.
        const int rate_step = random_.bounded(2) == 0 ? -5 : 5;
        child.trait_mutation_rate_percent = static_cast<std::uint8_t>(std::clamp(
            static_cast<int>(parent.trait_mutation_rate_percent) + rate_step, 20, 100));
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
    while (agents_.size() < minimum) {
        agents_.push_back(create_random_agent());
        ++introduced_agents_;
    }
    if (agents_.size() != before) brain_batch_dirty_ = true;
}

void Simulation::replenish_food_if_allowed()
{
    if (agents_.size() >= config_.food_population_threshold) return;
    const bool bootstrap = agents_.size() < config_.food_bootstrap_population_threshold;
    const std::uint64_t configured_target = bootstrap
        ? config_.bootstrap_food_count
        : (agents_.size() < config_.food_boost_population_threshold
                ? config_.boosted_food_count : config_.target_food_count);
    const std::size_t target = checked_size(configured_target, "food target");
    // Recovered populations keep excess food until agents consume it naturally.
    const std::size_t missing = target > food_.size() ? target - food_.size() : 0;
    const std::size_t additions = std::min(missing,
        checked_size(config_.maximum_new_food_per_tick, "maximum new food per tick"));
    for (std::size_t index = 0; index < additions; ++index) {
        food_.push_back(create_random_food(!bootstrap));
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
std::span<const BiomeRegion> Simulation::biomes() const noexcept { return biomes_; }
std::span<const TerrainCell> Simulation::terrain() const noexcept { return terrain_; }

double Simulation::light_level() const noexcept
{
    const double phase = static_cast<double>(
        current_tick_ % config_.day_night_cycle_ticks)
        / static_cast<double>(config_.day_night_cycle_ticks);
    return 0.5 - 0.5 * std::cos(full_turn * phase);
}

double Simulation::current_eye_range() const noexcept
{
    return config_.night_eye_range
        + (config_.day_eye_range - config_.night_eye_range) * light_level();
}

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
    return {.seed = config_.seed, .completed_ticks = current_tick_,
        .population = static_cast<std::uint64_t>(agents_.size()),
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
