#include "evobrain/checkpoint.hpp"

#include "evobrain/brain.hpp"
#include "evobrain/simulation.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace evobrain {
namespace {

constexpr std::array<char, 8> checkpoint_identifier {
    'E', 'V', 'O', 'B', 'R', 'A', 'I', 'N',
};
// Terrain is regenerated on load; changing the generation algorithm also changes
// the format version so an older save cannot silently acquire a different world.
constexpr std::uint32_t checkpoint_version = 16;

// Writes primitive values using a platform-independent little-endian encoding.
class BinaryWriter {
public:
    explicit BinaryWriter(std::ostream& output)
        : output_(output)
    {
    }

    // Writes one raw byte and fails immediately when the stream rejects it.
    void byte(const std::uint8_t value)
    {
        output_.put(static_cast<char>(value));
        if (!output_) {
            throw std::runtime_error("failed to write checkpoint");
        }
    }

    // Writes a 32-bit unsigned integer in little-endian byte order.
    void unsigned_32(const std::uint32_t value)
    {
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            byte(static_cast<std::uint8_t>(value >> shift));
        }
    }

    // Writes a 64-bit unsigned integer in little-endian byte order.
    void unsigned_64(const std::uint64_t value)
    {
        for (unsigned int shift = 0; shift < 64; shift += 8) {
            byte(static_cast<std::uint8_t>(value >> shift));
        }
    }

    // Writes the exact double bit pattern without textual conversion.
    void real(const double value)
    {
        unsigned_64(std::bit_cast<std::uint64_t>(value));
    }

private:
    std::ostream& output_;
};

// Reads primitive values from the checkpoint's explicit binary encoding.
class BinaryReader {
public:
    explicit BinaryReader(std::istream& input)
        : input_(input)
    {
    }

    // Reads one byte and reports truncation before returning partial state.
    std::uint8_t byte()
    {
        const int value = input_.get();
        if (value == std::char_traits<char>::eof()) {
            throw std::runtime_error("checkpoint is truncated");
        }
        return static_cast<std::uint8_t>(value);
    }

    // Reads one little-endian 32-bit unsigned integer.
    std::uint32_t unsigned_32()
    {
        std::uint32_t value = 0;
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            value |= static_cast<std::uint32_t>(byte()) << shift;
        }
        return value;
    }

    // Reads one little-endian 64-bit unsigned integer.
    std::uint64_t unsigned_64()
    {
        std::uint64_t value = 0;
        for (unsigned int shift = 0; shift < 64; shift += 8) {
            value |= static_cast<std::uint64_t>(byte()) << shift;
        }
        return value;
    }

    // Reads one exact double bit pattern from its serialized integer form.
    double real()
    {
        return std::bit_cast<double>(unsigned_64());
    }

    // Rejects unrecognized trailing data after the complete payload.
    void require_end()
    {
        if (input_.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("checkpoint contains trailing data");
        }
    }

private:
    std::istream& input_;
};

// Converts a serialized collection count before allocating native storage.
std::size_t collection_size(const std::uint64_t value)
{
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("checkpoint collection is too large");
    }
    return static_cast<std::size_t>(value);
}

// Writes the size preset and complete explicit configuration in version-sixteen order.
void write_config(BinaryWriter& writer, const SimulationConfig& config)
{
    writer.unsigned_64(config.seed);
    writer.byte(static_cast<std::uint8_t>(config.world_size));
    writer.unsigned_64(config.initial_population);
    writer.unsigned_64(config.minimum_population);
    writer.unsigned_64(config.target_food_count);
    writer.unsigned_64(config.food_population_threshold);
    writer.unsigned_64(config.food_bootstrap_population_threshold);
    writer.unsigned_64(config.bootstrap_food_count);
    writer.unsigned_64(config.food_boost_population_threshold);
    writer.unsigned_64(config.boosted_food_count);
    writer.unsigned_64(config.maximum_new_food_per_tick);
    writer.unsigned_64(config.food_regrowth_interval_ticks);
    writer.real(config.world_width);
    writer.real(config.world_height);
    writer.real(config.terrain_cell_size);
    writer.real(config.biome_region_minimum_size);
    writer.real(config.biome_region_maximum_size);
    writer.real(config.minimum_fertility);
    writer.real(config.sparse_fertility_maximum);
    writer.real(config.ordinary_fertility_minimum);
    writer.real(config.wetland_water_coverage_minimum);
    writer.real(config.wetland_water_coverage_maximum);
    writer.real(config.scattered_rock_probability);
    writer.real(config.cave_region_probability);
    writer.real(config.initial_energy);
    writer.real(config.food_energy);
    writer.real(config.food_regrowth_amount);
    writer.real(config.living_energy_cost);
    writer.real(config.movement_energy_cost);
    writer.real(config.reproduction_threshold);
    writer.real(config.maximum_movement_per_tick);
    writer.real(config.maximum_turn_per_tick);
    writer.real(config.agent_radius);
    writer.real(config.food_radius);
    writer.unsigned_64(config.day_night_cycle_ticks);
    writer.real(config.night_eye_range);
    writer.real(config.day_eye_range);
    writer.real(config.maximum_oxygen);
    writer.real(config.oxygen_refill_per_tick);
    writer.real(config.oxygen_drain_per_tick);
    writer.real(config.suffocation_energy_cost);
    writer.real(config.off_medium_speed_multiplier);
    writer.real(config.eat_threshold);
    writer.real(config.eat_attempt_energy_cost);
    writer.real(config.bite_amount_per_tick);
    writer.real(config.initial_brain_parameter_minimum);
    writer.real(config.initial_brain_parameter_maximum);
    writer.real(config.brain_parameter_minimum);
    writer.real(config.brain_parameter_maximum);
    writer.real(config.founder_mutation_rate_minimum);
    writer.real(config.founder_mutation_rate_maximum);
    writer.real(config.founder_mutation_strength_minimum);
    writer.real(config.founder_mutation_strength_maximum);
    writer.real(config.brain_mutation_scale);
    writer.real(config.mutation_rate_mutation_scale);
    writer.real(config.mutation_strength_mutation_scale);
}

// Restores explicit version-sixteen settings without reapplying size scaling.
SimulationConfig read_config(BinaryReader& reader)
{
    SimulationConfig config {.seed = reader.unsigned_64()};
    config.world_size = static_cast<WorldSize>(reader.byte());
    config.initial_population = reader.unsigned_64();
    config.minimum_population = reader.unsigned_64();
    config.target_food_count = reader.unsigned_64();
    config.food_population_threshold = reader.unsigned_64();
    config.food_bootstrap_population_threshold = reader.unsigned_64();
    config.bootstrap_food_count = reader.unsigned_64();
    config.food_boost_population_threshold = reader.unsigned_64();
    config.boosted_food_count = reader.unsigned_64();
    config.maximum_new_food_per_tick = reader.unsigned_64();
    config.food_regrowth_interval_ticks = reader.unsigned_64();
    config.world_width = reader.real();
    config.world_height = reader.real();
    config.terrain_cell_size = reader.real();
    config.biome_region_minimum_size = reader.real();
    config.biome_region_maximum_size = reader.real();
    config.minimum_fertility = reader.real();
    config.sparse_fertility_maximum = reader.real();
    config.ordinary_fertility_minimum = reader.real();
    config.wetland_water_coverage_minimum = reader.real();
    config.wetland_water_coverage_maximum = reader.real();
    config.scattered_rock_probability = reader.real();
    config.cave_region_probability = reader.real();
    config.initial_energy = reader.real();
    config.food_energy = reader.real();
    config.food_regrowth_amount = reader.real();
    config.living_energy_cost = reader.real();
    config.movement_energy_cost = reader.real();
    config.reproduction_threshold = reader.real();
    config.maximum_movement_per_tick = reader.real();
    config.maximum_turn_per_tick = reader.real();
    config.agent_radius = reader.real();
    config.food_radius = reader.real();
    config.day_night_cycle_ticks = reader.unsigned_64();
    config.night_eye_range = reader.real();
    config.day_eye_range = reader.real();
    config.maximum_oxygen = reader.real();
    config.oxygen_refill_per_tick = reader.real();
    config.oxygen_drain_per_tick = reader.real();
    config.suffocation_energy_cost = reader.real();
    config.off_medium_speed_multiplier = reader.real();
    config.eat_threshold = reader.real();
    config.eat_attempt_energy_cost = reader.real();
    config.bite_amount_per_tick = reader.real();
    config.initial_brain_parameter_minimum = reader.real();
    config.initial_brain_parameter_maximum = reader.real();
    config.brain_parameter_minimum = reader.real();
    config.brain_parameter_maximum = reader.real();
    config.founder_mutation_rate_minimum = reader.real();
    config.founder_mutation_rate_maximum = reader.real();
    config.founder_mutation_strength_minimum = reader.real();
    config.founder_mutation_strength_maximum = reader.real();
    config.brain_mutation_scale = reader.real();
    config.mutation_rate_mutation_scale = reader.real();
    config.mutation_strength_mutation_scale = reader.real();
    return config;
}

// Writes one agent without relying on compiler layout or structure padding.
void write_agent(BinaryWriter& writer, const Agent& agent)
{
    writer.unsigned_64(agent.id);
    writer.real(agent.position.x);
    writer.real(agent.position.y);
    writer.real(agent.direction);
    writer.real(agent.energy);
    writer.unsigned_64(agent.age);
    writer.unsigned_64(agent.generation);
    writer.real(agent.carnivore_tendency);
    writer.real(agent.water_adaptation);
    writer.real(agent.oxygen);
    writer.byte(agent.rock_contact ? 1 : 0);
    writer.real(agent.color.red);
    writer.real(agent.color.green);
    writer.real(agent.color.blue);
    writer.real(agent.mutation_rate);
    writer.real(agent.mutation_strength);
    writer.byte(agent.trait_mutation_rate_percent);
    writer.real(agent.prior_bite_damage);
    for (const double parameter : agent.brain) {
        writer.real(parameter);
    }
    writer.byte(agent.brain_structure.founder_fast_path);
    for (const std::uint8_t active : agent.brain_structure.hidden_active) {
        writer.byte(active);
    }
    for (const std::uint8_t enabled : agent.brain_structure.input_hidden_enabled) {
        writer.byte(enabled);
    }
    for (const std::uint8_t enabled : agent.brain_structure.hidden_output_enabled) {
        writer.byte(enabled);
    }
    for (const std::uint8_t enabled : agent.brain_structure.recurrent_enabled) {
        writer.byte(enabled);
    }
    for (const double weight : agent.brain_structure.recurrent_weights) {
        writer.real(weight);
    }
    for (const double value : agent.brain_state.previous_hidden) {
        writer.real(value);
    }
    for (const double value : agent.brain_state.next_hidden) {
        writer.real(value);
    }
}

// Reads the fields that precede the brain payload.
Agent read_agent_header(BinaryReader& reader)
{
    return Agent {
        .id = reader.unsigned_64(),
        .position = Vec2 {.x = reader.real(), .y = reader.real()},
        .direction = reader.real(),
        .energy = reader.real(),
        .age = reader.unsigned_64(),
        .generation = reader.unsigned_64(),
        .carnivore_tendency = reader.real(),
        .water_adaptation = reader.real(),
        .oxygen = reader.real(),
        .rock_contact = reader.byte() != 0,
        .color = {.red = reader.real(), .green = reader.real(), .blue = reader.real()},
        .mutation_rate = reader.real(),
        .mutation_strength = reader.real(),
        .trait_mutation_rate_percent = reader.byte(),
        .prior_bite_damage = reader.real(),
    };
}

// Reads one complete agent, including brain mutation parameters, from version sixteen.
Agent read_agent(BinaryReader& reader)
{
    Agent agent = read_agent_header(reader);
    for (double& parameter : agent.brain) {
        parameter = reader.real();
    }
    agent.brain_structure.founder_fast_path = reader.byte();
    for (std::uint8_t& active : agent.brain_structure.hidden_active) {
        active = reader.byte();
    }
    for (std::uint8_t& enabled : agent.brain_structure.input_hidden_enabled) {
        enabled = reader.byte();
    }
    for (std::uint8_t& enabled : agent.brain_structure.hidden_output_enabled) {
        enabled = reader.byte();
    }
    for (std::uint8_t& enabled : agent.brain_structure.recurrent_enabled) {
        enabled = reader.byte();
    }
    for (double& weight : agent.brain_structure.recurrent_weights) {
        weight = reader.real();
    }
    for (double& value : agent.brain_state.previous_hidden) {
        value = reader.real();
    }
    for (double& value : agent.brain_state.next_hidden) {
        value = reader.real();
    }
    return agent;
}

} // namespace

void save_checkpoint(const Simulation& simulation, std::ostream& output)
{
    const SimulationSnapshot snapshot = simulation.snapshot();
    BinaryWriter writer(output);
    for (const char value : checkpoint_identifier) {
        writer.byte(static_cast<std::uint8_t>(value));
    }
    writer.unsigned_32(checkpoint_version);
    write_config(writer, snapshot.config);
    writer.unsigned_64(snapshot.current_tick);
    writer.unsigned_64(snapshot.random_state);
    writer.unsigned_64(snapshot.next_agent_id);
    writer.unsigned_64(snapshot.next_food_id);
    writer.unsigned_64(snapshot.births);
    writer.unsigned_64(snapshot.introduced_agents);
    writer.unsigned_64(snapshot.deaths);
    writer.unsigned_64(snapshot.agents_eaten);

    writer.unsigned_64(static_cast<std::uint64_t>(snapshot.agents.size()));
    for (const Agent& agent : snapshot.agents) {
        write_agent(writer, agent);
    }

    writer.unsigned_64(static_cast<std::uint64_t>(snapshot.food.size()));
    for (const Food& item : snapshot.food) {
        writer.unsigned_64(item.id);
        writer.real(item.position.x);
        writer.real(item.position.y);
        writer.real(item.energy);
    }
}

Simulation load_checkpoint(std::istream& input)
{
    BinaryReader reader(input);
    for (const char expected : checkpoint_identifier) {
        if (reader.byte() != static_cast<std::uint8_t>(expected)) {
            throw std::runtime_error("invalid EvoBrainBot checkpoint identifier");
        }
    }
    const std::uint32_t version = reader.unsigned_32();
    if (version != checkpoint_version) {
        throw std::runtime_error("unsupported EvoBrainBot checkpoint version");
    }

    SimulationSnapshot snapshot {
        .config = read_config(reader),
        .current_tick = reader.unsigned_64(),
        .random_state = reader.unsigned_64(),
        .next_agent_id = reader.unsigned_64(),
        .next_food_id = reader.unsigned_64(),
        .births = reader.unsigned_64(),
        .introduced_agents = reader.unsigned_64(),
        .deaths = reader.unsigned_64(),
        .agents_eaten = reader.unsigned_64(),
    };

    const std::size_t agent_count = collection_size(reader.unsigned_64());
    snapshot.agents.reserve(agent_count);
    for (std::size_t index = 0; index < agent_count; ++index) {
        snapshot.agents.push_back(read_agent(reader));
    }

    const std::size_t food_count = collection_size(reader.unsigned_64());
    snapshot.food.reserve(food_count);
    for (std::size_t index = 0; index < food_count; ++index) {
        snapshot.food.push_back(Food {
            .id = reader.unsigned_64(),
            .position = Vec2 {.x = reader.real(), .y = reader.real()},
            .energy = reader.real(),
        });
    }

    reader.require_end();
    return Simulation::from_snapshot(std::move(snapshot));
}

} // namespace evobrain
