#include "world_renderer.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

namespace evobrain::viewer {
namespace {

constexpr float rectangle_shape = 0.0F;
constexpr float circle_shape = 1.0F;

struct Color {
    float red;
    float green;
    float blue;
    float alpha;
};

constexpr Color outer_fill {0.88F, 0.90F, 0.93F, 1.0F};
constexpr Color world_fill {1.0F, 1.0F, 1.0F, 1.0F};
constexpr Color outer_boundary {0.48F, 0.51F, 0.56F, 0.75F};
constexpr Color world_boundary {0.16F, 0.18F, 0.22F, 1.0F};
constexpr Color heading_color {0.04F, 0.22F, 0.48F, 1.0F};
constexpr Color food_color {static_cast<float>(plant_food_color.red),
    static_cast<float>(plant_food_color.green),
    static_cast<float>(plant_food_color.blue), 1.0F};
constexpr Color herbivore_marker {0.10F, 0.85F, 0.22F, 1.0F};
constexpr Color carnivore_marker {0.92F, 0.12F, 0.08F, 1.0F};
constexpr Color eye_geometry {0.18F, 0.72F, 0.95F, 0.32F};
constexpr Color eye_position {0.05F, 0.35F, 0.72F, 1.0F};
constexpr Color mouth_position {0.95F, 0.20F, 0.12F, 1.0F};
constexpr Color selection_glow {1.0F, 0.68F, 0.08F, 0.68F};
constexpr Color energy_background {0.08F, 0.10F, 0.12F, 0.92F};
constexpr Color energy_fill {0.16F, 0.72F, 0.28F, 1.0F};
constexpr Color debug_grid {0.12F, 0.42F, 0.72F, 0.32F};
constexpr Color debug_occupied {0.18F, 0.56F, 0.82F, 0.09F};
constexpr Color debug_query {1.0F, 0.62F, 0.08F, 0.14F};

// Reads one build-generated shader blob without runtime compilation.
std::vector<std::uint8_t> read_binary_file(
    const std::filesystem::path& path,
    std::string& error)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "Could not open compiled shader: " + path.string();
        return {};
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        error = "Compiled shader is empty: " + path.string();
        return {};
    }
    input.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!input.read(reinterpret_cast<char*>(bytes.data()), size)) {
        error = "Could not read compiled shader: " + path.string();
        return {};
    }
    return bytes;
}

// Creates a DXIL shader with the resource counts required by SDL_GPU.
SDL_GPUShader* create_shader(
    SDL_GPUDevice* device,
    const std::filesystem::path& path,
    const SDL_GPUShaderStage stage,
    std::string& error)
{
    const std::vector<std::uint8_t> code = read_binary_file(path, error);
    if (code.empty()) {
        return nullptr;
    }
    SDL_GPUShaderCreateInfo info {};
    info.code_size = code.size();
    info.code = code.data();
    info.entrypoint = "main";
    info.format = SDL_GPU_SHADERFORMAT_DXIL;
    info.stage = stage;
    info.num_uniform_buffers = stage == SDL_GPU_SHADERSTAGE_VERTEX ? 1 : 0;
    SDL_GPUShader* shader = SDL_CreateGPUShader(device, &info);
    if (shader == nullptr) {
        error = SDL_GetError();
    }
    return shader;
}

// Returns true when a screen-space shape can affect the clipped world view.
bool intersects_viewport(
    const float center_x,
    const float center_y,
    const float half_width,
    const float half_height,
    const PixelViewport& viewport) noexcept
{
    return center_x + half_width >= viewport.x
        && center_x - half_width <= viewport.x + viewport.width
        && center_y + half_height >= viewport.y
        && center_y - half_height <= viewport.y + viewport.height;
}

} // namespace

struct WorldRenderer::ShapeInstance {
    float center_x;
    float center_y;
    float half_width;
    float half_height;
    float rotation_cosine;
    float rotation_sine;
    float red;
    float green;
    float blue;
    float alpha;
    float shape;
    float padding;
};

WorldRenderer::~WorldRenderer()
{
    shutdown();
}

bool WorldRenderer::initialize(
    SDL_GPUDevice* device,
    const SDL_GPUTextureFormat target_format,
    const std::filesystem::path& shader_directory,
    std::string& error)
{
    shutdown();
    device_ = device;

    SDL_GPUShader* vertex_shader = create_shader(
        device_, shader_directory / L"world.vert.dxil", SDL_GPU_SHADERSTAGE_VERTEX, error);
    if (vertex_shader == nullptr) {
        shutdown();
        return false;
    }
    SDL_GPUShader* fragment_shader = create_shader(
        device_, shader_directory / L"world.frag.dxil", SDL_GPU_SHADERSTAGE_FRAGMENT, error);
    if (fragment_shader == nullptr) {
        SDL_ReleaseGPUShader(device_, vertex_shader);
        shutdown();
        return false;
    }

    const SDL_GPUVertexBufferDescription buffer_description {
        .slot = 0,
        .pitch = sizeof(ShapeInstance),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_INSTANCE,
        .instance_step_rate = 0,
    };
    const std::array<SDL_GPUVertexAttribute, 5> attributes {{
        {.location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
         .offset = offsetof(ShapeInstance, center_x)},
        {.location = 1, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
         .offset = offsetof(ShapeInstance, half_width)},
        {.location = 2, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
         .offset = offsetof(ShapeInstance, rotation_cosine)},
        {.location = 3, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
         .offset = offsetof(ShapeInstance, red)},
        {.location = 4, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT,
         .offset = offsetof(ShapeInstance, shape)},
    }};
    SDL_GPUColorTargetDescription color_target {};
    color_target.format = target_format;
    color_target.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA;
    color_target.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    color_target.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
    color_target.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
    color_target.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    color_target.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
    color_target.blend_state.enable_blend = true;

    SDL_GPUGraphicsPipelineCreateInfo pipeline_info {};
    pipeline_info.vertex_shader = vertex_shader;
    pipeline_info.fragment_shader = fragment_shader;
    pipeline_info.vertex_input_state.vertex_buffer_descriptions = &buffer_description;
    pipeline_info.vertex_input_state.num_vertex_buffers = 1;
    pipeline_info.vertex_input_state.vertex_attributes = attributes.data();
    pipeline_info.vertex_input_state.num_vertex_attributes =
        static_cast<std::uint32_t>(attributes.size());
    pipeline_info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    pipeline_info.target_info.color_target_descriptions = &color_target;
    pipeline_info.target_info.num_color_targets = 1;
    pipeline_ = SDL_CreateGPUGraphicsPipeline(device_, &pipeline_info);
    SDL_ReleaseGPUShader(device_, vertex_shader);
    SDL_ReleaseGPUShader(device_, fragment_shader);
    if (pipeline_ == nullptr) {
        error = SDL_GetError();
        shutdown();
        return false;
    }
    return true;
}

void WorldRenderer::shutdown() noexcept
{
    if (device_ != nullptr) {
        SDL_ReleaseGPUTransferBuffer(device_, transfer_buffer_);
        SDL_ReleaseGPUBuffer(device_, instance_buffer_);
        SDL_ReleaseGPUGraphicsPipeline(device_, pipeline_);
    }
    transfer_buffer_ = nullptr;
    instance_buffer_ = nullptr;
    pipeline_ = nullptr;
    device_ = nullptr;
    instance_capacity_ = 0;
    instance_count_ = 0;
    first_entity_instance_ = 0;
    entity_scissor_ = SDL_Rect {};
}

bool WorldRenderer::ensure_capacity(
    const std::uint32_t instance_count,
    std::string& error)
{
    if (instance_count <= instance_capacity_) {
        return true;
    }
    std::uint32_t capacity = std::max<std::uint32_t>(instance_capacity_, 1024);
    while (capacity < instance_count
        && capacity <= std::numeric_limits<std::uint32_t>::max() / 2) {
        capacity *= 2;
    }
    if (capacity < instance_count
        || capacity > std::numeric_limits<std::uint32_t>::max() / sizeof(ShapeInstance)) {
        error = "The world contains too many visible shapes to render.";
        return false;
    }

    SDL_GPUBufferCreateInfo buffer_info {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = capacity * static_cast<std::uint32_t>(sizeof(ShapeInstance)),
    };
    SDL_GPUTransferBufferCreateInfo transfer_info {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = buffer_info.size,
    };
    SDL_GPUBuffer* new_buffer = SDL_CreateGPUBuffer(device_, &buffer_info);
    SDL_GPUTransferBuffer* new_transfer =
        SDL_CreateGPUTransferBuffer(device_, &transfer_info);
    if (new_buffer == nullptr || new_transfer == nullptr) {
        SDL_ReleaseGPUBuffer(device_, new_buffer);
        SDL_ReleaseGPUTransferBuffer(device_, new_transfer);
        error = SDL_GetError();
        return false;
    }
    SDL_ReleaseGPUBuffer(device_, instance_buffer_);
    SDL_ReleaseGPUTransferBuffer(device_, transfer_buffer_);
    instance_buffer_ = new_buffer;
    transfer_buffer_ = new_transfer;
    instance_capacity_ = capacity;
    return true;
}

bool WorldRenderer::prepare(
    SDL_GPUCommandBuffer* command_buffer,
    const PixelViewport& viewport,
    const Camera& camera,
    const RenderSnapshot* snapshot,
    const WorldRenderOptions& options,
    std::string& error)
{
    if (viewport.width <= 0 || viewport.height <= 0) {
        instance_count_ = 0;
        return true;
    }
    const CameraViewport camera_viewport {
        .x = static_cast<double>(viewport.x),
        .y = static_cast<double>(viewport.y),
        .width = static_cast<double>(viewport.width),
        .height = static_cast<double>(viewport.height),
    };
    const double pixels_per_world = std::min(
        static_cast<double>(viewport.width) / camera.world_width(),
        static_cast<double>(viewport.height) / camera.world_height()) * camera.zoom();

    std::vector<ShapeInstance> instances;
    const std::size_t shapes_per_agent = options.show_agent_information ? 5 : 2;
    const std::size_t entity_count = snapshot == nullptr
        ? 0
        : snapshot->food.size() + snapshot->agents.size() * shapes_per_agent;
    const std::size_t debug_shape_estimate = snapshot != nullptr && options.show_debug
        ? snapshot->diagnostics.spatial_columns + snapshot->diagnostics.spatial_rows + 256
        : 0;
    instances.reserve(entity_count + debug_shape_estimate + 16);

    const auto add_screen_shape = [&](const float center_x,
                                      const float center_y,
                                      const float half_width,
                                      const float half_height,
                                      const float angle,
                                      const Color color,
                                      const float shape) {
        const float culling_half_extent = std::max(half_width, half_height);
        if (!intersects_viewport(
                center_x, center_y, culling_half_extent, culling_half_extent, viewport)) {
            return;
        }
        instances.push_back(ShapeInstance {
            .center_x = center_x,
            .center_y = center_y,
            .half_width = half_width,
            .half_height = half_height,
            .rotation_cosine = std::cos(angle),
            .rotation_sine = std::sin(angle),
            .red = color.red,
            .green = color.green,
            .blue = color.blue,
            .alpha = color.alpha,
            .shape = shape,
        });
    };
    const auto add_world_rectangle = [&](const double minimum_x,
                                         const double minimum_y,
                                         const double maximum_x,
                                         const double maximum_y,
                                         const Color color) {
        const Vec2 minimum = camera.world_to_screen(
            {.x = minimum_x, .y = minimum_y}, camera_viewport);
        const Vec2 maximum = camera.world_to_screen(
            {.x = maximum_x, .y = maximum_y}, camera_viewport);
        add_screen_shape(
            static_cast<float>((minimum.x + maximum.x) * 0.5),
            static_cast<float>((minimum.y + maximum.y) * 0.5),
            static_cast<float>((maximum.x - minimum.x) * 0.5),
            static_cast<float>((maximum.y - minimum.y) * 0.5),
            0.0F,
            color,
            rectangle_shape);
    };
    const auto add_boundary = [&](const double minimum_x,
                                  const double minimum_y,
                                  const double maximum_x,
                                  const double maximum_y,
                                  const Color color,
                                  const float width) {
        const Vec2 top_left = camera.world_to_screen(
            {.x = minimum_x, .y = minimum_y}, camera_viewport);
        const Vec2 bottom_right = camera.world_to_screen(
            {.x = maximum_x, .y = maximum_y}, camera_viewport);
        const float center_x = static_cast<float>((top_left.x + bottom_right.x) * 0.5);
        const float center_y = static_cast<float>((top_left.y + bottom_right.y) * 0.5);
        const float half_width = static_cast<float>((bottom_right.x - top_left.x) * 0.5);
        const float half_height = static_cast<float>((bottom_right.y - top_left.y) * 0.5);
        add_screen_shape(center_x, static_cast<float>(top_left.y), half_width, width, 0.0F,
            color, rectangle_shape);
        add_screen_shape(center_x, static_cast<float>(bottom_right.y), half_width, width, 0.0F,
            color, rectangle_shape);
        add_screen_shape(static_cast<float>(top_left.x), center_y, width, half_height, 0.0F,
            color, rectangle_shape);
        add_screen_shape(static_cast<float>(bottom_right.x), center_y, width, half_height, 0.0F,
            color, rectangle_shape);
    };

    const WorldBounds outer = camera.outer_bounds();
    add_world_rectangle(outer.minimum_x, outer.minimum_y,
        outer.maximum_x, outer.maximum_y, outer_fill);
    add_world_rectangle(0.0, 0.0,
        camera.world_width(), camera.world_height(), world_fill);
    add_boundary(outer.minimum_x, outer.minimum_y,
        outer.maximum_x, outer.maximum_y, outer_boundary, 1.0F);
    add_boundary(0.0, 0.0,
        camera.world_width(), camera.world_height(), world_boundary, 1.25F);

    if (snapshot != nullptr && options.show_debug
        && snapshot->diagnostics.spatial_columns > 0
        && snapshot->diagnostics.spatial_rows > 0) {
        const std::size_t columns = snapshot->diagnostics.spatial_columns;
        const std::size_t rows = snapshot->diagnostics.spatial_rows;
        const double cell_width = camera.world_width() / static_cast<double>(columns);
        const double cell_height = camera.world_height() / static_cast<double>(rows);
        std::vector<bool> occupied(columns * rows, false);
        const auto mark_occupied = [&](const double x, const double y) {
            const std::size_t column = std::min(columns - 1,
                static_cast<std::size_t>(x / cell_width));
            const std::size_t row = std::min(rows - 1,
                static_cast<std::size_t>(y / cell_height));
            occupied[row * columns + column] = true;
        };
        for (const AgentVisual& agent : snapshot->agents) mark_occupied(agent.x, agent.y);
        for (const FoodVisual& food : snapshot->food) mark_occupied(food.x, food.y);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                if (!occupied[row * columns + column]) continue;
                add_world_rectangle(column * cell_width, row * cell_height,
                    (column + 1) * cell_width, (row + 1) * cell_height,
                    debug_occupied);
            }
        }

        if (snapshot->selected_agent) {
            const double query_radius = snapshot->eye_range
                + std::hypot(0.60, 0.50) * snapshot->agent_radius
                + std::max(snapshot->agent_radius, snapshot->food_radius);
            const long long minimum_column = static_cast<long long>(std::floor(
                (snapshot->selected_agent->position.x - query_radius) / cell_width));
            const long long maximum_column = static_cast<long long>(std::floor(
                (snapshot->selected_agent->position.x + query_radius) / cell_width));
            const long long minimum_row = static_cast<long long>(std::floor(
                (snapshot->selected_agent->position.y - query_radius) / cell_height));
            const long long maximum_row = static_cast<long long>(std::floor(
                (snapshot->selected_agent->position.y + query_radius) / cell_height));
            const auto wrap_index = [](const long long value, const std::size_t count) {
                const long long signed_count = static_cast<long long>(count);
                const long long remainder = value % signed_count;
                return static_cast<std::size_t>(
                    remainder < 0 ? remainder + signed_count : remainder);
            };
            for (long long raw_row = minimum_row; raw_row <= maximum_row; ++raw_row) {
                const std::size_t row = wrap_index(raw_row, rows);
                for (long long raw_column = minimum_column;
                     raw_column <= maximum_column; ++raw_column) {
                    const std::size_t column = wrap_index(raw_column, columns);
                    add_world_rectangle(column * cell_width, row * cell_height,
                        (column + 1) * cell_width, (row + 1) * cell_height,
                        debug_query);
                }
            }
        }

        const float line_half_width = 0.5F;
        for (std::size_t column = 1; column < columns; ++column) {
            const Vec2 top = camera.world_to_screen(
                {.x = column * cell_width, .y = 0.0}, camera_viewport);
            const Vec2 bottom = camera.world_to_screen(
                {.x = column * cell_width, .y = camera.world_height()}, camera_viewport);
            add_screen_shape(static_cast<float>(top.x),
                static_cast<float>((top.y + bottom.y) * 0.5), line_half_width,
                static_cast<float>((bottom.y - top.y) * 0.5), 0.0F,
                debug_grid, rectangle_shape);
        }
        for (std::size_t row = 1; row < rows; ++row) {
            const Vec2 left = camera.world_to_screen(
                {.x = 0.0, .y = row * cell_height}, camera_viewport);
            const Vec2 right = camera.world_to_screen(
                {.x = camera.world_width(), .y = row * cell_height}, camera_viewport);
            add_screen_shape(static_cast<float>((left.x + right.x) * 0.5),
                static_cast<float>(left.y), static_cast<float>((right.x - left.x) * 0.5),
                line_half_width, 0.0F, debug_grid, rectangle_shape);
        }
    }
    first_entity_instance_ = static_cast<std::uint32_t>(instances.size());

    // Toroidal copies are positioned across an edge, then clipped back to the
    // configured world so no entity fragment is visible in the camera-only exterior.
    const Vec2 world_top_left = camera.world_to_screen(
        {.x = 0.0, .y = 0.0}, camera_viewport);
    const Vec2 world_bottom_right = camera.world_to_screen(
        {.x = camera.world_width(), .y = camera.world_height()}, camera_viewport);
    const int entity_left = std::clamp(
        static_cast<int>(std::floor(world_top_left.x)),
        viewport.x,
        viewport.x + viewport.width);
    const int entity_top = std::clamp(
        static_cast<int>(std::floor(world_top_left.y)),
        viewport.y,
        viewport.y + viewport.height);
    const int entity_right = std::clamp(
        static_cast<int>(std::ceil(world_bottom_right.x)),
        entity_left,
        viewport.x + viewport.width);
    const int entity_bottom = std::clamp(
        static_cast<int>(std::ceil(world_bottom_right.y)),
        entity_top,
        viewport.y + viewport.height);
    entity_scissor_ = SDL_Rect {
        .x = entity_left,
        .y = entity_top,
        .w = entity_right - entity_left,
        .h = entity_bottom - entity_top,
    };

    if (snapshot != nullptr) {
        const float food_radius = static_cast<float>(
            std::max(snapshot->food_radius * pixels_per_world, 1.0));
        const float agent_radius = static_cast<float>(
            std::max(snapshot->agent_radius * pixels_per_world, 2.0));
        const double wrap_margin = (agent_radius * 2.5) / pixels_per_world;

        const auto offsets_for = [&](const double coordinate, const double extent) {
            std::array<double, 3> offsets {0.0, 0.0, 0.0};
            std::size_t count = 1;
            if (coordinate < wrap_margin) {
                offsets[count++] = extent;
            }
            if (coordinate > extent - wrap_margin) {
                offsets[count++] = -extent;
            }
            return std::pair {offsets, count};
        };

        for (const FoodVisual& food : snapshot->food) {
            const auto [x_offsets, x_count] = offsets_for(food.x, camera.world_width());
            const auto [y_offsets, y_count] = offsets_for(food.y, camera.world_height());
            for (std::size_t x = 0; x < x_count; ++x) {
                for (std::size_t y = 0; y < y_count; ++y) {
                    const Vec2 screen = camera.world_to_screen(
                        {.x = food.x + x_offsets[x], .y = food.y + y_offsets[y]},
                        camera_viewport);
                    add_screen_shape(static_cast<float>(screen.x), static_cast<float>(screen.y),
                        food_radius, food_radius, 0.0F, food_color, circle_shape);
                }
            }
        }

        if (options.selected_agent_id) {
            for (const AgentVisual& agent : snapshot->agents) {
                if (agent.id != *options.selected_agent_id) {
                    continue;
                }
                const AgentScreenCopies copies =
                    agent_screen_copies(agent, camera, camera_viewport);
                for (std::size_t index = 0; index < copies.count; ++index) {
                    add_screen_shape(
                        static_cast<float>(copies.centers[index].x),
                        static_cast<float>(copies.centers[index].y),
                        agent_radius + 4.0F,
                        agent_radius + 4.0F,
                        0.0F,
                        selection_glow,
                        circle_shape);
                }
                break;
            }
        }

        // Draw headings before bodies so the line begins visually beneath the circle.
        for (const AgentVisual& agent : snapshot->agents) {
            const auto [x_offsets, x_count] = offsets_for(agent.x, camera.world_width());
            const auto [y_offsets, y_count] = offsets_for(agent.y, camera.world_height());
            for (std::size_t x = 0; x < x_count; ++x) {
                for (std::size_t y = 0; y < y_count; ++y) {
                    const Vec2 screen = camera.world_to_screen(
                        {.x = agent.x + x_offsets[x], .y = agent.y + y_offsets[y]},
                        camera_viewport);
                    const float line_length = agent_radius * 2.4F;
                    const float center_x = static_cast<float>(screen.x)
                        + std::cos(agent.direction) * line_length * 0.5F;
                    const float center_y = static_cast<float>(screen.y)
                        + std::sin(agent.direction) * line_length * 0.5F;
                    add_screen_shape(center_x, center_y, line_length * 0.5F,
                        std::max(0.75F, agent_radius * 0.18F), agent.direction,
                        heading_color, rectangle_shape);
                }
            }
        }
        for (const AgentVisual& agent : snapshot->agents) {
            const auto [x_offsets, x_count] = offsets_for(agent.x, camera.world_width());
            const auto [y_offsets, y_count] = offsets_for(agent.y, camera.world_height());
            for (std::size_t x = 0; x < x_count; ++x) {
                for (std::size_t y = 0; y < y_count; ++y) {
                    const Vec2 screen = camera.world_to_screen(
                        {.x = agent.x + x_offsets[x], .y = agent.y + y_offsets[y]},
                        camera_viewport);
                    add_screen_shape(static_cast<float>(screen.x),
                        static_cast<float>(screen.y), agent_radius, agent_radius,
                        0.0F, Color {agent.red, agent.green, agent.blue, 1.0F}, circle_shape);
                }
            }
        }
        if (options.show_agent_information) {
            const float bar_half_width = std::clamp(agent_radius * 1.8F, 6.0F, 32.0F);
            const float bar_half_height = std::clamp(agent_radius * 0.18F, 1.0F, 3.0F);
            for (const AgentVisual& agent : snapshot->agents) {
                const float fraction = static_cast<float>(std::clamp(
                    static_cast<double>(agent.energy) / snapshot->reproduction_threshold,
                    0.0,
                    1.0));
                const AgentScreenCopies copies =
                    agent_screen_copies(agent, camera, camera_viewport);
                for (std::size_t index = 0; index < copies.count; ++index) {
                    const float center_x = static_cast<float>(copies.centers[index].x);
                    const float center_y = static_cast<float>(copies.centers[index].y)
                        - agent_radius - bar_half_height - 2.0F;
                    add_screen_shape(center_x, center_y, bar_half_width,
                        bar_half_height, 0.0F, energy_background, rectangle_shape);
                    if (fraction > 0.0F) {
                        const float filled_half_width = bar_half_width * fraction;
                        // Anchor the fill at the left edge rather than shrinking
                        // toward the center as the agent spends energy.
                        const float filled_center_x = center_x - bar_half_width
                            + filled_half_width;
                        add_screen_shape(filled_center_x, center_y, filled_half_width,
                            bar_half_height, 0.0F, energy_fill, rectangle_shape);
                    }
                    const float marker_radius = std::max(1.5F, agent_radius * 0.32F);
                    add_screen_shape(center_x - agent_radius * 0.60F,
                        static_cast<float>(copies.centers[index].y) + agent_radius * 0.60F,
                        marker_radius, marker_radius, 0.0F,
                        agent.diet == Diet::herbivore ? herbivore_marker : carnivore_marker,
                        circle_shape);
                }
            }

            if (snapshot->selected_agent) {
                const SelectedAgentDetails& selected = *snapshot->selected_agent;
                const double forward_x = std::cos(selected.direction);
                const double forward_y = std::sin(selected.direction);
                const double left_x = -forward_y;
                const double left_y = forward_x;
                constexpr std::array<double, vision_ray_count> offsets {
                    -std::numbers::pi_v<double> / 2.0,
                    -std::numbers::pi_v<double> / 4.0, 0.0, 0.0,
                    std::numbers::pi_v<double> / 4.0,
                    std::numbers::pi_v<double> / 2.0,
                };
                const auto add_world_line = [&](const Vec2 start, const Vec2 end,
                                                const Color color, const float width) {
                    // Nine copies make a boundary-crossing toroidal segment reappear
                    // on the opposite side while the world scissor clips excess parts.
                    for (int x = -1; x <= 1; ++x) {
                        for (int y = -1; y <= 1; ++y) {
                            const Vec2 screen_start = camera.world_to_screen(
                                {.x = start.x + x * camera.world_width(),
                                    .y = start.y + y * camera.world_height()}, camera_viewport);
                            const Vec2 screen_end = camera.world_to_screen(
                                {.x = end.x + x * camera.world_width(),
                                    .y = end.y + y * camera.world_height()}, camera_viewport);
                            const float delta_x = static_cast<float>(screen_end.x - screen_start.x);
                            const float delta_y = static_cast<float>(screen_end.y - screen_start.y);
                            const float length = std::hypot(delta_x, delta_y);
                            add_screen_shape(static_cast<float>((screen_start.x + screen_end.x) * 0.5),
                                static_cast<float>((screen_start.y + screen_end.y) * 0.5),
                                length * 0.5F, width, std::atan2(delta_y, delta_x), color,
                                rectangle_shape);
                        }
                    }
                };
                const auto add_world_point = [&](const Vec2 point, const float radius,
                                                 const Color color) {
                    for (int x = -1; x <= 1; ++x) {
                        for (int y = -1; y <= 1; ++y) {
                            const Vec2 screen = camera.world_to_screen(
                                {.x = point.x + x * camera.world_width(),
                                    .y = point.y + y * camera.world_height()}, camera_viewport);
                            add_screen_shape(static_cast<float>(screen.x),
                                static_cast<float>(screen.y), radius, radius, 0.0F,
                                color, circle_shape);
                        }
                    }
                };
                for (std::size_t ray_index = 0; ray_index < vision_ray_count; ++ray_index) {
                    const double side = ray_index < rays_per_eye ? 0.50 : -0.50;
                    const Vec2 eye {
                        .x = selected.position.x + forward_x * snapshot->agent_radius * 0.60
                            + left_x * snapshot->agent_radius * side,
                        .y = selected.position.y + forward_y * snapshot->agent_radius * 0.60
                            + left_y * snapshot->agent_radius * side,
                    };
                    const double angle = selected.direction + offsets[ray_index];
                    const Vec2 end {.x = eye.x + std::cos(angle) * snapshot->eye_range,
                        .y = eye.y + std::sin(angle) * snapshot->eye_range};
                    add_world_line(eye, end, eye_geometry, 0.75F);
                    if (ray_index == 0 || ray_index == rays_per_eye) {
                        add_world_point(eye, std::max(1.5F, agent_radius * 0.18F),
                            eye_position);
                    }
                }
                const Vec2 mouth {.x = selected.position.x
                        + forward_x * snapshot->agent_radius,
                    .y = selected.position.y + forward_y * snapshot->agent_radius};
                add_world_point(mouth, std::max(2.0F, agent_radius * 0.22F),
                    mouth_position);
            }
        }
    }

    if (instances.size() > std::numeric_limits<std::uint32_t>::max()) {
        error = "The world contains too many visible shapes to render.";
        return false;
    }
    instance_count_ = static_cast<std::uint32_t>(instances.size());
    if (!ensure_capacity(instance_count_, error)) {
        return false;
    }

    void* mapped = SDL_MapGPUTransferBuffer(device_, transfer_buffer_, true);
    if (mapped == nullptr) {
        error = SDL_GetError();
        return false;
    }
    std::memcpy(mapped, instances.data(), instances.size() * sizeof(ShapeInstance));
    SDL_UnmapGPUTransferBuffer(device_, transfer_buffer_);

    SDL_GPUCopyPass* copy_pass = SDL_BeginGPUCopyPass(command_buffer);
    if (copy_pass == nullptr) {
        error = SDL_GetError();
        return false;
    }
    SDL_GPUTransferBufferLocation source {.transfer_buffer = transfer_buffer_, .offset = 0};
    SDL_GPUBufferRegion destination {
        .buffer = instance_buffer_,
        .offset = 0,
        .size = instance_count_ * static_cast<std::uint32_t>(sizeof(ShapeInstance)),
    };
    SDL_UploadToGPUBuffer(copy_pass, &source, &destination, true);
    SDL_EndGPUCopyPass(copy_pass);

    const std::array<float, 4> uniforms {
        static_cast<float>(viewport.target_width),
        static_cast<float>(viewport.target_height),
        0.0F,
        0.0F,
    };
    SDL_PushGPUVertexUniformData(
        command_buffer, 0, uniforms.data(), static_cast<std::uint32_t>(sizeof(uniforms)));
    return true;
}

void WorldRenderer::draw(
    SDL_GPURenderPass* render_pass,
    const PixelViewport& viewport) const
{
    if (instance_count_ == 0 || pipeline_ == nullptr) {
        return;
    }
    const SDL_Rect scissor {
        .x = viewport.x,
        .y = viewport.y,
        .w = viewport.width,
        .h = viewport.height,
    };
    const SDL_GPUBufferBinding binding {.buffer = instance_buffer_, .offset = 0};
    SDL_SetGPUScissor(render_pass, &scissor);
    SDL_BindGPUGraphicsPipeline(render_pass, pipeline_);
    SDL_BindGPUVertexBuffers(render_pass, 0, &binding, 1);
    SDL_DrawGPUPrimitives(render_pass, 6, first_entity_instance_, 0, 0);

    const std::uint32_t entity_count = instance_count_ - first_entity_instance_;
    if (entity_count != 0 && entity_scissor_.w > 0 && entity_scissor_.h > 0) {
        SDL_SetGPUScissor(render_pass, &entity_scissor_);
        SDL_DrawGPUPrimitives(
            render_pass, 6, entity_count, 0, first_entity_instance_);
    }
}

} // namespace evobrain::viewer
