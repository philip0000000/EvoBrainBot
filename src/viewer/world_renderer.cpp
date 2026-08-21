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
constexpr Color agent_color {0.08F, 0.40F, 0.75F, 1.0F};
constexpr Color heading_color {0.04F, 0.22F, 0.48F, 1.0F};
constexpr Color food_color {0.10F, 0.55F, 0.22F, 1.0F};
constexpr Color selection_glow {1.0F, 0.68F, 0.08F, 0.68F};
constexpr Color energy_background {0.08F, 0.10F, 0.12F, 0.92F};
constexpr Color energy_fill {0.16F, 0.72F, 0.28F, 1.0F};

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
    const double pixels_per_world = std::min(viewport.width, viewport.height) * camera.zoom();

    std::vector<ShapeInstance> instances;
    const std::size_t shapes_per_agent = options.show_agent_information ? 4 : 2;
    const std::size_t entity_count = snapshot == nullptr
        ? 0
        : snapshot->food.size() + snapshot->agents.size() * shapes_per_agent;
    instances.reserve(entity_count + 16);

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
    const auto add_boundary = [&](const double minimum,
                                  const double maximum,
                                  const Color color,
                                  const float width) {
        const Vec2 top_left = camera.world_to_screen(
            {.x = minimum, .y = minimum}, camera_viewport);
        const Vec2 bottom_right = camera.world_to_screen(
            {.x = maximum, .y = maximum}, camera_viewport);
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

    add_world_rectangle(Camera::outer_minimum, Camera::outer_minimum,
        Camera::outer_maximum, Camera::outer_maximum, outer_fill);
    add_world_rectangle(0.0, 0.0, 1.0, 1.0, world_fill);
    add_boundary(Camera::outer_minimum, Camera::outer_maximum, outer_boundary, 1.0F);
    add_boundary(0.0, 1.0, world_boundary, 1.25F);
    first_entity_instance_ = static_cast<std::uint32_t>(instances.size());

    // Toroidal copies are positioned across an edge, then clipped back to the
    // unit square so no entity fragment is visible in the camera-only exterior.
    const Vec2 world_top_left = camera.world_to_screen(
        {.x = 0.0, .y = 0.0}, camera_viewport);
    const Vec2 world_bottom_right = camera.world_to_screen(
        {.x = 1.0, .y = 1.0}, camera_viewport);
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
            std::clamp(2.0 * camera.zoom(), 1.0, 10.0));
        const float agent_radius = static_cast<float>(agent_visual_radius(camera.zoom()));
        const double wrap_margin = (agent_radius * 2.5) / pixels_per_world;

        const auto offsets_for = [&](const double coordinate) {
            std::array<double, 3> offsets {0.0, 0.0, 0.0};
            std::size_t count = 1;
            if (coordinate < wrap_margin) {
                offsets[count++] = 1.0;
            }
            if (coordinate > 1.0 - wrap_margin) {
                offsets[count++] = -1.0;
            }
            return std::pair {offsets, count};
        };

        for (const FoodVisual& food : snapshot->food) {
            const auto [x_offsets, x_count] = offsets_for(food.x);
            const auto [y_offsets, y_count] = offsets_for(food.y);
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
            const auto [x_offsets, x_count] = offsets_for(agent.x);
            const auto [y_offsets, y_count] = offsets_for(agent.y);
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
            const auto [x_offsets, x_count] = offsets_for(agent.x);
            const auto [y_offsets, y_count] = offsets_for(agent.y);
            for (std::size_t x = 0; x < x_count; ++x) {
                for (std::size_t y = 0; y < y_count; ++y) {
                    const Vec2 screen = camera.world_to_screen(
                        {.x = agent.x + x_offsets[x], .y = agent.y + y_offsets[y]},
                        camera_viewport);
                    add_screen_shape(static_cast<float>(screen.x),
                        static_cast<float>(screen.y), agent_radius, agent_radius,
                        0.0F, agent_color, circle_shape);
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
                }
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
