#pragma once

#include "evobrain/viewer/agent_selection.hpp"
#include "evobrain/viewer/camera.hpp"
#include "evobrain/viewer/simulation_worker.hpp"

#include <SDL3/SDL_gpu.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace evobrain::viewer {

// Defines the swapchain-pixel rectangle occupied by the world view.
struct PixelViewport {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    int target_width = 0;
    int target_height = 0;
};

// Supplies viewer-only choices that alter world presentation, never simulation.
struct WorldRenderOptions {
    bool show_agent_information = false;
    bool show_debug = false;
    std::optional<std::uint64_t> selected_agent_id;
};

// Batches the world, entities, headings, and boundaries into one GPU draw.
class WorldRenderer {
public:
    // Creates an empty renderer that owns no SDL_GPU resources.
    WorldRenderer() = default;

    // Releases resources if the renderer is destroyed after initialization.
    ~WorldRenderer();

    WorldRenderer(const WorldRenderer&) = delete;
    WorldRenderer& operator=(const WorldRenderer&) = delete;

    // Loads build-compiled shaders and creates the D3D12 graphics pipeline.
    [[nodiscard]] bool initialize(
        SDL_GPUDevice* device,
        SDL_GPUTextureFormat target_format,
        const std::filesystem::path& shader_directory,
        std::string& error);

    // Releases every buffer and pipeline owned by this renderer.
    void shutdown() noexcept;

    // Builds and uploads one batched frame before its render pass begins.
    [[nodiscard]] bool prepare(
        SDL_GPUCommandBuffer* command_buffer,
        const PixelViewport& viewport,
        const Camera& camera,
        const RenderSnapshot* snapshot,
        const WorldRenderOptions& options,
        std::string& error);

    // Issues the prepared frame as one instanced draw inside the world scissor.
    void draw(SDL_GPURenderPass* render_pass, const PixelViewport& viewport) const;

private:
    struct ShapeInstance;

    // Grows the reusable upload and GPU buffers without shrinking each frame.
    [[nodiscard]] bool ensure_capacity(std::uint32_t instance_count, std::string& error);

    SDL_GPUDevice* device_ = nullptr;
    SDL_GPUGraphicsPipeline* pipeline_ = nullptr;
    SDL_GPUBuffer* instance_buffer_ = nullptr;
    SDL_GPUTransferBuffer* transfer_buffer_ = nullptr;
    std::uint32_t instance_capacity_ = 0;
    std::uint32_t instance_count_ = 0;
    std::uint32_t first_entity_instance_ = 0;
    SDL_Rect entity_scissor_ {};
};

} // namespace evobrain::viewer
