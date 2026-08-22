#include "world_renderer.hpp"

#include <SDL3/SDL.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <string>

namespace {

constexpr std::uint32_t target_width = 2560;
constexpr std::uint32_t target_height = 1440;
constexpr SDL_GPUTextureFormat target_format =
    SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;

// Builds a stable, evenly distributed render workload without running simulation logic.
evobrain::viewer::RenderSnapshot make_snapshot(
    const std::size_t agent_count,
    const std::size_t food_count)
{
    evobrain::viewer::RenderSnapshot snapshot;
    snapshot.stats.population = agent_count;
    snapshot.stats.food = food_count;
    snapshot.reproduction_threshold = 1.0;
    snapshot.agents.reserve(agent_count);
    snapshot.food.reserve(food_count);

    const auto coordinate = [](const std::size_t index, const std::size_t count) {
        // Irrational-looking multipliers prevent rows of overlapping objects
        // while remaining exactly reproducible between benchmark runs.
        return static_cast<float>(
            std::fmod((static_cast<double>(index) + 0.5) * count, 1'000'003.0)
            / 1'000'003.0 * 2.5);
    };
    for (std::size_t index = 0; index < food_count; ++index) {
        snapshot.food.push_back({
            .x = coordinate(index, 7919),
            .y = coordinate(index, 1543),
        });
    }
    for (std::size_t index = 0; index < agent_count; ++index) {
        snapshot.agents.push_back({
            .id = static_cast<std::uint64_t>(index + 1),
            .x = coordinate(index, 3571),
            .y = coordinate(index, 6827),
            .direction = static_cast<float>(
                std::fmod(index * 0.6180339887498948, 1.0)
                * 2.0 * std::numbers::pi),
            .energy = static_cast<float>(std::fmod(index * 0.1732050807568877, 1.25)),
        });
    }
    return snapshot;
}

// Records one complete upload-and-draw frame into the offscreen D3D12 target.
bool render_frame(
    SDL_GPUDevice* device,
    SDL_GPUTexture* target,
    evobrain::viewer::WorldRenderer& renderer,
    const evobrain::viewer::PixelViewport& viewport,
    const evobrain::viewer::Camera& camera,
    const evobrain::viewer::RenderSnapshot& snapshot,
    const evobrain::viewer::WorldRenderOptions& options,
    std::string& error)
{
    SDL_GPUCommandBuffer* command_buffer = SDL_AcquireGPUCommandBuffer(device);
    if (command_buffer == nullptr) {
        error = SDL_GetError();
        return false;
    }
    if (!renderer.prepare(command_buffer, viewport, camera, &snapshot, options, error)) {
        SDL_CancelGPUCommandBuffer(command_buffer);
        return false;
    }

    SDL_GPUColorTargetInfo color_target {};
    color_target.texture = target;
    color_target.clear_color = SDL_FColor {0.035F, 0.040F, 0.050F, 1.0F};
    color_target.load_op = SDL_GPU_LOADOP_CLEAR;
    color_target.store_op = SDL_GPU_STOREOP_STORE;
    color_target.cycle = true;
    SDL_GPURenderPass* render_pass =
        SDL_BeginGPURenderPass(command_buffer, &color_target, 1, nullptr);
    if (render_pass == nullptr) {
        error = SDL_GetError();
        SDL_CancelGPUCommandBuffer(command_buffer);
        return false;
    }
    renderer.draw(render_pass, viewport);
    SDL_EndGPURenderPass(render_pass);
    if (!SDL_SubmitGPUCommandBuffer(command_buffer)) {
        error = SDL_GetError();
        return false;
    }
    return true;
}

// Measures the production renderer after warm-up and waits for GPU completion.
double run_scenario(
    SDL_GPUDevice* device,
    SDL_GPUTexture* target,
    evobrain::viewer::WorldRenderer& renderer,
    const evobrain::viewer::PixelViewport& viewport,
    const evobrain::viewer::Camera& camera,
    const std::size_t agents,
    const std::size_t food,
    const int measured_frames,
    const evobrain::viewer::WorldRenderOptions& options,
    std::string& error)
{
    const evobrain::viewer::RenderSnapshot snapshot = make_snapshot(agents, food);
    for (int frame = 0; frame < 10; ++frame) {
        if (!render_frame(
                device, target, renderer, viewport, camera, snapshot, options, error)) {
            return 0.0;
        }
    }
    if (!SDL_WaitForGPUIdle(device)) {
        error = SDL_GetError();
        return 0.0;
    }

    const auto start = std::chrono::steady_clock::now();
    for (int frame = 0; frame < measured_frames; ++frame) {
        if (!render_frame(
                device, target, renderer, viewport, camera, snapshot, options, error)) {
            return 0.0;
        }
    }
    if (!SDL_WaitForGPUIdle(device)) {
        error = SDL_GetError();
        return 0.0;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return measured_frames / seconds;
}

// Runs render-only acceptance and observation workloads on the D3D12 backend.
int run_benchmark()
{
    // SDL's D3D12 GPU backend belongs to the video subsystem even though this
    // benchmark deliberately creates no window or swapchain.
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << '\n';
        return EXIT_FAILURE;
    }
    SDL_GPUDevice* device =
        SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_DXIL, false, "direct3d12");
    if (device == nullptr) {
        std::cerr << "D3D12 device creation failed: " << SDL_GetError() << '\n';
        SDL_Quit();
        return EXIT_FAILURE;
    }
    if (!SDL_GPUTextureSupportsFormat(device, target_format,
            SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
        std::cerr << "The benchmark render-target format is unsupported.\n";
        SDL_DestroyGPUDevice(device);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    SDL_GPUTextureCreateInfo texture_info {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = target_format,
        .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
        .width = target_width,
        .height = target_height,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1,
    };
    SDL_GPUTexture* target = SDL_CreateGPUTexture(device, &texture_info);
    if (target == nullptr) {
        std::cerr << "Offscreen target creation failed: " << SDL_GetError() << '\n';
        SDL_DestroyGPUDevice(device);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    evobrain::viewer::WorldRenderer renderer;
    std::string error;
    const std::filesystem::path shader_directory =
        std::filesystem::u8path(SDL_GetBasePath()) / L"shaders";
    if (!renderer.initialize(device, target_format, shader_directory, error)) {
        std::cerr << "Renderer initialization failed: " << error << '\n';
        SDL_ReleaseGPUTexture(device, target);
        SDL_DestroyGPUDevice(device);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    const evobrain::viewer::PixelViewport viewport {
        .width = static_cast<int>(target_width),
        .height = static_cast<int>(target_height),
        .target_width = static_cast<int>(target_width),
        .target_height = static_cast<int>(target_height),
    };
    evobrain::viewer::Camera camera;
    camera.set_world_dimensions(2.5, 2.5, {
        .width = static_cast<double>(target_width),
        .height = static_cast<double>(target_height),
    });
    camera.reset({
        .width = static_cast<double>(target_width),
        .height = static_cast<double>(target_height),
    });

    const double acceptance_fps = run_scenario(
        device, target, renderer, viewport, camera, 50'000, 50'000, 120, {}, error);
    if (acceptance_fps == 0.0) {
        std::cerr << "Acceptance scenario failed: " << error << '\n';
    } else {
        std::cout << "50,000 agents + 50,000 food: "
                  << acceptance_fps << " render-only FPS\n";
    }
    const double observation_fps = acceptance_fps == 0.0 ? 0.0 : run_scenario(
        device, target, renderer, viewport, camera, 100'000, 100'000, 60, {}, error);
    if (acceptance_fps != 0.0 && observation_fps == 0.0) {
        std::cerr << "Observation scenario failed: " << error << '\n';
    } else if (observation_fps != 0.0) {
        std::cout << "100,000 agents + 100,000 food (informational): "
                  << observation_fps << " render-only FPS\n";
    }
    const double overlay_fps = acceptance_fps == 0.0 ? 0.0 : run_scenario(
        device,
        target,
        renderer,
        viewport,
        camera,
        50'000,
        50'000,
        60,
        {.show_agent_information = true, .selected_agent_id = 1},
        error);
    if (acceptance_fps != 0.0 && overlay_fps == 0.0) {
        std::cerr << "Agent-information scenario failed: " << error << '\n';
    } else if (overlay_fps != 0.0) {
        std::cout << "50,000 agents + 50,000 food + information overlays (informational): "
                  << overlay_fps << " render-only FPS\n";
    }

    renderer.shutdown();
    SDL_ReleaseGPUTexture(device, target);
    SDL_DestroyGPUDevice(device);
    SDL_Quit();
    // The larger workload is deliberately informational and cannot turn a
    // successful acceptance workload into a failed benchmark.
    return acceptance_fps >= 75.0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

// Runs the standalone render-only performance verification tool.
int main()
{
    return run_benchmark();
}
