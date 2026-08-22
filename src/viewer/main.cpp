#include "world_renderer.hpp"

#include "evobrain/viewer/camera.hpp"
#include "evobrain/viewer/simulation_worker.hpp"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlgpu3.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kDefaultWidth = 1600;
constexpr int kDefaultHeight = 900;
constexpr int kMinimumWidth = 960;
constexpr int kMinimumHeight = 600;

struct UiError {
    std::string summary;
    std::string detail;
    bool open_popup = true;
};

// Returns the user-facing name of one playback state for the window title.
const char* playback_state_name(const evobrain::viewer::PlaybackState playback)
{
    switch (playback) {
    case evobrain::viewer::PlaybackState::paused:
        return "Paused";
    case evobrain::viewer::PlaybackState::running:
        return "Running";
    case evobrain::viewer::PlaybackState::fast_forward:
        return "Fast-forward";
    }
    return "Unknown";
}

// Updates the native title only when its checkpoint or worker state has changed,
// returning SDL diagnostics if the native operation fails.
std::optional<std::string> update_window_title(
    SDL_Window* window,
    const std::string& filename,
    const evobrain::viewer::WorkerStatus& status,
    std::string& previous_title)
{
    std::string title = "EvoBrainBot Viewer \xE2\x80\x94 ";
    title += status.has_simulation ? filename : "No checkpoint";
    title += " \xE2\x80\x94 ";
    title += playback_state_name(status.playback);
    if (status.has_unsaved_changes) {
        title += " *";
    }
    if (title == previous_title) {
        return std::nullopt;
    }
    previous_title = title;
    if (!SDL_SetWindowTitle(window, title.c_str())) {
        return std::string {SDL_GetError()};
    }
    return std::nullopt;
}

struct DialogState {
    std::mutex mutex;
    bool ready = false;
    bool failed = false;
    std::string selected_path;
    std::string error;
};

struct ViewerUiResult {
    bool request_open = false;
    bool request_save = false;
    bool request_save_as = false;
    bool request_exit = false;
    bool toggle_run = false;
    bool step = false;
    bool toggle_fast_forward = false;
    bool confirm_replace = false;
    bool cancel_replace = false;
    bool confirm_overwrite = false;
    bool cancel_overwrite = false;
    bool exit_save = false;
    bool exit_discard = false;
    bool exit_cancel = false;
    std::optional<int> target_ticks_per_second;
    bool selection_requested = false;
    std::optional<std::uint64_t> selected_agent_id;
    evobrain::viewer::CameraViewport world_viewport;
};

struct WorldCanvasResult {
    evobrain::viewer::CameraViewport viewport;
    bool selection_requested = false;
    std::optional<std::uint64_t> selected_agent_id;
};

struct BrainLayerView {
    std::string_view name;
};

struct BrainNodeView {
    std::size_t layer = 0;
    std::string_view name;
    std::optional<double> bias;
};

struct BrainConnectionView {
    std::size_t source = 0;
    std::size_t target = 0;
    double weight = 0.0;
};

struct BrainViewState {
    ImVec2 pan {0.0F, 0.0F};
    float zoom = 1.0F;
    std::optional<std::uint64_t> agent_id;
    bool reset_requested = true;
};

// Native dialog callbacks can outlive a frame and run on another thread.
DialogState g_open_dialog_result;
DialogState g_save_dialog_result;

// Shows startup errors without requiring the viewer UI to have initialized.
int show_startup_error(const char* summary, const std::string& detail)
{
    const std::string message = std::string(summary) + "\n\n" + detail;
    SDL_ShowSimpleMessageBox(
        SDL_MESSAGEBOX_ERROR, "EvoBrainBot Viewer", message.c_str(), nullptr);
    return EXIT_FAILURE;
}

// Copies the asynchronous native file dialog result into process-lifetime state.
void SDLCALL open_dialog_callback(
    void*,
    const char* const* file_list,
    int)
{
    std::lock_guard lock(g_open_dialog_result.mutex);
    g_open_dialog_result.ready = true;
    g_open_dialog_result.failed = file_list == nullptr;
    g_open_dialog_result.error = file_list == nullptr ? SDL_GetError() : "";
    g_open_dialog_result.selected_path =
        file_list != nullptr && file_list[0] != nullptr ? file_list[0] : "";
}

// Copies the asynchronous native save dialog result into process-lifetime state.
void SDLCALL save_dialog_callback(
    void*,
    const char* const* file_list,
    int)
{
    std::lock_guard lock(g_save_dialog_result.mutex);
    g_save_dialog_result.ready = true;
    g_save_dialog_result.failed = file_list == nullptr;
    g_save_dialog_result.error = file_list == nullptr ? SDL_GetError() : "";
    g_save_dialog_result.selected_path =
        file_list != nullptr && file_list[0] != nullptr ? file_list[0] : "";
}

// Clears one asynchronous dialog result before the native dialog is shown.
void reset_dialog_state(DialogState& state)
{
    std::lock_guard lock(state.mutex);
    state.ready = false;
    state.failed = false;
    state.selected_path.clear();
    state.error.clear();
}

// Opens the Windows checkpoint picker; its static filter remains valid until callback.
void show_open_dialog(SDL_Window* window, const std::string& initial_location)
{
    static constexpr SDL_DialogFileFilter filters[] {
        {"EvoBrainBot checkpoints", "evo"},
        {"All files", "*"},
    };
    reset_dialog_state(g_open_dialog_result);
    SDL_ShowOpenFileDialog(
        open_dialog_callback,
        nullptr,
        window,
        filters,
        static_cast<int>(std::size(filters)),
        initial_location.empty() ? nullptr : initial_location.c_str(),
        false);
}

// Opens the Windows Save As picker with a complete suggested checkpoint path.
void show_save_dialog(SDL_Window* window, const std::string& suggested_path)
{
    static constexpr SDL_DialogFileFilter filters[] {
        {"EvoBrainBot checkpoints", "evo"},
        {"All files", "*"},
    };
    reset_dialog_state(g_save_dialog_result);
    SDL_ShowSaveFileDialog(
        save_dialog_callback,
        nullptr,
        window,
        filters,
        static_cast<int>(std::size(filters)),
        suggested_path.empty() ? nullptr : suggested_path.c_str());
}

// Returns one completed native dialog result, leaving cancellation as a no-op.
std::optional<std::pair<std::string, std::string>> take_dialog_result(
    DialogState& state,
    bool& dialog_open)
{
    std::lock_guard lock(state.mutex);
    if (!state.ready) {
        return std::nullopt;
    }
    dialog_open = false;
    state.ready = false;
    if (state.failed) {
        return std::pair {std::string {}, state.error};
    }
    return std::pair {state.selected_path, std::string {}};
}

// Converts a filesystem path to the UTF-8 representation expected by SDL and ImGui.
std::string path_to_utf8(const std::filesystem::path& path)
{
    const std::u8string encoded = path.u8string();
    return std::string(
        reinterpret_cast<const char*>(encoded.data()), encoded.size());
}

// Adds the checkpoint extension only when the selected filename has none.
std::filesystem::path with_checkpoint_extension(std::filesystem::path path)
{
    if (!path.has_extension()) {
        path += L".evo";
    }
    return path;
}

// Draws an in-window error with selectable technical details.
void draw_error_popup(std::optional<UiError>& error)
{
    if (!error) {
        return;
    }
    if (error->open_popup) {
        ImGui::OpenPopup("Viewer error");
        error->open_popup = false;
    }
    if (ImGui::BeginPopupModal("Viewer error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%s", error->summary.c_str());
        if (!error->detail.empty() && ImGui::CollapsingHeader("Technical details")) {
            ImGui::InputTextMultiline(
                "##technical-details",
                error->detail.data(),
                error->detail.size() + 1,
                ImVec2(560.0F, 120.0F),
                ImGuiInputTextFlags_ReadOnly);
        }
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
            error.reset();
        }
        ImGui::EndPopup();
    }
}

// Draws the world canvas and returns camera geometry plus any left-click selection.
WorldCanvasResult draw_world_canvas(
    evobrain::viewer::Camera& camera,
    const evobrain::viewer::RenderSnapshot* snapshot,
    const bool fast_forwarding,
    const ImVec2 size)
{
    WorldCanvasResult result;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0F, 0.0F, 0.0F, 0.0F));
    ImGui::BeginChild("World", size, ImGuiChildFlags_Borders);
    const ImVec2 canvas_position = ImGui::GetCursorScreenPos();
    const ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    ImGui::InvisibleButton(
        "World canvas",
        canvas_size,
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();

    result.viewport = evobrain::viewer::CameraViewport {
        .x = canvas_position.x,
        .y = canvas_position.y,
        .width = std::max(canvas_size.x, 1.0F),
        .height = std::max(canvas_size.y, 1.0F),
    };
    if (snapshot != nullptr) {
        camera.set_world_dimensions(
            snapshot->world_width, snapshot->world_height, result.viewport);
    }
    camera.viewport_changed(result.viewport);
    const ImGuiIO& io = ImGui::GetIO();
    if (hovered && io.MouseWheel != 0.0F) {
        camera.zoom_at(
            io.MousePos.x,
            io.MousePos.y,
            std::pow(1.2, io.MouseWheel),
            result.viewport);
    }
    if (hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
        camera.pan_pixels(io.MouseDelta.x, io.MouseDelta.y, result.viewport);
    }
    if (!fast_forwarding && snapshot != nullptr && snapshot->contains_world
        && ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        result.selection_requested = true;
        result.selected_agent_id = evobrain::viewer::select_agent_at_screen(
            *snapshot,
            camera,
            result.viewport,
            io.MousePos.x,
            io.MousePos.y);
    }
    if (fast_forwarding) {
        ImGui::GetWindowDrawList()->AddText(
            ImVec2(canvas_position.x + 12.0F, canvas_position.y + 12.0F),
            ImGui::GetColorU32(ImGuiCol_Text),
            "Fast-forwarding: world rendering is paused.");
    } else if (snapshot == nullptr) {
        ImGui::GetWindowDrawList()->AddText(
            ImVec2(canvas_position.x + 12.0F, canvas_position.y + 12.0F),
            ImGui::GetColorU32(ImGuiCol_TextDisabled),
            "Open an .evo checkpoint to begin.");
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
    return result;
}

// Returns squared screen distance from a point to one connection segment.
float distance_to_segment_squared(
    const ImVec2 point,
    const ImVec2 start,
    const ImVec2 end) noexcept
{
    const float segment_x = end.x - start.x;
    const float segment_y = end.y - start.y;
    const float length_squared = segment_x * segment_x + segment_y * segment_y;
    const float projection = length_squared > 0.0F
        ? std::clamp(((point.x - start.x) * segment_x
                         + (point.y - start.y) * segment_y)
                / length_squared,
            0.0F,
            1.0F)
        : 0.0F;
    const float closest_x = start.x + segment_x * projection;
    const float closest_y = start.y + segment_y * projection;
    const float delta_x = point.x - closest_x;
    const float delta_y = point.y - closest_y;
    return delta_x * delta_x + delta_y * delta_y;
}

// Draws an interactive layered graph without assuming a particular topology.
void draw_brain_canvas(
    const std::span<const BrainLayerView> layers,
    const std::span<const BrainNodeView> nodes,
    const std::span<const BrainConnectionView> connections,
    BrainViewState& state)
{
    constexpr float canvas_height = 270.0F;
    constexpr float horizontal_spacing = 210.0F;
    constexpr float vertical_spacing = 72.0F;
    if (layers.empty() || nodes.empty()) {
        ImGui::TextDisabled("No brain topology available");
        return;
    }
    if (ImGui::Button("Reset brain view")) {
        state.reset_requested = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Wheel: zoom  Middle-drag: pan");

    ImGui::BeginChild(
        "Brain canvas", ImVec2(0.0F, canvas_height), ImGuiChildFlags_Borders,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    const ImVec2 canvas_position = ImGui::GetCursorScreenPos();
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const ImVec2 canvas_size {
        std::max(available.x, 1.0F), std::max(available.y, 1.0F)};
    ImGui::InvisibleButton(
        "Brain canvas interaction",
        canvas_size,
        ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    const ImVec2 canvas_center {
        canvas_position.x + canvas_size.x * 0.5F,
        canvas_position.y + canvas_size.y * 0.5F};

    std::vector<std::size_t> nodes_per_layer(layers.size(), 0);
    for (const BrainNodeView& node : nodes) {
        if (node.layer < nodes_per_layer.size()) {
            ++nodes_per_layer[node.layer];
        }
    }
    std::size_t maximum_nodes = 1;
    for (const std::size_t count : nodes_per_layer) {
        maximum_nodes = std::max(maximum_nodes, count);
    }
    const float graph_width = (layers.size() - 1) * horizontal_spacing;
    const float graph_height = (maximum_nodes - 1) * vertical_spacing;
    if (state.reset_requested) {
        const float fit_width = canvas_size.x / std::max(graph_width + 190.0F, 1.0F);
        const float fit_height = canvas_size.y / std::max(graph_height + 100.0F, 1.0F);
        state.zoom = std::clamp(std::min(fit_width, fit_height), 0.1F, 1.5F);
        state.pan = ImVec2(0.0F, 0.0F);
        state.reset_requested = false;
    }

    const ImGuiIO& io = ImGui::GetIO();
    if (hovered && io.MouseWheel != 0.0F) {
        const float old_zoom = state.zoom;
        const ImVec2 graph_anchor {
            (io.MousePos.x - canvas_center.x - state.pan.x) / old_zoom,
            (io.MousePos.y - canvas_center.y - state.pan.y) / old_zoom};
        state.zoom = std::clamp(
            state.zoom * std::pow(1.2F, io.MouseWheel), 0.1F, 5.0F);
        state.pan = ImVec2(
            io.MousePos.x - canvas_center.x - graph_anchor.x * state.zoom,
            io.MousePos.y - canvas_center.y - graph_anchor.y * state.zoom);
    }
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
        state.pan.x += io.MouseDelta.x;
        state.pan.y += io.MouseDelta.y;
    }

    std::vector<std::size_t> next_node_in_layer(layers.size(), 0);
    std::vector<ImVec2> graph_positions(nodes.size());
    std::vector<ImVec2> screen_positions(nodes.size());
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const BrainNodeView& node = nodes[index];
        if (node.layer >= layers.size()) {
            continue;
        }
        const std::size_t order = next_node_in_layer[node.layer]++;
        const float layer_height = (nodes_per_layer[node.layer] - 1) * vertical_spacing;
        graph_positions[index] = ImVec2(
            node.layer * horizontal_spacing - graph_width * 0.5F,
            order * vertical_spacing - layer_height * 0.5F);
        screen_positions[index] = ImVec2(
            canvas_center.x + state.pan.x + graph_positions[index].x * state.zoom,
            canvas_center.y + state.pan.y + graph_positions[index].y * state.zoom);
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->PushClipRect(
        canvas_position,
        ImVec2(canvas_position.x + canvas_size.x, canvas_position.y + canvas_size.y),
        true);
    const auto connection_is_valid = [&](const BrainConnectionView& connection) {
        return connection.source < nodes.size()
            && connection.target < nodes.size()
            && nodes[connection.source].layer < layers.size()
            && nodes[connection.target].layer < layers.size();
    };
    double maximum_magnitude = 0.0;
    for (const BrainConnectionView& connection : connections) {
        if (connection_is_valid(connection)) {
            maximum_magnitude = std::max(
                maximum_magnitude, std::abs(connection.weight));
        }
    }
    maximum_magnitude = std::max(maximum_magnitude, 1e-12);
    std::optional<std::size_t> hovered_connection;
    float hovered_connection_distance = std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < connections.size(); ++index) {
        const BrainConnectionView& connection = connections[index];
        if (!connection_is_valid(connection)) {
            continue;
        }
        const float strength = static_cast<float>(
            std::abs(connection.weight) / maximum_magnitude);
        const float thickness = 1.0F + strength * 4.0F;
        const ImU32 color = connection.weight > 1e-9
            ? IM_COL32(64, 156, 255, 220)
            : connection.weight < -1e-9
                ? IM_COL32(255, 112, 72, 220)
                : IM_COL32(140, 140, 140, 150);
        draw_list->AddLine(
            screen_positions[connection.source],
            screen_positions[connection.target],
            color,
            thickness);
        if (hovered) {
            const float distance = distance_to_segment_squared(
                io.MousePos,
                screen_positions[connection.source],
                screen_positions[connection.target]);
            const float hover_radius = std::max(5.0F, thickness + 2.0F);
            if (distance <= hover_radius * hover_radius
                && distance < hovered_connection_distance) {
                hovered_connection = index;
                hovered_connection_distance = distance;
            }
        }
    }

    const float node_radius = std::clamp(7.0F * state.zoom, 4.0F, 11.0F);
    std::optional<std::size_t> hovered_node;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const BrainNodeView& node = nodes[index];
        // Malformed topology entries are ignored rather than drawing at the
        // default origin or indexing a nonexistent layer during hover.
        if (node.layer >= layers.size()) {
            continue;
        }
        const ImU32 node_color = node.layer == 0
            ? IM_COL32(170, 190, 215, 255)
            : node.layer + 1 == layers.size()
                ? IM_COL32(245, 202, 92, 255)
                : IM_COL32(174, 132, 224, 255);
        draw_list->AddCircleFilled(screen_positions[index], node_radius, node_color);
        const float font_size = std::clamp(ImGui::GetFontSize() * state.zoom, 10.0F, 18.0F);
        const ImVec2 text_size = ImGui::CalcTextSize(
            node.name.data(), node.name.data() + node.name.size());
        const float text_x = node.layer == 0
            ? screen_positions[index].x - node_radius - 5.0F - text_size.x
            : screen_positions[index].x + node_radius + 5.0F;
        draw_list->AddText(
            ImGui::GetFont(),
            font_size,
            ImVec2(text_x,
                screen_positions[index].y - font_size * 0.5F),
            ImGui::GetColorU32(ImGuiCol_Text),
            node.name.data(),
            node.name.data() + node.name.size());
        if (hovered) {
            const float delta_x = io.MousePos.x - screen_positions[index].x;
            const float delta_y = io.MousePos.y - screen_positions[index].y;
            if (delta_x * delta_x + delta_y * delta_y <= node_radius * node_radius) {
                hovered_node = index;
            }
        }
    }

    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        const float graph_x = layer * horizontal_spacing - graph_width * 0.5F;
        const ImVec2 layer_position {
            canvas_center.x + state.pan.x + graph_x * state.zoom,
            canvas_center.y + state.pan.y
                - (graph_height * 0.5F + 45.0F) * state.zoom};
        const ImVec2 text_size = ImGui::CalcTextSize(
            layers[layer].name.data(),
            layers[layer].name.data() + layers[layer].name.size());
        draw_list->AddText(
            ImVec2(layer_position.x - text_size.x * 0.5F, layer_position.y),
            ImGui::GetColorU32(ImGuiCol_TextDisabled),
            layers[layer].name.data(),
            layers[layer].name.data() + layers[layer].name.size());
    }
    draw_list->PopClipRect();

    if (hovered_node) {
        const BrainNodeView& node = nodes[*hovered_node];
        if (node.bias) {
            ImGui::SetTooltip("%.*s\nLayer: %.*s\nBias: %.17g",
                static_cast<int>(node.name.size()), node.name.data(),
                static_cast<int>(layers[node.layer].name.size()),
                layers[node.layer].name.data(), *node.bias);
        } else {
            ImGui::SetTooltip("%.*s\nLayer: %.*s",
                static_cast<int>(node.name.size()), node.name.data(),
                static_cast<int>(layers[node.layer].name.size()),
                layers[node.layer].name.data());
        }
    } else if (hovered_connection) {
        const BrainConnectionView& connection = connections[*hovered_connection];
        ImGui::SetTooltip("%.*s -> %.*s\nWeight: %.17g",
            static_cast<int>(nodes[connection.source].name.size()),
            nodes[connection.source].name.data(),
            static_cast<int>(nodes[connection.target].name.size()),
            nodes[connection.target].name.data(),
            connection.weight);
    }
    ImGui::EndChild();
}

// Adapts the fixed 26-to-8-to-3 brain to the topology-independent canvas.
void draw_brain_map(
    const evobrain::viewer::SelectedAgentDetails& agent,
    BrainViewState& state)
{
    constexpr std::array<BrainLayerView, 3> layers {{
        {.name = "Input layer"},
        {.name = "Hidden layer"},
        {.name = "Output layer"},
    }};
    constexpr std::array<std::string_view, evobrain::brain_input_count> input_names {{
        "Left -90 red", "Left -90 green", "Left -90 blue", "Left -90 proximity",
        "Left -45 red", "Left -45 green", "Left -45 blue", "Left -45 proximity",
        "Left 0 red", "Left 0 green", "Left 0 blue", "Left 0 proximity",
        "Right 0 red", "Right 0 green", "Right 0 blue", "Right 0 proximity",
        "Right +45 red", "Right +45 green", "Right +45 blue", "Right +45 proximity",
        "Right +90 red", "Right +90 green", "Right +90 blue", "Right +90 proximity",
        "Energy", "Prior bite damage",
    }};
    constexpr std::array<std::string_view, evobrain::brain_hidden_count> hidden_names {{
        "Hidden 1", "Hidden 2", "Hidden 3", "Hidden 4",
        "Hidden 5", "Hidden 6", "Hidden 7", "Hidden 8",
    }};
    constexpr std::array<std::string_view, evobrain::brain_output_count> output_names {{
        "Turn", "Move", "Eat",
    }};
    std::vector<BrainNodeView> nodes;
    nodes.reserve(evobrain::brain_input_count + evobrain::brain_hidden_count
        + evobrain::brain_output_count);
    for (const std::string_view name : input_names) nodes.push_back({.layer = 0, .name = name});
    for (std::size_t hidden = 0; hidden < evobrain::brain_hidden_count; ++hidden) {
        nodes.push_back({.layer = 1, .name = hidden_names[hidden],
            .bias = agent.brain[evobrain::hidden_bias_offset + hidden]});
    }
    for (std::size_t output = 0; output < evobrain::brain_output_count; ++output) {
        nodes.push_back({.layer = 2, .name = output_names[output],
            .bias = agent.brain[evobrain::output_bias_offset + output]});
    }
    std::vector<BrainConnectionView> connections;
    connections.reserve(evobrain::input_hidden_weight_count
        + evobrain::brain_hidden_count * evobrain::brain_output_count);
    for (std::size_t hidden = 0; hidden < evobrain::brain_hidden_count; ++hidden) {
        for (std::size_t input = 0; input < evobrain::brain_input_count; ++input) {
            connections.push_back({.source = input,
                .target = evobrain::brain_input_count + hidden,
                .weight = agent.brain[hidden * evobrain::brain_input_count + input]});
        }
    }
    for (std::size_t output = 0; output < evobrain::brain_output_count; ++output) {
        for (std::size_t hidden = 0; hidden < evobrain::brain_hidden_count; ++hidden) {
            connections.push_back({.source = evobrain::brain_input_count + hidden,
                .target = evobrain::brain_input_count + evobrain::brain_hidden_count + output,
                .weight = agent.brain[evobrain::hidden_output_weight_offset
                    + output * evobrain::brain_hidden_count + hidden]});
        }
    }
    if (state.agent_id != agent.id) {
        state.agent_id = agent.id;
        state.reset_requested = true;
    }
    draw_brain_canvas(layers, nodes, connections, state);

    ImGui::TextColored(ImVec4(0.25F, 0.61F, 1.0F, 1.0F), "Positive weight");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0F, 0.44F, 0.28F, 1.0F), "Negative weight");
    if (ImGui::CollapsingHeader(
            "Exact brain parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BeginChild(
            "Exact brain parameters table", ImVec2(0.0F, 155.0F),
            ImGuiChildFlags_Borders);
        if (ImGui::BeginTable("Brain weights", 3,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg
                    | ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableSetupColumn("From");
            ImGui::TableSetupColumn("To");
            ImGui::TableSetupColumn("Value");
            ImGui::TableHeadersRow();
            for (const BrainConnectionView& connection : connections) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(
                    nodes[connection.source].name.data(),
                    nodes[connection.source].name.data()
                        + nodes[connection.source].name.size());
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(
                    nodes[connection.target].name.data(),
                    nodes[connection.target].name.data()
                        + nodes[connection.target].name.size());
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%.6g", connection.weight);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%.17g", connection.weight);
                }
            }
            for (const BrainNodeView& node : nodes) {
                if (!node.bias) {
                    continue;
                }
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("Bias");
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(node.name.data(), node.name.data() + node.name.size());
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%.6g", *node.bias);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%.17g", *node.bias);
                }
            }
            ImGui::EndTable();
        }
        ImGui::EndChild();
    }
}

// Draws a draggable separator and updates the width of the pane on its right.
void draw_vertical_splitter(
    const float height,
    const float minimum_right_width,
    const float maximum_right_width,
    float& right_width)
{
    constexpr float splitter_width = 8.0F;
    ImGui::InvisibleButton("Information splitter", ImVec2(splitter_width, height));
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    if (hovered || active) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
    if (active) {
        // Moving the separator left grows the pane positioned to its right.
        right_width = std::clamp(
            right_width - ImGui::GetIO().MouseDelta.x,
            minimum_right_width,
            maximum_right_width);
    }

    const ImU32 color = ImGui::GetColorU32(
        active ? ImGuiCol_SeparatorActive
               : (hovered ? ImGuiCol_SeparatorHovered : ImGuiCol_Separator));
    const ImVec2 minimum = ImGui::GetItemRectMin();
    const ImVec2 maximum = ImGui::GetItemRectMax();
    const float center_x = (minimum.x + maximum.x) * 0.5F;
    ImGui::GetWindowDrawList()->AddLine(
        ImVec2(center_x, minimum.y), ImVec2(center_x, maximum.y), color, 1.0F);
}

// Draws the fixed one-window viewer UI and returns main-loop actions and geometry.
ViewerUiResult draw_viewer_shell(
    evobrain::viewer::SimulationWorker& worker,
    evobrain::viewer::Camera& camera,
    const std::string& filename,
    std::optional<UiError>& error,
    int& target_tps_edit,
    std::string& target_tps_validation,
    bool& show_agent_information,
    bool& show_debug,
    BrainViewState& brain_view,
    float& information_width,
    const bool confirm_replace,
    const bool confirm_overwrite,
    const bool confirm_exit,
    const std::string& overwrite_filename)
{
    ViewerUiResult result;
    const evobrain::viewer::WorkerStatus status = worker.status();
    const bool paused = status.playback == evobrain::viewer::PlaybackState::paused;
    const bool running = status.playback == evobrain::viewer::PlaybackState::running;
    const bool fast_forwarding =
        status.playback == evobrain::viewer::PlaybackState::fast_forward;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);

    constexpr ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_MenuBar;

    // The GPU world is rendered before ImGui. A transparent shell background
    // keeps the canvas visible while child panels and menu elements draw over it.
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0F, 0.0F, 0.0F, 0.0F));
    ImGui::Begin("EvoBrainBot Viewer", nullptr, window_flags);
    ImGui::PopStyleColor();
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open checkpoint...", "Ctrl+O")) {
                result.request_open = true;
            }
            ImGui::BeginDisabled(!status.has_simulation || !paused);
            if (ImGui::MenuItem("Save checkpoint", "Ctrl+S")) {
                result.request_save = true;
            }
            if (ImGui::MenuItem("Save checkpoint as...")) {
                result.request_save_as = true;
            }
            ImGui::EndDisabled();
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                result.request_exit = true;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    const auto snapshot = worker.latest_render_snapshot();
    if (ImGui::Button("Open checkpoint...")) {
        result.request_open = true;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!status.has_simulation);
    if (ImGui::Button(running || fast_forwarding ? "Pause" : "Run")) {
        result.toggle_run = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!status.has_simulation || !paused);
    if (ImGui::Button("Step")) {
        result.step = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!status.has_simulation);
    if (ImGui::Button(fast_forwarding ? "Pause fast-forward" : "Fast-forward")) {
        result.toggle_fast_forward = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0F);
    if (ImGui::InputInt(
            "Target TPS",
            &target_tps_edit,
            0,
            0)) {
        result.target_ticks_per_second = target_tps_edit;
    }
    ImGui::SameLine();
    const bool reset_requested = ImGui::Button("Reset view");
    if (!target_tps_validation.empty()) {
        ImGui::TextColored(
            ImVec4(1.0F, 0.38F, 0.32F, 1.0F), "%s", target_tps_validation.c_str());
    }

    constexpr float splitter_width = 8.0F;
    constexpr float minimum_world_width = 240.0F;
    constexpr float preferred_minimum_information_width = 300.0F;
    const ImVec2 content_size = ImGui::GetContentRegionAvail();
    const float maximum_information_width =
        std::max(content_size.x - splitter_width - minimum_world_width, 1.0F);
    const float minimum_information_width =
        std::min(preferred_minimum_information_width, maximum_information_width);
    information_width = std::clamp(
        information_width, minimum_information_width, maximum_information_width);
    const float layout_information_width = information_width;
    const ImVec2 world_size(
        std::max(content_size.x - layout_information_width - splitter_width, 1.0F),
        content_size.y);
    const WorldCanvasResult world = draw_world_canvas(
        camera, snapshot.get(), fast_forwarding, world_size);
    result.world_viewport = world.viewport;
    result.selection_requested = world.selection_requested;
    result.selected_agent_id = world.selected_agent_id;
    ImGui::SameLine(0.0F, 0.0F);
    draw_vertical_splitter(
        content_size.y,
        minimum_information_width,
        maximum_information_width,
        information_width);
    ImGui::SameLine(0.0F, 0.0F);
    ImGui::BeginChild(
        "Information", ImVec2(layout_information_width, 0.0F), ImGuiChildFlags_Borders);
    ImGui::Checkbox("Show agent information", &show_agent_information);
    ImGui::SameLine();
    ImGui::TextDisabled("(I)");
    ImGui::Checkbox("Show simulation debug", &show_debug);
    ImGui::SameLine();
    ImGui::TextDisabled("(D)");
    ImGui::Separator();
    ImGui::TextUnformatted("Selected agent");
    ImGui::Separator();
    if (snapshot == nullptr) {
        ImGui::TextDisabled("No checkpoint loaded");
    } else if (fast_forwarding && status.selected_agent_id) {
        ImGui::TextDisabled("Unavailable during Fast-forward");
    } else if (!snapshot->selected_agent) {
        ImGui::TextDisabled("No agent selected");
    } else {
        const auto& agent = *snapshot->selected_agent;
        ImGui::Text("ID: %llu", static_cast<unsigned long long>(agent.id));
        ImGui::Text("Energy: %.4f / %.4f", agent.energy,
            snapshot->reproduction_threshold);
        ImGui::Text("Age: %llu ticks", static_cast<unsigned long long>(agent.age));
        ImGui::Text("Generation: %llu",
            static_cast<unsigned long long>(agent.generation));
        ImGui::Text("Position: (%.4f, %.4f)", agent.position.x, agent.position.y);
        ImGui::Text("Direction: %.4f rad", agent.direction);
        ImGui::Text("Diet: %s", agent.diet == evobrain::Diet::herbivore
            ? "Herbivore" : "Carnivore");
        ImGui::ColorButton("Body color", ImVec4(static_cast<float>(agent.color.red),
            static_cast<float>(agent.color.green), static_cast<float>(agent.color.blue), 1.0F));
        ImGui::SameLine();
        ImGui::Text("RGB: %.6f, %.6f, %.6f", agent.color.red,
            agent.color.green, agent.color.blue);
        ImGui::Text("Mutation rate: %.8f", agent.mutation_rate);
        ImGui::Text("Mutation strength: %.8f", agent.mutation_strength);
        ImGui::Text("Prior bite damage: %.6f", agent.prior_bite_damage);
        ImGui::Separator();
        ImGui::TextUnformatted("Brain structure and weights");
        draw_brain_map(agent, brain_view);
    }
    ImGui::Separator();
    ImGui::TextUnformatted("Statistics");
    ImGui::Separator();
    if (snapshot == nullptr) {
        ImGui::TextDisabled("No checkpoint loaded");
    } else {
        ImGui::TextWrapped("File: %s", filename.c_str());
        ImGui::Text("Seed: %llu", static_cast<unsigned long long>(snapshot->stats.seed));
        ImGui::Text("Tick: %llu", static_cast<unsigned long long>(snapshot->stats.completed_ticks));
        ImGui::Text("Population: %llu", static_cast<unsigned long long>(snapshot->stats.population));
        ImGui::Text("Herbivores: %llu", static_cast<unsigned long long>(snapshot->stats.herbivores));
        ImGui::Text("Carnivores: %llu", static_cast<unsigned long long>(snapshot->stats.carnivores));
        ImGui::Text("Food: %llu", static_cast<unsigned long long>(snapshot->stats.food));
        ImGui::Text("Births: %llu", static_cast<unsigned long long>(snapshot->stats.births));
        ImGui::Text("Introduced agents: %llu",
            static_cast<unsigned long long>(snapshot->stats.introduced_agents));
        ImGui::Text("Deaths: %llu", static_cast<unsigned long long>(snapshot->stats.deaths));
        ImGui::Text("Agents eaten: %llu", static_cast<unsigned long long>(snapshot->stats.agents_eaten));
        ImGui::Separator();
        ImGui::Text("Requested TPS: %d", status.target_ticks_per_second);
        ImGui::Text("Actual TPS: %.1f", status.actual_ticks_per_second);
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("Zoom: %.2fx", camera.zoom());
        if (show_debug) {
            const auto& diagnostics = snapshot->diagnostics;
            ImGui::Separator();
            ImGui::TextUnformatted("Simulation debug");
            ImGui::Text("Spatial grid: %llu x %llu",
                static_cast<unsigned long long>(diagnostics.spatial_columns),
                static_cast<unsigned long long>(diagnostics.spatial_rows));
            ImGui::Text("Execution threads: %llu",
                static_cast<unsigned long long>(diagnostics.execution_threads));
            ImGui::Text("Tick: %.3f ms", diagnostics.total_milliseconds);
            ImGui::Text("Spatial index: %.3f ms", diagnostics.spatial_index_milliseconds);
            ImGui::Text("Sensing + brains: %.3f ms",
                diagnostics.sensing_brain_milliseconds);
            ImGui::Text("Movement: %.3f ms", diagnostics.movement_milliseconds);
            ImGui::Text("Bites: %.3f ms", diagnostics.bite_milliseconds);
            ImGui::Text("Lifecycle: %.3f ms", diagnostics.lifecycle_milliseconds);
            ImGui::Text("Vision candidates: %llu / %llu",
                static_cast<unsigned long long>(diagnostics.vision_candidate_tests),
                static_cast<unsigned long long>(diagnostics.vision_brute_force_tests));
            ImGui::Text("Bite candidates: %llu / %llu",
                static_cast<unsigned long long>(diagnostics.bite_candidate_tests),
                static_cast<unsigned long long>(diagnostics.bite_brute_force_tests));
        }
    }
    ImGui::EndChild();
    if (reset_requested) {
        camera.reset(result.world_viewport);
    }

    if (confirm_replace) {
        ImGui::OpenPopup("Replace advanced simulation?");
    }
    if (ImGui::BeginPopupModal(
            "Replace advanced simulation?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped(
            "The current simulation has unsaved changes. Open the selected checkpoint and discard them?");
        if (ImGui::Button("Open and discard")) {
            result.confirm_replace = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            result.cancel_replace = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (confirm_overwrite) {
        ImGui::OpenPopup("Replace checkpoint file?");
    }
    if (ImGui::BeginPopupModal(
            "Replace checkpoint file?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped(
            "A file named '%s' already exists. Replace it?", overwrite_filename.c_str());
        if (ImGui::Button("Replace")) {
            result.confirm_overwrite = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            result.cancel_overwrite = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (confirm_exit) {
        ImGui::OpenPopup("Unsaved simulation");
    }
    if (ImGui::BeginPopupModal(
            "Unsaved simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("Save the advanced simulation before closing the viewer?");
        if (ImGui::Button("Save and exit")) {
            result.exit_save = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Discard and exit")) {
            result.exit_discard = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            result.exit_cancel = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    draw_error_popup(error);
    ImGui::End();
    return result;
}

// Converts the logical ImGui world rectangle to a clamped swapchain-pixel scissor.
evobrain::viewer::PixelViewport pixel_viewport(
    const evobrain::viewer::CameraViewport& logical,
    const ImDrawData& draw_data,
    const int target_width,
    const int target_height)
{
    const float scale_x = draw_data.FramebufferScale.x;
    const float scale_y = draw_data.FramebufferScale.y;
    const int left = std::clamp(
        static_cast<int>(std::floor((logical.x - draw_data.DisplayPos.x) * scale_x)),
        0,
        target_width);
    const int top = std::clamp(
        static_cast<int>(std::floor((logical.y - draw_data.DisplayPos.y) * scale_y)),
        0,
        target_height);
    const int right = std::clamp(
        static_cast<int>(std::ceil(
            (logical.x + logical.width - draw_data.DisplayPos.x) * scale_x)),
        left,
        target_width);
    const int bottom = std::clamp(
        static_cast<int>(std::ceil(
            (logical.y + logical.height - draw_data.DisplayPos.y) * scale_y)),
        top,
        target_height);
    return evobrain::viewer::PixelViewport {
        .x = left,
        .y = top,
        .width = right - left,
        .height = bottom - top,
        .target_width = target_width,
        .target_height = target_height,
    };
}

// Runs the SDL event, Dear ImGui, and SDL_GPU render loop on the main thread.
int run_viewer()
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        return show_startup_error("SDL could not start.", SDL_GetError());
    }

    SDL_Window* window = SDL_CreateWindow(
        "EvoBrainBot Viewer",
        kDefaultWidth,
        kDefaultHeight,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN);
    if (window == nullptr) {
        const int result = show_startup_error(
            "The viewer window could not be created.", SDL_GetError());
        SDL_Quit();
        return result;
    }
    SDL_SetWindowMinimumSize(window, kMinimumWidth, kMinimumHeight);
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);

#if defined(NDEBUG)
    constexpr bool enable_gpu_debugging = false;
#else
    constexpr bool enable_gpu_debugging = true;
#endif
    SDL_GPUDevice* gpu_device = SDL_CreateGPUDevice(
        SDL_GPU_SHADERFORMAT_DXIL, enable_gpu_debugging, "direct3d12");
    if (gpu_device == nullptr) {
        const int result = show_startup_error(
            "A Direct3D 12 GPU device could not be created.", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return result;
    }
    if (!SDL_ClaimWindowForGPUDevice(gpu_device, window)) {
        const int result = show_startup_error(
            "The viewer window could not be connected to the GPU.", SDL_GetError());
        SDL_DestroyGPUDevice(gpu_device);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return result;
    }
    SDL_SetGPUSwapchainParameters(
        gpu_device, window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_VSYNC);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplSDL3_InitForSDLGPU(window);
    ImGui_ImplSDLGPU3_InitInfo init_info {};
    init_info.Device = gpu_device;
    init_info.ColorTargetFormat = SDL_GetGPUSwapchainTextureFormat(gpu_device, window);
    init_info.MSAASamples = SDL_GPU_SAMPLECOUNT_1;
    init_info.SwapchainComposition = SDL_GPU_SWAPCHAINCOMPOSITION_SDR;
    init_info.PresentMode = SDL_GPU_PRESENTMODE_VSYNC;
    ImGui_ImplSDLGPU3_Init(&init_info);

    evobrain::viewer::WorldRenderer world_renderer;
    std::string renderer_error;
    const std::filesystem::path shader_directory =
        std::filesystem::u8path(SDL_GetBasePath()) / L"shaders";
    if (!world_renderer.initialize(
            gpu_device, init_info.ColorTargetFormat, shader_directory, renderer_error)) {
        ImGui_ImplSDLGPU3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        SDL_ReleaseWindowFromGPUDevice(gpu_device, window);
        SDL_DestroyGPUDevice(gpu_device);
        SDL_DestroyWindow(window);
        const int result = show_startup_error(
            "The world renderer could not start.", renderer_error);
        SDL_Quit();
        return result;
    }

    evobrain::viewer::SimulationWorker worker;
    evobrain::viewer::Camera camera;
    std::optional<UiError> ui_error;
    std::string loaded_filename;
    std::string last_directory;
    std::optional<std::filesystem::path> save_path;
    std::optional<std::filesystem::path> pending_open_path;
    std::optional<std::filesystem::path> pending_overwrite_path;
    bool open_dialog_open = false;
    bool save_dialog_open = false;
    bool confirm_replace = false;
    bool confirm_overwrite = false;
    bool confirm_exit = false;
    bool exit_after_save = false;
    bool reset_camera_after_load = false;
    bool running = true;
    bool renderer_available = true;
    bool first_frame = true;
    bool show_agent_information = false;
    bool show_debug = false;
    BrainViewState brain_view;
    float information_width = 390.0F;
    int target_tps_edit = 60;
    std::string target_tps_validation;
    std::string previous_window_title;
    std::optional<UiError> fatal_runtime_error;
    SDL_ShowWindow(window);

    // Stops simulation work and records a GPU failure that cannot be presented
    // through ImGui because the current frame can no longer be submitted.
    const auto stop_after_gpu_failure = [&](const char* summary, std::string detail) {
        if (detail.empty()) {
            detail = "SDL did not provide additional error information.";
        }
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "%s %s", summary, detail.c_str());
        static_cast<void>(worker.pause());
        fatal_runtime_error = UiError {
            .summary = summary,
            .detail = std::move(detail),
        };
        running = false;
    };

    // Applies a validated replacement and updates session-only path state.
    const auto load_selected_checkpoint = [&](const std::filesystem::path& path) {
        const auto load_result = worker.load_checkpoint_file(path);
        if (!load_result) {
            ui_error = UiError {
                .summary = load_result.summary,
                .detail = load_result.detail,
            };
            return;
        }
        loaded_filename = path_to_utf8(path.filename());
        last_directory = path_to_utf8(path.parent_path());
        save_path.reset();
        reset_camera_after_load = true;
    };

    // Saves one paused state and establishes the path used by later Ctrl+S.
    const auto save_selected_checkpoint = [&] (
                                              const std::filesystem::path& path,
                                              const bool overwrite) {
        const auto save_result = worker.save_checkpoint_file(path, overwrite);
        if (!save_result) {
            ui_error = UiError {
                .summary = save_result.summary,
                .detail = save_result.detail,
            };
            exit_after_save = false;
            first_frame = true;
            return;
        }
        save_path = path;
        loaded_filename = path_to_utf8(path.filename());
        last_directory = path_to_utf8(path.parent_path());
        if (exit_after_save) {
            running = false;
        }
    };

    // Starts Save As with the required first-save "-continued.evo" suggestion.
    const auto begin_save_as = [&] {
        if (save_dialog_open) {
            return;
        }
        std::filesystem::path filename = std::filesystem::u8path(loaded_filename);
        std::filesystem::path suggested = L"continued.evo";
        if (!filename.empty()) {
            suggested = std::filesystem::path(
                filename.stem().wstring() + L"-continued.evo");
        }
        if (!last_directory.empty()) {
            suggested = std::filesystem::u8path(last_directory) / suggested;
        }
        save_dialog_open = true;
        show_save_dialog(window, path_to_utf8(suggested));
    };

    while (running) {
        if (const auto title_error = update_window_title(
                window, loaded_filename, worker.status(), previous_window_title)) {
            static_cast<void>(worker.pause());
            SDL_LogError(
                SDL_LOG_CATEGORY_APPLICATION,
                "The viewer window title could not be updated. %s",
                title_error->c_str());
            ui_error = UiError {
                .summary = "The viewer window title could not be updated.",
                .detail = *title_error,
            };
            first_frame = true;
        }
        std::optional<evobrain::viewer::WorkerFailure> worker_failure =
            worker.take_failure();
        std::optional<SDL_Event> waited_event;
        if (!first_frame && !worker_failure) {
            const auto wait_status = worker.status();
            SDL_Event event {};
            if (wait_status.playback == evobrain::viewer::PlaybackState::fast_forward) {
                if (SDL_WaitEventTimeout(&event, 250)) {
                    waited_event = event;
                }
            } else if (wait_status.playback == evobrain::viewer::PlaybackState::paused) {
                if (open_dialog_open || save_dialog_open) {
                    if (SDL_WaitEventTimeout(&event, 100)) {
                        waited_event = event;
                    }
                } else if (SDL_WaitEvent(&event)) {
                    waited_event = event;
                }
            }
        }
        first_frame = false;

        bool close_requested = false;
        const auto process_event = [&](SDL_Event& event) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT
                || (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED
                    && event.window.windowID == SDL_GetWindowID(window))) {
                close_requested = true;
            }
        };
        if (waited_event) {
            process_event(*waited_event);
        }
        SDL_Event event {};
        while (SDL_PollEvent(&event)) {
            process_event(event);
        }

        if (const auto dialog_result =
                take_dialog_result(g_open_dialog_result, open_dialog_open)) {
            if (!dialog_result->second.empty()) {
                ui_error = UiError {
                    .summary = "The checkpoint picker could not be opened.",
                    .detail = dialog_result->second,
                };
            } else if (!dialog_result->first.empty()) {
                const std::filesystem::path path =
                    std::filesystem::u8path(dialog_result->first);
                if (worker.status().has_unsaved_changes) {
                    pending_open_path = path;
                    confirm_replace = true;
                } else {
                    load_selected_checkpoint(path);
                }
            }
        }

        if (const auto dialog_result =
                take_dialog_result(g_save_dialog_result, save_dialog_open)) {
            if (!dialog_result->second.empty()) {
                ui_error = UiError {
                    .summary = "The checkpoint picker could not be opened.",
                    .detail = dialog_result->second,
                };
                exit_after_save = false;
            } else if (!dialog_result->first.empty()) {
                const std::filesystem::path path = with_checkpoint_extension(
                    std::filesystem::u8path(dialog_result->first));
                std::error_code exists_error;
                const bool exists = std::filesystem::exists(path, exists_error);
                if (exists_error) {
                    ui_error = UiError {
                        .summary = "The checkpoint destination could not be inspected.",
                        .detail = exists_error.message(),
                    };
                    exit_after_save = false;
                } else if (exists) {
                    pending_overwrite_path = path;
                    confirm_overwrite = true;
                } else {
                    save_selected_checkpoint(path, false);
                }
            } else {
                exit_after_save = false;
            }
        }

        if (worker_failure) {
            ui_error = UiError {
                .summary = worker_failure->summary,
                .detail = worker_failure->detail,
            };
        }

        ImGui_ImplSDLGPU3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        if (worker.status().playback == evobrain::viewer::PlaybackState::running) {
            // The worker publishes no more than one full entity copy for each
            // frame the main thread is actually going to render.
            worker.request_render_snapshot();
        }

        const ImGuiIO& io = ImGui::GetIO();
        const bool shortcuts_enabled = !io.WantTextInput;
        const bool shortcut_open = shortcuts_enabled && io.KeyCtrl
            && ImGui::IsKeyPressed(ImGuiKey_O, false);
        const bool shortcut_save = shortcuts_enabled && io.KeyCtrl
            && ImGui::IsKeyPressed(ImGuiKey_S, false);
        const bool shortcut_run = shortcuts_enabled
            && ImGui::IsKeyPressed(ImGuiKey_Space, false);
        const bool shortcut_step = shortcuts_enabled
            && ImGui::IsKeyPressed(ImGuiKey_Period, false);
        const bool shortcut_fast = shortcuts_enabled
            && ImGui::IsKeyPressed(ImGuiKey_F, false);
        const bool shortcut_reset = shortcuts_enabled
            && ImGui::IsKeyPressed(ImGuiKey_Home, false);
        const bool shortcut_agent_information =
            evobrain::viewer::agent_information_shortcut_pressed(
                ImGui::IsKeyPressed(ImGuiKey_I, false), io.WantTextInput);
        const bool shortcut_debug = shortcuts_enabled
            && ImGui::IsKeyPressed(ImGuiKey_D, false);
        const evobrain::viewer::WorkerStatus status_before_ui = worker.status();
        ViewerUiResult ui = draw_viewer_shell(
            worker,
            camera,
            loaded_filename,
            ui_error,
            target_tps_edit,
            target_tps_validation,
            show_agent_information,
            show_debug,
            brain_view,
            information_width,
            confirm_replace,
            confirm_overwrite,
            confirm_exit,
            pending_overwrite_path
                ? path_to_utf8(pending_overwrite_path->filename())
                : std::string {});
        if (shortcut_reset || reset_camera_after_load) {
            camera.reset(ui.world_viewport);
            reset_camera_after_load = false;
        }
        if (shortcut_agent_information) {
            show_agent_information = !show_agent_information;
        }
        if (shortcut_debug) {
            show_debug = !show_debug;
        }
        if (ui.selection_requested) {
            worker.select_agent(ui.selected_agent_id);
        }

        if ((ui.request_open || shortcut_open) && !open_dialog_open) {
            static_cast<void>(worker.pause());
            open_dialog_open = true;
            show_open_dialog(window, last_directory);
        }

        const auto show_operation_error = [&](const evobrain::viewer::OperationResult& result) {
            if (!result) {
                ui_error = UiError {
                    .summary = result.summary,
                    .detail = result.detail,
                };
                first_frame = true;
            }
        };
        evobrain::viewer::WorkerStatus status = worker.status();
        if ((ui.toggle_run || shortcut_run) && status.has_simulation) {
            if (status.playback == evobrain::viewer::PlaybackState::paused) {
                show_operation_error(worker.run());
            } else {
                show_operation_error(worker.pause());
            }
            status = worker.status();
        }
        if ((ui.step || shortcut_step) && status.has_simulation
            && status.playback == evobrain::viewer::PlaybackState::paused) {
            show_operation_error(worker.step());
            status = worker.status();
        }
        if ((ui.toggle_fast_forward || shortcut_fast) && status.has_simulation) {
            if (status.playback == evobrain::viewer::PlaybackState::fast_forward) {
                show_operation_error(worker.pause());
            } else {
                show_operation_error(worker.fast_forward());
            }
            status = worker.status();
        }
        if (ui.target_ticks_per_second) {
            const auto rate_result =
                worker.set_target_ticks_per_second(*ui.target_ticks_per_second);
            if (rate_result) {
                target_tps_validation.clear();
            } else {
                target_tps_validation = rate_result.summary;
                target_tps_edit = worker.status().target_ticks_per_second;
                first_frame = true;
            }
        }

        const bool save_requested = ui.request_save || shortcut_save;
        if (save_requested && status.has_simulation
            && status.playback == evobrain::viewer::PlaybackState::paused) {
            if (save_path) {
                save_selected_checkpoint(*save_path, true);
            } else {
                begin_save_as();
            }
        }
        if (ui.request_save_as && status.has_simulation
            && status.playback == evobrain::viewer::PlaybackState::paused) {
            begin_save_as();
        }

        if (ui.confirm_replace && pending_open_path) {
            confirm_replace = false;
            const std::filesystem::path selected = *pending_open_path;
            pending_open_path.reset();
            load_selected_checkpoint(selected);
        } else if (ui.cancel_replace) {
            confirm_replace = false;
            pending_open_path.reset();
        }

        if (ui.confirm_overwrite && pending_overwrite_path) {
            confirm_overwrite = false;
            const std::filesystem::path selected = *pending_overwrite_path;
            pending_overwrite_path.reset();
            save_selected_checkpoint(selected, true);
        } else if (ui.cancel_overwrite) {
            confirm_overwrite = false;
            pending_overwrite_path.reset();
            exit_after_save = false;
        }

        if (close_requested || ui.request_exit) {
            static_cast<void>(worker.pause());
            if (worker.status().has_unsaved_changes) {
                confirm_exit = true;
                first_frame = true;
            } else {
                running = false;
            }
        }
        if (ui.exit_save) {
            confirm_exit = false;
            exit_after_save = true;
            if (save_path) {
                save_selected_checkpoint(*save_path, true);
            } else {
                begin_save_as();
            }
        } else if (ui.exit_discard) {
            confirm_exit = false;
            running = false;
        } else if (ui.exit_cancel) {
            confirm_exit = false;
        }
        if (!(worker.status() == status_before_ui)) {
            // Synchronous commands were issued after the UI was built; render
            // one more frame before paused-mode event waiting updates labels.
            first_frame = true;
        }
        ImGui::Render();

        SDL_GPUCommandBuffer* command_buffer = SDL_AcquireGPUCommandBuffer(gpu_device);
        if (command_buffer == nullptr) {
            stop_after_gpu_failure(
                "A GPU command buffer could not be acquired.", SDL_GetError());
            continue;
        }
        ImGui_ImplSDLGPU3_PrepareDrawData(ImGui::GetDrawData(), command_buffer);

        SDL_GPUTexture* swapchain_texture = nullptr;
        std::uint32_t target_width = 0;
        std::uint32_t target_height = 0;
        if (!SDL_WaitAndAcquireGPUSwapchainTexture(
                command_buffer,
                window,
                &swapchain_texture,
                &target_width,
                &target_height)) {
            const std::string detail = SDL_GetError();
            SDL_CancelGPUCommandBuffer(command_buffer);
            stop_after_gpu_failure(
                "The next viewer frame could not be acquired.", detail);
            continue;
        }
        if (swapchain_texture != nullptr) {
            const auto world_pixels = pixel_viewport(
                ui.world_viewport,
                *ImGui::GetDrawData(),
                static_cast<int>(target_width),
                static_cast<int>(target_height));
            const auto snapshot = worker.latest_render_snapshot();
            const auto render_status = worker.status();
            const bool render_world = render_status.playback
                    != evobrain::viewer::PlaybackState::fast_forward
                && (!snapshot || snapshot->contains_world);
            if (renderer_available && render_world
                && !world_renderer.prepare(
                    command_buffer,
                    world_pixels,
                    camera,
                    snapshot.get(),
                    evobrain::viewer::WorldRenderOptions {
                        .show_agent_information = show_agent_information,
                        .show_debug = show_debug,
                        .selected_agent_id = render_status.selected_agent_id,
                    },
                    renderer_error)) {
                renderer_available = false;
                static_cast<void>(worker.pause());
                SDL_LogError(
                    SDL_LOG_CATEGORY_GPU,
                    "The world could not be rendered. %s",
                    renderer_error.c_str());
                ui_error = UiError {
                    .summary = "The world could not be rendered.",
                    .detail = renderer_error,
                };
                first_frame = true;
            }

            SDL_GPUColorTargetInfo color_target {};
            color_target.texture = swapchain_texture;
            color_target.clear_color = SDL_FColor {0.035F, 0.040F, 0.050F, 1.0F};
            color_target.load_op = SDL_GPU_LOADOP_CLEAR;
            color_target.store_op = SDL_GPU_STOREOP_STORE;

            SDL_GPURenderPass* render_pass =
                SDL_BeginGPURenderPass(command_buffer, &color_target, 1, nullptr);
            if (render_pass == nullptr) {
                const std::string detail = SDL_GetError();
                SDL_CancelGPUCommandBuffer(command_buffer);
                stop_after_gpu_failure(
                    "The viewer render pass could not be started.", detail);
                continue;
            }
            if (renderer_available && render_world) {
                world_renderer.draw(render_pass, world_pixels);
            }
            ImGui_ImplSDLGPU3_RenderDrawData(
                ImGui::GetDrawData(), command_buffer, render_pass);
            SDL_EndGPURenderPass(render_pass);
        }
        if (!SDL_SubmitGPUCommandBuffer(command_buffer)) {
            stop_after_gpu_failure(
                "The viewer frame could not be submitted to the GPU.", SDL_GetError());
        }
    }

    int exit_code = EXIT_SUCCESS;
    if (fatal_runtime_error) {
        const std::string message = fatal_runtime_error->summary + "\n\n"
            + fatal_runtime_error->detail;
        SDL_ShowSimpleMessageBox(
            SDL_MESSAGEBOX_ERROR,
            "EvoBrainBot Viewer",
            message.c_str(),
            window);
        exit_code = EXIT_FAILURE;
    }

    world_renderer.shutdown();
    ImGui_ImplSDLGPU3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_ReleaseWindowFromGPUDevice(gpu_device, window);
    SDL_DestroyGPUDevice(gpu_device);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return exit_code;
}

} // namespace

// Starts the Windows viewer; SDL provides the platform entry-point adapter.
int main(int, char**)
{
    return run_viewer();
}
