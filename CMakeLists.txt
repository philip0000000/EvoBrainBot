cmake_minimum_required(VERSION 3.24)

project(EvoBrainBot LANGUAGES CXX)

include(CTest)

if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(EVOBRAIN_VIEWER_DEFAULT ON)
else()
    set(EVOBRAIN_VIEWER_DEFAULT OFF)
endif()

option(
    EVOBRAIN_BUILD_VIEWER
    "Build the Windows x64 SDL_GPU simulation viewer"
    ${EVOBRAIN_VIEWER_DEFAULT}
)

add_library(evobrain_core STATIC
    "src/core/brain.cpp"
    "src/core/checkpoint.cpp"
    "src/core/random.cpp"
    "src/core/simulation.cpp"
)

target_include_directories(evobrain_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set_target_properties(evobrain_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(evobrain_core PRIVATE /EHsc)
endif()

add_executable(EvoBrainBot
    "src/runner/command_line.cpp"
    "src/runner/main.cpp"
)

target_link_libraries(EvoBrainBot PRIVATE evobrain_core)

set_target_properties(EvoBrainBot PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(EvoBrainBot PRIVATE /EHsc)
endif()

if(EVOBRAIN_BUILD_VIEWER)
    if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "EvoBrainBotViewer is supported only on 64-bit Windows")
    endif()

    enable_language(C)
    include(FetchContent)

    # The viewer owns its exact dependency versions so a fresh build does not
    # depend on machine-wide SDL or Dear ImGui installations.
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG 8e37db5e797b6167f3a00d697d816a684bd259c7
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(SDL3)

    FetchContent_Declare(
        DearImGui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG 01380c579715e62fb9a8d6ec0502c4ea83bfde6e
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(DearImGui)

    add_library(evobrain_viewer_ui STATIC
        "${dearimgui_SOURCE_DIR}/imgui.cpp"
        "${dearimgui_SOURCE_DIR}/imgui_draw.cpp"
        "${dearimgui_SOURCE_DIR}/imgui_tables.cpp"
        "${dearimgui_SOURCE_DIR}/imgui_widgets.cpp"
        "${dearimgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp"
        "${dearimgui_SOURCE_DIR}/backends/imgui_impl_sdlgpu3.cpp"
    )
    target_include_directories(evobrain_viewer_ui
        PUBLIC
            "${dearimgui_SOURCE_DIR}"
            "${dearimgui_SOURCE_DIR}/backends"
    )
    target_link_libraries(evobrain_viewer_ui PUBLIC SDL3::SDL3-static)
    set_target_properties(evobrain_viewer_ui PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    if(MSVC)
        target_compile_options(evobrain_viewer_ui PRIVATE /EHsc)
    endif()

    add_library(evobrain_viewer_support STATIC
        "src/viewer/camera.cpp"
        "src/viewer/simulation_worker.cpp"
    )
    target_link_libraries(evobrain_viewer_support PUBLIC evobrain_core)
    target_include_directories(evobrain_viewer_support
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    set_target_properties(evobrain_viewer_support PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    if(MSVC)
        target_compile_options(evobrain_viewer_support PRIVATE /EHsc)
    endif()

    find_program(EVOBRAIN_DXC_EXECUTABLE NAMES dxc.exe dxc REQUIRED)
    set(EVOBRAIN_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/compiled-shaders")
    set(EVOBRAIN_WORLD_VERTEX_SHADER
        "${EVOBRAIN_SHADER_OUTPUT_DIR}/world.vert.dxil")
    set(EVOBRAIN_WORLD_FRAGMENT_SHADER
        "${EVOBRAIN_SHADER_OUTPUT_DIR}/world.frag.dxil")
    add_custom_command(
        OUTPUT
            "${EVOBRAIN_WORLD_VERTEX_SHADER}"
            "${EVOBRAIN_WORLD_FRAGMENT_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${EVOBRAIN_SHADER_OUTPUT_DIR}"
        COMMAND "${EVOBRAIN_DXC_EXECUTABLE}"
            -T vs_6_0 -E main -Qstrip_debug -Qstrip_reflect
            -Fo "${EVOBRAIN_WORLD_VERTEX_SHADER}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/world.vert.hlsl"
        COMMAND "${EVOBRAIN_DXC_EXECUTABLE}"
            -T ps_6_0 -E main -Qstrip_debug -Qstrip_reflect
            -Fo "${EVOBRAIN_WORLD_FRAGMENT_SHADER}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/world.frag.hlsl"
        DEPENDS
            "shaders/world.vert.hlsl"
            "shaders/world.frag.hlsl"
        COMMENT "Compiling EvoBrainBotViewer DXIL shaders"
        VERBATIM
    )
    add_custom_target(evobrain_viewer_shaders
        DEPENDS
            "${EVOBRAIN_WORLD_VERTEX_SHADER}"
            "${EVOBRAIN_WORLD_FRAGMENT_SHADER}"
    )

    add_library(evobrain_viewer_render STATIC
        "src/viewer/world_renderer.cpp"
    )
    target_include_directories(evobrain_viewer_render
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/src/viewer"
    )
    target_link_libraries(
        evobrain_viewer_render
        PUBLIC
            evobrain_viewer_support
            SDL3::SDL3-static
    )
    set_target_properties(evobrain_viewer_render PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    if(MSVC)
        target_compile_options(evobrain_viewer_render PRIVATE /EHsc)
    endif()
    add_dependencies(evobrain_viewer_render evobrain_viewer_shaders)

    # Keeps each executable's runtime shader directory identical to the
    # portable install layout.
    function(evobrain_copy_viewer_shaders target)
        add_dependencies(${target} evobrain_viewer_shaders)
        add_custom_command(
            TARGET ${target}
            POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                "$<TARGET_FILE_DIR:${target}>/shaders"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${EVOBRAIN_WORLD_VERTEX_SHADER}"
                "${EVOBRAIN_WORLD_FRAGMENT_SHADER}"
                "$<TARGET_FILE_DIR:${target}>/shaders"
            VERBATIM
        )
    endfunction()

    add_executable(EvoBrainBotViewer WIN32
        "src/viewer/main.cpp"
    )
    target_link_libraries(
        EvoBrainBotViewer
        PRIVATE
            evobrain_core
            evobrain_viewer_render
            evobrain_viewer_ui
            SDL3::SDL3-static
    )
    set_target_properties(EvoBrainBotViewer PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    if(MSVC)
        target_compile_options(EvoBrainBotViewer PRIVATE /EHsc)
    endif()
    evobrain_copy_viewer_shaders(EvoBrainBotViewer)

    add_executable(evobrain_viewer_render_benchmark
        "tests/viewer_render_benchmark.cpp"
    )
    target_link_libraries(
        evobrain_viewer_render_benchmark
        PRIVATE
            evobrain_viewer_render
            SDL3::SDL3-static
    )
    set_target_properties(evobrain_viewer_render_benchmark PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    if(MSVC)
        target_compile_options(evobrain_viewer_render_benchmark PRIVATE /EHsc)
    endif()
    evobrain_copy_viewer_shaders(evobrain_viewer_render_benchmark)

    install(TARGETS EvoBrainBotViewer RUNTIME DESTINATION .)
    install(FILES
        "${EVOBRAIN_WORLD_VERTEX_SHADER}"
        "${EVOBRAIN_WORLD_FRAGMENT_SHADER}"
        DESTINATION shaders
    )
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md" DESTINATION .)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/viewer-performance.md"
        DESTINATION docs)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
        DESTINATION licenses RENAME EvoBrainBot-MIT.txt)
    install(FILES "${sdl3_SOURCE_DIR}/LICENSE.txt"
        DESTINATION licenses RENAME SDL3-zlib.txt)
    install(FILES "${dearimgui_SOURCE_DIR}/LICENSE.txt"
        DESTINATION licenses RENAME DearImGui-MIT.txt)

    if(BUILD_TESTING)
        add_executable(evobrain_viewer_tests
            "tests/viewer_worker_tests.cpp"
        )
        target_link_libraries(evobrain_viewer_tests PRIVATE evobrain_viewer_support)
        set_target_properties(evobrain_viewer_tests PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        if(MSVC)
            target_compile_options(evobrain_viewer_tests PRIVATE /EHsc)
        endif()
        add_test(NAME evobrain_viewer_tests COMMAND evobrain_viewer_tests)
    endif()
endif()

if(BUILD_TESTING)
    add_executable(evobrain_tests
        "tests/checkpoint_tests.cpp"
        "tests/simulation_tests.cpp"
    )

    target_link_libraries(evobrain_tests PRIVATE evobrain_core)

    set_target_properties(evobrain_tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    if(MSVC)
        target_compile_options(evobrain_tests PRIVATE /EHsc)
    endif()

    add_test(NAME evobrain_core_tests COMMAND evobrain_tests)
    add_test(
        NAME evobrain_runner_tests
        COMMAND "${CMAKE_COMMAND}"
            "-DEVOBRAINBOT_EXECUTABLE=$<TARGET_FILE:EvoBrainBot>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/runner_tests.cmake"
    )
endif()
