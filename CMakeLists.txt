cmake_minimum_required(VERSION 3.24)

project(EvoBrainBot LANGUAGES CXX)

include(CTest)

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
