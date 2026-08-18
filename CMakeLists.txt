cmake_minimum_required(VERSION 3.24)

project(EvoBrainBot LANGUAGES CXX)

add_executable(EvoBrainBot
    "src/main.cpp"
)

set_target_properties(EvoBrainBot PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
