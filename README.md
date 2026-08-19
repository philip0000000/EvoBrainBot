# EvoBrainBot

EvoBrainBot is an artificial-life simulation where autonomous agents evolve behavior through natural selection.

## Status

Early development.

The project is currently focused on establishing a clean and flexible foundation for the simulation.

## Goals

* Simulate evolving autonomous agents
* Explore artificial life and emergent behavior
* Support evolving agent brains and traits
* Prioritize performance and scalability
* Keep the architecture flexible as the project develops

## Technology

* C++20
* CMake

## Build

EvoBrainBot requires CMake 3.24 or newer and a C++20 compiler.

On Linux with GCC:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

On Windows from a Visual Studio developer shell:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Test

After building, run the core and command-line tests through CTest.

On Linux:

```sh
ctest --test-dir build --output-on-failure
```

On Windows with a multi-configuration generator:

```powershell
ctest --test-dir build --build-config Release --output-on-failure
```

## Run headless

The headless runner requires an explicit seed and tick count:

```sh
./build/EvoBrainBot run --seed 1234 --ticks 1000
```

With a Visual Studio generator on Windows, the executable is normally in the
selected configuration directory:

```powershell
.\build\Release\EvoBrainBot.exe run --seed 1234 --ticks 1000
```

Display the available commands or help for `run` with:

```sh
./build/EvoBrainBot --help
./build/EvoBrainBot run --help
```

On Windows with a Visual Studio generator:

```powershell
.\build\Release\EvoBrainBot.exe --help
.\build\Release\EvoBrainBot.exe run --help
```

## License

Licensed under the Apache License 2.0.
