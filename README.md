# EvoBrainBot

EvoBrainBot is an artificial-life simulation where autonomous agents evolve behavior through natural selection.

## Status

Early development.

The project currently provides a small deterministic headless simulation in
which agents sense food, move, spend energy, eat, die, reproduce, and inherit
mutated brain parameters.

The current mechanics and configuration values are provisional. They establish
a complete testable evolutionary loop and are expected to be refined after
viewer and experiment tooling is available.

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

Start a new simulation with an explicit seed and tick count:

```sh
./build/EvoBrainBot run --seed 1234 --ticks 1000
```

With a Visual Studio generator on Windows, the executable is normally in the
selected configuration directory:

```powershell
.\build\Release\EvoBrainBot.exe run --seed 1234 --ticks 1000
```

The final summary reports the completed tick count, living population, food,
reproduction births, random agent introductions, and deaths.

### Save and resume

Every successful `run` and `resume` command saves the complete final simulation
state. If `--checkpoint-out` is omitted, the state is written to
`autosave.evo` in the current directory, replacing an existing file with that
name.

Choose a different checkpoint path with `--checkpoint-out`:

```sh
./build/EvoBrainBot run --seed 1234 --ticks 1000 \
    --checkpoint-out state.evo
```

Resume the checkpoint for an additional number of ticks. The continued state is
also saved to `autosave.evo` unless another output path is supplied:

```sh
./build/EvoBrainBot resume --checkpoint-in state.evo --ticks 500 \
    --checkpoint-out continued.evo
```

On Windows with a Visual Studio generator:

```powershell
.\build\Release\EvoBrainBot.exe run --seed 1234 --ticks 1000 `
    --checkpoint-out state.evo

.\build\Release\EvoBrainBot.exe resume --checkpoint-in state.evo --ticks 500 `
    --checkpoint-out continued.evo
```

`--ticks` always means the ticks executed by the current command. A resumed run
obtains its original seed, configuration, accumulated statistics, entities, and
random-generator state from the checkpoint.

Checkpoints use a versioned binary format. Unsupported, incomplete, and invalid
checkpoint files are rejected instead of being partially loaded.

Display the available commands or help for `run` with:

```sh
./build/EvoBrainBot --help
./build/EvoBrainBot run --help
./build/EvoBrainBot resume --help
```

On Windows with a Visual Studio generator:

```powershell
.\build\Release\EvoBrainBot.exe --help
.\build\Release\EvoBrainBot.exe run --help
.\build\Release\EvoBrainBot.exe resume --help
```

## License

Licensed under the Apache License 2.0.
