# EvoBrainBot

EvoBrainBot is an artificial-life simulation where autonomous agents evolve behavior through natural selection.

## Status

Early development.

The project currently provides a deterministic headless simulation and a basic
Windows viewer. Agents sense food, move, spend energy, eat, die, reproduce, and
inherit mutated brain parameters.

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
* SDL 3 with SDL_GPU (Direct3D 12) for the Windows viewer
* Dear ImGui for the viewer interface

## Build

EvoBrainBot requires CMake 3.24 or newer and a C++20 compiler.

On Linux with GCC:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

On Windows from a Visual Studio x64 developer shell, the repository presets
build the viewer and headless executable together:

```powershell
cmake --preset x64-release
cmake --build out/build/x64-release
```

The x86 presets continue to build only the existing headless executable. SDL 3
and Dear ImGui are fetched at pinned revisions during an x64 viewer configure;
no machine-wide copies are required.

## Test

After building, run the core and command-line tests through CTest.

On Linux:

```sh
ctest --test-dir build --output-on-failure
```

On Windows with a repository preset:

```powershell
ctest --test-dir out/build/x64-release --output-on-failure
```

## Run the Windows viewer

`EvoBrainBotViewer` targets Windows 11 x64 and a Direct3D 12-capable GPU. There
is no software-rendering fallback. Windows 10 and other operating systems are
not acceptance targets for this first viewer, while the headless simulation
retains its existing platform and x86 build options.

Run the viewer from its build directory:

```powershell
.\out\build\x64-release\EvoBrainBotViewer.exe
```

The viewer starts empty and opens existing `.evo` checkpoints; it does not
create a new simulation. Checkpoints open paused, and a failed open or save
leaves the current simulation intact.

Controls:

* `Ctrl+O`: open a checkpoint
* `Ctrl+S`: save the paused checkpoint, opening Save As on its first save
* `Space`: run or pause
* `.`: advance one tick while paused
* `F`: enter or leave Fast-forward
* `I`: show or hide agent information overlays
* Left-click an agent: select it for inspection
* Left-click empty world space: clear agent selection
* Mouse wheel over the brain canvas: zoom the brain graph
* Middle-mouse drag in the brain canvas: pan the brain graph
* Drag the vertical separator: resize the simulation and information panes
* Mouse wheel: zoom around the pointer
* Middle-mouse drag: pan
* `Home`: reset the camera

Fast-forward advances without drawing the world and refreshes the interface
statistics periodically. The Target TPS input controls normal playback from 1
through 1000 ticks per second. Save As is available from the File menu and has
no keyboard shortcut.

The resizable right-side HUD shows the selected agent's identity, energy, age,
generation, position, direction, and four-input/two-output brain. Brain
connections show evolved weight signs and relative magnitudes. Hover nodes or
connections for exact information, use **Reset brain view** to fit the graph,
and expand the parameter table for numeric values. The canvas lays out arbitrary
layer and node counts, while the current simulation brain has four inputs, two
outputs, and no hidden layer. It shows structure and weights, not live brain
activity. The **Show agent information** control and `I` shortcut add energy
bars beside visible agents. Each bar uses the reproduction threshold as its
full reference level, not as a maximum-energy or health value. Selected-agent
details are unavailable during Fast-forward and return after pausing if the
selected stable ID survived.

### Portable viewer directory

Create the self-contained viewer directory after an x64 Release build:

```powershell
cmake --install out/build/x64-release
```

The preset installs to `out/install/x64-release`. The directory contains the
viewer executable, compiled DXIL shaders, this README, and the required project
and dependency license notices. It is a portable directory, not an installer;
the first version does not register `.evo` file associations. The target
Windows 11 development system already provides the Microsoft Visual C++ runtime.

### Render-only performance check

The developer benchmark measures the production renderer without advancing the
simulation. It renders the full world and headings at 2560 x 1440, first with
50,000 agents plus 50,000 food, then with an informational 100,000 plus 100,000
workload, and finally with 50,000 energy bars enabled:

```powershell
.\out\build\x64-release\evobrain_viewer_render_benchmark.exe
```

The required 50,000 plus 50,000 case exits unsuccessfully below 75 render-only
frames per second. Simulation throughput must be measured separately with the
headless executable because simulation speed is not a renderer acceptance
measurement. See `docs/viewer-performance.md` for the recorded acceptance run.

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

Licensed under the MIT License. See `LICENSE`.
