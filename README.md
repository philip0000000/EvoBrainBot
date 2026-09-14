# EvoBrainBot

EvoBrainBot is an artificial-life simulation where autonomous agents evolve behavior through natural selection.

## Status

Early development.

The project currently provides a deterministic headless simulation and a basic
Windows viewer. Agents sense their surroundings, move, breathe, spend energy,
eat plants or other agents, die, reproduce, and inherit mutated brains and traits.

The current mechanics and configuration values are provisional. They establish
a complete testable evolutionary loop and are expected to be refined after
viewer and experiment tooling is available.

See [`docs/brain-architecture.md`](docs/brain-architecture.md) for recurrent
semantics, mutation rules, backend selection, and bounded benchmark commands.
See [`docs/gpu-brain-optimization.md`](docs/gpu-brain-optimization.md) for the
implemented CUDA optimizations, their maintenance constraints, and measured
future optimization candidates.

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

An installed CUDA Toolkit is detected automatically and adds the optional GPU
brain backend. Disable it explicitly with `-DEVOBRAIN_ENABLE_CUDA=OFF`, or
override the default Turing-through-Hopper code targets with
`-DEVOBRAIN_CUDA_ARCHITECTURES=<CMake architecture list>`.

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
* `D`: show or hide spatial-index and simulation-performance diagnostics
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
generation, position, direction, carnivore tendency, a readable lung-adaptation
classification, exact water adaptation, oxygen, rock contact, evolved RGB color,
brain mutation rate and strength, trait mutation rate, prior-tick
bite damage, and 28-input/16-hidden/three-output brain. Brain
connections show evolved weight signs and relative magnitudes. The first eight
hidden neurons are active in founders, while eight gray dormant neurons can be
activated by evolution. Green recurrent connections carry previous-tick hidden
values, and self-connections are drawn as loops. Hover nodes or
connections for exact information, use **Reset brain view** to fit the graph,
and expand the parameter table for numeric values. The canvas lays out arbitrary
layer and node counts. The brain receives RGB and proximity from six literal
finite-range eye rays plus energy, prior-tick bite damage, oxygen reserve, and
rock contact, and outputs turn, move, and eat decisions. It shows structure and
weights, not live brain activity. The **Show agent information** control and `I`
shortcut add energy bars beside visible agents. For
the selected agent it also draws both eye positions, all six maximum-length eye
rays, and the mouth point; these overlays show geometry rather than current
perception or brain activity. Each energy bar uses the reproduction threshold
as its full reference level, not as a maximum-energy or health value.
Selected-agent details are unavailable during Fast-forward and return after
pausing if the selected stable ID survived.

The viewer renders fertility as a continuous terrain gradient: land changes
from brown to green and water from very dark navy to light blue. Wetland land
and water patches share their region's fertility. Rocks remain medium-light grey
with a darker outline, and the global day/night value changes terrain brightness.
These ground colors are observer aids only; agent vision sees grey rocks but not
terrain medium, fertility colors, or the rendered light level.

The paused **Brain backend** selector chooses CPU or an available CUDA GPU
backend. Backend choice is transient execution configuration and is not saved
in checkpoints.

The **Show simulation debug** control and `D` shortcut draw the transient
spatial-index grid without changing simulation state. Occupied cells are tinted,
and selecting an agent highlights the conservative cells queried for its six eye
rays. The information pane reports phase timings, active simulation threads, and
the broad-phase candidate counts compared with the former brute-force workload.

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

### Brain performance benchmark

The brain-only benchmark uses deterministic seed 5 by default and accepts only
the approved automatic tick counts of 100, 500, or 1,000:

```powershell
.\out\build\x64-release\evobrain_brain_benchmark.exe `
    --backend gpu --population 3000 --ticks 100 --seed 5 --mix mixed `
    --replacements-per-tick 1
```

Allowed populations are 250, 300, 2,000, 3,000, 5,000, and 30,000. The optional
`--replacements-per-tick` count models equal-count birth/death churn and cannot
exceed the selected population. Allowed mixes are
`feed-forward-8`, `recurrent-8`, `recurrent-16`, and `mixed`. Runs longer than
1,000 ticks are deliberately rejected. The 10,000-tick smoke/performance run is
manual and is performed by the user, never by automated tests or normal CI.
CPU and GPU results must be measured on the target hardware; small populations
can remain faster on CPU because CUDA launch and transfer costs dominate.

## Run headless

Start a new simulation with a random seed from 1 through 999. It runs until
`Q`, `q`, `SIGINT`, or `SIGTERM` requests a graceful stop:

```sh
./build/EvoBrainBot run
```

Supply a seed and tick limit for a reproducible finite run:

```sh
./build/EvoBrainBot run --seed 1234 --ticks 1000
```

Choose a size for a new world with `--world-size small|medium|large`:

```sh
./build/EvoBrainBot run medium.evo --world-size medium --seed 1234 --ticks 1000
```

Omitting the option selects small. `resume` rejects this option and restores the
saved size and explicit population/food settings without rescaling them.

CPU is the default brain backend. Select an available CUDA build explicitly for
new or resumed runs with `--brain-backend gpu`; unavailable GPU selection fails
instead of silently falling back:

```sh
./build/EvoBrainBot run --seed 1234 --ticks 1000 --brain-backend cpu
./build/EvoBrainBot resume state.evo --ticks 500 --brain-backend gpu
```

With a Visual Studio generator on Windows, the executable is normally in the
selected configuration directory:

```powershell
.\build\Release\EvoBrainBot.exe run --seed 1234 --ticks 1000
```

An attached terminal displays the checkpoint filename, selected seed, completed
ticks, total population, food, reproduction births, random agent introductions,
deaths, and agents killed through eating. `Q` or
`q` stops a finite run early as well as stopping an indefinite run. The final
status is printed after the checkpoint is saved.

### Simulation feature inventory

See [Simulation features](docs/simulation-features.md) for separate world and agent
tables, generation frequencies and calculations, inherited traits, diet energy,
and population/food settings.

Update that document whenever a simulation feature, ability, trait, sensor,
resource, generation rule, or lifecycle rule is added, removed, or materially
changed. Console options and viewer-only presentation are excluded.

### Save and resume

Project policy requires checkpoint writes to follow an explicit user save
command only. The viewer follows this policy through paused Save and Save As
actions, and brain backend changes and benchmarks never save. The headless
`run` command is also an explicit save workflow: it saves when its requested
tick limit is reached or when the user presses `Q` to stop and save. It does not
write periodic or background checkpoints.

Every successful `run` command saves the complete final simulation state when
its tick limit or a graceful stop ends training. If its optional checkpoint
argument is omitted, the state is written to `autosave.evo` in the current
directory, replacing an existing file with that name.

Choose a different checkpoint path with one positional argument. The checkpoint
may appear before, after, or between the options:

```sh
./build/EvoBrainBot run state.evo --ticks 1000 --seed 1234
```

If the final filename contains no dot, `run` appends `.evo`. For example,
`state` becomes `state.evo`, while `state.data` remains unchanged. Dots in
parent directory names do not prevent the extension from being appended.

Resume a checkpoint for an optional additional number of ticks. The completed
state atomically replaces the same checkpoint file:

```sh
./build/EvoBrainBot resume state.evo --ticks 500
```

On Windows with a Visual Studio generator:

```powershell
.\build\Release\EvoBrainBot.exe run state.evo --ticks 1000 --seed 1234

.\build\Release\EvoBrainBot.exe resume state.evo --ticks 500
```

For `resume`, the checkpoint path must match the existing filename exactly;
`.evo` is not appended automatically. `--ticks` means the maximum additional
ticks executed by the current command. Omit it to continue until `Q`, `q`,
`SIGINT`, or `SIGTERM` requests a graceful stop. The completed state atomically
replaces the input checkpoint. An attached terminal accepts `Q` or `q` without
Enter. Non-interactive Linux or RunPod jobs can stop gracefully through
`SIGINT` or `SIGTERM`. A resumed run obtains its original seed, configuration,
accumulated statistics, entities, and random-generator state from the
checkpoint.

Checkpoints use version 16 of the binary format and preserve world size, explicit
population/food settings, brain mutation parameters and inherited trait mutation rate, ecological traits,
oxygen, brain topology, and recurrent runtime memory in addition to the complete
simulation state. Older checkpoint versions are intentionally unsupported.
Unsupported, incomplete, or invalid files are rejected instead of being
partially loaded.

Display the consolidated command help by running the executable without
arguments or with `--help`:

```sh
./build/EvoBrainBot
./build/EvoBrainBot --help
```

On Windows with a Visual Studio generator:

```powershell
.\build\Release\EvoBrainBot.exe
.\build\Release\EvoBrainBot.exe --help
```

## License

Licensed under the MIT License. See `LICENSE`.
