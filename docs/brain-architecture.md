# Evolvable brain architecture

## Topology and activation

Every brain has a fixed capacity of 26 inputs, 12 clamped-linear hidden neurons,
and three clamped-linear outputs. Founders reproduce the original dense
26-to-8-to-3 feed-forward behavior: hidden neurons zero through seven are active,
neurons eight through eleven are dormant, and every recurrent connection is
disabled.

Dormant neurons have no behavioral effect. A successful activation mutation
enables the neuron together with at least one sensor input and one output
connection. Recurrent mutations may connect any active hidden neuron to any
active hidden neuron, including itself. Fixed capacities and byte masks avoid
per-agent topology allocation and give CPU and CUDA backends a common layout.

Mutation rate and mutation strength are always clamped to positive floors:

- Mutation rate: `0.0001`
- Mutation strength: `0.001`

These floors apply after inheritance and mutation and while restoring a
checkpoint, so a lineage cannot permanently lose the ability to mutate.

## Previous-tick recurrence

The general evaluator owns separate previous and next arrays of 12 hidden
values. During a tick, every recurrent edge reads only its source value from the
previous array. All next values are complete before they replace the previous
array. Hidden iteration order therefore cannot create same-tick feedback.

One recurrent edge advances a signal one connection per simulation tick. A
self-connection can retain a value, and a loop of several hidden neurons can
produce delayed or pulsing behavior. This provides memory without special LSTM,
oscillator, spiking, inhibitory, or other neuron classes.

Recurrent state belongs to a living agent rather than its genome. It persists
between ticks and in a checkpoint so resume is exact, but founders and newborn
children start with zero recurrent state instead of inheriting a parent's
memory.

## Evaluation backends

Sensing writes one contiguous input batch. Brain evaluation then consumes
contiguous parameter, topology, recurrent-state, and input arrays and produces
one contiguous output batch. The CPU backend uses a dense founder fast path and
a general masked recurrent path. Backend dispatch occurs once per batch.

The CUDA backend retains allocations and unchanged genomes on the device. Each
stable agent ID owns a persistent GPU slot in fixed-stride structure-of-arrays
storage. Deaths release slots, newborns reuse them, and one packed update batch
scatters only new genomes; unchanged weights and topology are not repacked or
uploaded. A rare full rebuild occurs on first use, backend replacement, capacity
growth, or excessive fragmentation. Current inputs are uploaded and outputs plus
recurrent state are downloaded through one ordered per-tick synchronization.
The rest of the simulation remains CPU-owned.

Retained agent IDs are assumed to keep identical weights and topology. If a
future feature changes a living agent's genome, it must explicitly upload that
changed genome or reset the CUDA cache before the next evaluation. See
[`gpu-brain-optimization.md`](gpu-brain-optimization.md) for the complete list
of implemented optimizations and possible future work.

CPU is the default. `--brain-backend cpu|gpu` selects execution for both new and
resumed headless runs. The viewer exposes the same selector only while paused.
Backend choice is not evolutionary state and is not stored in checkpoints.
Explicit GPU selection must fail clearly when CUDA is not available; it must
never silently select CPU.

Both backends use `double`. Repeated execution with the same seed,
configuration, and backend must be deterministic. Direct CPU/GPU outputs must
agree within absolute or relative tolerance `1e-12`. Long evolutionary runs can
eventually diverge across backends when a permitted floating-point difference
changes a later branch or selection outcome.

## Benchmarks

`evobrain_brain_benchmark` accepts only the bounded automatic matrix:

- Populations: 250, 300, 2,000, 3,000, 5,000, or 30,000
- Ticks: 100, 500, or 1,000
- Mixes: `feed-forward-8`, `recurrent-8`, `recurrent-12`, or `mixed`
- Optional equal-count churn: `--replacements-per-tick <count>`
- Default deterministic seed: 5

Example:

```powershell
.\out\build\x64-release\evobrain_brain_benchmark.exe `
    --backend gpu --population 3000 --ticks 500 --seed 5 --mix recurrent-12 `
    --replacements-per-tick 1
```

The churn option deterministically replaces that many stable identities per tick
to model same-count deaths and births. The command reports initialization time,
brain time, average time per tick, agent-brain evaluations per second, and a
deterministic output checksum. Record
the operating system, CPU/GPU model, build type, compiler, CUDA toolkit/driver,
and command line alongside results.

CUDA builds are optional. CMake detects an installed toolkit when
`EVOBRAIN_ENABLE_CUDA=ON` and otherwise produces a CPU-only build. The default
`EVOBRAIN_CUDA_ARCHITECTURES` value covers compute capabilities 7.5, 8.0, 8.6,
8.9, and 9.0 and can be overridden for a target RunPod image.

Automatic tests, normal CI, and automated benchmarks must never exceed 1,000
ticks. The user performs the documented 10,000-tick smoke/performance test
manually because it may take 10–20 minutes.

### Local reference measurement

On 2026-08-23, an x64 Release build with MSVC, CUDA 13.3, driver 610.88,
Ryzen 7 3700X, and RTX 2070 SUPER produced these representative seed-5 mixed
results over 1,000 ticks:

| Agents | CPU ms/tick | CUDA ms/tick | Faster backend |
| ---: | ---: | ---: | :--- |
| 300 | 0.066 | 0.308 | CPU |
| 3,000 | 0.242 | 0.570 | CPU |
| 30,000 | 5.625 | 4.093 | CUDA |

CPU and CUDA checksums matched exactly in these runs. The crossover is specific
to this hardware and workload; RunPod targets must be measured independently.

## Checkpoints and saving

Checkpoint version 4 stores expanded genomes and recurrent state but not backend
choice or diagnostics. Version-3 fixed brains load as eight-active/four-dormant
brains with zero recurrence.

The project policy is explicit-save-only: periodic, per-tick, background,
pause, crash-recovery, backend-change, benchmark, and RunPod-specific automatic
checkpoints are prohibited. The headless `run` workflow intentionally saves
after its requested tick limit or when the user presses `Q` to stop and save;
these are explicit parts of that command rather than background autosaves. This
brain work does not add another save path.
