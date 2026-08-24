# Optimization Architecture

This living design note describes how EvoBrainBot divides simulation work for
profiling and optimization. It records current boundaries and likely extension
points, not permanent technology commitments. The design should be revised as
agent counts, brain structures, simulation mechanics, and measurements evolve.

## Workload categories

| Area | Examples | Best initial approach | GPU suitability |
| --- | --- | --- | --- |
| World perception | Eyes, spatial queries, nearby agents and food | Better algorithms and CPU threading | High |
| Brain evaluation | Evaluating the shared brain topology for every agent | Batched CPU calculations | Very high |
| Action resolution | Movement, bites, conflicts, energy transfers, and deaths | Deterministic CPU processing | Medium |
| Ecosystem lifecycle | Food growth, births, mutation, and population rules | CPU; possibly GPU at very large scale | Low to medium |

A fifth level exists above one world: running many independent simulations.
Different worlds and seeds do not depend on one another, so they can be executed
concurrently or evaluated as larger hardware batches. This may be particularly
valuable on remote systems such as Runpod.

## Simulation phase boundaries

The simulation should preserve an explicit ordered pipeline:

1. Build or update the transient world-query representation.
2. Produce a batch of agent perceptions and internal inputs.
3. Evaluate agent brains and produce an action buffer.
4. Apply movement and action-attempt costs.
5. Resolve interactions and conflicts deterministically.
6. Apply ecosystem lifecycle changes.

Future features may add inputs, actions, entity types, or lifecycle rules without
removing these phase boundaries. An optimization backend should process batches
at a phase boundary instead of leaking hardware-specific code into individual
agents, eyes, bites, or reproduction rules.

## Current implementation

The current CPU implementation uses a transient toroidal spatial index as a
broad phase for vision and mouth targeting. Exact ray-circle intersections,
wrapping, physical overlap checks, layer order, and stable-ID tie-breaking remain
the authoritative narrow-phase rules.

Agent sensing and brain evaluation are separate batched phases. Sensing workers
read the same completed world state and write predetermined input slots; the
selected brain backend then consumes contiguous parameter, topology, state, and
input arrays and writes one output slot per agent.
Movement, bite aggregation, births, deaths, mutation, food updates, and
population introduction remain ordered deterministic CPU phases.

The spatial index, execution thread count, timings, and diagnostic counters are
not evolutionary state and are not stored in checkpoints. The viewer's `D`
overlay exposes these diagnostics without changing simulation behavior.

The CPU backend has a specialized dense 26-to-8-to-3 founder path and a general
path for recurrent brains or nine through 12 active hidden neurons. Both use
the same clamped-linear semantics, and recurrent edges read only the completed
previous-tick buffer. Genome arrays are rebuilt only after population or genome
changes; recurrent state and per-tick inputs/outputs use reusable contiguous
buffers.

## Optimization order

For each significant increase in ecosystem complexity or brain size:

1. Measure complete Release-mode workloads at representative populations.
2. Record time by simulation phase and identify the actual bottleneck.
3. Improve algorithms and remove unnecessary work.
4. Improve memory layout, batching, and temporary-buffer reuse where measured.
5. Add bounded CPU parallelism at independent phase boundaries.
6. Consider GPU execution only when the remaining workload is large and regular
   enough to outweigh transfer, synchronization, and dispatch costs.

Correctness, reproducibility, and observable behavior take priority over a faster
implementation that silently changes simulation mechanics.

## GPU considerations

### Brain evaluation

Brain evaluation is naturally parallel when many agents share one topology. A
GPU could evaluate large input, weight, and output batches efficiently. Small
brains and populations may remain faster on the CPU because each simulation tick
offers little work compared with GPU dispatch and data-transfer overhead.

GPU brain evaluation becomes more promising with larger or recurrent brains,
thousands of agents, many simultaneous worlds, or simulation data that remains
resident on the GPU across ticks.

### World perception

Spatial binning, neighborhood queries, ray intersection tests, and sensor-input
generation can also run on a GPU. These calculations are mostly independent per
agent. Dense environmental fields, if ever introduced as simulation mechanics,
could also make growth and sampling hardware-friendly.

Changing individual food objects into a food field would alter ecosystem
behavior and is therefore a separate simulation-design decision, not an implied
optimization requirement.

### Action resolution and lifecycle

Concurrent agents can target the same food or agent. Exact proportional energy
transfers, stable ordering, births, removals, and mutation require conflict
resolution. GPU implementations are possible through sorting and deterministic
reduction passes, but they are more complex than independent perception or brain
evaluation.

A first hybrid backend would likely keep action resolution and lifecycle work on
the CPU. Moving perception and brains to a GPU while returning an action buffer
could help only if per-tick transfers do not dominate the saved computation.

### Multiple simulations

Ticks within one world are sequential, but separate worlds are independent.
Batching many experiments can provide enough work to use a large GPU efficiently
without requiring one ecosystem to contain an extreme number of agents. This
model also supports comparisons between seeds, brain structures, mutation rules,
and ecosystem parameters.

## Current brain-backend decisions

- CPU is the default and remains available without a CUDA toolkit or NVIDIA GPU.
- CUDA is the optional GPU technology; explicit unavailable GPU selection is an error.
- Brain calculations use double precision without fast-math.
- Repeated runs are deterministic within one backend. Direct CPU/GPU evaluation
  must agree within an absolute or relative tolerance of `1e-12`; long evolutionary
  trajectories can still diverge after permitted floating-point differences affect
  later branching.
- Backend choice and diagnostics are transient and absent from checkpoints.
- Sensing, action resolution, and lifecycle remain on the CPU in this issue.
- CUDA assigns stable agent IDs to persistent fixed-stride device slots. Deaths
  release slots and newborns update only reused slots through one packed scatter
  batch, while unchanged structure-of-arrays genomes remain resident. One
  per-tick synchronization covers inputs, outputs, and recurrent state.

## Decisions intentionally left open

Profiling and future requirements must decide:

- the first population or brain size that justifies a GPU backend;
- the measured CPU/GPU crossover population on target hardware;
- whether non-brain world state should ever become GPU-resident;
- how multiple simulations should be scheduled and compared;
- whether later environmental fields belong in the ecosystem model.

See [viewer-performance.md](viewer-performance.md) for recorded performance
measurements. Those measurements are observations, not architectural limits.
