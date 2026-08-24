# GPU brain optimization

This document records the CUDA brain optimizations that are currently
implemented, the assumptions they introduce, and optimization ideas that have
not been implemented. It is a planning reference, not a commitment to implement
every candidate.

Performance depends on brain size, population, population churn, transfer cost,
and the target CPU and GPU. Future work must be selected through profiling and
bounded benchmarks on the hardware where the simulation will run.

## Implemented optimizations

| Optimization | Expected value | Best conditions | Constraint or risk | Notes |
| --- | --- | --- | --- | --- |
| Batched CUDA evaluation | High for sufficiently large populations | Enough agents to amortize launch and transfer overhead | Small populations can remain faster on CPU | One kernel launch evaluates the complete population batch |
| Persistent genome cache | Medium to high during population churn | Most agents survive between ticks | A retained agent ID is assumed to keep identical weights and topology | Only newborn or otherwise new genomes are normally uploaded |
| Stable device slots keyed by agent ID | Medium | Frequent births and deaths with many surviving agents | IDs must remain unique, nonzero, and stable for each living agent | Death releases a slot and a newborn can reuse it |
| Incremental packed genome updates | Medium | A small fraction of the population is new each tick | Runtime genome changes must also be explicitly reported | One scatter kernel applies all changed-slot uploads |
| Persistent device allocations | Medium | Repeated ticks with a stable or slowly growing capacity | Large capacity growth can require allocation and a complete rebuild | Capacity is retained when population later shrinks |
| Fixed-stride structure-of-arrays layout | Medium | Large batches reading the same genome component | Changing brain dimensions requires corresponding layout and kernel updates | Adjacent threads read adjacent agent values |
| Founder feed-forward fast path | Medium while simple brains are common | Founders and descendants with the exact original topology | Any topology change must disable or update the fast path | Avoids topology-mask and recurrent-state work |
| One CUDA thread per brain | Good for the current small brains | Many small independent 26-to-12-to-3 brains | Larger future brains may need cooperative threads | Keeps the current kernel and accumulation order simple |
| Reusable input, output, and recurrent-state buffers | Small to medium | Repeated evaluation with stable capacity | Inputs, outputs, and recurrent state still cross the CPU/GPU boundary | Avoids allocation on every tick |
| One ordered synchronization boundary per tick | Small to medium | Batched GPU work | The CPU must still wait before applying agent actions | Covers transfers and kernel completion together |
| Fixed double-precision semantics with fused multiply-add disabled | Correctness rather than speed | Deterministic CPU/GPU comparison | Gives up possible CUDA throughput | Preserves close numerical agreement with the CPU backend |

### Persistent-cache invariant

The persistent genome cache is the most important maintenance constraint. A
surviving agent currently inherits no runtime changes to its weights or
topology, so its cached device genome remains valid. A future feature such as
lifetime learning, brain damage, temporary connections, or environmental genome
effects must do one of the following before the next GPU brain evaluation:

1. Mark that agent's genome as changed and upload it into its existing slot.
2. Reset the complete CUDA genome cache.

Failing to do either can silently evaluate the agent with stale weights or
topology. Changes to recurrent memory alone are not genome changes; recurrent
state already has its own synchronization path.

## Not implemented

The value ratings below are hypotheses until measured with the bounded brain
benchmark and representative full simulations.

| Candidate | Potential value | Best conditions | Cost or consequence | Recommendation |
| --- | --- | --- | --- | --- |
| Per-agent dirty-genome uploads | High if living brains can change | Lifetime learning or other runtime genome modification | Adds explicit change tracking and more uploads | Add only when a feature requires mutable living genomes |
| Pinned host transfer buffers | Probably small | Measurements show pageable transfers are significant | More allocation and ownership complexity | Benchmark before implementing |
| CUDA Graphs | Small to medium | Repeated launches with a stable execution shape | Dynamic population updates complicate graph reuse | Benchmark launch overhead first |
| Overlapped transfers and multiple CUDA streams | Probably small for the current tick dependency | Substantial independent CPU work exists between brain phases | Adds synchronization complexity and may not hide required transfers | Profile the complete tick before implementing |
| Cooperative CUDA threads per brain | Potentially high for much larger brains | Future networks contain far more neurons or connections | Likely slower for the current tiny brains and requires a new kernel layout | Reconsider only after brain size grows materially |
| Topology bucketing or specialized recurrent kernels | Workload dependent | Populations contain large groups with similar sparse topologies | Sorting, dispatch, and maintenance overhead can exceed saved arithmetic | Use topology measurements before implementing |
| Sparse connection storage | Workload dependent | Evolved brains contain many disabled connections | Indirection can be slower than fixed masks for small or dense brains | Do not use unless profiling shows persistent sparsity |
| Optional single-precision (`float`) evaluation | Potentially high on GPU | Throughput matters more than current numerical agreement | Changes precision, determinism, checkpoint expectations, and evolution trajectories | Treat as a separate measured compatibility decision |
| Vision and brain on the GPU | Potentially large for large populations | Vision is a dominant cost and GPU-resident sensing can feed brains directly | Major simulation-boundary redesign; environmental features increase complexity | Implement as a separate focused issue after profiling |
| Agent actions or the complete simulation on the GPU | Potentially very large at scale | Most simulation phases can remain device-resident | Major architectural change with difficult dynamic data and determinism concerns | Not justified by brain-only measurements |

## Evaluation rules

- Keep CPU as the available reference backend.
- Do not add a GPU optimization solely from theoretical throughput estimates.
- Compare CPU and CUDA on deterministic seeded scenarios and representative
  population churn.
- Automatic tests and benchmarks must not exceed 1,000 ticks. The user performs
  longer smoke and performance runs manually.
- Record hardware, build mode, compiler, CUDA toolkit and driver, population,
  brain mix, churn, tick count, and seed with benchmark results.
- Prefer removable backend-local changes over optimizations that spread GPU
  assumptions into environment and agent-feature code.
