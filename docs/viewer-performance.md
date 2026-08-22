# Windows viewer performance verification

This record verifies the basic viewer's render acceptance target without
mixing simulation cost into the result. It contains no personally identifying
hardware details.

## Method

The x64 Release `evobrain_viewer_render_benchmark` uses the production SDL_GPU
D3D12 world renderer and build-generated DXIL shaders. It renders to an
offscreen 2560 x 1440 target with the complete 2.5 by 2.5 world visible. Each
agent has a body and heading, each food object is visible, and the same compact instanced
batch path used by the viewer uploads and draws every frame.

The required workload keeps agent information hidden to verify that optional
inspection rendering does not regress the existing baseline. A separate
informational workload enables energy bars and diet markers for all 50,000
agents and highlights one selected agent with its eye and mouth geometry. It
uses the same batched upload and draw path.

The benchmark performs warm-up frames, measures complete CPU preparation,
instance upload, and GPU drawing, then waits for GPU completion before
reporting. It does not create simulation ticks, a swapchain, or a visible
window. The required workload fails the process below 75 render-only frames per
second; the larger workload is informational.

## Recorded result

Measured again on 2026-08-22 on the target Windows 11 x64 development system
after introducing evolved colors and predator-prey inspection geometry:

| Workload | Result | Status |
| --- | ---: | --- |
| 50,000 agents + 50,000 food | 142.9 FPS | Pass (75 FPS required) |
| 100,000 agents + 100,000 food | 75.2 FPS | Informational |
| 50,000 agents + 50,000 food + information overlays | 78.3 FPS | Informational |

The viewer remains batched at these counts: world/background shapes use one
draw, while agent headings, bodies, selection glow, energy bars, and food use
one clipped entity draw. There is no draw call per object or energy bar.

## Separate simulation observation

As a separate product-side measurement, the x64 Release headless executable
completed 10,000 deterministic predator-prey simulation ticks in approximately
1.48 seconds (about 6,750 ticks per second) for its default seed-1234 starting
configuration after adding the toroidal spatial broad phase and bounded parallel
sensing/brain evaluation. The pre-optimization measurement was approximately
29.54 seconds. Both runs ended with the same 160 agents, 157 herbivores, three
carnivores, 1,000 food items, and lifecycle counters. This observation is not a
viewer render result and has no acceptance threshold.

Results can vary with system load and hardware. Re-run the renderer check with:

```powershell
.\out\build\x64-release\evobrain_viewer_render_benchmark.exe
```
