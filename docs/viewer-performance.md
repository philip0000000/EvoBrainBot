# Windows viewer performance verification

This record verifies the basic viewer's render acceptance target without
mixing simulation cost into the result. It contains no personally identifying
hardware details.

## Method

The x64 Release `evobrain_viewer_render_benchmark` uses the production SDL_GPU
D3D12 world renderer and build-generated DXIL shaders. It renders to an
offscreen 2560 x 1440 target with the complete world visible. Each agent has a
body and heading, each food object is visible, and the same compact instanced
batch path used by the viewer uploads and draws every frame.

The required workload keeps agent information hidden to verify that optional
inspection rendering does not regress the existing baseline. A separate
informational workload enables energy bars for all 50,000 agents and highlights
one selected agent. It uses the same batched upload and draw path.

The benchmark performs warm-up frames, measures complete CPU preparation,
instance upload, and GPU drawing, then waits for GPU completion before
reporting. It does not create simulation ticks, a swapchain, or a visible
window. The required workload fails the process below 75 render-only frames per
second; the larger workload is informational.

## Recorded result

Measured on 2026-08-21 on the target Windows 11 x64 development system:

| Workload | Result | Status |
| --- | ---: | --- |
| 50,000 agents + 50,000 food | 130.6 FPS | Pass (75 FPS required) |
| 100,000 agents + 100,000 food | 69.2 FPS | Informational |
| 50,000 agents + 50,000 food + energy bars | 77.5 FPS | Informational |

The viewer remains batched at these counts: world/background shapes use one
draw, while agent headings, bodies, selection glow, energy bars, and food use
one clipped entity draw. There is no draw call per object or energy bar.

## Separate simulation observation

As a separate product-side sanity check, the x64 Release headless executable
completed 10,000 deterministic simulation ticks in approximately 0.16 seconds
for its default seed-1234 starting configuration. This observation is not a
viewer render result and has no acceptance threshold in this issue.

Results can vary with system load and hardware. Re-run the renderer check with:

```powershell
.\out\build\x64-release\evobrain_viewer_render_benchmark.exe
```
