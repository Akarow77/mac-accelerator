# Research summary — 2026-09-29

[English](RESEARCH_SUMMARY.md) · [한국어](RESEARCH_SUMMARY.ko.md) · [Project](../README.md)

## Scope

Mac Accelerator is an independent Apple Silicon / Core ML benchmark app and
server, developed from an M2 MacBook Air + comma 3X experiment. It is not an
official comma/sunnypilot product, a native eGPU driver or a road-ready control
system. Models and device firmware are not distributed here.

## Implemented capabilities

- Independent app packaging, preferences and authentication identity.
- Authenticated, size-bounded transport and Mac-side observation-only replay.
- Opt-in NV12 preprocessing comparisons and bounded detached-device bench tools.
- Corrected per-request deadlines in the experimental transport test server.
- Experimental scatter/gather framing checks. Production size limit remains
  4 MiB; 8 MiB was an explicit benchmark override, not a production change.

The app has no live 3X camera hook, device parameter writer or vehicle-control
output publisher. GPU/ANE split experiments and the recurrent-history prototype
are not integrated into its inference server.

## Measurements and limitations

| Workload | Result | Scope |
| --- | --- | --- |
| Short 3X → Mac USB-NCM bulk test | ~470 Mbps; negotiated link 5 Gbps | Not Chestnut throughput or a cable bandwidth limit |
| Authenticated synthetic USB + Core ML CPU/NE | 132 warmup + 100 measured frames; scheduled-to-result mean 41.296 ms, max 44.849 ms; inference mean 29.291 ms | Excludes camera capture and calibrated preprocessing; not long-run qualification |
| Owned raw camera snapshot transfer | 4,804,608 bytes/image; scatter/gather mean 106.759 ms vs contiguous 119.530 ms and chunks 131.524 ms | Four samples/variant; not continuous or synchronized camera pairs |
| High-core-affinity raw transport ABBA | Mean 20.52 → 10.38 ms | No persistent policy or proof of coexistence with local modeld |
| Recurrent-history ring prototype | Maintenance ~0.526 → 0.052 ms; 1,200 exact sequence checks | Offline prototype, not full-model equivalence |

The 41 ms synthetic test and 107 ms snapshot test are different workloads.
They do not establish a continuous camera-to-output 50 ms path.
See [measured progress](PROGRESS_2026-09-29.md) and
[camera benchmark methodology](CAMERA_USB_BENCH_2026-09-29.md).

## Known power-stability issue

Whole-device resets were observed in the vehicle + direct Mac USB topology,
including on restored sunnypilot master. Operator observations report no resets
after Mac disconnection or in the vehicle-detached environment. These were not
matched-load/duration A/B tests. Saved boot logs include UVLO/SMPL indications;
this does not uniquely identify backfeed, a failed component or source selection.

Sampled USB-input readings included ~4.39 V / 1.35 A and, on another boot,
~4.84 V / 1.30 A. PD reported 5 V with a 3 A ceiling. These are driver snapshots,
not reset-time waveforms or Mac-connector measurements. A preserved panic's FIFO
chain matches GLINK internal IPC source, not necessarily the USB/camera path.
The relationship between the panic and power event is unconfirmed.

There is no validated electrical fix or supported data-only power setting.
Direct Mac USB use in the vehicle remains paused. See
[power status and source references](POWER_STABILITY_STATUS.md).

## Upstream architecture findings

The inspected Chestnut path uploads the NV12 image extent before GPU warp,
not pre-shrunk model tensors or the entire camera allocation. For the inspected
1928×1208 layout, the NV12 extent is 3,735,552 bytes/image versus 4,804,608 bytes
in the snapshot allocation. The 22.25% size difference is arithmetic, not an
implemented speedup. Two such images at 20 Hz require ~1.195 Gbps before overhead.
See [pinned source analysis and bandwidth calculations](USB_POWER_ISOLATION_RESEARCH_2026-09-29.md).

## Validation status

Continuous calibrated capture, recurrent accuracy, local-model coexistence,
disconnect/stale-result behavior and sustained thermal latency remain
unqualified. M1 and other M chips are untested. These results do not qualify
vehicle control. Raw logs, identifiers, network addresses, credentials, models
and camera images are excluded from public reports.
