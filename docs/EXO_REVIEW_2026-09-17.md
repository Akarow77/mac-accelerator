# EXO design review for the Mac accelerator

Reviewed official EXO commit `21a54c5ea0230a3bec1e1a786d200126c7e34ec6` through
GitHub's contents API. No EXO installation, service, MLX environment changes,
RDMA configuration, network changes or global memory-limit tuning was performed.

## Applicable ideas, with important limits

The [EXO README](https://github.com/exo-explore/exo/blob/21a54c5ea0230a3bec1e1a786d200126c7e34ec6/README.md)
describes multi-device model distribution and an MLX GPU execution path on Mac.
It is not an Apple GPU+ANE placement engine for this Core ML driving graph.

The inspected
[placement code](https://github.com/exo-explore/exo/blob/21a54c5ea0230a3bec1e1a786d200126c7e34ec6/src/exo/master/placement_utils.py)
allocates contiguous pipeline layers proportionally to nodes' available RAM and
checks per-node capacity. That solves a multi-machine capacity problem; GPU and
ANE sharing one Mac's memory are not two independent RAM pools. For this graph,
choose a boundary using measured stage duration, handoff bytes, numerical error
and paced output age. Layer/operator counts alone are not a cost model.

In [EXO's MLX pipeline code](https://github.com/exo-explore/exo/blob/21a54c5ea0230a3bec1e1a786d200126c7e34ec6/src/exo/worker/engines/mlx/auto_parallel.py),
first/last-layer wrappers handle receive/send; selected prefill sends are queued
then submitted with asynchronous evaluation. Explicit evaluation boundaries and
cache dependencies preserve order. It is not simply “make every operation async”:
there are deliberate synchronizations and collective operations. Our relevant
analogy is submit the next independent image stage before waiting for dependent
work, while retaining buffers until consumption completes. Copying the LLM
batching and all-gather protocol would add dependencies this driving pipeline
does not need.

[MLX streams](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html)
and [unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
are useful references for GPU producer design. They do not automatically make a
Core ML/ANE consumer share the same allocation or synchronization primitives.
That interop and storage lifetime require separate verification.

[MLX distributed documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
distinguishes TCP ring communication from JACCL/Thunderbolt RDMA on supported
Thunderbolt 5 Macs. A 3X USB-NCM connection is not that transport. IPv4 or DHCP
cannot turn it into RDMA. The documentation also warns that its optional fast
Metal synchronization can deadlock; do not enable it as an untested speed tweak.

## Evidence about current resource use

Our [frame-pipeline experiment](FRAME_PIPELINE_EXPERIMENT_2026-09-17.md) established
overlapping GPU/ANE prediction calls after correcting submission order. It does
not measure GPU occupancy, ANE utilization, memory bandwidth saturation or kernel
concurrency. Do not present API duration divided by wall time as hardware usage.

Mean timestamp decomposition of the final process experiment, milliseconds:

| Component | Saturated pipeline | 20Hz pipeline |
| --- | ---: | ---: |
| Admission → GPU call start (prepare/dispatch) | 1.321 | 2.137 |
| GPU prediction call | 13.096 | 16.338 |
| GPU end → ANE start (handoff/queue) | 7.204 | 1.064 |
| ANE prediction call | 17.254 | 21.169 |
| ANE end → completed session output | 1.183 | 1.677 |

These components include OS/API waiting, not just CPU execution. In particular,
the saturated handoff interval includes waiting for the prior ANE frame and is
not seven milliseconds of proven wasted memory copying. The unsplit model already
completed roughly 49fps under saturated input; at a 20Hz source, idle periods can
be normal slack, not a failure to exploit available hardware.

The installed `powermetrics` advertises CPU/GPU/ANE/thermal samplers. A single
noninteractive read-only probe was denied because sudo authentication was
unavailable; no hardware counters were collected. `macmon` is not installed. No
password was requested in chat and no privilege settings were changed.

## Changes and next bounded experiments

- Added `--stem-operations` to the offline pipeline benchmark so candidate cuts
  can be evaluated instead of fixing every chip to the first 40 operations.
  This is an experiment option, not an automatic app/runtime selector.
- Keep two admitted frames, ordered state updates and exclusive buffer ownership.
  Do not manufacture utilization by queuing stale driving frames or waiting for
  future frames before returning an otherwise-ready result.
- A native/asynchronous producer independent of Python's synchronous ANE call is
  the next candidate; measure submission, copies, queue time, GPU, ANE and full
  age separately. Zero-copy is a testable storage property, not a label.
- Compare a small number of cuts under identical 20Hz recorded inputs, warmup,
  thermal conditions and repeated order. Retain the unsplit path unless a
  candidate improves tail latency **and** passes numerical/state checks.
- Collect hardware counters with explicit local administrator authentication or
  a separately approved monitoring setup. Power is not interchangeable with
  occupancy, and whole-system readings include other applications.

The initial run used the 40-operation stem; subsequent follow-up measured cuts
9 and 17 as well (see the appended pipeline report). None beat its own unsplit
reference on paced latency.
EXO is a useful architecture reference, not evidence that this prototype is safe
for vehicle control or that M2 hardware performance has been exhausted.
