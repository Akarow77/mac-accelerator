# GPU frame N+1 / ANE frame N — M2 experiment

## What was actually built

`tools/benchmark_frame_pipeline.py` extracts two independent Core ML submodels.
The tested `stem` cut contains the first 40 image-dependent operations, with
frontier tensors `input_21_cast_fp16` and `input_23_cast_fp16`. Dependency analysis
excludes policy/recurrent inputs from the GPU stage. The ANE stage receives these
tensors plus current policy inputs and correctly ordered recurrent history.

Anticipated plans: GPU stage 40 GPU operations; ANE stage 635 Neural Engine
operations and one CPU concat. Constant operations have no preferred device.
These counts are not time percentages or hardware execution traces.

A separate GPU process uses two preallocated shared-memory slots (10,485,760 bytes
total). A bounded admission semaphore permits at most two frames including the
frame being consumed. Slots are not reused until synchronous ANE consumption
returns. Input preparation and GPU result copies still occur; Core ML may perform
additional copies internally. **This is not verified zero-copy.**

The saturated scheduler submits the next GPU frame before entering the parent's
synchronous ANE call. This matters: just using threads, or moving GPU prediction
to a process without fixing dispatch order, produced zero adjacent-call overlap.
No GPU/ANE hardware-counter or Instruments trace was collected; recorded intervals
prove overlapping prediction calls, not their exact simultaneous kernel activity.

The 20Hz mode never waits for an unavailable future frame merely to force overlap.
The transport protocol/server/app is unchanged by this experiment and still has
one in-flight request. This script is not a production network pipeline.

## Method and saturated result

MacBook Air M2, 132 warmup frames and 120 measured frames per case. Seven seeded
synthetic image pairs repeat; this is not a recorded driving route. All variants
use `CoreMLPolicySession` with stride 4 and independent, identically initialized
state. The CPU math-thread limits are one, and each prediction thread/process
requests user-interactive QoS. A temporary idle-sleep assertion covers the run.
Other app activity and temperature were not controlled.

Final run: `artifacts/frame-pipeline-stem-process-20260917-03.json`.

| Mode | Completed frames/s | Mean admission→output ms | p99 ms | Max ms |
| --- | ---: | ---: | ---: | ---: |
| Unsplit CPU+ANE, before | 49.26 | 20.28 | 22.54 | 22.87 |
| Split GPU→ANE, serial | 29.86 | 33.44 | 42.82 | 45.96 |
| Split GPU(N+1)/ANE(N), pipelined | 50.61 | 40.06 | 42.42 | 42.54 |
| Unsplit CPU+ANE, after | 48.72 | 20.51 | 23.25 | 23.32 |

Pipelining improved completed throughput about 69.5% over the same serial split.
All 119 adjacent measured frame pairs had overlapping GPU/ANE calls; mean overlap
was 12.30ms. GPU call mean was 13.10ms and ANE call mean 17.25ms. Output completion
interval mean was 19.76ms, distinct from the 40.06ms individual-frame age.

This supports the proposed frame-pipeline mechanism. It does not establish a
significant advantage over the unsplit model: that already delivered roughly
49fps, with about half the per-frame age. Admission occurs only when a slot is
available; saturated age excludes any upstream camera/source backlog. Paced
scheduled-to-output age is therefore also required.

## Paced 20Hz result and decision

Same final process, 132 warmup + 120 measured frames per case, starts scheduled
every 50ms. Scheduled age includes wake lateness, preparation, IPC, queue waiting,
inference and session output/state work, but **not camera or USB transport**.

| Mode | Mean scheduled→output ms | p99 ms | Max ms | >50ms |
| --- | ---: | ---: | ---: | ---: |
| Unsplit CPU+ANE | 28.48 | 31.96 | 35.79 | 0/120 |
| Split, process pipeline | 45.48 | 65.26 | 68.06 | 52/120 |

No adjacent prediction-call overlap was observed in this paced run. At 20Hz the
next image usually is not available while the current frame is processing;
shared-memory preparation/dispatch in the parent also remains subject to Python
scheduling. Do not add artificial next-frame waiting to manufacture overlap: it
would age the current result. Process pipeline output again exactly matched its
serial split after context fill.

**Do not adopt this split as the default.** It demonstrates saturated throughput
overlap, but performed worse on the relevant paced latency measurement. Preserve
the non-split CPU+ANE worker. Further work should examine a smaller/cheaper
boundary and a native independent producer with explicit buffer lifetime and
equivalence tests, not infer M2 hardware incapability from this prototype.

## Numerical and temporal limitations

Measured pipeline outputs were bit-for-bit equal to the same serial split for all
120 measured frames after 132 warmup frames. They were **not** identical to the
unsplit CPU+ANE model: maximum absolute difference was 0.21875. Different output
fields have different units; this number is not a driving-accuracy pass/fail
threshold. No acceptance threshold or real-route equivalence was established.

The previous raw graph backend-placement probes are a different experiment:
they were serial, with a simplified recurrence update, not this stride-4 streaming
pipeline. Their reports are retained in `artifacts/partition-20260917-af476k9f`.
Forcing policy operations to GPU produced an all-GPU plan and was rejected rather
than reported as GPU+ANE. Initial submodel extraction failed at Python's default
recursion depth; a subsequent shared-memory trial rejected a float32/float16 API
dtype mismatch. Both failures are retained, not treated as valid timing runs.

No vehicle access, deployment, auto-start or DHCP changes during measurement.
Research documentation and source may subsequently be synchronized to GitHub.
`roadReady=false`, `zeroCopyVerified=false`, `numericalQualified=false`.

Verification: all 76 unit tests passed, Ruff passed and `git diff --check` passed.
A separate authenticated localhost shadow run after the stride correction
completed 200 synthetic frames: 69 full-context samples, zero >50ms in that
window (mean 30.24ms, p99 34.24ms, max 35.31ms). Its log is
`artifacts/shadow-stride4-20260917-01.jsonl`. This checks the unchanged unsplit
worker's local plumbing, not the split pipeline, live camera transport or driving.
Test-created benchmark/server processes were confirmed stopped.

## Reproduce

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  caffeinate -i .coreml-venv/bin/python tools/benchmark_frame_pipeline.py \
  --cut stem --gpu-process --output artifacts/new-unique-pipeline-report.json
```

Requires the project's already-installed Core ML environment and local model
package. Existing report files are never overwritten. Temporary extracted GPU
packages and the worker/shared-memory objects are cleaned up after each case.
The script supports a `vision` cut for further research, but only the `stem`
frame pipeline above has been measured in this turn. Model extraction is expensive
and the short run is not thermal/endurance qualification.

## Smaller-cut follow-up

After the user requested another iteration, tested image-operation cuts 9 and 17
with the same 132-frame warmup and 120-frame measurement per mode. Added a paced
serial-split case to separate split cost from pipeline orchestration cost.

| Cut | 20Hz mode | Mean scheduled age ms | p99 ms | Max ms | >50ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 9 | Unsplit reference | 31.74 | 36.26 | 36.32 | 0/120 |
| 9 | Serial split | 36.39 | 40.49 | 50.93 | 1/120 |
| 9 | Process pipeline | 36.83 | 44.17 | 45.60 | 0/120 |
| 17 | Unsplit reference | 32.45 | 35.20 | 35.39 | 0/120 |
| 17 | Serial split | 40.50 | 48.02 | 72.31 | 1/120 |
| 17 | Process pipeline | 41.49 | 45.88 | 46.57 | 0/120 |

Reports: `artifacts/frame-pipeline-cut9-20260917-01.json` and
`artifacts/frame-pipeline-cut17-20260917-01.json`. Both pipelines matched their
serial splits exactly; max absolute differences from the unsplit model were
0.21875 and 0.34375 respectively. No driving equivalence threshold is established.
Different run temperatures/activity were not controlled; compare each candidate
with its own reference, not just against the earlier 40-op run. Neither smaller
cut beat its reference's paced latency. Keep the unsplit default and prioritize
the physical transport/readback investigation. Only 9, 17 and 40-op cuts have been
measured as frame pipelines, not a global search over all partitions.
