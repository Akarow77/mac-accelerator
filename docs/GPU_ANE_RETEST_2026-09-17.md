# Core ML ALL versus CPU + Neural Engine — 2026-09-17

## Scope and method

Measured the existing `artifacts/big_driving.mlpackage` on the user's M2 Mac,
using `benchmark_coreml.py`, not a camera/USB replay or a new zero-copy pipeline.
No 3X access, app configuration changes, model conversion or runtime code changes
were made. No existing accelerator server or benchmark process was observed at
the start. Other system activity and thermal conditions were not controlled.

Order: ALL, CPU_AND_NE, CPU_AND_NE, ALL. Each fresh process used the same seeded
synthetic model inputs, 40 warmup predictions and 600 measured predictions paced
at 20Hz (30 seconds). User-interactive QoS was confirmed in each process. The
test parent held an idle-sleep assertion and set VECLIB_MAXIMUM_THREADS=1 and
OPENBLAS_NUM_THREADS=1. Model configuration used the benchmark's default
specialization settings, identically across runs.

Timing surrounds `predict()` in the benchmark: synchronous Core ML prediction,
output shape/finite checks, and recurrent feature-history update. Model loading,
warmup, input generation, pacing sleep, camera preprocessing and network transfer
are outside this timer. The images remain synthetic and fixed while recurrent
feature state changes. These are not driving-accuracy or pure ANE-kernel timings.

Subsequent upstream review found that this raw benchmark's one-step-per-call
feature update is not the model's stride-4 streaming contract. These measurements
remain raw-graph timings, not upstream temporal-correctness validation. See
[the temporal correction](UPSTREAM_BIG_MODEL_REVIEW_2026-09-17.md).

## Measurements

| Order | Allowed compute units | Samples | Mean ms | p99 ms | Max ms | >50ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | ALL | 600 | 23.6902 | 27.2151 | 43.3506 | 0 |
| 2 | CPU_AND_NE | 600 | 23.1014 | 26.7592 | 49.1251 | 0 |
| 3 | CPU_AND_NE | 600 | 24.3095 | 26.3454 | 26.8839 | 0 |
| 4 | ALL | 600 | 24.0082 | 26.4111 | 45.2669 | 0 |

Combined ALL mean: 23.8492ms, maximum 45.2669ms, 0/1200 over 50ms.
Combined CPU_AND_NE mean: 23.7055ms, maximum 49.1251ms, 0/1200 over 50ms.
The mean difference is only 0.1437ms (about 0.6%); this short comparison does not
establish a meaningful speed advantage for ALL. Pooled p99 cannot be recovered
from the saved per-run percentile summaries and is deliberately not reported.

Pacing lateness is a separate measurement, not part of prediction duration. The
maximum observed scheduling lateness was 13.0889ms in run 2. Zero prediction
duration misses therefore do not mean zero scheduled-start-to-output misses.

Raw reports (local, ignored artifacts):

- `artifacts/gpu-ane-abba-20260917-ec_bde09/1-all.json`
- `artifacts/gpu-ane-abba-20260917-ec_bde09/2-cpu-and-ne.json`
- `artifacts/gpu-ane-abba-20260917-ec_bde09/3-cpu-and-ne.json`
- `artifacts/gpu-ane-abba-20260917-ec_bde09/4-all.json`

## Anticipated compute-device placement

After timing completed, loaded each configuration separately and queried
`MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=...)`.
All program functions and nested operation blocks were inspected using
`get_compute_device_usage_for_mlprogram_operation`.

Both ALL and CPU_AND_NE returned the same preferred-device operation counts:

| Preferred device | Operations |
| --- | ---: |
| Neural Engine | 675 |
| CPU | 1 (`ios18.concat`) |
| GPU | 0 |
| No device returned | 1190 (all `const`) |

The plan places convolution, linear, matrix multiplication, normalization and
other principal operators on the Neural Engine. The one CPU concat is a model
operation count, not a statement that the Python process does no CPU work.
Counts are not runtime-duration percentages. This is an anticipated compiler
plan, not an Instruments trace proving hardware activity on every prediction.

[Apple's Compute Plan documentation](https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html)
describes these APIs as execution-resource estimates. ALL allows GPU and ANE;
it does not force the model to use both.

## Conclusion and remaining gap

Retesting with GPU permitted produced approximately 24ms model-stage latency,
but the plan still preferred ANE and CPU, with no GPU-selected operators. This is
not evidence that a simultaneously GPU+ANE, zero-copy implementation was tested.
No forced model partitioning or shared-surface input implementation was added.
Any such implementation needs its own placement, output-equivalence, recurrent
state, memory-lifetime and completed-prediction latency checks.

`roadReady=false`; no end-to-end device or sustained driving qualification.
