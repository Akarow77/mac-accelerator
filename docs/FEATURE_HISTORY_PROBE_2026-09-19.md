# Recurrent-history buffer reuse: offline experiment

Recorded on 2026-09-19 from the local M2 measurement report
`artifacts/feature-history-ring-20260917-01.json` (not committed).
Reproduction tool: `tools/probe_feature_history.py`.

Status: **measured, not integrated into the inference server**. No device,
network configuration, model or vehicle-control changes were made.

## Candidate

The current server shifts a `[128, 32, 512]` float32 history on each frame,
then appends one feature. Replace the shift with a circular write position,
gathering the same 32 stride-4 entries into the existing model input before
appending the current feature. This removes a nominal 8,323,072-byte history
shift per frame; the 2,097,152-byte model-input gather remains.

This borrows the buffer-reuse principle discussed in
[the Chestnut transport review](CHESTNUT_TRANSPORT_REVIEW_2026-09-17.md), not its
USB implementation. It is neither GPU/ANE zero-copy nor a transport optimization.

## Measurement

CPU-only buffer maintenance, including input gather and append. Each block has
256 warmup iterations followed by 256 measured iterations; order is ABBA.
Feature generation, inference, camera capture and network work are excluded.

| Method | Mean ms | p99 ms | Max ms |
| --- | ---: | ---: | ---: |
| Shift, first block | 0.52310 | 0.66330 | 0.79958 |
| Ring, first block | 0.05226 | 0.07963 | 0.14446 |
| Ring, second block | 0.05184 | 0.07650 | 0.12813 |
| Shift, second block | 0.52957 | 0.66616 | 0.68971 |

Across equally sized blocks, this component averages 0.52633ms versus 0.05205ms,
approximately **0.47428ms saved per frame**. This is not a measured reduction in
end-to-end inference latency and does not resolve historical multi-ms stalls.

The prototype passed 1,200 exact history-sequence comparisons with strides 1,
2 and 4, multiple wraparounds and an explicit reset. These use small feature
arrays; the timing uses the actual history dimensions. It also checks that
appending does not mutate an already gathered input. This is not full-model
output equivalence or evidence of real-time safety.

## Before integration

- Test actual model outputs against the existing session on identical inputs.
- Verify session reset, failure recovery and input-buffer ownership/lifetimes.
- Repeat sustained shadow-only timing; retain deadlines and failure checks.
- Measure physical 3X capture/readback and USB transfer separately before
  attributing remaining latency to networking or attempting transfer overlap.

Example (use a new output filename; existing reports are never overwritten):

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .coreml-venv/bin/python \
  tools/probe_feature_history.py --output artifacts/feature-history-new.json
```
