# Shadow hardening verification — 2026-09-16

Device scope: MacBook Air M2 only. No ADB, 3X firmware, Params, live camera hook or
vehicle-control integration was used in this work.

## Corrections

1. Partial `sendmsg`/`send` calls now share one absolute deadline. Previously each
   partial operation could reuse a timeout, exceeding the intended total budget.
2. FP16-to-FP32 inference results now own immutable bytes. Previously returned
   memoryviews referenced the reusable output buffer, so retained results could
   change after a later inference.
3. Non-finite policy inputs are rejected before the server mutates recurrent
   queues or calls the model. The server CLI also requires an authentication key.
4. New observation-only replay has strict source ordering/clock checks, bounded
   private logs, exact input contract validation and latched failure. No automatic
   restart or output path into a vehicle exists.

## Tests

- 51 unit tests pass, including partial-send deadline, output ownership, source
  gap, future/stale timestamps, stale output, invalid dimensions, non-finite
  inputs/outputs, disconnect, log quota and simulated disk-full failures.
- Python lint and diff whitespace checks pass.
- Synthetic localhost shadow: 120/120 frames complete; 55 post-context samples
  have zero 50ms target misses.
- Synthetic localhost shadow: 600/600 frames over 30 seconds complete; all 600
  have zero measured 50ms target misses (535 post-context samples). Scheduled
  local playback to output averages **32.68ms**, maximum **41.16ms**. Request/response
  averages **29.12ms**, maximum **36.26ms**. This includes no camera readback or USB.
- Terminating the test-owned server after 1.5 seconds produces a logged, latched
  failure and nonzero observer exit. No reconnect occurs. The harness accepts both
  EOF and TCP reset/broken-pipe reports; the first harness attempt recognized EOF
  only and reported TCP reset as unexpected, although the observer stopped correctly.

The 600-frame run was not a controlled thermal benchmark; lightweight code checks
were also performed during the test. Repeated synthetic images do not exercise
real route diversity. Real recorded-warp CLI input handling is implemented but a
matching real warp+policy sequence was not newly replayed in this verification.

Logs explicitly keep `sustainedQualified=false` and `roadReady=false`. The 150ms
stale-stop limit allows diagnostics of 50ms misses; it is not a passing real-time
budget. No claim of bug-free execution, validated conversion fidelity, stable
hardware, or road readiness follows from this work.

## Remaining blocker

Live shadow needs an independently owned camera/preprocessing path whose resource
interference is measured against the unchanged local model. See
[the design gate](SHADOW_DESIGN.md). M2 performance is promising for this workload;
these tests do not establish that live integration can meet every deadline.
