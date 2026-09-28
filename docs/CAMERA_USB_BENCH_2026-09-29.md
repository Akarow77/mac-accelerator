# Detached 3X camera transport bench — 2026-09-29

Historical detached bench report. Subsequent direct-USB vehicle tests are paused
because of the unresolved [power instability](POWER_STABILITY_STATUS.md).

## Scope and status

Bench tools only, unqualified for vehicle control. Device was confirmed offroad;
user confirmed detached from vehicle. Neither onroad parameters, modeld nor
control outputs were changed. Only road/wide cameras were enabled, with
`DISABLE_DRIVER=1`, bounded camerad lifetime and no image files saved/uploaded.
Camera producer is stopped before mapped buffers are copied to owned bytes.
This is snapshot transport, not continuous capture, preprocessing or inference.

Temporary authenticated client modules/key are removed after each run. The
bounded high-core hold restores the original online state. These cleanups do
not revert bench code. Production protocol default remains **4 MiB**: the
8 MiB override exists only inside explicitly opted-in benchmark processes.

## Earlier synthetic inference result

`artifacts/usb-auth-coreml-20260929-03.json`: physical USB, authentication and
Big-model Core ML CPU_AND_NE, client affinity 4–7. Following 132 recurrent
warmup frames, 100 measured synthetic requests:

- Request RTT mean 41.110 ms, p99 44.406 ms, max 44.673 ms.
- Scheduled release to result mean 41.296 ms, max 44.849 ms; zero >50 ms.
- Inference mean 29.291 ms.

This excludes camera acquisition and calibrated preprocessing. It is not a
camera-to-output E2E measurement or a long-duration qualification. Earlier
failed harness runs exhausted a transport-server timeout carried between frames;
the bench server now starts a fresh deadline for every request.

## Real camera snapshots and size mismatch

Both road camera buffers: 1928 × 1208, stride 2048, UV offset 2490368,
**4804608 bytes** each, including padding. Initial probes correctly rejected
this size against the 4 MiB cap; their cameras/settings were cleaned up.
No frame-ID mismatch was observed in the diagnostic retry.

`usb-camera-snapshot-20260929-03.json` verified two complete snapshots over
1 MiB authenticated chunks, using cumulative SHA-256 acknowledgements.
Narrow 145.118 ms; wide 131.537 ms. Their frame IDs differ (77/78), so these
are not asserted to be a synchronized model-input pair.

## 8 MiB and copy-reduction comparisons

All variants reuse the same owned snapshots within a run. Single mode sends
one application message per camera, not a single USB packet. TCP segmentation
still occurs. Each acknowledgement verifies all bytes received so far.

Initial ABBA (`usb-camera-8mib-abba-20260929-01.json`, four observations per
variant): chunked ~127.469 ms vs contiguous single ~130.138 ms. A larger cap
alone did not show a clear improvement.

Follow-up `usb-camera-8mib-iovec-20260929-01.json` uses
chunked/single/iovec/iovec/single/chunked order, both cameras per pass:

| Mode (4 observations each) | Total + acknowledgement | Prepare | Send wall | Send thread CPU | Receive |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 MiB chunks, stop-and-wait | 131.524 ms | 13.457 | 93.877 | 76.897 | 17.756 |
| 8 MiB cap, contiguous single | 119.530 ms | 18.466 | 86.356 | 78.540 | 8.857 |
| 8 MiB cap, scatter/gather single | 106.759 ms | 5.986 | 87.687 | 80.192 | 7.227 |

Scatter/gather reuses existing `authenticate_payload_parts` and
`send_message_parts`: removes large concatenations without removing HMAC,
CRC, deadlines or acknowledgement validation. This is **not end-to-end
zero-copy**; there is still an owned snapshot copy, kernel work and receive-side
allocation/copy. Thread CPU includes kernel execution, not just Python.

Small samples, warm-cache/order effects and no concurrent local-model load:
these numbers are promising microbenchmark observations, not stable p99 claims.
The observed scatter/gather total is ~18.8% below chunked and ~10.7% below
contiguous single in this run. Raw send alone (~88 ms) remains too slow for
20 Hz camera pairs, before any inference. Merely raising the cap is insufficient.

## Verification and next steps

Every successful snapshot/variant matched cumulative full-image SHA-256.
Last run: peak CPU temperature 48.9°C, original CPU state restored, temporary
files removed, no camerad remained. New tests cover the unchanged 4 MiB default,
explicit 8 MiB boundary, rejection before oversized receive allocation, and
byte-identical authenticated scatter/gather framing for a 4804608-byte input.

Next: remove non-image padding and/or perform the exact calibrated model
transform on 3X before transfer, validate against reference tensors, measure
CPU/local-model interference, then test continuous shadow operation with
disconnects, stale results and thermal soak. No runtime deployment or road-test
approval follows from this snapshot benchmark.
