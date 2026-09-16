# Observation-only shadow: what works, what remains

See the [verification record](SHADOW_VALIDATION.md) for completed tests and limits.

## Decision

M2 compute is not established as the primary bottleneck. Historical Mac-only
Core ML timing passed 6,000 paced inference samples, while live copy/transport
failed timing. Local compute, localhost transport, wired transport and live camera
age are different measurements. Faster average inference alone cannot fix a
synchronous QCOM readback that blocks the local driving model.

The independent project now implements **synthetic / recorded-warp shadow replay**.
It does not implement live 3X camera capture. No firmware, Params or modeld changes
are needed or deployed. No remote model outputs enter a control path.

## Implemented boundaries

- Explicit run only; no auto-start, reconnect or hidden retry after failure.
- Single in-flight request; no unbounded queue, record looping or silent frame drops.
- Exact Big Model shape and frame-skip contract, expected source SHA and mandatory
  authentication. The source SHA still does not prove converted-package provenance.
- Strict frame sequence and monotonic playback timestamps. A gap or invalid clock
  latches failure, rather than carrying recurrent state across missing frames.
- 66 consecutive frames before context is reported. This is context length, not
  proof of numerical equivalence or model quality.
- 50ms age target counted separately from the 150ms stale-stop diagnostic limit.
  A 60ms frame is logged as a miss, not a timing pass. Repeated overload accumulates
  playback lateness and stops at the stale limit; no queue catch-up is hidden.
- Absolute send/receive deadlines across partial socket operations. These bound
  socket waiting, not every possible OS scheduler stall.
- Immutable returned FP16-to-FP32 result bytes: a later inference cannot rewrite
  an earlier frame's retained output.
- Non-finite policy values rejected before server recurrent buffers change.
- Private, exclusive-create, size-capped logs with space reserved for terminal
  failure. Disk-full errors close the client; no claim that a failed disk can log.
- Logs contain timing, context and output hashes, not images or control commands.
  `roadReady` and `sustainedQualified` are never asserted by this tool.

The source and observer currently share a replay process on the Mac, separate from
the inference server. A slow log write can delay replay, which will be counted or
stopped; it cannot block a vehicle model because no vehicle integration exists.

## Live capture design gate — not implemented

Do not reinsert `tensor.numpy()` into modeld, even with a timer checked afterward.
The timer cannot undo its first stall. Merely moving a function to another process
does not isolate shared GPU, memory bandwidth, camera-buffer ownership or power.

A future candidate is an independent, normal-priority camera stream consumer,
with independently owned bounded buffers and asynchronous preprocessing/transfer.
Actual VisionIPC buffer lifetime, reference-release behavior, stream pairing and
warp/calibration fidelity must first be established from the device implementation.
The candidate must not retain a camera buffer while waiting for a network response.
Raw camera transport or separate QCOM warp work may cost more bandwidth/compute;
neither path is assumed to be free or fast enough.

Before any powered bench capture trial:

1. Verify stable official 3X baseline; investigate the earlier unexplained reset
   separately. A software extraction does not settle its cause.
2. Measure observer off/on local model frame times, camera drops, temperatures and
   process scheduling with identical inputs. No FIFO priority or Params writes.
3. Prove buffer ownership and camera pair/calibration correctness, including camera
   restart and frame wrap/reset cases. Never compare clocks from separate machines
   by directly subtracting their monotonic timestamps.
4. Kill/disconnect each observer component, saturate its private queue, and fill its
   log quota. Verify the local model continues unaffected, not merely that the Mac
   worker exits. Device disconnection must not arm a future automatic retry.
5. Only then measure sustained real image payloads and recurrent numerical fidelity.
   UI green/service-ready is not an acceptance criterion for driving.

No critical-bug-free claim is made. The retained pickle metadata still requires
trusted files, conversion fidelity remains incompletely qualified, shared-resource
isolation has not been demonstrated, and 3X live shadow remains disabled.
