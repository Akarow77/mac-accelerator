# Remote inference protocol v2

This is the implemented bench protocol between a comma 3X and an Apple Silicon
Mac. It is not approved for on-road control.

## Split point

The interface accepts two contiguous `uint8[6,128,256]` warped image tensors.
This standalone project supplies synthetic inputs only; camera acquisition,
calibration and warp would belong to a separately reviewed external producer.
The Mac owns recurrent image, desire and feature queues and runs the driving policy.

Keeping the warp on the 3X reduces the 20 Hz payload from about 149.4 MB/s of
padded NV12 data to 393,216 bytes per inference before compression, or about
7.9 MB/s at 20 Hz. The Core ML Big Model response preserves its native 18,452
float16 values (about 36.9 KB); the 3X converts these to float32 before parsing.

## Framing and discovery

Every message starts with the 40-byte, network-order `SPMA` header defined in
`transport.py`: protocol version, message type, flags, session ID, frame ID,
capture monotonic timestamp, payload length, and CRC32. Payloads are capped at
4 MiB.

The client begins each connection with a nonce-bearing JSON `HELLO`. The server
returns an identity containing its device type, backend, model checkpoint,
source-model SHA-256, input/output shapes, output slices, frame skip, output length
and dtype, request length, and supported compression modes. The client rejects any identity that differs from
its configured expectations.

The source hash is currently a server declaration computed from `--onnx`; it is
not cryptographically bound to `--model` (the Core ML package). Authentication
does not by itself establish conversion provenance or numerical equivalence.

When a shared key is configured, both sides prove possession with HMAC-SHA256
over canonical handshake JSON. Every inference request and response also carries
an HMAC-SHA256 tag covering the protocol metadata and payload. The key must be at
least 32 bytes and is never stored in the repository.

## Inference request and response

The uncompressed request is exactly 393,264 bytes: two warped image tensors,
eight desire-pulse floats, two traffic-convention floats, and two action-delay
floats. `FLAG_ZSTD_REQUEST` losslessly compresses this request with Zstd level 1;
the declared uncompressed size remains fixed. `FLAG_RESET` resets recurrent
state and must be acknowledged by the response.

The response contains server receive, inference-start, and inference-end
monotonic timestamps followed by the model's declared little-endian output. The
header echoes the request session, frame, capture timestamp, and accepted flags.
Monotonic clocks on different machines are not compared; the client measures the
complete request round trip with its own clock.

Only one request may be in flight. Frame IDs must be sequential at the client and
strictly increasing at the server. Duplicate/backward frames are rejected. The
Core ML session uses `frame_skip=4` by default for the 32-step Big Model. Earlier
stride-2 replay reports do not validate the upstream temporal contract.
The synthetic client's transport
qualification does not establish camera context, recurrent accuracy or road readiness.

## Readiness and failure behavior

Loading has a separate cold-start timeout. The accelerator becomes ready only
after a configured number of consecutive frames finish within the active
deadline. Once active, a bad identity, malformed length, checksum or HMAC error,
stale response, non-finite output, disconnect, or deadline miss latches the
client into the failed state until an explicit reset. There is no ignition lifecycle
or vehicle parameter integration in this project.

The standalone client runs at normal scheduling priority. There is no FIFO request,
camera snapshot hook or local vehicle-model process here. On macOS the server
requests user-interactive QoS; that is not a hard-real-time guarantee. USB and
Mac-only tests do not establish a sustained end-to-end deadline.

Receive timeouts use one absolute deadline across header, payload, and fragmented
reads. The synthetic client defaults to a 50ms active round-trip deadline, with
separate startup qualification settings. `smoke_local.py` deliberately uses 500ms
to test plumbing and reports 50ms period misses separately; passing that smoke test
is not 50ms qualification. No live camera age or vehicle UI heartbeat is monitored
by this independent app. These are diagnostic, not safety, limits.

## Historical development stages

1. Loopback server/client on the Mac with synthetic warped inputs. Completed.
2. Recorded-route Core ML versus PyTorch validation. Completed for a short sample.
3. Wired comma 3X discovery and synthetic USB-NCM inference. Completed off-road.
4. Live camera-warp shadow integration. Failed timing; removed from production
   modeld in the original project, and not included in this standalone repository.
5. Sustained thermal and full-route qualification. Not passed.
6. Any control-path proposal requires a separate safety design and review.
