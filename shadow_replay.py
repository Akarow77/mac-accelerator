#!/usr/bin/env python3
"""Observation-only replay. No live capture, vehicle access, automatic retry or control output."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from accelerator_client import AcceleratorClient
from accelerator_protocol import read_auth_key
from macos_performance import configure_user_interactive_qos
from model_contract import BIG_MODEL_CONTEXT_FRAMES, BIG_MODEL_FRAME_SKIP
from transport import POLICY_INPUTS, WARPED_BYTES, WARPED_SHAPE


class BoundedLog:
  def __init__(self, path: Path, limit: int = 16 * 1024 * 1024):
    if limit < 8192:
      raise ValueError('log limit must allow a terminal record')
    self.limit, self.written = limit, 0
    self.file = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb')

  def write(self, row: dict, *, terminal: bool = False) -> None:
    data = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode()
    budget = self.limit if terminal else self.limit - 4096
    if self.written + len(data) > budget:
      raise RuntimeError('shadow log size limit reached')
    self.file.write(data)
    self.file.flush()
    self.written += len(data)

  def close(self):
    self.file.close()


class ShadowRecorder:
  """One in-flight request, strict source sequence, latched failures, no output publisher.

  capture_ns is a timestamp in THIS process host's monotonic clock. Replay uses
  scheduled local playback time, never a foreign device's monotonic timestamp.
  """
  def __init__(self, client, log, *, target_ms=50., max_age_ms=150.,
               context_frames=BIG_MODEL_CONTEXT_FRAMES, clock=time.monotonic_ns):
    if (not all(math.isfinite(x) and x > 0 for x in (target_ms, max_age_ms))
        or max_age_ms < target_ms or context_frames < 1):
      raise ValueError('invalid timing/context limits')
    self.client, self.log, self.clock = client, log, clock
    self.target_ms, self.max_age_ms, self.context_frames = target_ms, max_age_ms, context_frames
    self.next_frame = 0
    self.last_capture = -1
    self.failed = False
    self.eligible = self.misses = 0

  def fail(self, error):
    if self.failed:
      return
    self.failed = True
    self.client.close()
    self.log.write({'event': 'failed', 'frame': self.next_frame,
                    'reason': str(error)[:1000], 'roadReady': False}, terminal=True)

  def process(self, frame_id: int, capture_ns: int, pixels: bytes, policy: bytes, *, preprocessing=None, wake_lateness_ms=None):
    if self.failed:
      raise RuntimeError('shadow failure latched; start a new explicit run')
    try:
      now = self.clock()
      if frame_id != self.next_frame:
        raise ValueError('source gap, duplicate or out-of-order frame')
      if capture_ns <= self.last_capture or capture_ns > now:
        raise ValueError('invalid source clock or backward timestamp')
      deadline = capture_ns + int(self.max_age_ms * 1e6)
      if now >= deadline:
        raise TimeoutError('source frame already stale')
      if len(pixels) != WARPED_BYTES or len(policy) != POLICY_INPUTS.size:
        raise ValueError('invalid source dimensions')
      if not all(math.isfinite(value) for value in POLICY_INPUTS.unpack(policy)):
        raise ValueError('non-finite policy input')
      result = self.client.infer(pixels, policy, frame_id=frame_id, capture_ns=capture_ns,
                                 reset=frame_id == 0, deadline_ns=deadline)
      completed = self.clock()
      if completed >= deadline:
        raise TimeoutError('output became stale')
      values = np.frombuffer(result.output, dtype='<f4')
      if values.size != 18452 or not np.isfinite(values).all():
        raise ValueError('invalid shadow output')
      age_ms = (completed - capture_ns) / 1e6
      client_stages = {name: getattr(result, name) for name in ('prepare_ms', 'send_ms', 'receive_ms', 'validate_ms')}
      if any(not math.isfinite(value) or value < 0 for value in client_stages.values()):
        raise ValueError('invalid client stage timing')
      context = frame_id + 1 >= self.context_frames
      within = age_ms <= self.target_ms
      self.log.write({'event': 'frame', 'frame': frame_id, 'contextReady': context,
                      'scheduledPlaybackToOutputMs': age_ms,
                      'roundTripMs': result.round_trip_ms, 'inferenceMs': result.inference_ms,
                      'clientStagesMs': client_stages,
                      'serverPrepareMs': result.server_prepare_ms,
                      # Cross-host clock offsets cancel for each duration, but
                      # this residual also contains CPU/queue work, NOT just wire time.
                      'nonInferenceRoundTripMs': result.round_trip_ms - result.inference_ms,
                      'withinTimingTarget': within, 'roadReady': False,
                      'outputSha256': hashlib.sha256(result.output).hexdigest(), 'preprocessing': preprocessing,
                      'sourceWakeLatenessMs': wake_lateness_ms})
      self.next_frame += 1
      self.last_capture = capture_ns
      self.eligible += int(context)
      self.misses += int(context and not within)
    except Exception as error:
      self.fail(error)
      raise


def validate_identity(identity):
  expected = {'img': [1, 12, 128, 256], 'big_img': [1, 12, 128, 256],
              'desire_pulse': [1, 33, 8], 'traffic_convention': [1, 2],
              'action_t': [1, 2], 'features_buffer': [1, 32, 32, 512]}
  if (identity.frame_skip != BIG_MODEL_FRAME_SKIP or identity.input_shapes != expected
      or identity.output_shapes != {'outputs': [1, 18452]}):
    raise ValueError('shadow Big Model temporal/shape contract mismatch')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument('--synthetic', action='store_true')
  source.add_argument('--synthetic-nv12', action='store_true', help='include full owned NV12 pair preparation on this host')
  source.add_argument('--warps', type=Path, help='uint8 .npy [frames,2,6,128,256], allow_pickle=False')
  parser.add_argument('--policy', type=Path, help='matching float32 .npy [frames,12], required for recorded warps')
  parser.add_argument('--preprocess-profile', type=Path, help='same-hardware measured profile for --synthetic-nv12')
  parser.add_argument('--frames', type=int, default=200)
  parser.add_argument('--host', default='::1')
  parser.add_argument('--port', type=int, default=8066)
  parser.add_argument('--auth-key-file', type=Path, required=True)
  parser.add_argument('--expected-model-sha256', required=True)
  parser.add_argument('--log', type=Path, required=True, help='new private JSONL file; never overwritten')
  parser.add_argument('--max-log-bytes', type=int, default=16 * 1024 * 1024)
  args = parser.parse_args()
  if not 1 <= args.frames <= 72000 or bool(args.policy) != bool(args.warps):
    parser.error('frames must be 1..72000; --warps and --policy must be supplied together')
  if len(args.expected_model_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.expected_model_sha256):
    parser.error('expected SHA-256 must be 64 lowercase hexadecimal characters')
  if args.preprocess_profile and not args.synthetic_nv12:
    parser.error('--preprocess-profile requires --synthetic-nv12')
  if args.synthetic_nv12 and not args.preprocess_profile:
    parser.error('--synthetic-nv12 requires a verified --preprocess-profile from this Mac')
  warps = policy_rows = None
  if args.warps:
    warps = np.load(args.warps, mmap_mode='r', allow_pickle=False)
    policy_rows = np.load(args.policy, mmap_mode='r', allow_pickle=False)
    if warps.dtype != np.uint8 or warps.shape != (len(warps), *WARPED_SHAPE) or len(warps) < args.frames:
      parser.error('invalid warp shape/type/frame count; recordings are never looped')
    if policy_rows.dtype != np.dtype('<f4') or policy_rows.shape != (len(warps), 12):
      parser.error('policy must be matching float32 [frames,12]')
  key = read_auth_key(args.auth_key_file)
  if key is None:
    parser.error('authentication key required')
  log = BoundedLog(args.log, args.max_log_bytes)
  client = AcceleratorClient(args.host, args.port, auth_key=key, deadline_ms=150., qualification_timeout_ms=150.,
                             expected_backend='COREML_ANE', expected_output_floats=18452,
                             expected_model_sha256=args.expected_model_sha256)
  recorder = ShadowRecorder(client, log)
  gc_was_enabled = gc.isenabled()
  try:
    qos = configure_user_interactive_qos()
    preprocessor = transforms = nv12_frames = None
    preprocessing_profile = None
    if args.synthetic_nv12:
      from camera_preprocess import NV12Layout, OwnedCameraFrame, PairPreprocessor
      backend = 'cpu'
      reference_device = None
      if args.preprocess_profile:
        from profile_preprocess import validate_profile
        with args.preprocess_profile.open() as source_file:
          profile = json.load(source_file)
          choice = validate_profile(profile)
          reference_device = profile.get('mapReferenceDevice')
          preprocessing_profile = {'selectedPreprocessor': choice, 'hardware': profile['hardware']}
        backend = {'tinygrad-metal-gather': 'tinygrad-metal', 'native-cpu-gather': 'native-cpu',
                   'cpu-gather-cached': 'cpu'}[choice]
      layout = NV12Layout(1928, 1208, 2048, 1216, 608)
      preprocessor = PairPreprocessor(layout, layout, backend=backend, reference_device=reference_device)
      transforms = (np.array([[3.1, .03, 130], [.02, 3.9, 55], [.00001, -.00003, 1]], dtype=np.float32),
                    np.array([[2.8, -.02, 170], [.015, 3.5, 100], [-.00002, .00001, 1]], dtype=np.float32))
      preprocessor.warmup(transforms)
      rng = np.random.default_rng(24)
      nv12_frames = [tuple(rng.integers(0, 256, layout.nbytes, dtype=np.uint8).tobytes() for _ in range(2)) for _ in range(4)]
    source_name = 'synthetic-nv12' if args.synthetic_nv12 else 'synthetic' if args.synthetic else 'recorded-warps'
    log.write({'event': 'start', 'source': source_name,
               'sourceQoS': 'user-interactive' if qos else 'default',
               'preprocessingProfile': preprocessing_profile,
               'clock': 'local scheduled playback; NOT live camera EOF', 'targetMs': 50,
               'staleStopMs': 150, 'contextFrames': BIG_MODEL_CONTEXT_FRAMES, 'roadReady': False})
    validate_identity(client.connect())
    synthetic_pixels = np.random.default_rng(0).integers(0, 256, WARPED_SHAPE, dtype=np.uint8).tobytes()
    synthetic_policy = POLICY_INPUTS.pack(*([0.] * 8), 1., 0., .1, .1)
    # Setup/compilation/allocation is complete. Refcounted per-frame temporaries
    # remain bounded; don't schedule cyclic collection in the measured loop.
    gc.collect()
    gc.disable()
    origin = time.monotonic_ns()
    for frame in range(args.frames):
      scheduled = origin + frame * 50_000_000
      remaining = (scheduled - time.monotonic_ns()) / 1e9
      if remaining > 0:
        time.sleep(remaining)
      wake_lateness = max(0., (time.monotonic_ns() - scheduled) / 1e6)
      pixels = synthetic_pixels if warps is None else warps[frame].tobytes(order='C')
      preprocessing = None
      if preprocessor is not None:
        pair = nv12_frames[frame % len(nv12_frames)]
        pixels, preprocessing = preprocessor.prepare(OwnedCameraFrame(frame, scheduled, pair[0]),
                                                     OwnedCameraFrame(frame, scheduled, pair[1]), transforms)
      policy = synthetic_policy if policy_rows is None else policy_rows[frame].tobytes(order='C')
      recorder.process(frame, scheduled, pixels, policy, preprocessing=preprocessing, wake_lateness_ms=wake_lateness)
    log.write({'event': 'complete', 'frames': recorder.next_frame, 'contextFramesMeasured': recorder.eligible,
               'targetMisses': recorder.misses, 'contextEstablished': recorder.eligible > 0,
               'sampleTimingTargetPass': recorder.eligible > 0 and recorder.misses == 0,
               'sustainedQualified': False, 'roadReady': False}, terminal=True)
    print(f'Shadow replay completed: frames={recorder.next_frame}, contextSamples={recorder.eligible}, '
          f'targetMisses={recorder.misses}. No live capture or control outputs.')
  except KeyboardInterrupt as error:
    recorder.fail(error)
    raise SystemExit(130) from None
  except Exception as error:
    recorder.fail(error)
    print(f'Shadow stopped: {type(error).__name__}: {error}', file=sys.stderr)
    raise SystemExit(1) from None
  finally:
    if gc_was_enabled:
      gc.enable()
    client.close()
    log.close()


if __name__ == '__main__':
  main()
