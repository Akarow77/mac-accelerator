"""Synthetic, observation-only client for the physical authenticated USB bench.

No Params writes, camera subscriptions, vehicle messages or runtime installation.
150ms is a diagnostic abort limit, NOT the 50ms timing target or road acceptance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from accelerator_client import AcceleratorClient
from accelerator_protocol import read_auth_key
from model_contract import BIG_MODEL_FRAME_SKIP
from transport import POLICY_INPUTS, WARPED_BYTES


def summarize(values):
  ordered = sorted(values)
  return {'samples': len(values), 'meanMs': statistics.mean(values),
          'p99Ms': ordered[(99 * len(values) + 99) // 100 - 1], 'maxMs': max(values),
          'over50ms': sum(value > 50 for value in values)}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--host', required=True)
  parser.add_argument('--port', required=True, type=int)
  parser.add_argument('--key', required=True, type=Path)
  parser.add_argument('--backend', required=True, choices=('TRANSPORT_PROBE', 'COREML_ANE'))
  parser.add_argument('--sha', required=True)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--warmup', required=True, type=int, choices=(20, 132))
  parser.add_argument('--cpu-set', required=True, choices=('default', '4-7'))
  args = parser.parse_args()
  if args.cpu_set == '4-7':
    if not {4, 5, 6, 7}.issubset(os.sched_getaffinity(0)):
      raise RuntimeError('high cores unavailable')
    os.sched_setaffinity(0, {4, 5, 6, 7})
  pixels = np.random.default_rng(42).integers(0, 256, WARPED_BYTES, dtype=np.uint8).tobytes()
  policy = POLICY_INPUTS.pack(*([0.] * 8 + [1., 0., 0., 0.]))
  client = AcceleratorClient(args.host, args.port, auth_key=read_auth_key(args.key),
                             expected_backend=args.backend, expected_model_sha256=args.sha,
                             expected_checkpoint=args.checkpoint, expected_output_floats=18452,
                             deadline_ms=150., qualification_timeout_ms=150.)
  rows = []
  digest = hashlib.sha256()
  names = ('round_trip_ms', 'inference_ms', 'prepare_ms', 'send_ms', 'receive_ms',
           'validate_ms', 'server_prepare_ms')
  try:
    identity = client.connect()
    if args.backend == 'COREML_ANE':
      shapes = {'img': [1,12,128,256], 'big_img': [1,12,128,256], 'desire_pulse': [1,33,8],
                'traffic_convention': [1,2], 'action_t': [1,2], 'features_buffer': [1,32,32,512]}
      if identity.frame_skip != BIG_MODEL_FRAME_SKIP or identity.input_shapes != shapes:
        raise ValueError('model temporal/input contract mismatch')
      if args.warmup != 132:
        raise ValueError('model test requires full-history warmup')
    base = time.monotonic_ns()
    for frame in range(args.warmup + 100):
      if Path('/data/params/d/IsOffroad').read_text().strip() != '1':
        raise RuntimeError('left offroad')
      if args.cpu_set == '4-7' and set(os.sched_getaffinity(0)) != {4, 5, 6, 7}:
        raise RuntimeError('requested CPU affinity changed')
      target = base + frame * 50_000_000
      time.sleep(max(0, (target - time.monotonic_ns()) / 1e9))
      wake = time.monotonic_ns()
      result = client.infer(pixels, policy, frame_id=frame, capture_ns=target, reset=frame == 0,
                            deadline_ns=target + 150_000_000)
      end = time.monotonic_ns()
      if frame >= args.warmup:
        row = {name: getattr(result, name) for name in names}
        row.update(scheduled_to_result_ms=(end-target)/1e6, wake_lateness_ms=(wake-target)/1e6)
        rows.append(row)
        digest.update(result.output)
  finally:
    client.close()
  print(json.dumps({'status': 'measured-not-qualified', 'roadReady': False,
                    'affinity': sorted(os.sched_getaffinity(0)), 'warmup': args.warmup,
                    'abortLimitMs': 150, 'timingTargetMs': 50, 'outputDigest': digest.hexdigest(),
                    'stages': {name: summarize([row[name] for row in rows]) for name in rows[0]},
                    'rows': rows}, allow_nan=False))


if __name__ == '__main__':
  main()
