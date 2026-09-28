"""Offline only: compare shifted versus circular recurrent history, no model/device.

Read history BEFORE appending the current feature, exactly as the policy session
does. This does not modify the server or claim a GPU/ANE zero-copy interface.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np


class CircularHistory:
  def __init__(self, steps, stride, shape):
    if any(type(value) is not int or value < 1 for value in (steps, stride, *shape)):
      raise ValueError('history dimensions and stride must be positive integers')
    if not shape:
      raise ValueError('feature shape must not be empty')
    self.stride = stride
    self.storage = np.zeros((steps * stride, *shape), dtype=np.float32)
    self.base = np.arange(steps, dtype=np.intp) * stride
    self.indices = np.empty_like(self.base)
    self.head = 0

  def read_into(self, output):
    if (not isinstance(output, np.ndarray) or output.shape != (len(self.base), *self.storage.shape[1:])
        or output.dtype != np.float32 or not output.flags.writeable):
      raise ValueError('output must be writable float32 with the exact history shape')
    if np.shares_memory(output, self.storage):
      raise ValueError('output must not alias history storage')
    np.add(self.base, self.head, out=self.indices)
    np.remainder(self.indices, len(self.storage), out=self.indices)
    # No overlapping output; all indices explicitly reduced into valid range.
    # wrap avoids np.take's buffering associated with mode='raise'.
    np.take(self.storage, self.indices, axis=0, out=output, mode='wrap')

  def append(self, feature):
    if (not isinstance(feature, np.ndarray) or feature.shape != self.storage.shape[1:]
        or feature.dtype != np.float32):
      raise ValueError('feature must be float32 with the exact feature shape')
    if np.shares_memory(feature, self.storage):
      raise ValueError('feature must not alias history storage')
    np.copyto(self.storage[self.head], feature)
    self.head = (self.head + 1) % len(self.storage)

  def reset(self):
    self.storage.fill(0)
    self.head = 0


def verify():
  rng = np.random.default_rng(17)
  checks = 0
  for stride in (1, 2, 4):
    ring = CircularHistory(32, stride, (2, 3))
    legacy = np.zeros_like(ring.storage)
    output = np.empty((32, 2, 3), dtype=np.float32)
    for frame in range(400):
      if frame == 277:
        ring.reset()
        legacy.fill(0)
      ring.read_into(output)
      np.testing.assert_array_equal(output, legacy[::stride])
      retained = output.copy()
      feature = rng.standard_normal((2, 3)).astype(np.float32)
      legacy[:-1] = legacy[1:]
      legacy[-1] = feature
      ring.append(feature)
      np.testing.assert_array_equal(output, retained)
      checks += 1
  return checks


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  report = {'scope': 'CPU recurrent-buffer maintenance only; no model, network or camera',
            'roadReady': False, 'serverChanged': False}
  with args.output.open('x') as handle:
    try:
      report['exactSequenceChecks'] = verify()
      rng = np.random.default_rng(23)
      features = rng.standard_normal((256, 32, 512)).astype(np.float32)
      results = []
      for method in ('shift', 'ring', 'ring', 'shift'):
        ring = CircularHistory(32, 4, (32, 512))
        legacy = np.zeros_like(ring.storage)
        output = np.empty((32, 32, 512), dtype=np.float32)
        times = []
        for frame in range(512):
          feature = features[frame % len(features)]
          start = time.perf_counter_ns()
          if method == 'shift':
            np.copyto(output, legacy[::4])
            legacy[:-1] = legacy[1:]
            legacy[-1] = feature
          else:
            ring.read_into(output)
            ring.append(feature)
          elapsed = (time.perf_counter_ns() - start) / 1e6
          if frame >= 256:
            times.append(elapsed)
        results.append({'method': method, 'samples': len(times), 'meanMs': float(np.mean(times)),
                        'p99Ms': float(np.percentile(times, 99)), 'maxMs': max(times)})
      report.update(status='measured-not-integrated', abba=results,
                    avoidedShiftBytesPerFrame=127 * 32 * 512 * 4,
                    modelInputGatherBytesPerFrame=32 * 32 * 512 * 4)
    except Exception as error:
      report.update(status='failed', error=f'{type(error).__name__}: {error}')
      raise
    finally:
      json.dump(report, handle, indent=2, allow_nan=False)
      handle.write('\n')
      print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
