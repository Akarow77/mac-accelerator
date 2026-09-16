"""Compare completed CPU/GPU NV12 preprocessing, retaining device-specific evidence."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from camera_preprocess import GatherPlan, NV12Layout


def hardware_signature():
  def sysctl(name):
    if platform.system() != 'Darwin':
      return 'unavailable'
    return subprocess.check_output(['/usr/sbin/sysctl', '-n', name], text=True).strip()
  root = Path(__file__).resolve().parent
  code = hashlib.sha256(b''.join((root / path).read_bytes() for path in
                       ('camera_preprocess.py', 'backends/tinygrad_warp.py', 'backends/native_gather.py', 'native/gather.c'))).hexdigest()
  native = root / 'artifacts/libcamera_gather.dylib'
  installed = importlib.util.find_spec('tinygrad') is not None
  revision = None
  if installed:
    direct = importlib.metadata.distribution('tinygrad').read_text('direct_url.json')
    revision = json.loads(direct).get('vcs_info', {}).get('commit_id') if direct else None
  return {'chip': sysctl('machdep.cpu.brand_string'), 'memoryBytes': sysctl('hw.memsize'),
          'machine': platform.machine(), 'macOS': platform.mac_ver()[0],
          'numpy': np.__version__, 'python': platform.python_version(), 'preprocessCodeSha256': code,
          'nativeLibrarySha256': hashlib.sha256(native.read_bytes()).hexdigest() if native.is_file() else None,
          'tinygrad': importlib.metadata.version('tinygrad') if installed else None, 'tinygradRevision': revision,
          # tinygrad fills DEV=METAL lazily on Apple Silicon; normalize the
          # uninitialized equivalent so profile checks work before its import.
          'tinygradEnvironment': {key: os.environ.get(key, 'METAL' if key == 'DEV' else None)
                                 for key in ('DEV', 'JIT', 'BEAM', 'NOOPT', 'FLOAT16', 'SPLIT_REDUCEOP')}}


def summary(values):
  return {'meanMs': float(np.mean(values)), 'p50Ms': float(np.percentile(values, 50)),
          'p99Ms': float(np.percentile(values, 99)), 'maxMs': float(max(values))}


def select_candidate(candidates, budget_ms):
  eligible = [name for name, value in candidates.items()
              if value.get('verified') is True and value.get('samples', 0) >= 40
              and np.isfinite(value.get('p99Ms', np.inf)) and np.isfinite(value.get('maxMs', np.inf))
              and value['maxMs'] <= budget_ms]
  return min(eligible, key=lambda name: candidates[name]['p99Ms']) if eligible else None


def validate_profile(profile):
  if profile.get('hardware') != hardware_signature():
    raise ValueError('profile hardware/software mismatch; measure again on this Mac')
  if profile.get('mapReferenceDevice') != 'METAL':
    raise ValueError('unqualified NumPy-only map; tinygrad reference-map profile required')
  expected = select_candidate(profile['candidates'], profile['budgetMs'])
  if expected is None or expected != profile.get('selectedPreprocessor'):
    raise ValueError('no passing measured preprocessor')
  if expected not in ('cpu-gather-cached', 'native-cpu-gather', 'tinygrad-metal-gather'):
    raise ValueError('unsupported selected preprocessor')
  return expected


def benchmark(runs=120, include_metal=False):
  if runs < 40:
    raise ValueError('at least 40 samples required')
  layout = NV12Layout(1928, 1208, 2048, 1216, 608)
  transforms = (np.array([[3.1, .03, 130], [.02, 3.9, 55], [.00001, -.00003, 1]], dtype=np.float32),
                np.array([[2.8, -.02, 170], [.015, 3.5, 100], [-.00002, .00001, 1]], dtype=np.float32))
  start = time.perf_counter_ns()
  plans = [GatherPlan(layout, transform, reference_device='METAL' if include_metal else None) for transform in transforms]
  plan_ms = (time.perf_counter_ns() - start) / 1e6
  rng = np.random.default_rng(415)
  frames = [(rng.integers(0, 256, layout.nbytes, dtype=np.uint8).tobytes(),
             rng.integers(0, 256, layout.nbytes, dtype=np.uint8).tobytes()) for _ in range(4)]
  references = [np.stack([plan.run(frame) for plan, frame in zip(plans, pair)]).tobytes() for pair in frames]
  output = np.empty((2, 6, 128, 256), dtype=np.uint8)

  def cpu(pair):
    for index in range(2):
      plans[index].run(pair[index], output[index])
    return output.tobytes()

  def uncached(pair):
    for index in range(2):
      GatherPlan(layout, transforms[index], reference_device='METAL' if include_metal else None).run(pair[index], output[index])
    return output.tobytes()

  runners = {'cpu-gather-cached': cpu}
  if not include_metal:
    runners['cpu-gather-rebuild'] = uncached
  errors = {}
  if (Path(__file__).resolve().parent / 'artifacts/libcamera_gather.dylib').is_file():
    try:
      from backends.native_gather import NativeGather
      native = [NativeGather(plan) for plan in plans]
      def native_cpu(pair):
        for index in range(2):
          native[index].run(pair[index], output[index])
        return output.tobytes()
      runners['native-cpu-gather'] = native_cpu
    except Exception as error:
      errors['native-cpu-gather'] = f'{type(error).__name__}: {error}'
  if include_metal:
    try:
      from backends.tinygrad_warp import TinygradGather
      gpu = [TinygradGather(plan) for plan in plans]
      def metal(pair):
        return np.stack([gpu[index].run(pair[index]) for index in range(2)]).tobytes()
      runners['tinygrad-metal-gather'] = metal
    except Exception as error:
      errors['tinygrad-metal-gather'] = f'{type(error).__name__}: {error}'
  candidates = {}
  for name, run in runners.items():
    values = []
    try:
      for i in range(8 + runs):
        index = i % len(frames)
        start = time.perf_counter_ns()
        result = run(frames[index])
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if result != references[index]:
          raise ValueError(f'byte mismatch at sample {i}; candidate rejected')
        if i >= 8:
          values.append(elapsed)
      candidates[name] = {**summary(values), 'samples': len(values), 'verified': True}
    except Exception as error:
      candidates[name] = {'verified': False, 'samples': len(values), 'error': f'{type(error).__name__}: {error}'}
  candidates.update({name: {'verified': False, 'samples': 0, 'error': error} for name, error in errors.items()})
  return {'hardware': hardware_signature(), 'candidates': candidates, 'budgetMs': 5.,
          'selectedPreprocessor': select_candidate(candidates, 5.) if include_metal else None, 'planBuildMs': plan_ms,
          'mapReferenceDevice': 'METAL' if include_metal else None,
          'geometryScope': 'Metal-reference lookup; QCOM equivalence and live calibration not qualified',
          'inputBytesPerPair': 2 * layout.nbytes, 'outputBytesPerPair': len(references[0]),
          'scope': 'owned synthetic NV12 pair -> packed model bytes; no capture/USB/model inference',
          'roadReady': False, 'otherChipQualified': False}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runs', type=int, default=120)
  parser.add_argument('--include-metal', action='store_true')
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    parser.error('output exists; choose a new profile path')
  report = benchmark(args.runs, args.include_metal)
  with args.output.open('x') as output:
    json.dump(report, output, indent=2, allow_nan=False)
    output.write('\n')
  print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
  main()
