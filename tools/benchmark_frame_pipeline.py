"""Mac-only bounded GPU(N+1)/ANE(N) experiment. Never imported by the server.

Two independently compiled submodels; GPU inputs must depend only on images.
Host-visible NumPy handoff is NOT zero-copy. Timelines measure overlapping API
calls, not simultaneous silicon execution. No network, camera, or control output.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import queue
import sys
import threading
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coreml_inference_server import CoreMLPolicySession
from macos_performance import configure_user_interactive_qos
from model_contract import BIG_MODEL_CONTEXT_FRAMES, BIG_MODEL_FRAME_SKIP
from tools.probe_model_partition import graph_groups, plan_summary, summarize
from transport import POLICY_INPUTS, WARPED_SHAPE


def cut_frontier(spec, cut, stem_operations=40):
  operations, vision, _ = graph_groups(spec)
  if not 1 <= stem_operations <= len(vision):
    raise ValueError('stem operation count outside image-only graph')
  selected = vision if cut == 'vision' else set(sorted(vision)[:stem_operations])
  consumers = defaultdict(set)
  for index, (op, _) in enumerate(operations):
    for value in op.inputs.values():
      for arg in value.arguments:
        if arg.HasField('name'):
          consumers[arg.name].add(index)
  function = spec.mlProgram.functions['main']
  outputs = function.block_specializations[function.opset].outputs
  names = [value.name for index in sorted(selected) for value in operations[index][0].outputs
           if consumers[value.name] - selected or value.name in outputs]
  tail_inputs = [value.name for value in function.inputs if consumers[value.name] - selected]
  if not names or any(operations[index][1] - {'img', 'big_img'} for index in selected):
    raise ValueError('cut is not image-only')
  return names, tail_inputs


class FramePipeline:
  """Single producer, ordered consumer; at most two admitted frames in total."""
  def __init__(self, gpu, ane, frontier, tail_inputs, pixels, count, period=0.):
    self.gpu, self.ane, self.frontier, self.tail_inputs = gpu, ane, frontier, tail_inputs
    self.pixels, self.count, self.period = pixels, count, period
    self.ready = queue.Queue(maxsize=1)
    self.slots = threading.BoundedSemaphore(2)
    self.stop = threading.Event()
    self.error = None
    self.rows = []
    self.next_frame = 0
    self.thread = threading.Thread(target=self.produce, name='experimental-gpu-stage', daemon=True)

  def start(self):
    self.origin = time.perf_counter_ns()
    self.thread.start()

  def produce(self):
    configure_user_interactive_qos()
    history = np.zeros((BIG_MODEL_FRAME_SKIP + 1, *WARPED_SHAPE), dtype=np.uint8)
    try:
      for frame in range(self.count):
        while not self.stop.is_set() and not self.slots.acquire(timeout=.1):
          pass
        if self.stop.is_set():
          return
        scheduled = self.origin + int(frame * self.period * 1e9) if self.period else None
        if scheduled is not None and self.stop.wait(max(0., (scheduled - time.perf_counter_ns()) / 1e9)):
          return
        admitted = time.perf_counter_ns()
        history[:-1] = history[1:]
        history[-1] = self.pixels[frame % len(self.pixels)]
        # Owned buffers remain unchanged until the synchronous prediction returns.
        inputs = {name: history[::BIG_MODEL_FRAME_SKIP, camera].reshape(1, 12, 128, 256).astype(np.float32)
                  for camera, name in enumerate(('img', 'big_img'))}
        gpu_start = time.perf_counter_ns()
        result = self.gpu.predict(inputs)
        gpu_end = time.perf_counter_ns()
        # In process mode these timestamps surround the worker's actual predict,
        # excluding IPC. macOS perf_counter uses one system-wide monotonic clock.
        worker_interval = getattr(self.gpu, 'last_interval', None)
        if worker_interval is not None:
          gpu_start, gpu_end = worker_interval
        if set(result) != set(self.frontier) or any(not np.isfinite(v).all() for v in result.values()):
          raise ValueError('invalid GPU output')
        item = (frame, result, {'frame': frame, 'scheduledNs': scheduled, 'admittedNs': admitted,
                                'gpuStartNs': gpu_start, 'gpuEndNs': gpu_end})
        while not self.stop.is_set():
          try:
            self.ready.put(item, timeout=.1)
            break
          except queue.Full:
            pass
    except Exception as error:
      self.error = error
      self.stop.set()

  def predict(self, inputs):
    deadline = time.monotonic() + 15
    while True:
      if self.error is not None:
        raise RuntimeError('GPU stage failed') from self.error
      if time.monotonic() >= deadline:
        raise TimeoutError('GPU stage did not complete')
      try:
        frame, result, row = self.ready.get(timeout=.1)
        break
      except queue.Empty:
        continue
    if frame != self.next_frame:
      raise ValueError('pipeline frame sequence mismatch')
    if not self.period and frame + 1 < self.count and hasattr(self.gpu, 'wait_submitted'):
      self.gpu.wait_submitted(frame + 1)
    state = {name: inputs[name] for name in self.tail_inputs}
    row['aneStartNs'] = time.perf_counter_ns()
    output = self.ane.predict({**state, **result})
    row['aneEndNs'] = time.perf_counter_ns()
    self.rows.append(row)
    self.next_frame += 1
    # Release only after ANE returns, never while its input storage is in use.
    self.slots.release()
    return output

  def close(self):
    self.stop.set()
    self.thread.join(timeout=20)
    if self.thread.is_alive():
      raise RuntimeError('GPU call still running; terminate benchmark process, do not reuse buffers')


class SerialSplit:
  def __init__(self, gpu, ane, tail_inputs):
    self.gpu, self.ane, self.tail_inputs = gpu, ane, tail_inputs

  def predict(self, inputs):
    vision = self.gpu.predict({name: inputs[name] for name in ('img', 'big_img')})
    return self.ane.predict({**{name: inputs[name] for name in self.tail_inputs}, **vision})


def run(model, metadata, output_name, pixels, warmup, runs, *, period=0., pipeline=False):
  session = CoreMLPolicySession(model, metadata, output_name, BIG_MODEL_FRAME_SKIP, output_dtype='<f4')
  policy = POLICY_INPUTS.pack(*([0.] * 8), 1., 0., .1, .1)
  payloads = [item.tobytes() + policy for item in pixels]
  outputs, rows = [], []
  origin = time.perf_counter_ns()
  if pipeline:
    model.start()
    origin = model.origin
  try:
    for frame in range(warmup + runs):
      if period and not pipeline:
        time.sleep(max(0., (origin + int(frame * period * 1e9) - time.perf_counter_ns()) / 1e9))
      start = time.perf_counter_ns()
      result = session.infer(payloads[frame % len(payloads)])
      end = time.perf_counter_ns()
      if frame >= warmup:
        outputs.append(np.frombuffer(result, dtype='<f4'))
        row = dict(model.rows[-1]) if pipeline else {'frame': frame, 'admittedNs': start}
        row['completedNs'] = end
        rows.append(row)
  finally:
    if pipeline:
      model.close()
  intervals = [(b['completedNs'] - a['completedNs']) / 1e6 for a, b in zip(rows, rows[1:])]
  age = [(row['completedNs'] - row['admittedNs']) / 1e6 for row in rows]
  report = {'completedPerSecond': (runs - 1) / ((rows[-1]['completedNs'] - rows[0]['completedNs']) / 1e9),
            'admissionToOutput': summarize(age), 'completionIntervals': summarize(intervals), 'timeline': rows}
  if period:
    report['scheduledToOutput'] = summarize([
      (row['completedNs'] - (origin + int(row['frame'] * period * 1e9))) / 1e6 for row in rows])
  if pipeline:
    overlap = [max(0, min(a['aneEndNs'], b['gpuEndNs']) - max(a['aneStartNs'], b['gpuStartNs'])) / 1e6
               for a, b in zip(rows, rows[1:])]
    report['adjacentApiCallOverlap'] = {'pairs': len(overlap), 'nonzeroPairs': sum(x > 0 for x in overlap),
                                       'meanMs': float(np.mean(overlap)), 'maxMs': max(overlap),
                                       'hardwareConcurrencyVerified': False}
    report['gpuCall'] = summarize([(r['gpuEndNs'] - r['gpuStartNs']) / 1e6 for r in rows])
    report['aneCall'] = summarize([(r['aneEndNs'] - r['aneStartNs']) / 1e6 for r in rows])
  return report, np.stack(outputs)


def main():
  import coremltools as ct
  from coremltools.converters.mil.debugging_utils import extract_submodel
  root = Path(__file__).resolve().parents[1]
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--cut', choices=('stem', 'vision'), default='stem')
  parser.add_argument('--stem-operations', type=int, default=40,
                      help='offline cut search; choose by measured latency, not shared RAM proportions')
  parser.add_argument('--runs', type=int, default=120)
  parser.add_argument('--warmup', type=int, default=BIG_MODEL_CONTEXT_FRAMES)
  parser.add_argument('--gpu-process', action='store_true', help='isolate GPU prediction from the parent Python interpreter')
  args = parser.parse_args()
  if args.runs < 20 or args.warmup < BIG_MODEL_CONTEXT_FRAMES:
    parser.error('require at least 20 samples after 132 context frames')
  report = {'roadReady': False, 'zeroCopyVerified': False, 'numericalQualified': False,
            'scope': 'synthetic Mac-only split model; no camera or network', 'cut': args.cut,
            'frameSkip': BIG_MODEL_FRAME_SKIP, 'warmupFrames': args.warmup, 'maxAdmittedFrames': 2}
  report['gpuProcess'] = args.gpu_process
  report['stemOperations'] = args.stem_operations
  with args.output.open('x') as output:
    try:
      report['qos'] = configure_user_interactive_qos()
      with (root / 'models/big_driving_supercombo_metadata.pkl').open('rb') as source:
        metadata = pickle.load(source)
      path = str(root / 'artifacts/big_driving.mlpackage')
      base = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
      spec = base.get_spec()
      report['modelSpecSha256'] = hashlib.sha256(spec.SerializeToString(deterministic=True)).hexdigest()
      frontier, tail_inputs = cut_frontier(spec, args.cut, args.stem_operations)
      report['frontier'], report['tailInputs'] = frontier, tail_inputs
      output_name = spec.description.output[0].name
      gpu_source = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_GPU, skip_model_load=True)
      # Apple's extractor deep-copies the connected MIL graph recursively. This
      # model exceeds Python's default depth; only this offline process changes it.
      sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
      print('Extracting GPU image-only submodel', flush=True)
      gpu = extract_submodel(gpu_source, outputs=frontier, inputs=['img', 'big_img'])
      print('Extracting ANE dependent submodel', flush=True)
      ane = extract_submodel(base, outputs=[output_name], inputs=tail_inputs + frontier)
      report['gpuPlan'], report['anePlan'] = plan_summary(gpu), plan_summary(ane)
      if (not report['gpuPlan']['preferredDeviceOperations'].get('MLGPUComputeDevice')
          or not report['anePlan']['preferredDeviceOperations'].get('MLNeuralEngineComputeDevice')):
        raise RuntimeError('requested GPU/ANE partition not present in anticipated plans')
      pixels = np.random.default_rng(37).integers(0, 256, (7, *WARPED_SHAPE), dtype=np.uint8)
      report['runs'] = {}
      reference = split_reference = None
      cases = [('baseline-before', base, 0., False),
               ('split-serial', SerialSplit(gpu, ane, tail_inputs), 0., False),
               ('split-pipelined', None, 0., True), ('baseline-after', base, 0., False),
               ('baseline-20hz', base, .05, False),
               ('split-serial-20hz', SerialSplit(gpu, ane, tail_inputs), .05, False),
               ('split-pipelined-20hz', None, .05, True)]
      for name, model, period, pipeline in cases:
        print(f'Measuring {name}', flush=True)
        gpu_process = temporary = None
        try:
          if pipeline:
            active_gpu = gpu
            if args.gpu_process:
              from tools.coreml_gpu_process import GPUProcess
              temporary = tempfile.TemporaryDirectory(prefix='mac-frame-pipeline-')
              gpu_path = Path(temporary.name) / 'gpu.mlpackage'
              gpu.save(str(gpu_path))
              gpu_process = active_gpu = GPUProcess(gpu_path, gpu.get_spec())
              report['sharedMemoryBytes'] = 2 * gpu_process.stride
            model = FramePipeline(active_gpu, ane, frontier, tail_inputs, pixels, args.warmup + args.runs, period)
          result, predictions = run(model, metadata, output_name, pixels, args.warmup, args.runs,
                                     period=period, pipeline=pipeline)
        finally:
          if gpu_process is not None:
            gpu_process.close()
          if temporary is not None:
            temporary.cleanup()
        if reference is None:
          reference = predictions
        result['maxAbsVsBaseline'] = float(np.abs(predictions - reference).max())
        if name == 'split-serial':
          split_reference = predictions
        if pipeline:
          result['exactlyMatchesSerialSplit'] = bool(np.array_equal(predictions, split_reference))
          result['maxAbsVsSerialSplit'] = float(np.abs(predictions - split_reference).max())
        report['runs'][name] = result
        print(json.dumps({k: v for k, v in result.items() if k != 'timeline'}), flush=True)
        if pipeline and not result['exactlyMatchesSerialSplit']:
          raise RuntimeError('pipeline differs from serial split; reject this experiment')
      report['status'] = 'measured-not-qualified'
    except Exception as error:
      report['status'], report['error'] = 'failed', f'{type(error).__name__}: {error}'
      raise
    finally:
      json.dump(report, output, indent=2, allow_nan=False)
      output.write('\n')


if __name__ == '__main__':
  main()
