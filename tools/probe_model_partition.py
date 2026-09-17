"""Mac-only explicit Core ML backend-placement experiment; never used by the server.

Backend annotations are experimental hints. Require a mixed-device compute plan
before calling a candidate mixed. No zero-copy or inter-frame overlap is assumed.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_coreml import make_inputs
from macos_performance import configure_user_interactive_qos


def graph_groups(spec):
  function = spec.mlProgram.functions['main']
  block = function.block_specializations[function.opset]
  dependencies = {value.name: {value.name} for value in function.inputs}
  operations = []
  consumers = defaultdict(list)
  for index, op in enumerate(block.operations):
    if op.blocks:
      raise ValueError('nested control flow requires a separate dependency analysis')
    names = [arg.name for value in op.inputs.values() for arg in value.arguments if arg.HasField('name')]
    unknown = [name for name in names if name not in dependencies]
    if unknown:
      raise ValueError(f'unknown graph dependencies: {unknown}')
    parents = set().union(*(dependencies[name] for name in names))
    for name in names:
      consumers[name].append(index)
    for value in op.outputs:
      dependencies[value.name] = parents
    operations.append((op, parents))
  vision = {i for i, (_, deps) in enumerate(operations) if deps and deps <= {'img', 'big_img'}}
  frontier = []
  for i in sorted(vision):
    for value in operations[i][0].outputs:
      if any(user not in vision for user in consumers[value.name]) or value.name in block.outputs:
        dims = [d.constant.size for d in value.type.tensorType.dimensions]
        frontier.append({'name': value.name, 'shape': dims, 'dtype': value.type.tensorType.dataType})
  return operations, vision, frontier


def summarize(values):
  return {'samples': len(values), 'meanMs': float(np.mean(values)),
          'p99Ms': float(np.percentile(values, 99)), 'maxMs': float(max(values)),
          'over50Ms': sum(value > 50 for value in values)}


def plan_summary(model):
  from coremltools.models.compute_plan import MLComputePlan
  plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=model.compute_unit)
  counts = Counter()
  operators = defaultdict(Counter)
  def visit(block):
    for op in block.operations:
      usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
      device = type(usage.preferred_compute_device).__name__ if usage and usage.preferred_compute_device else 'unknown'
      counts[device] += 1
      operators[device][op.operator_name] += 1
      for child in op.blocks:
        visit(child)
  for function in plan.model_structure.program.functions.values():
    visit(function.block)
  return {'preferredDeviceOperations': dict(counts), 'operators': dict(operators),
          'scope': 'anticipated compute plan, not runtime tracing or time percentages'}


def main():
  import coremltools as ct
  from coremltools.models.ml_program.experimental.compute_plan_utils import set_intended_backends
  root = Path(__file__).resolve().parents[1]
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode', choices=('baseline', 'stem-gpu', 'vision-gpu', 'policy-gpu'), required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--runs', type=int, default=120)
  parser.add_argument('--warmup', type=int, default=40)
  parser.add_argument('--stem-operations', type=int, default=40)
  args = parser.parse_args()
  if args.runs < 20 or args.warmup < 1 or args.stem_operations < 1:
    parser.error('invalid sample/warmup/stem counts')
  report = {'mode': args.mode, 'roadReady': False, 'zeroCopyVerified': False,
            'interFrameOverlap': False, 'numericalQualified': False,
            'scope': 'synthetic model inputs; no camera, transport or device access'}
  # Refuse overwrites and reserve the report before expensive compilation.
  with args.output.open('x') as output:
    try:
      report['qos'] = configure_user_interactive_qos()
      model_path = root / 'artifacts/big_driving.mlpackage'
      metadata_path = root / 'models/big_driving_supercombo_metadata.pkl'
      with metadata_path.open('rb') as handle:
        metadata = pickle.load(handle)
      base = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
      spec = base.get_spec()
      report['modelSpecSha256'] = hashlib.sha256(spec.SerializeToString(deterministic=True)).hexdigest()
      operations, vision, frontier = graph_groups(spec)
      selected = (vision if args.mode == 'vision-gpu' else
                  set(sorted(vision)[:args.stem_operations]) if args.mode == 'stem-gpu' else
                  {i for i, (_, deps) in enumerate(operations) if 'features_buffer' in deps}
                  if args.mode == 'policy-gpu' else set())
      # Keep the final assembly op unannotated; do not force unsupported outputs.
      selected = {i for i in selected if not any(v.name in spec.mlProgram.functions['main'].block_specializations[
                  spec.mlProgram.functions['main'].opset].outputs for v in operations[i][0].outputs)}
      selected_names = {v.name for i in selected for v in operations[i][0].outputs}
      report['visionFrontier'] = frontier
      report['requestedGpuOperations'] = len(selected)
      started = time.perf_counter()
      model = base if not selected else set_intended_backends(
        base, lambda op: ['mps_graph'] if any(v.name in selected_names for v in op.outputs) else None)
      report['partitionCompileSeconds'] = time.perf_counter() - started
      report['plan'] = plan_summary(model)
      counts = report['plan']['preferredDeviceOperations']
      if selected and (not counts.get('MLGPUComputeDevice') or not counts.get('MLNeuralEngineComputeDevice')):
        raise RuntimeError('backend hints did not produce an anticipated GPU+ANE mixed plan')
      output_name = spec.description.output[0].name
      hidden = metadata['output_slices']['hidden_state']
      expected_shape = tuple(metadata['output_shapes']['outputs'])
      def predict(which, inputs):
        result = np.asarray(which.predict(inputs)[output_name])
        if result.shape != expected_shape or not np.isfinite(result).all():
          raise ValueError('invalid model output')
        return result
      def advance(inputs, result):
        history = inputs['features_buffer']
        history[:, :-1] = history[:, 1:]
        history[:, -1] = result[0, hidden].reshape(history[:, -1].shape)
      # Independent histories: numerical differences are allowed to propagate.
      reference_inputs = make_inputs(np.random.default_rng(17))
      candidate_inputs = {key: value.copy() for key, value in reference_inputs.items()}
      errors = []
      field_errors = defaultdict(list)
      for _ in range(70):
        reference = predict(base, reference_inputs)
        candidate = predict(model, candidate_inputs)
        difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
        errors.append(float(difference.max()))
        for name, section in metadata['output_slices'].items():
          field_errors[name].append(float(difference[:, section].max()))
        advance(reference_inputs, reference)
        advance(candidate_inputs, candidate)
      report['numericalComparison'] = {'frames': len(errors), 'maxAbs': max(errors),
                                       'finalMaxAbs': errors[-1], 'fieldMaxAbs': {k: max(v) for k, v in field_errors.items()},
                                       'scope': '70-frame synthetic independent histories; no driving acceptance threshold'}
      # Baseline and candidate use identical seed, warmup and recurrence semantics.
      inputs = make_inputs(np.random.default_rng(0))
      for _ in range(args.warmup):
        advance(inputs, predict(model, inputs))
      times, prediction_times = [], []
      span_start = time.perf_counter_ns()
      for _ in range(args.runs):
        start = time.perf_counter_ns()
        result = predict(model, inputs)
        predicted = time.perf_counter_ns()
        advance(inputs, result)
        end = time.perf_counter_ns()
        prediction_times.append((predicted - start) / 1e6)
        times.append((end - start) / 1e6)
      span = (time.perf_counter_ns() - span_start) / 1e9
      report['serialSaturated'] = {**summarize(times), 'completedPerSecond': args.runs / span,
                                   'predictionAndValidation': summarize(prediction_times), 'durationsMs': times}
      report['status'] = 'measured-not-qualified'
    except Exception as error:
      report['status'] = 'failed'
      report['error'] = f'{type(error).__name__}: {error}'
      raise
    finally:
      json.dump(report, output, indent=2, allow_nan=False)
      output.write('\n')
      print(json.dumps({k: v for k, v in report.items() if k != 'visionFrontier'}, indent=2), flush=True)


if __name__ == '__main__':
  main()
