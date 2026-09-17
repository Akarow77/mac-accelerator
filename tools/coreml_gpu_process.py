"""Offline experiment only: separate Core ML GPU process, two shared-memory slots.

This removes same-interpreter serialization, not Core ML's internal copies.
The caller owns slot lifetimes and must allow at most two outstanding frames.
"""

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import time
import threading

import numpy as np

from macos_performance import configure_user_interactive_qos


def views(memory, layout, stride, slot):
  return {name: np.ndarray(shape, dtype=dtype, buffer=memory.buf, offset=slot * stride + offset)
          for name, (shape, dtype, offset) in layout.items()}


def worker(connection, model_path, memory_name, layout, stride, input_names, output_names):
  import coremltools as ct
  memory = None
  try:
    configure_user_interactive_qos()
    memory = SharedMemory(name=memory_name)
    model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    connection.send(('ready',))
    while True:
      slot = connection.recv()
      if slot is None:
        return
      if slot not in (0, 1):
        raise ValueError('invalid shared-memory slot')
      buffers = views(memory, layout, stride, slot)
      started = time.perf_counter_ns()
      result = model.predict({name: buffers[name] for name in input_names})
      ended = time.perf_counter_ns()
      if set(result) != set(output_names):
        raise ValueError('GPU output names changed')
      for name in output_names:
        if result[name].shape != buffers[name].shape or not np.isfinite(result[name]).all():
          raise ValueError('invalid GPU result')
        # The Python prediction API can return float32 even for an FP16 spec.
        # Preserve it; never silently downcast the inter-stage tensor.
        np.copyto(buffers[name], result[name], casting='safe')
      connection.send(('done', slot, started, ended))
      del buffers, result
  except Exception as error:
    try:
      connection.send(('error', f'{type(error).__name__}: {error}'))
    except (BrokenPipeError, EOFError, OSError):
      pass
  finally:
    if memory is not None:
      memory.close()
    connection.close()


class GPUProcess:
  def __init__(self, model_path, spec):
    dtype_map = {65552: '<f2', 65568: '<f4'}
    self.layout, self.stride = {}, 0
    self.input_names = [v.name for v in spec.description.input]
    self.output_names = [v.name for v in spec.description.output]
    for feature in (*spec.description.input, *spec.description.output):
      array = feature.type.multiArrayType
      if array.dataType not in dtype_map or not array.shape or any(d <= 0 for d in array.shape):
        raise ValueError('shared-memory experiment requires fixed float16/float32 tensors')
      dtype = '<f4' if feature.name in self.output_names else dtype_map[array.dataType]
      shape = tuple(array.shape)
      self.layout[feature.name] = (shape, dtype, self.stride)
      self.stride += (int(np.prod(shape)) * np.dtype(dtype).itemsize + 127) // 128 * 128
    self.memory = SharedMemory(create=True, size=2 * self.stride)
    context = mp.get_context('spawn')
    self.connection, child = context.Pipe()
    self.process = context.Process(target=worker, args=(child, str(model_path), self.memory.name, self.layout,
                                                       self.stride, self.input_names, self.output_names), daemon=True)
    self.frame = 0
    self.submitted = -1
    self.submission = threading.Condition()
    self.last_interval = None
    try:
      self.process.start()
      child.close()
      if not self.connection.poll(120) or self.connection.recv() != ('ready',):
        raise RuntimeError('GPU worker did not become ready')
    except Exception:
      self.close()
      raise

  def predict(self, inputs):
    slot = self.frame % 2
    buffers = views(self.memory, self.layout, self.stride, slot)
    for name in self.input_names:
      np.copyto(buffers[name], inputs[name], casting='no')
    self.connection.send(slot)
    with self.submission:
      self.submitted = self.frame
      self.submission.notify_all()
    if not self.connection.poll(15):
      raise TimeoutError('GPU process did not complete')
    response = self.connection.recv()
    if response[0] != 'done' or response[1] != slot:
      raise RuntimeError(f'GPU process failed: {response}')
    self.last_interval = response[2:4]
    self.frame += 1
    # Views are valid until this slot is reused TWO frames later. FramePipeline's
    # semaphore holds the slot through synchronous ANE completion.
    return {name: buffers[name] for name in self.output_names}

  def wait_submitted(self, frame):
    # Let the parent producer dispatch the next GPU request BEFORE the main
    # thread enters a potentially GIL-holding ANE call. Saturated tests only.
    with self.submission:
      if not self.submission.wait_for(lambda: self.submitted >= frame, timeout=15):
        raise TimeoutError('next GPU frame was not submitted')

  def close(self):
    if self.process.pid is not None:
      if self.process.is_alive():
        try:
          self.connection.send(None)
        except (BrokenPipeError, OSError):
          pass
        self.process.join(3)
      if self.process.is_alive():
        self.process.terminate()
        self.process.join(3)
      if self.process.is_alive():
        self.process.kill()
        self.process.join(3)
    self.connection.close()
    self.memory.close()
    self.memory.unlink()
