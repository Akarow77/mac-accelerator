"""Bounds-checked native CPU gather with a source-bound local build."""

import ctypes
import hashlib
from pathlib import Path

import numpy as np


class NativeGather:
  def __init__(self, plan):
    root = Path(__file__).resolve().parents[1]
    self.library = ctypes.CDLL(str(root / 'artifacts/libcamera_gather.dylib'))
    build_id = self.library.mac_gather_build_id
    build_id.argtypes = []
    build_id.restype = ctypes.c_char_p
    expected = hashlib.sha256((root / 'native/gather.c').read_bytes()).hexdigest()
    if build_id().decode() != expected:
      raise ValueError('native gather source changed; rebuild before profiling')
    self.function = self.library.mac_gather_bytes
    self.function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                              ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    self.function.restype = ctypes.c_int
    self.plan = plan
    if (plan.indices.shape != plan.shape or plan.indices.dtype.kind not in 'iu' or
        np.any(plan.indices < 0) or np.any(plan.indices >= plan.layout.nbytes)):
      raise ValueError('invalid source indices')
    self.indices = np.array(plan.indices, dtype=np.uint32, order='C', copy=True)
    self.indices.flags.writeable = False

  def run(self, frame: bytes, output=None):
    if type(frame) is not bytes or len(frame) != self.plan.layout.nbytes:
      raise ValueError('expected independently owned frame bytes of exact length')
    if output is None:
      output = np.empty(self.plan.shape, dtype=np.uint8)
    if (not isinstance(output, np.ndarray) or output.dtype != np.uint8 or output.shape != self.plan.shape or
        not output.flags.c_contiguous or not output.flags.writeable):
      raise ValueError('invalid output buffer')
    # c_char_p pins the immutable bytes for this synchronous call, without copying
    # the entire NV12 allocation. The C function never writes to the source/map.
    source = ctypes.c_char_p(frame)
    result = self.function(source, len(frame), self.indices.ctypes.data, self.indices.size,
                            output.ctypes.data, output.nbytes)
    if result:
      raise ValueError(f'native gather rejected buffer/index bounds: {result}')
    return output
