"""Optional tinygrad Metal gather; no sunnypilot dependency, no serialized JIT."""

import numpy as np


def reference_indices(layout, transform, model_width=512, model_height=256, device='METAL'):
  """Compile lookup coordinates with upstream's actual tinygrad FP32 operations.

  Use an integer index image so upstream's warp/packing returns source indices.
  Work and host readback happen at setup/calibration change, not every frame.
  Derived from compile_modeld.make_frame_prepare (upstream MIT, see NOTICE).
  """
  from tinygrad import Tensor
  from tinygrad.helpers import Context

  def warp(src, matrix, width, height, sw, sh, stride):
    x = Tensor.arange(width).to(device).reshape(1, width).expand(height, width).reshape(-1)
    y = Tensor.arange(height).to(device).reshape(height, 1).expand(height, width).reshape(-1)
    sx = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    sy = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    scale = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    ix = (sx / scale).round().clip(0, sw - 1).cast('int')
    iy = (sy / scale).round().clip(0, sh - 1).cast('int')
    return src[iy * stride + ix]

  source = Tensor.arange(layout.nbytes).cast('int32').to(device).realize()
  matrix = Tensor(transform, device=device).realize()
  uv_matrix = matrix * Tensor([[1., 1., .5], [1., 1., .5], [2., 2., 1.]], device=device)
  offset = layout.stride * layout.y_height
  uv = source[offset:offset + layout.uv_height * layout.stride].reshape(layout.uv_height, layout.stride)
  with Context(SPLIT_REDUCEOP=0):
    y = warp(source[:layout.height * layout.stride], matrix, model_width, model_height,
             layout.width, layout.height, layout.stride).realize()
    u = warp(uv[:layout.height // 2, :layout.width:2].flatten(), uv_matrix, model_width // 2, model_height // 2,
             layout.width // 2, layout.height // 2, layout.width // 2).realize()
    v = warp(uv[:layout.height // 2, 1:layout.width:2].flatten(), uv_matrix, model_width // 2, model_height // 2,
             layout.width // 2, layout.height // 2, layout.width // 2).realize()
  packed = y.cat(u).cat(v).reshape(model_height * 3 // 2, model_width)
  indices = Tensor.cat(packed[0:model_height:2, 0::2], packed[1:model_height:2, 0::2],
                       packed[0:model_height:2, 1::2], packed[1:model_height:2, 1::2],
                       packed[model_height:model_height + model_height // 4].reshape(model_height // 2, model_width // 2),
                       packed[model_height + model_height // 4:model_height + model_height // 2].reshape(model_height // 2, model_width // 2),
                       dim=0).reshape(6, model_height // 2, model_width // 2).numpy()
  result = np.asarray(indices, dtype=np.intp)
  if np.any(result < 0) or np.any(result >= layout.nbytes):
    raise ValueError('reference lookup indices out of bounds')
  return result


class TinygradGather:
  def __init__(self, plan, device='METAL'):
    from tinygrad import Tensor
    from tinygrad.engine.jit import TinyJit
    self.Tensor, self.device, self.plan = Tensor, device, plan
    self.indices = Tensor(plan.indices.astype(np.int32), device=device).realize()
    self.jit = TinyJit(lambda frame: frame[self.indices].realize())

  def run(self, frame: bytes):
    if type(frame) is not bytes or len(frame) != self.plan.layout.nbytes:
      raise ValueError('expected owned NV12 bytes of exact layout size')
    source = self.Tensor(np.frombuffer(frame, dtype=np.uint8), device=self.device).realize()
    # Count input transfer, JIT execution AND completed host readback in timing.
    return self.jit(source).numpy().copy()
