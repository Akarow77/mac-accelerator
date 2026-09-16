"""Independent, CPU-addressable NV12 -> model-input preparation (no camera/GPU hooks).

Geometry follows openpilot's nearest-even, edge-clamped NV12 warp and six-plane
packing. Source: sunnypilot a5f44653/openpilot/selfdrive/modeld/compile_modeld.py.
The caller must supply independently owned immutable frame bytes. A VisionIPC
view is NOT ownership and is deliberately not accepted by the pair API.
"""

from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class NV12Layout:
  width: int
  height: int
  stride: int
  y_height: int
  uv_height: int

  def __post_init__(self):
    values = (self.width, self.height, self.stride, self.y_height, self.uv_height)
    if any(type(x) is not int or x <= 0 for x in values):
      raise ValueError('layout dimensions must be positive integers')
    if (self.width % 2 or self.height % 2 or self.stride < self.width
        or self.y_height < self.height or self.uv_height < self.height // 2 or self.nbytes > 32 * 1024 * 1024):
      raise ValueError('invalid or oversized NV12 layout')

  @property
  def nbytes(self):
    return self.stride * (self.y_height + self.uv_height)

  @classmethod
  def packed(cls, width, height):
    return cls(width, height, width, height, height // 2)


def checked_transform(transform):
  value = np.array(transform, dtype=np.float32, copy=True)
  if value.shape != (3, 3) or not np.isfinite(value).all():
    raise ValueError('transform must be finite float32 3x3')
  if abs(float(np.linalg.det(value))) < 1e-10:
    raise ValueError('singular transform')
  return value


def plane_coordinates(matrix, width, height, source_width, source_height):
  x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
  denominator = (matrix[2, 0] * x + matrix[2, 1] * y) + matrix[2, 2]
  if not np.isfinite(denominator).all() or np.any(np.abs(denominator) < 1e-8):
    raise ValueError('invalid projective denominator')
  with np.errstate(over='raise', invalid='raise', divide='raise'):
    sx = ((matrix[0, 0] * x + matrix[0, 1] * y) + matrix[0, 2]) / denominator
    sy = ((matrix[1, 0] * x + matrix[1, 1] * y) + matrix[1, 2]) / denominator
  if not np.isfinite(sx).all() or not np.isfinite(sy).all():
    raise ValueError('non-finite source coordinate')
  return (np.clip(np.rint(sx), 0, source_width - 1).astype(np.intp),
          np.clip(np.rint(sy), 0, source_height - 1).astype(np.intp))


class GatherPlan:
  def __init__(self, layout: NV12Layout, transform, model_width=512, model_height=256, *, reference_device=None):
    if (type(model_width) is not int or type(model_height) is not int or
        not 0 < model_width <= 2048 or not 0 < model_height <= 1024 or model_width % 2 or model_height % 2):
      raise ValueError('invalid model dimensions')
    self.layout = layout
    self.transform = checked_transform(transform)
    self.shape = (6, model_height // 2, model_width // 2)
    sx, sy = plane_coordinates(self.transform, model_width, model_height, layout.width, layout.height)
    y_indices = sy * layout.stride + sx
    uv_transform = self.transform * np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]], dtype=np.float32)
    ux, uy = plane_coordinates(uv_transform, model_width // 2, model_height // 2, layout.width // 2, layout.height // 2)
    uv_indices = layout.stride * layout.y_height + uy * layout.stride + 2 * ux
    self.indices = np.stack((y_indices[::2, ::2], y_indices[1::2, ::2],
                             y_indices[::2, 1::2], y_indices[1::2, 1::2], uv_indices, uv_indices + 1))
    if reference_device is not None:
      from backends.tinygrad_warp import reference_indices
      self.indices = reference_indices(layout, self.transform, model_width, model_height, reference_device)
    self.indices.flags.writeable = False
    self.transform.flags.writeable = False

  def run(self, frame: bytes, output: np.ndarray | None = None):
    if type(frame) is not bytes or len(frame) != self.layout.nbytes:
      raise ValueError('expected independently owned immutable NV12 bytes of exact layout size')
    if output is None:
      output = np.empty(self.shape, dtype=np.uint8)
    if output.dtype != np.uint8 or output.shape != self.shape or not output.flags.c_contiguous:
      raise ValueError('invalid output buffer')
    # A single gather handles Y packing and interleaved UV directly. No full-frame
    # U/V deinterleave, padding copy, GPU synchronization or YUV intermediate.
    np.take(np.frombuffer(frame, dtype=np.uint8), self.indices, out=output, mode='clip')
    return output


@dataclass(frozen=True)
class OwnedCameraFrame:
  frame_id: int
  eof_ns: int
  data: bytes

  def __post_init__(self):
    if type(self.frame_id) is not int or self.frame_id < 0 or type(self.eof_ns) is not int or self.eof_ns <= 0:
      raise ValueError('invalid frame identity or timestamp')
    if type(self.data) is not bytes:
      raise ValueError('camera buffer views are unsafe; owned bytes required')


class PairPreprocessor:
  """Synchronous observer-side preparation; never invoke inside a vehicle model.

  Keeps exactly one plan per camera. Calibration changes replace plans, not append
  to a growing pointer cache. Results own their bytes, not reusable array views.
  """
  def __init__(self, narrow_layout, wide_layout, *, backend='cpu', reference_device=None, max_pair_skew_ms=5., max_age_ms=100.):
    if not all(math.isfinite(x) and x > 0 for x in (max_pair_skew_ms, max_age_ms)):
      raise ValueError('invalid pair timing limits')
    self.layouts = (narrow_layout, wide_layout)
    if backend not in ('cpu', 'native-cpu', 'tinygrad-metal'):
      raise ValueError('unknown preprocessing backend')
    self.backend = backend
    self.reference_device = reference_device
    self.max_skew_ns = int(max_pair_skew_ms * 1e6)
    self.max_age_ns = int(max_age_ms * 1e6)
    self.plans = [None, None]
    self.runners = [None, None]
    self.last_ids = None
    self.last_eof = None
    self.failed = False
    self.warmed = False
    self.output = np.empty((2, 6, 128, 256), dtype=np.uint8)

  def _update_plans(self, transforms):
    if len(transforms) != 2:
      raise ValueError('exactly two transforms required')
    for index, transform in enumerate(transforms):
      matrix = checked_transform(transform)
      plan = self.plans[index]
      if plan is None or not np.array_equal(plan.transform, matrix):
        if self.warmed:
          raise ValueError('calibration changed; stop observer and warm a new plan explicitly')
        self.plans[index] = GatherPlan(self.layouts[index], matrix, reference_device=self.reference_device)
        if self.backend == 'tinygrad-metal':
          from backends.tinygrad_warp import TinygradGather
          self.runners[index] = TinygradGather(self.plans[index])
        elif self.backend == 'native-cpu':
          from backends.native_gather import NativeGather
          self.runners[index] = NativeGather(self.plans[index])

  def _gather(self, frames):
    for index, frame in enumerate(frames):
      if self.backend == 'cpu':
        self.plans[index].run(frame, self.output[index])
      elif self.backend == 'native-cpu':
        self.runners[index].run(frame, self.output[index])
      else:
        self.output[index] = self.runners[index].run(frame)
    return self.output.tobytes()

  def warmup(self, transforms):
    """Allocate plans and exercise kernels before accepting any stream frames."""
    if self.failed or self.last_ids is not None:
      raise RuntimeError('warmup is only allowed before a new observer run')
    try:
      self._update_plans(transforms)
      rng = np.random.default_rng(0)
      frames = tuple(rng.integers(0, 256, layout.nbytes, dtype=np.uint8).tobytes() for layout in self.layouts)
      expected = np.stack([plan.run(frame) for plan, frame in zip(self.plans, frames)]).tobytes()
      for _ in range(8):
        if self._gather(frames) != expected:
          raise ValueError('warmup output mismatch')
      self.warmed = True
    except Exception:
      self.failed = True
      raise

  def prepare(self, narrow, wide, transforms, *, clock=time.monotonic_ns):
    if self.failed:
      raise RuntimeError('preprocessor failure latched; explicitly create a new observer')
    try:
      if self.reference_device is not None and not self.warmed:
        raise RuntimeError('reference-map warmup required before accepting frames')
      start = clock()
      frames = (narrow, wide)
      ids = tuple(frame.frame_id for frame in frames)
      stamps = tuple(frame.eof_ns for frame in frames)
      if self.last_ids is not None and any(current != previous + 1 for current, previous in zip(ids, self.last_ids)):
        raise ValueError('camera sequence gap or duplicate')
      if self.last_eof is not None and any(current <= previous for current, previous in zip(stamps, self.last_eof)):
        raise ValueError('camera timestamp moved backward')
      if max(stamps) > start or start - min(stamps) > self.max_age_ns or abs(stamps[0] - stamps[1]) > self.max_skew_ns:
        raise ValueError('stale, future or unpaired camera timestamps')
      plan_start = clock()
      self._update_plans(transforms)
      plan_end = clock()
      result = self._gather(tuple(frame.data for frame in frames))
      end = clock()
      if end - min(stamps) > self.max_age_ns:
        raise TimeoutError('frame became stale during preparation')
      self.last_ids, self.last_eof = ids, stamps
      return result, {'planMs': (plan_end - plan_start) / 1e6,
                      'gatherAndOutputCopyMs': (end - plan_end) / 1e6,
                      'prepareMs': (end - start) / 1e6}
    except Exception:
      self.failed = True
      raise
