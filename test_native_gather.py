import ctypes
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from backends.native_gather import NativeGather
from camera_preprocess import GatherPlan, NV12Layout, OwnedCameraFrame, PairPreprocessor


@unittest.skipUnless((Path(__file__).parent / 'artifacts/libcamera_gather.dylib').is_file(),
                     'optional native library not built')
class NativeGatherTests(unittest.TestCase):
  def setUp(self):
    self.layout = NV12Layout(8, 4, 12, 6, 3)
    self.plan = GatherPlan(self.layout, np.eye(3), 8, 4)
    self.runner = NativeGather(self.plan)
    self.raw = np.arange(self.layout.nbytes, dtype=np.uint8).tobytes()

  def test_exact_random_padded_and_projective_output(self):
    rng = np.random.default_rng(31)
    for matrix in (np.eye(3), [[1, 0, .5], [0, 1, -.5], [0, 0, 1]],
                   [[.9, .1, 2], [.1, 1.1, -2], [.01, -.01, 1]]):
      plan = GatherPlan(self.layout, matrix, 8, 4)
      runner = NativeGather(plan)
      for _ in range(4):
        frame = rng.integers(0, 256, self.layout.nbytes, dtype=np.uint8).tobytes()
        np.testing.assert_array_equal(runner.run(frame), plan.run(frame))

  def test_frame_and_output_validation(self):
    for frame in (self.raw[:-1], memoryview(self.raw), bytearray(self.raw)):
      with self.assertRaises(ValueError):
        self.runner.run(frame)
    readonly = np.empty(self.plan.shape, dtype=np.uint8)
    readonly.flags.writeable = False
    for output in ([], readonly, np.empty(self.plan.shape), np.empty((1,), dtype=np.uint8)):
      with self.assertRaises(ValueError):
        self.runner.run(self.raw, output)

  def test_bad_indices_rejected_before_conversion(self):
    for value in (-1, self.layout.nbytes, 2**32):
      self.plan.indices = np.full(self.plan.shape, value, dtype=np.int64)
      with self.assertRaisesRegex(ValueError, 'indices'):
        NativeGather(self.plan)
    self.plan.indices = np.zeros(self.plan.shape, dtype=np.float32)
    with self.assertRaisesRegex(ValueError, 'indices'):
      NativeGather(self.plan)

  def test_c_bounds_checks_do_not_overwrite_destination(self):
    output = np.full(4, 123, dtype=np.uint8)
    bad = np.array([self.layout.nbytes], dtype=np.uint32)
    fn = self.runner.function
    source = ctypes.c_char_p(self.raw)
    self.assertEqual(fn(source, len(self.raw), bad.ctypes.data, 1, output.ctypes.data, 4), 2)
    self.assertEqual(fn(source, len(self.raw), bad.ctypes.data, 5, output.ctypes.data, 4), 1)
    self.assertEqual(fn(None, 0, bad.ctypes.data, 1, output.ctypes.data, 4), 1)
    self.assertEqual(output.tolist(), [123] * 4)

  def test_changed_source_requires_rebuild(self):
    with patch('backends.native_gather.hashlib.sha256') as digest:
      digest.return_value.hexdigest.return_value = 'different-source'
      with self.assertRaisesRegex(ValueError, 'rebuild'):
        NativeGather(self.plan)

  def test_pair_warmup_and_owned_output(self):
    layout = NV12Layout.packed(512, 256)
    processor = PairPreprocessor(layout, layout, backend='native-cpu')
    matrices = (np.eye(3), np.eye(3))
    processor.warmup(matrices)
    first = OwnedCameraFrame(0, 100, bytes(layout.nbytes))
    result, _ = processor.prepare(first, first, matrices, clock=lambda: 1000)
    second = OwnedCameraFrame(1, 200, bytes([27]) * layout.nbytes)
    changed, _ = processor.prepare(second, second, matrices, clock=lambda: 1000)
    self.assertEqual(result, bytes(393216))
    self.assertEqual(changed, bytes([27]) * 393216)


if __name__ == '__main__':
  unittest.main()
