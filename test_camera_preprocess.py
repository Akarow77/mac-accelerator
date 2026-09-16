import unittest
from unittest.mock import patch

import numpy as np

from camera_preprocess import GatherPlan, NV12Layout, OwnedCameraFrame, PairPreprocessor
from profile_preprocess import select_candidate, validate_profile


class CameraPreprocessTests(unittest.TestCase):
  def test_identity_and_nv12_chroma_order_with_padding(self):
    layout = NV12Layout(8, 4, 12, 6, 3)
    raw = np.full(layout.nbytes, 255, dtype=np.uint8)
    y = np.arange(32, dtype=np.uint8).reshape(4, 8)
    raw[:72].reshape(6, 12)[:4, :8] = y
    raw[72:].reshape(3, 12)[:2, :8:2] = 91
    raw[72:].reshape(3, 12)[:2, 1:8:2] = 173
    result = GatherPlan(layout, np.eye(3), 8, 4).run(raw.tobytes())
    expected = np.stack((y[::2, ::2], y[1::2, ::2], y[::2, 1::2], y[1::2, 1::2],
                         np.full((2, 4), 91, dtype=np.uint8), np.full((2, 4), 173, dtype=np.uint8)))
    np.testing.assert_array_equal(result, expected)

  def test_round_half_even_and_edge_clamp(self):
    layout = NV12Layout.packed(8, 4)
    raw = np.arange(layout.nbytes, dtype=np.uint8).tobytes()
    transform = np.array([[1, 0, .5], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    plan = GatherPlan(layout, transform, 8, 4)
    self.assertEqual(plan.indices[0, 0].tolist(), [0, 2, 4, 6])
    self.assertEqual(plan.indices[2, 0].tolist(), [2, 4, 6, 7])
    self.assertEqual(plan.run(raw).shape, (6, 2, 4))

  def test_zero_projective_denominator_rejected(self):
    matrix = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=np.float32)
    with self.assertRaisesRegex(ValueError, 'denominator'):
      GatherPlan(NV12Layout.packed(8, 4), matrix, 8, 4)

  def test_borrowed_camera_memory_rejected(self):
    with self.assertRaisesRegex(ValueError, 'owned'):
      OwnedCameraFrame(1, 1, memoryview(b'unsafe'))

  def test_layout_and_nonfinite_transform_rejected(self):
    with self.assertRaises(ValueError):
      NV12Layout(8, 4, 4, 4, 2)
    with self.assertRaises(ValueError):
      GatherPlan(NV12Layout.packed(8, 4), np.full((3, 3), np.nan))

  def test_pair_plan_reuse_invalidation_and_owned_output(self):
    layout = NV12Layout.packed(512, 256)
    processor = PairPreprocessor(layout, layout)
    raw = bytes(layout.nbytes)
    tfm = np.eye(3, dtype=np.float32)
    first, _ = processor.prepare(OwnedCameraFrame(0, 100, raw), OwnedCameraFrame(0, 100, raw), (tfm, tfm), clock=lambda: 1000)
    plan = processor.plans[0]
    processor.prepare(OwnedCameraFrame(1, 200, raw), OwnedCameraFrame(1, 200, raw), (tfm, tfm), clock=lambda: 1000)
    self.assertIs(processor.plans[0], plan)
    tfm[0, 2] = 1
    second, _ = processor.prepare(OwnedCameraFrame(2, 300, bytes([17]) * layout.nbytes),
                                  OwnedCameraFrame(2, 300, raw), (tfm, tfm), clock=lambda: 1000)
    self.assertIsNot(processor.plans[0], plan)
    self.assertEqual(len(processor.plans), 2)
    self.assertNotEqual(first, second)
    self.assertEqual(first, bytes(393216))

  def test_pair_skew_and_gap_latch_failure(self):
    layout = NV12Layout.packed(8, 4)
    processor = PairPreprocessor(layout, layout)
    raw = bytes(layout.nbytes)
    with self.assertRaisesRegex(ValueError, 'unpaired'):
      processor.prepare(OwnedCameraFrame(0, 1, raw), OwnedCameraFrame(0, 10_000_000, raw),
                        (np.eye(3), np.eye(3)), clock=lambda: 10_000_000)
    with self.assertRaisesRegex(RuntimeError, 'latched'):
      processor.prepare(None, None, ())

  def test_profile_requires_correct_output_and_tail_budget(self):
    candidates = {'bad-fast': {'verified': False, 'samples': 100, 'p99Ms': .1, 'maxMs': .1},
                  'tail-spike': {'verified': True, 'samples': 100, 'p99Ms': 1, 'maxMs': 9},
                  'pass': {'verified': True, 'samples': 100, 'p99Ms': 2, 'maxMs': 3}}
    self.assertEqual(select_candidate(candidates, 5), 'pass')
    self.assertIsNone(select_candidate(candidates, .01))

  def test_profile_from_other_chip_or_os_rejected(self):
    with patch('profile_preprocess.hardware_signature', return_value={'chip': 'M1'}):
      with self.assertRaisesRegex(ValueError, 'mismatch'):
        validate_profile({'hardware': {'chip': 'M2'}})

  def test_numpy_only_profile_is_not_accepted_as_upstream_equivalent(self):
    with patch('profile_preprocess.hardware_signature', return_value={'chip': 'M2'}):
      with self.assertRaisesRegex(ValueError, 'reference-map'):
        validate_profile({'hardware': {'chip': 'M2'}, 'mapReferenceDevice': None})

  def test_warmup_does_not_consume_frame_sequence(self):
    layout = NV12Layout.packed(512, 256)
    processor = PairPreprocessor(layout, layout)
    processor.warmup((np.eye(3), np.eye(3)))
    self.assertIsNone(processor.last_ids)
    frame = OwnedCameraFrame(0, 100, bytes(layout.nbytes))
    processor.prepare(frame, frame, (np.eye(3), np.eye(3)), clock=lambda: 1000)
    with self.assertRaisesRegex(RuntimeError, 'before'):
      processor.warmup((np.eye(3), np.eye(3)))

  def test_pair_sequence_gap_and_postprocessing_expiry_stop(self):
    layout = NV12Layout.packed(512, 256)
    processor = PairPreprocessor(layout, layout)
    frame = OwnedCameraFrame(0, 100, bytes(layout.nbytes))
    processor.prepare(frame, frame, (np.eye(3), np.eye(3)), clock=lambda: 1000)
    gap = OwnedCameraFrame(2, 200, frame.data)
    with self.assertRaisesRegex(ValueError, 'gap'):
      processor.prepare(gap, gap, (np.eye(3), np.eye(3)), clock=lambda: 1000)
    processor = PairPreprocessor(layout, layout)
    times = iter([1000, 1000, 1000, 101_000_100])
    with self.assertRaisesRegex(TimeoutError, 'stale'):
      processor.prepare(frame, frame, (np.eye(3), np.eye(3)), clock=lambda: next(times))

  def test_calibration_change_after_warmup_stops_instead_of_compiling_in_stream(self):
    layout = NV12Layout.packed(512, 256)
    processor = PairPreprocessor(layout, layout)
    processor.warmup((np.eye(3), np.eye(3)))
    changed = np.eye(3)
    changed[0, 2] = 1
    frame = OwnedCameraFrame(0, 100, bytes(layout.nbytes))
    with self.assertRaisesRegex(ValueError, 'calibration changed'):
      processor.prepare(frame, frame, (changed, np.eye(3)), clock=lambda: 1000)
    self.assertTrue(processor.failed)


if __name__ == '__main__':
  unittest.main()
