"""Offline prototype checks; importing this never changes the inference server."""

import unittest

import numpy as np

from tools.probe_feature_history import CircularHistory, verify


class FeatureHistoryTests(unittest.TestCase):
  def test_sequence_reset_and_wraparound(self):
    self.assertEqual(verify(), 1200)

  def test_actual_model_dimensions_and_owned_inputs(self):
    ring = CircularHistory(32, 4, (32, 512))
    legacy = np.zeros_like(ring.storage)
    output = np.empty((32, 32, 512), dtype=np.float32)
    feature = np.empty((32, 512), dtype=np.float32)
    for frame in range(140):
      ring.read_into(output)
      np.testing.assert_array_equal(output, legacy[::4])
      feature.fill(frame + 1)
      legacy[:-1] = legacy[1:]
      legacy[-1] = feature
      ring.append(feature)
      feature.fill(-999)  # caller reuse must not mutate retained features
    ring.reset()
    ring.read_into(output)
    self.assertEqual(ring.head, 0)
    self.assertFalse(np.any(output))
    self.assertFalse(np.any(ring.storage))

  def test_invalid_dimensions(self):
    for args in ((0, 4, (2,)), (1, 0, (2,)), (True, 4, (2,)),
                 (2, 1.5, (2,)), (2, 4, (0,)), (2, 4, ())):
      with self.subTest(args=args), self.assertRaises(ValueError):
        CircularHistory(*args)

  def test_bad_append_leaves_state_unchanged(self):
    ring = CircularHistory(2, 1, (3,))
    for feature in (np.ones(1, dtype=np.float32), np.ones(3, dtype=np.float64),
                    ring.storage[0], [1, 2, 3]):
      with self.subTest(feature=type(feature)), self.assertRaises(ValueError):
        ring.append(feature)
      self.assertEqual(ring.head, 0)
      self.assertFalse(np.any(ring.storage))

  def test_bad_output_leaves_state_unchanged(self):
    ring = CircularHistory(2, 1, (3,))
    readonly = np.empty((2, 3), dtype=np.float32)
    readonly.flags.writeable = False
    for output in (ring.storage, readonly, np.empty((2, 3), dtype=np.float64),
                   np.empty((1, 3), dtype=np.float32), []):
      with self.subTest(output=type(output)), self.assertRaises(ValueError):
        ring.read_into(output)
      self.assertEqual(ring.head, 0)
      self.assertFalse(np.any(ring.storage))

  def test_single_slot_reads_before_write(self):
    ring = CircularHistory(1, 1, (1,))
    output = np.empty((1, 1), dtype=np.float32)
    for frame in range(5):
      ring.read_into(output)
      np.testing.assert_array_equal(output, [[frame]])
      ring.append(np.array([frame + 1], dtype=np.float32))
      np.testing.assert_array_equal(output, [[frame]])


if __name__ == '__main__':
  unittest.main()
