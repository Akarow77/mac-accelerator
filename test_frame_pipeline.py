"""Bounded pipeline ordering/failure tests with fake stages, no GPU access."""

import threading
import unittest

import numpy as np

try:
  from tools.benchmark_frame_pipeline import FramePipeline
except ModuleNotFoundError as error:
  if error.name != 'coremltools':
    raise
  FramePipeline = None
from transport import WARPED_SHAPE


@unittest.skipIf(FramePipeline is None, 'coremltools is installed only in .coreml-venv')
class PipelineTests(unittest.TestCase):
  def test_next_image_stage_overlaps_current_policy_and_preserves_storage(self):
    next_started = threading.Event()
    calls = []
    class GPU:
      def predict(self, inputs):
        frame = len(calls)
        calls.append(inputs['img'].copy())
        if frame == 1:
          next_started.set()
        return {'feature': np.array([frame], dtype=np.float32)}
    class ANE:
      def predict(self, inputs):
        if inputs['feature'][0] == 0:
          if not next_started.wait(2):
            raise TimeoutError('next frame never started during current policy')
          np.testing.assert_array_equal(inputs['feature'], [0])
        return {'output': inputs['feature'].copy() + inputs['state']}
    pixels = np.stack([np.full(WARPED_SHAPE, i + 1, dtype=np.uint8) for i in range(5)])
    pipeline = FramePipeline(GPU(), ANE(), ['feature'], ['state'], pixels, 5)
    pipeline.start()
    try:
      results = [pipeline.predict({'state': np.array([100.])})['output'] for _ in range(5)]
    finally:
      pipeline.close()
    np.testing.assert_array_equal(np.array(results).ravel(), np.arange(100., 105.))
    self.assertEqual([row['frame'] for row in pipeline.rows], list(range(5)))
    self.assertTrue(np.all(calls[3][:, :6] == 0))
    self.assertTrue(np.all(calls[4][:, :6] == 1))
    self.assertTrue(np.all(calls[4][:, 6:] == 5))
    self.assertFalse(pipeline.thread.is_alive())

  def test_gpu_failure_propagates_without_retry(self):
    class Broken:
      def predict(self, inputs):
        raise ValueError('injected failure')
    pipeline = FramePipeline(Broken(), None, ['feature'], [], np.zeros((1, *WARPED_SHAPE), dtype=np.uint8), 1)
    pipeline.start()
    try:
      with self.assertRaisesRegex(RuntimeError, 'GPU stage failed'):
        pipeline.predict({})
    finally:
      pipeline.close()

  def test_only_two_frames_admitted_before_consumer(self):
    two_started = threading.Event()
    calls = []
    class GPU:
      def predict(self, inputs):
        calls.append(1)
        if len(calls) == 2:
          two_started.set()
        return {'feature': np.array([len(calls)])}
    pipeline = FramePipeline(GPU(), None, ['feature'], [], np.zeros((1, *WARPED_SHAPE), dtype=np.uint8), 100)
    pipeline.start()
    try:
      self.assertTrue(two_started.wait(2))
      # Producer cannot acquire a third slot until a consumer finishes.
      self.assertFalse(pipeline.slots.acquire(blocking=False))
      self.assertEqual(len(calls), 2)
    finally:
      pipeline.close()
    self.assertFalse(pipeline.thread.is_alive())

  def test_ane_failure_does_not_leave_producer_running(self):
    class GPU:
      def predict(self, inputs):
        return {'feature': np.array([1.])}
    class ANE:
      def predict(self, inputs):
        raise ValueError('policy failure')
    pipeline = FramePipeline(GPU(), ANE(), ['feature'], [], np.zeros((1, *WARPED_SHAPE), dtype=np.uint8), 100)
    pipeline.start()
    try:
      with self.assertRaisesRegex(ValueError, 'policy failure'):
        pipeline.predict({})
    finally:
      pipeline.close()
    self.assertFalse(pipeline.thread.is_alive())

  def test_saturated_dispatch_precedes_ane_without_waiting_for_next_gpu_result(self):
    submission = threading.Condition()
    current = [-1]
    policy_started = threading.Event()
    class GPU:
      def predict(self, inputs):
        with submission:
          current[0] += 1
          frame = current[0]
          submission.notify_all()
        if frame == 1 and not policy_started.wait(2):
          raise TimeoutError('policy did not run concurrently with next GPU stage')
        return {'feature': np.array([frame])}

      def wait_submitted(self, frame):
        with submission:
          if not submission.wait_for(lambda: current[0] >= frame, timeout=2):
            raise TimeoutError('dispatch was not made')
    class ANE:
      def predict(self, inputs):
        policy_started.set()
        return {'output': inputs['feature'].copy()}
    pipeline = FramePipeline(GPU(), ANE(), ['feature'], [], np.zeros((1, *WARPED_SHAPE), dtype=np.uint8), 2)
    pipeline.start()
    try:
      np.testing.assert_array_equal(pipeline.predict({})['output'], [0])
      np.testing.assert_array_equal(pipeline.predict({})['output'], [1])
    finally:
      pipeline.close()


if __name__ == '__main__':
  unittest.main()
