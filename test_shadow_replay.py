import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from shadow_replay import BoundedLog, ShadowRecorder, validate_identity
from model_contract import BIG_MODEL_CONTEXT_FRAMES, BIG_MODEL_FRAME_SKIP
from transport import POLICY_INPUTS, WARPED_BYTES


class ShadowTests(unittest.TestCase):
  def setUp(self):
    self.client, self.log = Mock(), Mock()
    self.client.infer.return_value = Mock(output=np.zeros(18452, dtype='<f4').tobytes(),
                                          round_trip_ms=25., inference_ms=20., prepare_ms=.2, send_ms=.3,
                                          receive_ms=24., validate_ms=.5, server_prepare_ms=.1)
    self.clock = Mock(side_effect=[1_000_000_000, 1_025_000_000])
    self.recorder = ShadowRecorder(self.client, self.log, clock=self.clock)
    self.pixels, self.policy = bytes(WARPED_BYTES), bytes(POLICY_INPUTS.size)

  def process(self, frame=0, stamp=1_000_000_000, **kwargs):
    self.recorder.process(frame, stamp, kwargs.get('pixels', self.pixels), kwargs.get('policy', self.policy))

  def test_good_frame_is_observation_only_without_context(self):
    self.process()
    row = self.log.write.call_args.args[0]
    self.assertFalse(row['roadReady'])
    self.assertFalse(row['contextReady'])
    self.assertTrue(row['withinTimingTarget'])
    self.assertEqual(row['clientStagesMs']['receive_ms'], 24.)
    self.assertEqual(row['nonInferenceRoundTripMs'], 5.)
    self.assertEqual(self.client.infer.call_args.kwargs['deadline_ns'], 1_150_000_000)

  def test_slow_frame_is_counted_not_marked_timing_pass(self):
    self.recorder.context_frames = 1
    self.clock.side_effect = [1_000_000_000, 1_060_000_000]
    self.process()
    self.assertFalse(self.log.write.call_args.args[0]['withinTimingTarget'])
    self.assertEqual(self.recorder.misses, 1)

  def test_source_gap_is_latched_before_inference(self):
    with self.assertRaisesRegex(ValueError, 'gap'):
      self.process(frame=1)
    self.client.infer.assert_not_called()
    self.client.close.assert_called_once()
    with self.assertRaisesRegex(RuntimeError, 'latched'):
      self.process()

  def test_stale_input_rejected_before_inference(self):
    with self.assertRaisesRegex(TimeoutError, 'already stale'):
      self.process(stamp=800_000_000)
    self.client.infer.assert_not_called()

  def test_future_clock_rejected_before_inference(self):
    with self.assertRaisesRegex(ValueError, 'clock'):
      self.process(stamp=1_000_000_001)
    self.client.infer.assert_not_called()

  def test_stale_output_is_latched(self):
    self.clock.side_effect = [1_000_000_000, 1_151_000_000]
    with self.assertRaisesRegex(TimeoutError, 'became stale'):
      self.process()
    self.assertEqual(self.log.write.call_args.args[0]['event'], 'failed')

  def test_nonfinite_policy_rejected(self):
    with self.assertRaisesRegex(ValueError, 'non-finite'):
      self.process(policy=POLICY_INPUTS.pack(*([float('nan')] * 12)))
    self.client.infer.assert_not_called()

  def test_invalid_dimensions_rejected(self):
    with self.assertRaisesRegex(ValueError, 'dimensions'):
      self.process(pixels=b'bad')
    self.client.infer.assert_not_called()

  def test_invalid_output_latched(self):
    self.client.infer.return_value.output = np.full(18452, np.inf, dtype='<f4').tobytes()
    with self.assertRaisesRegex(ValueError, 'output'):
      self.process()
    self.client.close.assert_called_once()

  def test_disconnect_closes_client_and_records_failure(self):
    self.client.infer.side_effect = EOFError('peer closed')
    with self.assertRaises(EOFError):
      self.process()
    self.assertTrue(self.recorder.failed)
    self.assertIn('peer closed', self.log.write.call_args.args[0]['reason'])

  def test_logging_failure_still_closes_client(self):
    self.log.write.side_effect = OSError('disk full')
    with self.assertRaises(OSError):
      self.process()
    self.client.close.assert_called_once()
    self.assertTrue(self.recorder.failed)

  def test_wrong_temporal_contract_rejected(self):
    with self.assertRaisesRegex(ValueError, 'contract'):
      validate_identity(Mock(frame_skip=1))

  def test_stride_four_contract_and_full_context(self):
    identity = Mock(frame_skip=BIG_MODEL_FRAME_SKIP,
                    input_shapes={'img': [1, 12, 128, 256], 'big_img': [1, 12, 128, 256],
                                  'desire_pulse': [1, 33, 8], 'traffic_convention': [1, 2],
                                  'action_t': [1, 2], 'features_buffer': [1, 32, 32, 512]},
                    output_shapes={'outputs': [1, 18452]})
    validate_identity(identity)
    identity.frame_skip = 2
    with self.assertRaisesRegex(ValueError, 'contract'):
      validate_identity(identity)
    self.assertEqual(self.recorder.context_frames, 132)
    self.assertEqual(BIG_MODEL_CONTEXT_FRAMES, 132)
    for frame in range(132):
      stamp = 1_000_000_000 + frame * 50_000_000
      self.clock.side_effect = [stamp, stamp + 25_000_000]
      self.process(frame=frame, stamp=stamp)
      self.assertEqual(self.log.write.call_args.args[0]['contextReady'], frame == 131)
    self.assertEqual(self.recorder.eligible, 1)

  def test_bounded_private_log_never_overwrites_and_reserves_failure(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'shadow.jsonl'
      log = BoundedLog(path, 8192)
      try:
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
          BoundedLog(path)
        with self.assertRaisesRegex(RuntimeError, 'limit'):
          log.write({'payload': 'x' * 4096})
        log.write({'event': 'failed'}, terminal=True)
      finally:
        log.close()
      self.assertEqual(json.loads(path.read_text()), {'event': 'failed'})


if __name__ == '__main__':
  unittest.main()
