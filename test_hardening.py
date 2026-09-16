import unittest
from unittest.mock import Mock, patch

import numpy as np

from accelerator_client import AcceleratorClient, AcceleratorIdentity
from transport import (DEVICE_TYPE, POLICY_INPUTS, REQUEST_BYTES, RESPONSE, TIMINGS,
                       WARPED_BYTES, _send_parts, send_message_parts)


class TransportHardeningTests(unittest.TestCase):
  def test_partial_send_uses_one_absolute_deadline(self):
    sock = Mock()
    sock.sendmsg.return_value = 1
    with patch('transport.time.monotonic_ns', side_effect=[1_000_000, 2_000_000, 11_000_000]):
      with self.assertRaisesRegex(TimeoutError, 'absolute deadline'):
        _send_parts(sock, [memoryview(b'123456')], deadline_ns=10_000_000)
    self.assertEqual(sock.sendmsg.call_count, 2)
    self.assertEqual(sock.settimeout.call_args_list[0].args[0], .009)
    self.assertEqual(sock.settimeout.call_args_list[1].args[0], .008)

  def test_implicit_socket_timeout_becomes_absolute(self):
    sock = Mock()
    sock.gettimeout.return_value = .01
    with patch('transport.time.monotonic_ns', return_value=100), patch('transport._send_parts') as send:
      send_message_parts(sock, RESPONSE, 0, 1, 0, 0, (b'x',))
    self.assertEqual(send.call_args.args[2], 10_000_100)

  def test_fp16_results_do_not_alias_later_inference(self):
    client = AcceleratorClient('::1', deadline_ms=100, expected_backend='COREML_ANE')
    client.socket = Mock()
    client.identity = AcceleratorIdentity(DEVICE_TYPE, 'COREML_ANE', '', 'a' * 64,
                                         {}, {}, 2, REQUEST_BYTES, output_dtype='float16')
    client._prepare_output_buffers(client.identity)
    def response(value, frame):
      return (0, client.session_id, frame, 1,
              TIMINGS.pack(1, 2, 3) + np.array([value, value], dtype='<f2').tobytes())
    with patch('accelerator_client.send_message_parts'), \
         patch('accelerator_client.recv_message', side_effect=[response(1, 0), response(2, 1)]):
      first = client.infer(bytes(WARPED_BYTES), bytes(POLICY_INPUTS.size), frame_id=0, capture_ns=1)
      saved = bytes(first.output)
      client.infer(bytes(WARPED_BYTES), bytes(POLICY_INPUTS.size), frame_id=1, capture_ns=1)
    self.assertIsInstance(first.output, bytes)
    self.assertEqual(first.output, saved)
    self.assertEqual(np.frombuffer(first.output, dtype='<f4').tolist(), [1, 1])


if __name__ == '__main__':
  unittest.main()
