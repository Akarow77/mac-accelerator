"""Transport bounds for the isolated 8 MiB camera experiment."""

import hashlib
import unittest
from unittest.mock import patch

import transport
from accelerator_protocol import authenticate_payload, authenticate_payload_parts, verify_payload


class Sink:
  def gettimeout(self):
    return None

  def sendmsg(self, parts):
    return sum(map(len, parts))


class CameraTransportTests(unittest.TestCase):
  def test_production_default_still_rejects_camera_buffer(self):
    self.assertEqual(transport.MAX_PAYLOAD_BYTES, 4 * 1024 * 1024)
    with self.assertRaises(transport.ProtocolError):
      transport.send_message(Sink(), transport.REQUEST, 0, 1, 0, 0, bytes(4804608))

  def test_explicit_bench_cap_accepts_camera_and_bounds_8mib(self):
    with patch.object(transport, 'MAX_PAYLOAD_BYTES', 8 * 1024 * 1024):
      for length in (4804608, 8 * 1024 * 1024):
        transport.send_message(Sink(), transport.REQUEST, 0, 1, 0, 0, bytes(length))
      with self.assertRaises(transport.ProtocolError):
        transport.send_message(Sink(), transport.REQUEST, 0, 1, 0, 0, bytes(8 * 1024 * 1024 + 1))
    self.assertEqual(transport.MAX_PAYLOAD_BYTES, 4 * 1024 * 1024)

  def test_oversized_receive_rejected_before_payload_allocation(self):
    header = transport.HEADER.pack(transport.MAGIC, transport.VERSION, transport.REQUEST,
                                   0, 1, 0, 0, 8 * 1024 * 1024 + 1, 0)
    with patch.object(transport, 'MAX_PAYLOAD_BYTES', 8 * 1024 * 1024), \
         patch.object(transport, 'recv_exact', return_value=header) as read:
      with self.assertRaises(transport.ProtocolError):
        transport.recv_message(Sink(), transport.REQUEST)
      read.assert_called_once_with(unittest.mock.ANY, transport.HEADER.size, None)

  def test_scatter_gather_auth_matches_contiguous_camera(self):
    pixels = bytes(4804608)
    parts = (b'header', memoryview(pixels))
    key = b'test-only-key' * 3
    expected = authenticate_payload(key, transport.REQUEST, 0, 1, 2, 3, b''.join(parts))
    actual = b''.join(authenticate_payload_parts(key, transport.REQUEST, 0, 1, 2, 3, parts))
    self.assertEqual(actual, expected)
    decoded = verify_payload(key, transport.REQUEST, 0, 1, 2, 3, actual)
    self.assertEqual(hashlib.sha256(decoded).digest(), hashlib.sha256(b''.join(parts)).digest())


if __name__ == '__main__':
  unittest.main()
