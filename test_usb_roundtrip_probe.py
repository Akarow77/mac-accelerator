"""Pure local helper tests; no device connection."""

import socket
import unittest
from unittest.mock import patch

from tools.probe_usb_roundtrip import CLIENT, client_bootstrap, read_exact, scoped_link_local


class UsbProbeTests(unittest.TestCase):
  def test_legacy_adb_bootstrap_preserves_source(self):
    namespace = {}
    expected = '왕복 "test"'
    exec(client_bootstrap(f'result = {expected!r}'), namespace)
    self.assertEqual(namespace['result'], expected)
    self.assertLess(len(client_bootstrap(CLIENT)), 3500)

  def test_embedded_device_client_compiles(self):
    compile(CLIENT, '<usb-bench-client>', 'exec')

  def test_absolute_receive_deadline(self):
    class Slow:
      def settimeout(self, value):
        self.timeout = value

      def recv_into(self, view):
        view[0] = 1
        return 1
    with patch('tools.probe_usb_roundtrip.time.monotonic', side_effect=[1., 1.5, 3.1]):
      with self.assertRaises(TimeoutError):
        read_exact(Slow(), bytearray(3))

  def test_requires_scoped_link_local(self):
    self.assertEqual(scoped_link_local('fe80::1%en7'), 'fe80::1%en7')
    for value in ('::1', 'fe80::1', 'fe80::1%', '2001:db8::1%en7', '127.0.0.1%en7'):
      with self.subTest(value=value), self.assertRaises(ValueError):
        scoped_link_local(value)

  def test_receives_into_owned_buffer(self):
    sender, receiver = socket.socketpair()
    with sender, receiver:
      sender.sendall(b'abcde')
      data = bytearray(5)
      read_exact(receiver, data)
      self.assertEqual(data, b'abcde')

  def test_rejects_truncated_data(self):
    sender, receiver = socket.socketpair()
    with sender, receiver:
      sender.sendall(b'ab')
      sender.shutdown(socket.SHUT_WR)
      with self.assertRaises(EOFError):
        read_exact(receiver, bytearray(5))

  def test_fragmented_reads(self):
    class Fragmented:
      def settimeout(self, value):
        if not 0 < value <= 2:
          raise ValueError('unbounded timeout')

      def recv_into(self, view):
        view[0] = 42
        return 1
    data = bytearray(5)
    read_exact(Fragmented(), data)
    self.assertEqual(data, b'*****')


if __name__ == '__main__':
  unittest.main()
