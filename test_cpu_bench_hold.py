"""Simulate sysfs; these tests never write device or host CPU controls."""

import contextlib
import io
import unittest
from unittest.mock import patch

from tools.hold_big_cores_bench import GUARD


class CpuHoldTests(unittest.TestCase):
  def simulate(self, states, temperatures, ticks):
    values = {f'/sys/devices/system/cpu/cpu{i}/online': '0' for i in range(4, 8)}
    state_iter, temperature_iter = iter(states), iter(temperatures)
    class FakePath:
      def __init__(self, path):
        self.path = path

      def read_text(self):
        if self.path.endswith('IsOffroad'):
          return next(state_iter)
        if self.path.endswith('/temp'):
          return str(next(temperature_iter))
        return values[self.path]

      def write_text(self, value):
        values[self.path] = value
    output = io.StringIO()
    with patch('pathlib.Path', FakePath), patch('os.geteuid', return_value=0), \
         patch('sys.argv', ['guard', '1']), patch('signal.signal'), \
         patch('time.monotonic', side_effect=ticks), contextlib.redirect_stdout(output):
      exec(compile(GUARD, '<guard-test>', 'exec'), {})
    return values, output.getvalue()

  def test_expiry_restores_original_offline_states(self):
    values, output = self.simulate(['1', '1'], [45000] * 8, [0, 2])
    self.assertEqual(set(values.values()), {'0'})
    self.assertIn('"restored": true', output)

  def test_onroad_transition_does_not_offline_cores(self):
    values, output = self.simulate(['1', '0', '0'], [45000] * 8, [0, 0.5])
    self.assertEqual(set(values.values()), {'1'})
    self.assertIn('left offroad', output)
    self.assertIn('"restored": false', output)

  def test_temperature_stop_restores_states(self):
    values, output = self.simulate(['1', '1', '1'], [45000] * 8 + [76000] * 8, [0, 0.5])
    self.assertEqual(set(values.values()), {'0'})
    self.assertIn('temperature limit', output)


if __name__ == '__main__':
  unittest.main()
