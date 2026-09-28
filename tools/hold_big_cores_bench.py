"""Explicit detached/offroad CPU-online bench lease; never changes governors.

Temporary device sysfs writes are intentional, user-authorized controls, not a
firmware edit. Restores original online states when still offroad. Never offlines
cores if the device has transitioned onroad; yields to its hardware manager.
"""

import argparse
import shlex
import subprocess

if __package__:
  from .probe_usb_roundtrip import client_bootstrap
else:
  from probe_usb_roundtrip import client_bootstrap


GUARD = r'''
import json, os, signal, sys, time
from pathlib import Path
duration = int(sys.argv[1])
if not 1 <= duration <= 120:
  raise ValueError('duration must be 1..120 seconds')
offroad = Path('/data/params/d/IsOffroad')
cores = [Path(f'/sys/devices/system/cpu/cpu{i}/online') for i in range(4, 8)]
zones = [Path(f'/sys/class/thermal/thermal_zone{i}/temp') for i in (1,2,3,4,7,8,9,10)]
def temps():
  values = [int(p.read_text()) / 1000 for p in zones]
  if not all(-10 < v < 120 for v in values):
    raise RuntimeError('invalid CPU temperature reading')
  return values
if os.geteuid() != 0 or offroad.read_text().strip() != '1':
  raise RuntimeError('requires root on a detached offroad device')
original = [p.read_text().strip() for p in cores]
if any(v not in ('0', '1') for v in original):
  raise RuntimeError('invalid online state')
initial = temps()
if max(initial) >= 75:
  raise RuntimeError('CPU too hot to start')
def stopped(signum, frame):
  global reason
  reason = f'signal {signum}'
  raise SystemExit(f'signal {signum}')
signal.signal(signal.SIGTERM, stopped)
signal.signal(signal.SIGINT, stopped)
peak = max(initial)
reason = 'duration complete'
writes = 0
try:
  for p in cores:
    p.write_text('1')
    writes += 1
  print(json.dumps({'status':'holding', 'pid':os.getpid(), 'seconds':duration, 'original':original,
                    'initialMaxCpuC':peak}), flush=True)
  end = time.monotonic() + duration
  while time.monotonic() < end:
    if offroad.read_text().strip() != '1':
      reason = 'left offroad; yield to hardware manager'
      break
    peak = max(peak, max(temps()))
    if peak >= 75:
      reason = 'temperature limit'
      break
    for p in cores:
      if p.read_text().strip() != '1':
        p.write_text('1')
        writes += 1
    time.sleep(0.25)
finally:
  restored = False
  errors = []
  if offroad.read_text().strip() == '1':
    for p, value in zip(cores, original):
      try:
        p.write_text(value)
      except OSError as error:
        errors.append(str(error))
    restored = not errors and [p.read_text().strip() for p in cores] == original
  print(json.dumps({'status':'released', 'reason':reason, 'restored':restored,
                    'peakCpuC':peak, 'onlineWrites':writes, 'restoreErrors':errors}), flush=True)
'''


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--serial', required=True)
  parser.add_argument('--seconds', type=int, choices=range(1, 121), default=75)
  args = parser.parse_args()
  command = shlex.join(['python3', '-u', '-c', client_bootstrap(GUARD), str(args.seconds)])
  subprocess.run(['/opt/homebrew/bin/adb', '-s', args.serial, 'exec-out', command],
                 check=True, timeout=args.seconds + 15)


if __name__ == '__main__':
  main()
