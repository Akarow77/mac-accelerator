"""Bounded, read-only ADB diagnostics; local private files, no images or keys."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


COMMANDS = {
  'boot.txt': 'date -u; uptime; cat /proc/sys/kernel/random/boot_id; cat /proc/uptime; uname -a; cat /etc/os-release; cat /VERSION',
  'git.txt': 'sudo -u comma git --no-pager -C /data/openpilot status --short --branch; sudo -u comma git --no-pager -C /data/openpilot log -1 --format="%H %s"; sudo -u comma git --no-pager -C /data/openpilot branch -v; sudo -u comma git -C /data/openpilot submodule status',
  'dmesg.txt': 'dmesg',
  'journal-current-boot.txt': 'journalctl -b -n 3000 --no-pager -o short-monotonic',
  'boot-history.txt': 'journalctl --list-boots --no-pager; last -x -n 20',
  'processes.txt': 'ps -eo pid,ppid,stat,comm; sudo -u comma env TERM=xterm tmux capture-pane -p -t comma:0.0 -S -500',
  'power-cpu.txt': 'for p in /sys/class/power_supply/*; do echo "$p"; cat "$p/uevent"; done; cat /sys/devices/system/cpu/online; for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "$p"; for f in scaling_governor scaling_min_freq scaling_max_freq scaling_cur_freq cpuinfo_max_freq; do echo "$f"; cat "$p/$f"; done; done; for p in /sys/class/thermal/thermal_zone*; do echo "$p"; cat "$p/type" "$p/temp"; done',
  'selected-state.txt': 'for f in IsOffroad DoReboot DoShutdown DoUninstall; do echo "$f"; cat "/data/params/d/$f"; echo; done; ip link show usb0; ip -6 addr show dev usb0',
  'startup-config.txt': 'cat /usr/comma/comma.sh /usr/comma/init.qcom.sh /data/openpilot/launch_env.sh',
}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--serial', required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  os.umask(0o077)
  args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
  adb = ['/opt/homebrew/bin/adb', '-s', args.serial]
  manifest = {'createdUnix': time.time(), 'scope': 'diagnostic logs only; no images, auth keys or full params',
              'commands': {}, 'copies': {}, 'files': {}}
  for name, command in COMMANDS.items():
    with (args.output / name).open('wb') as target:
      try:
        result = subprocess.run(adb + ['exec-out', command], stdout=target, stderr=subprocess.STDOUT, timeout=15)
        manifest['commands'][name] = {'returncode': result.returncode}
      except subprocess.TimeoutExpired:
        manifest['commands'][name] = {'timeout': True}
    print(name, manifest['commands'][name], flush=True)
  for remote in ('/sys/fs/pstore', '/tmp/launch_log', '/data/community/crashes'):
    try:
      result = subprocess.run(adb + ['pull', remote, str(args.output) + '/'], capture_output=True, text=True, timeout=20)
      manifest['copies'][remote] = {'returncode': result.returncode, 'summary': (result.stdout + result.stderr)[-1000:]}
    except subprocess.TimeoutExpired:
      manifest['copies'][remote] = {'timeout': True}
    print(remote, manifest['copies'][remote], flush=True)
  for path in sorted(args.output.rglob('*')):
    if path.is_file():
      path.chmod(0o600)
      manifest['files'][str(path.relative_to(args.output))] = {
        'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
  (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
  print(json.dumps({'output': str(args.output.resolve()), 'fileCount': len(manifest['files']),
                    'bytes': sum(x['bytes'] for x in manifest['files'].values())}), flush=True)


if __name__ == '__main__':
  main()
