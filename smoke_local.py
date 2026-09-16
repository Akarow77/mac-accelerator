#!/usr/bin/env python3
"""Isolated runtime smoke tests with diagnostic limits, NOT real-time qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--frames', type=int, default=20)
  parser.add_argument('--shadow-log', type=Path, help='run observation-only replay and retain its private log')
  parser.add_argument('--nv12', action='store_true', help='include owned synthetic NV12 pair preprocessing in shadow timing')
  parser.add_argument('--preprocess-profile', type=Path)
  parser.add_argument('--disconnect-after', type=float, help='shadow-only test: terminate this test server after N seconds')
  args = parser.parse_args()
  if args.frames < 1:
    parser.error('--frames must be positive')
  if args.disconnect_after is not None and (not args.shadow_log or not 0 < args.disconnect_after < 30):
    parser.error('--disconnect-after requires --shadow-log and 0 < seconds < 30')
  if (args.nv12 and not args.shadow_log) or (args.preprocess_profile and not args.nv12):
    parser.error('--nv12 requires --shadow-log; --preprocess-profile requires --nv12')
  root = Path(__file__).resolve().parent
  with (root / 'models/big_driving_supercombo.onnx').open('rb') as handle:
    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
  with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
    probe.bind(('::1', 0))
    port = probe.getsockname()[1]
  with tempfile.TemporaryDirectory(prefix='mac-accelerator-smoke-') as directory:
    temporary = Path(directory)
    key = temporary / 'auth.key'
    key.write_bytes(secrets.token_bytes(32))
    key.chmod(0o600)
    log = temporary / 'server.log'
    environment = {**os.environ, 'AUTH_KEY_FILE': str(key), 'PORT': str(port),
                   'MAC_ACCELERATOR_LOCAL_TEST': '1', 'PYTHONUNBUFFERED': '1',
                   'VECLIB_MAXIMUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    # Do not let an existing app/shell override this standalone project's models.
    for name in ('MODEL', 'ONNX', 'METADATA', 'COREML_TMPDIR', 'PYTHONPATH', 'PYTHONHOME'):
      environment.pop(name, None)
    with log.open('w') as output:
      server = subprocess.Popen([str(root / 'run_coreml_server.sh')], cwd=root,
                                env=environment, stdout=output, stderr=subprocess.STDOUT)
    timer = None
    try:
      expires = time.monotonic() + 90
      while 'listening on ' not in log.read_text():
        if server.poll() is not None or time.monotonic() > expires:
          raise RuntimeError('Server did not become ready:\n' + log.read_text()[-5000:])
        time.sleep(0.1)
      print('Independent localhost server ready; no device accessed.', flush=True)
      command = [
        sys.executable, str(root / 'synthetic_client.py'), '::1', '--port', str(port),
        '--auth-key-file', str(key), '--expected-model-sha256', digest,
        '--warmup-frames', '3', '--qualification-frames', '2', '--deadline-ms', '500',
        '--frames', str(args.frames), '--synthetic-random-prefix-bytes', '393216',
      ]
      if args.shadow_log:
        command = [sys.executable, str(root / 'shadow_replay.py'), '--synthetic-nv12' if args.nv12 else '--synthetic', '--port', str(port),
                   '--auth-key-file', str(key), '--expected-model-sha256', digest,
                   '--frames', str(args.frames), '--log', str(args.shadow_log.resolve())]
        if args.preprocess_profile:
          command.extend(['--preprocess-profile', str(args.preprocess_profile.resolve())])
      if args.disconnect_after is not None:
        timer = threading.Timer(args.disconnect_after, server.terminate)
        timer.start()
      result = subprocess.run(command, cwd=root, env=environment, timeout=60, check=False)
      if args.disconnect_after is not None:
        if result.returncode == 0 or not args.shadow_log.exists():
          raise RuntimeError('Disconnect injection did not produce a recorded failure')
        terminal = json.loads(args.shadow_log.read_text().splitlines()[-1])
        reason = terminal.get('reason', '')
        network_failure = any(marker in reason for marker in ('peer closed', 'Connection reset by peer', 'Broken pipe'))
        if terminal.get('event') != 'failed' or not network_failure or server.poll() is None:
          raise RuntimeError(f'Disconnect stopped for an unexpected reason: {terminal}')
        print('Injected server disconnect correctly latched failure; no automatic restart.')
        return
      if result.returncode:
        raise RuntimeError(f'Client failed with exit code {result.returncode}')
      print('Plumbing smoke passed. This is NOT a 50ms or road-use qualification.')
    finally:
      if timer is not None:
        timer.cancel()
        timer.join()
      if server.poll() is None:
        server.terminate()
        try:
          server.wait(timeout=5)
        except subprocess.TimeoutExpired:
          server.kill()
          server.wait(timeout=5)


if __name__ == '__main__':
  main()
