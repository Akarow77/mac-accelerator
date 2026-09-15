#!/usr/bin/env python3
"""Short isolated runtime test; 500ms diagnostic deadline, NOT timing qualification."""

import argparse
import hashlib
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--frames', type=int, default=20)
  args = parser.parse_args()
  if args.frames < 1:
    parser.error('--frames must be positive')
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
    try:
      expires = time.monotonic() + 90
      while 'listening on ' not in log.read_text():
        if server.poll() is not None or time.monotonic() > expires:
          raise RuntimeError('Server did not become ready:\n' + log.read_text()[-5000:])
        time.sleep(0.1)
      print('Independent localhost server ready; no device accessed.', flush=True)
      result = subprocess.run([
        sys.executable, str(root / 'synthetic_client.py'), '::1', '--port', str(port),
        '--auth-key-file', str(key), '--expected-model-sha256', digest,
        '--warmup-frames', '3', '--qualification-frames', '2', '--deadline-ms', '500',
        '--frames', str(args.frames), '--synthetic-random-prefix-bytes', '393216',
      ], cwd=root, env=environment, timeout=60, check=False)
      if result.returncode:
        raise RuntimeError(f'Client failed with exit code {result.returncode}')
      print('Plumbing smoke passed. This is NOT a 50ms or road-use qualification.')
    finally:
      if server.poll() is None:
        server.terminate()
        try:
          server.wait(timeout=5)
        except subprocess.TimeoutExpired:
          server.kill()
          server.wait(timeout=5)


if __name__ == '__main__':
  main()
