"""Detached 3X authenticated transport + Mac inference; temporary files only.

Uses unchanged current production framing/auth/client and CoreML serve_client.
High-core hold is explicit, bounded and restored; no onroad mode is enabled.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re
import secrets
import select
import shlex
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accelerator_protocol import authenticate_payload, server_handshake, verify_payload
from coreml_inference_server import CoreMLPolicySession, make_identity, serve_client, sha256_file
from macos_performance import configure_user_interactive_qos
from model_contract import BIG_MODEL_FRAME_SKIP
from tools.hold_big_cores_bench import GUARD
from tools.probe_usb_roundtrip import client_bootstrap, scoped_link_local
from transport import FLAG_RESET, REQUEST, REQUEST_BYTES, RESPONSE, TIMINGS, recv_message, send_message
import transport

ROOT = Path(__file__).resolve().parents[1]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--serial', required=True)
  parser.add_argument('--mac', required=True, type=scoped_link_local)
  parser.add_argument('--device', required=True, type=scoped_link_local)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--camera-snapshot-only', action='store_true',
                      help='real road-camera snapshots only; stops camerad before transfer, no model')
  parser.add_argument('--camera-compare-8mib', action='store_true',
                      help='bench-only 8 MiB cap; same snapshots in chunked/single/iovec/iovec/single/chunked order')
  args = parser.parse_args()
  if args.camera_compare_8mib:
    if not args.camera_snapshot_only:
      parser.error('--camera-compare-8mib requires --camera-snapshot-only')
    # Isolated benchmark process only; production transport default stays 4 MiB.
    transport.MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
  adb = ['/opt/homebrew/bin/adb', '-s', args.serial]
  report = {'scope': 'authenticated physical USB synthetic input; no camera or vehicle control',
            'roadReady': False, 'runs': [], 'status': 'starting'}
  remote = None
  guard = None
  guard_pid = None
  sources = [ROOT / (name + '.py') for name in
             ('accelerator_client', 'accelerator_protocol', 'compression', 'fallback', 'transport', 'model_contract')]
  sources.append(ROOT / 'tools/usb_authenticated_client.py')
  if args.camera_snapshot_only:
    sources.append(ROOT / 'tools/usb_camera_snapshot_client.py')
    report['scope'] = 'two real road-camera snapshots; no image files, cabin camera or inference'
  def shell(command, timeout=10):
    result = subprocess.run(adb + ['shell', command], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
      raise RuntimeError('ADB command failed: ' + (result.stderr or result.stdout)[-600:])
    return result.stdout.strip()
  with args.output.open('x') as output, tempfile.TemporaryDirectory(prefix='macbench-auth-') as temporary:
    try:
      if shell('cat /data/params/d/IsOffroad') != '1':
        raise RuntimeError('device is not offroad')
      identity = {}
      if not args.camera_snapshot_only:
        print('Loading existing trusted Core ML artifact...', flush=True)
        import coremltools as ct
        with (ROOT / 'models/big_driving_supercombo_metadata.pkl').open('rb') as f:
          metadata = pickle.load(f)
        model_sha = sha256_file(ROOT / 'models/big_driving_supercombo.onnx')
        model = ct.models.MLModel(str(ROOT / 'artifacts/big_driving.mlpackage'), compute_units=ct.ComputeUnit.CPU_AND_NE)
        name = model.get_spec().description.output[0].name
        identity = make_identity(metadata, model_sha)
        session = CoreMLPolicySession(model, metadata, name, BIG_MODEL_FRAME_SKIP)
        for _ in range(3):
          session.infer(bytes(REQUEST_BYTES))
        del session
        report['modelSha256'] = model_sha
        report['checkpoint'] = identity['model_checkpoint']
      key = secrets.token_bytes(32)
      key_file = Path(temporary) / 'bench.key'
      key_file.write_bytes(key)
      key_file.chmod(0o600)
      remote = shell('sudo -u comma mktemp -d /tmp/macbench-auth.XXXXXX')
      if not re.fullmatch(r'/tmp/macbench-auth\.[A-Za-z0-9]{6}', remote):
        raise RuntimeError('invalid remote temporary directory')
      files = sources + [key_file]
      subprocess.run(adb + ['push', *map(str, files), remote + '/'], check=True, capture_output=True, timeout=20)
      staged = [remote + '/' + path.name for path in files]
      shell(shlex.join(['chown', 'comma:comma', *staged]))
      shell(shlex.join(['chmod', '600', remote + '/bench.key']))
      guard_command = shlex.join(['python3', '-u', '-c', client_bootstrap(GUARD), '120'])
      guard = subprocess.Popen(adb + ['exec-out', guard_command], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
      if not select.select([guard.stdout], [], [], 10)[0]:
        raise TimeoutError('CPU hold did not start')
      held = json.loads(guard.stdout.readline())
      if held.get('status') != 'holding' or not isinstance(held.get('pid'), int):
        raise RuntimeError('invalid CPU hold acknowledgement')
      guard_pid = held['pid']
      report['holdStart'] = held
      runs = [('camera', '4-7')] if args.camera_snapshot_only else [('transport', 'default'), ('transport', '4-7'), ('coreml', '4-7')]
      for mode, cpu in runs:
        run = {'mode': mode, 'cpuSet': cpu}
        report['runs'].append(run)
        errors = []
        connections = []
        warmup = 0 if mode == 'camera' else (132 if mode == 'coreml' else 20)
        run_identity = identity if mode == 'coreml' else {
          **identity, 'backend': 'TRANSPORT_PROBE', 'model_sha256': '0' * 64,
          'model_checkpoint': 'no-model-physical-usb-probe'}
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
          listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
          listener.bind(socket.getaddrinfo(args.mac, 0, socket.AF_INET6, socket.SOCK_STREAM)[0][4])
          listener.listen(1)
          listener.settimeout(25 if mode == 'camera' else 10)
          def serve():
            try:
              configure_user_interactive_qos()
              conn, peer = listener.accept()
              connections.append(conn)
              with conn:
                if peer[0] != args.device.split('%')[0]:
                  raise ValueError('unexpected USB peer')
                if mode == 'camera':
                  conn.settimeout(3)
                  sid = None
                  frame, camera, offset = 0, 0, 0
                  digest = hashlib.sha256()
                  camera_meta = None
                  while camera < (12 if args.camera_compare_8mib else 2):
                    conn.settimeout(3)
                    flags, current_sid, number, capture, wire = recv_message(conn, REQUEST)
                    if sid is None:
                      sid = current_sid
                    if flags != 0 or current_sid != sid or number != frame:
                      raise ValueError('camera sequence mismatch')
                    payload = verify_payload(key, REQUEST, flags, sid, number, capture, wire)
                    if len(payload) < 4:
                      raise ValueError('missing camera header')
                    header_len = struct.unpack_from('!I', payload)[0]
                    if not 0 < header_len <= 4096 or len(payload) < 4 + header_len:
                      raise ValueError('invalid camera header')
                    meta = json.loads(payload[4:4+header_len])
                    pixels = memoryview(payload)[4+header_len:]
                    chunk_offset = meta.pop('offset')
                    if camera_meta is None:
                      camera_meta = meta
                    if (meta != camera_meta or meta['camera'] != ('narrow', 'wide')[camera % 2]
                        or not 0 < meta['bytes'] <= 16 * 1024 * 1024 or chunk_offset != offset
                        or not 0 < len(pixels) <= (8 if args.camera_compare_8mib else 1) * 1024 * 1024
                        or offset + len(pixels) > meta['bytes']):
                      raise ValueError('invalid camera metadata')
                    digest.update(pixels)
                    offset += len(pixels)
                    ack = json.dumps({'bytes': offset, 'sha256': digest.hexdigest()}).encode()
                    response = authenticate_payload(key, RESPONSE, 0, sid, number, capture, ack)
                    conn.settimeout(3)
                    send_message(conn, RESPONSE, 0, sid, number, capture, response)
                    frame += 1
                    if offset == meta['bytes']:
                      camera += 1
                      offset, camera_meta = 0, None
                      digest = hashlib.sha256()
                elif mode == 'coreml':
                  try:
                    serve_client(conn, peer, model, metadata, name, BIG_MODEL_FRAME_SKIP,
                                 run_identity, key, 2., 1000.)
                  except EOFError:
                    pass  # bounded client closes after its final frame
                else:
                  conn.settimeout(2)
                  conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                  conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
                  sid = server_handshake(conn, run_identity, key)
                  for frame in range(warmup + 100):
                    # Framing shrinks socket timeout while enforcing one-frame
                    # deadlines. Start a fresh budget for each paced request.
                    conn.settimeout(2)
                    flags, received_sid, number, capture, wire = recv_message(conn, REQUEST)
                    received = time.monotonic_ns()
                    if (received_sid != sid or number != frame or flags != (FLAG_RESET if frame == 0 else 0)):
                      raise ValueError('request sequence/flags mismatch')
                    payload = verify_payload(key, REQUEST, flags, sid, number, capture, wire)
                    if len(payload) != REQUEST_BYTES:
                      raise ValueError('request length mismatch')
                    ready = time.monotonic_ns()
                    response = authenticate_payload(key, RESPONSE, flags, sid, number, capture,
                                                    TIMINGS.pack(received, ready, ready) + bytes(18452 * 2))
                    send_message(conn, RESPONSE, flags, sid, number, capture, response)
            except Exception as error:
              errors.append(f'{type(error).__name__}: {error}')
          worker = threading.Thread(target=serve, daemon=True)
          worker.start()
          host = args.mac.split('%')[0] + '%' + args.device.split('%')[1]
          command = shlex.join(['sudo', '-u', 'comma', 'env', 'PYTHONDONTWRITEBYTECODE=1',
                               'OPENBLAS_NUM_THREADS=1', '/usr/local/venv/bin/python3',
                               remote + '/usb_authenticated_client.py', '--host', host,
                               '--port', str(listener.getsockname()[1]), '--key', remote + '/bench.key',
                               '--backend', run_identity['backend'], '--sha', run_identity['model_sha256'],
                               '--checkpoint', run_identity['model_checkpoint'], '--warmup', str(warmup), '--cpu-set', cpu])
          if mode == 'camera':
            command = shlex.join(['sudo', '-u', 'comma', 'env', 'PYTHONDONTWRITEBYTECODE=1',
                                 'PYTHONPATH=/data/openpilot', '/usr/local/venv/bin/python3',
                                 remote + '/usb_camera_snapshot_client.py', '--host', host,
                                 '--port', str(listener.getsockname()[1]), '--key', remote + '/bench.key'])
            if args.camera_compare_8mib:
              command += ' --compare-8mib'
          print(f'Testing {mode}, affinity={cpu}, warmup={warmup}...', flush=True)
          try:
            result = subprocess.run(adb + ['exec-out', command], capture_output=True, text=True, timeout=30)
            if result.returncode or not result.stdout.lstrip().startswith('{'):
              raise RuntimeError('client failed: ' + (result.stderr or result.stdout)[-1000:])
            run['result'] = json.loads(result.stdout)
          finally:
            for conn in connections:
              try:
                conn.shutdown(socket.SHUT_RDWR)
              except OSError:
                pass
            worker.join(11)
            run['serverErrors'] = list(errors)
            if errors:
              print(json.dumps({'serverErrors': errors}), flush=True)
          if worker.is_alive() or errors:
            raise RuntimeError(f'server did not finish cleanly: {errors}')
          print(json.dumps({'mode': mode, 'cpuSet': cpu,
                            'measurement': run['result'].get('stages', run['result'].get('rows'))}, allow_nan=False), flush=True)
      report['status'] = 'measured-not-qualified'
    except Exception as error:
      report['status'], report['error'] = 'failed', f'{type(error).__name__}: {error}'
      raise
    finally:
      if guard is not None:
        try:
          if guard_pid is not None and guard.poll() is None:
            shell(shlex.join(['kill', '-TERM', str(guard_pid)]))
          remaining, _ = guard.communicate(timeout=12)
          released = [json.loads(line) for line in remaining.splitlines() if line.startswith('{')]
          report['holdEnd'] = released
          if not released or not released[-1].get('restored'):
            report['status'] = 'cleanup-needs-attention'
        except Exception as error:
          report['holdCleanupError'] = str(error)[:300]
          report['status'] = 'cleanup-needs-attention'
      if remote and re.fullmatch(r'/tmp/macbench-auth\.[A-Za-z0-9]{6}', remote):
        try:
          shell(shlex.join(['rm', '-f', *[remote + '/' + path.name for path in sources], remote + '/bench.key']))
          shell(shlex.join(['rmdir', remote]))
          report['remoteTemporaryFilesRemoved'] = True
        except Exception as error:
          report['remoteCleanupError'] = str(error)[:300]
          report['status'] = 'cleanup-needs-attention'
      json.dump(report, output, indent=2, allow_nan=False)
      output.write('\n')
      print(json.dumps({'status': report['status'], 'holdEnd': report.get('holdEnd'),
                        'remoteTemporaryFilesRemoved': report.get('remoteTemporaryFilesRemoved')}), flush=True)


if __name__ == '__main__':
  main()
