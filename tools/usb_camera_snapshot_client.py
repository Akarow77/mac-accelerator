"""Two road-camera snapshots, captured offroad and sent only after camerad stops.

No image files, cabin camera, modeld hooks, controls, calibration or live inference.
Mapped buffers stay owned by the VisionIPC clients until copied after shutdown.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import struct
import subprocess
import time

from accelerator_protocol import authenticate_payload, authenticate_payload_parts, read_auth_key, verify_payload
from transport import REQUEST, RESPONSE, recv_message, send_message, send_message_parts
import transport


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--host', required=True)
  parser.add_argument('--port', type=int, required=True)
  parser.add_argument('--key', type=Path, required=True)
  parser.add_argument('--compare-8mib', action='store_true')
  args = parser.parse_args()
  if args.compare_8mib:
    transport.MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
  if Path('/data/params/d/IsOffroad').read_text().strip() != '1':
    raise RuntimeError('not offroad')
  if subprocess.run(['pgrep', '-x', 'camerad'], capture_output=True).returncode != 1:
    raise RuntimeError('existing camerad or process-check failure; refusing to interfere')
  if not {4,5,6,7}.issubset(os.sched_getaffinity(0)):
    raise RuntimeError('high cores unavailable')
  os.sched_setaffinity(0, {4,5,6,7})
  from msgq.visionipc import VisionIpcClient
  from openpilot.cereal.visionipc import VisionStreamType
  clients = [VisionIpcClient('camerad', kind, True) for kind in
             (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD)]
  env = dict(os.environ, DISABLE_DRIVER='1')
  for key in ('LOG_RAW_FRAMES', 'DEBUG_FRAMES', 'CTRL_EXP_FROM_PARAMS', 'DISABLE_ROAD', 'DISABLE_WIDE_ROAD'):
    env.pop(key, None)
  process = subprocess.Popen(['timeout', '--signal=INT', '--kill-after=3', '18', './camerad'],
                             cwd='/data/openpilot/openpilot/system/camerad', env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
  def stop_camera():
    if process.poll() is None:
      os.killpg(process.pid, signal.SIGINT)
      try:
        process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)
        raise RuntimeError('camera required forced shutdown; snapshot rejected')
  buffers = []
  metadata = []
  try:
    connect_deadline = time.monotonic() + 8
    for client in clients:
      while not client.connect(False):
        if process.poll() is not None or time.monotonic() > connect_deadline:
          raise TimeoutError('camera did not start')
        time.sleep(0.05)
    exposure_deadline = time.monotonic() + 4
    while time.monotonic() < exposure_deadline:
      if Path('/data/params/d/IsOffroad').read_text().strip() != '1':
        raise RuntimeError('left offroad')
      for client in clients:
        client.recv(100)
    for label, client in zip(('narrow', 'wide'), clients):
      buffer = client.recv(200)
      if buffer is None or not client.is_connected():
        raise RuntimeError('missing camera frame')
      buffers.append(buffer)
      metadata.append({'camera': label, 'frameId': client.frame_id, 'eofNs': client.timestamp_eof,
                       'width': buffer.width, 'height': buffer.height, 'stride': buffer.stride,
                       'uvOffset': buffer.uv_offset, 'bytes': len(buffer.data)})
  finally:
    stop_camera()
  if subprocess.run(['pgrep', '-x', 'camerad'], capture_output=True).returncode != 1:
    raise RuntimeError('camera still running; no buffer copied or transmitted')
  owned = []
  for buffer, meta in zip(buffers, metadata):
    if buffer.frame_id != meta['frameId']:
      raise RuntimeError(f"snapshot frame changed: {meta}, bufferFrameId={buffer.frame_id}")
    if not 0 < meta['bytes'] <= 16 * 1024 * 1024:
      raise RuntimeError(f'snapshot size invalid: {meta}')
    pixels = bytes(buffer.data)
    if buffer.frame_id != meta['frameId']:
      raise RuntimeError('snapshot changed during copy')
    owned.append(pixels)
  key = read_auth_key(args.key)
  session = secrets.randbits(64)
  rows = []
  with socket.create_connection((args.host, args.port), timeout=5) as conn:
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    frame = 0
    modes = ('chunked', 'single', 'single-iovec', 'single-iovec', 'single', 'chunked') if args.compare_8mib else ('chunked',)
    cases = [(pixels, {**meta, 'transferMode': mode, 'pass': repeat})
             for repeat, mode in enumerate(modes) for pixels, meta in zip(owned, metadata)]
    for pixels, meta in cases:
      if Path('/data/params/d/IsOffroad').read_text().strip() != '1':
        raise RuntimeError('left offroad')
      started = time.monotonic_ns()
      digest = hashlib.sha256()
      chunk_size = len(pixels) if meta['transferMode'].startswith('single') else 1024 * 1024
      preparation_ms = send_ms = send_cpu_ms = receive_ms = 0.
      for offset in range(0, len(pixels), chunk_size):
        preparing = time.monotonic_ns()
        chunk = memoryview(pixels)[offset:offset + chunk_size]
        header = json.dumps({**meta, 'offset': offset}).encode()
        capture = time.monotonic_ns()
        parts = (struct.pack('!I', len(header)), header, chunk)
        if meta['transferMode'] == 'single-iovec':
          wire_parts = authenticate_payload_parts(key, REQUEST, 0, session, frame, capture, parts)
        else:
          wire = authenticate_payload(key, REQUEST, 0, session, frame, capture, b''.join(parts))
        prepared = time.monotonic_ns()
        preparation_ms += (prepared - preparing) / 1e6
        conn.settimeout(3)
        cpu_started = time.thread_time_ns()
        if meta['transferMode'] == 'single-iovec':
          send_message_parts(conn, REQUEST, 0, session, frame, capture, wire_parts)
        else:
          send_message(conn, REQUEST, 0, session, frame, capture, wire)
        send_cpu_ms += (time.thread_time_ns() - cpu_started) / 1e6
        sent = time.monotonic_ns()
        send_ms += (sent - prepared) / 1e6
        conn.settimeout(3)
        flags, sid, number, stamp, reply = recv_message(conn, RESPONSE)
        receive_ms += (time.monotonic_ns() - sent) / 1e6
        if (flags, sid, number, stamp) != (0, session, frame, capture):
          raise RuntimeError('camera acknowledgement identity mismatch')
        result = json.loads(verify_payload(key, RESPONSE, flags, sid, number, stamp, reply))
        digest.update(chunk)
        if result != {'bytes': offset + len(chunk), 'sha256': digest.hexdigest()}:
          raise RuntimeError('camera bytes did not match acknowledgement')
        frame += 1
      rows.append({**meta, 'transferAndAckMs': (time.monotonic_ns()-started)/1e6,
                   'prepareMs': preparation_ms, 'sendMs': send_ms,
                   'sendThreadCpuMs': send_cpu_ms, 'receiveMs': receive_ms,
                   'verifiedBytes': len(pixels)})
  print(json.dumps({'status':'snapshot-transfer-verified', 'roadReady':False,
                    'cameraStoppedBeforeCopyAndTransfer':True, 'cabinCameraEnabled':False,
                    'imageFilesSaved':False, 'messageLimitBytes': transport.MAX_PAYLOAD_BYTES, 'rows':rows}))


if __name__ == '__main__':
  main()
