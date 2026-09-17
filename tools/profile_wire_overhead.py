"""Authenticated loopback transport and copy microbench; never a driving model.

Includes framing, HMAC, CRC, request/response socket transfer and client output
validation. No USB, NCM, camera readback, preprocessing or model inference.
"""

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import socket
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accelerator_client import AcceleratorClient
from accelerator_protocol import AUTH_METADATA, authenticate_payload, server_handshake, verify_payload
from transport import (DEVICE_TYPE, POLICY_INPUTS, REQUEST, REQUEST_BYTES, RESPONSE, TIMINGS, VERSION,
                       WARPED_BYTES, recv_message, send_message)


def stats(values):
  ordered = sorted(values)
  return {'samples': len(values), 'meanMs': statistics.mean(values), 'medianMs': statistics.median(values),
          'p99NearestRankMs': ordered[min(len(ordered) - 1, (99 * len(ordered) + 99) // 100 - 1)],
          'maxMs': max(values)}


def legacy_verify(key, wire):
  payload, tag = wire[:-32], wire[-32:]
  metadata = AUTH_METADATA.pack(VERSION, REQUEST, 0, 1, 1, 1)
  expected = hmac.new(key, b'SPMA-PAYLOAD-v2\0' + metadata + payload, hashlib.sha256).digest()
  expected = (payload + expected)[-32:]
  if not hmac.compare_digest(tag, expected):
    raise ValueError('legacy tag mismatch')
  return payload


def auth_profile(key, payload, runs):
  wire = authenticate_payload(key, REQUEST, 0, 1, 1, 1, payload)
  methods = {'legacy': lambda: legacy_verify(key, wire),
             'current': lambda: verify_payload(key, REQUEST, 0, 1, 1, 1, wire)}
  reports = []
  for name in ('legacy', 'current', 'current', 'legacy'):
    function = methods[name]
    if function() != payload:
      raise ValueError('authentication output mismatch')
    for _ in range(50):
      function()
    times = []
    for _ in range(runs):
      start = time.perf_counter_ns()
      function()
      times.append((time.perf_counter_ns() - start) / 1e6)
    reports.append({'method': name, **stats(times)})
  return reports


def loopback(family, key, pixels, runs, warmup):
  host = '127.0.0.1' if family == socket.AF_INET else '::1'
  listener = socket.socket(family, socket.SOCK_STREAM)
  listener.bind((host, 0))
  listener.listen(1)
  listener.settimeout(10)
  port = listener.getsockname()[1]
  errors = []
  identity = {'backend': 'TRANSPORT_PROBE', 'device_type': DEVICE_TYPE, 'protocol': VERSION,
              'model_checkpoint': 'no-model-loopback-probe', 'model_sha256': '0' * 64,
              'output_floats': 18452, 'output_dtype': 'float16', 'request_bytes': REQUEST_BYTES,
              'input_shapes': {}, 'output_shapes': {'outputs': [1, 18452]}, 'output_slices': {}}
  def serve():
    try:
      conn, _ = listener.accept()
      with conn:
        conn.settimeout(5)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        session = server_handshake(conn, identity, key)
        zero_output = bytes(18452 * 2)
        for frame in range(warmup + runs):
          flags, received_session, number, capture, wire = recv_message(conn, REQUEST)
          received = time.monotonic_ns()
          if received_session != session or number != frame:
            raise ValueError('probe sequence mismatch')
          payload = verify_payload(key, REQUEST, flags, session, number, capture, wire)
          if len(payload) != REQUEST_BYTES:
            raise ValueError('probe request size mismatch')
          ready = time.monotonic_ns()
          # Equal start/end expressly records ZERO model inference.
          response = authenticate_payload(key, RESPONSE, flags, session, number, capture,
                                          TIMINGS.pack(received, ready, ready) + zero_output)
          send_message(conn, RESPONSE, flags, session, number, capture, response)
    except Exception as error:
      errors.append(f'{type(error).__name__}: {error}')
    finally:
      listener.close()
  worker = threading.Thread(target=serve, daemon=True)
  worker.start()
  client = AcceleratorClient(host, port, auth_key=key, expected_backend='TRANSPORT_PROBE',
                             expected_output_floats=18452, expected_model_sha256='0' * 64,
                             deadline_ms=500., qualification_timeout_ms=500.)
  results = []
  try:
    client.connect()
    for frame in range(warmup + runs):
      result = client.infer(pixels, bytes(POLICY_INPUTS.size), frame_id=frame,
                            capture_ns=time.monotonic_ns(), reset=frame == 0)
      if frame >= warmup:
        results.append(result)
  finally:
    client.close()
    listener.close()
    worker.join(6)
  if worker.is_alive() or errors:
    raise RuntimeError(f'probe server failure: {errors}')
  return {name: stats([getattr(result, name) for result in results]) for name in
          ('round_trip_ms', 'prepare_ms', 'send_ms', 'receive_ms', 'validate_ms', 'server_prepare_ms')}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--runs', type=int, default=500)
  args = parser.parse_args()
  if not 20 <= args.runs <= 5000:
    parser.error('runs must be 20..5000')
  report = {'scope': 'synthetic uncompressed authenticated TCP loopback; no model, USB or camera',
            'roadReady': False, 'wireRequestPayloadBytes': REQUEST_BYTES + 32,
            'wireResponsePayloadBytes': TIMINGS.size + 18452 * 2 + 32}
  with args.output.open('x') as output:
    try:
      key, pixels = secrets.token_bytes(32), secrets.token_bytes(WARPED_BYTES)
      report['authABBA'] = auth_profile(key, pixels + bytes(POLICY_INPUTS.size), args.runs)
      report['ipv4'] = loopback(socket.AF_INET, key, pixels, args.runs, 50)
      report['ipv6'] = loopback(socket.AF_INET6, key, pixels, args.runs, 50)
      report['status'] = 'measured-not-qualified'
    except Exception as error:
      report['status'], report['error'] = 'failed', f'{type(error).__name__}: {error}'
      raise
    finally:
      json.dump(report, output, indent=2, allow_nan=False)
      output.write('\n')
      print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
