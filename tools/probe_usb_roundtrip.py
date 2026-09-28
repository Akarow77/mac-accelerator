"""Detached-device synthetic TCP probe. No model, camera, Params or device files.

Runs a bounded stdlib client through ADB and a one-client USB-scoped listener.
This is NOT the authenticated accelerator protocol or a driving qualification.
"""

import argparse
import base64
import ipaddress
import json
from pathlib import Path
import shlex
import socket
import subprocess
import threading
import time
import zlib


CLIENT = r'''
import json, os, socket, statistics, struct, sys, time
host, port, runs, warmup = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
buffers = int(sys.argv[5])
preconnect = bool(int(sys.argv[6]))
force_send = bool(int(sys.argv[7]))
cpu_set = sys.argv[8]
chunk_bytes = int(sys.argv[9])
if cpu_set == '4-7':
  selected = {4, 5, 6, 7}
  if not selected.issubset(os.sched_getaffinity(0)):
    raise ValueError('requested cores unavailable')
  os.sched_setaffinity(0, selected)
request = bytearray(393264)
chunk_size = chunk_bytes or len(request)
chunks = [memoryview(request)[offset:offset+chunk_size] for offset in range(0, len(request), chunk_size)]
response = bytearray(36960)
expected = bytearray(36960)
times = []
send_times = []
send_cpu_times = []
tcp_rows = []
address = socket.getaddrinfo(host, port, socket.AF_INET6, socket.SOCK_STREAM)[0][4]
with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as conn:
  conn.settimeout(5)
  if buffers and preconnect:
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffers)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffers)
  conn.connect(address)
  conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  if buffers:
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffers)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffers)
  if force_send:
    # Linux UAPI SO_SNDBUFFORCE=32, verified against device socket.h.
    # Bench only: requires CAP_NET_ADMIN; never changes a sysctl.
    conn.setsockopt(socket.SOL_SOCKET, 32, buffers)
  actual = {'send': conn.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
            'receive': conn.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)}
  for frame in range(runs + warmup):
    if frame == warmup:
      tcp_before = struct.unpack_from('=24I', conn.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104), 8)
    struct.pack_into('!I', request, 0, frame)
    struct.pack_into('!I', expected, 0, frame)
    cpu_start = time.thread_time()
    start = time.monotonic()
    deadline = start + 2
    for chunk in chunks:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise TimeoutError('send deadline')
      conn.settimeout(remaining)
      conn.sendall(chunk)
    sent = time.monotonic()
    send_cpu_ms = (time.thread_time() - cpu_start) * 1000
    view = memoryview(response)
    while view:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise TimeoutError('response deadline')
      conn.settimeout(remaining)
      n = conn.recv_into(view)
      if not n:
        raise EOFError('response truncated')
      view = view[n:]
    elapsed = (time.monotonic() - start) * 1000
    if response != expected:
      raise ValueError('response content mismatch')
    if frame >= warmup:
      times.append(elapsed)
      send_times.append((sent - start) * 1000)
      send_cpu_times.append(send_cpu_ms)
      # Stable first 104 bytes checked against this device's Linux tcp.h.
      tcp_rows.append(struct.unpack_from('=24I', conn.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104), 8))
    time.sleep(max(0, start + 0.05 - time.monotonic()))
ordered = sorted(times)
print(json.dumps({'clientCpuAffinity': sorted(os.sched_getaffinity(0)),
  'clientSocketBufferBytes': actual, 'samples': len(times), 'meanMs': statistics.mean(times),
  'sendMeanMs': statistics.mean(send_times),
  'sendThreadCpuMeanMs': statistics.mean(send_cpu_times),
  'tcpInfo': {'smoothedRttMeanUs': statistics.mean(row[15] for row in tcp_rows),
              'smoothedRttMaxUs': max(row[15] for row in tcp_rows),
              'sendCwndMinSegments': min(row[18] for row in tcp_rows),
              'sendCwndMaxSegments': max(row[18] for row in tcp_rows),
              'sendMssBytes': tcp_rows[-1][2],
              'retransmissionsDuringMeasuredFrames': tcp_rows[-1][23] - tcp_before[23]},
  'medianMs': statistics.median(times), 'p99NearestRankMs': ordered[(99*len(times)+99)//100-1],
  'maxMs': max(times), 'over50ms': sum(x > 50 for x in times)}))
'''


def read_exact(conn, data):
  deadline = time.monotonic() + 2
  view = memoryview(data)
  while view:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
      raise TimeoutError('request deadline')
    conn.settimeout(remaining)
    count = conn.recv_into(view)
    if not count:
      raise EOFError('request truncated')
    view = view[count:]


def scoped_link_local(value):
  address, sep, scope = value.partition('%')
  if not sep or not scope or not ipaddress.IPv6Address(address).is_link_local:
    raise ValueError('a scoped IPv6 link-local address is required')
  return value


def client_bootstrap(source):
  encoded = base64.b64encode(zlib.compress(source.encode('utf-8'))).decode('ascii')
  return f"import base64,zlib; exec(compile(zlib.decompress(base64.b64decode('{encoded}')), '<usb-probe>', 'exec'))"


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mac', required=True, type=scoped_link_local)
  parser.add_argument('--device', required=True, type=scoped_link_local)
  parser.add_argument('--serial', required=True)
  parser.add_argument('--adb', default='/opt/homebrew/bin/adb')
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--socket-buffers', type=int, choices=(0, 2097152), default=0,
                      help='0=OS defaults, 2097152=existing accelerator socket options')
  parser.add_argument('--preconnect-buffers', action='store_true',
                      help='also set buffers before connect/listen, for handshake comparison')
  parser.add_argument('--force-client-send-buffer', action='store_true',
                      help='Linux privileged bench socket only; no global kernel changes')
  parser.add_argument('--client-cpu-set', choices=('default', '4-7'), default='default',
                      help='detached bench client affinity only; never alters local model scheduling')
  parser.add_argument('--send-chunk-bytes', type=int, choices=(0, 16384, 65536), default=0,
                      help='0=one sendall; otherwise preallocated memoryview chunks, same frame deadline')
  args = parser.parse_args()
  if args.force_client_send_buffer and not args.socket_buffers:
    parser.error('--force-client-send-buffer requires --socket-buffers 2097152')
  report = {'scope': 'physical USB/NCM synthetic raw TCP, no authentication/model/camera',
            'requestBytes': 393264, 'responseBytes': 36960, 'roadReady': False,
            'warmupFrames': 20, 'measuredFrames': 100, 'targetPeriodMs': 50}
  report['requestedSocketBufferBytes'] = args.socket_buffers
  report['preconnectBuffers'] = args.preconnect_buffers
  report['forceClientSendBuffer'] = args.force_client_send_buffer
  report['clientCpuSet'] = args.client_cpu_set
  report['sendChunkBytes'] = args.send_chunk_bytes
  with args.output.open('x') as output:
    try:
      state = subprocess.run([args.adb, '-s', args.serial, 'shell',
                              'cat /data/params/d/IsOffroad'], capture_output=True, text=True,
                             timeout=5, check=True)
      if state.stdout.strip() != '1':
        raise RuntimeError('device must be offroad; no state change attempted')
      cpu_command = shlex.join(['python3', '-c',
                               'import json, os; print(json.dumps(sorted(os.sched_getaffinity(0))))'])
      cpu_result = subprocess.run([args.adb, '-s', args.serial, 'shell', cpu_command],
                                  capture_output=True, text=True, timeout=5, check=True)
      report['deviceCpuAffinityBefore'] = json.loads(cpu_result.stdout)
      if args.client_cpu_set == '4-7' and not {4, 5, 6, 7}.issubset(report['deviceCpuAffinityBefore']):
        raise RuntimeError('requested cores unavailable; no listener or client started')
      errors = []
      with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        if args.socket_buffers and args.preconnect_buffers:
          listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.socket_buffers)
        listener.bind(socket.getaddrinfo(args.mac, 0, socket.AF_INET6, socket.SOCK_STREAM)[0][4])
        listener.listen(1)
        listener.settimeout(10)
        def serve():
          try:
            conn, peer = listener.accept()
            with conn:
              if ipaddress.IPv6Address(peer[0]) != ipaddress.IPv6Address(args.device.split('%')[0]):
                raise ValueError('unexpected USB peer')
              conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
              if args.socket_buffers:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.socket_buffers)
              report['serverReceiveBufferBytes'] = conn.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
              request, response = bytearray(393264), bytearray(36960)
              expected = bytearray(393264)
              for frame in range(120):
                read_exact(conn, request)
                expected[:4] = frame.to_bytes(4, 'big')
                if request != expected:
                  raise ValueError('request content mismatch')
                response[:4] = request[:4]
                conn.settimeout(2)
                conn.sendall(response)
          except Exception as error:
            errors.append(f'{type(error).__name__}: {error}')
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        # Device must use its own scope name for the Mac's link-local address.
        host = args.mac.split('%')[0] + '%' + args.device.split('%')[1]
        # Keep the legacy ADB command short; exec-out cannot carry our stdin
        # on this device. No file upload and no dependency installation.
        bootstrap = client_bootstrap(CLIENT)
        command = shlex.join(['python3', '-u', '-c', bootstrap, host,
                              str(listener.getsockname()[1]), '100', '20', str(args.socket_buffers),
                              str(int(args.preconnect_buffers)), str(int(args.force_client_send_buffer)),
                              args.client_cpu_set, str(args.send_chunk_bytes)])
        try:
          result = subprocess.run([args.adb, '-s', args.serial, 'exec-out', command],
                                  capture_output=True, text=True, timeout=25, check=False)
        finally:
          thread.join(timeout=12)
        if result.returncode or result.stdout.lstrip().startswith('Traceback'):
          detail = (result.stderr.strip() or result.stdout.strip() or 'no diagnostic output').splitlines()[-1]
          raise RuntimeError(f'device client failed ({result.returncode}): {detail[:300]}')
        if thread.is_alive() or errors:
          raise RuntimeError(f'probe server incomplete: {errors}')
        report.update(status='measured-not-qualified', rtt=json.loads(result.stdout))
    except Exception as error:
      report.update(status='failed', error=f'{type(error).__name__}: {error}')
      raise
    finally:
      json.dump(report, output, indent=2, allow_nan=False)
      output.write('\n')
      print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
