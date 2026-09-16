"""Optional development-only geometry comparison with a pinned local upstream checkout.

Run with that checkout's prepared Python environment. This is not a runtime
dependency or camera consumer. Video paths stay local; only aggregate counts print.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--upstream-root', type=Path, required=True)
  parser.add_argument('--video', type=Path, required=True)
  parser.add_argument('--reference-map', action='store_true')
  parser.add_argument('--native', action='store_true', help='compare the locally built native gather')
  args = parser.parse_args()
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  sys.path.insert(1, str(args.upstream_root.resolve()))
  from camera_preprocess import GatherPlan, NV12Layout
  from openpilot.selfdrive.modeld.compile_modeld import NV12Frame, make_frame_prepare
  from tinygrad import Tensor
  from tinygrad.device import Device
  # Four decoded real-image frames; bounded output, no route identifiers in report.
  raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-i', str(args.video),
                                 '-frames:v', '4', '-f', 'rawvideo', '-pix_fmt', 'nv12', 'pipe:1'], timeout=60)
  layout = NV12Layout(1928, 1208, 2048, 1216, 608)
  packed_size = layout.width * layout.height * 3 // 2
  if len(raw) != packed_size * 4:
    raise ValueError('recording did not decode four full-size frames')
  matrices = [np.eye(3, dtype=np.float32),
              np.array([[3.1, .03, 130], [.02, 3.9, 55], [.00001, -.00003, 1]], dtype=np.float32),
              np.array([[2.8, -.02, 170], [.015, 3.5, 100], [-.00002, .00001, 1]], dtype=np.float32)]
  nv12 = NV12Frame(layout.width, layout.height, layout.stride, layout.y_height, layout.uv_height, layout.nbytes)
  reference = make_frame_prepare(nv12, 512, 256)
  comparisons = []
  for matrix_index, matrix in enumerate(matrices):
    plan = GatherPlan(layout, matrix, reference_device=Device.DEFAULT if args.reference_map else None)
    runner = plan
    if args.native:
      from backends.native_gather import NativeGather
      runner = NativeGather(plan)
    differences = 0
    max_error = 0
    for frame in range(4):
      source = np.frombuffer(raw, dtype=np.uint8, count=packed_size, offset=frame * packed_size)
      padded = np.zeros(layout.nbytes, dtype=np.uint8)
      y_size = layout.width * layout.height
      padded[:layout.stride * layout.y_height].reshape(layout.y_height, layout.stride)[:layout.height, :layout.width] = source[:y_size].reshape(layout.height, layout.width)
      padded[layout.stride * layout.y_height:].reshape(layout.uv_height, layout.stride)[:layout.height // 2, :layout.width] = source[y_size:].reshape(layout.height // 2, layout.width)
      expected = reference(Tensor(padded, device=Device.DEFAULT), Tensor(matrix, device=Device.DEFAULT)).numpy()
      actual = runner.run(padded.tobytes())
      differences += int(np.count_nonzero(actual != expected))
      max_error = max(max_error, int(np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16)))))
    comparisons.append({'transform': matrix_index, 'frames': 4, 'differentBytes': differences, 'maxByteError': max_error})
  report = {'referenceDevice': Device.DEFAULT, 'nativeGather': args.native, 'comparisons': comparisons,
            'exact': all(row['differentBytes'] == 0 for row in comparisons),
            'scope': 'decoded real images with three test transforms; not route-calibrated model validation'}
  print(json.dumps(report, indent=2))
  if not report['exact']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
