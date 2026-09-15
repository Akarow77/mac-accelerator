#!/usr/bin/env python3
"""Copy user-supplied model artifacts without importing or executing their metadata."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def import_artifacts(root: Path, onnx: Path, metadata: Path, coreml: Path | None = None) -> None:
  sources = [(onnx, root / 'models/big_driving_supercombo.onnx'),
             (metadata, root / 'models/big_driving_supercombo_metadata.pkl')]
  if coreml is not None:
    sources.append((coreml, root / 'artifacts/big_driving.mlpackage'))
  if not onnx.is_file() or not metadata.is_file():
    raise ValueError('ONNX and matching metadata must be existing files')
  with onnx.open('rb') as handle:
    prefix = handle.read(80)
  if onnx.stat().st_size < 1024 or prefix.startswith(b'version https://git-lfs'):
    raise ValueError('ONNX appears to be a Git LFS pointer or invalid small file')
  if coreml is not None and (not coreml.is_dir() or not (coreml / 'Manifest.json').is_file()):
    raise ValueError('Core ML input must be an existing .mlpackage with Manifest.json')
  manifest = root / 'models/import.json'
  for destination in [*(destination for _, destination in sources), manifest]:
    if destination.exists() or destination.is_symlink():
      raise FileExistsError(f'Preserving existing artifact: {destination}')
  for source, destination in sources:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
      shutil.copytree(source, destination)
    else:
      shutil.copy2(source, destination)
  with onnx.open('rb') as handle:
    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
  manifest.write_text(json.dumps({
    'source_onnx_sha256': digest,
    'coreml_imported': coreml is not None,
    'notice': 'Copy record only; does not prove conversion provenance or accuracy.',
  }, indent=2) + '\n')


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--onnx', type=Path, required=True)
  parser.add_argument('--metadata', type=Path, required=True,
                      help='matching trusted metadata; runtime loads pickle, which can execute code')
  parser.add_argument('--coreml', type=Path, help='optional previously converted .mlpackage')
  args = parser.parse_args()
  import_artifacts(Path(__file__).resolve().parent, args.onnx, args.metadata, args.coreml)
  print('Imported independent local copies. No firmware or device settings were changed.')


if __name__ == '__main__':
  main()
