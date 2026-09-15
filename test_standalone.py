import ast
from pathlib import Path
import plistlib
import tempfile
import unittest

from import_model import import_artifacts


ROOT = Path(__file__).resolve().parent


class StandaloneTests(unittest.TestCase):
  def test_no_vehicle_stack_imports(self):
    for path in ROOT.glob('*.py'):
      for node in ast.walk(ast.parse(path.read_text())):
        names = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                 else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
        for name in names:
          self.assertNotIn(name.split('.')[0], ('openpilot', 'sunnypilot', 'tinygrad', 'cereal'), str(path))

  def test_app_identity_and_paths_are_independent(self):
    with (ROOT / 'macos_app/Info.plist').open('rb') as handle:
      info = plistlib.load(handle)
    self.assertEqual(info['CFBundleIdentifier'], 'dev.akarow.mac-accelerator')
    self.assertEqual(info['CFBundleExecutable'], 'MacAccelerator')
    source = (ROOT / 'macos_app/MacAcceleratorApp.m').read_text()
    self.assertNotIn('tools/mac_accelerator/', source)
    self.assertNotIn('openpilot/', source)
    self.assertIn('@"Mac Accelerator"', source)
    self.assertIn('self.localTestButton.state = NSControlStateValueOn', source)

  def test_shell_defaults_are_local_and_project_relative(self):
    server = (ROOT / 'run_coreml_server.sh').read_text()
    self.assertIn('REPO_ROOT="$SCRIPT_DIR"', server)
    self.assertEqual(server.count('${MAC_ACCELERATOR_LOCAL_TEST:-1}'), 2)
    self.assertNotIn('openpilot/', server)

  def test_import_copies_without_unpickling_or_overwriting(self):
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      source = base / 'source'
      source.mkdir()
      onnx, metadata = source / 'model.onnx', source / 'metadata.pkl'
      onnx.write_bytes(b'x' * 2048)
      metadata.write_bytes(b'not a pickle; import must never execute it')
      package = source / 'model.mlpackage'
      package.mkdir()
      (package / 'Manifest.json').write_text('{}')
      destination = base / 'destination'
      import_artifacts(destination, onnx, metadata, package)
      copied = destination / 'models/big_driving_supercombo.onnx'
      self.assertEqual(copied.read_bytes(), onnx.read_bytes())
      self.assertFalse(copied.is_symlink())
      self.assertNotEqual(copied.stat().st_ino, onnx.stat().st_ino)
      with self.assertRaises(FileExistsError):
        import_artifacts(destination, onnx, metadata, package)

  def test_rejects_lfs_pointer_before_copy(self):
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      onnx, metadata = base / 'model.onnx', base / 'model.pkl'
      onnx.write_bytes(b'version https://git-lfs.github.com/spec/v1\n' + b'x' * 2048)
      metadata.write_bytes(b'x')
      destination = base / 'destination'
      with self.assertRaises(ValueError):
        import_artifacts(destination, onnx, metadata)
      self.assertFalse(destination.exists())


if __name__ == '__main__':
  unittest.main()
