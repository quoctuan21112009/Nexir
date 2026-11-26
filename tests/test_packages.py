import argparse
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nexir.compiler import NexirError
from nexir.packages import cpp_prefix, package_command


class PackageCommandTests(unittest.TestCase):
    def test_vcpkg_install_selects_matching_target_and_host_triplets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ('vcpkg.exe' if os.name == 'nt' else 'vcpkg')).touch()
            args = argparse.Namespace(kind='cpp', action='install', package='zlib', triplet='x64-mingw-static')
            with patch.dict(os.environ, {'VCPKG_ROOT': str(root)}), patch('nexir.packages._run') as run:
                package_command(args)
            command, = run.call_args.args
            env = run.call_args.kwargs['env']
            self.assertEqual(command[-2:], ['install', 'zlib:x64-mingw-static'])
            self.assertEqual(env['VCPKG_DEFAULT_TRIPLET'], 'x64-mingw-static')
            self.assertEqual(env['VCPKG_DEFAULT_HOST_TRIPLET'], 'x64-mingw-static')

    def test_vcpkg_build_prefix_requires_installed_headers_and_libs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / 'installed' / 'x64-mingw-static'
            with patch.dict(os.environ, {'VCPKG_ROOT': str(root)}):
                with self.assertRaises(NexirError):
                    cpp_prefix('x64-mingw-static')
                (prefix / 'include').mkdir(parents=True)
                (prefix / 'lib').mkdir()
                self.assertEqual(cpp_prefix('x64-mingw-static'), prefix)

    def test_package_specs_cannot_be_pip_options(self):
        args = argparse.Namespace(kind='python', action='install', package='--index-url', upgrade=False)
        with self.assertRaises(NexirError):
            package_command(args)


if __name__ == '__main__':
    unittest.main()
