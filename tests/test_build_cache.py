import os
from pathlib import Path
import shutil
import tempfile
import unittest

from nexir.build_cache import cached_object


@unittest.skipUnless(shutil.which("clang++") or shutil.which("g++"), "C++ compiler required")
class NativeObjectCacheTests(unittest.TestCase):
    def test_reuse_across_staging_directories_and_header_invalidation(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(os.environ, NEXIR_CACHE_DIR=str(root / "cache"))
            header = root / "value.h"
            header.write_text("#define VALUE 7\n")
            flags = ["-std=c++17", "-O2", "-I", str(root)]
            def build(name):
                stage = root / name
                stage.mkdir()
                src = stage / "program.cpp"
                src.write_text('#include "value.h"\nint value() { return VALUE; }\n')
                out = cached_object(compiler, flags, src, stage / "program.o", env)
                return out.read_bytes()
            first = build("first")
            self.assertEqual(first, build("second"))
            self.assertEqual(len(list((root / "cache" / "objects").glob("*.o"))), 1)
            header.write_text("#define VALUE 8\n")
            self.assertNotEqual(first, build("third"))
            self.assertEqual(len(list((root / "cache" / "objects").glob("*.o"))), 2)
