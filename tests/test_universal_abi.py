"""Tests for NEXIR Universal ABI: zero-copy Span, N-D arrays, nested List/Dict, global packages, and #![cap]."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nexir.cli import main as cli_main
from nexir.compiler import compile_source
from nexir.interop import import_nx


class UniversalABITests(unittest.TestCase):
    def test_zero_copy_bytearray_and_list_span(self):
        src = """
@export
def scale_and_sum(xs: MutSpan[f64]) -> f64:
    total: f64 = 0.0
    for i in range(len(xs)):
        xs[i] = xs[i] * 2.0
        total += xs[i]
    return total
"""
        with tempfile.TemporaryDirectory(prefix="nx-zc-") as td:
            p = Path(td) / "zc.nx"
            p.write_text(src, encoding="utf-8")
            mod = import_nx(p)
            try:
                import array
                buf = array.array("d", [1.0, 2.0, 3.0, 4.0])
                res = mod.scale_and_sum(buf)
                self.assertAlmostEqual(res, 20.0)
                # Zero-copy mutation happened directly on Python's buffer!
                self.assertEqual(list(buf), [2.0, 4.0, 6.0, 8.0])
            finally:
                mod.close()

    def test_nd_arrays_nested_lists_and_dicts_to_python(self):
        py_src = """
def sum_2d(mat):
    return sum(sum(row) for row in mat)

def sum_3d(cube):
    return sum(sum(sum(row) for row in plane) for plane in cube)

def inspect_nested(items, cfg):
    s = sum(d["v"] for d in items)
    t = sum(cfg["nums"]) + cfg["meta"]["offset"]
    return f"{s}:{t}"
"""
        nx_src = """
import py.ndhelper as nd

@cap("memory.alloc", "python.interop")
def main() -> i32:
    m2 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    m3 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    r2: f64 = as_f64(nd.sum_2d(m2))
    r3: i64 = as_i64(nd.sum_3d(m3))
    list_of_dicts = [{"v": 10}, {"v": 25}]
    nested_dict = {"nums": [1, 2, 3], "meta": {"offset": 100}}
    info: str = as_str(nd.inspect_nested(list_of_dicts, nested_dict))
    print(r2, r3, info)
    return 0
"""
        with tempfile.TemporaryDirectory(prefix="nx-nd-") as td:
            td_path = Path(td)
            (td_path / "ndhelper.py").write_text(py_src, encoding="utf-8")
            nx_file = td_path / "main.nx"
            nx_file.write_text(nx_src, encoding="utf-8")
            exe_file = td_path / ("main.exe" if os.name == "nt" else "main")
            self.assertEqual(cli_main(["build", str(nx_file), "-o", str(exe_file)]), 0)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(td_path) + os.pathsep + env.get("PYTHONPATH", "")
            if os.name == "nt":
                env["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + env.get("PATH", "")
            proc = subprocess.run([str(exe_file)], capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "21 36 35:106")

    def test_global_package_search_path_and_module_caps(self):
        with tempfile.TemporaryDirectory(prefix="nx-gpkg-") as td:
            td_path = Path(td)
            pkg_root = td_path / "global_pkgs"
            pkg_dir = pkg_root / "mygreeter"
            pkg_dir.mkdir(parents=True)
            (pkg_dir / "nx.toml").write_text('[package]\nname = "mygreeter"\nversion = "0.1.0"\ntrusted = true\n', encoding="utf-8")
            (pkg_dir / "__init__.nx").write_text(
                '@cap("memory.alloc")\ndef greet(name: str) -> str:\n    return "Hello " + name\n',
                encoding="utf-8",
            )
            app_dir = td_path / "app"
            app_dir.mkdir()
            app = app_dir / "main.nx"
            app.write_text('import mygreeter\n\ndef main() -> i32:\n    print(mygreeter.greet("Nexir"))\n    return 0\n', encoding="utf-8")
            exe = app_dir / ("main.exe" if os.name == "nt" else "main")
            old_np = os.environ.get("NEXIR_PATH")
            os.environ["NEXIR_PATH"] = str(pkg_root)
            try:
                self.assertEqual(cli_main(["build", str(app), "-o", str(exe)]), 0)
            finally:
                if old_np is None:
                    os.environ.pop("NEXIR_PATH", None)
                else:
                    os.environ["NEXIR_PATH"] = old_np
            proc = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "Hello Nexir")

        cap_src = '''#![cap("all")]
def main() -> i32:
    xs = [1, 2, 3]
    print(len(xs))
    return 0
'''
        cpp = compile_source(cap_src, require_main=True)
        self.assertIn("nx_main", cpp)

    def test_minimal_script_int_input_and_lockfree_channel_and_run_cache(self):
        from nexir.cache_mgr import CacheManager
        from nexir.compiler import Compiler

        # 1. Top-level script: a = int(input("A: ")); b = int(input("B: ")); print(a + b)
        script = 'a = int(input("A: "))\nb = int(input("B: "))\nprint(a + b)\n'
        comp = Compiler(script, require_main=True)
        inferred = comp._infer_minimal_caps(comp.entry_unit, [gexpr for _, gexpr, _ in comp.entry_unit.global_inits] + list(comp.entry_unit.top_stmts))
        self.assertIn("memory.alloc", inferred)
        self.assertNotIn("process.exec", inferred)
        self.assertNotIn("filesystem.admin", inferred)

        with tempfile.TemporaryDirectory(prefix="nx-script-") as td:
            td_path = Path(td)
            cache_dir = td_path / "cache"
            old_cache = os.environ.get("NEXIR_CACHE_DIR")
            os.environ["NEXIR_CACHE_DIR"] = str(cache_dir)
            try:
                p = td_path / "add_input.nx"
                p.write_text(script, encoding="utf-8")
                exe = td_path / ("add_input.exe" if os.name == "nt" else "add_input")
                self.assertEqual(cli_main(["build", str(p), "-o", str(exe)]), 0)
                proc = subprocess.run([str(exe)], input="17\n25\n", capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0)
                self.assertIn("A: B: 42", proc.stdout)

                # 2. Lock-free bounded Channel[i64](16)
                ch_src = """
import nx.thread as th

@cap("memory.alloc")
def main() -> i32:
    ch = Channel[i64](16)
    f1 = th.submit(lambda: None) if False else None
    for i in range(10):
        ch.send(i * 2)
    s: i64 = 0
    for i in range(10):
        s += value_or(ch.recv(), 0)
    ch.close()
    print(s)
    return 0
"""
                # Keep valid NX syntax for lock-free channel test
                ch_src = """
@cap("memory.alloc")
def main() -> i32:
    ch = Channel[i64](16)
    for i in range(10):
        ch.send(i * 2)
    s: i64 = 0
    for i in range(10):
        s += value_or(ch.recv(), 0)
    ch.close()
    print(s)
    return 0
"""
                ch_p = td_path / "chan.nx"
                ch_p.write_text(ch_src, encoding="utf-8")
                # First nx run compiles & caches; second nx run hits binary cache
                self.assertEqual(cli_main(["run", str(ch_p)]), 0)
                self.assertEqual(cli_main(["run", str(ch_p)]), 0)
                st = CacheManager(cache_dir).status()
                self.assertGreaterEqual(st["hits"], 1)
                self.assertGreaterEqual(st["counts"]["bins"], 1)
            finally:
                if old_cache is None:
                    os.environ.pop("NEXIR_CACHE_DIR", None)
                else:
                    os.environ["NEXIR_CACHE_DIR"] = old_cache

    def test_python_native_tuple_and_strict_native_ffi(self):
        from nexir.compiler import NexirError, compile_file

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # 1. Strict native FFI: calling c.foo.bar() without a header or @extern must fail cleanly
            bad_ffi = td_path / "bad_ffi.nx"
            bad_ffi.write_text(
                'import c.unknown_lib as ul\n\n@cap("ffi.c")\ndef main() -> i32:\n    x = ul.mystery_fn(1, 2)\n    return 0\n',
                encoding="utf-8",
            )
            with self.assertRaises(NexirError) as ctx:
                compile_file(bad_ffi)
            self.assertIn("missing native ABI signature", str(ctx.exception))

            # 2. Python-native tuple args + module attr objects (e.g. np.zeros((8, 8, 3), dtype=np.uint8), cv2.circle)
            try:
                import cv2  # noqa: F401
                import numpy  # noqa: F401
                has_cv = True
            except ImportError:
                has_cv = False

            if has_cv:
                py_native = td_path / "py_native_cv.nx"
                py_native.write_text(
                    'import py.numpy as np\n'
                    'import py.cv2 as cv2\n\n'
                    'img = np.zeros((16, 16, 3), dtype=np.uint8)\n'
                    'cv2.circle(img, (8, 8), 4, (0, 255, 0), -1)\n'
                    'gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n'
                    'print(gray.shape)\n',
                    encoding="utf-8",
                )
                self.assertEqual(cli_main(["run", str(py_native)]), 0)
            else:
                py_tuple = td_path / "py_tuple.nx"
                py_tuple.write_text(
                    'import py.builtins as bi\n\n'
                    's = bi.sum((10, 20, 12))\n'
                    'print(s)\n',
                    encoding="utf-8",
                )
                self.assertEqual(cli_main(["run", str(py_tuple)]), 0)


if __name__ == "__main__":
    unittest.main()
