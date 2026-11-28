"""End-to-end tests for top-level scripts, untyped/call globals, input(), **kwargs,
native HTTPS, auto-deps, polyglot auto-discovery, REPL, LLVM backend, ASM & cache manager.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nexir.cache_mgr import CacheManager, function_ast_hashes
from nexir.cli import main as cli_main
from nexir.interop import auto_discover_foreign_module, parse_c_header_signatures
from nexir.llvm_backend import compile_to_llvm_ir
from nexir.repl import run_repl


class Nx2ExtraTests(unittest.TestCase):
    def _run_nx(self, source: str, *, stdin_data: str | None = None, extra_args: list[str] | None = None, env_extra: dict[str, str] | None = None) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory(prefix="nx2-extra-") as td:
            td_path = Path(td)
            src_file = td_path / "prog.nx"
            src_file.write_text(source, encoding="utf-8")
            exe_file = td_path / ("prog.exe" if os.name == "nt" else "prog")
            rc = cli_main(["build", str(src_file), "-o", str(exe_file), *(extra_args or [])])
            self.assertEqual(rc, 0)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(td_path) + os.pathsep + env.get("PYTHONPATH", "")
            if env_extra:
                env.update(env_extra)
            proc = subprocess.run([str(exe_file)], input=stdin_data, capture_output=True, text=True, env=env)
            return proc.returncode, proc.stdout, proc.stderr

    def test_toplevel_script_untyped_and_call_globals_and_input(self):
        src = '''
def scale(v: i64) -> i64:
    return v * 3

base = 10
scaled = scale(base)
greeting = input("Name? ")
print(base, scaled, greeting)
'''
        rc, out, _ = self._run_nx(src, stdin_data="NexirUser\n")
        self.assertEqual(rc, 0)
        self.assertIn("Name? 10 30 NexirUser", out)

    def test_python_kwargs_unpacking(self):
        with tempfile.TemporaryDirectory(prefix="nx2-kwargs-") as td:
            td_path = Path(td)
            (td_path / "kwhelper.py").write_text(
                "def combine(a, b=0, **extra):\n    return f'{a}:{b}:' + ','.join(f'{k}={v}' for k, v in sorted(extra.items()))\n",
                encoding="utf-8",
            )
            src = '''
import py.kwhelper as kh

@cap("memory.alloc", "python.interop")
def main() -> i32:
    opts = {"x": 10, "y": 20}
    res: str = as_str(kh.combine(1, b=2, **opts))
    print(res)
    return 0
'''
            src_file = td_path / "app.nx"
            src_file.write_text(src, encoding="utf-8")
            exe_file = td_path / ("app.exe" if os.name == "nt" else "app")
            self.assertEqual(cli_main(["build", str(src_file), "-o", str(exe_file)]), 0)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(td_path) + os.pathsep + env.get("PYTHONPATH", "")
            if os.name == "nt":
                env["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + env.get("PATH", "")
            proc = subprocess.run([str(exe_file)], capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout.strip(), "1:2:x=10,y=20")

    def test_native_https_routing_and_sandbox(self):
        src = '''
import nx.net as net

@cap("memory.alloc", "network.connect")
def main() -> i32:
    r = net.http_get("https://example.com/")
    if is_err(r):
        print("ERR:", error(r))
    else:
        print("OK:", unwrap(r).status)
    return 0
'''
        rc, out, _ = self._run_nx(src, env_extra={"NX_SANDBOX": "1", "NX_DENY_NET": "1"})
        self.assertEqual(rc, 0)
        self.assertIn("sandbox denied https request", out)

    def test_auto_dependency_on_import_from_registry(self):
        with tempfile.TemporaryDirectory(prefix="nx2-autodep-") as td:
            td_path = Path(td)
            reg_pkg = td_path / "registry" / "nxmath" / "0.1.0"
            reg_pkg.mkdir(parents=True)
            (reg_pkg / "nxmath.nx").write_text("def add2(a: i64, b: i64) -> i64:\n    return a + b\n", encoding="utf-8")
            proj = td_path / "proj"
            proj.mkdir()
            app = proj / "main.nx"
            app.write_text("import nxmath\nprint(nxmath.add2(7, 8))\n", encoding="utf-8")
            exe = proj / ("main.exe" if os.name == "nt" else "main")
            old_reg = os.environ.get("NX_REGISTRY")
            os.environ["NX_REGISTRY"] = str(td_path / "registry")
            try:
                self.assertEqual(cli_main(["build", str(app), "-o", str(exe)]), 0)
            finally:
                if old_reg is None:
                    os.environ.pop("NX_REGISTRY", None)
                else:
                    os.environ["NX_REGISTRY"] = old_reg
            self.assertTrue((proj / ".nx_pkgs" / "nxmath" / "nxmath.nx").is_file())
            proc = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout.strip(), "15")

    def test_polyglot_auto_header_discovery(self):
        with tempfile.TemporaryDirectory(prefix="nx2-poly-") as td:
            td_path = Path(td)
            hdr = td_path / "fastmath.h"
            hdr.write_text("static inline double fast_mul(double a, double b) { return a * b; }\n", encoding="utf-8")
            sigs = parse_c_header_signatures(hdr)
            self.assertEqual(sigs["fast_mul"], (("f64", "f64"), "f64"))
            info = auto_discover_foreign_module("cpp", "fastmath", [td_path])
            self.assertEqual(info["header"], hdr)
            app = td_path / "main.nx"
            app.write_text('import cpp.fastmath as fm\n@cap("ffi.cpp")\ndef main() -> i32:\n    print(fm.fast_mul(2.5, 4.0))\n    return 0\n', encoding="utf-8")
            exe = td_path / ("main.exe" if os.name == "nt" else "main")
            self.assertEqual(cli_main(["build", str(app), "-o", str(exe)]), 0)
            proc = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout.strip(), "10")

    def test_llvm_backend_ir_emission(self):
        src = '''
def add(a: i64, b: i64) -> i64:
    return a + b

def main() -> i32:
    s: i64 = 0
    for i in range(4):
        s += add(i, 1)
    print(s)
    return 0
'''
        ll = compile_to_llvm_ir(src)
        self.assertIn("target triple =", ll)
        self.assertIn("define i64 @nx_add(i64 %arg_a, i64 %arg_b)", ll)
        self.assertIn("define i32 @main()", ll)
        self.assertIn("call i32 (i8*, ...) @printf", ll)

    def test_asm_backend_incremental_and_cache_manager(self):
        with tempfile.TemporaryDirectory(prefix="nx2-asm-") as td:
            td_path = Path(td)
            cache_dir = td_path / "cache"
            old_cache = os.environ.get("NEXIR_CACHE_DIR")
            os.environ["NEXIR_CACHE_DIR"] = str(cache_dir)
            try:
                src = "def sq(x: i64) -> i64:\n    return x * x\nprint(sq(9))\n"
                sfile = td_path / "app.nx"
                sfile.write_text(src, encoding="utf-8")
                asm_out = td_path / "app.s"
                exe_out = td_path / ("app.exe" if os.name == "nt" else "app")
                self.assertEqual(cli_main(["build", str(sfile), "--backend", "asm", "--emit-asm", str(asm_out), "-o", str(exe_out)]), 0)
                self.assertTrue(asm_out.is_file())
                self.assertGreater(asm_out.stat().st_size, 100)
                proc = subprocess.run([str(exe_out)], capture_output=True, text=True)
                self.assertEqual(proc.stdout.strip(), "81")
                # Second build hits incremental cache
                self.assertEqual(cli_main(["build", str(sfile), "--backend", "asm", "--incremental", "-o", str(exe_out)]), 0)
                mgr = CacheManager(cache_dir)
                st = mgr.status()
                self.assertGreaterEqual(st["hits"], 1)
                self.assertGreaterEqual(st["counts"]["asm"], 1)
                self.assertGreaterEqual(st["counts"]["inc"], 1)
                self.assertIn("sq", function_ast_hashes(src))
                mgr.tune(max_mb=0.0)
                self.assertEqual(mgr.status()["counts"]["asm"], 0)
                mgr.clean()
            finally:
                if old_cache is None:
                    os.environ.pop("NEXIR_CACHE_DIR", None)
                else:
                    os.environ["NEXIR_CACHE_DIR"] = old_cache

    def test_repl_session(self):
        inputs = iter(["x = 6", "def mul7(n: i64) -> i64:", "    return n * 7", "", "mul7(x)", ":quit"])
        outputs: list[str] = []

        def fake_run(src_path: Path) -> None:
            exe = src_path.with_suffix(".exe" if os.name == "nt" else "")
            self.assertEqual(cli_main(["build", str(src_path), "-o", str(exe)]), 0)
            proc = subprocess.run([str(exe)], capture_output=True, text=True)
            outputs.append(proc.stdout.strip())

        buf = io.StringIO()
        with patch("builtins.input", side_effect=lambda prompt="": next(inputs)), patch("sys.stdout", buf):
            rc = run_repl(fake_run)
        self.assertEqual(rc, 0)
        self.assertIn("42", outputs)


if __name__ == "__main__":
    unittest.main()
