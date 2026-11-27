"""Comprehensive NX 2.0 integration tests covering:
- class/inheritance/Result/Optional/enum/match
- i8..i128 / u8..u128 free numeric conversions
- Zero-leak cache-based NxStr / StringBuilder / format specifiers
- Multi-file & cross-language imports (Python direct, C++ direct, Rust/Go/C bindings, Python<-NX)
- Hybrid typing (static + dynamic)
- Deep File I/O & Networking
- Hybrid Async (nx.asyncio), ThreadPool/Channel/Atomic (nx.thread), GPU/SIMD (nx.gpu)
- Compile-time safety & Runtime sandbox, resource limits & audit log
- pkg.go.dev-style ecosystem (lockfile, sha256 verify, audit, doc HTML)
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

from nexir.cli import _build, main as cli_main
from nexir.compiler import NexirError, compile_file, compile_source
from nexir.ecosystem import ecosystem_command
from nexir.interop import generate_bindings, import_nx


class Nx2IntegrationTests(unittest.TestCase):
    def build_and_run(self, source: str, *, extra_files: dict[str, str] | None = None,
                      env_extra: dict[str, str] | None = None, debug: bool = False,
                      expected_code: int = 0) -> tuple[str, str, int]:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for rel, content in (extra_files or {}).items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
            main_nx = root / "main.nx"
            main_nx.write_text(source, encoding="utf-8")
            out = root / ("main.exe" if os.name == "nt" else "main")
            py_dll = _build(main_nx, out, debug=debug)
            env = os.environ.copy()
            if py_dll:
                if os.name == "nt":
                    env["PATH"] = str(py_dll) + os.pathsep + env.get("PATH", "")
                env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
            if env_extra:
                env.update(env_extra)
            proc = subprocess.run([str(out)], cwd=root, capture_output=True, text=True, env=env, timeout=30)
            if expected_code is not None:
                self.assertEqual(proc.returncode, expected_code, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
            return proc.stdout, proc.stderr, proc.returncode

    def test_numeric_128bit_and_free_conversions(self):
        src = '''
def add_128(a: i128, b: i32) -> i128:
    return a + b

def main() -> i32:
    big: i128 = 170141183460469231731687303715884105720
    small: i32 = 7
    res = add_128(big, small)
    x64: i64 = 100
    x32: i32 = x64
    xu8: u8 = 250
    xi16: i16 = xu8
    parsed = i128("100000000000000000000000")
    print(res, x32, xi16, parsed + 1)
    return 0
'''
        out, _, _ = self.build_and_run(src)
        self.assertEqual(
            out.strip(),
            "170141183460469231731687303715884105727 100 250 100000000000000000000001",
        )

    def test_memory_model_no_leak_and_cache_and_stringbuilder_fstrings(self):
        src = '''
import nx.debug

@cap("memory.alloc")
def make_many_strings() -> i64:
    acc: str = ""
    for i in range(5000):
        s = f"item-payload-{i:04d}-value-{3.14159:.2f}"
        if i == 4999:
            acc = s
    sb = StringBuilder(64)
    sb.append("ok:")
    sb.append(acc)
    res = sb.to_str()
    print(res)
    return nx.debug.live_blocks()

@cap("memory.alloc")
def main() -> i32:
    before = nx.debug.live_blocks()
    during = make_many_strings()
    after = nx.debug.live_blocks()
    hits = nx.debug.cache_hits()
    print(hits > 0, after <= before + 2)
    return 0
'''
        out, err, _ = self.build_and_run(src, env_extra={"NX_DEBUG_MEM": "1"})
        lines = out.strip().splitlines()
        self.assertEqual(lines[0], "ok:item-payload-4999-value-3.14")
        self.assertEqual(lines[1], "true true")
        self.assertIn("[NX-MEM]", err)
        self.assertIn("live_blocks=0", err)

    def test_class_inheritance_result_optional_and_propagation(self):
        src = '''
class Animal:
    name: str
    def speak(self) -> str:
        return self.name

class Dog(Animal):
    age: i32
    @cap("memory.alloc")
    def speak(self) -> str:
        return self.name + "!"

def find_dog(ok: bool) -> Optional[Dog]:
    if ok:
        return Dog("Rex", 3)
    return None

def load_dog(ok: bool) -> Dog raises:
    if not ok:
        raise "missing dog"
    return Dog("Max", 5)

@cap("memory.alloc")
def greet_dog(ok: bool) -> str raises:
    d = load_dog(ok)?
    return d.speak()

@cap("memory.alloc")
def main() -> i32:
    d1 = find_dog(True)
    if d1 is not None:
        print(d1.speak(), d1.age)
    d2 = find_dog(False)
    print(d2.is_none())
    try:
        msg = greet_dog(True)
        print(msg)
    except Err as e:
        print("err:", e)
    finally:
        print("fin1")
    try:
        msg = greet_dog(False)
        print(msg)
    except Err as e:
        print("caught:", e)
    return 0
'''
        out, _, _ = self.build_and_run(src)
        self.assertEqual(out.strip().splitlines(), ["Rex! 3", "true", "Max!", "fin1", "caught: missing dog"])

    def test_advanced_enum_and_exhaustive_match(self):
        src = '''
enum Shape:
    Circle(radius: f64)
    Rect(w: f64, h: f64)
    Point

    def area(self) -> f64:
        match self:
            case Shape.Circle(r):
                return 3.0 * r * r
            case Shape.Rect(w, h):
                return w * h
            case Shape.Point:
                return 0.0

def main() -> i32:
    c = Shape.Circle(2.0)
    r = Shape.Rect(3.0, 4.0)
    p = Shape.Point
    print(c.area(), r.area(), p.area())
    return 0
'''
        out, _, _ = self.build_and_run(src)
        self.assertEqual(out.strip(), "12 12 0")

        # Non-exhaustive match must fail at compile time!
        bad_src = '''
enum Light:
    Red
    Green
def main() -> i32:
    x = Light.Red
    match x:
        case Light.Red:
            print(1)
    return 0
'''
        with self.assertRaises(NexirError) as ctx:
            compile_source(bad_src)
        self.assertIn("non-exhaustive match", str(ctx.exception))

    def test_pythonic_collections_and_hybrid_typing(self):
        src = '''
def hybrid_add(a, b):
    return a + b

@cap("memory.alloc")
def main() -> i32:
    nums = [10, 20, 30]
    nums.append(40)
    nums.insert(1, 15)
    print(nums[-1], len(nums), sum(nums))
    d: Dict[str, i32] = {"a": 1, "b": 2}
    print(d.get("missing", 99))
    for k, v in d.items():
        if k == "a":
            print(k, v)
    for i, val in enumerate(nums):
        if i == 0:
            print(i, val)
    mixed = [1, "hi", True]
    print(len(mixed), mixed[1])
    print(hybrid_add(20, 22), hybrid_add("hy", "brid"))
    return 0
'''
        out, _, _ = self.build_and_run(src)
        self.assertEqual(out.strip().splitlines(), [
            "40 5 115",
            "99",
            "a 1",
            "0 10",
            "3 hi",
            "42 hybrid",
        ])

    def test_multifile_and_direct_cpp_and_python_without_wrappers(self):
        extra = {
            "math_mod.nx": '''
class Helper:
    val: i32
def helper(x: i32) -> i32:
    return x * 3
''',
            "other_mod.nx": '''
def helper(x: i32) -> i32:
    return x + 100
''',
        }
        src = '''
import math_mod
from other_mod import helper as other_helper
import cpp.cmath as cmath
import py.math as pymath
import py.statistics as pystats

@cap("memory.alloc", "ffi.cpp", "python.interop")
def main() -> i32:
    h = math_mod.Helper(7)
    print(math_mod.helper(h.val), other_helper(5))
    print(cmath.hypot(3.0, 4.0))
    py_val: f64 = pymath.sqrt(81.0)
    nums = [2.0, 4.0, 6.0, 8.0]
    mean_val: f64 = pystats.mean(nums)
    print(py_val, mean_val)
    return 0
'''
        out, _, _ = self.build_and_run(src, extra_files=extra)
        self.assertEqual(out.strip().splitlines(), ["21 105", "5", "9 5"])

    def test_bidirectional_python_nx_and_bindings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mod_nx = root / "fastlib.nx"
            mod_nx.write_text(
                "@export\n"
                "def fast_sum(a: i64, b: i64) -> i64:\n"
                "    return a + b\n"
                "@export\n"
                "def dot_sum(data: Span[f64]) -> f64:\n"
                "    acc: f64 = 0.0\n"
                "    for i in range(len(data)):\n"
                "        acc += data[i]\n"
                "    return acc\n",
                encoding="utf-8",
            )
            proxy = import_nx(mod_nx, cache_dir=root / "cache")
            try:
                self.assertEqual(proxy.fast_sum(17, 25), 42)
                self.assertAlmostEqual(proxy.dot_sum([1.5, 2.5, 3.0]), 7.0)
            finally:
                proxy.close()

            for lang in ("c", "cpp", "python", "rust", "go"):
                bpath = generate_bindings(mod_nx, lang, root / "bind")
                self.assertTrue(bpath.is_file())
                txt = bpath.read_text(encoding="utf-8")
                self.assertIn("fast_sum", txt)

    def test_deep_io_networking_and_sandbox_audit(self):
        # Start a local HTTP server for nx.net.http_get test
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"hello-nx-net"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass

        srv = HTTPServer(("127.0.0.1", 0), Handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            src = f'''
import nx.fs as fs
import nx.net as net
import nx.sandbox as sb

@cap("filesystem.read", "filesystem.write", "filesystem.copy", "filesystem.delete", "network.connect", "memory.alloc")
def run_io(port: i32) -> i32 raises:
    fs.write_text("a.txt", "hello world")
    fs.edit_file("a.txt", "world", "nexir")
    fs.copy_file("a.txt", "b.txt")
    txt = fs.read_text("b.txt")?
    info = fs.stat("b.txt")?
    print(txt, info.size)
    with open("c.txt", "w") as f:
        f.write("raii")
    print(read_text("c.txt"))
    fs.delete_file("a.txt")
    resp = net.http_get(f"http://127.0.0.1:{port}/")?
    print(resp.status, resp.body)
    return 0

@cap("filesystem.read", "filesystem.write", "filesystem.copy", "filesystem.delete", "network.connect", "memory.alloc")
def main() -> i32:
    try:
        rc = run_io({port})
        return rc
    except Err as e:
        print("err:", e)
        return 1
'''
            with tempfile.TemporaryDirectory() as folder:
                audit_file = Path(folder) / "audit.jsonl"
                out, _, _ = self.build_and_run(src, env_extra={"NX_AUDIT_LOG": str(audit_file)})
                lines = out.strip().splitlines()
                self.assertEqual(lines, ["hello nexir 11", "raii", "200 hello-nx-net"])
                self.assertTrue(audit_file.is_file())
                audit_text = audit_file.read_text(encoding="utf-8")
                self.assertIn("fs.access", audit_text)
                self.assertIn("net.access", audit_text)
        finally:
            srv.shutdown()
            srv.server_close()

        # Verify sandbox memory cap & path denial
        src_sb = '''
import nx.fs as fs
@cap("filesystem.read", "memory.alloc")
def main() -> i32:
    r = fs.read_text("secret.txt")
    if r.is_err():
        print("blocked:", r.error())
    return 0
'''
        out, _, _ = self.build_and_run(src_sb, env_extra={"NX_SANDBOX": "1", "NX_DENY_FS": "1"})
        self.assertIn("blocked: sandbox denied", out)

        mem_src = '''
@cap("memory.alloc")
def main() -> i32:
    big = zeros_f64(100000)
    print(len(big))
    return 0
'''
        _, err, code = self.build_and_run(mem_src, env_extra={"NX_SANDBOX": "1", "NX_MAX_MEMORY": "1024"}, expected_code=None)
        self.assertNotEqual(code, 0)
        self.assertIn("memory limit exceeded", err)

    def test_hybrid_async_thread_and_gpu_runtime(self):
        src = '''
import nx.asyncio as aio
import nx.thread as th
import nx.gpu as gpu

@cap("memory.alloc")
async def compute_part(x: i64) -> i64:
    await aio.sleep(5)
    return x * x

@cap("memory.alloc", "gpu.compute")
async def async_main() -> i32:
    f1 = compute_part(3)
    f2 = compute_part(4)
    r1, r2 = aio.gather(f1, f2)
    print("gather:", r1 + r2)

    ch = Channel[i64](4)
    ch.send(10)
    ch.send(20)
    v1 = ch.recv()
    v2 = ch.recv()
    print("chan:", value_or(v1, 0) + value_or(v2, 0))

    atom = th.AtomicI64(5)
    atom.add(7)
    print("atom:", atom.load(), th.cpu_count() >= 1)

    a = gpu.to_device([1.0, 2.0, 3.0, 4.0])
    b = gpu.to_device([2.0, 3.0, 4.0, 5.0])
    c = gpu.mul(a, b)
    s = gpu.reduce_sum(c)
    print("gpu_sum:", s)

    m1 = [1.0, 2.0, 3.0, 4.0]
    m2 = [2.0, 0.0, 1.0, 2.0]
    mm = await gpu.matmul_async(m1, m2, 2, 2, 2)
    print("matmul:", mm[0], mm[3])
    return 0

@cap("memory.alloc", "gpu.compute")
def main() -> i32:
    return aio.run(async_main())
'''
        out, _, _ = self.build_and_run(src)
        self.assertEqual(out.strip().splitlines(), [
            "gather: 25",
            "chan: 30",
            "atom: 12 true",
            "gpu_sum: 40",
            "matmul: 4 8",
        ])

    def test_debug_stack_trace_and_safety_checks(self):
        # Compile-time flow-sensitive move safety check
        moved_branch = '''
class Box:
    v: i32
@cap("memory.alloc")
def main() -> i32:
    a = owned(Box(1))
    if True:
        b = move(a)
    print(a.v)
    return 0
'''
        with self.assertRaises(NexirError) as ctx:
            compile_source(moved_branch)
        self.assertIn("use after move", str(ctx.exception))

        # Compile-time FFI security policy blocks dangerous header/symbol without process.exec
        bad_ffi = '''
import cpp.unistd as u
@cap("ffi.cpp")
def main() -> i32:
    return 0
'''
        with self.assertRaises(NexirError) as ctx2:
            compile_source(bad_ffi)
        self.assertIn("FFI security policy", str(ctx2.exception))

        # Debug mode runtime stack trace
        crash_src = '''
@cap("memory.alloc")
def boom() -> i32:
    arr = [1, 2]
    return arr[5]
@cap("memory.alloc")
def main() -> i32:
    print(boom())
    return 0
'''
        _, err, code = self.build_and_run(crash_src, debug=True, expected_code=None)
        self.assertNotEqual(code, 0)
        self.assertIn("NX Stack Trace", err)
        self.assertIn("boom", err)

    def test_pkg_go_dev_ecosystem_registry_lockfile_audit_docs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reg = root / "registry"
            lib_dir = root / "mylib"
            app_dir = root / "myapp"
            lib_dir.mkdir()
            app_dir.mkdir()

            old_reg = os.environ.get("NX_REGISTRY")
            os.environ["NX_REGISTRY"] = str(reg)
            try:
                ecosystem_command(argparse.Namespace(action="init", path=lib_dir, name="nx-math"))
                (lib_dir / "src" / "main.nx").write_text(
                    '"""Math utilities for NEXIR."""\n'
                    'def square(x: i32) -> i32:\n'
                    '    return x * x\n',
                    encoding="utf-8",
                )
                ecosystem_command(argparse.Namespace(action="publish", path=lib_dir))

                ecosystem_command(argparse.Namespace(action="init", path=app_dir, name="my-app"))
                ecosystem_command(argparse.Namespace(action="add", path=app_dir, package="nx-math", version="^0.1.0"))
                ecosystem_command(argparse.Namespace(action="install", path=app_dir))
                ecosystem_command(argparse.Namespace(action="verify", path=app_dir))
                ecosystem_command(argparse.Namespace(action="audit", path=app_dir))
                ecosystem_command(argparse.Namespace(action="doc", path=app_dir, format="html", serve=False))

                self.assertTrue((app_dir / "nx.lock").is_file())
                self.assertTrue((app_dir / "build" / "docs" / "index.html").is_file())
                doc_html = (app_dir / "build" / "docs" / "index.html").read_text(encoding="utf-8")
                self.assertIn("pkg.nx.dev", doc_html)

                # Tamper installed dependency and verify integrity check catches it!
                (app_dir / ".nx" / "packages" / "nx-math" / "src" / "main.nx").write_text("def evil():\n    pass\n", encoding="utf-8")
                with self.assertRaises(NexirError) as ctx:
                    ecosystem_command(argparse.Namespace(action="verify", path=app_dir))
                self.assertIn("sha256 mismatch", str(ctx.exception))
            finally:
                if old_reg is None:
                    os.environ.pop("NX_REGISTRY", None)
                else:
                    os.environ["NX_REGISTRY"] = old_reg

    def test_native_abi_c99_header_compiles(self):
        gcc = shutil.which("gcc") or str(Path(r"C:\msys64\ucrt64\bin\gcc.exe"))
        if not Path(gcc).is_file():
            self.skipTest("GCC unavailable for C99 ABI header test")
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "abi_check.c"
            out = Path(folder) / ("abi_check.exe" if os.name == "nt" else "abi_check")
            src.write_text(
                '#include "nx_native_abi.h"\n#include <stdio.h>\n'
                'int main(void) {\n'
                '  NxAbiValue v; v.tag = NX_ABI_I64; v.flags = 0; v.payload.i64 = 42;\n'
                '  if (NX_ABI_VERSION != 2 || v.payload.i64 != 42) return 1;\n'
                '  puts("abi ok"); return 0;\n}\n',
                encoding="utf-8",
            )
            env = os.environ.copy()
            if os.name == "nt":
                env["PATH"] = str(Path(gcc).parent) + os.pathsep + env.get("PATH", "")
            comp = subprocess.run(
                [gcc, "-std=c99", "-Wall", "-Wextra", "-I", str(Path(__file__).resolve().parents[1] / "nexir" / "include"), str(src), "-o", str(out)],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(comp.returncode, 0, comp.stderr)
            run = subprocess.run([str(out)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout.strip(), "abi ok")

    def test_vscode_and_folder_icon_extension_metadata(self):
        root = Path(__file__).resolve().parents[1]
        ext_dir = root / "editors" / "vscode"
        pkg = json.loads((ext_dir / "package.json").read_text(encoding="utf-8"))
        lang = pkg["contributes"]["languages"][0]
        self.assertEqual(lang["id"], "nexir")
        self.assertIn(".nx", lang["extensions"])
        self.assertIn("icon", lang)
        self.assertTrue((ext_dir / "icons" / "nexir.svg").is_file())
        self.assertTrue((ext_dir / "icons" / "nexir.png").is_file())
        self.assertTrue((ext_dir / "icons" / "nexir.ico").is_file())
        theme = json.loads((ext_dir / "icons" / "nexir-icon-theme.json").read_text(encoding="utf-8"))
        self.assertEqual(theme["fileExtensions"]["nx"], "_nexir")
        self.assertEqual(theme["languageIds"]["nexir"], "_nexir")
        self.assertEqual(cli_main(["icon", "status"]), 0)


if __name__ == "__main__":
    unittest.main()
