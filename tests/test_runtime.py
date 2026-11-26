import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from nexir.cli import _build
from nexir.compiler import NexirError, compile_source


class RuntimeTests(unittest.TestCase):
    def run_source(self, source: str, *, cpp_sources=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "program.nx"
            output = Path(folder) / ("program.exe" if os.name == "nt" else "program")
            path.write_text(source, encoding="utf-8")
            _build(path, output, cpp_sources=cpp_sources)
            return subprocess.run([str(output)], capture_output=True, text=True)

    def test_tagged_value_layout_and_copy(self):
        source = ('def main():\n    first: dyn = "old"\n    second: dyn = first\n'
                  '    first = "new"\n    print(first, second)\n'
                  '    number: dyn = 41\n    number = number + 1\n'
                  '    print(type_name(number), as_i64(number))\n')
        generated = compile_source(source)
        self.assertIn('static_assert(sizeof(NxValue) == 16', generated)
        self.assertIn('static_assert(alignof(NxValue) == 16', generated)
        result = self.run_source(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'new old\ni64 42\n')

    def test_dynamic_type_error_is_detected(self):
        result = self.run_source('def main():\n    value: dyn = "text"\n    print(as_i64(value))\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not i64', result.stderr)

    def test_ring_generation(self):
        source = ('@cap("memory.alloc")\ndef main():\n'
                  '    ring = ring_f64(2)\n    ring_write(ring, 0, 1.5)\n'
                  '    print(ring_valid(ring, 0), ring_read(ring, 0))\n'
                  '    ring_write(ring, 2, 3.5)\n'
                  '    print(ring_valid(ring, 0), ring_read(ring, 2))\n')
        result = self.run_source(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'true 1.5\nfalse 3.5\n')

    def test_ring_stale_read_fails(self):
        result = self.run_source('@cap("memory.alloc")\ndef main():\n'
                                 '    ring = ring_f64(2)\n    ring_write(ring, 0, 1.0)\n'
                                 '    ring_write(ring, 2, 2.0)\n    print(ring_read(ring, 0))\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stale or unwritten ring frame', result.stderr)

    def test_ring_allocation_requires_cap(self):
        with self.assertRaises(NexirError) as error:
            compile_source('def main():\n    ring = ring_f64(2)\n')
        self.assertIn('memory.alloc', str(error.exception))

    def test_cpp_external_function(self):
        with tempfile.TemporaryDirectory() as folder:
            cpp = Path(folder) / 'bridge.cpp'
            cpp.write_text('extern "C" double nx_square(double x) { return x*x; }\n', encoding='utf-8')
            result = self.run_source('@extern("nx_square")\ndef square(x: f64) -> f64:\n'
                                     '    pass\ndef main():\n    print(square(3.0))\n', cpp_sources=[cpp])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '9\n')

    def test_non_language_import_is_rejected(self):
        with self.assertRaises(NexirError):
            compile_source('import arbitrary_package\ndef main():\n    pass\n')

    def test_cli_emit_cpp_without_toolchain(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'program.nx'
            output = Path(folder) / 'program.cpp'
            source.write_text('def main():\n    print(42)\n', encoding='utf-8')
            result = subprocess.run([sys.executable, '-m', 'nexir', 'emit', str(source), '-o', str(output)],
                                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('nx_main', output.read_text(encoding='utf-8'))

    def test_portable_wire_format(self):
        gcc = shutil.which('gcc') or str(Path(r'C:\msys64\ucrt64\bin\gcc.exe'))
        if not Path(gcc).is_file():
            self.skipTest('GCC unavailable for C99 header test')
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'wire.c'
            output = Path(folder) / ('wire.exe' if os.name == 'nt' else 'wire')
            source.write_text(
                '#include "nx_value_wire.h"\n#include <stdio.h>\n'
                'int main(void) {\n'
                '  uint8_t wire[NX_WIRE_SIZE]; int64_t i = 0; double f = 0;\n'
                '  nx_wire_i64(wire, -42);\n'
                '  if (wire[0] != 1 || wire[8] != 214 || wire[15] != 255) return 1;\n'
                '  if (!nx_wire_read_i64(wire, &i) || i != -42) return 2;\n'
                '  if (!nx_wire_f64(wire, 3.5) || !nx_wire_read_f64(wire, &f) || f != 3.5) return 3;\n'
                '  puts("wire ok"); return 0;\n}\n', encoding='utf-8')
            env = os.environ.copy()
            if os.name == 'nt':
                env['PATH'] = str(Path(gcc).parent) + os.pathsep + env.get('PATH', '')
            compiled = subprocess.run([gcc, '-std=c99', '-I', str(Path(__file__).resolve().parents[1] / 'nexir' / 'include'),
                                       str(source), '-o', str(output)], capture_output=True, text=True, env=env)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, 'wire ok\n')

    def test_cli_init_and_run(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / 'starter'
            init = subprocess.run([sys.executable, '-m', 'nexir', 'init', str(project)],
                                  cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(init.returncode, 0, init.stderr)
            self.assertTrue((project / 'main.nx').exists())
            run = subprocess.run([sys.executable, '-m', 'nexir', 'run', str(project / 'main.nx')],
                                 cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout, 'Hello from NEXIR\n')


if __name__ == '__main__':
    unittest.main()
