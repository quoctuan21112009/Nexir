import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest

from nexir.cli import _build
from nexir.compiler import NexirError, compile_source


class ArrayCapabilityTests(unittest.TestCase):
    def build_run(self, source: str):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "case.nx"
            output = Path(folder) / ("case.exe" if os.name == "nt" else "case")
            path.write_text(source, encoding="utf-8")
            _build(path, output)
            return subprocess.run([str(output)], capture_output=True, text=True)

    def reject(self, source: str, fragment: str):
        with self.assertRaises(NexirError) as error:
            compile_source(source)
        self.assertIn(fragment, str(error.exception))

    def test_span_mutation_and_sum(self):
        result = self.build_run(
            'def sum_data(data: Span[f32]) -> f32:\n'
            '    total: f32 = f32(0)\n'
            '    for i in range(len(data)):\n        total += data[i]\n'
            '    return total\n'
            'def main():\n    values = [f32(1), f32(2), f32(3)]\n'
            '    values[1] = f32(4)\n    print(sum_data(values))\n'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "8\n")

    def test_bounds_check_read(self):
        result = self.build_run('def main():\n    values = [1.0, 2.0]\n    print(values[2])\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("index out of bounds", result.stderr)

    def test_bounds_check_write_negative(self):
        result = self.build_run('def main():\n    values = [1.0]\n    values[-1] = 4.0\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("index out of bounds", result.stderr)

    def test_readonly_span_cannot_be_mutated(self):
        self.reject('def change(data: Span[f32]):\n    data[0] = f32(2)\ndef main():\n    pass\n', "assignment requires MutSpan")

    def test_nested_array_cannot_escape_scope(self):
        self.reject('def main():\n    data = [1.0]\n    if True:\n        data = [2.0]\n',
                    "cannot rebind an outer Span")

    def test_allocation_requires_cap(self):
        self.reject('def main():\n    data = zeros_f32(3)\n', 'memory.alloc')

    def test_file_read_requires_two_caps(self):
        self.reject('def main():\n    print(read_text("a.txt"))\n', 'filesystem.read')
        self.reject('@cap("filesystem.read")\ndef main():\n    print(read_text("a.txt"))\n', 'memory.alloc')

    def test_network_requires_cap(self):
        self.reject('def main():\n    print(tcp_connect("localhost", 1))\n', 'network.connect')

    def test_network_connect_with_cap(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            result = self.build_run(f'@cap("network.connect")\ndef main():\n'
                                    f'    print(tcp_connect("127.0.0.1", {port}))\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "true\n")

    def test_transitive_capability(self):
        self.reject('@cap("memory.alloc")\ndef helper():\n    zeros_f32(1)\n'
                    'def main():\n    helper()\n', 'memory.alloc')

    def test_allocation_with_cap(self):
        result = self.build_run('@cap("memory.alloc")\ndef main():\n'
                                '    data = zeros_f64(2)\n    data[1] = 3.5\n    print(data[1])\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "3.5\n")

    def test_file_read_with_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            text = Path(folder) / "input.txt"
            text.write_text("hello", encoding="utf-8")
            source = f'@cap("filesystem.read", "memory.alloc")\ndef main():\n    print(read_text({str(text)!r}))\n'
            result = self.build_run(source)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "hello\n")


if __name__ == "__main__":
    unittest.main()
