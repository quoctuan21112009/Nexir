import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nexir.cli import _build
from nexir.compiler import NexirError, compile_source


class FrontendTests(unittest.TestCase):
    def assert_rejected(self, source: str, fragment: str) -> None:
        with self.assertRaises(NexirError) as caught:
            compile_source(source)
        self.assertIn(fragment, str(caught.exception))

    def test_type_mismatch(self):
        self.assert_rejected('def main():\n    n: i32 = "text"\n', "expected i32, got str")

    def test_missing_return(self):
        self.assert_rejected('def f() -> i32:\n    print(1)\ndef main():\n    print(f())\n', "may finish without returning")

    def test_unknown_function(self):
        self.assert_rejected('def main():\n    mystery()\n', "unknown function")

    def test_unsupported_decorator(self):
        self.assert_rejected('@agent.action\ndef main():\n    pass\n', "supported decorators")

    def test_reject_python_import(self):
        self.assert_rejected('import os\ndef main():\n    pass\n', "only function definitions")

    def test_undefined_variable(self):
        self.assert_rejected('def main():\n    print(x)\n', "unknown variable")

    def test_bool_condition_required(self):
        self.assert_rejected('def main():\n    if 1:\n        print(1)\n', "condition must be bool")

    def test_void_value_cannot_be_assigned(self):
        self.assert_rejected('def f():\n    pass\ndef main():\n    x = f()\n', "cannot assign a void result")

    def test_unsupported_class(self):
        self.assert_rejected('class C:\n    pass\ndef main():\n    pass\n', "only function definitions")

    def test_break_outside_loop(self):
        self.assert_rejected('def main():\n    break\n', "break is only valid inside a loop")

    def test_continue_outside_loop(self):
        self.assert_rejected('def main():\n    continue\n', "continue is only valid inside a loop")

    def test_block_local_is_not_visible_outside(self):
        self.assert_rejected('def main():\n    if True:\n        x = 1\n    print(x)\n', "unknown variable: x")


class NativeTests(unittest.TestCase):
    def build_and_run(self, source: str, *, expected_code: int = 0):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.nx"
            path.write_text(source, encoding="utf-8")
            output = Path(directory) / ("test.exe" if os.name == "nt" else "test")
            _build(path, output)
            # This process executes the native binary directly, without importing NEXIR.
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            proc = subprocess.run([str(output)], capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            return proc.stdout, proc.stderr

    def test_arithmetic_function_and_print(self):
        stdout, _ = self.build_and_run(
            'def add(a: i32, b: i32) -> i32:\n    return a + b\n'
            'def main():\n    print("NEXIR", add(10, 20))\n'
        )
        self.assertEqual(stdout, "NEXIR 30\n")

    def test_loop_and_if(self):
        stdout, _ = self.build_and_run(
            'def main():\n    total: i64 = 0\n'
            '    for i in range(1, 6):\n        if i % 2 == 0:\n'
            '            total += i\n    print(total)\n'
        )
        self.assertEqual(stdout, "6\n")

    def test_negative_step_and_while(self):
        stdout, _ = self.build_and_run(
            'def main():\n    total: i64 = 0\n'
            '    for i in range(5, 0, -2):\n        total += i\n'
            '    while total > 0:\n        total -= 3\n'
            '    print(total)\n'
        )
        self.assertEqual(stdout, "0\n")

    def test_break_continue_and_nested_if(self):
        stdout, _ = self.build_and_run(
            'def main():\n    total: i64 = 0\n'
            '    for i in range(10):\n'
            '        if i == 2:\n            continue\n'
            '        if i == 5:\n            break\n'
            '        total += i\n'
            '    n: i32 = 0\n'
            '    while n < 3:\n'
            '        n += 1\n'
            '        if n == 2:\n            continue\n'
            '        total += n\n'
            '    print(total)\n'
        )
        self.assertEqual(stdout, "12\n")

    def test_same_name_reassigned_after_if(self):
        stdout, _ = self.build_and_run(
            'def main():\n    if True:\n        x = 1\n'
            '    x = 2\n    print(x)\n'
        )
        self.assertEqual(stdout, "2\n")

    def test_strings_booleans_and_float(self):
        stdout, _ = self.build_and_run(
            'def main():\n    print("a\\\"b", "a" == "a", 1.5 + 2.0)\n'
        )
        self.assertEqual(stdout, 'a"b true 3.5\n')

    def test_main_exit_code(self):
        stdout, _ = self.build_and_run('def main() -> i32:\n    return 7\n', expected_code=7)
        self.assertEqual(stdout, "")

    def test_minimum_signed_literals(self):
        stdout, _ = self.build_and_run(
            'def main():\n    small: i32 = -2147483648\n'
            '    large: i64 = -9223372036854775808\n'
            '    print(small, large)\n'
        )
        self.assertEqual(stdout, "-2147483648 -9223372036854775808\n")

    def test_range_zero_step_fails_at_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zero.nx"
            path.write_text('def main():\n    for i in range(0, 3, 0):\n        print(i)\n', encoding="utf-8")
            output = Path(directory) / ("zero.exe" if os.name == "nt" else "zero")
            _build(path, output)
            proc = subprocess.run([str(output)], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("range step cannot be zero", proc.stderr)


if __name__ == "__main__":
    unittest.main()
