import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from nexir.cli import _build
from nexir.compiler import NexirError, compile_source


ROOT = Path(__file__).resolve().parents[1]


class LanguageExtensionTests(unittest.TestCase):
    def test_examples_compile_and_run(self):
        cases = {
            'structs.nx': '5\nPair released\n',
            'ownership.nx': '10\n11\nCounter released\n',
            'generics.nx': '5\n4\n',
            'generic_simd.nx': '5 4\n',
            'traits.nx': '42\n',
            'trait_methods.nx': '12\n',
            'trait_self.nx': '5\n',
            'simd_parallel.nx': '14 6 4\n',
            'result_errors.nx': '5\nsecond call failed\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            for filename, expected in cases.items():
                with self.subTest(filename=filename):
                    output = Path(directory) / (filename.removesuffix('.nx') + ('.exe' if os.name == 'nt' else ''))
                    _build(ROOT / 'examples' / filename, output)
                    result = subprocess.run([str(output)], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, expected)

    def assert_rejected(self, source, fragment):
        with self.assertRaises(NexirError) as error:
            compile_source(source)
        self.assertIn(fragment, str(error.exception))

    def test_fn_requires_return_annotation(self):
        self.assert_rejected('fn main():\n    pass\n', 'explicit return type')

    def test_fn_rejects_dynamic_local(self):
        self.assert_rejected('fn main() -> i32:\n    item: dyn = 1\n    return 0\n', 'cannot use dyn')

    def test_owned_use_after_move(self):
        self.assert_rejected(
            'struct Box:\n    value: i32\n'
            '@cap("memory.alloc")\nfn main() -> i32:\n'
            '    first = owned(Box(1))\n    second = move(first)\n'
            '    print(first.value)\n    return 0\n', 'use after move')

    def test_borrowed_cannot_mutate(self):
        self.assert_rejected(
            'struct Box:\n    value: i32\n'
            'fn change(box: borrowed[Box]) -> void:\n    box.value = 2\n'
            'fn main() -> i32:\n    return 0\n', 'read-only')

    def test_conflicting_borrows_in_one_call(self):
        self.assert_rejected(
            'struct Box:\n    value: i32\n'
            'fn both(a: mut[Box], b: borrowed[Box]) -> void:\n    pass\n'
            'fn main() -> i32:\n    item = Box(1)\n'
            '    both(item, item)\n    return 0\n', 'conflicting borrows')

    def test_trait_mismatch(self):
        self.assert_rejected(
            'trait HasValue:\n    fn value(self) -> i32:\n        pass\n'
            'struct Empty:\n    number: i32\n'
            'fn read[T: HasValue](item: T) -> i32:\n    return item.value()\n'
            'fn main() -> i32:\n    print(read(Empty(1)))\n    return 0\n',
            'does not implement trait')

    def test_generic_simd_width_mismatch(self):
        self.assert_rejected(
            'fn add[T: Number, size: Int](a: SIMD[T, size], b: SIMD[T, size]) -> SIMD[T, size]:\n'
            '    return a + b\n'
            'fn main() -> i32:\n'
            '    x = SIMD[f64, 2](1.0, 2.0)\n'
            '    y = SIMD[f64, 3](1.0, 2.0, 3.0)\n'
            '    z = add(x, y)\n    return 0\n',
            'one constant value')

    def test_parallel_cannot_break(self):
        self.assert_rejected(
            'fn main() -> i32:\n    for i in parallel_range(10):\n'
            '        break\n    return 0\n', 'cannot return, break or continue')

    def test_parallel_cannot_change_shared_scalar(self):
        self.assert_rejected(
            'fn main() -> i32:\n    total = 0\n'
            '    for i in parallel_range(10):\n        total += i\n'
            '    return total\n', 'cannot assign shared variables')

    def test_raise_requires_raises(self):
        self.assert_rejected('fn main() -> i32:\n    raise "bad"\n', 'raise requires a raises function')


if __name__ == '__main__':
    unittest.main()
