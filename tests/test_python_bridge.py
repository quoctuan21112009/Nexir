import unittest

from nexir.cli import _enable_python_startup
from nexir.compiler import compile_source
from nexir.python_bridge import PYTHON_BRIDGE_CPP


class PythonBridgeFrontendTests(unittest.TestCase):
    def test_existing_extern_abi_accepts_f64_span_bridge(self):
        source = (
            '@extern("nx_py_call_f64_span")\n'
            'def predict_batch(module: str, function: str, values: Span[f64]) -> f64:\n'
            '    pass\n'
            '@cap("memory.alloc")\n'
            'def main():\n'
            '    values = [1.0, 2.0]\n'
            '    print(predict_batch("ml_worker", "predict", values))\n'
        )
        generated = compile_source(source)
        self.assertIn("nx_py_call_f64_span", generated)
        self.assertIn("nx_span_f64", generated)

    def test_python_initialization_and_preloads_are_injected_once(self):
        generated = compile_source("def main():\n    pass\n")
        enabled = _enable_python_startup(generated, ["numpy", "sklearn"])
        self.assertEqual(enabled.count("nx_py_initialize();"), 1)
        self.assertEqual(enabled.count('nx_py_preload("numpy")'), 1)
        self.assertEqual(enabled.count('nx_py_preload("sklearn")'), 1)

    def test_bridge_uses_ctypes_from_address_and_caches_imports(self):
        self.assertIn('"from_address"', PYTHON_BRIDGE_CPP)
        self.assertIn("modules.emplace", PYTHON_BRIDGE_CPP)
        self.assertIn("functions.emplace", PYTHON_BRIDGE_CPP)
        self.assertIn("Py_Initialize()", PYTHON_BRIDGE_CPP)


if __name__ == "__main__":
    unittest.main()
