# Nexir CPython bridge (initial supported slice)

This feature embeds CPython into a native Nexir executable. Nexir still owns the
hot loops and numeric work; use one boundary crossing to hand a completed
`Span[f64]`/`MutSpan[f64]` batch to Python for an operation that needs an
existing Python package (for example a scikit-learn inference call).

## Nexir declaration

The bridge exports a C ABI function that takes the module name, callable name,
and a span. Declare it with Nexir's existing `@extern` feature:

```nx
@extern("nx_py_call_f64_span")
def predict_batch(module: str, function: str, values: Span[f64]) -> f64:
    pass

def main():
    values = [1.0, 2.0, 3.0]
    result = predict_batch("ml_worker", "predict", values)
    print(result)
```

To permit Python to write into a Nexir-owned buffer, declare
`@extern("nx_py_call_f64_mutspan")` and use `MutSpan[f64]`. Do not retain the
Python ctypes view beyond the call; its storage is owned by Nexir and remains
valid only while the native span is alive. Avoid concurrent access while
Python is using the buffer. The bridge currently supports one f64 span argument
and a numeric scalar return value.

In `ml_worker.py`, use NumPy's ctypes adapter to retain the same memory rather
than converting it to a new array:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # module is imported once and kept in sys.modules

def predict(values):
    view = np.ctypeslib.as_array(values)  # view over Nexir memory; no input copy
    return float(view.mean())
```

Whether a downstream sklearn estimator copies internally depends on that
estimator and its options; this bridge guarantees only that the input handed to
NumPy starts as a view over the Nexir buffer. Do not claim end-to-end zero-copy
for APIs that copy internally.

## Build and preload

With Python 3.10+ and a C++17 compiler available, the `--python` option asks
Nexir to get the matching Python include/library paths from `sysconfig`, compile
the bridge, initialize CPython before `main`, and link the interpreter. Repeat
`--python-preload` for modules that must import at startup:

```powershell
py -3 -m nexir build fast_ml.nx -o fast_ml.exe `
  --python --python-preload numpy --python-preload sklearn
```

The lower-level `--include-dir`, `--library-dir` and `--library` flags remain
available for toolchains with nonstandard Python installations. Do not mix
headers/import libraries from a different Python version or architecture than
the interpreter whose runtime will be loaded.

The executable is native, but it is **not a self-contained Python distribution**:
embedding links CPython and still requires a compatible CPython DLL/shared
library, the Python standard library, `ml_worker.py`, NumPy, sklearn, and their
native dependencies at runtime. `--python-preload` imports modules early; it
does not package them. Copying only `python3xx.dll` is not sufficient to make
NumPy/sklearn portable to a machine without their dependencies.

## Current limitations and follow-up work

This is an initial interop slice, not implementation of all planned Nexir
features. It does not yet provide generic Python objects, arbitrary Python
argument/return conversion, Python callbacks, automatic worker-file deployment,
or lifecycle/finalization APIs. Ownership/borrow checking, a security sandbox,
generic functions, a GPU backend, and a freestanding PLC runtime are separate
compiler/runtime projects and are not enabled by this bridge. In particular,
`@extern` is a native-code trust boundary, not a sandbox.

The preload list is intentionally explicit: a build that asks to preload a
missing module fails at startup instead of silently running without it. Python
exceptions are printed to stderr; a failed numeric call returns NaN.
