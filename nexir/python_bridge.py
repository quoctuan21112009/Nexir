"""CPython bridge source compiled into Nexir executables with ``--python``.

The ABI deliberately passes a Nexir f64 span by value. ctypes creates a view of
that exact buffer; Python workers can use numpy.ctypeslib.as_array(view) without
copying the input. The caller must keep the span alive for the duration of the
synchronous call and must not concurrently mutate it.
"""

PYTHON_BRIDGE_CPP = r'''#include <Python.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

struct nx_span_f64 { const double* ptr; std::size_t len; };
struct nx_mutspan_f64 { double* ptr; std::size_t len; };

namespace {
std::once_flag init_once;
std::unordered_map<std::string, PyObject*> modules;
std::unordered_map<std::string, PyObject*> functions;

void initialize_once() {
    Py_Initialize();
    // Let PyGILState_Ensure manage the GIL for every bridge entry, including
    // calls made later from a different native thread.
    PyEval_SaveThread();
}

bool ensure_python() {
    std::call_once(init_once, initialize_once);
    return Py_IsInitialized() != 0;
}

void report_python_error(const char* context) {
    std::fprintf(stderr, "NEXIR Python bridge: %s\\n", context);
    if (PyErr_Occurred()) PyErr_Print();
}

PyObject* load_module(const char* name) {
    auto found = modules.find(name);
    if (found != modules.end()) return found->second;
    PyObject* module = PyImport_ImportModule(name);
    if (!module) return nullptr;
    modules.emplace(name, module); // retain one strong reference for process lifetime
    return module;
}

PyObject* load_function(const char* module_name, const char* function_name) {
    const std::string key = std::string(module_name) + "\\n" + function_name;
    auto found = functions.find(key);
    if (found != functions.end()) return found->second;
    PyObject* module = load_module(module_name);
    if (!module) return nullptr;
    PyObject* callable = PyObject_GetAttrString(module, function_name);
    if (!callable) return nullptr;
    if (!PyCallable_Check(callable)) {
        Py_DECREF(callable);
        PyErr_Format(PyExc_TypeError, "%s.%s is not callable", module_name, function_name);
        return nullptr;
    }
    functions.emplace(key, callable); // cache the resolved callable too
    return callable;
}

PyObject* ctypes_view(const double* data, std::size_t length) {
    PyObject* ctypes = PyImport_ImportModule("ctypes");
    if (!ctypes) return nullptr;
    PyObject* scalar_type = PyObject_GetAttrString(ctypes, "c_double");
    Py_DECREF(ctypes);
    if (!scalar_type) return nullptr;
    PyObject* count = PyLong_FromSize_t(length);
    if (!count) { Py_DECREF(scalar_type); return nullptr; }
    PyObject* array_type = PyNumber_Multiply(scalar_type, count);
    Py_DECREF(scalar_type);
    Py_DECREF(count);
    if (!array_type) return nullptr;

    PyObject* view = nullptr;
    if (length == 0) {
        // from_address(0) is invalid even for a zero-length array.
        view = PyObject_CallObject(array_type, nullptr);
    } else if (!data) {
        PyErr_SetString(PyExc_ValueError, "non-empty Nexir span has a null data pointer");
    } else {
        PyObject* from_address = PyObject_GetAttrString(array_type, "from_address");
        PyObject* address = PyLong_FromVoidPtr(const_cast<double*>(data));
        if (from_address && address) view = PyObject_CallFunctionObjArgs(from_address, address, nullptr);
        Py_XDECREF(from_address);
        Py_XDECREF(address);
    }
    Py_DECREF(array_type);
    return view;
}

double call_f64(const char* module_name, const char* function_name,
                const double* data, std::size_t length) {
    if (!module_name || !function_name) {
        std::fprintf(stderr, "NEXIR Python bridge: module and function names are required\\n");
        return std::nan("");
    }
    if (!ensure_python()) return std::nan("");
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* callable = load_function(module_name, function_name);
    PyObject* view = callable ? ctypes_view(data, length) : nullptr;
    PyObject* result = (callable && view) ? PyObject_CallFunctionObjArgs(callable, view, nullptr) : nullptr;
    Py_XDECREF(view);
    double value = std::nan("");
    if (result) {
        value = PyFloat_AsDouble(result); // accepts Python and NumPy scalar numbers
        Py_DECREF(result);
        if (PyErr_Occurred()) {
            report_python_error("function must return a numeric scalar");
            value = std::nan("");
        }
    } else {
        report_python_error("call failed");
    }
    PyGILState_Release(gil);
    return value;
}
} // namespace

extern "C" void nx_py_initialize(void) {
    if (ensure_python()) {
        const PyGILState_STATE gil = PyGILState_Ensure();
        PyGILState_Release(gil);
    }
}

extern "C" int nx_py_preload(const char* module_name) {
    if (!module_name || !ensure_python()) return 0;
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* module = load_module(module_name);
    if (!module) report_python_error("module preload failed");
    PyGILState_Release(gil);
    return module ? 1 : 0;
}

extern "C" double nx_py_call_f64_span(const char* module_name,
                                       const char* function_name,
                                       nx_span_f64 data) {
    return call_f64(module_name, function_name, data.ptr, data.len);
}

extern "C" double nx_py_call_f64_mutspan(const char* module_name,
                                          const char* function_name,
                                          nx_mutspan_f64 data) {
    return call_f64(module_name, function_name, data.ptr, data.len);
}
'''
