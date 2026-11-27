"""CPython & NX Native ABI v2 bridge compiled into Nexir executables.

Supports:
1. Legacy @py_call / @py_fn zero-copy Span[f64]/MutSpan[f64] calls.
2. Direct C-extension & Python module invocation WITHOUT intermediate .py files
   via NX Native ABI v2 (arbitrary args, kwargs, return values, method calls,
   attributes, multi-dtype zero-copy ctypes/memoryview spans, and sandbox check).
"""

PYTHON_BRIDGE_CPP = r'''#include <Python.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct nx_span_f64 { const double* ptr; std::size_t len; };
struct nx_mutspan_f64 { double* ptr; std::size_t len; };

enum {
    NX_ABI_NULL   = 0,
    NX_ABI_BOOL   = 1,
    NX_ABI_I64    = 2,
    NX_ABI_U64    = 3,
    NX_ABI_I128   = 4,
    NX_ABI_U128   = 5,
    NX_ABI_F64    = 6,
    NX_ABI_STR    = 7,
    NX_ABI_BYTES  = 8,
    NX_ABI_SPAN   = 9,
    NX_ABI_LIST   = 10,
    NX_ABI_DICT   = 11,
    NX_ABI_OBJECT = 12,
    NX_ABI_PY_OBJ = 13,
    NX_ABI_ERROR  = 14
};

struct NxAbiWireVal {
    std::uint32_t tag;
    std::uint32_t flags;
    union {
        std::int64_t i64;
        std::uint64_t u64;
        double f64;
        void *ptr;
        struct { std::uint64_t lo; std::int64_t hi; } i128_v;
    } payload;
};

struct NxAbiWireList {
    std::size_t count;
    NxAbiWireVal *items;
};

struct NxAbiWireDict {
    std::size_t count;
    const char **keys;
    NxAbiWireVal *vals;
};

struct NxAbiWireSpan {
    void *ptr;
    std::size_t len;
    std::uint32_t dtype;
    std::uint32_t writable;
};

namespace {
std::once_flag init_once;
std::mutex bridge_mu;
std::unordered_map<std::string, PyObject*> modules;
std::unordered_map<std::string, PyObject*> functions;
int (*g_sandbox_module_check)(const char *) = nullptr;

void initialize_once() {
    Py_Initialize();
    PyEval_SaveThread();
}

bool ensure_python() {
    std::call_once(init_once, initialize_once);
    return Py_IsInitialized() != 0;
}

void report_python_error(const char* context) {
    std::fprintf(stderr, "NEXIR Python bridge: %s\n", context);
    if (PyErr_Occurred()) PyErr_Print();
}

char *dup_cstr(const char *s, std::size_t len) {
    char *out = static_cast<char *>(std::malloc(len + 1));
    if (!out) return nullptr;
    if (len) std::memcpy(out, s, len);
    out[len] = '\0';
    return out;
}

PyObject* load_module(const char* name) {
    if (g_sandbox_module_check && !g_sandbox_module_check(name)) {
        PyErr_Format(PyExc_PermissionError, "sandbox denied Python module: %s", name);
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(bridge_mu);
        auto found = modules.find(name);
        if (found != modules.end()) return found->second;
    }
    PyObject* module = PyImport_ImportModule(name);
    if (!module) return nullptr;
    std::lock_guard<std::mutex> lock(bridge_mu);
    auto [it, inserted] = modules.emplace(name, module);
    if (!inserted) {
        Py_DECREF(module);
        return it->second;
    }
    return module;
}

PyObject* load_function(const char* module_name, const char* function_name) {
    const std::string key = std::string(module_name) + "\n" + function_name;
    {
        std::lock_guard<std::mutex> lock(bridge_mu);
        auto found = functions.find(key);
        if (found != functions.end()) return found->second;
    }
    PyObject* module = load_module(module_name);
    if (!module) return nullptr;
    PyObject* callable = PyObject_GetAttrString(module, function_name);
    if (!callable) return nullptr;
    if (!PyCallable_Check(callable)) {
        Py_DECREF(callable);
        PyErr_Format(PyExc_TypeError, "%s.%s is not callable", module_name, function_name);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(bridge_mu);
    auto [it, inserted] = functions.emplace(key, callable);
    if (!inserted) {
        Py_DECREF(callable);
        return it->second;
    }
    return callable;
}

const char *ctypes_name_for_dtype(std::uint32_t dtype) {
    switch (dtype) {
        case 1: return "c_int8";
        case 2: return "c_int16";
        case 3: return "c_int32";
        case 4: return "c_int64";
        case 6: return "c_uint8";
        case 7: return "c_uint16";
        case 8: return "c_uint32";
        case 9: return "c_uint64";
        case 11: return "c_float";
        case 12: return "c_double";
        case 13: return "c_bool";
        default: return "c_double";
    }
}

PyObject* ctypes_view_typed(const void* data, std::size_t length, std::uint32_t dtype) {
    PyObject* ctypes = PyImport_ImportModule("ctypes");
    if (!ctypes) return nullptr;
    PyObject* scalar_type = PyObject_GetAttrString(ctypes, ctypes_name_for_dtype(dtype));
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
        view = PyObject_CallObject(array_type, nullptr);
    } else if (!data) {
        PyErr_SetString(PyExc_ValueError, "non-empty Nexir span has a null data pointer");
    } else {
        PyObject* from_address = PyObject_GetAttrString(array_type, "from_address");
        PyObject* address = PyLong_FromVoidPtr(const_cast<void*>(data));
        if (from_address && address) view = PyObject_CallFunctionObjArgs(from_address, address, nullptr);
        Py_XDECREF(from_address);
        Py_XDECREF(address);
    }
    Py_DECREF(array_type);
    return view;
}

PyObject* ctypes_view(const double* data, std::size_t length) {
    return ctypes_view_typed(data, length, 12);
}

PyObject *abi_to_py(const NxAbiWireVal &v) {
    switch (v.tag) {
        case NX_ABI_NULL: Py_RETURN_NONE;
        case NX_ABI_BOOL: return PyBool_FromLong(v.payload.i64 ? 1 : 0);
        case NX_ABI_I64: return PyLong_FromLongLong(v.payload.i64);
        case NX_ABI_U64: return PyLong_FromUnsignedLongLong(v.payload.u64);
        case NX_ABI_I128: {
            PyObject *hi_obj = PyLong_FromLongLong(v.payload.i128_v.hi);
            PyObject *shift = PyLong_FromLong(64);
            PyObject *shifted = (hi_obj && shift) ? PyNumber_Lshift(hi_obj, shift) : nullptr;
            PyObject *lo_obj = PyLong_FromUnsignedLongLong(v.payload.i128_v.lo);
            PyObject *res = (shifted && lo_obj) ? PyNumber_Or(shifted, lo_obj) : nullptr;
            Py_XDECREF(hi_obj); Py_XDECREF(shift); Py_XDECREF(shifted); Py_XDECREF(lo_obj);
            return res;
        }
        case NX_ABI_U128: {
            std::uint64_t hi_u = static_cast<std::uint64_t>(v.payload.i128_v.hi);
            PyObject *hi_obj = PyLong_FromUnsignedLongLong(hi_u);
            PyObject *shift = PyLong_FromLong(64);
            PyObject *shifted = (hi_obj && shift) ? PyNumber_Lshift(hi_obj, shift) : nullptr;
            PyObject *lo_obj = PyLong_FromUnsignedLongLong(v.payload.i128_v.lo);
            PyObject *res = (shifted && lo_obj) ? PyNumber_Or(shifted, lo_obj) : nullptr;
            Py_XDECREF(hi_obj); Py_XDECREF(shift); Py_XDECREF(shifted); Py_XDECREF(lo_obj);
            return res;
        }
        case NX_ABI_F64: return PyFloat_FromDouble(v.payload.f64);
        case NX_ABI_STR: {
            const char *s = static_cast<const char *>(v.payload.ptr);
            return PyUnicode_FromString(s ? s : "");
        }
        case NX_ABI_SPAN: {
            const auto *sp = static_cast<const NxAbiWireSpan *>(v.payload.ptr);
            if (!sp) Py_RETURN_NONE;
            return ctypes_view_typed(sp->ptr, sp->len, sp->dtype);
        }
        case NX_ABI_LIST: {
            const auto *lst = static_cast<const NxAbiWireList *>(v.payload.ptr);
            if (!lst) return PyList_New(0);
            PyObject *out = PyList_New(static_cast<Py_ssize_t>(lst->count));
            if (!out) return nullptr;
            for (std::size_t i = 0; i < lst->count; ++i) {
                PyObject *elem = abi_to_py(lst->items[i]);
                if (!elem) { Py_DECREF(out); return nullptr; }
                PyList_SET_ITEM(out, static_cast<Py_ssize_t>(i), elem);
            }
            return out;
        }
        case NX_ABI_DICT: {
            const auto *d = static_cast<const NxAbiWireDict *>(v.payload.ptr);
            if (!d) return PyDict_New();
            PyObject *out = PyDict_New();
            if (!out) return nullptr;
            for (std::size_t i = 0; i < d->count; ++i) {
                PyObject *val = abi_to_py(d->vals[i]);
                if (!val) { Py_DECREF(out); return nullptr; }
                PyDict_SetItemString(out, d->keys[i], val);
                Py_DECREF(val);
            }
            return out;
        }
        case NX_ABI_PY_OBJ: {
            PyObject *obj = static_cast<PyObject *>(v.payload.ptr);
            if (!obj) Py_RETURN_NONE;
            Py_INCREF(obj);
            return obj;
        }
        default:
            Py_RETURN_NONE;
    }
}

NxAbiWireVal py_to_abi(PyObject *obj) {
    NxAbiWireVal out{};
    if (!obj || obj == Py_None) {
        out.tag = NX_ABI_NULL;
        return out;
    }
    if (PyBool_Check(obj)) {
        out.tag = NX_ABI_BOOL;
        out.payload.i64 = (obj == Py_True) ? 1 : 0;
        return out;
    }
    if (PyLong_Check(obj)) {
        int overflow = 0;
        long long iv = PyLong_AsLongLongAndOverflow(obj, &overflow);
        if (!overflow && !PyErr_Occurred()) {
            out.tag = NX_ABI_I64;
            out.payload.i64 = static_cast<std::int64_t>(iv);
            return out;
        }
        PyErr_Clear();
        PyObject *shift = PyLong_FromLong(64);
        PyObject *mask = PyLong_FromUnsignedLongLong(0xFFFFFFFFFFFFFFFFULL);
        PyObject *hi_obj = shift ? PyNumber_Rshift(obj, shift) : nullptr;
        PyObject *lo_obj = mask ? PyNumber_And(obj, mask) : nullptr;
        long long hi_v = hi_obj ? PyLong_AsLongLong(hi_obj) : 0;
        unsigned long long lo_v = lo_obj ? PyLong_AsUnsignedLongLong(lo_obj) : 0;
        Py_XDECREF(shift); Py_XDECREF(mask); Py_XDECREF(hi_obj); Py_XDECREF(lo_obj);
        if (!PyErr_Occurred()) {
            out.tag = NX_ABI_I128;
            out.payload.i128_v.lo = static_cast<std::uint64_t>(lo_v);
            out.payload.i128_v.hi = static_cast<std::int64_t>(hi_v);
            return out;
        }
        PyErr_Clear();
        double dv = PyLong_AsDouble(obj);
        if (!PyErr_Occurred()) {
            out.tag = NX_ABI_F64;
            out.payload.f64 = dv;
            return out;
        }
        PyErr_Clear();
    }
    if (PyFloat_Check(obj)) {
        out.tag = NX_ABI_F64;
        out.payload.f64 = PyFloat_AsDouble(obj);
        return out;
    }
    if (PyUnicode_Check(obj)) {
        Py_ssize_t sz = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(obj, &sz);
        if (utf8 && sz >= 0) {
            out.tag = NX_ABI_STR;
            out.flags = 1;
            out.payload.ptr = dup_cstr(utf8, static_cast<std::size_t>(sz));
            return out;
        }
    }
    if (PyList_Check(obj) || PyTuple_Check(obj)) {
        Py_ssize_t n = PySequence_Size(obj);
        if (n >= 0) {
            auto *lst = static_cast<NxAbiWireList *>(std::malloc(sizeof(NxAbiWireList)));
            lst->count = static_cast<std::size_t>(n);
            lst->items = n ? static_cast<NxAbiWireVal *>(std::calloc(static_cast<std::size_t>(n), sizeof(NxAbiWireVal))) : nullptr;
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject *item = PySequence_GetItem(obj, i);
                lst->items[i] = py_to_abi(item);
                Py_XDECREF(item);
            }
            out.tag = NX_ABI_LIST;
            out.flags = 1;
            out.payload.ptr = lst;
            return out;
        }
    }
    if (PyDict_Check(obj)) {
        Py_ssize_t n = PyDict_Size(obj);
        auto *d = static_cast<NxAbiWireDict *>(std::malloc(sizeof(NxAbiWireDict)));
        d->count = static_cast<std::size_t>(n);
        d->keys = n ? static_cast<const char **>(std::calloc(static_cast<std::size_t>(n), sizeof(char *))) : nullptr;
        d->vals = n ? static_cast<NxAbiWireVal *>(std::calloc(static_cast<std::size_t>(n), sizeof(NxAbiWireVal))) : nullptr;
        PyObject *k = nullptr, *v = nullptr;
        Py_ssize_t pos = 0, idx = 0;
        while (PyDict_Next(obj, &pos, &k, &v) && idx < n) {
            PyObject *ks = PyObject_Str(k);
            Py_ssize_t ksz = 0;
            const char *kutf8 = ks ? PyUnicode_AsUTF8AndSize(ks, &ksz) : "";
            d->keys[idx] = dup_cstr(kutf8 ? kutf8 : "", kutf8 ? static_cast<std::size_t>(ksz) : 0);
            Py_XDECREF(ks);
            d->vals[idx] = py_to_abi(v);
            ++idx;
        }
        d->count = static_cast<std::size_t>(idx);
        out.tag = NX_ABI_DICT;
        out.flags = 1;
        out.payload.ptr = d;
        return out;
    }
    // Check numpy/generic float scalar or keep opaque PyObject* handle
    if (PyObject_HasAttrString(obj, "__float__") && !PySequence_Check(obj) && !PyMapping_Check(obj)) {
        double fv = PyFloat_AsDouble(obj);
        if (!PyErr_Occurred()) {
            out.tag = NX_ABI_F64;
            out.payload.f64 = fv;
            return out;
        }
        PyErr_Clear();
    }
    Py_INCREF(obj);
    out.tag = NX_ABI_PY_OBJ;
    out.flags = 1;
    out.payload.ptr = obj;
    return out;
}

NxAbiWireVal invoke_callable(PyObject *callable, std::size_t argc, const NxAbiWireVal *argv,
                             const char **kw_names, std::size_t kw_count) {
    NxAbiWireVal err{};
    err.tag = NX_ABI_ERROR;
    err.flags = 1;
    if (!callable) {
        err.payload.ptr = dup_cstr("callable is null", 16);
        return err;
    }
    std::size_t pos_count = argc >= kw_count ? (argc - kw_count) : 0;
    PyObject *args_tuple = PyTuple_New(static_cast<Py_ssize_t>(pos_count));
    if (!args_tuple) {
        err.payload.ptr = dup_cstr("failed to allocate tuple", 24);
        return err;
    }
    for (std::size_t i = 0; i < pos_count; ++i) {
        PyObject *item = abi_to_py(argv[i]);
        if (!item) { Py_DECREF(args_tuple); report_python_error("argument conversion failed"); err.payload.ptr = dup_cstr("arg error", 9); return err; }
        PyTuple_SET_ITEM(args_tuple, static_cast<Py_ssize_t>(i), item);
    }
    PyObject *kwargs_dict = nullptr;
    if (kw_count > 0 && kw_names) {
        kwargs_dict = PyDict_New();
        for (std::size_t i = 0; i < kw_count; ++i) {
            PyObject *val = abi_to_py(argv[pos_count + i]);
            if (val) {
                PyDict_SetItemString(kwargs_dict, kw_names[i], val);
                Py_DECREF(val);
            }
        }
    }
    PyObject *res = PyObject_Call(callable, args_tuple, kwargs_dict);
    Py_DECREF(args_tuple);
    Py_XDECREF(kwargs_dict);
    if (!res) {
        PyObject *ptype = nullptr, *pval = nullptr, *ptb = nullptr;
        PyErr_Fetch(&ptype, &pval, &ptb);
        PyObject *msg_obj = pval ? PyObject_Str(pval) : nullptr;
        const char *msg = msg_obj ? PyUnicode_AsUTF8(msg_obj) : "Python call failed";
        err.payload.ptr = dup_cstr(msg ? msg : "Python error", msg ? std::strlen(msg) : 12);
        Py_XDECREF(msg_obj);
        Py_XDECREF(ptype);
        Py_XDECREF(pval);
        Py_XDECREF(ptb);
        return err;
    }
    NxAbiWireVal out = py_to_abi(res);
    Py_DECREF(res);
    return out;
}

double call_f64(const char* module_name, const char* function_name,
                const double* data, std::size_t length) {
    if (!module_name || !function_name) {
        std::fprintf(stderr, "NEXIR Python bridge: module and function names are required\n");
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
        value = PyFloat_AsDouble(result);
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

char* call_str(const char* module_name, const char* function_name,
               const double* data, std::size_t length) {
    if (!ensure_python()) return nullptr;
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* callable = load_function(module_name, function_name);
    PyObject* view = callable ? ctypes_view(data, length) : nullptr;
    PyObject* result = (callable && view) ? PyObject_CallFunctionObjArgs(callable, view, nullptr) : nullptr;
    Py_XDECREF(view);
    char* output = nullptr;
    if (result && PyUnicode_Check(result)) {
        Py_ssize_t size = 0;
        const char* utf8 = PyUnicode_AsUTF8AndSize(result, &size);
        if (utf8 && size >= 0 && static_cast<std::size_t>(size) < std::numeric_limits<std::size_t>::max()) {
            output = dup_cstr(utf8, static_cast<std::size_t>(size));
            if (!output) PyErr_NoMemory();
        }
    } else if (result) {
        PyErr_SetString(PyExc_TypeError, "@py_call function must return str");
    }
    if (!output) report_python_error("string call failed");
    Py_XDECREF(result);
    PyGILState_Release(gil);
    return output;
}

nx_mutspan_f64 call_buffer(const char* module_name, const char* function_name,
                           const double* data, std::size_t length) {
    nx_mutspan_f64 output{nullptr, std::numeric_limits<std::size_t>::max()};
    if (!ensure_python()) return output;
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* callable = load_function(module_name, function_name);
    PyObject* view = callable ? ctypes_view(data, length) : nullptr;
    PyObject* result = (callable && view) ? PyObject_CallFunctionObjArgs(callable, view, nullptr) : nullptr;
    Py_XDECREF(view);
    PyObject* sequence = result ? PySequence_Fast(result, "@py_call buffer result must be a numeric sequence") : nullptr;
    if (sequence) {
        const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
        if (count >= 0 && static_cast<std::size_t>(count) <= std::numeric_limits<std::size_t>::max() / sizeof(double)) {
            double* values = count ? static_cast<double*>(std::malloc(static_cast<std::size_t>(count) * sizeof(double))) : nullptr;
            if (count == 0 || values) {
                bool valid = true;
                for (Py_ssize_t i = 0; i < count; ++i) {
                    values[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(sequence, i));
                    if (PyErr_Occurred()) { valid = false; break; }
                }
                if (valid) output = {values, static_cast<std::size_t>(count)};
                else std::free(values);
            } else PyErr_NoMemory();
        } else PyErr_SetString(PyExc_OverflowError, "Python buffer result is too large");
    }
    if (output.len == std::numeric_limits<std::size_t>::max()) report_python_error("buffer call failed");
    Py_XDECREF(sequence);
    Py_XDECREF(result);
    PyGILState_Release(gil);
    return output;
}
} // namespace

extern "C" void nx_py_initialize(void) {
    if (ensure_python()) {
        const PyGILState_STATE gil = PyGILState_Ensure();
        PyGILState_Release(gil);
    }
}

extern "C" int nx_py_abi_sandbox_hook(int (*fn)(const char *)) {
    g_sandbox_module_check = fn;
    return 1;
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
extern "C" char* nx_py_call_str_span(const char* module, const char* function, nx_span_f64 data) {
    return call_str(module, function, data.ptr, data.len);
}
extern "C" char* nx_py_call_str_mutspan(const char* module, const char* function, nx_mutspan_f64 data) {
    return call_str(module, function, data.ptr, data.len);
}
extern "C" nx_mutspan_f64 nx_py_call_buffer_span(const char* module, const char* function, nx_span_f64 data) {
    return call_buffer(module, function, data.ptr, data.len);
}
extern "C" nx_mutspan_f64 nx_py_call_buffer_mutspan(const char* module, const char* function, nx_mutspan_f64 data) {
    return call_buffer(module, function, data.ptr, data.len);
}

extern "C" NxAbiWireVal nx_py_abi_call(const char *module_name, const char *function_name,
                                       std::size_t argc, const NxAbiWireVal *argv,
                                       const char **kw_names, std::size_t kw_count) {
    NxAbiWireVal err{};
    err.tag = NX_ABI_ERROR;
    err.flags = 1;
    if (!module_name || !function_name || !ensure_python()) {
        err.payload.ptr = dup_cstr("Python runtime unavailable", 26);
        return err;
    }
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject *callable = load_function(module_name, function_name);
    NxAbiWireVal out = callable ? invoke_callable(callable, argc, argv, kw_names, kw_count) : [&]() {
        PyObject *pt = nullptr, *pv = nullptr, *pb = nullptr;
        PyErr_Fetch(&pt, &pv, &pb);
        PyObject *ms = pv ? PyObject_Str(pv) : nullptr;
        const char *m = ms ? PyUnicode_AsUTF8(ms) : "failed to load Python function";
        NxAbiWireVal e{}; e.tag = NX_ABI_ERROR; e.flags = 1;
        e.payload.ptr = dup_cstr(m ? m : "load error", m ? std::strlen(m) : 10);
        Py_XDECREF(ms); Py_XDECREF(pt); Py_XDECREF(pv); Py_XDECREF(pb);
        return e;
    }();
    PyGILState_Release(gil);
    return out;
}

extern "C" NxAbiWireVal nx_py_abi_method_call(void *py_obj, const char *method_name,
                                              std::size_t argc, const NxAbiWireVal *argv,
                                              const char **kw_names, std::size_t kw_count) {
    NxAbiWireVal err{};
    err.tag = NX_ABI_ERROR;
    err.flags = 1;
    if (!py_obj || !method_name || !ensure_python()) {
        err.payload.ptr = dup_cstr("null py_obj or method", 21);
        return err;
    }
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject *callable = PyObject_GetAttrString(static_cast<PyObject *>(py_obj), method_name);
    NxAbiWireVal out = callable ? invoke_callable(callable, argc, argv, kw_names, kw_count) : [&]() {
        PyErr_Clear();
        NxAbiWireVal e{}; e.tag = NX_ABI_ERROR; e.flags = 1;
        e.payload.ptr = dup_cstr("attribute/method not found", 26);
        return e;
    }();
    Py_XDECREF(callable);
    PyGILState_Release(gil);
    return out;
}

extern "C" NxAbiWireVal nx_py_abi_import_getattr(const char *module_name, const char *attr_name) {
    NxAbiWireVal err{};
    err.tag = NX_ABI_ERROR;
    err.flags = 1;
    if (!module_name || !attr_name || !ensure_python()) {
        err.payload.ptr = dup_cstr("Python runtime unavailable", 26);
        return err;
    }
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject *mod = load_module(module_name);
    if (!mod) {
        PyErr_Clear();
        PyGILState_Release(gil);
        err.payload.ptr = dup_cstr("module import failed", 20);
        return err;
    }
    PyObject *attr = PyObject_GetAttrString(mod, attr_name);
    if (!attr) {
        PyErr_Clear();
        PyGILState_Release(gil);
        err.payload.ptr = dup_cstr("module attribute not found", 26);
        return err;
    }
    NxAbiWireVal out = py_to_abi(attr);
    Py_DECREF(attr);
    PyGILState_Release(gil);
    return out;
}

extern "C" NxAbiWireVal nx_py_abi_getattr(void *py_obj, const char *attr_name) {
    NxAbiWireVal err{};
    err.tag = NX_ABI_ERROR;
    err.flags = 1;
    if (!py_obj || !attr_name || !ensure_python()) {
        err.payload.ptr = dup_cstr("null py_obj or attr", 19);
        return err;
    }
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject *attr = PyObject_GetAttrString(static_cast<PyObject *>(py_obj), attr_name);
    if (!attr) {
        PyErr_Clear();
        PyGILState_Release(gil);
        err.payload.ptr = dup_cstr("attribute not found", 19);
        return err;
    }
    NxAbiWireVal out = py_to_abi(attr);
    Py_DECREF(attr);
    PyGILState_Release(gil);
    return out;
}

extern "C" void nx_py_abi_release_val(NxAbiWireVal *val) {
    if (!val || !val->flags) return;
    if ((val->tag == NX_ABI_STR || val->tag == NX_ABI_ERROR) && val->payload.ptr) {
        std::free(val->payload.ptr);
    } else if (val->tag == NX_ABI_LIST && val->payload.ptr) {
        auto *lst = static_cast<NxAbiWireList *>(val->payload.ptr);
        for (std::size_t i = 0; i < lst->count; ++i) nx_py_abi_release_val(&lst->items[i]);
        std::free(lst->items);
        std::free(lst);
    } else if (val->tag == NX_ABI_DICT && val->payload.ptr) {
        auto *d = static_cast<NxAbiWireDict *>(val->payload.ptr);
        for (std::size_t i = 0; i < d->count; ++i) {
            std::free(const_cast<char *>(d->keys[i]));
            nx_py_abi_release_val(&d->vals[i]);
        }
        std::free(d->keys);
        std::free(d->vals);
        std::free(d);
    } else if (val->tag == NX_ABI_PY_OBJ && val->payload.ptr) {
        if (Py_IsInitialized()) {
            const PyGILState_STATE gil = PyGILState_Ensure();
            Py_DECREF(static_cast<PyObject *>(val->payload.ptr));
            PyGILState_Release(gil);
        }
    }
    val->tag = NX_ABI_NULL;
    val->flags = 0;
    val->payload.ptr = nullptr;
}
'''
