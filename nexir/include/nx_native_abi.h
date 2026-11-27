#ifndef NEXIR_NX_NATIVE_ABI_H
#define NEXIR_NX_NATIVE_ABI_H

/*
 * NX Native ABI v2 (C99/C++17 compatible stable binary interface)
 *
 * Provides:
 *   1. Portable tagged values, owned strings, spans, buffers, and error codes
 *      shared across NX, C, C++, Python C-API extensions, Rust, and Go.
 *   2. Direct runtime bridge for Python modules implemented in C/C++
 *      (CPython C-extensions via vectorcall/PyObject/Py_buffer) without needing
 *      wrapper .py files.
 *   3. Sandbox capability checks, memory accounting hooks, and module audit logs.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NX_ABI_VERSION 2u

#ifdef _WIN32
#  ifdef NEXIR_ABI_EXPORTS
#    define NX_ABI_EXPORT __declspec(dllexport)
#  else
#    define NX_ABI_EXPORT __declspec(dllexport)
#  endif
#else
#  define NX_ABI_EXPORT __attribute__((visibility("default")))
#endif

typedef enum NxAbiTypeTag {
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
} NxAbiTypeTag;

typedef enum NxAbiDType {
    NX_DTYPE_I8   = 1,
    NX_DTYPE_I16  = 2,
    NX_DTYPE_I32  = 3,
    NX_DTYPE_I64  = 4,
    NX_DTYPE_I128 = 5,
    NX_DTYPE_U8   = 6,
    NX_DTYPE_U16  = 7,
    NX_DTYPE_U32  = 8,
    NX_DTYPE_U64  = 9,
    NX_DTYPE_U128 = 10,
    NX_DTYPE_F32  = 11,
    NX_DTYPE_F64  = 12,
    NX_DTYPE_BOOL = 13
} NxAbiDType;

typedef struct NxAbiI128 {
    uint64_t lo;
    int64_t  hi;
} NxAbiI128;

typedef struct NxAbiU128 {
    uint64_t lo;
    uint64_t hi;
} NxAbiU128;

typedef struct NxAbiString {
    const char *data;
    size_t      len;
    uint32_t    flags; /* bit 0: owned heap, bit 1: static literal */
    uint32_t    reserved;
} NxAbiString;

typedef struct NxAbiSpan {
    void    *ptr;
    size_t   len;
    uint32_t dtype;    /* NxAbiDType */
    uint32_t writable; /* 0 = readonly Span, 1 = MutSpan */
} NxAbiSpan;

typedef struct NxAbiValue {
    uint32_t tag;      /* NxAbiTypeTag */
    uint32_t flags;
    union {
        int64_t   i64;
        uint64_t  u64;
        double    f64;
        void     *ptr;
        NxAbiI128 i128_inline; /* Note: keeps struct 24 bytes if embedded */
    } payload;
} NxAbiValue;

typedef struct NxAbiExportEntry {
    const char *name;
    const char *signature;
    void       *fn_ptr;
} NxAbiExportEntry;

typedef struct NxAbiModule {
    uint32_t                abi_version;
    const char             *name;
    const char             *version;
    size_t                  export_count;
    const NxAbiExportEntry *exports;
} NxAbiModule;

/* Host Runtime VTable exposed to plugins, native extensions, and CPython bridge */
typedef struct NxAbiHostApi {
    uint32_t abi_version;
    uint32_t struct_size;

    /* Cache-backed memory allocation (subject to sandbox limits) */
    void *(*alloc)(size_t size);
    void  (*dealloc)(void *ptr, size_t size);

    /* String lifecycle */
    NxAbiString (*str_new)(const char *utf8, size_t len);
    void        (*str_release)(NxAbiString s);

    /* Sandbox & Audit */
    int  (*sandbox_allow)(const char *capability, const char *target);
    void (*audit_record)(const char *event, const char *kind, const char *name,
                         const char *detail, int allowed);

    /* Direct CPython C-extension & module runtime calls */
    int         (*py_ensure_init)(void);
    void       *(*py_import_module)(const char *module_name);
    NxAbiValue  (*py_call_fast)(const char *module_name, const char *callable_name,
                                size_t argc, const NxAbiValue *argv,
                                const char **kw_names, size_t kw_count);
    NxAbiValue  (*py_getattr)(void *py_obj, const char *attr_name);
    NxAbiValue  (*py_method_call)(void *py_obj, const char *method_name,
                                  size_t argc, const NxAbiValue *argv,
                                  const char **kw_names, size_t kw_count);
    void        (*py_retain)(void *py_obj);
    void        (*py_release)(void *py_obj);
    void        (*value_release)(NxAbiValue *val);
} NxAbiHostApi;

/* Runtime entrypoints implemented by the NX runtime / CPython bridge */
NX_ABI_EXPORT const NxAbiHostApi *nx_abi_host_api(void);
NX_ABI_EXPORT uint32_t nx_abi_version(void);
NX_ABI_EXPORT void nx_abi_value_release(NxAbiValue *val);

#ifdef __cplusplus
}
#endif

#endif /* NEXIR_NX_NATIVE_ABI_H */
