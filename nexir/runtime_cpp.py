"""C++17 desktop runtime emitted into NEXIR 2.0 programs.

Composes:
- Cache-based zero-leak memory allocator & tracker (replaces global nx_owners)
- Owned SSO/cached string type (NxStr), NxStrBuilder, format specifiers
- Full 8..128-bit numeric conversions (i8..i128, u8..u128, f32, f64)
- Hybrid NxValue v2 (16-byte tagged value + list/dict/i128/pyobj)
- Sandbox, resource limits, OS isolation & structured module audit logger
- Deep cross-platform FS I/O & Networking (WinSock2 / POSIX / WSAPoll / kqueue / epoll)
- Hybrid Multi-Core Async Executor (nx.asyncio), ThreadPool/Channels (nx.thread), GPU/SIMD (nx.gpu)
- x86 AVX2/SSE2 SIMD lanes & Ring buffers
- NX Native ABI v2 runtime host API
"""

from .runtime_memory import RUNTIME_MEMORY_CPP
from .runtime_sandbox_io_net import RUNTIME_SANDBOX_IO_NET_CPP
from .runtime_hybrid import RUNTIME_HYBRID_CPP

_RUNTIME_HEADERS = r'''
#define _POSIX_C_SOURCE 200112L
#ifdef _WIN32
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0601
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#  define NX_EXPORT __declspec(dllexport)
#else
#  include <sys/types.h>
#  include <sys/socket.h>
#  include <sys/resource.h>
#  include <netdb.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#  include <dlfcn.h>
#  ifdef __APPLE__
#    include <sys/event.h>
#  else
#    include <sys/epoll.h>
#  endif
#  define NX_EXPORT __attribute__((visibility("default")))
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#  include <immintrin.h>
#  define NX_X86_SIMD 1
#  define NX_TARGET_AVX2 __attribute__((target("avx2")))
#  define NX_TARGET_SSE2 __attribute__((target("sse2")))
#endif
'''

_RUNTIME_SIMD_RING_ABI = r'''
// ============================================================================
// 14. SIMD Lanes (AVX2/SSE2 + Scalar Fallback) & Generation-Checked Ring
// ============================================================================
template <typename T, std::size_t N> struct nx_simd {
    alignas(32) std::array<T, N> lane{};
    T &at(std::int64_t i) { return lane[nx_bounds(i, N, false)]; }
    const T &at(std::int64_t i) const { return lane[nx_bounds(i, N, false)]; }
};

template <typename T, std::size_t N> static nx_simd<T, N> nx_simd_scalar(const nx_simd<T, N> &a, const nx_simd<T, N> &b, char op) {
    nx_simd<T, N> out;
#pragma omp simd
    for (std::size_t i = 0; i < N; ++i)
        out.lane[i] = op == '+' ? a.lane[i] + b.lane[i] : op == '-' ? a.lane[i] - b.lane[i] : a.lane[i] * b.lane[i];
    return out;
}

#ifdef NX_X86_SIMD
template <typename T, std::size_t N> NX_TARGET_AVX2 static nx_simd<T, N> nx_simd_avx2(const nx_simd<T, N> &a, const nx_simd<T, N> &b, char op) {
    nx_simd<T, N> out;
    std::size_t i = 0;
    if constexpr (std::is_same_v<T, float>) {
        for (; i + 8 <= N; i += 8) {
            auto x = _mm256_loadu_ps(a.lane.data() + i), y = _mm256_loadu_ps(b.lane.data() + i);
            auto z = op == '+' ? _mm256_add_ps(x, y) : op == '-' ? _mm256_sub_ps(x, y) : _mm256_mul_ps(x, y);
            _mm256_storeu_ps(out.lane.data() + i, z);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        for (; i + 4 <= N; i += 4) {
            auto x = _mm256_loadu_pd(a.lane.data() + i), y = _mm256_loadu_pd(b.lane.data() + i);
            auto z = op == '+' ? _mm256_add_pd(x, y) : op == '-' ? _mm256_sub_pd(x, y) : _mm256_mul_pd(x, y);
            _mm256_storeu_pd(out.lane.data() + i, z);
        }
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        for (; i + 8 <= N; i += 8) {
            auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.lane.data() + i));
            auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.lane.data() + i));
            auto z = op == '+' ? _mm256_add_epi32(x, y) : op == '-' ? _mm256_sub_epi32(x, y) : _mm256_mullo_epi32(x, y);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out.lane.data() + i), z);
        }
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        if (op != '*') for (; i + 4 <= N; i += 4) {
            auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.lane.data() + i));
            auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.lane.data() + i));
            auto z = op == '+' ? _mm256_add_epi64(x, y) : _mm256_sub_epi64(x, y);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out.lane.data() + i), z);
        }
    }
    for (; i < N; ++i)
        out.lane[i] = op == '+' ? a.lane[i] + b.lane[i] : op == '-' ? a.lane[i] - b.lane[i] : a.lane[i] * b.lane[i];
    return out;
}

template <typename T, std::size_t N> NX_TARGET_SSE2 static nx_simd<T, N> nx_simd_sse2(const nx_simd<T, N> &a, const nx_simd<T, N> &b, char op) {
    nx_simd<T, N> out;
    std::size_t i = 0;
    if constexpr (std::is_same_v<T, float>) {
        for (; i + 4 <= N; i += 4) {
            auto x = _mm_loadu_ps(a.lane.data() + i), y = _mm_loadu_ps(b.lane.data() + i);
            auto z = op == '+' ? _mm_add_ps(x, y) : op == '-' ? _mm_sub_ps(x, y) : _mm_mul_ps(x, y);
            _mm_storeu_ps(out.lane.data() + i, z);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        for (; i + 2 <= N; i += 2) {
            auto x = _mm_loadu_pd(a.lane.data() + i), y = _mm_loadu_pd(b.lane.data() + i);
            auto z = op == '+' ? _mm_add_pd(x, y) : op == '-' ? _mm_sub_pd(x, y) : _mm_mul_pd(x, y);
            _mm_storeu_pd(out.lane.data() + i, z);
        }
    }
    for (; i < N; ++i)
        out.lane[i] = op == '+' ? a.lane[i] + b.lane[i] : op == '-' ? a.lane[i] - b.lane[i] : a.lane[i] * b.lane[i];
    return out;
}
#endif

template <typename T, std::size_t N> static nx_simd<T, N> nx_simd_binary(const nx_simd<T, N> &a, const nx_simd<T, N> &b, char op) {
#ifdef NX_X86_SIMD
    if constexpr ((std::is_same_v<T, float> && N >= 8) || (std::is_same_v<T, double> && N >= 4) ||
                  (std::is_same_v<T, std::int32_t> && N >= 8) || (std::is_same_v<T, std::int64_t> && N >= 4)) {
        if ((op != '*' || !std::is_same_v<T, std::int64_t>) && __builtin_cpu_supports("avx2"))
            return nx_simd_avx2(a, b, op);
    }
    if constexpr ((std::is_same_v<T, float> && N >= 4) || (std::is_same_v<T, double> && N >= 2)) {
        if (__builtin_cpu_supports("sse2")) return nx_simd_sse2(a, b, op);
    }
#endif
    return nx_simd_scalar(a, b, op);
}
template <typename T, std::size_t N> static nx_simd<T, N> operator+(const nx_simd<T, N> &a, const nx_simd<T, N> &b) { return nx_simd_binary(a, b, '+'); }
template <typename T, std::size_t N> static nx_simd<T, N> operator-(const nx_simd<T, N> &a, const nx_simd<T, N> &b) { return nx_simd_binary(a, b, '-'); }
template <typename T, std::size_t N> static nx_simd<T, N> operator*(const nx_simd<T, N> &a, const nx_simd<T, N> &b) { return nx_simd_binary(a, b, '*'); }

struct nx_ring_slot_f64 { double value = 0.0; std::uint64_t generation = UINT64_MAX; };
class nx_ring_f64 {
public:
    explicit nx_ring_f64(std::size_t capacity) : slots_(capacity) {
        if (capacity == 0) nx_fail("ring capacity must be positive");
    }
    void write(std::int64_t absolute, double value) {
        if (absolute < 0) nx_fail("negative ring index");
        auto i = static_cast<std::uint64_t>(absolute);
        auto &slot = slots_[i % slots_.size()];
        slot.value = value;
        slot.generation = i / slots_.size();
    }
    bool valid(std::int64_t absolute) const {
        if (absolute < 0) return false;
        auto i = static_cast<std::uint64_t>(absolute);
        return slots_[i % slots_.size()].generation == i / slots_.size();
    }
    double read(std::int64_t absolute) const {
        if (!valid(absolute)) nx_fail("stale or unwritten ring frame");
        return slots_[static_cast<std::uint64_t>(absolute) % slots_.size()].value;
    }
private:
    nx_vec<nx_ring_slot_f64> slots_;
};
using nx_ring_ref_f64 = std::shared_ptr<nx_ring_f64>;
static nx_ring_ref_f64 nx_ring_new_f64(std::int64_t capacity) {
    if (capacity <= 0) nx_fail("ring capacity must be positive");
    return std::make_shared<nx_ring_f64>(static_cast<std::size_t>(capacity));
}

// ============================================================================
// 15. Native Shared Library Loader (Rust / Go / C / C++ ABI) & Wire Helpers
// ============================================================================
class NxNativeLibLoader {
public:
    static void *load_symbol(const char *kind, const char *lib_name, const char *symbol) {
        if (NxSandbox::instance().deny_ffi) {
            nx_audit_log_event("sandbox.deny", kind, lib_name, symbol, false);
            nx_fail("sandbox: native FFI loading denied");
        }
        std::string key = std::string(lib_name) + "::" + symbol;
        std::lock_guard<std::mutex> lock(mu());
        auto &sym_cache = symbols();
        auto sit = sym_cache.find(key);
        if (sit != sym_cache.end()) return sit->second;

        auto &lib_cache = libs();
        void *handle = nullptr;
        auto lit = lib_cache.find(lib_name);
        if (lit != lib_cache.end()) {
            handle = lit->second;
        } else {
#ifdef _WIN32
            handle = reinterpret_cast<void *>(LoadLibraryA(lib_name));
#else
            handle = dlopen(lib_name, RTLD_LAZY);
#endif
            if (!handle) {
                nx_audit_log_event("module.load", kind, lib_name, "failed to load native library", false);
                nx_fail_str("cannot load native library: " + std::string(lib_name));
            }
            lib_cache.emplace(lib_name, handle);
            nx_audit_log_event("module.load", kind, lib_name, "native library loaded", true);
        }
#ifdef _WIN32
        void *sym = reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol));
#else
        void *sym = dlsym(handle, symbol);
#endif
        if (!sym) nx_fail_str("missing native symbol: " + key);
        sym_cache.emplace(key, sym);
        return sym;
    }
private:
    static std::mutex &mu() { static std::mutex m; return m; }
    static std::unordered_map<std::string, void *> &libs() { static std::unordered_map<std::string, void *> l; return l; }
    static std::unordered_map<std::string, void *> &symbols() { static std::unordered_map<std::string, void *> s; return s; }
};

// Runtime C++ <-> NxAbiValue converters for direct CPython & Native ABI calls
enum NxAbiWireTags {
    NX_ABI_T_NULL = 0, NX_ABI_T_BOOL = 1, NX_ABI_T_I64 = 2, NX_ABI_T_U64 = 3,
    NX_ABI_T_I128 = 4, NX_ABI_T_U128 = 5, NX_ABI_T_F64 = 6, NX_ABI_T_STR = 7,
    NX_ABI_T_SPAN = 9, NX_ABI_T_LIST = 10, NX_ABI_T_DICT = 11, NX_ABI_T_PYOBJ = 13, NX_ABI_T_ERROR = 14
};
struct NxAbiWireVal {
    std::uint32_t tag = 0;
    std::uint32_t flags = 0;
    union {
        std::int64_t i64;
        std::uint64_t u64;
        double f64;
        void *ptr;
        struct { std::uint64_t lo; std::int64_t hi; } i128_v;
    } payload{};
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

extern "C" {
    int nx_py_abi_sandbox_hook(int (*fn)(const char *));
}

static inline NxAbiWireVal nx_to_abi_wire(bool v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_BOOL; w.payload.i64 = v ? 1 : 0; return w; }
static inline NxAbiWireVal nx_to_abi_wire(std::int8_t v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_I64; w.payload.i64 = v; return w; }
static inline NxAbiWireVal nx_to_abi_wire(std::int16_t v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_I64; w.payload.i64 = v; return w; }
static inline NxAbiWireVal nx_to_abi_wire(std::int32_t v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_I64; w.payload.i64 = v; return w; }
static inline NxAbiWireVal nx_to_abi_wire(std::int64_t v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_I64; w.payload.i64 = v; return w; }
static inline NxAbiWireVal nx_to_abi_wire(std::uint8_t v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_U64; w.payload.u64 = v; return w; }
static inline NxAbiWireVal nx_to_abi_wire(std::uint16_t v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_U64; w.payload.u64 = v; return w; }
static inline NxAbiWireVal nx_to_abi_wire(std::uint32_t v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_U64; w.payload.u64 = v; return w; }
static inline NxAbiWireVal nx_to_abi_wire(std::uint64_t v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_U64; w.payload.u64 = v; return w; }
static inline NxAbiWireVal nx_to_abi_wire(nx_i128 v) {
    NxAbiWireVal w{}; w.tag = NX_ABI_T_I128;
    w.payload.i128_v.lo = static_cast<std::uint64_t>(v);
    w.payload.i128_v.hi = static_cast<std::int64_t>(v >> 64);
    return w;
}
static inline NxAbiWireVal nx_to_abi_wire(nx_u128 v) {
    NxAbiWireVal w{}; w.tag = NX_ABI_T_U128;
    w.payload.i128_v.lo = static_cast<std::uint64_t>(v);
    w.payload.i128_v.hi = static_cast<std::int64_t>(v >> 64);
    return w;
}
static inline NxAbiWireVal nx_to_abi_wire(float v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_F64; w.payload.f64 = static_cast<double>(v); return w; }
static inline NxAbiWireVal nx_to_abi_wire(double v) { NxAbiWireVal w{}; w.tag = NX_ABI_T_F64; w.payload.f64 = v; return w; }
static inline NxAbiWireVal nx_to_abi_wire(const char *s) { NxAbiWireVal w{}; w.tag = NX_ABI_T_STR; w.payload.ptr = const_cast<char *>(s ? s : ""); return w; }
static inline NxAbiWireVal nx_to_abi_wire(const NxStr &s) { NxAbiWireVal w{}; w.tag = NX_ABI_T_STR; w.payload.ptr = const_cast<char *>(s.c_str()); return w; }
static inline NxAbiWireVal nx_to_abi_wire(const NxValue &v) {
    NxAbiWireVal w{};
    switch (v.type) {
        case 0: w.tag = NX_ABI_T_NULL; break;
        case 1: w.tag = NX_ABI_T_I64; w.payload.i64 = v.data.i; break;
        case 2: w.tag = NX_ABI_T_F64; w.payload.f64 = v.data.f; break;
        case 3: w.tag = NX_ABI_T_STR; w.payload.ptr = const_cast<char *>(std::static_pointer_cast<NxStr>(v.data.ptr->object)->c_str()); break;
        case 5: w.tag = NX_ABI_T_BOOL; w.payload.i64 = v.data.i; break;
        case 6: return nx_to_abi_wire(*std::static_pointer_cast<nx_i128>(v.data.ptr->object));
        case 9: w.tag = NX_ABI_T_PYOBJ; w.payload.ptr = v.data.ptr->object.get(); break;
        default: w.tag = NX_ABI_T_STR; w.payload.ptr = const_cast<char *>(nx_as_str(v).c_str()); break;
    }
    return w;
}
'''

PRELUDE = "\n".join([
    _RUNTIME_HEADERS,
    RUNTIME_MEMORY_CPP,
    RUNTIME_SANDBOX_IO_NET_CPP,
    RUNTIME_HYBRID_CPP,
    _RUNTIME_SIMD_RING_ABI,
])
