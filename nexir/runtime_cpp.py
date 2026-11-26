"""C++17 desktop runtime emitted into NEXIR programs.

Spans keep a two-field C ABI for native interoperability. Dynamic values use a
16-byte tagged value and an intrusive atomic count for owned heap payloads.
"""

PRELUDE = r'''
#define _POSIX_C_SOURCE 200112L
#include <algorithm>
#include <cctype>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <sstream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
template <typename T> struct nx_hash : std::hash<T> {};
template <typename T> struct nx_equal : std::equal_to<T> {};
template <> struct nx_hash<const char *> {
    std::size_t operator()(const char *value) const { return std::hash<std::string_view>{}(value); }
};
template <> struct nx_equal<const char *> {
    bool operator()(const char *a, const char *b) const { return std::strcmp(a, b) == 0; }
};
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define NX_X86_SIMD 1
#define NX_TARGET_AVX2 __attribute__((target("avx2")))
#define NX_TARGET_SSE2 __attribute__((target("sse2")))
#endif
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#define NX_EXPORT __declspec(dllexport)
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#define NX_EXPORT __attribute__((visibility("default")))
#endif

[[noreturn]] static void nx_fail(const char *message) {
    std::fprintf(stderr, "NEXIR: %s\n", message);
    std::exit(1);
}

template <typename T> struct nx_result {
    std::variant<T, std::string> state;
    explicit nx_result(T value) : state(std::in_place_index<0>, std::move(value)) {}
    explicit nx_result(std::string error) : state(std::in_place_index<1>, std::move(error)) {}
    bool ok() const { return state.index() == 0; }
    T value() const { if (!ok()) nx_fail("unwrap of error Result"); return std::get<0>(state); }
    const char *error() const { if (ok()) nx_fail("error of successful Result"); return std::get<1>(state).c_str(); }
};
template <typename T> static nx_result<T> nx_ok(T value) { return nx_result<T>(std::move(value)); }
template <typename T> static nx_result<T> nx_err(const char *message) { return nx_result<T>(std::string(message)); }

struct nx_span_i32 { const std::int32_t *ptr; std::size_t len; };
struct nx_mutspan_i32 { std::int32_t *ptr; std::size_t len; };
struct nx_span_i64 { const std::int64_t *ptr; std::size_t len; };
struct nx_mutspan_i64 { std::int64_t *ptr; std::size_t len; };
struct nx_span_f32 { const float *ptr; std::size_t len; };
struct nx_mutspan_f32 { float *ptr; std::size_t len; };
struct nx_span_f64 { const double *ptr; std::size_t len; };
struct nx_mutspan_f64 { double *ptr; std::size_t len; };

static std::size_t nx_bounds(std::int64_t index, std::size_t len);

template <typename T, std::size_t N> struct nx_simd {
    alignas(32) std::array<T, N> lane{};
    T &at(std::int64_t i) { return lane[nx_bounds(i, N)]; }
    const T &at(std::int64_t i) const { return lane[nx_bounds(i, N)]; }
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
template <typename F> static void nx_parallel_for(std::int64_t start, std::int64_t stop, F body) {
    if (stop <= start) return;
    if (start < 0 && stop > std::numeric_limits<std::int64_t>::max() + start)
        nx_fail("parallel_range is too large");
    const std::int64_t count = stop - start;
    const auto hardware = std::thread::hardware_concurrency();
    const std::int64_t workers = std::min<std::int64_t>(count, hardware ? hardware : 1);
    if (workers == 1) { for (std::int64_t i = start; i < stop; ++i) body(i); return; }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    const auto base = count / workers;
    const auto remainder = count % workers;
    for (std::int64_t worker = 0; worker < workers; ++worker) {
        const auto first = start + base * worker + std::min(worker, remainder);
        const auto last = start + base * (worker + 1) + std::min(worker + 1, remainder);
        threads.emplace_back([=, &body] { for (auto i = first; i < last; ++i) body(i); });
    }
    for (auto &thread : threads) thread.join();
}

static std::size_t nx_bounds(std::int64_t index, std::size_t len) {
    if (index < 0 || static_cast<std::uint64_t>(index) >= len) {
        std::fprintf(stderr, "NEXIR index out of bounds: %lld (length %zu)\n",
                     static_cast<long long>(index), len);
        std::exit(1);
    }
    return static_cast<std::size_t>(index);
}

template <typename T> class nx_buffer {
    std::vector<T> storage_;
public:
    explicit nx_buffer(std::int64_t count) {
        if (count < 0) nx_fail("negative Buffer length");
        storage_.resize(static_cast<std::size_t>(count));
    }
    std::size_t size() const { return storage_.size(); }
    T &at(std::int64_t index) { return storage_[nx_bounds(index, storage_.size())]; }
    const T &at(std::int64_t index) const { return storage_[nx_bounds(index, storage_.size())]; }
    T *data() { return storage_.data(); }
    const T *data() const { return storage_.data(); }
};

template <typename T> static T &nx_list_at(std::vector<T> &values, std::int64_t index) {
    return values[nx_bounds(index, values.size())];
}
template <typename T> static T nx_list_pop(std::vector<T> &values) {
    if (values.empty()) nx_fail("pop from empty List");
    T result = std::move(values.back());
    values.pop_back();
    return result;
}
template <typename T> static void nx_list_sort(std::vector<T> &values) { std::sort(values.begin(), values.end()); }
static void nx_list_sort(std::vector<const char *> &values) {
    std::sort(values.begin(), values.end(), [](const char *a, const char *b) { return std::strcmp(a, b) < 0; });
}
template <typename T> static bool nx_list_contains(const std::vector<T> &values, const T &value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}
static bool nx_list_contains(const std::vector<const char *> &values, const char *value) {
    return std::any_of(values.begin(), values.end(), [=](const char *item) { return std::strcmp(item, value) == 0; });
}
template <typename T> static bool nx_span_contains(const T *values, std::size_t count, T value) {
    if (count == 0) return false;
    return std::find(values, values + count, value) != values + count;
}
template <typename T> static std::vector<T> nx_slice(const T *values, std::size_t count,
        std::int64_t start, std::int64_t stop, std::int64_t step, bool has_start, bool has_stop) {
    if (step == 0) nx_fail("slice step cannot be zero");
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) nx_fail("slice is too large");
    const std::int64_t n = static_cast<std::int64_t>(count);
    const bool forward = step > 0;
    if (!has_start) start = forward ? 0 : n - 1;
    else if (start < 0) start += n;
    if (!has_stop) stop = forward ? n : -1;
    else if (stop < 0) stop += n;
    if (forward) {
        start = std::max<std::int64_t>(0, std::min(start, n));
        stop = std::max<std::int64_t>(0, std::min(stop, n));
    } else {
        start = std::max<std::int64_t>(-1, std::min(start, n - 1));
        stop = std::max<std::int64_t>(-1, std::min(stop, n - 1));
    }
    std::vector<T> result;
    if (forward) for (auto i = start; i < stop; i += step) result.push_back(values[i]);
    else for (auto i = start; i > stop; i += step) result.push_back(values[i]);
    return result;
}
template <typename K, typename V, typename H, typename E, typename Q> static V &nx_dict_at(std::unordered_map<K, V, H, E> &values, Q &&key) {
    K converted = std::forward<Q>(key);
    auto found = values.find(converted);
    if (found == values.end()) nx_fail("Dict key not found");
    return found->second;
}
template <typename T, typename H, typename E, typename Q> static void nx_set_remove(std::unordered_set<T, H, E> &values, Q &&value) {
    T converted = std::forward<Q>(value);
    if (!values.erase(converted)) nx_fail("Set element not found");
}
template <typename T, typename H, typename E, typename Q> static void nx_set_discard(std::unordered_set<T, H, E> &values, Q &&value) {
    T converted = std::forward<Q>(value);
    values.erase(converted);
}

static nx_span_i32 nx_as_span_i32(nx_mutspan_i32 s) { return {s.ptr, s.len}; }
static nx_span_i64 nx_as_span_i64(nx_mutspan_i64 s) { return {s.ptr, s.len}; }
static nx_span_f32 nx_as_span_f32(nx_mutspan_f32 s) { return {s.ptr, s.len}; }
static nx_span_f64 nx_as_span_f64(nx_mutspan_f64 s) { return {s.ptr, s.len}; }
static const std::int32_t *nx_at_i32(nx_span_i32 s, std::int64_t i) { return &s.ptr[nx_bounds(i, s.len)]; }
static const std::int64_t *nx_at_i64(nx_span_i64 s, std::int64_t i) { return &s.ptr[nx_bounds(i, s.len)]; }
static const float *nx_at_f32(nx_span_f32 s, std::int64_t i) { return &s.ptr[nx_bounds(i, s.len)]; }
static const double *nx_at_f64(nx_span_f64 s, std::int64_t i) { return &s.ptr[nx_bounds(i, s.len)]; }
static std::int32_t *nx_at_mut_i32(nx_mutspan_i32 s, std::int64_t i) { return &s.ptr[nx_bounds(i, s.len)]; }
static std::int64_t *nx_at_mut_i64(nx_mutspan_i64 s, std::int64_t i) { return &s.ptr[nx_bounds(i, s.len)]; }
static float *nx_at_mut_f32(nx_mutspan_f32 s, std::int64_t i) { return &s.ptr[nx_bounds(i, s.len)]; }
static double *nx_at_mut_f64(nx_mutspan_f64 s, std::int64_t i) { return &s.ptr[nx_bounds(i, s.len)]; }

static std::mutex nx_owner_mutex;
static std::vector<std::shared_ptr<void>> nx_owners;
template <typename T> static void nx_keep(std::shared_ptr<T> value) {
    std::lock_guard<std::mutex> lock(nx_owner_mutex);
    nx_owners.push_back(std::move(value));
}
static const char *nx_string_keep(std::string value) {
    auto owner = std::make_shared<std::string>(std::move(value));
    const char *result = owner->c_str();
    nx_keep(std::move(owner));
    return result;
}
static const char *nx_string_slice(const char *value, std::int64_t start, std::int64_t stop,
        std::int64_t step, bool has_start, bool has_stop) {
    auto chars = nx_slice(value, std::strlen(value), start, stop, step, has_start, has_stop);
    return nx_string_keep(std::string(chars.begin(), chars.end()));
}
static std::string nx_to_string(const char *value) { return value ? value : ""; }
static std::string nx_to_string(bool value) { return value ? "true" : "false"; }
template <typename T> static std::string nx_to_string(T value) {
    std::ostringstream output;
    output << value;
    return output.str();
}
static const char *nx_string_lower(const char *value) {
    std::string out(value);
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return nx_string_keep(std::move(out));
}
static const char *nx_string_upper(const char *value) {
    std::string out(value);
    for (char &c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return nx_string_keep(std::move(out));
}
static const char *nx_string_strip(const char *value) {
    std::string out(value);
    const auto begin = out.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return nx_string_keep("");
    const auto end = out.find_last_not_of(" \t\r\n");
    return nx_string_keep(out.substr(begin, end - begin + 1));
}
static const char *nx_string_replace(const char *value, const char *old_value, const char *new_value) {
    std::string out(value), needle(old_value), replacement(new_value);
    if (needle.empty()) nx_fail("replace requires a nonempty search string");
    std::size_t offset = 0;
    while ((offset = out.find(needle, offset)) != std::string::npos) {
        out.replace(offset, needle.size(), replacement);
        offset += replacement.size();
    }
    return nx_string_keep(std::move(out));
}
static std::vector<const char *> nx_string_split(const char *value, const char *delimiter) {
    std::string text(value), separator(delimiter);
    if (separator.empty()) nx_fail("split requires a nonempty separator");
    std::vector<const char *> out;
    std::size_t start = 0, pos;
    while ((pos = text.find(separator, start)) != std::string::npos) {
        out.push_back(nx_string_keep(text.substr(start, pos - start)));
        start = pos + separator.size();
    }
    out.push_back(nx_string_keep(text.substr(start)));
    return out;
}
static const char *nx_adopt_py_string(char *raw) {
    if (!raw) nx_fail("Python string call failed");
    std::unique_ptr<char, void(*)(void*)> allocation(raw, std::free);
    auto owner = std::make_shared<std::string>(allocation.get());
    const char *value = owner->c_str();
    nx_keep(std::move(owner));
    return value;
}
static std::unique_ptr<nx_buffer<double>> nx_buffer_from_py(nx_mutspan_f64 raw) {
    if (raw.len == std::numeric_limits<std::size_t>::max() ||
        raw.len > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        (raw.len && !raw.ptr)) nx_fail("Python buffer call failed");
    std::unique_ptr<double, void(*)(void*)> allocation(raw.ptr, std::free);
    auto owner = std::make_unique<nx_buffer<double>>(static_cast<std::int64_t>(raw.len));
    if (raw.len) std::copy(allocation.get(), allocation.get() + raw.len, owner->data());
    return owner;
}
template <typename T> static std::pair<T *, std::size_t> nx_make_storage(std::initializer_list<T> values) {
    auto owner = std::make_shared<std::vector<T>>(values);
    auto result = std::make_pair(owner->data(), owner->size());
    nx_keep(std::move(owner));
    return result;
}
template <typename T> static std::pair<T *, std::size_t> nx_make_zeros(std::int64_t count) {
    if (count < 0) nx_fail("negative array length");
    auto owner = std::make_shared<std::vector<T>>(static_cast<std::size_t>(count));
    auto result = std::make_pair(owner->data(), owner->size());
    nx_keep(std::move(owner));
    return result;
}
static nx_mutspan_i32 nx_list_i32(std::initializer_list<std::int32_t> values) {
    auto s = nx_make_storage(values); return {s.first, s.second};
}
static nx_mutspan_i64 nx_list_i64(std::initializer_list<std::int64_t> values) {
    auto s = nx_make_storage(values); return {s.first, s.second};
}
static nx_mutspan_f32 nx_list_f32(std::initializer_list<float> values) {
    auto s = nx_make_storage(values); return {s.first, s.second};
}
static nx_mutspan_f64 nx_list_f64(std::initializer_list<double> values) {
    auto s = nx_make_storage(values); return {s.first, s.second};
}
static nx_mutspan_f32 nx_zeros_f32(std::int64_t n) {
    auto s = nx_make_zeros<float>(n); return {s.first, s.second};
}
static nx_mutspan_f64 nx_zeros_f64(std::int64_t n) {
    auto s = nx_make_zeros<double>(n); return {s.first, s.second};
}
static const char *nx_read_text(const char *path) {
    std::FILE *file = std::fopen(path, "rb");
    if (!file) nx_fail("cannot read file");
    if (std::fseek(file, 0, SEEK_END) != 0) nx_fail("cannot seek file");
    long end = std::ftell(file);
    if (end < 0 || std::fseek(file, 0, SEEK_SET) != 0) nx_fail("cannot seek file");
    auto owner = std::make_shared<std::string>(static_cast<std::size_t>(end), '\0');
    if (std::fread(owner->data(), 1, static_cast<std::size_t>(end), file) != static_cast<std::size_t>(end)) nx_fail("cannot read file");
    std::fclose(file);
    const char *result = owner->c_str();
    nx_keep(std::move(owner));
    return result;
}
static bool nx_tcp_connect(const char *host, std::int32_t port) {
    if (port < 1 || port > 65535) return false;
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
#endif
    char service[8];
    std::snprintf(service, sizeof(service), "%d", static_cast<int>(port));
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *results = nullptr;
    bool connected = false;
    if (getaddrinfo(host, service, &hints, &results) == 0) {
        for (addrinfo *it = results; it; it = it->ai_next) {
#ifdef _WIN32
            SOCKET sock = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
            if (sock == INVALID_SOCKET) continue;
            if (connect(sock, it->ai_addr, static_cast<int>(it->ai_addrlen)) == 0) connected = true;
            closesocket(sock);
#else
            int sock = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
            if (sock < 0) continue;
            if (connect(sock, it->ai_addr, it->ai_addrlen) == 0) connected = true;
            close(sock);
#endif
            if (connected) break;
        }
        freeaddrinfo(results);
    }
#ifdef _WIN32
    WSACleanup();
#endif
    return connected;
}

// The payload is 8 bytes; owned objects point to an intrusive atomic control block.
struct nx_heap {
    std::atomic<std::uint32_t> references{1};
    std::shared_ptr<void> object;
    explicit nx_heap(std::shared_ptr<void> value) : object(std::move(value)) {}
};
struct alignas(16) NxValue {
    std::uint32_t type = 0; // 0 null, 1 i64, 2 f64, 3 string, 4 object
    std::uint32_t reserved = 0;
    union payload { std::int64_t i; double f; nx_heap *ptr; payload() : i(0) {} } data;

    NxValue() = default;
    NxValue(const NxValue &other) : type(other.type), reserved(0), data() {
        data = other.data;
        retain();
    }
    NxValue(NxValue &&other) noexcept : type(other.type), reserved(0), data() {
        data = other.data;
        other.type = 0;
        other.data.i = 0;
    }
    NxValue &operator=(const NxValue &other) {
        if (this != &other) { release(); type = other.type; data = other.data; retain(); }
        return *this;
    }
    NxValue &operator=(NxValue &&other) noexcept {
        if (this != &other) {
            release(); type = other.type; data = other.data;
            other.type = 0; other.data.i = 0;
        }
        return *this;
    }
    ~NxValue() { release(); }
    void retain() { if (type >= 3 && data.ptr) data.ptr->references.fetch_add(1, std::memory_order_relaxed); }
    void release() {
        if (type >= 3 && data.ptr && data.ptr->references.fetch_sub(1, std::memory_order_acq_rel) == 1) delete data.ptr;
    }
    static NxValue integer(std::int64_t n) { NxValue v; v.type = 1; v.data.i = n; return v; }
    static NxValue floating(double n) { NxValue v; v.type = 2; v.data.f = n; return v; }
    static NxValue string(const char *s) {
        NxValue v; v.type = 3;
        v.data.ptr = new nx_heap(std::make_shared<std::string>(s));
        return v;
    }
    static NxValue object(std::shared_ptr<void> owner) {
        NxValue v; v.type = 4; v.data.ptr = new nx_heap(std::move(owner)); return v;
    }
};
static_assert(sizeof(NxValue) == 16, "NxValue must be 16 bytes");
static_assert(alignof(NxValue) == 16, "NxValue must be 16-byte aligned");
static_assert(offsetof(NxValue, data) == 8, "payload offset must be 8");
static NxValue nx_box(std::int32_t n) { return NxValue::integer(n); }
static NxValue nx_box(std::int64_t n) { return NxValue::integer(n); }
static NxValue nx_box(float n) { return NxValue::floating(n); }
static NxValue nx_box(double n) { return NxValue::floating(n); }
static NxValue nx_box(const char *s) { return NxValue::string(s); }
static NxValue nx_box(bool n) { return NxValue::integer(n ? 1 : 0); }
static const char *nx_type_name(const NxValue &v) {
    switch (v.type) {
        case 0: return "null"; case 1: return "i64"; case 2: return "f64";
        case 3: return "str"; case 4: return "object";
        default: return "invalid";
    }
}
static std::int64_t nx_as_i64(const NxValue &v) {
    if (v.type != 1) nx_fail("dynamic value is not i64");
    return v.data.i;
}
static double nx_as_f64(const NxValue &v) {
    if (v.type == 1) return static_cast<double>(v.data.i);
    if (v.type == 2) return v.data.f;
    nx_fail("dynamic value is not numeric");
}
static NxValue nx_add_dyn(const NxValue &a, const NxValue &b) {
    if (a.type == 1 && b.type == 1) {
        if ((b.data.i > 0 && a.data.i > std::numeric_limits<std::int64_t>::max() - b.data.i) ||
            (b.data.i < 0 && a.data.i < std::numeric_limits<std::int64_t>::min() - b.data.i)) nx_fail("dynamic i64 overflow");
        return NxValue::integer(a.data.i + b.data.i);
    }
    if ((a.type == 1 || a.type == 2) && (b.type == 1 || b.type == 2))
        return NxValue::floating(nx_as_f64(a) + nx_as_f64(b));
    if (a.type == 3 && b.type == 3) {
        const auto &left = *std::static_pointer_cast<std::string>(a.data.ptr->object);
        const auto &right = *std::static_pointer_cast<std::string>(b.data.ptr->object);
        return NxValue::string((left + right).c_str());
    }
    nx_fail("unsupported dynamic + operands");
}
static void nx_print_value(const NxValue &v) {
    switch (v.type) {
        case 0: std::fputs("null", stdout); break;
        case 1: std::printf("%lld", static_cast<long long>(v.data.i)); break;
        case 2: std::printf("%.17g", v.data.f); break;
        case 3: std::fputs(std::static_pointer_cast<std::string>(v.data.ptr->object)->c_str(), stdout); break;
        case 4: std::fputs("<object>", stdout); break;
        default: nx_fail("invalid dynamic value");
    }
}

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
    std::vector<nx_ring_slot_f64> slots_;
};
using nx_ring_ref_f64 = std::shared_ptr<nx_ring_f64>;
static nx_ring_ref_f64 nx_ring_new_f64(std::int64_t capacity) {
    if (capacity <= 0) nx_fail("ring capacity must be positive");
    return std::make_shared<nx_ring_f64>(static_cast<std::size_t>(capacity));
}
'''
