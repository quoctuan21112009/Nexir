"""C++17 desktop runtime emitted into NEXIR programs.

Spans keep a two-field C ABI for native interoperability. Dynamic values use a
16-byte tagged value and an intrusive atomic count for owned heap payloads.
"""

PRELUDE = r'''
#define _POSIX_C_SOURCE 200112L
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
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

struct nx_span_i32 { const std::int32_t *ptr; std::size_t len; };
struct nx_mutspan_i32 { std::int32_t *ptr; std::size_t len; };
struct nx_span_i64 { const std::int64_t *ptr; std::size_t len; };
struct nx_mutspan_i64 { std::int64_t *ptr; std::size_t len; };
struct nx_span_f32 { const float *ptr; std::size_t len; };
struct nx_mutspan_f32 { float *ptr; std::size_t len; };
struct nx_span_f64 { const double *ptr; std::size_t len; };
struct nx_mutspan_f64 { double *ptr; std::size_t len; };

static std::size_t nx_bounds(std::int64_t index, std::size_t len) {
    if (index < 0 || static_cast<std::uint64_t>(index) >= len) {
        std::fprintf(stderr, "NEXIR index out of bounds: %lld (length %zu)\n",
                     static_cast<long long>(index), len);
        std::exit(1);
    }
    return static_cast<std::size_t>(index);
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
