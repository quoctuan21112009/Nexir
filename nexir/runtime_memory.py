"""C++17 memory cache, 128-bit numeric conversions, owned SSO/cached strings,
StringBuilder, format engine, RAII collections, and hybrid NxValue v2.
"""

RUNTIME_MEMORY_CPP = r'''
// ============================================================================
// 1. Debug Trace Stack & Panic Handler
// ============================================================================
struct NxTraceFrame {
    const char *file;
    const char *func;
    int line;
    NxTraceFrame *prev;
};
inline thread_local NxTraceFrame *nx_tls_trace_top = nullptr;

struct NxTraceGuard {
    NxTraceFrame frame;
    NxTraceGuard(const char *file, const char *func, int line) noexcept
        : frame{file, func, line, nx_tls_trace_top} {
        nx_tls_trace_top = &frame;
    }
    ~NxTraceGuard() noexcept {
        nx_tls_trace_top = frame.prev;
    }
    void set_line(int line) noexcept { frame.line = line; }
};

static void nx_dump_trace() noexcept {
    if (!nx_tls_trace_top) return;
    std::fprintf(stderr, "NX Stack Trace (most recent call first):\n");
    for (NxTraceFrame *cur = nx_tls_trace_top; cur; cur = cur->prev) {
        std::fprintf(stderr, "  at %s (%s:%d)\n",
                     cur->func ? cur->func : "<unknown>",
                     cur->file ? cur->file : "<unknown>",
                     cur->line);
    }
}

[[noreturn]] static void nx_fail(const char *message) {
    std::fprintf(stderr, "NEXIR: %s\n", message ? message : "fatal error");
    nx_dump_trace();
    std::exit(1);
}

[[noreturn]] static void nx_fail_str(const std::string &message) {
    nx_fail(message.c_str());
}

// ============================================================================
// 2. Sandbox Resource & Audit Declarations (implemented in runtime_sandbox)
// ============================================================================
static void nx_sandbox_on_alloc(std::size_t bytes);
static void nx_sandbox_on_free(std::size_t bytes) noexcept;
static void nx_audit_log_event(const char *event, const char *kind, const char *name,
                               const char *detail, bool allowed);

// ============================================================================
// 3. Cache-Based Memory Allocator & Leak Tracker (Replaces global nx_owners!)
// ============================================================================
struct NxMemStats {
    std::atomic<std::uint64_t> active_bytes{0};
    std::atomic<std::uint64_t> peak_bytes{0};
    std::atomic<std::uint64_t> total_allocs{0};
    std::atomic<std::uint64_t> total_frees{0};
    std::atomic<std::uint64_t> cache_hits{0};
    std::atomic<std::uint64_t> live_blocks{0};
    std::atomic<std::uint64_t> intern_hits{0};
};

inline NxMemStats &nx_mem_stats() {
    static NxMemStats stats;
    return stats;
}

inline std::atomic<bool> &nx_mem_cache_alive() noexcept {
    static std::atomic<bool> alive{true};
    return alive;
}

class NxMemCache {
public:
    // 9 size classes: 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384
    static constexpr std::size_t kBins = 9;
    static constexpr std::size_t kMaxLocalPerBin = 64;
    static constexpr std::size_t kMaxSharedPerBin = 256;

    static constexpr std::size_t bin_size(std::size_t idx) noexcept {
        constexpr std::size_t sizes[kBins] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 16384};
        return sizes[idx];
    }

    static std::size_t bin_index(std::size_t bytes) noexcept {
        if (bytes <= 32) return 0;
        if (bytes <= 64) return 1;
        if (bytes <= 128) return 2;
        if (bytes <= 256) return 3;
        if (bytes <= 512) return 4;
        if (bytes <= 1024) return 5;
        if (bytes <= 2048) return 6;
        if (bytes <= 4096) return 7;
        if (bytes <= 16384) return 8;
        return kBins;
    }

    struct Node { Node *next; };

    // POD ThreadCache (no non-trivial dtor) avoids MinGW/UCRT emutls + __cxa_thread_atexit race.
    struct ThreadCache {
        Node *bins[kBins]{};
        std::size_t counts[kBins]{};
    };

    ThreadCache &tls_cache() {
        thread_local ThreadCache *tc_ptr = nullptr;
        if (!tc_ptr) {
            auto owned = std::make_unique<ThreadCache>();
            ThreadCache *raw = owned.get();
            std::lock_guard<std::mutex> lock(mu_);
            thread_caches_.push_back(std::move(owned));
            tc_ptr = raw;
        }
        return *tc_ptr;
    }

    static NxMemCache &shared() {
        static NxMemCache instance;
        return instance;
    }

    void *allocate(std::size_t bytes, std::size_t *actual_cap = nullptr) {
        if (bytes == 0) bytes = 1;
        std::size_t idx = bin_index(bytes);
        std::size_t alloc_size = idx < kBins ? bin_size(idx) : bytes;
        if (actual_cap) *actual_cap = alloc_size;

        nx_sandbox_on_alloc(alloc_size);
        auto &st = nx_mem_stats();
        st.total_allocs.fetch_add(1, std::memory_order_relaxed);
        st.live_blocks.fetch_add(1, std::memory_order_relaxed);
        std::uint64_t cur = st.active_bytes.fetch_add(alloc_size, std::memory_order_relaxed) + alloc_size;
        std::uint64_t peak = st.peak_bytes.load(std::memory_order_relaxed);
        while (cur > peak && !st.peak_bytes.compare_exchange_weak(peak, cur, std::memory_order_relaxed)) {}

        if (idx < kBins && nx_mem_cache_alive().load(std::memory_order_acquire)) {
            ThreadCache &tc = tls_cache();
            if (tc.bins[idx]) {
                Node *n = tc.bins[idx];
                tc.bins[idx] = n->next;
                --tc.counts[idx];
                st.cache_hits.fetch_add(1, std::memory_order_relaxed);
                return static_cast<void *>(n);
            }
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (shared_bins_[idx]) {
                    Node *n = shared_bins_[idx];
                    shared_bins_[idx] = n->next;
                    --shared_counts_[idx];
                    st.cache_hits.fetch_add(1, std::memory_order_relaxed);
                    return static_cast<void *>(n);
                }
            }
        }
        void *mem = std::malloc(alloc_size);
        if (!mem) nx_fail("out of memory in NxMemCache::allocate");
        return mem;
    }

    void deallocate(void *ptr, std::size_t alloc_size) noexcept {
        if (!ptr) return;
        std::size_t idx = bin_index(alloc_size);
        std::size_t real_size = idx < kBins ? bin_size(idx) : alloc_size;
        auto &st = nx_mem_stats();
        st.total_frees.fetch_add(1, std::memory_order_relaxed);
        st.live_blocks.fetch_sub(1, std::memory_order_relaxed);
        st.active_bytes.fetch_sub(real_size, std::memory_order_relaxed);
        nx_sandbox_on_free(real_size);

        if (idx < kBins && nx_mem_cache_alive().load(std::memory_order_acquire)) {
            ThreadCache &tc = tls_cache();
            if (tc.counts[idx] < kMaxLocalPerBin) {
                Node *n = static_cast<Node *>(ptr);
                n->next = tc.bins[idx];
                tc.bins[idx] = n;
                ++tc.counts[idx];
                return;
            }
            std::lock_guard<std::mutex> lock(mu_);
            if (shared_counts_[idx] < kMaxSharedPerBin) {
                Node *n = static_cast<Node *>(ptr);
                n->next = shared_bins_[idx];
                shared_bins_[idx] = n;
                ++shared_counts_[idx];
                return;
            }
        }
        std::free(ptr);
    }

    void drain_shared() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        for (std::size_t i = 0; i < kBins; ++i) {
            while (shared_bins_[i]) {
                Node *n = shared_bins_[i];
                shared_bins_[i] = n->next;
                std::free(n);
            }
            shared_counts_[i] = 0;
        }
        for (auto &tc : thread_caches_) {
            if (!tc) continue;
            for (std::size_t i = 0; i < kBins; ++i) {
                while (tc->bins[i]) {
                    Node *n = tc->bins[i];
                    tc->bins[i] = n->next;
                    std::free(n);
                }
                tc->counts[i] = 0;
            }
        }
    }

    ~NxMemCache() {
        nx_mem_cache_alive().store(false, std::memory_order_release);
        drain_shared();
    }

private:
    std::mutex mu_;
    Node *shared_bins_[kBins]{};
    std::size_t shared_counts_[kBins]{};
    std::vector<std::unique_ptr<ThreadCache>> thread_caches_;
};

// Custom STL allocator routing containers/allocations through NxMemCache + Sandbox tracking
template <typename T>
class NxAllocator {
public:
    using value_type = T;
    NxAllocator() noexcept = default;
    template <typename U> NxAllocator(const NxAllocator<U> &) noexcept {}
    T *allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) nx_fail("allocation overflow");
        return static_cast<T *>(NxMemCache::shared().allocate(n * sizeof(T)));
    }
    void deallocate(T *p, std::size_t n) noexcept {
        NxMemCache::shared().deallocate(p, n * sizeof(T));
    }
    template <typename U> bool operator==(const NxAllocator<U> &) const noexcept { return true; }
    template <typename U> bool operator!=(const NxAllocator<U> &) const noexcept { return false; }
};

// ============================================================================
// 4. 128-bit & Multi-width Numeric Types + Checked Free Conversions
// ============================================================================
#if defined(__SIZEOF_INT128__)
using nx_i128 = __int128;
using nx_u128 = unsigned __int128;
#else
#error "NEXIR 2.0 requires __int128 compiler support or portable fallback"
#endif

static inline std::string nx_i128_to_std_string(nx_i128 val) {
    if (val == 0) return "0";
    bool neg = val < 0;
    nx_u128 mag = neg ? (static_cast<nx_u128>(-(val + 1)) + 1) : static_cast<nx_u128>(val);
    char buf[48];
    int pos = 47;
    buf[pos] = '\0';
    while (mag > 0) {
        buf[--pos] = static_cast<char>('0' + static_cast<unsigned>(mag % 10));
        mag /= 10;
    }
    if (neg) buf[--pos] = '-';
    return std::string(buf + pos);
}

static inline std::string nx_u128_to_std_string(nx_u128 val) {
    if (val == 0) return "0";
    char buf[48];
    int pos = 47;
    buf[pos] = '\0';
    while (val > 0) {
        buf[--pos] = static_cast<char>('0' + static_cast<unsigned>(val % 10));
        val /= 10;
    }
    return std::string(buf + pos);
}

static inline void nx_print_i128(nx_i128 val) {
    std::string s = nx_i128_to_std_string(val);
    std::fputs(s.c_str(), stdout);
}

static inline void nx_print_u128(nx_u128 val) {
    std::string s = nx_u128_to_std_string(val);
    std::fputs(s.c_str(), stdout);
}

static inline nx_i128 nx_parse_i128_raw(const char *s) {
    if (!s || !*s) nx_fail("cannot parse empty string as integer");
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    bool neg = false;
    if (*p == '-') { neg = true; ++p; }
    else if (*p == '+') { ++p; }
    if (*p < '0' || *p > '9') nx_fail("invalid integer literal string");
    nx_u128 acc = 0;
    const nx_u128 max_pos = (static_cast<nx_u128>(1) << 127) - 1;
    const nx_u128 max_neg = (static_cast<nx_u128>(1) << 127);
    const nx_u128 limit = neg ? max_neg : max_pos;
    for (; *p >= '0' && *p <= '9'; ++p) {
        unsigned digit = static_cast<unsigned>(*p - '0');
        if (acc > (limit - digit) / 10) nx_fail("integer overflow parsing i128");
        acc = acc * 10 + digit;
    }
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    if (*p != '\0') nx_fail("trailing characters in integer string");
    if (neg) {
        if (acc == max_neg) return static_cast<nx_i128>(acc);
        return -static_cast<nx_i128>(acc);
    }
    return static_cast<nx_i128>(acc);
}

static inline nx_u128 nx_parse_u128_raw(const char *s) {
    if (!s || !*s) nx_fail("cannot parse empty string as unsigned integer");
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    if (*p == '+') ++p;
    if (*p < '0' || *p > '9') nx_fail("invalid unsigned integer string");
    nx_u128 acc = 0;
    const nx_u128 limit = ~static_cast<nx_u128>(0);
    for (; *p >= '0' && *p <= '9'; ++p) {
        unsigned digit = static_cast<unsigned>(*p - '0');
        if (acc > (limit - digit) / 10) nx_fail("integer overflow parsing u128");
        acc = acc * 10 + digit;
    }
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    if (*p != '\0') nx_fail("trailing characters in unsigned integer string");
    return acc;
}

template <typename T>
struct nx_is_int128 : std::bool_constant<std::is_same_v<std::decay_t<T>, nx_i128> || std::is_same_v<std::decay_t<T>, nx_u128>> {};

template <typename T>
struct nx_is_numeric : std::bool_constant<std::is_arithmetic_v<std::decay_t<T>> || nx_is_int128<T>::value> {};

template <typename Dst, typename Src>
static inline Dst nx_num_cast(Src value) {
    using D = std::decay_t<Dst>;
    using S = std::decay_t<Src>;
    if constexpr (std::is_same_v<D, S>) {
        return value;
    } else if constexpr (std::is_same_v<D, bool>) {
        return value != 0;
    } else if constexpr (std::is_floating_point_v<D>) {
        return static_cast<D>(value);
    } else if constexpr (std::is_floating_point_v<S>) {
        if (!std::isfinite(value)) nx_fail("cannot convert non-finite float to integer");
        return static_cast<D>(value);
    } else {
        // Integer <-> Integer (including 8..128 bit signed/unsigned) with safe range check
        if constexpr (std::is_same_v<D, nx_i128>) {
            if constexpr (std::is_same_v<S, nx_u128>) {
                constexpr nx_u128 max_i128 = (static_cast<nx_u128>(1) << 127) - 1;
                if (value > max_i128) nx_fail("integer conversion overflow (u128 -> i128)");
            }
            return static_cast<nx_i128>(value);
        } else if constexpr (std::is_same_v<D, nx_u128>) {
            if constexpr (std::is_signed_v<S> || std::is_same_v<S, nx_i128>) {
                if (value < 0) nx_fail("negative value cannot convert to unsigned u128");
            }
            return static_cast<nx_u128>(value);
        } else {
            // Standard integral destination (i8..i64, u8..u64)
            constexpr nx_i128 dmin = static_cast<nx_i128>(std::numeric_limits<D>::min());
            constexpr nx_u128 dmax = static_cast<nx_u128>(std::numeric_limits<D>::max());
            if constexpr (std::is_same_v<S, nx_u128> || std::is_unsigned_v<S>) {
                if (static_cast<nx_u128>(value) > dmax) nx_fail("integer narrowing conversion overflow");
                return static_cast<D>(value);
            } else {
                nx_i128 sv = static_cast<nx_i128>(value);
                if (sv < dmin) nx_fail("integer narrowing conversion underflow");
                if (sv > 0 && static_cast<nx_u128>(sv) > dmax) nx_fail("integer narrowing conversion overflow");
                return static_cast<D>(sv);
            }
        }
    }
}

// Arithmetic promotion helpers for mixed numeric widths (including i128/u128)
static inline nx_i128 nx_pow_i128(nx_i128 base, nx_i128 exp) {
    if (exp < 0) nx_fail("negative exponent for integer power");
    nx_i128 res = 1;
    while (exp > 0) {
        if (exp & 1) res *= base;
        exp >>= 1;
        if (exp > 0) base *= base;
    }
    return res;
}

// ============================================================================
// 5. Owned String Type (NxStr) with SSO + RefCounted Cache + Interning
//    Replaces leaking const char* / nx_owners!
// ============================================================================
struct NxStrBlock {
    std::atomic<std::uint32_t> refs{1};
    std::uint32_t hash_val{0};
    std::size_t len{0};
    std::size_t alloc_size{0};
    char *data() noexcept { return reinterpret_cast<char *>(this + 1); }
    const char *data() const noexcept { return reinterpret_cast<const char *>(this + 1); }
};

class NxStrInternCache {
public:
    static constexpr std::size_t kSlots = 1024; // bounded ring/direct-mapped cache
    static constexpr std::size_t kMaxInternLen = 96;

    static NxStrInternCache &instance() {
        static NxStrInternCache cache;
        return cache;
    }

    static std::uint32_t fnv1a(const char *p, std::size_t n) noexcept {
        std::uint32_t h = 2166136261u;
        for (std::size_t i = 0; i < n; ++i) {
            h ^= static_cast<unsigned char>(p[i]);
            h *= 16777619u;
        }
        return h;
    }

    NxStrBlock *lookup_or_insert(const char *p, std::size_t n, std::uint32_t h) {
        std::size_t slot = h & (kSlots - 1);
        {
            std::lock_guard<std::mutex> lock(mu_);
            NxStrBlock *existing = slots_[slot];
            if (existing && existing->hash_val == h && existing->len == n &&
                std::memcmp(existing->data(), p, n) == 0) {
                existing->refs.fetch_add(1, std::memory_order_relaxed);
                nx_mem_stats().intern_hits.fetch_add(1, std::memory_order_relaxed);
                return existing;
            }
        }
        std::size_t total = sizeof(NxStrBlock) + n + 1;
        std::size_t cap = 0;
        void *mem = NxMemCache::shared().allocate(total, &cap);
        NxStrBlock *blk = new (mem) NxStrBlock();
        blk->hash_val = h;
        blk->len = n;
        blk->alloc_size = cap;
        if (n) std::memcpy(blk->data(), p, n);
        blk->data()[n] = '\0';

        if (n <= kMaxInternLen) {
            std::lock_guard<std::mutex> lock(mu_);
            NxStrBlock *old = slots_[slot];
            blk->refs.fetch_add(1, std::memory_order_relaxed); // cache holds 1 ref
            slots_[slot] = blk;
            if (old && old->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::size_t old_sz = old->alloc_size;
                old->~NxStrBlock();
                NxMemCache::shared().deallocate(old, old_sz);
            }
        }
        return blk;
    }

    void clear() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        for (std::size_t i = 0; i < kSlots; ++i) {
            if (slots_[i]) {
                NxStrBlock *old = slots_[i];
                slots_[i] = nullptr;
                if (old->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::size_t old_sz = old->alloc_size;
                    old->~NxStrBlock();
                    NxMemCache::shared().deallocate(old, old_sz);
                }
            }
        }
    }

    ~NxStrInternCache() { clear(); }

private:
    std::mutex mu_;
    NxStrBlock *slots_[kSlots]{};
};

class NxStr {
public:
    static constexpr std::size_t kSsoMax = 22;
    // Storage layout (24 bytes):
    // kind_ == 0: SSO inline buf_[23] (len_ = kind_tag & 0x1F)
    // kind_ == 1: static literal view (ptr_, len_)
    // kind_ == 2: refcounted heap NxStrBlock*
    enum Mode : std::uint8_t { MODE_SSO = 0, MODE_STATIC = 1, MODE_HEAP = 2 };

    NxStr() noexcept : mode_(MODE_SSO), sso_len_(0) { sso_buf_[0] = '\0'; }

    // Construct from string literal or C string
    NxStr(const char *s) : NxStr(s ? s : "", s ? std::strlen(s) : 0, false) {}

    static NxStr from_literal(const char *s, std::size_t n) noexcept {
        NxStr out;
        if (n <= kSsoMax) {
            out.mode_ = MODE_SSO;
            out.sso_len_ = static_cast<std::uint8_t>(n);
            if (n) std::memcpy(out.sso_buf_, s, n);
            out.sso_buf_[n] = '\0';
        } else {
            out.mode_ = MODE_STATIC;
            out.ext_.ptr = s;
            out.ext_.len = n;
        }
        return out;
    }

    static NxStr from_copy(const char *s, std::size_t n) {
        return NxStr(s ? s : "", n, true);
    }

    static NxStr from_std(const std::string &s) {
        return NxStr(s.data(), s.size(), true);
    }

    NxStr(const char *s, std::size_t n, bool intern = true) {
        if (!s) n = 0;
        if (n <= kSsoMax) {
            mode_ = MODE_SSO;
            sso_len_ = static_cast<std::uint8_t>(n);
            if (n) std::memcpy(sso_buf_, s, n);
            sso_buf_[n] = '\0';
        } else if (intern && n <= NxStrInternCache::kMaxInternLen) {
            mode_ = MODE_HEAP;
            std::uint32_t h = NxStrInternCache::fnv1a(s, n);
            heap_.blk = NxStrInternCache::instance().lookup_or_insert(s, n, h);
        } else {
            mode_ = MODE_HEAP;
            std::size_t total = sizeof(NxStrBlock) + n + 1;
            std::size_t cap = 0;
            void *mem = NxMemCache::shared().allocate(total, &cap);
            NxStrBlock *blk = new (mem) NxStrBlock();
            blk->hash_val = 0;
            blk->len = n;
            blk->alloc_size = cap;
            if (n) std::memcpy(blk->data(), s, n);
            blk->data()[n] = '\0';
            heap_.blk = blk;
        }
    }

    NxStr(const NxStr &other) noexcept : mode_(other.mode_) {
        if (mode_ == MODE_SSO) {
            sso_len_ = other.sso_len_;
            std::memcpy(sso_buf_, other.sso_buf_, sso_len_ + 1);
        } else if (mode_ == MODE_STATIC) {
            ext_ = other.ext_;
        } else {
            heap_ = other.heap_;
            if (heap_.blk) heap_.blk->refs.fetch_add(1, std::memory_order_relaxed);
        }
    }

    NxStr(NxStr &&other) noexcept : mode_(other.mode_) {
        if (mode_ == MODE_SSO) {
            sso_len_ = other.sso_len_;
            std::memcpy(sso_buf_, other.sso_buf_, sso_len_ + 1);
        } else if (mode_ == MODE_STATIC) {
            ext_ = other.ext_;
        } else {
            heap_ = other.heap_;
            other.mode_ = MODE_SSO;
            other.sso_len_ = 0;
            other.sso_buf_[0] = '\0';
        }
    }

    NxStr &operator=(const NxStr &other) noexcept {
        if (this == &other) return *this;
        release();
        mode_ = other.mode_;
        if (mode_ == MODE_SSO) {
            sso_len_ = other.sso_len_;
            std::memcpy(sso_buf_, other.sso_buf_, sso_len_ + 1);
        } else if (mode_ == MODE_STATIC) {
            ext_ = other.ext_;
        } else {
            heap_ = other.heap_;
            if (heap_.blk) heap_.blk->refs.fetch_add(1, std::memory_order_relaxed);
        }
        return *this;
    }

    NxStr &operator=(NxStr &&other) noexcept {
        if (this == &other) return *this;
        release();
        mode_ = other.mode_;
        if (mode_ == MODE_SSO) {
            sso_len_ = other.sso_len_;
            std::memcpy(sso_buf_, other.sso_buf_, sso_len_ + 1);
        } else if (mode_ == MODE_STATIC) {
            ext_ = other.ext_;
        } else {
            heap_ = other.heap_;
            other.mode_ = MODE_SSO;
            other.sso_len_ = 0;
            other.sso_buf_[0] = '\0';
        }
        return *this;
    }

    ~NxStr() noexcept { release(); }

    const char *c_str() const noexcept {
        if (mode_ == MODE_SSO) return sso_buf_;
        if (mode_ == MODE_STATIC) return ext_.ptr;
        return heap_.blk ? heap_.blk->data() : "";
    }

    std::size_t size() const noexcept {
        if (mode_ == MODE_SSO) return sso_len_;
        if (mode_ == MODE_STATIC) return ext_.len;
        return heap_.blk ? heap_.blk->len : 0;
    }

    bool empty() const noexcept { return size() == 0; }
    std::string_view view() const noexcept { return std::string_view(c_str(), size()); }
    operator std::string_view() const noexcept { return view(); }

    NxStr operator+(const NxStr &rhs) const {
        std::size_t l1 = size(), l2 = rhs.size();
        if (l1 == 0) return rhs;
        if (l2 == 0) return *this;
        std::size_t total_len = l1 + l2;
        if (total_len <= kSsoMax) {
            NxStr out;
            out.mode_ = MODE_SSO;
            out.sso_len_ = static_cast<std::uint8_t>(total_len);
            std::memcpy(out.sso_buf_, c_str(), l1);
            std::memcpy(out.sso_buf_ + l1, rhs.c_str(), l2);
            out.sso_buf_[total_len] = '\0';
            return out;
        }
        std::size_t total = sizeof(NxStrBlock) + total_len + 1;
        std::size_t cap = 0;
        void *mem = NxMemCache::shared().allocate(total, &cap);
        NxStrBlock *blk = new (mem) NxStrBlock();
        blk->hash_val = 0;
        blk->len = total_len;
        blk->alloc_size = cap;
        std::memcpy(blk->data(), c_str(), l1);
        std::memcpy(blk->data() + l1, rhs.c_str(), l2);
        blk->data()[total_len] = '\0';
        NxStr out;
        out.mode_ = MODE_HEAP;
        out.heap_.blk = blk;
        return out;
    }

    bool operator==(const NxStr &rhs) const noexcept { return view() == rhs.view(); }
    bool operator!=(const NxStr &rhs) const noexcept { return view() != rhs.view(); }
    bool operator<(const NxStr &rhs) const noexcept { return view() < rhs.view(); }
    bool operator<=(const NxStr &rhs) const noexcept { return view() <= rhs.view(); }
    bool operator>(const NxStr &rhs) const noexcept { return view() > rhs.view(); }
    bool operator>=(const NxStr &rhs) const noexcept { return view() >= rhs.view(); }

    NxStr at_char(std::int64_t idx) const {
        std::int64_t n = static_cast<std::int64_t>(size());
        if (idx < 0) idx += n;
        if (idx < 0 || idx >= n) nx_fail("string index out of bounds");
        char ch = c_str()[idx];
        return NxStr(&ch, 1, false);
    }

    NxStr lower() const {
        std::size_t n = size();
        const char *p = c_str();
        std::string tmp(n, '\0');
        for (std::size_t i = 0; i < n; ++i)
            tmp[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(p[i])));
        return NxStr(tmp.data(), n, false);
    }
    NxStr upper() const {
        std::size_t n = size();
        const char *p = c_str();
        std::string tmp(n, '\0');
        for (std::size_t i = 0; i < n; ++i)
            tmp[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(p[i])));
        return NxStr(tmp.data(), n, false);
    }
    NxStr strip(const char *chars = " \t\r\n") const {
        auto v = view();
        std::size_t b = v.find_first_not_of(chars);
        if (b == std::string_view::npos) return NxStr();
        std::size_t e = v.find_last_not_of(chars);
        return NxStr(v.data() + b, e - b + 1, false);
    }
    NxStr lstrip(const char *chars = " \t\r\n") const {
        auto v = view();
        std::size_t b = v.find_first_not_of(chars);
        if (b == std::string_view::npos) return NxStr();
        return NxStr(v.data() + b, v.size() - b, false);
    }
    NxStr rstrip(const char *chars = " \t\r\n") const {
        auto v = view();
        std::size_t e = v.find_last_not_of(chars);
        if (e == std::string_view::npos) return NxStr();
        return NxStr(v.data(), e + 1, false);
    }
    NxStr replace(const NxStr &old_s, const NxStr &new_s) const {
        if (old_s.empty()) nx_fail("replace requires a nonempty search string");
        auto v = view();
        auto ov = old_s.view();
        auto nv = new_s.view();
        std::string out;
        out.reserve(v.size());
        std::size_t pos = 0, found;
        while ((found = v.find(ov, pos)) != std::string_view::npos) {
            out.append(v.data() + pos, found - pos);
            out.append(nv.data(), nv.size());
            pos = found + ov.size();
        }
        out.append(v.data() + pos, v.size() - pos);
        return NxStr(out.data(), out.size(), false);
    }
    bool startswith(const NxStr &prefix) const noexcept {
        auto v = view(), pv = prefix.view();
        return v.size() >= pv.size() && std::memcmp(v.data(), pv.data(), pv.size()) == 0;
    }
    bool endswith(const NxStr &suffix) const noexcept {
        auto v = view(), sv = suffix.view();
        return v.size() >= sv.size() && std::memcmp(v.data() + (v.size() - sv.size()), sv.data(), sv.size()) == 0;
    }
    std::int64_t find(const NxStr &needle, std::int64_t start = 0) const noexcept {
        auto v = view();
        if (start < 0) start = std::max<std::int64_t>(0, static_cast<std::int64_t>(v.size()) + start);
        if (static_cast<std::size_t>(start) > v.size()) return -1;
        auto pos = v.find(needle.view(), static_cast<std::size_t>(start));
        return pos == std::string_view::npos ? -1 : static_cast<std::int64_t>(pos);
    }
    std::int64_t count(const NxStr &needle) const noexcept {
        if (needle.empty()) return static_cast<std::int64_t>(size() + 1);
        auto v = view(), nv = needle.view();
        std::int64_t cnt = 0;
        std::size_t pos = 0;
        while ((pos = v.find(nv, pos)) != std::string_view::npos) {
            ++cnt;
            pos += nv.size();
        }
        return cnt;
    }
    bool isdigit() const noexcept {
        auto v = view();
        if (v.empty()) return false;
        for (unsigned char c : v) if (!std::isdigit(c)) return false;
        return true;
    }
    bool isalpha() const noexcept {
        auto v = view();
        if (v.empty()) return false;
        for (unsigned char c : v) if (!std::isalpha(c)) return false;
        return true;
    }
    bool contains(const NxStr &sub) const noexcept {
        return view().find(sub.view()) != std::string_view::npos;
    }
    NxStr repeat(std::int64_t n) const {
        if (n <= 0 || empty()) return NxStr();
        std::size_t l = size();
        std::size_t total_len = l * static_cast<std::size_t>(n);
        std::string tmp;
        tmp.reserve(total_len);
        for (std::int64_t i = 0; i < n; ++i) tmp.append(c_str(), l);
        return NxStr(tmp.data(), tmp.size(), false);
    }

private:
    void release() noexcept {
        if (mode_ == MODE_HEAP && heap_.blk) {
            if (heap_.blk->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::size_t sz = heap_.blk->alloc_size;
                heap_.blk->~NxStrBlock();
                NxMemCache::shared().deallocate(heap_.blk, sz);
            }
            heap_.blk = nullptr;
        }
    }

    Mode mode_;
    std::uint8_t sso_len_{0};
    union {
        char sso_buf_[kSsoMax + 1];
        struct { const char *ptr; std::size_t len; } ext_;
        struct { NxStrBlock *blk; } heap_;
    };
};

// Specialize std::hash and nx_hash/nx_equal for NxStr
namespace std {
template <> struct hash<NxStr> {
    std::size_t operator()(const NxStr &s) const noexcept {
        return std::hash<std::string_view>{}(s.view());
    }
};
} // namespace std

template <typename T> struct nx_hash : std::hash<T> {};
template <typename T> struct nx_equal : std::equal_to<T> {};
template <> struct nx_hash<const char *> {
    std::size_t operator()(const char *value) const { return std::hash<std::string_view>{}(value ? value : ""); }
};
template <> struct nx_equal<const char *> {
    bool operator()(const char *a, const char *b) const { return std::strcmp(a, b) == 0; }
};
template <> struct nx_hash<NxStr> {
    std::size_t operator()(const NxStr &v) const noexcept { return std::hash<std::string_view>{}(v.view()); }
};
template <> struct nx_equal<NxStr> {
    bool operator()(const NxStr &a, const NxStr &b) const noexcept { return a == b; }
};

// ============================================================================
// 6. String Formatting & NxStrBuilder (Eliminates allocation chains!)
// ============================================================================
static inline std::string nx_to_std_string(const NxStr &v) { return std::string(v.view()); }
static inline std::string nx_to_std_string(const char *v) { return v ? std::string(v) : std::string(); }
static inline std::string nx_to_std_string(const std::string &v) { return v; }
static inline std::string nx_to_std_string(bool v) { return v ? "true" : "false"; }
static inline std::string nx_to_std_string(nx_i128 v) { return nx_i128_to_std_string(v); }
static inline std::string nx_to_std_string(nx_u128 v) { return nx_u128_to_std_string(v); }
static inline std::string nx_to_std_string(std::int8_t v) { return std::to_string(static_cast<int>(v)); }
static inline std::string nx_to_std_string(std::uint8_t v) { return std::to_string(static_cast<unsigned>(v)); }
static inline std::string nx_to_std_string(std::int16_t v) { return std::to_string(static_cast<int>(v)); }
static inline std::string nx_to_std_string(std::uint16_t v) { return std::to_string(static_cast<unsigned>(v)); }
static inline std::string nx_to_std_string(std::int32_t v) { return std::to_string(v); }
static inline std::string nx_to_std_string(std::uint32_t v) { return std::to_string(v); }
static inline std::string nx_to_std_string(std::int64_t v) { return std::to_string(v); }
static inline std::string nx_to_std_string(std::uint64_t v) { return std::to_string(v); }
static inline std::string nx_to_std_string(float v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    return std::string(buf);
}
static inline std::string nx_to_std_string(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}
template <typename T>
static inline auto nx_to_std_string(const std::optional<T> &opt) -> std::string {
    if (!opt.has_value()) return "None";
    return nx_to_std_string(*opt);
}
template <typename T>
static inline auto nx_to_std_string(const std::unique_ptr<T> &ptr) -> std::string {
    if (!ptr) return "None";
    return nx_to_std_string(*ptr);
}
template <typename T>
static inline auto nx_to_std_string(const T &obj) -> decltype(obj.nx___str__().c_str(), std::string()) {
    return std::string(obj.nx___str__().view());
}

static inline std::string nx_format_apply_spec(const std::string &raw, const char *spec, bool is_num, double fval, nx_i128 ival, bool is_float) {
    if (!spec || !*spec) return raw;
    std::string sp(spec);
    char align = 0;
    char fill = ' ';
    std::size_t idx = 0;
    if (sp.size() >= 2 && (sp[1] == '<' || sp[1] == '>' || sp[1] == '^')) {
        fill = sp[0]; align = sp[1]; idx = 2;
    } else if (!sp.empty() && (sp[0] == '<' || sp[0] == '>' || sp[0] == '^')) {
        align = sp[0]; idx = 1;
    }
    bool zero_pad = false;
    if (idx < sp.size() && sp[idx] == '0' && align == 0) {
        zero_pad = true; fill = '0'; align = '>'; ++idx;
    }
    std::size_t width = 0;
    while (idx < sp.size() && std::isdigit(static_cast<unsigned char>(sp[idx]))) {
        width = width * 10 + static_cast<std::size_t>(sp[idx] - '0');
        ++idx;
    }
    bool comma = false;
    if (idx < sp.size() && sp[idx] == ',') { comma = true; ++idx; }
    int precision = -1;
    if (idx < sp.size() && sp[idx] == '.') {
        ++idx;
        precision = 0;
        while (idx < sp.size() && std::isdigit(static_cast<unsigned char>(sp[idx]))) {
            precision = precision * 10 + (sp[idx] - '0');
            ++idx;
        }
    }
    char kind = (idx < sp.size()) ? sp[idx] : '\0';
    std::string body = raw;
    if (is_num) {
        if (kind == 'f' || kind == 'F' || kind == 'e' || kind == 'E' || kind == 'g' || kind == '%') {
            int p = precision >= 0 ? precision : 6;
            double v = (kind == '%') ? fval * 100.0 : fval;
            char fmt_buf[32];
            char fchar = (kind == '%') ? 'f' : kind;
            std::snprintf(fmt_buf, sizeof(fmt_buf), "%%.%d%c", p, fchar);
            char out[128];
            std::snprintf(out, sizeof(out), fmt_buf, v);
            body = out;
            if (kind == '%') body.push_back('%');
        } else if (kind == 'x' || kind == 'X' || kind == 'o' || kind == 'b') {
            bool neg = ival < 0;
            nx_u128 mag = neg ? (static_cast<nx_u128>(-(ival + 1)) + 1) : static_cast<nx_u128>(ival);
            unsigned base = (kind == 'o') ? 8u : (kind == 'b') ? 2u : 16u;
            const char *digits = (kind == 'X') ? "0123456789ABCDEF" : "0123456789abcdef";
            if (mag == 0) body = "0";
            else {
                std::string tmp;
                while (mag > 0) { tmp.push_back(digits[mag % base]); mag /= base; }
                std::reverse(tmp.begin(), tmp.end());
                body = neg ? ("-" + tmp) : tmp;
            }
        } else if (precision >= 0 && is_float) {
            char fmt_buf[32];
            std::snprintf(fmt_buf, sizeof(fmt_buf), "%%.%df", precision);
            char out[128];
            std::snprintf(out, sizeof(out), fmt_buf, fval);
            body = out;
        }
        if (comma) {
            auto dot = body.find('.');
            std::size_t end_i = (dot == std::string::npos) ? body.size() : dot;
            std::size_t start_i = (!body.empty() && body[0] == '-') ? 1 : 0;
            std::string grouped;
            int cnt = 0;
            for (std::size_t k = end_i; k > start_i; --k) {
                if (cnt && cnt % 3 == 0) grouped.push_back(',');
                grouped.push_back(body[k - 1]);
                ++cnt;
            }
            std::reverse(grouped.begin(), grouped.end());
            body = body.substr(0, start_i) + grouped + body.substr(end_i);
        }
    }
    if (width > body.size()) {
        std::size_t pad = width - body.size();
        if (zero_pad && !body.empty() && body[0] == '-') {
            body = "-" + std::string(pad, '0') + body.substr(1);
        } else if (align == '<' || (!align && !is_num)) {
            body.append(pad, fill);
        } else if (align == '^') {
            std::size_t left = pad / 2, right = pad - left;
            body = std::string(left, fill) + body + std::string(right, fill);
        } else {
            body = std::string(pad, fill) + body;
        }
    }
    return body;
}

template <typename T>
static inline std::string nx_format_value(const T &val, const char *spec) {
    if (!spec || !*spec) return nx_to_std_string(val);
    if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
        return nx_format_apply_spec(nx_to_std_string(val), spec, false, 0.0, 0, false);
    } else if constexpr (nx_is_numeric<T>::value) {
        bool is_flt = std::is_floating_point_v<std::decay_t<T>>;
        double fv = static_cast<double>(val);
        nx_i128 iv = is_flt ? static_cast<nx_i128>(fv) : static_cast<nx_i128>(val);
        return nx_format_apply_spec(nx_to_std_string(val), spec, true, fv, iv, is_flt);
    } else {
        return nx_format_apply_spec(nx_to_std_string(val), spec, false, 0.0, 0, false);
    }
}

class NxStrBuilder {
public:
    NxStrBuilder() = default;
    explicit NxStrBuilder(std::size_t cap) { buf_.reserve(cap); }

    NxStrBuilder &append(const NxStr &s) { buf_.append(s.c_str(), s.size()); return *this; }
    NxStrBuilder &append(const char *s) { if (s) buf_.append(s); return *this; }
    NxStrBuilder &append(const std::string &s) { buf_.append(s); return *this; }
    NxStrBuilder &append(std::string_view s) { buf_.append(s.data(), s.size()); return *this; }
    template <typename T>
    NxStrBuilder &append(const T &v) { buf_.append(nx_to_std_string(v)); return *this; }

    template <typename T>
    NxStrBuilder &append_fmt(const T &v, const char *spec) {
        buf_.append(nx_format_value(v, spec));
        return *this;
    }

    NxStrBuilder &append_line(const NxStr &s) { append(s); buf_.push_back('\n'); return *this; }
    template <typename T>
    NxStrBuilder &append_line(const T &v) { append(v); buf_.push_back('\n'); return *this; }

    void reserve(std::int64_t cap) { if (cap > 0) buf_.reserve(static_cast<std::size_t>(cap)); }
    void clear() noexcept { buf_.clear(); }
    std::int64_t size() const noexcept { return static_cast<std::int64_t>(buf_.size()); }
    NxStr to_str() const { return NxStr(buf_.data(), buf_.size(), false); }
    NxStr finish() {
        NxStr out(buf_.data(), buf_.size(), false);
        return out;
    }

private:
    std::string buf_;
};

static inline std::string nx_to_std_string(const NxStrBuilder &sb) {
    return std::string(sb.to_str().view());
}

static inline NxStr nx_string_slice(const NxStr &s, std::int64_t start, std::int64_t stop,
                                    std::int64_t step, bool has_start, bool has_stop) {
    if (step == 0) nx_fail("slice step cannot be zero");
    const std::int64_t n = static_cast<std::int64_t>(s.size());
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
    const char *p = s.c_str();
    std::string out;
    if (forward) {
        if (step == 1 && stop > start) return NxStr(p + start, static_cast<std::size_t>(stop - start), false);
        for (auto i = start; i < stop; i += step) out.push_back(p[i]);
    } else {
        for (auto i = start; i > stop; i += step) out.push_back(p[i]);
    }
    return NxStr(out.data(), out.size(), false);
}

// ============================================================================
// 7. Spans, Buffers, Collections & Result/Optional for Arbitrary Classes/Enums
// ============================================================================
static std::size_t nx_bounds(std::int64_t index, std::size_t len, bool allow_negative = false) {
    if (allow_negative && index < 0) index += static_cast<std::int64_t>(len);
    if (index < 0 || static_cast<std::uint64_t>(index) >= len) {
        std::fprintf(stderr, "NEXIR index out of bounds: %lld (length %zu)\n",
                     static_cast<long long>(index), len);
        nx_dump_trace();
        std::exit(1);
    }
    return static_cast<std::size_t>(index);
}

struct nx_ok_tag {};
struct nx_err_tag {};

template <typename T>
struct nx_result {
    std::variant<T, NxStr> state;
    nx_result(nx_ok_tag, T value) : state(std::in_place_index<0>, std::move(value)) {}
    nx_result(nx_err_tag, NxStr error) : state(std::in_place_index<1>, std::move(error)) {}
    bool ok() const noexcept { return state.index() == 0; }
    bool is_err() const noexcept { return state.index() == 1; }
    T &value_ref() { if (!ok()) nx_fail_str("unwrap of error Result: " + std::string(error().view())); return std::get<0>(state); }
    const T &value_ref() const { if (!ok()) nx_fail_str("unwrap of error Result: " + std::string(error().view())); return std::get<0>(state); }
    T value() { if (!ok()) nx_fail_str("unwrap of error Result: " + std::string(error().view())); return std::move(std::get<0>(state)); }
    T value_or(T fallback) { return ok() ? std::move(std::get<0>(state)) : std::move(fallback); }
    NxStr error() const { if (ok()) nx_fail("error of successful Result"); return std::get<1>(state); }
};

struct nx_result_void {
    bool is_ok_val = true;
    NxStr err_msg;
    nx_result_void() = default;
    explicit nx_result_void(NxStr err) : is_ok_val(false), err_msg(std::move(err)) {}
    bool ok() const noexcept { return is_ok_val; }
    bool is_err() const noexcept { return !is_ok_val; }
    void value() const { if (!is_ok_val) nx_fail_str("unwrap of error Result: " + std::string(err_msg.view())); }
    NxStr error() const { if (is_ok_val) nx_fail("error of successful Result"); return err_msg; }
};

template <typename T, typename E>
struct nx_result_err {
    std::variant<T, E> state;
    nx_result_err(nx_ok_tag, T value) : state(std::in_place_index<0>, std::move(value)) {}
    nx_result_err(nx_err_tag, E error) : state(std::in_place_index<1>, std::move(error)) {}
    bool ok() const noexcept { return state.index() == 0; }
    bool is_err() const noexcept { return state.index() == 1; }
    T value() { if (!ok()) nx_fail("unwrap of typed error Result"); return std::move(std::get<0>(state)); }
    T value_or(T fallback) { return ok() ? std::move(std::get<0>(state)) : std::move(fallback); }
    E error() const { if (ok()) nx_fail("error of successful Result"); return std::get<1>(state); }
};

template <typename T> static nx_result<T> nx_rt_ok(T value) { return nx_result<T>(nx_ok_tag{}, std::move(value)); }
static inline nx_result_void nx_rt_ok_void() { return nx_result_void(); }
template <typename T> static nx_result<T> nx_rt_err(const NxStr &message) { return nx_result<T>(nx_err_tag{}, message); }
template <typename T> static nx_result<T> nx_rt_err(const char *message) { return nx_result<T>(nx_err_tag{}, NxStr(message)); }
static inline nx_result_void nx_rt_err_void(const NxStr &message) { return nx_result_void(message); }
template <typename T, typename E> static nx_result_err<T, E> nx_rt_ok_e(T value) { return nx_result_err<T, E>(nx_ok_tag{}, std::move(value)); }
template <typename T, typename E> static nx_result_err<T, E> nx_rt_err_e(E err) { return nx_result_err<T, E>(nx_err_tag{}, std::move(err)); }

template <typename T> static nx_result<T> nx_ok(T value) { return nx_rt_ok<T>(std::move(value)); }
static inline nx_result_void nx_ok_void() { return nx_rt_ok_void(); }
template <typename T> static nx_result<T> nx_err(const NxStr &message) { return nx_rt_err<T>(message); }
template <typename T> static nx_result<T> nx_err(const char *message) { return nx_rt_err<T>(message); }
static inline nx_result_void nx_err_void(const NxStr &message) { return nx_rt_err_void(message); }
template <typename T, typename E> static nx_result_err<T, E> nx_ok_e(T value) { return nx_rt_ok_e<T, E>(std::move(value)); }
template <typename T, typename E> static nx_result_err<T, E> nx_err_e(E err) { return nx_rt_err_e<T, E>(std::move(err)); }

struct nx_span_i8 { const std::int8_t *ptr; std::size_t len; };
struct nx_mutspan_i8 { std::int8_t *ptr; std::size_t len; };
struct nx_span_i16 { const std::int16_t *ptr; std::size_t len; };
struct nx_mutspan_i16 { std::int16_t *ptr; std::size_t len; };
struct nx_span_i32 { const std::int32_t *ptr; std::size_t len; };
struct nx_mutspan_i32 { std::int32_t *ptr; std::size_t len; };
struct nx_span_i64 { const std::int64_t *ptr; std::size_t len; };
struct nx_mutspan_i64 { std::int64_t *ptr; std::size_t len; };
struct nx_span_u8 { const std::uint8_t *ptr; std::size_t len; };
struct nx_mutspan_u8 { std::uint8_t *ptr; std::size_t len; };
struct nx_span_u16 { const std::uint16_t *ptr; std::size_t len; };
struct nx_mutspan_u16 { std::uint16_t *ptr; std::size_t len; };
struct nx_span_u32 { const std::uint32_t *ptr; std::size_t len; };
struct nx_mutspan_u32 { std::uint32_t *ptr; std::size_t len; };
struct nx_span_u64 { const std::uint64_t *ptr; std::size_t len; };
struct nx_mutspan_u64 { std::uint64_t *ptr; std::size_t len; };
struct nx_span_i128 { const nx_i128 *ptr; std::size_t len; };
struct nx_mutspan_i128 { nx_i128 *ptr; std::size_t len; };
struct nx_span_u128 { const nx_u128 *ptr; std::size_t len; };
struct nx_mutspan_u128 { nx_u128 *ptr; std::size_t len; };
struct nx_span_f32 { const float *ptr; std::size_t len; };
struct nx_mutspan_f32 { float *ptr; std::size_t len; };
struct nx_span_f64 { const double *ptr; std::size_t len; };
struct nx_mutspan_f64 { double *ptr; std::size_t len; };

template <typename T>
using nx_vec = std::vector<T, NxAllocator<T>>;

template <typename K, typename V, typename H = nx_hash<K>, typename E = nx_equal<K>>
using nx_map = std::unordered_map<K, V, H, E, NxAllocator<std::pair<const K, V>>>;

template <typename T, typename H = nx_hash<T>, typename E = nx_equal<T>>
using nx_set = std::unordered_set<T, H, E, NxAllocator<T>>;

template <typename T> class nx_buffer {
    nx_vec<T> storage_;
public:
    explicit nx_buffer(std::int64_t count) {
        if (count < 0) nx_fail("negative Buffer length");
        storage_.resize(static_cast<std::size_t>(count));
    }
    std::size_t size() const noexcept { return storage_.size(); }
    T &at(std::int64_t index) { return storage_[nx_bounds(index, storage_.size(), false)]; }
    const T &at(std::int64_t index) const { return storage_[nx_bounds(index, storage_.size(), false)]; }
    T *data() noexcept { return storage_.data(); }
    const T *data() const noexcept { return storage_.data(); }
    nx_vec<T> &raw() noexcept { return storage_; }
};

template <typename T> static T &nx_list_at(nx_vec<T> &values, std::int64_t index) {
    return values[nx_bounds(index, values.size(), true)];
}
template <typename T> static const T &nx_list_at(const nx_vec<T> &values, std::int64_t index) {
    return values[nx_bounds(index, values.size(), true)];
}
template <typename T> static T nx_list_pop(nx_vec<T> &values, std::int64_t index = -1) {
    if (values.empty()) nx_fail("pop from empty List");
    std::size_t idx = nx_bounds(index, values.size(), true);
    T result = std::move(values[idx]);
    values.erase(values.begin() + static_cast<std::ptrdiff_t>(idx));
    return result;
}
template <typename T> static void nx_list_insert(nx_vec<T> &values, std::int64_t index, T val) {
    std::int64_t n = static_cast<std::int64_t>(values.size());
    if (index < 0) index += n;
    if (index < 0) index = 0;
    if (index > n) index = n;
    values.insert(values.begin() + static_cast<std::ptrdiff_t>(index), std::move(val));
}
template <typename T> static void nx_list_extend(nx_vec<T> &values, const nx_vec<T> &other) {
    values.insert(values.end(), other.begin(), other.end());
}
template <typename T> static void nx_list_remove(nx_vec<T> &values, const T &val) {
    auto it = std::find(values.begin(), values.end(), val);
    if (it == values.end()) nx_fail("List.remove(x): x not in list");
    values.erase(it);
}
template <typename T> static std::int64_t nx_list_index(const nx_vec<T> &values, const T &val) {
    auto it = std::find(values.begin(), values.end(), val);
    if (it == values.end()) nx_fail("List.index(x): x not in list");
    return static_cast<std::int64_t>(it - values.begin());
}
template <typename T> static std::int64_t nx_list_count(const nx_vec<T> &values, const T &val) {
    return static_cast<std::int64_t>(std::count(values.begin(), values.end(), val));
}
template <typename T> static void nx_list_reverse(nx_vec<T> &values) {
    std::reverse(values.begin(), values.end());
}
template <typename T> static void nx_list_sort(nx_vec<T> &values, bool rev = false) {
    if (rev) std::sort(values.begin(), values.end(), std::greater<T>());
    else std::sort(values.begin(), values.end());
}
template <typename T> static nx_vec<T> nx_list_sorted(nx_vec<T> values, bool rev = false) {
    nx_list_sort(values, rev);
    return values;
}
template <typename T> static nx_vec<T> nx_list_repeat(const nx_vec<T> &values, std::int64_t n) {
    nx_vec<T> out;
    if (n <= 0 || values.empty()) return out;
    out.reserve(values.size() * static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) out.insert(out.end(), values.begin(), values.end());
    return out;
}
template <typename T> static nx_vec<T> nx_list_concat(const nx_vec<T> &a, const nx_vec<T> &b) {
    nx_vec<T> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}
template <typename T> static bool nx_list_contains(const nx_vec<T> &values, const T &value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}
template <typename T> static bool nx_span_contains(const T *values, std::size_t count, T value) {
    if (count == 0) return false;
    return std::find(values, values + count, value) != values + count;
}
template <typename T> static nx_vec<T> nx_slice(const T *values, std::size_t count,
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
    nx_vec<T> result;
    if (forward) for (auto i = start; i < stop; i += step) result.push_back(values[i]);
    else for (auto i = start; i > stop; i += step) result.push_back(values[i]);
    return result;
}
static inline nx_vec<NxStr> nx_string_split(const NxStr &text, const NxStr &delimiter = NxStr(" ")) {
    auto sv = text.view();
    auto dv = delimiter.view();
    if (dv.empty()) nx_fail("split requires a nonempty separator");
    nx_vec<NxStr> out;
    std::size_t start = 0, pos;
    while ((pos = sv.find(dv, start)) != std::string_view::npos) {
        out.push_back(NxStr(sv.data() + start, pos - start, false));
        start = pos + dv.size();
    }
    out.push_back(NxStr(sv.data() + start, sv.size() - start, false));
    return out;
}
static inline NxStr nx_string_join(const NxStr &sep, const nx_vec<NxStr> &items) {
    if (items.empty()) return NxStr();
    std::size_t est = sep.size() * (items.size() - 1);
    for (const auto &it : items) est += it.size();
    NxStrBuilder sb(est);
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) sb.append(sep);
        sb.append(items[i]);
    }
    return sb.finish();
}

template <typename K, typename V, typename H, typename E, typename Q>
static V &nx_dict_at(nx_map<K, V, H, E> &values, Q &&key) {
    K converted = std::forward<Q>(key);
    auto found = values.find(converted);
    if (found == values.end()) nx_fail("Dict key not found");
    return found->second;
}
template <typename K, typename V, typename H, typename E, typename Q>
static V nx_dict_get(const nx_map<K, V, H, E> &values, Q &&key, V fallback) {
    K converted = std::forward<Q>(key);
    auto found = values.find(converted);
    return found == values.end() ? std::move(fallback) : found->second;
}
template <typename K, typename V, typename H, typename E, typename Q>
static V nx_dict_pop(nx_map<K, V, H, E> &values, Q &&key) {
    K converted = std::forward<Q>(key);
    auto found = values.find(converted);
    if (found == values.end()) nx_fail("Dict.pop: key not found");
    V out = std::move(found->second);
    values.erase(found);
    return out;
}
template <typename K, typename V, typename H, typename E, typename Q>
static V nx_dict_setdefault(nx_map<K, V, H, E> &values, Q &&key, V fallback) {
    K converted = std::forward<Q>(key);
    auto [it, inserted] = values.emplace(std::move(converted), std::move(fallback));
    return it->second;
}
template <typename K, typename V, typename H, typename E>
static nx_vec<K> nx_dict_keys(const nx_map<K, V, H, E> &values) {
    nx_vec<K> out;
    out.reserve(values.size());
    for (const auto &kv : values) out.push_back(kv.first);
    return out;
}
template <typename K, typename V, typename H, typename E>
static nx_vec<V> nx_dict_values(const nx_map<K, V, H, E> &values) {
    nx_vec<V> out;
    out.reserve(values.size());
    for (const auto &kv : values) out.push_back(kv.second);
    return out;
}
template <typename K, typename V, typename H, typename E>
static nx_vec<std::tuple<K, V>> nx_dict_items(const nx_map<K, V, H, E> &values) {
    nx_vec<std::tuple<K, V>> out;
    out.reserve(values.size());
    for (const auto &kv : values) out.emplace_back(kv.first, kv.second);
    return out;
}
template <typename K, typename V, typename H, typename E>
static void nx_dict_update(nx_map<K, V, H, E> &dst, const nx_map<K, V, H, E> &src) {
    for (const auto &kv : src) dst[kv.first] = kv.second;
}
template <typename T, typename H, typename E, typename Q>
static void nx_set_remove(nx_set<T, H, E> &values, Q &&value) {
    T converted = std::forward<Q>(value);
    if (!values.erase(converted)) nx_fail("Set element not found");
}
template <typename T, typename H, typename E, typename Q>
static void nx_set_discard(nx_set<T, H, E> &values, Q &&value) {
    T converted = std::forward<Q>(value);
    values.erase(converted);
}

// Span / Buffer helpers (RAII-friendly!)
static inline nx_span_i8 nx_as_span_i8(nx_mutspan_i8 s) { return {s.ptr, s.len}; }
static inline nx_span_i16 nx_as_span_i16(nx_mutspan_i16 s) { return {s.ptr, s.len}; }
static inline nx_span_i32 nx_as_span_i32(nx_mutspan_i32 s) { return {s.ptr, s.len}; }
static inline nx_span_i64 nx_as_span_i64(nx_mutspan_i64 s) { return {s.ptr, s.len}; }
static inline nx_span_u8 nx_as_span_u8(nx_mutspan_u8 s) { return {s.ptr, s.len}; }
static inline nx_span_u16 nx_as_span_u16(nx_mutspan_u16 s) { return {s.ptr, s.len}; }
static inline nx_span_u32 nx_as_span_u32(nx_mutspan_u32 s) { return {s.ptr, s.len}; }
static inline nx_span_u64 nx_as_span_u64(nx_mutspan_u64 s) { return {s.ptr, s.len}; }
static inline nx_span_i128 nx_as_span_i128(nx_mutspan_i128 s) { return {s.ptr, s.len}; }
static inline nx_span_u128 nx_as_span_u128(nx_mutspan_u128 s) { return {s.ptr, s.len}; }
static inline nx_span_f32 nx_as_span_f32(nx_mutspan_f32 s) { return {s.ptr, s.len}; }
static inline nx_span_f64 nx_as_span_f64(nx_mutspan_f64 s) { return {s.ptr, s.len}; }

template <typename T> static inline nx_span_i32 nx_vec_span_i32(const nx_vec<T> &v) { return {v.data(), v.size()}; }
static inline nx_span_i8 nx_vec_span(const nx_vec<std::int8_t> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_i8 nx_vec_mutspan(nx_vec<std::int8_t> &v) { return {v.data(), v.size()}; }
static inline nx_span_i16 nx_vec_span(const nx_vec<std::int16_t> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_i16 nx_vec_mutspan(nx_vec<std::int16_t> &v) { return {v.data(), v.size()}; }
static inline nx_span_i32 nx_vec_span(const nx_vec<std::int32_t> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_i32 nx_vec_mutspan(nx_vec<std::int32_t> &v) { return {v.data(), v.size()}; }
static inline nx_span_i64 nx_vec_span(const nx_vec<std::int64_t> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_i64 nx_vec_mutspan(nx_vec<std::int64_t> &v) { return {v.data(), v.size()}; }
static inline nx_span_u8 nx_vec_span(const nx_vec<std::uint8_t> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_u8 nx_vec_mutspan(nx_vec<std::uint8_t> &v) { return {v.data(), v.size()}; }
static inline nx_span_u16 nx_vec_span(const nx_vec<std::uint16_t> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_u16 nx_vec_mutspan(nx_vec<std::uint16_t> &v) { return {v.data(), v.size()}; }
static inline nx_span_u32 nx_vec_span(const nx_vec<std::uint32_t> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_u32 nx_vec_mutspan(nx_vec<std::uint32_t> &v) { return {v.data(), v.size()}; }
static inline nx_span_u64 nx_vec_span(const nx_vec<std::uint64_t> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_u64 nx_vec_mutspan(nx_vec<std::uint64_t> &v) { return {v.data(), v.size()}; }
static inline nx_span_i128 nx_vec_span(const nx_vec<nx_i128> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_i128 nx_vec_mutspan(nx_vec<nx_i128> &v) { return {v.data(), v.size()}; }
static inline nx_span_u128 nx_vec_span(const nx_vec<nx_u128> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_u128 nx_vec_mutspan(nx_vec<nx_u128> &v) { return {v.data(), v.size()}; }
static inline nx_span_f32 nx_vec_span(const nx_vec<float> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_f32 nx_vec_mutspan(nx_vec<float> &v) { return {v.data(), v.size()}; }
static inline nx_span_f64 nx_vec_span(const nx_vec<double> &v) { return {v.data(), v.size()}; }
static inline nx_mutspan_f64 nx_vec_mutspan(nx_vec<double> &v) { return {v.data(), v.size()}; }

template <typename SpanT> static inline auto nx_span_at(SpanT s, std::int64_t i) -> decltype(&s.ptr[0]) {
    return &s.ptr[nx_bounds(i, s.len, false)];
}

static inline nx_vec<float> nx_zeros_f32(std::int64_t n) {
    if (n < 0) nx_fail("negative array length");
    return nx_vec<float>(static_cast<std::size_t>(n), 0.0f);
}
static inline nx_vec<double> nx_zeros_f64(std::int64_t n) {
    if (n < 0) nx_fail("negative array length");
    return nx_vec<double>(static_cast<std::size_t>(n), 0.0);
}
static inline nx_vec<std::int64_t> nx_zeros_i64(std::int64_t n) {
    if (n < 0) nx_fail("negative array length");
    return nx_vec<std::int64_t>(static_cast<std::size_t>(n), 0);
}
static inline nx_vec<std::int32_t> nx_zeros_i32(std::int64_t n) {
    if (n < 0) nx_fail("negative array length");
    return nx_vec<std::int32_t>(static_cast<std::size_t>(n), 0);
}

static inline NxStr nx_adopt_py_string(char *raw) {
    if (!raw) nx_fail("Python string call failed");
    std::unique_ptr<char, void(*)(void*)> allocation(raw, std::free);
    return NxStr(allocation.get(), std::strlen(allocation.get()), false);
}
static inline std::unique_ptr<nx_buffer<double>> nx_buffer_from_py(nx_mutspan_f64 raw) {
    if (raw.len == std::numeric_limits<std::size_t>::max() ||
        raw.len > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        (raw.len && !raw.ptr)) nx_fail("Python buffer call failed");
    std::unique_ptr<double, void(*)(void*)> allocation(raw.ptr, std::free);
    auto owner = std::make_unique<nx_buffer<double>>(static_cast<std::int64_t>(raw.len));
    if (raw.len) std::copy(allocation.get(), allocation.get() + raw.len, owner->data());
    return owner;
}

// Pythonic builtin aggregations (sum, min, max, abs, round, any, all)
template <typename T> static inline T nx_builtin_sum(const nx_vec<T> &v) {
    T acc{};
    for (const auto &x : v) acc += x;
    return acc;
}
template <typename SpanT> static inline auto nx_builtin_sum_span(SpanT s) {
    using Elem = std::remove_const_t<std::remove_pointer_t<decltype(s.ptr)>>;
    Elem acc{};
    for (std::size_t i = 0; i < s.len; ++i) acc += s.ptr[i];
    return acc;
}
template <typename T> static inline T nx_builtin_min(const nx_vec<T> &v) {
    if (v.empty()) nx_fail("min() arg is an empty sequence");
    return *std::min_element(v.begin(), v.end());
}
template <typename T> static inline T nx_builtin_max(const nx_vec<T> &v) {
    if (v.empty()) nx_fail("max() arg is an empty sequence");
    return *std::max_element(v.begin(), v.end());
}
template <typename T> static inline T nx_builtin_abs(T v) {
    if constexpr (std::is_unsigned_v<T> || std::is_same_v<T, nx_u128>) return v;
    else return v < 0 ? -v : v;
}
static inline bool nx_builtin_any(const nx_vec<bool> &v) {
    for (bool b : v) if (b) return true;
    return false;
}
static inline bool nx_builtin_all(const nx_vec<bool> &v) {
    for (bool b : v) if (!b) return false;
    return true;
}

// ============================================================================
// 8. Hybrid Dynamic Value (NxValue v2) - 16 bytes, Cache-managed, Zero-leak!
//    type tags:
//      0: null/None
//      1: i64
//      2: f64
//      3: str (heap NxStr)
//      4: object (shared_ptr<void>)
//      5: bool
//      6: i128 (heap nx_i128)
//      7: list (heap nx_vec<NxValue>)
//      8: dict (heap nx_map<NxStr, NxValue>)
//      9: pyobj (void* PyObject* with custom deleter)
// ============================================================================
struct nx_heap {
    std::atomic<std::uint32_t> references{1};
    std::uint32_t kind{0};
    std::shared_ptr<void> object;
    explicit nx_heap(std::shared_ptr<void> value, std::uint32_t k = 0) : kind(k), object(std::move(value)) {}
};

struct alignas(16) NxValue {
    std::uint32_t type = 0;
    std::uint32_t reserved = 0;
    union payload { std::int64_t i; double f; nx_heap *ptr; payload() : i(0) {} } data;

    NxValue() = default;
    NxValue(const NxValue &other) : type(other.type), reserved(other.reserved), data() {
        data = other.data;
        retain();
    }
    NxValue(NxValue &&other) noexcept : type(other.type), reserved(other.reserved), data() {
        data = other.data;
        other.type = 0;
        other.reserved = 0;
        other.data.i = 0;
    }
    NxValue &operator=(const NxValue &other) {
        if (this != &other) { release(); type = other.type; reserved = other.reserved; data = other.data; retain(); }
        return *this;
    }
    NxValue &operator=(NxValue &&other) noexcept {
        if (this != &other) {
            release(); type = other.type; reserved = other.reserved; data = other.data;
            other.type = 0; other.reserved = 0; other.data.i = 0;
        }
        return *this;
    }
    ~NxValue() { release(); }

    static bool is_heap_type(std::uint32_t t) noexcept { return t == 3 || t == 4 || t >= 6; }
    void retain() noexcept {
        if (is_heap_type(type) && data.ptr) data.ptr->references.fetch_add(1, std::memory_order_relaxed);
    }
    void release() noexcept {
        if (is_heap_type(type) && data.ptr && data.ptr->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete data.ptr;
        }
        data.ptr = nullptr;
    }

    static NxValue null() noexcept { return NxValue(); }
    static NxValue boolean(bool b) noexcept { NxValue v; v.type = 5; v.data.i = b ? 1 : 0; return v; }
    static NxValue integer(std::int64_t n) noexcept { NxValue v; v.type = 1; v.data.i = n; return v; }
    static NxValue floating(double n) noexcept { NxValue v; v.type = 2; v.data.f = n; return v; }
    static NxValue int128(nx_i128 n) {
        if (n >= std::numeric_limits<std::int64_t>::min() && n <= std::numeric_limits<std::int64_t>::max())
            return integer(static_cast<std::int64_t>(n));
        NxValue v; v.type = 6;
        v.data.ptr = new nx_heap(std::make_shared<nx_i128>(n), 6);
        return v;
    }
    static NxValue string(const NxStr &s) {
        NxValue v; v.type = 3;
        v.data.ptr = new nx_heap(std::make_shared<NxStr>(s), 3);
        return v;
    }
    static NxValue string(const char *s) { return string(NxStr(s)); }
    static NxValue object(std::shared_ptr<void> owner) {
        NxValue v; v.type = 4; v.data.ptr = new nx_heap(std::move(owner), 4); return v;
    }
    static NxValue list(nx_vec<NxValue> items = {}) {
        NxValue v; v.type = 7;
        v.data.ptr = new nx_heap(std::make_shared<nx_vec<NxValue>>(std::move(items)), 7);
        return v;
    }
    static NxValue dict(nx_map<NxStr, NxValue> entries = {}) {
        NxValue v; v.type = 8;
        v.data.ptr = new nx_heap(std::make_shared<nx_map<NxStr, NxValue>>(std::move(entries)), 8);
        return v;
    }
    static NxValue pyobj(std::shared_ptr<void> holder) {
        NxValue v; v.type = 9;
        v.data.ptr = new nx_heap(std::move(holder), 9);
        return v;
    }
};
static_assert(sizeof(NxValue) == 16, "NxValue must be 16 bytes");
static_assert(alignof(NxValue) == 16, "NxValue must be 16-byte aligned");
static_assert(offsetof(NxValue, data) == 8, "payload offset must be 8");

static inline NxValue nx_box(std::nullptr_t) { return NxValue::null(); }
static inline NxValue nx_box(bool b) { return NxValue::boolean(b); }
static inline NxValue nx_box(std::int8_t n) { return NxValue::integer(n); }
static inline NxValue nx_box(std::int16_t n) { return NxValue::integer(n); }
static inline NxValue nx_box(std::int32_t n) { return NxValue::integer(n); }
static inline NxValue nx_box(std::int64_t n) { return NxValue::integer(n); }
static inline NxValue nx_box(std::uint8_t n) { return NxValue::integer(n); }
static inline NxValue nx_box(std::uint16_t n) { return NxValue::integer(n); }
static inline NxValue nx_box(std::uint32_t n) { return NxValue::integer(n); }
static inline NxValue nx_box(std::uint64_t n) { return NxValue::int128(static_cast<nx_i128>(n)); }
static inline NxValue nx_box(nx_i128 n) { return NxValue::int128(n); }
static inline NxValue nx_box(nx_u128 n) { return NxValue::int128(nx_num_cast<nx_i128>(n)); }
static inline NxValue nx_box(float n) { return NxValue::floating(n); }
static inline NxValue nx_box(double n) { return NxValue::floating(n); }
static inline NxValue nx_box(const char *s) { return NxValue::string(s); }
static inline NxValue nx_box(const NxStr &s) { return NxValue::string(s); }
static inline NxValue nx_box(const NxValue &v) { return v; }
template <typename T> static inline NxValue nx_box(const nx_vec<T> &items) {
    nx_vec<NxValue> boxed;
    boxed.reserve(items.size());
    for (const auto &it : items) boxed.push_back(nx_box(it));
    return NxValue::list(std::move(boxed));
}
template <typename V> static inline NxValue nx_box(const nx_map<NxStr, V> &entries) {
    nx_map<NxStr, NxValue> boxed;
    for (const auto &kv : entries) boxed[kv.first] = nx_box(kv.second);
    return NxValue::dict(std::move(boxed));
}

static inline NxStr nx_type_name(const NxValue &v) {
    switch (v.type) {
        case 0: return NxStr("None");
        case 1: return NxStr("i64");
        case 2: return NxStr("f64");
        case 3: return NxStr("str");
        case 4: return NxStr("object");
        case 5: return NxStr("bool");
        case 6: return NxStr("i128");
        case 7: return NxStr("list");
        case 8: return NxStr("dict");
        case 9: return NxStr("pyobj");
        default: return NxStr("invalid");
    }
}

static inline bool nx_as_bool(const NxValue &v) {
    switch (v.type) {
        case 0: return false;
        case 1: return v.data.i != 0;
        case 2: return v.data.f != 0.0;
        case 3: return !std::static_pointer_cast<NxStr>(v.data.ptr->object)->empty();
        case 5: return v.data.i != 0;
        case 6: return *std::static_pointer_cast<nx_i128>(v.data.ptr->object) != 0;
        case 7: return !std::static_pointer_cast<nx_vec<NxValue>>(v.data.ptr->object)->empty();
        case 8: return !std::static_pointer_cast<nx_map<NxStr, NxValue>>(v.data.ptr->object)->empty();
        default: return true;
    }
}
static inline std::int64_t nx_as_i64(const NxValue &v) {
    if (v.type == 1) return v.data.i;
    if (v.type == 5) return v.data.i;
    if (v.type == 6) return nx_num_cast<std::int64_t>(*std::static_pointer_cast<nx_i128>(v.data.ptr->object));
    if (v.type == 2) return static_cast<std::int64_t>(v.data.f);
    nx_fail("dynamic value is not i64");
}
static inline nx_i128 nx_as_i128(const NxValue &v) {
    if (v.type == 1 || v.type == 5) return static_cast<nx_i128>(v.data.i);
    if (v.type == 6) return *std::static_pointer_cast<nx_i128>(v.data.ptr->object);
    if (v.type == 2) return static_cast<nx_i128>(v.data.f);
    nx_fail("dynamic value is not integer");
}
static inline double nx_as_f64(const NxValue &v) {
    if (v.type == 1 || v.type == 5) return static_cast<double>(v.data.i);
    if (v.type == 2) return v.data.f;
    if (v.type == 6) return static_cast<double>(*std::static_pointer_cast<nx_i128>(v.data.ptr->object));
    nx_fail("dynamic value is not numeric");
}
static inline NxStr nx_as_str(const NxValue &v) {
    if (v.type == 3) return *std::static_pointer_cast<NxStr>(v.data.ptr->object);
    if (v.type == 0) return NxStr("None");
    if (v.type == 1) return NxStr(std::to_string(v.data.i).c_str());
    if (v.type == 2) return NxStr(nx_to_std_string(v.data.f).c_str());
    if (v.type == 5) return NxStr(v.data.i ? "true" : "false");
    if (v.type == 6) return NxStr(nx_i128_to_std_string(*std::static_pointer_cast<nx_i128>(v.data.ptr->object)).c_str());
    if (v.type == 7) {
        const auto &lst = *std::static_pointer_cast<nx_vec<NxValue>>(v.data.ptr->object);
        NxStrBuilder sb;
        sb.append("[");
        for (std::size_t i = 0; i < lst.size(); ++i) {
            if (i) sb.append(", ");
            sb.append(nx_as_str(lst[i]));
        }
        sb.append("]");
        return sb.finish();
    }
    if (v.type == 8) {
        const auto &d = *std::static_pointer_cast<nx_map<NxStr, NxValue>>(v.data.ptr->object);
        NxStrBuilder sb;
        sb.append("{");
        bool first = true;
        for (const auto &kv : d) {
            if (!first) sb.append(", ");
            first = false;
            sb.append(kv.first).append(": ").append(nx_as_str(kv.second));
        }
        sb.append("}");
        return sb.finish();
    }
    if (v.type == 9) return NxStr("<pyobj>");
    return NxStr("<object>");
}

template <typename T>
static inline T nx_unbox(const NxValue &v) {
    using D = std::decay_t<T>;
    if constexpr (std::is_same_v<D, NxValue>) return v;
    else if constexpr (std::is_same_v<D, bool>) return nx_as_bool(v);
    else if constexpr (std::is_same_v<D, NxStr>) return nx_as_str(v);
    else if constexpr (std::is_floating_point_v<D>) return static_cast<D>(nx_as_f64(v));
    else if constexpr (nx_is_numeric<D>::value) return nx_num_cast<D>(nx_as_i128(v));
    else nx_fail("unsupported dynamic unbox type");
}

static inline std::string nx_to_std_string(const NxValue &v) {
    return std::string(nx_as_str(v).view());
}

static inline NxValue nx_dyn_add(const NxValue &a, const NxValue &b) {
    if (a.type == 1 && b.type == 1) {
        nx_i128 sum = static_cast<nx_i128>(a.data.i) + static_cast<nx_i128>(b.data.i);
        return NxValue::int128(sum);
    }
    if ((a.type == 1 || a.type == 6) && (b.type == 1 || b.type == 6)) {
        return NxValue::int128(nx_as_i128(a) + nx_as_i128(b));
    }
    if ((a.type == 1 || a.type == 2 || a.type == 6) && (b.type == 1 || b.type == 2 || b.type == 6)) {
        return NxValue::floating(nx_as_f64(a) + nx_as_f64(b));
    }
    if (a.type == 3 && b.type == 3) {
        return NxValue::string(*std::static_pointer_cast<NxStr>(a.data.ptr->object) +
                               *std::static_pointer_cast<NxStr>(b.data.ptr->object));
    }
    if (a.type == 7 && b.type == 7) {
        const auto &la = *std::static_pointer_cast<nx_vec<NxValue>>(a.data.ptr->object);
        const auto &lb = *std::static_pointer_cast<nx_vec<NxValue>>(b.data.ptr->object);
        return NxValue::list(nx_list_concat(la, lb));
    }
    nx_fail("unsupported dynamic + operands");
}
static inline NxValue nx_add_dyn(const NxValue &a, const NxValue &b) { return nx_dyn_add(a, b); }

static inline NxValue nx_dyn_sub(const NxValue &a, const NxValue &b) {
    if ((a.type == 1 || a.type == 6) && (b.type == 1 || b.type == 6))
        return NxValue::int128(nx_as_i128(a) - nx_as_i128(b));
    if ((a.type == 1 || a.type == 2 || a.type == 6) && (b.type == 1 || b.type == 2 || b.type == 6))
        return NxValue::floating(nx_as_f64(a) - nx_as_f64(b));
    nx_fail("unsupported dynamic - operands");
}
static inline NxValue nx_dyn_mul(const NxValue &a, const NxValue &b) {
    if ((a.type == 1 || a.type == 6) && (b.type == 1 || b.type == 6))
        return NxValue::int128(nx_as_i128(a) * nx_as_i128(b));
    if ((a.type == 1 || a.type == 2 || a.type == 6) && (b.type == 1 || b.type == 2 || b.type == 6))
        return NxValue::floating(nx_as_f64(a) * nx_as_f64(b));
    if (a.type == 3 && (b.type == 1 || b.type == 6))
        return NxValue::string(std::static_pointer_cast<NxStr>(a.data.ptr->object)->repeat(nx_as_i64(b)));
    nx_fail("unsupported dynamic * operands");
}
static inline NxValue nx_dyn_div(const NxValue &a, const NxValue &b) {
    if ((a.type == 1 || a.type == 2 || a.type == 6) && (b.type == 1 || b.type == 2 || b.type == 6)) {
        double d = nx_as_f64(b);
        if (d == 0.0) nx_fail("dynamic division by zero");
        return NxValue::floating(nx_as_f64(a) / d);
    }
    nx_fail("unsupported dynamic / operands");
}
static inline NxValue nx_dyn_mod(const NxValue &a, const NxValue &b) {
    if ((a.type == 1 || a.type == 6) && (b.type == 1 || b.type == 6)) {
        nx_i128 bv = nx_as_i128(b);
        if (bv == 0) nx_fail("dynamic modulo by zero");
        return NxValue::int128(nx_as_i128(a) % bv);
    }
    nx_fail("unsupported dynamic % operands");
}
static inline int nx_dyn_cmp(const NxValue &a, const NxValue &b) {
    if (a.type == 0 && b.type == 0) return 0;
    if (a.type == 3 && b.type == 3) {
        const auto &sa = *std::static_pointer_cast<NxStr>(a.data.ptr->object);
        const auto &sb = *std::static_pointer_cast<NxStr>(b.data.ptr->object);
        return sa == sb ? 0 : (sa < sb ? -1 : 1);
    }
    if ((a.type == 1 || a.type == 6) && (b.type == 1 || b.type == 6)) {
        auto ia = nx_as_i128(a), ib = nx_as_i128(b);
        return ia == ib ? 0 : (ia < ib ? -1 : 1);
    }
    if ((a.type == 1 || a.type == 2 || a.type == 6) && (b.type == 1 || b.type == 2 || b.type == 6)) {
        double fa = nx_as_f64(a), fb = nx_as_f64(b);
        return fa == fb ? 0 : (fa < fb ? -1 : 1);
    }
    if (a.type == 5 && b.type == 5) {
        return a.data.i == b.data.i ? 0 : (a.data.i < b.data.i ? -1 : 1);
    }
    return a.type == b.type ? 0 : (a.type < b.type ? -1 : 1);
}
static inline bool nx_dyn_eq(const NxValue &a, const NxValue &b) {
    if (a.type != b.type && !((a.type == 1 || a.type == 2 || a.type == 6) && (b.type == 1 || b.type == 2 || b.type == 6)))
        return false;
    return nx_dyn_cmp(a, b) == 0;
}
static inline std::int64_t nx_dyn_len(const NxValue &v) {
    if (v.type == 3) return static_cast<std::int64_t>(std::static_pointer_cast<NxStr>(v.data.ptr->object)->size());
    if (v.type == 7) return static_cast<std::int64_t>(std::static_pointer_cast<nx_vec<NxValue>>(v.data.ptr->object)->size());
    if (v.type == 8) return static_cast<std::int64_t>(std::static_pointer_cast<nx_map<NxStr, NxValue>>(v.data.ptr->object)->size());
    nx_fail("object of dynamic type has no len()");
}
static inline NxValue nx_dyn_getitem(const NxValue &container, const NxValue &key) {
    if (container.type == 7) {
        const auto &lst = *std::static_pointer_cast<nx_vec<NxValue>>(container.data.ptr->object);
        return nx_list_at(lst, nx_as_i64(key));
    }
    if (container.type == 8) {
        auto &d = *std::static_pointer_cast<nx_map<NxStr, NxValue>>(container.data.ptr->object);
        return nx_dict_at(d, nx_as_str(key));
    }
    if (container.type == 3) {
        return NxValue::string(std::static_pointer_cast<NxStr>(container.data.ptr->object)->at_char(nx_as_i64(key)));
    }
    nx_fail("dynamic object is not subscriptable");
}
static inline void nx_dyn_setitem(NxValue &container, const NxValue &key, const NxValue &val) {
    if (container.type == 7) {
        auto &lst = *std::static_pointer_cast<nx_vec<NxValue>>(container.data.ptr->object);
        nx_list_at(lst, nx_as_i64(key)) = val;
        return;
    }
    if (container.type == 8) {
        auto &d = *std::static_pointer_cast<nx_map<NxStr, NxValue>>(container.data.ptr->object);
        d[nx_as_str(key)] = val;
        return;
    }
    nx_fail("dynamic object does not support item assignment");
}
static inline void nx_dyn_append(NxValue &container, const NxValue &val) {
    if (container.type != 7) nx_fail("append requires dynamic list");
    std::static_pointer_cast<nx_vec<NxValue>>(container.data.ptr->object)->push_back(val);
}

static inline void nx_print_value(const NxValue &v) {
    if (v.type == 0) { std::fputs("None", stdout); return; }
    if (v.type == 1) { std::printf("%lld", static_cast<long long>(v.data.i)); return; }
    if (v.type == 2) { std::printf("%.17g", v.data.f); return; }
    if (v.type == 5) { std::fputs(v.data.i ? "true" : "false", stdout); return; }
    if (v.type == 6) { nx_print_i128(*std::static_pointer_cast<nx_i128>(v.data.ptr->object)); return; }
    std::fputs(nx_as_str(v).c_str(), stdout);
}
'''
