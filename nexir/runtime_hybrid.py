"""C++17 Hybrid Async Runtime (nx.asyncio), Bounded ThreadPool (nx.thread),
and Cross-Platform GPU/SIMD Workload Dispatch (nx.gpu).
"""

RUNTIME_HYBRID_CPP = r'''
// ============================================================================
// 12. Hybrid Multi-Core Async Runtime, ThreadPool, Channels & GPU Dispatch
//     Cross-platform: Windows (WSAPoll/LoadLibrary), macOS (kqueue/dlopen),
//     Linux (epoll/dlopen).
// ============================================================================

inline thread_local bool nx_tls_in_worker = false;
static bool nx_executor_try_step();

template <typename T>
class nx_Future {
    struct State {
        std::mutex mu;
        std::condition_variable cv;
        bool ready = false;
        bool has_err = false;
        NxStr err_msg;
        std::optional<T> val;
    };
    std::shared_ptr<State> state_;
public:
    nx_Future() : state_(std::make_shared<State>()) {}
    void set_value(T v) {
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->val = std::move(v);
        state_->ready = true;
        state_->cv.notify_all();
    }
    void set_error(const NxStr &msg) {
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->err_msg = msg;
        state_->has_err = true;
        state_->ready = true;
        state_->cv.notify_all();
    }
    bool is_done() const {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->ready;
    }
    T get() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(state_->mu);
                if (state_->ready) {
                    if (state_->has_err) nx_fail_str("async task failed: " + std::string(state_->err_msg.view()));
                    return std::move(*state_->val);
                }
                if (!nx_tls_in_worker) {
                    state_->cv.wait(lock, [this]() { return state_->ready; });
                    if (state_->has_err) nx_fail_str("async task failed: " + std::string(state_->err_msg.view()));
                    return std::move(*state_->val);
                }
            }
            // Work-stealing assist: never block a worker thread while subtasks wait in queue!
            if (!nx_executor_try_step()) {
                std::unique_lock<std::mutex> lock(state_->mu);
                state_->cv.wait_for(lock, std::chrono::milliseconds(1), [this]() { return state_->ready; });
            }
        }
    }
    nx_result<T> try_result() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(state_->mu);
                if (state_->ready) {
                    if (state_->has_err) return nx_err<T>(state_->err_msg);
                    return nx_ok(std::move(*state_->val));
                }
                if (!nx_tls_in_worker) {
                    state_->cv.wait(lock, [this]() { return state_->ready; });
                    if (state_->has_err) return nx_err<T>(state_->err_msg);
                    return nx_ok(std::move(*state_->val));
                }
            }
            if (!nx_executor_try_step()) {
                std::unique_lock<std::mutex> lock(state_->mu);
                state_->cv.wait_for(lock, std::chrono::milliseconds(1), [this]() { return state_->ready; });
            }
        }
    }
};

template <>
class nx_Future<void> {
    struct State {
        std::mutex mu;
        std::condition_variable cv;
        bool ready = false;
        bool has_err = false;
        NxStr err_msg;
    };
    std::shared_ptr<State> state_;
public:
    nx_Future() : state_(std::make_shared<State>()) {}
    void set_value() {
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->ready = true;
        state_->cv.notify_all();
    }
    void set_error(const NxStr &msg) {
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->err_msg = msg;
        state_->has_err = true;
        state_->ready = true;
        state_->cv.notify_all();
    }
    bool is_done() const {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->ready;
    }
    void get() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(state_->mu);
                if (state_->ready) {
                    if (state_->has_err) nx_fail_str("async task failed: " + std::string(state_->err_msg.view()));
                    return;
                }
                if (!nx_tls_in_worker) {
                    state_->cv.wait(lock, [this]() { return state_->ready; });
                    if (state_->has_err) nx_fail_str("async task failed: " + std::string(state_->err_msg.view()));
                    return;
                }
            }
            if (!nx_executor_try_step()) {
                std::unique_lock<std::mutex> lock(state_->mu);
                state_->cv.wait_for(lock, std::chrono::milliseconds(1), [this]() { return state_->ready; });
            }
        }
    }
};

class NxHybridExecutor {
public:
    static NxHybridExecutor &instance() {
        static NxHybridExecutor exec;
        return exec;
    }

    std::size_t worker_count() const noexcept { return workers_.size(); }

    void configure(std::int64_t requested_workers, bool cooperative_mode) {
        std::lock_guard<std::mutex> lock(cfg_mu_);
        cooperative_ = cooperative_mode;
        if (requested_workers > 0) {
            std::size_t cap = static_cast<std::size_t>(requested_workers);
            auto sb_cap = NxSandbox::instance().max_threads;
            if (sb_cap > 0 && cap > sb_cap) cap = static_cast<std::size_t>(sb_cap);
            ensure_started(cap);
        }
    }

    bool try_step() {
        std::function<void()> task;
        {
            std::lock_guard<std::mutex> lock(queue_mu_);
            if (queue_.empty()) return false;
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        if (task) task();
        return true;
    }

    template <typename F>
    auto submit(F &&fn) -> nx_Future<decltype(fn())> {
        ensure_default();
        using R = decltype(fn());
        nx_Future<R> fut;
        auto task = [f = std::forward<F>(fn), fut]() mutable {
            if constexpr (std::is_void_v<R>) {
                f();
                fut.set_value();
            } else {
                fut.set_value(f());
            }
        };
        {
            std::lock_guard<std::mutex> lock(queue_mu_);
            if (cooperative_) {
                coop_queue_.push_back(std::move(task));
            } else {
                queue_.push_back(std::move(task));
            }
        }
        cv_.notify_one();
        return fut;
    }

    void parallel_for(std::int64_t start, std::int64_t stop, const std::function<void(std::int64_t)> &body) {
        if (stop <= start) return;
        if (start < 0 && stop > std::numeric_limits<std::int64_t>::max() + start)
            nx_fail("parallel_range is too large");
        const std::int64_t count = stop - start;
        if (nx_tls_in_worker || count <= 1) {
            for (std::int64_t i = start; i < stop; ++i) body(i);
            return;
        }
        ensure_default();
        const std::size_t nworkers = std::min<std::size_t>(static_cast<std::size_t>(count), workers_.size() + 1);
        if (nworkers <= 1) {
            for (std::int64_t i = start; i < stop; ++i) body(i);
            return;
        }
        std::vector<nx_Future<void>> futures;
        futures.reserve(nworkers - 1);
        const std::int64_t base = count / static_cast<std::int64_t>(nworkers);
        const std::int64_t rem = count % static_cast<std::int64_t>(nworkers);
        for (std::size_t w = 0; w < nworkers - 1; ++w) {
            const std::int64_t wi = static_cast<std::int64_t>(w);
            const std::int64_t first = start + base * wi + std::min(wi, rem);
            const std::int64_t last = start + base * (wi + 1) + std::min(wi + 1, rem);
            futures.push_back(submit([first, last, &body]() {
                for (std::int64_t i = first; i < last; ++i) body(i);
            }));
        }
        const std::int64_t last_wi = static_cast<std::int64_t>(nworkers - 1);
        const std::int64_t last_first = start + base * last_wi + std::min(last_wi, rem);
        for (std::int64_t i = last_first; i < stop; ++i) body(i);
        for (auto &f : futures) f.get();
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(queue_mu_);
            if (stopping_) return;
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto &t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
        if (coop_thread_.joinable()) coop_thread_.join();
    }

    ~NxHybridExecutor() { shutdown(); }

private:
    NxHybridExecutor() = default;

    void ensure_default() {
        std::call_once(start_flag_, [this]() {
            NxSandbox::instance().init_from_env();
            std::size_t hw = std::thread::hardware_concurrency();
            if (hw == 0) hw = 2;
            auto sb_cap = NxSandbox::instance().max_threads;
            if (sb_cap > 0 && hw > sb_cap) hw = static_cast<std::size_t>(sb_cap);
            spawn_workers(hw);
        });
    }

    void ensure_started(std::size_t n) {
        ensure_default();
        if (n > workers_.size() && !stopping_) {
            spawn_workers(n - workers_.size());
        }
    }

    void spawn_workers(std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            workers_.emplace_back([this]() { worker_loop(false); });
        }
        if (!coop_started_) {
            coop_started_ = true;
            coop_thread_ = std::thread([this]() { worker_loop(true); });
        }
    }

    void worker_loop(bool coop_only) {
        nx_tls_in_worker = true;
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mu_);
                cv_.wait(lock, [this, coop_only]() {
                    if (stopping_) return true;
                    return coop_only ? !coop_queue_.empty() : !queue_.empty();
                });
                if (stopping_ && (coop_only ? coop_queue_.empty() : queue_.empty())) return;
                auto &q = coop_only ? coop_queue_ : queue_;
                if (q.empty()) continue;
                task = std::move(q.front());
                q.pop_front();
            }
            if (task) task();
        }
    }

    std::once_flag start_flag_;
    std::mutex cfg_mu_;
    std::mutex queue_mu_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> queue_;
    std::deque<std::function<void()>> coop_queue_;
    std::vector<std::thread> workers_;
    std::thread coop_thread_;
    bool coop_started_ = false;
    bool cooperative_ = false;
    bool stopping_ = false;
};

static bool nx_executor_try_step() {
    return NxHybridExecutor::instance().try_step();
}

template <typename F>
static void nx_parallel_for(std::int64_t start, std::int64_t stop, F body) {
    NxHybridExecutor::instance().parallel_for(start, stop, body);
}

// Cross-platform non-blocking I/O readiness helper (WSAPoll / kqueue / epoll)
static inline bool nx_io_wait_readable(nx_socket_t fd, int timeout_ms) {
#ifdef _WIN32
    WSAPOLLFD pfd{};
    pfd.fd = fd;
    pfd.events = POLLRDNORM;
    int rc = WSAPoll(&pfd, 1, timeout_ms);
    return rc > 0 && (pfd.revents & (POLLRDNORM | POLLHUP | POLLERR)) != 0;
#elif defined(__APPLE__)
    int kq = kqueue();
    if (kq < 0) return true;
    struct kevent ev{};
    EV_SET(&ev, fd, EVFILT_READ, EV_ADD | EV_ONESHOT, 0, 0, nullptr);
    struct timespec ts{timeout_ms / 1000, (timeout_ms % 1000) * 1000000L};
    struct kevent out{};
    int rc = kevent(kq, &ev, 1, &out, 1, timeout_ms >= 0 ? &ts : nullptr);
    close(kq);
    return rc > 0;
#else
    int ep = epoll_create1(0);
    if (ep < 0) return true;
    struct epoll_event ev{};
    ev.events = EPOLLIN | EPOLLERR | EPOLLHUP;
    ev.data.fd = fd;
    epoll_ctl(ep, EPOLL_CTL_ADD, fd, &ev);
    struct epoll_event out{};
    int rc = epoll_wait(ep, &out, 1, timeout_ms);
    close(ep);
    return rc > 0;
#endif
}

// nx.asyncio helpers
static inline nx_Future<void> nx_async_sleep(std::int64_t ms) {
    return NxHybridExecutor::instance().submit([ms]() {
        if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    });
}
static inline nx_Future<nx_result<NxStr>> nx_async_read_text(const NxStr &path) {
    return NxHybridExecutor::instance().submit([path]() { return nx_fs_read_text(path); });
}
static inline nx_Future<bool> nx_async_tcp_connect(const NxStr &host, std::int32_t port) {
    return NxHybridExecutor::instance().submit([host, port]() { return nx_tcp_connect(host, port); });
}
static inline nx_Future<nx_result<nx_HttpResponse>> nx_async_http_get(const NxStr &url) {
    return NxHybridExecutor::instance().submit([url]() { return nx_http_get(url); });
}
template <typename T>
static inline T nx_async_run(nx_Future<T> fut) { return fut.get(); }
static inline void nx_async_run(nx_Future<void> fut) { fut.get(); }
template <typename T>
static inline T nx_await(nx_Future<T> fut) { return fut.get(); }
static inline void nx_await(nx_Future<void> fut) { fut.get(); }
template <typename T>
static inline nx_vec<T> nx_async_gather_vec(nx_vec<nx_Future<T>> futs) {
    nx_vec<T> out;
    out.reserve(futs.size());
    for (auto &f : futs) out.push_back(f.get());
    return out;
}
template <typename... Fs>
static inline auto nx_async_gather(Fs &&...futs) {
    return std::tuple<decltype(futs.get())...>(futs.get()...);
}
static inline void nx_asyncio_configure(std::int64_t workers, bool cooperative) {
    NxHybridExecutor::instance().configure(workers, cooperative);
}
static inline std::int64_t nx_asyncio_workers() {
    NxHybridExecutor::instance().configure(0, false);
    std::size_t hw = std::thread::hardware_concurrency();
    return static_cast<std::int64_t>(hw ? hw : 1);
}

// nx.thread primitives: cpu_count, ThreadPool, Channel[T], Mutex[T], AtomicI64
static inline std::int64_t nx_thread_cpu_count() {
    std::size_t hw = std::thread::hardware_concurrency();
    return static_cast<std::int64_t>(hw ? hw : 1);
}

template <typename T>
class nx_Channel {
    struct State {
        std::mutex mu;
        std::condition_variable cv_send;
        std::condition_variable cv_recv;
        std::deque<T> queue;
        std::size_t cap = 0; // 0 = unbounded
        bool closed = false;
    };
    std::shared_ptr<State> state_;
public:
    explicit nx_Channel(std::int64_t capacity = 0) : state_(std::make_shared<State>()) {
        state_->cap = capacity > 0 ? static_cast<std::size_t>(capacity) : 0;
    }
    bool nx_send(T val) {
        std::unique_lock<std::mutex> lock(state_->mu);
        if (state_->cap > 0) {
            state_->cv_send.wait(lock, [this]() { return state_->closed || state_->queue.size() < state_->cap; });
        }
        if (state_->closed) return false;
        state_->queue.push_back(std::move(val));
        state_->cv_recv.notify_one();
        return true;
    }
    std::optional<T> nx_recv() {
        std::unique_lock<std::mutex> lock(state_->mu);
        state_->cv_recv.wait(lock, [this]() { return state_->closed || !state_->queue.empty(); });
        if (state_->queue.empty()) return std::nullopt;
        T v = std::move(state_->queue.front());
        state_->queue.pop_front();
        state_->cv_send.notify_one();
        return v;
    }
    std::optional<T> nx_try_recv() {
        std::lock_guard<std::mutex> lock(state_->mu);
        if (state_->queue.empty()) return std::nullopt;
        T v = std::move(state_->queue.front());
        state_->queue.pop_front();
        state_->cv_send.notify_one();
        return v;
    }
    void nx_close() {
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->closed = true;
        state_->cv_send.notify_all();
        state_->cv_recv.notify_all();
    }
};

template <typename T>
class nx_Mutex {
    struct State {
        std::mutex mu;
        T value;
        explicit State(T v) : value(std::move(v)) {}
    };
    std::shared_ptr<State> state_;
public:
    explicit nx_Mutex(T initial) : state_(std::make_shared<State>(std::move(initial))) {}
    T nx_get() {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->value;
    }
    void nx_set(T v) {
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->value = std::move(v);
    }
    template <typename F>
    auto nx_with_lock(F &&fn) {
        std::lock_guard<std::mutex> lock(state_->mu);
        return fn(state_->value);
    }
};

class nx_AtomicI64 {
    std::shared_ptr<std::atomic<std::int64_t>> atom_;
public:
    explicit nx_AtomicI64(std::int64_t init = 0) : atom_(std::make_shared<std::atomic<std::int64_t>>(init)) {}
    std::int64_t nx_load() const noexcept { return atom_->load(std::memory_order_seq_cst); }
    void nx_store(std::int64_t v) noexcept { atom_->store(v, std::memory_order_seq_cst); }
    std::int64_t nx_add(std::int64_t delta) noexcept { return atom_->fetch_add(delta, std::memory_order_seq_cst) + delta; }
    std::int64_t nx_sub(std::int64_t delta) noexcept { return atom_->fetch_sub(delta, std::memory_order_seq_cst) - delta; }
};

class nx_ThreadPool {
    std::size_t threads_;
public:
    explicit nx_ThreadPool(std::int64_t workers = 0) {
        std::size_t w = workers > 0 ? static_cast<std::size_t>(workers) : static_cast<std::size_t>(nx_thread_cpu_count());
        threads_ = w;
        NxHybridExecutor::instance().configure(static_cast<std::int64_t>(w), false);
    }
    std::int64_t nx_workers() const noexcept { return static_cast<std::int64_t>(threads_); }
    template <typename F>
    auto nx_submit(F &&fn) { return NxHybridExecutor::instance().submit(std::forward<F>(fn)); }
};

// ============================================================================
// 13. GPU Workload Dispatch (CUDA/OpenCL Dynamic Probe + Multi-Core SIMD Fallback)
// ============================================================================
enum class NxGpuBackendKind { CPU_SIMD = 0, OPENCL = 1, CUDA = 2 };

class NxGpuRuntime {
public:
    NxGpuBackendKind backend = NxGpuBackendKind::CPU_SIMD;
    std::string backend_name = "cpu-simd";
    int device_count = 1;

    static NxGpuRuntime &instance() {
        static NxGpuRuntime rt;
        return rt;
    }

private:
    NxGpuRuntime() {
        probe();
    }

    void probe() {
#ifdef _WIN32
        if (HMODULE cu = LoadLibraryA("nvcuda.dll")) {
            using CuInitFn = int(__stdcall *)(unsigned int);
            using CuCntFn = int(__stdcall *)(int *);
            auto cuInit = reinterpret_cast<CuInitFn>(reinterpret_cast<void *>(GetProcAddress(cu, "cuInit")));
            auto cuCnt = reinterpret_cast<CuCntFn>(reinterpret_cast<void *>(GetProcAddress(cu, "cuDeviceGetCount")));
            int cnt = 0;
            if (cuInit && cuCnt && cuInit(0) == 0 && cuCnt(&cnt) == 0 && cnt > 0) {
                backend = NxGpuBackendKind::CUDA;
                backend_name = "cuda";
                device_count = cnt;
                nx_audit_log_event("module.load", "gpu", "nvcuda.dll", "CUDA driver detected", true);
                return;
            }
            FreeLibrary(cu);
        }
        if (HMODULE ocl = LoadLibraryA("OpenCL.dll")) {
            backend = NxGpuBackendKind::OPENCL;
            backend_name = "opencl";
            device_count = 1;
            nx_audit_log_event("module.load", "gpu", "OpenCL.dll", "OpenCL runtime detected", true);
            FreeLibrary(ocl);
            return;
        }
#else
        if (void *cu = dlopen("libcuda.so.1", RTLD_LAZY)) {
            using CuInitFn = int(*)(unsigned int);
            using CuCntFn = int(*)(int *);
            auto cuInit = reinterpret_cast<CuInitFn>(dlsym(cu, "cuInit"));
            auto cuCnt = reinterpret_cast<CuCntFn>(dlsym(cu, "cuDeviceGetCount"));
            int cnt = 0;
            if (cuInit && cuCnt && cuInit(0) == 0 && cuCnt(&cnt) == 0 && cnt > 0) {
                backend = NxGpuBackendKind::CUDA;
                backend_name = "cuda";
                device_count = cnt;
                nx_audit_log_event("module.load", "gpu", "libcuda.so.1", "CUDA driver detected", true);
                return;
            }
            dlclose(cu);
        }
        if (void *ocl = dlopen("libOpenCL.so.1", RTLD_LAZY)) {
            backend = NxGpuBackendKind::OPENCL;
            backend_name = "opencl";
            device_count = 1;
            nx_audit_log_event("module.load", "gpu", "libOpenCL.so.1", "OpenCL runtime detected", true);
            dlclose(ocl);
            return;
        }
#endif
        backend = NxGpuBackendKind::CPU_SIMD;
        backend_name = "cpu-simd";
        device_count = 1;
    }
};

template <typename T>
class nx_GpuBuffer {
    nx_vec<T> host_mirror_;
public:
    nx_GpuBuffer() = default;
    explicit nx_GpuBuffer(std::int64_t count) {
        if (count < 0) nx_fail("negative GpuBuffer length");
        host_mirror_.resize(static_cast<std::size_t>(count));
    }
    explicit nx_GpuBuffer(nx_vec<T> data) : host_mirror_(std::move(data)) {}
    std::int64_t size() const noexcept { return static_cast<std::int64_t>(host_mirror_.size()); }
    T *data() noexcept { return host_mirror_.data(); }
    const T *data() const noexcept { return host_mirror_.data(); }
    nx_vec<T> to_host() const { return host_mirror_; }
};

static inline NxStr nx_gpu_backend_name() {
    return NxStr::from_std(NxGpuRuntime::instance().backend_name);
}
static inline bool nx_gpu_is_available() {
    return NxGpuRuntime::instance().backend != NxGpuBackendKind::CPU_SIMD;
}
static inline std::int64_t nx_gpu_device_count() {
    return static_cast<std::int64_t>(NxGpuRuntime::instance().device_count);
}
template <typename T>
static inline nx_GpuBuffer<T> nx_gpu_to_device(const nx_vec<T> &data) {
    return nx_GpuBuffer<T>(data);
}
template <typename T>
static inline nx_vec<T> nx_gpu_to_host(const nx_GpuBuffer<T> &buf) {
    return buf.to_host();
}
template <typename T>
static inline nx_GpuBuffer<T> nx_gpu_add(const nx_GpuBuffer<T> &a, const nx_GpuBuffer<T> &b) {
    if (a.size() != b.size()) nx_fail("GpuBuffer size mismatch in gpu.add");
    std::int64_t n = a.size();
    nx_GpuBuffer<T> out(n);
    const T *pa = a.data(), *pb = b.data();
    T *po = out.data();
    nx_parallel_for(0, n, [pa, pb, po](std::int64_t i) { po[i] = pa[i] + pb[i]; });
    return out;
}
template <typename T>
static inline nx_GpuBuffer<T> nx_gpu_mul(const nx_GpuBuffer<T> &a, const nx_GpuBuffer<T> &b) {
    if (a.size() != b.size()) nx_fail("GpuBuffer size mismatch in gpu.mul");
    std::int64_t n = a.size();
    nx_GpuBuffer<T> out(n);
    const T *pa = a.data(), *pb = b.data();
    T *po = out.data();
    nx_parallel_for(0, n, [pa, pb, po](std::int64_t i) { po[i] = pa[i] * pb[i]; });
    return out;
}
template <typename T>
static inline T nx_gpu_reduce_sum(const nx_GpuBuffer<T> &buf) {
    const T *p = buf.data();
    std::int64_t n = buf.size();
    if (n == 0) return T{};
    std::size_t nw = std::max<std::size_t>(1, NxHybridExecutor::instance().worker_count());
    if (n < 4096 || nw <= 1) {
        T acc{};
        for (std::int64_t i = 0; i < n; ++i) acc += p[i];
        return acc;
    }
    nx_vec<T> partials(nw, T{});
    nx_parallel_for(0, static_cast<std::int64_t>(nw), [&](std::int64_t w) {
        std::int64_t first = (n * w) / static_cast<std::int64_t>(nw);
        std::int64_t last = (n * (w + 1)) / static_cast<std::int64_t>(nw);
        T local{};
        for (std::int64_t i = first; i < last; ++i) local += p[i];
        partials[static_cast<std::size_t>(w)] = local;
    });
    T total{};
    for (const T &x : partials) total += x;
    return total;
}
static inline nx_vec<double> nx_gpu_matmul(const nx_vec<double> &a, const nx_vec<double> &b,
                                           std::int64_t m, std::int64_t k, std::int64_t n) {
    if (m <= 0 || k <= 0 || n <= 0) nx_fail("matmul dimensions must be positive");
    if (static_cast<std::int64_t>(a.size()) != m * k || static_cast<std::int64_t>(b.size()) != k * n)
        nx_fail("matmul buffer dimensions mismatch");
    nx_vec<double> out(static_cast<std::size_t>(m * n), 0.0);
    const double *pa = a.data(), *pb = b.data();
    double *po = out.data();
    nx_parallel_for(0, m, [pa, pb, po, k, n](std::int64_t row) {
        for (std::int64_t p = 0; p < k; ++p) {
            double av = pa[row * k + p];
#pragma omp simd
            for (std::int64_t col = 0; col < n; ++col) {
                po[row * n + col] += av * pb[p * n + col];
            }
        }
    });
    return out;
}
static inline nx_Future<nx_vec<double>> nx_gpu_matmul_async(const nx_vec<double> &a, const nx_vec<double> &b,
                                                            std::int64_t m, std::int64_t k, std::int64_t n) {
    return NxHybridExecutor::instance().submit([a, b, m, k, n]() {
        return nx_gpu_matmul(a, b, m, k, n);
    });
}
'''
