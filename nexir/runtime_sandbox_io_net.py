"""C++17 Sandbox enforcement, resource limits, audit logging, deep FS I/O, and networking."""

RUNTIME_SANDBOX_IO_NET_CPP = r'''
// ============================================================================
// 9. Sandbox Policy, Resource Limits, OS Isolation & Module Audit Logger
// ============================================================================
struct NxAuditEntry {
    std::int64_t ts_ms;
    std::string event;
    std::string kind;
    std::string name;
    std::string detail;
    bool allowed;
};

class NxSandbox {
public:
    bool enabled = false;
    bool deny_fs = false;
    bool deny_net = false;
    bool deny_py = false;
    bool deny_ffi = false;
    bool audit_stderr = false;
    bool debug_mem = false;
    std::uint64_t max_memory_bytes = 0;   // 0 = unlimited
    std::uint64_t max_allocations = 0;    // 0 = unlimited
    std::uint64_t max_open_files = 256;
    std::uint64_t max_sockets = 128;
    std::uint64_t max_threads = 0;        // 0 = hardware_concurrency
    std::uint64_t timeout_ms = 0;         // 0 = unlimited
    std::vector<std::string> allow_read_paths;
    std::vector<std::string> allow_write_paths;
    std::vector<std::string> allow_delete_paths;
    std::vector<std::string> allow_hosts;
    std::vector<int> allow_ports;
    std::vector<std::string> allow_py_modules;
    std::string audit_log_path;

    std::atomic<std::uint64_t> open_files{0};
    std::atomic<std::uint64_t> open_sockets{0};
    std::mutex mu;
    std::vector<NxAuditEntry> audit_entries;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    static NxSandbox &instance() {
        static NxSandbox sb;
        return sb;
    }

    static std::vector<std::string> split_env_list(const char *val) {
        std::vector<std::string> out;
        if (!val || !*val) return out;
        std::string s(val);
        std::size_t start = 0;
        while (start < s.size()) {
            std::size_t pos = s.find(';', start);
            if (pos == std::string::npos) pos = s.find(',', start);
            std::string item = (pos == std::string::npos) ? s.substr(start) : s.substr(start, pos - start);
            if (!item.empty()) out.push_back(item);
            if (pos == std::string::npos) break;
            start = pos + 1;
        }
        return out;
    }

    static std::uint64_t parse_u64_env(const char *name, std::uint64_t fallback = 0) {
        const char *v = std::getenv(name);
        if (!v || !*v) return fallback;
        char *end = nullptr;
        unsigned long long val = std::strtoull(v, &end, 10);
        if (!end) return fallback;
        if (*end == 'k' || *end == 'K') val *= 1024ULL;
        else if (*end == 'm' || *end == 'M') val *= 1024ULL * 1024ULL;
        else if (*end == 'g' || *end == 'G') val *= 1024ULL * 1024ULL * 1024ULL;
        return static_cast<std::uint64_t>(val);
    }

    void init_from_env() {
        std::call_once(init_flag_, [this]() {
            const char *sb = std::getenv("NX_SANDBOX");
            if (sb && (*sb == '1' || *sb == 't' || *sb == 'T')) enabled = true;
            if (const char *v = std::getenv("NX_DENY_FS")) deny_fs = (*v == '1');
            if (const char *v = std::getenv("NX_DENY_NET")) deny_net = (*v == '1');
            if (const char *v = std::getenv("NX_DENY_PY")) deny_py = (*v == '1');
            if (const char *v = std::getenv("NX_DENY_FFI")) deny_ffi = (*v == '1');
            if (const char *v = std::getenv("NX_AUDIT_STDERR")) audit_stderr = (*v == '1');
            if (const char *v = std::getenv("NX_DEBUG_MEM")) debug_mem = (*v == '1');
            if (const char *v = std::getenv("NX_AUDIT_LOG")) audit_log_path = v;

            max_memory_bytes = parse_u64_env("NX_MAX_MEMORY", 0);
            max_allocations = parse_u64_env("NX_MAX_ALLOCS", 0);
            max_open_files = parse_u64_env("NX_MAX_FILES", 256);
            max_sockets = parse_u64_env("NX_MAX_SOCKETS", 128);
            max_threads = parse_u64_env("NX_MAX_THREADS", 0);
            timeout_ms = parse_u64_env("NX_TIMEOUT_MS", 0);

            allow_read_paths = split_env_list(std::getenv("NX_ALLOW_READ"));
            allow_write_paths = split_env_list(std::getenv("NX_ALLOW_WRITE"));
            allow_delete_paths = split_env_list(std::getenv("NX_ALLOW_DELETE"));
            allow_hosts = split_env_list(std::getenv("NX_ALLOW_HOSTS"));
            allow_py_modules = split_env_list(std::getenv("NX_ALLOW_PY_MODULES"));
            for (const auto &p : split_env_list(std::getenv("NX_ALLOW_PORTS"))) {
                allow_ports.push_back(std::atoi(p.c_str()));
            }

            if (timeout_ms > 0) {
                std::uint64_t limit = timeout_ms;
                std::thread([limit]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(limit));
                    std::fprintf(stderr, "NEXIR sandbox: execution timeout exceeded (%llu ms)\n",
                                 static_cast<unsigned long long>(limit));
                    std::_Exit(124);
                }).detach();
            }

#ifdef _WIN32
            if (enabled && std::getenv("NX_SANDBOX_OS") && *std::getenv("NX_SANDBOX_OS") == '1') {
                HANDLE job = CreateJobObjectW(nullptr, nullptr);
                if (job) {
                    JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli{};
                    jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                    jeli.BasicLimitInformation.ActiveProcessLimit = 1;
                    if (max_memory_bytes > 0) {
                        jeli.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
                        jeli.ProcessMemoryLimit = static_cast<SIZE_T>(max_memory_bytes);
                    }
                    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
                    AssignProcessToJobObject(job, GetCurrentProcess());
                }
            }
#else
            if (enabled && std::getenv("NX_SANDBOX_OS") && *std::getenv("NX_SANDBOX_OS") == '1') {
                if (max_memory_bytes > 0) {
                    struct rlimit rl{static_cast<rlim_t>(max_memory_bytes), static_cast<rlim_t>(max_memory_bytes)};
                    setrlimit(RLIMIT_AS, &rl);
                }
                if (max_open_files > 0) {
                    struct rlimit rl{static_cast<rlim_t>(max_open_files), static_cast<rlim_t>(max_open_files)};
                    setrlimit(RLIMIT_NOFILE, &rl);
                }
            }
#endif
        });
    }

    static std::string json_escape(const std::string &s) {
        std::string out;
        out.reserve(s.size() + 4);
        for (unsigned char c : s) {
            if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(static_cast<char>(c)); }
            else if (c == '\n') out.append("\\n");
            else if (c == '\r') out.append("\\r");
            else if (c == '\t') out.append("\\t");
            else if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out.append(buf);
            } else {
                out.push_back(static_cast<char>(c));
            }
        }
        return out;
    }

    void record_audit(const char *event, const char *kind, const char *name,
                      const char *detail, bool allowed) {
        init_from_env();
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        NxAuditEntry entry{
            static_cast<std::int64_t>(now_ms),
            event ? event : "",
            kind ? kind : "",
            name ? name : "",
            detail ? detail : "",
            allowed
        };
        std::string line = "{\"ts_ms\":" + std::to_string(entry.ts_ms) +
            ",\"event\":\"" + json_escape(entry.event) +
            "\",\"kind\":\"" + json_escape(entry.kind) +
            "\",\"name\":\"" + json_escape(entry.name) +
            "\",\"detail\":\"" + json_escape(entry.detail) +
            "\",\"status\":\"" + std::string(allowed ? "allowed" : "denied") + "\"}";

        std::lock_guard<std::mutex> lock(mu);
        if (audit_entries.size() < 4096) audit_entries.push_back(entry);
        if (audit_stderr) {
            std::fprintf(stderr, "[NX-AUDIT] %s\n", line.c_str());
        }
        if (!audit_log_path.empty()) {
            std::FILE *f = std::fopen(audit_log_path.c_str(), "ab");
            if (f) {
                std::fwrite(line.data(), 1, line.size(), f);
                std::fputc('\n', f);
                std::fclose(f);
            }
        }
    }

    static std::string canonicalize_for_check(const std::string &raw) {
        std::error_code ec;
        std::filesystem::path p(raw);
        auto abs_p = std::filesystem::weakly_canonical(p, ec);
        std::string res = ec ? p.lexically_normal().string() : abs_p.string();
#ifdef _WIN32
        std::transform(res.begin(), res.end(), res.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
        return res;
    }

    static bool is_device_or_system_path(const std::string &p) {
        std::string low = p;
        std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (low.rfind("\\\\.\\", 0) == 0 || low.rfind("\\\\?\\", 0) == 0) return true;
        if (low.rfind("/dev/", 0) == 0 || low.rfind("/proc/", 0) == 0 || low.rfind("/sys/", 0) == 0) return true;
        return false;
    }

    static bool path_within_any(const std::string &target, const std::vector<std::string> &prefixes) {
        if (prefixes.empty()) return false;
        std::string norm_t = canonicalize_for_check(target);
        for (const auto &pref : prefixes) {
            std::string norm_p = canonicalize_for_check(pref);
            if (norm_t == norm_p) return true;
            if (norm_t.size() > norm_p.size() && norm_t.compare(0, norm_p.size(), norm_p) == 0) {
                char sep = norm_t[norm_p.size()];
                if (sep == '/' || sep == '\\' || (!norm_p.empty() && (norm_p.back() == '/' || norm_p.back() == '\\')))
                    return true;
            }
        }
        return false;
    }

    bool check_fs(const char *op, const std::string &path) {
        init_from_env();
        if (is_device_or_system_path(path)) {
            if (enabled || !allow_read_paths.empty()) {
                record_audit("sandbox.deny", "fs", op, path.c_str(), false);
                return false;
            }
        }
        if (deny_fs) {
            record_audit("sandbox.deny", "fs", op, path.c_str(), false);
            return false;
        }
        std::string op_s(op ? op : "");
        if (op_s == "read" && !allow_read_paths.empty()) {
            if (!path_within_any(path, allow_read_paths) && !path_within_any(path, allow_write_paths)) {
                record_audit("sandbox.deny", "fs", op, path.c_str(), false);
                return false;
            }
        } else if ((op_s == "write" || op_s == "edit" || op_s == "mkdir" || op_s == "copy") && !allow_write_paths.empty()) {
            if (!path_within_any(path, allow_write_paths)) {
                record_audit("sandbox.deny", "fs", op, path.c_str(), false);
                return false;
            }
        } else if ((op_s == "delete" || op_s == "chmod") && (!allow_delete_paths.empty() || !allow_write_paths.empty())) {
            const auto &list = !allow_delete_paths.empty() ? allow_delete_paths : allow_write_paths;
            if (!path_within_any(path, list)) {
                record_audit("sandbox.deny", "fs", op, path.c_str(), false);
                return false;
            }
        } else if (enabled) {
            // In strict sandbox mode, if no allowlist was specified for write/delete, default deny write/delete outside CWD
            if (op_s == "write" || op_s == "delete" || op_s == "chmod") {
                std::error_code ec;
                std::string cwd = std::filesystem::current_path(ec).string();
                if (!ec && !path_within_any(path, {cwd})) {
                    record_audit("sandbox.deny", "fs", op, path.c_str(), false);
                    return false;
                }
            }
        }
        record_audit("fs.access", "fs", op, path.c_str(), true);
        return true;
    }

    bool check_net(const char *op, const std::string &host, int port) {
        init_from_env();
        if (deny_net) {
            record_audit("sandbox.deny", "net", op, (host + ":" + std::to_string(port)).c_str(), false);
            return false;
        }
        if (!allow_hosts.empty()) {
            bool ok = std::find(allow_hosts.begin(), allow_hosts.end(), host) != allow_hosts.end();
            if (!ok) {
                record_audit("sandbox.deny", "net", op, (host + ":" + std::to_string(port)).c_str(), false);
                return false;
            }
        }
        if (!allow_ports.empty() && port > 0) {
            bool ok = std::find(allow_ports.begin(), allow_ports.end(), port) != allow_ports.end();
            if (!ok) {
                record_audit("sandbox.deny", "net", op, (host + ":" + std::to_string(port)).c_str(), false);
                return false;
            }
        }
        record_audit("net.access", "net", op, (host + ":" + std::to_string(port)).c_str(), true);
        return true;
    }

    bool check_py_module(const char *mod) {
        init_from_env();
        std::string name(mod ? mod : "");
        if (deny_py) {
            record_audit("sandbox.deny", "python", name.c_str(), "python interop disabled by sandbox", false);
            return false;
        }
        if (!allow_py_modules.empty()) {
            std::string root = name.substr(0, name.find('.'));
            bool ok = std::find(allow_py_modules.begin(), allow_py_modules.end(), name) != allow_py_modules.end() ||
                      std::find(allow_py_modules.begin(), allow_py_modules.end(), root) != allow_py_modules.end();
            if (!ok) {
                record_audit("sandbox.deny", "python", name.c_str(), "module not in allowlist", false);
                return false;
            }
        } else if (enabled) {
            // In strict sandbox mode, block dangerous Python OS/process modules by default
            static const char *const kBlocked[] = {"os", "subprocess", "ctypes", "shutil", "socket", "importlib", "sys"};
            std::string root = name.substr(0, name.find('.'));
            for (const char *b : kBlocked) {
                if (root == b) {
                    record_audit("sandbox.deny", "python", name.c_str(), "blocked privileged python module in sandbox", false);
                    return false;
                }
            }
        }
        record_audit("module.load", "python", name.c_str(), "cpython bridge", true);
        return true;
    }

private:
    std::once_flag init_flag_;
};

static void nx_sandbox_on_alloc(std::size_t bytes) {
    auto &sb = NxSandbox::instance();
    sb.init_from_env();
    auto &st = nx_mem_stats();
    if (sb.max_allocations > 0 && st.total_allocs.load(std::memory_order_relaxed) >= sb.max_allocations) {
        sb.record_audit("sandbox.deny", "memory", "alloc_count", "max_allocations exceeded", false);
        nx_fail("sandbox: max allocation count exceeded");
    }
    if (sb.max_memory_bytes > 0) {
        std::uint64_t active = st.active_bytes.load(std::memory_order_relaxed);
        if (active + bytes > sb.max_memory_bytes) {
            sb.record_audit("sandbox.deny", "memory", "max_memory", "max_memory_bytes exceeded", false);
            nx_fail("sandbox: memory limit exceeded");
        }
    }
}

static void nx_sandbox_on_free(std::size_t) noexcept {}

static void nx_audit_log_event(const char *event, const char *kind, const char *name,
                               const char *detail, bool allowed) {
    NxSandbox::instance().record_audit(event, kind, name, detail, allowed);
}

static inline void nx_sandbox_require_fs(const char *op, const NxStr &path) {
    if (!NxSandbox::instance().check_fs(op, std::string(path.view()))) {
        nx_fail_str("sandbox: filesystem access denied (" + std::string(op) + ": " + std::string(path.view()) + ")");
    }
}

static inline void nx_sandbox_require_net(const char *op, const NxStr &host, int port) {
    if (!NxSandbox::instance().check_net(op, std::string(host.view()), port)) {
        nx_fail_str("sandbox: network access denied (" + std::string(op) + ": " + std::string(host.view()) + ":" + std::to_string(port) + ")");
    }
}

// NX API functions for nx.sandbox & nx.debug
static inline void nx_sandbox_enable() { NxSandbox::instance().init_from_env(); NxSandbox::instance().enabled = true; }
static inline void nx_sandbox_set_max_memory(std::int64_t bytes) {
    if (bytes <= 0) nx_fail("max_memory must be positive");
    auto &sb = NxSandbox::instance();
    sb.init_from_env();
    std::uint64_t req = static_cast<std::uint64_t>(bytes);
    if (sb.max_memory_bytes == 0 || req < sb.max_memory_bytes) sb.max_memory_bytes = req;
}
static inline void nx_sandbox_allow_read(const NxStr &path) {
    auto &sb = NxSandbox::instance();
    sb.init_from_env();
    sb.allow_read_paths.push_back(std::string(path.view()));
}
static inline void nx_sandbox_allow_write(const NxStr &path) {
    auto &sb = NxSandbox::instance();
    sb.init_from_env();
    sb.allow_write_paths.push_back(std::string(path.view()));
}
static inline void nx_sandbox_deny_network() {
    auto &sb = NxSandbox::instance();
    sb.init_from_env();
    sb.deny_net = true;
}
static inline void nx_sandbox_allow_host(const NxStr &host) {
    auto &sb = NxSandbox::instance();
    sb.init_from_env();
    sb.allow_hosts.push_back(std::string(host.view()));
}
static inline void nx_sandbox_set_audit_log(const NxStr &path) {
    auto &sb = NxSandbox::instance();
    sb.init_from_env();
    std::lock_guard<std::mutex> lock(sb.mu);
    sb.audit_log_path = std::string(path.view());
}
static inline nx_vec<NxStr> nx_sandbox_audit_log() {
    auto &sb = NxSandbox::instance();
    sb.init_from_env();
    std::lock_guard<std::mutex> lock(sb.mu);
    nx_vec<NxStr> out;
    out.reserve(sb.audit_entries.size());
    for (const auto &e : sb.audit_entries) {
        out.push_back(NxStr::from_std(e.event + "|" + e.kind + "|" + e.name + "|" + (e.allowed ? "allowed" : "denied") + "|" + e.detail));
    }
    return out;
}

static inline std::int64_t nx_debug_active_bytes() { return static_cast<std::int64_t>(nx_mem_stats().active_bytes.load()); }
static inline std::int64_t nx_debug_peak_bytes() { return static_cast<std::int64_t>(nx_mem_stats().peak_bytes.load()); }
static inline std::int64_t nx_debug_cache_hits() { return static_cast<std::int64_t>(nx_mem_stats().cache_hits.load()); }
static inline std::int64_t nx_debug_live_blocks() { return static_cast<std::int64_t>(nx_mem_stats().live_blocks.load()); }
static inline std::int64_t nx_debug_intern_hits() { return static_cast<std::int64_t>(nx_mem_stats().intern_hits.load()); }
static inline NxStr nx_debug_mem_report() {
    const auto &st = nx_mem_stats();
    NxStrBuilder sb;
    sb.append("active_bytes=").append(static_cast<std::int64_t>(st.active_bytes.load()))
      .append(" peak_bytes=").append(static_cast<std::int64_t>(st.peak_bytes.load()))
      .append(" allocs=").append(static_cast<std::int64_t>(st.total_allocs.load()))
      .append(" frees=").append(static_cast<std::int64_t>(st.total_frees.load()))
      .append(" cache_hits=").append(static_cast<std::int64_t>(st.cache_hits.load()))
      .append(" intern_hits=").append(static_cast<std::int64_t>(st.intern_hits.load()))
      .append(" live_blocks=").append(static_cast<std::int64_t>(st.live_blocks.load()));
    return sb.finish();
}

struct NxDebugMemGuard {
    ~NxDebugMemGuard() {
        NxStrInternCache::instance().clear();
        if (NxSandbox::instance().debug_mem) {
            std::fprintf(stderr, "[NX-MEM] %s\n", nx_debug_mem_report().c_str());
        }
    }
};

// ============================================================================
// 10. Deep File I/O (nx.fs / nx.io)
// ============================================================================
struct nx_FileInfo {
    std::int64_t nx_size = 0;
    std::int64_t nx_mtime = 0;
    bool nx_is_dir = false;
    bool nx_is_file = false;
    bool nx_readonly = false;
    NxStr nx___str__() const {
        NxStrBuilder sb;
        sb.append("FileInfo(size=").append(nx_size).append(", is_dir=").append(nx_is_dir).append(")");
        return sb.finish();
    }
};

static inline NxStr nx_read_text(const NxStr &path) {
    nx_sandbox_require_fs("read", path);
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) nx_fail_str("cannot read file: " + std::string(path.view()));
    if (std::fseek(file, 0, SEEK_END) != 0) { std::fclose(file); nx_fail("cannot seek file"); }
    long end = std::ftell(file);
    if (end < 0 || std::fseek(file, 0, SEEK_SET) != 0) { std::fclose(file); nx_fail("cannot seek file"); }
    std::string buf(static_cast<std::size_t>(end), '\0');
    if (end > 0 && std::fread(buf.data(), 1, static_cast<std::size_t>(end), file) != static_cast<std::size_t>(end)) {
        std::fclose(file);
        nx_fail("cannot read file");
    }
    std::fclose(file);
    return NxStr(buf.data(), buf.size(), false);
}

static inline nx_result<NxStr> nx_fs_read_text(const NxStr &path) {
    if (!NxSandbox::instance().check_fs("read", std::string(path.view())))
        return nx_err<NxStr>("sandbox denied file read");
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) return nx_err<NxStr>("cannot open file for reading");
    if (std::fseek(file, 0, SEEK_END) != 0) { std::fclose(file); return nx_err<NxStr>("cannot seek file"); }
    long end = std::ftell(file);
    if (end < 0 || std::fseek(file, 0, SEEK_SET) != 0) { std::fclose(file); return nx_err<NxStr>("cannot seek file"); }
    std::string buf(static_cast<std::size_t>(end), '\0');
    if (end > 0 && std::fread(buf.data(), 1, static_cast<std::size_t>(end), file) != static_cast<std::size_t>(end)) {
        std::fclose(file);
        return nx_err<NxStr>("failed reading file");
    }
    std::fclose(file);
    return nx_ok(NxStr(buf.data(), buf.size(), false));
}

static inline nx_result<nx_vec<std::uint8_t>> nx_fs_read_bytes(const NxStr &path) {
    if (!NxSandbox::instance().check_fs("read", std::string(path.view())))
        return nx_err<nx_vec<std::uint8_t>>("sandbox denied file read");
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) return nx_err<nx_vec<std::uint8_t>>("cannot open file");
    std::fseek(file, 0, SEEK_END);
    long end = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (end < 0) { std::fclose(file); return nx_err<nx_vec<std::uint8_t>>("cannot determine file size"); }
    nx_vec<std::uint8_t> out(static_cast<std::size_t>(end));
    if (end > 0 && std::fread(out.data(), 1, static_cast<std::size_t>(end), file) != static_cast<std::size_t>(end)) {
        std::fclose(file);
        return nx_err<nx_vec<std::uint8_t>>("read error");
    }
    std::fclose(file);
    return nx_ok(std::move(out));
}

static inline nx_result<bool> nx_fs_write_text(const NxStr &path, const NxStr &content) {
    if (!NxSandbox::instance().check_fs("write", std::string(path.view())))
        return nx_err<bool>("sandbox denied file write");
    std::FILE *file = std::fopen(path.c_str(), "wb");
    if (!file) return nx_err<bool>("cannot open file for writing");
    if (!content.empty() && std::fwrite(content.c_str(), 1, content.size(), file) != content.size()) {
        std::fclose(file);
        return nx_err<bool>("write error");
    }
    std::fclose(file);
    return nx_ok(true);
}

static inline nx_result<bool> nx_fs_write_bytes(const NxStr &path, const nx_vec<std::uint8_t> &data) {
    if (!NxSandbox::instance().check_fs("write", std::string(path.view())))
        return nx_err<bool>("sandbox denied file write");
    std::FILE *file = std::fopen(path.c_str(), "wb");
    if (!file) return nx_err<bool>("cannot open file for writing");
    if (!data.empty() && std::fwrite(data.data(), 1, data.size(), file) != data.size()) {
        std::fclose(file);
        return nx_err<bool>("write error");
    }
    std::fclose(file);
    return nx_ok(true);
}

static inline nx_result<bool> nx_fs_append_text(const NxStr &path, const NxStr &content) {
    if (!NxSandbox::instance().check_fs("write", std::string(path.view())))
        return nx_err<bool>("sandbox denied file append");
    std::FILE *file = std::fopen(path.c_str(), "ab");
    if (!file) return nx_err<bool>("cannot open file for appending");
    if (!content.empty() && std::fwrite(content.c_str(), 1, content.size(), file) != content.size()) {
        std::fclose(file);
        return nx_err<bool>("append error");
    }
    std::fclose(file);
    return nx_ok(true);
}

static inline nx_result<bool> nx_fs_copy_file(const NxStr &src, const NxStr &dst) {
    if (!NxSandbox::instance().check_fs("read", std::string(src.view())))
        return nx_err<bool>("sandbox denied copy source read");
    if (!NxSandbox::instance().check_fs("copy", std::string(dst.view())))
        return nx_err<bool>("sandbox denied copy destination write");
    std::error_code ec;
    bool ok = std::filesystem::copy_file(src.c_str(), dst.c_str(),
                                         std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) return nx_err<bool>(NxStr::from_std("copy failed: " + ec.message()));
    return nx_ok(ok);
}

static inline nx_result<bool> nx_fs_move_file(const NxStr &src, const NxStr &dst) {
    if (!NxSandbox::instance().check_fs("delete", std::string(src.view())) ||
        !NxSandbox::instance().check_fs("write", std::string(dst.view())))
        return nx_err<bool>("sandbox denied file move");
    std::error_code ec;
    std::filesystem::rename(src.c_str(), dst.c_str(), ec);
    if (ec) return nx_err<bool>(NxStr::from_std("move failed: " + ec.message()));
    return nx_ok(true);
}

static inline nx_result<bool> nx_fs_delete_file(const NxStr &path) {
    if (!NxSandbox::instance().check_fs("delete", std::string(path.view())))
        return nx_err<bool>("sandbox denied file delete");
    std::error_code ec;
    bool removed = std::filesystem::remove(path.c_str(), ec);
    if (ec) return nx_err<bool>(NxStr::from_std("delete failed: " + ec.message()));
    return nx_ok(removed);
}

static inline nx_result<bool> nx_fs_edit_file(const NxStr &path, const NxStr &target, const NxStr &replacement) {
    if (!NxSandbox::instance().check_fs("read", std::string(path.view())) ||
        !NxSandbox::instance().check_fs("edit", std::string(path.view())))
        return nx_err<bool>("sandbox denied file edit");
    auto r = nx_fs_read_text(path);
    if (r.is_err()) return nx_err<bool>(r.error());
    NxStr content = r.value();
    if (!content.contains(target)) return nx_err<bool>("edit target text not found in file");
    NxStr updated = content.replace(target, replacement);
    return nx_fs_write_text(path, updated);
}

static inline bool nx_fs_exists(const NxStr &path) {
    nx_sandbox_require_fs("read", path);
    std::error_code ec;
    return std::filesystem::exists(path.c_str(), ec);
}
static inline bool nx_fs_is_file(const NxStr &path) {
    nx_sandbox_require_fs("read", path);
    std::error_code ec;
    return std::filesystem::is_regular_file(path.c_str(), ec);
}
static inline bool nx_fs_is_dir(const NxStr &path) {
    nx_sandbox_require_fs("read", path);
    std::error_code ec;
    return std::filesystem::is_directory(path.c_str(), ec);
}
static inline nx_result<bool> nx_fs_mkdir(const NxStr &path) {
    if (!NxSandbox::instance().check_fs("mkdir", std::string(path.view())))
        return nx_err<bool>("sandbox denied mkdir");
    std::error_code ec;
    std::filesystem::create_directories(path.c_str(), ec);
    if (ec) return nx_err<bool>(NxStr::from_std("mkdir failed: " + ec.message()));
    return nx_ok(true);
}
static inline nx_result<nx_vec<NxStr>> nx_fs_list_dir(const NxStr &path) {
    if (!NxSandbox::instance().check_fs("read", std::string(path.view())))
        return nx_err<nx_vec<NxStr>>("sandbox denied list_dir");
    std::error_code ec;
    nx_vec<NxStr> out;
    for (const auto &entry : std::filesystem::directory_iterator(path.c_str(), ec)) {
        out.push_back(NxStr::from_std(entry.path().filename().string()));
    }
    if (ec) return nx_err<nx_vec<NxStr>>(NxStr::from_std("list_dir failed: " + ec.message()));
    std::sort(out.begin(), out.end());
    return nx_ok(std::move(out));
}
static inline nx_result<nx_FileInfo> nx_fs_stat(const NxStr &path) {
    if (!NxSandbox::instance().check_fs("read", std::string(path.view())))
        return nx_err<nx_FileInfo>("sandbox denied stat");
    std::error_code ec;
    auto st = std::filesystem::status(path.c_str(), ec);
    if (ec || !std::filesystem::exists(st)) return nx_err<nx_FileInfo>("file does not exist");
    nx_FileInfo info;
    info.nx_is_dir = std::filesystem::is_directory(st);
    info.nx_is_file = std::filesystem::is_regular_file(st);
    info.nx_size = info.nx_is_file ? static_cast<std::int64_t>(std::filesystem::file_size(path.c_str(), ec)) : 0;
    auto ft = std::filesystem::last_write_time(path.c_str(), ec);
    info.nx_mtime = ec ? 0 : static_cast<std::int64_t>(ft.time_since_epoch().count());
    info.nx_readonly = (st.permissions() & std::filesystem::perms::owner_write) == std::filesystem::perms::none;
    return nx_ok(info);
}
static inline nx_result<bool> nx_fs_chmod(const NxStr &path, std::int64_t mode) {
    if (!NxSandbox::instance().check_fs("chmod", std::string(path.view())))
        return nx_err<bool>("sandbox denied chmod");
    std::error_code ec;
    std::filesystem::permissions(path.c_str(), static_cast<std::filesystem::perms>(mode & 0777), ec);
    if (ec) return nx_err<bool>(NxStr::from_std("chmod failed: " + ec.message()));
    return nx_ok(true);
}

class nx_File {
public:
    nx_File() = default;
    ~nx_File() { nx_close(); }
    nx_File(const nx_File &) = delete;
    nx_File &operator=(const nx_File &) = delete;
    nx_File(nx_File &&other) noexcept : fp_(other.fp_), writable_(other.writable_) { other.fp_ = nullptr; }
    nx_File &operator=(nx_File &&other) noexcept {
        if (this != &other) { nx_close(); fp_ = other.fp_; writable_ = other.writable_; other.fp_ = nullptr; }
        return *this;
    }

    static nx_File open(const NxStr &path, const NxStr &mode) {
        std::string m(mode.view());
        bool w = (m.find('w') != std::string::npos || m.find('a') != std::string::npos || m.find('+') != std::string::npos);
        nx_sandbox_require_fs(w ? "write" : "read", path);
        auto &sb = NxSandbox::instance();
        if (sb.open_files.load() >= sb.max_open_files) nx_fail("sandbox: max open files exceeded");
        std::string cmode = m;
        if (cmode.find('b') == std::string::npos) cmode.push_back('b');
        std::FILE *f = std::fopen(path.c_str(), cmode.c_str());
        if (!f) nx_fail_str("cannot open file: " + std::string(path.view()));
        sb.open_files.fetch_add(1, std::memory_order_relaxed);
        nx_File out;
        out.fp_ = f;
        out.writable_ = w;
        return out;
    }

    NxStr nx_read() {
        if (!fp_) nx_fail("read on closed File");
        long cur = std::ftell(fp_);
        std::fseek(fp_, 0, SEEK_END);
        long end = std::ftell(fp_);
        std::fseek(fp_, cur, SEEK_SET);
        if (end <= cur) return NxStr();
        std::size_t rem = static_cast<std::size_t>(end - cur);
        std::string buf(rem, '\0');
        std::size_t got = std::fread(buf.data(), 1, rem, fp_);
        buf.resize(got);
        return NxStr(buf.data(), buf.size(), false);
    }

    NxStr nx_read_line() {
        if (!fp_) nx_fail("read_line on closed File");
        std::string line;
        char buf[512];
        while (std::fgets(buf, sizeof(buf), fp_)) {
            line.append(buf);
            if (!line.empty() && line.back() == '\n') break;
        }
        return NxStr(line.data(), line.size(), false);
    }

    std::int64_t nx_write(const NxStr &text) {
        if (!fp_) nx_fail("write on closed File");
        if (text.empty()) return 0;
        std::size_t n = std::fwrite(text.c_str(), 1, text.size(), fp_);
        return static_cast<std::int64_t>(n);
    }

    void nx_flush() { if (fp_) std::fflush(fp_); }
    void nx_close() {
        if (fp_) {
            std::fclose(fp_);
            fp_ = nullptr;
            NxSandbox::instance().open_files.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    nx_File &nx___enter__() { return *this; }
    void nx___exit__() { nx_close(); }

private:
    std::FILE *fp_ = nullptr;
    bool writable_ = false;
};

// ============================================================================
// 11. Deep Cross-Platform Networking (nx.net)
// ============================================================================
#ifdef _WIN32
using nx_socket_t = SOCKET;
static constexpr nx_socket_t NX_INVALID_SOCK = INVALID_SOCKET;
static inline void nx_close_sock(nx_socket_t s) { if (s != INVALID_SOCKET) closesocket(s); }
static inline void nx_net_init_once() {
    static std::once_flag flag;
    std::call_once(flag, []() { WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa); });
}
#else
using nx_socket_t = int;
static constexpr nx_socket_t NX_INVALID_SOCK = -1;
static inline void nx_close_sock(nx_socket_t s) { if (s >= 0) close(s); }
static inline void nx_net_init_once() {}
#endif

static inline void nx_set_sock_timeout(nx_socket_t s, int timeout_ms) {
    if (timeout_ms <= 0) return;
#ifdef _WIN32
    DWORD tv = static_cast<DWORD>(timeout_ms);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&tv), sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&tv), sizeof(tv));
#else
    struct timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

static inline bool nx_tcp_connect(const NxStr &host, std::int32_t port) {
    if (port < 1 || port > 65535) return false;
    nx_sandbox_require_net("connect", host, port);
    nx_net_init_once();
    char service[8];
    std::snprintf(service, sizeof(service), "%d", static_cast<int>(port));
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *results = nullptr;
    bool connected = false;
    if (getaddrinfo(host.c_str(), service, &hints, &results) == 0) {
        for (addrinfo *it = results; it; it = it->ai_next) {
            nx_socket_t sock = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
            if (sock == NX_INVALID_SOCK) continue;
            if (connect(sock, it->ai_addr, static_cast<int>(it->ai_addrlen)) == 0) connected = true;
            nx_close_sock(sock);
            if (connected) break;
        }
        freeaddrinfo(results);
    }
    return connected;
}

class nx_TcpStream {
public:
    nx_TcpStream() = default;
    explicit nx_TcpStream(nx_socket_t fd, NxStr peer = NxStr()) : fd_(fd), peer_(std::move(peer)) {}
    ~nx_TcpStream() { nx_close(); }
    nx_TcpStream(const nx_TcpStream &) = delete;
    nx_TcpStream &operator=(const nx_TcpStream &) = delete;
    nx_TcpStream(nx_TcpStream &&other) noexcept : fd_(other.fd_), peer_(std::move(other.peer_)) {
        other.fd_ = NX_INVALID_SOCK;
    }
    nx_TcpStream &operator=(nx_TcpStream &&other) noexcept {
        if (this != &other) { nx_close(); fd_ = other.fd_; peer_ = std::move(other.peer_); other.fd_ = NX_INVALID_SOCK; }
        return *this;
    }

    static nx_result<nx_TcpStream> connect_to(const NxStr &host, std::int32_t port, std::int32_t timeout_ms = 5000) {
        if (port < 1 || port > 65535) return nx_err<nx_TcpStream>("invalid port");
        if (!NxSandbox::instance().check_net("connect", std::string(host.view()), port))
            return nx_err<nx_TcpStream>("sandbox denied tcp connect");
        auto &sb = NxSandbox::instance();
        if (sb.open_sockets.load() >= sb.max_sockets) return nx_err<nx_TcpStream>("sandbox: max sockets exceeded");
        nx_net_init_once();
        char service[8];
        std::snprintf(service, sizeof(service), "%d", static_cast<int>(port));
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo *results = nullptr;
        if (getaddrinfo(host.c_str(), service, &hints, &results) != 0)
            return nx_err<nx_TcpStream>("DNS lookup failed");
        nx_socket_t sock = NX_INVALID_SOCK;
        for (addrinfo *it = results; it; it = it->ai_next) {
            nx_socket_t cand = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
            if (cand == NX_INVALID_SOCK) continue;
            nx_set_sock_timeout(cand, timeout_ms);
            if (connect(cand, it->ai_addr, static_cast<int>(it->ai_addrlen)) == 0) {
                sock = cand;
                break;
            }
            nx_close_sock(cand);
        }
        freeaddrinfo(results);
        if (sock == NX_INVALID_SOCK) return nx_err<nx_TcpStream>("TCP connection failed");
        sb.open_sockets.fetch_add(1, std::memory_order_relaxed);
        return nx_ok(nx_TcpStream(sock, host + NxStr(":") + NxStr(service)));
    }

    std::int64_t nx_send(const NxStr &data) {
        if (fd_ == NX_INVALID_SOCK) nx_fail("send on closed TcpStream");
        const char *p = data.c_str();
        std::size_t rem = data.size();
        std::size_t sent_total = 0;
        while (rem > 0) {
            int n = send(fd_, p + sent_total, static_cast<int>(rem), 0);
            if (n <= 0) break;
            sent_total += static_cast<std::size_t>(n);
            rem -= static_cast<std::size_t>(n);
        }
        return static_cast<std::int64_t>(sent_total);
    }

    NxStr nx_recv(std::int64_t max_bytes = 4096) {
        if (fd_ == NX_INVALID_SOCK) nx_fail("recv on closed TcpStream");
        if (max_bytes <= 0) return NxStr();
        std::string buf(static_cast<std::size_t>(max_bytes), '\0');
        int n = recv(fd_, buf.data(), static_cast<int>(max_bytes), 0);
        if (n <= 0) return NxStr();
        return NxStr(buf.data(), static_cast<std::size_t>(n), false);
    }

    NxStr nx_recv_all(std::int64_t limit = 1048576) {
        if (fd_ == NX_INVALID_SOCK) nx_fail("recv_all on closed TcpStream");
        std::string out;
        char chunk[4096];
        while (static_cast<std::int64_t>(out.size()) < limit) {
            int n = recv(fd_, chunk, sizeof(chunk), 0);
            if (n <= 0) break;
            out.append(chunk, static_cast<std::size_t>(n));
        }
        return NxStr(out.data(), out.size(), false);
    }

    void nx_set_timeout(std::int32_t ms) {
        if (fd_ != NX_INVALID_SOCK) nx_set_sock_timeout(fd_, ms);
    }

    NxStr nx_peer_addr() const { return peer_; }

    void nx_close() {
        if (fd_ != NX_INVALID_SOCK) {
            nx_close_sock(fd_);
            fd_ = NX_INVALID_SOCK;
            NxSandbox::instance().open_sockets.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    nx_socket_t raw_fd() const noexcept { return fd_; }

private:
    nx_socket_t fd_ = NX_INVALID_SOCK;
    NxStr peer_;
};

class nx_TcpListener {
public:
    nx_TcpListener() = default;
    ~nx_TcpListener() { nx_close(); }
    nx_TcpListener(const nx_TcpListener &) = delete;
    nx_TcpListener &operator=(const nx_TcpListener &) = delete;
    nx_TcpListener(nx_TcpListener &&other) noexcept : fd_(other.fd_), port_(other.port_) {
        other.fd_ = NX_INVALID_SOCK;
    }
    nx_TcpListener &operator=(nx_TcpListener &&other) noexcept {
        if (this != &other) { nx_close(); fd_ = other.fd_; port_ = other.port_; other.fd_ = NX_INVALID_SOCK; }
        return *this;
    }

    static nx_result<nx_TcpListener> bind_addr(const NxStr &host, std::int32_t port) {
        if (port < 0 || port > 65535) return nx_err<nx_TcpListener>("invalid port");
        if (!NxSandbox::instance().check_net("listen", std::string(host.view()), port))
            return nx_err<nx_TcpListener>("sandbox denied tcp listen");
        nx_net_init_once();
        char service[8];
        std::snprintf(service, sizeof(service), "%d", static_cast<int>(port));
        addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE;
        addrinfo *results = nullptr;
        if (getaddrinfo(host.empty() ? nullptr : host.c_str(), service, &hints, &results) != 0)
            return nx_err<nx_TcpListener>("getaddrinfo failed for bind");
        nx_socket_t sock = NX_INVALID_SOCK;
        for (addrinfo *it = results; it; it = it->ai_next) {
            sock = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
            if (sock == NX_INVALID_SOCK) continue;
            int opt = 1;
            setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&opt), sizeof(opt));
            if (bind(sock, it->ai_addr, static_cast<int>(it->ai_addrlen)) == 0 && listen(sock, 64) == 0) {
                break;
            }
            nx_close_sock(sock);
            sock = NX_INVALID_SOCK;
        }
        freeaddrinfo(results);
        if (sock == NX_INVALID_SOCK) return nx_err<nx_TcpListener>("bind/listen failed");
        sockaddr_in sa{};
        socklen_t slen = sizeof(sa);
        std::int32_t bound_port = port;
        if (getsockname(sock, reinterpret_cast<sockaddr *>(&sa), &slen) == 0) {
            bound_port = ntohs(sa.sin_port);
        }
        NxSandbox::instance().open_sockets.fetch_add(1, std::memory_order_relaxed);
        nx_TcpListener lst;
        lst.fd_ = sock;
        lst.port_ = bound_port;
        return nx_ok(std::move(lst));
    }

    nx_result<nx_TcpStream> nx_accept() {
        if (fd_ == NX_INVALID_SOCK) return nx_err<nx_TcpStream>("accept on closed listener");
        sockaddr_in cli{};
        socklen_t clen = sizeof(cli);
        nx_socket_t conn = accept(fd_, reinterpret_cast<sockaddr *>(&cli), &clen);
        if (conn == NX_INVALID_SOCK) return nx_err<nx_TcpStream>("accept failed");
        NxSandbox::instance().open_sockets.fetch_add(1, std::memory_order_relaxed);
        char ip[64] = {};
        inet_ntop(AF_INET, &cli.sin_addr, ip, sizeof(ip));
        return nx_ok(nx_TcpStream(conn, NxStr(ip) + NxStr(":") + NxStr(std::to_string(ntohs(cli.sin_port)).c_str())));
    }

    std::int32_t nx_local_port() const noexcept { return port_; }

    void nx_close() {
        if (fd_ != NX_INVALID_SOCK) {
            nx_close_sock(fd_);
            fd_ = NX_INVALID_SOCK;
            NxSandbox::instance().open_sockets.fetch_sub(1, std::memory_order_relaxed);
        }
    }

private:
    nx_socket_t fd_ = NX_INVALID_SOCK;
    std::int32_t port_ = 0;
};

static inline nx_result<nx_vec<NxStr>> nx_dns_resolve(const NxStr &host) {
    if (!NxSandbox::instance().check_net("dns", std::string(host.view()), 0))
        return nx_err<nx_vec<NxStr>>("sandbox denied dns lookup");
    nx_net_init_once();
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *res = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0)
        return nx_err<nx_vec<NxStr>>("DNS resolve failed");
    nx_vec<NxStr> ips;
    for (addrinfo *it = res; it; it = it->ai_next) {
        char buf[INET6_ADDRSTRLEN] = {};
        if (it->ai_family == AF_INET) {
            auto *sa = reinterpret_cast<sockaddr_in *>(it->ai_addr);
            inet_ntop(AF_INET, &sa->sin_addr, buf, sizeof(buf));
        } else if (it->ai_family == AF_INET6) {
            auto *sa = reinterpret_cast<sockaddr_in6 *>(it->ai_addr);
            inet_ntop(AF_INET6, &sa->sin6_addr, buf, sizeof(buf));
        }
        if (buf[0]) {
            NxStr ip(buf);
            if (!nx_list_contains(ips, ip)) ips.push_back(ip);
        }
    }
    freeaddrinfo(res);
    return nx_ok(std::move(ips));
}

struct nx_HttpResponse {
    std::int32_t nx_status = 0;
    NxStr nx_body;
    NxStr nx_headers;
    bool nx_ok_status() const noexcept { return nx_status >= 200 && nx_status < 300; }
    NxStr nx___str__() const {
        NxStrBuilder sb;
        sb.append("HttpResponse(status=").append(nx_status).append(", len=").append(static_cast<std::int64_t>(nx_body.size())).append(")");
        return sb.finish();
    }
};

static inline nx_result<nx_HttpResponse> nx_https_request_native(
    const NxStr &method, const std::string &host, int port, const std::string &path,
    const NxStr &body, const NxStr &content_type) {
    if (!NxSandbox::instance().check_net("https", host, port))
        return nx_err<nx_HttpResponse>("sandbox denied https request");
#ifdef _WIN32
    HMODULE wh = LoadLibraryA("winhttp.dll");
    if (!wh) return nx_err<nx_HttpResponse>("failed to load winhttp.dll for native HTTPS");
    using HINTERNET = void *;
    using FnOpen = HINTERNET(WINAPI *)(const wchar_t *, unsigned long, const wchar_t *, const wchar_t *, unsigned long);
    using FnConnect = HINTERNET(WINAPI *)(HINTERNET, const wchar_t *, unsigned short, unsigned long);
    using FnOpenReq = HINTERNET(WINAPI *)(HINTERNET, const wchar_t *, const wchar_t *, const wchar_t *, const wchar_t *, const wchar_t **, unsigned long);
    using FnSendReq = int(WINAPI *)(HINTERNET, const wchar_t *, unsigned long, void *, unsigned long, unsigned long, std::uintptr_t);
    using FnRecvResp = int(WINAPI *)(HINTERNET, void *);
    using FnQueryHdr = int(WINAPI *)(HINTERNET, unsigned long, const wchar_t *, void *, unsigned long *, unsigned long *);
    using FnQueryAvail = int(WINAPI *)(HINTERNET, unsigned long *);
    using FnReadData = int(WINAPI *)(HINTERNET, void *, unsigned long, unsigned long *);
    using FnClose = int(WINAPI *)(HINTERNET);

    auto pOpen = reinterpret_cast<FnOpen>(reinterpret_cast<void *>(GetProcAddress(wh, "WinHttpOpen")));
    auto pConnect = reinterpret_cast<FnConnect>(reinterpret_cast<void *>(GetProcAddress(wh, "WinHttpConnect")));
    auto pOpenReq = reinterpret_cast<FnOpenReq>(reinterpret_cast<void *>(GetProcAddress(wh, "WinHttpOpenRequest")));
    auto pSendReq = reinterpret_cast<FnSendReq>(reinterpret_cast<void *>(GetProcAddress(wh, "WinHttpSendRequest")));
    auto pRecvResp = reinterpret_cast<FnRecvResp>(reinterpret_cast<void *>(GetProcAddress(wh, "WinHttpReceiveResponse")));
    auto pQueryHdr = reinterpret_cast<FnQueryHdr>(reinterpret_cast<void *>(GetProcAddress(wh, "WinHttpQueryHeaders")));
    auto pQueryAvail = reinterpret_cast<FnQueryAvail>(reinterpret_cast<void *>(GetProcAddress(wh, "WinHttpQueryDataAvailable")));
    auto pReadData = reinterpret_cast<FnReadData>(reinterpret_cast<void *>(GetProcAddress(wh, "WinHttpReadData")));
    auto pClose = reinterpret_cast<FnClose>(reinterpret_cast<void *>(GetProcAddress(wh, "WinHttpCloseHandle")));
    if (!pOpen || !pConnect || !pOpenReq || !pSendReq || !pRecvResp || !pQueryHdr || !pQueryAvail || !pReadData || !pClose) {
        FreeLibrary(wh);
        return nx_err<nx_HttpResponse>("winhttp.dll missing required symbols");
    }
    auto to_wide = [](const std::string &s) -> std::wstring {
        if (s.empty()) return std::wstring();
        int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
        std::wstring w(static_cast<std::size_t>(n > 0 ? n : 0), L'\0');
        if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), w.data(), n);
        return w;
    };
    std::wstring wagent = L"NEXIR/2.0";
    std::wstring whost = to_wide(host);
    std::wstring wpath = to_wide(path);
    std::wstring wmeth = to_wide(std::string(method.view()));
    std::wstring whdr;
    if (!body.empty()) {
        whdr = L"Content-Type: " + to_wide(std::string(content_type.view())) + L"\r\n";
    }
    HINTERNET hSession = pOpen(wagent.c_str(), 0 /* WINHTTP_ACCESS_TYPE_DEFAULT_PROXY */, nullptr, nullptr, 0);
    if (!hSession) { FreeLibrary(wh); return nx_err<nx_HttpResponse>("WinHttpOpen failed"); }
    HINTERNET hConnect = pConnect(hSession, whost.c_str(), static_cast<unsigned short>(port), 0);
    if (!hConnect) { pClose(hSession); FreeLibrary(wh); return nx_err<nx_HttpResponse>("WinHttpConnect failed"); }
    constexpr unsigned long kSecure = 0x00800000UL;
    HINTERNET hReq = pOpenReq(hConnect, wmeth.c_str(), wpath.c_str(), nullptr, nullptr, nullptr, kSecure);
    if (!hReq) { pClose(hConnect); pClose(hSession); FreeLibrary(wh); return nx_err<nx_HttpResponse>("WinHttpOpenRequest failed"); }
    void *bptr = body.empty() ? nullptr : const_cast<char *>(body.c_str());
    unsigned long blen = static_cast<unsigned long>(body.size());
    int ok = pSendReq(hReq, whdr.empty() ? nullptr : whdr.c_str(), static_cast<unsigned long>(whdr.size()), bptr, blen, blen, 0);
    if (ok) ok = pRecvResp(hReq, nullptr);
    if (!ok) {
        pClose(hReq); pClose(hConnect); pClose(hSession); FreeLibrary(wh);
        return nx_err<nx_HttpResponse>("HTTPS request failed");
    }
    unsigned long status_code = 0;
    unsigned long slen = sizeof(status_code);
    constexpr unsigned long kStatusNum = 19 | 0x20000000UL; // WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER
    pQueryHdr(hReq, kStatusNum, nullptr, &status_code, &slen, nullptr);

    std::string resp_body;
    unsigned long avail = 0;
    while (pQueryAvail(hReq, &avail) && avail > 0) {
        std::string chunk(avail, '\0');
        unsigned long read_bytes = 0;
        if (!pReadData(hReq, chunk.data(), avail, &read_bytes) || read_bytes == 0) break;
        chunk.resize(read_bytes);
        resp_body += chunk;
    }
    pClose(hReq); pClose(hConnect); pClose(hSession); FreeLibrary(wh);
    nx_HttpResponse resp;
    resp.nx_status = static_cast<std::int32_t>(status_code);
    resp.nx_headers = NxStr("HTTPS/WinHTTP");
    resp.nx_body = NxStr::from_std(resp_body);
    return nx_ok(std::move(resp));
#else
    // POSIX native TLS via OpenSSL (libssl) loaded dynamically
    void *ssl_lib = dlopen("libssl.so.3", RTLD_LAZY);
    if (!ssl_lib) ssl_lib = dlopen("libssl.so", RTLD_LAZY);
    if (!ssl_lib) ssl_lib = dlopen("libssl.dylib", RTLD_LAZY);
    if (!ssl_lib) return nx_err<nx_HttpResponse>("HTTPS requires OpenSSL libssl on POSIX");
    using FnCtxNew = void *(*)(const void *);
    using FnMethod = const void *(*)();
    using FnCtxFree = void (*)(void *);
    using FnSslNew = void *(*)(void *);
    using FnSslFree = void (*)(void *);
    using FnSetFd = int (*)(void *, int);
    using FnConnect = int (*)(void *);
    using FnWrite = int (*)(void *, const void *, int);
    using FnRead = int (*)(void *, void *, int);
    using FnCtrl = long (*)(void *, int, long, void *);
    using FnSetDefaultVerify = int (*)(void *);
    using FnSetVerify = void (*)(void *, int, void *);
    using FnSet1Host = int (*)(void *, const char *);
    using FnGetVerifyResult = long (*)(void *);
    auto pMethod = reinterpret_cast<FnMethod>(dlsym(ssl_lib, "TLS_client_method"));
    auto pCtxNew = reinterpret_cast<FnCtxNew>(dlsym(ssl_lib, "SSL_CTX_new"));
    auto pCtxFree = reinterpret_cast<FnCtxFree>(dlsym(ssl_lib, "SSL_CTX_free"));
    auto pSslNew = reinterpret_cast<FnSslNew>(dlsym(ssl_lib, "SSL_new"));
    auto pSslFree = reinterpret_cast<FnSslFree>(dlsym(ssl_lib, "SSL_free"));
    auto pSetFd = reinterpret_cast<FnSetFd>(dlsym(ssl_lib, "SSL_set_fd"));
    auto pConnect = reinterpret_cast<FnConnect>(dlsym(ssl_lib, "SSL_connect"));
    auto pWrite = reinterpret_cast<FnWrite>(dlsym(ssl_lib, "SSL_write"));
    auto pRead = reinterpret_cast<FnRead>(dlsym(ssl_lib, "SSL_read"));
    auto pCtrl = reinterpret_cast<FnCtrl>(dlsym(ssl_lib, "SSL_ctrl"));
    auto pSetDefaultVerify = reinterpret_cast<FnSetDefaultVerify>(dlsym(ssl_lib, "SSL_CTX_set_default_verify_paths"));
    auto pSetVerify = reinterpret_cast<FnSetVerify>(dlsym(ssl_lib, "SSL_set_verify"));
    auto pSet1Host = reinterpret_cast<FnSet1Host>(dlsym(ssl_lib, "SSL_set1_host"));
    auto pGetVerifyResult = reinterpret_cast<FnGetVerifyResult>(dlsym(ssl_lib, "SSL_get_verify_result"));
    if (!pMethod || !pCtxNew || !pCtxFree || !pSslNew || !pSslFree || !pSetFd || !pConnect || !pWrite || !pRead || !pSetDefaultVerify || !pSetVerify || !pSet1Host || !pGetVerifyResult) {
        dlclose(ssl_lib);
        return nx_err<nx_HttpResponse>("OpenSSL missing required TLS/verification symbols");
    }
    auto conn = nx_TcpStream::connect_to(NxStr::from_std(host), port, 5000);
    if (conn.is_err()) { dlclose(ssl_lib); return nx_err<nx_HttpResponse>(conn.error()); }
    auto stream = conn.value();
    void *ctx = pCtxNew(pMethod());
    if (ctx) pSetDefaultVerify(ctx);
    void *ssl = ctx ? pSslNew(ctx) : nullptr;
    if (!ssl) { if (ctx) pCtxFree(ctx); stream.nx_close(); dlclose(ssl_lib); return nx_err<nx_HttpResponse>("SSL_new failed"); }
    pSetVerify(ssl, 1 /* SSL_VERIFY_PEER */, nullptr);
    pSet1Host(ssl, host.c_str());
    if (pCtrl) pCtrl(ssl, 55 /* SSL_CTRL_SET_TLSEXT_HOSTNAME */, 0, const_cast<char *>(host.c_str()));
    pSetFd(ssl, static_cast<int>(stream.raw_fd()));
    if (pConnect(ssl) <= 0 || pGetVerifyResult(ssl) != 0 /* X509_V_OK */) {
        pSslFree(ssl); pCtxFree(ctx); stream.nx_close(); dlclose(ssl_lib);
        return nx_err<nx_HttpResponse>("TLS handshake or certificate/hostname verification failed");
    }
    NxStrBuilder req;
    req.append(method).append(" ").append(path).append(" HTTP/1.1\r\nHost: ").append(host).append("\r\nConnection: close\r\nUser-Agent: NEXIR/2.0\r\n");
    if (!body.empty()) {
        req.append("Content-Type: ").append(content_type).append("\r\nContent-Length: ").append(static_cast<std::int64_t>(body.size())).append("\r\n");
    }
    req.append("\r\n");
    if (!body.empty()) req.append(body);
    NxStr rstr = req.finish();
    pWrite(ssl, rstr.c_str(), static_cast<int>(rstr.size()));
    std::string raw;
    char buf[4096];
    int nr = 0;
    while ((nr = pRead(ssl, buf, sizeof(buf))) > 0) raw.append(buf, static_cast<std::size_t>(nr));
    pSslFree(ssl); pCtxFree(ctx); stream.nx_close(); dlclose(ssl_lib);
    std::size_t hdr_end = raw.find("\r\n\r\n");
    if (hdr_end == std::string::npos) return nx_err<nx_HttpResponse>("malformed HTTPS response");
    std::string hdr = raw.substr(0, hdr_end);
    std::string bdy = raw.substr(hdr_end + 4);
    int code = 0;
    std::size_t sp = hdr.find(' ');
    if (sp != std::string::npos) code = std::atoi(hdr.c_str() + sp + 1);
    nx_HttpResponse resp;
    resp.nx_status = code;
    resp.nx_headers = NxStr::from_std(hdr);
    resp.nx_body = NxStr::from_std(bdy);
    return nx_ok(std::move(resp));
#endif
}

static inline nx_result<nx_HttpResponse> nx_http_request(const NxStr &method, const NxStr &url,
                                                         const NxStr &body = NxStr(),
                                                         const NxStr &content_type = NxStr("application/json")) {
    std::string u(url.view());
    bool is_https = u.rfind("https://", 0) == 0;
    bool is_http = u.rfind("http://", 0) == 0;
    if (!is_http && !is_https) return nx_err<nx_HttpResponse>("nx_http_request supports http:// and https:// URLs natively");
    std::string rest = u.substr(is_https ? 8 : 7);
    std::size_t slash = rest.find('/');
    std::string host_port = slash == std::string::npos ? rest : rest.substr(0, slash);
    std::string path = slash == std::string::npos ? "/" : rest.substr(slash);
    std::size_t colon = host_port.find(':');
    std::string host = colon == std::string::npos ? host_port : host_port.substr(0, colon);
    int port = colon == std::string::npos ? (is_https ? 443 : 80) : std::atoi(host_port.substr(colon + 1).c_str());
    if (is_https) {
        return nx_https_request_native(method, host, port, path, body, content_type);
    }

    auto conn = nx_TcpStream::connect_to(NxStr::from_std(host), port, 5000);
    if (conn.is_err()) return nx_err<nx_HttpResponse>(conn.error());
    auto stream = conn.value();

    NxStrBuilder req;
    req.append(method).append(" ").append(path).append(" HTTP/1.1\r\n")
       .append("Host: ").append(host_port).append("\r\n")
       .append("Connection: close\r\n")
       .append("User-Agent: NEXIR/2.0\r\n");
    if (!body.empty()) {
        req.append("Content-Type: ").append(content_type).append("\r\n")
           .append("Content-Length: ").append(static_cast<std::int64_t>(body.size())).append("\r\n");
    }
    req.append("\r\n");
    if (!body.empty()) req.append(body);
    stream.nx_send(req.finish());
    NxStr raw = stream.nx_recv_all();
    stream.nx_close();

    std::string rs(raw.view());
    std::size_t hdr_end = rs.find("\r\n\r\n");
    if (hdr_end == std::string::npos) return nx_err<nx_HttpResponse>("malformed HTTP response");
    std::string hdr = rs.substr(0, hdr_end);
    std::string bdy = rs.substr(hdr_end + 4);
    int code = 0;
    std::size_t sp = hdr.find(' ');
    if (sp != std::string::npos) code = std::atoi(hdr.c_str() + sp + 1);
    nx_HttpResponse resp;
    resp.nx_status = code;
    resp.nx_headers = NxStr::from_std(hdr);
    resp.nx_body = NxStr::from_std(bdy);
    return nx_ok(std::move(resp));
}
static inline nx_result<nx_HttpResponse> nx_http_get(const NxStr &url) {
    return nx_http_request(NxStr("GET"), url);
}
static inline nx_result<nx_HttpResponse> nx_http_post(const NxStr &url, const NxStr &body,
                                                      const NxStr &content_type = NxStr("application/json")) {
    return nx_http_request(NxStr("POST"), url, body, content_type);
}
'''
