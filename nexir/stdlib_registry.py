"""Standard library module registry for `import nx.<module>`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StdFunc:
    cpp_expr: str       # template or function name; if contains '{args}' uses format, else fn(args)
    params: tuple[str, ...]  # empty or pattern; None-like special handled in compiler
    result: str
    caps: frozenset[str] = frozenset()
    cpp_header: str | None = None
    variadic: bool = False


STD_MODULES: dict[str, dict[str, StdFunc]] = {
    "nx.asyncio": {
        "sleep": StdFunc("nx_async_sleep", ("i64",), "Future[void]"),
        "read_text": StdFunc("nx_async_read_text", ("str",), "Future[Result[str]]", frozenset({"filesystem.read", "memory.alloc"})),
        "tcp_connect": StdFunc("nx_async_tcp_connect", ("str", "i32"), "Future[bool]", frozenset({"network.connect"})),
        "http_get": StdFunc("nx_async_http_get", ("str",), "Future[Result[HttpResponse]]", frozenset({"network.connect", "memory.alloc"})),
        "configure": StdFunc("nx_asyncio_configure", ("i64", "bool"), "void"),
        "workers": StdFunc("nx_asyncio_workers", (), "i64"),
        "run": StdFunc("nx_async_run", ("ANY",), "UNWRAP_FUTURE"),
        "gather": StdFunc("nx_async_gather", ("ANY",), "GATHER", variadic=True),
    },
    "nx.thread": {
        "cpu_count": StdFunc("nx_thread_cpu_count", (), "i64"),
    },
    "nx.fs": {
        "read_text": StdFunc("nx_fs_read_text", ("str",), "Result[str]", frozenset({"filesystem.read", "memory.alloc"})),
        "read_bytes": StdFunc("nx_fs_read_bytes", ("str",), "Result[List[u8]]", frozenset({"filesystem.read", "memory.alloc"})),
        "write_text": StdFunc("nx_fs_write_text", ("str", "str"), "Result[bool]", frozenset({"filesystem.write"})),
        "write_bytes": StdFunc("nx_fs_write_bytes", ("str", "List[u8]"), "Result[bool]", frozenset({"filesystem.write"})),
        "append_text": StdFunc("nx_fs_append_text", ("str", "str"), "Result[bool]", frozenset({"filesystem.write"})),
        "copy_file": StdFunc("nx_fs_copy_file", ("str", "str"), "Result[bool]", frozenset({"filesystem.read", "filesystem.copy"})),
        "move_file": StdFunc("nx_fs_move_file", ("str", "str"), "Result[bool]", frozenset({"filesystem.write", "filesystem.delete"})),
        "delete_file": StdFunc("nx_fs_delete_file", ("str",), "Result[bool]", frozenset({"filesystem.delete"})),
        "edit_file": StdFunc("nx_fs_edit_file", ("str", "str", "str"), "Result[bool]", frozenset({"filesystem.read", "filesystem.write", "memory.alloc"})),
        "exists": StdFunc("nx_fs_exists", ("str",), "bool", frozenset({"filesystem.read"})),
        "is_file": StdFunc("nx_fs_is_file", ("str",), "bool", frozenset({"filesystem.read"})),
        "is_dir": StdFunc("nx_fs_is_dir", ("str",), "bool", frozenset({"filesystem.read"})),
        "mkdir": StdFunc("nx_fs_mkdir", ("str",), "Result[bool]", frozenset({"filesystem.write"})),
        "list_dir": StdFunc("nx_fs_list_dir", ("str",), "Result[List[str]]", frozenset({"filesystem.read", "memory.alloc"})),
        "stat": StdFunc("nx_fs_stat", ("str",), "Result[FileInfo]", frozenset({"filesystem.read"})),
        "chmod": StdFunc("nx_fs_chmod", ("str", "i64"), "Result[bool]", frozenset({"filesystem.admin"})),
        "open": StdFunc("nx_File::open", ("str", "str"), "File", frozenset({"filesystem.read"})),
    },
    "nx.io": {
        "open": StdFunc("nx_File::open", ("str", "str"), "File", frozenset({"filesystem.read"})),
        "read_text": StdFunc("nx_fs_read_text", ("str",), "Result[str]", frozenset({"filesystem.read", "memory.alloc"})),
        "write_text": StdFunc("nx_fs_write_text", ("str", "str"), "Result[bool]", frozenset({"filesystem.write"})),
        "copy_file": StdFunc("nx_fs_copy_file", ("str", "str"), "Result[bool]", frozenset({"filesystem.read", "filesystem.copy"})),
        "edit_file": StdFunc("nx_fs_edit_file", ("str", "str", "str"), "Result[bool]", frozenset({"filesystem.read", "filesystem.write", "memory.alloc"})),
        "delete_file": StdFunc("nx_fs_delete_file", ("str",), "Result[bool]", frozenset({"filesystem.delete"})),
    },
    "nx.net": {
        "tcp_connect": StdFunc("nx_tcp_connect", ("str", "i32"), "bool", frozenset({"network.connect"})),
        "connect": StdFunc("nx_TcpStream::connect_to", ("str", "i32"), "Result[TcpStream]", frozenset({"network.connect"})),
        "listen": StdFunc("nx_TcpListener::bind_addr", ("str", "i32"), "Result[TcpListener]", frozenset({"network.listen"})),
        "dns_resolve": StdFunc("nx_dns_resolve", ("str",), "Result[List[str]]", frozenset({"network.dns", "memory.alloc"})),
        "http_get": StdFunc("nx_http_get", ("str",), "Result[HttpResponse]", frozenset({"network.connect", "memory.alloc"})),
        "http_post": StdFunc("nx_http_post", ("str", "str"), "Result[HttpResponse]", frozenset({"network.connect", "memory.alloc"})),
        "http_request": StdFunc("nx_http_request", ("str", "str", "str"), "Result[HttpResponse]", frozenset({"network.connect", "memory.alloc"})),
    },
    "nx.gpu": {
        "backend_name": StdFunc("nx_gpu_backend_name", (), "str", frozenset({"gpu.compute"})),
        "is_available": StdFunc("nx_gpu_is_available", (), "bool", frozenset({"gpu.compute"})),
        "device_count": StdFunc("nx_gpu_device_count", (), "i64", frozenset({"gpu.compute"})),
        "to_device": StdFunc("nx_gpu_to_device", ("ANY_LIST",), "GPU_BUF_FROM_LIST", frozenset({"gpu.compute", "memory.alloc"})),
        "to_host": StdFunc("nx_gpu_to_host", ("ANY_GPU_BUF",), "LIST_FROM_GPU_BUF", frozenset({"gpu.compute", "memory.alloc"})),
        "add": StdFunc("nx_gpu_add", ("ANY_GPU_BUF", "ANY_GPU_BUF"), "SAME_AS_ARG0", frozenset({"gpu.compute", "memory.alloc"})),
        "mul": StdFunc("nx_gpu_mul", ("ANY_GPU_BUF", "ANY_GPU_BUF"), "SAME_AS_ARG0", frozenset({"gpu.compute", "memory.alloc"})),
        "reduce_sum": StdFunc("nx_gpu_reduce_sum", ("ANY_GPU_BUF",), "ELEM_OF_GPU_BUF", frozenset({"gpu.compute"})),
        "matmul": StdFunc("nx_gpu_matmul", ("List[f64]", "List[f64]", "i64", "i64", "i64"), "List[f64]", frozenset({"gpu.compute", "memory.alloc"})),
        "matmul_async": StdFunc("nx_gpu_matmul_async", ("List[f64]", "List[f64]", "i64", "i64", "i64"), "Future[List[f64]]", frozenset({"gpu.compute", "memory.alloc"})),
    },
    "nx.sandbox": {
        "enable": StdFunc("nx_sandbox_enable", (), "void"),
        "set_max_memory": StdFunc("nx_sandbox_set_max_memory", ("i64",), "void"),
        "allow_read": StdFunc("nx_sandbox_allow_read", ("str",), "void"),
        "allow_write": StdFunc("nx_sandbox_allow_write", ("str",), "void"),
        "allow_host": StdFunc("nx_sandbox_allow_host", ("str",), "void"),
        "deny_network": StdFunc("nx_sandbox_deny_network", (), "void"),
        "set_audit_log": StdFunc("nx_sandbox_set_audit_log", ("str",), "void"),
        "audit_log": StdFunc("nx_sandbox_audit_log", (), "List[str]", frozenset({"memory.alloc"})),
    },
    "nx.debug": {
        "active_bytes": StdFunc("nx_debug_active_bytes", (), "i64"),
        "peak_bytes": StdFunc("nx_debug_peak_bytes", (), "i64"),
        "cache_hits": StdFunc("nx_debug_cache_hits", (), "i64"),
        "intern_hits": StdFunc("nx_debug_intern_hits", (), "i64"),
        "live_blocks": StdFunc("nx_debug_live_blocks", (), "i64"),
        "mem_report": StdFunc("nx_debug_mem_report", (), "str", frozenset({"memory.alloc"})),
    },
    "nx.math": {
        "sqrt": StdFunc("std::sqrt", ("f64",), "f64"),
        "sin": StdFunc("std::sin", ("f64",), "f64"),
        "cos": StdFunc("std::cos", ("f64",), "f64"),
        "tan": StdFunc("std::tan", ("f64",), "f64"),
        "floor": StdFunc("std::floor", ("f64",), "f64"),
        "ceil": StdFunc("std::ceil", ("f64",), "f64"),
        "pow": StdFunc("std::pow", ("f64", "f64"), "f64"),
        "log": StdFunc("std::log", ("f64",), "f64"),
        "exp": StdFunc("std::exp", ("f64",), "f64"),
        "fabs": StdFunc("std::fabs", ("f64",), "f64"),
    },
    "nx.sys": {
        "platform": StdFunc('NxStr([](){ return sizeof(void*) == 8 ? "64bit" : "32bit"; }())', (), "str"),
    },
}

# Known safe C++ STL / standard headers and common functions for direct `import cpp.<header>` without wrapper files
CPP_KNOWN_HEADERS: dict[str, dict[str, tuple[tuple[str, ...], str]]] = {
    "cmath": {
        "hypot": (("f64", "f64"), "f64"),
        "sqrt": (("f64",), "f64"),
        "sin": (("f64",), "f64"),
        "cos": (("f64",), "f64"),
        "tan": (("f64",), "f64"),
        "pow": (("f64", "f64"), "f64"),
        "exp": (("f64",), "f64"),
        "log": (("f64",), "f64"),
        "floor": (("f64",), "f64"),
        "ceil": (("f64",), "f64"),
        "fabs": (("f64",), "f64"),
        "fmod": (("f64", "f64"), "f64"),
    },
    "cstdlib": {
        "abs": (("i64",), "i64"),
        "llabs": (("i64",), "i64"),
    },
    "algorithm": {},
    "numeric": {},
}
