"""Compile-time Security Gate: Capabilities, FFI policy, Sandbox defaults, and Audit Log."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time


CAPABILITIES = {
    "memory.alloc",
    "filesystem.read",
    "filesystem.write",
    "filesystem.copy",
    "filesystem.delete",
    "filesystem.admin",
    "network.connect",
    "network.listen",
    "network.dns",
    "ffi.c",
    "ffi.cpp",
    "ffi.rust",
    "ffi.go",
    "python.interop",
    "gpu.compute",
    "process.exec",
    "env.read",
}

DANGEROUS_CPP_HEADERS = {
    "windows.h", "winbase.h", "unistd.h", "sys/syscall.h", "spawn.h",
    "dlfcn.h", "process.h", "fcntl.h", "sys/ptrace.h",
}

DANGEROUS_FFI_SYMBOLS = {
    "system", "popen", "pclose", "execv", "execve", "execl", "execlp",
    "WinExec", "ShellExecuteA", "ShellExecuteW", "CreateProcessA", "CreateProcessW",
    "gets", "strcpy", "strcat", "sprintf", "vsprintf",
}

DANGEROUS_PY_MODULES = {
    "os", "subprocess", "ctypes", "shutil", "socket", "importlib", "pty", "fcntl",
}

HIGH_RISK_CAPABILITIES = {
    "ffi.c", "ffi.cpp", "ffi.rust", "ffi.go", "python.interop",
    "process.exec", "filesystem.delete", "filesystem.admin", "network.listen",
}


@dataclass
class AuditRecord:
    ts_ms: int
    event: str
    kind: str
    name: str
    path: str
    sha256: str
    capabilities: list[str]
    status: str


class CompileAuditLog:
    """Tracks every module/FFI binding loaded during compilation."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def record(self, event: str, kind: str, name: str, path: str = "",
               sha256: str = "", capabilities: list[str] | None = None,
               status: str = "allowed") -> None:
        self.records.append(AuditRecord(
            ts_ms=int(time.time() * 1000),
            event=event,
            kind=kind,
            name=name,
            path=path,
            sha256=sha256,
            capabilities=sorted(set(capabilities or ())),
            status=status,
        ))

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps({
            "ts_ms": r.ts_ms,
            "event": r.event,
            "kind": r.kind,
            "name": r.name,
            "path": r.path,
            "sha256": r.sha256,
            "capabilities": r.capabilities,
            "status": r.status,
        }) for r in self.records) + ("\n" if self.records else "")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_cpp_header(raw_header: str, *, allow_unsafe: bool = False) -> tuple[str, bool]:
    local = raw_header.startswith('"') and raw_header.endswith('"')
    path = raw_header[1:-1] if local else raw_header
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", path) or ".." in path or path.startswith("/"):
        raise ValueError("invalid C++ header path")
    norm = path.lower()
    if (norm in DANGEROUS_CPP_HEADERS or f"{norm}.h" in DANGEROUS_CPP_HEADERS) and not allow_unsafe:
        raise ValueError(f"FFI security policy blocks dangerous header '{path}' without @cap(\"process.exec\")")
    return path, local


def validate_ffi_symbol(symbol: str, *, allow_process: bool = False) -> None:
    leaf = symbol.split("::")[-1]
    if leaf in DANGEROUS_FFI_SYMBOLS and not allow_process:
        raise ValueError(f"FFI security policy blocks unsafe/process symbol '{symbol}' without @cap(\"process.exec\")")


def validate_python_module(module_name: str, *, strict_sandbox: bool = False,
                           allowlist: set[str] | None = None) -> None:
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module_name):
        raise ValueError(f"invalid Python module name: {module_name}")
    root = module_name.split(".")[0]
    if allowlist is not None and module_name not in allowlist and root not in allowlist:
        raise ValueError(f"sandbox blocks Python module '{module_name}' (not in allowlist)")
    if strict_sandbox and root in DANGEROUS_PY_MODULES:
        raise ValueError(f"sandbox blocks privileged Python module '{module_name}'")
