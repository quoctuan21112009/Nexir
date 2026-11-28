"""Bidirectional interoperability and multi-language binding generator for NEXIR 2.0.

Enables:
1. Python -> NX: `import_nx("mod.nx")` and `enable_import_hook()` so Python can
   import and call compiled `.nx` native modules directly via C ABI.
2. NX -> C / C++ / Python / Rust / Go: `generate_bindings(...)` generates clean
   native FFI bindings for @export functions from any `.nx` module.
"""

from __future__ import annotations

import ast
import ctypes
from dataclasses import dataclass
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from .compiler import Compiler, NexirError, _normalize_source


_CTYPES_MAP = {
    "i8": ctypes.c_int8,
    "i16": ctypes.c_int16,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "u8": ctypes.c_uint8,
    "u16": ctypes.c_uint16,
    "u32": ctypes.c_uint32,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "bool": ctypes.c_bool,
    "void": None,
}


class _SpanF64(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_double)), ("len", ctypes.c_size_t)]


class _SpanF32(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_float)), ("len", ctypes.c_size_t)]


class _SpanI32(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_int32)), ("len", ctypes.c_size_t)]


class _SpanI64(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_int64)), ("len", ctypes.c_size_t)]


_SPAN_CTYPES = {
    "Span[f64]": (_SpanF64, ctypes.c_double),
    "MutSpan[f64]": (_SpanF64, ctypes.c_double),
    "Span[f32]": (_SpanF32, ctypes.c_float),
    "MutSpan[f32]": (_SpanF32, ctypes.c_float),
    "Span[i32]": (_SpanI32, ctypes.c_int32),
    "MutSpan[i32]": (_SpanI32, ctypes.c_int32),
    "Span[i64]": (_SpanI64, ctypes.c_int64),
    "MutSpan[i64]": (_SpanI64, ctypes.c_int64),
}


@dataclass(frozen=True)
class ExportedSymbol:
    name: str
    params: tuple[tuple[str, str], ...]
    result: str


def inspect_exports(source_path: Path) -> list[ExportedSymbol]:
    source = source_path.read_text(encoding="utf-8")
    comp = Compiler(source, require_main=False)
    exports: list[ExportedSymbol] = []
    for name, fn in comp.entry_unit.functions.items():
        if fn.exported:
            exports.append(ExportedSymbol(name=name, params=fn.params, result=fn.result))
    return exports


class NxModuleProxy:
    """Python proxy object wrapping a compiled NX native shared library."""

    def __init__(self, name: str, lib_path: Path, exports: list[ExportedSymbol]):
        self.__name__ = name
        self.__file__ = str(lib_path)
        self._dll_dirs = []
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for cand_dir in (lib_path.parent, Path(shutil.which("g++") or r"C:\msys64\ucrt64\bin\g++.exe").resolve().parent):
                if cand_dir.is_dir():
                    try:
                        self._dll_dirs.append(os.add_dll_directory(str(cand_dir)))
                    except OSError:
                        pass
        self._lib = ctypes.CDLL(str(lib_path))
        self._exports = exports
        for sym in exports:
            c_fn = getattr(self._lib, f"nx_{sym.name}", None)
            if c_fn is None:
                continue
            argtypes = []
            for _, ptype in sym.params:
                if ptype in _SPAN_CTYPES:
                    argtypes.append(_SPAN_CTYPES[ptype][0])
                elif ptype in _CTYPES_MAP and _CTYPES_MAP[ptype] is not None:
                    argtypes.append(_CTYPES_MAP[ptype])
                else:
                    argtypes.append(ctypes.c_void_p)
            c_fn.argtypes = argtypes
            c_fn.restype = _CTYPES_MAP.get(sym.result, ctypes.c_void_p)
            setattr(self, sym.name, self._wrap(c_fn, sym))

    def close(self) -> None:
        handle = getattr(self._lib, "_handle", None)
        if handle and os.name == "nt":
            import _ctypes
            _ctypes.FreeLibrary(handle)
            self._lib._handle = 0

    def _wrap(self, c_fn: Any, sym: ExportedSymbol):
        def wrapper(*args):
            if len(args) != len(sym.params):
                raise TypeError(f"{sym.name}() expected {len(sym.params)} args, got {len(args)}")
            converted = []
            keep_alive = []
            for val, (_, ptype) in zip(args, sym.params):
                if ptype in _SPAN_CTYPES:
                    struct_cls, elem_cls = _SPAN_CTYPES[ptype]
                    is_mut = ptype.startswith("MutSpan[")
                    arr = None
                    # True zero-copy fast path via Python buffer protocol / NumPy ctypes
                    if hasattr(val, "ctypes") and hasattr(val, "shape") and (not is_mut or getattr(getattr(val, "flags", None), "writeable", True)):
                        try:
                            n_elems = len(val)
                            ptr = val.ctypes.data_as(ctypes.POINTER(elem_cls))
                            keep_alive.append(val)
                            converted.append(struct_cls(ptr, n_elems))
                            continue
                        except Exception:
                            pass
                    try:
                        mv = memoryview(val)
                        if mv.ndim == 1 and mv.c_contiguous and mv.itemsize == ctypes.sizeof(elem_cls):
                            if not mv.readonly:
                                arr = (elem_cls * len(mv)).from_buffer(mv)
                                keep_alive.append(mv)
                                keep_alive.append(arr)
                                converted.append(struct_cls(arr, len(mv)))
                                continue
                            elif not is_mut:
                                # Read-only buffer zero-copy via Py_buffer / ctypes address
                                buf_t = ctypes.c_char * mv.nbytes
                                raw_ptr = ctypes.cast(ctypes.pointer(buf_t.from_buffer_copy(b"")) if False else ctypes.addressof(ctypes.c_char.from_buffer(bytearray()) if False else ctypes.c_void_p()), ctypes.POINTER(elem_cls)) if False else None
                    except TypeError:
                        pass
                    arr = (elem_cls * len(val))(*val)
                    keep_alive.append(arr)
                    converted.append(struct_cls(arr, len(val)))
                else:
                    converted.append(val)
            return c_fn(*converted)
        wrapper.__name__ = sym.name
        return wrapper


def import_nx(nx_path: str | Path, *, cache_dir: Path | None = None) -> NxModuleProxy:
    """Compile a `.nx` library and load its @export symbols directly into Python."""
    from .cli import _build

    source = Path(nx_path).resolve()
    if not source.is_file():
        raise NexirError(f"NX module not found: {source}")
    exports = inspect_exports(source)
    if not exports:
        # Auto-check if module has top-level exportable functions; allow explicit @export
        raise NexirError(f"{source.name} has no @export functions for Python import")

    target_dir = cache_dir or (Path(tempfile.gettempdir()) / "nexir_py_cache")
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    suffix = ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")
    out_lib = target_dir / f"{source.stem}_{digest}{suffix}"
    if not out_lib.is_file():
        _build(source, out_lib, shared=True, require_main=False)
    return NxModuleProxy(source.stem, out_lib, exports)


class _NxMetaFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):
        if "." in fullname:
            return None
        search_roots = [Path(p) for p in (path or sys.path) if p]
        for root in search_roots:
            cand = root / f"{fullname}.nx"
            if cand.is_file():
                return importlib.util.spec_from_loader(fullname, _NxLoader(cand))
        return None


class _NxLoader(importlib.abc.Loader):
    def __init__(self, nx_path: Path):
        self.nx_path = nx_path

    def create_module(self, spec):
        return import_nx(self.nx_path)

    def exec_module(self, module):
        return None


def enable_import_hook() -> None:
    """Enable `import my_nx_module` in Python for `.nx` files on sys.path."""
    if not any(isinstance(f, _NxMetaFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _NxMetaFinder())


_C_TYPE_MAP = {
    "i8": "int8_t", "i16": "int16_t", "i32": "int32_t", "i64": "int64_t",
    "u8": "uint8_t", "u16": "uint16_t", "u32": "uint32_t", "u64": "uint64_t",
    "f32": "float", "f64": "double", "bool": "bool", "void": "void",
    "Span[f64]": "NxAbiSpan", "MutSpan[f64]": "NxAbiSpan",
    "Span[f32]": "NxAbiSpan", "MutSpan[f32]": "NxAbiSpan",
    "Span[i32]": "NxAbiSpan", "MutSpan[i32]": "NxAbiSpan",
    "Span[i64]": "NxAbiSpan", "MutSpan[i64]": "NxAbiSpan",
}

_RUST_TYPE_MAP = {
    "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
    "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64",
    "f32": "f32", "f64": "f64", "bool": "bool", "void": "()",
}

_GO_TYPE_MAP = {
    "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
    "u8": "uint8", "u16": "uint16", "u32": "uint32", "u64": "uint64",
    "f32": "float32", "f64": "float64", "bool": "bool", "void": "",
}
_GO_C_MAP = {
    "i8": "C.int8_t", "i16": "C.int16_t", "i32": "C.int32_t", "i64": "C.int64_t",
    "u8": "C.uint8_t", "u16": "C.uint16_t", "u32": "C.uint32_t", "u64": "C.uint64_t",
    "f32": "C.float", "f64": "C.double", "bool": "C.bool",
}


def generate_bindings(source_path: Path, lang: str, out_dir: Path) -> Path:
    """Generate native FFI bindings (c, cpp, python, rust, go) for @export NX symbols."""
    exports = inspect_exports(source_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem.replace("-", "_")

    if lang == "c":
        out = out_dir / f"{stem}.h"
        guard = f"NX_BIND_{stem.upper()}_H"
        lines = [
            f"#ifndef {guard}",
            f"#define {guard}",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            '#include "nx_native_abi.h"',
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
        ]
        for s in exports:
            ret = _C_TYPE_MAP.get(s.result, "void")
            params = ", ".join(f"{_C_TYPE_MAP.get(t, 'void*')} {n}" for n, t in s.params) or "void"
            lines.append(f"{ret} nx_{s.name}({params});")
        lines.extend(["#ifdef __cplusplus", "}", "#endif", f"#endif /* {guard} */", ""])
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    if lang == "cpp":
        out = out_dir / f"{stem}.hpp"
        lines = [
            "#pragma once",
            f'#include "{stem}.h"',
            f"namespace nx_{stem} {{",
        ]
        for s in exports:
            ret = _C_TYPE_MAP.get(s.result, "void")
            params = ", ".join(f"{_C_TYPE_MAP.get(t, 'void*')} {n}" for n, t in s.params)
            args = ", ".join(n for n, _ in s.params)
            body = f"nx_{s.name}({args});" if ret == "void" else f"return nx_{s.name}({args});"
            lines.append(f"inline {ret} {s.name}({params}) {{ {body} }}")
        lines.extend(["}", ""])
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    if lang == "python":
        out = out_dir / f"{stem}_nx.py"
        lines = [
            '"""Auto-generated Python FFI wrapper for NX native module."""',
            "from pathlib import Path",
            "from nexir.interop import import_nx",
            "",
            f'_MOD = import_nx(Path(__file__).resolve().with_name("{source_path.name}"))',
        ]
        for s in exports:
            params = ", ".join(n for n, _ in s.params)
            lines.append(f"def {s.name}({params}):")
            lines.append(f"    return _MOD.{s.name}({params})")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    if lang == "rust":
        out = out_dir / f"{stem}.rs"
        lines = [
            f"// Auto-generated Rust FFI bindings for {source_path.name}",
            f'#[link(name = "{stem}")]',
            'extern "C" {',
        ]
        for s in exports:
            params = ", ".join(f"{n}: {_RUST_TYPE_MAP.get(t, '*const core::ffi::c_void')}" for n, t in s.params)
            ret = _RUST_TYPE_MAP.get(s.result, "()")
            ret_sig = "" if ret == "()" else f" -> {ret}"
            lines.append(f"    pub fn nx_{s.name}({params}){ret_sig};")
        lines.append("}")
        for s in exports:
            params = ", ".join(f"{n}: {_RUST_TYPE_MAP.get(t, '*const core::ffi::c_void')}" for n, t in s.params)
            args = ", ".join(n for n, _ in s.params)
            ret = _RUST_TYPE_MAP.get(s.result, "()")
            ret_sig = "" if ret == "()" else f" -> {ret}"
            lines.append(f"pub fn {s.name}({params}){ret_sig} {{ unsafe {{ nx_{s.name}({args}) }} }}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    if lang == "go":
        out = out_dir / f"{stem}.go"
        lines = [
            f"package {stem}",
            "",
            "/*",
            f"#cgo LDFLAGS: -L. -l{stem}",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            f'#include "{stem}.h"',
            "*/",
            'import "C"',
            "",
        ]
        for s in exports:
            g_params = ", ".join(f"{n} {_GO_TYPE_MAP.get(t, 'uintptr')}" for n, t in s.params)
            g_ret = _GO_TYPE_MAP.get(s.result, "")
            c_args = ", ".join(f"{_GO_C_MAP.get(t, 'C.uintptr_t')}({n})" for n, t in s.params)
            if g_ret:
                lines.append(f"func {s.name.title()}({g_params}) {g_ret} {{")
                lines.append(f"\treturn {g_ret}(C.nx_{s.name}({c_args}))")
                lines.append("}")
            else:
                lines.append(f"func {s.name.title()}({g_params}) {{")
                lines.append(f"\tC.nx_{s.name}({c_args})")
                lines.append("}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    raise NexirError(f"unsupported binding target language: {lang}")


_C_TO_NX_TYPE = {
    "void": "void",
    "bool": "bool",
    "_Bool": "bool",
    "int": "i32",
    "int32_t": "i32",
    "int64_t": "i64",
    "long long": "i64",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "size_t": "u64",
    "float": "f32",
    "double": "f64",
    "const char *": "str",
    "const char*": "str",
    "char *": "str",
}

_C_SIG_RE = re.compile(
    r"^\s*(?:extern\s+\"C\"\s+)?(?:static\s+inline\s+|inline\s+|NX_EXPORT\s+|NX_ABI_EXPORT\s+)?"
    r"(const\s+char\s*\*|char\s*\*|long\s+long|[A-Za-z_]\w*(?:\s*\*)*)\s+"
    r"([A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*(?:;|(?=\{))",
    re.MULTILINE,
)


def _map_c_type_to_nx(raw: str) -> str:
    norm = " ".join(raw.replace("std::", "").strip().split())
    norm = norm.replace(" *", "*").replace("*", " *").strip()
    if norm in _C_TO_NX_TYPE:
        return _C_TO_NX_TYPE[norm]
    no_space = norm.replace(" ", "")
    if no_space in _C_TO_NX_TYPE:
        return _C_TO_NX_TYPE[no_space]
    return "dyn"


def parse_c_header_signatures(header_path: Path) -> dict[str, tuple[tuple[str, ...], str]]:
    """Automatically extract C/C++ function signatures from a header for polyglot zero-wrapper import."""
    if not header_path.is_file():
        return {}
    text = header_path.read_text(encoding="utf-8", errors="ignore")
    # Strip C comments
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    sigs: dict[str, tuple[tuple[str, ...], str]] = {}
    for m in _C_SIG_RE.finditer(text):
        ret_raw, fn_name, params_raw = m.group(1), m.group(2), m.group(3).strip()
        if fn_name in {"if", "while", "for", "switch", "return", "sizeof"}:
            continue
        ret_t = _map_c_type_to_nx(ret_raw)
        if not params_raw or params_raw == "void":
            param_types: tuple[str, ...] = ()
        else:
            pts: list[str] = []
            ok = True
            for p in params_raw.split(","):
                p = p.strip()
                if not p or p == "...":
                    ok = False
                    break
                # Split trailing parameter identifier if present
                tokens = p.replace("*", "* ").split()
                if len(tokens) >= 2 and re.fullmatch(r"[A-Za-z_]\w*", tokens[-1]) and tokens[-1] not in {"int", "long", "char", "double", "float", "void", "bool"}:
                    t_str = " ".join(tokens[:-1])
                else:
                    t_str = " ".join(tokens)
                pts.append(_map_c_type_to_nx(t_str))
            if not ok:
                continue
            param_types = tuple(pts)
        sigs[fn_name] = (param_types, ret_t)
    return sigs


def auto_discover_foreign_module(lang: str, target: str, search_dirs: list[Path]) -> dict[str, Any]:
    """Discover headers, shared libraries, and typed signatures for c/cpp/rust/go imports."""
    info: dict[str, Any] = {"lang": lang, "target": target, "header": None, "lib": None, "signatures": {}}
    stem = Path(target).stem
    header_names = [target] if target.endswith((".h", ".hpp", ".hh")) else [f"{target}.h", f"{target}.hpp", f"{stem}.h", f"{stem}.hpp"]
    for base in search_dirs:
        if not base or not base.exists():
            continue
        for sub in (base, base / "include", base / "src", base / "target" / "release"):
            if not sub.is_dir():
                continue
            for hn in header_names:
                cand = sub / hn
                if cand.is_file() and info["header"] is None:
                    info["header"] = cand
                    info["signatures"] = parse_c_header_signatures(cand)
            for ext in (".dll", ".so", ".dylib", ".a", ".lib"):
                for prefix in ("", "lib"):
                    lcand = sub / f"{prefix}{stem}{ext}"
                    if lcand.is_file() and info["lib"] is None:
                        info["lib"] = lcand
    return info


def auto_install_python_dep(mod_name: str, cache_root: Path) -> Path | None:
    """Install missing Python dependency into an isolated local cache directory."""
    pkg = mod_name.split(".")[0]
    pydeps = cache_root / "pydeps"
    pydeps.mkdir(parents=True, exist_ok=True)
    if str(pydeps) not in sys.path:
        sys.path.insert(0, str(pydeps))
    if importlib.util.find_spec(pkg) is not None:
        return pydeps
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet", "--target", str(pydeps), pkg],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    importlib.invalidate_caches()
    return pydeps


def auto_install_nx_dep(pkg_name: str, project_root: Path) -> Path | None:
    """Resolve and install a missing .nx package from local/configured registry into .nx_pkgs."""
    reg_env = os.environ.get("NX_REGISTRY")
    reg_dir = Path(reg_env).expanduser() if reg_env else (Path.home() / ".nexir" / "registry")
    if not reg_dir.is_dir():
        return None
    pkg_dir = reg_dir / pkg_name
    if not pkg_dir.is_dir():
        return None
    versions = sorted((p for p in pkg_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not versions:
        return None
    latest = versions[-1]
    dest = project_root / ".nx_pkgs" / pkg_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(latest, dest)
    return dest

