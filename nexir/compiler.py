"""NEXIR 2.0 Frontend, Multi-File / Cross-Language Module Loader, and C++17 Backend.

Supports:
- Pythonic syntax: `class` (replaces `struct`), `def` (`fn` removed), `enum` (simple & ADT),
  `trait`, `async def` / `await`, `match` / `case`, `with`, `try/except as/finally`, `?` propagation.
- Multi-file `.nx` imports with per-module C++ namespacing (no symbol collisions).
- Direct Python (`import py.<mod>` / installed modules) and C++ (`import cpp.<hdr>`) without wrapper files.
- Direct C/Rust/Go shared library imports (`import c.<lib>`, `import rust.<lib>`, `import go.<lib>`).
- Standard library (`import nx.asyncio`, `nx.thread`, `nx.fs`, `nx.io`, `nx.net`, `nx.gpu`, `nx.sandbox`, `nx.debug`, `nx.math`).
- Debug mode (`#line`, `NxTraceGuard`, memory/leak diagnostics) & Compile Audit Log.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import importlib.util
import io
from pathlib import Path
import re
import tokenize
from typing import Any

from .compiler_body import BodyGenerator, c_string
from .runtime_cpp import PRELUDE
from .security import (
    CAPABILITIES,
    CompileAuditLog,
    sha256_file,
    validate_cpp_header,
    validate_ffi_symbol,
    validate_python_module,
)
from .stdlib_registry import STD_MODULES
from .typesystem import (
    BASE_TYPES,
    INTEGERS,
    NUMERICS,
    canonical_type_name,
    channel_element,
    dict_parts,
    future_element,
    gpu_buffer_element,
    list_element,
    mutex_element,
    optional_element,
    resolve_type_node,
    result_parts,
    set_element,
    simd_parts,
    tuple_parts,
)


class NexirError(Exception):
    def __init__(self, message: str, node: ast.AST | None = None, *, file: str | None = None):
        self.message = message
        self.line = getattr(node, "lineno", None)
        self.column = getattr(node, "col_offset", None)
        self.file = file
        super().__init__(message)

    def __str__(self) -> str:
        prefix = f"{self.file}:" if self.file else ""
        if self.line is None:
            return f"{prefix}{self.message}" if prefix else self.message
        col = (self.column + 1) if self.column is not None else 1
        return f"{prefix}line {self.line}, column {col}: {self.message}"


@dataclass(frozen=True)
class Function:
    node: ast.FunctionDef | ast.AsyncFunctionDef
    params: tuple[tuple[str, str], ...]
    result: str
    caps: frozenset[str]
    exported: bool
    extern_symbol: str | None = None
    owner: str | None = None
    generic: tuple[tuple[str, str], ...] = ()
    raises: bool = False
    py_call: tuple[str, str] | None = None
    defaults: tuple[ast.expr | None, ...] = ()
    cpp_symbol: str | None = None
    is_async: bool = False
    mutates_self: bool = True
    is_virtual: bool = False
    is_override: bool = False


@dataclass
class ClassDefInfo:
    name: str
    qualified_name: str
    node: ast.ClassDef
    fields: tuple[tuple[str, str], ...]
    field_defaults: dict[str, ast.expr]
    methods: dict[str, Function]
    base: str | None = None
    generic: tuple[tuple[str, str], ...] = ()
    traits: tuple[str, ...] = ()


@dataclass
class EnumDefInfo:
    name: str
    qualified_name: str
    node: ast.ClassDef
    variants: dict[str, tuple[tuple[str, str], ...]]
    simple_values: dict[str, int]
    methods: dict[str, Function]
    is_simple: bool = False


@dataclass
class ImportBinding:
    alias: str
    kind: str  # "nx" | "nx_std" | "py" | "cpp" | "c" | "rust" | "go"
    target_name: str
    target_unit: ModuleUnit | None = None


@dataclass
class ModuleUnit:
    name: str
    path: Path | None
    prefix: str
    source: str
    tree: ast.Module = field(default_factory=lambda: ast.Module(body=[], type_ignores=[]))
    declarations: dict[int, str] = field(default_factory=dict)
    generics: dict[int, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    raises_lines: set[int] = field(default_factory=set)
    line_map: dict[int, int] = field(default_factory=dict)
    imports: dict[str, ImportBinding] = field(default_factory=dict)
    imported_symbols: dict[str, tuple[str, Any, str]] = field(default_factory=dict)
    traits: dict[str, dict[str, tuple[tuple[str, ...], str]]] = field(default_factory=dict)
    enums: dict[str, EnumDefInfo] = field(default_factory=dict)
    classes: dict[str, ClassDefInfo] = field(default_factory=dict)
    globals: dict[str, tuple[str, str]] = field(default_factory=dict)
    functions: dict[str, Function] = field(default_factory=dict)

    def cpp_symbol(self, local_name: str) -> str:
        return f"nx_{self.prefix}{local_name}"

    def original_line(self, lineno: int) -> int:
        return self.line_map.get(lineno, lineno)


RESERVED_NAMES = {
    "Buffer", "List", "Dict", "Set", "Tuple", "Optional", "Result", "Future",
    "Channel", "Mutex", "GpuBuffer", "StringBuilder", "File", "FileInfo",
    "TcpStream", "TcpListener", "HttpResponse", "ThreadPool", "AtomicI64",
    "buffer_span", "buffer_mutspan", "print", "range", "parallel_range",
    "len", "sum", "min", "max", "abs", "sorted", "any", "all",
    "zeros_f32", "zeros_f64", "zeros_i32", "zeros_i64",
    "read_text", "open", "tcp_connect", "type_name",
    "as_i64", "as_i128", "as_f64", "as_str", "as_bool", "is_none",
    "ring_f64", "ring_write", "ring_read", "ring_valid",
    "owned", "borrowed", "mut", "move", "is_ok", "is_err", "unwrap", "unwrap_or", "error", "value_or",
    "enumerate", "zip",
}


def _normalize_source(source: str, *, filename: str | None = None) -> tuple[str, dict[int, str], dict[int, tuple[tuple[str, str], ...]], set[int], dict[int, int]]:
    """Translate NX-only declarations (enum/trait/generics/raises/?) and reject legacy `struct` and `fn`."""
    lines = source.splitlines(keepends=True)
    declarations: dict[int, str] = {}
    generics: dict[int, tuple[tuple[str, str], ...]] = {}
    raises_lines: set[int] = set()
    first_tok_on_line: set[int] = set()
    qmark_positions: list[tuple[int, int]] = []

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError as exc:
        raise NexirError(f"tokenization error: {exc}", file=filename) from exc

    for token in tokens:
        if token.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE,
                          tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER}:
            continue
        if token.type == tokenize.OP and token.string == "?":
            qmark_positions.append(token.start)
            continue
        line = token.start[0]
        if line in first_tok_on_line:
            continue
        first_tok_on_line.add(line)

        if token.type == tokenize.NAME:
            if token.string == "struct":
                raise NexirError("struct has been replaced by class in NX 2.0", ast.Pass(lineno=line, col_offset=token.start[1]), file=filename)
            if token.string == "fn":
                raise NexirError("fn has been removed in NX 2.0; use def instead", ast.Pass(lineno=line, col_offset=token.start[1]), file=filename)

    # Apply postfix `?` -> `.__nx_propagate__()` in reverse line/col order before line rewrites
    for qline, qcol in reversed(qmark_positions):
        ln = lines[qline - 1]
        lines[qline - 1] = ln[:qcol] + ".__nx_propagate__()" + ln[qcol + 1:]

    # Second pass on line headers (trait, enum, class[T], def[T], async def[T], raises)
    in_enum_indent: int | None = None
    for line_idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        cur_indent = len(raw_line) - len(stripped)
        if in_enum_indent is not None:
            if cur_indent <= in_enum_indent:
                in_enum_indent = None
            else:
                # Allow ADT enum variants written as `Circle(radius: f64)` -> `Circle(radius=f64)`
                m_var = re.match(r"^(\s*[A-Za-z_]\w*\s*\()([^)]*)\)(\s*(?:#.*)?\r?\n?)$", raw_line)
                if m_var and ":" in m_var.group(2) and not stripped.startswith("def "):
                    v_head, v_args, v_tail = m_var.groups()
                    converted_args = re.sub(r"([A-Za-z_]\w*)\s*:\s*([^,)]+)", r"\1=\2", v_args)
                    lines[line_idx - 1] = f"{v_head}{converted_args}){v_tail}"
                    continue
        m_decl = re.match(r"^(\s*)(trait|enum)\s+([A-Za-z_]\w*)(.*)$", raw_line, re.DOTALL)
        if m_decl:
            indent, kw, name, rest = m_decl.groups()
            declarations[line_idx] = kw
            lines[line_idx - 1] = f"{indent}class {name}{rest}"
            if kw == "enum":
                in_enum_indent = len(indent)
            continue

        m_cls = re.match(r"^(\s*class\s+[A-Za-z_]\w*)\[([^\]]+)\](.*)$", raw_line, re.DOTALL)
        if m_cls:
            head, gen_raw, tail = m_cls.groups()
            generics[line_idx] = _parse_generic_params(gen_raw, line_idx, filename)
            lines[line_idx - 1] = f"{head}{tail}"
            continue

        m_def = re.match(r"^(\s*(?:async\s+)?def\s+[A-Za-z_]\w*)(?:\[([^\]]+)\])?(\s*\(.*\)\s*->\s*)(.+?)(\s+raises)(\s*:.*)$", raw_line, re.DOTALL)
        if m_def:
            head, gen_raw, mid, ret_expr, _, tail = m_def.groups()
            if gen_raw:
                generics[line_idx] = _parse_generic_params(gen_raw, line_idx, filename)
            raises_lines.add(line_idx)
            lines[line_idx - 1] = f"{head}{mid}Result[{ret_expr.strip()}]{tail}"
            continue

        m_def_gen = re.match(r"^(\s*(?:async\s+)?def\s+[A-Za-z_]\w*)\[([^\]]+)\](\s*\(.*)$", raw_line, re.DOTALL)
        if m_def_gen:
            head, gen_raw, tail = m_def_gen.groups()
            generics[line_idx] = _parse_generic_params(gen_raw, line_idx, filename)
            lines[line_idx - 1] = f"{head}{tail}"

    line_map = {i: i for i in range(1, len(lines) + 1)}
    return "".join(lines), declarations, generics, raises_lines, line_map


def _parse_generic_params(raw: str, line: int, filename: str | None) -> tuple[tuple[str, str], ...]:
    params: list[tuple[str, str]] = []
    for item in raw.split(","):
        part = [p.strip() for p in item.split(":")]
        if len(part) != 2 or not part[0].isidentifier() or not part[1].isidentifier():
            raise NexirError("generic parameters must use NAME: BOUND", ast.Pass(lineno=line, col_offset=0), file=filename)
        params.append((part[0], part[1]))
    return tuple(params)


class Compiler:
    def __init__(
        self,
        source: str,
        *,
        entry_path: Path | None = None,
        require_main: bool = True,
        debug: bool = False,
        audit_log: CompileAuditLog | None = None,
        strict_sandbox: bool = False,
        allowed_py_modules: set[str] | None = None,
    ):
        self.entry_path = entry_path.resolve() if entry_path else None
        self.require_main = require_main
        self.debug = debug
        self.audit_log = audit_log or CompileAuditLog()
        self.strict_sandbox = strict_sandbox
        self.allowed_py_modules = allowed_py_modules

        self.types: dict[str, str] = dict(BASE_TYPES)
        self.cpp_headers: set[str] = set()
        self.uses_python: bool = False
        self.py_preloads: set[str] = set()
        self.native_libs: dict[str, str] = {}  # lib -> kind ("c" | "rust" | "go")
        self.traits: dict[str, dict[str, tuple[tuple[str, ...], str]]] = {}
        self.classes: dict[str, ClassDefInfo] = {}
        self.enums: dict[str, EnumDefInfo] = {}
        self.subclasses: dict[str, str] = {}  # child -> base
        self.generic_names: set[str] = set()

        self._loaded_units: dict[Path, ModuleUnit] = {}
        self._unit_order: list[ModuleUnit] = []
        self._mod_seq = 0

        entry_name = entry_path.stem if entry_path else "<string>"
        self.entry_unit = self._load_unit(entry_name, source, entry_path, prefix="")
        self._resolve_imports_recursive(self.entry_unit)
        self._compile_units()

    def check_identifier(self, name: str, node: ast.AST | None = None) -> str:
        if not name.isidentifier() or name.startswith("nx_") or name in self.types or name in RESERVED_NAMES:
            raise NexirError(f"reserved name: {name}", node)
        return "nx_" + name

    def cpp_type(self, typ: str) -> str:
        if typ not in self.types:
            self.ensure_type(typ)
        return self.types[typ]

    def ensure_type(self, typ: str, node: ast.AST | None = None) -> None:
        if typ in self.types:
            return
        if typ == "None":
            return
        if typ in self.classes:
            self.types[typ] = self.classes[typ].qualified_name
            return
        if typ in self.enums:
            self.types[typ] = self.enums[typ].qualified_name
            return
        le = list_element(typ)
        if le:
            self.ensure_type(le, node)
            self.types[typ] = f"nx_vec<{self.types[le]}>"
            return
        se = set_element(typ)
        if se:
            self.ensure_type(se, node)
            ct = self.types[se]
            self.types[typ] = f"nx_set<{ct}, nx_hash<{ct}>, nx_equal<{ct}>>"
            return
        dp = dict_parts(typ)
        if dp:
            self.ensure_type(dp[0], node)
            self.ensure_type(dp[1], node)
            ck, cv = self.types[dp[0]], self.types[dp[1]]
            self.types[typ] = f"nx_map<{ck}, {cv}, nx_hash<{ck}>, nx_equal<{ck}>>"
            return
        tp = tuple_parts(typ)
        if tp:
            for p in tp:
                self.ensure_type(p, node)
            self.types[typ] = "std::tuple<" + ", ".join(self.types[p] for p in tp) + ">"
            return
        oe = optional_element(typ)
        if oe:
            self.ensure_type(oe, node)
            self.types[typ] = self.types[oe] if oe.startswith("owned[") else f"std::optional<{self.types[oe]}>"
            return
        rp = result_parts(typ)
        if rp:
            self.ensure_type(rp[0], node)
            if rp[1] is None:
                self.types[typ] = "nx_result_void" if rp[0] == "void" else f"nx_result<{self.types[rp[0]]}>"
            else:
                self.ensure_type(rp[1], node)
                self.types[typ] = f"nx_result_err<{self.types[rp[0]]}, {self.types[rp[1]]}>"
            return
        fe = future_element(typ)
        if fe:
            self.ensure_type(fe, node)
            self.types[typ] = f"nx_Future<{self.types[fe]}>"
            return
        ce = channel_element(typ)
        if ce:
            self.ensure_type(ce, node)
            self.types[typ] = f"nx_Channel<{self.types[ce]}>"
            return
        me = mutex_element(typ)
        if me:
            self.ensure_type(me, node)
            self.types[typ] = f"nx_Mutex<{self.types[me]}>"
            return
        ge = gpu_buffer_element(typ)
        if ge:
            self.ensure_type(ge, node)
            self.types[typ] = f"nx_GpuBuffer<{self.types[ge]}>"
            return
        if typ.startswith(("owned[", "borrowed[", "mut[")):
            inner = typ[typ.index("[") + 1:-1]
            self.ensure_type(inner, node)
            if typ.startswith("owned["):
                self.types[typ] = f"std::unique_ptr<{self.types[inner]}>"
            elif typ.startswith("borrowed["):
                self.types[typ] = f"const {self.types[inner]}&"
            else:
                self.types[typ] = f"{self.types[inner]}&"
            return

    def all_class_fields(self, cls_name: str) -> list[tuple[str, str, bool]]:
        cls = self.classes[cls_name]
        fields: list[tuple[str, str, bool]] = []
        if cls.base and cls.base in self.classes:
            fields.extend(self.all_class_fields(cls.base))
        for fn, ft in cls.fields:
            fields.append((fn, ft, fn in cls.field_defaults))
        return fields

    def _load_unit(self, name: str, source: str, path: Path | None, *, prefix: str) -> ModuleUnit:
        norm, decls, gens, raises, line_map = _normalize_source(source, filename=name)
        try:
            tree = ast.parse(norm)
        except SyntaxError as exc:
            raise NexirError(f"syntax error: {exc.msg} at line {exc.lineno}, column {exc.offset}", file=name) from exc
        unit = ModuleUnit(
            name=name, path=path, prefix=prefix, source=source,
            tree=tree, declarations=decls, generics=gens, raises_lines=raises, line_map=line_map,
        )
        if path is not None:
            self._loaded_units[path] = unit
            self.audit_log.record("module.load", "nx", name, str(path), sha256_file(path), status="allowed")
        else:
            self.audit_log.record("module.load", "nx", name, "<memory>", status="allowed")
        self._unit_order.append(unit)
        return unit

    def _find_nx_module(self, mod_parts: list[str], from_dir: Path | None, level: int) -> Path | None:
        rel_path = Path(*mod_parts).with_suffix(".nx")
        pkg_init = Path(*mod_parts) / "__init__.nx"
        roots: list[Path] = []
        if from_dir is not None:
            base = from_dir
            for _ in range(max(0, level - 1)):
                base = base.parent
            roots.append(base)
        if self.entry_path is not None:
            eroot = self.entry_path.parent
            roots.extend([eroot, eroot / "src", eroot / "nx_modules", eroot / ".nx" / "packages"])
            # Also search project root if entry is in src/
            roots.extend([eroot.parent, eroot.parent / "nx_modules", eroot.parent / ".nx" / "packages"])
        home_pkg = Path.home() / ".nexir" / "packages"
        if home_pkg.is_dir():
            roots.append(home_pkg)
            for pkg_dir in home_pkg.iterdir():
                if pkg_dir.is_dir():
                    roots.append(pkg_dir / "src")
                    roots.append(pkg_dir)
        for root in roots:
            for cand in (root / rel_path, root / pkg_init):
                if cand.is_file():
                    return cand.resolve()
        return None

    def _resolve_import_target(self, raw_name: str, from_dir: Path | None, level: int, node: ast.AST) -> tuple[str, str, ModuleUnit | None]:
        # 1. Standard library nx.*
        if raw_name in STD_MODULES:
            self.audit_log.record("module.load", "nx_std", raw_name, "builtin", status="allowed")
            return "nx_std", raw_name, None
        # 2. Direct Python module py.*
        if raw_name.startswith("py."):
            py_mod = raw_name[3:]
            try:
                validate_python_module(py_mod, strict_sandbox=self.strict_sandbox, allowlist=self.allowed_py_modules)
            except ValueError as exc:
                self.audit_log.record("module.load", "python", py_mod, str(exc), status="denied")
                raise NexirError(str(exc), node) from exc
            self.audit_log.record("module.load", "python", py_mod, "cpython-abi", status="allowed")
            return "py", py_mod, None
        # 3. Direct C++ header cpp.*
        if raw_name.startswith("cpp."):
            hdr = raw_name[4:].replace(".", "/")
            try:
                path, local = validate_cpp_header(hdr, allow_unsafe=False)
            except ValueError as exc:
                self.audit_log.record("module.load", "cpp", hdr, str(exc), status="denied")
                raise NexirError(str(exc), node) from exc
            self.cpp_headers.add(f'#include "{path}"' if local else f"#include <{path}>")
            self.audit_log.record("module.load", "cpp", path, "header", status="allowed")
            return "cpp", path, None
        # 4. Native C / Rust / Go shared library
        for prefix_lang in ("c.", "rust.", "go."):
            if raw_name.startswith(prefix_lang):
                kind = prefix_lang[:-1]
                lib_name = raw_name[len(prefix_lang):]
                if not re.fullmatch(r"[A-Za-z0-9_.\-]+", lib_name) or ".." in lib_name:
                    raise NexirError(f"invalid {kind} native library name: {lib_name}", node)
                self.native_libs[lib_name] = kind
                self.audit_log.record("module.load", kind, lib_name, "shared-lib", status="allowed")
                return kind, lib_name, None
        # 5. Local / package .nx file
        parts = [p for p in raw_name.split(".") if p]
        nx_file = self._find_nx_module(parts, from_dir, level) if parts else None
        if nx_file is not None:
            if nx_file in self._loaded_units:
                return "nx", nx_file.stem, self._loaded_units[nx_file]
            self._mod_seq += 1
            safe_mod = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
            prefix = f"m{self._mod_seq}_{safe_mod}_"
            try:
                src = nx_file.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise NexirError(f"source file must be UTF-8: {nx_file.name}", node) from exc
            sub_unit = self._load_unit(nx_file.name, src, nx_file, prefix=prefix)
            self._resolve_imports_recursive(sub_unit)
            return "nx", nx_file.stem, sub_unit
        # 6. Fallback: installed Python module (only when level == 0 and valid Python module exists!)
        if level == 0 and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", raw_name):
            root_mod = raw_name.split(".")[0]
            try:
                spec = importlib.util.find_spec(root_mod)
            except (ModuleNotFoundError, ValueError):
                spec = None
            if spec is not None:
                try:
                    validate_python_module(raw_name, strict_sandbox=self.strict_sandbox, allowlist=self.allowed_py_modules)
                except ValueError as exc:
                    self.audit_log.record("module.load", "python", raw_name, str(exc), status="denied")
                    raise NexirError(str(exc), node) from exc
                self.audit_log.record("module.load", "python", raw_name, "cpython-abi", status="allowed")
                return "py", raw_name, None
        raise NexirError(f"module not found: {raw_name}", node)

    def _resolve_imports_recursive(self, unit: ModuleUnit) -> None:
        from_dir = unit.path.parent if unit.path else None
        for node in unit.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    kind, tgt_name, tgt_unit = self._resolve_import_target(alias.name, from_dir, 0, node)
                    bind_name = alias.asname or (alias.name.split(".")[-1] if "." in alias.name else alias.name)
                    unit.imports[bind_name] = ImportBinding(bind_name, kind, tgt_name, tgt_unit)
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                if node.level > 0 and not mod_name:
                    # `from . import foo`
                    for alias in node.names:
                        if alias.name == "*":
                            raise NexirError("wildcard relative import requires module name", node)
                        kind, tgt_name, tgt_unit = self._resolve_import_target(alias.name, from_dir, node.level, node)
                        bind_name = alias.asname or alias.name
                        unit.imports[bind_name] = ImportBinding(bind_name, kind, tgt_name, tgt_unit)
                    continue
                kind, tgt_name, tgt_unit = self._resolve_import_target(mod_name, from_dir, node.level, node)
                for alias in node.names:
                    if alias.name == "*":
                        if kind != "nx" or tgt_unit is None:
                            raise NexirError("wildcard import (*) is supported for .nx modules only", node)
                        unit.imports[f"__star_{tgt_name}"] = ImportBinding("*", kind, tgt_name, tgt_unit)
                    else:
                        as_name = alias.asname or alias.name
                        unit.imported_symbols[as_name] = (kind, tgt_unit if kind == "nx" else tgt_name, alias.name)

    def _compile_units(self) -> None:
        # Step A: Collect traits & generic bounds & class/enum names across all units
        for unit in self._unit_order:
            for node in unit.tree.body:
                if isinstance(node, ast.ClassDef) and unit.declarations.get(node.lineno) == "trait":
                    self.check_identifier(node.name, node)
                    if node.name in self.traits:
                        raise NexirError(f"duplicate trait declaration: {node.name}", node)
                    if node.decorator_list or node.bases or node.keywords:
                        raise NexirError("trait does not support decorators or inheritance", node)
                    methods: dict[str, tuple[tuple[str, ...], str]] = {}
                    for member in node.body:
                        if isinstance(member, ast.Pass):
                            continue
                        if (not isinstance(member, ast.FunctionDef) or member.decorator_list
                                or not member.args.args or member.args.args[0].arg != "self"
                                or member.args.args[0].annotation is not None
                                or len(member.body) != 1 or not isinstance(member.body[0], ast.Pass)
                                or member.args.defaults or member.args.kwonlyargs or member.args.vararg or member.args.kwarg):
                            raise NexirError("trait methods require typed parameters and a pass body", member)
                        self.check_identifier(member.name, member)
                        params = []
                        for arg in member.args.args[1:]:
                            typ = "Self" if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "Self" else resolve_type_node(arg.annotation, default="", types=self.types, error_cls=NexirError)
                            if typ not in NUMERICS | {"bool", "str", "Self"}:
                                raise NexirError("trait parameter must be numeric, bool, str or Self", arg)
                            params.append(typ)
                        res = "Self" if isinstance(member.returns, ast.Name) and member.returns.id == "Self" else resolve_type_node(member.returns, types=self.types, error_cls=NexirError)
                        if res not in NUMERICS | {"bool", "Self"}:
                            raise NexirError("trait method result must be numeric, bool or Self", member)
                        methods[member.name] = (tuple(params), res)
                    self.traits[node.name] = methods
                    unit.traits[node.name] = methods

        for unit in self._unit_order:
            for params in unit.generics.values():
                for param, bound in params:
                    if bound not in {"Number", "Integer", "Int"} | self.traits.keys():
                        raise NexirError(f"unsupported generic bound: {bound}")
                    if param in self.types and param not in self.generic_names:
                        raise NexirError(f"generic type name is reserved: {param}")
                    if bound == "Int":
                        self.types[f"$const:{param}"] = f"nx_{param}"
                    else:
                        self.types[param] = param
                        self.generic_names.add(param)
                        if bound in {"Number", "Integer"}:
                            self.types[f"$numeric:{param}"] = "true"

        # Step B: Register class & enum names first so forward/self-references & Optional[Class]/Result[Class] work
        for unit in self._unit_order:
            for node in unit.tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                decl = unit.declarations.get(node.lineno, "class")
                if decl == "trait":
                    continue
                self.check_identifier(node.name, node)
                qname = node.name if unit is self.entry_unit else f"{unit.prefix}{node.name}"
                cpp_id = unit.cpp_symbol(node.name)
                if decl == "enum":
                    if node.name in unit.enums or node.name in unit.classes:
                        raise NexirError(f"duplicate enum: {node.name}", node)
                    self.types[qname] = cpp_id
                    if unit is self.entry_unit:
                        self.types[node.name] = cpp_id
                    edef = EnumDefInfo(node.name, qname, node, {}, {}, {})
                    unit.enums[node.name] = edef
                    self.enums[qname] = edef
                else:
                    if node.name in unit.classes or node.name in unit.enums:
                        raise NexirError(f"duplicate or reserved class: {node.name}", node)
                    self.types[qname] = cpp_id
                    self.types[f"owned[{qname}]"] = f"std::unique_ptr<{cpp_id}>"
                    self.types[f"borrowed[{qname}]"] = f"const {cpp_id}&"
                    self.types[f"mut[{qname}]"] = f"{cpp_id}&"
                    cdef = ClassDefInfo(node.name, qname, node, (), {}, {})
                    unit.classes[node.name] = cdef
                    self.classes[qname] = cdef

        # Expand star imports (`from mod import *`) for classes/enums/functions/globals after pass registration
        for unit in self._unit_order:
            for key, imp in list(unit.imports.items()):
                if key.startswith("__star_") and imp.target_unit is not None:
                    tgt = imp.target_unit
                    for cn, cd in tgt.classes.items():
                        unit.classes.setdefault(cn, cd)
                        unit.imported_symbols.setdefault(cn, ("nx", tgt, cn))
                    for en, ed in tgt.enums.items():
                        unit.enums.setdefault(en, ed)
                    for fn_name in tgt.functions:
                        if fn_name != "main":
                            unit.imported_symbols.setdefault(fn_name, ("nx", tgt, fn_name))
                    for gn in tgt.globals:
                        unit.imported_symbols.setdefault(gn, ("nx", tgt, gn))
            # Also handle explicit `from mod import Class/Enum`
            for sym_name, (kind, tgt, orig_name) in list(unit.imported_symbols.items()):
                if kind == "nx" and isinstance(tgt, ModuleUnit):
                    if orig_name in tgt.classes:
                        unit.classes[sym_name] = tgt.classes[orig_name]
                    if orig_name in tgt.enums:
                        unit.enums[sym_name] = tgt.enums[orig_name]

        # Step C: Parse enum bodies, class bodies, globals, and functions for each unit
        for unit in self._unit_order:
            self._parse_unit_members(unit)

        if self.require_main:
            if "main" not in self.entry_unit.functions:
                raise NexirError("program requires def main()")
            main_fn = self.entry_unit.functions["main"]
            if main_fn.params or main_fn.result not in {"void", "i32"}:
                raise NexirError("main must take no arguments and return void or i32", main_fn.node)

    def _parse_unit_members(self, unit: ModuleUnit) -> None:
        # 1. File-level globals/constants
        for node in unit.tree.body:
            if not isinstance(node, ast.AnnAssign):
                continue
            if not isinstance(node.target, ast.Name) or node.value is None:
                raise NexirError("file-level variables require a typed name and constant", node)
            name = node.target.id
            self.check_identifier(name, node)
            typ = resolve_type_node(node.annotation, types=self.types, classes=set(self.classes), enums=set(self.enums), error_cls=NexirError)
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, MemoryError) as exc:
                raise NexirError("file-level initializer must be a constant", node.value) from exc
            if type(value) is str and typ == "str":
                code = f"NxStr::from_literal({c_string(value)}, {len(value.encode('utf-8'))})"
            elif type(value) is bool and typ == "bool":
                code = "true" if value else "false"
            elif type(value) is int and typ in INTEGERS:
                code = f"{value}LL" if typ in {"i64", "i128"} else str(value)
            elif type(value) is float and typ in {"f32", "f64"}:
                code = repr(value)
            elif value is None and typ.startswith("Optional["):
                code = "std::nullopt"
            else:
                raise NexirError("file-level constant does not match its type", node.value)
            if name in unit.globals:
                raise NexirError(f"duplicate file-level variable: {name}", node)
            unit.globals[name] = (typ, code)

        # 2. Enums
        for ename, edef in unit.enums.items():
            if edef.node not in unit.tree.body:
                continue  # imported
            node = edef.node
            if node.decorator_list or node.bases or node.keywords:
                raise NexirError("enum does not support decorators or inheritance", node)
            variants: dict[str, tuple[tuple[str, str], ...]] = {}
            simple_vals: dict[str, int] = {}
            methods: dict[str, Function] = {}
            next_val = 0
            has_payload = False
            for member in node.body:
                if isinstance(member, ast.Pass):
                    continue
                if isinstance(member, ast.Expr) and isinstance(member.value, ast.Name):
                    vname = member.value.id
                    self.check_identifier(vname, member)
                    variants[vname] = ()
                    simple_vals[vname] = next_val
                    next_val += 1
                    continue
                if isinstance(member, ast.Expr) and isinstance(member.value, ast.Call) and isinstance(member.value.func, ast.Name):
                    has_payload = True
                    vname = member.value.func.id
                    self.check_identifier(vname, member)
                    v_params: list[tuple[str, str]] = []
                    for arg in member.value.args:
                        if isinstance(arg, ast.Name):
                            vt = resolve_type_node(arg, types=self.types, classes=set(self.classes), enums=set(self.enums), error_cls=NexirError)
                            v_params.append((f"v{len(v_params)}", vt))
                        else:
                            raise NexirError("enum variant payload uses Type or name: Type", arg)
                    for kw in member.value.keywords:
                        if not kw.arg:
                            raise NexirError("invalid enum variant field", kw)
                        vt = resolve_type_node(kw.value, types=self.types, classes=set(self.classes), enums=set(self.enums), error_cls=NexirError)
                        v_params.append((kw.arg, vt))
                    variants[vname] = tuple(v_params)
                    continue
                if isinstance(member, ast.Assign) and len(member.targets) == 1 and isinstance(member.targets[0], ast.Name) and isinstance(member.value, ast.Constant) and type(member.value.value) is int:
                    vname = member.targets[0].id
                    self.check_identifier(vname, member)
                    variants[vname] = ()
                    simple_vals[vname] = member.value.value
                    next_val = member.value.value + 1
                    continue
                if isinstance(member, ast.FunctionDef):
                    m = self._parse_method(edef.qualified_name, member, unit)
                    methods[member.name] = m
                    continue
                # Also support `Circle(r: f64)` written in Python as AnnAssign? Wait: `Circle(radius: f64)` in Python syntax is INVALID Python syntax unless rewritten OR written `Circle(radius=f64)` or `Circle(f64)`!
                raise NexirError("enum body supports Variant, Variant(T...), Variant = int, or methods", member)
            if not variants:
                raise NexirError("enum requires at least one variant", node)
            edef.variants = variants
            edef.simple_values = simple_vals
            edef.methods = methods
            edef.is_simple = not has_payload

        # 3. Classes
        for cname, cdef in unit.classes.items():
            if cdef.node not in unit.tree.body:
                continue  # imported
            node = cdef.node
            if node.decorator_list or node.keywords:
                raise NexirError("class does not support decorators or keyword bases", node)
            base_name: str | None = None
            traits_list: list[str] = []
            for base in node.bases:
                if not isinstance(base, ast.Name):
                    raise NexirError("class base must be a class or trait name", base)
                if base.id in self.traits:
                    traits_list.append(base.id)
                elif base.id in unit.classes:
                    if base_name is not None:
                        raise NexirError("class supports single inheritance plus traits", base)
                    base_name = unit.classes[base.id].qualified_name
                    self.subclasses[cdef.qualified_name] = base_name
                else:
                    raise NexirError(f"unknown base class or trait: {base.id}", base)
            cdef.base = base_name
            cdef.traits = tuple(traits_list)
            cdef.generic = unit.generics.get(node.lineno, ())

            fields: list[tuple[str, str]] = []
            field_defaults: dict[str, ast.expr] = {}
            methods: dict[str, Function] = {}
            for member in node.body:
                if isinstance(member, ast.Pass):
                    continue
                if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                    field = member.target.id
                    self.check_identifier(field, member)
                    typ = resolve_type_node(member.annotation, types=self.types, classes=set(self.classes), enums=set(self.enums), error_cls=NexirError)
                    if typ in self.generic_names:
                        raise NexirError("class fields cannot use an unbound function generic type", member)
                    if typ in {"void", f"borrowed[{cdef.qualified_name}]", f"mut[{cdef.qualified_name}]"} or typ.startswith(("borrowed[", "mut[")):
                        raise NexirError("class field requires a value or owned type", member)
                    if typ == cdef.qualified_name:
                        raise NexirError("class cannot contain itself by value; use owned[Class]", member)
                    if field in dict(fields) or field in methods:
                        raise NexirError(f"duplicate class member: {field}", member)
                    fields.append((field, typ))
                    if member.value is not None:
                        field_defaults[field] = member.value
                    continue
                if isinstance(member, ast.FunctionDef):
                    m = self._parse_method(cdef.qualified_name, member, unit)
                    if member.name in methods or member.name in dict(fields):
                        raise NexirError(f"duplicate class member: {member.name}", member)
                    methods[member.name] = m
                    continue
                raise NexirError("class body supports typed fields and methods only", member)
            cdef.fields = tuple(fields)
            cdef.field_defaults = field_defaults
            cdef.methods = methods

            # Verify trait conformance for explicitly declared traits `class Foo(MyTrait):`
            for tr in cdef.traits:
                for mname, (tparams, tres) in self.traits[tr].items():
                    cm = methods.get(mname)
                    if cm is None:
                        raise NexirError(f"class {cname} missing trait {tr} method {mname}", node)
                    exp_res = cdef.qualified_name if tres == "Self" else tres
                    exp_params = tuple(cdef.qualified_name if p == "Self" else p for p in tparams)
                    if cm.result != exp_res or tuple(t for _, t in cm.params[1:]) != exp_params:
                        raise NexirError(f"class {cname}.{mname} signature does not match trait {tr}", cm.node)

        # 4. Functions (`def` and `async def`)
        for node in unit.tree.body:
            if isinstance(node, (ast.AnnAssign, ast.Import, ast.ImportFrom, ast.ClassDef)):
                continue
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise NexirError("only function, class, enum, trait, import, and variable declarations are allowed at module level", node)
            self.check_identifier(node.name, node)
            if node.name in unit.functions or node.name in unit.classes or node.name in unit.enums or node.name in unit.traits or node.name in unit.globals:
                raise NexirError(f"duplicate function: {node.name}", node)
            fn = self._parse_function(node, unit)
            unit.functions[node.name] = fn

    def _parse_method(self, owner: str, node: ast.FunctionDef, unit: ModuleUnit) -> Function:
        if node.name not in {"__init__", "__del__", "__str__", "__repr__", "__add__", "__sub__", "__mul__", "__truediv__", "__eq__", "__getitem__", "__setitem__", "__len__", "__enter__", "__exit__"}:
            self.check_identifier(node.name, node)
        if not node.args.args or node.args.args[0].arg != "self" or node.args.args[0].annotation is not None:
            raise NexirError("class method must start with unannotated self", node)
        args = node.args
        if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.kw_defaults:
            raise NexirError("only positional or default-value method parameters are supported", node)
        caps: set[str] = set()
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Name) or dec.func.id != "cap" or dec.keywords or not dec.args:
                raise NexirError("class methods support @cap(...) only", dec)
            for item in dec.args:
                if not isinstance(item, ast.Constant) or type(item.value) is not str or item.value not in CAPABILITIES:
                    raise NexirError("unknown method capability", item)
                caps.add(item.value)
        params = [("self", owner)]
        for arg in args.args[1:]:
            self.check_identifier(arg.arg, arg)
            typ = resolve_type_node(arg.annotation, default="dyn", types=self.types, classes=set(self.classes), enums=set(self.enums), error_cls=NexirError)
            if typ == "void":
                raise NexirError("method parameters require a non-void type", arg)
            if arg.arg in [n for n, _ in params]:
                raise NexirError(f"duplicate parameter: {arg.arg}", arg)
            params.append((arg.arg, typ))
        res = resolve_type_node(node.returns, types=self.types, classes=set(self.classes), enums=set(self.enums), error_cls=NexirError)
        raises = node.lineno in unit.raises_lines
        if node.name in {"__init__", "__del__"} and res != "void":
            raise NexirError("constructor/destructor must return void", node)
        if node.name == "__del__" and len(params) != 1:
            raise NexirError("__del__ takes only self", node)
        # Check if method mutates self fields so const methods work on borrowed[Class]
        mutates = node.name in {"__init__", "__del__", "__setitem__"} or any(
            isinstance(st, (ast.Assign, ast.AugAssign))
            and isinstance(getattr(st, "target", getattr(st, "targets", [None])[0]), ast.Attribute)
            and isinstance(getattr(st, "target", getattr(st, "targets", [None])[0]).value, ast.Name)
            and getattr(st, "target", getattr(st, "targets", [None])[0]).value.id == "self"
            for st in ast.walk(node)
        )
        return Function(node, tuple(params), res, frozenset(caps), False, owner=owner, raises=raises, mutates_self=mutates)

    def _parse_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, unit: ModuleUnit) -> Function:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        generic = unit.generics.get(node.lineno, ())
        allowed_generic = dict(generic)
        caps: set[str] = set()
        exported = False
        extern_symbol = None
        py_call = None
        py_fn_used = False
        cpp_symbol = None
        seen: set[str] = set()

        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id in {"native", "export"}:
                label = dec.id
                exported |= label == "export"
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "cap" and not dec.keywords:
                label = "cap"
                if not dec.args:
                    raise NexirError("@cap requires at least one capability", dec)
                for item in dec.args:
                    if not isinstance(item, ast.Constant) or type(item.value) is not str or item.value not in CAPABILITIES:
                        raise NexirError("unknown capability; allowed: " + ", ".join(sorted(CAPABILITIES)), item)
                    caps.add(item.value)
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "extern" and not dec.keywords:
                label = "extern"
                if len(dec.args) != 1 or not isinstance(dec.args[0], ast.Constant) or type(dec.args[0].value) is not str:
                    raise NexirError("@extern requires one C symbol name", dec)
                symbol = dec.args[0].value
                if not symbol.isascii() or not symbol.isidentifier():
                    raise NexirError("@extern symbol must be an ASCII identifier", dec)
                try:
                    validate_ffi_symbol(symbol, allow_process=("process.exec" in caps))
                except ValueError as exc:
                    raise NexirError(str(exc), dec) from exc
                extern_symbol = symbol
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id in {"py_call", "py_fn"} and not dec.keywords:
                label = dec.func.id
                if len(dec.args) != 2 or any(not isinstance(it, ast.Constant) or type(it.value) is not str for it in dec.args):
                    raise NexirError("@py_call requires module and function string names", dec)
                mod_n, fn_n = (it.value for it in dec.args)
                try:
                    validate_python_module(mod_n, strict_sandbox=self.strict_sandbox, allowlist=self.allowed_py_modules)
                except ValueError as exc:
                    raise NexirError(str(exc), dec) from exc
                if not re.fullmatch(r"[A-Za-z_]\w*", fn_n):
                    raise NexirError("@py_call function name must be a Python identifier", dec)
                py_call = (mod_n, fn_n)
                py_fn_used = label == "py_fn"
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "cpp_header" and not dec.keywords:
                label = "cpp_header"
                if len(dec.args) != 1 or not isinstance(dec.args[0], ast.Constant) or type(dec.args[0].value) is not str:
                    raise NexirError("@cpp_header expects one header string", dec)
                try:
                    path, local = validate_cpp_header(dec.args[0].value, allow_unsafe=("process.exec" in caps))
                except ValueError as exc:
                    raise NexirError(str(exc), dec) from exc
                self.cpp_headers.add(f'#include "{path}"' if local else f"#include <{path}>")
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "cpp_fn" and not dec.keywords:
                label = "cpp_fn"
                if len(dec.args) != 1 or not isinstance(dec.args[0], ast.Constant) or type(dec.args[0].value) is not str:
                    raise NexirError("@cpp_fn expects one C++ symbol", dec)
                cpp_symbol = dec.args[0].value
                if not re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", cpp_symbol):
                    raise NexirError("@cpp_fn expects a simple C++ qualified name", dec)
                try:
                    validate_ffi_symbol(cpp_symbol, allow_process=("process.exec" in caps))
                except ValueError as exc:
                    raise NexirError(str(exc), dec) from exc
            else:
                raise NexirError("unsupported decorator", dec)
            if label in seen and label != "cpp_header":
                raise NexirError(f"duplicate @{label}", dec)
            seen.add(label)

        args = node.args
        if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.kw_defaults:
            raise NexirError("only positional or named fixed parameters are supported", node)
        params: list[tuple[str, str]] = []
        for arg in args.args:
            self.check_identifier(arg.arg, arg)
            # Hybrid typing: unannotated parameter defaults to `dyn`!
            typ = resolve_type_node(arg.annotation, default="dyn", types=self.types, classes=set(self.classes), enums=set(self.enums), error_cls=NexirError)
            if typ == "void":
                raise NexirError("parameters require a non-void type", arg)
            if typ in self.generic_names and typ not in allowed_generic:
                raise NexirError(f"unbound generic type: {typ}", arg)
            if arg.arg in [n for n, _ in params]:
                raise NexirError(f"duplicate parameter: {arg.arg}", arg)
            params.append((arg.arg, typ))

        defaults: tuple[ast.expr | None, ...] = (None,) * (len(params) - len(args.defaults)) + tuple(args.defaults)
        for item, (_, typ) in zip(defaults, params):
            if item is not None:
                try:
                    dv = ast.literal_eval(item)
                except (ValueError, TypeError, SyntaxError, MemoryError) as exc:
                    raise NexirError("default must be a constant matching the parameter type", item) from exc
                if not (
                    type(dv) is str and typ == "str"
                    or type(dv) is bool and typ == "bool"
                    or type(dv) is int and typ in INTEGERS
                    or type(dv) is float and typ in {"f32", "f64"}
                    or dv is None and typ.startswith("Optional[")
                ):
                    raise NexirError("default must be a constant matching the parameter type", item)

        if node.returns is None:
            has_val_ret = any(isinstance(s, ast.Return) and s.value is not None for s in ast.walk(node))
            result = "dyn" if has_val_ret else "void"
        else:
            result = resolve_type_node(node.returns, types=self.types, classes=set(self.classes), enums=set(self.enums), error_cls=NexirError)
        if result in self.generic_names and result not in allowed_generic:
            raise NexirError(f"unbound generic type: {result}", node.returns)
        if is_async:
            result = f"Future[{result}]"
            self.ensure_type(result, node)
        raises = node.lineno in unit.raises_lines
        if raises and (exported or extern_symbol):
            raise NexirError("raises functions cannot use @export or @extern", node)
        if any(t.startswith("Span[") or t.startswith("MutSpan[") for t in [result]) and not is_async:
            raise NexirError("returning Span is not supported until lifetime checking is available", node.returns)
        if result.startswith(("borrowed[", "mut[")):
            raise NexirError("returning a borrow is not supported until lifetime checking is available", node.returns)
        if exported and (result not in NUMERICS | {"bool", "void"} or any(t not in NUMERICS | {"bool"} and not t.startswith(("Span[", "MutSpan[")) for _, t in params)):
            raise NexirError("@export supports numeric, bool and Span signatures only", node)
        if extern_symbol and (exported or caps or result == "dyn" or any(t == "dyn" or t == "Ring[f64]" for _, t in params)):
            raise NexirError("@extern requires a plain C ABI signature without @export or @cap", node)
        if extern_symbol and (len(node.body) != 1 or not isinstance(node.body[0], ast.Pass)):
            raise NexirError("@extern declaration body must be pass", node)
        if generic and (exported or extern_symbol):
            raise NexirError("generic functions cannot use @export or @extern", node)
        if py_call:
            if cpp_symbol or exported or extern_symbol or generic or raises or any(defaults) or len(params) != 1 or params[0][1] not in {"Span[f64]", "MutSpan[f64]"} or result not in {"f64", "str", "owned[Buffer[f64]]"}:
                raise NexirError("@py_call requires one f64 Span/MutSpan argument and f64, str or owned[Buffer[f64]] result", node)
            if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
                raise NexirError("@py_call declaration body must be pass", node)
            if result != "f64":
                caps.add("memory.alloc")
            if py_fn_used and "python.interop" not in caps:
                raise NexirError('@py_fn requires @cap("python.interop")', node)
            self.uses_python = True
            self.py_preloads.add(py_call[0])
        if cpp_symbol:
            if exported or extern_symbol or generic or raises or any(defaults) or result not in NUMERICS | {"bool", "void", "str"} or any(t not in NUMERICS | {"bool", "str"} for _, t in params):
                raise NexirError("@cpp_fn requires a primitive C++ signature without other FFI decorators", node)
            if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
                raise NexirError("@cpp_fn declaration body must be pass", node)
            if "ffi.cpp" not in caps:
                raise NexirError('@cpp_fn requires @cap("ffi.cpp")', node)

        return Function(
            node, tuple(params), result, frozenset(caps), exported, extern_symbol,
            generic=generic, raises=raises, py_call=py_call, defaults=defaults,
            cpp_symbol=cpp_symbol, is_async=is_async,
        )

    def _emit_python_abi_cpp(self) -> list[str]:
        return [
            "/* NX_PY_CALL */",
            'extern "C" double nx_py_call_f64_span(const char*, const char*, nx_span_f64);',
            'extern "C" double nx_py_call_f64_mutspan(const char*, const char*, nx_mutspan_f64);',
            'extern "C" char* nx_py_call_str_span(const char*, const char*, nx_span_f64);',
            'extern "C" char* nx_py_call_str_mutspan(const char*, const char*, nx_mutspan_f64);',
            'extern "C" nx_mutspan_f64 nx_py_call_buffer_span(const char*, const char*, nx_span_f64);',
            'extern "C" nx_mutspan_f64 nx_py_call_buffer_mutspan(const char*, const char*, nx_mutspan_f64);',
            'extern "C" NxAbiWireVal nx_py_abi_call(const char*, const char*, std::size_t, const NxAbiWireVal*, const char**, std::size_t);',
            'extern "C" NxAbiWireVal nx_py_abi_method_call(void*, const char*, std::size_t, const NxAbiWireVal*, const char**, std::size_t);',
            'extern "C" NxAbiWireVal nx_py_abi_getattr(void*, const char*);',
            'extern "C" NxAbiWireVal nx_py_abi_import_getattr(const char*, const char*);',
            'extern "C" void nx_py_abi_release_val(NxAbiWireVal*);',
            'extern "C" void *nx_py_abi_import(const char*);',
            "static int nx_py_sandbox_callback(const char *mod) { return NxSandbox::instance().check_py_module(mod) ? 1 : 0; }",
            "static NxValue nx_py_wire_to_nxvalue(const NxAbiWireVal &w) {",
            "    switch (w.tag) {",
            "        case NX_ABI_T_NULL: return NxValue::null();",
            "        case NX_ABI_T_BOOL: return NxValue::boolean(w.payload.i64 != 0);",
            "        case NX_ABI_T_I64: return NxValue::integer(w.payload.i64);",
            "        case NX_ABI_T_U64: return NxValue::int128(static_cast<nx_i128>(w.payload.u64));",
            "        case NX_ABI_T_I128: {",
            "            nx_i128 val = (static_cast<nx_i128>(w.payload.i128_v.hi) << 64) | static_cast<nx_u128>(w.payload.i128_v.lo);",
            "            return NxValue::int128(val);",
            "        }",
            "        case NX_ABI_T_F64: return NxValue::floating(w.payload.f64);",
            "        case NX_ABI_T_STR: return NxValue::string(static_cast<const char *>(w.payload.ptr));",
            "        case NX_ABI_T_LIST: {",
            "            const auto *lst = static_cast<const NxAbiWireList *>(w.payload.ptr);",
            "            nx_vec<NxValue> out;",
            "            if (lst) { out.reserve(lst->count); for (std::size_t i = 0; i < lst->count; ++i) out.push_back(nx_py_wire_to_nxvalue(lst->items[i])); }",
            "            return NxValue::list(std::move(out));",
            "        }",
            "        case NX_ABI_T_DICT: {",
            "            const auto *d = static_cast<const NxAbiWireDict *>(w.payload.ptr);",
            "            nx_map<NxStr, NxValue> out;",
            "            if (d) { for (std::size_t i = 0; i < d->count; ++i) out[NxStr(d->keys[i])] = nx_py_wire_to_nxvalue(d->vals[i]); }",
            "            return NxValue::dict(std::move(out));",
            "        }",
            "        case NX_ABI_T_PYOBJ: {",
            "            void *obj = w.payload.ptr;",
            "            NxAbiWireVal *owned_box = new NxAbiWireVal(w);",
            "            return NxValue::pyobj(std::shared_ptr<void>(obj, [owned_box](void *) { nx_py_abi_release_val(owned_box); delete owned_box; }));",
            "        }",
            '        case NX_ABI_T_ERROR: nx_fail_str(std::string("Python error: ") + (w.payload.ptr ? static_cast<const char*>(w.payload.ptr) : "unknown"));',
            "        default: return NxValue::null();",
            "    }",
            "}",
            "static NxValue nx_py_dyn_method(const NxValue &recv, const char *method, std::size_t argc, const NxAbiWireVal *argv, const char **kws, std::size_t kwc) {",
            '    if (recv.type != 9 || !recv.data.ptr) nx_fail("method call requires Python object");',
            "    NxAbiWireVal res = nx_py_abi_method_call(recv.data.ptr->object.get(), method, argc, argv, kws, kwc);",
            "    NxValue out = nx_py_wire_to_nxvalue(res);",
            "    nx_py_abi_release_val(&res);",
            "    return out;",
            "}",
            "static NxValue nx_py_dyn_getattr(const NxValue &recv, const char *attr) {",
            '    if (recv.type != 9 || !recv.data.ptr) nx_fail("attribute access requires Python object");',
            "    NxAbiWireVal res = nx_py_abi_getattr(recv.data.ptr->object.get(), attr);",
            "    NxValue out = nx_py_wire_to_nxvalue(res);",
            "    nx_py_abi_release_val(&res);",
            "    return out;",
            "}",
            "static NxValue nx_py_abi_get_attr_mod(const char *mod, const char *attr) {",
            "    NxAbiWireVal w = nx_py_abi_import_getattr(mod, attr);",
            "    NxValue out = nx_py_wire_to_nxvalue(w);",
            "    nx_py_abi_release_val(&w);",
            "    return out;",
            "}",
            *[f"/* NX_PY_PRELOAD {m} */" for m in sorted(self.py_preloads)],
        ]

    def compile(self) -> str:
        # Generate enum/class/function bodies first so self.uses_python, self.py_preloads,
        # self.cpp_headers and dynamic self.types entries are all collected before header emit!
        decl_out: list[str] = []
        body_out: list[str] = []

        # Forward declare all enums and classes
        for unit in self._unit_order:
            for edef in unit.enums.values():
                if edef.node in unit.tree.body:
                    decl_out.append(f"struct {unit.cpp_symbol(edef.name)};")
            for cdef in unit.classes.values():
                if cdef.node in unit.tree.body:
                    decl_out.append(f"struct {unit.cpp_symbol(cdef.name)};")

        # Globals
        for unit in self._unit_order:
            for name, (typ, value) in unit.globals.items():
                qualifier = " const" if name.isupper() else ""
                decl_out.append(f"static {self.cpp_type(typ)}{qualifier} {unit.cpp_symbol(name)} = {value};")

        # Function forward declarations
        for unit in self._unit_order:
            for name, fn in unit.functions.items():
                emit_name = f"nx_{name}" if (unit is self.entry_unit or fn.exported or fn.extern_symbol) else unit.cpp_symbol(name)
                if fn.generic:
                    decl_out.append("template <" + ", ".join(f"std::size_t nx_{p}" if b == "Int" else f"typename {p}" for p, b in fn.generic) + ">")
                params = ", ".join(f"{self.cpp_type(t)} nx_{n}" for n, t in fn.params) or "void"
                if fn.extern_symbol:
                    decl_out.append(f'extern "C" {self.cpp_type(fn.result)} {fn.extern_symbol}({params});')
                    if fn.extern_symbol == emit_name:
                        continue
                linkage = 'extern "C" NX_EXPORT' if fn.exported else "static"
                decl_out.append(f"{linkage} {self.cpp_type(fn.result)} {emit_name}({params});")

        # Emit Enum definitions (with variants, std::variant payload, ==, __str__, methods!)
        for unit in self._unit_order:
            for ename, edef in unit.enums.items():
                if edef.node not in unit.tree.body:
                    continue
                cpp_e = unit.cpp_symbol(ename)
                body_out.append(f"struct {cpp_e} {{")
                tags = ", ".join(edef.variants.keys())
                body_out.append(f"    enum class Tag {{ {tags} }} tag{{Tag::{next(iter(edef.variants))}}};")
                v_structs = []
                for vname, vparams in edef.variants.items():
                    fields_s = " ".join(f"{self.cpp_type(vt)} f{i}{{}};" for i, (_, vt) in enumerate(vparams))
                    eq_checks = " && ".join(f"f{i} == o.f{i}" for i in range(len(vparams))) or "(void)o, true"
                    body_out.append(f"    struct Variant_{vname} {{ {fields_s} bool operator==(const Variant_{vname} &o) const {{ return {eq_checks}; }} }};")
                    v_structs.append(f"Variant_{vname}")
                body_out.append(f"    std::variant<{', '.join(v_structs)}> data{{Variant_{next(iter(edef.variants))}{{}}}};")
                for vname, vparams in edef.variants.items():
                    psig = ", ".join(f"{self.cpp_type(vt)} a{i}" for i, (_, vt) in enumerate(vparams))
                    pinit = ", ".join(f"a{i}" for i in range(len(vparams)))
                    body_out.append(f"    static {cpp_e} {vname}({psig}) {{ {cpp_e} o; o.tag = Tag::{vname}; o.data = Variant_{vname}{{{pinit}}}; return o; }}")
                body_out.append(f"    bool operator==(const {cpp_e} &o) const {{ return tag == o.tag && data == o.data; }}")
                body_out.append(f"    bool operator!=(const {cpp_e} &o) const {{ return !(*this == o); }}")
                body_out.append("    NxStr nx___str__() const {")
                body_out.append("        switch (tag) {")
                for vname, vparams in edef.variants.items():
                    body_out.append(f"            case Tag::{vname}: return NxStr({c_string(f'{ename}.{vname}')});")
                body_out.append("        }")
                body_out.append(f'        return NxStr({c_string(ename)});')
                body_out.append("    }")
                for mname, method in edef.methods.items():
                    body = BodyGenerator(self, unit, method, error_cls=NexirError)
                    self_q = "" if method.mutates_self else "const "
                    body.emit(f"{self_q}{cpp_e}& nx_self = *this;")
                    always_ret = body.block(method.node.body)
                    if method.result != "void" and not always_ret:
                        raise NexirError(f"enum method {ename}.{mname} may finish without returning {method.result}", method.node)
                    if method.result == "void":
                        body.emit("return;")
                    mparams = ", ".join(f"{self.cpp_type(t)} nx_{n}" for n, t in method.params[1:])
                    const_q = "" if method.mutates_self else " const"
                    body_out.extend([f"    {self.cpp_type(method.result)} nx_{mname}({mparams}){const_q} {{", *body.lines, "    }"])
                body_out.append("};")

        # Emit Class definitions (in topological order of inheritance!)
        emitted_classes: set[str] = set()
        def emit_class(unit: ModuleUnit, cdef: ClassDefInfo) -> None:
            if cdef.qualified_name in emitted_classes:
                return
            if cdef.base and cdef.base in self.classes:
                base_cdef = self.classes[cdef.base]
                base_unit = next(u for u in self._unit_order if base_cdef.name in u.classes and u.classes[base_cdef.name] is base_cdef)
                emit_class(base_unit, base_cdef)
            emitted_classes.add(cdef.qualified_name)
            cpp_c = unit.cpp_symbol(cdef.name)
            base_clause = f" : public {self.cpp_type(cdef.base)}" if cdef.base else ""
            if cdef.generic:
                body_out.append("template <" + ", ".join(f"std::size_t nx_{p}" if b == "Int" else f"typename {p}" for p, b in cdef.generic) + ">")
            body_out.append(f"struct {cpp_c}{base_clause} {{")
            for field, typ in cdef.fields:
                if field in cdef.field_defaults:
                    dv_gen = BodyGenerator(self, unit, Function(ast.FunctionDef(name="__init__", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[]), (), "void", frozenset(), False), error_cls=NexirError)
                    dv_cpp = dv_gen.argument(cdef.field_defaults[field], typ)
                    body_out.append(f"    {self.cpp_type(typ)} nx_{field} = {dv_cpp};")
                else:
                    body_out.append(f"    {self.cpp_type(typ)} nx_{field}{{}};")
            has_custom_str = "__str__" in cdef.methods
            has_del = "__del__" in cdef.methods
            has_init = "__init__" in cdef.methods
            is_polymorphic = bool(cdef.base or any(b.base == cdef.qualified_name for b in self.classes.values()))
            if is_polymorphic and not has_del:
                body_out.append(f"    virtual ~{cpp_c}() = default;")
            if not has_init and (is_polymorphic or cdef.field_defaults):
                all_f = self.all_class_fields(cdef.qualified_name)
                body_out.append(f"    {cpp_c}() = default;")
                if all_f:
                    cparams = ", ".join(f"{self.cpp_type(ft)} nx_p_{fn_n}" for fn_n, ft, _ in all_f)
                    cassigns = " ".join(f"this->nx_{fn_n} = std::move(nx_p_{fn_n});" for fn_n, _, _ in all_f)
                    body_out.append(f"    {cpp_c}({cparams}) {{ {cassigns} }}")
            for mname, method in cdef.methods.items():
                body = BodyGenerator(self, unit, method, error_cls=NexirError)
                self_q = "" if method.mutates_self else "const "
                body.emit(f"{self_q}{cpp_c}& nx_self = *this;")
                always_ret = body.block(method.node.body)
                if method.result != "void" and not always_ret:
                    raise NexirError(f"method {cdef.name}.{mname} may finish without returning {method.result}", method.node)
                if method.result == "void":
                    body.emit("return;")
                mparams = ", ".join(f"{self.cpp_type(t)} nx_{n}" for n, t in method.params[1:])
                if mname == "__init__":
                    sig = f"{cpp_c}({mparams})"
                elif mname == "__del__":
                    virt = "virtual " if (cdef.base or any(b.base == cdef.qualified_name for b in self.classes.values())) else ""
                    sig = f"{virt}~{cpp_c}()"
                else:
                    is_base = any(b.base == cdef.qualified_name for b in self.classes.values())
                    is_ovr = cdef.base is not None
                    virt = "virtual " if (is_base or is_ovr) else ""
                    const_q = "" if method.mutates_self else " const"
                    sig = f"{virt}{self.cpp_type(method.result)} nx_{mname}({mparams}){const_q}"
                body_out.extend([f"    {sig} {{", *body.lines, "    }"])
            if not has_custom_str:
                body_out.append("    NxStr nx___str__() const {")
                body_out.append("        NxStrBuilder sb;")
                body_out.append(f'        sb.append({c_string(cdef.name + "(")});')
                for fi, (fn_n, _) in enumerate(cdef.fields):
                    if fi:
                        body_out.append('        sb.append(", ");')
                    body_out.append(f'        sb.append({c_string(fn_n + "=")}).append(nx_{fn_n});')
                body_out.append('        sb.append(")");')
                body_out.append("        return sb.finish();")
                body_out.append("    }")
            body_out.append("};")

        for unit in self._unit_order:
            for cname, cdef in unit.classes.items():
                if cdef.node in unit.tree.body:
                    emit_class(unit, cdef)

        for trait, methods in self.traits.items():
            tests = [
                f"decltype(std::declval<T&>().nx_{m}({', '.join('std::declval<' + ('T' if typ == 'Self' else self.cpp_type(typ)) + '>()' for typ in ps)}))"
                for m, (ps, _) in methods.items()
            ]
            detection = ", ".join(tests)
            checks = " && ".join(
                f"std::is_same_v<{test}, {('T' if res == 'Self' else self.cpp_type(res))}>"
                for test, (_, res) in zip(tests, methods.values())
            ) or "true"
            body_out.append(f"template <typename T, typename = void> struct nx_trait_{trait} : std::false_type {{}};")
            body_out.append(f"template <typename T> struct nx_trait_{trait}<T, std::void_t<{detection}>> : std::bool_constant<{checks}> {{}};")
        body_out.append("")

        for unit in self._unit_order:
            for name, fn in unit.functions.items():
                if unit is not self.entry_unit and name == "main":
                    continue
                emit_name = f"nx_{name}" if (unit is self.entry_unit or fn.exported or fn.extern_symbol) else unit.cpp_symbol(name)
                body = BodyGenerator(self, unit, fn, error_cls=NexirError)
                body_out.extend(body.generate(emit_name, name))
                body_out.append("")

        out = ["/* Generated by NEXIR 2.0 C++17 backend. Do not edit. */", *sorted(self.cpp_headers), PRELUDE, ""]
        if self.uses_python:
            out.extend(self._emit_python_abi_cpp())
        out.extend(decl_out)
        out.extend(body_out)

        if self.require_main:
            hook = "nx_py_abi_sandbox_hook(nx_py_sandbox_callback); " if self.uses_python else ""
            if self.entry_unit.functions["main"].result == "i32":
                out.append(f"int main(void) {{ NxDebugMemGuard __nx_dbg_mem; NxSandbox::instance().init_from_env(); {hook}int __rc = nx_main(); NxHybridExecutor::instance().shutdown(); return __rc; }}")
            else:
                out.append(f"int main(void) {{ NxDebugMemGuard __nx_dbg_mem; NxSandbox::instance().init_from_env(); {hook}nx_main(); NxHybridExecutor::instance().shutdown(); return 0; }}")
        return "\n".join(out) + "\n"

    def _guaranteed_ret(self, nodes: list[ast.stmt]) -> bool:
        from .compiler_body import guaranteed_return
        return guaranteed_return(nodes)


def compile_source(source: str, *, debug: bool = False, require_main: bool = True) -> str:
    return Compiler(source, require_main=require_main, debug=debug).compile()


def compile_file(
    path: Path,
    *,
    debug: bool = False,
    require_main: bool = True,
    audit_log: CompileAuditLog | None = None,
    strict_sandbox: bool = False,
    allowed_py_modules: set[str] | None = None,
) -> str:
    path = path.resolve()
    if path.suffix != ".nx":
        raise NexirError("source file must end in .nx")
    if not path.is_file():
        raise NexirError(f"source file not found: {path}")
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise NexirError("source file must be UTF-8") from exc
    return Compiler(
        source,
        entry_path=path,
        require_main=require_main,
        debug=debug,
        audit_log=audit_log,
        strict_sandbox=strict_sandbox,
        allowed_py_modules=allowed_py_modules,
    ).compile()
