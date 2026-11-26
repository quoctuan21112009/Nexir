"""Strict NEXIR frontend and C++17 backend.

Python's AST supplies indentation and expression parsing during bootstrap. Every
accepted syntax node is checked below; arbitrary Python is never executed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import io
from pathlib import Path
import re
import tokenize
from .runtime_cpp import PRELUDE


TYPES = {"i32": "int32_t", "i64": "int64_t", "f32": "float", "f64": "double", "bool": "bool", "str": "const char *", "void": "void", "dyn": "NxValue", "Ring[f64]": "nx_ring_ref_f64"}
for _element in ("i32", "i64", "f32", "f64"):
    TYPES[f"Span[{_element}]"] = f"nx_span_{_element}"
    TYPES[f"MutSpan[{_element}]"] = f"nx_mutspan_{_element}"
    TYPES[f"Buffer[{_element}]"] = f"nx_buffer<{TYPES[_element]}>"
    TYPES[f"owned[Buffer[{_element}]]"] = f"std::unique_ptr<nx_buffer<{TYPES[_element]}>>"
    TYPES[f"borrowed[Buffer[{_element}]]"] = f"const nx_buffer<{TYPES[_element]}>&"
    TYPES[f"mut[Buffer[{_element}]]"] = f"nx_buffer<{TYPES[_element]}>&"
for _element in ("i32", "i64", "f32", "f64", "str"):
    TYPES[f"List[{_element}]"] = f"std::vector<{TYPES[_element]}>"
    TYPES[f"Set[{_element}]"] = f"std::unordered_set<{TYPES[_element]}, nx_hash<{TYPES[_element]}>, nx_equal<{TYPES[_element]}>>"
    for _value in ("i32", "i64", "f32", "f64", "bool", "str"):
        TYPES[f"Dict[{_element},{_value}]"] = f"std::unordered_map<{TYPES[_element]}, {TYPES[_value]}, nx_hash<{TYPES[_element]}>, nx_equal<{TYPES[_element]}>>"
INTEGERS = {"i32", "i64"}
NUMERICS = INTEGERS | {"f32", "f64"}
CAPABILITIES = {"filesystem.read", "network.connect", "memory.alloc", "ffi.cpp", "python.interop"}


def _span_element(typ: str) -> str | None:
    if typ.startswith("Span[") or typ.startswith("MutSpan["):
        return typ[typ.index("[") + 1:-1]
    return None


def _list_element(typ: str) -> str | None:
    return typ[5:-1] if typ.startswith("List[") and typ.endswith("]") else None


def _set_element(typ: str) -> str | None:
    return typ[4:-1] if typ.startswith("Set[") and typ.endswith("]") else None


def _dict_parts(typ: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"Dict\[(i32|i64|f32|f64|str),(i32|i64|f32|f64|bool|str)\]", typ)
    return (match.group(1), match.group(2)) if match else None


def _tuple_parts(typ: str) -> tuple[str, ...] | None:
    if not typ.startswith("Tuple[") or not typ.endswith("]"):
        return None
    return tuple(typ[6:-1].split(","))


def _simd_parts(typ: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"SIMD\[([A-Za-z_]\w*),([A-Za-z_]\w*|\d+)\]", typ)
    return (match.group(1), match.group(2)) if match else None


def _buffer_parts(typ: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(owned|borrowed|mut)\[Buffer\[(i32|i64|f32|f64)\]\]", typ)
    return (match.group(1), match.group(2)) if match else None


class NexirError(Exception):
    def __init__(self, message: str, node: ast.AST | None = None):
        self.message = message
        self.line = getattr(node, "lineno", None)
        self.column = getattr(node, "col_offset", None)
        super().__init__(message)

    def __str__(self) -> str:
        if self.line is None:
            return self.message
        return f"line {self.line}, column {self.column + 1}: {self.message}"


@dataclass(frozen=True)
class Function:
    node: ast.FunctionDef
    params: tuple[tuple[str, str], ...]
    result: str
    caps: frozenset[str]
    exported: bool
    extern_symbol: str | None = None
    owner: str | None = None
    strict: bool = False
    generic: tuple[tuple[str, str], ...] = ()
    raises: bool = False
    py_call: tuple[str, str] | None = None
    defaults: tuple[ast.expr | None, ...] = ()
    cpp_symbol: str | None = None


@dataclass(frozen=True)
class Struct:
    node: ast.ClassDef
    fields: tuple[tuple[str, str], ...]
    methods: dict[str, Function]


def _normalize_source(source: str) -> tuple[str, dict[int, str], dict[int, tuple[tuple[str, str], ...]], set[int]]:
    """Translate NX-only declarations while leaving strings/comments untouched."""
    lines = source.splitlines(keepends=True)
    declarations: dict[int, str] = {}
    generics: dict[int, tuple[tuple[str, str], ...]] = {}
    raises_lines: set[int] = set()
    first: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE,
                              tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER}:
                continue
            line = token.start[0]
            if line in first:
                continue
            first.add(line)
            if token.type == tokenize.NAME and token.string in {"fn", "struct", "trait"}:
                declarations[line] = token.string
                replacement = "def" if token.string == "fn" else "class"
                start, end = token.start[1], token.end[1]
                lines[line - 1] = lines[line - 1][:start] + replacement + lines[line - 1][end:]
                if token.string == "fn":
                    match = re.match(r"^(\s*def\s+[A-Za-z_]\w*)\[([^\]]+)\](\s*\()", lines[line - 1])
                    if match:
                        params: list[tuple[str, str]] = []
                        for item in match.group(2).split(","):
                            part = item.strip().split(":")
                            if len(part) != 2 or not part[0].strip().isidentifier() or not part[1].strip().isidentifier():
                                raise NexirError("generic parameters must use NAME: BOUND")
                            params.append((part[0].strip(), part[1].strip()))
                        generics[line] = tuple(params)
                        lines[line - 1] = match.group(1) + match.group(3) + lines[line - 1][match.end():]
                    raises_match = re.match(r"^(\s*def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*->\s*)([A-Za-z_]\w*)(\s+)raises(\s*:)", lines[line - 1])
                    if raises_match:
                        raises_lines.add(line)
                        lines[line - 1] = (raises_match.group(1) + f"Result[{raises_match.group(2)}]" + raises_match.group(4)
                                           + lines[line - 1][raises_match.end():])
    except tokenize.TokenError as exc:
        raise NexirError(f"tokenization error: {exc}") from exc
    return "".join(lines), declarations, generics, raises_lines


def _type(node: ast.expr | None, *, default: str = "void", types: dict[str, str] | None = None) -> str:
    types = TYPES if types is None else types
    if node is None:
        return default
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "Result" and isinstance(node.slice, ast.Name):
        element = node.slice.id
        if element not in NUMERICS | {"bool", "str"}:
            raise NexirError("Result currently requires a numeric, bool or str value type", node)
        typ = f"Result[{element}]"
        types[typ] = f"nx_result<{types[element]}>"
        return typ
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "Optional" and isinstance(node.slice, ast.Name):
        element = node.slice.id
        if element not in NUMERICS | {"bool", "str"}:
            raise NexirError("Optional requires a known value type", node)
        typ = f"Optional[{element}]"
        types[typ] = f"std::optional<{types[element]}>"
        return typ
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "Dict" and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2 and all(isinstance(part, ast.Name) for part in node.slice.elts):
        typ = f"Dict[{node.slice.elts[0].id},{node.slice.elts[1].id}]"
        if typ in types:
            return typ
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "Tuple" and isinstance(node.slice, ast.Tuple) and 2 <= len(node.slice.elts) <= 8 and all(isinstance(part, ast.Name) for part in node.slice.elts):
        parts = [part.id for part in node.slice.elts]
        if any(part not in NUMERICS | {"bool", "str"} for part in parts):
            raise NexirError("Tuple supports primitive value types", node)
        typ = "Tuple[" + ",".join(parts) + "]"
        types[typ] = "std::tuple<" + ", ".join(types[part] for part in parts) + ">"
        return typ
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "SIMD" and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
        element, width = node.slice.elts
        if not isinstance(element, ast.Name) or (element.id not in NUMERICS and f"$numeric:{element.id}" not in types):
            raise NexirError("SIMD requires a numeric element type", node)
        if isinstance(width, ast.Constant) and type(width.value) is int and 1 <= width.value <= 1024:
            width_name, width_cpp = str(width.value), str(width.value)
        elif isinstance(width, ast.Name) and f"$const:{width.id}" in types:
            width_name, width_cpp = width.id, types[f"$const:{width.id}"]
        else:
            raise NexirError("SIMD width must be 1 to 1024 or a generic Int parameter", node)
        typ = f"SIMD[{element.id},{width_name}]"
        types[typ] = f"nx_simd<{types[element.id]}, {width_cpp}>"
        return typ
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Name):
        typ = f"{node.value.id}[{node.slice.id}]"
        if typ in types:
            return typ
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in {"owned", "borrowed", "mut"} and isinstance(node.slice, ast.Subscript):
        inner = _type(node.slice, types=types)
        typ = f"{node.value.id}[{inner}]"
        if typ in types:
            return typ
    if not isinstance(node, ast.Name) or node.id not in types:
        raise NexirError("expected one of: " + ", ".join(types), node)
    return node.id


def _compatible(actual: str, expected: str) -> bool:
    return (actual == expected or (expected == "dyn" and actual in NUMERICS | {"str", "bool"})
            or (expected.startswith("Optional[") and (actual == "None" or actual == expected[9:-1]))
            or (actual, expected) in {("i32", "i64"), ("f32", "f64")}
            or actual.startswith("MutSpan[") and expected == "Span[" + actual[8:])


def _coerce(value: str, actual: str, expected: str) -> str:
    if expected.startswith("Optional[") and actual != expected:
        return "std::nullopt" if actual == "None" else f"std::optional<{TYPES[actual]}>({value})"
    if expected == "dyn" and actual != "dyn":
        return f"nx_box({value})"
    if actual.startswith("MutSpan[") and expected == "Span[" + actual[8:]:
        elem = _span_element(actual)
        return f"nx_as_span_{elem}({value})"
    return value


def _c_string(value: str) -> str:
    # Fixed-width octal escapes are unambiguous next to hex digits.
    return '"' + "".join(
        chr(b) if 32 <= b < 127 and b not in (34, 92) else f"\\{b:03o}"
        for b in value.encode("utf-8")
    ) + '"'


def _name(name: str, node: ast.AST | None = None) -> str:
    if not name.isidentifier() or name.startswith("nx_") or name in TYPES or name in {"Buffer", "List", "Dict", "Set", "Tuple", "Optional", "buffer_span", "buffer_mutspan", "print", "range", "parallel_range", "len", "zeros_f32", "zeros_f64", "read_text", "tcp_connect", "type_name", "as_i64", "as_f64", "ring_f64", "ring_write", "ring_read", "ring_valid", "owned", "move", "is_ok", "unwrap", "error", "value_or"}:
        raise NexirError(f"reserved name: {name}", node)
    return "nx_" + name


class Compiler:
    def __init__(self, source: str, imports: set[str] | None = None):
        self.imports = imports or set()
        self.cpp_headers: set[str] = set()
        source, self.declarations, self.generics, self.raises_lines = _normalize_source(source)
        try:
            self.tree = ast.parse(source)
        except SyntaxError as exc:
            raise NexirError(f"syntax error: {exc.msg} at line {exc.lineno}, column {exc.offset}") from exc
        self.types = dict(TYPES)
        trait_declarations = [node for node in self.tree.body if isinstance(node, ast.ClassDef) and self.declarations.get(node.lineno) == "trait"]
        self.trait_nodes = {node.name: node for node in trait_declarations}
        if len(self.trait_nodes) != len(trait_declarations):
            raise NexirError("duplicate trait declaration")
        self.traits: dict[str, dict[str, tuple[tuple[str, ...], str]]] = {}
        for name, node in self.trait_nodes.items():
            _name(name, node)
            if node.decorator_list or node.bases or node.keywords:
                raise NexirError("trait does not support decorators or inheritance", node)
            methods: dict[str, tuple[tuple[str, ...], str]] = {}
            for member in node.body:
                if isinstance(member, ast.Pass):
                    continue
                if not isinstance(member, ast.FunctionDef) or member.decorator_list or not member.args.args or member.args.args[0].arg != "self" or member.args.args[0].annotation is not None or len(member.body) != 1 or not isinstance(member.body[0], ast.Pass) or member.args.defaults or member.args.kwonlyargs or member.args.vararg or member.args.kwarg:
                    raise NexirError("trait methods require typed parameters and a pass body", member)
                _name(member.name, member)
                params = []
                for arg in member.args.args[1:]:
                    typ = "Self" if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "Self" else _type(arg.annotation, default="", types=self.types)
                    if typ not in NUMERICS | {"bool", "str", "Self"}:
                        raise NexirError("trait parameter must be numeric, bool, str or Self", arg)
                    params.append(typ)
                result = "Self" if isinstance(member.returns, ast.Name) and member.returns.id == "Self" else _type(member.returns, types=self.types)
                if result not in NUMERICS | {"bool", "Self"}:
                    raise NexirError("trait method result must be numeric, bool or Self", member)
                if member.name in methods:
                    raise NexirError(f"duplicate trait method: {member.name}", member)
                methods[member.name] = (tuple(params), result)
            self.traits[name] = methods
        generic_names: set[str] = set()
        for params in self.generics.values():
            for param, bound in params:
                if bound not in {"Number", "Integer", "Int"} | self.traits.keys():
                    raise NexirError(f"unsupported generic bound: {bound}")
                if param in self.types and param not in generic_names:
                    raise NexirError(f"generic type name is reserved: {param}")
                if bound == "Int":
                    self.types[f"$const:{param}"] = f"nx_{param}"
                else:
                    self.types[param] = param
                    generic_names.add(param)
                    if bound in {"Number", "Integer"}:
                        self.types[f"$numeric:{param}"] = "true"
        self.generic_names = generic_names
        self.globals: dict[str, tuple[str, str]] = {}
        for node in self.tree.body:
            if not isinstance(node, ast.AnnAssign):
                continue
            if not isinstance(node.target, ast.Name) or node.value is None:
                raise NexirError("file-level variables require a typed name and constant", node)
            name = node.target.id
            _name(name, node)
            typ = _type(node.annotation, types=self.types)
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, MemoryError) as exc:
                raise NexirError("file-level initializer must be a constant", node.value) from exc
            if type(value) is str and typ == "str":
                code = _c_string(value)
            elif type(value) is bool and typ == "bool":
                code = "true" if value else "false"
            elif type(value) is int and typ in INTEGERS:
                code = str(value)
            elif type(value) is float and typ in {"f32", "f64"}:
                code = repr(value)
            elif value is None and typ.startswith("Optional["):
                code = "std::nullopt"
            else:
                raise NexirError("file-level constant does not match its type", node.value)
            if name in self.globals:
                raise NexirError(f"duplicate file-level variable: {name}", node)
            self.globals[name] = (typ, code)
        self.structs: dict[str, Struct] = {}
        self.functions: dict[str, Function] = {}
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and self.declarations.get(node.lineno) == "struct":
                _name(node.name, node)
                if node.name in self.types or node.name in self.structs or node.name in self.traits:
                    raise NexirError(f"duplicate or reserved struct: {node.name}", node)
                self.types[node.name] = f"nx_{node.name}"
                self.types[f"owned[{node.name}]"] = f"std::unique_ptr<nx_{node.name}>"
                self.types[f"borrowed[{node.name}]"] = f"const nx_{node.name}&"
                self.types[f"mut[{node.name}]"] = f"nx_{node.name}&"
                self.structs[node.name] = Struct(node, (), {})
        for name, struct in list(self.structs.items()):
            node = struct.node
            if node.decorator_list or node.bases or node.keywords:
                raise NexirError("struct does not support decorators or inheritance", node)
            fields: list[tuple[str, str]] = []
            methods: dict[str, Function] = {}
            for member in node.body:
                if isinstance(member, ast.Pass):
                    continue
                if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name) and member.value is None:
                    field = member.target.id
                    _name(field, member)
                    typ = _type(member.annotation, types=self.types)
                    if typ in self.generic_names:
                        raise NexirError("struct fields cannot use a function generic type", member)
                    if typ in {"void", f"borrowed[{name}]", f"mut[{name}]"} or typ.startswith(("borrowed[", "mut[")):
                        raise NexirError("struct field requires a value or owned type", member)
                    if typ == name:
                        raise NexirError("struct cannot contain itself by value; use owned[Struct]", member)
                    if field in dict(fields) or field in methods:
                        raise NexirError(f"duplicate struct member: {field}", member)
                    fields.append((field, typ))
                    continue
                if isinstance(member, ast.FunctionDef):
                    method = self._parse_method(name, member)
                    if member.name in methods or member.name in dict(fields):
                        raise NexirError(f"duplicate struct member: {member.name}", member)
                    methods[member.name] = method
                    continue
                raise NexirError("struct body supports fields and methods only", member)
            self.structs[name] = Struct(node, tuple(fields), methods)
        for node in self.tree.body:
            if isinstance(node, ast.AnnAssign):
                continue
            if isinstance(node, ast.ClassDef) and node.name in self.traits and self.declarations.get(node.lineno) == "trait":
                continue
            if isinstance(node, ast.ClassDef) and node.name in self.structs and self.declarations.get(node.lineno) == "struct":
                continue
            if not isinstance(node, ast.FunctionDef):
                raise NexirError("only function definitions are allowed at module level", node)
            _name(node.name, node)
            if node.name in self.functions or node.name in self.structs or node.name in self.traits or node.name in self.globals:
                raise NexirError(f"duplicate function: {node.name}", node)
            strict = self.declarations.get(node.lineno) == "fn"
            generic = self.generics.get(node.lineno, ())
            allowed_generic = dict(generic)
            caps: set[str] = set()
            exported = False
            extern_symbol = None
            py_call = None
            py_fn_used = False
            cpp_symbol = None
            seen: set[str] = set()
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id in {"native", "export"}:
                    label = decorator.id
                    exported |= label == "export"
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "cap" and not decorator.keywords:
                    label = "cap"
                    if not decorator.args:
                        raise NexirError("@cap requires at least one capability", decorator)
                    for item in decorator.args:
                        if not isinstance(item, ast.Constant) or type(item.value) is not str or item.value not in CAPABILITIES:
                            raise NexirError("unknown capability; allowed: " + ", ".join(sorted(CAPABILITIES)), item)
                        caps.add(item.value)
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "extern" and not decorator.keywords:
                    label = "extern"
                    if len(decorator.args) != 1 or not isinstance(decorator.args[0], ast.Constant) or type(decorator.args[0].value) is not str:
                        raise NexirError("@extern requires one C symbol name", decorator)
                    symbol = decorator.args[0].value
                    if not symbol.isascii() or not symbol.isidentifier():
                        raise NexirError("@extern symbol must be an ASCII identifier", decorator)
                    extern_symbol = symbol
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id in {"py_call", "py_fn"} and not decorator.keywords:
                    label = decorator.func.id
                    if len(decorator.args) != 2 or any(not isinstance(item, ast.Constant) or type(item.value) is not str for item in decorator.args):
                        raise NexirError('@py_call requires module and function string names', decorator)
                    module, function = (item.value for item in decorator.args)
                    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) or not re.fullmatch(r"[A-Za-z_]\w*", function):
                        raise NexirError('@py_call names must be Python identifiers', decorator)
                    py_call = (module, function)
                    py_fn_used = label == "py_fn"
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "cpp_header" and not decorator.keywords:
                    label = "cpp_header"
                    if len(decorator.args) != 1 or not isinstance(decorator.args[0], ast.Constant) or type(decorator.args[0].value) is not str:
                        raise NexirError('@cpp_header expects one header string', decorator)
                    header = decorator.args[0].value
                    local = header.startswith('"') and header.endswith('"')
                    path = header[1:-1] if local else header
                    if not re.fullmatch(r"[A-Za-z0-9_./-]+", path) or ".." in path or path.startswith("/"):
                        raise NexirError("invalid C++ header path", decorator)
                    self.cpp_headers.add(f'#include "{path}"' if local else f"#include <{path}>")
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "cpp_fn" and not decorator.keywords:
                    label = "cpp_fn"
                    if len(decorator.args) != 1 or not isinstance(decorator.args[0], ast.Constant) or type(decorator.args[0].value) is not str:
                        raise NexirError('@cpp_fn expects one C++ symbol', decorator)
                    cpp_symbol = decorator.args[0].value
                    if not re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", cpp_symbol):
                        raise NexirError("@cpp_fn expects a simple C++ qualified name", decorator)
                else:
                    raise NexirError("unsupported decorator", decorator)
                if label in seen and label != "cpp_header":
                    raise NexirError(f"duplicate @{label}", decorator)
                seen.add(label)
            args = node.args
            if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.kw_defaults:
                raise NexirError("only positional or named fixed parameters are supported", node)
            params = []
            for arg in args.args:
                _name(arg.arg, arg)
                typ = _type(arg.annotation, default="", types=self.types)
                if typ in {"", "void"}:
                    raise NexirError("parameters require a non-void type", arg)
                if typ in self.generic_names and typ not in allowed_generic:
                    raise NexirError(f"unbound generic type: {typ}", arg)
                if strict and typ == "dyn":
                    raise NexirError("fn parameters cannot use dyn", arg)
                if arg.arg in [n for n, _ in params]:
                    raise NexirError(f"duplicate parameter: {arg.arg}", arg)
                params.append((arg.arg, typ))
            defaults: tuple[ast.expr | None, ...] = (None,) * (len(params) - len(args.defaults)) + tuple(args.defaults)
            for item, (_, typ) in zip(defaults, params):
                if item is not None:
                    try:
                        default_value = ast.literal_eval(item)
                    except (ValueError, TypeError, SyntaxError, MemoryError) as exc:
                        raise NexirError("default must be a constant matching the parameter type", item) from exc
                    if not (type(default_value) is str and typ == "str" or type(default_value) is bool and typ == "bool" or type(default_value) is int and typ in INTEGERS or type(default_value) is float and typ in {"f32", "f64"} or default_value is None and typ.startswith("Optional[")):
                        raise NexirError("default must be a constant matching the parameter type", item)
            if strict and node.returns is None:
                raise NexirError("fn requires an explicit return type", node)
            result = _type(node.returns, types=self.types)
            if result in self.generic_names and result not in allowed_generic:
                raise NexirError(f"unbound generic type: {result}", node.returns)
            raises = node.lineno in self.raises_lines
            if raises and (exported or extern_symbol):
                raise NexirError("raises functions cannot use @export or @extern", node)
            if strict and result == "dyn":
                raise NexirError("fn cannot return dyn", node)
            if _span_element(result):
                raise NexirError("returning Span is not supported until lifetime checking is available", node.returns)
            if result.startswith(("borrowed[", "mut[")):
                raise NexirError("returning a borrow is not supported until lifetime checking is available", node.returns)
            if exported and (result not in NUMERICS | {"bool", "void"} or any(t not in NUMERICS | {"bool"} and not _span_element(t) for _, t in params)):
                raise NexirError("@export supports numeric, bool and Span signatures only", node)
            if extern_symbol and (exported or caps or result == "dyn" or any(t == "dyn" or t == "Ring[f64]" for _, t in params)):
                raise NexirError("@extern requires a plain C ABI signature without @export or @cap", node)
            if extern_symbol and (len(node.body) != 1 or not isinstance(node.body[0], ast.Pass)):
                raise NexirError("@extern declaration body must be pass", node)
            if generic and (exported or extern_symbol):
                raise NexirError("generic functions cannot use @export or @extern", node)
            if py_call:
                if cpp_symbol or exported or extern_symbol or generic or raises or any(defaults) or len(params) != 1 or params[0][1] not in {"Span[f64]", "MutSpan[f64]"} or result not in {"f64", "str", "owned[Buffer[f64]]"}:
                    raise NexirError('@py_call requires one f64 Span/MutSpan argument and f64, str or owned[Buffer[f64]] result', node)
                if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
                    raise NexirError('@py_call declaration body must be pass', node)
                if result != "f64":
                    caps.add("memory.alloc")
                if py_fn_used and "python.interop" not in caps:
                    raise NexirError('@py_fn requires @cap("python.interop")', node)
            if cpp_symbol:
                if exported or extern_symbol or generic or raises or any(defaults) or result not in NUMERICS | {"bool", "void", "str"} or any(typ not in NUMERICS | {"bool", "str"} for _, typ in params):
                    raise NexirError("@cpp_fn requires a primitive C++ signature without other FFI decorators", node)
                if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
                    raise NexirError("@cpp_fn declaration body must be pass", node)
                if "ffi.cpp" not in caps:
                    raise NexirError('@cpp_fn requires @cap("ffi.cpp")', node)
            self.functions[node.name] = Function(node, tuple(params), result, frozenset(caps), exported, extern_symbol, strict=strict, generic=generic, raises=raises, py_call=py_call, defaults=defaults, cpp_symbol=cpp_symbol)
        if "main" not in self.functions:
            raise NexirError("program requires def main()")
        main = self.functions["main"]
        if main.params or main.result not in {"void", "i32"}:
            raise NexirError("main must take no arguments and return void or i32", main.node)

    def _parse_method(self, owner: str, node: ast.FunctionDef) -> Function:
        if node.name not in {"__init__", "__del__"}:
            _name(node.name, node)
        if not node.args.args or node.args.args[0].arg != "self" or node.args.args[0].annotation is not None:
            raise NexirError("struct method must start with unannotated self", node)
        args = node.args
        if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults or args.kw_defaults:
            raise NexirError("only required positional method parameters are supported", node)
        caps: set[str] = set()
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Name) or decorator.func.id != "cap" or decorator.keywords or not decorator.args:
                raise NexirError("struct methods support @cap(...) only", decorator)
            for item in decorator.args:
                if not isinstance(item, ast.Constant) or type(item.value) is not str or item.value not in CAPABILITIES:
                    raise NexirError("unknown method capability", item)
                caps.add(item.value)
        params = [("self", owner)]
        for arg in args.args[1:]:
            _name(arg.arg, arg)
            typ = _type(arg.annotation, default="", types=self.types)
            if typ in self.generic_names:
                raise NexirError("struct methods cannot use function generic types", arg)
            if typ in {"", "void"}:
                raise NexirError("method parameters require a non-void type", arg)
            if arg.arg in [n for n, _ in params]:
                raise NexirError(f"duplicate parameter: {arg.arg}", arg)
            params.append((arg.arg, typ))
        strict = self.declarations.get(node.lineno) == "fn"
        if strict and node.returns is None:
            raise NexirError("fn requires an explicit return type", node)
        result = _type(node.returns, types=self.types)
        if result in self.generic_names:
            raise NexirError("struct methods cannot use function generic types", node.returns)
        if node.name in {"__init__", "__del__"} and result != "void":
            raise NexirError("constructor/destructor must return void", node)
        if node.name == "__del__" and len(params) != 1:
            raise NexirError("__del__ takes only self", node)
        if _span_element(result) or result.startswith(("borrowed[", "mut[")):
            raise NexirError("method cannot return a borrowed view", node)
        if self.generics.get(node.lineno):
            raise NexirError("generic struct methods are not supported yet", node)
        return Function(node, tuple(params), result, frozenset(caps), False, owner=owner, strict=strict)
    def compile(self) -> str:
        out = ["/* Generated by NEXIR C++17 backend. Do not edit. */", *sorted(self.cpp_headers), PRELUDE, ""]
        for name, (typ, value) in self.globals.items():
            qualifier = " const" if name.isupper() else ""
            out.append(f"static {self.types[typ]}{qualifier} nx_{name} = {value};")
        if any(fn.py_call for fn in self.functions.values()):
            out.extend([
                "/* NX_PY_CALL */",
                'extern "C" double nx_py_call_f64_span(const char*, const char*, nx_span_f64);',
                'extern "C" double nx_py_call_f64_mutspan(const char*, const char*, nx_mutspan_f64);',
                'extern "C" char* nx_py_call_str_span(const char*, const char*, nx_span_f64);',
                'extern "C" char* nx_py_call_str_mutspan(const char*, const char*, nx_mutspan_f64);',
                'extern "C" nx_mutspan_f64 nx_py_call_buffer_span(const char*, const char*, nx_span_f64);',
                'extern "C" nx_mutspan_f64 nx_py_call_buffer_mutspan(const char*, const char*, nx_mutspan_f64);',
            ])
            out.extend(f"/* NX_PY_PRELOAD {module} */" for module in sorted({fn.py_call[0] for fn in self.functions.values() if fn.py_call}))
        for name in self.structs:
            out.append(f"struct nx_{name};")
        for name, fn in self.functions.items():
            if fn.generic:
                out.append("template <" + ", ".join(f"std::size_t nx_{param}" if bound == "Int" else f"typename {param}" for param, bound in fn.generic) + ">")
            params = ", ".join(f"{self.types[t]} nx_{n}" for n, t in fn.params) or "void"
            if fn.extern_symbol:
                out.append(f'extern "C" {self.types[fn.result]} {fn.extern_symbol}({params});')
                if fn.extern_symbol == f"nx_{name}":
                    continue
            linkage = 'extern "C" NX_EXPORT' if fn.exported else "static"
            out.append(f"{linkage} {self.types[fn.result]} nx_{name}({params});")
        for name, struct in self.structs.items():
            out.append(f"struct nx_{name} {{")
            for field, typ in struct.fields:
                if typ in self.structs and list(self.structs).index(typ) > list(self.structs).index(name):
                    raise NexirError(f"struct field {field} refers to a later struct; reorder declarations", struct.node)
                out.append(f"    {self.types[typ]} nx_{field}{{}};")
            for method_name, method in struct.methods.items():
                body = _Body(self, method)
                body.emit(f"nx_{name}& nx_self = *this;")
                body.block(method.node.body)
                if method.result != "void" and not _guaranteed_return(method.node.body):
                    raise NexirError(f"method {name}.{method_name} may finish without returning {method.result}", method.node)
                if method.result == "void":
                    body.emit("return;")
                params = ", ".join(f"{self.types[t]} nx_{n}" for n, t in method.params[1:])
                if method_name == "__init__":
                    signature = f"nx_{name}({params})"
                elif method_name == "__del__":
                    signature = f"~nx_{name}()"
                else:
                    signature = f"{self.types[method.result]} nx_{method_name}({params})"
                out.extend([f"    {signature} {{", *body.lines, "    }"])
            out.append("};")
        for trait, methods in self.traits.items():
            tests = [f"decltype(std::declval<T&>().nx_{method}({', '.join('std::declval<' + ('T' if typ == 'Self' else self.types[typ]) + '>()' for typ in params)}))" for method, (params, _) in methods.items()]
            detection = ", ".join(tests)
            checks = " && ".join(f"std::is_same_v<{test}, {('T' if result == 'Self' else self.types[result])}>" for test, (_, result) in zip(tests, methods.values())) or "true"
            out.append(f"template <typename T, typename = void> struct nx_trait_{trait} : std::false_type {{}};")
            out.append(f"template <typename T> struct nx_trait_{trait}<T, std::void_t<{detection}>> : std::bool_constant<{checks}> {{}};")
        out.append("")
        for name, fn in self.functions.items():
            body = _Body(self, fn)
            out.extend(body.generate(name))
            out.append("")
        ret = "return nx_main();" if self.functions["main"].result == "i32" else "nx_main(); return 0;"
        out.append(f"int main(void) {{ {ret} }}")
        return "\n".join(out) + "\n"


class _Body:
    def __init__(self, compiler: Compiler, function: Function):
        self.compiler = compiler
        self.fn = function
        self.scopes: list[dict[str, str]] = [dict(function.params)]
        self.lines: list[str] = []
        self.depth = 1
        self.loop_depth = 0
        self.serial = 0
        self.moved: set[str] = set()

    def numeric(self, typ: str) -> bool:
        return typ in NUMERICS or dict(self.fn.generic).get(typ) in {"Number", "Integer"}

    def emit(self, line: str) -> None:
        self.lines.append("    " * self.depth + line)

    def lookup(self, name: str, node: ast.AST) -> str:
        if name in self.moved:
            raise NexirError(f"use after move: {name}", node)
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        if name in self.compiler.globals:
            return self.compiler.globals[name][0]
        raise NexirError(f"unknown variable: {name}", node)

    def member(self, node: ast.Attribute, *, mutable: bool = False) -> tuple[str, str, str]:
        value, typ = self.expr(node.value)
        owner = typ
        operator = "."
        if typ.startswith("owned["):
            owner = typ[6:-1]
            operator = "->"
        elif typ.startswith(("borrowed[", "mut[")):
            owner = typ[typ.index("[") + 1:-1]
            if mutable and typ.startswith("borrowed["):
                raise NexirError("borrowed value is read-only", node)
        if owner not in self.compiler.structs:
            raise NexirError("member access requires a struct", node)
        return f"({value}){operator}nx_{node.attr}", owner, typ

    def argument(self, node: ast.expr, expected: str) -> str:
        value, actual = self.expr(node)
        if expected.startswith(("borrowed[", "mut[")):
            owner = expected[expected.index("[") + 1:-1]
            if not isinstance(node, (ast.Name, ast.Attribute)):
                raise NexirError("borrow requires a named value", node)
            if actual == f"owned[{owner}]":
                return f"(*{value})"
            if actual == expected:
                return value
            if actual == owner or (expected.startswith("borrowed[") and actual == f"mut[{owner}]"):
                return value
            raise NexirError(f"expected {expected}, got {actual}", node)
        if not _compatible(actual, expected):
            raise NexirError(f"expected {expected}, got {actual}", node)
        if expected.startswith("owned[") and isinstance(node, ast.Name):
            raise NexirError("owned value must be passed with move(...) or freshly constructed", node)
        return _coerce(value, actual, expected)

    def check_borrow_aliases(self, args: list[ast.expr], params: tuple[tuple[str, str], ...]) -> None:
        seen: dict[str, str] = {}
        for arg, (_, expected) in zip(args, params):
            if not isinstance(arg, ast.Name) or not expected.startswith(("borrowed[", "mut[")):
                continue
            mode = "mut" if expected.startswith("mut[") else "borrowed"
            previous = seen.get(arg.id)
            if previous and (mode == "mut" or previous == "mut"):
                raise NexirError(f"conflicting borrows of {arg.id} in one call", arg)
            seen[arg.id] = mode

    def require_caps(self, required: set[str] | frozenset[str], node: ast.AST) -> None:
        missing = set(required) - self.fn.caps
        if missing:
            raise NexirError(f"function {self.fn.node.name} needs @cap({', '.join(repr(c) for c in sorted(missing))})", node)

    def index(self, node: ast.Subscript, *, mutable: bool = False) -> tuple[str, str]:
        container, typ = self.expr(node.value)
        dict_types = _dict_parts(typ)
        if dict_types:
            key = self.argument(node.slice, dict_types[0])
            return (f"({container})[{key}]" if mutable else f"nx_dict_at({container}, {key})"), dict_types[1]
        if typ in self.compiler.structs and not mutable:
            method = self.compiler.structs[typ].methods.get("__getitem__")
            if method and len(method.params) == 2:
                self.require_caps(method.caps, node)
                index = self.argument(node.slice, method.params[1][1])
                return f"({container}).nx___getitem__({index})", method.result
        tuple_parts = _tuple_parts(typ)
        if tuple_parts and not mutable:
            if not isinstance(node.slice, ast.Constant) or type(node.slice.value) is not int or not 0 <= node.slice.value < len(tuple_parts):
                raise NexirError("Tuple index must be a nonnegative constant within bounds", node.slice)
            return f"std::get<{node.slice.value}>({container})", tuple_parts[node.slice.value]
        list_element = _list_element(typ)
        if list_element:
            index, index_type = self.expr(node.slice)
            if index_type not in INTEGERS:
                raise NexirError("List index must be an integer", node.slice)
            return f"nx_list_at({container}, {index})", list_element
        buffer = _buffer_parts(typ)
        if buffer:
            mode, element = buffer
            if mutable and mode == "borrowed":
                raise NexirError("borrowed Buffer is read-only", node)
            index, index_type = self.expr(node.slice)
            if index_type not in INTEGERS:
                raise NexirError("Buffer index must be an integer", node.slice)
            operator = "->" if mode == "owned" else "."
            return f"({container}){operator}at({index})", element
        simd = _simd_parts(typ)
        if simd:
            index, index_type = self.expr(node.slice)
            if index_type not in INTEGERS:
                raise NexirError("SIMD index must be an integer", node.slice)
            return f"({container}).at({index})", simd[0]
        element = _span_element(typ)
        if element is None or (mutable and not typ.startswith("MutSpan[")):
            raise NexirError("indexing requires Span; assignment requires MutSpan", node)
        index, index_type = self.expr(node.slice)
        if index_type not in INTEGERS:
            raise NexirError("array index must be an integer", node.slice)
        access = "nx_at_mut" if typ.startswith("MutSpan[") else "nx_at"
        return f"(*{access}_{element}({container}, (int64_t)({index})))", element

    def slice_expr(self, node: ast.Subscript) -> tuple[str, str]:
        self.require_caps({"memory.alloc"}, node)
        container, typ = self.expr(node.value)
        element = _list_element(typ) or _span_element(typ)
        buffer = _buffer_parts(typ)
        if buffer:
            element = buffer[1]
        if not element and typ != "str":
            raise NexirError("slicing requires List, Span, Buffer or str", node)
        fields = []
        for part in (node.slice.lower, node.slice.upper, node.slice.step):
            if part is None:
                fields.append("0")
            else:
                value, actual = self.expr(part)
                if actual not in INTEGERS:
                    raise NexirError("slice indices must be integers", part)
                fields.append(value)
        if node.slice.step is None:
            fields[2] = "1"
        args = f"{fields[0]}, {fields[1]}, {fields[2]}, {'true' if node.slice.lower is not None else 'false'}, {'true' if node.slice.upper is not None else 'false'}"
        if typ == "str":
            return f"nx_string_slice({container}, {args})", "str"
        pointer = f"({container}).data()" if _list_element(typ) else f"({container}).ptr" if _span_element(typ) else f"({container}){'->' if buffer and buffer[0] == 'owned' else '.'}data()"
        length = f"({container}).size()" if _list_element(typ) else f"({container}).len" if _span_element(typ) else f"({container}){'->' if buffer and buffer[0] == 'owned' else '.'}size()"
        return f"nx_slice({pointer}, {length}, {args})", f"List[{element}]"

    def comprehension(self, node: ast.ListComp | ast.DictComp) -> tuple[str, str]:
        self.require_caps({"memory.alloc"}, node)
        if len(node.generators) != 1:
            raise NexirError("comprehension supports one for clause", node)
        clause = node.generators[0]
        if clause.is_async or not isinstance(clause.target, ast.Name) or not isinstance(clause.iter, ast.Name):
            raise NexirError("comprehension requires a named iterable and item", clause)
        name = clause.target.id
        _name(name, clause.target)
        container, typ = self.expr(clause.iter)
        element = _list_element(typ) or _span_element(typ)
        buffer = _buffer_parts(typ)
        if buffer:
            element = buffer[1]
        if not element:
            raise NexirError("comprehension requires List, Span or Buffer", clause.iter)
        self.scopes.append({name: element})
        try:
            checks = [self.expr(item) for item in clause.ifs]
            if any(actual != "bool" for _, actual in checks):
                raise NexirError("comprehension filter must be bool", node)
            if isinstance(node, ast.ListComp):
                value, value_type = self.expr(node.elt)
                result_type = f"List[{value_type}]"
                if result_type not in self.compiler.types:
                    raise NexirError("List comprehension result requires a primitive type", node.elt)
                action = f"nx_out.push_back({value});"
            else:
                key, key_type = self.expr(node.key)
                value, value_type = self.expr(node.value)
                result_type = f"Dict[{key_type},{value_type}]"
                if result_type not in self.compiler.types:
                    raise NexirError("Dict comprehension requires supported key and value types", node)
                action = f"nx_out[{key}] = {value};"
        finally:
            self.scopes.pop()
        self.serial += 1
        suffix = self.serial
        size = f"({container}).size()" if _list_element(typ) else f"({container}).len" if _span_element(typ) else f"({container}){'->' if buffer and buffer[0] == 'owned' else '.'}size()"
        item = f"({container})[nx_index_{suffix}]" if _list_element(typ) else f"({container}).ptr[nx_index_{suffix}]" if _span_element(typ) else f"({container}){'->' if buffer and buffer[0] == 'owned' else '.'}at(nx_index_{suffix})"
        guard = "" if not checks else "if (" + " && ".join(code for code, _ in checks) + ") "
        code = f"([&]() {{ {self.compiler.types[result_type]} nx_out; for (std::size_t nx_index_{suffix} = 0; nx_index_{suffix} < {size}; ++nx_index_{suffix}) {{ {self.compiler.types[element]} nx_{name} = {item}; {guard}{action} }} return nx_out; }}())"
        return code, result_type

    def array_literal(self, node: ast.List) -> tuple[str, str]:
        if not node.elts:
            raise NexirError("empty list requires zeros_f32(0) or zeros_f64(0)", node)
        elements = [self.expr(item) for item in node.elts]
        if any(t not in NUMERICS for _, t in elements):
            raise NexirError("lists currently contain numeric values only", node)
        types = {t for _, t in elements}
        typ = "f64" if "f64" in types else "f32" if "f32" in types else "i64" if "i64" in types else "i32"
        values = ", ".join(code for code, _ in elements)
        return f"nx_list_{typ}({{{values}}})", f"MutSpan[{typ}]"

    def expr(self, node: ast.expr) -> tuple[str, str]:
        if isinstance(node, (ast.ListComp, ast.DictComp)):
            return self.comprehension(node)
        if isinstance(node, ast.Tuple):
            if not 2 <= len(node.elts) <= 8:
                raise NexirError("Tuple literal requires 2 to 8 values", node)
            values = [self.expr(item) for item in node.elts]
            parts = [typ for _, typ in values]
            if any(part not in NUMERICS | {"bool", "str"} for part in parts):
                raise NexirError("Tuple supports primitive value types", node)
            typ = "Tuple[" + ",".join(parts) + "]"
            self.compiler.types[typ] = "std::tuple<" + ", ".join(self.compiler.types[part] for part in parts) + ">"
            return f"{self.compiler.types[typ]}{{{', '.join(value for value, _ in values)}}}", typ
        if isinstance(node, ast.Dict):
            self.require_caps({"memory.alloc"}, node)
            if not node.keys or any(key is None for key in node.keys):
                raise NexirError("Dict literal requires typed key-value entries", node)
            keys = [self.expr(key) for key in node.keys if key is not None]
            values = [self.expr(value) for value in node.values]
            key_type, value_type = keys[0][1], values[0][1]
            typ = f"Dict[{key_type},{value_type}]"
            if typ not in self.compiler.types or any(t != key_type for _, t in keys) or any(t != value_type for _, t in values):
                raise NexirError("Dict literal requires uniform supported key and value types", node)
            pairs = ", ".join("{" + key + ", " + value + "}" for (key, _), (value, _) in zip(keys, values))
            return f"{self.compiler.types[typ]}{{{pairs}}}", typ
        if isinstance(node, ast.Set):
            self.require_caps({"memory.alloc"}, node)
            values = [self.expr(item) for item in node.elts]
            element = values[0][1]
            typ = f"Set[{element}]"
            if typ not in self.compiler.types or any(t != element for _, t in values):
                raise NexirError("Set literal requires uniform supported element types", node)
            return f"{self.compiler.types[typ]}{{{', '.join(value for value, _ in values)}}}", typ
        if isinstance(node, ast.List):
            return self.array_literal(node)
        if isinstance(node, ast.JoinedStr):
            self.require_caps({"memory.alloc"}, node)
            pieces = []
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    pieces.append(f"std::string({_c_string(part.value)})")
                elif isinstance(part, ast.FormattedValue) and part.format_spec is None and part.conversion in {-1, 115, 114}:
                    value, typ = self.expr(part.value)
                    if typ not in NUMERICS | {"bool", "str"}:
                        raise NexirError("f-string value must be numeric, bool or str", part)
                    pieces.append(f"nx_to_string({value})")
                else:
                    raise NexirError("unsupported f-string format", part)
            return f"nx_string_keep({' + '.join(pieces) if pieces else 'std::string()'})", "str"
        if isinstance(node, ast.IfExp):
            cond, ct = self.expr(node.test)
            yes, yt = self.expr(node.body)
            no, nt = self.expr(node.orelse)
            if ct != "bool" or yt != nt:
                raise NexirError("conditional expression requires bool and equal branch types", node)
            return f"({cond} ? {yes} : {no})", yt
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "std::nullopt", "None"
            if type(node.value) is bool:
                return ("true" if node.value else "false", "bool")
            if type(node.value) is int:
                n = node.value
                if not -(2**63) <= n < 2**63:
                    raise NexirError("integer literal exceeds i64", node)
                return (str(n) + ("LL" if not -(2**31) <= n < 2**31 else ""), "i32" if -(2**31) <= n < 2**31 else "i64")
            if type(node.value) is float:
                return (repr(node.value), "f64")
            if type(node.value) is str:
                return (_c_string(node.value), "str")
            raise NexirError("unsupported literal", node)
        if isinstance(node, ast.Name):
            return ("nx_" + node.id, self.lookup(node.id, node))
        if isinstance(node, ast.Attribute):
            value, owner, receiver_type = self.member(node)
            field = dict(self.compiler.structs[owner].fields).get(node.attr)
            if field is None:
                raise NexirError(f"unknown struct field: {owner}.{node.attr}", node)
            if receiver_type.startswith("borrowed[") and field.startswith(("owned[", "MutSpan[")):
                raise NexirError("borrowed value cannot expose a mutable owned field", node)
            return value, field
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                return self.slice_expr(node)
            return self.index(node)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) and type(node.operand.value) is int:
                n = -node.operand.value
                if not -(2**63) <= n < 2**63:
                    raise NexirError("integer literal exceeds i64", node)
                if n == -(2**63):
                    return ("(-9223372036854775807LL - 1LL)", "i64")
                if n == -(2**31):
                    return ("(-2147483647 - 1)", "i32")
                return (f"({n}{'LL' if n < -(2**31) else ''})", "i32" if n >= -(2**31) else "i64")
            value, typ = self.expr(node.operand)
            if isinstance(node.op, ast.Not) and typ == "bool":
                return (f"(!{value})", "bool")
            if isinstance(node.op, (ast.UAdd, ast.USub)) and self.numeric(typ):
                sign = "+" if isinstance(node.op, ast.UAdd) else "-"
                return (f"({sign}{value})", typ)
            raise NexirError("invalid unary operation", node)
        if isinstance(node, ast.BinOp):
            left, lt = self.expr(node.left)
            right, rt = self.expr(node.right)
            if isinstance(node.op, ast.Add) and lt == rt == "str":
                self.require_caps({"memory.alloc"}, node)
                return f"nx_string_keep(std::string({left}) + {right})", "str"
            overloaded = {ast.Add: "__add__", ast.Sub: "__sub__", ast.Mult: "__mul__"}.get(type(node.op))
            if lt in self.compiler.structs and overloaded in self.compiler.structs[lt].methods:
                method = self.compiler.structs[lt].methods[overloaded]
                if len(method.params) != 2 or not _compatible(rt, method.params[1][1]):
                    raise NexirError("operator argument type does not match struct method", node)
                self.require_caps(method.caps, node)
                return f"({left}).nx_{overloaded}({right})", method.result
            if _simd_parts(lt) and lt == rt:
                op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}.get(type(node.op))
                if op is None:
                    raise NexirError("SIMD supports +, -, * between equal types", node)
                return f"({left} {op} {right})", lt
            if "dyn" in {lt, rt}:
                if not isinstance(node.op, ast.Add) or not _compatible(lt, "dyn") or not _compatible(rt, "dyn"):
                    raise NexirError("dynamic values currently support + only", node)
                return (f"nx_add_dyn({_coerce(left, lt, 'dyn')}, {_coerce(right, rt, 'dyn')})", "dyn")
            if not self.numeric(lt) or not self.numeric(rt):
                raise NexirError("arithmetic requires numeric operands", node)
            if (lt in dict(self.fn.generic) or rt in dict(self.fn.generic)) and lt != rt:
                raise NexirError("generic arithmetic requires operands of the same type", node)
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%"}.get(type(node.op))
            if op is None or (op == "%" and (lt not in INTEGERS or rt not in INTEGERS)):
                raise NexirError("unsupported arithmetic operation", node)
            if op in {"/", "%"} and isinstance(node.right, ast.Constant) and node.right.value == 0:
                raise NexirError("division by zero", node.right)
            typ = lt if lt in dict(self.fn.generic) else "f64" if "f64" in {lt, rt} else "f32" if "f32" in {lt, rt} else "i64" if "i64" in {lt, rt} else "i32"
            return (f"({left} {op} {right})", typ)
        if isinstance(node, ast.BoolOp):
            op = " && " if isinstance(node.op, ast.And) else " || " if isinstance(node.op, ast.Or) else None
            vals = [self.expr(v) for v in node.values]
            if op is None or any(t != "bool" for _, t in vals):
                raise NexirError("boolean operations require bool operands", node)
            return ("(" + op.join(c for c, _ in vals) + ")", "bool")
        if isinstance(node, ast.Compare):
            if len(node.ops) > 1 and any(not isinstance(item, (ast.Name, ast.Constant)) for item in node.comparators[:-1]):
                raise NexirError("chained comparison middle values must be names or constants", node)
            values = [self.expr(item) for item in [node.left, *node.comparators]]
            checks = []
            for op_node, (left, lt), (right, rt) in zip(node.ops, values, values[1:]):
                if isinstance(op_node, (ast.Is, ast.IsNot)) and ((lt.startswith("Optional[") and rt == "None") or (rt.startswith("Optional[") and lt == "None")):
                    optional = left if lt.startswith("Optional[") else right
                    checks.append(f"(!({optional}).has_value())" if isinstance(op_node, ast.Is) else f"({optional}).has_value()")
                    continue
                if isinstance(op_node, (ast.In, ast.NotIn)):
                    element = _list_element(rt) or _set_element(rt) or _span_element(rt)
                    map_types = _dict_parts(rt)
                    if map_types:
                        element = map_types[0]
                    if lt == rt == "str":
                        check = f"(std::strstr({right}, {left}) != nullptr)"
                    elif element == lt and (_set_element(rt) or map_types):
                        check = f"(({right}).find({left}) != ({right}).end())"
                    elif element == lt and _list_element(rt):
                        check = f"nx_list_contains({right}, {left})"
                    elif element == lt and _span_element(rt):
                        check = f"nx_span_contains(({right}).ptr, ({right}).len, {left})"
                    else:
                        raise NexirError("in requires a compatible str, List, Set, Dict or Span", node)
                    checks.append(f"(!{check})" if isinstance(op_node, ast.NotIn) else check)
                    continue
                op = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}.get(type(op_node))
                if op is None or not (lt == rt or self.numeric(lt) and self.numeric(rt)):
                    raise NexirError("incompatible comparison", node)
                if lt in self.compiler.structs and op in {"==", "!="} and "__eq__" in self.compiler.structs[lt].methods:
                    method = self.compiler.structs[lt].methods["__eq__"]
                    if method.result != "bool" or len(method.params) != 2 or method.params[1][1] != rt:
                        raise NexirError("__eq__ must accept the same struct and return bool", node)
                    check = f"({left}).nx___eq__({right})"
                    checks.append(f"(!{check})" if op == "!=" else check)
                elif lt == "str":
                    checks.append(f"(std::strcmp({left}, {right}) {op} 0)")
                else:
                    checks.append(f"({left} {op} {right})")
            return ("(" + " && ".join(checks) + ")", "bool")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "value_or" and len(node.args) == 2 and not node.keywords:
                optional, typ = self.expr(node.args[0])
                if not typ.startswith("Optional["):
                    raise NexirError("value_or expects Optional[T]", node.args[0])
                element = typ[9:-1]
                fallback = self.argument(node.args[1], element)
                return f"({optional}).value_or({fallback})", element
            if isinstance(node.func, ast.Subscript) and not node.keywords:
                typ = _type(node.func, types=self.compiler.types)
                if _set_element(typ) or _dict_parts(typ):
                    self.require_caps({"memory.alloc"}, node)
                    if not node.args:
                        return f"{self.compiler.types[typ]}{{}}", typ
                    if len(node.args) == 1:
                        value, actual = self.expr(node.args[0])
                        if actual == typ:
                            return value, typ
                    raise NexirError(f"{typ} expects no arguments or a matching literal", node)
                if _tuple_parts(typ):
                    parts = _tuple_parts(typ)
                    if len(node.args) != len(parts):
                        raise NexirError(f"{typ} expects {len(parts)} values", node)
                    values = [self.argument(item, part) for item, part in zip(node.args, parts)]
                    return f"{self.compiler.types[typ]}{{{', '.join(values)}}}", typ
                list_element = _list_element(typ)
                if list_element:
                    self.require_caps({"memory.alloc"}, node)
                    if len(node.args) == 0:
                        return f"{self.compiler.types[typ]}{{}}", typ
                    if len(node.args) != 1 or not isinstance(node.args[0], ast.List):
                        raise NexirError("List[T] expects no arguments or one list literal", node)
                    values = [self.argument(item, list_element) for item in node.args[0].elts]
                    return f"{self.compiler.types[typ]}{{{', '.join(values)}}}", typ
                if typ.startswith("Buffer["):
                    self.require_caps({"memory.alloc"}, node)
                    if len(node.args) != 1:
                        raise NexirError("Buffer[T] expects one length", node)
                    length, actual = self.expr(node.args[0])
                    if actual not in INTEGERS:
                        raise NexirError("Buffer length must be an integer", node.args[0])
                    return f"std::make_unique<{self.compiler.types[typ]}>({length})", f"owned[{typ}]"
                simd = _simd_parts(typ)
                if not simd:
                    raise NexirError("only SIMD type constructors are callable", node)
                element, width = simd
                if not width.isdigit():
                    raise NexirError("SIMD constructor requires a concrete width", node)
                if len(node.args) != int(width):
                    raise NexirError(f"{typ} expects {width} lane values", node)
                args = [self.argument(arg, element) for arg in node.args]
                return f"{self.compiler.types[typ]}{{{{{', '.join(args)}}}}}", typ
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in self.compiler.imports:
                return self.expr(ast.copy_location(ast.Call(func=ast.Name(id=node.func.attr, ctx=ast.Load()), args=node.args, keywords=node.keywords), node))
            if isinstance(node.func, ast.Attribute) and not node.keywords:
                receiver, receiver_type = self.expr(node.func.value)
                element = _list_element(receiver_type)
                if element:
                    method = node.func.attr
                    if method in {"append", "push"} and len(node.args) == 1:
                        self.require_caps({"memory.alloc"}, node)
                        value = self.argument(node.args[0], element)
                        return f"({receiver}).push_back({value})", "void"
                    if method == "pop" and not node.args:
                        return f"nx_list_pop({receiver})", element
                    if method == "sort" and not node.args:
                        return f"nx_list_sort({receiver})", "void"
                    raise NexirError("List supports append(value), push(value), pop() and sort()", node)
                set_element = _set_element(receiver_type)
                if set_element:
                    method = node.func.attr
                    if method == "add" and len(node.args) == 1:
                        self.require_caps({"memory.alloc"}, node)
                        value = self.argument(node.args[0], set_element)
                        return f"({receiver}).insert({value})", "void"
                    if method in {"remove", "discard"} and len(node.args) == 1:
                        value = self.argument(node.args[0], set_element)
                        return f"nx_set_{method}({receiver}, {value})", "void"
                    raise NexirError("Set supports add, remove and discard", node)
                if receiver_type == "str":
                    method = node.func.attr
                    if method in {"lower", "upper", "strip"} and not node.args:
                        self.require_caps({"memory.alloc"}, node)
                        return f"nx_string_{method}({receiver})", "str"
                    if method == "replace" and len(node.args) == 2:
                        self.require_caps({"memory.alloc"}, node)
                        old = self.argument(node.args[0], "str")
                        new = self.argument(node.args[1], "str")
                        return f"nx_string_replace({receiver}, {old}, {new})", "str"
                    if method == "split" and len(node.args) <= 1:
                        self.require_caps({"memory.alloc"}, node)
                        delimiter = self.argument(node.args[0], "str") if node.args else '" "'
                        return f"nx_string_split({receiver}, {delimiter})", "List[str]"
                    raise NexirError("unsupported str method", node)
                owner = receiver_type
                operator = "."
                generic_bound = dict(self.fn.generic).get(receiver_type)
                if generic_bound in self.compiler.traits:
                    signature = self.compiler.traits[generic_bound].get(node.func.attr)
                    if signature is None or len(node.args) != len(signature[0]):
                        raise NexirError(f"trait {generic_bound} has no matching method {node.func.attr}", node)
                    args = [self.argument(arg, receiver_type if expected == "Self" else expected) for arg, expected in zip(node.args, signature[0])]
                    result = receiver_type if signature[1] == "Self" else signature[1]
                    return f"({receiver}).nx_{node.func.attr}({', '.join(args)})", result
                if receiver_type.startswith("owned["):
                    owner = receiver_type[6:-1]
                    operator = "->"
                elif receiver_type.startswith(("borrowed[", "mut[")):
                    if receiver_type.startswith("borrowed["):
                        raise NexirError("methods on borrowed values require read-only method support", node.func)
                    owner = receiver_type[receiver_type.index("[") + 1:-1]
                if owner not in self.compiler.structs:
                    raise NexirError("method call requires a struct", node.func)
                method = self.compiler.structs[owner].methods.get(node.func.attr)
                if method is None or node.func.attr in {"__init__", "__del__"}:
                    raise NexirError(f"unknown struct method: {owner}.{node.func.attr}", node)
                self.require_caps(method.caps, node)
                if len(node.args) != len(method.params) - 1:
                    raise NexirError(f"{owner}.{node.func.attr} expects {len(method.params) - 1} arguments", node)
                args = [self.argument(arg, expected) for arg, (_, expected) in zip(node.args, method.params[1:])]
                return f"({receiver}){operator}nx_{node.func.attr}({', '.join(args)})", method.result
            if not isinstance(node.func, ast.Name):
                raise NexirError("only named functions are callable", node)
            name = node.func.id
            if name in self.compiler.structs:
                if node.keywords:
                    raise NexirError("struct constructors require positional arguments", node)
                struct = self.compiler.structs[name]
                initializer = struct.methods.get("__init__")
                expected = initializer.params[1:] if initializer else struct.fields
                if len(node.args) != len(expected):
                    raise NexirError(f"{name} expects {len(expected)} constructor arguments", node)
                args = [self.argument(arg, typ) for arg, (_, typ) in zip(node.args, expected)]
                braces = "(" if initializer else "{"
                close = ")" if initializer else "}"
                return f"nx_{name}{braces}{', '.join(args)}{close}", name
            if name == "owned":
                self.require_caps({"memory.alloc"}, node)
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Call) or not isinstance(node.args[0].func, ast.Name) or node.args[0].func.id not in self.compiler.structs:
                    raise NexirError("owned expects a struct constructor", node)
                value, typ = self.expr(node.args[0])
                return f"std::unique_ptr<nx_{typ}>(new {value})", f"owned[{typ}]"
            if name == "move":
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                    raise NexirError("move expects one owned variable", node)
                value, typ = self.expr(node.args[0])
                if not typ.startswith("owned["):
                    raise NexirError("move requires an owned value", node)
                self.moved.add(node.args[0].id)
                return f"std::move({value})", typ
            if name in {"buffer_span", "buffer_mutspan"}:
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                    raise NexirError(f"{name} requires a named Buffer owner", node)
                value, typ = self.expr(node.args[0])
                parts = _buffer_parts(typ)
                if not parts or (name == "buffer_mutspan" and parts[0] == "borrowed"):
                    raise NexirError(f"{name} requires a compatible owned or borrowed Buffer", node)
                mode, element = parts
                operator = "->" if mode == "owned" else "."
                view = "MutSpan" if name == "buffer_mutspan" else "Span"
                return f"{self.compiler.types[f'{view}[{element}]']}{{({value}){operator}data(), ({value}){operator}size()}}", f"{view}[{element}]"
            if name in {"is_ok", "unwrap", "error"}:
                if len(node.args) != 1:
                    raise NexirError(f"{name} expects one Result", node)
                value, typ = self.expr(node.args[0])
                if not typ.startswith("Result["):
                    raise NexirError(f"{name} requires Result[T]", node.args[0])
                if name == "is_ok":
                    return f"({value}).ok()", "bool"
                if name == "error":
                    if not isinstance(node.args[0], ast.Name):
                        raise NexirError("error requires a named Result to preserve string lifetime", node.args[0])
                    return f"({value}).error()", "str"
                return f"({value}).value()", typ[7:-1]
            if name == "type_name":
                if len(node.args) != 1:
                    raise NexirError("type_name expects one dyn value", node)
                value, typ = self.expr(node.args[0])
                if typ != "dyn":
                    raise NexirError("type_name requires dyn", node.args[0])
                return f"nx_type_name({value})", "str"
            if name in {"as_i64", "as_f64"}:
                if len(node.args) != 1:
                    raise NexirError(f"{name} expects one dyn value", node)
                value, typ = self.expr(node.args[0])
                if typ != "dyn":
                    raise NexirError(f"{name} requires dyn", node.args[0])
                return f"nx_{name}({value})", name[3:]
            if name == "ring_f64":
                self.require_caps({"memory.alloc"}, node)
                if len(node.args) != 1:
                    raise NexirError("ring_f64 expects capacity", node)
                size, typ = self.expr(node.args[0])
                if typ not in INTEGERS:
                    raise NexirError("ring capacity must be integer", node.args[0])
                return f"nx_ring_new_f64({size})", "Ring[f64]"
            if name in {"ring_write", "ring_read", "ring_valid"}:
                count = 3 if name == "ring_write" else 2
                if len(node.args) != count:
                    raise NexirError(f"{name} expects {count} arguments", node)
                ring, ring_typ = self.expr(node.args[0])
                index, index_typ = self.expr(node.args[1])
                if ring_typ != "Ring[f64]" or index_typ not in INTEGERS:
                    raise NexirError(f"{name} expects Ring[f64] and integer index", node)
                if name == "ring_write":
                    value, value_typ = self.expr(node.args[2])
                    if value_typ not in NUMERICS:
                        raise NexirError("ring_write value must be numeric", node.args[2])
                    return f"({ring})->write({index}, {value})", "void"
                if name == "ring_read":
                    return f"({ring})->read({index})", "f64"
                return f"({ring})->valid({index})", "bool"
            if name in {"f32", "f64", "i32", "i64"}:
                if len(node.args) != 1:
                    raise NexirError(f"{name} expects one numeric argument", node)
                value, actual = self.expr(node.args[0])
                if not self.numeric(actual):
                    raise NexirError("numeric cast requires a number", node.args[0])
                return f"(({TYPES[name]})({value}))", name
            if name == "len":
                if len(node.args) != 1:
                    raise NexirError("len expects one Span", node)
                value, typ = self.expr(node.args[0])
                if _list_element(typ):
                    return f"static_cast<int64_t>(({value}).size())", "i64"
                if _set_element(typ) or _dict_parts(typ):
                    return f"static_cast<int64_t>(({value}).size())", "i64"
                if _tuple_parts(typ):
                    return str(len(_tuple_parts(typ))), "i64"
                if typ == "str":
                    return f"static_cast<int64_t>(std::strlen({value}))", "i64"
                buffer = _buffer_parts(typ)
                if buffer:
                    operator = "->" if buffer[0] == "owned" else "."
                    return f"static_cast<int64_t>(({value}){operator}size())", "i64"
                simd = _simd_parts(typ)
                if simd:
                    width = simd[1]
                    return (self.compiler.types[f"$const:{width}"] if not width.isdigit() else width), "i64"
                if not _span_element(typ):
                    raise NexirError("len requires Span, MutSpan or SIMD", node.args[0])
                return f"((int64_t)({value}).len)", "i64"
            if name in {"zeros_f32", "zeros_f64"}:
                self.require_caps({"memory.alloc"}, node)
                if len(node.args) != 1:
                    raise NexirError(f"{name} expects one length", node)
                value, typ = self.expr(node.args[0])
                if typ not in INTEGERS:
                    raise NexirError("array length must be an integer", node.args[0])
                elem = name.removeprefix("zeros_")
                return f"nx_{name}({value})", f"MutSpan[{elem}]"
            if name == "read_text":
                self.require_caps({"filesystem.read", "memory.alloc"}, node)
                if len(node.args) != 1:
                    raise NexirError("read_text expects one path", node)
                value, typ = self.expr(node.args[0])
                if typ != "str":
                    raise NexirError("read_text path must be str", node.args[0])
                return f"nx_read_text({value})", "str"
            if name == "tcp_connect":
                self.require_caps({"network.connect"}, node)
                if len(node.args) != 2:
                    raise NexirError("tcp_connect expects host and port", node)
                host, ht = self.expr(node.args[0])
                port, pt = self.expr(node.args[1])
                if ht != "str" or pt != "i32":
                    raise NexirError("tcp_connect expects (str, i32)", node)
                return f"nx_tcp_connect({host}, {port})", "bool"
            if name not in self.compiler.functions:
                raise NexirError(f"unknown function: {name}", node)
            fn = self.compiler.functions[name]
            self.require_caps(fn.caps, node)
            if len(node.args) > len(fn.params):
                raise NexirError(f"{name} expects at most {len(fn.params)} arguments", node)
            bound: list[ast.expr | None] = [*node.args] + [None] * (len(fn.params) - len(node.args))
            names = [param for param, _ in fn.params]
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg not in names:
                    raise NexirError(f"unknown keyword argument for {name}", keyword)
                position = names.index(keyword.arg)
                if bound[position] is not None:
                    raise NexirError(f"duplicate argument: {keyword.arg}", keyword)
                bound[position] = keyword.value
            for i, (item, default) in enumerate(zip(bound, fn.defaults)):
                if item is None:
                    bound[i] = default
            if any(item is None for item in bound):
                raise NexirError(f"missing required argument for {name}", node)
            call_args = [item for item in bound if item is not None]
            self.check_borrow_aliases(call_args, fn.params)
            if fn.generic:
                inferred: dict[str, str] = {}
                args = []
                bounds = dict(fn.generic)
                for arg, (_, expected) in zip(call_args, fn.params):
                    if expected in bounds:
                        value, actual = self.expr(arg)
                        bound = bounds[expected]
                        if bound in self.compiler.traits:
                            struct = self.compiler.structs.get(actual)
                            required = self.compiler.traits[bound]
                            if struct is None or any(
                                method not in struct.methods
                                or struct.methods[method].result != (actual if result == "Self" else result)
                                or tuple(typ for _, typ in struct.methods[method].params[1:]) != tuple(actual if typ == "Self" else typ for typ in params)
                                for method, (params, result) in required.items()
                            ):
                                raise NexirError(f"{actual} does not implement trait {bound}", arg)
                        elif not self.numeric(actual) or (bound == "Integer" and actual not in INTEGERS and dict(self.fn.generic).get(actual) != "Integer"):
                            raise NexirError(f"{name} requires {bound} for {expected}, got {actual}", arg)
                        if expected in inferred and inferred[expected] != actual:
                            raise NexirError(f"generic {expected} must have one concrete type", arg)
                        inferred[expected] = actual
                        args.append(value)
                    elif _simd_parts(expected) and any(param in expected for param in bounds):
                        value, actual = self.expr(arg)
                        expected_parts = _simd_parts(expected)
                        actual_parts = _simd_parts(actual)
                        if actual_parts is None:
                            raise NexirError(f"expected {expected}, got {actual}", arg)
                        expected_element, expected_width = expected_parts
                        actual_element, actual_width = actual_parts
                        if expected_element in bounds:
                            if bounds[expected_element] not in {"Number", "Integer"} or not self.numeric(actual_element):
                                raise NexirError(f"SIMD generic {expected_element} requires a numeric type", arg)
                            if expected_element in inferred and inferred[expected_element] != actual_element:
                                raise NexirError(f"generic {expected_element} must have one concrete type", arg)
                            inferred[expected_element] = actual_element
                        elif expected_element != actual_element:
                            raise NexirError(f"expected {expected}, got {actual}", arg)
                        if expected_width in bounds and bounds[expected_width] == "Int":
                            if expected_width in inferred and inferred[expected_width] != actual_width:
                                raise NexirError(f"generic {expected_width} must have one constant value", arg)
                            inferred[expected_width] = actual_width
                        elif expected_width != actual_width:
                            raise NexirError(f"expected {expected}, got {actual}", arg)
                        args.append(value)
                    else:
                        args.append(self.argument(arg, expected))
                if any(param not in inferred for param, _ in fn.generic):
                    raise NexirError(f"cannot infer generic type for {name}", node)
                result = inferred.get(fn.result, fn.result)
                parts = _simd_parts(result)
                if parts:
                    result = f"SIMD[{inferred.get(parts[0], parts[0])},{inferred.get(parts[1], parts[1])}]"
                return (f"nx_{name}({', '.join(args)})", result)
            args = []
            for arg, (_, expected) in zip(call_args, fn.params):
                args.append(self.argument(arg, expected))
            return (f"nx_{name}({', '.join(args)})", fn.result)
        raise NexirError(f"unsupported expression: {type(node).__name__}", node)

    def print_stmt(self, node: ast.Call) -> None:
        if node.keywords:
            raise NexirError("print does not accept keyword arguments", node)
        for i, arg in enumerate(node.args):
            if i:
                self.emit('fputc(\' \', stdout);')
            value, typ = self.expr(arg)
            fmt = {"i32": "%d", "i64": "%lld", "f32": "%.9g", "f64": "%.17g"}.get(typ)
            if fmt:
                cast = "(long long)" if typ == "i64" else ""
                self.emit(f'printf("{fmt}", {cast}{value});')
            elif typ == "bool":
                self.emit(f'fputs({value} ? "true" : "false", stdout);')
            elif typ == "str":
                self.emit(f"fputs({value}, stdout);")
            elif typ == "dyn":
                self.emit(f"nx_print_value({value});")
            else:
                raise NexirError("print cannot display void", arg)
        self.emit("fputc('\\n', stdout);")

    def block(self, nodes: list[ast.stmt]) -> bool:
        always_returns = False
        for node in nodes:
            if always_returns:
                raise NexirError("unreachable statement after return", node)
            always_returns = self.stmt(node)
        return always_returns

    def nested(self, nodes: list[ast.stmt]) -> bool:
        self.scopes.append({})
        self.depth += 1
        result = self.block(nodes)
        self.depth -= 1
        self.moved.difference_update(self.scopes[-1])
        self.scopes.pop()
        return result

    def stmt(self, node: ast.stmt) -> bool:
        if isinstance(node, ast.Global):
            if any(name not in self.compiler.globals for name in node.names):
                raise NexirError("global names must be declared at file level", node)
            return False
        if isinstance(node, (ast.Break, ast.Continue)):
            if self.loop_depth == 0:
                raise NexirError(f"{type(node).__name__.lower()} is only valid inside a loop", node)
            self.emit("break;" if isinstance(node, ast.Break) else "continue;")
            return False
        if isinstance(node, ast.Pass):
            self.emit(";")
            return False
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "print":
                self.print_stmt(call)
            else:
                value, typ = self.expr(call)
                self.emit(f"{value};")
            return False
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name) or node.value is None:
                raise NexirError("typed declaration must have a name and initializer", node)
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in {"buffer_span", "buffer_mutspan"}:
                raise NexirError("Buffer view must be passed directly; it cannot be stored", node.value)
            name = node.target.id
            _name(name, node)
            if any(name in scope for scope in self.scopes):
                raise NexirError(f"variable already defined: {name}", node)
            typ = _type(node.annotation, types=self.compiler.types)
            if typ == "void":
                raise NexirError("variable cannot have void type", node)
            if self.fn.strict and typ == "dyn":
                raise NexirError("fn local variables cannot use dyn", node)
            if typ.startswith(("borrowed[", "mut[")):
                raise NexirError("borrowed and mut are parameter types only", node)
            value = self.argument(node.value, typ)
            self.scopes[-1][name] = typ
            self.emit(f"{self.compiler.types[typ]} nx_{name} = {value};")
            return False
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in {"buffer_span", "buffer_mutspan"}:
                raise NexirError("Buffer view must be passed directly; it cannot be stored", node.value)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
                target, owner, _ = self.member(node.targets[0], mutable=True)
                expected = dict(self.compiler.structs[owner].fields).get(node.targets[0].attr)
                if expected is None:
                    raise NexirError(f"unknown struct field: {owner}.{node.targets[0].attr}", node.targets[0])
                value = self.argument(node.value, expected)
                self.emit(f"{target} = {value};")
                return False
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                target, expected = self.index(node.targets[0], mutable=True)
                value, actual = self.expr(node.value)
                if not _compatible(actual, expected):
                    raise NexirError(f"expected {expected}, got {actual}", node.value)
                self.emit(f"{target} = {value};")
                return False
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise NexirError("assignment requires one variable", node)
            name = node.targets[0].id
            _name(name, node)
            if name.isupper() and name in self.compiler.globals:
                raise NexirError(f"cannot assign to file constant: {name}", node)
            value, actual = self.expr(node.value)
            if actual == "void":
                raise NexirError("cannot assign a void result", node.value)
            if self.fn.strict and actual == "dyn":
                raise NexirError("fn local variables cannot use dyn", node.value)
            if name in self.moved:
                expected = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
                if expected is None or not _compatible(actual, expected):
                    raise NexirError(f"cannot reinitialize moved variable {name} with {actual}", node)
                self.moved.remove(name)
                self.emit(f"nx_{name} = {_coerce(value, actual, expected)};")
                return False
            try:
                expected = self.lookup(name, node)
            except NexirError:
                if actual == "None":
                    raise NexirError("None requires an Optional[T] annotation", node.value)
                if actual.startswith(("borrowed[", "mut[")):
                    raise NexirError("cannot store a borrow", node.value)
                if actual.startswith("owned[") and isinstance(node.value, ast.Name):
                    raise NexirError("owned value must be moved", node.value)
                self.scopes[-1][name] = actual
                self.emit(f"{self.compiler.types[actual]} nx_{name} = {value};")
                return False
            if not _compatible(actual, expected):
                raise NexirError(f"expected {expected}, got {actual}", node.value)
            if _span_element(expected) and name not in self.scopes[-1]:
                raise NexirError("cannot rebind an outer Span from a nested block", node)
            if expected.startswith("owned[") and isinstance(node.value, ast.Name):
                raise NexirError("owned value must be moved", node.value)
            self.moved.discard(name)
            self.emit(f"nx_{name} = {_coerce(value, actual, expected)};")
            return False
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Attribute):
                target, owner, _ = self.member(node.target, mutable=True)
                typ = dict(self.compiler.structs[owner].fields).get(node.target.attr)
                value, actual = self.expr(node.value)
                op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(node.op))
                if typ not in NUMERICS or op is None or not _compatible(actual, typ):
                    raise NexirError("invalid struct field augmented assignment", node)
                self.emit(f"{target} {op}= {value};")
                return False
            if isinstance(node.target, ast.Subscript):
                target, typ = self.index(node.target, mutable=True)
                value, actual = self.expr(node.value)
                op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(node.op))
                if op is None or not _compatible(actual, typ):
                    raise NexirError("invalid array augmented assignment", node)
                self.emit(f"{target} {op}= {value};")
                return False
            if not isinstance(node.target, ast.Name):
                raise NexirError("augmented assignment requires a variable", node)
            name = node.target.id
            typ = self.lookup(name, node.target)
            if name.isupper() and name in self.compiler.globals:
                raise NexirError(f"cannot assign to file constant: {name}", node)
            value, actual = self.expr(node.value)
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(node.op))
            if op is None or typ not in NUMERICS or not _compatible(actual, typ):
                raise NexirError("invalid augmented assignment", node)
            self.emit(f"nx_{name} {op}= {value};")
            return False
        if isinstance(node, ast.Return):
            if node.value is None:
                if self.fn.result != "void":
                    raise NexirError(f"return requires {self.fn.result}", node)
                self.emit("return;")
            else:
                value, actual = self.expr(node.value)
                expected = self.fn.result[7:-1] if self.fn.raises else self.fn.result
                if not _compatible(actual, expected):
                    raise NexirError(f"return expected {expected}, got {actual}", node)
                if self.fn.result.startswith("owned[") and isinstance(node.value, ast.Name):
                    value = f"std::move({value})"
                value = _coerce(value, actual, expected)
                self.emit(f"return nx_ok<{self.compiler.types[expected]}>({value});" if self.fn.raises else f"return {value};")
            return True
        if isinstance(node, ast.Raise):
            if not self.fn.raises or node.exc is None or node.cause is not None:
                raise NexirError("raise requires a raises function and one error string", node)
            value, typ = self.expr(node.exc)
            if typ != "str":
                raise NexirError("raise error must be str", node.exc)
            result = self.fn.result[7:-1]
            self.emit(f"return nx_err<{self.compiler.types[result]}>({value});")
            return True
        if isinstance(node, ast.Try):
            if len(node.handlers) != 1 or node.handlers[0].type is not None or node.handlers[0].name or node.orelse or node.finalbody or not node.body:
                raise NexirError("try supports one bare except and no else/finally", node)
            first = node.body[0]
            if isinstance(first, ast.Assign) and len(first.targets) == 1 and isinstance(first.targets[0], ast.Name):
                name, expression = first.targets[0].id, first.value
                annotation = None
            elif isinstance(first, ast.AnnAssign) and isinstance(first.target, ast.Name) and first.value is not None:
                name, expression = first.target.id, first.value
                annotation = _type(first.annotation, types=self.compiler.types)
            else:
                raise NexirError("try must start with a named Result assignment", first)
            _name(name, first)
            if any(name in scope for scope in self.scopes):
                raise NexirError(f"variable already defined: {name}", first)
            value, typ = self.expr(expression)
            if not typ.startswith("Result["):
                raise NexirError("try assignment requires a Result[T] call", expression)
            element = typ[7:-1]
            if annotation and annotation != element:
                raise NexirError(f"try value must have type {element}", first)
            self.serial += 1
            temp = f"nx_try_{self.serial}"
            self.emit(f"auto {temp} = {value};")
            self.emit(f"if ({temp}.ok()) {{")
            self.scopes.append({name: element})
            self.depth += 1
            self.emit(f"{self.compiler.types[element]} nx_{name} = {temp}.value();")
            success = self.block(node.body[1:])
            self.depth -= 1
            self.moved.difference_update(self.scopes[-1])
            self.scopes.pop()
            self.emit("} else {")
            failure = self.nested(node.handlers[0].body)
            self.emit("}")
            return success and failure
        if isinstance(node, ast.If):
            condition, typ = self.expr(node.test)
            if typ != "bool":
                raise NexirError("if condition must be bool", node.test)
            self.emit(f"if ({condition}) {{")
            yes = self.nested(node.body)
            self.emit("}")
            if node.orelse:
                self.emit("else {")
                no = self.nested(node.orelse)
                self.emit("}")
                return yes and no
            return False
        if isinstance(node, ast.While):
            if node.orelse:
                raise NexirError("while-else is not supported", node)
            condition, typ = self.expr(node.test)
            if typ != "bool":
                raise NexirError("while condition must be bool", node.test)
            self.emit(f"while ({condition}) {{")
            self.loop_depth += 1
            self.nested(node.body)
            self.loop_depth -= 1
            self.emit("}")
            return False
        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Name) and not node.orelse:
                name = node.target.id
                _name(name, node.target)
                if any(name in scope for scope in self.scopes):
                    raise NexirError(f"loop variable already defined: {name}", node.target)
                container, typ = self.expr(node.iter)
                element = _list_element(typ) or _span_element(typ)
                buffer = _buffer_parts(typ)
                if buffer:
                    element = buffer[1]
                map_types = _dict_parts(typ)
                set_element = _set_element(typ)
                if map_types or set_element:
                    element = map_types[0] if map_types else set_element
                    self.emit(f"for (const auto &nx_entry : {container}) {{")
                    self.scopes.append({name: element})
                    self.depth += 1
                    self.loop_depth += 1
                    self.emit(f"{self.compiler.types[element]} nx_{name} = nx_entry{'.first' if map_types else ''};")
                    self.block(node.body)
                    self.loop_depth -= 1
                    self.depth -= 1
                    self.moved.difference_update(self.scopes[-1])
                    self.scopes.pop()
                    self.emit("}")
                    return False
                if typ == "str":
                    element = "str"
                    self.require_caps({"memory.alloc"}, node)
                if element:
                    self.serial += 1
                    suffix = self.serial
                    size = f"({container}).size()" if _list_element(typ) else f"({container}).len" if _span_element(typ) else f"({container}){'->' if buffer and buffer[0] == 'owned' else '.'}size()" if buffer else f"std::strlen({container})"
                    value = f"({container})[nx_index_{suffix}]" if _list_element(typ) or typ == "str" else f"({container}).ptr[nx_index_{suffix}]" if _span_element(typ) else f"({container}){'->' if buffer and buffer[0] == 'owned' else '.'}at(nx_index_{suffix})"
                    if typ == "str":
                        value = f"nx_string_keep(std::string(1, {value}))"
                    self.emit(f"for (std::size_t nx_index_{suffix} = 0; nx_index_{suffix} < {size}; ++nx_index_{suffix}) {{")
                    self.scopes.append({name: element})
                    self.depth += 1
                    self.loop_depth += 1
                    self.emit(f"{self.compiler.types[element]} nx_{name} = {value};")
                    self.block(node.body)
                    self.loop_depth -= 1
                    self.depth -= 1
                    self.moved.difference_update(self.scopes[-1])
                    self.scopes.pop()
                    self.emit("}")
                    return False
            if node.orelse or not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name) or node.iter.func.id not in {"range", "parallel_range"} or node.iter.keywords:
                raise NexirError("for supports range(...) or parallel_range(...) only", node)
            parallel = node.iter.func.id == "parallel_range"
            name = node.target.id
            _name(name, node.target)
            if any(name in scope for scope in self.scopes):
                raise NexirError(f"loop variable already defined: {name}", node.target)
            args = [self.expr(a) for a in node.iter.args]
            if len(args) not in ({1, 2} if parallel else {1, 2, 3}) or any(t not in INTEGERS for _, t in args):
                raise NexirError(("parallel_range expects one or two" if parallel else "range expects one to three") + " integer arguments", node.iter)
            start, stop, step = ("0", args[0][0], "1") if len(args) == 1 else (args[0][0], args[1][0], args[2][0] if len(args) == 3 else "1")
            self.serial += 1
            suffix = self.serial
            if parallel:
                if any(isinstance(child, (ast.Return, ast.Break, ast.Continue)) for statement in node.body for child in ast.walk(statement)):
                    raise NexirError("parallel_range body cannot return, break or continue", node)
                if any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "move" for statement in node.body for child in ast.walk(statement)):
                    raise NexirError("parallel_range body cannot move owned values", node)
                for statement in node.body:
                    for child in ast.walk(statement):
                        if isinstance(child, (ast.Assign, ast.AugAssign)):
                            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                            for target in targets:
                                if isinstance(target, (ast.Name, ast.Attribute)):
                                    raise NexirError("parallel_range cannot assign shared variables or struct fields", target)
                self.emit(f"nx_parallel_for({start}, {stop}, [&](int64_t nx_{name}) {{")
                self.scopes.append({name: "i64"})
                self.depth += 1
                self.block(node.body)
                self.depth -= 1
                self.moved.difference_update(self.scopes[-1])
                self.scopes.pop()
                self.emit("});")
                return False
            self.emit(f"int64_t nx_start_{suffix} = {start}, nx_stop_{suffix} = {stop}, nx_step_{suffix} = {step};")
            self.emit(f"if (nx_step_{suffix} == 0) {{ fputs(\"range step cannot be zero\\n\", stderr); exit(1); }}")
            self.emit(f"for (int64_t nx_index_{suffix} = nx_start_{suffix}; (nx_step_{suffix} > 0 ? nx_index_{suffix} < nx_stop_{suffix} : nx_index_{suffix} > nx_stop_{suffix}); nx_index_{suffix} += nx_step_{suffix}) {{")
            self.scopes.append({name: "i64"})
            self.depth += 1
            self.loop_depth += 1
            self.emit(f"int64_t nx_{name} = nx_index_{suffix};")
            self.block(node.body)
            self.loop_depth -= 1
            self.depth -= 1
            self.moved.difference_update(self.scopes[-1])
            self.scopes.pop()
            self.emit("}")
            return False
        raise NexirError(f"unsupported statement: {type(node).__name__}", node)

    def generate(self, name: str) -> list[str]:
        params = ", ".join(f"{self.compiler.types[t]} nx_{n}" for n, t in self.fn.params) or "void"
        if self.fn.cpp_symbol:
            args = ", ".join(f"nx_{n}" for n, _ in self.fn.params)
            call = f"{self.fn.cpp_symbol}({args})"
            statement = f"{call};" if self.fn.result == "void" else f"return {call};"
            return [f"static inline {self.compiler.types[self.fn.result]} nx_{name}({params}) {{ {statement} }}"]
        if self.fn.py_call:
            module, function = self.fn.py_call
            argument = "nx_" + self.fn.params[0][0]
            suffix = "mutspan" if self.fn.params[0][1] == "MutSpan[f64]" else "span"
            result = self.fn.result
            kind = "f64" if result == "f64" else "str" if result == "str" else "buffer"
            call = f"nx_py_call_{kind}_{suffix}({_c_string(module)}, {_c_string(function)}, {argument})"
            if result == "str":
                call = f"nx_adopt_py_string({call})"
            elif result == "owned[Buffer[f64]]":
                call = f"nx_buffer_from_py({call})"
            return [f"static {self.compiler.types[result]} nx_{name}({params}) {{ return {call}; }}"]
        if self.fn.extern_symbol:
            if self.fn.extern_symbol == f"nx_{name}":
                return []
            args = ", ".join(f"nx_{n}" for n, _ in self.fn.params)
            call = f"{self.fn.extern_symbol}({args})"
            statement = f"{call}; return;" if self.fn.result == "void" else f"return {call};"
            return [f"static {self.compiler.types[self.fn.result]} nx_{name}({params}) {{ {statement} }}"]
        self.block(self.fn.node.body)
        if self.fn.result != "void" and not _guaranteed_return(self.fn.node.body):
            raise NexirError(f"function {name} may finish without returning {self.fn.result}", self.fn.node)
        if self.fn.result == "void":
            self.emit("return;")
        if self.fn.generic:
            asserts = [f"static_assert({('nx_' + param + ' > 0 && nx_' + param + ' <= 1024') if bound == 'Int' else ('nx_trait_' + bound + '<' + param + '>::value') if bound in self.compiler.traits else ('std::is_' + ('integral' if bound == 'Integer' else 'arithmetic') + '_v<' + param + '>')}, \"{param} must satisfy {bound}\");" for param, bound in self.fn.generic]
            self.lines = ["    " + line for line in asserts] + self.lines
        linkage = 'extern "C" NX_EXPORT' if self.fn.exported else "static"
        prefix = ["template <" + ", ".join(f"std::size_t nx_{param}" if bound == "Int" else f"typename {param}" for param, bound in self.fn.generic) + ">"] if self.fn.generic else []
        return [*prefix, f"{linkage} {self.compiler.types[self.fn.result]} nx_{name}({params}) {{", *self.lines, "}"]


def _guaranteed_return(nodes: list[ast.stmt]) -> bool:
    for node in nodes:
        if isinstance(node, (ast.Return, ast.Raise)):
            return True
        if isinstance(node, ast.If) and node.orelse and _guaranteed_return(node.body) and _guaranteed_return(node.orelse):
            return True
        if isinstance(node, ast.Try) and node.handlers and _guaranteed_return(node.body) and all(_guaranteed_return(handler.body) for handler in node.handlers):
            return True
    return False


def compile_source(source: str) -> str:
    return Compiler(source).compile()


def compile_file(path: Path) -> str:
    if path.suffix != ".nx":
        raise NexirError("source file must end in .nx")
    imported: set[str] = set()
    loaded: set[Path] = set()
    active: set[Path] = set()

    def load(source_path: Path) -> str:
        source_path = source_path.resolve()
        if source_path in active:
            raise NexirError(f"cyclic import involving {source_path.name}")
        if source_path in loaded:
            return ""
        if not source_path.is_file():
            raise NexirError(f"module not found: {source_path.name}")
        active.add(source_path)
        try:
            source = source_path.read_text(encoding="utf-8")
            normalized, _, _, _ = _normalize_source(source)
            tree = ast.parse(normalized)
            lines = source.splitlines(keepends=True)
            dependencies = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    if len(node.names) != 1 or node.names[0].asname or not node.names[0].name.isidentifier():
                        raise NexirError("import supports one sibling module name", node)
                    module = node.names[0].name
                    imported.add(module)
                    dependencies.append(load(source_path.parent / f"{module}.nx"))
                    lines[node.lineno - 1] = "\n"
            loaded.add(source_path)
            return "\n".join(dependencies) + "\n" + "".join(lines)
        except UnicodeDecodeError as exc:
            raise NexirError("source file must be UTF-8") from exc
        except SyntaxError as exc:
            raise NexirError(f"syntax error in {source_path.name}: {exc.msg} at line {exc.lineno}") from exc
        finally:
            active.remove(source_path)

    return Compiler(load(path), imported).compile()
