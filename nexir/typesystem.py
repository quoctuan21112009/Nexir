"""NEXIR 2.0 Type System: 8..128-bit integers, floats, owned strings, classes, enums,
hybrid static/dynamic typing, collections, Result/Optional for classes, and async/GPU types.
"""

from __future__ import annotations

import ast
import re


SIGNED_INTEGERS = {"i8", "i16", "i32", "i64", "i128"}
UNSIGNED_INTEGERS = {"u8", "u16", "u32", "u64", "u128"}
INTEGERS = SIGNED_INTEGERS | UNSIGNED_INTEGERS
FLOATS = {"f32", "f64"}
NUMERICS = INTEGERS | FLOATS

INT_BITS = {
    "i8": 8, "u8": 8,
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32,
    "i64": 64, "u64": 64,
    "i128": 128, "u128": 128,
}

INT_BOUNDS = {
    "i8": (-(2**7), 2**7 - 1),
    "i16": (-(2**15), 2**15 - 1),
    "i32": (-(2**31), 2**31 - 1),
    "i64": (-(2**63), 2**63 - 1),
    "i128": (-(2**127), 2**127 - 1),
    "u8": (0, 2**8 - 1),
    "u16": (0, 2**16 - 1),
    "u32": (0, 2**32 - 1),
    "u64": (0, 2**64 - 1),
    "u128": (0, 2**128 - 1),
}

TYPE_ALIASES = {
    "int": "i64",
    "float": "f64",
}

BASE_TYPES: dict[str, str] = {
    "i8": "std::int8_t",
    "i16": "std::int16_t",
    "i32": "std::int32_t",
    "i64": "std::int64_t",
    "i128": "nx_i128",
    "u8": "std::uint8_t",
    "u16": "std::uint16_t",
    "u32": "std::uint32_t",
    "u64": "std::uint64_t",
    "u128": "nx_u128",
    "f32": "float",
    "f64": "double",
    "bool": "bool",
    "str": "NxStr",
    "void": "void",
    "dyn": "NxValue",
    "StringBuilder": "NxStrBuilder",
    "Ring[f64]": "nx_ring_ref_f64",
    "File": "nx_File",
    "FileInfo": "nx_FileInfo",
    "TcpStream": "nx_TcpStream",
    "TcpListener": "nx_TcpListener",
    "HttpResponse": "nx_HttpResponse",
    "ThreadPool": "nx_ThreadPool",
    "AtomicI64": "nx_AtomicI64",
}

for _el in sorted(NUMERICS):
    BASE_TYPES[f"Span[{_el}]"] = f"nx_span_{_el}"
    BASE_TYPES[f"MutSpan[{_el}]"] = f"nx_mutspan_{_el}"
    BASE_TYPES[f"Buffer[{_el}]"] = f"nx_buffer<{BASE_TYPES[_el]}>"
    BASE_TYPES[f"owned[Buffer[{_el}]]"] = f"std::unique_ptr<nx_buffer<{BASE_TYPES[_el]}>>"
    BASE_TYPES[f"borrowed[Buffer[{_el}]]"] = f"const nx_buffer<{BASE_TYPES[_el]}>&"
    BASE_TYPES[f"mut[Buffer[{_el}]]"] = f"nx_buffer<{_el if False else BASE_TYPES[_el]}>&"
    BASE_TYPES[f"GpuBuffer[{_el}]"] = f"nx_GpuBuffer<{BASE_TYPES[_el]}>"

# Types treated as Thread-Safe (Send + Sync) for compile-time data race checks
BUILTIN_SEND_TYPES = NUMERICS | {"bool", "str", "void", "AtomicI64", "Ring[f64]"}

# Move-only types (cannot be implicitly copied, must be moved)
MOVE_ONLY_PREFIXES = ("owned[",)
MOVE_ONLY_NAMES = {"File", "TcpStream", "TcpListener"}


def canonical_type_name(name: str) -> str:
    return TYPE_ALIASES.get(name, name)


def span_element(typ: str) -> str | None:
    if (typ.startswith("Span[") or typ.startswith("MutSpan[")) and typ.endswith("]"):
        return typ[typ.index("[") + 1:-1]
    return None


def list_element(typ: str) -> str | None:
    return typ[5:-1] if typ.startswith("List[") and typ.endswith("]") else None


def set_element(typ: str) -> str | None:
    return typ[4:-1] if typ.startswith("Set[") and typ.endswith("]") else None


def gpu_buffer_element(typ: str) -> str | None:
    return typ[10:-1] if typ.startswith("GpuBuffer[") and typ.endswith("]") else None


def future_element(typ: str) -> str | None:
    return typ[7:-1] if typ.startswith("Future[") and typ.endswith("]") else None


def channel_element(typ: str) -> str | None:
    return typ[8:-1] if typ.startswith("Channel[") and typ.endswith("]") else None


def mutex_element(typ: str) -> str | None:
    return typ[6:-1] if typ.startswith("Mutex[") and typ.endswith("]") else None


def optional_element(typ: str) -> str | None:
    return typ[9:-1] if typ.startswith("Optional[") and typ.endswith("]") else None


def result_parts(typ: str) -> tuple[str, str | None] | None:
    if not typ.startswith("Result[") or not typ.endswith("]"):
        return None
    inner = typ[7:-1]
    parts = split_top_level_args(inner)
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def dict_parts(typ: str) -> tuple[str, str] | None:
    if not typ.startswith("Dict[") or not typ.endswith("]"):
        return None
    parts = split_top_level_args(typ[5:-1])
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def tuple_parts(typ: str) -> tuple[str, ...] | None:
    if not typ.startswith("Tuple[") or not typ.endswith("]"):
        return None
    parts = split_top_level_args(typ[6:-1])
    return tuple(parts) if len(parts) >= 2 else None


def simd_parts(typ: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"SIMD\[([A-Za-z_]\w*),([A-Za-z_]\w*|\d+)\]", typ)
    return (match.group(1), match.group(2)) if match else None


def buffer_parts(typ: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(owned|borrowed|mut)\[Buffer\[([A-Za-z0-9_]+)\]\]", typ)
    if match and match.group(2) in NUMERICS:
        return match.group(1), match.group(2)
    return None


def split_top_level_args(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def promote_numeric(a: str, b: str) -> str:
    """Promote two numeric types to their common arithmetic type."""
    if a == b:
        return a
    if "f64" in (a, b):
        return "f64"
    if "f32" in (a, b):
        # f32 + i64/i128 promotes to f64 to avoid precision loss, f32 + smaller int -> f32
        if any(x in ("i64", "u64", "i128", "u128") for x in (a, b)):
            return "f64"
        return "f32"
    if "i128" in (a, b):
        return "i128"
    if "u128" in (a, b):
        return "i128" if any(x in SIGNED_INTEGERS for x in (a, b)) else "u128"
    if "i64" in (a, b):
        return "i64"
    if "u64" in (a, b):
        return "i128" if any(x in SIGNED_INTEGERS for x in (a, b)) else "u64"
    if "i32" in (a, b):
        return "i32"
    if "u32" in (a, b):
        return "i64" if any(x in SIGNED_INTEGERS for x in (a, b)) else "u32"
    if "i16" in (a, b):
        return "i16"
    if "u16" in (a, b):
        return "i32" if any(x in SIGNED_INTEGERS for x in (a, b)) else "u16"
    if "i8" in (a, b):
        return "i8"
    return "u8"


def is_widening_numeric(src: str, dst: str) -> bool:
    if src == dst:
        return True
    if src == "f32" and dst == "f64":
        return True
    if src in SIGNED_INTEGERS and dst in SIGNED_INTEGERS:
        return INT_BITS[src] <= INT_BITS[dst]
    if src in UNSIGNED_INTEGERS and dst in UNSIGNED_INTEGERS:
        return INT_BITS[src] <= INT_BITS[dst]
    if src in UNSIGNED_INTEGERS and dst in SIGNED_INTEGERS:
        return INT_BITS[src] < INT_BITS[dst]
    if src in INTEGERS and dst == "f64" and INT_BITS[src] <= 32:
        return True
    if src in INTEGERS and dst == "f32" and INT_BITS[src] <= 16:
        return True
    return False


def compatible_types(actual: str, expected: str, *, subclasses: dict[str, str] | None = None) -> bool:
    if actual == expected:
        return True
    if expected == "dyn" and actual != "void":
        return True
    if actual == "dyn" and expected in NUMERICS | {"bool", "str"}:
        return True
    opt = optional_element(expected)
    if opt is not None:
        if actual == "None":
            return True
        if compatible_types(actual, opt, subclasses=subclasses):
            return True
    if actual in NUMERICS and expected in NUMERICS:
        return True
    if actual.startswith("List[") and (expected.startswith("Span[") or expected.startswith("MutSpan[")):
        return list_element(actual) == span_element(expected)
    if actual.startswith("MutSpan[") and expected == "Span[" + actual[8:]:
        return True
    # Subclass -> Base class upcast (when using owned or reference or value)
    if subclasses and actual in subclasses:
        cur = actual
        while cur in subclasses:
            cur = subclasses[cur]
            if cur == expected:
                return True
    return False


def coerce_cpp(value: str, actual: str, expected: str, types: dict[str, str]) -> str:
    if actual == expected:
        return value
    opt = optional_element(expected)
    if opt is not None:
        if opt.startswith("owned["):
            return "nullptr" if actual == "None" else value
        return "std::nullopt" if actual == "None" else f"std::optional<{types[opt]}>({coerce_cpp(value, actual, opt, types)})"
    if expected == "dyn" and actual != "dyn":
        return f"nx_box({value})"
    if actual == "dyn" and expected != "dyn":
        return f"nx_unbox<{types[expected]}>({value})"
    if actual in NUMERICS and expected in NUMERICS:
        if is_widening_numeric(actual, expected):
            return f"static_cast<{types[expected]}>({value})"
        return f"nx_num_cast<{types[expected]}>({value})"
    if actual.startswith("List[") and expected.startswith("Span["):
        return f"nx_vec_span({value})"
    if actual.startswith("List[") and expected.startswith("MutSpan["):
        return f"nx_vec_mutspan({value})"
    if actual.startswith("MutSpan[") and expected == "Span[" + actual[8:]:
        elem = span_element(actual)
        return f"nx_as_span_{elem}({value})"
    return value


def resolve_type_node(node: ast.expr | None, *, default: str = "void",
                      types: dict[str, str], classes: set[str] | None = None,
                      enums: set[str] | None = None, error_cls) -> str:
    classes = classes or set()
    enums = enums or set()
    if node is None:
        return default
    if isinstance(node, ast.Constant) and node.value is None:
        return "void"
    if isinstance(node, ast.Name):
        name = canonical_type_name(node.id)
        if name not in types:
            raise error_cls(f"unknown type: {node.id}", node)
        return name
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        # Module-qualified type e.g. mod.MyClass
        qual = f"{node.value.id}.{node.attr}"
        if qual in types:
            return qual
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        head = node.value.id
        if head == "Result":
            elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            if len(elts) == 1:
                inner = resolve_type_node(elts[0], types=types, classes=classes, enums=enums, error_cls=error_cls)
                typ = f"Result[{inner}]"
                types[typ] = "nx_result_void" if inner == "void" else f"nx_result<{types[inner]}>"
                return typ
            if len(elts) == 2:
                ok_t = resolve_type_node(elts[0], types=types, classes=classes, enums=enums, error_cls=error_cls)
                err_t = resolve_type_node(elts[1], types=types, classes=classes, enums=enums, error_cls=error_cls)
                typ = f"Result[{ok_t},{err_t}]"
                types[typ] = f"nx_result_err<{types[ok_t]}, {types[err_t]}>"
                return typ
            raise error_cls("Result takes 1 or 2 type arguments", node)
        if head == "Optional":
            inner = resolve_type_node(node.slice, types=types, classes=classes, enums=enums, error_cls=error_cls)
            if inner == "void":
                raise error_cls("Optional cannot wrap void", node)
            typ = f"Optional[{inner}]"
            if inner.startswith("owned["):
                types[typ] = types[inner]
            else:
                types[typ] = f"std::optional<{types[inner]}>"
            return typ
        if head == "List":
            inner = resolve_type_node(node.slice, types=types, classes=classes, enums=enums, error_cls=error_cls)
            if inner == "void":
                raise error_cls("List cannot hold void", node)
            typ = f"List[{inner}]"
            types[typ] = f"nx_vec<{types[inner]}>"
            return typ
        if head == "Set":
            inner = resolve_type_node(node.slice, types=types, classes=classes, enums=enums, error_cls=error_cls)
            if inner not in INTEGERS | FLOATS | {"str", "bool"} and inner not in enums:
                raise error_cls("Set requires a hashable primitive, str, or enum type", node)
            typ = f"Set[{inner}]"
            cpp_t = types[inner]
            types[typ] = f"nx_set<{cpp_t}, nx_hash<{cpp_t}>, nx_equal<{cpp_t}>>"
            return typ
        if head == "Dict":
            if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) != 2:
                raise error_cls("Dict requires Dict[KeyType, ValueType]", node)
            k_t = resolve_type_node(node.slice.elts[0], types=types, classes=classes, enums=enums, error_cls=error_cls)
            v_t = resolve_type_node(node.slice.elts[1], types=types, classes=classes, enums=enums, error_cls=error_cls)
            if k_t not in INTEGERS | FLOATS | {"str", "bool"} and k_t not in enums:
                raise error_cls("Dict key must be numeric, bool, str, or enum", node)
            if v_t == "void":
                raise error_cls("Dict value cannot be void", node)
            typ = f"Dict[{k_t},{v_t}]"
            ck, cv = types[k_t], types[v_t]
            types[typ] = f"nx_map<{ck}, {cv}, nx_hash<{ck}>, nx_equal<{ck}>>"
            return typ
        if head == "Tuple":
            elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            if not 2 <= len(elts) <= 8:
                raise error_cls("Tuple supports 2 to 8 elements", node)
            parts = [resolve_type_node(p, types=types, classes=classes, enums=enums, error_cls=error_cls) for p in elts]
            if any(p == "void" for p in parts):
                raise error_cls("Tuple cannot contain void", node)
            typ = "Tuple[" + ",".join(parts) + "]"
            types[typ] = "std::tuple<" + ", ".join(types[p] for p in parts) + ">"
            return typ
        if head == "Future":
            inner = resolve_type_node(node.slice, types=types, classes=classes, enums=enums, error_cls=error_cls)
            typ = f"Future[{inner}]"
            types[typ] = f"nx_Future<{types[inner]}>"
            return typ
        if head == "Channel":
            inner = resolve_type_node(node.slice, types=types, classes=classes, enums=enums, error_cls=error_cls)
            typ = f"Channel[{inner}]"
            types[typ] = f"nx_Channel<{types[inner]}>"
            return typ
        if head == "Mutex":
            inner = resolve_type_node(node.slice, types=types, classes=classes, enums=enums, error_cls=error_cls)
            typ = f"Mutex[{inner}]"
            types[typ] = f"nx_Mutex<{types[inner]}>"
            return typ
        if head == "GpuBuffer":
            inner = resolve_type_node(node.slice, types=types, classes=classes, enums=enums, error_cls=error_cls)
            if inner not in NUMERICS:
                raise error_cls("GpuBuffer requires a numeric element type", node)
            typ = f"GpuBuffer[{inner}]"
            types[typ] = f"nx_GpuBuffer<{types[inner]}>"
            return typ
        if head == "SIMD":
            if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) != 2:
                raise error_cls("SIMD requires SIMD[ElementType, Width]", node)
            element, width = node.slice.elts
            if not isinstance(element, ast.Name):
                raise error_cls("SIMD requires a numeric element type", node)
            el_name = canonical_type_name(element.id)
            if el_name not in NUMERICS and f"$numeric:{el_name}" not in types:
                raise error_cls("SIMD requires a numeric element type", node)
            if isinstance(width, ast.Constant) and type(width.value) is int and 1 <= width.value <= 1024:
                width_name, width_cpp = str(width.value), str(width.value)
            elif isinstance(width, ast.Name) and f"$const:{width.id}" in types:
                width_name, width_cpp = width.id, types[f"$const:{width.id}"]
            else:
                raise error_cls("SIMD width must be 1 to 1024 or a generic Int parameter", node)
            typ = f"SIMD[{el_name},{width_name}]"
            types[typ] = f"nx_simd<{types[el_name]}, {width_cpp}>"
            return typ
        if head in {"owned", "borrowed", "mut"}:
            inner = resolve_type_node(node.slice, types=types, classes=classes, enums=enums, error_cls=error_cls)
            typ = f"{head}[{inner}]"
            if typ in types:
                return typ
            if inner in classes:
                if head == "owned":
                    types[typ] = f"std::unique_ptr<{types[inner]}>"
                elif head == "borrowed":
                    types[typ] = f"const {types[inner]}&"
                else:
                    types[typ] = f"{types[inner]}&"
                return typ
            raise error_cls(f"{head}[...] requires Buffer or class type", node)
        if head in {"Span", "MutSpan", "Buffer", "Ring"}:
            inner = resolve_type_node(node.slice, types=types, classes=classes, enums=enums, error_cls=error_cls)
            typ = f"{head}[{inner}]"
            if typ in types:
                return typ
        if head in classes:
            # Generic class instantiation e.g. Box[i32]
            elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            parts = [resolve_type_node(p, types=types, classes=classes, enums=enums, error_cls=error_cls) for p in elts]
            typ = f"{head}[{','.join(parts)}]"
            if typ not in types:
                types[typ] = f"{types[head]}<{', '.join(types[p] for p in parts)}>"
            return typ
    raise error_cls("expected valid NEXIR type annotation", node)
