"""NEXIR 2.0 Statement/Expression Safety Checker & C++17 Code Generator."""

from __future__ import annotations

import ast
import copy
import re
from typing import TYPE_CHECKING

from .security import validate_ffi_symbol
from .stdlib_registry import CPP_KNOWN_HEADERS, STD_MODULES
from .typesystem import (
    INT_BOUNDS,
    INTEGERS,
    MOVE_ONLY_NAMES,
    NUMERICS,
    UNSIGNED_INTEGERS,
    buffer_parts,
    canonical_type_name,
    channel_element,
    coerce_cpp,
    compatible_types,
    dict_parts,
    future_element,
    gpu_buffer_element,
    list_element,
    mutex_element,
    optional_element,
    promote_numeric,
    resolve_type_node,
    result_parts,
    set_element,
    simd_parts,
    span_element,
    tuple_parts,
)

if TYPE_CHECKING:
    from .compiler import Compiler, Function, ModuleUnit


def c_string(value: str) -> str:
    return '"' + "".join(
        chr(b) if 32 <= b < 127 and b not in (34, 92) else f"\\{b:03o}"
        for b in value.encode("utf-8")
    ) + '"'


def nx_str_literal(value: str) -> str:
    raw = value.encode("utf-8")
    return f"NxStr::from_literal({c_string(value)}, {len(raw)})"


def _flatten_str_concat(node: ast.BinOp) -> list[ast.expr] | None:
    if not isinstance(node.op, ast.Add):
        return None
    parts: list[ast.expr] = []
    def collect(n: ast.expr) -> None:
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            collect(n.left)
            collect(n.right)
        else:
            parts.append(n)
    collect(node)
    return parts if len(parts) >= 3 else None


def guaranteed_return(nodes: list[ast.stmt]) -> bool:
    for node in nodes:
        if isinstance(node, (ast.Return, ast.Raise)):
            return True
        if isinstance(node, ast.If) and node.orelse and guaranteed_return(node.body) and guaranteed_return(node.orelse):
            return True
        if isinstance(node, ast.Try) and node.handlers and guaranteed_return(node.body) and all(guaranteed_return(h.body) for h in node.handlers):
            return True
        if isinstance(node, ast.Match) and node.cases and all(guaranteed_return(c.body) for c in node.cases):
            # Conservative check if a wildcard case exists
            if any(isinstance(c.pattern, ast.MatchAs) and c.pattern.name is None and c.pattern.pattern is None and c.guard is None for c in node.cases):
                return True
    return False


class BodyGenerator:
    def __init__(self, compiler: "Compiler", unit: "ModuleUnit", fn: "Function", *, error_cls):
        self.compiler = compiler
        self.unit = unit
        self.fn = fn
        self.error_cls = error_cls
        self.scopes: list[dict[str, str]] = [dict(fn.params)]
        self.narrowed: list[dict[str, str]] = [{}]
        self.moved: set[str] = set()
        self.lines: list[str] = []
        self.depth = 1
        self.loop_depth = 0
        self.serial = 0
        self.in_method = fn.owner is not None

    def numeric(self, typ: str) -> bool:
        return typ in NUMERICS or dict(self.fn.generic).get(typ) in {"Number", "Integer"}

    def emit(self, line: str) -> None:
        self.lines.append("    " * self.depth + line)

    def emit_debug_line(self, node: ast.AST) -> None:
        if self.compiler.debug and hasattr(node, "lineno"):
            src_line = self.unit.original_line(node.lineno)
            self.emit(f"__nx_trace.set_line({src_line});")

    def _cpp_type(self, typ: str) -> str:
        return self.compiler.cpp_type(typ)

    def _check_name(self, name: str, node: ast.AST | None = None) -> str:
        return self.compiler.check_identifier(name, node)

    def _is_move_only(self, typ: str) -> bool:
        return typ.startswith("owned[") or typ in MOVE_ONLY_NAMES

    def _is_send_type(self, typ: str) -> bool:
        if typ in NUMERICS | {"bool", "str", "void", "AtomicI64", "Ring[f64]"}:
            return True
        if typ.startswith(("Channel[", "Mutex[", "Future[", "GpuBuffer[", "owned[Buffer[", "borrowed[")):
            return True
        if typ.startswith("Optional["):
            return self._is_send_type(typ[9:-1])
        return False

    def lookup(self, name: str, node: ast.AST) -> str:
        if name in self.moved:
            raise self.error_cls(f"use after move: {name}", node)
        for nm in reversed(self.narrowed):
            if name in nm:
                return nm[name]
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        if name in self.unit.globals:
            return self.unit.globals[name][0]
        if name in self.unit.imported_symbols:
            sym = self.unit.imported_symbols[name]
            if sym[0] == "global":
                return sym[1].globals[sym[2]][0]
        raise self.error_cls(f"unknown variable: {name}", node)

    def _var_cpp(self, name: str) -> str:
        # Check if narrowed Optional
        is_narrowed = any(name in nm for nm in self.narrowed)
        # Check if file global in current unit or imported symbol
        raw = None
        for scope in reversed(self.scopes):
            if name in scope:
                raw = f"nx_{name}"
                break
        if raw is None:
            if name in self.unit.globals:
                raw = self.unit.cpp_symbol(name)
            elif name in self.unit.imported_symbols and self.unit.imported_symbols[name][0] == "global":
                _, tgt_unit, tgt_name = self.unit.imported_symbols[name]
                raw = tgt_unit.cpp_symbol(tgt_name)
            else:
                raw = f"nx_{name}"
        if is_narrowed:
            # Find base type
            base = None
            for scope in reversed(self.scopes):
                if name in scope:
                    base = scope[name]
                    break
            if base and base.startswith("Optional["):
                inner = base[9:-1]
                if inner.startswith("owned["):
                    return raw
                return f"(*{raw})"
        return raw

    def require_caps(self, required: set[str] | frozenset[str], node: ast.AST) -> None:
        missing = set(required) - self.fn.caps
        if missing:
            raise self.error_cls(
                f"function {self.fn.node.name} needs @cap({', '.join(repr(c) for c in sorted(missing))})",
                node,
            )

    def argument(self, node: ast.expr, expected: str) -> str:
        value, actual = self.expr(node)
        if expected.startswith(("borrowed[", "mut[")):
            owner = expected[expected.index("[") + 1:-1]
            if not isinstance(node, (ast.Name, ast.Attribute)):
                raise self.error_cls("borrow requires a named value", node)
            if actual == f"owned[{owner}]":
                return f"(*{value})"
            if actual == expected:
                return value
            if actual == owner or (expected.startswith("borrowed[") and actual == f"mut[{owner}]"):
                return value
            raise self.error_cls(f"expected {expected}, got {actual}", node)
        if not compatible_types(actual, expected, subclasses=self.compiler.subclasses):
            raise self.error_cls(f"expected {expected}, got {actual}", node)
        if self._is_move_only(expected) and isinstance(node, ast.Name):
            raise self.error_cls(f"{expected} value must be passed with move(...) or freshly constructed", node)
        # Compile-time constant integer range check
        if isinstance(node, ast.Constant) and type(node.value) is int and expected in INT_BOUNDS:
            lo, hi = INT_BOUNDS[expected]
            if not (lo <= node.value <= hi):
                raise self.error_cls(f"integer literal {node.value} out of range for {expected}", node)
        if expected == "str" and actual == "str":
            return value
        return coerce_cpp(value, actual, expected, self.compiler.types)

    def check_borrow_aliases(self, args: list[ast.expr], params: tuple[tuple[str, str], ...]) -> None:
        seen: dict[str, str] = {}
        for arg, (_, expected) in zip(args, params):
            if not isinstance(arg, ast.Name) or not expected.startswith(("borrowed[", "mut[")):
                continue
            mode = "mut" if expected.startswith("mut[") else "borrowed"
            prev = seen.get(arg.id)
            if prev and (mode == "mut" or prev == "mut"):
                raise self.error_cls(f"conflicting borrows of {arg.id} in one call", arg)
            seen[arg.id] = mode

    def _resolve_class(self, typ: str) -> tuple[str, str]:
        """Return (canonical_class_name, deref_operator) for class, owned[Class], borrowed[Class], mut[Class]."""
        if typ.startswith("owned["):
            return typ[6:-1], "->"
        if typ.startswith(("borrowed[", "mut[")):
            return typ[typ.index("[") + 1:-1], "."
        return typ, "."

    def member(self, node: ast.Attribute, *, mutable: bool = False) -> tuple[str, str, str]:
        value, typ = self.expr(node.value)
        owner, operator = self._resolve_class(typ)
        if mutable and typ.startswith("borrowed["):
            raise self.error_cls("borrowed value is read-only", node)
        if owner in {"FileInfo", "HttpResponse"}:
            return f"({value}){operator}nx_{node.attr}", owner, typ
        if owner not in self.compiler.classes:
            raise self.error_cls(f"member access requires a class (got {typ})", node)
        return f"({value}){operator}nx_{node.attr}", owner, typ

    def _find_class_field(self, cls_name: str, field_name: str) -> str | None:
        if cls_name == "FileInfo":
            return {"size": "i64", "mtime": "i64", "is_dir": "bool", "is_file": "bool", "readonly": "bool"}.get(field_name)
        if cls_name == "HttpResponse":
            return {"status": "i32", "body": "str", "headers": "str"}.get(field_name)
        cur: str | None = cls_name
        while cur and cur in self.compiler.classes:
            cls = self.compiler.classes[cur]
            fmap = dict(cls.fields)
            if field_name in fmap:
                return fmap[field_name]
            cur = cls.base
        return None

    def _find_class_method(self, cls_name: str, method_name: str) -> "Function | None":
        cur: str | None = cls_name
        while cur and cur in self.compiler.classes:
            cls = self.compiler.classes[cur]
            if method_name in cls.methods:
                return cls.methods[method_name]
            cur = cls.base
        return None

    def index(self, node: ast.Subscript, *, mutable: bool = False) -> tuple[str, str]:
        container, typ = self.expr(node.value)
        if typ == "dyn":
            key_v, key_t = self.expr(node.slice)
            return f"nx_dyn_getitem({container}, {coerce_cpp(key_v, key_t, 'dyn', self.compiler.types)})", "dyn"
        d_parts = dict_parts(typ)
        if d_parts:
            key = self.argument(node.slice, d_parts[0])
            return (f"({container})[{key}]" if mutable else f"nx_dict_at({container}, {key})"), d_parts[1]
        if typ in self.compiler.classes:
            dunder = "__setitem__" if mutable else "__getitem__"
            method = self._find_class_method(typ, dunder)
            if not mutable and method and len(method.params) == 2:
                self.require_caps(method.caps, node)
                idx = self.argument(node.slice, method.params[1][1])
                return f"({container}).nx___getitem__({idx})", method.result
        t_parts = tuple_parts(typ)
        if t_parts and not mutable:
            if not isinstance(node.slice, ast.Constant) or type(node.slice.value) is not int or not 0 <= node.slice.value < len(t_parts):
                raise self.error_cls("Tuple index must be a nonnegative constant within bounds", node.slice)
            return f"std::get<{node.slice.value}>({container})", t_parts[node.slice.value]
        l_elem = list_element(typ)
        if l_elem:
            idx, idx_t = self.expr(node.slice)
            if idx_t not in INTEGERS:
                raise self.error_cls("List index must be an integer", node.slice)
            return f"nx_list_at({container}, static_cast<std::int64_t>({idx}))", l_elem
        if typ == "str" and not mutable:
            idx, idx_t = self.expr(node.slice)
            if idx_t not in INTEGERS:
                raise self.error_cls("str index must be an integer", node.slice)
            return f"({container}).at_char(static_cast<std::int64_t>({idx}))", "str"
        buf = buffer_parts(typ)
        if buf:
            mode, elem = buf
            if mutable and mode == "borrowed":
                raise self.error_cls("borrowed Buffer is read-only", node)
            idx, idx_t = self.expr(node.slice)
            if idx_t not in INTEGERS:
                raise self.error_cls("Buffer index must be an integer", node.slice)
            op = "->" if mode == "owned" else "."
            return f"({container}){op}at(static_cast<std::int64_t>({idx}))", elem
        s_parts = simd_parts(typ)
        if s_parts:
            idx, idx_t = self.expr(node.slice)
            if idx_t not in INTEGERS:
                raise self.error_cls("SIMD index must be an integer", node.slice)
            if isinstance(node.slice, ast.Constant) and type(node.slice.value) is int and s_parts[1].isdigit():
                if not 0 <= node.slice.value < int(s_parts[1]):
                    raise self.error_cls("SIMD index out of compile-time bounds", node.slice)
            return f"({container}).at(static_cast<std::int64_t>({idx}))", s_parts[0]
        elem = span_element(typ)
        if elem is None or (mutable and not typ.startswith("MutSpan[")):
            raise self.error_cls("indexing requires Span, List, Dict, Buffer, SIMD or str; assignment requires MutSpan, List or Dict", node)
        idx, idx_t = self.expr(node.slice)
        if idx_t not in INTEGERS:
            raise self.error_cls("array index must be an integer", node.slice)
        return f"(*nx_span_at({container}, static_cast<std::int64_t>({idx})))", elem

    def slice_expr(self, node: ast.Subscript) -> tuple[str, str]:
        self.require_caps({"memory.alloc"}, node)
        container, typ = self.expr(node.value)
        elem = list_element(typ) or span_element(typ)
        buf = buffer_parts(typ)
        if buf:
            elem = buf[1]
        if not elem and typ not in {"str", "dyn"}:
            raise self.error_cls("slicing requires List, Span, Buffer, str or dyn", node)
        assert isinstance(node.slice, ast.Slice)
        fields = []
        for part in (node.slice.lower, node.slice.upper, node.slice.step):
            if part is None:
                fields.append("0")
            else:
                val, act = self.expr(part)
                if act not in INTEGERS:
                    raise self.error_cls("slice indices must be integers", part)
                fields.append(f"static_cast<std::int64_t>({val})")
        if node.slice.step is None:
            fields[2] = "1"
        args = f"{fields[0]}, {fields[1]}, {fields[2]}, {'true' if node.slice.lower is not None else 'false'}, {'true' if node.slice.upper is not None else 'false'}"
        if typ == "dyn":
            has_lo = "true" if node.slice.lower is not None else "false"
            has_hi = "true" if node.slice.upper is not None else "false"
            if self.compiler.uses_python:
                return (
                    f"([&]() {{ NxValue _c = {container}; "
                    f"if (_c.type == 9 && _c.data.ptr) {{ "
                    f"NxAbiWireVal _sa[3]; "
                    f"_sa[0] = {has_lo} ? nx_to_abi_wire(static_cast<std::int64_t>({fields[0]})) : NxAbiWireVal{{NX_ABI_T_NULL, 0, {{}}}}; "
                    f"_sa[1] = {has_hi} ? nx_to_abi_wire(static_cast<std::int64_t>({fields[1]})) : NxAbiWireVal{{NX_ABI_T_NULL, 0, {{}}}}; "
                    f"_sa[2] = nx_to_abi_wire(static_cast<std::int64_t>({fields[2]})); "
                    f"NxAbiWireVal _sl = nx_py_abi_call(\"builtins\", \"slice\", 3, _sa, nullptr, 0); "
                    f"NxValue _res = nx_py_dyn_method(_c, \"__getitem__\", 1, &_sl, nullptr, 0); "
                    f"nx_py_abi_release_val(&_sl); return _res; }} "
                    f"if (_c.type == 3 && _c.data.ptr) return NxValue::string(nx_string_slice(*std::static_pointer_cast<NxStr>(_c.data.ptr->object), {args})); "
                    f"if (_c.type == 7 && _c.data.ptr) {{ const auto &_v = *std::static_pointer_cast<nx_vec<NxValue>>(_c.data.ptr->object); return NxValue::list(nx_list_slice(_v.data(), _v.size(), {args})); }} "
                    f"nx_fail(\"dyn object does not support slicing\"); }}())",
                    "dyn",
                )
            return (
                f"([&]() {{ NxValue _c = {container}; "
                f"if (_c.type == 3 && _c.data.ptr) return NxValue::string(nx_string_slice(*std::static_pointer_cast<NxStr>(_c.data.ptr->object), {args})); "
                f"if (_c.type == 7 && _c.data.ptr) {{ const auto &_v = *std::static_pointer_cast<nx_vec<NxValue>>(_c.data.ptr->object); return NxValue::list(nx_list_slice(_v.data(), _v.size(), {args})); }} "
                f"nx_fail(\"dyn object does not support slicing\"); }}())",
                "dyn",
            )
        if typ == "str":
            return f"nx_string_slice({container}, {args})", "str"
        ptr = f"({container}).data()" if list_element(typ) else f"({container}).ptr" if span_element(typ) else f"({container}){'->' if buf and buf[0] == 'owned' else '.'}data()"
        length = f"({container}).size()" if list_element(typ) else f"({container}).len" if span_element(typ) else f"({container}){'->' if buf and buf[0] == 'owned' else '.'}size()"
        return f"nx_slice({ptr}, {length}, {args})", f"List[{elem}]"

    def comprehension(self, node: ast.ListComp | ast.DictComp) -> tuple[str, str]:
        self.require_caps({"memory.alloc"}, node)
        if len(node.generators) != 1:
            raise self.error_cls("comprehension supports one for clause", node)
        clause = node.generators[0]
        if clause.is_async or not isinstance(clause.target, ast.Name):
            raise self.error_cls("comprehension requires a named loop variable", clause)
        name = clause.target.id
        self._check_name(name, clause.target)
        # Support iterating over range(...) or named/expression iterable!
        is_range = isinstance(clause.iter, ast.Call) and isinstance(clause.iter.func, ast.Name) and clause.iter.func.id == "range" and not clause.iter.keywords
        if is_range:
            r_args = [self.expr(a) for a in clause.iter.args]
            if len(r_args) not in {1, 2, 3} or any(t not in INTEGERS for _, t in r_args):
                raise self.error_cls("range expects 1 to 3 integer arguments", clause.iter)
            start, stop, step = ("0", r_args[0][0], "1") if len(r_args) == 1 else (r_args[0][0], r_args[1][0], r_args[2][0] if len(r_args) == 3 else "1")
            element = "i64"
            container = None
            typ = "range"
        else:
            container, typ = self.expr(clause.iter)
            element = list_element(typ) or span_element(typ)
            buf = buffer_parts(typ)
            if buf:
                element = buf[1]
            if not element:
                raise self.error_cls("comprehension requires List, Span, Buffer or range", clause.iter)
        self.scopes.append({name: element})
        try:
            checks = [self.expr(item) for item in clause.ifs]
            if any(act != "bool" for _, act in checks):
                raise self.error_cls("comprehension filter must be bool", node)
            if isinstance(node, ast.ListComp):
                val, val_t = self.expr(node.elt)
                result_type = f"List[{val_t}]"
                action = f"nx_out.push_back({val});"
            else:
                k, kt = self.expr(node.key)
                v, vt = self.expr(node.value)
                result_type = f"Dict[{kt},{vt}]"
                action = f"nx_out[{k}] = {v};"
        finally:
            self.scopes.pop()
        self.compiler.ensure_type(result_type, node)
        self.serial += 1
        sfx = self.serial
        guard = "" if not checks else "if (" + " && ".join(c for c, _ in checks) + ") "
        if is_range:
            code = (
                f"([&]() {{ {self._cpp_type(result_type)} nx_out; "
                f"std::int64_t nx_cs_{sfx} = static_cast<std::int64_t>({start}), nx_ce_{sfx} = static_cast<std::int64_t>({stop}), nx_cp_{sfx} = static_cast<std::int64_t>({step}); "
                f"if (nx_cp_{sfx} == 0) nx_fail(\"range step cannot be zero\"); "
                f"for (std::int64_t nx_{name} = nx_cs_{sfx}; (nx_cp_{sfx} > 0 ? nx_{name} < nx_ce_{sfx} : nx_{name} > nx_ce_{sfx}); nx_{name} += nx_cp_{sfx}) {{ "
                f"{guard}{action} }} return nx_out; }}())"
            )
            return code, result_type
        buf = buffer_parts(typ)
        size = f"({container}).size()" if list_element(typ) else f"({container}).len" if span_element(typ) else f"({container}){'->' if buf and buf[0] == 'owned' else '.'}size()"
        item = f"({container})[nx_index_{sfx}]" if list_element(typ) else f"({container}).ptr[nx_index_{sfx}]" if span_element(typ) else f"({container}){'->' if buf and buf[0] == 'owned' else '.'}at(nx_index_{sfx})"
        code = (
            f"([&]() {{ {self._cpp_type(result_type)} nx_out; "
            f"auto &&nx_src_{sfx} = {container}; "
            f"for (std::size_t nx_index_{sfx} = 0; nx_index_{sfx} < {size.replace(container, f'nx_src_{sfx}')}; ++nx_index_{sfx}) {{ "
            f"{self._cpp_type(element)} nx_{name} = {item.replace(container, f'nx_src_{sfx}')}; {guard}{action} }} return nx_out; }}())"
        )
        return code, result_type

    def array_literal(self, node: ast.List, *, expected_hint: str | None = None) -> tuple[str, str]:
        self.require_caps({"memory.alloc"}, node)
        if not node.elts:
            if expected_hint and list_element(expected_hint):
                return f"{self._cpp_type(expected_hint)}{{}}", expected_hint
            self.compiler.ensure_type("List[dyn]", node)
            return f"{self._cpp_type('List[dyn]')}{{}}", "List[dyn]"
        elements = [self.expr(item) for item in node.elts]
        types_set = {t for _, t in elements}
        if expected_hint and list_element(expected_hint):
            elem_t = list_element(expected_hint)
            assert elem_t is not None
            coerced = [coerce_cpp(c, t, elem_t, self.compiler.types) if compatible_types(t, elem_t, subclasses=self.compiler.subclasses) else None for c, t in elements]
            if all(c is not None for c in coerced):
                self.compiler.ensure_type(expected_hint, node)
                return f"{self._cpp_type(expected_hint)}{{{', '.join(c for c in coerced if c is not None)}}}", expected_hint
        if len(types_set) == 1:
            elem_t = next(iter(types_set))
            typ = f"List[{elem_t}]"
            self.compiler.ensure_type(typ, node)
            return f"{self._cpp_type(typ)}{{{', '.join(c for c, _ in elements)}}}", typ
        if all(t in NUMERICS for t in types_set):
            elem_t = "f64" if "f64" in types_set else "f32" if "f32" in types_set else "i128" if "i128" in types_set else "i64" if "i64" in types_set else "i32"
            typ = f"List[{elem_t}]"
            self.compiler.ensure_type(typ, node)
            coerced = [coerce_cpp(c, t, elem_t, self.compiler.types) for c, t in elements]
            return f"{self._cpp_type(typ)}{{{', '.join(coerced)}}}", typ
        # Heterogeneous list -> List[dyn] (hybrid Pythonic list!)
        self.compiler.ensure_type("List[dyn]", node)
        boxed = [coerce_cpp(c, t, "dyn", self.compiler.types) for c, t in elements]
        return f"{self._cpp_type('List[dyn]')}{{{', '.join(boxed)}}}", "List[dyn]"

    def _fstring(self, node: ast.JoinedStr) -> tuple[str, str]:
        self.require_caps({"memory.alloc"}, node)
        self.serial += 1
        sfx = self.serial
        stmts = [f"NxStrBuilder nx_sb_{sfx};"]
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                stmts.append(f"nx_sb_{sfx}.append({c_string(part.value)});")
            elif isinstance(part, ast.FormattedValue):
                val, typ = self.expr(part.value)
                if typ == "void":
                    raise self.error_cls("f-string expression cannot be void", part)
                spec_str = ""
                if part.format_spec is not None:
                    if isinstance(part.format_spec, ast.JoinedStr) and all(isinstance(v, ast.Constant) and isinstance(v.value, str) for v in part.format_spec.values):
                        spec_str = "".join(v.value for v in part.format_spec.values)  # type: ignore[misc]
                    else:
                        raise self.error_cls("dynamic f-string format_spec is not supported", part.format_spec)
                if spec_str:
                    stmts.append(f"nx_sb_{sfx}.append_fmt({val}, {c_string(spec_str)});")
                else:
                    stmts.append(f"nx_sb_{sfx}.append({val});")
            else:
                raise self.error_cls("unsupported f-string component", part)
        stmts.append(f"return nx_sb_{sfx}.finish();")
        return f"([&]() {{ {' '.join(stmts)} }}())", "str"

    def _call_stdlib(self, mod_name: str, attr: str, node: ast.Call) -> tuple[str, str]:
        mod = STD_MODULES[mod_name]
        if mod_name == "nx.thread":
            if attr == "ThreadPool":
                if len(node.args) > 1 or node.keywords:
                    raise self.error_cls("ThreadPool takes 0 or 1 worker count argument", node)
                w = self.argument(node.args[0], "i64") if node.args else "0"
                return f"nx_ThreadPool({w})", "ThreadPool"
            if attr == "Channel":
                raise self.error_cls("use Channel[T](capacity) to construct a typed Channel", node)
            if attr == "Mutex":
                if len(node.args) != 1 or node.keywords:
                    raise self.error_cls("Mutex(initial) expects 1 argument", node)
                val, vt = self.expr(node.args[0])
                typ = f"Mutex[{vt}]"
                self.compiler.ensure_type(typ, node)
                return f"nx_Mutex<{self._cpp_type(vt)}>({val})", typ
            if attr == "AtomicI64":
                if len(node.args) > 1 or node.keywords:
                    raise self.error_cls("AtomicI64 takes 0 or 1 initial i64", node)
                v = self.argument(node.args[0], "i64") if node.args else "0"
                return f"nx_AtomicI64({v})", "AtomicI64"
        if mod_name == "nx.math" and attr in {"min", "max"}:
            if len(node.args) != 2:
                raise self.error_cls(f"nx.math.{attr} takes 2 numeric arguments", node)
            a, at = self.expr(node.args[0])
            b, bt = self.expr(node.args[1])
            if at not in NUMERICS or bt not in NUMERICS:
                raise self.error_cls("min/max require numeric arguments", node)
            rt = promote_numeric(at, bt)
            fn_name = "std::min" if attr == "min" else "std::max"
            return f"{fn_name}({coerce_cpp(a, at, rt, self.compiler.types)}, {coerce_cpp(b, bt, rt, self.compiler.types)})", rt

        if attr not in mod:
            raise self.error_cls(f"{mod_name} has no function {attr}", node)
        spec = mod[attr]
        self.require_caps(spec.caps, node)
        if spec.variadic and attr == "gather":
            if not node.args or node.keywords:
                raise self.error_cls("asyncio.gather expects at least one Future argument", node)
            args_cpp = []
            parts = []
            for a in node.args:
                c, t = self.expr(a)
                fe = future_element(t)
                if fe is None or fe == "void":
                    raise self.error_cls("asyncio.gather expects non-void Future[T] arguments", a)
                args_cpp.append(f"std::move({c})" if isinstance(a, ast.Name) else c)
                parts.append(fe)
            if len(parts) == 1:
                return f"({args_cpp[0]}).get()", parts[0]
            tup_t = "Tuple[" + ",".join(parts) + "]"
            self.compiler.ensure_type(tup_t, node)
            return f"nx_async_gather({', '.join(args_cpp)})", tup_t

        if len(node.args) != len(spec.params) and not (mod_name == "nx.net" and attr in {"connect", "http_post", "http_request"}):
            # Allow optional default parameters for connect(host, port, timeout), http_post(url, body, content_type)
            if not (mod_name == "nx.net" and attr == "connect" and len(node.args) == 3):
                raise self.error_cls(f"{mod_name}.{attr} expects {len(spec.params)} arguments", node)

        # Polymorphic stdlib calls
        if spec.params == ("ANY",) and attr == "run":
            val, vt = self.expr(node.args[0])
            fe = future_element(vt)
            if fe is None:
                raise self.error_cls("asyncio.run expects a Future[T]", node.args[0])
            return f"nx_async_run({f'std::move({val})' if isinstance(node.args[0], ast.Name) else val})", fe
        if spec.params == ("ANY_LIST",) and attr == "to_device":
            val, vt = self.expr(node.args[0])
            el = list_element(vt)
            if el not in NUMERICS:
                raise self.error_cls("gpu.to_device expects List[Numeric]", node.args[0])
            rt = f"GpuBuffer[{el}]"
            self.compiler.ensure_type(rt, node)
            return f"nx_gpu_to_device({val})", rt
        if attr == "to_host":
            val, vt = self.expr(node.args[0])
            el = gpu_buffer_element(vt)
            if not el:
                raise self.error_cls("gpu.to_host expects GpuBuffer[T]", node.args[0])
            rt = f"List[{el}]"
            self.compiler.ensure_type(rt, node)
            return f"nx_gpu_to_host({val})", rt
        if attr in {"add", "mul"} and spec.params == ("ANY_GPU_BUF", "ANY_GPU_BUF"):
            a, at = self.expr(node.args[0])
            b, bt = self.expr(node.args[1])
            if not gpu_buffer_element(at) or at != bt:
                raise self.error_cls(f"gpu.{attr} expects matching GpuBuffer[T] arguments", node)
            return f"{spec.cpp_expr}({a}, {b})", at
        if attr == "reduce_sum":
            val, vt = self.expr(node.args[0])
            el = gpu_buffer_element(vt)
            if not el:
                raise self.error_cls("gpu.reduce_sum expects GpuBuffer[T]", node.args[0])
            return f"nx_gpu_reduce_sum({val})", el

        if mod_name == "nx.net" and attr == "connect" and len(node.args) == 3:
            c_args = [self.argument(node.args[0], "str"), self.argument(node.args[1], "i32"), self.argument(node.args[2], "i32")]
        else:
            c_args = [self.argument(a, p) for a, p in zip(node.args, spec.params)]
        self.compiler.ensure_type(spec.result, node)
        return f"{spec.cpp_expr}({', '.join(c_args)})", spec.result

    def _call_python_direct(self, py_mod: str, fn_name: str, node: ast.Call) -> tuple[str, str]:
        self.require_caps({"python.interop", "memory.alloc"}, node)
        self.compiler.uses_python = True
        self.compiler.py_preloads.add(py_mod)
        return self._emit_py_abi_call(f"nx_py_abi_call({c_string(py_mod)}, {c_string(fn_name)}", node)

    def _emit_py_abi_call(self, prefix_call: str, node: ast.Call) -> tuple[str, str]:
        self.serial += 1
        sfx = self.serial
        arg_vals: list[str] = []
        keep_stmts: list[str] = [f"NxAbiWireArena nx_arena_{sfx};"]
        kw_names: list[str] = []
        for idx, arg in enumerate(node.args):
            c, t = self.expr(arg)
            arg_vals.append(self._marshal_to_abi_wire(c, t, f"{sfx}_{idx}", keep_stmts))
        for kidx, kw in enumerate(node.keywords):
            if kw.arg is None:
                c, t = self.expr(kw.value)
                dp = dict_parts(t)
                if not ((dp and dp[0] == "str") or t == "dyn"):
                    raise self.error_cls("**kwargs unpacking requires Dict[str, T] or dyn dict", kw)
                kw_names.append("**")
                arg_vals.append(self._marshal_to_abi_wire(c, t, f"{sfx}_kw{kidx}", keep_stmts))
            else:
                kw_names.append(kw.arg)
                c, t = self.expr(kw.value)
                arg_vals.append(self._marshal_to_abi_wire(c, t, f"{sfx}_kw{kidx}", keep_stmts))
        total_args = len(arg_vals)
        kw_count = len(kw_names)
        arr_init = f"NxAbiWireVal nx_pargs_{sfx}[] = {{{', '.join(arg_vals)}}};" if total_args else ""
        kw_init = f"const char* nx_pkw_{sfx}[] = {{{', '.join(c_string(k) for k in kw_names)}}};" if kw_count else ""
        argv_ptr = f"nx_pargs_{sfx}" if total_args else "nullptr"
        kw_ptr = f"nx_pkw_{sfx}" if kw_count else "nullptr"
        body = (
            f"([&]() {{ "
            f"{' '.join(keep_stmts)} {arr_init} {kw_init} "
            f"NxAbiWireVal nx_res_{sfx} = {prefix_call}, {total_args}, {argv_ptr}, {kw_ptr}, {kw_count}); "
            f"NxValue nx_out_{sfx} = nx_py_wire_to_nxvalue(nx_res_{sfx}); "
            f"nx_py_abi_release_val(&nx_res_{sfx}); "
            f"return nx_out_{sfx}; }}())"
        )
        return body, "dyn"

    def _marshal_to_abi_wire(self, expr_cpp: str, typ: str, tag: str, keep_stmts: list[str]) -> str:
        el = span_element(typ)
        if el:
            dtype_map = {
                "i8": 1, "i16": 2, "i32": 3, "i64": 4, "u8": 6, "u16": 7, "u32": 8, "u64": 9,
                "f32": 11, "f64": 12,
            }
            dt = dtype_map.get(el, 12)
            writable = 1 if typ.startswith("MutSpan[") else 0
            keep_stmts.append(f"auto nx_spv_{tag} = {expr_cpp};")
            keep_stmts.append(f"NxAbiWireSpan nx_sp_{tag}{{const_cast<void*>(static_cast<const void*>(nx_spv_{tag}.ptr)), nx_spv_{tag}.len, {dt}u, {writable}u}};")
            keep_stmts.append(f"NxAbiWireVal nx_w_{tag}{{}}; nx_w_{tag}.tag = NX_ABI_T_SPAN; nx_w_{tag}.payload.ptr = &nx_sp_{tag};")
            return f"nx_w_{tag}"
        le = list_element(typ)
        if le:
            return f"nx_arena_{tag.split('_')[0]}.pack({expr_cpp})"
        tp = tuple_parts(typ)
        if tp:
            return f"nx_arena_{tag.split('_')[0]}.pack({expr_cpp})"
        dp = dict_parts(typ)
        if dp and dp[0] == "str":
            return f"nx_arena_{tag.split('_')[0]}.pack({expr_cpp})"
        if typ == "str":
            return f"nx_arena_{tag.split('_')[0]}.pack({expr_cpp})"
        if typ == "dyn":
            return f"nx_arena_{tag.split('_')[0]}.pack({expr_cpp})"
        return f"nx_to_abi_wire({expr_cpp})"

    def _call_cpp_direct(self, header: str, sym: str, node: ast.Call, *, expected_hint: str | None = None) -> tuple[str, str]:
        self.require_caps({"ffi.cpp"}, node)
        try:
            validate_ffi_symbol(sym, allow_process=("process.exec" in self.fn.caps))
        except ValueError as exc:
            raise self.error_cls(str(exc), node) from exc
        known = CPP_KNOWN_HEADERS.get(header, {}).get(sym)
        if known:
            params_t, ret_t = known
            if len(node.args) != len(params_t):
                raise self.error_cls(f"cpp.{header}.{sym} expects {len(params_t)} arguments", node)
            args = [self.argument(a, pt) for a, pt in zip(node.args, params_t)]
            return f"std::{sym}({', '.join(args)})", ret_t
        auto_sig = self.compiler.entry_unit.foreign_sigs.get(("cpp", header), {}).get(sym)
        if auto_sig:
            params_t, ret_t = auto_sig
            if len(node.args) == len(params_t):
                args = [f"({self.argument(a, pt)}).c_str()" if pt == "str" else self.argument(a, pt) for a, pt in zip(node.args, params_t)]
                call = f"{sym}({', '.join(args)})"
                return (f"NxStr({call})", "str") if ret_t == "str" else (call, ret_t)
        args = []
        for a in node.args:
            c, t = self.expr(a)
            args.append(f"({c}).c_str()" if t == "str" else c)
        ret = expected_hint or "dyn"
        call = f"{sym}({', '.join(args)})"
        if ret == "dyn":
            return f"nx_box({call})", "dyn"
        return call, ret

    def _call_native_lib(self, kind: str, lib_name: str, sym: str, node: ast.Call, *, expected_hint: str | None = None) -> tuple[str, str]:
        cap = f"ffi.{kind}" if kind in {"c", "rust", "go"} else "ffi.cpp"
        self.require_caps({cap}, node)
        try:
            validate_ffi_symbol(sym, allow_process=("process.exec" in self.fn.caps))
        except ValueError as exc:
            raise self.error_cls(str(exc), node) from exc
        auto_sig = self.compiler.entry_unit.foreign_sigs.get((kind, lib_name), {}).get(sym)
        if not auto_sig:
            raise self.error_cls(
                f"missing native ABI signature for {kind}.{lib_name}.{sym}; "
                f"provide a C header/binding or declare an explicit @extern function",
                node,
            )
        params_t, ret_nx = auto_sig
        if len(node.args) != len(params_t):
            raise self.error_cls(
                f"{kind}.{lib_name}.{sym} expects {len(params_t)} arguments, got {len(node.args)}",
                node,
            )
        arg_vals = []
        arg_types = []
        for a, pt in zip(node.args, params_t):
            c = self.argument(a, pt)
            arg_vals.append(f"({c}).c_str()" if pt == "str" else c)
            arg_types.append("const char *" if pt == "str" else self._cpp_type(pt))
        ret_cpp = self._cpp_type(ret_nx)
        fn_ptr_sig = f"{ret_cpp}(*)({', '.join(arg_types) or 'void'})"
        call = f"reinterpret_cast<{fn_ptr_sig}>(NxNativeLibLoader::load_symbol({c_string(kind)}, {c_string(lib_name)}, {c_string(sym)}))({', '.join(arg_vals)})"
        if ret_nx == "str":
            return f"NxStr({call})", "str"
        return call, ret_nx

    def expr(self, node: ast.expr, *, expected_hint: str | None = None) -> tuple[str, str]:
        if isinstance(node, (ast.ListComp, ast.DictComp)):
            return self.comprehension(node)
        if isinstance(node, ast.Tuple):
            if not 2 <= len(node.elts) <= 8:
                raise self.error_cls("Tuple literal requires 2 to 8 values", node)
            vals = [self.expr(it) for it in node.elts]
            parts = [t for _, t in vals]
            if any(p == "void" for p in parts):
                raise self.error_cls("Tuple cannot contain void", node)
            typ = "Tuple[" + ",".join(parts) + "]"
            self.compiler.ensure_type(typ, node)
            return f"{self._cpp_type(typ)}{{{', '.join(c for c, _ in vals)}}}", typ
        if isinstance(node, ast.Dict):
            self.require_caps({"memory.alloc"}, node)
            if not node.keys:
                if expected_hint and dict_parts(expected_hint):
                    self.compiler.ensure_type(expected_hint, node)
                    return f"{self._cpp_type(expected_hint)}{{}}", expected_hint
                self.compiler.ensure_type("Dict[str,dyn]", node)
                return f"{self._cpp_type('Dict[str,dyn]')}{{}}", "Dict[str,dyn]"
            if any(k is None for k in node.keys):
                raise self.error_cls("Dict unpacking literals are not supported", node)
            keys = [self.expr(k) for k in node.keys if k is not None]
            vals = [self.expr(v) for v in node.values]
            k_t, v_t = keys[0][1], vals[0][1]
            if expected_hint and dict_parts(expected_hint):
                ek, ev = dict_parts(expected_hint)  # type: ignore[misc]
                if all(compatible_types(kt, ek) for _, kt in keys) and all(compatible_types(vt, ev) for _, vt in vals):
                    k_t, v_t = ek, ev
            elif any(vt != v_t for _, vt in vals):
                v_t = "dyn"
            typ = f"Dict[{k_t},{v_t}]"
            self.compiler.ensure_type(typ, node)
            pairs = ", ".join(
                "{" + coerce_cpp(kc, kt, k_t, self.compiler.types) + ", " + coerce_cpp(vc, vt, v_t, self.compiler.types) + "}"
                for (kc, kt), (vc, vt) in zip(keys, vals)
            )
            return f"{self._cpp_type(typ)}{{{pairs}}}", typ
        if isinstance(node, ast.Set):
            self.require_caps({"memory.alloc"}, node)
            vals = [self.expr(it) for it in node.elts]
            el = vals[0][1]
            typ = f"Set[{el}]"
            self.compiler.ensure_type(typ, node)
            if any(t != el for _, t in vals):
                raise self.error_cls("Set literal requires uniform supported element types", node)
            return f"{self._cpp_type(typ)}{{{', '.join(c for c, _ in vals)}}}", typ
        if isinstance(node, ast.List):
            return self.array_literal(node, expected_hint=expected_hint)
        if isinstance(node, ast.JoinedStr):
            return self._fstring(node)
        if isinstance(node, ast.Await):
            if not self.fn.is_async:
                raise self.error_cls("await can only be used inside async def", node)
            val, vt = self.expr(node.value)
            fe = future_element(vt)
            if fe is None:
                raise self.error_cls("await requires a Future[T]", node.value)
            arg = f"std::move({val})" if isinstance(node.value, ast.Name) else val
            return f"nx_await({arg})", fe
        if isinstance(node, ast.IfExp):
            cond, ct = self.expr(node.test)
            yes, yt = self.expr(node.body)
            no, nt = self.expr(node.orelse)
            if ct == "dyn":
                cond = f"nx_as_bool({cond})"
                ct = "bool"
            if ct != "bool":
                raise self.error_cls("conditional expression condition must be bool", node)
            if yt in NUMERICS and nt in NUMERICS and yt != nt:
                rt = promote_numeric(yt, nt)
                return f"({cond} ? {coerce_cpp(yes, yt, rt, self.compiler.types)} : {coerce_cpp(no, nt, rt, self.compiler.types)})", rt
            if yt != nt:
                raise self.error_cls("conditional expression branches must have compatible types", node)
            return f"({cond} ? {yes} : {no})", yt
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "std::nullopt", "None"
            if type(node.value) is bool:
                return ("true" if node.value else "false", "bool")
            if type(node.value) is int:
                n = node.value
                if -(2**31) <= n < 2**31:
                    return str(n), "i32"
                if -(2**63) <= n < 2**63:
                    return f"{n}LL", "i64"
                if -(2**127) <= n < 2**127:
                    lo = n & ((1 << 64) - 1)
                    hi = n >> 64
                    return f"((static_cast<nx_i128>({hi}LL) << 64) | static_cast<nx_u128>({lo}ULL))", "i128"
                if 0 <= n < 2**128:
                    lo = n & ((1 << 64) - 1)
                    hi = n >> 64
                    return f"((static_cast<nx_u128>({hi}ULL) << 64) | static_cast<nx_u128>({lo}ULL))", "u128"
                raise self.error_cls("integer literal exceeds 128-bit range", node)
            if type(node.value) is float:
                return repr(node.value), "f64"
            if type(node.value) is str:
                return nx_str_literal(node.value), "str"
            raise self.error_cls("unsupported literal", node)
        if isinstance(node, ast.Name):
            typ = self.lookup(node.id, node)
            return self._var_cpp(node.id), typ
        if isinstance(node, ast.Attribute):
            # Check Enum.VARIANT (unit variant) or mod.Enum.VARIANT or mod.CONST
            if isinstance(node.value, ast.Name):
                base_name = node.value.id
                if base_name in self.unit.enums:
                    enum_def = self.unit.enums[base_name]
                    if node.attr not in enum_def.variants:
                        raise self.error_cls(f"unknown enum variant: {base_name}.{node.attr}", node)
                    v_params = enum_def.variants[node.attr]
                    if v_params:
                        raise self.error_cls(f"{base_name}.{node.attr} requires arguments", node)
                    return f"{self.unit.cpp_symbol(base_name)}::{node.attr}()", base_name
                if base_name in self.unit.imports:
                    imp = self.unit.imports[base_name]
                    if imp.kind == "nx":
                        tgt = imp.target_unit
                        assert tgt is not None
                        if node.attr in tgt.globals:
                            return tgt.cpp_symbol(node.attr), tgt.globals[node.attr][0]
                    elif imp.kind == "py":
                        self.require_caps({"python.interop", "memory.alloc"}, node)
                        self.compiler.uses_python = True
                        self.compiler.py_preloads.add(imp.target_name)
                        return f"nx_py_abi_get_attr_mod({c_string(imp.target_name)}, {c_string(node.attr)})", "dyn"
                    elif imp.kind == "nx_std" and imp.target_name == "nx.math":
                        if node.attr == "pi":
                            return "3.14159265358979323846", "f64"
                        if node.attr == "e":
                            return "2.71828182845904523536", "f64"
            # Dynamic pyobj / dyn attribute access
            if isinstance(node.value, ast.Name) or True:
                # Try resolving object
                try:
                    val, typ = self.expr(node.value)
                except self.error_cls:
                    raise
                if typ == "dyn":
                    self.compiler.uses_python = True
                    return f"nx_py_dyn_getattr({val}, {c_string(node.attr)})", "dyn"
                owner, operator = self._resolve_class(typ)
                if owner not in self.compiler.classes and owner not in {"FileInfo", "HttpResponse"}:
                    raise self.error_cls(f"member access requires a class (got {typ})", node)
                field_t = self._find_class_field(owner, node.attr)
                if field_t is None:
                    raise self.error_cls(f"unknown class field: {owner}.{node.attr}", node)
                if typ.startswith("borrowed[") and field_t.startswith(("owned[", "MutSpan[")):
                    raise self.error_cls("borrowed value cannot expose a mutable owned field", node)
                return f"({val}){operator}nx_{node.attr}", field_t
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                return self.slice_expr(node)
            return self.index(node)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) and type(node.operand.value) is int:
                n = -node.operand.value
                if n == -(2**31):
                    return "(-2147483647 - 1)", "i32"
                if n == -(2**63):
                    return "(-9223372036854775807LL - 1LL)", "i64"
                if -(2**31) <= n < 2**31:
                    return f"({n})", "i32"
                if -(2**63) <= n < 2**63:
                    return f"({n}LL)", "i64"
                if -(2**127) <= n < 2**127:
                    lo = n & ((1 << 64) - 1)
                    hi = n >> 64
                    return f"((static_cast<nx_i128>({hi}LL) << 64) | static_cast<nx_u128>({lo}ULL))", "i128"
                raise self.error_cls("integer literal exceeds i128", node)
            value, typ = self.expr(node.operand)
            if isinstance(node.op, ast.Not) and (typ == "bool" or typ == "dyn"):
                if typ == "dyn":
                    return f"(!nx_as_bool({value}))", "bool"
                return f"(!{value})", "bool"
            if isinstance(node.op, (ast.UAdd, ast.USub)) and self.numeric(typ):
                sign = "+" if isinstance(node.op, ast.UAdd) else "-"
                return f"({sign}{value})", typ
            if isinstance(node.op, ast.Invert) and typ in INTEGERS:
                return f"(~{value})", typ
            raise self.error_cls("invalid unary operation", node)
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.BoolOp):
            op = " && " if isinstance(node.op, ast.And) else " || " if isinstance(node.op, ast.Or) else None
            vals = [self.expr(v) for v in node.values]
            if op is None or any(t not in {"bool", "dyn"} for _, t in vals):
                raise self.error_cls("boolean operations require bool operands", node)
            parts = [f"nx_as_bool({c})" if t == "dyn" else c for c, t in vals]
            return "(" + op.join(parts) + ")", "bool"
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Call):
            return self._call(node, expected_hint=expected_hint)
        raise self.error_cls(f"unsupported expression: {type(node).__name__}", node)

    def _binop(self, node: ast.BinOp) -> tuple[str, str]:
        # Optimize chained string + concatenation with NxStrBuilder!
        flat = _flatten_str_concat(node)
        if flat is not None:
            evaled = [self.expr(p) for p in flat]
            if all(t == "str" for _, t in evaled):
                self.require_caps({"memory.alloc"}, node)
                self.serial += 1
                sfx = self.serial
                appends = " ".join(f"nx_sb_{sfx}.append({c});" for c, _ in evaled)
                return f"([&]() {{ NxStrBuilder nx_sb_{sfx}; {appends} return nx_sb_{sfx}.finish(); }}())", "str"

        left, lt = self.expr(node.left)
        right, rt = self.expr(node.right)
        if isinstance(node.op, ast.Add) and lt == rt == "str":
            self.require_caps({"memory.alloc"}, node)
            return f"({left} + {right})", "str"
        if isinstance(node.op, ast.Mult) and lt == "str" and rt in INTEGERS:
            self.require_caps({"memory.alloc"}, node)
            return f"({left}).repeat(static_cast<std::int64_t>({right}))", "str"
        if isinstance(node.op, ast.Add) and list_element(lt) and lt == rt:
            self.require_caps({"memory.alloc"}, node)
            return f"nx_list_concat({left}, {right})", lt
        if isinstance(node.op, ast.Mult) and list_element(lt) and rt in INTEGERS:
            self.require_caps({"memory.alloc"}, node)
            return f"nx_list_repeat({left}, static_cast<std::int64_t>({right}))", lt

        overloaded = {ast.Add: "__add__", ast.Sub: "__sub__", ast.Mult: "__mul__", ast.Div: "__truediv__"}.get(type(node.op))
        if lt in self.compiler.classes:
            method = self._find_class_method(lt, overloaded or "")
            if method is not None:
                if len(method.params) != 2 or not compatible_types(rt, method.params[1][1]):
                    raise self.error_cls("operator argument type does not match class method", node)
                self.require_caps(method.caps, node)
                return f"({left}).nx_{overloaded}({coerce_cpp(right, rt, method.params[1][1], self.compiler.types)})", method.result

        if simd_parts(lt) and lt == rt:
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}.get(type(node.op))
            if op is None:
                raise self.error_cls("SIMD supports +, -, * between equal types", node)
            return f"({left} {op} {right})", lt

        if "dyn" in {lt, rt}:
            op_fn = {ast.Add: "nx_dyn_add", ast.Sub: "nx_dyn_sub", ast.Mult: "nx_dyn_mul", ast.Div: "nx_dyn_div", ast.Mod: "nx_dyn_mod"}.get(type(node.op))
            if op_fn is None or not compatible_types(lt, "dyn") or not compatible_types(rt, "dyn"):
                raise self.error_cls("unsupported dynamic binary operation", node)
            return f"{op_fn}({coerce_cpp(left, lt, 'dyn', self.compiler.types)}, {coerce_cpp(right, rt, 'dyn', self.compiler.types)})", "dyn"

        if not self.numeric(lt) or not self.numeric(rt):
            raise self.error_cls(f"arithmetic requires numeric operands (got {lt} and {rt})", node)
        if (lt in dict(self.fn.generic) or rt in dict(self.fn.generic)) and lt != rt:
            raise self.error_cls("generic arithmetic requires operands of the same type", node)

        if isinstance(node.op, ast.FloorDiv):
            if lt in INTEGERS and rt in INTEGERS:
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    raise self.error_cls("division by zero", node.right)
                typ = promote_numeric(lt, rt)
                return f"({coerce_cpp(left, lt, typ, self.compiler.types)} / {coerce_cpp(right, rt, typ, self.compiler.types)})", typ
            typ = promote_numeric(lt, rt)
            return f"std::floor({coerce_cpp(left, lt, typ, self.compiler.types)} / {coerce_cpp(right, rt, typ, self.compiler.types)})", typ

        if isinstance(node.op, ast.Pow):
            if lt in INTEGERS and rt in INTEGERS:
                typ = promote_numeric(lt, rt)
                if typ in {"i128", "u128"}:
                    return f"static_cast<{self._cpp_type(typ)}>(nx_pow_i128(static_cast<nx_i128>({left}), static_cast<nx_i128>({right})))", typ
                return f"static_cast<{self._cpp_type(typ)}>(std::pow(static_cast<double>({left}), static_cast<double>({right})))", typ
            return f"std::pow(static_cast<double>({left}), static_cast<double>({right}))", "f64"

        op_map = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%",
            ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
        }
        op = op_map.get(type(node.op))
        if op is None:
            raise self.error_cls("unsupported arithmetic operation", node)
        if op in {"%", "&", "|", "^", "<<", ">>"} and (lt not in INTEGERS or rt not in INTEGERS):
            raise self.error_cls(f"operator {op} requires integer operands", node)
        if op in {"/", "%"} and isinstance(node.right, ast.Constant) and node.right.value == 0:
            raise self.error_cls("division by zero", node.right)
        typ = lt if lt in dict(self.fn.generic) else promote_numeric(lt, rt)
        cl = coerce_cpp(left, lt, typ, self.compiler.types) if lt != typ else left
        cr = coerce_cpp(right, rt, typ, self.compiler.types) if rt != typ else right
        return f"({cl} {op} {cr})", typ

    def _compare(self, node: ast.Compare) -> tuple[str, str]:
        if len(node.ops) > 1 and any(not isinstance(item, (ast.Name, ast.Constant)) for item in node.comparators[:-1]):
            raise self.error_cls("chained comparison middle values must be names or constants", node)
        values = [self.expr(item) for item in [node.left, *node.comparators]]
        checks = []
        for op_node, (left, lt), (right, rt) in zip(node.ops, values, values[1:]):
            if isinstance(op_node, (ast.Is, ast.IsNot)) and ((optional_element(lt) and rt == "None") or (optional_element(rt) and lt == "None")):
                opt_expr = left if optional_element(lt) else right
                opt_typ = lt if optional_element(lt) else rt
                inner = optional_element(opt_typ)
                if inner and inner.startswith("owned["):
                    checks.append(f"({opt_expr} == nullptr)" if isinstance(op_node, ast.Is) else f"({opt_expr} != nullptr)")
                else:
                    checks.append(f"(!({opt_expr}).has_value())" if isinstance(op_node, ast.Is) else f"({opt_expr}).has_value()")
                continue
            if isinstance(op_node, (ast.Is, ast.IsNot)) and ("dyn" in {lt, rt}) and "None" in {lt, rt}:
                d_expr = left if lt == "dyn" else right
                checks.append(f"(({d_expr}).type == 0)" if isinstance(op_node, ast.Is) else f"(({d_expr}).type != 0)")
                continue
            if isinstance(op_node, (ast.In, ast.NotIn)):
                elem = list_element(rt) or set_element(rt) or span_element(rt)
                m_parts = dict_parts(rt)
                if m_parts:
                    elem = m_parts[0]
                if lt == rt == "str":
                    check = f"({right}).contains({left})"
                elif elem and compatible_types(lt, elem) and (set_element(rt) or m_parts):
                    check = f"(({right}).find({coerce_cpp(left, lt, elem, self.compiler.types)}) != ({right}).end())"
                elif elem and compatible_types(lt, elem) and list_element(rt):
                    check = f"nx_list_contains({right}, {coerce_cpp(left, lt, elem, self.compiler.types)})"
                elif elem and compatible_types(lt, elem) and span_element(rt):
                    check = f"nx_span_contains(({right}).ptr, ({right}).len, {coerce_cpp(left, lt, elem, self.compiler.types)})"
                else:
                    raise self.error_cls("in requires compatible str, List, Set, Dict or Span", node)
                checks.append(f"(!{check})" if isinstance(op_node, ast.NotIn) else check)
                continue
            op = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}.get(type(op_node))
            if op is None:
                raise self.error_cls("unsupported comparison operator", node)
            if "dyn" in {lt, rt}:
                dl = coerce_cpp(left, lt, "dyn", self.compiler.types)
                dr = coerce_cpp(right, rt, "dyn", self.compiler.types)
                if op in {"==", "!="}:
                    checks.append(f"{'!' if op == '!=' else ''}nx_dyn_eq({dl}, {dr})")
                else:
                    checks.append(f"(nx_dyn_cmp({dl}, {dr}) {op} 0)")
                continue
            if not (lt == rt or (self.numeric(lt) and self.numeric(rt))):
                raise self.error_cls(f"incompatible comparison between {lt} and {rt}", node)
            if lt in self.compiler.classes and op in {"==", "!="}:
                eq_m = self._find_class_method(lt, "__eq__")
                if eq_m:
                    if eq_m.result != "bool" or len(eq_m.params) != 2 or eq_m.params[1][1] != rt:
                        raise self.error_cls("__eq__ must accept the same class and return bool", node)
                    chk = f"({left}).nx___eq__({right})"
                    checks.append(f"(!{chk})" if op == "!=" else chk)
                    continue
            if self.numeric(lt) and self.numeric(rt) and lt != rt:
                common = promote_numeric(lt, rt)
                checks.append(f"({coerce_cpp(left, lt, common, self.compiler.types)} {op} {coerce_cpp(right, rt, common, self.compiler.types)})")
            else:
                checks.append(f"({left} {op} {right})")
        return "(" + " && ".join(checks) + ")", "bool"

    def _call(self, node: ast.Call, *, expected_hint: str | None = None) -> tuple[str, str]:
        # Handle postfix `?` error propagation rewritten as expr.__nx_propagate__()
        if isinstance(node.func, ast.Attribute) and node.func.attr == "__nx_propagate__" and not node.args and not node.keywords:
            if not self.fn.raises and not result_parts(self.fn.result):
                raise self.error_cls("`?` error propagation requires a `raises` or Result-returning function", node)
            res_expr, res_typ = self.expr(node.func.value)
            rparts = result_parts(res_typ)
            if rparts is None:
                raise self.error_cls("`?` requires a Result[T] expression", node)
            ok_t, err_t = rparts
            fn_rparts = result_parts(self.fn.result)
            fn_ok = fn_rparts[0] if fn_rparts else "void"
            self.serial += 1
            sfx = self.serial
            err_ret = f"nx_rt_err_void(nx_q_{sfx}.error())" if fn_ok == "void" else f"nx_rt_err<{self._cpp_type(fn_ok)}>(nx_q_{sfx}.error())"
            self.emit(f"auto nx_q_{sfx} = {res_expr};")
            self.emit(f"if (nx_q_{sfx}.is_err()) return {err_ret};")
            if ok_t == "void":
                return "(void)0", "void"
            return f"nx_q_{sfx}.value()", ok_t

        # super().__init__(...)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "__init__"
                and isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "super"):
            if not self.in_method or not self.fn.owner:
                raise self.error_cls("super() is only valid inside a class method", node)
            cls = self.compiler.classes[self.fn.owner]
            if not cls.base:
                raise self.error_cls(f"class {self.fn.owner} has no base class", node)
            base_cls = self.compiler.classes[cls.base]
            init_m = base_cls.methods.get("__init__")
            expected = init_m.params[1:] if init_m else base_cls.fields
            if len(node.args) != len(expected):
                raise self.error_cls(f"super().__init__ expects {len(expected)} arguments", node)
            args = [self.argument(a, t) for a, (_, t) in zip(node.args, expected)]
            if init_m:
                return f"nx_{cls.base}::operator=(nx_{cls.base}({', '.join(args)}))", "void"
            return f"nx_{cls.base}::operator=(nx_{cls.base}{{{', '.join(args)}}})", "void"

        if isinstance(node.func, ast.Name) and node.func.id == "value_or" and len(node.args) == 2 and not node.keywords:
            opt, typ = self.expr(node.args[0])
            inner = optional_element(typ)
            if not inner:
                raise self.error_cls("value_or expects Optional[T]", node.args[0])
            fb = self.argument(node.args[1], inner)
            if inner.startswith("owned["):
                return f"({opt} ? std::move({opt}) : {fb})", inner
            return f"({opt}).value_or({fb})", inner

        # Generic type constructors: List[T](), Dict[K,V](), Set[T](), Tuple[...](...), Buffer[T](n), Channel[T](cap), Mutex[T](val), GpuBuffer[T](n), SIMD[T,N](...), Class[T](...)
        if isinstance(node.func, ast.Subscript) and not node.keywords:
            typ = resolve_type_node(node.func, types=self.compiler.types, classes=set(self.compiler.classes), enums=set(self.compiler.enums), error_cls=self.error_cls)
            self.compiler.ensure_type(typ, node)
            if set_element(typ) or dict_parts(typ):
                self.require_caps({"memory.alloc"}, node)
                if not node.args:
                    return f"{self._cpp_type(typ)}{{}}", typ
                if len(node.args) == 1:
                    v, act = self.expr(node.args[0])
                    if act == typ:
                        return v, typ
                raise self.error_cls(f"{typ} expects no arguments or a matching literal", node)
            if tuple_parts(typ):
                parts = tuple_parts(typ)
                assert parts is not None
                if len(node.args) != len(parts):
                    raise self.error_cls(f"{typ} expects {len(parts)} values", node)
                vals = [self.argument(it, p) for it, p in zip(node.args, parts)]
                return f"{self._cpp_type(typ)}{{{', '.join(vals)}}}", typ
            le = list_element(typ)
            if le:
                self.require_caps({"memory.alloc"}, node)
                if len(node.args) == 0:
                    return f"{self._cpp_type(typ)}{{}}", typ
                if len(node.args) == 1 and isinstance(node.args[0], ast.List):
                    vals = [self.argument(it, le) for it in node.args[0].elts]
                    return f"{self._cpp_type(typ)}{{{', '.join(vals)}}}", typ
                raise self.error_cls("List[T] expects no arguments or one list literal", node)
            if typ.startswith("Buffer["):
                self.require_caps({"memory.alloc"}, node)
                if len(node.args) != 1:
                    raise self.error_cls("Buffer[T] expects one length", node)
                length, act = self.expr(node.args[0])
                if act not in INTEGERS:
                    raise self.error_cls("Buffer length must be an integer", node.args[0])
                return f"std::make_unique<{self._cpp_type(typ)}>(static_cast<std::int64_t>({length}))", f"owned[{typ}]"
            ce = channel_element(typ)
            if ce:
                self.require_caps({"memory.alloc"}, node)
                cap = self.argument(node.args[0], "i64") if node.args else "0"
                return f"nx_Channel<{self._cpp_type(ce)}>({cap})", typ
            me = mutex_element(typ)
            if me:
                if len(node.args) != 1:
                    raise self.error_cls("Mutex[T] expects 1 initial value", node)
                val = self.argument(node.args[0], me)
                return f"nx_Mutex<{self._cpp_type(me)}>({val})", typ
            ge = gpu_buffer_element(typ)
            if ge:
                self.require_caps({"gpu.compute", "memory.alloc"}, node)
                if len(node.args) != 1:
                    raise self.error_cls("GpuBuffer[T] expects 1 length", node)
                length = self.argument(node.args[0], "i64")
                return f"nx_GpuBuffer<{self._cpp_type(ge)}>({length})", typ
            sp = simd_parts(typ)
            if sp:
                el, width = sp
                if not width.isdigit():
                    raise self.error_cls("SIMD constructor requires a concrete width", node)
                if len(node.args) != int(width):
                    raise self.error_cls(f"{typ} expects {width} lane values", node)
                args = [self.argument(arg, el) for arg in node.args]
                return f"{self._cpp_type(typ)}{{{{{', '.join(args)}}}}}", typ
            # Generic class instantiation e.g. Box[i32](10)
            base_cls = typ.split("[")[0]
            if base_cls in self.compiler.classes:
                cls = self.compiler.classes[base_cls]
                init_m = cls.methods.get("__init__")
                args = [self.expr(a)[0] for a in node.args]
                braces = "(" if init_m else "{"
                close = ")" if init_m else "}"
                return f"{self._cpp_type(typ)}{braces}{', '.join(args)}{close}", typ
            raise self.error_cls("unsupported generic type constructor", node)

        # Module-qualified calls: nx.std.fn(...), mod.fn(...), py_mod.fn(...), cpp_hdr.fn(...), Enum.Variant(...)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Name):
            # Support `import nx.debug` -> `nx.debug.live_blocks()` or `import nx.fs` -> `nx.fs.read_text(...)`
            pkg_root = node.func.value.value.id
            sub_mod = node.func.value.attr
            attr = node.func.attr
            full_std = f"{pkg_root}.{sub_mod}"
            if full_std in STD_MODULES and (full_std in {imp.target_name for imp in self.unit.imports.values() if imp.kind == "nx_std"}):
                return self._call_stdlib(full_std, attr, node)
            if pkg_root == "py" and sub_mod in {imp.target_name for imp in self.unit.imports.values() if imp.kind == "py"}:
                return self._call_python_direct(sub_mod, attr, node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            head = node.func.value.id
            attr = node.func.attr
            if head in self.unit.enums:
                enum_def = self.unit.enums[head]
                if attr not in enum_def.variants:
                    raise self.error_cls(f"unknown enum variant: {head}.{attr}", node)
                v_params = enum_def.variants[attr]
                if len(node.args) != len(v_params):
                    raise self.error_cls(f"{head}.{attr} expects {len(v_params)} arguments", node)
                args = [self.argument(a, pt) for a, (_, pt) in zip(node.args, v_params)]
                return f"{self.unit.cpp_symbol(head)}::{attr}({', '.join(args)})", head
            if head in self.unit.imports:
                imp = self.unit.imports[head]
                if imp.kind == "nx_std":
                    return self._call_stdlib(imp.target_name, attr, node)
                if imp.kind == "py":
                    return self._call_python_direct(imp.target_name, attr, node)
                if imp.kind == "cpp":
                    return self._call_cpp_direct(imp.target_name, attr, node, expected_hint=expected_hint)
                if imp.kind in {"c", "rust", "go"}:
                    return self._call_native_lib(imp.kind, imp.target_name, attr, node, expected_hint=expected_hint)
                if imp.kind == "nx":
                    tgt = imp.target_unit
                    assert tgt is not None
                    if attr in tgt.classes:
                        return self._construct_class(tgt, attr, node)
                    if attr in tgt.functions:
                        return self._call_nx_function(tgt, attr, node)
                    raise self.error_cls(f"module {head} has no member {attr}", node)

        # Method calls on values / objects
        if isinstance(node.func, ast.Attribute):
            return self._call_method(node, expected_hint=expected_hint)

        if not isinstance(node.func, ast.Name):
            raise self.error_cls("only named functions or methods are callable", node)

        name = node.func.id
        # Check imported symbols (`from mod import func` / `from py.math import sqrt` / `from cpp.cmath import hypot` / `from nx.fs import read_text`)
        if name in self.unit.imported_symbols:
            kind, target, orig_name = self.unit.imported_symbols[name]
            if kind == "nx_std":
                return self._call_stdlib(target, orig_name, node)
            if kind == "py":
                return self._call_python_direct(target, orig_name, node)
            if kind == "cpp":
                return self._call_cpp_direct(target, orig_name, node, expected_hint=expected_hint)
            if kind in {"c", "rust", "go"}:
                return self._call_native_lib(kind, target, orig_name, node, expected_hint=expected_hint)
            if kind == "nx":
                if orig_name in target.classes:
                    return self._construct_class(target, orig_name, node)
                return self._call_nx_function(target, orig_name, node)

        # Builtin classes / StringBuilder / constructors
        if name == "StringBuilder":
            self.require_caps({"memory.alloc"}, node)
            if not node.args:
                return "NxStrBuilder()", "StringBuilder"
            if len(node.args) == 1:
                cap = self.argument(node.args[0], "i64")
                return f"NxStrBuilder(static_cast<std::size_t>({cap}))", "StringBuilder"
            raise self.error_cls("StringBuilder takes 0 or 1 capacity argument", node)

        if name in self.unit.classes:
            return self._construct_class(self.unit, name, node)

        if name == "owned":
            self.require_caps({"memory.alloc"}, node)
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Call) or not isinstance(node.args[0].func, ast.Name):
                raise self.error_cls("owned expects a class constructor call", node)
            cname = node.args[0].func.id
            if cname not in self.compiler.classes and cname not in self.unit.imported_symbols:
                raise self.error_cls("owned expects a class constructor", node)
            val, typ = self.expr(node.args[0])
            self.compiler.ensure_type(f"owned[{typ}]", node)
            return f"std::unique_ptr<{self._cpp_type(typ)}>(new {val})", f"owned[{typ}]"

        if name == "move":
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                raise self.error_cls("move expects one owned variable", node)
            val, typ = self.expr(node.args[0])
            if not self._is_move_only(typ):
                raise self.error_cls("move requires an owned/move-only value", node)
            self.moved.add(node.args[0].id)
            return f"std::move({val})", typ

        if name in {"buffer_span", "buffer_mutspan"}:
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                raise self.error_cls(f"{name} requires a named Buffer owner", node)
            val, typ = self.expr(node.args[0])
            parts = buffer_parts(typ)
            if not parts or (name == "buffer_mutspan" and parts[0] == "borrowed"):
                raise self.error_cls(f"{name} requires a compatible owned or borrowed Buffer", node)
            mode, elem = parts
            op = "->" if mode == "owned" else "."
            view = "MutSpan" if name == "buffer_mutspan" else "Span"
            return f"{self._cpp_type(f'{view}[{elem}]')}{{({val}){op}data(), ({val}){op}size()}}", f"{view}[{elem}]"

        if name in {"is_ok", "is_err", "unwrap", "error", "unwrap_or"}:
            if name == "unwrap_or":
                if len(node.args) != 2:
                    raise self.error_cls("unwrap_or expects (Result/Optional, default)", node)
                v, typ = self.expr(node.args[0])
                rp = result_parts(typ)
                op = optional_element(typ)
                if rp:
                    fb = self.argument(node.args[1], rp[0])
                    return f"({v}).value_or({fb})", rp[0]
                if op:
                    fb = self.argument(node.args[1], op)
                    return f"({v}).value_or({fb})", op
                raise self.error_cls("unwrap_or expects Result or Optional", node.args[0])
            if len(node.args) != 1:
                raise self.error_cls(f"{name} expects one Result or Optional", node)
            val, typ = self.expr(node.args[0])
            rp = result_parts(typ)
            op = optional_element(typ)
            if op and name in {"is_ok", "unwrap"}:
                if name == "is_ok":
                    return f"({val} != nullptr)" if op.startswith("owned[") else f"({val}).has_value()", "bool"
                return (f"std::move({val})" if op.startswith("owned[") else f"({val}).value()"), op
            if rp is None:
                raise self.error_cls(f"{name} requires Result[T]", node.args[0])
            if name == "is_ok":
                return f"({val}).ok()", "bool"
            if name == "is_err":
                return f"({val}).is_err()", "bool"
            if name == "error":
                return f"({val}).error()", "str" if rp[1] is None else rp[1]
            if rp[0] == "void":
                return f"({val}).value()", "void"
            return f"({val}).value()", rp[0]

        if name in {"type_name", "as_i64", "as_i128", "as_f64", "as_str", "as_bool", "is_none"}:
            if len(node.args) != 1:
                raise self.error_cls(f"{name} expects one dyn value", node)
            val, typ = self.expr(node.args[0])
            if typ != "dyn":
                if name == "type_name":
                    return nx_str_literal(typ), "str"
                raise self.error_cls(f"{name} requires dyn", node.args[0])
            if name == "is_none":
                return f"(({val}).type == 0)", "bool"
            ret_map = {"type_name": "str", "as_i64": "i64", "as_i128": "i128", "as_f64": "f64", "as_str": "str", "as_bool": "bool"}
            return f"nx_{name}({val})", ret_map[name]

        if name == "input":
            self.require_caps({"memory.alloc"}, node)
            if len(node.args) > 1 or node.keywords:
                raise self.error_cls("input() expects 0 or 1 positional prompt argument", node)
            if not node.args:
                return "nx_builtin_input()", "str"
            prompt_cpp = self.argument(node.args[0], "str")
            return f"nx_builtin_input({prompt_cpp})", "str"

        if name == "ring_f64":
            self.require_caps({"memory.alloc"}, node)
            if len(node.args) != 1:
                raise self.error_cls("ring_f64 expects capacity", node)
            size, typ = self.expr(node.args[0])
            if typ not in INTEGERS:
                raise self.error_cls("ring capacity must be integer", node.args[0])
            return f"nx_ring_new_f64(static_cast<std::int64_t>({size}))", "Ring[f64]"
        if name in {"ring_write", "ring_read", "ring_valid"}:
            count = 3 if name == "ring_write" else 2
            if len(node.args) != count:
                raise self.error_cls(f"{name} expects {count} arguments", node)
            ring, rt = self.expr(node.args[0])
            idx, it = self.expr(node.args[1])
            if rt != "Ring[f64]" or it not in INTEGERS:
                raise self.error_cls(f"{name} expects Ring[f64] and integer index", node)
            if name == "ring_write":
                v = self.argument(node.args[2], "f64")
                return f"({ring})->write(static_cast<std::int64_t>({idx}), {v})", "void"
            if name == "ring_read":
                return f"({ring})->read(static_cast<std::int64_t>({idx}))", "f64"
            return f"({ring})->valid(static_cast<std::int64_t>({idx}))", "bool"

        # Numeric & scalar type conversions: i8..i128, u8..u128, f32, f64, int, float, bool, str
        canon = canonical_type_name(name)
        if canon in NUMERICS:
            if len(node.args) != 1:
                raise self.error_cls(f"{name} expects one argument", node)
            val, act = self.expr(node.args[0])
            if act == "str":
                if canon == "i128":
                    return f"nx_parse_i128_raw(({val}).c_str())", "i128"
                if canon == "u128":
                    return f"nx_parse_u128_raw(({val}).c_str())", "u128"
                if canon in INTEGERS:
                    return f"nx_num_cast<{self._cpp_type(canon)}>(nx_parse_i128_raw(({val}).c_str()))", canon
                return f"static_cast<{self._cpp_type(canon)}>(std::stod(std::string(({val}).view())))", canon
            if act == "dyn":
                return f"nx_unbox<{self._cpp_type(canon)}>({val})", canon
            if act == "bool":
                return f"static_cast<{self._cpp_type(canon)}>({val} ? 1 : 0)", canon
            if not self.numeric(act):
                raise self.error_cls(f"{name}() requires a numeric, bool, str or dyn argument", node.args[0])
            return f"nx_num_cast<{self._cpp_type(canon)}>({val})", canon
        if name == "str":
            if len(node.args) != 1:
                raise self.error_cls("str() expects one argument", node)
            val, act = self.expr(node.args[0])
            if act == "str":
                return val, "str"
            self.require_caps({"memory.alloc"}, node)
            return f"NxStr::from_std(nx_to_std_string({val}))", "str"
        if name == "bool":
            if len(node.args) != 1:
                raise self.error_cls("bool() expects one argument", node)
            val, act = self.expr(node.args[0])
            if act == "bool":
                return val, "bool"
            if act == "dyn":
                return f"nx_as_bool({val})", "bool"
            if self.numeric(act):
                return f"({val} != 0)", "bool"
            if act == "str":
                return f"(!({val}).empty())", "bool"
            if list_element(act) or dict_parts(act) or set_element(act):
                return f"(!({val}).empty())", "bool"
            raise self.error_cls(f"cannot convert {act} to bool", node)

        # Builtin sequence functions: len, sum, min, max, abs, sorted, any, all
        if name == "len":
            if len(node.args) != 1:
                raise self.error_cls("len expects one argument", node)
            val, typ = self.expr(node.args[0])
            if typ == "dyn":
                return f"nx_dyn_len({val})", "i64"
            if list_element(typ) or set_element(typ) or dict_parts(typ) or typ == "str" or typ == "StringBuilder":
                return f"static_cast<std::int64_t>(({val}).size())", "i64"
            tp = tuple_parts(typ)
            if tp:
                return str(len(tp)), "i64"
            buf = buffer_parts(typ)
            if buf:
                op = "->" if buf[0] == "owned" else "."
                return f"static_cast<std::int64_t>(({val}){op}size())", "i64"
            sp = simd_parts(typ)
            if sp:
                w = sp[1]
                return (self.compiler.types[f"$const:{w}"] if not w.isdigit() else w), "i64"
            if span_element(typ):
                return f"static_cast<std::int64_t>(({val}).len)", "i64"
            if typ in self.compiler.classes and self._find_class_method(typ, "__len__"):
                m = self._find_class_method(typ, "__len__")
                assert m is not None
                return f"static_cast<std::int64_t>(({val}).nx___len__())", "i64"
            raise self.error_cls(f"object of type {typ} has no len()", node.args[0])

        if name == "sum":
            if len(node.args) != 1:
                raise self.error_cls("sum expects one List or Span", node)
            val, typ = self.expr(node.args[0])
            el = list_element(typ)
            if el and el in NUMERICS:
                return f"nx_builtin_sum({val})", el
            sel = span_element(typ)
            if sel:
                return f"nx_builtin_sum_span({val})", sel
            raise self.error_cls("sum requires numeric List or Span", node.args[0])
        if name in {"min", "max"}:
            if len(node.args) == 1:
                val, typ = self.expr(node.args[0])
                el = list_element(typ)
                if el and el in NUMERICS:
                    return f"nx_builtin_{name}({val})", el
                raise self.error_cls(f"{name} requires numeric List or two arguments", node)
            if len(node.args) == 2:
                a, at = self.expr(node.args[0])
                b, bt = self.expr(node.args[1])
                if at in NUMERICS and bt in NUMERICS:
                    rt = promote_numeric(at, bt)
                    fn_c = "std::min" if name == "min" else "std::max"
                    return f"{fn_c}({coerce_cpp(a, at, rt, self.compiler.types)}, {coerce_cpp(b, bt, rt, self.compiler.types)})", rt
            raise self.error_cls(f"{name} expects 1 numeric List or 2 numbers", node)
        if name == "abs":
            if len(node.args) != 1:
                raise self.error_cls("abs expects one numeric argument", node)
            val, typ = self.expr(node.args[0])
            if not self.numeric(typ):
                raise self.error_cls("abs requires a numeric value", node)
            return f"nx_builtin_abs({val})", typ
        if name == "sorted":
            self.require_caps({"memory.alloc"}, node)
            if len(node.args) != 1:
                raise self.error_cls("sorted expects one List", node)
            val, typ = self.expr(node.args[0])
            if not list_element(typ):
                raise self.error_cls("sorted requires a List", node)
            return f"nx_list_sorted({val})", typ
        if name in {"any", "all"}:
            if len(node.args) != 1:
                raise self.error_cls(f"{name} expects List[bool]", node)
            val = self.argument(node.args[0], "List[bool]")
            return f"nx_builtin_{name}({val})", "bool"

        if name in {"zeros_f32", "zeros_f64", "zeros_i32", "zeros_i64"}:
            self.require_caps({"memory.alloc"}, node)
            if len(node.args) != 1:
                raise self.error_cls(f"{name} expects one length", node)
            val, typ = self.expr(node.args[0])
            if typ not in INTEGERS:
                raise self.error_cls("array length must be an integer", node.args[0])
            elem = name.removeprefix("zeros_")
            rt = f"List[{elem}]"
            self.compiler.ensure_type(rt, node)
            return f"nx_{name}(static_cast<std::int64_t>({val}))", rt

        if name == "read_text":
            self.require_caps({"filesystem.read", "memory.alloc"}, node)
            if len(node.args) != 1:
                raise self.error_cls("read_text expects one path", node)
            val = self.argument(node.args[0], "str")
            return f"nx_read_text({val})", "str"
        if name == "open":
            self.require_caps({"filesystem.read"}, node)
            if len(node.args) not in {1, 2}:
                raise self.error_cls("open expects path and optional mode", node)
            path_e = self.argument(node.args[0], "str")
            mode_e = self.argument(node.args[1], "str") if len(node.args) == 2 else nx_str_literal("r")
            if len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                if any(ch in node.args[1].value for ch in ("w", "a", "+")):
                    self.require_caps({"filesystem.write"}, node)
            return f"nx_File::open({path_e}, {mode_e})", "File"
        if name == "tcp_connect":
            self.require_caps({"network.connect"}, node)
            if len(node.args) != 2:
                raise self.error_cls("tcp_connect expects host and port", node)
            host = self.argument(node.args[0], "str")
            port = self.argument(node.args[1], "i32")
            return f"nx_tcp_connect({host}, {port})", "bool"

        if name in self.unit.functions:
            return self._call_nx_function(self.unit, name, node)

        raise self.error_cls(f"unknown function: {name}", node)

    def _construct_class(self, target_unit: "ModuleUnit", name: str, node: ast.Call) -> tuple[str, str]:
        cls = target_unit.classes[name]
        if node.keywords:
            raise self.error_cls("class constructors currently take positional arguments", node)
        init_m = cls.methods.get("__init__")
        if init_m:
            expected = init_m.params[1:]
            self.require_caps(init_m.caps, node)
            if len(node.args) != len(expected):
                raise self.error_cls(f"{name} expects {len(expected)} constructor arguments", node)
            args = [self.argument(a, t) for a, (_, t) in zip(node.args, expected)]
            return f"{target_unit.cpp_symbol(name)}({', '.join(args)})", cls.qualified_name
        # Default aggregate construction supporting field defaults
        all_fields = self.compiler.all_class_fields(cls.qualified_name)
        req_count = sum(1 for _, _, has_def in all_fields if not has_def)
        if not (req_count <= len(node.args) <= len(all_fields)):
            raise self.error_cls(f"{name} expects {req_count} to {len(all_fields)} constructor arguments", node)
        args = [self.argument(a, ft) for a, (_, ft, _) in zip(node.args, all_fields)]
        needs_paren = cls.base is not None or bool(cls.field_defaults) or any(b.base == cls.qualified_name for b in self.compiler.classes.values())
        if len(args) == len(all_fields) and not needs_paren:
            return f"{target_unit.cpp_symbol(name)}{{{', '.join(args)}}}", cls.qualified_name
        if len(args) == len(all_fields) and needs_paren:
            return f"{target_unit.cpp_symbol(name)}({', '.join(args)})", cls.qualified_name
        # Use helper lambda/ctor when some defaulted fields are omitted
        self.serial += 1
        sfx = self.serial
        assigns = " ".join(f"nx_obj_{sfx}.nx_{fn} = {a};" for a, (fn, _, _) in zip(args, all_fields))
        return f"([&]() {{ {target_unit.cpp_symbol(name)} nx_obj_{sfx}; {assigns} return nx_obj_{sfx}; }}())", cls.qualified_name

    def _call_nx_function(self, target_unit: "ModuleUnit", name: str, node: ast.Call) -> tuple[str, str]:
        fn = target_unit.functions[name]
        if not getattr(target_unit, "is_trusted", False):
            self.require_caps(fn.caps, node)
        if len(node.args) > len(fn.params):
            raise self.error_cls(f"{name} expects at most {len(fn.params)} arguments", node)
        bound: list[ast.expr | None] = [*node.args] + [None] * (len(fn.params) - len(node.args))
        names = [p for p, _ in fn.params]
        for kw in node.keywords:
            if kw.arg is None or kw.arg not in names:
                raise self.error_cls(f"unknown keyword argument for {name}", kw)
            pos = names.index(kw.arg)
            if bound[pos] is not None:
                raise self.error_cls(f"duplicate argument: {kw.arg}", kw)
            bound[pos] = kw.value
        for i, (it, default) in enumerate(zip(bound, fn.defaults)):
            if it is None:
                bound[i] = default
        if any(it is None for it in bound):
            raise self.error_cls(f"missing required argument for {name}", node)
        call_args = [it for it in bound if it is not None]
        self.check_borrow_aliases(call_args, fn.params)
        cpp_sym = f"nx_{name}" if (target_unit is self.compiler.entry_unit or fn.exported or fn.extern_symbol) else target_unit.cpp_symbol(name)

        if fn.generic:
            inferred: dict[str, str] = {}
            args = []
            bounds = dict(fn.generic)
            for arg, (_, expected) in zip(call_args, fn.params):
                if expected in bounds:
                    value, actual = self.expr(arg)
                    bound = bounds[expected]
                    if bound in self.compiler.traits:
                        cls_def = self.compiler.classes.get(actual)
                        required = self.compiler.traits[bound]
                        if cls_def is None or any(
                            self._find_class_method(actual, m) is None
                            or self._find_class_method(actual, m).result != (actual if res == "Self" else res)  # type: ignore[union-attr]
                            or tuple(t for _, t in self._find_class_method(actual, m).params[1:]) != tuple(actual if t == "Self" else t for t in ps)  # type: ignore[union-attr]
                            for m, (ps, res) in required.items()
                        ):
                            raise self.error_cls(f"{actual} does not implement trait {bound}", arg)
                    elif not self.numeric(actual) or (bound == "Integer" and actual not in INTEGERS and dict(self.fn.generic).get(actual) != "Integer"):
                        raise self.error_cls(f"{name} requires {bound} for {expected}, got {actual}", arg)
                    if expected in inferred and inferred[expected] != actual:
                        raise self.error_cls(f"generic {expected} must have one concrete type", arg)
                    inferred[expected] = actual
                    args.append(value)
                elif simd_parts(expected) and any(p in expected for p in bounds):
                    value, actual = self.expr(arg)
                    ep = simd_parts(expected)
                    ap = simd_parts(actual)
                    if ap is None or ep is None:
                        raise self.error_cls(f"expected {expected}, got {actual}", arg)
                    ee, ew = ep
                    ae, aw = ap
                    if ee in bounds:
                        if bounds[ee] not in {"Number", "Integer"} or not self.numeric(ae):
                            raise self.error_cls(f"SIMD generic {ee} requires a numeric type", arg)
                        if ee in inferred and inferred[ee] != ae:
                            raise self.error_cls(f"generic {ee} must have one concrete type", arg)
                        inferred[ee] = ae
                    elif ee != ae:
                        raise self.error_cls(f"expected {expected}, got {actual}", arg)
                    if ew in bounds and bounds[ew] == "Int":
                        if ew in inferred and inferred[ew] != aw:
                            raise self.error_cls(f"generic {ew} must have one constant value", arg)
                        inferred[ew] = aw
                    elif ew != aw:
                        raise self.error_cls(f"expected {expected}, got {actual}", arg)
                    args.append(value)
                else:
                    args.append(self.argument(arg, expected))
            if any(p not in inferred for p, _ in fn.generic):
                raise self.error_cls(f"cannot infer generic type for {name}", node)
            result = inferred.get(fn.result, fn.result)
            parts = simd_parts(result)
            if parts:
                result = f"SIMD[{inferred.get(parts[0], parts[0])},{inferred.get(parts[1], parts[1])}]"
                self.compiler.ensure_type(result, node)
            return f"{cpp_sym}({', '.join(args)})", result

        args = [self.argument(a, exp) for a, (_, exp) in zip(call_args, fn.params)]
        return f"{cpp_sym}({', '.join(args)})", fn.result

    def _call_method(self, node: ast.Call, *, expected_hint: str | None = None) -> tuple[str, str]:
        assert isinstance(node.func, ast.Attribute)
        receiver, rtype = self.expr(node.func.value)
        method = node.func.attr

        # Dynamic / Python object method call!
        if rtype == "dyn":
            if method == "append" and len(node.args) == 1 and not node.keywords:
                v, vt = self.expr(node.args[0])
                return f"nx_dyn_append({receiver}, {coerce_cpp(v, vt, 'dyn', self.compiler.types)})", "void"
            self.require_caps({"python.interop"}, node)
            self.compiler.uses_python = True
            return self._emit_py_abi_call(f"nx_py_dyn_method({receiver}, {c_string(method)}", node)

        # StringBuilder methods
        if rtype == "StringBuilder":
            if method in {"append", "append_line"} and len(node.args) == 1:
                v, vt = self.expr(node.args[0])
                if vt == "void":
                    raise self.error_cls("cannot append void to StringBuilder", node)
                return f"({receiver}).{method}({v})", "void"
            if method == "reserve" and len(node.args) == 1:
                return f"({receiver}).reserve({self.argument(node.args[0], 'i64')})", "void"
            if method == "clear" and not node.args:
                return f"({receiver}).clear()", "void"
            if method in {"to_str", "build"} and not node.args:
                return f"({receiver}).to_str()", "str"
            raise self.error_cls(f"unknown StringBuilder method: {method}", node)

        # File / TcpStream / TcpListener / HttpResponse methods
        if rtype in {"File", "TcpStream", "TcpListener", "HttpResponse", "FileInfo"}:
            if rtype == "File":
                if method == "read" and not node.args:
                    self.require_caps({"memory.alloc"}, node)
                    return f"({receiver}).nx_read()", "str"
                if method == "read_line" and not node.args:
                    self.require_caps({"memory.alloc"}, node)
                    return f"({receiver}).nx_read_line()", "str"
                if method == "write" and len(node.args) == 1:
                    self.require_caps({"filesystem.write"}, node)
                    return f"({receiver}).nx_write({self.argument(node.args[0], 'str')})", "i64"
                if method in {"flush", "close"} and not node.args:
                    return f"({receiver}).nx_{method}()", "void"
            if rtype == "TcpStream":
                if method == "send" and len(node.args) == 1:
                    return f"({receiver}).nx_send({self.argument(node.args[0], 'str')})", "i64"
                if method == "recv" and len(node.args) <= 1:
                    self.require_caps({"memory.alloc"}, node)
                    mx = self.argument(node.args[0], "i64") if node.args else "4096"
                    return f"({receiver}).nx_recv({mx})", "str"
                if method == "recv_all" and not node.args:
                    self.require_caps({"memory.alloc"}, node)
                    return f"({receiver}).nx_recv_all()", "str"
                if method == "set_timeout" and len(node.args) == 1:
                    return f"({receiver}).nx_set_timeout({self.argument(node.args[0], 'i32')})", "void"
                if method == "peer_addr" and not node.args:
                    return f"({receiver}).nx_peer_addr()", "str"
                if method == "close" and not node.args:
                    return f"({receiver}).nx_close()", "void"
            if rtype == "TcpListener":
                if method == "accept" and not node.args:
                    return f"({receiver}).nx_accept()", "Result[TcpStream]"
                if method == "local_port" and not node.args:
                    return f"({receiver}).nx_local_port()", "i32"
                if method == "close" and not node.args:
                    return f"({receiver}).nx_close()", "void"
            if rtype == "HttpResponse" and method == "ok" and not node.args:
                return f"({receiver}).nx_ok_status()", "bool"
            raise self.error_cls(f"unknown method {rtype}.{method}", node)

        # Future[T] / Channel[T] / Mutex[T] / AtomicI64 / ThreadPool methods
        if future_element(rtype) is not None:
            fe = future_element(rtype)
            assert fe is not None
            if method == "get" and not node.args:
                return f"({receiver}).get()", fe
            if method == "is_done" and not node.args:
                return f"({receiver}).is_done()", "bool"
            if method == "try_result" and not node.args:
                rt = f"Result[{fe}]"
                self.compiler.ensure_type(rt, node)
                return f"({receiver}).try_result()", rt
        if channel_element(rtype) is not None:
            ce = channel_element(rtype)
            assert ce is not None
            if method == "send" and len(node.args) == 1:
                return f"({receiver}).nx_send({self.argument(node.args[0], ce)})", "bool"
            if method in {"recv", "try_recv"} and not node.args:
                ot = f"Optional[{ce}]"
                self.compiler.ensure_type(ot, node)
                return f"({receiver}).nx_{method}()", ot
            if method == "close" and not node.args:
                return f"({receiver}).nx_close()", "void"
        if mutex_element(rtype) is not None:
            me = mutex_element(rtype)
            assert me is not None
            if method == "get" and not node.args:
                return f"({receiver}).nx_get()", me
            if method == "set" and len(node.args) == 1:
                return f"({receiver}).nx_set({self.argument(node.args[0], me)})", "void"
        if rtype == "AtomicI64":
            if method == "load" and not node.args:
                return f"({receiver}).nx_load()", "i64"
            if method == "store" and len(node.args) == 1:
                return f"({receiver}).nx_store({self.argument(node.args[0], 'i64')})", "void"
            if method in {"add", "sub"} and len(node.args) == 1:
                return f"({receiver}).nx_{method}({self.argument(node.args[0], 'i64')})", "i64"
        if rtype == "ThreadPool":
            if method == "workers" and not node.args:
                return f"({receiver}).nx_workers()", "i64"

        # Result / Optional methods
        if result_parts(rtype):
            rp = result_parts(rtype)
            assert rp is not None
            if method == "is_ok" and not node.args:
                return f"({receiver}).ok()", "bool"
            if method == "is_err" and not node.args:
                return f"({receiver}).is_err()", "bool"
            if method == "unwrap" and not node.args:
                return f"({receiver}).value()", rp[0]
            if method == "unwrap_or" and len(node.args) == 1:
                return f"({receiver}).value_or({self.argument(node.args[0], rp[0])})", rp[0]
            if method == "error" and not node.args:
                return f"({receiver}).error()", rp[1] or "str"
        if optional_element(rtype):
            oe = optional_element(rtype)
            assert oe is not None
            if method == "is_some" and not node.args:
                return (f"({receiver} != nullptr)" if oe.startswith("owned[") else f"({receiver}).has_value()"), "bool"
            if method == "is_none" and not node.args:
                return (f"({receiver} == nullptr)" if oe.startswith("owned[") else f"(!({receiver}).has_value())"), "bool"
            if method == "unwrap" and not node.args:
                return (f"std::move({receiver})" if oe.startswith("owned[") else f"({receiver}).value()"), oe
            if method == "unwrap_or" and len(node.args) == 1:
                fb = self.argument(node.args[0], oe)
                return (f"({receiver} ? std::move({receiver}) : {fb})" if oe.startswith("owned[") else f"({receiver}).value_or({fb})"), oe

        # List[T] methods
        le = list_element(rtype)
        if le:
            if method in {"append", "push"} and len(node.args) == 1:
                self.require_caps({"memory.alloc"}, node)
                return f"({receiver}).push_back({self.argument(node.args[0], le)})", "void"
            if method == "extend" and len(node.args) == 1:
                self.require_caps({"memory.alloc"}, node)
                return f"nx_list_extend({receiver}, {self.argument(node.args[0], rtype)})", "void"
            if method == "insert" and len(node.args) == 2:
                self.require_caps({"memory.alloc"}, node)
                return f"nx_list_insert({receiver}, {self.argument(node.args[0], 'i64')}, {self.argument(node.args[1], le)})", "void"
            if method == "pop" and len(node.args) <= 1:
                idx = self.argument(node.args[0], "i64") if node.args else "-1"
                return f"nx_list_pop({receiver}, {idx})", le
            if method == "remove" and len(node.args) == 1:
                return f"nx_list_remove({receiver}, {self.argument(node.args[0], le)})", "void"
            if method == "index" and len(node.args) == 1:
                return f"nx_list_index({receiver}, {self.argument(node.args[0], le)})", "i64"
            if method == "count" and len(node.args) == 1:
                return f"nx_list_count({receiver}, {self.argument(node.args[0], le)})", "i64"
            if method == "reverse" and not node.args:
                return f"nx_list_reverse({receiver})", "void"
            if method == "clear" and not node.args:
                return f"({receiver}).clear()", "void"
            if method == "copy" and not node.args:
                self.require_caps({"memory.alloc"}, node)
                return f"{self._cpp_type(rtype)}({receiver})", rtype
            if method == "sort" and not node.args:
                return f"nx_list_sort({receiver}, false)", "void"
            if method == "sort" and len(node.keywords) == 1 and node.keywords[0].arg == "reverse":
                return f"nx_list_sort({receiver}, {self.argument(node.keywords[0].value, 'bool')})", "void"
            raise self.error_cls(f"unsupported List method: {method}", node)

        # Dict[K, V] methods
        dp = dict_parts(rtype)
        if dp:
            kt, vt = dp
            if method == "get" and len(node.args) in {1, 2}:
                key = self.argument(node.args[0], kt)
                if len(node.args) == 2:
                    fb = self.argument(node.args[1], vt)
                    return f"nx_dict_get({receiver}, {key}, {fb})", vt
                return f"nx_dict_get({receiver}, {key}, {self._cpp_type(vt)}{{}})", vt
            if method == "pop" and len(node.args) == 1:
                return f"nx_dict_pop({receiver}, {self.argument(node.args[0], kt)})", vt
            if method == "setdefault" and len(node.args) == 2:
                self.require_caps({"memory.alloc"}, node)
                return f"nx_dict_setdefault({receiver}, {self.argument(node.args[0], kt)}, {self.argument(node.args[1], vt)})", vt
            if method == "keys" and not node.args:
                self.require_caps({"memory.alloc"}, node)
                rt = f"List[{kt}]"
                self.compiler.ensure_type(rt, node)
                return f"nx_dict_keys({receiver})", rt
            if method == "values" and not node.args:
                self.require_caps({"memory.alloc"}, node)
                rt = f"List[{vt}]"
                self.compiler.ensure_type(rt, node)
                return f"nx_dict_values({receiver})", rt
            if method == "items" and not node.args:
                self.require_caps({"memory.alloc"}, node)
                tt = f"Tuple[{kt},{vt}]"
                rt = f"List[{tt}]"
                self.compiler.ensure_type(tt, node)
                self.compiler.ensure_type(rt, node)
                return f"nx_dict_items({receiver})", rt
            if method == "update" and len(node.args) == 1:
                self.require_caps({"memory.alloc"}, node)
                return f"nx_dict_update({receiver}, {self.argument(node.args[0], rtype)})", "void"
            if method == "clear" and not node.args:
                return f"({receiver}).clear()", "void"
            raise self.error_cls(f"unsupported Dict method: {method}", node)

        # Set[T] methods
        se = set_element(rtype)
        if se:
            if method == "add" and len(node.args) == 1:
                self.require_caps({"memory.alloc"}, node)
                return f"({receiver}).insert({self.argument(node.args[0], se)})", "void"
            if method in {"remove", "discard"} and len(node.args) == 1:
                return f"nx_set_{method}({receiver}, {self.argument(node.args[0], se)})", "void"
            if method == "clear" and not node.args:
                return f"({receiver}).clear()", "void"
            raise self.error_cls(f"unsupported Set method: {method}", node)

        # str methods
        if rtype == "str":
            if method in {"lower", "upper"} and not node.args:
                self.require_caps({"memory.alloc"}, node)
                return f"({receiver}).{method}()", "str"
            if method in {"strip", "lstrip", "rstrip"} and len(node.args) <= 1:
                self.require_caps({"memory.alloc"}, node)
                if node.args:
                    ch = self.argument(node.args[0], "str")
                    return f"({receiver}).{method}(({ch}).c_str())", "str"
                return f"({receiver}).{method}()", "str"
            if method == "replace" and len(node.args) == 2:
                self.require_caps({"memory.alloc"}, node)
                return f"({receiver}).replace({self.argument(node.args[0], 'str')}, {self.argument(node.args[1], 'str')})", "str"
            if method == "split" and len(node.args) <= 1:
                self.require_caps({"memory.alloc"}, node)
                delim = self.argument(node.args[0], "str") if node.args else 'NxStr(" ")'
                return f"nx_string_split({receiver}, {delim})", "List[str]"
            if method == "join" and len(node.args) == 1:
                self.require_caps({"memory.alloc"}, node)
                return f"nx_string_join({receiver}, {self.argument(node.args[0], 'List[str]')})", "str"
            if method in {"startswith", "endswith"} and len(node.args) == 1:
                return f"({receiver}).{method}({self.argument(node.args[0], 'str')})", "bool"
            if method == "find" and len(node.args) in {1, 2}:
                st = self.argument(node.args[1], "i64") if len(node.args) == 2 else "0"
                return f"({receiver}).find({self.argument(node.args[0], 'str')}, {st})", "i64"
            if method == "count" and len(node.args) == 1:
                return f"({receiver}).count({self.argument(node.args[0], 'str')})", "i64"
            if method in {"isdigit", "isalpha"} and not node.args:
                return f"({receiver}).{method}()", "bool"
            raise self.error_cls(f"unsupported str method: {method}", node)

        # Trait-bounded generic method call
        g_bound = dict(self.fn.generic).get(rtype)
        if g_bound in self.compiler.traits:
            sig = self.compiler.traits[g_bound].get(method)
            if sig is None or len(node.args) != len(sig[0]):
                raise self.error_cls(f"trait {g_bound} has no matching method {method}", node)
            args = [self.argument(a, rtype if exp == "Self" else exp) for a, exp in zip(node.args, sig[0])]
            res = rtype if sig[1] == "Self" else sig[1]
            return f"({receiver}).nx_{method}({', '.join(args)})", res

        # Enum methods
        if rtype in self.compiler.enums:
            edef = self.compiler.enums[rtype]
            m = edef.methods.get(method)
            if m is None:
                raise self.error_cls(f"unknown enum method: {rtype}.{method}", node)
            self.require_caps(m.caps, node)
            if len(node.args) != len(m.params) - 1:
                raise self.error_cls(f"{rtype}.{method} expects {len(m.params) - 1} arguments", node)
            args = [self.argument(a, exp) for a, (_, exp) in zip(node.args, m.params[1:])]
            return f"({receiver}).nx_{method}({', '.join(args)})", m.result

        # Class methods (including inherited & read-only borrowed support!)
        owner, operator = self._resolve_class(rtype)
        if owner not in self.compiler.classes:
            raise self.error_cls(f"method call requires a class or collection (got {rtype})", node.func)
        m = self._find_class_method(owner, method)
        if m is None or method in {"__init__", "__del__"}:
            raise self.error_cls(f"unknown class method: {owner}.{method}", node)
        if rtype.startswith("borrowed[") and m.mutates_self:
            raise self.error_cls(f"cannot call mutating method {owner}.{method} on borrowed value", node.func)
        self.require_caps(m.caps, node)
        if len(node.args) != len(m.params) - 1:
            raise self.error_cls(f"{owner}.{method} expects {len(m.params) - 1} arguments", node)
        args = [self.argument(a, exp) for a, (_, exp) in zip(node.args, m.params[1:])]
        return f"({receiver}){operator}nx_{method}({', '.join(args)})", m.result

    def print_stmt(self, node: ast.Call) -> None:
        sep = ' " "'
        end = ' "\\n"'
        for kw in node.keywords:
            if kw.arg == "sep" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                sep = " " + c_string(kw.value.value)
            elif kw.arg == "end" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                end = " " + c_string(kw.value.value)
            else:
                raise self.error_cls("print only supports constant string sep= and end=", kw)
        for i, arg in enumerate(node.args):
            if i:
                self.emit(f"std::fputs({sep.strip()}, stdout);")
            val, typ = self.expr(arg)
            if typ == "void":
                raise self.error_cls("print cannot display void", arg)
            if typ == "i128":
                self.emit(f"nx_print_i128({val});")
            elif typ == "u128":
                self.emit(f"nx_print_u128({val});")
            elif typ in {"i8", "i16", "i32"}:
                self.emit(f'std::printf("%d", static_cast<int>({val}));')
            elif typ in {"u8", "u16", "u32"}:
                self.emit(f'std::printf("%u", static_cast<unsigned>({val}));')
            elif typ == "i64":
                self.emit(f'std::printf("%lld", static_cast<long long>({val}));')
            elif typ == "u64":
                self.emit(f'std::printf("%llu", static_cast<unsigned long long>({val}));')
            elif typ == "f32":
                self.emit(f'std::printf("%.9g", static_cast<double>({val}));')
            elif typ == "f64":
                self.emit(f'std::printf("%.17g", {val});')
            elif typ == "bool":
                self.emit(f'std::fputs({val} ? "true" : "false", stdout);')
            elif typ == "str":
                self.emit(f"std::fputs(({val}).c_str(), stdout);")
            elif typ == "dyn":
                self.emit(f"nx_print_value({val});")
            else:
                self.emit(f"std::fputs(nx_to_std_string({val}).c_str(), stdout);")
        self.emit(f"std::fputs({end.strip()}, stdout);")

    def block(self, nodes: list[ast.stmt]) -> bool:
        always_returns = False
        for node in nodes:
            if always_returns:
                raise self.error_cls("unreachable statement after return", node)
            always_returns = self.stmt(node)
        return always_returns

    def nested(self, nodes: list[ast.stmt], *, extra_narrow: dict[str, str] | None = None) -> bool:
        self.scopes.append({})
        self.narrowed.append(dict(extra_narrow or {}))
        self.depth += 1
        res = self.block(nodes)
        self.depth -= 1
        self.moved.difference_update(self.scopes[-1])
        self.narrowed.pop()
        self.scopes.pop()
        return res

    def _detect_none_guard(self, test: ast.expr) -> tuple[str, bool] | None:
        """Detect `x is not None` -> (x, True) or `x is None` -> (x, False) for flow narrowing."""
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
            if isinstance(test.left, ast.Name) and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None:
                if isinstance(test.ops[0], ast.IsNot):
                    return test.left.id, True
                if isinstance(test.ops[0], ast.Is):
                    return test.left.id, False
        return None

    def _unpack_tuple_assign(self, target: ast.Tuple, value_node: ast.expr, node: ast.stmt) -> bool:
        val, vtyp = self.expr(value_node)
        tp = tuple_parts(vtyp)
        if not tp or len(tp) != len(target.elts):
            raise self.error_cls(f"tuple unpacking requires Tuple of length {len(target.elts)}", node)
        self.serial += 1
        sfx = self.serial
        self.emit(f"auto nx_tup_{sfx} = {val};")
        for idx, (elt, etyp) in enumerate(zip(target.elts, tp)):
            if not isinstance(elt, ast.Name):
                raise self.error_cls("tuple unpacking targets must be variable names", elt)
            name = elt.id
            self._check_name(name, elt)
            elem_val = f"std::get<{idx}>(nx_tup_{sfx})"
            try:
                existing = self.lookup(name, elt)
                if not compatible_types(etyp, existing):
                    raise self.error_cls(f"expected {existing}, got {etyp}", elt)
                self.emit(f"{self._var_cpp(name)} = {coerce_cpp(elem_val, etyp, existing, self.compiler.types)};")
            except self.error_cls:
                self.scopes[-1][name] = etyp
                self.emit(f"{self._cpp_type(etyp)} nx_{name} = {elem_val};")
        return False

    def _match_stmt(self, node: ast.Match) -> bool:
        subj, styp = self.expr(node.subject)
        self.serial += 1
        sfx = self.serial
        self.emit(f"auto nx_m_{sfx} = {subj};")

        is_enum = styp in self.compiler.enums
        edef = self.compiler.enums.get(styp)
        covered_variants: set[str] = set()
        has_wildcard = False
        all_return = True

        for ci, case in enumerate(node.cases):
            branch_kw = "if" if ci == 0 else "else if"
            pat = case.pattern
            extra_vars: dict[str, str] = {}
            bind_lines: list[str] = []
            cond = "true"

            if isinstance(pat, ast.MatchAs) and pat.pattern is None:
                if pat.name is None:
                    if case.guard is None:
                        has_wildcard = True
                    cond = "true"
                else:
                    self._check_name(pat.name, pat)
                    extra_vars[pat.name] = styp
                    bind_lines.append(f"{self._cpp_type(styp)} nx_{pat.name} = nx_m_{sfx};")
                    if case.guard is None:
                        has_wildcard = True
                    cond = "true"
            elif isinstance(pat, ast.MatchValue):
                pv, pt = self.expr(pat.value)
                if is_enum:
                    raise self.error_cls("match enum variants using Enum.Variant(...)", pat)
                if not compatible_types(pt, styp):
                    raise self.error_cls(f"match case type {pt} incompatible with {styp}", pat)
                cond = f"(nx_m_{sfx} == {coerce_cpp(pv, pt, styp, self.compiler.types)})"
            elif isinstance(pat, ast.MatchSingleton):
                if pat.value is None:
                    oe = optional_element(styp)
                    if oe:
                        cond = f"(nx_m_{sfx} == nullptr)" if oe.startswith("owned[") else f"(!nx_m_{sfx}.has_value())"
                    elif styp == "dyn":
                        cond = f"(nx_m_{sfx}.type == 0)"
                    else:
                        raise self.error_cls("case None requires Optional or dyn subject", pat)
                else:
                    if styp != "bool":
                        raise self.error_cls("case True/False requires bool subject", pat)
                    cond = f"(nx_m_{sfx} == {'true' if pat.value else 'false'})"
            elif isinstance(pat, ast.MatchClass):
                # Handles Enum.Variant(a, b) or Ok(v) / Err(e)
                if isinstance(pat.cls, ast.Attribute) and isinstance(pat.cls.value, ast.Name) and is_enum and pat.cls.value.id == styp:
                    vname = pat.cls.attr
                    assert edef is not None
                    if vname not in edef.variants:
                        raise self.error_cls(f"unknown variant {styp}.{vname}", pat)
                    v_fields = edef.variants[vname]
                    if len(pat.patterns) != len(v_fields):
                        raise self.error_cls(f"{styp}.{vname} expects {len(v_fields)} pattern bindings", pat)
                    if case.guard is None:
                        covered_variants.add(vname)
                    tag_id = f"{self.unit.cpp_symbol(styp)}::Tag::{vname}"
                    cond = f"(nx_m_{sfx}.tag == {tag_id})"
                    for fi, (subpat, (_, ft)) in enumerate(zip(pat.patterns, v_fields)):
                        if isinstance(subpat, ast.MatchAs) and subpat.pattern is None and subpat.name:
                            self._check_name(subpat.name, subpat)
                            extra_vars[subpat.name] = ft
                            bind_lines.append(f"{self._cpp_type(ft)} nx_{subpat.name} = std::get<{self.unit.cpp_symbol(styp)}::Variant_{vname}>(nx_m_{sfx}.data).f{fi};")
                        elif isinstance(subpat, ast.MatchAs) and subpat.name is None and subpat.pattern is None:
                            pass
                        else:
                            raise self.error_cls("match destructuring supports variable names or _", subpat)
                elif isinstance(pat.cls, ast.Name) and pat.cls.id in {"Ok", "Err"} and result_parts(styp):
                    rp = result_parts(styp)
                    assert rp is not None
                    is_ok = pat.cls.id == "Ok"
                    cond = f"(nx_m_{sfx}.ok())" if is_ok else f"(nx_m_{sfx}.is_err())"
                    if len(pat.patterns) == 1 and isinstance(pat.patterns[0], ast.MatchAs) and pat.patterns[0].name:
                        bname = pat.patterns[0].name
                        self._check_name(bname, pat.patterns[0])
                        btyp = rp[0] if is_ok else (rp[1] or "str")
                        extra_vars[bname] = btyp
                        expr_get = f"nx_m_{sfx}.value()" if is_ok else f"nx_m_{sfx}.error()"
                        bind_lines.append(f"{self._cpp_type(btyp)} nx_{bname} = {expr_get};")
                else:
                    raise self.error_cls("unsupported match class pattern", pat)
            else:
                raise self.error_cls("unsupported match pattern", pat)

            self.scopes.append(extra_vars)
            self.depth += 1
            guard_cpp = ""
            if case.guard is not None:
                gc, gt = self.expr(case.guard)
                if gt != "bool":
                    raise self.error_cls("match guard must be bool", case.guard)
                guard_cpp = gc
            self.depth -= 1
            self.scopes.pop()

            if case.guard is not None and not bind_lines:
                full_cond = f"({cond} && ({guard_cpp}))"
                bind_check = ""
            elif case.guard is not None:
                full_cond = cond
                bind_check = guard_cpp
            else:
                full_cond = cond
                bind_check = ""

            is_final_exhaustive = (
                ci == len(node.cases) - 1
                and not bind_check
                and (has_wildcard or (is_enum and edef is not None and len(covered_variants) == len(edef.variants)))
            )
            if ci > 0 and is_final_exhaustive:
                self.emit("else {")
            else:
                self.emit(f"{branch_kw} ({full_cond}) {{")
            self.scopes.append(extra_vars)
            self.narrowed.append({})
            self.depth += 1
            for bl in bind_lines:
                self.emit(bl)
            if bind_check:
                self.emit(f"if ({bind_check}) {{")
                self.depth += 1
            ret = self.block(case.body)
            all_return = all_return and ret
            if bind_check:
                self.depth -= 1
                self.emit("}")
                all_return = False
            self.depth -= 1
            self.moved.difference_update(self.scopes[-1])
            self.narrowed.pop()
            self.scopes.pop()
            self.emit("}")

        if is_enum and edef is not None and not has_wildcard:
            missing = [v for v in edef.variants if v not in covered_variants]
            if missing:
                raise self.error_cls(f"non-exhaustive match on {styp}; missing variants: {', '.join(missing)}", node)
        return all_return and (has_wildcard or (is_enum and edef is not None and set(edef.variants) <= covered_variants))

    def stmt(self, node: ast.stmt) -> bool:
        self.emit_debug_line(node)
        if isinstance(node, ast.Global):
            if any(name not in self.unit.globals for name in node.names):
                raise self.error_cls("global names must be declared at file level", node)
            return False
        if isinstance(node, (ast.Break, ast.Continue)):
            if self.loop_depth == 0:
                raise self.error_cls(f"{type(node).__name__.lower()} is only valid inside a loop", node)
            self.emit("break;" if isinstance(node, ast.Break) else "continue;")
            return False
        if isinstance(node, ast.Pass):
            self.emit(";")
            return False
        if isinstance(node, ast.Match):
            # Pre-rewrite unit enum attribute MatchValue -> MatchClass-equivalent handling in helper
            # Let's check unit enum cases cleanly by rewriting MatchValue(Attribute) -> handled cleanly
            return self._match_stmt_patched(node)
        if isinstance(node, ast.With):
            if len(node.items) != 1 or node.items[0].optional_vars is None or not isinstance(node.items[0].optional_vars, ast.Name):
                raise self.error_cls("with supports `with expr as name:`", node)
            item = node.items[0]
            var_name = item.optional_vars.id  # type: ignore[union-attr]
            self._check_name(var_name, item.optional_vars)
            val, typ = self.expr(item.context_expr)
            if typ != "File" and not (typ in self.compiler.classes and self._find_class_method(typ, "__enter__") and self._find_class_method(typ, "__exit__")):
                raise self.error_cls("with target must be File or class implementing __enter__/__exit__", item.context_expr)
            self.serial += 1
            sfx = self.serial
            self.emit("{")
            self.scopes.append({var_name: typ})
            self.narrowed.append({})
            self.depth += 1
            self.emit(f"auto nx_ctx_{sfx} = {val};")
            if typ == "File":
                self.emit(f"nx_File &nx_{var_name} = nx_ctx_{sfx}.nx___enter__();")
                # Also guard RAII exit!
                self.emit(f"struct NxWithGuard_{sfx} {{ nx_File &ref; ~NxWithGuard_{sfx}() {{ ref.nx___exit__(); }} }} nx_wg_{sfx}{{nx_ctx_{sfx}}};")
            else:
                enter_m = self._find_class_method(typ, "__enter__")
                assert enter_m is not None
                self.scopes[-1][var_name] = enter_m.result
                self.emit(f"auto nx_{var_name} = nx_ctx_{sfx}.nx___enter__();")
                self.emit(f"struct NxWithGuard_{sfx} {{ {self._cpp_type(typ)} &ref; ~NxWithGuard_{sfx}() {{ ref.nx___exit__(); }} }} nx_wg_{sfx}{{nx_ctx_{sfx}}};")
            ret = self.block(node.body)
            self.depth -= 1
            self.moved.difference_update(self.scopes[-1])
            self.narrowed.pop()
            self.scopes.pop()
            self.emit("}")
            return ret
        if isinstance(node, ast.Delete):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Subscript):
                raise self.error_cls("del supports Dict key deletion `del d[k]`", node)
            sub = node.targets[0]
            container, ctyp = self.expr(sub.value)
            dp = dict_parts(ctyp)
            if not dp:
                raise self.error_cls("del target must be Dict[K, V]", sub)
            key = self.argument(sub.slice, dp[0])
            self.emit(f"({container}).erase({key});")
            return False
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "print":
                self.print_stmt(call)
            else:
                val, _ = self.expr(call)
                self.emit(f"{val};")
            return False
        if isinstance(node, ast.Expr):
            # Standalone await expr etc.
            val, _ = self.expr(node.value)
            self.emit(f"{val};")
            return False
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name) or node.value is None:
                raise self.error_cls("typed declaration must have a name and initializer", node)
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in {"buffer_span", "buffer_mutspan"}:
                raise self.error_cls("Buffer view must be passed directly; it cannot be stored", node.value)
            name = node.target.id
            self._check_name(name, node)
            if any(name in scope for scope in self.scopes):
                raise self.error_cls(f"variable already defined: {name}", node)
            typ = resolve_type_node(node.annotation, types=self.compiler.types, classes=set(self.compiler.classes), enums=set(self.compiler.enums), error_cls=self.error_cls)
            if typ == "void":
                raise self.error_cls("variable cannot have void type", node)
            if typ.startswith(("borrowed[", "mut[")):
                raise self.error_cls("borrowed and mut are parameter types only", node)
            self.compiler.ensure_type(typ, node)
            if isinstance(node.value, (ast.List, ast.Dict)):
                val_raw, val_t = self.expr(node.value, expected_hint=typ)
                if not compatible_types(val_t, typ, subclasses=self.compiler.subclasses):
                    raise self.error_cls(f"expected {typ}, got {val_t}", node.value)
                value = coerce_cpp(val_raw, val_t, typ, self.compiler.types)
            elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id in self.unit.imports and self.unit.imports[node.value.func.value.id].kind in {"cpp", "c", "rust", "go"}:
                val_raw, val_t = self._call(node.value, expected_hint=typ)
                value = coerce_cpp(val_raw, val_t, typ, self.compiler.types)
            else:
                value = self.argument(node.value, typ)
            self.scopes[-1][name] = typ
            self.emit(f"{self._cpp_type(typ)} nx_{name} = {value};")
            return False
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in {"buffer_span", "buffer_mutspan"}:
                raise self.error_cls("Buffer view must be passed directly; it cannot be stored", node.value)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
                return self._unpack_tuple_assign(node.targets[0], node.value, node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
                target, owner, _ = self.member(node.targets[0], mutable=True)
                expected = self._find_class_field(owner, node.targets[0].attr)
                if expected is None:
                    raise self.error_cls(f"unknown class field: {owner}.{node.targets[0].attr}", node.targets[0])
                value = self.argument(node.value, expected)
                self.emit(f"{target} = {value};")
                return False
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                sub = node.targets[0]
                target, expected = self.index(sub, mutable=True)
                container_t = self.expr(sub.value)[1]
                if container_t == "dyn":
                    key_c, key_t = self.expr(sub.slice)
                    val_c, val_t = self.expr(node.value)
                    cont_c, _ = self.expr(sub.value)
                    self.emit(f"nx_dyn_setitem({cont_c}, {coerce_cpp(key_c, key_t, 'dyn', self.compiler.types)}, {coerce_cpp(val_c, val_t, 'dyn', self.compiler.types)});")
                    return False
                val_c, act_t = self.expr(node.value)
                if not compatible_types(act_t, expected, subclasses=self.compiler.subclasses):
                    raise self.error_cls(f"expected {expected}, got {act_t}", node.value)
                self.emit(f"{target} = {coerce_cpp(val_c, act_t, expected, self.compiler.types)};")
                return False
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise self.error_cls("assignment requires one variable or tuple", node)
            name = node.targets[0].id
            self._check_name(name, node)
            if name.isupper() and name in self.unit.globals:
                raise self.error_cls(f"cannot assign to file constant: {name}", node)
            value, actual = self.expr(node.value)
            if actual == "void":
                raise self.error_cls("cannot assign a void result", node.value)
            if name in self.moved:
                expected = next((s[name] for s in reversed(self.scopes) if name in s), None)
                if expected is None or not compatible_types(actual, expected, subclasses=self.compiler.subclasses):
                    raise self.error_cls(f"cannot reinitialize moved variable {name} with {actual}", node)
                self.moved.remove(name)
                self.emit(f"nx_{name} = {coerce_cpp(value, actual, expected, self.compiler.types)};")
                return False
            try:
                expected = self.lookup(name, node)
            except self.error_cls:
                if actual == "None":
                    raise self.error_cls("None requires an Optional[T] or dyn annotation", node.value)
                if actual.startswith(("borrowed[", "mut[")):
                    raise self.error_cls("cannot store a borrow", node.value)
                if self._is_move_only(actual) and isinstance(node.value, ast.Name):
                    raise self.error_cls("owned value must be moved with move(...)", node.value)
                self.scopes[-1][name] = actual
                self.compiler.ensure_type(actual, node)
                self.emit(f"{self._cpp_type(actual)} nx_{name} = {value};")
                return False
            if not compatible_types(actual, expected, subclasses=self.compiler.subclasses):
                raise self.error_cls(f"expected {expected}, got {actual}", node.value)
            if span_element(expected) and name not in self.scopes[-1]:
                raise self.error_cls("cannot rebind an outer Span from a nested block", node)
            if self._is_move_only(expected) and isinstance(node.value, ast.Name):
                raise self.error_cls("owned value must be moved with move(...)", node.value)
            self.moved.discard(name)
            self.emit(f"{self._var_cpp(name)} = {coerce_cpp(value, actual, expected, self.compiler.types)};")
            return False
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Attribute):
                target, owner, _ = self.member(node.target, mutable=True)
                typ = self._find_class_field(owner, node.target.attr)
                value, actual = self.expr(node.value)
                op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(node.op))
                if typ not in NUMERICS or op is None or not compatible_types(actual, typ):
                    raise self.error_cls("invalid class field augmented assignment", node)
                self.emit(f"{target} {op}= {coerce_cpp(value, actual, typ, self.compiler.types)};")
                return False
            if isinstance(node.target, ast.Subscript):
                target, typ = self.index(node.target, mutable=True)
                value, actual = self.expr(node.value)
                op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(node.op))
                if op is None or not compatible_types(actual, typ):
                    raise self.error_cls("invalid array augmented assignment", node)
                self.emit(f"{target} {op}= {coerce_cpp(value, actual, typ, self.compiler.types)};")
                return False
            if not isinstance(node.target, ast.Name):
                raise self.error_cls("augmented assignment requires a variable", node)
            name = node.target.id
            typ = self.lookup(name, node.target)
            if name.isupper() and name in self.unit.globals:
                raise self.error_cls(f"cannot assign to file constant: {name}", node)
            value, actual = self.expr(node.value)
            if typ == "str" and isinstance(node.op, ast.Add) and actual == "str":
                self.require_caps({"memory.alloc"}, node)
                self.emit(f"{self._var_cpp(name)} = ({self._var_cpp(name)} + {value});")
                return False
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(node.op))
            if op is None or typ not in NUMERICS or not compatible_types(actual, typ):
                raise self.error_cls("invalid augmented assignment", node)
            self.emit(f"{self._var_cpp(name)} {op}= {coerce_cpp(value, actual, typ, self.compiler.types)};")
            return False
        if isinstance(node, ast.Return):
            fn_ret = future_element(self.fn.result) if self.fn.is_async else self.fn.result
            assert fn_ret is not None
            if node.value is None:
                if fn_ret == "Result[void]":
                    self.emit("return nx_rt_ok_void();")
                    return True
                if fn_ret != "void":
                    raise self.error_cls(f"return requires {fn_ret}", node)
                self.emit("return;")
                return True
            value, actual = self.expr(node.value)
            rparts = result_parts(fn_ret)
            expected = rparts[0] if (self.fn.raises and rparts) else fn_ret
            if not compatible_types(actual, expected, subclasses=self.compiler.subclasses):
                raise self.error_cls(f"return expected {expected}, got {actual}", node)
            if (self._is_move_only(expected) or self.fn.raises) and isinstance(node.value, ast.Name):
                value = f"std::move({value})"
            value = coerce_cpp(value, actual, expected, self.compiler.types)
            if self.fn.raises and rparts:
                if rparts[1] is not None:
                    self.emit(f"return nx_rt_ok_e<{self._cpp_type(expected)}, {self._cpp_type(rparts[1])}>({value});")
                else:
                    self.emit(f"return nx_rt_ok<{self._cpp_type(expected)}>({value});")
            else:
                self.emit(f"return {value};")
            return True
        if isinstance(node, ast.Raise):
            rparts = result_parts(self.fn.result)
            if not self.fn.raises or rparts is None or node.exc is None or node.cause is not None:
                raise self.error_cls("raise requires a raises function and one error value", node)
            value, typ = self.expr(node.exc)
            ok_t, err_t = rparts
            expected_err = err_t or "str"
            if not compatible_types(typ, expected_err):
                raise self.error_cls(f"raise error must be {expected_err}, got {typ}", node.exc)
            coerced = coerce_cpp(value, typ, expected_err, self.compiler.types)
            if ok_t == "void":
                self.emit(f"return nx_rt_err_void({coerced});")
            elif err_t is not None:
                self.emit(f"return nx_rt_err_e<{self._cpp_type(ok_t)}, {self._cpp_type(err_t)}>({coerced});")
            else:
                self.emit(f"return nx_rt_err<{self._cpp_type(ok_t)}>({coerced});")
            return True
        if isinstance(node, ast.Try):
            return self._try_stmt(node)
        if isinstance(node, ast.If):
            guard = self._detect_none_guard(node.test)
            cond, ct = self.expr(node.test)
            if ct == "dyn":
                cond = f"nx_as_bool({cond})"
                ct = "bool"
            if ct != "bool":
                raise self.error_cls("if condition must be bool", node.test)
            extra_yes: dict[str, str] = {}
            extra_no: dict[str, str] = {}
            if guard:
                gname, is_not_none = guard
                try:
                    gtyp = self.lookup(gname, node.test)
                    oe = optional_element(gtyp)
                    if oe:
                        if is_not_none:
                            extra_yes[gname] = oe
                        else:
                            extra_no[gname] = oe
                except self.error_cls:
                    pass
            moved_before = set(self.moved)
            self.emit(f"if ({cond}) {{")
            yes = self.nested(node.body, extra_narrow=extra_yes)
            moved_yes = set(self.moved)
            self.emit("}")
            no = False
            moved_no = set(moved_before)
            if node.orelse:
                self.moved = set(moved_before)
                self.emit("else {")
                no = self.nested(node.orelse, extra_narrow=extra_no)
                moved_no = set(self.moved)
                self.emit("}")
            # Flow-sensitive join: if either non-returning branch moved a variable, mark moved
            if yes and not no:
                self.moved = moved_no
                if guard and not guard[1]:
                    # `if x is None: return` -> narrow x for rest of block!
                    gtyp = self.lookup(guard[0], node.test)
                    oe = optional_element(gtyp)
                    if oe:
                        self.narrowed[-1][guard[0]] = oe
            elif no and not yes:
                self.moved = moved_yes
            else:
                self.moved = moved_yes | moved_no
            return yes and no
        if isinstance(node, ast.While):
            if node.orelse:
                raise self.error_cls("while-else is not supported", node)
            cond, ct = self.expr(node.test)
            if ct == "dyn":
                cond = f"nx_as_bool({cond})"
                ct = "bool"
            if ct != "bool":
                raise self.error_cls("while condition must be bool", node.test)
            outer_vars = {n for s in self.scopes for n in s}
            moved_before = set(self.moved)
            self.emit(f"while ({cond}) {{")
            self.loop_depth += 1
            self.nested(node.body)
            self.loop_depth -= 1
            self.emit("}")
            newly_moved = (self.moved - moved_before) & outer_vars
            if newly_moved:
                raise self.error_cls(f"cannot move outer variable inside loop: {', '.join(sorted(newly_moved))}", node)
            return False
        if isinstance(node, ast.For):
            return self._for_stmt(node)
        raise self.error_cls(f"unsupported statement: {type(node).__name__}", node)

    def _match_stmt_patched(self, node: ast.Match) -> bool:
        # Convert MatchValue(Attribute(Name(E), variant)) into MatchClass(Attribute(Name(E), variant), [])
        patched_cases = []
        for c in node.cases:
            if isinstance(c.pattern, ast.MatchValue) and isinstance(c.pattern.value, ast.Attribute) and isinstance(c.pattern.value.value, ast.Name):
                nc = copy.copy(c)
                nc.pattern = ast.MatchClass(cls=c.pattern.value, patterns=[], kwd_attrs=[], kwd_patterns=[])
                ast.copy_location(nc.pattern, c.pattern)
                patched_cases.append(nc)
            else:
                patched_cases.append(c)
        patched = copy.copy(node)
        patched.cases = patched_cases
        return self._match_stmt(patched)

    def _try_stmt(self, node: ast.Try) -> bool:
        if len(node.handlers) != 1 or node.orelse:
            raise self.error_cls("try supports one except clause (with optional `as err`) and optional finally", node)
        handler = node.handlers[0]
        if handler.type is not None and not (isinstance(handler.type, ast.Name) and handler.type.id in {"Err", "Exception", "Error"}):
            raise self.error_cls("except supports bare except or Err/Exception", handler.type)
        err_var = handler.name
        if err_var:
            self._check_name(err_var, handler)

        self.serial += 1
        sfx = self.serial
        if node.finalbody:
            # Emit RAII finally guard so finally runs on return too!
            self.emit("{")
            self.depth += 1
            self.emit(f"auto nx_fin_{sfx} = [&]() {{")
            self.nested(node.finalbody)
            self.emit("};")
            self.emit(f"struct NxFinGuard_{sfx} {{ decltype(nx_fin_{sfx}) &fn; ~NxFinGuard_{sfx}() {{ fn(); }} }} nx_fg_{sfx}{{nx_fin_{sfx}}};")

        first = node.body[0] if node.body else None
        if isinstance(first, ast.Assign) and len(first.targets) == 1 and isinstance(first.targets[0], ast.Name):
            name, expr_n = first.targets[0].id, first.value
            ann = None
        elif isinstance(first, ast.AnnAssign) and isinstance(first.target, ast.Name) and first.value is not None:
            name, expr_n = first.target.id, first.value
            ann = resolve_type_node(first.annotation, types=self.compiler.types, classes=set(self.compiler.classes), enums=set(self.compiler.enums), error_cls=self.error_cls)
        else:
            raise self.error_cls("try must start with a Result[T] assignment", first or node)

        self._check_name(name, first)
        if any(name in s for s in self.scopes):
            raise self.error_cls(f"variable already defined: {name}", first)
        val, typ = self.expr(expr_n)
        rp = result_parts(typ)
        if rp is None:
            raise self.error_cls("try assignment requires a Result[T] expression", expr_n)
        elem, err_t = rp
        if ann and ann != elem:
            raise self.error_cls(f"try value must have type {elem}", first)

        self.emit(f"auto nx_try_{sfx} = {val};")
        self.emit(f"if (nx_try_{sfx}.ok()) {{")
        self.scopes.append({name: elem})
        self.narrowed.append({})
        self.depth += 1
        self.emit(f"{self._cpp_type(elem)} nx_{name} = nx_try_{sfx}.value();")
        success = self.block(node.body[1:])
        self.depth -= 1
        self.moved.difference_update(self.scopes[-1])
        self.narrowed.pop()
        self.scopes.pop()
        self.emit("} else {")
        err_scope = {err_var: (err_t or "str")} if err_var else {}
        self.scopes.append(err_scope)
        self.narrowed.append({})
        self.depth += 1
        if err_var:
            self.emit(f"{self._cpp_type(err_t or 'str')} nx_{err_var} = nx_try_{sfx}.error();")
        failure = self.block(handler.body)
        self.depth -= 1
        self.moved.difference_update(self.scopes[-1])
        self.narrowed.pop()
        self.scopes.pop()
        self.emit("}")
        if node.finalbody:
            self.depth -= 1
            self.emit("}")
        return success and failure

    def _for_stmt(self, node: ast.For) -> bool:
        if node.orelse:
            raise self.error_cls("for-else is not supported", node)
        # Handle `for i, val in enumerate(seq):` and `for a, b in zip(s1, s2):` and `for k, v in d.items():`
        if isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2 and all(isinstance(e, ast.Name) for e in node.target.elts):
            n0, n1 = node.target.elts[0].id, node.target.elts[1].id  # type: ignore[attr-defined]
            self._check_name(n0, node.target.elts[0])
            self._check_name(n1, node.target.elts[1])
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "enumerate" and len(node.iter.args) == 1:
                cont, ctyp = self.expr(node.iter.args[0])
                elem = list_element(ctyp) or span_element(ctyp) or ("str" if ctyp == "str" else None)
                if not elem:
                    raise self.error_cls("enumerate requires List, Span or str", node.iter)
                self.serial += 1
                sfx = self.serial
                self.emit(f"auto &&nx_en_{sfx} = {cont};")
                sz = f"nx_en_{sfx}.size()" if list_element(ctyp) or ctyp == "str" else f"nx_en_{sfx}.len"
                it = f"nx_en_{sfx}.at_char(nx_i_{sfx})" if ctyp == "str" else (f"nx_en_{sfx}[nx_i_{sfx}]" if list_element(ctyp) else f"nx_en_{sfx}.ptr[nx_i_{sfx}]")
                self.emit(f"for (std::size_t nx_i_{sfx} = 0; nx_i_{sfx} < {sz}; ++nx_i_{sfx}) {{")
                self.scopes.append({n0: "i64", n1: elem})
                self.narrowed.append({})
                self.depth += 1
                self.loop_depth += 1
                self.emit(f"std::int64_t nx_{n0} = static_cast<std::int64_t>(nx_i_{sfx});")
                self.emit(f"{self._cpp_type(elem)} nx_{n1} = {it};")
                self.block(node.body)
                self.loop_depth -= 1
                self.depth -= 1
                self.moved.difference_update(self.scopes[-1])
                self.narrowed.pop()
                self.scopes.pop()
                self.emit("}")
                return False
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "zip" and len(node.iter.args) == 2:
                c0, t0 = self.expr(node.iter.args[0])
                c1, t1 = self.expr(node.iter.args[1])
                e0 = list_element(t0) or span_element(t0)
                e1 = list_element(t1) or span_element(t1)
                if not e0 or not e1:
                    raise self.error_cls("zip requires two List or Span iterables", node.iter)
                self.serial += 1
                sfx = self.serial
                self.emit(f"auto &&nx_z0_{sfx} = {c0}; auto &&nx_z1_{sfx} = {c1};")
                sz0 = f"nx_z0_{sfx}.size()" if list_element(t0) else f"nx_z0_{sfx}.len"
                sz1 = f"nx_z1_{sfx}.size()" if list_element(t1) else f"nx_z1_{sfx}.len"
                i0 = f"nx_z0_{sfx}[nx_i_{sfx}]" if list_element(t0) else f"nx_z0_{sfx}.ptr[nx_i_{sfx}]"
                i1 = f"nx_z1_{sfx}[nx_i_{sfx}]" if list_element(t1) else f"nx_z1_{sfx}.ptr[nx_i_{sfx}]"
                self.emit(f"for (std::size_t nx_i_{sfx} = 0, nx_lim_{sfx} = std::min({sz0}, {sz1}); nx_i_{sfx} < nx_lim_{sfx}; ++nx_i_{sfx}) {{")
                self.scopes.append({n0: e0, n1: e1})
                self.narrowed.append({})
                self.depth += 1
                self.loop_depth += 1
                self.emit(f"{self._cpp_type(e0)} nx_{n0} = {i0};")
                self.emit(f"{self._cpp_type(e1)} nx_{n1} = {i1};")
                self.block(node.body)
                self.loop_depth -= 1
                self.depth -= 1
                self.moved.difference_update(self.scopes[-1])
                self.narrowed.pop()
                self.scopes.pop()
                self.emit("}")
                return False
            # Tuple unpacking from iterable of Tuple[T0, T1] (e.g. d.items())
            cont, ctyp = self.expr(node.iter)
            le = list_element(ctyp)
            tp = tuple_parts(le) if le else None
            if tp and len(tp) == 2:
                self.serial += 1
                sfx = self.serial
                self.emit(f"for (const auto &nx_it_{sfx} : {cont}) {{")
                self.scopes.append({n0: tp[0], n1: tp[1]})
                self.narrowed.append({})
                self.depth += 1
                self.loop_depth += 1
                self.emit(f"{self._cpp_type(tp[0])} nx_{n0} = std::get<0>(nx_it_{sfx});")
                self.emit(f"{self._cpp_type(tp[1])} nx_{n1} = std::get<1>(nx_it_{sfx});")
                self.block(node.body)
                self.loop_depth -= 1
                self.depth -= 1
                self.moved.difference_update(self.scopes[-1])
                self.narrowed.pop()
                self.scopes.pop()
                self.emit("}")
                return False
            raise self.error_cls("for tuple unpacking requires enumerate, zip, or List[Tuple[A,B]]", node)

        if not isinstance(node.target, ast.Name):
            raise self.error_cls("for target must be a variable name or 2-tuple", node.target)
        name = node.target.id
        self._check_name(name, node.target)
        if any(name in s for s in self.scopes):
            raise self.error_cls(f"loop variable already defined: {name}", node.target)

        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id in {"range", "parallel_range"} and not node.iter.keywords:
            parallel = node.iter.func.id == "parallel_range"
            args = [self.expr(a) for a in node.iter.args]
            if len(args) not in ({1, 2} if parallel else {1, 2, 3}) or any(t not in INTEGERS for _, t in args):
                raise self.error_cls(("parallel_range expects 1 or 2" if parallel else "range expects 1 to 3") + " integer arguments", node.iter)
            start, stop, step = ("0", args[0][0], "1") if len(args) == 1 else (args[0][0], args[1][0], args[2][0] if len(args) == 3 else "1")
            self.serial += 1
            sfx = self.serial
            if parallel:
                if any(isinstance(ch, (ast.Return, ast.Break, ast.Continue)) for st in node.body for ch in ast.walk(st)):
                    raise self.error_cls("parallel_range body cannot return, break or continue", node)
                if any(isinstance(ch, ast.Call) and isinstance(ch.func, ast.Name) and ch.func.id == "move" for st in node.body for ch in ast.walk(st)):
                    raise self.error_cls("parallel_range body cannot move owned values", node)
                for st in node.body:
                    for ch in ast.walk(st):
                        if isinstance(ch, (ast.Assign, ast.AugAssign)):
                            tgts = ch.targets if isinstance(ch, ast.Assign) else [ch.target]
                            for tg in tgts:
                                if isinstance(tg, (ast.Name, ast.Attribute)):
                                    raise self.error_cls("parallel_range cannot assign shared variables or class fields", tg)
                self.emit(f"nx_parallel_for(static_cast<std::int64_t>({start}), static_cast<std::int64_t>({stop}), [&](std::int64_t nx_{name}) {{")
                self.scopes.append({name: "i64"})
                self.narrowed.append({})
                self.depth += 1
                self.block(node.body)
                self.depth -= 1
                self.moved.difference_update(self.scopes[-1])
                self.narrowed.pop()
                self.scopes.pop()
                self.emit("});")
                return False
            self.emit(f"std::int64_t nx_start_{sfx} = static_cast<std::int64_t>({start}), nx_stop_{sfx} = static_cast<std::int64_t>({stop}), nx_step_{sfx} = static_cast<std::int64_t>({step});")
            self.emit(f'if (nx_step_{sfx} == 0) nx_fail("range step cannot be zero");')
            self.emit(f"for (std::int64_t nx_index_{sfx} = nx_start_{sfx}; (nx_step_{sfx} > 0 ? nx_index_{sfx} < nx_stop_{sfx} : nx_index_{sfx} > nx_stop_{sfx}); nx_index_{sfx} += nx_step_{sfx}) {{")
            self.scopes.append({name: "i64"})
            self.narrowed.append({})
            self.depth += 1
            self.loop_depth += 1
            self.emit(f"std::int64_t nx_{name} = nx_index_{sfx};")
            self.block(node.body)
            self.loop_depth -= 1
            self.depth -= 1
            self.moved.difference_update(self.scopes[-1])
            self.narrowed.pop()
            self.scopes.pop()
            self.emit("}")
            return False

        # Iterate over List, Span, MutSpan, Buffer, Set, Dict, str, or dyn
        cont, ctyp = self.expr(node.iter)
        self.serial += 1
        sfx = self.serial
        mp = dict_parts(ctyp)
        se = set_element(ctyp)
        if mp or se:
            elem = mp[0] if mp else se
            assert elem is not None
            self.emit(f"for (const auto &nx_entry_{sfx} : {cont}) {{")
            self.scopes.append({name: elem})
            self.narrowed.append({})
            self.depth += 1
            self.loop_depth += 1
            self.emit(f"{self._cpp_type(elem)} nx_{name} = nx_entry_{sfx}{'.first' if mp else ''};")
            self.block(node.body)
            self.loop_depth -= 1
            self.depth -= 1
            self.moved.difference_update(self.scopes[-1])
            self.narrowed.pop()
            self.scopes.pop()
            self.emit("}")
            return False
        elem = list_element(ctyp) or span_element(ctyp)
        buf = buffer_parts(ctyp)
        if buf:
            elem = buf[1]
        if ctyp == "str":
            elem = "str"
            self.require_caps({"memory.alloc"}, node)
        if elem:
            self.emit(f"auto &&nx_it_{sfx} = {cont};")
            sz = f"nx_it_{sfx}.size()" if (list_element(ctyp) or ctyp == "str") else f"nx_it_{sfx}.len" if span_element(ctyp) else f"nx_it_{sfx}{'->' if buf and buf[0] == 'owned' else '.'}size()"
            val = f"nx_it_{sfx}.at_char(nx_index_{sfx})" if ctyp == "str" else f"nx_it_{sfx}[nx_index_{sfx}]" if list_element(ctyp) else f"nx_it_{sfx}.ptr[nx_index_{sfx}]" if span_element(ctyp) else f"nx_it_{sfx}{'->' if buf and buf[0] == 'owned' else '.'}at(nx_index_{sfx})"
            self.emit(f"for (std::size_t nx_index_{sfx} = 0; nx_index_{sfx} < {sz}; ++nx_index_{sfx}) {{")
            self.scopes.append({name: elem})
            self.narrowed.append({})
            self.depth += 1
            self.loop_depth += 1
            self.emit(f"{self._cpp_type(elem)} nx_{name} = {val};")
            self.block(node.body)
            self.loop_depth -= 1
            self.depth -= 1
            self.moved.difference_update(self.scopes[-1])
            self.narrowed.pop()
            self.scopes.pop()
            self.emit("}")
            return False
        raise self.error_cls("for supports range, parallel_range, List, Span, Buffer, Dict, Set or str", node)

    def generate(self, emit_name: str, source_name: str) -> list[str]:
        if self.fn.cpp_symbol:
            params = ", ".join(f"{self._cpp_type(t)} nx_{n}" for n, t in self.fn.params) or "void"
            args = ", ".join(f"nx_{n}" for n, _ in self.fn.params)
            call = f"{self.fn.cpp_symbol}({args})"
            stmt = f"{call};" if self.fn.result == "void" else f"return {call};"
            return [f"static inline {self._cpp_type(self.fn.result)} {emit_name}({params}) {{ {stmt} }}"]
        if self.fn.py_call:
            params = ", ".join(f"{self._cpp_type(t)} nx_{n}" for n, t in self.fn.params) or "void"
            mod, func = self.fn.py_call
            arg = "nx_" + self.fn.params[0][0]
            sfx = "mutspan" if self.fn.params[0][1] == "MutSpan[f64]" else "span"
            res = self.fn.result
            kind = "f64" if res == "f64" else "str" if res == "str" else "buffer"
            call = f"nx_py_call_{kind}_{sfx}({c_string(mod)}, {c_string(func)}, {arg})"
            if res == "str":
                call = f"nx_adopt_py_string({call})"
            elif res == "owned[Buffer[f64]]":
                call = f"nx_buffer_from_py({call})"
            return [f"static {self._cpp_type(res)} {emit_name}({params}) {{ return {call}; }}"]
        if self.fn.extern_symbol:
            params = ", ".join(f"{self._cpp_type(t)} nx_{n}" for n, t in self.fn.params) or "void"
            if self.fn.extern_symbol == emit_name:
                return []
            args = ", ".join(f"nx_{n}" for n, _ in self.fn.params)
            call = f"{self.fn.extern_symbol}({args})"
            stmt = f"{call}; return;" if self.fn.result == "void" else f"return {call};"
            return [f"static {self._cpp_type(self.fn.result)} {emit_name}({params}) {{ {stmt} }}"]

        params = ", ".join(f"{self._cpp_type(t)} nx_{n}" for n, t in self.fn.params) or "void"
        if self.compiler.debug:
            self.emit(f"NxTraceGuard __nx_trace({c_string(self.unit.name)}, {c_string(source_name)}, {self.unit.original_line(self.fn.node.lineno)});")
        always_ret = self.block(self.fn.node.body)

        inner_ret = future_element(self.fn.result) if self.fn.is_async else self.fn.result
        assert inner_ret is not None
        if inner_ret != "void" and not always_ret:
            raise self.error_cls(f"function {source_name} may finish without returning {inner_ret}", self.fn.node)
        if inner_ret == "void":
            self.emit("return;")
        elif inner_ret == "Result[void]" and not guaranteed_return(self.fn.node.body):
            self.emit("return nx_rt_ok_void();")

        if self.fn.is_async:
            # Wrap body inside NxHybridExecutor::instance().submit([=]() -> InnerRet { ... })
            ret_cpp = self._cpp_type(inner_ret)
            fut_cpp = self._cpp_type(self.fn.result)
            capture_args = ", ".join(f"nx_{n}" for n, _ in self.fn.params)
            body_lines = [f"    return NxHybridExecutor::instance().submit([{capture_args}]() -> {ret_cpp} {{", *("    " + ln for ln in self.lines), "    });"]
            linkage = "static"
            prefix = ["template <" + ", ".join(f"std::size_t nx_{p}" if b == "Int" else f"typename {p}" for p, b in self.fn.generic) + ">"] if self.fn.generic else []
            return [*prefix, f"{linkage} {fut_cpp} {emit_name}({params}) {{", *body_lines, "}"]

        if self.fn.generic:
            asserts = [
                f"static_assert({('nx_' + p + ' > 0 && nx_' + p + ' <= 1024') if b == 'Int' else ('nx_trait_' + b + '<' + p + '>::value') if b in self.compiler.traits else ('std::is_' + ('integral' if b == 'Integer' else 'arithmetic') + '_v<' + p + '> || nx_is_int128<' + p + '>::value')}, \"{p} must satisfy {b}\");"
                for p, b in self.fn.generic
            ]
            self.lines = ["    " + ln for ln in asserts] + self.lines
        linkage = 'extern "C" NX_EXPORT' if self.fn.exported else "static"
        prefix = ["template <" + ", ".join(f"std::size_t nx_{p}" if b == "Int" else f"typename {p}" for p, b in self.fn.generic) + ">"] if self.fn.generic else []
        return [*prefix, f"{linkage} {self._cpp_type(self.fn.result)} {emit_name}({params}) {{", *self.lines, "}"]
