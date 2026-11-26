"""Strict NEXIR frontend and C++17 backend.

Python's AST supplies indentation and expression parsing during bootstrap. Every
accepted syntax node is checked below; arbitrary Python is never executed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from .runtime_cpp import PRELUDE


TYPES = {"i32": "int32_t", "i64": "int64_t", "f32": "float", "f64": "double", "bool": "bool", "str": "const char *", "void": "void", "dyn": "NxValue", "Ring[f64]": "nx_ring_ref_f64"}
for _element in ("i32", "i64", "f32", "f64"):
    TYPES[f"Span[{_element}]"] = f"nx_span_{_element}"
    TYPES[f"MutSpan[{_element}]"] = f"nx_mutspan_{_element}"
INTEGERS = {"i32", "i64"}
NUMERICS = INTEGERS | {"f32", "f64"}
CAPABILITIES = {"filesystem.read", "network.connect", "memory.alloc"}


def _span_element(typ: str) -> str | None:
    if typ.startswith("Span[") or typ.startswith("MutSpan["):
        return typ[typ.index("[") + 1:-1]
    return None


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


def _type(node: ast.expr | None, *, default: str = "void") -> str:
    if node is None:
        return default
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Name):
        typ = f"{node.value.id}[{node.slice.id}]"
        if typ in TYPES:
            return typ
    if not isinstance(node, ast.Name) or node.id not in TYPES:
        raise NexirError("expected one of: " + ", ".join(TYPES), node)
    return node.id


def _compatible(actual: str, expected: str) -> bool:
    return (actual == expected or (expected == "dyn" and actual in NUMERICS | {"str", "bool"})
            or (actual, expected) in {("i32", "i64"), ("f32", "f64")}
            or actual.startswith("MutSpan[") and expected == "Span[" + actual[8:])


def _coerce(value: str, actual: str, expected: str) -> str:
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
    if not name.isidentifier() or name.startswith("nx_") or name in TYPES or name in {"print", "range", "len", "zeros_f32", "zeros_f64", "read_text", "tcp_connect", "type_name", "as_i64", "as_f64", "ring_f64", "ring_write", "ring_read", "ring_valid"}:
        raise NexirError(f"reserved name: {name}", node)
    return "nx_" + name


class Compiler:
    def __init__(self, source: str):
        try:
            self.tree = ast.parse(source)
        except SyntaxError as exc:
            raise NexirError(f"syntax error: {exc.msg} at line {exc.lineno}, column {exc.offset}") from exc
        self.functions: dict[str, Function] = {}
        for node in self.tree.body:
            if not isinstance(node, ast.FunctionDef):
                raise NexirError("only function definitions are allowed at module level", node)
            _name(node.name, node)
            if node.name in self.functions:
                raise NexirError(f"duplicate function: {node.name}", node)
            caps: set[str] = set()
            exported = False
            extern_symbol = None
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
                else:
                    raise NexirError("supported decorators: @native, @export, @cap(...), @extern(...) only", decorator)
                if label in seen:
                    raise NexirError(f"duplicate @{label}", decorator)
                seen.add(label)
            args = node.args
            if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults or args.kw_defaults:
                raise NexirError("only required positional parameters are supported", node)
            params = []
            for arg in args.args:
                _name(arg.arg, arg)
                typ = _type(arg.annotation, default="")
                if typ in {"", "void"}:
                    raise NexirError("parameters require a non-void type", arg)
                if arg.arg in [n for n, _ in params]:
                    raise NexirError(f"duplicate parameter: {arg.arg}", arg)
                params.append((arg.arg, typ))
            result = _type(node.returns)
            if _span_element(result):
                raise NexirError("returning Span is not supported until lifetime checking is available", node.returns)
            if exported and (result not in NUMERICS | {"bool", "void"} or any(t not in NUMERICS | {"bool"} and not _span_element(t) for _, t in params)):
                raise NexirError("@export supports numeric, bool and Span signatures only", node)
            if extern_symbol and (exported or caps or result == "dyn" or any(t == "dyn" or t == "Ring[f64]" for _, t in params)):
                raise NexirError("@extern requires a plain C ABI signature without @export or @cap", node)
            if extern_symbol and (len(node.body) != 1 or not isinstance(node.body[0], ast.Pass)):
                raise NexirError("@extern declaration body must be pass", node)
            self.functions[node.name] = Function(node, tuple(params), result, frozenset(caps), exported, extern_symbol)
        if "main" not in self.functions:
            raise NexirError("program requires def main()")
        main = self.functions["main"]
        if main.params or main.result not in {"void", "i32"}:
            raise NexirError("main must take no arguments and return void or i32", main.node)
    def compile(self) -> str:
        out = ["/* Generated by NEXIR C++17 backend. Do not edit. */", PRELUDE, ""]
        for name, fn in self.functions.items():
            params = ", ".join(f"{TYPES[t]} nx_{n}" for n, t in fn.params) or "void"
            if fn.extern_symbol:
                out.append(f'extern "C" {TYPES[fn.result]} {fn.extern_symbol}({params});')
                if fn.extern_symbol == f"nx_{name}":
                    continue
            linkage = 'extern "C" NX_EXPORT' if fn.exported else "static"
            out.append(f"{linkage} {TYPES[fn.result]} nx_{name}({params});")
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

    def emit(self, line: str) -> None:
        self.lines.append("    " * self.depth + line)

    def lookup(self, name: str, node: ast.AST) -> str:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise NexirError(f"unknown variable: {name}", node)

    def require_caps(self, required: set[str] | frozenset[str], node: ast.AST) -> None:
        missing = set(required) - self.fn.caps
        if missing:
            raise NexirError(f"function {self.fn.node.name} needs @cap({', '.join(repr(c) for c in sorted(missing))})", node)

    def index(self, node: ast.Subscript, *, mutable: bool = False) -> tuple[str, str]:
        container, typ = self.expr(node.value)
        element = _span_element(typ)
        if element is None or (mutable and not typ.startswith("MutSpan[")):
            raise NexirError("indexing requires Span; assignment requires MutSpan", node)
        index, index_type = self.expr(node.slice)
        if index_type not in INTEGERS:
            raise NexirError("array index must be an integer", node.slice)
        access = "nx_at_mut" if typ.startswith("MutSpan[") else "nx_at"
        return f"(*{access}_{element}({container}, (int64_t)({index})))", element

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
        if isinstance(node, ast.List):
            return self.array_literal(node)
        if isinstance(node, ast.Constant):
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
        if isinstance(node, ast.Subscript):
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
            if isinstance(node.op, (ast.UAdd, ast.USub)) and typ in NUMERICS:
                sign = "+" if isinstance(node.op, ast.UAdd) else "-"
                return (f"({sign}{value})", typ)
            raise NexirError("invalid unary operation", node)
        if isinstance(node, ast.BinOp):
            left, lt = self.expr(node.left)
            right, rt = self.expr(node.right)
            if "dyn" in {lt, rt}:
                if not isinstance(node.op, ast.Add) or not _compatible(lt, "dyn") or not _compatible(rt, "dyn"):
                    raise NexirError("dynamic values currently support + only", node)
                return (f"nx_add_dyn({_coerce(left, lt, 'dyn')}, {_coerce(right, rt, 'dyn')})", "dyn")
            if lt not in NUMERICS or rt not in NUMERICS:
                raise NexirError("arithmetic requires numeric operands", node)
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%"}.get(type(node.op))
            if op is None or (op == "%" and (lt not in INTEGERS or rt not in INTEGERS)):
                raise NexirError("unsupported arithmetic operation", node)
            if op in {"/", "%"} and isinstance(node.right, ast.Constant) and node.right.value == 0:
                raise NexirError("division by zero", node.right)
            typ = "f64" if "f64" in {lt, rt} else "f32" if "f32" in {lt, rt} else "i64" if "i64" in {lt, rt} else "i32"
            return (f"({left} {op} {right})", typ)
        if isinstance(node, ast.BoolOp):
            op = " && " if isinstance(node.op, ast.And) else " || " if isinstance(node.op, ast.Or) else None
            vals = [self.expr(v) for v in node.values]
            if op is None or any(t != "bool" for _, t in vals):
                raise NexirError("boolean operations require bool operands", node)
            return ("(" + op.join(c for c, _ in vals) + ")", "bool")
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise NexirError("chained comparisons are not supported", node)
            left, lt = self.expr(node.left)
            right, rt = self.expr(node.comparators[0])
            op = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}.get(type(node.ops[0]))
            if op is None or not (lt == rt or lt in NUMERICS and rt in NUMERICS):
                raise NexirError("incompatible comparison", node)
            if lt == "str":
                if op not in {"==", "!="}:
                    raise NexirError("strings support == and != only", node)
                return (f"(strcmp({left}, {right}) {op} 0)", "bool")
            return (f"({left} {op} {right})", "bool")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise NexirError("only positional calls to named functions are supported", node)
            name = node.func.id
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
                if actual not in NUMERICS:
                    raise NexirError("numeric cast requires a number", node.args[0])
                return f"(({TYPES[name]})({value}))", name
            if name == "len":
                if len(node.args) != 1:
                    raise NexirError("len expects one Span", node)
                value, typ = self.expr(node.args[0])
                if not _span_element(typ):
                    raise NexirError("len requires Span or MutSpan", node.args[0])
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
            if len(node.args) != len(fn.params):
                raise NexirError(f"{name} expects {len(fn.params)} arguments", node)
            args = []
            for arg, (_, expected) in zip(node.args, fn.params):
                value, actual = self.expr(arg)
                if not _compatible(actual, expected):
                    raise NexirError(f"expected {expected}, got {actual}", arg)
                args.append(_coerce(value, actual, expected))
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
        self.scopes.pop()
        return result

    def stmt(self, node: ast.stmt) -> bool:
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
            name = node.target.id
            _name(name, node)
            if any(name in scope for scope in self.scopes):
                raise NexirError(f"variable already defined: {name}", node)
            typ = _type(node.annotation)
            if typ == "void":
                raise NexirError("variable cannot have void type", node)
            value, actual = self.expr(node.value)
            if actual == "void":
                raise NexirError("cannot assign a void result", node.value)
            if not _compatible(actual, typ):
                raise NexirError(f"expected {typ}, got {actual}", node.value)
            self.scopes[-1][name] = typ
            self.emit(f"{TYPES[typ]} nx_{name} = {_coerce(value, actual, typ)};")
            return False
        if isinstance(node, ast.Assign):
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
            value, actual = self.expr(node.value)
            if actual == "void":
                raise NexirError("cannot assign a void result", node.value)
            try:
                expected = self.lookup(name, node)
            except NexirError:
                self.scopes[-1][name] = actual
                self.emit(f"{TYPES[actual]} nx_{name} = {value};")
                return False
            if not _compatible(actual, expected):
                raise NexirError(f"expected {expected}, got {actual}", node.value)
            if _span_element(expected) and name not in self.scopes[-1]:
                raise NexirError("cannot rebind an outer Span from a nested block", node)
            self.emit(f"nx_{name} = {_coerce(value, actual, expected)};")
            return False
        if isinstance(node, ast.AugAssign):
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
                if not _compatible(actual, self.fn.result):
                    raise NexirError(f"return expected {self.fn.result}, got {actual}", node)
                self.emit(f"return {_coerce(value, actual, self.fn.result)};")
            return True
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
            if node.orelse or not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range" or node.iter.keywords:
                raise NexirError("for supports only 'for name in range(...)'", node)
            name = node.target.id
            _name(name, node.target)
            if any(name in scope for scope in self.scopes):
                raise NexirError(f"loop variable already defined: {name}", node.target)
            args = [self.expr(a) for a in node.iter.args]
            if len(args) not in {1, 2, 3} or any(t not in INTEGERS for _, t in args):
                raise NexirError("range expects one to three integer arguments", node.iter)
            start, stop, step = ("0", args[0][0], "1") if len(args) == 1 else (args[0][0], args[1][0], args[2][0] if len(args) == 3 else "1")
            self.serial += 1
            suffix = self.serial
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
            self.scopes.pop()
            self.emit("}")
            return False
        raise NexirError(f"unsupported statement: {type(node).__name__}", node)

    def generate(self, name: str) -> list[str]:
        params = ", ".join(f"{TYPES[t]} nx_{n}" for n, t in self.fn.params) or "void"
        if self.fn.extern_symbol:
            if self.fn.extern_symbol == f"nx_{name}":
                return []
            args = ", ".join(f"nx_{n}" for n, _ in self.fn.params)
            call = f"{self.fn.extern_symbol}({args})"
            statement = f"{call}; return;" if self.fn.result == "void" else f"return {call};"
            return [f"static {TYPES[self.fn.result]} nx_{name}({params}) {{ {statement} }}"]
        self.block(self.fn.node.body)
        if self.fn.result != "void" and not _guaranteed_return(self.fn.node.body):
            raise NexirError(f"function {name} may finish without returning {self.fn.result}", self.fn.node)
        if self.fn.result == "void":
            self.emit("return;")
        linkage = 'extern "C" NX_EXPORT' if self.fn.exported else "static"
        return [f"{linkage} {TYPES[self.fn.result]} nx_{name}({params}) {{", *self.lines, "}"]


def _guaranteed_return(nodes: list[ast.stmt]) -> bool:
    for node in nodes:
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, ast.If) and node.orelse and _guaranteed_return(node.body) and _guaranteed_return(node.orelse):
            return True
    return False


def compile_source(source: str) -> str:
    return Compiler(source).compile()


def compile_file(path: Path) -> str:
    if path.suffix != ".nx":
        raise NexirError("source file must end in .nx")
    try:
        return compile_source(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise NexirError("source file must be UTF-8") from exc
