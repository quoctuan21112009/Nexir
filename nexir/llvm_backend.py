"""Independent LLVM IR backend for NEXIR 2.0 (emits standalone .ll IR and compiles via clang/llc when available)."""

from __future__ import annotations

import ast
import platform
import shutil
import subprocess
from pathlib import Path

from .compiler import Compiler, Function, ModuleUnit, NexirError
from .typesystem import INTEGERS, NUMERICS, SIGNED_INTEGERS, UNSIGNED_INTEGERS


def _llvm_target_triple() -> str:
    sys_name = platform.system().lower()
    mach = platform.machine().lower()
    arch = "aarch64" if ("arm" in mach or "aarch64" in mach) else "x86_64"
    if sys_name == "windows":
        return f"{arch}-w64-mingw32"
    if sys_name == "darwin":
        return f"{arch}-apple-darwin"
    return f"{arch}-unknown-linux-gnu"


def _llvm_type(nx_type: str) -> str:
    mapping = {
        "void": "void",
        "bool": "i1",
        "i8": "i8",
        "u8": "i8",
        "i16": "i16",
        "u16": "i16",
        "i32": "i32",
        "u32": "i32",
        "i64": "i64",
        "u64": "i64",
        "i128": "i128",
        "u128": "i128",
        "f32": "float",
        "f64": "double",
        "str": "i8*",
    }
    if nx_type in mapping:
        return mapping[nx_type]
    return "i8*"


def _int_bits(nx_type: str) -> int:
    if nx_type in {"i8", "u8"}:
        return 8
    if nx_type in {"i16", "u16"}:
        return 16
    if nx_type in {"i32", "u32"}:
        return 32
    if nx_type in {"i64", "u64"}:
        return 64
    if nx_type in {"i128", "u128"}:
        return 128
    if nx_type == "bool":
        return 1
    return 64


def _c_escape_llvm(text: str) -> tuple[str, int]:
    raw = text.encode("utf-8") + b"\x00"
    out = []
    for b in raw:
        if 32 <= b <= 126 and b not in (34, 92):
            out.append(chr(b))
        else:
            out.append(f"\\{b:02X}")
    return "".join(out), len(raw)


class _LLVMFuncEmitter:
    def __init__(self, mod: "LLVMModuleEmitter", unit: ModuleUnit, fn: Function) -> None:
        self.mod = mod
        self.unit = unit
        self.fn = fn
        self.lines: list[str] = []
        self.tmp_id = 0
        self.lbl_id = 0
        self.locals: dict[str, tuple[str, str]] = {}  # name -> (ptr_reg, nx_type)
        self.loop_stack: list[tuple[str, str]] = []  # (continue_lbl, break_lbl)
        self.terminated = False

    def new_reg(self, prefix: str = "t") -> str:
        self.tmp_id += 1
        return f"%{prefix}{self.tmp_id}"

    def new_lbl(self, prefix: str = "L") -> str:
        self.lbl_id += 1
        return f"{prefix}{self.lbl_id}"

    def emit(self, line: str) -> None:
        if not self.terminated:
            self.lines.append(f"  {line}")

    def emit_label(self, lbl: str) -> None:
        self.lines.append(f"{lbl}:")
        self.terminated = False

    def cast(self, reg: str, src_t: str, dst_t: str) -> str:
        if src_t == dst_t:
            return reg
        lt_src = _llvm_type(src_t)
        lt_dst = _llvm_type(dst_t)
        if lt_src == lt_dst:
            return reg
        out = self.new_reg("cvt")
        if src_t in INTEGERS | {"bool"} and dst_t in INTEGERS | {"bool"}:
            sb, db = _int_bits(src_t), _int_bits(dst_t)
            if dst_t == "bool":
                self.emit(f"{out} = icmp ne {lt_src} {reg}, 0")
            elif db > sb:
                op = "zext" if (src_t in UNSIGNED_INTEGERS or src_t == "bool") else "sext"
                self.emit(f"{out} = {op} {lt_src} {reg} to {lt_dst}")
            else:
                self.emit(f"{out} = trunc {lt_src} {reg} to {lt_dst}")
            return out
        if src_t in INTEGERS and dst_t in {"f32", "f64"}:
            op = "uitofp" if src_t in UNSIGNED_INTEGERS else "sitofp"
            self.emit(f"{out} = {op} {lt_src} {reg} to {lt_dst}")
            return out
        if src_t in {"f32", "f64"} and dst_t in INTEGERS:
            op = "fptoui" if dst_t in UNSIGNED_INTEGERS else "fptosi"
            self.emit(f"{out} = {op} {lt_src} {reg} to {lt_dst}")
            return out
        if src_t == "f32" and dst_t == "f64":
            self.emit(f"{out} = fpext float {reg} to double")
            return out
        if src_t == "f64" and dst_t == "f32":
            self.emit(f"{out} = fptrunc double {reg} to float")
            return out
        return reg

    def expr(self, node: ast.expr, hint: str | None = None) -> tuple[str, str]:
        if isinstance(node, ast.Constant):
            v = node.value
            if type(v) is bool:
                return ("1" if v else "0"), "bool"
            if type(v) is int:
                t = hint if hint in INTEGERS else "i64"
                return str(v), t
            if type(v) is float:
                t = hint if hint in {"f32", "f64"} else "f64"
                return repr(float(v)), t
            if type(v) is str:
                return self.mod.intern_str(v), "str"
        if isinstance(node, ast.Name):
            if node.id in self.locals:
                ptr, t = self.locals[node.id]
                r = self.new_reg("v")
                self.emit(f"{r} = load {_llvm_type(t)}, {_llvm_type(t)}* {ptr}")
                return r, t
            if node.id in self.unit.globals:
                t, _ = self.unit.globals[node.id]
                r = self.new_reg("g")
                self.emit(f"{r} = load {_llvm_type(t)}, {_llvm_type(t)}* @{self.unit.cpp_symbol(node.id)}")
                return r, t
            raise NexirError(f"LLVM backend: unknown identifier {node.id}", node)
        if isinstance(node, ast.UnaryOp):
            r, t = self.expr(node.operand, hint)
            out = self.new_reg("u")
            if isinstance(node.op, ast.USub):
                if t in {"f32", "f64"}:
                    self.emit(f"{out} = fneg {_llvm_type(t)} {r}")
                else:
                    self.emit(f"{out} = sub {_llvm_type(t)} 0, {r}")
                return out, t
            if isinstance(node.op, ast.Not):
                rb = self.cast(r, t, "bool")
                self.emit(f"{out} = xor i1 {rb}, 1")
                return out, "bool"
        if isinstance(node, ast.BinOp):
            lr, lt = self.expr(node.left, hint)
            rr, rt = self.expr(node.right, lt if lt in NUMERICS else hint)
            res_t = "f64" if "f64" in {lt, rt} else ("f32" if "f32" in {lt, rt} else lt)
            lr = self.cast(lr, lt, res_t)
            rr = self.cast(rr, rt, res_t)
            out = self.new_reg("b")
            lt_ll = _llvm_type(res_t)
            is_fp = res_t in {"f32", "f64"}
            if isinstance(node.op, ast.Add):
                self.emit(f"{out} = {'fadd' if is_fp else 'add'} {lt_ll} {lr}, {rr}")
            elif isinstance(node.op, ast.Sub):
                self.emit(f"{out} = {'fsub' if is_fp else 'sub'} {lt_ll} {lr}, {rr}")
            elif isinstance(node.op, ast.Mult):
                self.emit(f"{out} = {'fmul' if is_fp else 'mul'} {lt_ll} {lr}, {rr}")
            elif isinstance(node.op, ast.Div):
                if is_fp:
                    self.emit(f"{out} = fdiv {lt_ll} {lr}, {rr}")
                else:
                    op = "udiv" if res_t in UNSIGNED_INTEGERS else "sdiv"
                    self.emit(f"{out} = {op} {lt_ll} {lr}, {rr}")
            elif isinstance(node.op, ast.Mod):
                op = "frem" if is_fp else ("urem" if res_t in UNSIGNED_INTEGERS else "srem")
                self.emit(f"{out} = {op} {lt_ll} {lr}, {rr}")
            else:
                raise NexirError(f"LLVM backend: unsupported binary op {type(node.op).__name__}", node)
            return out, res_t
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            lr, lt = self.expr(node.left)
            rr, rt = self.expr(node.right, lt)
            cmp_t = "f64" if "f64" in {lt, rt} else ("f32" if "f32" in {lt, rt} else lt)
            lr = self.cast(lr, lt, cmp_t)
            rr = self.cast(rr, rt, cmp_t)
            out = self.new_reg("cmp")
            op = node.ops[0]
            if cmp_t in {"f32", "f64"}:
                fmap = {ast.Eq: "oeq", ast.NotEq: "one", ast.Lt: "olt", ast.LtE: "ole", ast.Gt: "ogt", ast.GtE: "oge"}
                self.emit(f"{out} = fcmp {fmap[type(op)]} {_llvm_type(cmp_t)} {lr}, {rr}")
            else:
                u = cmp_t in UNSIGNED_INTEGERS
                imap = {
                    ast.Eq: "eq", ast.NotEq: "ne",
                    ast.Lt: "ult" if u else "slt", ast.LtE: "ule" if u else "sle",
                    ast.Gt: "ugt" if u else "sgt", ast.GtE: "uge" if u else "sge",
                }
                self.emit(f"{out} = icmp {imap[type(op)]} {_llvm_type(cmp_t)} {lr}, {rr}")
            return out, "bool"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname == "print":
                for idx, a in enumerate(node.args):
                    ar, at = self.expr(a)
                    if idx > 0:
                        sp = self.mod.intern_str(" ")
                        self.emit(f"call i32 (i8*, ...) @printf(i8* {sp})")
                    if at == "str":
                        fmt = self.mod.intern_str("%s")
                        self.emit(f"call i32 (i8*, ...) @printf(i8* {fmt}, i8* {ar})")
                    elif at in {"f32", "f64"}:
                        ar64 = self.cast(ar, at, "f64")
                        fmt = self.mod.intern_str("%.17g")
                        self.emit(f"call i32 (i8*, ...) @printf(i8* {fmt}, double {ar64})")
                    elif at == "bool":
                        tstr = self.mod.intern_str("true")
                        fstr = self.mod.intern_str("false")
                        sel = self.new_reg("bstr")
                        self.emit(f"{sel} = select i1 {ar}, i8* {tstr}, i8* {fstr}")
                        fmt = self.mod.intern_str("%s")
                        self.emit(f"call i32 (i8*, ...) @printf(i8* {fmt}, i8* {sel})")
                    else:
                        ar64 = self.cast(ar, at, "i64")
                        fmt = self.mod.intern_str("%lld")
                        self.emit(f"call i32 (i8*, ...) @printf(i8* {fmt}, i64 {ar64})")
                nl = self.mod.intern_str("\n")
                self.emit(f"call i32 (i8*, ...) @printf(i8* {nl})")
                return "0", "void"
            if fname in self.unit.functions:
                callee = self.unit.functions[fname]
                cargs = []
                for a, (_, pt) in zip(node.args, callee.params):
                    ar, at = self.expr(a, pt)
                    ar = self.cast(ar, at, pt)
                    cargs.append(f"{_llvm_type(pt)} {ar}")
                sym = f"nx_{fname}" if (self.unit is self.mod.compiler.entry_unit or callee.exported) else self.unit.cpp_symbol(fname)
                if callee.result == "void":
                    self.emit(f"call void @{sym}({', '.join(cargs)})")
                    return "0", "void"
                out = self.new_reg("call")
                self.emit(f"{out} = call {_llvm_type(callee.result)} @{sym}({', '.join(cargs)})")
                return out, callee.result
        raise NexirError(f"LLVM backend: unsupported expression {type(node).__name__}", node)

    def stmt(self, node: ast.stmt) -> None:
        if self.terminated:
            return
        if isinstance(node, ast.Pass):
            return
        if isinstance(node, ast.Expr):
            self.expr(node.value)
            return
        if isinstance(node, ast.Return):
            if node.value is None or self.fn.result == "void":
                self.emit("ret void")
            else:
                r, t = self.expr(node.value, self.fn.result)
                r = self.cast(r, t, self.fn.result)
                self.emit(f"ret {_llvm_type(self.fn.result)} {r}")
            self.terminated = True
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if not isinstance(target, ast.Name) or node.value is None:
                raise NexirError("LLVM backend supports simple variable assignment", node)
            r, t = self.expr(node.value)
            if target.id not in self.locals:
                ptr = self.new_reg(f"var_{target.id}_")
                self.emit(f"{ptr} = alloca {_llvm_type(t)}")
                self.locals[target.id] = (ptr, t)
            ptr, vt = self.locals[target.id]
            r = self.cast(r, t, vt)
            self.emit(f"store {_llvm_type(vt)} {r}, {_llvm_type(vt)}* {ptr}")
            return
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            ptr, vt = self.locals[node.target.id]
            cur = self.new_reg("cur")
            self.emit(f"{cur} = load {_llvm_type(vt)}, {_llvm_type(vt)}* {ptr}")
            rr, rt = self.expr(node.value, vt)
            rr = self.cast(rr, rt, vt)
            out = self.new_reg("aug")
            is_fp = vt in {"f32", "f64"}
            op_map = {ast.Add: "fadd" if is_fp else "add", ast.Sub: "fsub" if is_fp else "sub", ast.Mult: "fmul" if is_fp else "mul"}
            self.emit(f"{out} = {op_map[type(node.op)]} {_llvm_type(vt)} {cur}, {rr}")
            self.emit(f"store {_llvm_type(vt)} {out}, {_llvm_type(vt)}* {ptr}")
            return
        if isinstance(node, ast.If):
            cr, ct = self.expr(node.test, "bool")
            cr = self.cast(cr, ct, "bool")
            lt, lf, le = self.new_lbl("if_then"), self.new_lbl("if_else"), self.new_lbl("if_end")
            self.emit(f"br i1 {cr}, label %{lt}, label %{lf}")
            self.emit_label(lt)
            for s in node.body:
                self.stmt(s)
            if not self.terminated:
                self.emit(f"br label %{le}")
            self.emit_label(lf)
            for s in node.orelse:
                self.stmt(s)
            if not self.terminated:
                self.emit(f"br label %{le}")
            self.emit_label(le)
            return
        if isinstance(node, ast.While):
            lc, lb, le = self.new_lbl("wh_cond"), self.new_lbl("wh_body"), self.new_lbl("wh_end")
            self.emit(f"br label %{lc}")
            self.emit_label(lc)
            cr, ct = self.expr(node.test, "bool")
            cr = self.cast(cr, ct, "bool")
            self.emit(f"br i1 {cr}, label %{lb}, label %{le}")
            self.emit_label(lb)
            self.loop_stack.append((lc, le))
            for s in node.body:
                self.stmt(s)
            self.loop_stack.pop()
            if not self.terminated:
                self.emit(f"br label %{lc}")
            self.emit_label(le)
            return
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            args = node.iter.args
            start_r, stop_r = ("0", "i64"), ("0", "i64")
            if len(args) == 1:
                stop_r = self.expr(args[0], "i64")
            else:
                start_r = self.expr(args[0], "i64")
                stop_r = self.expr(args[1], "i64")
            s_reg = self.cast(start_r[0], start_r[1], "i64")
            e_reg = self.cast(stop_r[0], stop_r[1], "i64")
            if node.target.id not in self.locals:
                ptr = self.new_reg(f"iv_{node.target.id}_")
                self.emit(f"{ptr} = alloca i64")
                self.locals[node.target.id] = (ptr, "i64")
            ptr, _ = self.locals[node.target.id]
            self.emit(f"store i64 {s_reg}, i64* {ptr}")
            lc, lb, li, le = self.new_lbl("for_c"), self.new_lbl("for_b"), self.new_lbl("for_inc"), self.new_lbl("for_e")
            self.emit(f"br label %{lc}")
            self.emit_label(lc)
            iv = self.new_reg("iv")
            self.emit(f"{iv} = load i64, i64* {ptr}")
            cond = self.new_reg("fcmp")
            self.emit(f"{cond} = icmp slt i64 {iv}, {e_reg}")
            self.emit(f"br i1 {cond}, label %{lb}, label %{le}")
            self.emit_label(lb)
            self.loop_stack.append((li, le))
            for s in node.body:
                self.stmt(s)
            self.loop_stack.pop()
            if not self.terminated:
                self.emit(f"br label %{li}")
            self.emit_label(li)
            cur = self.new_reg("iv2")
            nxt = self.new_reg("ivn")
            self.emit(f"{cur} = load i64, i64* {ptr}")
            self.emit(f"{nxt} = add i64 {cur}, 1")
            self.emit(f"store i64 {nxt}, i64* {ptr}")
            self.emit(f"br label %{lc}")
            self.emit_label(le)
            return
        if isinstance(node, ast.Break) and self.loop_stack:
            self.emit(f"br label %{self.loop_stack[-1][1]}")
            self.terminated = True
            return
        if isinstance(node, ast.Continue) and self.loop_stack:
            self.emit(f"br label %{self.loop_stack[-1][0]}")
            self.terminated = True
            return
        raise NexirError(f"LLVM backend: unsupported statement {type(node).__name__}", node)

    def emit_function(self, sym: str) -> list[str]:
        ret_ll = _llvm_type(self.fn.result)
        params_ll = ", ".join(f"{_llvm_type(t)} %arg_{n}" for n, t in self.fn.params)
        out = [f"define {ret_ll} @{sym}({params_ll}) {{", "entry:"]
        for n, t in self.fn.params:
            ptr = self.new_reg(f"p_{n}_")
            self.emit(f"{ptr} = alloca {_llvm_type(t)}")
            self.emit(f"store {_llvm_type(t)} %arg_{n}, {_llvm_type(t)}* {ptr}")
            self.locals[n] = (ptr, t)
        for s in self.fn.node.body:
            self.stmt(s)
        if not self.terminated:
            if self.fn.result == "void":
                self.emit("ret void")
            elif self.fn.result in {"f32", "f64"}:
                self.emit(f"ret {ret_ll} 0.0")
            else:
                self.emit(f"ret {ret_ll} 0")
        out.extend(self.lines)
        out.append("}")
        return out


class LLVMModuleEmitter:
    def __init__(self, compiler: Compiler) -> None:
        self.compiler = compiler
        self.strings: dict[str, tuple[str, int, str]] = {}
        self.str_idx = 0

    def intern_str(self, text: str) -> str:
        if text not in self.strings:
            self.str_idx += 1
            sym = f"@.str.{self.str_idx}"
            esc, nbytes = _c_escape_llvm(text)
            self.strings[text] = (sym, nbytes, esc)
        sym, nbytes, _ = self.strings[text]
        return f"getelementptr inbounds ([{nbytes} x i8], [{nbytes} x i8]* {sym}, i64 0, i64 0)"

    def emit(self) -> str:
        fn_blocks: list[str] = []
        for unit in self.compiler._unit_order:
            for name, fn in unit.functions.items():
                if unit is not self.compiler.entry_unit and name == "main":
                    continue
                sym = f"nx_{name}" if (unit is self.compiler.entry_unit or fn.exported) else unit.cpp_symbol(name)
                em = _LLVMFuncEmitter(self, unit, fn)
                fn_blocks.extend(em.emit_function(sym))
                fn_blocks.append("")

        # Top-level script or main wrapper
        if "main" in self.compiler.entry_unit.functions and not self.compiler.entry_unit.top_stmts:
            mres = self.compiler.entry_unit.functions["main"].result
            if mres == "i32":
                fn_blocks.extend(["define i32 @main() {", "entry:", "  %rc = call i32 @nx_main()", "  ret i32 %rc", "}"])
            else:
                fn_blocks.extend(["define i32 @main() {", "entry:", "  call void @nx_main()", "  ret i32 0", "}"])
        elif self.compiler.entry_unit.top_stmts:
            synth = Function(
                ast.FunctionDef(name="main", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=list(self.compiler.entry_unit.top_stmts), decorator_list=[]),
                (), "i32", frozenset(), False,
            )
            em = _LLVMFuncEmitter(self, self.compiler.entry_unit, synth)
            fn_blocks.extend(em.emit_function("main"))

        header = [
            "; Generated by NEXIR 2.0 Independent LLVM IR Backend",
            f'target triple = "{_llvm_target_triple()}"',
            "",
            "declare i32 @printf(i8*, ...)",
            "declare i32 @puts(i8*)",
            "declare i8* @malloc(i64)",
            "declare void @free(i8*)",
            "",
        ]
        for unit in self.compiler._unit_order:
            for gname, (gt, _) in unit.globals.items():
                zero = "0.0" if gt in {"f32", "f64"} else ("null" if gt == "str" else "0")
                header.append(f"@{unit.cpp_symbol(gname)} = internal global {_llvm_type(gt)} {zero}")
        for _, (sym, nbytes, esc) in self.strings.items():
            header.append(f'{sym} = private unnamed_addr constant [{nbytes} x i8] c"{esc}", align 1')
        header.append("")
        return "\n".join(header + fn_blocks) + "\n"


def compile_to_llvm_ir(source: str, *, entry_path: Path | None = None, require_main: bool = True) -> str:
    comp = Compiler(source, entry_path=entry_path, require_main=require_main)
    return LLVMModuleEmitter(comp).emit()


def find_llvm_tools() -> tuple[str | None, str | None]:
    return shutil.which("clang"), shutil.which("llc")


def build_with_llvm(ll_path: Path, output_path: Path, *, emit_asm: bool = False, opt_level: str = "2") -> Path:
    clang, llc = find_llvm_tools()
    if emit_asm:
        if llc:
            subprocess.run([llc, f"-O{opt_level}", "-filetype=asm", str(ll_path), "-o", str(output_path)], check=True)
            return output_path
        if clang:
            subprocess.run([clang, f"-O{opt_level}", "-S", str(ll_path), "-o", str(output_path)], check=True)
            return output_path
        raise NexirError("LLVM backend requested ASM output, but neither `llc` nor `clang` was found in PATH")
    if clang:
        subprocess.run([clang, f"-O{opt_level}", str(ll_path), "-o", str(output_path)], check=True)
        return output_path
    raise NexirError("LLVM backend requires `clang` (or `llc`) in PATH to link native executables; use `--emit-llvm` to inspect generated .ll IR")
