"""Interactive Python-like REPL for NEXIR 2.0 (`nx repl`)."""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile

from . import __version__
from .compiler import NexirError, compile_source
from .llvm_backend import compile_to_llvm_ir


def _is_incomplete(buf: str) -> bool:
    stripped = buf.rstrip()
    if stripped.endswith(":"):
        return True
    try:
        ast.parse(buf)
        return False
    except SyntaxError as exc:
        msg = str(exc.msg).lower()
        return any(tok in msg for tok in ("eof", "expected an indented block", "was never closed"))


def _classify_snippet(src: str) -> tuple[str, str]:
    """Return (kind, code): kind in {'import', 'def', 'stmt', 'expr'}."""
    tree = ast.parse(src)
    if len(tree.body) == 1:
        node = tree.body[0]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "import", src
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return "def", src
        if isinstance(node, ast.Expr) and not (
            isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "print"
        ):
            return "expr", src
    return "stmt", src


def run_repl(run_fn) -> int:
    """Launch an interactive NEXIR 2.0 REPL using `run_fn(source_path, ...)`."""
    print(f"NEXIR {__version__} Interactive REPL (type :help for commands, :quit to exit)")
    imports: list[str] = []
    defs: list[str] = []
    stmts: list[str] = []
    buf: list[str] = []

    while True:
        prompt = "...   " if buf else "nx>>> "
        try:
            line = input(prompt)
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt")
            buf.clear()
            continue

        s = line.strip()
        if not buf and s in {":quit", ":exit", "exit()", "quit()"}:
            break
        if not buf and s == ":help":
            print("Commands: :vars (show session), :clear (reset), :llvm (show LLVM IR), :cpp (show C++), :quit")
            continue
        if not buf and s == ":clear":
            imports.clear()
            defs.clear()
            stmts.clear()
            print("Session cleared.")
            continue
        if not buf and s == ":vars":
            print("\n".join([*imports, *defs, *stmts]) or "(empty session)")
            continue
        if not buf and s in {":llvm", ":cpp"}:
            prog = "\n".join([*imports, *defs, *stmts, "pass" if not stmts else ""])
            try:
                if s == ":llvm":
                    print(compile_to_llvm_ir(prog, require_main=True))
                else:
                    print(compile_source(prog, require_main=True)[-1500:])
            except NexirError as exc:
                print(f"Error: {exc}")
            continue

        if buf and s == "":
            src = "\n".join(buf)
            buf.clear()
        else:
            buf.append(line)
            src = "\n".join(buf)
            if _is_incomplete(src):
                continue
            buf.clear()

        if not src.strip():
            continue

        try:
            kind, code = _classify_snippet(src)
        except SyntaxError as exc:
            print(f"SyntaxError: {exc}")
            continue

        prev_len = len(stmts)
        if kind == "import":
            imports.append(code)
            prog = "\n".join([*imports, *defs, *stmts, "pass" if not stmts else ""])
        elif kind == "def":
            defs.append(code)
            prog = "\n".join([*imports, *defs, *stmts, "pass" if not stmts else ""])
        elif kind == "expr":
            prog = "\n".join([*imports, *defs, *stmts, f"print({code})"])
        else:
            stmts.append(code)
            prog = "\n".join([*imports, *defs, *stmts])

        try:
            compile_source(prog, require_main=True)
            if kind in {"stmt", "expr"}:
                with tempfile.TemporaryDirectory(prefix="nx-repl-") as td:
                    src_file = Path(td) / "repl.nx"
                    src_file.write_text(prog + "\n", encoding="utf-8")
                    run_fn(src_file)
        except Exception as exc:
            print(f"Error: {exc}")
            if kind == "import":
                imports.pop()
            elif kind == "def":
                defs.pop()
            elif kind == "stmt" and len(stmts) > prev_len:
                stmts.pop()
    return 0
