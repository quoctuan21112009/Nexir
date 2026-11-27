"""pkg.go.dev-style Documentation Generator & Local Registry Server for NEXIR 2.0."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import partial
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

from .compiler import Compiler, NexirError


@dataclass
class DocSymbol:
    kind: str  # "function" | "class" | "enum" | "trait"
    name: str
    signature: str
    doc: str
    caps: list[str]
    exported: bool = False


@dataclass
class DocPackage:
    name: str
    version: str
    module_path: str
    description: str
    files: dict[str, list[DocSymbol]]


def extract_file_docs(nx_path: Path) -> list[DocSymbol]:
    source = nx_path.read_text(encoding="utf-8")
    comp = Compiler(source, entry_path=nx_path, require_main=False)
    unit = comp.entry_unit
    symbols: list[DocSymbol] = []

    for node in unit.tree.body:
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            decl = unit.declarations.get(node.lineno, "class")
            if decl == "trait":
                symbols.append(DocSymbol("trait", node.name, f"trait {node.name}", doc, []))
            elif decl == "enum":
                edef = unit.enums.get(node.name)
                variants = ", ".join(edef.variants.keys()) if edef else ""
                symbols.append(DocSymbol("enum", node.name, f"enum {node.name}({variants})", doc, []))
            else:
                cdef = unit.classes.get(node.name)
                fields = ", ".join(f"{fn}: {ft}" for fn, ft in (cdef.fields if cdef else ()))
                symbols.append(DocSymbol("class", node.name, f"class {node.name}({fields})", doc, []))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            fn = unit.functions.get(node.name)
            if fn:
                params = ", ".join(f"{n}: {t}" for n, t in fn.params)
                async_p = "async " if fn.is_async else ""
                sig = f"{async_p}def {node.name}({params}) -> {fn.result}"
                symbols.append(DocSymbol("function", node.name, sig, doc, sorted(fn.caps), fn.exported))
    return symbols


def generate_docs(project_root: Path, out_dir: Path, *, format_kind: str = "html") -> Path:
    """Generate pkg.go.dev-style documentation for a NEXIR project."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    manifest_path = project_root / "nx.toml"
    pkg_name = project_root.name
    pkg_ver = "0.1.0"
    mod_path = f"nx.dev/local/{pkg_name}"
    desc = "NEXIR native package"
    if manifest_path.is_file():
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        pkg = data.get("package", {})
        pkg_name = pkg.get("name", pkg_name)
        pkg_ver = pkg.get("version", pkg_ver)
        mod_path = pkg.get("module", f"nx.dev/local/{pkg_name}")
        desc = pkg.get("description", desc)

    nx_files = sorted(project_root.rglob("*.nx"))
    # Exclude build/cache directories
    nx_files = [p for p in nx_files if not any(part in {".git", ".build-env", "build", ".nx"} for part in p.parts)]
    files_docs: dict[str, list[DocSymbol]] = {}
    for nf in nx_files:
        rel = nf.relative_to(project_root).as_posix()
        try:
            files_docs[rel] = extract_file_docs(nf)
        except NexirError:
            files_docs[rel] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    if format_kind == "md":
        md_lines = [f"# {pkg_name} v{pkg_ver}", "", f"Module: `{mod_path}`", "", desc, ""]
        for rel, syms in files_docs.items():
            md_lines.append(f"## `{rel}`")
            for s in syms:
                caps = f" `@cap({', '.join(s.caps)})`" if s.caps else ""
                exp = " `[export]`" if s.exported else ""
                md_lines.append(f"- **{s.kind}** `{s.signature}`{caps}{exp}")
                if s.doc:
                    md_lines.append(f"  - {s.doc}")
            md_lines.append("")
        out_path = out_dir / "README.md"
        out_path.write_text("\n".join(md_lines), encoding="utf-8")
        return out_path

    # Generate pkg.go.dev-style HTML index
    nav_items = "".join(f'<li><a href="#file-{html.escape(rel)}">{html.escape(rel)}</a></li>' for rel in files_docs)
    sections = []
    for rel, syms in files_docs.items():
        cards = []
        for s in syms:
            badges = "".join(f'<span class="badge cap">@cap("{html.escape(c)}")</span>' for c in s.caps)
            if s.exported:
                badges += '<span class="badge export">@export</span>'
            doc_html = f'<p class="doc">{html.escape(s.doc)}</p>' if s.doc else ""
            cards.append(
                f'<div class="symbol" data-name="{html.escape(s.name.lower())}">'
                f'<div class="sym-head"><span class="kind">{html.escape(s.kind)}</span>'
                f'<code>{html.escape(s.signature)}</code>{badges}</div>{doc_html}</div>'
            )
        sections.append(f'<section id="file-{html.escape(rel)}"><h2>{html.escape(rel)}</h2>{"".join(cards) or "<p>No public symbols</p>"}</section>')

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(pkg_name)} v{html.escape(pkg_ver)} - pkg.nx.dev</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; color: #1f2937; background: #f8fafc; }}
header {{ background: #0f172a; color: #f8fafc; padding: 18px 28px; display: flex; justify-content: space-between; align-items: center; }}
header h1 {{ margin: 0; font-size: 20px; }}
.mod-path {{ color: #38bdf8; font-family: monospace; font-size: 14px; }}
.layout {{ display: grid; grid-template-columns: 260px 1fr; max-width: 1200px; margin: 24px auto; gap: 24px; padding: 0 20px; }}
aside {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; height: fit-content; }}
aside input {{ width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; margin-bottom: 12px; }}
aside ul {{ list-style: none; padding: 0; margin: 0; }}
aside li {{ margin: 6px 0; }}
aside a {{ color: #0284c7; text-decoration: none; font-family: monospace; font-size: 13px; }}
main {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 24px; }}
.symbol {{ border-bottom: 1px solid #f1f5f9; padding: 12px 0; }}
.sym-head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
.kind {{ font-size: 11px; text-transform: uppercase; font-weight: 700; background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; }}
.badge {{ font-size: 11px; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
.badge.cap {{ background: #fef3c7; color: #92400e; }}
.badge.export {{ background: #dcfce7; color: #166534; }}
code {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: 14px; color: #0f172a; }}
.doc {{ margin: 6px 0 0 0; color: #475569; font-size: 14px; }}
</style>
</head>
<body>
<header>
  <div><h1>{html.escape(pkg_name)} <small>v{html.escape(pkg_ver)}</small></h1><div class="mod-path">{html.escape(mod_path)}</div></div>
  <div>{html.escape(desc)}</div>
</header>
<div class="layout">
  <aside>
    <input id="filter" type="search" placeholder="Filter symbols..." oninput="for(const el of document.querySelectorAll('.symbol')) el.style.display=el.dataset.name.includes(this.value.toLowerCase())?'':'none'">
    <ul>{nav_items}</ul>
  </aside>
  <main>{"".join(sections)}</main>
</div>
</body>
</html>
"""
    out_path = out_dir / "index.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


def serve_docs(doc_dir: Path, port: int = 8080) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(doc_dir))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as srv:
        print(f"Serving NEXIR package docs at http://127.0.0.1:{port}/")
        srv.serve_forever()
