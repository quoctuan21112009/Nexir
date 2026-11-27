"""Command line interface for the NEXIR 2.0 native compiler, debugger, sandbox & package manager."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from . import __version__
from .build_cache import cached_object
from .compiler import NexirError, compile_file
from .docs_gen import generate_docs, serve_docs
from .ecosystem import ecosystem_command
from .interop import generate_bindings
from .packages import cpp_prefix, default_triplet, package_command, python_info
from .python_bridge import PYTHON_BRIDGE_CPP
from .security import CompileAuditLog


def _compiler() -> str:
    explicit = os.environ.get("NEXIR_CXX")
    if explicit:
        found = shutil.which(explicit)
        if found:
            return found
        if Path(explicit).is_file():
            return str(Path(explicit).resolve())
        raise NexirError(f"NEXIR_CXX compiler not found: {explicit}")
    for name in ("clang++", "g++", "c++"):
        found = shutil.which(name)
        if found:
            return found
    msys_gcc = Path(r"C:\msys64\ucrt64\bin\g++.exe")
    if msys_gcc.is_file():
        return str(msys_gcc)
    raise NexirError("C++17 compiler missing; install Clang++ or G++, or set NEXIR_CXX")


def _output_path(source: Path, requested: Path | None, *, shared: bool = False) -> Path:
    suffix = (".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")) if shared else (".exe" if os.name == "nt" else "")
    return (requested or source.with_suffix(suffix)).resolve()


def _python_link_settings() -> tuple[Path, Path, str, Path]:
    info = python_info()
    include, library_dir, library, dll_dir = (info.get(key) for key in ("include", "library_dir", "library", "dll_dir"))
    if not all((include, library_dir, library, dll_dir)):
        raise NexirError("cannot determine Python linker flags")
    if os.name != "nt":
        if library.startswith("lib"):
            library = library[3:]
        library = re.sub(r"\.so(?:\.[0-9.]+)?$", "", library)
        for suffix in (".dylib", ".so", ".a"):
            if library.endswith(suffix):
                library = library[:-len(suffix)]
                break
    if not Path(library_dir).is_dir():
        raise NexirError(f"Python library directory does not exist: {library_dir}")
    return Path(include), Path(library_dir), library, Path(dll_dir)


def _cpp_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _enable_python_startup(code: str, preload: list[str]) -> str:
    marker = "int main(void) {"
    if marker not in code:
        raise NexirError("cannot locate generated C++ entry point for Python initialization")
    startup = ['extern "C" void nx_py_initialize(void);', 'extern "C" int nx_py_preload(const char*);', marker, "    nx_py_initialize();"]
    startup.extend(f"    if (!nx_py_preload({_cpp_string(name)})) return 1;" for name in preload)
    return code.replace(marker, "\n".join(startup) + "\n   ", 1)


def _build(
    source: Path,
    output: Path,
    *,
    emit_c: Path | None = None,
    shared: bool = False,
    cpp_sources: list[Path] | None = None,
    include_dirs: list[Path] | None = None,
    library_dirs: list[Path] | None = None,
    libraries: list[str] | None = None,
    objects: list[Path] | None = None,
    static_runtime: bool = True,
    icon: Path | None = None,
    python: bool = False,
    python_preload: list[str] | None = None,
    vcpkg_triplet: str | None = None,
    require_main: bool = True,
    debug: bool = False,
    sanitize: bool = False,
    strict_sandbox: bool = False,
    allowed_py_modules: set[str] | None = None,
    audit_log_out: Path | None = None,
    profile: str = "balanced",
    use_cache: bool = True,
) -> Path | None:
    audit = CompileAuditLog()
    code = compile_file(
        source,
        debug=debug,
        require_main=require_main,
        audit_log=audit,
        strict_sandbox=strict_sandbox,
        allowed_py_modules=allowed_py_modules,
    )
    python = python or "/* NX_PY_CALL */" in code
    automatic_preload = re.findall(r"/\* NX_PY_PRELOAD ([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*) \*/", code)
    preload = list(dict.fromkeys([*automatic_preload, *(python_preload or [])]))
    if preload and not python:
        raise NexirError("--python-preload requires --python")
    if any(not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) for module in preload):
        raise NexirError("invalid Python module name in preload list")
    if python and require_main:
        code = _enable_python_startup(code, preload)

    if audit_log_out:
        audit_log_out.parent.mkdir(parents=True, exist_ok=True)
        audit_log_out.write_text(audit.to_jsonl(), encoding="utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    if emit_c:
        emit_c.parent.mkdir(parents=True, exist_ok=True)
        emit_c.write_text(code, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="nexir-", dir=output.parent) as directory:
        runtime_python_dir: Path | None = None
        c_file, staged_output = Path(directory) / "program.cpp", Path(directory) / output.name
        c_file.write_text(code, encoding="utf-8")
        compiler = _compiler()
        env = os.environ.copy()
        if os.name == "nt":
            env["PATH"] = str(Path(compiler).resolve().parent) + os.pathsep + env.get("PATH", "")

        profiles = {"balanced": "-O2", "fast": "-O3", "size": "-Os"}
        if profile not in profiles:
            raise NexirError(f"unknown optimization profile: {profile}")
        opt_flags = ["-g", "-O0", "-DNX_DEBUG=1"] if debug else [profiles[profile]]
        command = [compiler, "-std=c++17", *opt_flags, "-Wall", "-Wextra", "-Wno-unused-function", "-Wno-unknown-pragmas"]
        if not debug and not sanitize and profile == "size" and not shared:
            command.extend(["-ffunction-sections", "-fdata-sections"])
        if sanitize and os.name != "nt":
            command.extend(["-fsanitize=address,undefined", "-fno-omit-frame-pointer"])
        if "g++" in Path(compiler).name:
            command.append("-fopenmp-simd")
        if os.name != "nt":
            command.append("-pthread")
        if os.name == "nt" and static_runtime and not shared:
            command.append("-static")
        if shared:
            command.append("-shared")
            if os.name != "nt":
                command.append("-fPIC")
        effective_includes = [Path(__file__).parent / "include", *list(include_dirs or [])]
        effective_library_dirs = list(library_dirs or [])
        effective_libraries = list(libraries or [])
        if vcpkg_triplet is not None:
            prefix = cpp_prefix(vcpkg_triplet)
            effective_includes.append(prefix / "include")
            effective_library_dirs.append(prefix / "lib")
        source_files = list(cpp_sources or [])
        if python:
            py_include, py_library_dir, py_library, py_dll_dir = _python_link_settings()
            runtime_python_dir = py_dll_dir
            effective_includes.append(py_include)
            effective_library_dirs.append(py_library_dir)
            effective_libraries.append(py_library)
            bridge = Path(directory) / "nexir_python_bridge.cpp"
            bridge.write_text(PYTHON_BRIDGE_CPP, encoding="utf-8")
            source_files.append(bridge)
            if os.name == "nt":
                env["PATH"] = str(py_dll_dir) + os.pathsep + env.get("PATH", "")
        for include_dir in effective_includes:
            command.extend(["-I", str(include_dir.resolve())])
        # Cache generated translation units separately; external sources still use
        # the normal compiler path, preserving their include and __FILE__ semantics.
        if use_cache and not debug and not sanitize:
            compile_flags = [flag for flag in command[1:] if flag not in {"-shared", "-static"}]
            generated = [c_file] + ([bridge] if python else [])
            for index, unit in enumerate(generated):
                obj = cached_object(compiler, compile_flags, unit, Path(directory) / f"nx-unit-{index}.o", env)
                command.append(str(obj))
            if python:
                source_files.remove(bridge)
        else:
            command.append(str(c_file))
        command.extend(str(path.resolve()) for path in source_files)
        command.extend(str(path.resolve()) for path in objects or [])
        if os.name == "nt" and not shared:
            icon_file = icon or Path(__file__).parent / "assets" / "nx-icon.ico"
            windres = shutil.which("windres") or str(Path(compiler).resolve().parent / "windres.exe")
            if icon_file.is_file() and Path(windres).is_file():
                rc_file, icon_object = Path(directory) / "icon.rc", Path(directory) / "icon.o"
                rc_file.write_text(f'1 ICON "{icon_file.resolve().as_posix()}"\n', encoding="utf-8")
                result = subprocess.run([windres, "-i", str(rc_file), "-o", str(icon_object)], capture_output=True, text=True, errors="replace", env=env)
                if result.returncode:
                    raise NexirError("icon resource failed: " + result.stderr.strip())
                command.append(str(icon_object))
        for library_dir in effective_library_dirs:
            command.extend(["-L", str(library_dir.resolve())])
        command.extend("-l" + name for name in effective_libraries)
        if os.name == "nt":
            command.append("-lws2_32")
        else:
            command.append("-ldl")
        if not debug and not sanitize and profile == "size" and not shared:
            command.append("-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections")
        command.extend(["-o", str(staged_output)])
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", env=env)
        if result.returncode:
            raise NexirError(f"C++ backend failed (exit {result.returncode}):\n" + (result.stderr or result.stdout).strip())
        os.replace(staged_output, output)
        return runtime_python_dir


def _add_sandbox_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sandbox", action="store_true", help="enable strict runtime sandbox")
    p.add_argument("--max-memory", help="max heap/cache bytes (e.g. 64M, 256M)")
    p.add_argument("--max-allocs", type=int, help="max allocation count")
    p.add_argument("--timeout", type=int, help="execution timeout in milliseconds")
    p.add_argument("--allow-read", action="append", default=[], help="allow reading inside path")
    p.add_argument("--allow-write", action="append", default=[], help="allow writing inside path")
    p.add_argument("--allow-host", action="append", default=[], help="allow network host")
    p.add_argument("--deny-net", action="store_true", help="deny all network access")
    p.add_argument("--deny-fs", action="store_true", help="deny all filesystem access")
    p.add_argument("--deny-ffi", action="store_true", help="deny runtime dynamic native library loading (not linked native code)")
    p.add_argument("--allow-py", action="append", default=[], help="allowlist Python modules")
    p.add_argument("--audit-log", type=Path, help="write JSONL module/sandbox audit log")
    p.add_argument("--audit-stderr", action="store_true", help="print audit log lines to stderr")
    p.add_argument("--trace-mem", action="store_true", help="print memory cache/leak stats at exit")


def _apply_sandbox_env(args: argparse.Namespace, env: dict[str, str]) -> None:
    if getattr(args, "sandbox", False):
        env["NX_SANDBOX"] = "1"
    if getattr(args, "max_memory", None):
        env["NX_MAX_MEMORY"] = str(args.max_memory)
    if getattr(args, "max_allocs", None):
        env["NX_MAX_ALLOCS"] = str(args.max_allocs)
    if getattr(args, "timeout", None):
        env["NX_TIMEOUT_MS"] = str(args.timeout)
    if getattr(args, "allow_read", None):
        env["NX_ALLOW_READ"] = ";".join(args.allow_read)
    if getattr(args, "allow_write", None):
        env["NX_ALLOW_WRITE"] = ";".join(args.allow_write)
    if getattr(args, "allow_host", None):
        env["NX_ALLOW_HOSTS"] = ";".join(args.allow_host)
    if getattr(args, "deny_net", False):
        env["NX_DENY_NET"] = "1"
    if getattr(args, "deny_fs", False):
        env["NX_DENY_FS"] = "1"
    if getattr(args, "deny_ffi", False):
        env["NX_DENY_FFI"] = "1"
    if getattr(args, "allow_py", None):
        env["NX_ALLOW_PY_MODULES"] = ";".join(args.allow_py)
    if getattr(args, "audit_log", None):
        env["NX_AUDIT_LOG"] = str(args.audit_log.resolve())
    if getattr(args, "audit_stderr", False):
        env["NX_AUDIT_STDERR"] = "1"
    if getattr(args, "trace_mem", False):
        env["NX_DEBUG_MEM"] = "1"


def _handle_icon_command(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    assets_dir = Path(__file__).resolve().parent / "assets"
    meipass = getattr(sys, "_MEIPASS", None)
    ext_src = (Path(meipass) / "editors" / "vscode") if meipass and (Path(meipass) / "editors" / "vscode").is_dir() else (repo_root / "editors" / "vscode")
    editor_roots = (
        Path.home() / ".vscode" / "extensions",
        Path.home() / ".antigravity" / "extensions",
        Path.home() / ".antigravity-ide" / "extensions",
    )
    if args.action == "register":
        if os.name != "nt":
            print("Windows Explorer icon registration is only applicable on Windows; use `nx icon install-editor` for editor icons.")
            return
        import ctypes
        import winreg

        ico_src = assets_dir / "nx-icon.ico"
        if not ico_src.is_file():
            raise NexirError(f"icon asset missing: {ico_src}")
        dest_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Programs" / "Nexir"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_ico = dest_dir / "nx-icon.ico"
        shutil.copy2(ico_src, dest_ico)

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\.nx") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "Nexir.SourceFile")
            winreg.SetValueEx(k, "PerceivedType", 0, winreg.REG_SZ, "text")
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\Nexir.SourceFile") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "NEXIR Source File")
            winreg.SetValueEx(k, "FriendlyTypeName", 0, winreg.REG_SZ, "NEXIR Source File")
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\Nexir.SourceFile\DefaultIcon") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, f"{dest_ico},0")

        # Notify Windows Explorer to refresh file icons in open folder views
        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0x0000, None, None)
        print(f"Registered .nx Windows Explorer icon -> {dest_ico}")
        return

    if args.action == "install-editor":
        if not ext_src.is_dir():
            raise NexirError(f"editor extension directory missing: {ext_src}")
        installed_paths = []
        for editor_dir in editor_roots:
            if editor_dir.parent.exists() or editor_dir.exists():
                editor_dir.mkdir(parents=True, exist_ok=True)
                target = editor_dir / f"quoctuan21112009.nexir-language-{__version__}"
                for stale_name in (f"nexir-language-{__version__}", "quoctuan21112009.nexir-language-0.3.0"):
                    stale = editor_dir / stale_name
                    if stale.exists() and stale != target:
                        shutil.rmtree(stale, ignore_errors=True)
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(ext_src, target)
                installed_paths.append(str(target))
                # If legacy nexir-runner exists in .antigravity/extensions, update its icons/manifest too
                legacy_runner = editor_dir / "nexir-runner"
                if legacy_runner.is_dir():
                    shutil.copy2(ext_src / "icons" / "nexir.svg", legacy_runner / "nexir.svg")
                    shutil.copy2(ext_src / "icons" / "nexir-icon-theme.json", legacy_runner / "nx-icon-theme.json")
                    shutil.copy2(ext_src / "syntaxes" / "nexir.tmLanguage.json", legacy_runner / "nexir.tmLanguage.json")
                    # Fix path inside legacy nx-icon-theme.json so ./nexir.svg resolves in root folder
                    theme_txt = (ext_src / "icons" / "nexir-icon-theme.json").read_text(encoding="utf-8")
                    (legacy_runner / "nx-icon-theme.json").write_text(theme_txt, encoding="utf-8")
                    pkg = json.loads((legacy_runner / "package.json").read_text(encoding="utf-8-sig"))
                    pkg["version"] = __version__
                    pkg.setdefault("contributes", {})
                    if pkg["contributes"].get("languages"):
                        pkg["contributes"]["languages"][0]["icon"] = {"light": "./nexir.svg", "dark": "./nexir.svg"}
                    pkg["contributes"]["iconThemes"] = [
                        {"id": "nexir-file-icons", "label": "Nexir File Icons", "path": "./nx-icon-theme.json"}
                    ]
                    (legacy_runner / "package.json").write_text(json.dumps(pkg, indent=2) + "\n", encoding="utf-8")
                    installed_paths.append(str(legacy_runner))
        for p in installed_paths:
            print(f"Installed editor .nx icon extension -> {p}")
        return

    if args.action == "status":
        win_icon = None
        if os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\Nexir.SourceFile\DefaultIcon") as k:
                    win_icon, _ = winreg.QueryValueEx(k, "")
            except OSError:
                win_icon = None
        editors = []
        for editor_dir in editor_roots:
            if editor_dir.is_dir():
                for child in sorted(editor_dir.iterdir()):
                    if "nexir" in child.name.lower():
                        editors.append(str(child))
        print(json.dumps({"windows_explorer_icon": win_icon, "editor_extensions": editors}, indent=2))
        return
    raise NexirError(f"unknown icon action: {args.action}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexir", description="NEXIR 2.0 native compiler & ecosystem")
    parser.add_argument("--version", action="version", version=f"NEXIR {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="compile .nx to native executable or shared library")
    build.add_argument("source", type=Path)
    build.add_argument("-o", "--output", type=Path)
    build.add_argument("--emit-cpp", "--emit-c", dest="emit_cpp", type=Path)
    build.add_argument("--shared", action="store_true")
    build.add_argument("--no-main", action="store_true")
    build.add_argument("--debug", "-g", action="store_true")
    build.add_argument("--profile", choices=["balanced", "fast", "size"], default="balanced")
    build.add_argument("--no-cache", action="store_true", help="bypass native object cache")
    build.add_argument("--sanitize", action="store_true")
    build.add_argument("--cpp", action="append", type=Path, default=[])
    build.add_argument("--include-dir", action="append", type=Path, default=[])
    build.add_argument("--library-dir", action="append", type=Path, default=[])
    build.add_argument("--library", action="append", default=[])
    build.add_argument("--object", action="append", type=Path, default=[])
    build.add_argument("--vcpkg", nargs="?", const=default_triplet(), metavar="TRIPLET")
    build.add_argument("--dynamic-runtime", action="store_true")
    build.add_argument("--icon", type=Path)
    build.add_argument("--python", action="store_true")
    build.add_argument("--python-preload", action="append", default=[], metavar="MODULE")
    build.add_argument("--emit-bindings", choices=["c", "cpp", "python", "rust", "go"], help="also emit FFI bindings")
    _add_sandbox_args(build)

    check = sub.add_parser("check", help="parse, type-check, and safety-check .nx source")
    check.add_argument("source", type=Path)
    check.add_argument("--audit-log", type=Path)

    emit = sub.add_parser("emit", help="generate C++17 source without linking")
    emit.add_argument("source", type=Path)
    emit.add_argument("-o", "--output", type=Path, required=True)
    emit.add_argument("--debug", "-g", action="store_true")

    run = sub.add_parser("run", help="compile and run .nx source")
    run.add_argument("source", type=Path)
    run.add_argument("--debug", "-g", action="store_true")
    run.add_argument("--profile", choices=["balanced", "fast", "size"], default="balanced")
    run.add_argument("--no-cache", action="store_true", help="bypass native object cache")
    _add_sandbox_args(run)
    run.add_argument("args", nargs=argparse.REMAINDER)

    debug_cmd = sub.add_parser("debug", help="compile with debug trace & memory diagnostics and run")
    debug_cmd.add_argument("source", type=Path)
    debug_cmd.add_argument("--gdb", action="store_true", help="launch under gdb if available")
    _add_sandbox_args(debug_cmd)
    debug_cmd.add_argument("args", nargs=argparse.REMAINDER)

    bind_cmd = sub.add_parser("bind", help="generate cross-language FFI bindings (c, cpp, python, rust, go)")
    bind_cmd.add_argument("source", type=Path)
    bind_cmd.add_argument("--lang", required=True, choices=["c", "cpp", "python", "rust", "go"])
    bind_cmd.add_argument("-o", "--out-dir", type=Path, default=Path("bindings"))

    doc_cmd = sub.add_parser("doc", help="generate pkg.go.dev-style documentation for a project")
    doc_cmd.add_argument("--path", type=Path, default=Path("."))
    doc_cmd.add_argument("-o", "--out-dir", type=Path, default=Path("build/docs"))
    doc_cmd.add_argument("--format", choices=["html", "md"], default="html")
    doc_cmd.add_argument("--serve", action="store_true")
    doc_cmd.add_argument("--port", type=int, default=8080)

    init = sub.add_parser("init", help="create a starter .nx project")
    init.add_argument("directory", type=Path)

    icon_cmd = sub.add_parser("icon", help="manage .nx folder & editor file icon integration")
    icon_sub = icon_cmd.add_subparsers(dest="action", required=True)
    icon_sub.add_parser("register", help="register .nx file icon in Windows Explorer folders")
    icon_sub.add_parser("install-editor", help="install VS Code / Antigravity .nx icon extension")
    icon_sub.add_parser("status", help="show .nx icon registration status")

    sub.add_parser("doctor", help="show compiler availability")

    pkg = sub.add_parser("pkg", help="manage external and Nexir-native packages")
    pkg_sub = pkg.add_subparsers(dest="kind", required=True)
    py_pkg = pkg_sub.add_parser("python")
    py_sub = py_pkg.add_subparsers(dest="action", required=True)
    py_install = py_sub.add_parser("install"); py_install.add_argument("package"); py_install.add_argument("--upgrade", action="store_true")
    py_show = py_sub.add_parser("show"); py_show.add_argument("package")

    cpp_pkg = pkg_sub.add_parser("cpp")
    cpp_sub = cpp_pkg.add_subparsers(dest="action", required=True)
    cpp_sub.add_parser("bootstrap")
    for action in ("install", "remove"):
        cmd = cpp_sub.add_parser(action); cmd.add_argument("package"); cmd.add_argument("--triplet")
    cpp_path = cpp_sub.add_parser("path"); cpp_path.add_argument("--triplet")

    rust_pkg = pkg_sub.add_parser("rust", help="build a Cargo C-ABI library")
    rust_sub = rust_pkg.add_subparsers(dest="action", required=True)
    rust_build = rust_sub.add_parser("build")
    rust_build.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    rust_build.add_argument("--target", help="explicit Rust target triple")
    rust_build.add_argument("--offline", action="store_true")

    nx_pkg = pkg_sub.add_parser("nx", help="manage Nexir-native packages & lockfiles")
    nx_sub = nx_pkg.add_subparsers(dest="action", required=True)
    nx_init = nx_sub.add_parser("init"); nx_init.add_argument("--path", type=Path, default=Path(".")); nx_init.add_argument("--name")
    for action in ("validate", "list", "resolve", "install", "publish", "verify", "audit", "graph"):
        cmd = nx_sub.add_parser(action); cmd.add_argument("--path", type=Path, default=Path("."))
    nx_add = nx_sub.add_parser("add"); nx_add.add_argument("package"); nx_add.add_argument("version", nargs="?"); nx_add.add_argument("--path", type=Path, default=Path("."))
    nx_rem = nx_sub.add_parser("remove"); nx_rem.add_argument("package"); nx_rem.add_argument("--path", type=Path, default=Path("."))
    nx_search = nx_sub.add_parser("search"); nx_search.add_argument("query", nargs="?", default=""); nx_search.add_argument("--path", type=Path, default=Path("."))
    nx_doc = nx_sub.add_parser("doc"); nx_doc.add_argument("--path", type=Path, default=Path(".")); nx_doc.add_argument("--format", choices=["html", "md"], default="html"); nx_doc.add_argument("--serve", action="store_true"); nx_doc.add_argument("--port", type=int, default=8080)

    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            print(f"NEXIR {__version__}\nFrontend Python: {sys.version.split()[0]}\nC++17 compiler: {_compiler()}")
            return 0
        if args.command == "pkg":
            ecosystem_command(args) if args.kind == "nx" else package_command(args)
            return 0
        if args.command == "doc":
            out = generate_docs(args.path.resolve(), args.out_dir.resolve(), format_kind=args.format)
            print(out)
            if args.serve:
                serve_docs(args.out_dir.resolve(), port=args.port)
            return 0
        if args.command == "bind":
            out = generate_bindings(args.source.resolve(), args.lang, args.out_dir.resolve())
            print(out)
            return 0
        if args.command == "icon":
            _handle_icon_command(args)
            return 0
        if args.command == "init":
            directory = args.directory.resolve()
            directory.mkdir(parents=True, exist_ok=True)
            starter = directory / "main.nx"
            if starter.exists():
                raise NexirError(f"refusing to overwrite {starter}")
            starter.write_text('def main():\n    print("Hello from NEXIR")\n', encoding="utf-8")
            print(starter)
            return 0

        source = args.source.resolve()
        if args.command == "check":
            audit = CompileAuditLog()
            compile_file(source, audit_log=audit)
            if args.audit_log:
                args.audit_log.write_text(audit.to_jsonl(), encoding="utf-8")
            print(f"OK: {source}")
            return 0
        if args.command == "emit":
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(compile_file(source, debug=args.debug), encoding="utf-8")
            print(output)
            return 0
        if args.command == "build":
            output = _output_path(source, args.output, shared=args.shared)
            _build(
                source,
                output,
                emit_c=args.emit_cpp,
                shared=args.shared,
                cpp_sources=args.cpp,
                include_dirs=args.include_dir,
                library_dirs=args.library_dir,
                libraries=args.library,
                objects=args.object,
                static_runtime=not args.dynamic_runtime,
                icon=args.icon,
                python=args.python,
                python_preload=args.python_preload,
                vcpkg_triplet=args.vcpkg,
                require_main=not (args.shared or args.no_main),
                debug=args.debug,
                sanitize=args.sanitize,
                profile=args.profile,
                use_cache=not args.no_cache,
                strict_sandbox=args.sandbox,
                allowed_py_modules=set(args.allow_py) if args.allow_py else None,
                audit_log_out=args.audit_log,
            )
            if args.emit_bindings:
                generate_bindings(source, args.emit_bindings, output.parent)
            print(output)
            return 0
        if args.command in {"run", "debug"}:
            is_dbg = args.command == "debug" or getattr(args, "debug", False)
            with tempfile.TemporaryDirectory(prefix="nexir-run-") as directory:
                output = Path(directory) / ("program.exe" if os.name == "nt" else "program")
                python_dll_dir = _build(
                    source,
                    output,
                    debug=is_dbg,
                    profile=getattr(args, "profile", "balanced"),
                    use_cache=not getattr(args, "no_cache", False),
                    strict_sandbox=args.sandbox,
                    allowed_py_modules=set(args.allow_py) if args.allow_py else None,
                    audit_log_out=args.audit_log,
                )
                env = os.environ.copy()
                if python_dll_dir is not None:
                    if os.name == "nt":
                        env["PATH"] = str(python_dll_dir) + os.pathsep + env.get("PATH", "")
                    env["PYTHONPATH"] = str(source.parent) + os.pathsep + env.get("PYTHONPATH", "")
                _apply_sandbox_env(args, env)
                if args.command == "debug":
                    env.setdefault("NX_DEBUG_MEM", "1")
                if getattr(args, "gdb", False):
                    gdb = shutil.which("gdb")
                    if not gdb:
                        raise NexirError("gdb not found on PATH")
                    return subprocess.run([gdb, "-ex", "run", "-ex", "bt", "--args", str(output), *args.args], env=env).returncode
                return subprocess.run([str(output), *args.args], env=env).returncode
        return 1
    except (NexirError, OSError) as exc:
        print(f"nexir: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
