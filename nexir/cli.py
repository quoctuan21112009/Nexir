"""Command line interface for the NEXIR bootstrap compiler."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from . import __version__
from .compiler import NexirError, compile_file
from .packages import cpp_prefix, default_triplet, package_command, python_info
from .python_bridge import PYTHON_BRIDGE_CPP


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
    suffix = (".dll" if os.name == "nt" else ".so") if shared else (".exe" if os.name == "nt" else "")
    output = requested or source.with_suffix(suffix)
    return output.resolve()


def _python_link_settings() -> tuple[Path, Path, str, Path]:
    """Return CPython include dir, library dir/name, and runtime DLL directory."""
    info = python_info()
    include = info.get("include")
    if not include:
        raise NexirError("cannot determine Python include directory")
    library_dir_value = info.get("library_dir")
    library = info.get("library")
    dll_dir_value = info.get("dll_dir")
    if not library_dir_value or not library or not dll_dir_value:
        raise NexirError("cannot determine Python linker flags")
    library_dir = Path(library_dir_value)
    dll_dir = Path(dll_dir_value)
    if os.name != "nt":
        if library.startswith("lib"):
            library = library[3:]
        library = re.sub(r"\.so(?:\.[0-9.]+)?$", "", library)
        for suffix in (".dylib", ".so", ".a"):
            if library.endswith(suffix):
                library = library[:-len(suffix)]
                break
    if not library_dir.is_dir():
        raise NexirError(f"Python library directory does not exist: {library_dir}")
    return Path(include), library_dir, library, dll_dir


def _cpp_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _enable_python_startup(code: str, preload: list[str]) -> str:
    marker = "int main(void) {"
    if marker not in code:
        raise NexirError("cannot locate generated C++ entry point for Python initialization")
    startup = [
        'extern "C" void nx_py_initialize(void);',
        'extern "C" int nx_py_preload(const char*);',
        marker,
        "    nx_py_initialize();",
    ]
    startup.extend(
        f"    if (!nx_py_preload({_cpp_string(name)})) return 1;" for name in preload
    )
    return code.replace(marker, "\n".join(startup), 1)


def _build(source: Path, output: Path, *, emit_c: Path | None = None, shared: bool = False,
           cpp_sources: list[Path] | None = None, include_dirs: list[Path] | None = None,
           library_dirs: list[Path] | None = None, libraries: list[str] | None = None,
           objects: list[Path] | None = None, static_runtime: bool = True,
           icon: Path | None = None, python: bool = False,
           python_preload: list[str] | None = None,
           vcpkg_triplet: str | None = None) -> Path | None:
    code = compile_file(source)
    python = python or "/* NX_PY_CALL */" in code
    automatic_preload = re.findall(r"/\* NX_PY_PRELOAD ([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*) \*/", code)
    preload = list(dict.fromkeys([*automatic_preload, *(python_preload or [])]))
    if preload and not python:
        raise NexirError("--python-preload requires --python")
    for module in preload:
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
            raise NexirError(f"invalid Python module name for preload: {module!r}")
    if python:
        code = _enable_python_startup(code, preload)
    output.parent.mkdir(parents=True, exist_ok=True)
    if emit_c:
        emit_c.parent.mkdir(parents=True, exist_ok=True)
        emit_c.write_text(code, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nexir-", dir=output.parent) as directory:
        runtime_python_dir: Path | None = None
        c_file = Path(directory) / "program.cpp"
        staged_output = Path(directory) / output.name
        c_file.write_text(code, encoding="utf-8")
        compiler = _compiler()
        env = os.environ.copy()
        if os.name == "nt":
            env["PATH"] = str(Path(compiler).resolve().parent) + os.pathsep + env.get("PATH", "")
        command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wno-unused-function", "-Wno-unknown-pragmas"]
        if "g++" in Path(compiler).name:
            command.append("-fopenmp-simd")
        if os.name != "nt":
            command.append("-pthread")
        if os.name == "nt" and static_runtime:
            command.append("-static")
        if shared:
            command.extend(["-shared", "-fPIC"])
        effective_includes = list(include_dirs or [])
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
            bridge_file = Path(directory) / "nexir_python_bridge.cpp"
            bridge_file.write_text(PYTHON_BRIDGE_CPP, encoding="utf-8")
            source_files.append(bridge_file)
            if os.name == "nt":
                env["PATH"] = str(py_dll_dir) + os.pathsep + env.get("PATH", "")
        for include_dir in effective_includes:
            command.extend(["-I", str(include_dir.resolve())])
        command.append(str(c_file))
        command.extend(str(path.resolve()) for path in source_files)
        command.extend(str(path.resolve()) for path in objects or [])
        if os.name == "nt" and not shared:
            icon_file = icon or Path(__file__).parent / "assets" / "nx-icon.ico"
            windres = shutil.which("windres") or str(Path(compiler).resolve().parent / "windres.exe")
            if icon_file.is_file() and Path(windres).is_file():
                rc_file = Path(directory) / "icon.rc"
                icon_object = Path(directory) / "icon.o"
                rc_file.write_text(f'1 ICON "{icon_file.resolve().as_posix()}"\n', encoding="utf-8")
                icon_result = subprocess.run([windres, "-i", str(rc_file), "-o", str(icon_object)],
                                             capture_output=True, text=True, errors="replace", env=env)
                if icon_result.returncode:
                    raise NexirError("icon resource failed: " + icon_result.stderr.strip())
                command.append(str(icon_object))
        for library_dir in effective_library_dirs:
            command.extend(["-L", str(library_dir.resolve())])
        command.extend("-l" + name for name in effective_libraries)
        if os.name == "nt":
            command.append("-lws2_32")
        command.extend(["-o", str(staged_output)])
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", env=env)
        if result.returncode:
            raise NexirError(f"C++ backend failed (exit {result.returncode}):\n" + (result.stderr or result.stdout).strip())
        os.replace(staged_output, output)
        return runtime_python_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexir", description="NEXIR native compiler")
    parser.add_argument("--version", action="version", version=f"NEXIR {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="compile .nx to native executable")
    build.add_argument("source", type=Path)
    build.add_argument("-o", "--output", type=Path)
    build.add_argument("--emit-cpp", "--emit-c", dest="emit_cpp", type=Path, help="also write generated C++ for inspection")
    build.add_argument("--shared", action="store_true", help="build a shared library for @export functions")
    build.add_argument("--cpp", action="append", type=Path, default=[], help="link an additional C++ source file")
    build.add_argument("--include-dir", action="append", type=Path, default=[])
    build.add_argument("--library-dir", action="append", type=Path, default=[])
    build.add_argument("--library", action="append", default=[])
    build.add_argument("--object", action="append", type=Path, default=[], help="link a precompiled object, including CUDA objects")
    build.add_argument("--vcpkg", nargs="?", const=default_triplet(), metavar="TRIPLET",
                       help="add vcpkg include/lib paths for a triplet; still specify --library")
    build.add_argument("--dynamic-runtime", action="store_true", help="do not statically link the C++ runtime on Windows")
    build.add_argument("--icon", type=Path, help="Windows executable icon (.ico)")
    build.add_argument("--python", action="store_true", help="embed CPython and link the Nexir zero-copy f64 span bridge")
    build.add_argument("--python-preload", action="append", default=[], metavar="MODULE",
                       help="import a Python module once at process startup (repeatable; requires --python)")
    check = sub.add_parser("check", help="parse and type-check .nx source")
    check.add_argument("source", type=Path)
    emit = sub.add_parser("emit", help="generate C++17 source without linking")
    emit.add_argument("source", type=Path)
    emit.add_argument("-o", "--output", type=Path, required=True)
    run = sub.add_parser("run", help="compile and run .nx source")
    run.add_argument("source", type=Path)
    run.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to executable")
    init = sub.add_parser("init", help="create a starter .nx project")
    init.add_argument("directory", type=Path)
    sub.add_parser("doctor", help="show compiler availability")
    pkg = sub.add_parser("pkg", help="install external Python or C++ libraries")
    pkg_sub = pkg.add_subparsers(dest="kind", required=True)
    py_pkg = pkg_sub.add_parser("python", help="use pip with the external Python interpreter")
    py_pkg_sub = py_pkg.add_subparsers(dest="action", required=True)
    py_install = py_pkg_sub.add_parser("install")
    py_install.add_argument("package")
    py_install.add_argument("--upgrade", action="store_true")
    py_show = py_pkg_sub.add_parser("show")
    py_show.add_argument("package")
    cpp_pkg = pkg_sub.add_parser("cpp", help="use vcpkg with MinGW triplets")
    cpp_pkg_sub = cpp_pkg.add_subparsers(dest="action", required=True)
    cpp_pkg_sub.add_parser("bootstrap", help="clone and initialize vcpkg")
    for action in ("install", "remove"):
        command = cpp_pkg_sub.add_parser(action)
        command.add_argument("package")
        command.add_argument("--triplet")
    cpp_path = cpp_pkg_sub.add_parser("path")
    cpp_path.add_argument("--triplet")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            print(f"NEXIR {__version__}")
            print(f"Frontend Python: {sys.version.split()[0]}" + (" (bundled)" if getattr(sys, "frozen", False) else ""))
            print(f"C++17 compiler: {_compiler()}")
            return 0
        if args.command == "pkg":
            package_command(args)
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
            compile_file(source)
            print(f"OK: {source}")
            return 0
        if args.command == "emit":
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(compile_file(source), encoding="utf-8")
            print(output)
            return 0
        if args.command == "build":
            output = _output_path(source, args.output, shared=args.shared)
            _build(source, output, emit_c=args.emit_cpp, shared=args.shared,
                   cpp_sources=args.cpp, include_dirs=args.include_dir,
                   library_dirs=args.library_dir, libraries=args.library, objects=args.object,
                   static_runtime=not args.dynamic_runtime, icon=args.icon,
                   python=args.python, python_preload=args.python_preload,
                   vcpkg_triplet=args.vcpkg)
            print(output)
            return 0
        with tempfile.TemporaryDirectory(prefix="nexir-run-") as directory:
            output = Path(directory) / ("program.exe" if os.name == "nt" else "program")
            python_dll_dir = _build(source, output)
            env = os.environ.copy()
            if python_dll_dir is not None:
                if os.name == "nt":
                    env["PATH"] = str(python_dll_dir) + os.pathsep + env.get("PATH", "")
                env["PYTHONPATH"] = str(source.parent) + os.pathsep + env.get("PYTHONPATH", "")
            return subprocess.run([str(output), *args.args], env=env).returncode
    except (NexirError, OSError) as exc:
        print(f"nexir: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
